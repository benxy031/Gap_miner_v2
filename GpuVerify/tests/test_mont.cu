/* Direct cross-check of the CIOS Montgomery layer against GMP. */
#include "mont.cuh"
#include "gv.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

template <int N32>
__global__ void mont_test_kernel(const uint32_t *__restrict__ g_a,
                                 const uint32_t *__restrict__ g_b,
                                 const uint32_t *__restrict__ g_n,
                                 const uint32_t *__restrict__ g_np0,
                                 uint32_t *__restrict__ g_r, int count)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count)
        return;
    uint32_t a[N32], b[N32], n[N32], r[N32];
#pragma unroll 4
    for (int i = 0; i < N32; i++) {
        a[i] = g_a[(size_t)t * N32 + i];
        b[i] = g_b[(size_t)t * N32 + i];
        n[i] = g_n[(size_t)t * N32 + i];
    }
    mont_mul<N32>(r, a, b, n, g_np0[t]);
#pragma unroll 4
    for (int i = 0; i < N32; i++)
        g_r[(size_t)t * N32 + i] = r[i];
}

template <int N32>
static int run_case(int count, int seed)
{
    srand(seed);
    uint32_t *a = (uint32_t *)malloc((size_t)count * N32 * 4);
    uint32_t *b = (uint32_t *)malloc((size_t)count * N32 * 4);
    uint32_t *n = (uint32_t *)malloc((size_t)count * N32 * 4);
    uint32_t *np0 = (uint32_t *)malloc((size_t)count * 4);
    uint32_t *r = (uint32_t *)malloc((size_t)count * N32 * 4);

    mpz_t A, B, N, T, R, RINV, MOD, EXPECT, GOT;
    mpz_init(A);
    mpz_init(B);
    mpz_init(N);
    mpz_init(T);
    mpz_init(R);
    mpz_init(RINV);
    mpz_init(MOD);
    mpz_init(EXPECT);
    mpz_init(GOT);
    mpz_setbit(MOD, 32);

    int bad = 0;
    mpz_setbit(R, 32 * N32);
    for (int i = 0; i < count; i++) {
        int bits = 32 * N32 - 1;
        /* random odd modulus with the top bit set */
        mpz_setbit(N, bits);
        mpz_setbit(N, bits - 1 - (rand() % (N32 > 4 ? N32 * 8 : 16)));
        mpz_setbit(N, 0);
        /* random a, b < N */
        mpz_set_ui(A, (unsigned long)rand());
        mpz_mul_2exp(A, A, 64);
        mpz_add_ui(A, A, (unsigned long)rand());
        mpz_mod(A, A, N);
        mpz_set_ui(B, (unsigned long)rand());
        mpz_mul_2exp(B, B, 64);
        mpz_add_ui(B, B, (unsigned long)rand());
        mpz_mod(B, B, N);

        /* expected: A*B*R^-1 mod N with R = 2^(32*N32) */
        mpz_invert(RINV, R, N);
        mpz_mul(EXPECT, A, B);
        mpz_mul(EXPECT, EXPECT, RINV);
        mpz_mod(EXPECT, EXPECT, N);

        /* np0 = -N^-1 mod 2^32 */
        mpz_set_ui(T, mpz_get_ui(N));
        mpz_invert(T, T, MOD);
        np0[i] = (uint32_t)(0 - mpz_get_ui(T));

        size_t cnt;
        mpz_export(a + (size_t)i * N32, &cnt, -1, 4, 0, 0, A);
        mpz_export(b + (size_t)i * N32, &cnt, -1, 4, 0, 0, B);
        mpz_export(n + (size_t)i * N32, &cnt, -1, 4, 0, 0, N);
    }

    uint32_t *d_a, *d_b, *d_n, *d_np0, *d_r;
    cudaMalloc(&d_a, (size_t)count * N32 * 4);
    cudaMalloc(&d_b, (size_t)count * N32 * 4);
    cudaMalloc(&d_n, (size_t)count * N32 * 4);
    cudaMalloc(&d_np0, (size_t)count * 4);
    cudaMalloc(&d_r, (size_t)count * N32 * 4);
    cudaMemcpy(d_a, a, (size_t)count * N32 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, (size_t)count * N32 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, n, (size_t)count * N32 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_np0, np0, (size_t)count * 4, cudaMemcpyHostToDevice);

    mont_test_kernel<N32><<<(count + 255) / 256, 256>>>(d_a, d_b, d_n, d_np0,
                                                       d_r, count);
    cudaDeviceSynchronize();
    cudaMemcpy(r, d_r, (size_t)count * N32 * 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < count && bad < 3; i++) {
        /* rebuild A, B, N from the host copies and recompute the
         * expectation (EXPECT above held only the last iteration) */
        mpz_import(A, N32, -1, 4, 0, 0, a + (size_t)i * N32);
        mpz_import(B, N32, -1, 4, 0, 0, b + (size_t)i * N32);
        mpz_import(N, N32, -1, 4, 0, 0, n + (size_t)i * N32);
        mpz_invert(RINV, R, N);
        mpz_mul(EXPECT, A, B);
        mpz_mul(EXPECT, EXPECT, RINV);
        mpz_mod(EXPECT, EXPECT, N);
        mpz_import(GOT, N32, -1, 4, 0, 0, r + (size_t)i * N32);
        if (mpz_cmp(GOT, EXPECT) != 0) {
            bad++;
            gmp_printf("MONT MISMATCH N32=%d i=%d\n  A=%Zx\n  B=%Zx\n  N=%Zx\n  got=%Zx\n  exp=%Zx\n",
                       N32, i, A, B, N, GOT, EXPECT);
        }
    }
    printf("N32=%d: %d cases, %d mismatches\n", N32, count, bad);

    mpz_clear(A);
    mpz_clear(B);
    mpz_clear(N);
    mpz_clear(T);
    mpz_clear(R);
    mpz_clear(RINV);
    mpz_clear(MOD);
    mpz_clear(EXPECT);
    mpz_clear(GOT);
    free(a);
    free(b);
    free(n);
    free(np0);
    free(r);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_n);
    cudaFree(d_np0);
    cudaFree(d_r);
    return bad;
}

int main(void)
{
    int bad = 0;
    bad += run_case<16>(2000, 1);
    bad += run_case<32>(2000, 2);
    bad += run_case<64>(2000, 3);
    bad += run_case<128>(1000, 4);
    bad += run_case<256>(500, 5);
    bad += run_case<512>(100, 6);
    bad += run_case<1024>(50, 7);
    printf(bad ? "MONT TEST: %d FAILURES\n" : "MONT TEST: all passed\n", bad);
    return bad != 0;
}
