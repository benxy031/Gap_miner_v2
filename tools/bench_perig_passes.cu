/* bench_perig_passes.cu - WHERE does perig's 1.14 ns/test go?
 *
 * Follow-up to tools/bench_perig_setup.cu, which showed that the per-candidate
 * SETUP (montgomeryInverse64 + ciosConstants128 = one 128/128 modulo) is only
 * 3.8 % of the test, so PGN's stateful-reciprocal amortisation cannot explain
 * their per-test edge.  This harness splits the rest:
 *
 *   arm K=1  : setup + ONE exponentiation pass + final normalise/compare
 *              (a faithful copy of ciosFermatTest128_hi - VERIFIED below
 *               against the production function on every candidate)
 *   arm K=2/4: the same arm with the exponent loops repeated K times, each pass
 *              seeded by the previous result (a real dependency, so the
 *              compiler cannot CSE the extra passes away)
 *
 * Then   t(K) = fixed + K * t_pass   ->  t_pass from the slope, fixed from the
 * intercept, and t_pass / (number of CIOS squarings per pass, printed) is the
 * per-squaring cost.  Comparing `fixed` and `t_pass` against the measured
 * 1.14 ns/test says whether a 1.5x per-test win has to come from the squaring
 * core (PGN's lazy normalisation + PTX carries) or from the fixed part.
 *
 * Build:  nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *             tools/bench_perig_passes.cu -o bin/bench_perig_passes
 * Run:    bin/bench_perig_passes [N=1048576] [iters=50] [reps=3]
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "perig.cuh"            /* production test + its building blocks */

/* The exponent loops of ciosFermatTest128_hi, verbatim, wrapped in a K-pass
   chain.  Nothing here may differ from the production code at K=1: that is
   what the parity gate below checks. */
__device__ __forceinline__ bool perig_K(uint64_t n_lo, uint64_t n_hi, int K) {
    uint64_t res_lo, res_hi, one_lo, one_hi;
    int bit;
    uint64_t mmagic = montgomeryInverse64(n_lo);
    ciosConstants128(n_lo, n_hi, &one_lo, &one_hi);
    res_hi = one_hi;
    res_lo = one_lo;

    /* K>1 chains each pass from the previous RESULT (so the extra passes
       cannot be CSE'd away).  `one_lo/one_hi` must stay untouched: the final
       comparison below is against the Montgomery representation of 1. */
    uint64_t seed_lo = one_lo, seed_hi = one_hi;

    for (int pass = 0; pass < K; pass++) {
        res_lo = seed_lo;
        res_hi = seed_hi;
        if (n_hi == 0) {
            bit = 64 - my_clz64(n_lo);
            uint64_t msb_bits = bit < 5 ? bit - 1 : 3;
            uint64_t msb_mask = (1 << msb_bits) - 1;
            bit -= msb_bits;
            my_shld64(&res_hi, &res_lo, (n_lo >> bit) & msb_mask);
        } else {
            bit = 64 - my_clz64(n_hi);
            uint64_t msb_bits = bit < 4 ? bit : 3;
            uint64_t msb_mask = (1 << msb_bits) - 1;
            bit -= msb_bits;
            my_shld64(&res_hi, &res_lo, (n_hi >> bit) & msb_mask);

            while (bit >= 3) {
                bit -= 3;
                ciosModSquare3_128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
                my_shld64(&res_hi, &res_lo, ((n_hi >> bit) & 7));
            }
            while (bit) {
                bit -= 1;
                ciosModSquare128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
                my_shld64(&res_hi, &res_lo, ((n_hi >> bit) & 1));
            }
            bit = 64;
        }
        while (bit >= 5) {
            bit -= 3;
            ciosModSquare3_128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
            my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 7));
        }
        while (bit > 1) {
            bit -= 1;
            ciosModSquare128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
            my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 1));
        }
        /* chain the next pass from this result (defeats CSE; K=1 unaffected) */
        seed_lo = res_lo;
        seed_hi = res_hi;
    }

    ciosSubtract128(&res_lo, &res_hi, n_lo, n_hi);
    uint64_t legendre = ((n_lo >> 1) ^ (n_lo >> 2)) & 1;
    uint64_t m1_lo, m1_hi;
    uint8_t c;
    c = my_sbb64(0, n_lo, one_lo, &m1_lo);
    my_sbb64(c, n_hi, one_hi, &m1_hi);
    return ((res_lo == (legendre ? m1_lo : one_lo)) &&
            (res_hi == (legendre ? m1_hi : one_hi)));
}

template <int K>
__global__ static void k_pass(const uint64_t *lo, const uint64_t *hi,
                              uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = perig_K(lo[i], hi[i], K) ? 1 : 0;
}

__global__ static void k_prod(const uint64_t *lo, const uint64_t *hi,
                              uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

typedef void (*kfn)(const uint64_t *, const uint64_t *, uint8_t *, uint32_t);

static double bench(kfn k, const uint64_t *dlo, const uint64_t *dhi,
                    uint8_t *dres, uint32_t n, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);   /* warm */
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms * 1e-3 / (double)iters;
}

/* number of CIOS squarings the production loop performs for n_hi != 0 */
static int count_squarings(uint64_t n_hi) {
    int bit = 64 - __builtin_clzll(n_hi);
    int msb_bits = bit < 4 ? bit : 3;
    bit -= msb_bits;
    int cnt = 0;
    if (bit >= 3) { int k = bit / 3; cnt += 3 * k; bit -= 3 * k; }
    cnt += bit;                       /* while (bit) singles */
    bit = 64;
    while (bit >= 5) { bit -= 3; cnt += 3; }
    while (bit > 1) { bit -= 1; cnt += 1; }
    return cnt;
}

int main(int argc, char **argv) {
    uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    int iters  = (argc > 2) ? atoi(argv[2]) : 50;
    int reps   = (argc > 3) ? atoi(argv[3]) : 3;

    uint64_t *hlo = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)N * 8);
    if (!hlo || !hhi) { fprintf(stderr, "alloc failed\n"); return 1; }

    const uint64_t HI = 10;                 /* 2e20 = 10 * 2^64 + lo */
    unsigned __int128 two_e20 = (unsigned __int128)2000000000000000000ULL * 100u;
    uint64_t base_lo = (uint64_t)(two_e20 - ((unsigned __int128)HI << 64));
    for (uint32_t i = 0; i < N; i++) {
        hlo[i] = base_lo + 2ull * i + 1ull;
        hhi[i] = HI;
    }

    uint64_t *dlo, *dhi;
    uint8_t  *dres;
    cudaMalloc(&dlo, (size_t)N * 8);
    cudaMalloc(&dhi, (size_t)N * 8);
    cudaMalloc(&dres, (size_t)N);
    cudaMemcpy(dlo, hlo, (size_t)N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)N * 8, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nsur = count_squarings(HI);
    printf("device: %s (%d SMs) N=%u iters=%d reps=%d  hi=%llu  squarings/pass=%d\n",
           prop.name, prop.multiProcessorCount, N, iters, reps,
           (unsigned long long)HI, nsur);

    /* ---- GATE: my K=1 copy must agree with the production test everywhere --- */
    {
        uint8_t *a = (uint8_t *)malloc(N), *b = (uint8_t *)malloc(N);
        k_prod<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaDeviceSynchronize();
        cudaMemcpy(a, dres, N, cudaMemcpyDeviceToHost);
        k_pass<1><<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaDeviceSynchronize();
        cudaMemcpy(b, dres, N, cudaMemcpyDeviceToHost);
        uint32_t diff = 0, primes = 0;
        for (uint32_t i = 0; i < N; i++) { if (a[i] != b[i]) diff++; primes += a[i]; }
        printf("parity K=1 copy vs production: %u/%u primes, %u difference%s\n",
               primes, N, diff, diff ? "s  <-- copy is NOT faithful" : "");
        free(a); free(b);
    }

    double t[5] = {0, 0, 0, 0, 0};
    for (int r = 0; r < reps; r++) {
        double x = bench(k_pass<1>, dlo, dhi, dres, N, iters);
        if (!t[1] || x < t[1]) t[1] = x;
        x = bench(k_pass<2>, dlo, dhi, dres, N, iters);
        if (!t[2] || x < t[2]) t[2] = x;
        x = bench(k_pass<4>, dlo, dhi, dres, N, iters);
        if (!t[4] || x < t[4]) t[4] = x;
    }
    double x = bench(k_prod, dlo, dhi, dres, N, iters);
    t[0] = x;

    double tpass = (t[4] - t[2]) / 2.0;          /* slope: seconds per exponent pass */
    double fixed_s = t[1] - tpass;               /* intercept: setup + compare + ... */
    double rf = (double)N / t[0];

    printf("\narm                ms/launch   ns/test\n");
    printf("production         %9.3f   %8.3f\n", t[0] * 1e3, 1e9 * t[0] / N);
    printf("copy K=1           %9.3f   %8.3f\n", t[1] * 1e3, 1e9 * t[1] / N);
    printf("copy K=2           %9.3f   %8.3f\n", t[2] * 1e3, 1e9 * t[2] / N);
    printf("copy K=4           %9.3f   %8.3f\n", t[4] * 1e3, 1e9 * t[4] / N);
    printf("\nslope  = %.3f ns per exponent pass (= %.1f ps per squaring, %d squarings)\n",
           1e9 * tpass / N, 1e12 * tpass / N / nsur, nsur);
    printf("intercept (setup + final normalise/compare + loop overhead) = %.3f ns/test"
           "  (%.1f %% of the test)\n",
           1e9 * fixed_s / N, 100.0 * fixed_s / t[1]);
    printf("exponentiation share = %.1f %% of the test\n",
           100.0 * (t[1] - fixed_s) / t[1]);
    printf("production rate = %.4e tests/s\n", rf);

    cudaFree(dlo); cudaFree(dhi); cudaFree(dres);
    free(hlo); free(hhi);
    return 0;
}
