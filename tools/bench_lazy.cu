/* bench_lazy.cu - gate + A/B for OUR lazy-Barrett test (tools/p0_test_lazy.cuh)
 * against the production perig test, on the same candidate stream.
 *
 * GATE (must pass before any speed number means anything):
 *   bit-exact verdict parity vs perig over >= 1e6 candidates in the band
 *   n = 10*2^64 + low (the band both implementations are specialised for).
 * Then: tests/s for both arms.
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *            tools/bench_lazy.cu -o bin/bench_lazy
 * Run:   bin/bench_lazy [N=1048576] [iters=50] [reps=3]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "p0_test_lazy.cuh"   /* ours */
#include "perig.cuh"          /* production */

__global__ static void k_lazy(const uint64_t *lo, uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = p0lazy::test<10u>(lo[i]) ? 1 : 0;
}

__global__ static void k_perig(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

__global__ static void k_setup(const uint64_t *lo, uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const p0lazy::U96 nn{{(uint32_t)lo[i], (uint32_t)(lo[i] >> 32), 10u}};
    p0lazy::U96 r{};
    const p0lazy::U128 u = p0lazy::reciprocal192(nn, &r);
    res[i] = (uint8_t)((u.w[0] ^ u.w[1] ^ u.w[2] ^ u.w[3] ^ r.w[0]) & 1u);
}

static double bench(int which, const uint64_t *dlo, const uint64_t *dhi,
                    uint8_t *dres, uint32_t n, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    if (which == 0)      k_lazy<<<(n + 255) / 256, 256>>>(dlo, dres, n);
    else if (which == 2) k_setup<<<(n + 255) / 256, 256>>>(dlo, dres, n);
    else                 k_perig<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) {
        if (which == 0)      k_lazy<<<(n + 255) / 256, 256>>>(dlo, dres, n);
        else if (which == 2) k_setup<<<(n + 255) / 256, 256>>>(dlo, dres, n);
        else                 k_perig<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms * 1e-3 / (double)iters;
}

int main(int argc, char **argv) {
    uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    int iters = (argc > 2) ? atoi(argv[2]) : 50;
    int reps  = (argc > 3) ? atoi(argv[3]) : 3;

    uint64_t *hlo = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)N * 8);
    if (!hlo || !hhi) { fprintf(stderr, "alloc failed\n"); return 1; }
    const uint64_t HI = 10;
    unsigned __int128 two_e20 = (unsigned __int128)2000000000000000000ULL * 100u;
    uint64_t base_lo = (uint64_t)(two_e20 - ((unsigned __int128)HI << 64));
    for (uint32_t i = 0; i < N; i++) { hlo[i] = base_lo + 2ull * i + 1ull; hhi[i] = HI; }

    uint64_t *dlo, *dhi; uint8_t *dres;
    cudaMalloc(&dlo, (size_t)N * 8); cudaMalloc(&dhi, (size_t)N * 8); cudaMalloc(&dres, N);
    cudaMemcpy(dlo, hlo, (size_t)N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)N * 8, cudaMemcpyHostToDevice);
    uint8_t *h1 = (uint8_t *)malloc(N), *h2 = (uint8_t *)malloc(N);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (%d SMs)  N=%u  band=2e20 (hi=%llu)\n",
           prop.name, prop.multiProcessorCount, N, (unsigned long long)HI);

    /* ---- GATE: verdict parity ------------------------------------------- */
    k_lazy<<<(N + 255) / 256, 256>>>(dlo, dres, N);
    cudaError_t e1 = cudaDeviceSynchronize();
    if (e1 != cudaSuccess) printf("k_lazy launch/sync error: %s\n", cudaGetErrorString(e1));
    cudaMemcpy(h1, dres, N, cudaMemcpyDeviceToHost);
    /* host path of the SAME function: isolates device codegen from logic */
    {
        int host_only = 0, dev_only = 0, show = 0;
        for (uint32_t i = 0; i < N; i++) {
            int hall = p0lazy::test<10u>(hlo[i]) ? 1 : 0;
            if (hall != h1[i]) {
                if (hall) host_only++; else dev_only++;
                if (show++ < 5)
                    printf("  host/device mismatch low=%llu host=%d device=%d\n",
                           (unsigned long long)hlo[i], hall, (int)h1[i]);
            }
        }
        printf("host-vs-device of p0lazy::test: %d mismatch(es) (host-only %d, device-only %d)\n",
               host_only + dev_only, host_only, dev_only);
    }
    k_perig<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N); cudaDeviceSynchronize();
    cudaMemcpy(h2, dres, N, cudaMemcpyDeviceToHost);
    uint32_t primes = 0, diff = 0, lazyOnly = 0, perigOnly = 0;
    for (uint32_t i = 0; i < N; i++) {
        primes += h2[i];
        if (h1[i] != h2[i]) { diff++; if (h1[i]) lazyOnly++; else perigOnly++; }
    }
    printf("GATE lazy vs perig: %u/%u primes, %u difference(s) (lazy-only %u, perig-only %u)%s\n",
           primes, N, diff, lazyOnly, perigOnly, diff ? "   <-- FAIL" : "   PASS");

    double bl = 1e9, bp = 1e9, bs = 1e9;
    for (int r = 0; r < reps; r++) {
        double t = bench(0, dlo, dhi, dres, N, iters); if (t < bl) bl = t;
        t = bench(1, dlo, dhi, dres, N, iters);        if (t < bp) bp = t;
        t = bench(2, dlo, dhi, dres, N, iters);        if (t < bs) bs = t;
    }
    double rl = (double)N / bl, rp = (double)N / bp, rs = (double)N / bs;
    printf("\narm        ms/launch   tests/s      ns/test\n");
    printf("lazy(our)  %9.3f   %.4e   %.3f\n", bl * 1e3, rl, 1e9 / rl);
    printf("perig      %9.3f   %.4e   %.3f\n", bp * 1e3, rp, 1e9 / rp);
    printf("setup only %9.3f   %.4e   %.3f   (%.1f%% of lazy)\n",
           bs * 1e3, rs, 1e9 / rs, 100.0 * bs / bl);
    printf("\nlazy / perig = %.3fx   (lazy minus setup = %.3fx)\n",
           rl / rp, (double)N / (bl - bs) / rp);

    cudaFree(dlo); cudaFree(dhi); cudaFree(dres);
    free(hlo); free(hhi); free(h1); free(h2);
    return 0;
}
