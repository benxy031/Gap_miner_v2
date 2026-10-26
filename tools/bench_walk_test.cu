/* bench_walk_test.cu - the ONLY A/B that matters for the walk engine: the
   CURRENT production test (perig ciosFermatTest128_hi, base-2 Euler/Jacobi)
   versus the exact 2x64-bit CIOS candidate (mr128_64_base2_lazy_u128), on
   identical ~96-bit odd candidates, on the production card.

   Both return 1 = probably prime / 0 = composite.  Correctness of each has
   been established separately (mr128_bench --validate, perig GMP range tests);
   this harness measures SPEED only, at the walk's real geometry.

   Build:  nvcc -O3 -arch=sm_86 tools/bench_walk_test.cu -o bin/bench_walk_test
   Run:    bin/bench_walk_test [N] [iters]
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "perig.cuh"            /* ciosFermatTest128_hi (current production) */
#include "mr128_64_kernel.cuh"  /* mr128_64_base2_lazy_u128 (candidate)      */

__global__ static void k_perig(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

__global__ static void k_lazy(const uint64_t *lo, const uint64_t *hi,
                              uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = (uint8_t)mr128_64_base2_lazy_u128(lo[i], hi[i]);
}

typedef void (*kfn)(const uint64_t *, const uint64_t *, uint8_t *, uint32_t);

static float bench(kfn k, const uint64_t *dlo, const uint64_t *dhi,
                   uint8_t *dres, uint32_t n, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);   /* warm */
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}

int main(int argc, char **argv) {
    uint32_t n = (argc > 1) ? (uint32_t)strtoul(argv[1], 0, 10) : (1u << 20);
    int iters = (argc > 2) ? atoi(argv[2]) : 8;

    uint64_t *hlo = (uint64_t *)malloc((size_t)n * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)n * 8);
    srand(20261024);
    for (uint32_t i = 0; i < n; i++) {
        /* ~96-bit odd candidates: top limb in [2^31, 2^32), i.e. value in
           [2^95, 2^96) -- exactly the walk's size class. */
        uint64_t a = (uint64_t)rand() * (uint64_t)rand();
        uint64_t b = (uint64_t)rand() * (uint64_t)rand();
        hlo[i] = a | 1u;                     /* odd */
        hhi[i] = (b & 0x7fffffffULL) | 0x40000000ULL;  /* bit 94-95 set */
    }
    uint64_t *dlo, *dhi; uint8_t *dres;
    cudaMalloc(&dlo, (size_t)n * 8);
    cudaMalloc(&dhi, (size_t)n * 8);
    cudaMalloc(&dres, n);
    cudaMemcpy(dlo, hlo, (size_t)n * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)n * 8, cudaMemcpyHostToDevice);

    /* sanity: the two tests must agree on a sample (both are base-2 tests on
       the same candidates; perig is Euler, lazy is strong MR -- they can only
       differ on Euler pseudoprimes, so report the count, don't demand 0). */
    k_perig<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaDeviceSynchronize();
    uint8_t *h1 = (uint8_t *)malloc(n);
    cudaMemcpy(h1, dres, n, cudaMemcpyDeviceToHost);
    k_lazy<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaDeviceSynchronize();
    uint8_t *h2 = (uint8_t *)malloc(n);
    cudaMemcpy(h2, dres, n, cudaMemcpyDeviceToHost);
    uint64_t diff = 0, p1 = 0, p2 = 0;
    for (uint32_t i = 0; i < n; i++) { diff += (h1[i] != h2[i]); p1 += h1[i]; p2 += h2[i]; }
    printf("N=%u  primes: perig=%llu lazy=%llu  disagree=%llu (%.4f%%)\n",
           n, (unsigned long long)p1, (unsigned long long)p2,
           (unsigned long long)diff, 100.0 * diff / n);

    float tp = bench(k_perig, dlo, dhi, dres, n, iters);
    float tl = bench(k_lazy, dlo, dhi, dres, n, iters);
    printf("perig  (current) : %.3f ms  -> %.3e tests/s (%.3f ns)\n",
           tp, n / (tp / 1e3), 1e6 * tp / n);
    printf("lazy   (mr128_64): %.3f ms  -> %.3e tests/s (%.3f ns)\n",
           tl, n / (tl / 1e3), 1e6 * tl / n);
    printf("lazy / perig     : %.3fx  (%s)\n", tp / tl,
           tl < tp ? "lazy is FASTER" : "lazy is SLOWER");
    return 0;
}
