/* bench_lazy4.cu - gate + A/B for the FOUR-limb lazy-Barrett test
 * (tools/p0_test_lazy.cuh, p0lazy::test4) against the production perig test,
 * on the same candidate stream, in the campaign band (n ~ 1e32 = 2^106.3).
 *
 * GATE (must pass before any speed number means anything):
 *   - host and device code of the SAME function agree (isolates codegen),
 *   - verdict parity vs perig over >= 1e6 candidates (perig is exact there).
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *            tools/bench_lazy4.cu -o bin/bench_lazy4
 * Run:   bin/bench_lazy4 [N=1048576] [iters=30] [reps=3]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "p0_test_lazy.cuh"   /* ours: test4 */
#include "perig.cuh"          /* production */

__global__ static void k_lazy4(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = p0lazy::test4(lo[i], hi[i]) ? 1 : 0;
}

__global__ static void k_perig(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

__global__ static void k_setup(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const p0lazy::U128 nn{{(uint32_t)lo[i], (uint32_t)(lo[i] >> 32),
                           (uint32_t)hi[i], (uint32_t)(hi[i] >> 32)}};
    p0lazy::U192 r{};
    const p0lazy::U160 u = p0lazy::reciprocal256(nn, &r);
    res[i] = (uint8_t)((u.w[0] ^ u.w[1] ^ u.w[2] ^ u.w[3] ^ u.w[4] ^ r.w[0]) & 1u);
}

__global__ static void k_perig_setup(const uint64_t *lo, const uint64_t *hi,
                                   uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* perig's once-per-modulus constants: -n^-1 mod 2^64 and 2^128 mod n */
    const uint64_t mm = montgomeryInverse64(lo[i]);
    uint64_t one_lo = 0, one_hi = 0;
    ciosConstants128(lo[i], hi[i], &one_lo, &one_hi);
    res[i] = (uint8_t)((mm ^ one_lo ^ one_hi) & 1u);
}

static double bench(int which, const uint64_t *dlo, const uint64_t *dhi,
                    uint8_t *dres, uint32_t n, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const unsigned blk = (n + 255) / 256;
    if (which == 0)      k_lazy4<<<blk, 256>>>(dlo, dhi, dres, n);
    else if (which == 2) k_setup<<<blk, 256>>>(dlo, dhi, dres, n);
    else if (which == 3) k_perig_setup<<<blk, 256>>>(dlo, dhi, dres, n);
    else                 k_perig<<<blk, 256>>>(dlo, dhi, dres, n);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) {
        if (which == 0)      k_lazy4<<<blk, 256>>>(dlo, dhi, dres, n);
        else if (which == 2) k_setup<<<blk, 256>>>(dlo, dhi, dres, n);
        else if (which == 3) k_perig_setup<<<blk, 256>>>(dlo, dhi, dres, n);
        else                 k_perig<<<blk, 256>>>(dlo, dhi, dres, n);
    }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms * 1e-3 / (double)iters;
}

int main(int argc, char **argv) {
    const uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    const int iters = (argc > 2) ? atoi(argv[2]) : 30;
    const int reps  = (argc > 3) ? atoi(argv[3]) : 3;

    uint64_t *hlo = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)N * 8);
    if (!hlo || !hhi) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* base = 1e32, odd; candidate i = base + 2i (no 128-bit carry in range) */
    const unsigned __int128 base =
        (unsigned __int128)100000000000000000ull * 1000000000000000ull;   /* 1e32 */
    for (uint32_t i = 0; i < N; i++) {
        const unsigned __int128 v = base + (unsigned __int128)(2ull * i) + 1ull;
        hlo[i] = (uint64_t)v;
        hhi[i] = (uint64_t)(v >> 64);
    }

    uint64_t *dlo, *dhi; uint8_t *dres;
    cudaMalloc(&dlo, (size_t)N * 8); cudaMalloc(&dhi, (size_t)N * 8);
    cudaMalloc(&dres, N);
    cudaMemcpy(dlo, hlo, (size_t)N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)N * 8, cudaMemcpyHostToDevice);
    uint8_t *h1 = (uint8_t *)malloc(N), *h2 = (uint8_t *)malloc(N);

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (%d SMs)  N=%u  band=1e32 (2^106.3)\n",
           prop.name, prop.multiProcessorCount, N);

    /* ---- GATE 1: host vs device, same function ------------------------- */
    k_lazy4<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
    cudaError_t e1 = cudaDeviceSynchronize();
    if (e1 != cudaSuccess) printf("k_lazy4 launch/sync error: %s\n", cudaGetErrorString(e1));
    cudaMemcpy(h1, dres, N, cudaMemcpyDeviceToHost);
    {
        int host_only = 0, dev_only = 0, show = 0;
        for (uint32_t i = 0; i < N; i++) {
            const int hall = p0lazy::test4(hlo[i], hhi[i]) ? 1 : 0;
            if (hall != h1[i]) {
                if (hall) host_only++; else dev_only++;
                if (show++ < 5)
                    printf("  host/device mismatch lo=%llu hi=%llu host=%d device=%d\n",
                           (unsigned long long)hlo[i], (unsigned long long)hhi[i],
                           hall, (int)h1[i]);
            }
        }
        printf("host-vs-device of p0lazy::test4: %d mismatch(es) (host-only %d, device-only %d)\n",
               host_only + dev_only, host_only, dev_only);
    }

    /* ---- GATE 2: verdict parity vs perig ------------------------------- */
    k_perig<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N); cudaDeviceSynchronize();
    cudaMemcpy(h2, dres, N, cudaMemcpyDeviceToHost);
    uint32_t primes = 0, diff = 0, lazyOnly = 0, perigOnly = 0;
    for (uint32_t i = 0; i < N; i++) {
        primes += h2[i];
        if (h1[i] != h2[i]) { diff++; if (h1[i]) lazyOnly++; else perigOnly++; }
    }
    printf("GATE lazy4 vs perig: %u/%u primes, %u difference(s) (lazy-only %u, perig-only %u)%s\n",
           primes, N, diff, lazyOnly, perigOnly, diff ? "   <-- FAIL" : "   PASS");

    double bl = 1e9, bp = 1e9, bs = 1e9, bq = 1e9;
    for (int r = 0; r < reps; r++) {
        double t = bench(0, dlo, dhi, dres, N, iters); if (t < bl) bl = t;
        t = bench(1, dlo, dhi, dres, N, iters);        if (t < bp) bp = t;
        t = bench(2, dlo, dhi, dres, N, iters);        if (t < bs) bs = t;
        t = bench(3, dlo, dhi, dres, N, iters);        if (t < bq) bq = t;
    }
    const double rl = (double)N / bl, rp = (double)N / bp, rs = (double)N / bs;
    const double rq = (double)N / bq;
    printf("\narm            ms/launch   tests/s      ns/test\n");
    printf("lazy4(our)     %9.3f   %.4e   %.3f\n", bl * 1e3, rl, 1e9 / rl);
    printf("perig          %9.3f   %.4e   %.3f\n", bp * 1e3, rp, 1e9 / rp);
    printf("ours setup     %9.3f   %.4e   %.3f   (%.1f%% of lazy4)\n",
           bs * 1e3, rs, 1e9 / rs, 100.0 * bs / bl);
    printf("perig setup    %9.3f   %.4e   %.3f   (%.1f%% of perig)\n",
           bq * 1e3, rq, 1e9 / rq, 100.0 * bq / bp);
    printf("\nlazy4 / perig = %.3fx   (ladders only: %.3fx)\n",
           rl / rp, (double)N / (bl - bs) / ((double)N / (bp - bq)));

    cudaFree(dlo); cudaFree(dhi); cudaFree(dres);
    free(hlo); free(hhi); free(h1); free(h2);
    return diff ? 1 : 0;
}
