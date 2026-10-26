/* bench_sq_rate.cu - how fast is ONE ciosModSquare128, and how close is that to
 * the SM's instruction-issue floor?
 *
 * Context: tools/bench_perig_passes.cu showed that 96 % of the production
 * primality test is the exponentiation loop (16.9 ps per squaring, 64 squarings
 * per test) and that setup, comparison and register pressure are all ~4 % or
 * absent.  So the only place a 1.5x per-test win could still come from is the
 * squaring itself.  This harness measures it in ISOLATION (a dependent chain,
 * no exponent bookkeeping, no branches) and is meant to be run under
 * `ncu` to get the exact instruction count per squaring.
 *
 * Build:  nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *             tools/bench_sq_rate.cu -o bin/bench_sq_rate
 * Run:    bin/bench_sq_rate [Nthreads=262144] [iters=20] [reps=3]
 * Profile: ncu --kernel-name k_sq --launch-count 1 \
 *             --metrics smsp__inst_executed.sum,sm__cycles_elapsed.avg \
 *             ./bin/bench_sq_rate 65536 1 1
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "perig.cuh"

#define SQ_PER_THREAD 256

__global__ static void k_sq(const uint64_t *lo, const uint64_t *hi,
                            uint64_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t n_lo = lo[i], n_hi = hi[i];
    uint64_t mmagic = montgomeryInverse64(n_lo);
    uint64_t rlo, rhi;
    ciosConstants128(n_lo, n_hi, &rlo, &rhi);
    for (int k = 0; k < SQ_PER_THREAD; k++)
        ciosModSquare128(&rlo, &rhi, n_lo, n_hi, mmagic);
    res[i] = rlo ^ rhi;
}

/* TWO independent chains per thread: same total work, but the compiler can
   interleave them, so a LATENCY-bound squaring chain gets free ILP.  This is
   the shape a pipeline change would use (two candidates per thread). */
__global__ static void k_sq2(const uint64_t *lo, const uint64_t *hi,
                             uint64_t *res, uint32_t half) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;
    uint32_t j = i + half;
    uint64_t na_lo = lo[i], na_hi = hi[i], nb_lo = lo[j], nb_hi = hi[j];
    uint64_t ma = montgomeryInverse64(na_lo), mb = montgomeryInverse64(nb_lo);
    uint64_t alo, ahi, blo, bhi;
    ciosConstants128(na_lo, na_hi, &alo, &ahi);
    ciosConstants128(nb_lo, nb_hi, &blo, &bhi);
    for (int k = 0; k < SQ_PER_THREAD; k++) {
        ciosModSquare128(&alo, &ahi, na_lo, na_hi, ma);
        ciosModSquare128(&blo, &bhi, nb_lo, nb_hi, mb);
    }
    res[i] = alo ^ ahi;
    res[j] = blo ^ bhi;
}

/* FOUR independent chains per thread */
__global__ static void k_sq4(const uint64_t *lo, const uint64_t *hi,
                             uint64_t *res, uint32_t quarter) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= quarter) return;
    uint64_t nl[4], nh[4], mg[4], rl[4], rh[4];
    for (int q = 0; q < 4; q++) {
        uint32_t j = i + (uint32_t)q * quarter;
        nl[q] = lo[j]; nh[q] = hi[j];
        mg[q] = montgomeryInverse64(nl[q]);
        ciosConstants128(nl[q], nh[q], &rl[q], &rh[q]);
    }
    for (int k = 0; k < SQ_PER_THREAD; k++)
        for (int q = 0; q < 4; q++)
            ciosModSquare128(&rl[q], &rh[q], nl[q], nh[q], mg[q]);
    for (int q = 0; q < 4; q++)
        res[i + (uint32_t)q * quarter] = rl[q] ^ rh[q];
}

int main(int argc, char **argv) {
    uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 18);
    int iters  = (argc > 2) ? atoi(argv[2]) : 20;
    int reps   = (argc > 3) ? atoi(argv[3]) : 3;

    uint64_t *hlo = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hres = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *href = (uint64_t *)malloc((size_t)N * 8);
    if (!hlo || !hhi || !hres || !href) { fprintf(stderr, "alloc failed\n"); return 1; }

    const uint64_t HI = 10;                  /* 2e20 = 10 * 2^64 + lo */
    unsigned __int128 two_e20 = (unsigned __int128)2000000000000000000ULL * 100u;
    uint64_t base_lo = (uint64_t)(two_e20 - ((unsigned __int128)HI << 64));
    for (uint32_t i = 0; i < N; i++) {
        hlo[i] = base_lo + 2ull * i + 1ull;
        hhi[i] = HI;
    }

    uint64_t *dlo, *dhi, *dres;
    cudaMalloc(&dlo, (size_t)N * 8);
    cudaMalloc(&dhi, (size_t)N * 8);
    cudaMalloc(&dres, (size_t)N * 8);
    cudaMemcpy(dlo, hlo, (size_t)N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)N * 8, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s (%d SMs)  N=%u threads x %d squarings = %.3e squarings/launch\n",
           prop.name, prop.multiProcessorCount, N, SQ_PER_THREAD,
           (double)N * SQ_PER_THREAD);

    /* ---- falsification: the device chain must equal the HOST chain ---------- */
    {
        const uint32_t K = 8;
        k_sq<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaDeviceSynchronize();
        cudaMemcpy(hres, dres, N * 8, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (uint32_t i = 0; i < K; i++) {       /* host reference chain */
            uint64_t mmagic = montgomeryInverse64(hlo[i]);
            uint64_t rlo, rhi;
            ciosConstants128(hlo[i], hhi[i], &rlo, &rhi);
            for (int k = 0; k < SQ_PER_THREAD; k++)
                ciosModSquare128(&rlo, &rhi, hlo[i], hhi[i], mmagic);
            href[i] = rlo ^ rhi;
            if (href[i] != hres[i]) bad++;
        }
        printf("chain check (device vs host, %u threads x %d squarings): %s\n",
               K, SQ_PER_THREAD, bad ? "MISMATCH" : "all match");
    }

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a);
        for (int it = 0; it < iters; it++)
            k_sq<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        double t = (double)ms * 1e-3 / iters;
        if (t < best) best = t;
    }

    double sq = (double)N * SQ_PER_THREAD;
    double rate = sq / best;
    printf("\n1 chain/thread : %.3f ms/launch  ->  %.4e squarings/s  (%.1f ps, card time)\n",
           best * 1e3, rate, 1e12 / rate);

    /* ILP arms: same total squarings, half/quarter the threads */
    double best2 = 1e9, best4 = 1e9;
    for (int r = 0; r < reps; r++) {
        cudaEventRecord(a);
        for (int it = 0; it < iters; it++)
            k_sq2<<<(N / 2 + 255) / 256, 256>>>(dlo, dhi, dres, N / 2);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        double t = (double)ms * 1e-3 / iters;
        if (t < best2) best2 = t;
        cudaEventRecord(a);
        for (int it = 0; it < iters; it++)
            k_sq4<<<(N / 4 + 255) / 256, 256>>>(dlo, dhi, dres, N / 4);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
        t = (double)ms * 1e-3 / iters;
        if (t < best4) best4 = t;
    }
    double rate2 = sq / best2, rate4 = sq / best4;
    printf("2 chains/thread: %.3f ms/launch  ->  %.4e squarings/s  (%.1f ps)  = %.2fx\n",
           best2 * 1e3, rate2, 1e12 / rate2, rate2 / rate);
    printf("4 chains/thread: %.3f ms/launch  ->  %.4e squarings/s  (%.1f ps)  = %.2fx\n",
           best4 * 1e3, rate4, 1e12 / rate4, rate4 / rate);
    printf("per SM: %.4e squarings/s   at %.2f GHz (clock reported by CUDA)\n",
           rate / prop.multiProcessorCount, prop.clockRate * 1e-6);
    {
        double per_clk = (rate / prop.multiProcessorCount) / ((double)prop.clockRate * 1e3);
        printf("per-SM squarings per clock = %.3f  ->  issue slots/cycle/SM used ="
               " %.1f if one squaring is 15 instructions, %.1f if 25\n",
               per_clk, per_clk * 15.0, per_clk * 25.0);
    }

    cudaFree(dlo); cudaFree(dhi); cudaFree(dres);
    free(hlo); free(hhi); free(hres); free(href);
    return 0;
}
