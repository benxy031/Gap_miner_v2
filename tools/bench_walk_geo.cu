/*
 * bench_walk_geo.cu - is the walk engine limited by its ARITHMETIC or by its
 * OCCUPANCY at the campaign geometry?
 *
 * WHY: at 1e32 / --walk-primes 150000 / --walk-batch 96 the walk stage costs
 * 2.92 s per 1e12 integers for 1.64e9 tests = 5.62e8 tests/s (= 1.78 ns/test).
 * The production geometry is P0_WK_GRID=46 P0_WK_BLOCK=512, i.e. ONE CTA per SM
 * and 512 threads/SM = 33 % occupancy, chosen by a CO-RUN sweep (where more
 * walk CTAs steal SM issue slots from the mark kernel).  A flat standalone
 * bench of the same test reports ~8.9e8 tests/s at 96 bits.  So the question
 * is whether the walk's 5.62e8 is its arithmetic ceiling at that occupancy or
 * whether occupancy itself is the limit.
 *
 * This probe runs the PRODUCTION test (perig ciosFermatTest128_hi, the same
 * one w_test_mode selects above 2^75) on odd candidates in the real campaign
 * band (1e32, so ~107 bits) and sweeps the launch geometry:
 *     blocks x threads = 46x512, 46x1024, 92x512, 92x1024, 138x512, 184x512
 * Each thread runs a grid-stride loop over the candidate array, so the rate is
 * a throughput at that occupancy, not a launch-overhead artefact.
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools tools/bench_walk_geo.cu -o bin/bench_walk_geo
 * Usage: ./bin/bench_walk_geo [n] [reps]
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "perig.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

/* One thread per candidate, grid-stride so a chosen grid = a chosen occupancy. */
__global__ void k_perig_geo(const uint64_t *__restrict__ lo, const uint64_t *__restrict__ hi,
                            uint8_t *__restrict__ res, uint32_t n, uint32_t stride) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t step = stride * blockDim.x;
    for (; i < n; i += step) res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

static float bench(int blocks, int threads, const uint64_t *dlo, const uint64_t *dhi,
                   uint8_t *dres, uint32_t n, int reps) {
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, 0));
        k_perig_geo<<<blocks, threads>>>(dlo, dhi, dres, n, (uint32_t)blocks);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    return best;
}

int main(int argc, char **argv) {
    uint32_t n    = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 21);
    int reps      = (argc > 2) ? atoi(argv[2]) : 5;

    /* candidates in the campaign band: n = 1e32 + k, odd, spread over the
       low 64 bits so consecutive threads test unrelated values */
    std::vector<uint64_t> hlo(n), hhi(n);
    const unsigned __int128 base = (unsigned __int128)10000000000000000ULL * 10000000000000000ULL;
    uint64_t blo = (uint64_t)base, bhi = (uint64_t)(base >> 64);
    uint64_t s = 88172645463325252ull;
    for (uint32_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        uint64_t lo = blo + (s & 0xffffffffffffffull) * 2ull;   /* even stride, odd after |1 */
        hhi[i] = bhi + ((lo < blo) ? 1ull : 0ull);
        hlo[i] = lo | 1ull;
    }
    uint64_t *dlo, *dhi; uint8_t *dres;
    CK(cudaMalloc(&dlo, (size_t)n * 8)); CK(cudaMalloc(&dhi, (size_t)n * 8)); CK(cudaMalloc(&dres, n));
    CK(cudaMemcpy(dlo, hlo.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dhi, hhi.data(), (size_t)n * 8, cudaMemcpyHostToDevice));

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    double sms = p.multiProcessorCount, clk = p.clockRate * 1e3;

    /* verdict sanity: count probable primes so a broken band is visible */
    k_perig_geo<<<46, 512>>>(dlo, dhi, dres, n, 46);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> h(n);
    CK(cudaMemcpy(h.data(), dres, n, cudaMemcpyDeviceToHost));
    uint64_t np = 0; for (uint32_t i = 0; i < n; i++) np += h[i];
    printf("band 1e32 (~107 bit), N=%u, probable primes=%llu (%.3f%%)\n",
           n, (unsigned long long)np, 100.0 * np / n);
    printf("%-12s %10s %14s %14s %10s\n", "geometry", "ms", "tests/s", "ns/test", "cyc/SM");

    struct { int b, t; } geo[] = {
        {46, 512}, {46, 1024}, {92, 512}, {92, 1024}, {138, 512}, {184, 512}, {184, 1024},
    };
    for (auto &g : geo) {
        float ms = bench(g.b, g.t, dlo, dhi, dres, n, reps);
        double ts = (double)n / (ms / 1e3);
        char nm[24]; snprintf(nm, sizeof nm, "%dx%d", g.b, g.t);
        printf("%-12s %10.3f %14.3e %14.3f %10.2f\n",
               nm, ms, ts, ms * 1e6 / n, ts / (sms * clk));
    }
    return 0;
}
