/*
 * bench_p0sieve.cu - probe: can the GPU beat the CPU at bitmap MARKING for
 * the Phase-0 scanner geometry?
 *
 * Why: the live scanner (tools/phase0_scan_gpu.cu) now spends ~53% of its
 * wall in CPU marking (wheel tile + bucket marking of primes 17..P; measured
 * ~4.3e9 marks/s aggregate at P=3e4 on an i3-10100).  A GPU marker only helps
 * if it beats that INCLUDING the per-segment cycle (memset + kernel + D2H of
 * the 64 KB bitmap, because the scanner's survivor walk stays on the CPU).
 *
 * Geometry (identical to the scanner): segment = 2^20 integers = 2^19
 * odd-value SLOTS (one bit per slot); primes 3..P; slot s of a segment with
 * first odd value v0 holds value v0 + 2s.
 *
 * Work items: one item per (prime, 64-hit chunk) so every thread's serial
 * chain is bounded (the gapminer mark-kernel lesson: the naive one-thread-per-
 * prime shape is straggler-warp bound, 25-38x slower; chunked items fixed it).
 * Prime p has hits_max = half/p + 2 marks per segment; >64 hits are split.
 *
 * Kernel per item: p; v0 mod p (base split in two u64 words + host-precomputed
 * 2^64 mod p); first hit slot s0 = (-v0 mod p) * inv2 mod p; then mark slots
 * s0 + k*p, k in [k0, k0+64), while s < half.  Fire-and-forget atomicOr into
 * the segment bitmap (bit i = "value divisible by a prime <= P").
 *
 * Verification: --verify K downloads K segment bitmaps and compares them
 * bit-exactly against a naive CPU marker over the same primes (full parity
 * check of the item scheme, including the chunk boundaries).
 *
 * Build: make bin/bench_p0sieve WITH_CUDA=1
 * Usage: ./bin/bench_p0sieve [--primes P] [--segs N] [--verify K] [--tpb T]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include "p0_types.h"
#ifndef PHASE0_KERNEL_DLL
#include "p0_mark.cuh"
#else
/* Windows kernel-DLL split (README_WINDOWS.md): kernels are in phase0gpu.dll */
#include "phase0gpu_api.h"
#endif

#define SEG_BITS 20
#define SEG      (1u << SEG_BITS)          /* integers per segment */
#define HALF     (SEG >> 1)                /* odd-value slots */
#define BM_WORDS ((HALF + 63) >> 6)        /* bitmap words per segment */
#define BM_BYTES ((size_t)BM_WORDS * 8)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* item: prime index + first hit index within the segment (P0Item in p0_mark.cuh) */

/* naive CPU marker for the verify pass: no wheel, no buckets */
static void naive_mark(uint64_t *bm, uint64_t v0,
                       const std::vector<uint32_t> &primes) {
    memset(bm, 0, BM_BYTES);
    for (uint32_t p : primes) {
        /* first multiple of p that is >= v0 and odd (p odd: adjust parity) */
        uint64_t m = (v0 / p) * p;
        if (m < v0) m += p;
        if (((v0 ^ m) & 1ull) != 0) m += p;    /* need same parity as v0 */
        uint64_t s = (m - v0) >> 1;
        for (; s < HALF; s += p) bm[s >> 6] |= 1ULL << (s & 63u);
    }
}

static int check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[bench_p0sieve] %s: %s\n", what, cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    uint32_t P = 30000u;
    int segs = 2000, verify = 1;
    int tpb = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--primes") && i + 1 < argc) P = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--segs") && i + 1 < argc) segs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verify") && i + 1 < argc) verify = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tpb") && i + 1 < argc) tpb = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--primes P] [--segs N] [--verify K] [--tpb T]\n", argv[0]); return 2; }
    }
    if (P < 100 || segs < 1) return 2;

    /* ---- primes 3..P (odd table) ---- */
    std::vector<uint32_t> primes;
    {
        std::vector<uint8_t> comp(P + 1, 0);
        for (uint64_t x = 2; x * x <= P; x++)
            if (!comp[x]) for (uint64_t m = x * x; m <= P; m += x) comp[m] = 1;
        for (uint32_t x = 3; x <= P; x += 2) if (!comp[x]) primes.push_back(x);
    }
    printf("[bench_p0sieve] primes 3..%u: %zu\n", P, primes.size());

    /* ---- item table: chunks of <= P0_CHUNK hits ---- */
    std::vector<P0Item> items;
    for (size_t pi = 0; pi < primes.size(); pi++) {
        uint64_t hits = (uint64_t)HALF / primes[pi] + 2;
        uint32_t chunks = (uint32_t)((hits + P0_CHUNK - 1) / P0_CHUNK);
        for (uint32_t c = 0; c < chunks; c++) items.push_back(P0Item{(uint32_t)pi, c * P0_CHUNK});
    }
    printf("[bench_p0sieve] items per segment: %zu (max %u marks each)\n",
           items.size(), P0_CHUNK);

    /* ---- host copies ---- */
    std::vector<uint64_t> h_primes(primes.size()), h_r64(primes.size()), h_invp(primes.size());
    for (size_t i = 0; i < primes.size(); i++) {
        h_primes[i] = primes[i];
        h_r64[i] = ((__uint128_t)1 << 64) % primes[i];
        h_invp[i] = ~0ull / primes[i];
    }

    P0Item *d_items; uint64_t *d_primes, *d_r64, *d_invp;
    uint64_t *d_bm;
    CUDA_CHECK(cudaMalloc(&d_items, items.size() * sizeof(P0Item)));
    CUDA_CHECK(cudaMalloc(&d_primes, h_primes.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_r64, h_r64.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_invp, h_invp.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bm, BM_BYTES));
    CUDA_CHECK(cudaMemcpy(d_items, items.data(), items.size() * sizeof(P0Item), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_primes, h_primes.data(), h_primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_r64, h_r64.data(), h_r64.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invp, h_invp.data(), h_invp.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[bench_p0sieve] device: %s (%d SMs), tpb=%d\n", prop.name, prop.multiProcessorCount, tpb);

    /* ---- verify K segment bitmaps against the naive CPU marker ---- */
    if (verify > 0) {
        std::vector<uint64_t> h_gpu(BM_WORDS), h_cpu(BM_WORDS);
        uint64_t v0 = 200000000000000000001ull;   /* 2e20+1, odd */
        int bad = 0;
        for (int s = 0; s < verify; s++) {
            uint64_t base = v0 + (uint64_t)s * SEG;
            uint64_t base_lo = base, base_hi = 0;   /* < 2^64 for the probe */
            CUDA_CHECK(cudaMemset(d_bm, 0, BM_BYTES));
#ifdef PHASE0_KERNEL_DLL
            if (p0gpu_p0_mark(d_items, (uint32_t)items.size(), d_primes, d_r64,
                              d_invp, HALF, base_lo, base_hi,
                              (uint32_t)(base_lo % 30u), d_bm,
                              (uint32_t)((items.size() + tpb - 1) / tpb), tpb,
                              NULL) != 0)
                return 1;
#else
            p0_mark_kernel<<<(uint32_t)((items.size() + tpb - 1) / tpb), tpb>>>(
                d_items, (uint32_t)items.size(), d_primes, d_r64, d_invp, HALF,
                base_lo, base_hi, p0_vis_mask30((uint32_t)(base_lo % 30u)), d_bm);
#endif
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_bm, BM_BYTES, cudaMemcpyDeviceToHost));
            naive_mark(h_cpu.data(), base, primes);
            size_t diff = 0;
            for (size_t w = 0; w < BM_WORDS; w++) {
                uint64_t x = h_gpu[w] ^ h_cpu[w];
                if (x) diff += __builtin_popcountll(x);
            }
            if (diff) { fprintf(stderr, "[bench_p0sieve] VERIFY MISMATCH seg %d: %zu bits differ\n", s, diff); bad++; }
        }
        printf("[bench_p0sieve] verify: %d segment(s) %s\n", verify,
               bad ? "FAILED" : "bit-exact vs naive CPU marker");
        if (bad) return 1;
    }

    /* ---- timed loop: per segment = memset + kernel + D2H (scanner cycle) ---- */
    {
        std::vector<uint64_t> h_bm(BM_WORDS);
        uint64_t base0 = 200000000000000000001ull;
        const uint32_t grid = (uint32_t)((items.size() + tpb - 1) / tpb);

        /* warmup */
        CUDA_CHECK(cudaMemset(d_bm, 0, BM_BYTES));
#ifdef PHASE0_KERNEL_DLL
        if (p0gpu_p0_mark(d_items, (uint32_t)items.size(), d_primes, d_r64,
                          d_invp, HALF, base0, 0,
                          (uint32_t)(base0 % 30u), d_bm, grid, tpb, NULL) != 0)
            return 1;
#else
        p0_mark_kernel<<<grid, tpb>>>(d_items, (uint32_t)items.size(), d_primes, d_r64, d_invp, HALF, base0, 0, p0_vis_mask30((uint32_t)(base0 % 30u)), d_bm);
#endif
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int s = 0; s < segs; s++) {
            uint64_t base = base0 + (uint64_t)s * SEG;
            CUDA_CHECK(cudaMemset(d_bm, 0, BM_BYTES));
#ifdef PHASE0_KERNEL_DLL
            if (p0gpu_p0_mark(d_items, (uint32_t)items.size(), d_primes, d_r64,
                              d_invp, HALF, base, 0, (uint32_t)(base % 30u),
                              d_bm, grid, tpb, NULL) != 0)
                return 1;
#else
            p0_mark_kernel<<<grid, tpb>>>(d_items, (uint32_t)items.size(), d_primes, d_r64, d_invp, HALF, base, 0, p0_vis_mask30((uint32_t)(base % 30u)), d_bm);
#endif
            CUDA_CHECK(cudaMemcpy(h_bm.data(), d_bm, BM_BYTES, cudaMemcpyDeviceToHost));
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

        double marks = 0;
        for (size_t pi = 0; pi < primes.size(); pi++)
            marks += (double)HALF / primes[pi] + 1.0;      /* +1: first-hit each */
        double total_marks = marks * segs;
        double secs = ms / 1000.0;
        printf("[bench_p0sieve] %d segments (%.3e ints) in %.3f s\n",
               segs, (double)segs * SEG, secs);
        printf("[bench_p0sieve] %.3e marks/s   %.3e ints/s   %.1f us/segment\n",
               total_marks / secs, (double)segs * SEG / secs, ms * 1000.0 / segs);
        printf("[bench_p0sieve] projection: 1e14 ints = %.2f h of GPU marking "
               "(vs CPU ~%.2f h at the scanner's measured 1.45 s per 1.25e9 ints/thread)\n",
               (1e14 / ((double)segs * SEG / secs)) / 3600.0,
               (1e14 / 1.25e9 * 1.45 / 8.0) / 3600.0);
    }

    cudaFree(d_items); cudaFree(d_primes); cudaFree(d_r64); cudaFree(d_invp); cudaFree(d_bm);
    return 0;
}
