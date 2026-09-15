/*
 * bench_mark.cu -- dev tool: isolated cost model of the GPU bitmap MARK kernel
 * used by the fused CRT pipeline (new_src/gpu/gpu_sieve.cu).
 *
 * Why: the fused path spends ~31-33% of wall in the mark kernel, but three
 * plausible explanations failed to fit the miner-level A/B data:
 *   (1) atomic count -- cutting atomics by ~4.5x made marking SLOWER (-15% end
 *       to end, see docs/GPU_SCREEN_ANALYSIS.md, measured 2026-09-14);
 *   (2) per-thread critical path (one thread per prime walks W/p slots per
 *       row, so p=3 has the longest serial chain);
 *   (3) per-thread setup / residue sweep.
 * The miner cannot separate them (changing one factor moves the bottleneck
 * elsewhere).  This tool varies ONE factor at a time on the real kernel shape
 * (one thread per prime, row loop inside, per-multiple atomicOr into an odd
 * slot bitmap) and reports us/call plus derived op rates.
 *
 * Variants (same marking semantics, different parallelization):
 *   ref       one thread per prime, rows in the inner loop (production shape)
 *   ref_small same shape but only for primes <= W (the only primes that can
 *             mark any slot); isolates the cost of the large-prime threads
 *   nostore   ref shape with the atomicOr replaced by a dummy accumulate
 *             (isolates loop/warp cost from bitmap write traffic)
 *   split     one work item per (small prime, row, chunk): identical marking
 *             ops, but the serial chain per thread is chunk_slots/p instead
 *             of rows*W/p
 *   real      the PRODUCTION API (gpu_sieve_mark_rows_from_base) driven with
 *             the same geometry -- the fidelity anchor
 *
 * Build:  make bin/bench_mark WITH_CUDA=1     (nvcc, CUDA_ARCH=sm_86)
 * Usage:  ./bin/bench_mark [--primes N] [--rows R] [--window W] [--iters I]
 *                          [--chunks C] [--no-real]
 *
 * This is a MICROBENCHMARK: it does not verify sieve results, it only times
 * the marking kernels.  Correctness of any production change must still be
 * proven by tests/test_gpu_sieve.c (bit-exact CPU vs GPU parity) and by
 * MINING_JUMP2_VERIFY end-to-end.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>

/* Real production implementation, included so the bench can (a) drive the
   public API as a fidelity anchor and (b) never drift from the shipped
   kernel shape. */
#include "gpu_sieve.cu"

#define TPB 128U

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* Mark every slot q = pos + k*p < W in bitmap row wb (odd-slot space).
   Same body as the production helper gpu_sieve_mark_progression. */
__device__ static __forceinline__ void mark_per_multiple(uint64_t *wb,
                                                         uint64_t W,
                                                         uint64_t p,
                                                         uint64_t pos) {
    for (; pos < W; pos += p) {
        atomicOr((unsigned long long *)&wb[pos >> 6],
                 (unsigned long long)(1ULL << (pos & 63U)));
    }
}

/* Same loop, no bitmap traffic: the last position is accumulated into a dummy
   slot so the compiler cannot delete the loop.  Isolates issue/setup cost. */
__device__ static __forceinline__ void mark_per_multiple_nowrite(
    uint64_t *sink, uint64_t W, uint64_t p, uint64_t pos) {
    uint64_t acc = 0;
    for (; pos < W; pos += p) {
        acc += pos & 63U;
    }
    if (acc == (uint64_t)-1) atomicOr((unsigned long long *)sink, 1ULL);
}

/* ref: mirrors gpu_sieve_rows_mark_dense -- one thread per prime, rows walked
   in the inner loop with an incremental slot decrement between rows. */
__global__ static void k_ref(uint64_t *arena, uint64_t stride_words, uint64_t W,
                             const uint64_t *primes, const uint64_t *pos0,
                             uint64_t n, uint32_t rows, int write) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                   (uint64_t)threadIdx.x;
    if (idx >= n) return;
    uint64_t p = primes[idx];
    if (p < 3U) return;
    uint64_t pos = pos0[idx] % p;
    uint64_t dec = (p >> 2) + 1U;   /* plausible per-row drop, cost-neutral */
    if (dec >= p) dec -= p;
    uint64_t *wb = arena;
    for (uint32_t m = 0; m < rows; m++, wb += stride_words) {
        if (write) {
            mark_per_multiple(wb, W, p, pos);
        } else {
            mark_per_multiple_nowrite(arena, W, p, pos);
        }
        pos = (pos >= dec) ? pos - dec : pos - dec + p;
    }
}

/* split: work item per (small prime, row, chunk).  Each (prime,row) lattice is
   partitioned into contiguous chunk ranges, so the total marking op count is
   identical to ref_small -- only the parallelization changes.  Only primes
   p <= W are chunked: the 99.94% of the table with p > W cannot mark a slot in
   one row, so giving them work items only buys launch overhead. */
__global__ static void k_split(uint64_t *arena, uint64_t stride_words,
                               uint64_t W, const uint64_t *primes,
                               const uint64_t *pos0, uint64_t n_small,
                               uint32_t rows, uint32_t chunks,
                               uint64_t chunk_slots, uint64_t dec) {
    uint64_t item = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                    (uint64_t)threadIdx.x;
    uint64_t total = n_small * (uint64_t)rows * (uint64_t)chunks;
    (void)dec;
    if (item >= total) return;

    uint32_t c = (uint32_t)(item % chunks);
    uint64_t rest = item / chunks;
    uint32_t m = (uint32_t)(rest % rows);
    uint64_t pi = rest / rows;

    uint64_t p = primes[pi];
    if (p < 3U) return;

    /* Same per-prime per-row decrement as k_ref. */
    uint64_t dec_p = (p >> 2) + 1U;
    if (dec_p >= p) dec_p -= p;

    uint64_t begin = (uint64_t)c * chunk_slots;
    if (begin >= W) return;
    uint64_t end = begin + chunk_slots;
    if (end > W) end = W;

    /* First marked slot >= begin for this (prime,row) lattice.  The per-row
       residue is arbitrary here (timing only), but it must be < p. */
    /* Same per-row lattice as ref: pos_m = pos0 - m*dec_p (mod p). */
    uint64_t rem = (pos0[pi] + p - ((uint64_t)m * dec_p) % p) % p;
    if (rem >= p) rem -= p;
    /* First marked slot >= begin on this row's lattice.  NOTE: when begin is
       below the row's residue the first mark IS rem (not rem+p) -- getting
       this branch wrong silently drops ~2% of the marks while looking fast. */
    uint64_t first;
    if (begin <= rem) {
        first = rem;
    } else {
        uint64_t off = begin - rem;
        uint64_t k = (off + p - 1U) / p;
        first = rem + k * p;
    }
    if (first >= W) return;

    uint64_t *wb = arena + (size_t)m * stride_words;
    for (uint64_t q = first; q < end; q += p) {
        atomicOr((unsigned long long *)&wb[q >> 6],
                 (unsigned long long)(1ULL << (q & 63U)));
    }
}

/* ---- host helpers ------------------------------------------------------- */

static std::vector<uint64_t> first_primes(size_t n) {
    /* Upper bound for the n-th prime (n >= 6). */
    double dn = (double)n;
    size_t limit = (size_t)(dn * (std::log(dn) + std::log(std::log(dn)))) + 64;
    if (limit < 1024) limit = 1024;
    std::vector<uint8_t> comp(limit + 1, 0);
    for (size_t i = 2; i * i <= limit; i++) {
        if (!comp[i]) {
            for (size_t j = i * i; j <= limit; j += i) comp[j] = 1;
        }
    }
    std::vector<uint64_t> out;
    out.reserve(n);
    for (size_t i = 2; i <= limit && out.size() < n; i++) {
        if (!comp[i]) out.push_back((uint64_t)i);
    }
    return out;
}

static int parse_u64(const char *s, uint64_t *out) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return 0;
    *out = (uint64_t)v;
    return 1;
}

/* ---- launch thunks (timed with CUDA events, no host round trip inside) --- */

struct RefArgs {
    uint64_t *arena; uint64_t stride; uint64_t W;
    const uint64_t *primes; const uint64_t *pos0;
    uint64_t n; uint32_t rows; uint64_t blocks; int write;
};

static void launch_ref(void *a) {
    RefArgs *r = (RefArgs *)a;
    k_ref<<<(unsigned)r->blocks, TPB>>>(r->arena, r->stride, r->W, r->primes,
                                        r->pos0, r->n, r->rows, r->write);
}

struct SplitArgs {
    uint64_t *arena; uint64_t stride; uint64_t W;
    const uint64_t *primes; const uint64_t *pos0;
    uint64_t n_small; uint32_t rows; uint32_t chunks; uint64_t chunk_slots;
    uint64_t dec; uint64_t blocks;
};

static void launch_split(void *a) {
    SplitArgs *s = (SplitArgs *)a;
    k_split<<<(unsigned)s->blocks, TPB>>>(s->arena, s->stride, s->W, s->primes,
                                          s->pos0, s->n_small, s->rows,
                                          s->chunks, s->chunk_slots, s->dec);
}

/* Run one variant on a ZEROED arena and hash the resulting bitmap.  Used to
   prove that a fast variant marks exactly the same slots as ref_small: a
   "fast" variant that marks nothing would otherwise look like a huge win. */
static void run_and_hash(const char *label, void (*launch)(void *), void *arg,
                         uint64_t *d_arena, size_t arena_words,
                         std::vector<uint64_t> *host_out) {
    CUDA_CHECK(cudaMemset(d_arena, 0, arena_words * sizeof(uint64_t)));
    launch(arg);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_out->data(), d_arena,
                          arena_words * sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    uint64_t bits = 0;
    for (size_t i = 0; i < arena_words; i++) {
        bits += (uint64_t)__builtin_popcountll((*host_out)[i]);
    }
    printf("  %-26s marked bits = %llu\n", label, (unsigned long long)bits);
}

static double bench_launch(const char *label, void (*launch)(void *), void *arg,
                           int iters) {
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    launch(arg);   /* warmup */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++) launch(arg);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    double us = (double)ms * 1000.0 / (double)iters;
    printf("  %-26s %9.2f us/call\n", label, us);
    return us;
}

int main(int argc, char **argv) {
    uint64_t n_primes = 2000000;
    uint64_t W = 10175;     /* odd-slot count, shift512 p75 production shape */
    uint32_t rows = 8;      /* rows = (2^shift - nadd0)/P, capped by batch */
    uint32_t chunks = 8;
    int iters = 30;
    int use_real = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--primes") && i + 1 < argc)
            parse_u64(argv[++i], &n_primes);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) {
            uint64_t v = 0;
            parse_u64(argv[++i], &v);
            rows = (uint32_t)v;
        } else if (!strcmp(argv[i], "--window") && i + 1 < argc)
            parse_u64(argv[++i], &W);
        else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) {
            uint64_t v = 0;
            parse_u64(argv[++i], &v);
            chunks = (uint32_t)v;
        } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            uint64_t v = 0;
            parse_u64(argv[++i], &v);
            iters = (int)v;
        } else if (!strcmp(argv[i], "--no-real")) {
            use_real = 0;
        } else {
            fprintf(stderr, "usage: %s [--primes N] [--rows R] [--window W]"
                            " [--chunks C] [--iters I] [--no-real]\n",
                    argv[0]);
            return 2;
        }
    }
    if (chunks < 1) chunks = 1;
    if (rows < 1) rows = 1;

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    CUDA_CHECK(cudaSetDevice(dev));
    printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);

    std::vector<uint64_t> hprimes = first_primes((size_t)n_primes);
    if (hprimes.size() < n_primes) {
        fprintf(stderr, "prime generation short: %zu < %llu\n", hprimes.size(),
                (unsigned long long)n_primes);
        return 1;
    }
    /* Primes that can mark a slot at all: p <= W.  The table is sorted, so
       this is a prefix of the table. */
    uint64_t n_small = 0;
    while (n_small < hprimes.size() && hprimes[n_small] <= W) n_small++;

    std::vector<uint64_t> hpos0(hprimes.size());
    for (size_t i = 0; i < hprimes.size(); i++) {
        /* Pseudo-random per-prime start slot.  It must depend on the INDEX:
           (p * K) % p == 0 would make every thread start at slot 0 and pile
           all atomics on the first word (a 10x bench artifact). */
        uint64_t h = ((uint64_t)i + 1ULL) * 2654435761ULL + 12345ULL;
        hpos0[i] = h % hprimes[i];
    }

    uint64_t words = (W + 63U) / 64U;
    uint64_t stride_words = words;
    size_t arena_words = (size_t)rows * (size_t)stride_words;

    uint64_t *d_primes = NULL, *d_pos0 = NULL, *d_arena = NULL;
    CUDA_CHECK(cudaMalloc(&d_primes, hprimes.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_pos0, hpos0.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_arena, arena_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_primes, hprimes.data(),
                          hprimes.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos0, hpos0.data(),
                          hpos0.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_arena, 0, arena_words * sizeof(uint64_t)));

    uint64_t primes_u64 = (uint64_t)hprimes.size();
    uint64_t blocks_ref_all = (primes_u64 + TPB - 1U) / TPB;
    uint64_t blocks_ref_small = (n_small + TPB - 1U) / TPB;
    uint64_t total_split = n_small * (uint64_t)rows * (uint64_t)chunks;
    uint64_t blocks_split = (total_split + TPB - 1U) / TPB;
    uint64_t chunk_slots = (W + chunks - 1U) / chunks;
    if (chunk_slots == 0) chunk_slots = 1;

    double ops_small = 0.0;
    for (uint64_t i = 0; i < n_small; i++)
        ops_small += (double)rows * (double)W / (double)hprimes[i];
    double ops_all = ops_small;
    for (uint64_t i = n_small; i < primes_u64; i++)
        ops_all += (double)rows * (double)W / (double)hprimes[i];

    printf("geometry: primes=%llu (p<=W: %llu) rows=%u W=%llu words/row=%llu\n",
           (unsigned long long)primes_u64, (unsigned long long)n_small, rows,
           (unsigned long long)W, (unsigned long long)words);
    printf("ops/call: small-prime shape=%.0f, full-table shape=%.0f "
           "(large-prime tail=%.0f)\n",
           ops_small, ops_all, ops_all - ops_small);
    printf("grids: ref_all=%llu, ref_small=%llu, split=%llu blocks "
           "(chunks=%u, chunk_slots=%llu)\n",
           (unsigned long long)blocks_ref_all,
           (unsigned long long)blocks_ref_small,
           (unsigned long long)blocks_split, chunks,
           (unsigned long long)chunk_slots);

    printf("bench kernels (%d iters each; bitmaps NOT cleared between iters):\n",
           iters);
    RefArgs ra_all, ra_small, ra_nw;
    ra_all.arena = d_arena; ra_all.stride = stride_words; ra_all.W = W;
    ra_all.primes = d_primes; ra_all.pos0 = d_pos0; ra_all.n = primes_u64;
    ra_all.rows = rows; ra_all.blocks = blocks_ref_all; ra_all.write = 1;
    ra_small = ra_all;
    ra_small.n = n_small;
    ra_small.blocks = blocks_ref_small;
    ra_nw = ra_all;
    ra_nw.write = 0;

    SplitArgs sa;
    sa.arena = d_arena; sa.stride = stride_words; sa.W = W;
    sa.primes = d_primes; sa.pos0 = d_pos0; sa.n_small = n_small;
    sa.rows = rows; sa.chunks = chunks; sa.chunk_slots = chunk_slots;
    sa.dec = 5;   /* same constant per-row decrement as ref */
    sa.blocks = blocks_split;

    double ref_all_us = bench_launch("ref (full prime table)", launch_ref,
                                     &ra_all, iters);
    double ref_small_us = bench_launch("ref_small (p<=W only)", launch_ref,
                                       &ra_small, iters);
    double nw_us = bench_launch("nostore (loop cost only)", launch_ref, &ra_nw,
                                iters);
    double sp_us = bench_launch("split (p<=W, chunked)", launch_split, &sa,
                                iters);

    /* Semantic equivalence of the fast variant: a "win" that marks nothing
       would look identical in the timing table, so compare bitmaps. */
    {
        printf("bitmap equivalence check (zeroed arena, one launch each):\n");
        std::vector<uint64_t> hb_a(arena_words, 0), hb_b(arena_words, 0);
        run_and_hash("ref_small", launch_ref, &ra_small, d_arena, arena_words,
                     &hb_a);
        run_and_hash("split", launch_split, &sa, d_arena, arena_words, &hb_b);
        size_t mism = 0;
        uint64_t xorbits = 0;
        for (size_t i = 0; i < arena_words; i++) {
            if (hb_a[i] != hb_b[i]) mism++;
            xorbits += (uint64_t)__builtin_popcountll(hb_a[i] ^ hb_b[i]);
        }
        printf("  words differing = %zu / %zu, differing bits = %llu\n", mism,
               arena_words, (unsigned long long)xorbits);
        if (mism == 0)
            printf("  VERDICT: split marks the IDENTICAL bitmap (bit-exact)\n");
        else
            printf("  VERDICT: MISMATCH -- split is not equivalent; do not "
                   "trust its timing\n");
    }

    /* Semantic equivalence of the fast variant (same slot set as ref_small). */
    {
        printf("bitmap equivalence check (zeroed arena, one launch each):\n");
        std::vector<uint64_t> hb_a(arena_words, 0), hb_b(arena_words, 0);
        run_and_hash("ref_small", launch_ref, &ra_small, d_arena, arena_words,
                     &hb_a);
        run_and_hash("split", launch_split, &sa, d_arena, arena_words, &hb_b);
        size_t mism = 0;
        uint64_t xorbits = 0;
        for (size_t i = 0; i < arena_words; i++) {
            if (hb_a[i] != hb_b[i]) mism++;
            xorbits += (uint64_t)__builtin_popcountll(hb_a[i] ^ hb_b[i]);
        }
        printf("  words differing = %zu / %zu, differing bits = %llu\n", mism,
               arena_words, (unsigned long long)xorbits);
        if (mism == 0)
            printf("  VERDICT: split marks the IDENTICAL bitmap (bit-exact)\n");
        else
            printf("  VERDICT: MISMATCH -- split is NOT equivalent, do not "
                   "trust its timing\n");
    }

    if (use_real) {
        /* Fidelity anchor: drive the REAL production API with the same
           geometry.  GPU_SIEVE_TIMING=1 exposes the row-mark kernel time
           (CUDA events inside gpu_sieve.cu). */
        printf("real production API (gpu_sieve_mark_rows_from_base):\n");
        setenv("GPU_SIEVE_TIMING", "1", 1);
        gpu_sieve_ctx *ctx = gpu_sieve_init(dev, (size_t)primes_u64, W);
        if (!ctx) {
            fprintf(stderr, "gpu_sieve_init failed\n");
            return 1;
        }
        uint64_t base_limbs[12], step_limbs[12];
        for (int i = 0; i < 12; i++) {
            base_limbs[i] = 0x1234567890abcdefULL * (uint64_t)(i + 1);
            step_limbs[i] = 0xfedcba0987654321ULL * (uint64_t)(i + 1);
        }
        base_limbs[0] |= 1ULL;   /* odd base -> first_odd_offset 0 */
        step_limbs[0] |= 1ULL;   /* odd row stride */
        uint64_t first_odd_offset = 0;

        int ok = gpu_sieve_mark_rows_from_base(
            ctx, W, first_odd_offset, base_limbs, step_limbs, 12, rows,
            hprimes.data(), hprimes.data(), (size_t)primes_u64);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!ok) {
            printf("  (real API refused this geometry -- capacity check)\n");
        } else {
            cudaEvent_t e0, e1;
            CUDA_CHECK(cudaEventCreate(&e0));
            CUDA_CHECK(cudaEventCreate(&e1));
            uint64_t mark_before = gpu_sieve_accounted_mark_us(ctx);
            CUDA_CHECK(cudaEventRecord(e0));
            for (int i = 0; i < iters; i++) {
                gpu_sieve_mark_rows_from_base(
                    ctx, W, first_odd_offset, base_limbs, step_limbs, 12, rows,
                    hprimes.data(), hprimes.data(), (size_t)primes_u64);
            }
            CUDA_CHECK(cudaEventRecord(e1));
            CUDA_CHECK(cudaEventSynchronize(e1));
            uint64_t mark_after = gpu_sieve_accounted_mark_us(ctx);
            float ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            double real_us = (double)ms * 1000.0 / (double)iters;
            double mark_us = (double)(mark_after - mark_before) /
                             (double)iters;
            printf("  %-26s %9.2f us/call  (H2D limbs + residues + mark)\n",
                   "real row-mark API", real_us);
            printf("  %-26s %9.2f us/call  (mark kernel only)\n",
                   "  of which mark kernel", mark_us);
            printf("  %-26s %9.2f us/call\n", "ref bench copy", ref_all_us);
            printf("  %-26s %5.2fx\n", "real/ref_all ratio",
                   ref_all_us > 0 ? real_us / ref_all_us : 0.0);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    }

    printf("\nsummary (%d iters, W=%llu, rows=%u):\n", iters,
           (unsigned long long)W, rows);
    printf("  marking ops (p<=W)       : %.0f\n", ops_small);
    printf("  ref_small vs ref_all     : %8.2f vs %8.2f us  (%.0f%% -- cost of "
           "the large-prime threads)\n",
           ref_small_us, ref_all_us,
           ref_all_us > 0 ? 100.0 * ref_small_us / ref_all_us : 0.0);
    printf("  nostore                  : %8.2f us  (%.0f%% of ref_all -- the "
           "loop/warp cost)\n", nw_us,
           ref_all_us > 0 ? 100.0 * nw_us / ref_all_us : 0.0);
    printf("  split                    : %8.2f us  (%.2fx vs ref_small)\n",
           sp_us, sp_us > 0 ? ref_small_us / sp_us : 0.0);
    printf("  ns per marking op (ref_small): %.2f\n",
           ops_small > 0 ? ref_small_us * 1000.0 / ops_small : 0.0);

    cudaFree(d_primes);
    cudaFree(d_pos0);
    cudaFree(d_arena);
    (void)primes_u64;
    return 0;
}
