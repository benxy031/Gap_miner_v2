/* walk_bench.cu - Phase-0 walk engine v1: jump-by-minGap walk over OUR census
   sieve bitmap, with our exact MR as the walk test.

   Pipeline per benchmark range:
     1) host: primes to P, item table (same geometry as the census, segment
        2^24 numbers), upload
     2) GPU: mark every segment of the range (reuses p0_mark_kernel: candidates
        are CLEAR bits; 3..13 unfiltered, 17..P vis-filtered -> the bitmap is
        IDENTICAL to the production census bitmap)
     3) GPU: one walk launch: each thread owns a contiguous slot-slice and walks
        it independently: boot = first MR-prime >= slice start; then repeat
        { jump p+T; scan candidates backward; MR-test each; if we reach p again
        -> gap: scan forward for the upper prime, record; else p = found prime }.
     4) host: write gaps (census .log format), print counters.

   Usage: walk_bench START_DEC N_NUMBERS MIN_GAP [grid] [block] [P]
   Modes: env WALK_NOMARK=1 skips the sieve (walk-only timing on an old bitmap).
   The range must be a multiple of 2^24 numbers.*

   Build: nvcc -O3 -arch=sm_86 -std=c++17 -I/tmp/p0probe \
            -I/home/dejan/Git/gapminer_v2/tools walk_bench.cu -o walk_bench \
            -lcudart -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#include "p0_mark.cuh"
#include "mr68_kernel.cuh"
#include "perig.cuh"

#ifdef DUMMYMR
/* Diagnostic: replace the real MR with a cheap ~50%-true predicate that keeps the
   walk's control flow and test counts similar, to measure the MR share of walk cost. */
__device__ static inline int mr_base2_dummy(uint32_t n[3]) { return ((n[0] >> 5) & 1u) == 0u; }
#define mr_base2(n) mr_base2_dummy(n)
#endif

typedef unsigned __int128 u128;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

#define SEG_BITS 24

typedef struct { uint64_t slot; uint32_t gap; uint32_t pad; } GapRec;

/* ---- walk device code -------------------------------------------------- */
__device__ static inline int64_t w_nextCand(const uint64_t *bm, uint64_t nwords, int64_t pos) {
    if (pos < 0) return -1;
    uint64_t w = (uint64_t)pos >> 6;
    if (w >= nwords) return -1;
    uint64_t word = ~bm[w] & (~0ULL << ((uint32_t)pos & 63u));
    while (!word) {
        if (++w >= nwords) return -1;
        word = ~bm[w];
    }
    return (int64_t)(w << 6) + __ffsll((long long)word) - 1;
}
__device__ static inline int64_t w_prevCand(const uint64_t *bm, int64_t pos) {
    if (pos < 0) return -1;
    int64_t w = pos >> 6;
    uint64_t word = ~bm[w] & (~0ULL >> (63 - ((uint32_t)pos & 63u)));
    while (!word) {
        if (--w < 0) return -1;
        word = ~bm[w];
    }
    return (w << 6) + 63 - __clzll((long long)word);
}
__device__ static inline void w_make_n(uint64_t v0_lo, uint64_t v0_hi, uint64_t slot, uint32_t n[3]) {
    uint64_t lo2 = v0_lo + (slot << 1);
    n[0] = (uint32_t)lo2;
    n[1] = (uint32_t)(lo2 >> 32);
    n[2] = (uint32_t)v0_hi + (uint32_t)(lo2 < v0_lo);
}

/* walk_kernel4: IDENTICAL semantics to walk_kernel, but all slot arithmetic is
   int32 (slots < 2^31 for a 2^30-number block) to cut register pressure and
   raise occupancy. */
__device__ static inline int w32_next(const uint64_t *bm, uint32_t nwords, int pos) {
    if (pos < 0) return -1;
    uint32_t w = (uint32_t)pos >> 6;
    if (w >= nwords) return -1;
    uint64_t word = ~bm[w] & (~0ULL << ((uint32_t)pos & 63u));
    while (!word) {
        if (++w >= nwords) return -1;
        word = ~bm[w];
    }
    return (int)(w << 6) + (int)__ffsll((long long)word) - 1;
}
__device__ static inline int w32_prev(const uint64_t *bm, int pos) {
    if (pos < 0) return -1;
    int w = pos >> 6;
    uint64_t word = ~bm[w] & (~0ULL >> (63 - ((uint32_t)pos & 63u)));
    while (!word) {
        if (--w < 0) return -1;
        word = ~bm[w];
    }
    return (w << 6) + 63 - (int)__clzll((long long)word);
}

__global__ static void walk_kernel4(const uint64_t *__restrict__ bm, uint64_t nwords64,
                                    uint64_t v0_lo, uint64_t v0_hi,
                                    uint32_t minGapSlots, uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    uint32_t nwords = (uint32_t)nwords64;
    uint32_t total = gridDim.x * blockDim.x;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t sliceLen = nwords * 64u / total;
    int lo = (int)(tid * sliceLen);
    int hi = (tid + 1 == total) ? (int)(nwords * 64u) : lo + (int)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    unsigned long long tests = 0, jumps = 0;
    uint32_t n[3];
    int c = w32_next(bm, nwords, lo);
    while (c >= 0) {
        w_make_n(v0_lo, v0_hi, (uint64_t)c, n);
        tests++;
        if (mr_base2(n)) break;
        c = w32_next(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int p = c;
    const int end = (int)(nwords * 64u);
    while (p < hi) {
        int jumpPos = p + (int)minGapSlots;
        if (jumpPos >= end) break;
        jumps++;
        int s = jumpPos - 1;
        int foundGap = 0;
        for (;;) {
            s = w32_prev(bm, s);
            if (s < 0) goto done;
            if (s == p) { foundGap = 1; break; }
            w_make_n(v0_lo, v0_hi, (uint64_t)s, n);
            tests++;
            if (mr_base2(n)) { p = s; break; }
            s--;
        }
        if (foundGap) {
            int q = jumpPos;
            for (;;) {
                q = w32_next(bm, nwords, q);
                if (q < 0) goto done;
                w_make_n(v0_lo, v0_hi, (uint64_t)q, n);
                tests++;
                if (mr_base2(n)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = (uint64_t)p; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    break;
                }
                q++;
            }
        }
    }
done:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

/* walk_kernel5: FLAT (PGS-shape) walk. Identical semantics to walk_kernel, but
   the per-jump inner loop is flattened into ONE free-running loop: each thread
   steps (scan-to-next-candidate, test, jump-if-prime) through its whole slice.
   No per-jump warp-wide stopping point -> no E[max32] geometric divergence tax.
   PGS's findGaps has exactly this shape; expected ~3x over the nested version. */
__global__ static void walk_kernel5(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t v0_lo, uint64_t v0_hi,
                                    uint32_t minGapSlots, uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    uint64_t total = (uint64_t)gridDim.x * blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sliceLen = nwords * 64 / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    const int64_t end = (int64_t)(nwords * 64);
    unsigned long long tests = 0, jumps = 0;
    uint32_t n[3];
    int64_t c = w_nextCand(bm, nwords, lo);
    while (c >= 0) {
        w_make_n(v0_lo, v0_hi, (uint64_t)c, n);
        tests++;
        if (mr_base2(n)) break;
        c = w_nextCand(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;                      /* anchor prime */
    int64_t s = p + (int64_t)minGapSlots - 1;   /* scan cursor (descending) */
    if (s >= end) { atomicAdd(&stats[0], tests); return; }
    for (;;) {
        if (p >= hi) goto done;         /* slice bound (v1's `while (p < hi)`) */
        s = w_prevCand(bm, s);
        if (s < 0) goto done;
        if (s == p) {
            /* no candidate in (p, p+minGap) is prime: GAP from p */
            int64_t q = p + (int64_t)minGapSlots;
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done;
                w_make_n(v0_lo, v0_hi, (uint64_t)q, n);
                tests++;
                if (mr_base2(n)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = (uint64_t)p; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    jumps++;
                    s = p + (int64_t)minGapSlots - 1;
                    if (s >= end) goto done;
                    break;
                }
                q++;
            }
            continue;
        }
        w_make_n(v0_lo, v0_hi, (uint64_t)s, n);
        tests++;
        if (mr_base2(n)) {
            p = s;                      /* new anchor: jump up from here */
            jumps++;
            s = p + (int64_t)minGapSlots - 1;
            if (s >= end) goto done;
        } else {
            s--;                        /* composite: keep scanning down */
        }
    }
done:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

/* walk_kernel6: identical to walk_kernel5 but the primality test is the
   PGS/Perig Euler-Legendre base-2 test (ciosFermatTest128) run on the LOW
   64 bits of the candidate, with the (constant within the work unit) high
   word folded in at compile time via HIGH_64. */
__global__ static void walk_kernel6(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t v0_lo, uint64_t v0_hi,
                                    uint32_t minGapSlots, uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    (void)v0_hi;
    uint64_t total = (uint64_t)gridDim.x * blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sliceLen = nwords * 64 / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    const int64_t end = (int64_t)(nwords * 64);
    unsigned long long tests = 0, jumps = 0;
    int64_t c = w_nextCand(bm, nwords, lo);
    while (c >= 0) {
        tests++;
        if (ciosFermatTest128(v0_lo + 2ULL * (uint64_t)c)) break;
        c = w_nextCand(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;                      /* anchor prime */
    int64_t s = p + (int64_t)minGapSlots - 1;   /* scan cursor (descending) */
    if (s >= end) { atomicAdd(&stats[0], tests); return; }
    for (;;) {
        if (p >= hi) goto done;
        s = w_prevCand(bm, s);
        if (s < 0) goto done;
        if (s == p) {
            int64_t q = p + (int64_t)minGapSlots;
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done;
                tests++;
                if (ciosFermatTest128(v0_lo + 2ULL * (uint64_t)q)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = (uint64_t)p; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    jumps++;
                    s = p + (int64_t)minGapSlots - 1;
                    if (s >= end) goto done;
                    break;
                }
                q++;
            }
            continue;
        }
        tests++;
        if (ciosFermatTest128(v0_lo + 2ULL * (uint64_t)s)) {
            p = s;
            jumps++;
            s = p + (int64_t)minGapSlots - 1;
            if (s >= end) goto done;
        } else {
            s--;
        }
    }
done:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

/* --- class-30 walk (walk_kernel7 copy) --------------------------------- */
__device__ __constant__ uint32_t c30_cls8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
#ifdef WK7_DUMMY
#define WK7_TEST(v) 1
#else
#define WK7_TEST(v) ciosFermatTest128(v)
#endif

__global__ static void walk_kernel7(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t Alo, uint64_t Ahi,
                                    const uint32_t *__restrict__ wtab,
                                    uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    (void)Ahi;
    __shared__ uint32_t scls[8];
        uint64_t total = (uint64_t)gridDim.x * blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nslots = nwords * 64;
    uint64_t sliceLen = nslots / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)nslots : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    const int64_t end = (int64_t)nslots;
    unsigned long long tests = 0, jumps = 0;
    int64_t c = w_nextCand(bm, nwords, lo);
    while (c >= 0) {
        tests++;
        uint64_t v = Alo + 30ull * ((uint64_t)c >> 3) + (uint64_t)c30_cls8[c & 7u];
        if (WK7_TEST(v)) break;
        c = w_nextCand(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;
    int64_t s = p + (int64_t)wtab[p & 7];
    if (s >= end) { atomicAdd(&stats[0], tests); return; }
    for (;;) {
        if (p >= hi) goto done7;
        s = w_prevCand(bm, s);
        if (s < 0) goto done7;
        if (s == p) {
            int64_t q = p + (int64_t)wtab[p & 7];
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done7;
                tests++;
                uint64_t vq = Alo + 30ull * ((uint64_t)q >> 3) + (uint64_t)c30_cls8[q & 7u];
                if (WK7_TEST(vq)) {
                    uint64_t vp = Alo + 30ull * ((uint64_t)p >> 3) + (uint64_t)c30_cls8[p & 7u];
                    uint32_t gap = (uint32_t)(vq - vp);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = vp; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    jumps++;
                    s = p + (int64_t)wtab[p & 7];
                    if (s >= end) goto done7;
                    break;
                }
                q++;
            }
            continue;
        }
        tests++;
        uint64_t vs = Alo + 30ull * ((uint64_t)s >> 3) + (uint64_t)c30_cls8[s & 7u];
        if (WK7_TEST(vs)) {
            p = s;
            jumps++;
            s = p + (int64_t)wtab[p & 7];
            if (s >= end) goto done7;
        } else {
            s--;
        }
    }
done7:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

/* walk_kernel3: warp-cooperative walk. One warp owns a slice; per jump the warp
   tests candidates in flat warp rounds (lane j = j-th candidate below jumpPos,
   descending), so one warp MR pass replaces the serial per-candidate loop.
   Semantics identical to walk_kernel (largest prime < jumpPos becomes the anchor). */
__device__ static inline int64_t wk3_next_prime_up(const uint64_t * __restrict__ bm,
        uint64_t nwords, int64_t from, uint64_t v0_lo, uint64_t v0_hi,
        unsigned long long *tests, int lane) {
    int64_t sc = from;
    while (sc < (int64_t)(nwords * 64)) {
        uint64_t rw = (uint64_t)sc >> 6;
        uint64_t bits = ~bm[rw];
        uint32_t lb = (uint32_t)(sc & 63);
        if (lb) bits &= ~0ULL << lb;
        int64_t cs = -1;
        uint64_t t = bits;
        for (int k = 0; k <= lane; k++) {
            if (!t) break;
            int q = __ffsll((long long)t) - 1;
            if (k == lane) cs = q;
            t &= t - 1;
        }
        int pass = 0;
        if (cs >= 0) {
            uint32_t n[3];
            w_make_n(v0_lo, v0_hi, (uint64_t)((rw << 6) + (uint64_t)cs), n);
            (*tests)++;
            pass = mr_base2(n);
        }
        unsigned m = __ballot_sync(0xffffffffu, pass);
        if (m) {
            int win = __ffs(m) - 1;
            int wcs = __shfl_sync(0xffffffffu, (int)cs, win);
            return (int64_t)(rw << 6) + wcs;
        }
        sc = (int64_t)((rw + 1) << 6);
    }
    return -1;
}

__global__ static void walk_kernel3(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t v0_lo, uint64_t v0_hi,
                                    uint32_t minGapSlots, uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    int lane = (int)(threadIdx.x & 31u);
    uint64_t wid = (uint64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    uint64_t nwarps = (uint64_t)gridDim.x * (blockDim.x >> 5);
    if (nwarps == 0) return;
    uint64_t sliceLen = nwords * 64 / nwarps;
    int64_t lo = (int64_t)(wid * sliceLen);
    int64_t hi = (wid + 1 == nwarps) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    unsigned long long tests = 0, jumps = 0;
    int64_t p = wk3_next_prime_up(bm, nwords, lo, v0_lo, v0_hi, &tests, lane);
    if (p < 0 || p >= hi) { atomicAdd(&stats[0], tests); return; }
    uint64_t pword = (uint64_t)p >> 6;
    while (p < hi) {
        int64_t jumpPos = p + (int64_t)minGapSlots;
        if (jumpPos >= (int64_t)(nwords * 64)) break;
        if (lane == 0) jumps++;
        int foundGap = 0; int64_t pnew = -1;
        int64_t sc = jumpPos;
        while (!foundGap && pnew < 0) {
            uint64_t rw = (uint64_t)(sc - 1) >> 6;
            uint64_t bits = ~bm[rw];
            if (rw == pword) {
                uint32_t pb = (uint32_t)(p & 63);
                bits &= (pb == 63) ? 0ULL : (~0ULL << (pb + 1));
            }
            uint32_t hb = (uint32_t)((sc - 1) & 63);
            if (hb < 63) bits &= (1ULL << (hb + 1)) - 1;
            int64_t cs = -1;
            uint64_t t = bits;
            for (int k = 0; k <= lane; k++) {
                if (!t) break;
                int q = 63 - __clzll((long long)t);
                if (k == lane) cs = q;
                t &= ~(1ULL << q);
            }
            int pass = 0;
            if (cs >= 0) {
                uint32_t n[3];
                w_make_n(v0_lo, v0_hi, (uint64_t)((rw << 6) + (uint64_t)cs), n);
                tests++;
                pass = mr_base2(n);
            }
            unsigned m = __ballot_sync(0xffffffffu, pass);
            if (m) {
                int win = __ffs(m) - 1;
                int wcs = __shfl_sync(0xffffffffu, (int)cs, win);
                pnew = (int64_t)(rw << 6) + wcs;
            } else {
                if (rw <= pword) foundGap = 1;
                else sc = (int64_t)(rw << 6);
            }
        }
        if (foundGap) {
            int64_t q = wk3_next_prime_up(bm, nwords, jumpPos, v0_lo, v0_hi, &tests, lane);
            if (q < 0) goto done;
            if (lane == 0) {
                uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                if (gap >= minGapVal) {
                    uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                    if (idx < cap - 1) { out[idx + 1].slot = (uint64_t)p; out[idx + 1].gap = gap; }
                }
            }
            p = q;
        } else {
            p = pnew;
        }
        pword = (uint64_t)p >> 6;
    }
done:
    atomicAdd(&stats[0], tests);
    if (lane == 0) atomicAdd(&stats[1], jumps);
}

/* walk_kernel2: same semantics as walk_kernel, but each jump collects up to WK_K
   candidates and tests them in a FLAT warp-uniform loop (WK_K unconditional MRs)
   to avoid the warp-divergence amplification of the per-candidate loop. */
#define WK_K 3
__global__ static void walk_kernel2(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t v0_lo, uint64_t v0_hi,
                                    uint32_t minGapSlots, uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    uint64_t total = (uint64_t)gridDim.x * blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sliceLen = nwords * 64 / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    unsigned long long tests = 0, jumps = 0;
    uint32_t n[3];
    int64_t c = w_nextCand(bm, nwords, lo);
    while (c >= 0) {
        w_make_n(v0_lo, v0_hi, (uint64_t)c, n);
        tests++;
        if (mr_base2(n)) break;
        c = w_nextCand(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;
    while (p < hi) {
        int64_t jumpPos = p + (int64_t)minGapSlots;
        if (jumpPos >= (int64_t)(nwords * 64)) break;
        jumps++;
        int64_t cand[WK_K];
        int nc = 0, hitP = 0;
        int64_t s = jumpPos - 1;
        while (nc < WK_K) {
            s = w_prevCand(bm, s);
            if (s < 0) goto done;
            cand[nc++] = s;
            if (s == p) { hitP = 1; break; }
            s--;
        }
        int prim[WK_K];
        for (int i = 0; i < WK_K; i++) {
            if (i < nc && cand[i] > p) {
                w_make_n(v0_lo, v0_hi, (uint64_t)cand[i], n);
                tests++;
                prim[i] = mr_base2(n);
            } else prim[i] = 0;
        }
        int sel = -1;
        for (int i = 0; i < WK_K; i++) if (prim[i]) { sel = i; break; }
        if (sel >= 0) {
            p = cand[sel];
            int64_t nw = (p + (int64_t)minGapSlots) >> 6;
            if (nw < (int64_t)nwords) asm volatile("prefetch.global.L1 [%0];" :: "l"(bm + nw));
            continue;
        }
        if (!hitP) {
            int64_t t = cand[nc - 1] - 1;
            for (;;) {
                t = w_prevCand(bm, t);
                if (t < 0) goto done;
                if (t == p) { hitP = 1; break; }
                w_make_n(v0_lo, v0_hi, (uint64_t)t, n);
                tests++;
                if (mr_base2(n)) {
                    p = t;
                    int64_t nw = (p + (int64_t)minGapSlots) >> 6;
                    if (nw < (int64_t)nwords) asm volatile("prefetch.global.L1 [%0];" :: "l"(bm + nw));
                    break;
                }
                t--;
            }
        }
        if (hitP) {
            int64_t q = jumpPos;
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done;
                w_make_n(v0_lo, v0_hi, (uint64_t)q, n);
                tests++;
                if (mr_base2(n)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = (uint64_t)p; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    int64_t nw = (p + (int64_t)minGapSlots) >> 6;
                    if (nw < (int64_t)nwords) asm volatile("prefetch.global.L1 [%0];" :: "l"(bm + nw));
                    break;
                }
                q++;
            }
        }
    }
done:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

__global__ static void walk_kernel(const uint64_t *__restrict__ bm, uint64_t nwords,
                                   uint64_t v0_lo, uint64_t v0_hi,
                                   uint32_t minGapSlots, uint32_t minGapVal,
                                   GapRec * __restrict__ out, uint32_t cap,
                                   unsigned long long * __restrict__ stats) {
    uint64_t total = (uint64_t)gridDim.x * blockDim.x;
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t sliceLen = nwords * 64 / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;

    unsigned long long tests = 0, jumps = 0;

    /* boot: first MR-prime >= lo */
    int64_t c = w_nextCand(bm, nwords, lo);
    uint32_t n[3];
    while (c >= 0) {
        w_make_n(v0_lo, v0_hi, (uint64_t)c, n);
        tests++;
        if (mr_base2(n)) break;
        c = w_nextCand(bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;   /* slot of the last confirmed prime */

    while (p < hi) {
        int64_t jumpPos = p + (int64_t)minGapSlots;
        if (jumpPos >= (int64_t)(nwords * 64)) break;   /* range edge: host stitches */
        jumps++;
        /* backward scan starts at jumpPos-1 (EXCLUSIVE): a prime exactly at
           jumpPos means the gap from p is exactly minGap and must be recorded
           by the forward scan (same convention as the PGS walk). */
        int64_t s = jumpPos - 1;
        int foundGap = 0;
        for (;;) {
            s = w_prevCand(bm, s);
            if (s < 0) goto done;               /* cannot happen (p is a candidate) */
            if (s == p) { foundGap = 1; break; }
            w_make_n(v0_lo, v0_hi, (uint64_t)s, n);
            tests++;
            if (mr_base2(n)) { p = s; break; }  /* new anchor */
            s--;                                /* composite: keep scanning down */
        }
        if (foundGap) {
            /* no candidate in (p, jumpPos] -> upper prime is > jumpPos */
            int64_t q = jumpPos;
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done;           /* range edge */
                w_make_n(v0_lo, v0_hi, (uint64_t)q, n);
                tests++;
                if (mr_base2(n)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) {
                            out[idx + 1].slot = (uint64_t)p;
                            out[idx + 1].gap = gap;
                        }
                    }
                    p = q;
                    break;
                }
                q++;
            }
        }
    }
done:
    atomicAdd(&stats[0], tests);
    atomicAdd(&stats[1], jumps);
}

/* ---- host -------------------------------------------------------------- */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: walk_bench START_DEC N_NUMBERS MIN_GAP [grid] [block] [P]\n");
        return 2;
    }
    u128 start = 0;
    for (const char *s = argv[1]; *s; s++) start = start * 10 + (u128)(*s - '0');
    uint64_t N = strtoull(argv[2], NULL, 10);
    uint32_t minGapVal = (uint32_t)strtoul(argv[3], NULL, 10);
    uint32_t grid = (argc > 4) ? (uint32_t)strtoul(argv[4], NULL, 10) : 184;
    uint32_t block = (argc > 5) ? (uint32_t)strtoul(argv[5], NULL, 10) : 512;
    uint32_t P = (argc > 6) ? (uint32_t)strtoul(argv[6], NULL, 10) : 10000000u;
    int nomark = getenv("WALK_NOMARK") != NULL;

    const uint64_t segNum = 1ULL << SEG_BITS;
    if (N % segNum) { fprintf(stderr, "N must be a multiple of 2^24\n"); return 2; }
    uint64_t nseg = N / segNum;
    uint64_t half = N >> 1;
    uint64_t nwords = half >> 6;
    int cls_mode = getenv("LOADBM_CLASS") != NULL;
    if (cls_mode) nwords = (size_t)(N / 240u);   /* class slots: (N/30)*8/64 = N/240 */

    /* host primes + item table (same geometry as gpu_sieve_init) */
    uint8_t *sv = (uint8_t *)calloc((size_t)P + 1, 1);
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)P / 2 + 64));
    size_t np = 0;
    for (uint64_t i = 2; i <= (uint64_t)P; i++)
        if (!sv[i]) { hp[np++] = i;
            if (i * i <= (uint64_t)P) for (uint64_t j = i * i; j <= (uint64_t)P; j += i) sv[j] = 1; }
    uint64_t halfSeg = segNum >> 1;
    size_t cap = 0;
    for (size_t i = 0; i < np; i++) {
        if (hp[i] == 2) continue;          /* p=2 is parity-degenerate (odd slots) */
        cap += (size_t)((halfSeg / hp[i] + 2 + P0_CHUNK - 1) / P0_CHUNK);
    }
    P0Item *items = (P0Item *)malloc(cap * sizeof(P0Item));
    size_t ni = 0;
    for (size_t i = 0; i < np; i++) {
        if (hp[i] == 2) continue;
        uint64_t hits = halfSeg / hp[i] + 2;
        uint32_t chunks = (uint32_t)((hits + P0_CHUNK - 1) / P0_CHUNK);
        for (uint32_t cc = 0; cc < chunks; cc++) { items[ni].pidx = (uint32_t)i; items[ni].k0 = cc * P0_CHUNK; ni++; }
    }
    uint64_t *h_r = (uint64_t *)malloc(np * sizeof(uint64_t));
    uint64_t *h_iv = (uint64_t *)malloc(np * sizeof(uint64_t));
    for (size_t i = 0; i < np; i++) {
        h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]);
        h_iv[i] = ~0ULL / hp[i];
    }
    printf("[walk] start=%s N=%llu minGap=%u grid=%u block=%u P=%u items=%zu\n",
           argv[1], (unsigned long long)N, minGapVal, grid, block, P, ni);

    P0Item *d_items; uint64_t *d_p, *d_r, *d_iv, *d_bm; GapRec *d_out; unsigned long long *d_stats;
    cudaMalloc(&d_items, ni * sizeof(P0Item));
    cudaMalloc(&d_p, np * sizeof(uint64_t));
    cudaMalloc(&d_r, np * sizeof(uint64_t));
    cudaMalloc(&d_iv, np * sizeof(uint64_t));
    cudaMalloc(&d_bm, nwords * 8);
    cudaMalloc(&d_out, (size_t)(1 << 20) * sizeof(GapRec));
    cudaMalloc(&d_stats, 4 * sizeof(unsigned long long));
    cudaMemcpy(d_items, items, ni * sizeof(P0Item), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, hp, np * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * sizeof(uint64_t), cudaMemcpyHostToDevice);

    u128 v0 = start | 1;                       /* first odd >= start */
    uint64_t v0lo = (uint64_t)v0, v0hi = (uint64_t)(v0 >> 64);

    /* ---- sieve all segments ---- */
    double t0 = now_s(), t_mark = 0;
    if (getenv("LOADBM")) {
        uint64_t *tmp = (uint64_t *)malloc(nwords * 8);
        FILE *f = fopen("/tmp/eng_bm.bin", "rb");
        if (!f) { fprintf(stderr, "no eng_bm.bin\n"); return 1; }
        if (fread(tmp, 1, nwords * 8, f) != nwords * 8) { fprintf(stderr, "short read\n"); return 1; }
        fclose(f);
        cudaMemcpy(d_bm, tmp, nwords * 8, cudaMemcpyHostToDevice);
        free(tmp);
        printf("[walk] LOADBM: loaded /tmp/eng_bm.bin\n");
        t_mark = 0;
    } else if (!nomark) {
        cudaMemset(d_bm, 0, nwords * 8);
        uint32_t mgrid = (uint32_t)((ni + 255) / 256);
        for (uint64_t s = 0; s < nseg; s++) {
            u128 sv0 = start + (u128)s * segNum; sv0 |= 1;
            uint32_t vis = p0_vis_mask30((uint32_t)(sv0 % 30u));
            p0_mark_kernel<<<mgrid, 256>>>(d_items, (uint32_t)ni, d_p, d_r, d_iv,
                                           halfSeg, (uint64_t)sv0, (uint64_t)(sv0 >> 64),
                                           vis, d_bm + s * (halfSeg >> 6));
        }
        cudaDeviceSynchronize();
        t_mark = now_s() - t0;
    }
    double t1 = now_s();
    cudaMemset(d_out, 0, sizeof(GapRec));      /* counter at out[0].gap */
    cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long));
    if (getenv("DUMPBM")) {
        uint64_t *tmp = (uint64_t *)malloc(nwords * 8);
        cudaMemcpy(tmp, d_bm, nwords * 8, cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/bench_bm.bin", "wb");
        fwrite(tmp, 1, nwords * 8, f); fclose(f);
        printf("[walk] dumped /tmp/bench_bm.bin\n");
    }
    if (nomark) cudaMemcpy(d_stats, d_stats, 8, cudaMemcpyDeviceToDevice);
    uint32_t *d_wtab = NULL;
    u128 A30 = 0;
    if (cls_mode) {
        A30 = (v0 / 30u) * 30u;
        uint8_t cp[30]; const uint32_t C8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
        for (int i = 0; i < 30; i++) cp[i] = 0;
        for (int i = 0; i < 8; i++) cp[C8[i]] = 1;
        uint32_t wtab[8];
        for (int c = 0; c < 8; c++) {
            uint32_t w = 0;
            for (uint32_t t = 1; t <= minGapVal - 1u; t++)
                if (cp[(C8[c] + t) % 30u]) w++;
            wtab[c] = w;
        }
        cudaMalloc(&d_wtab, 8 * 4);
        cudaMemcpy(d_wtab, wtab, 8 * 4, cudaMemcpyHostToDevice);
    }
    if (cls_mode)
        walk_kernel7<<<grid, block>>>(d_bm, nwords, (uint64_t)A30, (uint64_t)(A30 >> 64),
                                      d_wtab, minGapVal, d_out, 1 << 20, d_stats);
    else if (getenv("WK5"))
        walk_kernel5<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                      minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    else if (getenv("WK6"))
        walk_kernel6<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                      minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    else if (getenv("WK4"))
        walk_kernel4<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                      minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    else if (getenv("WK3"))
        walk_kernel3<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                      minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    else if (getenv("WK2"))
        walk_kernel2<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                      minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    else
        walk_kernel<<<grid, block>>>(d_bm, nwords, v0lo, v0hi,
                                     minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[walk] %s\n", cudaGetErrorString(e)); return 1; }
    double t_walk = now_s() - t1;

    unsigned long long stats[2] = {0, 0};
    cudaMemcpy(stats, d_stats, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    uint32_t ng = 0;
    cudaMemcpy(&ng, &d_out[0].gap, 4, cudaMemcpyDeviceToHost);
    if (ng > (1u << 20) - 1) ng = (1u << 20) - 1;
    GapRec *res = (GapRec *)malloc((ng ? ng : 1) * sizeof(GapRec));
    if (ng) cudaMemcpy(res, d_out + 1, ng * sizeof(GapRec), cudaMemcpyDeviceToHost);

    char path[256];
    snprintf(path, sizeof path, "/tmp/walk_gaps_%s.txt", argv[1]);
    FILE *f = fopen(path, "w");
    for (uint32_t i = 0; i < ng; i++) {
        u128 val = ((u128)v0hi << 64 | v0lo) + 2 * (u128)res[i].slot;
        char buf[48];
        int k = 0;
        while (val) { buf[k++] = (char)('0' + (int)(val % 10)); val /= 10; }
        if (!k) buf[k++] = '0';
        while (k--) fputc(buf[k], f);
        fprintf(f, " %u\n", res[i].gap);
    }
    fclose(f);

    double bps_sieve = t_mark > 0 ? (double)N / t_mark / 1e9 : 0;
    double bps_walk = (double)N / t_walk / 1e9;
    double bps_tot = (double)N / (t_mark + t_walk) / 1e9;
    printf("[walk] sieve=%.3f s (%.1f B/s)  walk=%.3f s (%.1f B/s)  total=%.3f s (%.1f B/s)\n",
           t_mark, bps_sieve, t_walk, bps_walk, t_mark + t_walk, bps_tot);
    printf("[walk] tests=%llu jumps=%llu tests/jump=%.2f  gaps=%u -> %s\n",
           stats[0], stats[1], stats[1] ? (double)stats[0] / (double)stats[1] : 0.0, ng, path);
    return 0;
}
