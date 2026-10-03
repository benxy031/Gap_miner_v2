/* walk_engine.cu - integrated Phase-0 walk engine v2: tile-shared sieve +
   jump-by-minGap walk with our MR, PIPELINED over blocks (sieve block k+1 on
   stream A while walking block k on stream B).

   Blocks: NB blocks of BLOCK_NUM numbers. Bitmap: 1 bit per odd slot (our
   census format). Tile sieve geometry/code = tools/tile_bench.cu (chunked
   items, shared atomics, plain-store flush). Walk = tools/walk_bench.cu.

   Usage: walk_engine START N MIN_GAP [tile_bits] [sg_grid] [sg_block]
                       [wk_grid] [wk_block] [P]
   Prints per-stage and end-to-end B/s. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#include "mr68_kernel.cuh"
#include "perig.cuh"

typedef unsigned __int128 u128;
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

/* ================= tile sieve (from tile_bench.cu) ================= */
__device__ static inline uint64_t t_fastmod64(uint64_t x, uint64_t p, uint64_t invp) {
    uint64_t q = __umul64hi(x, invp);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}
__global__ void tile_sieve_kernel(const uint64_t * __restrict__ primes,
                                  const uint64_t * __restrict__ invp,
                                  const uint64_t * __restrict__ r64,
                                  const uint32_t * __restrict__ itemPidx,
                                  const uint32_t * __restrict__ itemK0,
                                  uint32_t nitems,
                                  uint64_t v0_lo, uint64_t v0_hi,
                                  uint64_t tileSlots, uint64_t ntiles,
                                  uint64_t * __restrict__ g_bm,
                                  uint32_t tileWords,
                                  int wheel_on,
                                  uint64_t v0m3, uint64_t v0m5, uint64_t v0m7,
                                  uint64_t v0m11, uint64_t v0m13) {
    extern __shared__ uint64_t sh[];
    __shared__ uint64_t pat[236];
    __shared__ uint64_t rp[5];
    const uint64_t wp[5] = {3u, 5u, 7u, 11u, 13u};
    const uint64_t v0m[5] = {v0m3, v0m5, v0m7, v0m11, v0m13};
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0;
        __syncthreads();
        uint64_t off2 = tile * tileSlots * 2;
        u128 tv = ((u128)v0_hi << 64 | v0_lo) + (u128)off2;
        uint64_t t_lo = (uint64_t)tv; uint64_t t_hi = (uint64_t)(tv >> 64);
        for (uint32_t it = threadIdx.x; it < nitems; it += blockDim.x) {
            int pi = (int)itemPidx[it];
            uint64_t p = primes[pi], pv = invp[pi];
            uint64_t vh = t_fastmod64(t_hi, p, pv);
            uint64_t vl = t_fastmod64(t_lo, p, pv);
            uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
            uint64_t d = vm ? (p - vm) : 0;
            uint64_t s = t_fastmod64(d * ((p + 1) >> 1), p, pv)
                       + (uint64_t)itemK0[it] * 64u * p;
            uint32_t j = 0;
            for (; j < 64u && s < tileSlots; j++, s += p)
                atomicOr(((unsigned int *)sh) + (uint32_t)(s >> 5), 1u << (s & 31u));
            (void)j;
        }
        __syncthreads();
        if (wheel_on) {
            /* wheel pass: odd multiples of {3,5,7,11,13} by OR-ing a phase-shifted
               period-15015 pattern (no atomics; their item entries are removed) */
            if (threadIdx.x == 0) {
                uint64_t g0 = tile * tileSlots;
                for (int i = 0; i < 5; i++) {
                    uint64_t p = wp[i];
                    uint64_t B = (v0m[i] + 2u * (g0 % p)) % p;
                    rp[i] = ((p - B) % p) * ((p + 1u) >> 1) % p;
                }
            }
            __syncthreads();
            for (uint32_t w = threadIdx.x; w < 236u; w += blockDim.x) pat[w] = 0ull;
            __syncthreads();
            for (int i = 0; i < 5; i++) {
                uint64_t p = wp[i], r = rp[i];
                for (uint64_t k = threadIdx.x; ; k += blockDim.x) {
                    uint64_t j2 = r + k * p;
                    if (j2 >= 15079u) break;
                    atomicOr((unsigned long long *)&pat[j2 >> 6], 1ull << (j2 & 63));
                }
                __syncthreads();
            }
            __syncthreads();
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) {
                uint64_t idx = ((uint64_t)64u * w) % 15015u;
                uint32_t k = (uint32_t)(idx >> 6), r2 = (uint32_t)(idx & 63);
                uint64_t val = (pat[k] >> r2) | (r2 ? (pat[k + 1] << (64 - r2)) : 0ull);
                sh[w] |= val;
            }
            __syncthreads();
        }
        uint64_t *out = g_bm + tile * tileWords;
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x)
            out[w] = sh[w];
        __syncthreads();
    }
}

/* ================= walk (from walk_bench.cu) ================= */
typedef struct { uint64_t slot; uint32_t gap; uint32_t pad; } GapRec;

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
/* FUSED sieve+walk launch (PGS kernelBoth style): the launch's CTAs are
   partitioned - blockIdx.x < sg_ctas run the tile sieve of block `k` into one
   bitmap buffer; the rest run the FLAT walk of block k-1 over the other buffer.
   Double buffering + serial launches => the walk is one full launch behind the
   sieve; the SM scheduler naturally mixes both roles. */
__global__ static void tile_walk_fused(
    /* sieve role */
    const uint64_t * __restrict__ primes,
    const uint64_t * __restrict__ invp,
    const uint64_t * __restrict__ r64,
    const uint32_t * __restrict__ itemPidx,
    const uint32_t * __restrict__ itemK0,
    uint32_t nitems,
    uint64_t sv0_lo, uint64_t sv0_hi,
    uint64_t sieveTiles, uint64_t tilesPerBlock,
    uint64_t * __restrict__ sieve_bm, uint32_t tileWords,
    uint32_t sg_ctas,
    /* walk role */
    const uint64_t * __restrict__ walk_bm, uint64_t walk_words,
    uint64_t wv0_lo, uint64_t wv0_hi,
    uint32_t minGapSlots, uint32_t minGapVal,
    GapRec * __restrict__ out, uint32_t cap,
    unsigned long long * __restrict__ stats) {
    extern __shared__ uint64_t sh[];
    /* Role assignment INTERLEAVED by CTA index (Bresenham): a prefix split
       (blockIdx < sg_ctas) makes the GPU -- which dispatches CTAs in order --
       start every walk CTA only after all sieve CTAs retire, killing overlap.
       Here exactly wk_ctas CTAs (spread across the whole grid) are walk CTAs,
       so both roles are resident from the first wave. */
    {
    uint32_t wk_ctas = (uint32_t)gridDim.x - sg_ctas;
    uint32_t rw0 = (uint32_t)(((uint64_t)blockIdx.x * wk_ctas) / gridDim.x);
    uint32_t rw1 = (uint32_t)(((uint64_t)(blockIdx.x + 1) * wk_ctas) / gridDim.x);
    if (rw1 == rw0) {
        uint32_t rank = blockIdx.x - rw1; /* sieve CTA number */
        if (rank >= sg_ctas) return;      /* safety; never taken when M=sg+wk */
        for (uint64_t tile = rank; tile < sieveTiles; tile += sg_ctas) {
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0;
            __syncthreads();
            /* NOTE: tilesPerBlock parameter carries the per-tile SLOT count */
            u128 tv = ((u128)sv0_hi << 64 | sv0_lo)
                    + ((u128)tile * tilesPerBlock * 2u);
            uint64_t t_lo = (uint64_t)tv; uint64_t t_hi = (uint64_t)(tv >> 64);
            for (uint32_t it = threadIdx.x; it < nitems; it += blockDim.x) {
                int pi = (int)itemPidx[it];
                uint64_t p = primes[pi], pv = invp[pi];
                uint64_t vh = t_fastmod64(t_hi, p, pv);
                uint64_t vl = t_fastmod64(t_lo, p, pv);
                uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
                uint64_t d = vm ? (p - vm) : 0;
                uint64_t s = t_fastmod64(d * ((p + 1) >> 1), p, pv)
                           + (uint64_t)itemK0[it] * 64u * p;
                uint32_t j = 0;
                for (; j < 64u && s < tilesPerBlock; j++, s += p)
                    atomicOr(((unsigned int *)sh) + (uint32_t)(s >> 5), 1u << (s & 31u));
                (void)j;
            }
            __syncthreads();
            uint64_t *o = sieve_bm + tile * tileWords;
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x)
                o[w] = sh[w];
            __syncthreads();
        }
        return;
    }
    /* ---- walk role: FLAT walk of block k-1 (identical to walk_kernel5) ---- */
    uint64_t nwords = walk_words;
    uint64_t total = (uint64_t)wk_ctas * blockDim.x;
    if (total == 0) return;
    uint64_t tid = (uint64_t)(rw1 - 1) * blockDim.x + threadIdx.x;
    uint64_t sliceLen = nwords * 64 / total;
    int64_t lo = (int64_t)(tid * sliceLen);
    int64_t hi = (tid + 1 == total) ? (int64_t)(nwords * 64) : lo + (int64_t)sliceLen;
    if (lo >= hi || sliceLen == 0) return;
    const int64_t end = (int64_t)(nwords * 64);
    unsigned long long tests = 0, jumps = 0;
    uint32_t n[3];
    int64_t c = w_nextCand(walk_bm, nwords, lo);
    while (c >= 0) {
        w_make_n(wv0_lo, wv0_hi, (uint64_t)c, n);
        tests++;
        if (mr_base2(n)) break;
        c = w_nextCand(walk_bm, nwords, c + 1);
    }
    if (c < 0) { atomicAdd(&stats[0], tests); return; }
    int64_t p = c;
    int64_t s = p + (int64_t)minGapSlots - 1;
    if (s >= end) { atomicAdd(&stats[0], tests); return; }
    for (;;) {
        if (p >= hi) goto done;
        s = w_prevCand(walk_bm, s);
        if (s < 0) goto done;
        if (s == p) {
            int64_t q = p + (int64_t)minGapSlots;
            for (;;) {
                q = w_nextCand(walk_bm, nwords, q);
                if (q < 0) goto done;
                w_make_n(wv0_lo, wv0_hi, (uint64_t)q, n);
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
        w_make_n(wv0_lo, wv0_hi, (uint64_t)s, n);
        tests++;
        if (mr_base2(n)) {
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
}

/* ---- class-30 mode ----------------------------------------------------
   slot s <-> value A + 30*(s>>3) + CLS[s&7]; only numbers coprime to 30
   exist as bits, so primes 3 and 5 mark nothing and every other prime marks
   1/p of the slots (vs 1/p of a 2x larger odd-slot space). */
__device__ __constant__ uint32_t c30_cls[8] = {1, 7, 11, 13, 17, 19, 23, 29};
__device__ __constant__ int8_t c30_clsidx[30];
__device__ __constant__ uint32_t c30_perm[30];
__device__ __constant__ uint32_t c30_pinv[30];

__global__ static void class30_sieve_kernel(const uint64_t * __restrict__ primes,
                                            const uint64_t * __restrict__ invp,
                                            const uint64_t * __restrict__ r64,
                                            const uint32_t * __restrict__ itemPidx,
                                            const uint32_t * __restrict__ itemK0,
                                            uint32_t nitems,
                                            uint64_t Alo, uint64_t Ahi,
                                            uint64_t tileSlots, uint64_t ntiles,
                                            uint64_t * __restrict__ g_bm,
                                            uint32_t tileWords,
                                            const uint32_t * __restrict__ wpidx, int wheel_on) {
    extern __shared__ uint64_t sh[];
    __shared__ uint64_t wpat[127];
    __shared__ uint32_t wpj[24];
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0;
        __syncthreads();
        u128 tv = ((u128)Ahi << 64 | Alo) + (u128)((tile * tileSlots) >> 3) * 30u + 1u;
        uint64_t t_lo = (uint64_t)tv, t_hi = (uint64_t)(tv >> 64);
        if (wheel_on && threadIdx.x < 3u) {
            int pi = (int)wpidx[threadIdx.x];
            uint64_t p = primes[pi], pv = invp[pi];
            uint64_t vh = t_fastmod64(t_hi, p, pv);
            uint64_t vl = t_fastmod64(t_lo, p, pv);
            uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
            uint64_t d = vm ? (p - vm) : 0;
            uint64_t phi1 = 1u + d;
            uint64_t r0 = phi1 % 30u;
            uint64_t pm30 = p % 30u;
            uint32_t mask = c30_perm[pm30];
            uint64_t kadj = 0;
            while (!((mask >> ((r0 + kadj * pm30) % 30u)) & 1u)) kadj++;
            uint64_t phi = phi1 + kadj * p;
            uint64_t r = phi % 30u;
            uint64_t pos = (phi / 30u) * 8u + (uint64_t)c30_clsidx[r];
            uint32_t step[8];
            uint64_t j = (uint64_t)c30_clsidx[(r * c30_pinv[pm30]) % 30u];
            uint64_t ph = phi;
            for (int s = 0; s < 8; s++) {
                uint64_t j2 = (j + 1u) & 7u;
                uint64_t dm = (uint64_t)c30_cls[j2] - (uint64_t)c30_cls[j];
                if (j2 == 0u) dm += 30u;
                uint64_t ph2 = ph + dm * p;
                step[s] = (uint32_t)((ph2 / 30u) * 8u + (uint64_t)c30_clsidx[ph2 % 30u]
                                   - ((ph / 30u) * 8u + (uint64_t)c30_clsidx[ph % 30u]));
                ph = ph2; j = j2;
            }
            uint32_t cum = 0;
            for (int s = 0; s < 8; s++) { wpj[threadIdx.x * 8 + s] = (uint32_t)pos + cum; cum += step[s]; }
        }
        {
        /* contiguous per-thread runs over the prime-major item table, reusing the
           per-prime setup (fastmods + bump + 8 slot steps) across its chunks */
        uint32_t nthreads = blockDim.x;
        uint32_t per = nitems / nthreads, rem = nitems - per * nthreads;
        uint32_t it0 = threadIdx.x * per + (threadIdx.x < rem ? threadIdx.x : rem);
        uint32_t itn = per + (threadIdx.x < rem ? 1u : 0u);
        int cur_pi = -1;
        uint32_t cur_p = 0;
        uint32_t pos_base = 0;
        uint32_t step[8];
        for (uint32_t it = it0; it < it0 + itn; it++) {
            int pi = (int)itemPidx[it];
            uint32_t k0 = itemK0[it];
            if (pi != cur_pi) {
                cur_pi = pi;
                uint64_t p = primes[pi];
                cur_p = (uint32_t)p;
                uint64_t pv = invp[pi];
                uint64_t vh = t_fastmod64(t_hi, p, pv);
                uint64_t vl = t_fastmod64(t_lo, p, pv);
                uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
                uint64_t d = vm ? (p - vm) : 0;
                uint32_t phi1 = (uint32_t)(1u + d);
                uint32_t r0 = phi1 % 30u;
                uint32_t pm30 = (uint32_t)(p % 30u);
                uint32_t mask = c30_perm[pm30];
                uint32_t kadj = 0;
                while (!((mask >> ((r0 + kadj * pm30) % 30u)) & 1u)) kadj++;
                uint32_t phi = phi1 + kadj * cur_p;
                uint32_t r = phi % 30u;
                pos_base = (phi / 30u) * 8u + (uint32_t)c30_clsidx[r];
                uint32_t j = (uint32_t)c30_clsidx[(r * c30_pinv[pm30]) % 30u];
                uint32_t ph = phi;
                for (int st = 0; st < 8; st++) {
                    uint32_t j2 = (j + 1u) & 7u;
                    uint32_t dm = (uint32_t)c30_cls[j2] - (uint32_t)c30_cls[j];
                    if (j2 == 0u) dm += 30u;
                    uint32_t ph2 = ph + dm * cur_p;
                    step[st] = (ph2 / 30u) * 8u + (uint32_t)c30_clsidx[ph2 % 30u]
                             - ((ph / 30u) * 8u + (uint32_t)c30_clsidx[ph % 30u]);
                    ph = ph2; j = j2;
                }
            }
            uint32_t p32 = pos_base + 64u * cur_p * k0;
            if (p32 >= (uint32_t)tileSlots) continue;
            for (int r8 = 0; r8 < 8; r8++) {
#pragma unroll
                for (int st = 0; st < 8; st++) {
                    if (p32 < (uint32_t)tileSlots)
                        atomicOr(((unsigned int *)sh) + (p32 >> 5), 1u << (p32 & 31u));
                    p32 += step[st];
                }
            }
        }
        }
        __syncthreads();
        if (wheel_on) {
            const uint64_t P = 8008ull, ASZ = 8128ull;
            for (uint32_t w = threadIdx.x; w < 127u; w += blockDim.x) wpat[w] = 0ull;
            __syncthreads();
            for (int wi = 0; wi < 3; wi++) {
                uint64_t p = (wi == 0) ? 7u : (wi == 1) ? 11u : 13u;
                uint64_t stride8 = 8u * p;
                uint64_t nt = ASZ / stride8 + 2u;
                for (uint64_t idx = threadIdx.x; idx < 8u * nt; idx += blockDim.x) {
                    uint64_t j = idx & 7u, t = idx >> 3;
                    uint64_t pos = (uint64_t)wpj[wi * 8 + (int)j] + t * stride8;
                    if (pos >= ASZ) continue;
                    uint64_t q = pos % P;
                    atomicOr((unsigned long long *)&wpat[q >> 6], 1ull << (q & 63u));
                    if (q < 64u) atomicOr((unsigned long long *)&wpat[(q + P) >> 6], 1ull << ((q + 8u) & 63u));
                }
            }
            __syncthreads();
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) {
                uint64_t idx = ((uint64_t)64u * w) % P;
                uint32_t k = (uint32_t)(idx >> 6), r2 = (uint32_t)(idx & 63u);
                uint64_t val = (wpat[k] >> r2) | (r2 ? (wpat[k + 1] << (64 - r2)) : 0ull);
                sh[w] |= val;
            }
            __syncthreads();
        }
        uint64_t *out = g_bm + tile * tileWords;
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x)
            out[w] = sh[w];
        __syncthreads();
    }
}

/* class-30 walk: same flat structure as walk_kernel6, but slot->value uses
   the class map and the window advance comes from wtab[class of anchor]. */
__global__ static void walk_kernel7(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t Alo, uint64_t Ahi,
                                    const uint32_t *__restrict__ wtab,
                                    uint32_t minGapVal,
                                    GapRec * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
    (void)Ahi;
    __shared__ uint32_t scls[8];
    if (threadIdx.x < 8u) scls[threadIdx.x] = c30_cls[threadIdx.x];
    __syncthreads();
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
        uint64_t v = Alo + 30ull * ((uint64_t)c >> 3) + (uint64_t)scls[c & 7u];
        if (ciosFermatTest128(v)) break;
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
                uint64_t vq = Alo + 30ull * ((uint64_t)q >> 3) + (uint64_t)scls[q & 7u];
                if (ciosFermatTest128(vq)) {
                    uint64_t vp = Alo + 30ull * ((uint64_t)p >> 3) + (uint64_t)scls[p & 7u];
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
        uint64_t vs = Alo + 30ull * ((uint64_t)s >> 3) + (uint64_t)scls[s & 7u];
        if (ciosFermatTest128(vs)) {
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

/* walk_kernel5: FLAT (PGS-shape) walk - identical semantics, but the per-jump
   inner loop is flattened into one free-running loop per thread (no per-jump
   warp-wide stopping point -> no E[max32] geometric divergence tax). */
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
                        if (idx < cap - 1) { out[idx + 1].slot = v0_lo + 2ULL * (uint64_t)p; out[idx + 1].gap = gap; }
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

/* walk_kernel6: identical to walk_kernel5, but the primality test is the
   PGS/Perig Euler-Legendre base-2 test (ciosFermatTest128) on the LOW 64 bits
   of the candidate with the (constant) high word folded in at compile time
   through HIGH_64. */
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
    int64_t p = c;
    int64_t s = p + (int64_t)minGapSlots - 1;
    if (s >= end) { atomicAdd(&stats[0], tests); return; }
    for (;;) {
        if (p >= hi) goto done6;
        s = w_prevCand(bm, s);
        if (s < 0) goto done6;
        if (s == p) {
            int64_t q = p + (int64_t)minGapSlots;
            for (;;) {
                q = w_nextCand(bm, nwords, q);
                if (q < 0) goto done6;
                tests++;
                if (ciosFermatTest128(v0_lo + 2ULL * (uint64_t)q)) {
                    uint32_t gap = (uint32_t)((uint64_t)(q - p) << 1);
                    if (gap >= minGapVal) {
                        uint32_t idx = atomicAdd((uint32_t *)&out[0].gap, 1u);
                        if (idx < cap - 1) { out[idx + 1].slot = v0_lo + 2ULL * (uint64_t)p; out[idx + 1].gap = gap; }
                    }
                    p = q;
                    jumps++;
                    s = p + (int64_t)minGapSlots - 1;
                    if (s >= end) goto done6;
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
            if (s >= end) goto done6;
        } else {
            s--;
        }
    }
done6:
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
    int64_t c = w_nextCand(bm, nwords, lo);
    uint32_t n[3];
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
        int64_t s = jumpPos - 1;
        int foundGap = 0;
        for (;;) {
            s = w_prevCand(bm, s);
            if (s < 0) goto done;
            if (s == p) { foundGap = 1; break; }
            w_make_n(v0_lo, v0_hi, (uint64_t)s, n);
            tests++;
            if (mr_base2(n)) { p = s; break; }
            s--;
        }
        if (foundGap) {
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

/* ================= host ================= */
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: walk_engine START N MIN_GAP [tile_bits] [sg_grid] [sg_block] [wk_grid] [wk_block] [P]\n"); return 2; }
    u128 start = 0;
    for (const char *s = argv[1]; *s; s++) start = start * 10 + (u128)(*s - '0');
    uint64_t N = strtoull(argv[2], NULL, 10);
    uint32_t minGapVal = (uint32_t)strtoul(argv[3], NULL, 10);
    int tile_bits = (argc > 4) ? atoi(argv[4]) : 19;
    int sg_grid = (argc > 5) ? atoi(argv[5]) : 138;
    int sg_block = (argc > 6) ? atoi(argv[6]) : 512;
    int wk_grid = (argc > 7) ? atoi(argv[7]) : 184;
    int wk_block = (argc > 8) ? atoi(argv[8]) : 512;
    uint32_t P = (argc > 9) ? (uint32_t)strtoul(argv[9], NULL, 10) : 100000u;

    const uint64_t blockNum = 1ULL << 30;            /* numbers per pipeline block */
    int class_mode = getenv("CLASS30") != NULL;
    int cwheel = getenv("WHEEL713") != NULL;
    const uint64_t blockNum30 = 30ull * (1ull << 25);      /* 1,006,632,960 numbers */
    const uint64_t blkReq = class_mode ? blockNum30 : blockNum;
    if (N % blkReq) { fprintf(stderr, "N must be a multiple of %llu\n", (unsigned long long)blkReq); return 2; }
    uint64_t NB = N / blkReq;
    uint64_t tileNum = 1ULL << tile_bits;
    uint64_t tileSlots = tileNum >> 1;
    uint32_t tileWords = (uint32_t)((tileSlots + 63) / 64);
    if (blockNum % tileNum) { fprintf(stderr, "blockNum %% tileNum != 0\n"); return 2; }
    uint64_t tilesPerBlock = blockNum / tileNum;

    uint8_t *sv = (uint8_t *)calloc((size_t)P + 1, 1);
    for (uint64_t i = 4; i <= (uint64_t)P; i += 2) sv[i] = 1;   /* evens are composite */
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)P / 2 + 64));
    size_t np = 0;
    for (uint64_t i = 3; i <= (uint64_t)P; i++)
        if (!sv[i]) { hp[np++] = i;
            if (i * i <= (uint64_t)P) for (uint64_t j = i * i; j <= (uint64_t)P; j += i) sv[j] = 1; }
    uint64_t *h_iv = (uint64_t *)malloc(np * sizeof(uint64_t));
    uint64_t *h_r = (uint64_t *)malloc(np * sizeof(uint64_t));
    for (size_t i = 0; i < np; i++) { h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]); h_iv[i] = ~0ULL / hp[i]; }
    size_t icap = 0;
    for (size_t i = 0; i < np; i++) icap += (size_t)((tileSlots / hp[i] + 64) / 64);
    int wheel_on = getenv("WHEEL13") != NULL;
    uint32_t *h_ip = (uint32_t *)malloc(icap * 4), *h_ik = (uint32_t *)malloc(icap * 4);
    size_t nii = 0;
    for (size_t i = 0; i < np; i++) {
        if (wheel_on && hp[i] <= 13u) continue;      /* wheel pass marks these */
        uint64_t hits = tileSlots / hp[i] + 1;
        uint32_t chunks = (uint32_t)((hits + 63) / 64);
        for (uint32_t c = 0; c < chunks; c++) { h_ip[nii] = (uint32_t)i; h_ik[nii] = c; nii++; }
    }
    uint64_t *d_p, *d_iv, *d_r, *d_bm; uint32_t *d_ip, *d_ik;
    GapRec *d_out; unsigned long long *d_stats;
    uint64_t blockWords = blockNum / 2 / 64;
    cudaMalloc(&d_p, np * 8); cudaMalloc(&d_iv, np * 8); cudaMalloc(&d_r, np * 8);
    cudaMalloc(&d_ip, nii * 4); cudaMalloc(&d_ik, nii * 4);
    cudaMalloc(&d_bm, blockWords * 8 * (class_mode ? 1ull : NB));
    cudaMalloc(&d_out, (size_t)(1 << 20) * sizeof(GapRec));
    cudaMalloc(&d_stats, 4 * sizeof(unsigned long long));
    cudaMemcpy(d_p, hp, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip, h_ip, nii * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ik, h_ik, nii * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(GapRec));
    cudaMemset(d_stats, 0, 4 * sizeof(unsigned long long));

    cudaStream_t sA, sB;
    cudaStreamCreate(&sA); cudaStreamCreate(&sB);
    cudaEvent_t ev[64];
    if (NB > 64) { fprintf(stderr, "NB>64\n"); return 2; }
    for (uint64_t k = 0; k < NB; k++) cudaEventCreateWithFlags(&ev[k], cudaEventDisableTiming);
    size_t shbytes = (size_t)tileWords * 8;
    cudaFuncSetAttribute(tile_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
    cudaFuncSetAttribute(tile_walk_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);

    /* ---- class-30 mode setup ---------------------------------------- */
    const uint64_t cWords = (1ull << 28) / 64;             /* 32 MB bitmap / block */
    const uint64_t tilesPerBlockC = (1ull << 28) / tileSlots;
    uint32_t *d_ip2 = NULL, *d_ik2 = NULL, *d_wtab = NULL, *d_wpidx = NULL;
    uint64_t *d_bm2 = NULL;
    size_t nii2 = 0;
    uint64_t NBc = 0;
    if (class_mode) {
        NBc = NB;
        uint8_t cp[30]; int8_t clsidx[30];
        for (int i = 0; i < 30; i++) { cp[i] = 0; clsidx[i] = -1; }
        const uint32_t CLS8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
        for (int i = 0; i < 8; i++) { cp[CLS8[i]] = 1; clsidx[CLS8[i]] = (int8_t)i; }
        uint32_t perm[30] = {0}, pinv[30] = {0};
        for (int pm = 0; pm < 30; pm++) {
            if (!cp[pm]) continue;
            uint32_t m = 0;
            for (int j = 0; j < 8; j++) m |= 1u << ((pm * (int)CLS8[j]) % 30);
            perm[pm] = m;
            for (int x = 1; x < 30; x++) if ((pm * x) % 30 == 1) pinv[pm] = (uint32_t)x;
        }
        cudaMemcpyToSymbol(c30_clsidx, clsidx, sizeof(clsidx));
        cudaMemcpyToSymbol(c30_perm, perm, sizeof(perm));
        cudaMemcpyToSymbol(c30_pinv, pinv, sizeof(pinv));
        /* wtab[c] = number of coprimes strictly between v_c and v_c+minGapVal */
        uint32_t wtab[8];
        for (int c = 0; c < 8; c++) {
            uint32_t w = 0;
            for (uint32_t t = 1; t <= minGapVal - 1u; t++)
                if (cp[(CLS8[c] + t) % 30u]) w++;
            wtab[c] = w;
        }
        cudaMalloc(&d_wtab, 8 * 4);
        cudaMemcpy(d_wtab, wtab, 8 * 4, cudaMemcpyHostToDevice);
        /* class item list: same primes but skipping 3 and 5 (and 7..13 with wheel) */
        size_t icap2 = 0;
        for (size_t i = 0; i < np; i++) {
            if (hp[i] <= 5u) continue;
            if (cwheel && hp[i] <= 13u) continue;
            icap2 += (size_t)((tileSlots / hp[i] + 2 + 63) / 64);
        }
        uint32_t *h_ip2 = (uint32_t *)malloc(icap2 * 4), *h_ik2 = (uint32_t *)malloc(icap2 * 4);
        for (size_t i = 0; i < np; i++) {
            if (hp[i] <= 5u) continue;
            if (cwheel && hp[i] <= 13u) continue;
            uint64_t marks = tileSlots / hp[i] + 2;
            uint32_t chunks = (uint32_t)((marks + 63) / 64);
            for (uint32_t c = 0; c < chunks; c++) { h_ip2[nii2] = (uint32_t)i; h_ik2[nii2] = c; nii2++; }
        }
        uint32_t h_wpidx[3] = {0, 0, 0};
        for (size_t i = 0; i < np; i++) {
            if (hp[i] == 7u) h_wpidx[0] = (uint32_t)i;
            if (hp[i] == 11u) h_wpidx[1] = (uint32_t)i;
            if (hp[i] == 13u) h_wpidx[2] = (uint32_t)i;
        }
        cudaMalloc(&d_wpidx, 3 * 4);
        cudaMemcpy(d_wpidx, h_wpidx, 3 * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_ip2, nii2 * 4); cudaMalloc(&d_ik2, nii2 * 4);
        cudaMalloc(&d_bm2, (size_t)cWords * 8 * NBc);
        cudaMemcpy(d_ip2, h_ip2, nii2 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ik2, h_ik2, nii2 * 4, cudaMemcpyHostToDevice);
        free(h_ip2); free(h_ik2);
        cudaFuncSetAttribute(class30_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
        printf("[engine] CLASS30: block=%llu numbers slots/blk=%llu items=%zu tiles/blk=%llu wtab=%u/%u/%u/%u/%u/%u/%u/%u\n",
               (unsigned long long)blockNum30, (unsigned long long)(1ull << 28), nii2,
               (unsigned long long)tilesPerBlockC,
               wtab[0], wtab[1], wtab[2], wtab[3], wtab[4], wtab[5], wtab[6], wtab[7]);
    }

    printf("[engine] start=%s N=%llu minGap=%u NB=%llu tile=%llu tiles/blk=%llu items=%zu sg=%dx%d wk=%dx%d P=%u\n",
           argv[1], (unsigned long long)N, minGapVal, (unsigned long long)NB,
           (unsigned long long)tileNum, (unsigned long long)tilesPerBlock, nii,
           sg_grid, sg_block, wk_grid, wk_block, P);

    double t0 = now_s();
    double t_sieve = 0;
    if (class_mode) {
        /* batched walk: one walk launch per WALK_BATCH contiguous blocks.
           The walk is tail-dominated at M=1 (slowest thread sets the wall);
           long slices average the per-thread work => ~2x (measured 203->441 B/s
           at T=1260, M=16).  Pipeline is preserved at BATCH granularity: all
           sieves enqueue on sA (one event per batch), walks on sB wait that event,
           so sieve batch b+1 runs concurrently with walk batch b. */
        uint32_t wb = (uint32_t)strtoul(getenv("WALK_BATCH") ? getenv("WALK_BATCH") : "16", NULL, 10);
        if (wb == 0) wb = 1;
        if ((uint64_t)wb > NBc) wb = (uint32_t)NBc;
        uint32_t nb = (uint32_t)((NBc + wb - 1) / wb);
        if (nb > 64) { fprintf(stderr, "[engine] WALK_BATCH too small (nb=%u)\n", nb); return 1; }
        for (uint64_t k = 0; k < NBc; k++) {
            u128 av = start + (u128)k * blockNum30;
            u128 A = (av / 30u) * 30u;
            uint64_t *bmk = d_bm2 + k * cWords;
            class30_sieve_kernel<<<sg_grid, sg_block, shbytes, sA>>>(d_p, d_iv, d_r, d_ip2, d_ik2,
                (uint32_t)nii2, (uint64_t)A, (uint64_t)(A >> 64), tileSlots, tilesPerBlockC,
                bmk, tileWords, d_wpidx, cwheel);
            uint64_t kend = k + 1;
            if (kend % wb == 0 || kend == NBc)
                cudaEventRecord(ev[k / wb], sA);
        }
        for (uint32_t b = 0; b < nb; b++) {
            uint64_t k0 = (uint64_t)b * wb;
            uint64_t m = (NBc - k0 < wb) ? (NBc - k0) : (uint64_t)wb;
            u128 av = start + (u128)k0 * blockNum30;
            u128 A = (av / 30u) * 30u;
            if (!getenv("NOSYNC")) cudaStreamWaitEvent(sB, ev[b], 0);
            walk_kernel7<<<wk_grid, wk_block, 0, sB>>>(d_bm2 + k0 * cWords, m * cWords,
                (uint64_t)A, (uint64_t)(A >> 64), d_wtab, minGapVal,
                d_out, 1 << 20, d_stats);
        }
    } else if (getenv("FUSED")) {
        /* PGS kernelBoth style: one launch per step, CTA-partitioned roles,
           double-buffered bitmaps, serial launches. Walk of block k-1 runs
           concurrently with the sieve of block k inside the same launch. */
        uint64_t *bm0 = d_bm, *bm1 = d_bm + blockWords;
        uint32_t sg_ctas = (uint32_t)sg_grid;
        uint32_t wk_ctas = (uint32_t)wk_grid;
        {
            u128 bv0 = start | 1;
            tile_walk_fused<<<sg_ctas, sg_block, shbytes, sB>>>(
                d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
                (uint64_t)bv0, (uint64_t)(bv0 >> 64),
                tilesPerBlock, tileSlots, bm0, tileWords, sg_ctas,
                bm0, blockWords, 0, 0, 0, 0, d_out, 1 << 20, d_stats);
        }
        for (uint64_t k = 1; k < NB; k++) {
            u128 sv = start + (u128)k * blockNum; sv |= 1;
            u128 wv = start + (u128)(k - 1) * blockNum; wv |= 1;
            uint64_t *sb = (k & 1) ? bm1 : bm0;
            uint64_t *wb = ((k - 1) & 1) ? bm1 : bm0;
            tile_walk_fused<<<sg_ctas + wk_ctas, sg_block, shbytes, sB>>>(
                d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
                (uint64_t)sv, (uint64_t)(sv >> 64),
                tilesPerBlock, tileSlots, sb, tileWords, sg_ctas,
                wb, blockWords, (uint64_t)wv, (uint64_t)(wv >> 64),
                minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
        }
        {
            u128 wv = start + (u128)(NB - 1) * blockNum; wv |= 1;
            uint64_t *wb = ((NB - 1) & 1) ? bm1 : bm0;
            tile_walk_fused<<<wk_ctas, sg_block, shbytes, sB>>>(
                d_p, d_iv, d_r, d_ip, d_ik, 0,
                0, 0, 0, tileSlots, bm0, tileWords, 0,
                wb, blockWords, (uint64_t)wv, (uint64_t)(wv >> 64),
                minGapVal >> 1, minGapVal, d_out, 1 << 20, d_stats);
        }
    } else {
    /* pipeline: sieve block k on sA; walk block k on sB after ev[k].
       Launches INTERLEAVED (sieve k, walk k, sieve k+1, ...) so the GPU can
       overlap walk k with sieve k+1; two separate loops serialize them. */
    for (uint64_t k = 0; k < NB; k++) {
        u128 bv0 = start + (u128)k * blockNum; bv0 |= 1;
        uint64_t *bmk = d_bm + k * blockWords;
        uint64_t *pk_first = NULL; (void)pk_first;
        double ta = now_s();
        tile_sieve_kernel<<<sg_grid, sg_block, shbytes, sA>>>(d_p, d_iv, d_r, d_ip, d_ik,
            (uint32_t)nii, (uint64_t)bv0, (uint64_t)(bv0 >> 64), tileSlots, tilesPerBlock,
            bmk, tileWords,
            wheel_on, (uint64_t)(bv0 % 3u), (uint64_t)(bv0 % 5u), (uint64_t)(bv0 % 7u),
            (uint64_t)(bv0 % 11u), (uint64_t)(bv0 % 13u));
        cudaEventRecord(ev[k], sA);
        if (!getenv("NOSYNC")) cudaStreamWaitEvent(sB, ev[k], 0);
        if (getenv("WK6")) {
            walk_kernel6<<<wk_grid, wk_block, 0, sB>>>(d_bm + k * blockWords, blockWords,
                (uint64_t)bv0, (uint64_t)(bv0 >> 64), minGapVal >> 1, minGapVal,
                d_out, 1 << 20, d_stats);
        } else if (getenv("SAMESTREAM")) {
            walk_kernel5<<<wk_grid, wk_block, 0, sA>>>(d_bm + k * blockWords, blockWords,
                (uint64_t)bv0, (uint64_t)(bv0 >> 64), minGapVal >> 1, minGapVal,
                d_out, 1 << 20, d_stats);
        } else {
            walk_kernel5<<<wk_grid, wk_block, 0, sB>>>(d_bm + k * blockWords, blockWords,
                (uint64_t)bv0, (uint64_t)(bv0 >> 64), minGapVal >> 1, minGapVal,
                d_out, 1 << 20, d_stats);
        }
        (void)ta;
    }
    }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[engine] %s\n", cudaGetErrorString(e)); return 1; }
    double t_all = now_s() - t0;

    if (getenv("DUMPBM")) {
        if (class_mode) {
            uint64_t *tmp = (uint64_t *)malloc((size_t)cWords * 8 * NBc);
            cudaMemcpy(tmp, d_bm2, (size_t)cWords * 8 * NBc, cudaMemcpyDeviceToHost);
            FILE *f = fopen("/tmp/eng_bm.bin", "wb");
            fwrite(tmp, 1, (size_t)cWords * 8 * NBc, f);
            fclose(f);
            printf("[engine] dumped /tmp/eng_bm.bin (class, %llu bytes, NB=%llu)\n",
                   (unsigned long long)((size_t)cWords * 8 * NBc), (unsigned long long)NBc);
            free(tmp);
        } else {
        uint64_t *tmp = (uint64_t *)malloc(blockWords * 8);
        cudaMemcpy(tmp, d_bm, blockWords * 8, cudaMemcpyDeviceToHost);
        FILE *f = fopen("/tmp/eng_bm.bin", "wb");
        fwrite(tmp, 1, blockWords * 8, f);
        fclose(f);
        printf("[engine] dumped /tmp/eng_bm.bin (%llu bytes)\n", (unsigned long long)(blockWords * 8));
        free(tmp);
        }
    }

    unsigned long long stats[2] = {0,0};
    cudaMemcpy(stats, d_stats, 16, cudaMemcpyDeviceToHost);
    uint32_t ng = 0;
    cudaMemcpy(&ng, &d_out[0].gap, 4, cudaMemcpyDeviceToHost);
    if (getenv("DUMP_GAPS")) {
        uint32_t nd = ng < ((1u << 20) - 1u) ? ng : ((1u << 20) - 1u);
        GapRec *hg = (GapRec *)malloc((size_t)nd * sizeof(GapRec));
        if (nd) cudaMemcpy(hg, d_out + 1, (size_t)nd * sizeof(GapRec), cudaMemcpyDeviceToHost);
        FILE *f = fopen(getenv("DUMP_GAPS"), "w");
        if (f) {
            for (uint32_t i = 0; i < nd; i++) fprintf(f, "%llu %u\n", (unsigned long long)hg[i].slot, hg[i].gap);
            fclose(f);
        }
        free(hg);
        printf("[engine] dumped %u gaps to %s\n", nd, getenv("DUMP_GAPS"));
    }
    double N_report = class_mode ? (double)(NBc * blockNum30) : (double)N;
    double bps = N_report / t_all / 1e9;
    printf("[engine] total=%.3f s  -> %.1f B/s   (tests=%llu jumps=%llu tests/jump=%.2f gaps=%u)%s\n",
           t_all, bps, stats[0], stats[1], stats[1] ? (double)stats[0]/stats[1] : 0.0, ng,
           class_mode ? "  [CLASS30]" : "");
    (void)t_sieve;
    return 0;
}
