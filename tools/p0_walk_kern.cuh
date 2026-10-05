/*
 * p0_walk_kern.cuh - class-30 bitmap sieve + batched jump-walk kernels for
 * the Phase-0 scanner's --engine walk (moved verbatim from
 * tools/phase0_scan_gpu.cu so the Windows kernel-DLL split can compile them
 * without the host code).
 *
 * Ported from tools/walk_engine.cu (Phase-0 parity prototype; gates: sieve
 * VERIFY arbiter false=0 miss=0 of 35.8M slots, 22/22 set-gate vs the
 * odd-slot engine, NB=1 identical).
 * block = 30*2^25 = 1,006,632,960 numbers = 2^28 slots (32 MB bitmap),
 * tile = 2^19 slots (64 KB shared), wheel {7,11,13} as a precomputed group
 * pattern, item table = (prime, 64-mark chunk) rows with per-prime setup
 * reuse in contiguous per-thread runs.
 *
 * PORTABILITY: no __int128 anywhere - MSVC (the Windows host compiler nvcc
 * requires) has no 128-bit integer type, and the host pass of nvcc parses
 * this file.  The two 128-bit expressions this code needs are written with
 * __umul64hi + an explicit carry chain instead (identical results; the Linux
 * build compiles the same source).
 */
#ifndef P0_WALK_KERN_CUH
#define P0_WALK_KERN_CUH

#include <stdint.h>
#include "p0_types.h"   /* w_gaprec_t */
#include "perig.cuh"    /* ciosFermatTest128_hi */

__device__ __constant__ uint32_t c30_cls[8] = {1, 7, 11, 13, 17, 19, 23, 29};
__device__ __constant__ int8_t c30_clsidx[30];
__device__ __constant__ uint32_t c30_perm[30];
__device__ __constant__ uint32_t c30_pinv[30];

/* x * invp >> 64 (Granlund-Montgomery quotient estimate); was
   ((u128)x * (u128)invp) >> 64. */
__device__ static inline uint64_t t_fastmod64(uint64_t x, uint64_t p, uint64_t invp) {
    uint64_t q = __umul64hi(x, invp);
    uint64_t r = x - q * p;
    while (r >= p) r -= p;
    return r;
}

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
        /* tv = (Ahi:Alo) + ((tile*tileSlots)>>3)*30 + 1 as a 64-bit pair.
           The addend is < 2^35 (tile*tileSlots < 2^28), so the 128-bit sum
           is one carry-propagating add (was a u128 expression). */
        uint64_t t_lo = Alo + ((tile * tileSlots) >> 3) * 30u;
        uint64_t t_hi = Ahi + ((t_lo < Alo) ? 1ULL : 0ULL);
        t_lo += 1ULL;
        t_hi += (t_lo == 0ULL) ? 1ULL : 0ULL;
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
            for (int st = 0; st < 8; st++) {
                uint64_t j2 = (j + 1u) & 7u;
                uint64_t dm = (uint64_t)c30_cls[j2] - (uint64_t)c30_cls[j];
                if (j2 == 0u) dm += 30u;
                uint64_t ph2 = ph + dm * p;
                step[st] = (uint32_t)((ph2 / 30u) * 8u + (uint64_t)c30_clsidx[ph2 % 30u]
                                    - ((ph / 30u) * 8u + (uint64_t)c30_clsidx[ph % 30u]));
                ph = ph2; j = j2;
            }
            uint32_t cum = 0;
            for (int st = 0; st < 8; st++) { wpj[threadIdx.x * 8 + st] = (uint32_t)pos + cum; cum += step[st]; }
        }
        {
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
            uint32_t p32 = pos_base + 256u * cur_p * k0;
            if (p32 >= (uint32_t)tileSlots) continue;
            for (int r8 = 0; r8 < 32; r8++) {
                int done = 0;
#pragma unroll
                for (int st = 0; st < 8; st++) {
                    /* positions are strictly monotone (step >= +1), so once
                       past the tile the rest of the chunk is skipped instead
                       of walking 64 guarded iterations - the dominant item
                       cost at deep sieve depths, and identical marks. */
                    if (p32 >= (uint32_t)tileSlots) { done = 1; break; }
                    atomicOr(((unsigned int *)sh) + (p32 >> 5), 1u << (p32 & 31u));
                    p32 += step[st];
                }
                if (done) break;
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

/* class-30 walk: slot->value uses the class map, the window advance comes
   from wtab[class of anchor]; primality = Perig base-2 Euler test with the
   scan range's high word passed at runtime (ciosFermatTest128_hi). */
__global__ static void walk_kernel7(const uint64_t *__restrict__ bm, uint64_t nwords,
                                    uint64_t Alo, uint64_t Ahi,
                                    const uint32_t *__restrict__ wtab,
                                    uint32_t minGapVal,
                                    w_gaprec_t * __restrict__ out, uint32_t cap,
                                    unsigned long long * __restrict__ stats) {
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
        if (ciosFermatTest128_hi(v, Ahi)) break;
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
                if (ciosFermatTest128_hi(vq, Ahi)) {
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
        if (ciosFermatTest128_hi(vs, Ahi)) {
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

#endif /* P0_WALK_KERN_CUH */
