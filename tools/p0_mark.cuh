/*
 * p0_mark.cuh - GPU bitmap-marking kernel for the Phase-0 scanner geometry.
 * Shared by tools/bench_p0sieve.cu (rate probe + bit-exact verification) and
 * the integrated path in tools/phase0_scan_gpu.cu (--gpu-sieve).
 *
 * Geometry: a segment of up to 2^seg_bits integers = up to 2^(seg_bits-1)
 * odd-value SLOTS, one bit per slot.  Slot s of a segment whose first (odd)
 * candidate is v0 holds the value v0 + 2s.  Bit set = "value divisible by a
 * prime <= P".
 *
 * Work items: one item per (prime, P0_CHUNK-hit chunk) so no thread carries a
 * long serial chain (the gapminer mark-kernel straggler lesson: one-thread-
 * per-prime was 25-38x slower; chunked items fixed it).
 *
 * Kernel per item: first hit slot s0 solves v0 + 2s = 0 (mod p), i.e.
 * s0 = (-v0 mod p) * inv2 (mod p) with inv2 = (p+1)/2; v0 mod p comes from
 * its two 64-bit words and a host-precomputed r64 = 2^64 mod p.  Then fire-
 * and-forget atomicOr over up to P0_CHUNK slots s0 + k*p.
 *
 * Redundant-mark filter (2026-10-01, extended): a slot whose value v0+2s is
 * divisible by ANY wheel prime {3,5,7,11,13} is already marked by the wheel,
 * so items for primes p >= 17 SKIP those slots.  Production always runs the
 * wheel first, so the bitmap is unchanged; the filter removes ~62 % of the
 * marking work (keep fraction 2/3*4/5*6/7*10/11*12/13 = 38.4 %).  Measured
 * 2026-10-01 (RTX 3070): probe 243 -> 173 us/segment; scanner 1-thread event
 * split wheel+mark 0.134 -> 0.106 s per 1e10; 1e11 marginal A/B 0.605 ->
 * 0.585 s per 1e10 (-3.5 %); cross-binary log diff 0 lines; bench --verify
 * bit-exact.  The mark kernel is atomicOr-throughput-bound (~40 G/s): time
 * tracks the atomic COUNT, not the ALU work.  Primes <= 13 are never filtered -
 * a caller without a wheel (the probe) relies on their marks.
 *
 * Include from .cu translation units only.
 */

#ifndef P0_MARK_CUH
#define P0_MARK_CUH

#include <stdint.h>
#include "p0_types.h"   /* P0Item, P0_CHUNK, p0_vis_mask30 (host-safe) */

/* x mod p for x < 2^64 via a host-precomputed reciprocal invp = (2^64-1)/p
   (Granlund-Montgomery: q is off by at most 1, one fixup suffices). */
__device__ static __forceinline__ uint64_t p0_fastmod64(uint64_t x, uint64_t p,
                                                         uint64_t invp) {
    uint64_t q = __umul64hi(x, invp);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}

__global__ static void p0_mark_kernel(const P0Item * __restrict__ items,
                                      uint32_t nitems,
                                      const uint64_t * __restrict__ primes,
                                      const uint64_t * __restrict__ r64,
                                      const uint64_t * __restrict__ invp,
                                      uint64_t half,
                                      uint64_t v0_lo, uint64_t v0_hi,
                                      uint32_t vis_mask30,
                                      uint64_t * __restrict__ bm) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nitems) return;
    uint32_t pi = items[i].pidx;
    uint64_t p = primes[pi];
    uint64_t pv = invp[pi];
    uint64_t vh = p0_fastmod64(v0_hi, p, pv);
    uint64_t vl = p0_fastmod64(v0_lo, p, pv);
    uint64_t vm = p0_fastmod64(vh * r64[pi] + vl, p, pv);
    uint64_t d = vm ? (p - vm) : 0;            /* -v0 mod p */
    uint64_t s0 = p0_fastmod64(d * ((p + 1) >> 1), p, pv);
    uint64_t s = s0 + (uint64_t)items[i].k0 * p;
    if (p > 13u) {
        /* Skip every hit whose slot the wheel already covers: 3,5 via
           vis_mask30 (host-built for this v0); 7/11/13 via the slot VALUE
           v = v0+2s modulo q, tracked incrementally with step +2p.  The test
           MUST be on the value - an s-mod-q form skipped the wrong slots and
           silently lost marks (caught by bench_p0sieve --verify). */
        uint32_t p30 = (uint32_t)(p % 30u);
        uint32_t r = (uint32_t)(s % 30u);
        /* v0 mod q from its two 64-bit words; R64_q = 2^64 mod q = 2/5/3 */
        uint32_t v07 = (uint32_t)(((v0_hi % 7u) * 2u + (v0_lo % 7u)) % 7u);
        uint32_t v011 = (uint32_t)(((v0_hi % 11u) * 5u + (v0_lo % 11u)) % 11u);
        uint32_t v013 = (uint32_t)(((v0_hi % 13u) * 3u + (v0_lo % 13u)) % 13u);
        uint32_t c7 = (uint32_t)((2u * (p % 7u)) % 7u);
        uint32_t c11 = (uint32_t)((2u * (p % 11u)) % 11u);
        uint32_t c13 = (uint32_t)((2u * (p % 13u)) % 13u);
        uint32_t w7 = (uint32_t)((v07 + 2u * (uint32_t)(s % 7u)) % 7u);
        uint32_t w11 = (uint32_t)((v011 + 2u * (uint32_t)(s % 11u)) % 11u);
        uint32_t w13 = (uint32_t)((v013 + 2u * (uint32_t)(s % 13u)) % 13u);
        for (uint32_t j = 0; j < P0_CHUNK && s < half; j++, s += p) {
            if (((vis_mask30 >> r) & 1u) && w7 && w11 && w13)
                atomicOr((unsigned long long *)&bm[s >> 6],
                         (unsigned long long)(1ULL << (s & 63u)));
            r += p30;
            if (r >= 30u) r -= 30u;
            w7 += c7; if (w7 >= 7u) w7 -= 7u;
            w11 += c11; if (w11 >= 11u) w11 -= 11u;
            w13 += c13; if (w13 >= 13u) w13 -= 13u;
        }
    } else {
        for (uint32_t j = 0; j < P0_CHUNK && s < half; j++, s += p) {
            atomicOr((unsigned long long *)&bm[s >> 6],
                     (unsigned long long)(1ULL << (s & 63u)));
        }
    }
}

/* host: 30-bit mask of slot residues s%30 whose value v0+2s is NOT divisible
   by 3 or 5 (the slots the {3,5} wheel covers); pass v0 mod 30.  Defined in
   the host-safe p0_types.h (see there). */

/* Wheel fill: write the {3,5,7,11,13} pattern into the segment bitmap.
   Doubles as the memset - every word the walk reads is written here.
   wpat2 = doubled un-rotated pattern (bit c set iff c mod 3|5|7|11|13 == 0),
   period wheel_p slots; inv2 = inverse of 2 mod wheel_p; r64p = 2^64 mod p.
   Rotation: shift = (v0 mod P) * inv2 mod P (v0 = segment's first odd value,
   split in two 64-bit words), so bit i means "v0 + 2i divisible by a wheel
   prime".  Matches the CPU tile bit-for-bit (verified by --check). */
__global__ static void p0_wheel_kernel(const uint64_t * __restrict__ wpat2,
                                       uint32_t wheel_p, uint32_t inv2,
                                       uint64_t r64p,
                                       uint64_t v0_lo, uint64_t v0_hi,
                                       uint32_t nwords,
                                       uint64_t * __restrict__ bm) {
    uint32_t wi = blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= nwords) return;
    uint64_t vh = v0_hi % wheel_p, vl = v0_lo % wheel_p;
    uint64_t vmod = (vh * r64p + vl) % wheel_p;
    uint32_t shift = (uint32_t)((vmod * inv2) % wheel_p);
    uint32_t o = (uint32_t)(((uint64_t)wi * 64u + shift) % wheel_p);
    uint32_t b = o & 63u, src = o >> 6;
    uint64_t w = wpat2[src] >> b;
    if (b) w |= wpat2[src + 1] << (64 - b);
    bm[wi] = w;
}

#endif /* P0_MARK_CUH */
