#pragma once
/* p0_test_u96.cuh - OUR Euler/Jacobi base-2 test for the wall band
 *     2^95 <= n <= 2^96 - 2^79 - 1      (Ahi = n >> 64 in [2^31, 2^32 - 2^15))
 *
 * WHAT IT IS
 *   The same predicate as perig (tools/perig.cuh):
 *       2^((n-1)/2) mod n  ==  (+1) or (-1), the sign chosen by the Jacobi
 *       bit ((n>>1) ^ (n>>2)) & 1,
 *   for a 96-bit modulus held as three 32-bit limbs, evaluated with a lazy
 *   Montgomery representative (R = 2^128).  Verdicts are bit-for-bit those of
 *   perig and of GMP (tools/bench_u96.cu gates both).
 *
 * WHY IT IS FASTER THAN perig HERE (measured 2026-10-30, RTX 3070, sm_86)
 *   perig works on two 64-bit limbs: ~96 SASS instructions per modular
 *   squaring (30 IMAD.WIDE, 36 registers).  This version needs ~43:
 *     - one FUSED column-wise (product-scanning) square + Montgomery reduction:
 *       9 square products + 12 reduction products + 4 quotient digits;
 *     - every product is accumulated into a (lo, hi) pair with the PTX triple
 *           mad.lo.cc.u32 / madc.hi.cc.u32 / addc.u32 cw,cw,0
 *       which ptxas fuses into ONE in-place IMAD.WIDE.U32 with the carry-out
 *       left in a predicate; two consecutive carry counters merge into one
 *       IADD3.X.  (The plain-C++ idiom  s = acc + p; cw += (s < acc)  is far
 *       worse: ~2x the instructions, see lab notes in the PGN review doc.);
 *     - the last squaring of every 4-bit exponent window folds the window's
 *       "multiply by 2^v" into the Montgomery constant (see step<true>), so
 *       the exponent costs 92 squarings and no multiplications.
 *   Measured standalone (1M consecutive odd candidates at Ahi = 4282402316,
 *   `bench_u96 time 1048576 200`, 1920 MHz): 0.91 ns/test vs perig 1.57 ns/test
 *   (1.70-1.72x; absolutes drift with the clock state, the ratio does not),
 *   26 vs 38 registers.
 *
 * INVARIANTS (all columns are computed EXACTLY - there is no carry-fold
 * assumption beyond the lazy bound)
 *   - n odd; np = -n^-1 mod 2^32.
 *   - x < n + 2^79 < 2^96 on entry and exit of every step: a step returns
 *     (x^2 + Q n) / 2^(128 - v) with Q < 2^(128 - v), hence
 *         out < x^2 / 2^(128 - v) + n < 2^(64 + v) + n <= n + 2^79   (v <= 15)
 *     which fits three limbs iff n < 2^96 - 2^79.  That is the ONLY upper
 *     domain limit; above it callers must use perig (exact to 2^120).
 *   - v-step: q3 is masked to (32 - v) bits, so x^2 + Q n == 0 mod 2^(128-v)
 *     and the division is exact (a funnel shift).
 *   - exponent e = (n-1)/2 = n >> 1 has 95 bits for n >= 2^95: the top 3 bits
 *     seed the accumulator by conditional doublings of R mod n, the other 92
 *     bits are 23 windows of 4 bits.  Below 2^95 the seed value is just
 *     smaller (0..3 instead of 4..7), so the verdicts stay exact (probed
 *     2026-10-30: 2M consecutive odd n at each of 8 high words from 2^70 to
 *     2^95, 0 differences vs perig), but the fixed 92 squarings are then more
 *     than the exponent needs; the production domain therefore stays n >= 2^95.
 *
 * NO code from the PGN snapshot is used here.  The ideas taken from reading
 * it are only "specialise per bit width" and "hand-schedule the carries";
 * the algorithm (product scanning + variable-R window) and the code are ours.
 */

#include <stdint.h>
#include "perig.cuh"   /* ciosConstants128: R mod n */

#ifdef __CUDACC__
#define P0U_HD __host__ __device__ __forceinline__
#else
#define P0U_HD inline
#endif

namespace p0u96 {

typedef uint32_t u32;
typedef uint64_t u64;

P0U_HD u32 fshl(u32 lo, u32 hi, u32 s)
{
#ifdef __CUDA_ARCH__
    return __funnelshift_l(lo, hi, s);
#else
    s &= 31;
    return s ? (hi << s) | (lo >> (32 - s)) : hi;
#endif
}

/* -n0^-1 mod 2^32 (n0 odd) */
P0U_HD u32 neginv32(u32 n0)
{
    u32 x = (3u * n0) ^ 2u;      /* 5 bits */
    x *= 2u - n0 * x;            /* 10 */
    x *= 2u - n0 * x;            /* 20 */
    x *= 2u - n0 * x;            /* 40 >= 32 */
    return 0u - x;
}

/* (hi:lo) += a*b ; the carry out of the 64-bit add is counted in cw.
   The accumulator pair must be updated IN PLACE for ptxas to emit a single
   IMAD.WIDE.U32 (see the header comment). */
P0U_HD void macc(u32 &lo, u32 &hi, u32 &cw, u32 a, u32 b)
{
#ifdef __CUDA_ARCH__
    asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\t"
        "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
        "addc.u32 %2, %2, 0;"
        : "+r"(lo), "+r"(hi), "+r"(cw) : "r"(a), "r"(b));
#else
    u64 acc = ((u64)hi << 32) | lo;
    u64 s = acc + (u64)a * b;
    cw += (s < acc);
    lo = (u32)s; hi = (u32)(s >> 32);
#endif
}

/* out = (x^2 + Q n) / 2^(128 - v)   (v = 0 unless VAR).  Q is chosen digit by
   digit so that the low 128 - v bits cancel. */
template <bool VAR>
P0U_HD void step(u32 &x0, u32 &x1, u32 &x2,
                 u32 n0, u32 n1, u32 n2, u32 np, u32 v)
{
    const u32 a0 = x0, a1 = x1, a2 = x2;
    u32 L, H, C, q0, q1, q2, q3;
    u64 p = (u64)a0 * a0;
    L = (u32)p; H = (u32)(p >> 32);
    q0 = L * np;
    C = 0;
    macc(L, H, C, q0, n0);               /* column 0: L becomes 0 */
    L = H; H = C; C = 0;                 /* column 1 */
    macc(L, H, C, a0, a1);
    macc(L, H, C, a0, a1);
    macc(L, H, C, q0, n1);
    q1 = L * np;
    macc(L, H, C, q1, n0);
    L = H; H = C; C = 0;                 /* column 2 */
    macc(L, H, C, a1, a1);
    macc(L, H, C, a0, a2);
    macc(L, H, C, a0, a2);
    macc(L, H, C, q0, n2);
    macc(L, H, C, q1, n1);
    q2 = L * np;
    macc(L, H, C, q2, n0);
    L = H; H = C; C = 0;                 /* column 3 */
    macc(L, H, C, a1, a2);
    macc(L, H, C, a1, a2);
    macc(L, H, C, q1, n2);
    macc(L, H, C, q2, n1);
    q3 = L * np;
    if (VAR) q3 &= 0xffffffffu >> v;
    macc(L, H, C, q3, n0);
    const u32 w3 = L;
    L = H; H = C; C = 0;                 /* column 4 */
    macc(L, H, C, a2, a2);
    macc(L, H, C, q2, n2);
    macc(L, H, C, q3, n1);
    const u32 w4 = L;
    /* column 5: the result is < 2^96, so this cannot overflow 64 bits */
    u64 t = (((u64)C << 32) | H) + (u64)q3 * n2;
    const u32 w5 = (u32)t, w6 = (u32)(t >> 32);
    if (VAR) {
        x0 = fshl(w3, w4, v);
        x1 = fshl(w4, w5, v);
        x2 = fshl(w5, w6, v);
    } else {
        x0 = w4; x1 = w5; x2 = w6;
    }
}

/* x * R^-1 mod n (lazy).  Plain C++: used only for the residue gate. */
P0U_HD void from_mont(u32 &x0, u32 &x1, u32 &x2,
                      u32 n0, u32 n1, u32 n2, u32 np)
{
    u32 s[8] = {x0, x1, x2, 0, 0, 0, 0, 0};
    const u32 nn[3] = {n0, n1, n2};
    for (int i = 0; i < 4; i++) {
        u32 q = s[i] * np;
        u64 c = 0;
        for (int j = 0; j < 3; j++) {
            u64 tt = (u64)q * nn[j] + s[i + j] + c;
            s[i + j] = (u32)tt; c = tt >> 32;
        }
        for (int j = i + 3; j < 8; j++) {
            u64 tt = (u64)s[j] + c;
            s[j] = (u32)tt; c = tt >> 32;
        }
    }
    x0 = s[4]; x1 = s[5]; x2 = s[6];
}

/* r -= n if r >= n  (r < 2n assumed) */
P0U_HD void csub3(u32 &r0, u32 &r1, u32 &r2, u32 n0, u32 n1, u32 n2)
{
    u64 d0 = (u64)r0 - n0;
    u64 d1 = (u64)r1 - n1 - (u32)(d0 >> 63);
    u64 d2 = (u64)r2 - n2 - (u32)(d1 >> 63);
    if ((d2 >> 63) == 0) { r0 = (u32)d0; r1 = (u32)d1; r2 = (u32)d2; }
}

/* n = n0 + n1*2^32 + n2*2^64, odd, 2^95 <= n <= 2^96 - 2^79 - 1.
   If resid != 0 it receives 2^e mod n (fully reduced) for the residue gate. */
P0U_HD bool test(u32 n0, u32 n1, u32 n2, u32 *resid = 0)
{
    const u32 np = neginv32(n0);
    const u64 m_lo = ((u64)n1 << 32) | n0;
    u64 r_lo, r_hi;
    ciosConstants128(m_lo, (u64)n2, &r_lo, &r_hi);      /* R mod n */
    u32 y0 = (u32)r_lo, y1 = (u32)(r_lo >> 32), y2 = (u32)r_hi;
    const u32 r10 = y0, r11 = y1, r12 = y2;              /* 1 in Montgomery form */
    const u32 e0 = fshl(n0, n1, 31), e1 = fshl(n1, n2, 31), e2 = n2 >> 1;
    /* seed: the top 3 exponent bits P (4..7) -> 2^P in Montgomery form */
    const u32 P = e2 >> 28;
    for (u32 i = 0; i < P; i++) {
        u32 c = y2 >> 31;
        y2 = fshl(y1, y2, 1); y1 = fshl(y0, y1, 1); y0 <<= 1;
        u64 d0 = (u64)y0 - n0;
        u64 d1 = (u64)y1 - n1 - (u32)(d0 >> 63);
        u64 d2 = (u64)y2 - n2 - (u32)(d1 >> 63);
        bool ge = c | ((d2 >> 63) == 0);
        if (ge) { y0 = (u32)d0; y1 = (u32)d1; y2 = (u32)d2; }
    }
    u32 w = e2 << 4;                                     /* 28 bits left in e2 */
#pragma unroll 1
    for (int k = 0; k < 7; k++) {
        u32 v = w >> 28; w <<= 4;
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<true >(y0, y1, y2, n0, n1, n2, np, v);
    }
    w = e1;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        u32 v = w >> 28; w <<= 4;
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<true >(y0, y1, y2, n0, n1, n2, np, v);
    }
    w = e0;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        u32 v = w >> 28; w <<= 4;
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<false>(y0, y1, y2, n0, n1, n2, np, 0);
        step<true >(y0, y1, y2, n0, n1, n2, np, v);
    }
    if (resid) {
        u32 z0 = y0, z1 = y1, z2 = y2;
        from_mont(z0, z1, z2, n0, n1, n2, np);
        csub3(z0, z1, z2, n0, n1, n2);
        resid[0] = z0; resid[1] = z1; resid[2] = z2;
    }
    csub3(y0, y1, y2, n0, n1, n2);
    /* compare with +1 (r1) or -1 (n - r1) in Montgomery form */
    u32 t0 = r10, t1 = r11, t2 = r12;
    if (((n0 >> 1) ^ (n0 >> 2)) & 1) {
        u64 d0 = (u64)n0 - t0;
        u64 d1 = (u64)n1 - t1 - (u32)(d0 >> 63);
        u64 d2 = (u64)n2 - t2 - (u32)(d1 >> 63);
        t0 = (u32)d0; t1 = (u32)d1; t2 = (u32)d2;
    }
    return (y0 == t0) & (y1 == t1) & (y2 == t2);
}

/* walk entry: n = hi*2^64 + v */
P0U_HD bool test_hi(u64 v, u32 hi)
{
    return test((u32)v, (u32)(v >> 32), hi);
}

} /* namespace p0u96 */
