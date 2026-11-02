#pragma once
/* Base-2 Euler/Jacobi test for odd 2^105 <= n < 2^108.
   Four 32-bit limbs use fused product-scanning Montgomery steps with R=2^128.
   The test has the same predicate as perig; bench_u108 checks its residues
   against GMP and its verdicts against perig. */
#include "p0_test_u96.cuh"

namespace p0u108 {

using p0u96::u32;
using p0u96::u64;
using p0u96::fshl;
using p0u96::macc;
using p0u96::neginv32;

/* R=2^128, 2^105 <= n < 2^108.  If x < 2^109, a v-step
   (v <= 15) returns x^2/2^(128-v) + Q*n/2^(128-v) < n+2^105.
   Thus x < 2n and x < 2^109 are invariants. */
template <bool VAR>
P0U_HD void step(u32 &x0, u32 &x1, u32 &x2, u32 &x3,
                 u32 n0, u32 n1, u32 n2, u32 n3, u32 np, u32 v)
{
    const u32 a0 = x0, a1 = x1, a2 = x2, a3 = x3;
    const u32 twice_a3 = a3 << 1;
    u32 L, H, C, q0, q1, q2, q3;
    const u64 p = (u64)a0 * a0;
    L = (u32)p; H = (u32)(p >> 32);
    q0 = L * np;
    C = 0;
    macc(L, H, C, q0, n0);
    L = H; H = C; C = 0;
    macc(L, H, C, a0, a1);
    macc(L, H, C, a0, a1);
    macc(L, H, C, q0, n1);
    q1 = L * np;
    macc(L, H, C, q1, n0);
    L = H; H = C; C = 0;
    macc(L, H, C, a1, a1);
    macc(L, H, C, a0, a2);
    macc(L, H, C, a0, a2);
    macc(L, H, C, q0, n2);
    macc(L, H, C, q1, n1);
    q2 = L * np;
    macc(L, H, C, q2, n0);
    L = H; H = C; C = 0;
    macc(L, H, C, a0, twice_a3);
    macc(L, H, C, a1, a2);
    macc(L, H, C, a1, a2);
    macc(L, H, C, q0, n3);
    macc(L, H, C, q1, n2);
    macc(L, H, C, q2, n1);
    q3 = L * np;
    if (VAR) q3 &= 0xffffffffu >> v;
    macc(L, H, C, q3, n0);
    const u32 w3 = L;
    L = H; H = C; C = 0;
    macc(L, H, C, a1, twice_a3);
    macc(L, H, C, a2, a2);
    macc(L, H, C, q1, n3);
    macc(L, H, C, q2, n2);
    macc(L, H, C, q3, n1);
    const u32 w4 = L;
    L = H; H = C; C = 0;
    macc(L, H, C, a2, twice_a3);
    macc(L, H, C, q2, n3);
    macc(L, H, C, q3, n2);
    const u32 w5 = L;
    L = H; H = C; C = 0;
    macc(L, H, C, a3, a3);
    macc(L, H, C, q3, n3);
    const u32 w6 = L, w7 = H;
    if (VAR) {
        x0 = fshl(w3, w4, v);
        x1 = fshl(w4, w5, v);
        x2 = fshl(w5, w6, v);
        x3 = fshl(w6, w7, v);
    } else {
        x0 = w4; x1 = w5; x2 = w6; x3 = w7;
    }
}

P0U_HD void csub4(u32 &r0, u32 &r1, u32 &r2, u32 &r3,
                  u32 n0, u32 n1, u32 n2, u32 n3)
{
    u64 d0 = (u64)r0 - n0;
    u64 d1 = (u64)r1 - n1 - (u32)(d0 >> 63);
    u64 d2 = (u64)r2 - n2 - (u32)(d1 >> 63);
    u64 d3 = (u64)r3 - n3 - (u32)(d2 >> 63);
    if ((d3 >> 63) == 0) {
        r0 = (u32)d0; r1 = (u32)d1; r2 = (u32)d2; r3 = (u32)d3;
    }
}

P0U_HD void from_mont(u32 &x0, u32 &x1, u32 &x2, u32 &x3,
                      u32 n0, u32 n1, u32 n2, u32 n3, u32 np)
{
    u32 s[9] = {x0, x1, x2, x3, 0, 0, 0, 0, 0};
    const u32 nn[4] = {n0, n1, n2, n3};
    for (int i = 0; i < 4; i++) {
        u32 q = s[i] * np;
        u64 c = 0;
        for (int j = 0; j < 4; j++) {
            u64 t = (u64)q * nn[j] + s[i + j] + c;
            s[i + j] = (u32)t; c = t >> 32;
        }
        for (int j = i + 4; j < 9; j++) {
            u64 t = (u64)s[j] + c;
            s[j] = (u32)t; c = t >> 32;
        }
    }
    x0 = s[4]; x1 = s[5]; x2 = s[6]; x3 = s[7];
}

P0U_HD void window(u32 &y0, u32 &y1, u32 &y2, u32 &y3,
                   u32 n0, u32 n1, u32 n2, u32 n3, u32 np, u32 v)
{
    step<false>(y0, y1, y2, y3, n0, n1, n2, n3, np, 0);
    step<false>(y0, y1, y2, y3, n0, n1, n2, n3, np, 0);
    step<false>(y0, y1, y2, y3, n0, n1, n2, n3, np, 0);
    step<true>(y0, y1, y2, y3, n0, n1, n2, n3, np, v);
}

P0U_HD bool test(u32 n0, u32 n1, u32 n2, u32 n3, u32 *resid = 0)
{
    const u32 np = neginv32(n0);
    const u64 mlo = ((u64)n1 << 32) | n0;
    const u64 mhi = ((u64)n3 << 32) | n2;
    u64 rlo, rhi;
    ciosConstants128(mlo, mhi, &rlo, &rhi);
    u32 y0 = (u32)rlo, y1 = (u32)(rlo >> 32);
    u32 y2 = (u32)rhi, y3 = (u32)(rhi >> 32);
    const u32 r0 = y0, r1 = y1, r2 = y2, r3 = y3;
    const u32 e0 = fshl(n0, n1, 31), e1 = fshl(n1, n2, 31);
    const u32 e2 = fshl(n2, n3, 31), e3 = n3 >> 1;
    const u32 seed = e3 >> 8;
    for (u32 i = 0; i < seed; i++) {
        y3 = fshl(y2, y3, 1); y2 = fshl(y1, y2, 1);
        y1 = fshl(y0, y1, 1); y0 <<= 1;
        csub4(y0, y1, y2, y3, n0, n1, n2, n3);
    }
    u32 w = e3 << 24;
#pragma unroll 1
    for (int k = 0; k < 2; k++) {
        u32 v = w >> 28; w <<= 4;
        window(y0, y1, y2, y3, n0, n1, n2, n3, np, v);
    }
    w = e2;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        u32 v = w >> 28; w <<= 4;
        window(y0, y1, y2, y3, n0, n1, n2, n3, np, v);
    }
    w = e1;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        u32 v = w >> 28; w <<= 4;
        window(y0, y1, y2, y3, n0, n1, n2, n3, np, v);
    }
    w = e0;
#pragma unroll 1
    for (int k = 0; k < 8; k++) {
        u32 v = w >> 28; w <<= 4;
        window(y0, y1, y2, y3, n0, n1, n2, n3, np, v);
    }
    if (resid) {
        u32 z0 = y0, z1 = y1, z2 = y2, z3 = y3;
        from_mont(z0, z1, z2, z3, n0, n1, n2, n3, np);
        csub4(z0, z1, z2, z3, n0, n1, n2, n3);
        resid[0] = z0; resid[1] = z1; resid[2] = z2; resid[3] = z3;
    }
    csub4(y0, y1, y2, y3, n0, n1, n2, n3);
    u32 t0 = r0, t1 = r1, t2 = r2, t3 = r3;
    if (((n0 >> 1) ^ (n0 >> 2)) & 1) {
        u64 d0 = (u64)n0 - t0;
        u64 d1 = (u64)n1 - t1 - (u32)(d0 >> 63);
        u64 d2 = (u64)n2 - t2 - (u32)(d1 >> 63);
        u64 d3 = (u64)n3 - t3 - (u32)(d2 >> 63);
        t0 = (u32)d0; t1 = (u32)d1; t2 = (u32)d2; t3 = (u32)d3;
    }
    return (y0 == t0) & (y1 == t1) & (y2 == t2) & (y3 == t3);
}

P0U_HD bool test_hi(u64 lo, u64 hi)
{
    return test((u32)lo, (u32)(lo >> 32), (u32)hi, (u32)(hi >> 32));
}

} /* namespace p0u108 */
