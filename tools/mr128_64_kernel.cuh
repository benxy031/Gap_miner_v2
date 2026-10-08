/*
 * mr128_64_kernel.cuh - exact base-2 strong probable prime test for
 * 2^96 < n < 2^128 as a 2 x 64-bit CIOS Montgomery kernel.
 *
 * WHY: the 4 x 32-bit kernel (mr128_kernel.cuh) is exact but pays 16
 * 32x32->64 products per squaring.  The walk engine's perig test does the same
 * job with 2 x 64-bit limbs in far fewer instructions - measured 1.67x faster
 * per test (tools/mr128_bench.cu) - but its MSB-first ladder is only exact up
 * to 120 bits (the Montgomery constant 2^128 mod n is fine; see
 * docs/PHASE0_scan_bench.md § 25.2).  This header keeps the wide-limb
 * arithmetic and replaces the exponential logic with the exact staircase of
 * mr128_kernel.cuh:
 *   - ONE = 2^128 mod n from a double-precision quotient estimate plus one
 *     correction of n,
 *   - d = (n-1) >> s built and normalised explicitly, walked with a
 *     left-to-right ladder that uses a doubling for zero bits,
 *   - the standard strong-probable-prime verdict on the squaring sequence.
 *
 * Arithmetic: 64x64->128 through uint128_t (native __int128 where the compiler
 * has it, the portable shim in perig.cuh where it does not).  The products
 * cannot overflow the 128-bit accumulator: with 64-bit limbs,
 * a*b + t + c <= (2^64-1)^2 + 2*(2^64-1) = 2^128 - 1.
 *
 * Limit: 2^96 < n < 2^128, n odd.  q = floor(2^128/n) must fit the quotient
 * estimate, i.e. n > 2^96; below that the 3 x 32-bit kernel covers the range.
 *
 * Include from .cu translation units, or from a host harness with plain g++
 * (perig.cuh defines the CUDA qualifiers away outside nvcc).
 */

#ifndef MR128_64_KERNEL_CUH
#define MR128_64_KERNEL_CUH

#include <stdint.h>
#include "perig.cuh"     /* uint128_t (native or shim), montgomeryInverse64 */

/* d = a - b - bin, returns the borrow out.  Branchless on purpose: the
   conditional form pushed this kernel into local memory (48 B/thread), and the
   spill traffic cost more than the extra integer ops. */
__device__ __forceinline__ static uint64_t mr128_64_sub(uint64_t a, uint64_t b,
                                                        uint64_t bin, uint64_t *d) {
    *d = a - b - bin;
    return (uint64_t)((a < b) | (bin & (a == b)));
}

__device__ __forceinline__ static int mr128_64_clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __clzll(x);
#else
    int c = 0;
    while (c < 64 && !(x & (1ULL << 63))) { x <<= 1; c++; }
    return c;
#endif
}

/* high 64 bits of a*b: the dedicated mul.hi.u64 on the GPU (the __int128
   form compiles both halves even when the low half is discarded) */
__device__ __forceinline__ static uint64_t mr128_64_mulhi(uint64_t a, uint64_t b) {
#if defined(__CUDA_ARCH__)
    return __umul64hi(a, b);
#else
    return (uint64_t)(((uint128_t)a * (uint128_t)b) >> 64);
#endif
}

/* r = a*b*R^-1 mod n, R = 2^128, result < n; n0inv = -n^-1 mod 2^64.
   The same CIOS staircase as mr68/mr128 (GMP-validated there) in 64-bit
   limbs: the 4-word accumulator holds the running sum (< 2^256). */
__device__ __forceinline__ static void mr128_64_mont_mul(uint64_t *r,
                                                         const uint64_t *a,
                                                         const uint64_t *b,
                                                         const uint64_t *n,
                                                         uint64_t n0inv) {
    /* scalar accumulator (an array with constant indices still ended up in
       local memory on this toolchain: 48 B/thread) */
    uint64_t t0 = 0U, t1 = 0U, t2 = 0U, t3 = 0U;
    uint128_t p, s;

#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint64_t bi = b[i];
        uint64_t c;
        p = (uint128_t)a[0] * (uint128_t)bi + (uint128_t)t0;
        t0 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        p = (uint128_t)a[1] * (uint128_t)bi + (uint128_t)t1 + (uint128_t)c;
        t1 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        s = (uint128_t)t2 + (uint128_t)c;
        t2 = (uint64_t)s;
        t3 += (uint64_t)(s >> 64);

        const uint64_t m = t0 * n0inv;
        uint128_t q = (uint128_t)m * (uint128_t)n[0] + (uint128_t)t0;
        c = (uint64_t)(q >> 64);                 /* t0 cancels exactly */
        p = (uint128_t)m * (uint128_t)n[1] + (uint128_t)t1 + (uint128_t)c;
        t0 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        s = (uint128_t)t2 + (uint128_t)c;
        t1 = (uint64_t)s;
        t2 = t3 + (uint64_t)(s >> 64);
        t3 = 0U;
    }
    uint64_t sub0, sub1, b0, b1;
    b0 = mr128_64_sub(t0, n[0], 0U, &sub0);
    b1 = mr128_64_sub(t1, n[1], b0, &sub1);
    int need = (t2 != 0U) | (b1 == 0U);
    r[0] = need ? sub0 : t0;
    r[1] = need ? sub1 : t1;
}

/* r = 2*a mod n */
__device__ __forceinline__ static void mr128_64_dbl_mod(uint64_t *r,
                                                        const uint64_t *a,
                                                        const uint64_t *n) {
    uint64_t ovf = a[1] >> 63;
    uint64_t t0 = a[0] << 1;
    uint64_t t1 = (a[1] << 1) | (a[0] >> 63);
    uint64_t sub0, sub1, b0, b1;
    b0 = mr128_64_sub(t0, n[0], 0U, &sub0);
    b1 = mr128_64_sub(t1, n[1], b0, &sub1);
    int need = (ovf != 0U) | (b1 == 0U);
    r[0] = need ? sub0 : t0;
    r[1] = need ? sub1 : t1;
}

/* ONE = 2^128 mod n (n > 2^96 keeps q = floor(2^128/n) far below 2^64) */
__device__ __forceinline__ static void mr128_64_one(uint64_t *one,
                                                    const uint64_t *n) {
    double dn = (double)n[1] * 18446744073709551616.0 + (double)n[0];
    uint64_t qh = (uint64_t)(340282366920938463463374607431768211456.0 / dn);
    uint128_t cy = 0;
    uint64_t qn[3];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        uint128_t p = (uint128_t)qh * (uint128_t)n[j] + cy;
        qn[j] = (uint64_t)p;
        cy = p >> 64;
    }
    qn[2] = (uint64_t)cy;                      /* qh*n < 2^32 * 2^128 */
    uint64_t ovf = qn[2];
    uint64_t b0, b1, o0, o1;
    b0 = mr128_64_sub(0U, qn[0], 0U, &o0);
    b1 = mr128_64_sub(0U, qn[1], b0, &o1);
    (void)b1;
    one[0] = o0;
    one[1] = o1;
    if (ovf) {                                 /* overshoot: one += n */
        uint64_t c2 = 0;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint128_t s = (uint128_t)one[j] + (uint128_t)n[j] + (uint128_t)c2;
            one[j] = (uint64_t)s;
            c2 = (uint64_t)(s >> 64);
        }
    } else {
        uint64_t s0, s1, d0, d1;               /* residual may be >= n */
        d0 = mr128_64_sub(one[0], n[0], 0U, &s0);
        d1 = mr128_64_sub(one[1], n[1], d0, &s1);
        if (!d1) { one[0] = s0; one[1] = s1; }
    }
}

/* base-2 strong probable prime on 2 x 64-bit limbs: 0 = composite,
   1 = probable prime (n odd, 2^96 < n < 2^128). */
__device__ static int mr128_64_base2(const uint64_t *nin) {
    uint64_t n[2], one[2], neg[2], two[2], y[2];
    uint64_t n0inv = montgomeryInverse64(nin[0]);
    n[0] = nin[0];
    n[1] = nin[1];

    int b = (64 - mr128_64_clz64(n[1])) + 64;   /* bit length of n */

    mr128_64_one(one, n);
    {
        uint64_t b0, nb1;
        b0 = mr128_64_sub(n[0], one[0], 0U, &neg[0]);
        nb1 = mr128_64_sub(n[1], one[1], b0, &neg[1]);
        (void)nb1;
    }
    mr128_64_dbl_mod(two, one, n);

    uint64_t nm1[2];
    nm1[1] = n[1] - ((n[0] == 0U) ? 1U : 0U);
    nm1[0] = n[0] - 1U;
    int s = 0;                                   /* n-1 = 2^s * d */
    {
        uint64_t w = nm1[0];
        if (w == 0U) { s = 64; w = nm1[1]; }
        while ((w & 1U) == 0U) { w >>= 1; s++; }
    }
    uint64_t dsh[2];
    {
        int sh = s & 63, wi = s >> 6;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint64_t lo = (j + wi < 2) ? nm1[j + wi] : 0U;
            uint64_t hi = (j + wi + 1 < 2) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint64_t)((lo >> sh) | (hi << (64 - sh)));
        }
    }
    int dbits = b - s;
    while (dbits > 0 &&
           ((dsh[(dbits - 1) >> 6] >> ((dbits - 1) & 63)) & 1ULL) == 0U)
        dbits--;

    /* MSB-first ladder: the exponent register walks left (one bit per step),
       so the loop is a test plus a shift, and a zero bit costs a doubling
       instead of a Montgomery multiply. */
    uint64_t xlo = dsh[0], xhi = dsh[1];
    {
        int sh = 128 - dbits + 1;
        if (sh >= 128) { xlo = 0; xhi = 0; }
        else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
        else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
    }
#pragma unroll 1
    for (int j = 0; j < 2; j++) y[j] = two[j];
    for (int e = dbits - 2; e >= 0; e--) {
        mr128_64_mont_mul(y, y, y, n, n0inv);
        if ((uint64_t)(xhi >> 63)) mr128_64_dbl_mod(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    if ((y[0] == one[0] && y[1] == one[1]) ||
        (y[0] == neg[0] && y[1] == neg[1])) return 1;
    for (int r = 0; r < s - 1; r++) {
        mr128_64_mont_mul(y, y, y, n, n0inv);
        if (y[0] == neg[0] && y[1] == neg[1]) return 1;
        if (y[0] == one[0] && y[1] == one[1]) return 0;
    }
    return 0;
}

/* ---- lazy variant: same math, residue kept in [0, 2n) through the ladder ---
   CIOS reduction of a*b (< 4n^2) leaves a residue < 4n^2/R + n < 2n whenever
   n < 2^126 (4n < R = 2^128), and the doublings below re-enter the same band
   via a precomputed 2n, so the entire ladder runs without a single
   compare/subtract per squaring.  Canonical form is produced only where the
   verdict compares.  Measured +37% over the canonical kernel
   (tools/mr128_bench.cu, variant 13 vs 10).  EXACT for n < 2^126 only; the
   caller must dispatch n >= 2^126 to mr128_64_base2. */
__device__ __forceinline__ static void mr128_64_mont_mul_lazy(uint64_t *r,
                                                              const uint64_t *a,
                                                              const uint64_t *b,
                                                              const uint64_t *n,
                                                              uint64_t n0inv) {
    uint64_t t0 = 0U, t1 = 0U, t2 = 0U, t3 = 0U;
    uint128_t p, s;
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const uint64_t bi = b[i];
        uint64_t c;
        p = (uint128_t)a[0] * (uint128_t)bi + (uint128_t)t0;
        t0 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        p = (uint128_t)a[1] * (uint128_t)bi + (uint128_t)t1 + (uint128_t)c;
        t1 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        s = (uint128_t)t2 + (uint128_t)c;
        t2 = (uint64_t)s;
        t3 += (uint64_t)(s >> 64);

        const uint64_t m = t0 * n0inv;
        uint128_t q = (uint128_t)m * (uint128_t)n[0] + (uint128_t)t0;
        c = (uint64_t)(q >> 64);
        p = (uint128_t)m * (uint128_t)n[1] + (uint128_t)t1 + (uint128_t)c;
        t0 = (uint64_t)p;
        c = (uint64_t)(p >> 64);
        s = (uint128_t)t2 + (uint128_t)c;
        t1 = (uint64_t)s;
        t2 = t3 + (uint64_t)(s >> 64);
        t3 = 0U;
    }
    (void)t2;                                /* provably 0 for n < 2^126 */
    r[0] = t0;
    r[1] = t1;
}

/* r = 2*a mod 2n: subtract the precomputed 2n when needed (input and output
   both in [0, 2n), so the lazy ladder never re-canonicalises) */
__device__ __forceinline__ static void mr128_64_dbl_lazy(uint64_t *r,
                                                         const uint64_t *a,
                                                         const uint64_t *n2) {
    uint64_t t0 = a[0] << 1;
    uint64_t t1 = (a[1] << 1) | (a[0] >> 63);
    uint64_t sub0, sub1, b0, b1;
    b0 = mr128_64_sub(t0, n2[0], 0U, &sub0);
    b1 = mr128_64_sub(t1, n2[1], b0, &sub1);
    if (!b1) { t0 = sub0; t1 = sub1; }
    r[0] = t0;
    r[1] = t1;
}

/* base-2 strong probable prime, lazy ladder: exact for 2^96 < n < 2^126 */
__device__ static int mr128_64_base2_lazy(const uint64_t *nin) {
    uint64_t n[2], one[2], neg[2], two[2], y[2];
    uint64_t n0inv = montgomeryInverse64(nin[0]);
    n[0] = nin[0];
    n[1] = nin[1];

    int b = (64 - mr128_64_clz64(n[1])) + 64;
    mr128_64_one(one, n);
    {
        uint64_t b0, nb1;
        b0 = mr128_64_sub(n[0], one[0], 0U, &neg[0]);
        nb1 = mr128_64_sub(n[1], one[1], b0, &neg[1]);
        (void)nb1;
    }
    mr128_64_dbl_mod(two, one, n);

    uint64_t nm1[2];
    nm1[1] = n[1] - ((n[0] == 0U) ? 1U : 0U);
    nm1[0] = n[0] - 1U;
    int s = 0;
    {
        uint64_t w = nm1[0];
        if (w == 0U) { s = 64; w = nm1[1]; }
        while ((w & 1U) == 0U) { w >>= 1; s++; }
    }
    uint64_t dsh[2];
    {
        int sh = s & 63, wi = s >> 6;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint64_t lo = (j + wi < 2) ? nm1[j + wi] : 0U;
            uint64_t hi = (j + wi + 1 < 2) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint64_t)((lo >> sh) | (hi << (64 - sh)));
        }
    }
    int dbits = b - s;
    while (dbits > 0 &&
           ((dsh[(dbits - 1) >> 6] >> ((dbits - 1) & 63)) & 1ULL) == 0U)
        dbits--;

    uint64_t xlo = dsh[0], xhi = dsh[1];
    {
        int sh = 128 - dbits + 1;
        if (sh >= 128) { xlo = 0; xhi = 0; }
        else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
        else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
    }
    const uint64_t n2[2] = { n[0] << 1, (n[1] << 1) | (n[0] >> 63) };
    y[0] = two[0]; y[1] = two[1];
    for (int e = dbits - 2; e >= 0; e--) {
        mr128_64_mont_mul_lazy(y, y, y, n, n0inv);   /* y stays < 2n */
        if ((uint64_t)(xhi >> 63)) mr128_64_dbl_lazy(y, y, n2);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    /* verdict compares need the canonical form; y < 2n so one subtract */
    uint64_t s0, s1, b0, b1;
    b0 = mr128_64_sub(y[0], n[0], 0U, &s0);
    b1 = mr128_64_sub(y[1], n[1], b0, &s1);
    if (!b1) { y[0] = s0; y[1] = s1; }
    if ((y[0] == one[0] && y[1] == one[1]) ||
        (y[0] == neg[0] && y[1] == neg[1])) return 1;
    for (int r = 0; r < s - 1; r++) {
        mr128_64_mont_mul_lazy(y, y, y, n, n0inv);
        b0 = mr128_64_sub(y[0], n[0], 0U, &s0);
        b1 = mr128_64_sub(y[1], n[1], b0, &s1);
        if (!b1) { y[0] = s0; y[1] = s1; }
        if (y[0] == neg[0] && y[1] == neg[1]) return 1;
        if (y[0] == one[0] && y[1] == one[1]) return 0;
    }
    return 0;
}

__device__ static int mr128_64_base2_lazy_u128(uint64_t lo, uint64_t hi) {
    uint64_t n[2] = {lo, hi};
    return mr128_64_base2_lazy(n);
}

/* 128-bit entry point: value = (hi << 64) | lo */
__device__ static int mr128_64_base2_u128(uint64_t lo, uint64_t hi) {
    uint64_t n[2] = {lo, hi};
    return mr128_64_base2(n);
}

#if defined(__CUDACC__)

__global__ static void mr128_64_kernel(const uint64_t * __restrict__ cands,
                                       uint8_t * __restrict__ results,
                                       uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    results[i] = (uint8_t)mr128_64_base2(cands + (size_t)i * 2);
}

#endif /* __CUDACC__ */

#endif /* MR128_64_KERNEL_CUH */
