/*
 * mr128_64_sqr_kernel.cuh - exact base-2 test for 2^96 < n < 2^128 built on the
 * lean square of the walk engine's perig test (perig.cuh, ciosModSquare128),
 * with its two range defects fixed:
 *
 *   1. the doubling 2*n_hi is a 65-bit multiplier.  perig adds it as
 *      `n_lo * (n_hi + n_hi)`, where the uint64 addition wraps for
 *      n >= 2^127 - the square is then simply wrong.  Here the doubling is
 *      split into a low word and a top bit and both terms are accumulated.
 *   2. perig's MSB-first ladder walks the exponent in 3-bit groups with a
 *      128-bit shift register, which is only exact while 8*res < 2^128; the
 *      ladder below normalises d = (n-1) >> s explicitly and walks it one bit
 *      per step, using a doubling for zero bits.
 *
 * Structure kept from perig (this is where its speed comes from): the square
 * is accumulated in 128-bit `unsigned __int128`-style steps (uint128_t here,
 * native or shim depending on the compiler) so the carry chain lives in one
 * register pair, and the two Montgomery reduction steps are interleaved with
 * the hi^2 term of the square so the accumulator never grows past three words.
 * Unlike perig, every step is canonicalised (< n), because the MR verdict
 * compares against the Montgomery forms of +-1 and a 2-bit guard would make
 * those comparisons ambiguous.
 *
 * Limit: 2^96 < n < 2^128, n odd.
 */

#ifndef MR128_64_SQR_KERNEL_CUH
#define MR128_64_SQR_KERNEL_CUH

#include <stdint.h>
#include "perig.cuh"     /* uint128_t (native or shim), montgomeryInverse64 */

__device__ __forceinline__ static uint64_t mrsq_sub(uint64_t a, uint64_t b,
                                                    uint64_t bin, uint64_t *d) {
    uint64_t t = a - b;
    uint64_t bo = (a < b) ? 1U : 0U;
    if (bin) {
        if (t == 0U) bo = 1U;
        t -= 1U;
    }
    *d = t;
    return bo;
}

__device__ __forceinline__ static int mrsq_clz64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __clzll(x);
#else
    int c = 0;
    while (c < 64 && !(x & (1ULL << 63))) { x <<= 1; c++; }
    return c;
#endif
}

/* r = a^2 * R^-1 mod n, R = 2^128, result < n; a < n, n0inv = -n^-1 mod 2^64.
   Square = lo^2 + 2*lo*hi*2^64 + hi^2*2^128, reduced in two interleaved steps
   (each REDC step divides by 2^64, so hi^2 enters between them). */
__device__ __forceinline__ static void mrsq_sqr(uint64_t *r, const uint64_t *a,
                                                const uint64_t *n, uint64_t n0inv) {
    const uint64_t lo = a[0], hi = a[1];
    uint128_t cc, cs;
    uint64_t t0, t1, t2, m;

    cc = (uint128_t)lo * (uint128_t)lo;                 /* lo^2 -> words 0..1 */
    t0 = (uint64_t)cc;
    cc >>= 64;
    {                                                   /* 2*lo*hi (65-bit factor) */
        uint64_t d_lo = hi << 1, d_top = hi >> 63;
        cc += (uint128_t)lo * (uint128_t)d_lo;          /* lands on words 1..2 */
        t1 = (uint64_t)cc;
        cc >>= 64;
        cc += (uint128_t)lo * (uint128_t)d_top;         /* the 65th bit */
        t2 = (uint64_t)cc;
        cc >>= 64;                                      /* must be 0 here */
    }

    m = t0 * n0inv;                                     /* REDC step 1 */
    cs = (uint128_t)m * (uint128_t)n[0] + (uint128_t)t0;
    cs >>= 64;
    cs += (uint128_t)m * (uint128_t)n[1] + (uint128_t)t1;
    t0 = (uint64_t)cs;
    cs >>= 64;
    cs += (uint128_t)t2;
    t1 = (uint64_t)cs;
    cs >>= 64;
    t2 = (uint64_t)cs;

    cc = (uint128_t)hi * (uint128_t)hi;                 /* hi^2, words 1..2 now */
    cc += (uint128_t)t1;
    t1 = (uint64_t)cc;
    cc >>= 64;
    cc += (uint128_t)t2;
    t2 = (uint64_t)cc;

    m = t0 * n0inv;                                     /* REDC step 2 */
    cs = (uint128_t)m * (uint128_t)n[0] + (uint128_t)t0;
    cs >>= 64;
    cs += (uint128_t)m * (uint128_t)n[1] + (uint128_t)t1;
    t0 = (uint64_t)cs;
    cs >>= 64;
    cs += (uint128_t)t2;
    t1 = (uint64_t)cs;

    uint64_t sub0, sub1, b0, b1;
    b0 = mrsq_sub(t0, n[0], 0U, &sub0);
    b1 = mrsq_sub(t1, n[1], b0, &sub1);
    int need = (b1 == 0U);
    r[0] = need ? sub0 : t0;
    r[1] = need ? sub1 : t1;
}

/* r = 2*a mod n */
__device__ __forceinline__ static void mrsq_dbl(uint64_t *r, const uint64_t *a,
                                                const uint64_t *n) {
    uint64_t ovf = a[1] >> 63;
    uint64_t t0 = a[0] << 1;
    uint64_t t1 = (a[1] << 1) | (a[0] >> 63);
    uint64_t sub0, sub1, b0, b1;
    b0 = mrsq_sub(t0, n[0], 0U, &sub0);
    b1 = mrsq_sub(t1, n[1], b0, &sub1);
    int need = (ovf != 0U) | (b1 == 0U);
    r[0] = need ? sub0 : t0;
    r[1] = need ? sub1 : t1;
}

/* ONE = 2^128 mod n (n > 2^96 keeps q far below 2^64) */
__device__ __forceinline__ static void mrsq_one(uint64_t *one, const uint64_t *n) {
    double dn = (double)n[1] * 18446744073709551616.0 + (double)n[0];
    uint64_t qh = (uint64_t)(340282366920938463463374607431768211456.0 / dn);
    uint128_t cy = 0;
    uint64_t qn[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        uint128_t p = (uint128_t)qh * (uint128_t)n[j] + cy;
        qn[j] = (uint64_t)p;
        cy = p >> 64;
    }
    uint64_t ovf = (uint64_t)cy;
    uint64_t b0, b1, o0, o1;
    b0 = mrsq_sub(0U, qn[0], 0U, &o0);
    b1 = mrsq_sub(0U, qn[1], b0, &o1);
    (void)b1;
    one[0] = o0;
    one[1] = o1;
    if (ovf) {
        uint64_t c2 = 0;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint128_t s = (uint128_t)one[j] + (uint128_t)n[j] + (uint128_t)c2;
            one[j] = (uint64_t)s;
            c2 = (uint64_t)(s >> 64);
        }
    } else {
        uint64_t s0, s1, d0, d1;
        d0 = mrsq_sub(one[0], n[0], 0U, &s0);
        d1 = mrsq_sub(one[1], n[1], d0, &s1);
        if (!d1) { one[0] = s0; one[1] = s1; }
    }
}

/* base-2 strong probable prime, 2 x 64-bit limbs: 0 = composite, 1 = probable
   prime.  n odd, 2^96 < n < 2^128.  Exponent handling is the validated
   staircase of mr128_64_kernel.cuh; only the squaring differs. */
__device__ static int mrsq_base2(const uint64_t *nin) {
    uint64_t n[2], one[2], neg[2], y[2];
    const uint64_t n0inv = montgomeryInverse64(nin[0]);
    n[0] = nin[0];
    n[1] = nin[1];

    const int nbits = (64 - mrsq_clz64(n[1])) + 64;

    mrsq_one(one, n);
    {
        uint64_t b0, nb;
        b0 = mrsq_sub(n[0], one[0], 0U, &neg[0]);
        nb = mrsq_sub(n[1], one[1], b0, &neg[1]);
        (void)nb;
    }
    mrsq_dbl(y, one, n);                      /* y = mont(2) */

    uint64_t nm1[2];
    nm1[1] = n[1] - ((n[0] == 0U) ? 1U : 0U);
    nm1[0] = n[0] - 1U;
    int s = 0;                                /* n-1 = 2^s * d */
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
    int dbits = nbits - s;
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
    for (int e = dbits - 2; e >= 0; e--) {
        mrsq_sqr(y, y, n, n0inv);
        if ((uint64_t)(xhi >> 63)) mrsq_dbl(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    if ((y[0] == one[0] && y[1] == one[1]) ||
        (y[0] == neg[0] && y[1] == neg[1])) return 1;
    for (int r = 0; r < s - 1; r++) {
        mrsq_sqr(y, y, n, n0inv);
        if (y[0] == neg[0] && y[1] == neg[1]) return 1;
        if (y[0] == one[0] && y[1] == one[1]) return 0;
    }
    return 0;
}

__device__ static int mrsq_base2_u128(uint64_t lo, uint64_t hi) {
    uint64_t n[2] = {lo, hi};
    return mrsq_base2(n);
}

#if defined(__CUDACC__)

__global__ static void mrsq_kernel(const uint64_t * __restrict__ c,
                                   uint8_t * __restrict__ r, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    r[i] = (uint8_t)mrsq_base2(c + (size_t)i * 2);
}

#endif /* __CUDACC__ */

#endif /* MR128_64_SQR_KERNEL_CUH */
