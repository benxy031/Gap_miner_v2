/*
 * mr128_kernel.cuh - base-2 strong probable prime test for any odd n < 2^128,
 * as a 4 x 32-bit CIOS Montgomery kernel (the NW=3 design of
 * mr68_kernel.cuh, one limb wider and with the 128-bit entry points the walk
 * engine needs).
 *
 * WHY IT EXISTS
 *   The walk engine's perig test (perig.cuh, ciosFermatTest128_hi) is exact up
 *   to 120 bits but returns FALSE NEGATIVES from 121 bits on - it rejects real
 *   primes, which silently splits a gap and reports a gap that spans prime
 *   numbers (tools/perig_range.cpp reproduces this on the host: 0 bad
 *   verdicts at <= 120 bits, all primes rejected at >= 124 bits, and the
 *   engine reported gaps that this test cannot see).  A missed prime cannot be
 *   caught by the host's GMP re-verification, so the test itself must be exact
 *   over the whole container.
 *
 * DESIGN
 *   Same arithmetic as the GMP-validated mr68_kernel.cuh / bench_cios_mr.cu:
 *   CIOS Montgomery with 32-bit limbs and a 64-bit accumulator, n0inv =
 *   -n^-1 mod 2^32 by Newton, ONE = 2^128 mod n from a double-precision
 *   quotient estimate plus one correction, and a left-to-right binary ladder
 *   that uses dbl_mod for zero bits (a cheap add, not a full multiply).
 *   Exact for any odd n < 2^128: the accumulator is 4+2 limbs, every product
 *   is a 32x32->64 IMAD pair, and no intermediate exceeds 2^160.
 *
 *   The core is __host__ __device__ so the same code is validated on the host
 *   against GMP (tools/probes/mr128_range.cpp) before it ever runs on the GPU.
 *
 * Limit: 2^96 < n < 2^128, n odd.  The lower bound comes from the ONE constant:
 * q = floor(2^128/n) is multiplied by full 32-bit limbs, so q must stay below
 * 2^32 (q*n[j] < 2^64), i.e. n > 2^96.  Below that the 3 x 32-bit kernel
 * (mr68_kernel.cuh) and the walk engine's perig test cover the range.
 *
 * Include this header from .cu translation units, or from a host harness with a
 * plain g++ (the qualifier shim below handles that, no -D flags needed).
 */

#ifndef MR128_KERNEL_CUH
#define MR128_KERNEL_CUH

/* Host-compilable: outside nvcc the CUDA qualifiers mean nothing, so the host
   validation harnesses (tools/mr128_range.cpp) build with a plain
     g++ -O2 -I tools tools/mr128_range.cpp -lgmp
   instead of needing -D__host__= -D__device__= ... on the command line. */
#if !defined(__CUDACC__)
#  ifndef __device__
#    define __device__
#  endif
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __forceinline__
#    define __forceinline__ inline
#  endif
#  ifndef __global__
#    define __global__
#  endif
#endif

#include <stdint.h>

#define MR128_LIMBS 4                       /* 32-bit limbs (128-bit container) */

/* leading-zero count, device intrinsic on the GPU and a portable loop in the
   host validation harness (this header is compiled by both). */
__device__ __forceinline__ static int mr128_clz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __clz(x);
#else
    int c = 0;
    while (c < 32 && !(x & 0x80000000u)) { x <<= 1; c++; }
    return c;
#endif
}

/* r = a*b*R^-1 mod n with R = 2^128, result < n; n0inv = -n^-1 mod 2^32. */
__device__ __forceinline__ static void mr128_mont_mul(uint32_t *r,
                                                      const uint32_t *a,
                                                      const uint32_t *b,
                                                      const uint32_t *n,
                                                      uint32_t n0inv) {
    uint32_t t[MR128_LIMBS + 2];
#pragma unroll
    for (int i = 0; i < MR128_LIMBS + 2; i++) t[i] = 0U;

#pragma unroll
    for (int i = 0; i < MR128_LIMBS; i++) {
        uint64_t c = 0;
        const uint64_t bi = (uint64_t)b[i];
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            uint64_t p = (uint64_t)a[j] * bi + (uint64_t)t[j] + c;
            t[j] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s0 = (uint64_t)t[MR128_LIMBS] + c;
        uint64_t s1 = (uint64_t)t[MR128_LIMBS + 1] + (s0 >> 32);
        t[MR128_LIMBS] = (uint32_t)s0;
        t[MR128_LIMBS + 1] = (uint32_t)s1;

        uint32_t m = t[0] * n0inv;
        c = ((uint64_t)t[0] + (uint64_t)m * (uint64_t)n[0]) >> 32;
#pragma unroll
        for (int j = 1; j < MR128_LIMBS; j++) {
            uint64_t p = (uint64_t)m * (uint64_t)n[j] + (uint64_t)t[j] + c;
            t[j - 1] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s2 = (uint64_t)t[MR128_LIMBS] + c;
        t[MR128_LIMBS - 1] = (uint32_t)s2;
        t[MR128_LIMBS] = (uint32_t)((uint64_t)t[MR128_LIMBS + 1] + (s2 >> 32));
        t[MR128_LIMBS + 1] = 0U;
    }

    uint32_t borrow = 0, sub[MR128_LIMBS];
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (t[MR128_LIMBS] != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) r[j] = need ? sub[j] : t[j];
}

/* r = 2*a mod n */
__device__ __forceinline__ static void mr128_dbl_mod(uint32_t *r,
                                                     const uint32_t *a,
                                                     const uint32_t *n) {
    uint32_t c = 0, t[MR128_LIMBS];
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) {
        uint64_t s = (uint64_t)a[j] * 2ULL + (uint64_t)c;
        t[j] = (uint32_t)s;
        c = (uint32_t)(s >> 32);
    }
    uint32_t borrow = 0, sub[MR128_LIMBS];
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (c != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) r[j] = need ? sub[j] : t[j];
}

/* base-2 strong probable prime on 4 x 32-bit limbs: 0 = composite, 1 = prime.
   n-1 = 2^s * d; y = 2^d in Montgomery form, then squarings looking for -1. */
__device__ static int mr128_base2(const uint32_t *nin) {
    uint32_t n[MR128_LIMBS], one[MR128_LIMBS], neg[MR128_LIMBS],
             y[MR128_LIMBS];
    uint32_t n0inv = 1U;
#pragma unroll
    for (int k = 0; k < 5; k++) n0inv = n0inv * (2U - nin[0] * n0inv);
    n0inv = (uint32_t)(0U - n0inv);
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) n[j] = nin[j];

    int b = 0;                                   /* bit length of n */
#pragma unroll
    for (int j = MR128_LIMBS - 1; j >= 0; j--)
        if (b == 0 && n[j] != 0U) b = j * 32 + (32 - mr128_clz32(n[j]));

    /* ONE = 2^128 mod n: q = floor(2^128 / n) from a double estimate (relative
       error ~2^-53 gives |dq| <= 1 since q < 2^32), the 128-bit residual
       (0 - q*n) mod 2^128 corrected with one add or subtract of n. */
    {
        double dn = (double)n[3];
        dn = dn * 4294967296.0 + (double)n[2];
        dn = dn * 4294967296.0 + (double)n[1];
        dn = dn * 4294967296.0 + (double)n[0];
        uint64_t q = (uint64_t)(340282366920938463463374607431768211456.0 / dn);
        uint32_t qn[MR128_LIMBS];
        uint64_t cy = 0;
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            uint64_t p = (uint64_t)q * (uint64_t)n[j] + cy;
            qn[j] = (uint32_t)p;
            cy = p >> 32;
        }
        uint32_t ovf = (uint32_t)cy;              /* q*n >= 2^128 */
        uint32_t bor = 0;
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            uint64_t d64 = 0ULL - (uint64_t)qn[j] - (uint64_t)bor;
            one[j] = (uint32_t)d64;
            bor = (uint32_t)((d64 >> 32) & 1ULL);
        }
        if (ovf) {
            uint32_t c2 = 0;
#pragma unroll
            for (int j = 0; j < MR128_LIMBS; j++) {
                uint64_t s64 = (uint64_t)one[j] + (uint64_t)n[j] + (uint64_t)c2;
                one[j] = (uint32_t)s64;
                c2 = (uint32_t)(s64 >> 32);
            }
        } else {
            uint32_t b2 = 0, sub[MR128_LIMBS];
#pragma unroll
            for (int j = 0; j < MR128_LIMBS; j++) {
                uint64_t d64 = (uint64_t)one[j] - (uint64_t)n[j] - (uint64_t)b2;
                sub[j] = (uint32_t)d64;
                b2 = (uint32_t)((d64 >> 32) & 1ULL);
            }
            if (!b2) {
#pragma unroll
                for (int j = 0; j < MR128_LIMBS; j++) one[j] = sub[j];
            }
        }
    }
    {
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            uint64_t d = (uint64_t)n[j] - (uint64_t)one[j] - (uint64_t)borrow;
            neg[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
    uint32_t two[MR128_LIMBS];
    mr128_dbl_mod(two, one, n);

    uint32_t nm1[MR128_LIMBS];
    {
        uint64_t d = (uint64_t)n[0] - 1ULL;
        nm1[0] = (uint32_t)d;
        uint32_t borrow = (uint32_t)((d >> 32) & 1ULL);
#pragma unroll
        for (int j = 1; j < MR128_LIMBS; j++) {
            uint64_t dd = (uint64_t)n[j] - (uint64_t)borrow;
            nm1[j] = (uint32_t)dd;
            borrow = (uint32_t)((dd >> 32) & 1ULL);
        }
    }
    int s = 0;                                   /* n-1 = 2^s * d */
    {
        int j = 0;
        while (j < MR128_LIMBS && nm1[j] == 0U) { s += 32; j++; }
        if (j < MR128_LIMBS) {
            uint32_t w = nm1[j];
            while ((w & 1U) == 0U) { w >>= 1; s++; }
        }
    }
    uint32_t dsh[MR128_LIMBS];
    {
        int sh = s & 31, wi = s >> 5;
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            uint32_t lo = (j + wi < MR128_LIMBS) ? nm1[j + wi] : 0U;
            uint32_t hi = (j + wi + 1 < MR128_LIMBS) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint32_t)((lo >> sh) | (hi << (32 - sh)));
        }
    }
    int dbits = b - s;
    while (dbits > 0 &&
           ((dsh[(dbits - 1) >> 5] >> ((dbits - 1) & 31)) & 1U) == 0U)
        dbits--;

    /* Binary ladder: the exponent d walks a 128-bit register whose MSB sits at
       bit 127, so each step is one test plus one shift (no dynamically indexed
       register array).  y starts as 2 (Montgomery form of the base). */
    uint64_t xlo = (uint64_t)dsh[0] | ((uint64_t)dsh[1] << 32);
    uint64_t xhi = (uint64_t)dsh[2] | ((uint64_t)dsh[3] << 32);
    {
        int sh = 128 - dbits + 1;
        if (sh >= 128) { xlo = 0; xhi = 0; }
        else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
        else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
    }
#pragma unroll 1
    for (int j = 0; j < MR128_LIMBS; j++) y[j] = two[j];
    for (int e = dbits - 2; e >= 0; e--) {
        mr128_mont_mul(y, y, y, n, n0inv);
        if ((uint32_t)(xhi >> 63)) mr128_dbl_mod(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    int is_one = 1, is_neg = 1;
#pragma unroll
    for (int j = 0; j < MR128_LIMBS; j++) {
        if (y[j] != one[j]) is_one = 0;
        if (y[j] != neg[j]) is_neg = 0;
    }
    if (is_one || is_neg) return 1;
    for (int r = 0; r < s - 1; r++) {
        mr128_mont_mul(y, y, y, n, n0inv);
        int got_neg = 1, got_one = 1;
#pragma unroll
        for (int j = 0; j < MR128_LIMBS; j++) {
            if (y[j] != neg[j]) got_neg = 0;
            if (y[j] != one[j]) got_one = 0;
        }
        if (got_neg) return 1;
        if (got_one) return 0;
    }
    return 0;
}

/* 128-bit entry points: value = (hi << 64) | lo. */
__device__ static int mr128_base2_u128(uint64_t lo, uint64_t hi) {
    uint32_t n[MR128_LIMBS];
    n[0] = (uint32_t)lo;
    n[1] = (uint32_t)(lo >> 32);
    n[2] = (uint32_t)hi;
    n[3] = (uint32_t)(hi >> 32);
    return mr128_base2(n);
}

/* value = base + 2*off, base = (base_hi << 64) | base_lo: the walk engine's
   candidate shape, so the caller only ships a 32-bit offset. */
__device__ static int mr128_base2_base_off(uint64_t base_lo, uint64_t base_hi,
                                           uint32_t off) {
    uint64_t d = (uint64_t)off << 1;
    uint64_t lo = base_lo + d;
    uint64_t hi = base_hi + (lo < base_lo ? 1ULL : 0ULL);
    return mr128_base2_u128(lo, hi);
}

#if defined(__CUDACC__)   /* kernels: only under nvcc (the core above is also
                            compiled by the host validation harness) */

__global__ static void mr128_kernel(const uint32_t * __restrict__ cands,
                                    uint8_t * __restrict__ results,
                                    uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    results[i] = (uint8_t)mr128_base2(cands + (size_t)i * MR128_LIMBS);
}

/* Walk-engine stage: candidates are base + 2*offs[i]; probable primes set
   their bit in the verdict bitmap (the only device-to-host copy). */
__global__ static void mr128_from_offsets(uint64_t base_lo, uint64_t base_hi,
                                          const uint32_t * __restrict__ offs,
                                          const uint32_t * __restrict__ cntp,
                                          uint64_t * __restrict__ out) {
    const uint32_t cnt = *cntp;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += stride) {
        uint32_t off = offs[i];
        if (mr128_base2_base_off(base_lo, base_hi, off))
            atomicOr((unsigned long long *)&out[off >> 6], 1ULL << (off & 63u));
    }
}

#endif /* __CUDACC__ */

#endif /* MR128_KERNEL_CUH */
