/*
 * mr128_bench.cu - --validate / --bench harness for the 128-bit walk-engine
 * primality test, plus the experimental kernel variants used to speed it up.
 *
 * WHY: the walk engine spends a large share of its time in the base-2 test
 * (tools/mr128_kernel.cuh above 2^120, perig below).  Before changing that
 * code the cost has to be measured and every variant has to pass a GMP
 * cross-check on the SAME candidate stream, so a speedup cannot be bought with
 * a wrong verdict.
 *
 * Modes:
 *   ./bin/mr128_bench --validate N [variant]   GMP cross-check, must be 0 bad
 *   ./bin/mr128_bench --bench N iters [variant] [tpb]
 *
 * Variants (all base-2 strong probable prime, i.e. no false positives):
 *   0  production: mr128_kernel.cuh (4 x 32-bit CIOS, one thread per candidate)
 *   1  production loop: two candidates per thread (independent chains)
 *   2  experimental: separated square + REDC (10 products instead of 16)
 *   3  experimental: 2 per thread, variant 2 core
 *
 * Build: make bin/mr128_bench WITH_CUDA=1
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <gmp.h>

#include "mr128_kernel.cuh"
#include "perig.cuh"          /* 2 x 64-bit reference: exactness breaks >120 bits, speed is the yardstick */
#include "mr128_64_kernel.cuh" /* exact 2 x 64-bit kernel (the candidate replacement) */

typedef unsigned __int128 u128;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[mr128] %s: %s\n", what, cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

/* ---------------- experimental variant 2: SOS + symmetric square -------------
   a^2 is built from the 10 distinct products (off-diagonal doubled) into an
   8 x 32-bit product, then reduced with 4 x 4 schoolbook REDC steps.  The
   Montgomery ladder changes shape: square, then conditionally double. */

static __device__ __forceinline__ void add_prod(uint32_t *w, uint32_t p_lo,
                                                uint32_t p_hi, int pos) {
    uint64_t s = (uint64_t)w[pos] + p_lo;
    w[pos] = (uint32_t)s;
    uint64_t c = s >> 32;
    s = (uint64_t)w[pos + 1] + p_hi + c;
    w[pos + 1] = (uint32_t)s;
    c = s >> 32;
    int k = pos + 2;
    while (c && k < 9) {
        s = (uint64_t)w[k] + c;
        w[k] = (uint32_t)s;
        c = s >> 32;
        k++;
    }
}

/* 4 x 32-bit square, symmetric: 10 products (t[i+j] += a_i a_j, doubled i != j) */
static __device__ __forceinline__ void sqr256(const uint32_t *a, uint32_t *w) {
#pragma unroll
    for (int i = 0; i < 9; i++) w[i] = 0U;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = i; j < 4; j++) {
            uint64_t p = (uint64_t)a[i] * (uint64_t)a[j];
            add_prod(w, (uint32_t)p, (uint32_t)(p >> 32), i + j);
            if (i != j) add_prod(w, (uint32_t)p, (uint32_t)(p >> 32), i + j);
        }
    }
}

/* Montgomery reduction of an 8-word value w[] (256 bits) modulo n, R = 2^128 */
static __device__ __forceinline__ void redc256(uint32_t *r, uint32_t *w,
                                               const uint32_t *n, uint32_t n0inv) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t m = w[i] * n0inv;
        uint64_t c = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t s = (uint64_t)m * (uint64_t)n[j] + (uint64_t)w[i + j] + c;
            w[i + j] = (uint32_t)s;
            c = s >> 32;
        }
        int k = i + 4;
        while (c && k < 9) {
            uint64_t s = (uint64_t)w[k] + c;
            w[k] = (uint32_t)s;
            c = s >> 32;
            k++;
        }
    }
    /* result = w[4..8], < 2n (so w[8] is 0 or 1): one conditional subtraction */
    uint32_t sub[4], borrow = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        uint64_t d = (uint64_t)w[4 + j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (w[8] != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < 4; j++) r[j] = need ? sub[j] : w[4 + j];
}

/* variant 2 core: same MR structure as mr128_base2, different mul/square */
static __device__ int mr128_v2(const uint32_t *nin) {
    uint32_t n[4], one[4], neg[4], two[4], y[4];
    uint32_t w[9];
    uint32_t n0inv = 1U;
#pragma unroll
    for (int k = 0; k < 5; k++) n0inv = n0inv * (2U - nin[0] * n0inv);
    n0inv = (uint32_t)(0U - n0inv);
#pragma unroll
    for (int j = 0; j < 4; j++) n[j] = nin[j];

    int b = 0;
#pragma unroll
    for (int j = 3; j >= 0; j--)
        if (b == 0 && n[j] != 0U) b = j * 32 + (32 - mr128_clz32(n[j]));

    {
        double dn = (double)n[3];
        dn = dn * 4294967296.0 + (double)n[2];
        dn = dn * 4294967296.0 + (double)n[1];
        dn = dn * 4294967296.0 + (double)n[0];
        uint64_t q = (uint64_t)(340282366920938463463374607431768211456.0 / dn);
        uint32_t qn[4];
        uint64_t cy = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t p = (uint64_t)q * (uint64_t)n[j] + cy;
            qn[j] = (uint32_t)p;
            cy = p >> 32;
        }
        uint32_t ovf = (uint32_t)cy, bor = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t d64 = 0ULL - (uint64_t)qn[j] - (uint64_t)bor;
            one[j] = (uint32_t)d64;
            bor = (uint32_t)((d64 >> 32) & 1ULL);
        }
        if (ovf) {
            uint32_t c2 = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t s64 = (uint64_t)one[j] + (uint64_t)n[j] + (uint64_t)c2;
                one[j] = (uint32_t)s64;
                c2 = (uint32_t)(s64 >> 32);
            }
        } else {
            uint32_t b2 = 0, sub[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t d64 = (uint64_t)one[j] - (uint64_t)n[j] - (uint64_t)b2;
                sub[j] = (uint32_t)d64;
                b2 = (uint32_t)((d64 >> 32) & 1ULL);
            }
            if (!b2) {
#pragma unroll
                for (int j = 0; j < 4; j++) one[j] = sub[j];
            }
        }
    }
    {
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t d = (uint64_t)n[j] - (uint64_t)one[j] - (uint64_t)borrow;
            neg[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
    mr128_dbl_mod(two, one, n);

    uint32_t nm1[4];
    {
        uint64_t d = (uint64_t)n[0] - 1ULL;
        nm1[0] = (uint32_t)d;
        uint32_t borrow = (uint32_t)((d >> 32) & 1ULL);
#pragma unroll
        for (int j = 1; j < 4; j++) {
            uint64_t dd = (uint64_t)n[j] - (uint64_t)borrow;
            nm1[j] = (uint32_t)dd;
            borrow = (uint32_t)((dd >> 32) & 1ULL);
        }
    }
    int s = 0;
    {
        int j = 0;
        while (j < 4 && nm1[j] == 0U) { s += 32; j++; }
        if (j < 4) {
            uint32_t v = nm1[j];
            while ((v & 1U) == 0U) { v >>= 1; s++; }
        }
    }
    uint32_t dsh[4];
    {
        int sh = s & 31, wi = s >> 5;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t lo = (j + wi < 4) ? nm1[j + wi] : 0U;
            uint32_t hi = (j + wi + 1 < 4) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint32_t)((lo >> sh) | (hi << (32 - sh)));
        }
    }
    int dbits = b - s;
    while (dbits > 0 && ((dsh[(dbits - 1) >> 5] >> ((dbits - 1) & 31)) & 1U) == 0U)
        dbits--;

    uint64_t xlo = (uint64_t)dsh[0] | ((uint64_t)dsh[1] << 32);
    uint64_t xhi = (uint64_t)dsh[2] | ((uint64_t)dsh[3] << 32);
    {
        int sh = 128 - dbits + 1;
        if (sh >= 128) { xlo = 0; xhi = 0; }
        else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
        else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
    }
#pragma unroll 1
    for (int j = 0; j < 4; j++) y[j] = two[j];
    for (int e = dbits - 2; e >= 0; e--) {
        sqr256(y, w);
        redc256(y, w, n, n0inv);
        if ((uint32_t)(xhi >> 63)) mr128_dbl_mod(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    int is_one = 1, is_neg = 1;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        if (y[j] != one[j]) is_one = 0;
        if (y[j] != neg[j]) is_neg = 0;
    }
    if (is_one || is_neg) return 1;
    for (int r = 0; r < s - 1; r++) {
        sqr256(y, w);
        redc256(y, w, n, n0inv);
        int got_neg = 1, got_one = 1;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            if (y[j] != neg[j]) got_neg = 0;
            if (y[j] != one[j]) got_one = 0;
        }
        if (got_neg) return 1;
        if (got_one) return 0;
    }
    return 0;
}

/* ---------------- variant 11: SOS square + REDC (2 x 64-bit) ----------------
   The ladder only ever squares, so the product needs just three multiplies
   (a0^2, a0*a1 doubled, a1^2) instead of the four generic CIOS performs, and
   the reduction consumes a plain 5-word square.  The doubling of a0*a1 is done
   on the 129-bit product (shift lo/hi plus a top bit) rather than by doubling
   the multiplicand, which is what makes perig's version inexact above 2^127:
   there a1 + a1 wraps a 64-bit multiplier. */
static __device__ __forceinline__ void mr128_64_sqr_redc(uint64_t *r,
                                                         const uint64_t *a,
                                                         const uint64_t *n,
                                                         uint64_t n0inv) {
    uint64_t t[5];
    uint128_t p, s;
    uint64_t c, c2;

    /* columns: a0^2 -> words 0..1, 2*a0*a1 -> words 1..3, a1^2 -> words 2..3 */
    p = (uint128_t)a[0] * (uint128_t)a[0];
    t[0] = (uint64_t)p;
    c = (uint64_t)(p >> 64);                       /* high(a0^2) lands on word 1 */

    p = (uint128_t)a[0] * (uint128_t)a[1];         /* 2*a0*a1 as a 129-bit value */
    {
        uint64_t lo = (uint64_t)p, hi = (uint64_t)(p >> 64);
        uint64_t d0 = lo << 1, d1 = (hi << 1) | (lo >> 63), dtop = hi >> 63;
        s = (uint128_t)d0 + (uint128_t)c;          /* word 1 */
        t[1] = (uint64_t)s;
        c2 = (uint64_t)(s >> 64);
        s = (uint128_t)d1 + (uint128_t)c2;         /* word 2 */
        t[2] = (uint64_t)s;
        c2 = (uint64_t)(s >> 64);
        s = (uint128_t)dtop + (uint128_t)c2;       /* word 3 */
        t[3] = (uint64_t)s;
        t[4] = (uint64_t)(s >> 64);
    }

    p = (uint128_t)a[1] * (uint128_t)a[1];         /* a1^2 -> words 2..3 */
    s = (uint128_t)t[2] + (uint128_t)(uint64_t)p;
    t[2] = (uint64_t)s;
    c = (uint64_t)(s >> 64);
    s = (uint128_t)t[3] + (uint128_t)(uint64_t)(p >> 64) + (uint128_t)c;
    t[3] = (uint64_t)s;
    t[4] += (uint64_t)(s >> 64);

#pragma unroll
    for (int i = 0; i < 2; i++) {                  /* two REDC limbs */
        uint64_t m = t[0] * n0inv;
        s = (uint128_t)m * (uint128_t)n[0] + (uint128_t)t[0];
        c = (uint64_t)(s >> 64);
        s = (uint128_t)m * (uint128_t)n[1] + (uint128_t)t[1] + (uint128_t)c;
        t[0] = (uint64_t)s;
        c = (uint64_t)(s >> 64);
        s = (uint128_t)t[2] + (uint128_t)c;
        t[1] = (uint64_t)s;
        c2 = (uint64_t)(s >> 64);
        s = (uint128_t)t[3] + (uint128_t)c2;
        t[2] = (uint64_t)s;
        t[3] = t[4] + (uint64_t)(s >> 64);
        t[4] = 0U;
    }
    uint64_t sub0, sub1, b0, b1;
    b0 = mr128_64_sub(t[0], n[0], 0U, &sub0);
    b1 = mr128_64_sub(t[1], n[1], b0, &sub1);
    int need = (t[2] != 0U) | (b1 == 0U);
    r[0] = need ? sub0 : t[0];
    r[1] = need ? sub1 : t[1];
}

/* variant 11 core: same ladder as mr128_64_base2, squaring through SOS+REDC */
static __device__ int mr128_64_base2_sqr(const uint64_t *nin) {
    uint64_t n[2], one[2], neg[2], two[2], y[2];
    uint64_t n0inv = montgomeryInverse64(nin[0]);
    n[0] = nin[0]; n[1] = nin[1];
    int b = (64 - mr128_64_clz64(n[1])) + 64;
    mr128_64_one(one, n);
    {
        uint64_t b0 = mr128_64_sub(n[0], one[0], 0U, &neg[0]);
        uint64_t nb = mr128_64_sub(n[1], one[1], b0, &neg[1]);
        (void)nb;
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
    y[0] = two[0]; y[1] = two[1];
    for (int e = dbits - 2; e >= 0; e--) {
        mr128_64_sqr_redc(y, y, n, n0inv);
        if ((uint64_t)(xhi >> 63)) mr128_64_dbl_mod(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    if ((y[0] == one[0] && y[1] == one[1]) ||
        (y[0] == neg[0] && y[1] == neg[1])) return 1;
    for (int r = 0; r < s - 1; r++) {
        mr128_64_sqr_redc(y, y, n, n0inv);
        if (y[0] == neg[0] && y[1] == neg[1]) return 1;
        if (y[0] == one[0] && y[1] == one[1]) return 0;
    }
    return 0;
}

static __device__ int mr128_64_base2_sqr(uint64_t lo, uint64_t hi) {
    uint64_t n[2] = {lo, hi};
    return mr128_64_base2_sqr(n);
}

/* ---------------- kernels ---------------- */

__global__ static void k_v0(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                            uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    r[i] = (uint8_t)mr128_base2(c + (size_t)i * 4);
}

/* two candidates per thread: two independent Montgomery chains */
__global__ static void k_v1(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                            uint32_t count) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (i + 1 < count) {
        int a = mr128_base2(c + (size_t)i * 4);
        int b = mr128_base2(c + (size_t)(i + 1) * 4);
        r[i] = (uint8_t)a;
        r[i + 1] = (uint8_t)b;
    } else if (i < count) {
        r[i] = (uint8_t)mr128_base2(c + (size_t)i * 4);
    }
}

__global__ static void k_v2(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                            uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    r[i] = (uint8_t)mr128_v2(c + (size_t)i * 4);
}

__global__ static void k_v3(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                            uint32_t count) {
    uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2u;
    if (i + 1 < count) {
        int a = mr128_v2(c + (size_t)i * 4);
        int b = mr128_v2(c + (size_t)(i + 1) * 4);
        r[i] = (uint8_t)a;
        r[i + 1] = (uint8_t)b;
    } else if (i < count) {
        r[i] = (uint8_t)mr128_v2(c + (size_t)i * 4);
    }
}

/* variant 13: the lazy-ladder kernel from mr128_64_kernel.cuh
   (mr128_64_base2_lazy) — kept in the harness to A/B against v10. */
__global__ static void k_v13(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                             uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t lo = (uint64_t)c[(size_t)i * 4 + 0] | ((uint64_t)c[(size_t)i * 4 + 1] << 32);
    uint64_t hi = (uint64_t)c[(size_t)i * 4 + 2] | ((uint64_t)c[(size_t)i * 4 + 3] << 32);
    r[i] = (uint8_t)mr128_64_base2_lazy_u128(lo, hi);
}

__global__ static void k_v9(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                            uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t lo = (uint64_t)c[(size_t)i * 4 + 0] | ((uint64_t)c[(size_t)i * 4 + 1] << 32);
    uint64_t hi = (uint64_t)c[(size_t)i * 4 + 2] | ((uint64_t)c[(size_t)i * 4 + 3] << 32);
    r[i] = (uint8_t)(ciosFermatTest128_hi(lo, hi) ? 1 : 0);
}

/* variant 10: the exact 2 x 64-bit kernel */
__global__ static void k_v10(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                             uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t lo = (uint64_t)c[(size_t)i * 4 + 0] | ((uint64_t)c[(size_t)i * 4 + 1] << 32);
    uint64_t hi = (uint64_t)c[(size_t)i * 4 + 2] | ((uint64_t)c[(size_t)i * 4 + 3] << 32);
    r[i] = (uint8_t)mr128_64_base2_u128(lo, hi);
}

__global__ static void k_v11(const uint32_t * __restrict__ c, uint8_t * __restrict__ r,
                             uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t lo = (uint64_t)c[(size_t)i * 4 + 0] | ((uint64_t)c[(size_t)i * 4 + 1] << 32);
    uint64_t hi = (uint64_t)c[(size_t)i * 4 + 2] | ((uint64_t)c[(size_t)i * 4 + 3] << 32);
    r[i] = (uint8_t)mr128_64_base2_sqr(lo, hi);
}

static void launch(int variant, uint32_t batch, int tpb,
                   const uint32_t *d_c, uint8_t *d_r) {
    uint32_t grid = (batch + (uint32_t)tpb - 1) / (uint32_t)tpb;
    switch (variant) {
        case 0: k_v0<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 1: k_v1<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 2: k_v2<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 9: k_v9<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 10: k_v10<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 11: k_v11<<<grid, tpb>>>(d_c, d_r, batch); break;
        case 13: k_v13<<<grid, tpb>>>(d_c, d_r, batch); break;
        default: k_v3<<<grid, tpb>>>(d_c, d_r, batch); break;
    }
}

/* candidates: random odd numbers in [2^120, 2^128) - the shape the walk engine
   tests above 2^120 (one high word shared by a whole slice, varied low word),
   plus real primes so a "reject everything" kernel cannot pass. */
static void fill_candidates(uint32_t *cands, uint32_t count, int bits) {
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 20261008u);
    mpz_t x;
    mpz_init(x);
    for (uint32_t i = 0; i < count; i++) {
        mpz_urandomb(x, rs, bits);
        if (i % 4 == 0) {                        /* quarter: real primes */
            mpz_setbit(x, bits - 1);
            mpz_nextprime(x, x);
        } else {                                 /* rest: random odd */
            mpz_setbit(x, bits - 1);
            mpz_setbit(x, 0);
        }
        uint64_t w[4] = {0, 0, 0, 0};
        mpz_export(w, NULL, -1, 8, 0, 0, x);
        cands[(size_t)i * 4 + 0] = (uint32_t)w[0];
        cands[(size_t)i * 4 + 1] = (uint32_t)(w[0] >> 32);
        cands[(size_t)i * 4 + 2] = (uint32_t)w[1];
        cands[(size_t)i * 4 + 3] = (uint32_t)(w[1] >> 32);
    }
    mpz_clear(x);
    gmp_randclear(rs);
}

static int run_validate(uint32_t count, int variant, int bits) {
    printf("[mr128] --validate %u candidates, %d bits, variant %d ...\n", count, bits, variant);
    uint32_t *h_c = (uint32_t *)malloc((size_t)count * 4 * sizeof(uint32_t));
    uint8_t *h_r = (uint8_t *)malloc(count);
    uint32_t *d_c = NULL;
    uint8_t *d_r = NULL;
    if (!h_c || !h_r) { fprintf(stderr, "[mr128] host OOM\n"); return 1; }
    fill_candidates(h_c, count, bits);
    if (check_cuda(cudaMalloc(&d_c, (size_t)count * 4 * sizeof(uint32_t)), "malloc c")) return 1;
    if (check_cuda(cudaMalloc(&d_r, count), "malloc r")) return 1;
    if (check_cuda(cudaMemcpy(d_c, h_c, (size_t)count * 4 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice), "H2D")) return 1;
    launch(variant, count, 128, d_c, d_r);
    if (check_cuda(cudaGetLastError(), "launch")) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "sync")) return 1;
    if (check_cuda(cudaMemcpy(h_r, d_r, count, cudaMemcpyDeviceToHost), "D2H")) return 1;

    mpz_t n;
    mpz_init(n);
    size_t gpu_p = 0, gmp_p = 0, bad = 0, bad_comp = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t w[2] = {
            (uint64_t)h_c[(size_t)i * 4 + 0] | ((uint64_t)h_c[(size_t)i * 4 + 1] << 32),
            (uint64_t)h_c[(size_t)i * 4 + 2] | ((uint64_t)h_c[(size_t)i * 4 + 3] << 32)
        };
        mpz_import(n, 2, -1, 8, 0, 0, w);
        int gmp = mpz_probab_prime_p(n, 30) >= 1;
        if (h_r[i]) gpu_p++;
        if (gmp) gmp_p++;
        if ((int)h_r[i] != gmp) {
            bad++;
            if (h_r[i] && !gmp) bad_comp++;
        }
    }
    printf("[mr128] kernel primes=%zu  GMP primes=%zu  mismatches=%zu (kernel-prime-but-composite=%zu)\n",
           gpu_p, gmp_p, bad, bad_comp);
    printf("[mr128] %s\n", bad ? "VALIDATE FAILED" : "VALIDATE PASS (0 mismatches vs GMP)");
    mpz_clear(n);
    free(h_c); free(h_r); cudaFree(d_c); cudaFree(d_r);
    return bad ? 1 : 0;
}

static int run_bench(uint32_t batch, int iters, int variant, int tpb, int bits) {
    cudaFree(0);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        printf("[mr128] %s (sm_%d%d, %d SMs, %d threads/SM)\n", prop.name, prop.major,
               prop.minor, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);

    cudaFuncAttributes fa;
    const void *fn = (variant == 0) ? (const void *)k_v0
                   : (variant == 1) ? (const void *)k_v1
                   : (variant == 2) ? (const void *)k_v2
                   : (variant == 9) ? (const void *)k_v9
                   : (variant == 10) ? (const void *)k_v10
                   : (variant == 11) ? (const void *)k_v11
                   : (variant == 13) ? (const void *)k_v13
                   : (variant == 13) ? (const void *)k_v13 : (const void *)k_v3;
    if (cudaFuncGetAttributes(&fa, fn) == cudaSuccess)
        printf("[mr128] variant %d: regs/thread=%d  maxThreadsPerBlock=%d  localBytes=%zu\n",
               variant, fa.numRegs, fa.maxThreadsPerBlock, fa.localSizeBytes);

    uint32_t *h_c = (uint32_t *)malloc((size_t)batch * 4 * sizeof(uint32_t));
    uint32_t *d_c = NULL;
    uint8_t *d_r = NULL;
    if (!h_c) return 1;
    fill_candidates(h_c, batch, bits);
    if (check_cuda(cudaMalloc(&d_c, (size_t)batch * 4 * sizeof(uint32_t)), "malloc c")) return 1;
    if (check_cuda(cudaMalloc(&d_r, batch), "malloc r")) return 1;
    if (check_cuda(cudaMemcpy(d_c, h_c, (size_t)batch * 4 * sizeof(uint32_t),
                              cudaMemcpyHostToDevice), "H2D")) return 1;

    launch(variant, batch, tpb, d_c, d_r);
    if (check_cuda(cudaDeviceSynchronize(), "warmup")) return 1;

    double t0 = now_s();
    for (int k = 0; k < iters; k++) {
        launch(variant, batch, tpb, d_c, d_r);
        if (check_cuda(cudaPeekAtLastError(), "launch")) return 1;
    }
    if (check_cuda(cudaDeviceSynchronize(), "sync")) return 1;
    double dt = now_s() - t0;

    double total = (double)batch * (double)iters;
    printf("[mr128] variant %d  bits=%d  batch=%u  iters=%d  tpb=%d\n",
           variant, bits, batch, iters, tpb);
    printf("[mr128] wall=%.4f s  throughput=%.4e tests/s  (%.2f ns/test)\n",
           dt, total / dt, dt * 1e9 / total);
    free(h_c); cudaFree(d_c); cudaFree(d_r);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--validate")) {
        uint32_t n = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 20000u;
        int variant = (argc > 3) ? atoi(argv[3]) : 0;
        int bits = (argc > 4) ? atoi(argv[4]) : 121;
        cudaFree(0);
        return run_validate(n, variant, bits);
    }
    if (argc >= 2 && !strcmp(argv[1], "--bench")) {
        uint32_t batch = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 262144u;
        int iters = (argc > 3) ? atoi(argv[3]) : 20;
        int variant = (argc > 4) ? atoi(argv[4]) : 0;
        int tpb = (argc > 5) ? atoi(argv[5]) : 128;
        int bits = (argc > 6) ? atoi(argv[6]) : 121;
        cudaFree(0);
        return run_bench(batch, iters, variant, tpb, bits);
    }
    fprintf(stderr, "usage: %s --validate N [variant] [bits]\n"
                    "       %s --bench BATCH ITERS [variant] [tpb] [bits]\n", argv[0], argv[0]);
    return 2;
}
