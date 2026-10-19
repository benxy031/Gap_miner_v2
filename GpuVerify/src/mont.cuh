/*
 * GpuVerify - CIOS Montgomery arithmetic on 32-bit limbs.
 * Fresh implementation of HAC algorithm 14.36; cross-checked against GMP
 * by tests/test_mont (thousands of random cases at several sizes).
 */
#pragma once

#include <stdint.h>

/*
 * r = a * b * 2^(-32*N32) mod n, n odd, np0 = -n^(-1) mod 2^32.
 * Arrays are N32 limbs. Operands are zero-padded to N32 limbs; a, b must
 * be < n. r may alias a or b. The result is fully reduced (< n).
 */
template <int N32>
__device__ __forceinline__ void mont_mul(uint32_t *r, const uint32_t *a,
                                         const uint32_t *b, const uint32_t *n,
                                         uint32_t np0)
{
    uint32_t t[N32 + 2];
#pragma unroll 4
    for (int i = 0; i < N32 + 2; i++)
        t[i] = 0;

    /* OUTER loop stays rolled ON PURPOSE: fully unrolling the N32^2 nest
     * segfaults nvcc 12.4 (cicc) at large limb counts. */
#pragma unroll 1
    for (int i = 0; i < N32; i++) {
        uint32_t ai = a[i];
        uint64_t carry = 0;
#pragma unroll 4
        for (int j = 0; j < N32; j++) {
            uint64_t cur = (uint64_t)t[j] + (uint64_t)ai * (uint64_t)b[j] + carry;
            t[j] = (uint32_t)cur;
            carry = cur >> 32;
        }
        uint64_t cur = (uint64_t)t[N32] + carry;
        t[N32] = (uint32_t)cur;
        carry = cur >> 32;
        t[N32 + 1] = (uint32_t)((uint64_t)t[N32 + 1] + carry);

        uint32_t q = t[0] * np0; /* t[0] * (-n^-1) makes t+ q*n divisible by 2^32 */

        uint64_t cur2 = 0;
#pragma unroll 4
        for (int j = 0; j < N32; j++) {
            uint64_t c = (uint64_t)t[j] + (uint64_t)q * (uint64_t)n[j] + cur2;
            t[j] = (uint32_t)c;
            cur2 = c >> 32;
        }
        cur2 = (uint64_t)t[N32] + cur2;
        t[N32] = (uint32_t)cur2;
        cur2 >>= 32;
        cur2 = (uint64_t)t[N32 + 1] + cur2;
        t[N32 + 1] = (uint32_t)cur2;

        /* shift right by one limb */
#pragma unroll 4
        for (int j = 0; j <= N32; j++)
            t[j] = t[j + 1];
        t[N32 + 1] = 0;
    }

    /* conditional subtraction: t < 2n here, so at most two subtractions */
#pragma unroll 1
    for (int rep = 0; rep < 2; rep++) {
        bool ge = (t[N32] | t[N32 + 1]) != 0;
        if (!ge) {
#pragma unroll 1
            for (int j = N32 - 1; j >= 0; j--) {
                if (t[j] != n[j]) {
                    ge = t[j] > n[j];
                    break;
                }
            }
        }
        if (!ge)
            break;
        uint64_t borrow = 0;
#pragma unroll 4
        for (int j = 0; j < N32; j++) {
            uint64_t sub = (uint64_t)n[j] + borrow;
            uint64_t v = (uint64_t)t[j];
            t[j] = (uint32_t)(v - sub);
            borrow = (v < sub) ? 1 : 0;
        }
        uint64_t v = (uint64_t)t[N32] | ((uint64_t)t[N32 + 1] << 32);
        v -= borrow;
        t[N32] = (uint32_t)v;
        t[N32 + 1] = (uint32_t)(v >> 32);
    }

#pragma unroll 4
    for (int j = 0; j < N32; j++)
        r[j] = t[j];
}

template <int N32>
__device__ __forceinline__ void mont_sqr(uint32_t *r, const uint32_t *a,
                                         const uint32_t *n, uint32_t np0)
{
    mont_mul<N32>(r, a, a, n, np0);
}

template <int N32>
__device__ __forceinline__ bool limb_eq(const uint32_t *a, const uint32_t *b)
{
    uint32_t x = 0;
#pragma unroll 4
    for (int i = 0; i < N32; i++)
        x |= a[i] ^ b[i];
    return x == 0;
}
