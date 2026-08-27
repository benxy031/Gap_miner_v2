/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Fixed-limb CPU Montgomery primality tests (ported from cpugapminer's
 * src/primality_utils.c and refactored for gapminer_v2):
 *
 *  - CIOS Montgomery multiplication in __uint128_t, little-endian limbs.
 *  - ADX/BMI2 (MULX + ADCX/ADOX) exact-limb inner loops with runtime CPUID
 *    detection and a portable CIOS fallback.
 *  - 4-bit fixed-window left-to-right exponentiation (~2x fewer multiplies
 *    than binary square-and-multiply; matches GMP's k-ary window strategy).
 *  - Fermat (2^(n-1) ≡ 1) and Euler–Plumb (2^((n-1)/2) ≡ ±1) tests for
 *    1..20 limbs (64..1280 bits, Gapcoin shifts 0..1024).
 *
 * Refactor vs cpugapminer:
 *  - dead bucket-layer montmul/montsqr variants and TUNED macros removed
 *    (verified unreferenced even in cpugapminer),
 *  - env overrides renamed CPUGAP_CPU_WINDOW_* -> GAPMINER_CPU_WINDOW_*,
 *  - mpz adapters (export/add/precomp) for the worker CPU paths.
 */

#include "primality_limbs.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#include <cpuid.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
static int g_primality_runtime_adx_checked = 0;
static int g_primality_runtime_adx_enabled = 1;

static inline int primality_detect_adx_bmi2_runtime(void)
{
    unsigned int eax, ebx, ecx, edx;
    const unsigned int bit_bmi2 = 1u << 8;
    const unsigned int bit_adx = 1u << 19;

    if (!__get_cpuid_max(0, NULL))
        return 0;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return 0;
    return ((ebx & bit_bmi2) != 0) && ((ebx & bit_adx) != 0);
}

static inline void primality_runtime_init_adx_once(void)
{
    if (!g_primality_runtime_adx_checked) {
        g_primality_runtime_adx_enabled = primality_detect_adx_bmi2_runtime();
        g_primality_runtime_adx_checked = 1;
    }
}
#else
static inline void primality_runtime_init_adx_once(void)
{
}
#endif

/* n_prime = -(n^{-1}) mod 2^64 for odd n. */
static inline uint64_t mont_ninv(uint64_t n) {
    uint64_t x = 1;
    x *= 2 - n * x;
    x *= 2 - n * x;
    x *= 2 - n * x;
    x *= 2 - n * x;
    x *= 2 - n * x;
    x *= 2 - n * x;
    return -x;
}

/* Montgomery product: a * b * R^{-1} mod n. */
static inline uint64_t mont_mul(uint64_t a, uint64_t b,
                                uint64_t n, uint64_t np) {
    __uint128_t ab = (__uint128_t)a * b;
    uint64_t ab_lo = (uint64_t)ab;
    uint64_t ab_hi = (uint64_t)(ab >> 64);
    uint64_t m = ab_lo * np;
    __uint128_t mn = (__uint128_t)m * n;
    uint64_t mn_lo = (uint64_t)mn;
    uint64_t mn_hi = (uint64_t)(mn >> 64);
    uint64_t carry = (ab_lo + mn_lo) < ab_lo ? 1u : 0u;
    uint64_t u = ab_hi + mn_hi + carry;
    return u >= n ? u - n : u;
}

/* R^2 mod n = 2^128 mod n. */
static inline uint64_t mont_R2(uint64_t n) {
    uint64_t r = (-(uint64_t)n) % n;
    return (uint64_t)(((__uint128_t)r * r) % n);
}

/* Strong (Miller-Rabin) pseudoprime test for base a modulo n. */
static int strong_mrt(uint64_t n, uint64_t a,
                      uint64_t np, uint64_t R2,
                      uint64_t d, int s) {
    uint64_t one_m = mont_mul(1, R2, n, np);
    uint64_t nm1_m = mont_mul(n - 1, R2, n, np);
    uint64_t b = mont_mul(a % n, R2, n, np);
    uint64_t x = one_m;
    uint64_t e = d;
    while (e) {
        if (e & 1) x = mont_mul(x, b, n, np);
        b = mont_mul(b, b, n, np);
        e >>= 1;
    }
    if (x == one_m || x == nm1_m) return 1;
    for (int r = 1; r < s; r++) {
        x = mont_mul(x, x, n, np);
        if (x == nm1_m) return 1;
    }
    return 0;
}

int primality_miller_rabin_u64(uint64_t n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (!(n & 1) || n % 3 == 0) return 0;

    static const uint64_t small[] = {5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (size_t i = 0; i < sizeof(small) / sizeof(*small); ++i) {
        if (n == small[i]) return 1;
        if (n % small[i] == 0) return 0;
    }

    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) {
        d >>= 1;
        s++;
    }

    uint64_t np = mont_ninv(n);
    uint64_t R2 = mont_R2(n);
    static const uint64_t bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (size_t i = 0; i < 7; i++) {
        uint64_t a = bases[i] % n;
        if (a == 0) continue;
        if (!strong_mrt(n, a, np, R2, d, s)) return 0;
    }
    return 1;
}

int primality_fast_fermat_u64(uint64_t n) {
    if (n < 4) return n >= 2;
    if (!(n & 1)) return 0;

    uint64_t d = n - 1;
    int s = 0;
    while (!(d & 1)) {
        d >>= 1;
        s++;
    }

    uint64_t np = mont_ninv(n);
    uint64_t R2 = mont_R2(n);
    if (!strong_mrt(n, 2, np, R2, d, s)) return 0;
    if (!strong_mrt(n, 3, np, R2, d, s)) return 0;

    return 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * Multi-limb CIOS Montgomery core.
 *
 * Mirrors cpugapminer's CPU path (and the GPU kernel in gpu_fermat.cu)
 * with dynamic limb count, little-endian 64-bit words.
 * Supports up to PRIMALITY_CPU_MAX_LIMBS limbs (1280 bits).
 * ═══════════════════════════════════════════════════════════════════ */

/* Multiply-accumulate: *acc += a × b + carry.  Returns high 64 bits. */
static inline uint64_t cpu_mac(uint64_t *acc, uint64_t a, uint64_t b,
                                uint64_t carry)
{
    unsigned __int128 p = (unsigned __int128)a * b + carry;
    uint64_t lo = (uint64_t)p;
    uint64_t hi = (uint64_t)(p >> 64);
#if defined(__x86_64__) || defined(_M_X64)
    unsigned long long out = 0;
    unsigned char c = _addcarry_u64(0,
                                    (unsigned long long)(*acc),
                                    (unsigned long long)lo,
                                    &out);
    *acc = (uint64_t)out;
    return hi + (uint64_t)c;
#else
    uint64_t prev = *acc;
    *acc = prev + lo;
    hi += (*acc < prev);
    return hi;
#endif
}

/* a >= b  (nl limbs, big-endian comparison) */
static inline int cpu_gte_n(const uint64_t *a, const uint64_t *b, int nl)
{
    for (int i = nl - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1; /* equal */
}

/* r = a − b  (nl limbs, unsigned, no overflow check) */
static inline void cpu_sub_n(uint64_t *r, const uint64_t *a,
                             const uint64_t *b, int nl)
{
    uint64_t borrow = 0;
    for (int i = 0; i < nl; i++) {
        uint64_t ai = a[i], bi = b[i];
        uint64_t d   = ai - bi;
        uint64_t b1  = (ai < bi);
        uint64_t d2  = d - borrow;
        uint64_t b2  = (d < borrow);
        r[i]   = d2;
        borrow = b1 + b2;
    }
}

/* a = 2a mod n  (modular doubling) */
static inline void cpu_moddbl_n(uint64_t *a, const uint64_t *n, int nl)
{
    uint64_t carry = 0;
    for (int i = 0; i < nl; i++) {
        uint64_t v = a[i];
        a[i]  = (v << 1) | carry;
        carry = v >> 63;
    }
    if (carry || cpu_gte_n(a, n, nl))
        cpu_sub_n(a, a, n, nl);
}

/* r = R mod n,  R = 2^(64*nl).
 *
 * Start from r = 2^topbit, where topbit is the highest set bit in n.
 * Since n is odd and has bit topbit set, 2^topbit < n, so this seed is
 * already a valid reduced value in [0, n).  Then double exactly
 * (64*nl - topbit) times modulo n to reach 2^(64*nl) mod n.
 *
 * Do not subtract n from the seed: that underflows in unsigned limb
 * arithmetic when 2^topbit < n and corrupts the representation. */
static inline void cpu_rmodn_n(uint64_t *r, const uint64_t *n, int nl)
{
    /* nl is significant-limb count; top limb is non-zero. */
    int top_limb = nl - 1;
    int top_bit_in_limb = 63 - __builtin_clzll(n[top_limb]);
    int topbit = top_limb * 64 + top_bit_in_limb;

    /* r = 2^topbit (single bit), guaranteed < n */
    for (int i = 0; i < nl; i++) r[i] = 0;
    r[top_limb] = (uint64_t)1 << top_bit_in_limb;

    /* double (64*nl - topbit) times -> r = R mod n */
    int remaining = 64 * nl - topbit;
    for (int i = 0; i < remaining; i++)
        cpu_moddbl_n(r, n, nl);
}

/* Montgomery multiplication: r = a · b · R⁻¹ mod n  (CIOS algorithm)
 * Requires n odd, 0 ≤ a, b < n < R = 2^(64*nl). */
static inline void cpu_montmul_n(uint64_t *r,
                                 const uint64_t *a,
                                 const uint64_t *b,
                                 const uint64_t *n,
                                 uint64_t ninv, int nl)
{
    uint64_t tbuf[2 * PRIMALITY_CPU_MAX_LIMBS + 4];
    for (int i = 0; i < 2 * nl + 4; i++) tbuf[i] = 0;
    uint64_t *t = tbuf;

    for (int i = 0; i < nl; i++) {
        uint64_t c = 0;
        for (int j = 0; j < nl; j++)
            c = cpu_mac(&t[j], a[i], b[j], c);
        uint64_t old = t[nl];
        t[nl] += c;
        t[nl + 1] += (t[nl] < old);

        uint64_t m = t[0] * ninv;
        c = 0;
        for (int j = 0; j < nl; j++)
            c = cpu_mac(&t[j], m, n[j], c);
        old = t[nl];
        t[nl] += c;
        t[nl + 1] += (t[nl] < old);

        /* Logical left-shift by one limb (drop t[0], append 0)
           without copying nl+1 words each iteration. */
        t++;
        t[nl + 1] = 0;
    }

    if (t[nl] || cpu_gte_n(t, n, nl))
        cpu_sub_n(r, t, n, nl);
    else
        for (int i = 0; i < nl; i++) r[i] = t[i];
}

#if defined(__x86_64__) || defined(_M_X64)
/*
 * ADX CIOS Montgomery inner loop: t[0..nl] += ai * b[0..nl-1].
 *
 * Pointer-based ping-pong carry: t and b are pointer register inputs with
 * register-indirect addressing; %%rax is hardcoded scratch, %%rcx is pinned
 * to 0 for the flush.  CF/OF are live across the entire asm block.
 *
 * Ping-pong carry: _SPA outputs new hi into %[c1] (reads carry via OF),
 *                 _SPB outputs new hi into %[c0].
 * Saves one movq per step vs a single carry register.
 * Even-NL sequence ends with _SPB → final hi in c0 → _FLUSH_PP.
 * Odd-NL  sequence ends with _SPA → final hi in c1 → _FLUSH_PA.
 * mulx does NOT affect flags, so new-hi can be written into c0/c1 without
 * disturbing the CF/OF chains.
 */
#define _SPA(j) \
    "mulx "#j"*8(%[bp]), %[lo], %[c1]  \n\t" \
    "movq "#j"*8(%[tp]), %%rax          \n\t" \
    "adox %[c0],         %%rax          \n\t" \
    "adcx %[lo],         %%rax          \n\t" \
    "movq %%rax,         "#j"*8(%[tp])  \n\t"
#define _SPB(j) \
    "mulx "#j"*8(%[bp]), %[lo], %[c0]  \n\t" \
    "movq "#j"*8(%[tp]), %%rax          \n\t" \
    "adox %[c1],         %%rax          \n\t" \
    "adcx %[lo],         %%rax          \n\t" \
    "movq %%rax,         "#j"*8(%[tp])  \n\t"

/* Flush when final hi is in c0 (even NL, ends with _SPB): */
#define _FLUSH_PP(nl_bytes, nl1_bytes) \
    "movq  "#nl_bytes"(%[tp]),  %%rax    \n\t" \
    "adox  %[c0],    %%rax               \n\t" \
    "adcx  %%rcx,    %%rax               \n\t" \
    "movq  %%rax,    "#nl_bytes"(%[tp])  \n\t" \
    "movq  "#nl1_bytes"(%[tp]), %%rax    \n\t" \
    "adox  %%rcx,    %%rax               \n\t" \
    "adcx  %%rcx,    %%rax               \n\t" \
    "movq  %%rax,    "#nl1_bytes"(%[tp]) \n\t"
/* Flush when final hi is in c1 (odd NL, ends with _SPA): */
#define _FLUSH_PA(nl_bytes, nl1_bytes) \
    "movq  "#nl_bytes"(%[tp]),  %%rax    \n\t" \
    "adox  %[c1],    %%rax               \n\t" \
    "adcx  %%rcx,    %%rax               \n\t" \
    "movq  %%rax,    "#nl_bytes"(%[tp])  \n\t" \
    "movq  "#nl1_bytes"(%[tp]), %%rax    \n\t" \
    "adox  %%rcx,    %%rax               \n\t" \
    "adcx  %%rcx,    %%rax               \n\t" \
    "movq  %%rax,    "#nl1_bytes"(%[tp]) \n\t"

#define ADX_ADDMUL_PP(name, steps, flush, nl_bytes, nl1_bytes) \
__attribute__((always_inline)) \
static inline void name(uint64_t *t, uint64_t ai, const uint64_t *b) \
{ \
    uint64_t lo, c0, c1; \
    __asm__ volatile( \
        "xorl %%ecx,  %%ecx         \n\t" \
        "xorl %k[c0], %k[c0]        \n\t" \
        "xorl %k[c1], %k[c1]        \n\t" \
        steps \
        flush(nl_bytes, nl1_bytes) \
        : [lo]"=&r"(lo), [c0]"=&r"(c0), [c1]"=&r"(c1) \
        : "d"(ai), [tp]"r"(t), [bp]"r"(b) \
        : "cc", "rax", "rcx", "memory" \
    ); \
}

/* NL=2 (even): _SPA(0) _SPB(1) → ends B → c0=hi1 → FLUSH_PP */
ADX_ADDMUL_PP(adx_addmul_2,
    _SPA(0) _SPB(1),
    _FLUSH_PP, 16, 24)
/* NL=3 (odd): ...→ _SPA(2) → c1=hi2 → FLUSH_PA */
ADX_ADDMUL_PP(adx_addmul_3,
    _SPA(0) _SPB(1) _SPA(2),
    _FLUSH_PA, 24, 32)
ADX_ADDMUL_PP(adx_addmul_4,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3),
    _FLUSH_PP, 32, 40)
ADX_ADDMUL_PP(adx_addmul_5,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4),
    _FLUSH_PA, 40, 48)
ADX_ADDMUL_PP(adx_addmul_6,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5),
    _FLUSH_PP, 48, 56)
ADX_ADDMUL_PP(adx_addmul_7,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6),
    _FLUSH_PA, 56, 64)
ADX_ADDMUL_PP(adx_addmul_8,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7),
    _FLUSH_PP, 64, 72)
ADX_ADDMUL_PP(adx_addmul_9,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8),
    _FLUSH_PA, 72, 80)
ADX_ADDMUL_PP(adx_addmul_10,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9),
    _FLUSH_PP, 80, 88)
ADX_ADDMUL_PP(adx_addmul_11,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10),
    _FLUSH_PA, 88, 96)
ADX_ADDMUL_PP(adx_addmul_12,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11),
    _FLUSH_PP, 96, 104)
ADX_ADDMUL_PP(adx_addmul_13,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12),
    _FLUSH_PA, 104, 112)
ADX_ADDMUL_PP(adx_addmul_14,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13),
    _FLUSH_PP, 112, 120)
ADX_ADDMUL_PP(adx_addmul_15,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14),
    _FLUSH_PA, 120, 128)
ADX_ADDMUL_PP(adx_addmul_16,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14) _SPB(15),
    _FLUSH_PP, 128, 136)
ADX_ADDMUL_PP(adx_addmul_17,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14) _SPB(15) _SPA(16),
    _FLUSH_PA, 136, 144)
ADX_ADDMUL_PP(adx_addmul_18,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14) _SPB(15) _SPA(16) _SPB(17),
    _FLUSH_PP, 144, 152)
ADX_ADDMUL_PP(adx_addmul_19,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14) _SPB(15) _SPA(16) _SPB(17) _SPA(18),
    _FLUSH_PA, 152, 160)
ADX_ADDMUL_PP(adx_addmul_20,
    _SPA(0) _SPB(1) _SPA(2) _SPB(3) _SPA(4) _SPB(5) _SPA(6) _SPB(7) _SPA(8) _SPB(9) _SPA(10) _SPB(11) _SPA(12) _SPB(13) _SPA(14) _SPB(15) _SPA(16) _SPB(17) _SPA(18) _SPB(19),
    _FLUSH_PP, 160, 168)
/* NL=1 (odd): _SPA(0) → c1=hi0 → FLUSH_PA */
ADX_ADDMUL_PP(adx_addmul_1,
    _SPA(0),
    _FLUSH_PA, 8, 16)

#undef ADX_ADDMUL_PP
#undef _SPA
#undef _SPB
#undef _FLUSH_PP
#undef _FLUSH_PA

__attribute__((always_inline))
static inline void adx_addmul(uint64_t *t, uint64_t ai, const uint64_t *b, int nl)
{
    switch (nl) {
        case  1: adx_addmul_1 (t, ai, b); break;
        case  2: adx_addmul_2 (t, ai, b); break;
        case  3: adx_addmul_3 (t, ai, b); break;
        case  4: adx_addmul_4 (t, ai, b); break;
        case  5: adx_addmul_5 (t, ai, b); break;
        case  6: adx_addmul_6 (t, ai, b); break;
        case  7: adx_addmul_7 (t, ai, b); break;
        case  8: adx_addmul_8 (t, ai, b); break;
        case  9: adx_addmul_9 (t, ai, b); break;
        case 10: adx_addmul_10(t, ai, b); break;
        case 11: adx_addmul_11(t, ai, b); break;
        case 12: adx_addmul_12(t, ai, b); break;
        case 13: adx_addmul_13(t, ai, b); break;
        case 14: adx_addmul_14(t, ai, b); break;
        case 15: adx_addmul_15(t, ai, b); break;
        case 16: adx_addmul_16(t, ai, b); break;
        case 17: adx_addmul_17(t, ai, b); break;
        case 18: adx_addmul_18(t, ai, b); break;
        case 19: adx_addmul_19(t, ai, b); break;
        default: adx_addmul_20(t, ai, b); break;
    }
}

/*
 * Per-exact-NL specialized Montgomery multiply.
 * Unlike the runtime-NL variants, these hardcode NL so GCC can:
 *   - constant-fold the outer loop to NL iterations (no loop counter)
 *   - replace the adx_addmul() switch with a direct adx_addmul_N call
 *   - elide the tbuf zero-init beyond 2*NL+2 words
 */
#define DECL_MONTMUL_EXACT_ADX(FN, NL, ADDMULFN) \
__attribute__((noinline)) \
static void FN(uint64_t *r, \
               const uint64_t *a, \
               const uint64_t *b, \
               const uint64_t *n, \
               uint64_t ninv) \
{ \
    uint64_t tbuf[2 * (NL) + 4] = {0}; \
    uint64_t *t = tbuf; \
    for (int i = 0; i < (NL); i++) { \
        ADDMULFN(t, a[i], b); \
        uint64_t m = t[0] * ninv; \
        ADDMULFN(t, m, n); \
        t++; \
        t[(NL) + 1] = 0; \
    } \
    if (t[NL] || cpu_gte_n(t, n, NL)) \
        cpu_sub_n(r, t, n, NL); \
    else \
        for (int i = 0; i < (NL); i++) r[i] = t[i]; \
}

DECL_MONTMUL_EXACT_ADX(cpu_montmul_n2,  2,  adx_addmul_2)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n3,  3,  adx_addmul_3)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n4,  4,  adx_addmul_4)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n5,  5,  adx_addmul_5)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n6,  6,  adx_addmul_6)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n7,  7,  adx_addmul_7)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n8,  8,  adx_addmul_8)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n9,  9,  adx_addmul_9)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n10, 10, adx_addmul_10)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n11, 11, adx_addmul_11)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n12, 12, adx_addmul_12)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n13, 13, adx_addmul_13)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n14, 14, adx_addmul_14)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n15, 15, adx_addmul_15)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n16, 16, adx_addmul_16)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n17, 17, adx_addmul_17)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n18, 18, adx_addmul_18)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n19, 19, adx_addmul_19)
DECL_MONTMUL_EXACT_ADX(cpu_montmul_n20, 20, adx_addmul_20)

/* Dispatch to exact-NL montmul; portable CIOS when ADX is unavailable. */
#define MONTMUL_EXACT_DISPATCH(r, a, b, n, ninv, nlimbs) \
    do { \
        primality_runtime_init_adx_once(); \
        if (!g_primality_runtime_adx_enabled) { \
            cpu_montmul_n((r), (a), (b), (n), (ninv), (nlimbs)); \
        } else { \
            switch (nlimbs) { \
            case  2: cpu_montmul_n2 (r,a,b,n,ninv); break; \
            case  3: cpu_montmul_n3 (r,a,b,n,ninv); break; \
            case  4: cpu_montmul_n4 (r,a,b,n,ninv); break; \
            case  5: cpu_montmul_n5 (r,a,b,n,ninv); break; \
            case  6: cpu_montmul_n6 (r,a,b,n,ninv); break; \
            case  7: cpu_montmul_n7 (r,a,b,n,ninv); break; \
            case  8: cpu_montmul_n8 (r,a,b,n,ninv); break; \
            case  9: cpu_montmul_n9 (r,a,b,n,ninv); break; \
            case 10: cpu_montmul_n10(r,a,b,n,ninv); break; \
            case 11: cpu_montmul_n11(r,a,b,n,ninv); break; \
            case 12: cpu_montmul_n12(r,a,b,n,ninv); break; \
            case 13: cpu_montmul_n13(r,a,b,n,ninv); break; \
            case 14: cpu_montmul_n14(r,a,b,n,ninv); break; \
            case 15: cpu_montmul_n15(r,a,b,n,ninv); break; \
            case 16: cpu_montmul_n16(r,a,b,n,ninv); break; \
            case 17: cpu_montmul_n17(r,a,b,n,ninv); break; \
            case 18: cpu_montmul_n18(r,a,b,n,ninv); break; \
            case 19: cpu_montmul_n19(r,a,b,n,ninv); break; \
            default: cpu_montmul_n20(r,a,b,n,ninv); break; \
            } \
        } \
    } while (0)

/* ADX CIOS montmul with ping-pong carry is already optimal; SOS square
 * Phase-2 bit-shift dependency chain costs more than the cross-product
 * savings on ADX hardware, so squaring uses montmul on x86-64. */
#define MONTSQR_EXACT_DISPATCH(r, a, b, n, ninv, nlimbs) \
    MONTMUL_EXACT_DISPATCH((r), (a), (b), (n), (ninv), (nlimbs))

#else /* non-x86: portable CIOS + SOS squaring */

/*
 * Non-ADX exact-NL Montgomery squaring: SOS triangle shortcut via cpu_mac.
 * Phase 1: upper-triangle cross products.  NL(NL-1)/2 multiplications.
 * Phase 2: double tbuf in-place (NL constant → GCC fully unrolls at -O3).
 * Phase 3: diagonal a[i]^2 added to tbuf[2i]; carry ripples to tbuf[2i+2].
 * Phase 4: SOS Montgomery reduction.  NL² multiplications.
 * Total muls: NL(NL-1)/2 + NL + NL² ≈ 3NL²/2  vs  2NL² for CIOS montmul.
 */
#define DECL_MONTSQR_EXACT(NL) \
__attribute__((noinline)) \
static void cpu_montsqr_n##NL(uint64_t *r, \
                               const uint64_t *a, \
                               const uint64_t *b_unused, \
                               const uint64_t *n, \
                               uint64_t ninv) \
{ \
    (void)b_unused; \
    uint64_t tbuf[2*(NL)+4] = {0}; \
    for (int i = 0; i < (NL)-1; i++) { \
        uint64_t ai = a[i], c = 0; \
        for (int j = i+1; j < (NL); j++) \
            c = cpu_mac(&tbuf[i+j], ai, a[j], c); \
        uint64_t old = tbuf[i+(NL)]; \
        tbuf[i+(NL)] += c; \
        tbuf[i+(NL)+1] += (tbuf[i+(NL)] < old); \
    } \
    { uint64_t cy = 0; \
      for (int k = 0; k < 2*(NL)+2; k++) { \
          uint64_t ncy = tbuf[k] >> 63; \
          tbuf[k] = (tbuf[k] << 1) | cy; cy = ncy; } \
      tbuf[2*(NL)+2] = cy; } \
    for (int i = 0; i < (NL); i++) { \
        uint64_t hi = cpu_mac(&tbuf[2*i], a[i], a[i], 0); \
        uint64_t old1 = tbuf[2*i+1]; tbuf[2*i+1] += hi; \
        if (tbuf[2*i+1] < old1) { \
            uint64_t old2 = ++tbuf[2*i+2]; \
            if (old2 == 0) tbuf[2*i+3]++; \
        } \
    } \
    for (int i = 0; i < (NL); i++) { \
        uint64_t m = tbuf[i] * ninv, c = 0; \
        for (int j = 0; j < (NL); j++) \
            c = cpu_mac(&tbuf[i+j], m, n[j], c); \
        uint64_t old = tbuf[i+(NL)]; \
        tbuf[i+(NL)] += c; \
        tbuf[i+(NL)+1] += (tbuf[i+(NL)] < old); \
    } \
    uint64_t *t = tbuf + (NL); \
    if (t[NL] || cpu_gte_n(t, n, (NL))) \
        cpu_sub_n(r, t, n, (NL)); \
    else \
        for (int i = 0; i < (NL); i++) r[i] = t[i]; \
}

DECL_MONTSQR_EXACT(2)
DECL_MONTSQR_EXACT(3)
DECL_MONTSQR_EXACT(4)
DECL_MONTSQR_EXACT(5)
DECL_MONTSQR_EXACT(6)
DECL_MONTSQR_EXACT(7)
DECL_MONTSQR_EXACT(8)
DECL_MONTSQR_EXACT(9)
DECL_MONTSQR_EXACT(10)
DECL_MONTSQR_EXACT(11)
DECL_MONTSQR_EXACT(12)
DECL_MONTSQR_EXACT(13)
DECL_MONTSQR_EXACT(14)
DECL_MONTSQR_EXACT(15)
DECL_MONTSQR_EXACT(16)
DECL_MONTSQR_EXACT(17)
DECL_MONTSQR_EXACT(18)
DECL_MONTSQR_EXACT(19)
DECL_MONTSQR_EXACT(20)

/* SOS squaring threshold: noinline call overhead exceeds savings for small
 * NL.  NL≤9: generic CIOS montmul; NL≥10: SOS (benchmarked +13-26% for
 * NL=10-18 on non-ADX hardware). */
#define MONTSQR_EXACT_DISPATCH(r, a, b, n, ninv, nlimbs) \
    do { \
        switch (nlimbs) { \
        case 10: cpu_montsqr_n10(r,a,b,n,ninv); break; \
        case 11: cpu_montsqr_n11(r,a,b,n,ninv); break; \
        case 12: cpu_montsqr_n12(r,a,b,n,ninv); break; \
        case 13: cpu_montsqr_n13(r,a,b,n,ninv); break; \
        case 14: cpu_montsqr_n14(r,a,b,n,ninv); break; \
        case 15: cpu_montsqr_n15(r,a,b,n,ninv); break; \
        case 16: cpu_montsqr_n16(r,a,b,n,ninv); break; \
        case 17: cpu_montsqr_n17(r,a,b,n,ninv); break; \
        case 18: cpu_montsqr_n18(r,a,b,n,ninv); break; \
        case 19: cpu_montsqr_n19(r,a,b,n,ninv); break; \
        case 20: cpu_montsqr_n20(r,a,b,n,ninv); break; \
        default: MONTMUL_EXACT_DISPATCH((r),(a),(b),(n),(ninv),(nlimbs)); break; \
        } \
    } while (0)

#endif /* __x86_64__ */

/* Fallback for non-x86 builds: generic CIOS montmul. */
#ifndef MONTMUL_EXACT_DISPATCH
#define MONTMUL_EXACT_DISPATCH(r, a, b, n, ninv, nlimbs) \
    cpu_montmul_n((r), (a), (b), (n), (ninv), (nlimbs))
#endif

/* If no squaring dispatch was defined (portable non-x86 build), fall back
   to the montmul dispatch. */
#ifndef MONTSQR_EXACT_DISPATCH
#define MONTSQR_EXACT_DISPATCH(r, a, b, n, ninv, nlimbs) \
    MONTMUL_EXACT_DISPATCH((r), (a), (b), (n), (ninv), (nlimbs))
#endif

static int fermat_u64_exact(uint64_t n)
{
    if (n < 4) return n >= 2;
    if ((n & 1) == 0) return 0;

    uint64_t e = n - 1;
    uint64_t acc = 1 % n;
    uint64_t base = 2 % n;
    while (e) {
        if (e & 1)
            acc = (uint64_t)(((__uint128_t)acc * base) % n);
        base = (uint64_t)(((__uint128_t)base * base) % n);
        e >>= 1;
    }
    return acc == 1;
}

/* Base-2 Euler–Plumb criterion for a single 64-bit limb. */
static inline int euler_u64(uint64_t n)
{
    if (n < 4) return n >= 2;
    if (!(n & 1)) return 0;
    uint64_t e = (n - 1) >> 1;
    uint64_t acc = 1, base = 2 % n;
    while (e) {
        if (e & 1) acc = (uint64_t)(((__uint128_t)acc * base) % n);
        base = (uint64_t)(((__uint128_t)base * base) % n);
        e >>= 1;
    }
    return acc == 1 || acc == n - 1;
}

/*
 * 4-bit fixed-window left-to-right exponentiation:
 *   Precompute win[i] = base^(2i+1) mod n, i=0..7  (8 odd powers)
 *   Scan exponent MSB→LSB, extracting 4-bit chunks w:
 *     - If w == 0: square 4 times
 *     - Else: find trailing zero count z of w, square (4-z) times,
 *             multiply by win[(w>>z)>>1], square z more times.
 */
#define FERMAT_WIN  4                  /* window width in bits   */
#define FERMAT_WINSZ (1 << FERMAT_WIN) /* 16 entries: powers 1..15 */
#define FERMAT_WIN_MIN 3
#define FERMAT_WIN_MAX 5
#define FERMAT_WINSZ_MAX (1 << FERMAT_WIN_MAX)

/* Helper: extract 4-bit chunk starting at bit position `bit` in e[],
 * scanning from top limb downward.  Returns value 0..15. */
static inline uint32_t cpu_get_bits4(const uint64_t *e, int bit) {
    int limb = bit >> 6;
    int off  = bit & 63;
    uint32_t w = (uint32_t)(e[limb] >> off) & 0xF;
    /* If the 4-bit window straddles a limb boundary, pick up the remaining
     * bits from the next-higher limb. */
    if (off > 60 && limb + 1 < PRIMALITY_CPU_MAX_LIMBS)
        w |= (uint32_t)(e[limb + 1] << (64 - off)) & 0xF;
    return w;
}

/* Generic little-endian bit extraction for runtime-selected window sizes. */
static inline uint32_t cpu_get_bits_w(const uint64_t *e, int bit_lo,
                                      int win_bits, int nlimbs) {
    uint32_t w = 0;
    for (int k = 0; k < win_bits; k++) {
        int b = bit_lo + k;
        int limb = b >> 6;
        if (limb >= nlimbs) break;
        w |= (uint32_t)(((e[limb] >> (b & 63)) & 1u) << k);
    }
    return w;
}

/* Temporary runtime controls for adaptive CPU window selection:
 *   GAPMINER_CPU_WINDOW_OVERRIDE=3|4|5  -> force fixed window size
 *   GAPMINER_CPU_WINDOW_LOG=1           -> log selected window once per nlimbs */
static int g_cpu_window_cfg_checked = 0;
static int g_cpu_window_override = 0; /* 0=auto, else 3..5 */
static int g_cpu_window_log = 0;
static unsigned int g_cpu_window_logged_mask = 0;

static inline void cpu_window_cfg_init_once(void) {
    if (g_cpu_window_cfg_checked)
        return;

    const char *ov = getenv("GAPMINER_CPU_WINDOW_OVERRIDE");
    if (ov && *ov) {
        int v = atoi(ov);
        if (v >= FERMAT_WIN_MIN && v <= FERMAT_WIN_MAX)
            g_cpu_window_override = v;
    }

    const char *lg = getenv("GAPMINER_CPU_WINDOW_LOG");
    if (lg && *lg && strcmp(lg, "0") != 0)
        g_cpu_window_log = 1;

    g_cpu_window_cfg_checked = 1;
}

/* Adaptive CPU window policy: default to the exact-NL 4-bit path (dedicated
 * per-NL specializations); win=3/5 routes through the generic dynwin loop.
 * Env override allows forcing 3/5 for experiments. */
static inline int cpu_window_bits_for_nlimbs(int nlimbs) {
    cpu_window_cfg_init_once();

    int w;
    if (g_cpu_window_override >= FERMAT_WIN_MIN &&
        g_cpu_window_override <= FERMAT_WIN_MAX) {
        w = g_cpu_window_override;
    } else {
        (void)nlimbs;
        w = FERMAT_WIN;
    }

    if (g_cpu_window_log && nlimbs > 0 && nlimbs <= 31) {
        unsigned int bit = 1u << (unsigned int)nlimbs;
        if ((g_cpu_window_logged_mask & bit) == 0u) {
            g_cpu_window_logged_mask |= bit;
            fprintf(stderr,
                    "[cpu-window] nlimbs=%d -> win=%d (%s)\n",
                    nlimbs, w,
                    (g_cpu_window_override ? "override" : "auto"));
            fflush(stderr);
        }
    }

    return w;
}

static inline int cpu_find_msb_n(const uint64_t *a, int nlimbs)
{
    int top = nlimbs - 1;
    while (top > 0 && a[top] == 0)
        top--;
    return top * 64 + 63 - __builtin_clzll(a[top]);
}

static int cpu_pow2_window_from_precomp(uint64_t *res,
                                        const primality_exact_precomp_t *precomp,
                                        const uint64_t *exp,
                                        int msb)
{
    int nlimbs = precomp->nlimbs;
    const uint64_t *n = precomp->n;
    uint64_t ninv = precomp->ninv;

    memcpy(res, precomp->base_m, (size_t)nlimbs * sizeof(uint64_t));
    int bit = msb - 1;
    while (bit >= 0) {
        if (bit < FERMAT_WIN - 1) {
            MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
            if ((exp[bit >> 6] >> (bit & 63)) & 1u)
                MONTMUL_EXACT_DISPATCH(res, res, precomp->base_m, n, ninv, nlimbs);
            bit--;
        } else {
            uint32_t w = cpu_get_bits4(exp, bit - (FERMAT_WIN - 1));
            if (w == 0) {
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= FERMAT_WIN;
            } else {
                int z = __builtin_ctz(w);
                int sq = FERMAT_WIN - z;
                for (int step = 0; step < sq; step++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTMUL_EXACT_DISPATCH(res, res, precomp->win[(w >> z) >> 1], n, ninv, nlimbs);
                for (int step = 0; step < z; step++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= FERMAT_WIN;
            }
        }
    }
    return 1;
}

int primality_exact_precomp_init(primality_exact_precomp_t *precomp,
                                 const uint64_t *n, int nlimbs)
{
    uint64_t one_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t base2_m[PRIMALITY_CPU_MAX_LIMBS];

    if (!precomp || !n)
        return 0;
    if (nlimbs <= 0 || nlimbs > PRIMALITY_CPU_MAX_LIMBS)
        return 0;
    if (nlimbs == 1 || (n[0] & 1u) == 0)
        return 0;

    memset(precomp, 0, sizeof(*precomp));
    precomp->nlimbs = nlimbs;
    precomp->win_bits = cpu_window_bits_for_nlimbs(nlimbs);
    if (precomp->win_bits != FERMAT_WIN)
        return 0;

    memcpy(precomp->n, n, (size_t)nlimbs * sizeof(uint64_t));
    precomp->ninv = mont_ninv(n[0]);

    cpu_rmodn_n(one_m, n, nlimbs);
    memcpy(precomp->base_m, one_m, (size_t)nlimbs * sizeof(uint64_t));
    cpu_moddbl_n(precomp->base_m, n, nlimbs);

    memcpy(precomp->win[0], precomp->base_m, (size_t)nlimbs * sizeof(uint64_t));
    MONTSQR_EXACT_DISPATCH(base2_m, precomp->base_m, precomp->base_m,
                           precomp->n, precomp->ninv, nlimbs);
    for (int idx = 1; idx < PRIMALITY_PRECOMP_WIN_ODD_COUNT; idx++)
        MONTMUL_EXACT_DISPATCH(precomp->win[idx], precomp->win[idx - 1], base2_m,
                               precomp->n, precomp->ninv, nlimbs);

    memcpy(precomp->fermat_exp, n, (size_t)nlimbs * sizeof(uint64_t));
    precomp->fermat_exp[0] -= 1;
    memcpy(precomp->nm1, precomp->fermat_exp, (size_t)nlimbs * sizeof(uint64_t));
    precomp->fermat_msb = cpu_find_msb_n(precomp->fermat_exp, nlimbs);

    memcpy(precomp->euler_exp, precomp->nm1, (size_t)nlimbs * sizeof(uint64_t));
    for (int idx = 0; idx < nlimbs - 1; idx++)
        precomp->euler_exp[idx] = (precomp->euler_exp[idx] >> 1) |
                                  (precomp->euler_exp[idx + 1] << 63);
    precomp->euler_exp[nlimbs - 1] >>= 1;
    precomp->euler_msb = cpu_find_msb_n(precomp->euler_exp, nlimbs);
    return 1;
}

int fermat_test_cpu_nlimbs_precomp(const primality_exact_precomp_t *precomp)
{
    uint64_t res[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t one[PRIMALITY_CPU_MAX_LIMBS] = {1};

    if (!precomp || precomp->nlimbs <= 1 || precomp->win_bits != FERMAT_WIN)
        return 0;

    cpu_pow2_window_from_precomp(res, precomp, precomp->fermat_exp, precomp->fermat_msb);
    MONTMUL_EXACT_DISPATCH(res, res, one, precomp->n, precomp->ninv, precomp->nlimbs);
    if (res[0] != 1)
        return 0;
    for (int idx = 1; idx < precomp->nlimbs; idx++)
        if (res[idx] != 0)
            return 0;
    return 1;
}

int euler_test_cpu_nlimbs_precomp(const primality_exact_precomp_t *precomp)
{
    uint64_t res[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t one[PRIMALITY_CPU_MAX_LIMBS] = {1};

    if (!precomp || precomp->nlimbs <= 1 || precomp->win_bits != FERMAT_WIN)
        return 0;

    cpu_pow2_window_from_precomp(res, precomp, precomp->euler_exp, precomp->euler_msb);
    MONTMUL_EXACT_DISPATCH(res, res, one, precomp->n, precomp->ninv, precomp->nlimbs);
    if (res[0] == 1) {
        int ok = 1;
        for (int idx = 1; idx < precomp->nlimbs; idx++) {
            if (res[idx] != 0) {
                ok = 0;
                break;
            }
        }
        if (ok)
            return 1;
    }
    for (int idx = 0; idx < precomp->nlimbs; idx++)
        if (res[idx] != precomp->nm1[idx])
            return 0;
    return 1;
}

static int fermat_test_cpu_nlimbs_dynwin(const uint64_t *n, int nlimbs, int win_bits)
{
    if (nlimbs <= 0 || nlimbs > PRIMALITY_CPU_MAX_LIMBS) return 0;
    if (nlimbs == 1) return fermat_u64_exact(n[0]);
    if ((n[0] & 1) == 0) return 0;
    if (win_bits < FERMAT_WIN_MIN || win_bits > FERMAT_WIN_MAX)
        win_bits = FERMAT_WIN;

    uint64_t ninv = mont_ninv(n[0]);
    uint64_t one_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t base_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t base2_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t res[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t e[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t one[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t win[FERMAT_WINSZ_MAX / 2][PRIMALITY_CPU_MAX_LIMBS];

    int win_odd_count = 1 << (win_bits - 1);

    cpu_rmodn_n(one_m, n, nlimbs);
    memcpy(base_m, one_m, (size_t)nlimbs * sizeof(uint64_t));
    cpu_moddbl_n(base_m, n, nlimbs);

    memcpy(win[0], base_m, (size_t)nlimbs * sizeof(uint64_t));
    MONTSQR_EXACT_DISPATCH(base2_m, base_m, base_m, n, ninv, nlimbs);
    for (int i = 1; i < win_odd_count; i++)
        MONTMUL_EXACT_DISPATCH(win[i], win[i - 1], base2_m, n, ninv, nlimbs);

    memcpy(e, n, (size_t)nlimbs * sizeof(uint64_t));
    e[0] -= 1;

    int top = nlimbs - 1;
    while (top > 0 && e[top] == 0) top--;
    int msb = top * 64 + 63 - __builtin_clzll(e[top]);

    memcpy(res, base_m, (size_t)nlimbs * sizeof(uint64_t));
    int bit = msb - 1;
    while (bit >= 0) {
        if (bit < win_bits - 1) {
            MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
            if ((e[bit >> 6] >> (bit & 63)) & 1u)
                MONTMUL_EXACT_DISPATCH(res, res, base_m, n, ninv, nlimbs);
            bit--;
        } else {
            uint32_t w = cpu_get_bits_w(e, bit - (win_bits - 1), win_bits, nlimbs);
            if (w == 0) {
                for (int i = 0; i < win_bits; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= win_bits;
            } else {
                int z = __builtin_ctz(w);
                int sq = win_bits - z;
                for (int i = 0; i < sq; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTMUL_EXACT_DISPATCH(res, res, win[(w >> z) >> 1], n, ninv, nlimbs);
                for (int i = 0; i < z; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= win_bits;
            }
        }
    }

    memset(one, 0, (size_t)nlimbs * sizeof(uint64_t));
    one[0] = 1;
    MONTMUL_EXACT_DISPATCH(res, res, one, n, ninv, nlimbs);
    if (res[0] != 1) return 0;
    for (int i = 1; i < nlimbs; i++) if (res[i] != 0) return 0;
    return 1;
}

static int euler_test_cpu_nlimbs_dynwin(const uint64_t *n, int nlimbs, int win_bits)
{
    if (nlimbs <= 0 || nlimbs > PRIMALITY_CPU_MAX_LIMBS) return 0;
    if (nlimbs == 1) return euler_u64(n[0]);
    if ((n[0] & 1) == 0) return 0;
    if (win_bits < FERMAT_WIN_MIN || win_bits > FERMAT_WIN_MAX)
        win_bits = FERMAT_WIN;

    uint64_t ninv = mont_ninv(n[0]);
    uint64_t one_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t base_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t base2_m[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t res[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t e[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t one[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t nm1[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t one_std[PRIMALITY_CPU_MAX_LIMBS];
    uint64_t win[FERMAT_WINSZ_MAX / 2][PRIMALITY_CPU_MAX_LIMBS];

    int win_odd_count = 1 << (win_bits - 1);

    cpu_rmodn_n(one_m, n, nlimbs);
    memcpy(base_m, one_m, (size_t)nlimbs * sizeof(uint64_t));
    cpu_moddbl_n(base_m, n, nlimbs);

    memcpy(win[0], base_m, (size_t)nlimbs * sizeof(uint64_t));
    MONTSQR_EXACT_DISPATCH(base2_m, base_m, base_m, n, ninv, nlimbs);
    for (int i = 1; i < win_odd_count; i++)
        MONTMUL_EXACT_DISPATCH(win[i], win[i - 1], base2_m, n, ninv, nlimbs);

    memcpy(e, n, (size_t)nlimbs * sizeof(uint64_t));
    e[0] -= 1;
    for (int i = 0; i < nlimbs - 1; i++)
        e[i] = (e[i] >> 1) | (e[i + 1] << 63);
    e[nlimbs - 1] >>= 1;

    int top = nlimbs - 1;
    while (top > 0 && e[top] == 0) top--;
    int msb = top * 64 + 63 - __builtin_clzll(e[top]);

    memcpy(res, base_m, (size_t)nlimbs * sizeof(uint64_t));
    int bit = msb - 1;
    while (bit >= 0) {
        if (bit < win_bits - 1) {
            MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
            if ((e[bit >> 6] >> (bit & 63)) & 1u)
                MONTMUL_EXACT_DISPATCH(res, res, base_m, n, ninv, nlimbs);
            bit--;
        } else {
            uint32_t w = cpu_get_bits_w(e, bit - (win_bits - 1), win_bits, nlimbs);
            if (w == 0) {
                for (int i = 0; i < win_bits; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= win_bits;
            } else {
                int z = __builtin_ctz(w);
                int sq = win_bits - z;
                for (int i = 0; i < sq; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                MONTMUL_EXACT_DISPATCH(res, res, win[(w >> z) >> 1], n, ninv, nlimbs);
                for (int i = 0; i < z; i++)
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, nlimbs);
                bit -= win_bits;
            }
        }
    }

    memset(one, 0, (size_t)nlimbs * sizeof(uint64_t));
    one[0] = 1;
    MONTMUL_EXACT_DISPATCH(res, res, one, n, ninv, nlimbs);

    if (res[0] == 1) {
        int ok = 1;
        for (int i = 1; i < nlimbs; i++) if (res[i] != 0) { ok = 0; break; }
        if (ok) return 1;
    }

    memset(one_std, 0, (size_t)nlimbs * sizeof(uint64_t));
    one_std[0] = 1;
    cpu_sub_n(nm1, n, one_std, nlimbs);
    for (int i = 0; i < nlimbs; i++)
        if (res[i] != nm1[i]) return 0;
    return 1;
}

/*
 * Per-exact-NL Fermat test.  NL is a compile-time constant, so
 * MONTMUL_EXACT_DISPATCH resolves to a direct cpu_montmul_nN() call,
 * eliminating the runtime adx_addmul() switch in the inner loop.
 */
#define DECL_FERMAT_EXACT(NL) \
__attribute__((noinline)) \
static int fermat_test_cpu_nlimbs_##NL(const uint64_t *n) \
{ \
    if ((n[0] & 1) == 0) return 0; \
    if ((NL) == 1) return fermat_u64_exact(n[0]); \
    uint64_t ninv = mont_ninv(n[0]); \
    uint64_t one_m[NL], base_m[NL], base2_m[NL]; \
    uint64_t win[FERMAT_WINSZ / 2][NL]; \
    cpu_rmodn_n(one_m, n, (NL)); \
    memcpy(base_m, one_m, (NL) * sizeof(uint64_t)); \
    cpu_moddbl_n(base_m, n, (NL)); \
    memcpy(win[0], base_m, (NL) * sizeof(uint64_t)); \
    MONTSQR_EXACT_DISPATCH(base2_m, base_m, base_m, n, ninv, (NL)); \
    for (int _w = 1; _w < FERMAT_WINSZ / 2; _w++) \
        MONTMUL_EXACT_DISPATCH(win[_w], win[_w-1], base2_m, n, ninv, (NL)); \
    uint64_t e[NL]; \
    memcpy(e, n, (NL) * sizeof(uint64_t)); \
    e[0] -= 1; \
    int top = (NL) - 1; \
    while (top > 0 && e[top] == 0) top--; \
    int msb = top * 64 + 63 - __builtin_clzll(e[top]); \
    uint64_t res[NL]; \
    memcpy(res, base_m, (NL) * sizeof(uint64_t)); \
    int bit = msb - 1; \
    while (bit >= 0) { \
        if (bit < FERMAT_WIN - 1) { \
            MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
            if ((e[bit >> 6] >> (bit & 63)) & 1) \
                MONTMUL_EXACT_DISPATCH(res, res, base_m, n, ninv, (NL)); \
            bit--; \
        } else { \
            uint32_t w = cpu_get_bits4(e, bit - (FERMAT_WIN - 1)); \
            if (w == 0) { \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                bit -= FERMAT_WIN; \
            } else { \
                int z = __builtin_ctz(w); \
                int sq = FERMAT_WIN - z; \
                for (int _s = 0; _s < sq; _s++) \
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTMUL_EXACT_DISPATCH(res, res, win[(w >> z) >> 1], n, ninv, (NL)); \
                for (int _s = 0; _s < z; _s++) \
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                bit -= FERMAT_WIN; \
            } \
        } \
    } \
    uint64_t one[NL]; \
    memset(one, 0, (NL) * sizeof(uint64_t)); \
    one[0] = 1; \
    MONTMUL_EXACT_DISPATCH(res, res, one, n, ninv, (NL)); \
    if (res[0] != 1) return 0; \
    for (int i = 1; i < (NL); i++) if (res[i] != 0) return 0; \
    return 1; \
}

DECL_FERMAT_EXACT(2)
DECL_FERMAT_EXACT(3)
DECL_FERMAT_EXACT(4)
DECL_FERMAT_EXACT(5)
DECL_FERMAT_EXACT(6)
DECL_FERMAT_EXACT(7)
DECL_FERMAT_EXACT(8)
DECL_FERMAT_EXACT(9)
DECL_FERMAT_EXACT(10)
DECL_FERMAT_EXACT(11)
DECL_FERMAT_EXACT(12)
DECL_FERMAT_EXACT(13)
DECL_FERMAT_EXACT(14)
DECL_FERMAT_EXACT(15)
DECL_FERMAT_EXACT(16)
DECL_FERMAT_EXACT(17)
DECL_FERMAT_EXACT(18)
DECL_FERMAT_EXACT(19)
DECL_FERMAT_EXACT(20)

int fermat_test_cpu_nlimbs(const uint64_t *n, int nlimbs)
{
    if (nlimbs <= 0 || nlimbs > PRIMALITY_CPU_MAX_LIMBS)
        return 0;
    if (nlimbs == 1)
        return fermat_u64_exact(n[0]);

    int win_bits = cpu_window_bits_for_nlimbs(nlimbs);
    if (win_bits != FERMAT_WIN)
        return fermat_test_cpu_nlimbs_dynwin(n, nlimbs, win_bits);

    switch (nlimbs) {
    case 2:  return fermat_test_cpu_nlimbs_2(n);
    case 3:  return fermat_test_cpu_nlimbs_3(n);
    case 4:  return fermat_test_cpu_nlimbs_4(n);
    case 5:  return fermat_test_cpu_nlimbs_5(n);
    case 6:  return fermat_test_cpu_nlimbs_6(n);
    case 7:  return fermat_test_cpu_nlimbs_7(n);
    case 8:  return fermat_test_cpu_nlimbs_8(n);
    case 9:  return fermat_test_cpu_nlimbs_9(n);
    case 10: return fermat_test_cpu_nlimbs_10(n);
    case 11: return fermat_test_cpu_nlimbs_11(n);
    case 12: return fermat_test_cpu_nlimbs_12(n);
    case 13: return fermat_test_cpu_nlimbs_13(n);
    case 14: return fermat_test_cpu_nlimbs_14(n);
    case 15: return fermat_test_cpu_nlimbs_15(n);
    case 16: return fermat_test_cpu_nlimbs_16(n);
    case 17: return fermat_test_cpu_nlimbs_17(n);
    case 18: return fermat_test_cpu_nlimbs_18(n);
    case 19: return fermat_test_cpu_nlimbs_19(n);
    case 20: return fermat_test_cpu_nlimbs_20(n);
    default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * Multi-limb CPU Euler–Plumb test: 2^((n-1)/2) ≡ ±1 (mod n)
 *
 * Halves the squaring count vs the Fermat test by using the exponent
 * (n-1)/2 instead of (n-1).  The final comparison is done in Montgomery
 * form, avoiding one extra Montgomery multiply per call.
 *
 * one_m  = R mod n           (Montgomery form of 1)
 * nm1_m  = n − one_m         (Montgomery form of n−1)
 * Accept iff res == one_m  OR  res == nm1_m.
 * ═══════════════════════════════════════════════════════════════════ */

#define DECL_EULER_EXACT(NL) \
__attribute__((noinline)) \
static int euler_test_cpu_nlimbs_##NL(const uint64_t *n) \
{ \
    if ((n[0] & 1) == 0) return 0; \
    if ((NL) == 1) return euler_u64(n[0]); \
    uint64_t ninv = mont_ninv(n[0]); \
    uint64_t one_m[NL], base_m[NL], base2_m[NL]; \
    uint64_t win[FERMAT_WINSZ / 2][NL]; \
    cpu_rmodn_n(one_m, n, (NL)); \
    memcpy(base_m, one_m, (NL) * sizeof(uint64_t)); \
    cpu_moddbl_n(base_m, n, (NL)); \
    memcpy(win[0], base_m, (NL) * sizeof(uint64_t)); \
    MONTSQR_EXACT_DISPATCH(base2_m, base_m, base_m, n, ninv, (NL)); \
    for (int _w = 1; _w < FERMAT_WINSZ / 2; _w++) \
        MONTMUL_EXACT_DISPATCH(win[_w], win[_w-1], base2_m, n, ninv, (NL)); \
    uint64_t e[NL]; \
    memcpy(e, n, (NL) * sizeof(uint64_t)); \
    e[0] -= 1; \
    for (int _i = 0; _i < (NL) - 1; _i++) \
        e[_i] = (e[_i] >> 1) | (e[_i + 1] << 63); \
    e[(NL) - 1] >>= 1; \
    int top = (NL) - 1; \
    while (top > 0 && e[top] == 0) top--; \
    int msb = top * 64 + 63 - __builtin_clzll(e[top]); \
    uint64_t res[NL]; \
    memcpy(res, base_m, (NL) * sizeof(uint64_t)); \
    int bit = msb - 1; \
    while (bit >= 0) { \
        if (bit < FERMAT_WIN - 1) { \
            MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
            if ((e[bit >> 6] >> (bit & 63)) & 1) \
                MONTMUL_EXACT_DISPATCH(res, res, base_m, n, ninv, (NL)); \
            bit--; \
        } else { \
            uint32_t w = cpu_get_bits4(e, bit - (FERMAT_WIN - 1)); \
            if (w == 0) { \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                bit -= FERMAT_WIN; \
            } else { \
                int z = __builtin_ctz(w); \
                int sq = FERMAT_WIN - z; \
                for (int _s = 0; _s < sq; _s++) \
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                MONTMUL_EXACT_DISPATCH(res, res, win[(w >> z) >> 1], n, ninv, (NL)); \
                for (int _s = 0; _s < z; _s++) \
                    MONTSQR_EXACT_DISPATCH(res, res, res, n, ninv, (NL)); \
                bit -= FERMAT_WIN; \
            } \
        } \
    } \
    uint64_t one_conv[NL]; \
    memset(one_conv, 0, (NL) * sizeof(uint64_t)); \
    one_conv[0] = 1; \
    MONTMUL_EXACT_DISPATCH(res, res, one_conv, n, ninv, (NL)); \
    if (res[0] == 1) { \
        int ok = 1; \
        for (int _i = 1; _i < (NL); _i++) if (res[_i]) { ok = 0; break; } \
        if (ok) return 1; \
    } \
    uint64_t nm1[NL], one_std[NL]; \
    memset(one_std, 0, (NL) * sizeof(uint64_t)); \
    one_std[0] = 1; \
    cpu_sub_n(nm1, n, one_std, (NL)); \
    int eq = 1; \
    for (int _i = 0; _i < (NL); _i++) if (res[_i] != nm1[_i]) { eq = 0; break; } \
    return eq; \
}

DECL_EULER_EXACT(2)
DECL_EULER_EXACT(3)
DECL_EULER_EXACT(4)
DECL_EULER_EXACT(5)
DECL_EULER_EXACT(6)
DECL_EULER_EXACT(7)
DECL_EULER_EXACT(8)
DECL_EULER_EXACT(9)
DECL_EULER_EXACT(10)
DECL_EULER_EXACT(11)
DECL_EULER_EXACT(12)
DECL_EULER_EXACT(13)
DECL_EULER_EXACT(14)
DECL_EULER_EXACT(15)
DECL_EULER_EXACT(16)
DECL_EULER_EXACT(17)
DECL_EULER_EXACT(18)
DECL_EULER_EXACT(19)
DECL_EULER_EXACT(20)

int euler_test_cpu_nlimbs(const uint64_t *n, int nlimbs)
{
    if (nlimbs <= 0 || nlimbs > PRIMALITY_CPU_MAX_LIMBS)
        return 0;
    if (nlimbs == 1)
        return euler_u64(n[0]);

    int win_bits = cpu_window_bits_for_nlimbs(nlimbs);
    if (win_bits != FERMAT_WIN)
        return euler_test_cpu_nlimbs_dynwin(n, nlimbs, win_bits);

    switch (nlimbs) {
    case 2:  return euler_test_cpu_nlimbs_2(n);
    case 3:  return euler_test_cpu_nlimbs_3(n);
    case 4:  return euler_test_cpu_nlimbs_4(n);
    case 5:  return euler_test_cpu_nlimbs_5(n);
    case 6:  return euler_test_cpu_nlimbs_6(n);
    case 7:  return euler_test_cpu_nlimbs_7(n);
    case 8:  return euler_test_cpu_nlimbs_8(n);
    case 9:  return euler_test_cpu_nlimbs_9(n);
    case 10: return euler_test_cpu_nlimbs_10(n);
    case 11: return euler_test_cpu_nlimbs_11(n);
    case 12: return euler_test_cpu_nlimbs_12(n);
    case 13: return euler_test_cpu_nlimbs_13(n);
    case 14: return euler_test_cpu_nlimbs_14(n);
    case 15: return euler_test_cpu_nlimbs_15(n);
    case 16: return euler_test_cpu_nlimbs_16(n);
    case 17: return euler_test_cpu_nlimbs_17(n);
    case 18: return euler_test_cpu_nlimbs_18(n);
    case 19: return euler_test_cpu_nlimbs_19(n);
    case 20: return euler_test_cpu_nlimbs_20(n);
    default: return 0;
    }
}

int primality_cpu_adx_compiled(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#else
    return 0;
#endif
}

int primality_cpu_adx_enabled(void)
{
#if defined(__x86_64__) || defined(_M_X64)
    primality_runtime_init_adx_once();
    return g_primality_runtime_adx_enabled;
#else
    return 0;
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * mpz adapters + limb helpers for the worker CPU paths.
 *
 * Workers keep candidates as mpz_t (GMP).  The limb tests run on
 * little-endian uint64_t arrays, so candidates are exported once and —
 * because sieve offsets ascend within a window — updated by 64-bit
 * delta increments between consecutive candidates instead of a full
 * GMP conversion per candidate.
 * ═══════════════════════════════════════════════════════════════════ */

int primality_limbs_effective_nl(const uint64_t *limbs, int max_nl) {
    int nl = max_nl;
    while (nl > 1 && limbs[nl - 1] == 0)
        nl--;
    return nl;
}

/* Add 64-bit delta to little-endian limb array in place.
   Returns 1 on success, 0 if carry overflows past max_nl. */
int primality_limbs_add_u64(uint64_t *limbs, int max_nl, uint64_t delta) {
    if (delta == 0)
        return 1;

    uint64_t old = limbs[0];
    limbs[0] = old + delta;
    uint64_t carry = (limbs[0] < old) ? 1u : 0u;
    int i = 1;
    while (carry && i < max_nl) {
        old = limbs[i];
        limbs[i] = old + carry;
        carry = (limbs[i] < old) ? 1u : 0u;
        i++;
    }
    return carry ? 0 : 1;
}

/* Export a non-negative mpz into little-endian 64-bit limbs.
   Returns the effective limb count (1..max_nl), or 0 when the number
   does not fit (or is negative). */
int primality_limbs_export(const mpz_t n, uint64_t *limbs, int max_nl) {
    if (!n || !limbs || max_nl <= 0)
        return 0;
    if (mpz_sgn(n) < 0)
        return 0;
    if (mpz_sizeinbase(n, 2) > (size_t)max_nl * 64U)
        return 0;

    memset(limbs, 0, (size_t)max_nl * sizeof(uint64_t));
    size_t count = (size_t)max_nl;
    mpz_export(limbs, &count, -1 /* LSW first */, sizeof(uint64_t),
               0 /* native endian */, 0 /* no nails */, n);
    return primality_limbs_effective_nl(limbs, max_nl);
}

int primality_euler_limbs_mpz(const mpz_t n) {
    uint64_t limbs[PRIMALITY_CPU_MAX_LIMBS];
    int nl = primality_limbs_export(n, limbs, PRIMALITY_CPU_MAX_LIMBS);
    if (nl <= 0)
        return 0;
    return euler_test_cpu_nlimbs(limbs, nl);
}

int primality_fermat_limbs_mpz(const mpz_t n) {
    uint64_t limbs[PRIMALITY_CPU_MAX_LIMBS];
    int nl = primality_limbs_export(n, limbs, PRIMALITY_CPU_MAX_LIMBS);
    if (nl <= 0)
        return 0;
    return fermat_test_cpu_nlimbs(limbs, nl);
}
