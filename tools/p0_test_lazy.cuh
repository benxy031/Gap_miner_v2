#pragma once
/* p0_test_lazy.cuh - OUR lazy-Barrett Euler/Jacobi test (32-bit limbs).
 *
 * WHAT IT IS
 *   A base-2 Euler/Euler-Jacobi probable-prime test for a modulus
 *       n = n0 + n1*2^32 + HI*2^64          (HI a compile-time constant)
 *   that keeps the running representative LAZY (< ~12n) so a squaring needs
 *   only five 32x32->64 wide multiplies and a ONE-DIGIT Barrett correction.
 *   The predicate is identical to perig's (tools/perig.cuh):
 *       2^((n-1)/2) mod n  ==  (+1) or (-1), the sign chosen by the
 *       Jacobi bit ((n>>1) ^ (n>>2)) & 1.
 *   It is therefore directly A/B-able against perig and gated bit-exactly
 *   against it on >= 1e6 candidates (tools/bench_lazy.cu).
 *
 * WHY THIS SHAPE (measured 2026-10-26, on our own card)
 *   The reference engine's advantage is NOT carry chains or a cached
 *   reciprocal (both measured at ~4 % on our code): it is that a 3x32-bit
 *   representation with a SMALL TOP LIMB collapses the square to five wide
 *   multiplies and the reduction to one quotient digit - see
 *   docs/PGN_SNAPSHOT_REVIEW_2026-10-20.md and the head-to-head bench.
 *   Our perig pays 128-bit __int128 products plus an explicit normalisation.
 *
 * INVARIANTS (checked by the bench, not by hope)
 *   - the doubling/reduction keeps a < 12n, so a.w2 <= 12*HI and a square is
 *     at most 2^152 (five 32-bit words);
 *   - u = floor(2^160 / n) with its remainder used as the exponent seed;
 *   - NO normalisation inside the 3-bit groups; one final one-digit pass.
 *
 * NO code from the PGN snapshot is used here; the technique (Barrett with a
 * truncated high product, lazy representatives) is public and is cited.
 */

#include <stdint.h>

#ifdef __CUDACC__
#define P0L_HD __host__ __device__ __forceinline__
#else
#define P0L_HD inline
#endif

namespace p0lazy {

struct U96 { uint32_t w[3]; };
struct U192 { uint32_t w[6]; };

/* portable count-leading-zeros for a 64-bit value.  The host branch must be
   plain C++: MSVC (the Windows DLL build's host compiler) has no
   __builtin_clzll, and it parses this body even though the hot path never
   calls it (only tools/dbg_lazy.cpp does). */
P0L_HD int clz64(uint64_t x) {
#ifdef __CUDA_ARCH__
    return __clzll((long long)x);
#else
    if (!x) return 64;
    int n = 0;
    while (!(x & (1ull << 63))) { x <<= 1; ++n; }
    return n;
#endif
}

P0L_HD int ge96(const U96 &a, const U96 &b) {
    for (int i = 2; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] > b.w[i];
    return 1;
}
/* a -= b, returns the final borrow (0 = no borrow) */
P0L_HD uint32_t sub96(U96 &a, const U96 &b) {
    uint32_t c = 0;
    for (int i = 0; i < 3; ++i) {
        uint64_t d = (uint64_t)a.w[i] - b.w[i] - c;
        a.w[i] = (uint32_t)d;
        c = (uint32_t)(d >> 63);
    }
    return c;
}
P0L_HD uint32_t add96(U96 &a, const U96 &b) {
    uint64_t c = 0;
    for (int i = 0; i < 3; ++i) {
        uint64_t s = (uint64_t)a.w[i] + b.w[i] + c;
        a.w[i] = (uint32_t)s;
        c = s >> 32;
    }
    return (uint32_t)c;
}
P0L_HD U96 shiftSmall96(const U96 &a, unsigned s) {
    if (!s) return a;
    return U96{{(uint32_t)(a.w[0] << s),
                (uint32_t)((a.w[1] << s) | (a.w[0] >> (32 - s))),
                (uint32_t)((a.w[2] << s) | (a.w[1] >> (32 - s)))}};
}
P0L_HD U96 twice96(const U96 &a) { return shiftSmall96(a, 1); }

P0L_HD U192 shl192(const U96 &n, unsigned bits) {
    U192 r{};
    const unsigned words = bits / 32, part = bits % 32;
    for (unsigned i = 0; i < 3; ++i) {
        r.w[i + words] |= (uint32_t)(n.w[i] << part);
        if (part) r.w[i + words + 1] |= (uint32_t)(n.w[i] >> (32 - part));
    }
    return r;
}
P0L_HD U192 mulDigit192(const U192 &a, uint32_t d) {
    U192 r{};
    uint64_t c = 0;
    for (int i = 0; i < 6; ++i) {
        uint64_t p = (uint64_t)a.w[i] * d + c;
        r.w[i] = (uint32_t)p;
        c = p >> 32;
    }
    return r;
}
P0L_HD uint32_t sub192(U192 &a, const U192 &b) {
    uint32_t c = 0;
    for (int i = 0; i < 6; ++i) {
        uint64_t d = (uint64_t)a.w[i] - b.w[i] - c;
        a.w[i] = (uint32_t)d;
        c = (uint32_t)(d >> 63);
    }
    return c;
}
P0L_HD uint32_t add192(U192 &a, const U192 &b) {
    uint64_t c = 0;
    for (int i = 0; i < 6; ++i) {
        uint64_t s = (uint64_t)a.w[i] + b.w[i] + c;
        a.w[i] = (uint32_t)s;
        c = s >> 32;
    }
    return (uint32_t)c;
}
P0L_HD int ge192(const U192 &a, const U192 &b) {
    for (int i = 5; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] > b.w[i];
    return 1;
}

/* ---- containers shared by both instantiations -------------------------- */
struct U128 { uint32_t w[4]; };

/* two-word accumulator for one product column: lo and hi are each bounded by
   (number of terms) * 2^32, so no 64-bit overflow is possible */
struct Acc2 { uint64_t lo, hi; };
P0L_HD void accMul(Acc2 &t, uint32_t x, uint32_t y) {
    const uint64_t p = (uint64_t)x * y;
    t.lo += (uint32_t)p;
    t.hi += (p >> 32);
}
P0L_HD uint64_t accCarry(const Acc2 &t) { return t.hi + (t.lo >> 32); }

struct U224 { uint32_t w[7]; };
P0L_HD U224 shl224(const U96 &n, unsigned bits) {
    U224 r{};
    const unsigned words = bits / 32, part = bits % 32;
    for (unsigned i = 0; i < 3; ++i) {
        r.w[i + words] |= (uint32_t)(n.w[i] << part);
        if (part) r.w[i + words + 1] |= (uint32_t)(n.w[i] >> (32 - part));
    }
    return r;
}
P0L_HD int ge224(const U224 &a, const U224 &b) {
    for (int i = 6; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] > b.w[i];
    return 1;
}
P0L_HD uint32_t sub224(U224 &a, const U224 &b) {
    uint32_t c = 0;
    for (int i = 0; i < 7; ++i) {
        uint64_t d = (uint64_t)a.w[i] - b.w[i] - c;
        a.w[i] = (uint32_t)d;
        c = (uint32_t)(d >> 63);
    }
    return c;
}
P0L_HD uint32_t add224(U224 &a, const U224 &b) {
    uint64_t c = 0;
    for (int i = 0; i < 7; ++i) {
        uint64_t s = (uint64_t)a.w[i] + b.w[i] + c;
        a.w[i] = (uint32_t)s;
        c = s >> 32;
    }
    return (uint32_t)c;
}
P0L_HD U224 mulDigit224(const U224 &a, uint32_t d) {
    U224 r{};
    uint64_t c = 0;
    for (int i = 0; i < 7; ++i) {
        uint64_t p = (uint64_t)a.w[i] * d + c;
        r.w[i] = (uint32_t)p;
        c = p >> 32;
    }
    return r;
}

/* u = floor(2^192/n) (four words) and rem = 2^192 mod n, eight radix-2^16
   quotient digits.  The float only HINTS a digit; every accepted digit is
   proven by exact 224-bit subtraction/add-back, so nothing is decided by
   floating point.
 *
 * WHY 192 AND NOT 160: the Barrett quotient is an underestimate, and the size
 * of that underestimate is the truncation error A/2^R with A = S>>64 ~ (kn)^2
 * / 2^64.  With R = 160-64 = 96 that term grows like (n/2^64)^2: it is 0.0009
 * for n ~ 1e20 (invisible, which is why the first version passed its gates)
 * but O(10) for n ~ 2^75, so each squaring added several multiples of n to the
 * residue, the residue grew, the error grew with it, and the thing DIVERGED -
 * the representative left the 96-bit container and silently wrapped.  With
 * R = 128 the term is A/2^128, i.e. 2^-32 across the whole band, so the
 * residue stays within a few n and the band below 2^75 is safe. */
P0L_HD U128 reciprocal192(const U96 &n, U96 *remOut) {
    U224 rem{};
    rem.w[6] = 1u;                                  /* 2^192 */
    U128 q{};
    const float den = (float)n.w[2] * 4294967296.0f + (float)n.w[1] +
                      (float)n.w[0] * 0x1p-32f;
    const float rh = 1.0f / den;
    for (int k = 7; k >= 0; --k) {
        const unsigned sh = (unsigned)k * 16;
        const int top = (16 * k + 83) / 32;
        const float hiScale = (k & 1) ? 281474976710656.0f : 4294967296.0f;
        const float loScale = (k & 1) ? 65536.0f : 1.0f;
        const float est = (float)rem.w[top] * hiScale + (float)rem.w[top - 1] * loScale;
        const float hv = est * rh;
        uint32_t digit = hv >= 65535.0f ? 65535u : (hv > 0.0f ? (uint32_t)hv : 0u);
        const U224 unit = shl224(n, sh);
        const U224 prod = mulDigit224(unit, digit);
        int neg = -((int)sub224(rem, prod));
        while (neg < 0) {                 /* digit too big: add one unit back */
            neg += (int)add224(rem, unit);
            --digit;
        }
        while (ge224(rem, unit)) {        /* digit too small: subtract more */
            sub224(rem, unit);
            ++digit;
        }
        q.w[k / 2] |= digit << ((k % 2) * 16);
    }
    if (remOut) *remOut = U96{{rem.w[0], rem.w[1], rem.w[2]}};
    return q;
}

/* One lazy squaring: exact 96-bit square (five words) then a ONE-word Barrett
   quotient taken from the corner of (S>>64) * u.
   Invariants: n = n2*2^64 + low with n2 <= 2^11, and the representative
   a * 2^v < 2^80 so the square stays inside five words. */
P0L_HD U96 squareLazy(const U96 &a, const U96 &n, const U128 &u) {
    const uint64_t p00 = (uint64_t)a.w[0] * a.w[0];
    const uint64_t p01 = (uint64_t)a.w[0] * a.w[1];
    const uint64_t p02 = (uint64_t)a.w[0] * a.w[2];
    const uint64_t p11 = (uint64_t)a.w[1] * a.w[1];
    const uint64_t p12 = (uint64_t)a.w[1] * a.w[2];
    /* a2 stays tiny (<= ~30 for a lazy representative below ~3n), so a2*a2
       fits one 32-bit word: a plain 32-bit multiply instead of a wide one. */
    const uint32_t p22 = a.w[2] * a.w[2];
    /* column sums */
    uint32_t a0 = (uint32_t)p00;
    uint64_t s = (p00 >> 32) + 2ull * (uint32_t)p01;
    uint32_t a1 = (uint32_t)s;
    s = (s >> 32) + 2ull * (p01 >> 32) + 2ull * (uint32_t)p02 + (uint32_t)p11;
    uint32_t a2 = (uint32_t)s;
    s = (s >> 32) + 2ull * (p02 >> 32) + (p11 >> 32) + 2ull * (uint32_t)p12;
    uint32_t a3 = (uint32_t)s;
    s = (s >> 32) + 2ull * (p12 >> 32) + p22;
    uint32_t a4 = (uint32_t)s;

    /* Barrett quotient with R = 128: q = ((S>>64) * u) >> 128, so the product
       words 4..6 are q's three words.  A = (a2,a3,a4) is three words and u is
       four, so the corner needs columns 3,4,5 only: 3+2+1 = six multiplies.
       Columns 0..2 are dropped, which only UNDER-estimates q (the safe
       direction) and costs at most ~2 units. */
    Acc2 c3{}, c4{}, c5{};
    accMul(c3, a2, u.w[3]); accMul(c3, a3, u.w[2]); accMul(c3, a4, u.w[1]);
    accMul(c4, a3, u.w[3]); accMul(c4, a4, u.w[2]);
    accMul(c5, a4, u.w[3]);
    uint64_t cy = accCarry(c3);
    uint64_t t = c4.lo + cy;  const uint32_t q0 = (uint32_t)t;  cy = c4.hi + (t >> 32);
    t = c5.lo + cy;           const uint32_t q1 = (uint32_t)t;
    const uint32_t q2 = (uint32_t)(c5.hi + (t >> 32));

    /* low three words of q*n, subtracted from the low three words of S: the
       residue r' = S - q*n is >= 0 and < ~4n < 2^77, so the truncation is
       exact and no add-back is needed.  Six multiplies. */
    Acc2 m0{}, m1{}, m2{};
    accMul(m0, q0, n.w[0]);
    accMul(m1, q1, n.w[0]); accMul(m1, q0, n.w[1]);
    accMul(m2, q2, n.w[0]); accMul(m2, q1, n.w[1]); accMul(m2, q0, n.w[2]);
    cy = accCarry(m0);
    const uint32_t w0 = (uint32_t)m0.lo;
    t = m1.lo + cy; const uint32_t w1 = (uint32_t)t; cy = m1.hi + (t >> 32);
    t = m2.lo + cy; const uint32_t w2 = (uint32_t)t;
    U96 out{{a0, a1, a2}};
    sub96(out, U96{{w0, w1, w2}});
    /* NO normalisation here: the residue is r' = S - q*n < ~3n and the next
       squaring is valid for any representative below 2^76 (Barrett's quotient
       stays within 2 of floor(S/n) at every step, so the bound is stable).
       Normalising on every squaring cost ~2 three-word subtractions per square
       and made the test SLOWER than perig; one normalisation at the end of the
       ladder is enough. */
    return out;
}

/* final normalisation: n = HI*2^64 + low means n/2^64 is in [HI, HI+1), so
   dividing the top word by HI+1 UNDER-estimates floor(a/n) - which is the safe
   direction.  Dividing by HI over-estimates and wraps the value (that was the
   last defect: every prime came back wrong).  The top limb is read from the
   modulus, so ONE body serves both a compile-time HI (the band-specialised
   bench and its gates) and a runtime one (the walk-engine dispatch). */
P0L_HD U96 normalizeFinal(U96 a, const U96 &n) {
    const uint32_t q = a.w[2] / (n.w[2] + 1u);
    uint64_t p = (uint64_t)q * n.w[0];
    U96 m{{(uint32_t)p, 0, 0}};
    p = (uint64_t)q * n.w[1] + (p >> 32);
    m.w[1] = (uint32_t)p;
    m.w[2] = n.w[2] * q + (uint32_t)(p >> 32);
    sub96(a, m);
    if (ge96(a, n)) sub96(a, n);
    return a;
}

/* The test itself: n = low + hi*2^64, Euler/Euler-Jacobi base 2.
   BAND: 1 <= hi < 2^11, i.e. 2^64 <= n < 2^75.  Three words hold the lazy
   representative a < 12n < 2^79 and the Barrett quotient corner (A = S>>64
   must stay three words).  A wider modulus needs the four-limb instantiation
   below, which measured SLOWER than perig at 107 bits. */
P0L_HD bool test_hi(uint64_t low, uint32_t HI) {
    if (!(low & 1)) return false;
    const U96 n{{(uint32_t)low, (uint32_t)(low >> 32), HI}};
    U96 rem{};
    const U128 u = reciprocal192(n, &rem);
    (void)rem;

    /* seed: start from 2 and run a plain binary exponentiation over the bits of
       e = (n-1)/2.  The 3-bit grouping of the exponent is a throughput detail;
       correctness first - the lever under test is the SQUARING, and the plain
       loop exercises it exactly as often as the grouped form does. */
    /* exponent e = (n-1)/2 where n = HI*2^64 + low.  This is a 67-bit number
       for HI = 10: e = (HI/2)*2^64 + (low-1)/2 when HI is even.  Using only
       (low-1)>>1 asks the wrong question - it was the defect that made every
       prime come back composite.  Split it into a high and a low half and run
       the ladder over both. */
    const uint64_t e_lo = (HI & 1u) ? ((low - 1) >> 1) + (1ull << 63)
                                    : ((low - 1) >> 1);
    const uint32_t e_hi = HI >> 1;          /* HI even -> exact; HI odd would
                                               need a carry that HI=10 avoids */
    U96 a{{1, 0, 0}};                       /* accumulate from 1 */
    if (e_hi) {
        int hb = -1;                        /* portable top bit of e_hi */
        for (uint32_t t = e_hi; t; t >>= 1) hb++;
        for (int bit = hb; bit >= 0; --bit) {
            a = squareLazy(a, n, u);
            if ((e_hi >> bit) & 1u) a = twice96(a);
        }
    }
    for (int bit = 63; bit >= 0; --bit) {
        a = squareLazy(a, n, u);
        if ((e_lo >> bit) & 1u) a = twice96(a);
    }
    a = normalizeFinal(a, n);
    /* fully reduce before comparing: the representative may still be a lazy
       multiple of n (congruent but not canonical), and the chain-level
       differential showed exactly that. */
    for (int g = 0; g < 24 && ge96(a, n); ++g) sub96(a, n);

    U96 want{{1, 0, 0}};
    if ((((low >> 1) ^ (low >> 2)) & 1)) {          /* Jacobi bit of 2 */
        want = n;
        sub96(want, U96{{1, 0, 0}});
    }
    return a.w[0] == want.w[0] && a.w[1] == want.w[1] && a.w[2] == want.w[2];
}

/* compile-time top limb: the same body with HI folded in, for the bench and
   the band-specialised gates (tools/bench_lazy.cu, tools/dbg_lazy.cpp) */
template <uint32_t HI>
P0L_HD bool test(uint64_t low) { return test_hi(low, HI); }

/* ==========================================================================
 * 4-limb instantiation: n = n0 + n1*2^32 + n2*2^64 + n3*2^96, n3 < 2^11
 * (the 1e32 campaign band: n = 1e32 -> n3 = 1266).
 *
 * Same lazy-Barrett idea, one limb wider.  The numbers that fix the shape:
 *   S = a^2  < 2^221        (a < 16n, see the invariant note at the end)
 *   A = S>>64 < 2^157       => FIVE 32-bit words A0..A4 = s2..s6
 *   u = floor(2^256/n)      => five words (2^147..2^160 for this band)
 *   q = (A*u) >> 192        => four words, and only columns 5..8 of the
 *                              product are needed, a 4+3+2+1 = 10-multiply
 *                              corner.  Columns 0..4 are dropped, which only
 *                              UNDER-estimates q (the safe direction); the
 *                              residue grows to at most 8n, still < 2^128,
 *                              so the low-word subtraction stays exact.
 *   q*n (low four words)    => 1+2+3+4 = 10 multiplies
 * => 10 (square) + 10 (quotient) + 10 (q*n) = 30 wide multiplies per squaring.
 *
 * DIFFERENCE FROM THE 3-LIMB VERSION: the exponent is walked ONE BIT at a
 * time, not in 3-bit groups.  A 3-bit group multiplies the lazy representative
 * by up to 128, and this container has only 128-107 = 21 bits of headroom, so
 * the five-word A assumption (A/2^192 negligible) would break; one-bit steps
 * keep a < 16n.  The extra cost is one four-word shift per bit.
 *
 * MEASURED VERDICT (2026-10-26, RTX 3070, 1M candidates, band 1e32):
 *   this four-limb test is 0.72x perig (2.494 vs 1.788 ns/test) - SLOWER.
 *   Do not dispatch it in place of perig.  The three-limb version is 1.37x
 *   perig in ITS band (2e20).  The difference is purely the limb count: perig
 *   always moves two 64-bit limbs (~30 IMADs per squaring) while ours costs
 *   18 IMADs at three limbs and 30 at four, plus four-word carry chains
 *   instead of native 64-bit adds.  See docs/PHASE0_scan_bench.md section 37.
 *   It is kept because it is gated and it is the honest boundary measurement.
 * ==========================================================================
 */

struct U160 { uint32_t w[5]; };

P0L_HD int ge128(const U128 &a, const U128 &b) {
    for (int i = 3; i >= 0; --i)
        if (a.w[i] != b.w[i]) return a.w[i] > b.w[i];
    return 1;
}
P0L_HD uint32_t sub128(U128 &a, const U128 &b) {
    uint32_t c = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t d = (uint64_t)a.w[i] - b.w[i] - c;
        a.w[i] = (uint32_t)d;
        c = (uint32_t)(d >> 63);
    }
    return c;
}
/* branchless a*2: the exponent walk uses it for every set bit */
P0L_HD U128 twice128(const U128 &a) {
    return U128{{(uint32_t)(a.w[0] << 1),
                 (uint32_t)((a.w[1] << 1) | (a.w[0] >> 31)),
                 (uint32_t)((a.w[2] << 1) | (a.w[1] >> 31)),
                 (uint32_t)((a.w[3] << 1) | (a.w[2] >> 31))}};
}
/* a - b without touching a (used to walk a running multiple back down) */
P0L_HD U128 sub128_ret(const U128 &a, const U128 &b) {
    U128 r = a;
    sub128(r, b);
    return r;
}

/* u = floor(2^256/n) and (optionally) rem = 2^256 mod n, in ten radix-2^16
   quotient digits.
 *
 * The state is a SCALED remainder: S_k = (2^256 - U_k*n) >> 16k, where U_k is
 * the quotient formed so far.  That makes every digit the same plain
 * "floor(S/n)" with ZERO shifted copies of n, so nothing is indexed by a
 * runtime offset.  (An earlier version shifted n by 16k each round; the
 * runtime word index forced the whole 288-bit carrier into LOCAL MEMORY and
 * made the setup 54% of the test instead of ~3%.)
 *
 *   S starts at 2^256 >> 144 = 2^112, and S_{k-1} = (S_k - d_k*n) * 2^16,
 *   so 0 <= S_k < n*2^16 always and every digit is < 2^16.
 *   The float only HINTS the digit; every accepted digit is proven by exact
 *   128-bit subtraction and corrected by adding/subtracting single copies of
 *   n, so no floating point decides anything.
 */
P0L_HD uint32_t add128(U128 &a, const U128 &b) {
    uint64_t c = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t s = (uint64_t)a.w[i] + b.w[i] + c;
        a.w[i] = (uint32_t)s;
        c = s >> 32;
    }
    return (uint32_t)c;
}
/* d * n for a 16-bit d; the final carry is provably 0 (d*n < 2^123) */
P0L_HD U128 mulSmall128(const U128 &n, uint32_t d) {
    U128 p{};
    uint64_t c = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t v = (uint64_t)n.w[i] * d + c;
        p.w[i] = (uint32_t)v;
        c = v >> 32;
    }
    return p;
}
P0L_HD U128 shl16_128(const U128 &a) {
    return U128{{(uint32_t)(a.w[0] << 16),
                 (uint32_t)((a.w[1] << 16) | (a.w[0] >> 16)),
                 (uint32_t)((a.w[2] << 16) | (a.w[1] >> 16)),
                 (uint32_t)((a.w[3] << 16) | (a.w[2] >> 16))}};
}

P0L_HD U160 reciprocal256(const U128 &n, U192 *remOut) {
    U128 S{{0, 0, 0, 0x00010000u}};          /* 2^256 >> 144 = 2^112 */
    const uint64_t nt = (uint64_t)n.w[2] | ((uint64_t)n.w[3] << 32);
    const float rn = 1.0f / (float)nt;
    U160 q{};
    for (int k = 9; k >= 0; --k) {
        const uint64_t top = (uint64_t)S.w[2] | ((uint64_t)S.w[3] << 32);
        const float hv = (float)top * rn;
        uint32_t d = hv >= 65536.0f ? 65535u : (hv > 0.0f ? (uint32_t)hv : 0u);
        U128 p = mulSmall128(n, d);
        while (d && ge128(p, S)) {           /* estimated one (or few) too big */
            --d;
            p = sub128_ret(p, n);
        }
        sub128(S, p);
        while (ge128(S, n)) {                /* and one (or few) too small */
            sub128(S, n);
            ++d;
        }
        q.w[k >> 1] |= d << ((k & 1) * 16);
        if (k) S = shl16_128(S);
    }
    if (remOut) *remOut = U192{{S.w[0], S.w[1], S.w[2], S.w[3], 0, 0}};
    return q;
}

/* One lazy squaring for four limbs.  Precondition: a < 16n (so a has four
   words, a.w3 < 2^16, S = a^2 < 2^221 and A = S>>64 has five words). */
P0L_HD U128 squareLazy4(const U128 &a, const U128 &n, const U160 &u) {
    const uint64_t p00 = (uint64_t)a.w[0] * a.w[0];
    const uint64_t p11 = (uint64_t)a.w[1] * a.w[1];
    const uint64_t p22 = (uint64_t)a.w[2] * a.w[2];
    const uint64_t p33 = (uint64_t)a.w[3] * a.w[3];   /* a.w3 < 2^16 -> fits 32 */
    const uint64_t p01 = (uint64_t)a.w[0] * a.w[1];
    const uint64_t p02 = (uint64_t)a.w[0] * a.w[2];
    const uint64_t p03 = (uint64_t)a.w[0] * a.w[3];
    const uint64_t p12 = (uint64_t)a.w[1] * a.w[2];
    const uint64_t p13 = (uint64_t)a.w[1] * a.w[3];
    const uint64_t p23 = (uint64_t)a.w[2] * a.w[3];

    /* exact 128-bit square, seven words; each column sum stays below 2^36 */
    uint64_t c = p00 >> 32;
    const uint32_t s0 = (uint32_t)p00;
    uint64_t t = c + 2ull * (uint32_t)p01;
    const uint32_t s1 = (uint32_t)t;
    c = (t >> 32) + 2ull * (p01 >> 32);
    t = c + 2ull * (uint32_t)p02 + (uint32_t)p11;
    const uint32_t s2 = (uint32_t)t;
    c = (t >> 32) + 2ull * (p02 >> 32) + (p11 >> 32);
    t = c + 2ull * (uint32_t)p03 + 2ull * (uint32_t)p12;
    const uint32_t s3 = (uint32_t)t;
    c = (t >> 32) + 2ull * (p03 >> 32) + 2ull * (p12 >> 32);
    t = c + 2ull * (uint32_t)p13 + (uint32_t)p22;
    const uint32_t s4 = (uint32_t)t;
    c = (t >> 32) + 2ull * (p13 >> 32) + (p22 >> 32);
    t = c + 2ull * (uint32_t)p23;
    const uint32_t s5 = (uint32_t)t;
    c = (t >> 32) + 2ull * (p23 >> 32);
    t = c + (uint32_t)p33;
    const uint32_t s6 = (uint32_t)t;
    /* (t >> 32) + (p33 >> 32) is provably zero under the precondition */

    /* quotient corner: A = (s2..s6) = S>>64, u has five words */
    Acc2 c5{}, c6{}, c7{}, c8{};
    accMul(c5, s3, u.w[4]); accMul(c5, s4, u.w[3]);
    accMul(c5, s5, u.w[2]); accMul(c5, s6, u.w[1]);
    accMul(c6, s4, u.w[4]); accMul(c6, s5, u.w[3]); accMul(c6, s6, u.w[2]);
    accMul(c7, s5, u.w[4]); accMul(c7, s6, u.w[3]);
    accMul(c8, s6, u.w[4]);
    uint64_t cy = accCarry(c5);
    uint64_t v = c6.lo + cy;  const uint32_t q0 = (uint32_t)v;  cy = c6.hi + (v >> 32);
    v = c7.lo + cy;           const uint32_t q1 = (uint32_t)v;  cy = c7.hi + (v >> 32);
    v = c8.lo + cy;           const uint32_t q2 = (uint32_t)v;
    const uint32_t q3 = (uint32_t)(c8.hi + (v >> 32));

    /* low four words of q*n, subtracted from the low four words of S */
    Acc2 m0{}, m1{}, m2{}, m3{};
    accMul(m0, q0, n.w[0]);
    accMul(m1, q1, n.w[0]); accMul(m1, q0, n.w[1]);
    accMul(m2, q2, n.w[0]); accMul(m2, q1, n.w[1]); accMul(m2, q0, n.w[2]);
    accMul(m3, q3, n.w[0]); accMul(m3, q2, n.w[1]);
    accMul(m3, q1, n.w[2]); accMul(m3, q0, n.w[3]);
    cy = accCarry(m0);
    const uint32_t w0 = (uint32_t)m0.lo;
    v = m1.lo + cy; const uint32_t w1 = (uint32_t)v; cy = m1.hi + (v >> 32);
    v = m2.lo + cy; const uint32_t w2 = (uint32_t)v; cy = m2.hi + (v >> 32);
    v = m3.lo + cy; const uint32_t w3 = (uint32_t)v;
    (void)(m3.hi + (v >> 32));

    /* r' = S - q*n is >= 0 and < 8n, so its low four words ARE the value */
    U128 out{{s0, s1, s2, s3}};
    sub128(out, U128{{w0, w1, w2, w3}});
    return out;
}

/* n = lo64 + hi64*2^64; precondition 2^96 <= n < 2^107 (top limb < 2^11) */
P0L_HD bool test4(uint64_t lo64, uint64_t hi64) {
    const U128 n{{(uint32_t)lo64, (uint32_t)(lo64 >> 32),
                  (uint32_t)hi64, (uint32_t)(hi64 >> 32)}};
    if (!(n.w[0] & 1u)) return false;
    U192 r256{};
    const U160 u = reciprocal256(n, &r256);
    (void)r256;

    /* e = (n-1)/2 is a 106-bit number: (hi64>>1) * 2^64 + e_lo */
    const uint64_t e_lo = (hi64 & 1u) ? (((lo64 - 1) >> 1) | (1ull << 63))
                                      : ((lo64 - 1) >> 1);
    const uint64_t e_hi = hi64 >> 1;

    U128 a{{1, 0, 0, 0}};
    {   /* high part of the exponent, MSB first */
        int hb = -1;
        for (uint64_t t = e_hi; t; t >>= 1) ++hb;
        for (int bit = hb; bit >= 0; --bit) {
            a = squareLazy4(a, n, u);
            if ((e_hi >> bit) & 1u) a = twice128(a);
        }
    }
    for (int bit = 63; bit >= 0; --bit) {
        a = squareLazy4(a, n, u);
        if ((e_lo >> bit) & 1u) a = twice128(a);
    }

    /* canonicalise: q = a.w3/(n.w3+1) is a provable under-estimate of
       floor(a/n), then at most a couple of subtractions remain */
    {
        const uint32_t qj = a.w[3] / (n.w[3] + 1u);
        Acc2 j0{}, j1{}, j2{}, j3{};
        accMul(j0, qj, n.w[0]);
        accMul(j1, qj, n.w[1]);
        accMul(j2, qj, n.w[2]);
        accMul(j3, qj, n.w[3]);
        uint64_t cc = accCarry(j0);
        const uint32_t k0 = (uint32_t)j0.lo;
        uint64_t vv = j1.lo + cc; const uint32_t k1 = (uint32_t)vv; cc = j1.hi + (vv >> 32);
        vv = j2.lo + cc;          const uint32_t k2 = (uint32_t)vv; cc = j2.hi + (vv >> 32);
        vv = j3.lo + cc;          const uint32_t k3 = (uint32_t)vv;
        sub128(a, U128{{k0, k1, k2, k3}});
    }
    for (int g = 0; g < 8 && ge128(a, n); ++g) sub128(a, n);

    U128 want{{1, 0, 0, 0}};
    if ((((lo64 >> 1) ^ (lo64 >> 2)) & 1u)) {        /* Jacobi bit of 2 */
        want = n;
        sub128(want, U128{{1, 0, 0, 0}});
    }
    return a.w[0] == want.w[0] && a.w[1] == want.w[1] &&
           a.w[2] == want.w[2] && a.w[3] == want.w[3];
}

}   /* namespace p0lazy */
#undef P0L_HD
