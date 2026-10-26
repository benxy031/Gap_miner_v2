/* dbg_lazy4.cpp - GMP ground truth for the FOUR-limb lazy-Barrett test
 * (tools/p0_test_lazy.cuh, namespace p0lazy, test4 / squareLazy4 /
 * reciprocal256).
 *
 * Three gates, each printed with its sample size so a vacuous pass is visible:
 *   1. reciprocal256:  u == floor(2^256/n)  AND  rem == 2^256 mod n
 *   2. squareLazy4:    r == a^2 mod n       AND  0 <= r < 16n  (lazy bound)
 *   3. test4:          verdict == GMP Euler reference, plus a prime count
 *
 * Build: g++ -O2 -I tools tools/dbg_lazy4.cpp -o bin/dbg_lazy4 -lgmp
 * Run:   bin/dbg_lazy4 [candidates=400]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>

#include "p0_test_lazy.cuh"

static int g_fail = 0;

static void fail(const char *what, mpz_t n, unsigned long long extra) {
    gmp_printf("  FAIL %s  n=%Zd  (extra=%llu)\n", what, n, extra);
    if (++g_fail > 12) { printf("  ... too many failures, stopping\n"); exit(1); }
}

/* GMP reference for the predicate test4 / ciosFermatTest128 implement:
       2^((n-1)/2) mod n == (jacobi(2,n) == +1 ? 1 : n-1) */
static int ref_euler(const mpz_t n) {
    mpz_t e, x, two, one, m1;
    mpz_inits(e, x, two, one, m1, NULL);
    mpz_set_ui(two, 2);
    mpz_set_ui(one, 1);
    mpz_sub_ui(e, n, 1);
    mpz_fdiv_q_2exp(e, e, 1);
    mpz_powm(x, two, e, n);
    mpz_sub_ui(m1, n, 1);
    const unsigned r8 = mpz_fdiv_ui(n, 8);
    const int jac = (r8 == 1u || r8 == 7u) ? 1 : -1;
    const int ok = (jac == 1) ? (mpz_cmp(x, one) == 0) : (mpz_cmp(x, m1) == 0);
    mpz_clears(e, x, two, one, m1, NULL);
    return ok;
}

/* lift a GMP value into four 32-bit limbs (low word first) */
static p0lazy::U128 to_u128(const mpz_t a) {
    p0lazy::U128 r{{0, 0, 0, 0}};
    mpz_t t;
    mpz_init_set(t, a);
    for (int k = 0; k < 4; k++) {
        r.w[k] = (uint32_t)mpz_get_ui(t);
        mpz_fdiv_q_2exp(t, t, 32);
    }
    mpz_clear(t);
    return r;
}
static void from_u128(mpz_t out, const uint32_t *w, int nw) {
    mpz_set_ui(out, 0);
    for (int k = nw - 1; k >= 0; --k) {
        mpz_mul_2exp(out, out, 32);
        mpz_add_ui(out, out, w[k]);
    }
}

int main(int argc, char **argv) {
    const unsigned long long N = (argc > 1) ? strtoull(argv[1], NULL, 10) : 400ull;

    /* ---------------- gate 1 + 2: reciprocal and squaring ------------- */
    mpz_t n, w, u_ref, r_ref, a, sq_ref, got, diff, bound, nh, t;
    mpz_inits(n, w, u_ref, r_ref, a, sq_ref, got, diff, bound, nh, t, NULL);
    mpz_set_ui(w, 1);
    mpz_mul_2exp(w, w, 256);             /* w = 2^256 */

    /* start just above 1e32 (the campaign band) */
    mpz_set_str(n, "100000000000000000000000000000000", 10);
    mpz_add_ui(n, n, 1);                 /* make it odd */
    if (mpz_even_p(n)) mpz_add_ui(n, n, 1);

    unsigned long long bad_recip = 0, bad_sq = 0, bad_bound = 0, sq_samples = 0;
    unsigned long long nmod = 0;
    for (unsigned long long i = 0; i < N; i++) {
        mpz_add_ui(n, n, 2);
        mpz_fdiv_q_2exp(nh, n, 64);
        const uint64_t hi = mpz_get_ui(nh);
        if (hi >> 43) continue;                            /* outside 2^107 */
        const uint64_t lo = mpz_get_ui(n);
        nmod++;

        const p0lazy::U128 nn{{(uint32_t)lo, (uint32_t)(lo >> 32),
                               (uint32_t)hi, (uint32_t)(hi >> 32)}};
        /* gate 1: reciprocal - exact quotient AND remainder */
        p0lazy::U192 rem{};
        const p0lazy::U160 u = p0lazy::reciprocal256(nn, &rem);
        mpz_fdiv_q(u_ref, w, n);
        mpz_fdiv_r(r_ref, w, n);
        from_u128(got, u.w, 5);
        if (mpz_cmp(got, u_ref) != 0) {
            bad_recip++;
            if (bad_recip < 4) fail("reciprocal256 quotient", n, i);
        }
        from_u128(got, rem.w, 6);
        if (mpz_cmp(got, r_ref) != 0) {
            bad_recip++;
            if (bad_recip < 4) fail("reciprocal256 remainder", n, i);
        }

        /* gate 2: one lazy squaring over a, 2a, .. 16a - the top of that
           range is where the laziness bound (residue < 16n) is really hit */
        mpz_set_ui(a, mpz_fdiv_ui(n, 1000003u) + 1u);      /* varied, < n */
        for (int mult = 1; mult <= 16; mult++) {
            mpz_mul_ui(t, a, (unsigned long)mult);
            const p0lazy::U128 av = to_u128(t);
            if (av.w[3] >= 0x10000u) continue;             /* not representable */
            const p0lazy::U128 rr = p0lazy::squareLazy4(av, nn, u);
            mpz_mul(sq_ref, t, t);
            mpz_fdiv_r(sq_ref, sq_ref, n);
            from_u128(got, rr.w, 4);
            mpz_fdiv_r(diff, got, n);
            mpz_sub(diff, diff, sq_ref);
            if (mpz_sgn(diff) != 0) {
                bad_sq++;
                if (bad_sq < 4) fail("squareLazy4 residue", n, (unsigned long long)mult);
            }
            mpz_mul_ui(bound, n, 16);
            if (mpz_cmp(got, bound) >= 0) {
                bad_bound++;
                if (bad_bound < 4) fail("squareLazy4 laziness bound (16n)", n,
                                       (unsigned long long)mult);
            }
            sq_samples++;
        }
    }
    printf("gate 1 reciprocal256 : %llu moduli, %llu quotient/remainder mismatches%s\n",
           nmod, bad_recip, bad_recip ? "   <-- FAIL" : "   PASS");
    printf("gate 2 squareLazy4   : %llu samples, %llu residue mismatches, %llu bound violations%s\n",
           sq_samples, bad_sq, bad_bound,
           (bad_sq || bad_bound) ? "   <-- FAIL" : "   PASS");

    /* ---------------- gate 3: the full test vs GMP --------------------- */
    mpz_set_str(n, "100000000000000000000000000000000", 10);
    if (mpz_even_p(n)) mpz_add_ui(n, n, 1);
    unsigned long long primes = 0, bad_verdict = 0, checked = 0;
    for (unsigned long long i = 0; i < N; i++) {
        mpz_add_ui(n, n, 2);
        mpz_fdiv_q_2exp(nh, n, 64);
        const uint64_t hi = mpz_get_ui(nh);
        if (hi >> 43) continue;
        const uint64_t lo = mpz_get_ui(n);
        const int want = ref_euler(n);
        const int gotv = p0lazy::test4(lo, hi) ? 1 : 0;
        primes += (unsigned long long)want;
        checked++;
        if (want != gotv) {
            bad_verdict++;
            if (bad_verdict < 4)
                fail(want ? "test4 rejected a prime" : "test4 accepted a composite", n, i);
        }
    }
    printf("gate 3 test4         : %llu candidates, %llu Euler-passing, %llu verdict mismatches%s\n",
           checked, primes, bad_verdict, bad_verdict ? "   <-- FAIL" : "   PASS");
    if (primes == 0) printf("gate 3 is VACUOUS: the sample contains no Euler-passing value\n");

    const int bad = (bad_recip || bad_sq || bad_bound || bad_verdict);
    printf("\n%s\n", bad ? "SOME CHECKS FAILED" : "ALL CHECKS PASS");
    mpz_clears(n, w, u_ref, r_ref, a, sq_ref, got, diff, bound, nh, t, NULL);
    return bad ? 1 : 0;
}
