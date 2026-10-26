/* dbg_lazy_band.cpp - GMP gate for the RUNTIME-top-limb entry point that the
 * walk-engine MODE 3 dispatch calls (p0lazy::test_hi), swept across the whole
 * band the dispatch enables: 1 <= hi < 2^11  =>  2^64 <= n < 2^75.
 *
 * The band-specialised gates (tools/dbg_lazy.cpp) only ever test hi = 10.  MODE 3
 * feeds a runtime hi, so this harness walks hi from 1 (n just above 2^64, the
 * edge where the reciprocal is largest) to 2047 (n just below 2^75, the edge
 * where the representative is largest).
 *
 * Build: g++ -O2 -I tools tools/dbg_lazy_band.cpp -o bin/dbg_lazy_band -lgmp
 * Run:   bin/dbg_lazy_band [per_hi=2000]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>

#include "p0_test_lazy.cuh"

/* 2^((n-1)/2) mod n == (jacobi(2,n) == +1 ? 1 : n-1) - the predicate test_hi
   implements, and the same one perig implements */
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

int main(int argc, char **argv) {
    const unsigned long long per_hi = (argc > 1) ? strtoull(argv[1], NULL, 10) : 2000ull;
    /* hi values chosen to hit both edges and the interior of the band */
    const uint32_t his[] = { 1u, 2u, 3u, 5u, 10u, 17u, 64u, 255u, 1000u, 2047u };
    const int nh = (int)(sizeof(his) / sizeof(his[0]));

    mpz_t n;
    mpz_init(n);
    unsigned long long total = 0, primes = 0, bad = 0, shown = 0;

    for (int h = 0; h < nh; h++) {
        const uint32_t hi = his[h];
        unsigned long long hp = 0, hbad = 0;
        /* low sweeps an odd arithmetic progression; restart per hi so the
           sample is identical in shape for every top limb */
        uint64_t low = 0x9e3779b97f4a7c15ull | 1ull;
        for (unsigned long long i = 0; i < per_hi; i++, low += 2ull) {
            const int got = p0lazy::test_hi(low, hi) ? 1 : 0;
            mpz_set_ui(n, hi);
            mpz_mul_2exp(n, n, 64);
            mpz_add_ui(n, n, (unsigned long)low);
            const int want = ref_euler(n);
            hp += (unsigned long long)want;
            if (want != got) {
                hbad++;
                if (shown++ < 5)
                    gmp_printf("  MISMATCH hi=%u low=%llu want=%d got=%d n=%Zd\n",
                               hi, (unsigned long long)low, want, got, n);
            }
        }
        total += per_hi;
        primes += hp;
        bad += hbad;
        printf("  hi=%-5u : %llu candidates, %llu Euler-passing, %llu mismatches%s\n",
               hi, per_hi, hp, hbad, hbad ? "   <-- FAIL" : "");
    }
    printf("\nband gate test_hi: %llu candidates, %llu Euler-passing, %llu mismatches%s\n",
           total, primes, bad, bad ? "   <-- FAIL" : "   PASS");
    if (!primes) printf("VACUOUS: no Euler-passing value in the sample\n");
    mpz_clear(n);
    return bad ? 1 : 0;
}
