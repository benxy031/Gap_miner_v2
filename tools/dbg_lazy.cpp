/* dbg_lazy.c - HOST debug harness for tools/p0_test_lazy.cuh (GMP reference).
 *
 * Three checks, in the order that localises a defect fastest:
 *   (a) reciprocal:  u*n + rem == 2^160  and 0 <= rem < n
 *   (b) squaring:    squareLazy(x) == x^2 mod n   (as an integer in [0,12n))
 *   (c) full test:   p0lazy::test<10>(low)  vs  mpz_powm(2,(n-1)/2,n) in {1,n-1}
 *
 * Build: g++ -O2 -std=c++17 -Itools tools/dbg_lazy.c -o bin/dbg_lazy -lgmp
 * Run:   bin/dbg_lazy [count=5]
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gmp.h>

#include "p0_test_lazy.cuh"

using p0lazy::U96;

static void u96_to_mpz(const U96 &a, mpz_t out) {
    mpz_set_ui(out, a.w[2]);
    mpz_mul_2exp(out, out, 32); mpz_add_ui(out, out, a.w[1]);
    mpz_mul_2exp(out, out, 32); mpz_add_ui(out, out, a.w[0]);
}
static void u96_from_mpz(U96 &a, const mpz_t v) {
    mpz_t t; mpz_init_set(t, v);
    a.w[0] = (uint32_t)mpz_get_ui(t); mpz_fdiv_q_2exp(t, t, 32);
    a.w[1] = (uint32_t)mpz_get_ui(t); mpz_fdiv_q_2exp(t, t, 32);
    a.w[2] = (uint32_t)mpz_get_ui(t);
    mpz_clear(t);
}
static void print96(const char *tag, const U96 &a) {
    mpz_t t; mpz_init(t); u96_to_mpz(a, t);
    gmp_printf("  %s = %Zd\n", tag, t); mpz_clear(t);
}

int main(int argc, char **argv) {
    int count = (argc > 1) ? atoi(argv[1]) : 5;

    /* candidate base in the 2e20 band, same as the GPU bench */
    mpz_t two_e20; mpz_init(two_e20);
    mpz_ui_pow_ui(two_e20, 10, 20); mpz_mul_ui(two_e20, two_e20, 2);
    unsigned long long base_lo = mpz_get_ui(two_e20);   /* low 64 bits */
    mpz_clear(two_e20);

    mpz_t N, U, R, S, want, got, t1, t2;
    mpz_inits(N, U, R, S, want, got, t1, t2, NULL);

    int bad_a = 0, bad_b = 0, bad_c = 0, tested = 0, primes_seen = 0, shown = 0;
    for (int k = 0; k < count; k++) {
        uint64_t low = (uint64_t)base_lo + (uint64_t)(2 * k) + 1ull;
        U96 n{{(uint32_t)low, (uint32_t)(low >> 32), 10u}};
        u96_to_mpz(n, N);
        tested++;

        /* (a) reciprocal ------------------------------------------------ */
        U96 rem{};
        p0lazy::U128 u = p0lazy::reciprocal192(n, &rem);
        mpz_set_ui(U, u.w[3]); mpz_mul_2exp(U, U, 32); mpz_add_ui(U, U, u.w[2]);
        mpz_mul_2exp(U, U, 32); mpz_add_ui(U, U, u.w[1]);
        mpz_mul_2exp(U, U, 32); mpz_add_ui(U, U, u.w[0]);
        u96_to_mpz(rem, R);
        mpz_mul(t1, U, N); mpz_add(t1, t1, R);       /* u*n + rem */
        mpz_set_ui(t2, 1); mpz_mul_2exp(t2, t2, 192); /* 2^192 */
        int a_ok = (mpz_cmp(t1, t2) == 0) && (mpz_cmp(R, N) < 0) && (mpz_sgn(R) >= 0);
        if (!a_ok && bad_a++ < 2) {
            printf("[a] FAIL low=%llu\n", (unsigned long long)low);
            gmp_printf("     u*n+rem = %Zd\n     2^192   = %Zd\n     rem<n? %d\n",
                       t1, t2, mpz_cmp(R, N));
        }

        /* (b) squaring -------------------------------------------------- */
        /* Probe BOTH regimes: small values AND values near n (the exponent
           loop squares numbers up to ~n, where the top word of the square is
           nonzero - small probes alone leave that untested, which is exactly
           how a wrong reduction passed the first version of this gate). */
        mpz_t tx; mpz_init(tx);
        unsigned long long probes_small[] = {2ull, 3ull, 5ull, 0x123456789ull, 0xdeadbeef1ull};
        int nprobe = 0;
        for (unsigned ip = 0; ip <= 9; ip++) {
            if (ip < 5) {
                mpz_set_ui(tx, probes_small[ip]);
            } else if (ip == 5) {
                mpz_sub_ui(tx, N, 1);                    /* n-1 */
            } else if (ip == 6) {
                mpz_fdiv_q_2exp(tx, N, 1);               /* n/2 */
            } else if (ip == 7) {
                mpz_fdiv_q_ui(tx, N, 3);                 /* n/3 */
            } else if (ip == 8) {
                mpz_mul_ui(tx, N, 7); mpz_fdiv_q_2exp(tx, tx, 3);  /* 7n/8 */
            } else {
                mpz_mul_ui(tx, N, 3); mpz_add_ui(tx, tx, 5);       /* 3n+5 */
            }
            if (mpz_cmp(tx, N) >= 0) {                      /* keep a < n */
                mpz_mod(tx, tx, N);
                if (mpz_sgn(tx) == 0) continue;
            }
            U96 x{}; u96_from_mpz(x, tx);
            U96 q = p0lazy::squareLazy(x, n, u);
            u96_to_mpz(q, got);
            mpz_mul(want, tx, tx); mpz_mod(want, want, N);
            mpz_mod(t2, got, N);
            int ok = (mpz_cmp(t2, want) == 0);
            mpz_t L; mpz_init(L); mpz_mul_ui(L, N, 8);      /* got < 8n */
            ok = ok && (mpz_cmp(got, L) < 0);
            mpz_clear(L);
            nprobe++;
            if (!ok && bad_b++ < 3) {
                gmp_printf("[b] FAIL low=%llu x=%Zd\n     got=%Zd\n     want(x^2 mod n)=%Zd\n",
                           (unsigned long long)low, tx, got, want);
            }
        }
        mpz_clear(tx);

        /* (c) full test ------------------------------------------------- */
        mpz_sub_ui(t1, N, 1); mpz_fdiv_q_2exp(t1, t1, 1);   /* e = (n-1)/2 */
        mpz_set_ui(t2, 2); mpz_powm(want, t2, t1, N);
        mpz_set_ui(t1, 1); mpz_sub(t2, N, t1);              /* n-1 */
        int want_prime = (mpz_cmp(want, t1) == 0) || (mpz_cmp(want, t2) == 0);
        int got_prime = p0lazy::test<10u>(low) ? 1 : 0;
        if (want_prime) primes_seen++;
        if (want_prime != got_prime && bad_c++ < 3 && shown++ < 2) {
            gmp_printf("[c] FAIL low=%llu  want=%d got=%d  2^((n-1)/2) mod n = %Zd\n",
                       (unsigned long long)low, want_prime, got_prime, want);
            /* follow the header's own chain to the final comparison */
            const U96 nn{{(uint32_t)low, (uint32_t)(low >> 32), 10u}};
            float hint2 = 0.0f; U96 rem2{};
            p0lazy::U128 u2 = p0lazy::reciprocal192(nn, &rem2);
            (void)hint2;
            uint64_t ee = (low - 1) >> 1;
            U96 aa{{1, 0, 0}};
            int tb = 63 - p0lazy::clz64(ee);
            for (int bit = tb; bit >= 0; --bit) {
                aa = p0lazy::squareLazy(aa, nn, u2);
                if ((ee >> bit) & 1u) aa = p0lazy::twice96(aa);
            }
            U96 fin = p0lazy::normalizeFinal(aa, nn);
            for (int g2 = 0; g2 < 24 && p0lazy::ge96(fin, nn); ++g2) p0lazy::sub96(fin, nn);
            mpz_t pre, finz, wantz;
            mpz_inits(pre, finz, wantz, NULL);
            u96_to_mpz(aa, pre); u96_to_mpz(fin, finz);
            /* what the header compares against */
            U96 w{{1, 0, 0}};
            if (((low >> 1) ^ (low >> 2)) & 1u) { w = nn; p0lazy::sub96(w, U96{{1, 0, 0}}); }
            u96_to_mpz(w, wantz);
            gmp_printf("     raw=%Zd\n     reduced=%Zd\n     want=%Zd  (true residue=%Zd)\n",
                       pre, finz, wantz, want);
            int eq = (mpz_cmp(finz, wantz) == 0);
            printf("     compare fin==want? %d   (fin mod n == 1 or n-1? ", eq);
            mpz_mod(t1, finz, N);
            mpz_set_ui(t2, 1);
            int r1 = (mpz_cmp(t1, t2) == 0);
            mpz_sub_ui(t2, N, 1);
            int r2 = (mpz_cmp(t1, t2) == 0);
            printf("%d)\n", r1 || r2);
            mpz_clears(pre, finz, wantz, NULL);
        }
    }

    printf("\ntested %d candidates (%d prime): (a) reciprocal fails=%d  (b) square fails=%d  "
           "(c) test fails=%d\n", tested, primes_seen, bad_a, bad_b, bad_c);
    printf("%s\n", (bad_a || bad_b || bad_c) ? "NOT READY - fix the first failing check"
                                             : (primes_seen ? "ALL CHECKS PASS"
                                                            : "VACUOUS - no prime in the sample"));
    mpz_clears(N, U, R, S, want, got, t1, t2, NULL);
    return (bad_a || bad_b || bad_c) ? 1 : 0;
}
