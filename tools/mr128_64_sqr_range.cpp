/* Host validation of the lean-square 2 x 64-bit test against GMP, plus a unit
   test of the square itself (mrsq_sqr vs the validated mont_mul on a==b).
   Build: g++ -O2 -I tools tools/mr128_64_sqr_range.cpp -lgmp -o mr128_64_sqr_range */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <stdint.h>
#include "mr128_64_sqr_kernel.cuh"
#include "mr128_64_kernel.cuh"

static int unit_square(int rounds) {
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 7u);
    mpz_t n, a; mpz_init(n); mpz_init(a);
    int bad = 0;
    for (int i = 0; i < rounds; i++) {
        int bits = 97 + (i % 32);
        mpz_urandomb(n, rs, bits - 1); mpz_setbit(n, bits - 1); mpz_setbit(n, 0);
        mpz_urandomb(a, rs, bits - 1); mpz_mod(a, a, n);
        uint64_t nw[4] = {0}, aw[4] = {0};
        mpz_export(nw, NULL, -1, 8, 0, 0, n);
        mpz_export(aw, NULL, -1, 8, 0, 0, a);
        uint64_t nn[2] = {nw[0], nw[1]}, aa[2] = {aw[0], aw[1]};
        uint64_t inv = montgomeryInverse64(nn[0]);
        uint64_t s1[2], s2[2];
        mrsq_sqr(s1, aa, nn, inv);           /* lean square  */
        mr128_64_mont_mul(s2, aa, aa, nn, inv);  /* validated mul (a==b) */
        if (s1[0] != s2[0] || s1[1] != s2[1]) {
            if (bad < 3)
                printf("  square mismatch bits=%d n=%s -> %016lx%016lx vs %016lx%016lx\n",
                       bits, mpz_get_str(NULL, 10, n), s1[1], s1[0], s2[1], s2[0]);
            bad++;
        }
    }
    printf("unit square (mrsq_sqr == mont_mul(a,a)): %d rounds, %d mismatches\n", rounds, bad);
    mpz_clear(n); mpz_clear(a); gmp_randclear(rs);
    return bad;
}

int main(int argc, char **argv) {
    int per = (argc > 1) ? atoi(argv[1]) : 3000;
    int bad = unit_square(400);
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 20261008u);
    mpz_t x; mpz_init(x);
    printf("%5s %7s %7s %10s %10s\n", "bits", "primes", "rejP", "composite", "accC");
    for (int bits = 98; bits <= 128; bits += 2) {
        size_t np = 0, rej = 0, nc = 0, acc = 0;
        for (int i = 0; i < per; i++) {
            mpz_urandomb(x, rs, bits - 1); mpz_setbit(x, bits - 2); mpz_nextprime(x, x);
            if (mpz_sizeinbase(x, 2) > (size_t)bits) { i--; continue; }
            uint64_t w[4] = {0,0,0,0}; mpz_export(w, NULL, -1, 8, 0, 0, x);
            int gmp = mpz_probab_prime_p(x, 30) >= 1;
            int k = mrsq_base2_u128(w[0], w[1]);
            np++; if (gmp && !k) rej++;
            mpz_urandomb(x, rs, bits); mpz_setbit(x, bits - 1); mpz_setbit(x, 0);
            if (mpz_probab_prime_p(x, 30) >= 1) { i--; continue; }
            mpz_export(w, NULL, -1, 8, 0, 0, x);
            k = mrsq_base2_u128(w[0], w[1]);
            nc++; if (k) acc++;
        }
        printf("%5d %7zu %7zu %10zu %10zu\n", bits, np, rej, nc, acc);
    }
    /* the CPU-relevant corner: n just below 2^128 (the 2*n_hi wrap that
       breaks perig) */
    {
        size_t np = 0, rej = 0, nc = 0, acc = 0;
        mpz_t v; mpz_init(v);
        for (int i = 0; i < 4000; i++) {
            mpz_set_str(v, "340282366920938463463374607431768200000", 10);
            mpz_add_ui(v, v, (unsigned long)(i * 30 + 1));
            if (mpz_probab_prime_p(v, 30) < 1) { continue; }
            uint64_t w[4] = {0,0,0,0}; mpz_export(w, NULL, -1, 8, 0, 0, v);
            int k = mrsq_base2_u128(w[0], w[1]);
            np++; if (!k) rej++;
        }
        printf("near-2^128 primes (n_hi >= 2^63): %zu checked, %zu rejected\n", np, rej);
        mpz_clear(v);
    }
    return bad ? 1 : 0;
}
