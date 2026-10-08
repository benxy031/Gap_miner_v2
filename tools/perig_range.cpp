/* Isolate ciosFermatTest128_hi (the walk engine's primality test) from the
   engine: compare its verdict against GMP across bit lengths 90..128. */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <stdint.h>
#include "perig.cuh"

int main(void) {
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 20261008u);
    mpz_t x; mpz_init(x);
    printf("%5s %8s %8s %10s %10s\n", "bits", "primes", "rejP", "composite", "accC");
    for (int bits = 88; bits <= 128; bits += 4) {
        size_t nprime = 0, rejP = 0, ncomp = 0, accC = 0;
        for (int i = 0; i < 2000; i++) {
            /* a real prime of this bit length */
            mpz_urandomb(x, rs, bits - 1); mpz_setbit(x, bits - 2); mpz_nextprime(x, x);
            if (mpz_sizeinbase(x, 2) > (size_t)bits) { i--; continue; }
            uint64_t lo = mpz_get_ui(x) | ((mpz_tstbit(x, 32) ? 1ULL : 0ULL) << 32);
            /* rebuild the 128-bit value from limbs without __int128 helpers */
            uint64_t w[2];
            mpz_export(w, NULL, -1, 8, 0, 0, x);
            uint64_t l = w[0], h = (mpz_sizeinbase(x, 2) > 64) ? w[1] : 0;
            int gmp = mpz_probab_prime_p(x, 30) >= 1;
            int k = ciosFermatTest128_hi(l, h) ? 1 : 0;
            if (gmp) { nprime++; if (!k) rejP++; }
            /* a random odd composite of this bit length */
            mpz_urandomb(x, rs, bits); mpz_setbit(x, bits - 1); mpz_setbit(x, 0);
            if (mpz_probab_prime_p(x, 30) >= 1) { i--; continue; }
            mpz_export(w, NULL, -1, 8, 0, 0, x);
            l = w[0]; h = (mpz_sizeinbase(x, 2) > 64) ? w[1] : 0;
            k = ciosFermatTest128_hi(l, h) ? 1 : 0;
            ncomp++; if (k) accC++;
        }
        printf("%5d %8zu %8zu %10zu %10zu\n", bits, nprime, rejP, ncomp, accC);
    }
    return 0;
}
