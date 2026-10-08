/* Host validation of the 4 x 32-bit CIOS base-2 test against GMP, across the
   whole 96..128-bit container.  Compile:
     g++ -O2 -D__host__= -D__device__= -D__forceinline__=inline -I tools \
         tools/probes/mr128_range.cpp -lgmp -o mr128_range                              */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <stdint.h>
#include "mr128_kernel.cuh"

int main(int argc, char **argv) {
    int per = (argc > 1) ? atoi(argv[1]) : 3000;
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 20261008u);
    mpz_t x; mpz_init(x);
    printf("%5s %7s %7s %10s %10s\n", "bits", "primes", "rejP", "composite", "accC");
    for (int bits = 96; bits <= 128; bits += 4) {
        size_t nprime = 0, rejP = 0, ncomp = 0, accC = 0;
        for (int i = 0; i < per; i++) {
            mpz_urandomb(x, rs, bits - 1); mpz_setbit(x, bits - 2); mpz_nextprime(x, x);
            if (mpz_sizeinbase(x, 2) > (size_t)bits) { i--; continue; }
            uint64_t w[4] = {0,0,0,0}; mpz_export(w, NULL, -1, 8, 0, 0, x);
            int gmp = mpz_probab_prime_p(x, 30) >= 1;
            int k = mr128_base2_u128(w[0], w[1]);
            nprime++; if (gmp && !k) rejP++;
            mpz_urandomb(x, rs, bits); mpz_setbit(x, bits - 1); mpz_setbit(x, 0);
            if (mpz_probab_prime_p(x, 30) >= 1) { i--; continue; }
            mpz_export(w, NULL, -1, 8, 0, 0, x);
            k = mr128_base2_u128(w[0], w[1]);
            ncomp++; if (k) accC++;
        }
        printf("%5d %7zu %7zu %10zu %10zu\n", bits, nprime, rejP, ncomp, accC);
    }
    /* the engine's exact shape: small low word, value = 2^120 + k*30 */
    {
        mpz_t base; mpz_init(base); mpz_set_ui(base, 1);
        size_t rej = 0, acc = 0, np = 0, nc = 0;
        for (int k = 0; k < 4000; k++) {
            mpz_set_str(base, "1329227995784915872903807060280", 10);   /* ~2^120 */
            mpz_add_ui(base, base, (unsigned long)(k * 30));
            uint64_t w[4] = {0,0,0,0}; mpz_export(w, NULL, -1, 8, 0, 0, base);
            int gmp = mpz_probab_prime_p(base, 30) >= 1;
            int kk = mr128_base2_u128(w[0], w[1]);
            if (gmp) { np++; if (!kk) rej++; } else { nc++; if (kk) acc++; }
        }
        printf("shape (2^120 + k*30, k<4000): primes=%zu rejP=%zu composites=%zu accC=%zu\n",
               np, rej, nc, acc);
    }
    return 0;
}
