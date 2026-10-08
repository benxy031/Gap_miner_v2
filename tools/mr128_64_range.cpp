/* Host validation of the 2 x 64-bit exact base-2 test against GMP across the
   whole 96..128-bit container, plus the walk engine's candidate shape.
   Build: g++ -O2 -I tools tools/mr128_64_range.cpp -lgmp -o mr128_64_range     */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <stdint.h>
#include "mr128_64_kernel.cuh"

int main(int argc, char **argv) {
    int per = (argc > 1) ? atoi(argv[1]) : 3000;
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
            int k = mr128_64_base2_u128(w[0], w[1]);
            np++; if (gmp && !k) rej++;
            mpz_urandomb(x, rs, bits); mpz_setbit(x, bits - 1); mpz_setbit(x, 0);
            if (mpz_probab_prime_p(x, 30) >= 1) { i--; continue; }
            mpz_export(w, NULL, -1, 8, 0, 0, x);
            k = mr128_64_base2_u128(w[0], w[1]);
            nc++; if (k) acc++;
        }
        printf("%5d %7zu %7zu %10zu %10zu\n", bits, np, rej, nc, acc);
    }
    /* walk engine shape: shared high word, low word = base + k*30 */
    {
        mpz_t v; mpz_init(v);
        size_t rej = 0, acc = 0, np = 0, nc = 0;
        for (int k = 0; k < 20000; k++) {
            mpz_set_str(v, "1329227995784915872903807060280", 10);
            mpz_add_ui(v, v, (unsigned long)(k * 30));
            uint64_t w[4] = {0,0,0,0}; mpz_export(w, NULL, -1, 8, 0, 0, v);
            int gmp = mpz_probab_prime_p(v, 30) >= 1;
            int kk = mr128_64_base2_u128(w[0], w[1]);
            if (gmp) { np++; if (!kk) rej++; } else { nc++; if (kk) acc++; }
        }
        printf("shape (2^120 + k*30, k<20000): primes=%zu rejP=%zu composites=%zu accC=%zu\n",
               np, rej, nc, acc);
        mpz_clear(v);
    }
    return 0;
}
