/* Independent GMP reference: full prime sequence over [start, start+len),
   prints "lower gap" for every gap >= mingap.  Used to gate the 128-bit walk. */
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s START LENGTH MINGAP\n", argv[0]); return 2; }
    mpz_t a, b, end; unsigned long long gapmin = strtoull(argv[3], NULL, 10);
    mpz_init_set_str(a, argv[1], 10); mpz_init(b); mpz_init(end);
    mpz_init_set_str(end, argv[1], 10); mpz_add_ui(end, end, strtoull(argv[2], NULL, 10));
    unsigned long long n = 0;
    for (;;) {
        mpz_nextprime(b, a);
        if (mpz_cmp(b, end) > 0) break;
        mpz_t d; mpz_init(d); mpz_sub(d, b, a);
        if (mpz_cmp_ui(d, gapmin) >= 0) { gmp_printf("%Zd %Zd\n", a, d); n++; }
        mpz_clear(d);
        mpz_set(a, b);
    }
    fprintf(stderr, "gaps>=%llu: %llu\n", gapmin, n);
    return 0;
}
