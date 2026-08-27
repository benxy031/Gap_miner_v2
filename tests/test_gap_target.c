/*
 * Integration Test: targeted-merit gap search (covering + CRT + check)
 *
 * Verifies:
 *   1. crt_solve returns x = residues[i] (mod primes[i]).
 *   2. gap_target_build's survivors match a brute-force covering.
 *   3. gap_target_check matches the GMP ground truth (nextprime gap) for
 *      bases that satisfy the residues.
 */

#include "../new_src/gap_target.h"

#include <stdio.h>
#include <string.h>

static int gmp_is_prime(void *ctx, const mpz_t n) {
    (void)ctx;
    return mpz_probab_prime_p(n, 25) > 0;
}

static int test_crt_solve(void) {
    const uint64_t primes[] = {3, 5, 7};
    const uint64_t residues[] = {1, 2, 3};
    mpz_t primorial, x;
    mpz_init_set_ui(primorial, 3 * 5 * 7);
    mpz_init(x);

    crt_solve(x, primes, residues, 3, primorial);

    int ok = (mpz_fdiv_ui(x, 3) == 1 &&
              mpz_fdiv_ui(x, 5) == 2 &&
              mpz_fdiv_ui(x, 7) == 3);
    printf("  crt_solve x=%s (%s)\n",
           mpz_get_str(NULL, 10, x), ok ? "OK" : "FAIL");

    mpz_clear(primorial);
    mpz_clear(x);
    return ok ? 0 : 1;
}

static int test_build(void) {
    const uint64_t primes[] = {2, 3, 5, 7};
    const uint64_t residues[] = {1, 1, 1, 1};  /* base = 1 (mod 210) */
    struct gap_target gt;

    if (!gap_target_build(&gt, 30, primes, residues, 4)) {
        fprintf(stderr, "FAIL: gap_target_build\n");
        return 1;
    }

    /* Brute force: j in [1,29], covered iff (1+j) divisible by 2,3,5,7. */
    uint64_t expected[] = {10, 12, 16, 18, 22, 28};
    int ok = (gt.n_survivors == 6);
    for (size_t i = 0; ok && i < gt.n_survivors; i++) {
        ok = (gt.survivors[i] == expected[i]);
    }
    printf("  survivors: n=%zu %s\n", gt.n_survivors, ok ? "OK" : "FAIL");

    gap_target_free(&gt);
    return ok ? 0 : 1;
}

static int test_check(void) {
    const uint64_t primes[] = {2, 3, 5, 7};
    const uint64_t residues[] = {1, 1, 1, 1};  /* base = 1 (mod 210) */
    struct gap_target gt;
    if (!gap_target_build(&gt, 30, primes, residues, 4)) {
        fprintf(stderr, "FAIL: build for check\n");
        return 1;
    }

    int failures = 0;
    int saw_valid = 0;
    mpz_t base, next;
    mpz_init_set_ui(base, 211);  /* 211 = 1 mod 210, prime */
    mpz_init(next);

    for (int k = 0; k < 200; k++) {
        mpz_nextprime(next, base);
        mpz_t diff;
        mpz_init(diff);
        mpz_sub(diff, next, base);
        /* Valid only if base itself is prime AND the gap to next >= gap_size. */
        int expect_valid = (mpz_probab_prime_p(base, 25) > 0) &&
                           (mpz_cmp_ui(diff, 30) >= 0);
        mpz_clear(diff);

        int got = gap_target_check(&gt, base, gmp_is_prime, NULL);
        if (got != expect_valid) {
            fprintf(stderr, "FAIL: check at base %s (got %d want %d)\n",
                    mpz_get_str(NULL, 10, base), got, expect_valid);
            failures++;
        }
        if (got) saw_valid = 1;

        mpz_add_ui(base, base, 210);  /* stay in the 1 mod 210 class */
    }

    printf("  check matched GMP over 200 bases (saw_valid=%d)\n", saw_valid);

    mpz_clear(base);
    mpz_clear(next);
    gap_target_free(&gt);
    return (failures || !saw_valid) ? 1 : 0;
}

int main(void) {
    int failures = 0;
    failures += test_crt_solve();
    failures += test_build();
    failures += test_check();

    if (failures == 0) {
        printf("PASS: targeted-merit gap search\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d checks failed\n", failures);
    return 1;
}
