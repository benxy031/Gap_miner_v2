/*
 * Integration Test: CRT (primorial) sieve set and generator
 *
 * Verifies:
 *   1. crt_set_primes_for_shift returns the maximal prime count whose
 *      primorial is < 2^shift, for shifts 256/512/1024.
 *   2. crt_set_resieve's survivor count matches the Mertens expectation
 *      (size * prod(1-1/p)) within a sane tolerance.
 *   3. crt_set_search_offset finds a lower-survivor offset than offset 0.
 *   4. save/load round-trips the set (same primorial, offset, survivors).
 */

#include "../new_src/crt_set.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>

static int test_primes_for_shift(void) {
    static const uint32_t shifts[] = {256, 512, 1024};
    int failures = 0;

    for (size_t k = 0; k < sizeof(shifts) / sizeof(shifts[0]); k++) {
        uint32_t shift = shifts[k];
        uint32_t n = crt_set_primes_for_shift(shift);
        if (n == 0) {
            fprintf(stderr, "FAIL: no primes for shift %u\n", shift);
            failures++;
            continue;
        }

        struct crt_set cs;
        if (!crt_set_init(&cs, shift, 1000, n)) {
            fprintf(stderr, "FAIL: init shift %u\n", shift);
            failures++;
            continue;
        }

        mpz_t limit;
        mpz_init_set_ui(limit, 1);
        mpz_mul_2exp(limit, limit, shift);

        if (mpz_cmp(cs.primorial, limit) >= 0) {
            fprintf(stderr, "FAIL: primorial >= 2^shift for shift %u\n", shift);
            failures++;
        }

        /* n must be maximal: n+1 primes must push the primorial to >= 2^shift. */
        struct crt_set cs2;
        if (crt_set_init(&cs2, shift, 1000, n + 1)) {
            if (mpz_cmp(cs2.primorial, limit) < 0) {
                fprintf(stderr, "FAIL: n not maximal for shift %u\n", shift);
                failures++;
            }
            crt_set_free(&cs2);
        }

        printf("  shift %u -> %u primes, primorial %llu bits\n",
               shift, n, (unsigned long long)cs.bit_size);

        mpz_clear(limit);
        crt_set_free(&cs);
    }
    return failures;
}

static int test_resieve_expectation(void) {
    int failures = 0;
    struct crt_set cs;
    if (!crt_set_init(&cs, 512, 20000, crt_set_primes_for_shift(512))) {
        fprintf(stderr, "FAIL: init for resieve\n");
        return 1;
    }

    /* The survivor count of a single (especially offset-0) window fluctuates;
       the expectation size * prod(1-1/p) is an average over random offsets.
       Average over many random offsets and compare. */
    const uint32_t samples = 100;
    uint64_t total = 0;
    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, 42);
    mpz_t cand;
    mpz_init(cand);
    for (uint32_t i = 0; i < samples; i++) {
        mpz_urandomm(cand, state, cs.primorial);
        mpz_set(cs.offset, cand);
        total += crt_set_resieve(&cs);
    }
    mpz_clear(cand);
    gmp_randclear(state);

    double avg = (double)total / samples;
    double rel = fabs(avg - cs.avg_candidates) / cs.avg_candidates;

    printf("  avg survivors over %u offsets = %.1f, expected = %.1f, rel_dev = %.4f\n",
           samples, avg, cs.avg_candidates, rel);

    if (rel > 0.05) {
        fprintf(stderr, "FAIL: survivor average deviates >5%% from expectation\n");
        failures++;
    }
    crt_set_free(&cs);
    return failures;
}

static int test_search_offset(void) {
    struct crt_set cs;
    if (!crt_set_init(&cs, 512, 20000, crt_set_primes_for_shift(512))) {
        fprintf(stderr, "FAIL: init for search\n");
        return 1;
    }

    uint64_t zero_survivors = cs.n_candidates; /* offset = 0 */
    crt_set_search_offset(&cs, 500, 12345);
    uint64_t best_survivors = cs.n_candidates;

    printf("  survivors@0=%llu survivors@best=%llu\n",
           (unsigned long long)zero_survivors,
           (unsigned long long)best_survivors);

    crt_set_free(&cs);
    return (best_survivors > zero_survivors) ? 1 : 0;
}

static int test_roundtrip(void) {
    struct crt_set a, b;
    uint32_t n = crt_set_primes_for_shift(512);
    if (!crt_set_init(&a, 512, 20000, n)) {
        fprintf(stderr, "FAIL: init a\n");
        return 1;
    }
    crt_set_search_offset(&a, 200, 999);
    if (crt_set_save(&a, "/tmp/test_crt_set.txt") != 0) {
        fprintf(stderr, "FAIL: save\n");
        crt_set_free(&a);
        return 1;
    }
    if (crt_set_load(&b, "/tmp/test_crt_set.txt") != 0) {
        fprintf(stderr, "FAIL: load\n");
        crt_set_free(&a);
        return 1;
    }

    int ok = (a.n_primes == b.n_primes && a.size == b.size &&
              mpz_cmp(a.primorial, b.primorial) == 0 &&
              mpz_cmp(a.offset, b.offset) == 0 &&
              a.n_candidates == b.n_candidates);
    if (!ok) {
        fprintf(stderr, "FAIL: round-trip mismatch\n");
    }
    printf("  round-trip %s (n_primes=%u size=%llu candidates=%llu)\n",
           ok ? "OK" : "MISMATCH", b.n_primes,
           (unsigned long long)b.size, (unsigned long long)b.n_candidates);

    crt_set_free(&a);
    crt_set_free(&b);
    remove("/tmp/test_crt_set.txt");
    return ok ? 0 : 1;
}

int main(void) {
    int failures = 0;
    failures += test_primes_for_shift();
    failures += test_resieve_expectation();
    failures += test_search_offset();
    failures += test_roundtrip();

    if (failures == 0) {
        printf("PASS: CRT set and generator\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d checks failed\n", failures);
    return 1;
}
