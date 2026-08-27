/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gap_target.h"

#include <stdlib.h>
#include <string.h>

int crt_solve(mpz_t x, const uint64_t *primes, const uint64_t *residues,
              size_t n, const mpz_t primorial) {
    if (!primes || !residues || n == 0) {
        return -1;
    }

    mpz_set_ui(x, 0);
    for (size_t i = 0; i < n; i++) {
        uint64_t p = primes[i];
        mpz_t mi, mi_mod, inv, term;
        mpz_inits(mi, mi_mod, inv, term, NULL);

        mpz_divexact_ui(mi, primorial, p);   /* mi = primorial / p */
        mpz_mod_ui(mi_mod, mi, p);           /* mi mod p */
        mpz_set_ui(term, p);
        mpz_invert(inv, mi_mod, term);       /* inv = (mi mod p)^-1 mod p */

        mpz_mul_ui(mi, mi, residues[i]);     /* mi * r_i */
        mpz_mul(mi, mi, inv);                /* mi * r_i * inv */
        mpz_add(x, x, mi);
        mpz_mod(x, x, primorial);

        mpz_clears(mi, mi_mod, inv, term, NULL);
    }
    return 0;
}

int gap_target_build(struct gap_target *gt, uint64_t gap_size,
                     const uint64_t *primes, const uint64_t *residues,
                     size_t n_primes) {
    if (!gt || gap_size < 2 || !primes || !residues || n_primes == 0) {
        return 0;
    }

    uint8_t *composite = (uint8_t *)calloc((size_t)gap_size, 1);
    if (!composite) {
        return 0;
    }

    for (size_t i = 0; i < n_primes; i++) {
        uint64_t p = primes[i];
        uint64_t r = residues[i] % p;
        /* j is covered iff (r + j) % p == 0  ->  j == (p - r) % p (mod p) */
        uint64_t first = (p - r) % p;
        for (uint64_t j = first; j < gap_size; j += p) {
            composite[j] = 1;
        }
    }

    size_t count = 0;
    for (uint64_t j = 1; j < gap_size; j++) {
        if (!composite[j]) {
            count++;
        }
    }

    gt->survivors = (uint64_t *)malloc((count ? count : 1) * sizeof(uint64_t));
    if (!gt->survivors) {
        free(composite);
        return 0;
    }
    size_t k = 0;
    for (uint64_t j = 1; j < gap_size; j++) {
        if (!composite[j]) {
            gt->survivors[k++] = j;
        }
    }
    gt->n_survivors = k;
    gt->gap_size = gap_size;

    free(composite);
    return 1;
}

void gap_target_free(struct gap_target *gt) {
    if (!gt) return;
    free(gt->survivors);
    gt->survivors = NULL;
    gt->n_survivors = 0;
}

int gap_target_check(const struct gap_target *gt, const mpz_t base,
                     int (*is_prime)(void *ctx, const mpz_t n), void *ctx) {
    if (!gt || !base || !is_prime) {
        return 0;
    }

    if (!is_prime(ctx, base)) {
        return 0;
    }

    mpz_t cand;
    mpz_init(cand);
    int valid = 1;
    for (size_t k = 0; k < gt->n_survivors; k++) {
        mpz_set(cand, base);
        mpz_add_ui(cand, cand, gt->survivors[k]);
        if (is_prime(ctx, cand)) {
            valid = 0;
            break;
        }
    }
    mpz_clear(cand);
    return valid;
}
