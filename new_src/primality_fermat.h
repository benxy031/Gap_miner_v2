/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Primality Testing: CPU Fermat Test
 *
 * Fermat's Little Theorem: if p is prime, then a^(p-1) ≡ 1 (mod p)
 * Test with multiple bases (2, 3, 5, 7, 11, 13, ...)
 */

#ifndef PRIMALITY_FERMAT_H
#define PRIMALITY_FERMAT_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Fermat test context for reusing scratch space */
struct fermat_ctx {
    mpz_t n;              /* Number to test */
    mpz_t n_minus_1;      /* n - 1 (precomputed) */
    mpz_t result;         /* Scratch space for modexp */
};

/* Initialize Fermat context */
void fermat_ctx_init(struct fermat_ctx *ctx);

/* Test with specific base: a^(n-1) mod n == 1 */
int fermat_test_base(struct fermat_ctx *ctx, mpz_t n, uint64_t base);

/* Multi-round Fermat test (bases: 2, 3, 5, 7, ...) */
int fermat_test_probable_prime(mpz_t n, uint32_t rounds);

/* Cleanup */
void fermat_ctx_free(struct fermat_ctx *ctx);

#endif /* PRIMALITY_FERMAT_H */
