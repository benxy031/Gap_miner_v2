/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Primality Testing: Euler's Criterion
 *
 * Euler's Criterion (2x faster than Fermat):
 * For odd p, (a/p) = a^((p-1)/2) mod p
 * where (a/p) is the Legendre symbol
 *
 * If (2/p) ≠ ±1, then p is composite (high probability)
 */

#ifndef PRIMALITY_EULER_H
#define PRIMALITY_EULER_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

struct euler_context {
	mpz_t exponent;
	mpz_t result;
	mpz_t n_minus_one;
	mpz_t base;
};

void euler_context_init(struct euler_context *context);
void euler_context_clear(struct euler_context *context);

/* Euler's criterion test with base 2 */
int euler_criterion_base2(mpz_t n);

/* Euler's criterion test with base a */
int euler_criterion(mpz_t n, uint64_t base);

/* Quick filter: test with Euler, then Fermat if needed */
int euler_quick_probable_prime(mpz_t n);

/* Quick filter using caller-owned GMP temporaries. */
int euler_quick_probable_prime_with_context(struct euler_context *context,
											 const mpz_t n);

#endif /* PRIMALITY_EULER_H */
