/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Targeted-merit gap search (covering-system construction).
 *
 * For a target gap size G and a prime table, pick residues r_p (1 <= r_p < p)
 * so that the gap start x satisfies x = r_p (mod p).  A position j in the gap
 * interior is then covered (forced composite) iff r_p + j = 0 (mod p) for some
 * prime p.  The survivors (uncovered offsets) are the positions that still
 * need a primality test: the gap at x is valid iff x is prime AND x + off is
 * composite for every survivor offset off.
 *
 * Honest scope note: for a SINGLE fixed gap start x this is a sieve — primes
 * larger than the largest sieve prime can never be covered, so about
 * G/ln(x) survivor primes remain and P(valid) = exp(-G/ln(x)); it does not
 * reduce the e^merit bound for a targeted single-gap search.
 *
 * The mining boost is a different regime: aligning the scan window per header
 * (base ≡ -o_i mod p_i) makes the whole [1, G) interior composite, so the
 * window is ~20% prime-poor and the consecutive-prime gap tail stretches by
 * ~1.27x — P(gap >= m) ≈ exp(-m/1.27) instead of Cramer exp(-m), about 190x
 * more merit-22 gaps (confirmed against a live node; see crt_runtime.h).
 */

#ifndef GAP_TARGET_H
#define GAP_TARGET_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>   /* before gmp.h: FILE in its prototypes */
#include <gmp.h>

struct gap_target {
    uint64_t gap_size;
    uint64_t *survivors;    /* offsets in [1, gap_size-1], ascending */
    size_t n_survivors;
};

/* Chinese Remainder Theorem: solve x = residues[i] (mod primes[i]).
 * primorial = product of primes; result is in [0, primorial). */
int crt_solve(mpz_t x, const uint64_t *primes, const uint64_t *residues,
              size_t n, const mpz_t primorial);

/* Build the covering for a gap of the given size with the given residues.
 * Survivor offsets j are those where (residues[i] + j) % primes[i] != 0 for
 * every prime.  Allocates gt->survivors. */
int gap_target_build(struct gap_target *gt, uint64_t gap_size,
                     const uint64_t *primes, const uint64_t *residues,
                     size_t n_primes);

void gap_target_free(struct gap_target *gt);

/* Verify a gap at base: base is (probable) prime via is_prime, and every
 * base + survivors[k] is composite.  Returns 1 iff the gap is valid. */
int gap_target_check(const struct gap_target *gt, const mpz_t base,
                     int (*is_prime)(void *ctx, const mpz_t n), void *ctx);

#endif /* GAP_TARGET_H */
