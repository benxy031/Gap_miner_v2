/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CRT (primorial) sieve set for one shift.
 *
 * A crt_set precomputes, for a given shift, a primorial P# (product of the
 * first n_primes) and a chosen residue `offset` (mod P#).  The miner then
 * scans [hash << shift + offset, + size) and tests only the survivors
 * (positions coprime to P#).  This is the primorial pre-sieve used by the
 * original Gapcoin miner, exposed here as a loadable per-shift file.
 *
 * Honest scope note: a FIXED residue mod P# is a plain pre-sieve — by
 * Dirichlet equidistribution it leaves the prime density inside the scan
 * interval unchanged, so on its own it only trims the per-window test cost
 * by a constant factor.  The covering-system boost (the ~190x merit-22 gain
 * confirmed against a live node) comes from a related but distinct technique:
 * aligning base ≡ -o_i (mod p_i) per header so the scan window is prime-poor
 * and the gap-length tail stretches by ~1.27x (see gap_target.h and
 * crt_runtime.h).  This module does not implement that alignment.
 */

#ifndef CRT_SET_H
#define CRT_SET_H

#include <stdio.h>   /* must precede gmp.h: FILE is used in its prototypes */
#include <stdint.h>
#include <gmp.h>

struct crt_set {
    uint32_t shift;          /* target shift (256..1024) */
    uint32_t n_primes;       /* number of primes in the primorial */
    uint64_t *primes;        /* the first n_primes primes (owned) */

    mpz_t primorial;         /* product of primes[0..n_primes) */
    mpz_t offset;            /* chosen residue mod primorial */

    uint64_t size;           /* window size (number of positions) */
    uint64_t bit_size;       /* ceil(log2(primorial)) = minimum shift */

    uint64_t n_candidates;   /* survivors in [offset, offset+size) */
    double avg_candidates;   /* size * prod(1 - 1/p): expected survivors */

    uint64_t *bitmap;        /* bit t = 1 if (offset+t) is composite */
    uint64_t bitmap_words;
};

/* Largest n such that the product of the first n primes is < 2^shift. */
uint32_t crt_set_primes_for_shift(uint32_t shift);

/* Initialize a crt_set for a shift with a given window size and prime count.
 * Allocates the prime table, primorial and bitmap; sets offset to 0. */
int crt_set_init(struct crt_set *cs, uint32_t shift, uint64_t size,
                 uint32_t n_primes);

void crt_set_free(struct crt_set *cs);

/* Rebuild the bitmap for the current offset and return the survivor count.
 * Bit t (0 <= t < size) is set if (offset + t) is divisible by a sieving
 * prime.  n_candidates is updated to the survivor count. */
uint64_t crt_set_resieve(struct crt_set *cs);

/* Randomly sample `trials` offsets (mod primorial) and keep the one with the
 * fewest survivors.  Deterministic for a fixed seed.  Returns 0 on success. */
int crt_set_search_offset(struct crt_set *cs, uint64_t trials, uint64_t seed);

/* Persistence: text format (one field per line, offset in decimal). */
int crt_set_save(const struct crt_set *cs, const char *path);
int crt_set_load(struct crt_set *cs, const char *path);

#endif /* CRT_SET_H */
