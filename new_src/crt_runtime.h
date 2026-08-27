/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CRT runtime: load a covering CRT file, build the static composite template,
 * and solve the CRT alignment for a block header.  This is the sieve-side of
 * the CRT pre-selection boost: the template pre-covers the gap-target interior
 * with composites, so the aligned window is prime-poor and its prime-gap tail
 * is heavier than the Cramer e^-m model.
 */

#ifndef CRT_RUNTIME_H
#define CRT_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <gmp.h>

struct crt_runtime {
    uint32_t n_primes;
    uint64_t *primes;        /* 2,3,5,... (n_primes entries, owned) */
    uint64_t *offsets;       /* o_i: position d ≡ o_i (mod p) is covered */

    uint32_t shift;          /* design shift from file */
    double   merit;          /* design merit from file */
    uint64_t gap_target;     /* G (interior [1, G) covered) */
    uint64_t window;         /* scan window = 2 * gap_target (min 10000) */
    uint64_t n_candidates;   /* survivors declared in the file */

    mpz_t primorial;         /* product of primes with offset != 0 */
    uint32_t adj;            /* base parity: candidate = base0 + nadd0 - adj */

    uint8_t *template;       /* odd-slot composite bitmap over [1, window) */
    uint64_t template_words;
    uint64_t n_survivors;    /* survivors recomputed from the template */
};

/* Load a text CRT file (the format written by gen_crt). */
int crt_runtime_load(struct crt_runtime *rt, const char *path);

void crt_runtime_free(struct crt_runtime *rt);

/* Build the static odd-slot composite template for the scan window.
 * slot s (0-based) covers odd offset t = 2*s+1; it is marked composite when
 * t ≡ offsets[i] (mod primes[i]) for some CRT prime (base ≡ -offsets[i]).
 * Recomputes n_survivors. */
int crt_runtime_build_template(struct crt_runtime *rt);

/* Set the mining scan window (2 * needed_gap) and rebuild the template.
 * The template marks primes uniformly for offsets >= gap_target, so the
 * covering only concentrates coverage inside [1, gap_target). */
int crt_runtime_set_window(struct crt_runtime *rt, uint64_t window);

/* Solve the CRT alignment nadd0 so that (h256<<shift) + nadd0 ≡ -offsets[i]
 * (mod primes[i]) for every prime with offset != 0.  Returns 0 on success. */
int crt_runtime_align(mpz_t nadd0, const uint8_t h256[32], uint32_t shift,
                      const struct crt_runtime *rt);

/* Fill survivor_offsets[] (ascending odd offsets t in [1, window)) with the
 * template survivors; returns the count (== rt->n_survivors). */
uint64_t crt_runtime_survivors(const struct crt_runtime *rt,
                               uint64_t *survivor_offsets);

#endif /* CRT_RUNTIME_H */
