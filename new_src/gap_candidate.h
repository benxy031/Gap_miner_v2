/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gap Candidate Management
 *
 * Lazy materialization model:
 * - base: immutable base value (set by the caller)
 * - offsets: sieve-extracted offsets (computed once)
 * - Candidate P(i) = base + offsets[i]  (materialized on demand)
 */

#ifndef GAP_CANDIDATE_H
#define GAP_CANDIDATE_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Gap candidate descriptor: stores immutable base + extracted offsets */
struct gap_candidate {
    /* Immutable base information */
    mpz_t base;                   /* Base value (set by the caller) */
    uint32_t shift;               /* Shift (tracking only; not used in materialization) */
    uint32_t nonce;               /* Nonce for tracking */

    /* Sieve results */
    uint64_t *offsets;            /* Array of offsets (row indices) */
    uint32_t count;               /* Number of offsets */
    uint32_t capacity;            /* Allocated size */

    /* Precomputed stats */
    double candidate_density;     /* count / ln(base): survivor-density heuristic */
};

/* Initialize candidate container (empty) */
void gap_candidate_init(struct gap_candidate *gc, uint32_t shift, uint32_t nonce);

/* Allocate space for offsets */
void gap_candidate_reserve(struct gap_candidate *gc, uint32_t capacity);

/* Add offset to candidate list */
void gap_candidate_add_offset(struct gap_candidate *gc, uint64_t offset);

/* Finalize: compute candidate_density = count / ln(base) */
void gap_candidate_finalize(struct gap_candidate *gc);

/* Access candidate at index: materializes full value */
void gap_candidate_get(const struct gap_candidate *gc, uint32_t idx, mpz_t dest);

/* Cleanup */
void gap_candidate_free(struct gap_candidate *gc);

#endif /* GAP_CANDIDATE_H */
