/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gap_candidate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

void gap_candidate_init(struct gap_candidate *gc, uint32_t shift, uint32_t nonce) {
    if (!gc) return;
    
    mpz_init(gc->base);
    gc->shift = shift;
    gc->nonce = nonce;
    
    gc->offsets = NULL;
    gc->count = 0;
    gc->capacity = 0;
    gc->candidate_density = 0.0;
}

void gap_candidate_reserve(struct gap_candidate *gc, uint32_t capacity) {
    if (!gc || capacity == 0) return;
    
    if (gc->capacity >= capacity) return;  /* Already sufficient */
    
    gc->offsets = (uint64_t *)realloc(gc->offsets, capacity * sizeof(uint64_t));
    gc->capacity = capacity;
}

void gap_candidate_add_offset(struct gap_candidate *gc, uint64_t offset) {
    if (!gc) return;
    
    if (gc->count >= gc->capacity) {
        /* Auto-expand by 50% */
        uint32_t new_cap = gc->capacity + (gc->capacity / 2) + 100;
        gap_candidate_reserve(gc, new_cap);
    }
    
    gc->offsets[gc->count++] = offset;
}

void gap_candidate_finalize(struct gap_candidate *gc) {
    if (!gc || gc->count == 0) {
        gc->candidate_density = 0.0;
        return;
    }
    
    /* Density heuristic: count / ln(base) */
    double base_d = mpz_get_d(gc->base);
    if (base_d > 1.0) {
        gc->candidate_density = (double)gc->count / log(base_d);
    } else {
        gc->candidate_density = 0.0;
    }
}

void gap_candidate_get(const struct gap_candidate *gc, uint32_t idx, mpz_t dest) {
    if (!gc || idx >= gc->count) {
        mpz_set_ui(dest, 0);
        return;
    }
    
    /* Materialize: P(idx) = base + offsets[idx] */
    mpz_set(dest, gc->base);
    mpz_add_ui(dest, dest, gc->offsets[idx]);
}

void gap_candidate_free(struct gap_candidate *gc) {
    if (!gc) return;
    
    mpz_clear(gc->base);
    
    if (gc->offsets) {
        free(gc->offsets);
        gc->offsets = NULL;
    }
    
    gc->count = 0;
    gc->capacity = 0;
}
