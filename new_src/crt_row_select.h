/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef CRT_ROW_SELECT_H
#define CRT_ROW_SELECT_H

#include <stddef.h>
#include <stdint.h>

/* Prepare powers of the conditional composite probability for one sieve depth. */
int crt_row_score_prepare(double *powers, size_t count, double prime_probability);

/* Score qualifying gaps in an odd-slot composite bitmap.  prefix has at least
   ceil(odd_slots/64)+1 entries; powers has at least odd_slots+1 entries. */
int crt_row_score_bitmap(const uint64_t *bitmap, uint64_t odd_slots,
                         uint64_t first_odd_offset, uint64_t gap,
                         uint64_t owned_limit, double prime_probability,
                         const double *powers, size_t powers_count,
                         uint32_t *prefix, size_t prefix_count,
                         double *score);

/* Deterministically select the highest scores; equal scores prefer lower rows. */
int crt_row_select_best(const double *scores, uint32_t rows, uint32_t keep,
                        uint64_t *mask);

#endif
