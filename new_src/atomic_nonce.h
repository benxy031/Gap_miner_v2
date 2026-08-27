/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Atomic Nonce Management (Lock-Free)
 *
 * Provides atomic counter for lock-free race across workers.
 * Architecture: per-GPU nonce with atomic increment (CAS loop).
 */

#ifndef ATOMIC_NONCE_H
#define ATOMIC_NONCE_H

#include <stdint.h>
#include <stdatomic.h>

/* Atomic nonce state */
struct atomic_nonce {
    _Atomic(uint32_t) current;    /* Current nonce value */
    uint32_t max_value;           /* Upper bound (e.g., 2^32-1) */
};

/* Initialize atomic nonce */
void atomic_nonce_init(struct atomic_nonce *nonce, uint32_t start, uint32_t max);

/* Get and advance nonce (returns old value, increments for next caller) */
uint32_t atomic_nonce_next(struct atomic_nonce *nonce);

/* Claim up to requested values below limit and return the first claimed value. */
uint32_t atomic_nonce_claim(struct atomic_nonce *nonce, uint32_t limit,
                            uint32_t requested, uint32_t *claimed_count);

/* Check if nonce has reached max */
int atomic_nonce_exhausted(struct atomic_nonce *nonce);

/* Reset for new block */
void atomic_nonce_reset(struct atomic_nonce *nonce, uint32_t start);

#endif /* ATOMIC_NONCE_H */
