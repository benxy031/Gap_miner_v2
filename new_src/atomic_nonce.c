/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Atomic Nonce Implementation (Lock-Free)
 *
 * Uses C11 atomics for lock-free nonce advancement.
 */

#include "atomic_nonce.h"
#include <string.h>

void atomic_nonce_init(struct atomic_nonce *nonce, uint32_t start, uint32_t max) {
    if (!nonce) return;
    atomic_store(&nonce->current, start);
    nonce->max_value = max;
}

uint32_t atomic_nonce_next(struct atomic_nonce *nonce) {
    if (!nonce) return 0;
    
    uint32_t old_value;
    uint32_t new_value;
    
    /* CAS loop: keep trying until we successfully increment */
    do {
        old_value = atomic_load(&nonce->current);
        
        /* Check if exhausted */
        if (old_value >= nonce->max_value) {
            return nonce->max_value;  /* Signal exhaustion */
        }
        
        new_value = old_value + 1;
    } while (!atomic_compare_exchange_strong(&nonce->current, &old_value, new_value));
    
    return old_value;
}

uint32_t atomic_nonce_claim(struct atomic_nonce *nonce, uint32_t limit,
                            uint32_t requested, uint32_t *claimed_count) {
    if (claimed_count) *claimed_count = 0;
    if (!nonce || !claimed_count || requested == 0) return limit;

    if (limit > nonce->max_value) limit = nonce->max_value;

    uint32_t old_value;
    uint32_t new_value;
    uint32_t claim_size;
    do {
        old_value = atomic_load(&nonce->current);
        if (old_value >= limit) return limit;

        claim_size = limit - old_value;
        if (claim_size > requested) claim_size = requested;
        new_value = old_value + claim_size;
    } while (!atomic_compare_exchange_strong(&nonce->current, &old_value,
                                              new_value));

    *claimed_count = claim_size;
    return old_value;
}

int atomic_nonce_exhausted(struct atomic_nonce *nonce) {
    if (!nonce) return 1;
    return atomic_load(&nonce->current) >= nonce->max_value;
}

void atomic_nonce_reset(struct atomic_nonce *nonce, uint32_t start) {
    if (!nonce) return;
    atomic_store(&nonce->current, start);
}
