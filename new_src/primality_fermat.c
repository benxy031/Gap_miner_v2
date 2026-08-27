/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Fermat Test Implementation
 *
 * Note: Not deterministic alone, but useful for quick filtering.
 * Always combine with BPSW for verification.
 */

#include "primality_fermat.h"
#include <stdlib.h>

void fermat_ctx_init(struct fermat_ctx *ctx) {
    if (!ctx) return;
    mpz_init(ctx->n);
    mpz_init(ctx->n_minus_1);
    mpz_init(ctx->result);
}

int fermat_test_base(struct fermat_ctx *ctx, mpz_t n, uint64_t base) {
    if (!ctx) return 0;
    
    /* Compute n - 1 */
    mpz_sub_ui(ctx->n_minus_1, n, 1);
    
    /* Compute base^(n-1) mod n */
    mpz_set_ui(ctx->result, base);
    mpz_powm(ctx->result, ctx->result, ctx->n_minus_1, n);
    
    /* Check if result == 1 */
    if (mpz_cmp_ui(ctx->result, 1) == 0) {
        return 1;  /* Probably prime */
    }
    
    return 0;  /* Definitely composite */
}

int fermat_test_probable_prime(mpz_t n, uint32_t rounds) {
    if (!n || rounds == 0) return 0;
    
    /* Handle small cases */
    if (mpz_cmp_ui(n, 2) < 0) return 0;
    if (mpz_cmp_ui(n, 2) == 0) return 1;
    if (mpz_even_p(n)) return 0;  /* Even → composite */
    
    struct fermat_ctx ctx;
    fermat_ctx_init(&ctx);
    mpz_set(ctx.n, n);
    
    /* Test with first 'rounds' small primes (2, 3, 5, 7, 11, ...) */
    uint64_t bases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    size_t base_count = sizeof(bases) / sizeof(bases[0]);
    
    if (rounds > base_count) rounds = base_count;
    
    for (uint32_t i = 0; i < rounds; i++) {
        if (mpz_cmp_ui(n, bases[i]) == 0) {
            /* n is one of the bases itself → prime */
            fermat_ctx_free(&ctx);
            return 1;
        }
        
        if (!fermat_test_base(&ctx, n, bases[i])) {
            /* Failed test → definitely composite */
            fermat_ctx_free(&ctx);
            return 0;
        }
    }
    
    fermat_ctx_free(&ctx);
    return 1;  /* Probably prime (probabilistic) */
}

void fermat_ctx_free(struct fermat_ctx *ctx) {
    if (!ctx) return;
    mpz_clear(ctx->n);
    mpz_clear(ctx->n_minus_1);
    mpz_clear(ctx->result);
}
