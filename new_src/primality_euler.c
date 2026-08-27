/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Euler's Criterion Implementation
 *
 * Euler's Criterion (2x faster than Fermat):
 * (2/p) = 2^((p-1)/2) mod p
 * For prime p: (2/p) = 1 if p ≡ 1,7 (mod 8), else -1 if p ≡ 3,5 (mod 8)
 */

#include "primality_euler.h"
#include <stdlib.h>

void euler_context_init(struct euler_context *context) {
    if (!context) return;
    mpz_init(context->exponent);
    mpz_init(context->result);
    mpz_init(context->n_minus_one);
    mpz_init(context->base);
}

void euler_context_clear(struct euler_context *context) {
    if (!context) return;
    mpz_clear(context->exponent);
    mpz_clear(context->result);
    mpz_clear(context->n_minus_one);
    mpz_clear(context->base);
}

int euler_criterion_base2(mpz_t n) {
    if (!n) return 0;
    
    /* Handle small cases */
    if (mpz_cmp_ui(n, 2) <= 0) return 0;
    if (mpz_even_p(n)) return 0;
    
    /* Compute (n-1)/2 */
    mpz_t exp, result;
    mpz_init(exp);
    mpz_init(result);
    
    mpz_sub_ui(exp, n, 1);
    mpz_fdiv_q_2exp(exp, exp, 1);  /* (n-1)/2 */
    
    /* Compute 2^((n-1)/2) mod n. */
    mpz_set_ui(result, 2);
    mpz_powm(result, result, exp, n);
    
    /* Result should be 1 or n-1 (which is -1 mod n) */
    int ret = 0;
    if (mpz_cmp_ui(result, 1) == 0) {
        ret = 1;  /* Consistent with being prime (quadratic residue) */
    } else {
        mpz_set(exp, n);
        mpz_sub_ui(exp, exp, 1);  /* exp = n - 1 */
        if (mpz_cmp(result, exp) == 0) {
            ret = 1;  /* Result is -1 mod n (quadratic non-residue) */
        }
    }
    
    mpz_clear(exp);
    mpz_clear(result);
    
    return ret;
}

int euler_criterion(mpz_t n, uint64_t base) {
    if (!n || base == 0) return 0;
    
    /* Handle small cases */
    if (mpz_cmp_ui(n, 2) <= 0) return 0;
    if (mpz_even_p(n)) return 0;
    
    /* Compute (n-1)/2 */
    mpz_t exp, result, base_mpz;
    mpz_init(exp);
    mpz_init(result);
    mpz_init_set_ui(base_mpz, base);
    
    mpz_sub_ui(exp, n, 1);
    mpz_fdiv_q_2exp(exp, exp, 1);  /* (n-1)/2 */
    
    /* Compute base^((n-1)/2) mod n */
    mpz_powm(result, base_mpz, exp, n);
    
    /* Result should be 1 or n-1 */
    int ret = 0;
    if (mpz_cmp_ui(result, 1) == 0) {
        ret = 1;
    } else {
        mpz_set(exp, n);
        mpz_sub_ui(exp, exp, 1);  /* exp = n - 1 */
        if (mpz_cmp(result, exp) == 0) {
            ret = 1;
        }
    }
    
    mpz_clear(exp);
    mpz_clear(result);
    mpz_clear(base_mpz);
    
    return ret;
}

int euler_quick_probable_prime(mpz_t n) {
    struct euler_context context;
    euler_context_init(&context);
    int result = euler_quick_probable_prime_with_context(&context, n);
    euler_context_clear(&context);
    return result;
}

int euler_quick_probable_prime_with_context(struct euler_context *context,
                                             const mpz_t n) {
    if (!context || !n) return 0;

    if (mpz_cmp_ui(n, 2) <= 0 || mpz_even_p(n)) return 0;

    mpz_sub_ui(context->exponent, n, 1);
    mpz_fdiv_q_2exp(context->exponent, context->exponent, 1);
    mpz_sub_ui(context->n_minus_one, n, 1);

    mpz_set_ui(context->base, 2);
    mpz_powm(context->result, context->base, context->exponent, n);
    if (mpz_cmp_ui(context->result, 1) != 0 &&
        mpz_cmp(context->result, context->n_minus_one) != 0) {
        return 0;
    }

    mpz_set_ui(context->base, 3);
    mpz_powm(context->result, context->base, context->exponent, n);
    return mpz_cmp_ui(context->result, 1) == 0 ||
           mpz_cmp(context->result, context->n_minus_one) == 0;
}
