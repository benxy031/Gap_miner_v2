/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Baillie-PSW Test Implementation
 *
 * Strong probable-prime combination of:
 * 1. Miller-Rabin with base 2 (strong pseudoprime test)
 * 2. Strong Lucas-Selfridge test
 *
 * No composite is known to pass both tests.
 */

#include "primality_bpsw.h"
#include <stdlib.h>

static void mod_divide_by_two(mpz_t result, const mpz_t value,
                              const mpz_t modulus) {
    mpz_mod(result, value, modulus);
    if (mpz_odd_p(result)) {
        mpz_add(result, result, modulus);
    }
    mpz_fdiv_q_2exp(result, result, 1);
}

/* Miller-Rabin with base 2: write n-1 = 2^r * d */
int miller_rabin_base2(mpz_t n) {
    if (!n) return 0;
    
    /* Handle small cases */
    if (mpz_cmp_ui(n, 2) < 0) return 0;
    if (mpz_cmp_ui(n, 2) == 0) return 1;
    if (mpz_even_p(n)) return 0;
    
    /* Write n-1 = 2^r * d where d is odd */
    mpz_t d, n_minus_1, x;
    mpz_init(d);
    mpz_init(n_minus_1);
    mpz_init(x);
    
    mpz_sub_ui(n_minus_1, n, 1);
    mpz_set(d, n_minus_1);
    
    /* Factor out all 2s */
    int r = 0;
    while (mpz_even_p(d)) {
        mpz_fdiv_q_2exp(d, d, 1);
        r++;
    }
    
    /* Compute x = 2^d mod n */
    mpz_set_ui(x, 2);
    mpz_powm(x, x, d, n);
    
    /* Check: x == 1 or x == n-1 */
    if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, n_minus_1) == 0) {
        mpz_clear(d);
        mpz_clear(n_minus_1);
        mpz_clear(x);
        return 1;
    }
    
    /* Square x repeatedly r-1 times */
    for (int i = 0; i < r - 1; i++) {
        mpz_mul(x, x, x);
        mpz_mod(x, x, n);
        
        if (mpz_cmp(x, n_minus_1) == 0) {
            mpz_clear(d);
            mpz_clear(n_minus_1);
            mpz_clear(x);
            return 1;
        }
    }
    
    mpz_clear(d);
    mpz_clear(n_minus_1);
    mpz_clear(x);
    
    return 0;  /* Composite */
}

/* Strong Lucas-Selfridge probable-prime test. */
int lucas_lehmer_test(mpz_t n) {
    if (!n) return 0;
    
    /* Handle small cases */
    if (mpz_cmp_ui(n, 2) < 0) return 0;
    if (mpz_cmp_ui(n, 2) == 0) return 1;
    if (mpz_even_p(n)) return 0;
    
    if (mpz_perfect_square_p(n)) return 0;

    mpz_t d, abs_d, q, n_plus_one, odd_part;
    mpz_t u, v, q_to_k, u_double, v_double, q_double;
    mpz_t next_u, next_v, next_q, scratch_a, scratch_b;
    mpz_inits(d, abs_d, q, n_plus_one, odd_part,
              u, v, q_to_k, u_double, v_double, q_double,
              next_u, next_v, next_q, scratch_a, scratch_b, NULL);

    mpz_set_si(d, 5);
    for (;;) {
        int jacobi = mpz_jacobi(d, n);
        if (jacobi == -1) {
            break;
        }
        if (jacobi == 0) {
            mpz_abs(abs_d, d);
            int result = mpz_cmp(abs_d, n) == 0;
            mpz_clears(d, abs_d, q, n_plus_one, odd_part,
                       u, v, q_to_k, u_double, v_double, q_double,
                       next_u, next_v, next_q, scratch_a, scratch_b, NULL);
            return result;
        }

        if (mpz_sgn(d) > 0) {
            mpz_add_ui(d, d, 2);
            mpz_neg(d, d);
        } else {
            mpz_neg(d, d);
            mpz_add_ui(d, d, 2);
        }
    }

    /* Selfridge chooses P = 1 and Q = (1 - D) / 4. */
    mpz_ui_sub(q, 1, d);
    mpz_fdiv_q_ui(q, q, 4);

    mpz_add_ui(n_plus_one, n, 1);
    mp_bitcnt_t powers_of_two = mpz_scan1(n_plus_one, 0);
    mpz_fdiv_q_2exp(odd_part, n_plus_one, powers_of_two);

    /* U_1 = 1, V_1 = P = 1 and Q^1 = Q. */
    mpz_set_ui(u, 1);
    mpz_set_ui(v, 1);
    mpz_mod(q_to_k, q, n);

    size_t bit_count = mpz_sizeinbase(odd_part, 2);
    for (size_t bit = bit_count - 1; bit-- > 0;) {
        /* Double k: U_2k = U_k V_k, V_2k = V_k^2 - 2Q^k. */
        mpz_mul(u_double, u, v);
        mpz_mod(u_double, u_double, n);

        mpz_mul(v_double, v, v);
        mpz_mul_ui(scratch_a, q_to_k, 2);
        mpz_sub(v_double, v_double, scratch_a);
        mpz_mod(v_double, v_double, n);

        mpz_mul(q_double, q_to_k, q_to_k);
        mpz_mod(q_double, q_double, n);

        if (mpz_tstbit(odd_part, bit)) {
            /* Advance 2k to 2k+1. */
            mpz_add(scratch_a, u_double, v_double);
            mod_divide_by_two(next_u, scratch_a, n);

            mpz_mul(scratch_a, d, u_double);
            mpz_add(scratch_a, scratch_a, v_double);
            mod_divide_by_two(next_v, scratch_a, n);

            mpz_mul(next_q, q_double, q);
            mpz_mod(next_q, next_q, n);
        } else {
            mpz_set(next_u, u_double);
            mpz_set(next_v, v_double);
            mpz_set(next_q, q_double);
        }

        mpz_set(u, next_u);
        mpz_set(v, next_v);
        mpz_set(q_to_k, next_q);
    }

    int result = mpz_cmp_ui(u, 0) == 0 || mpz_cmp_ui(v, 0) == 0;
    for (mp_bitcnt_t round = 1; !result && round < powers_of_two; round++) {
        mpz_mul(v, v, v);
        mpz_mul_ui(scratch_a, q_to_k, 2);
        mpz_sub(v, v, scratch_a);
        mpz_mod(v, v, n);

        mpz_mul(q_to_k, q_to_k, q_to_k);
        mpz_mod(q_to_k, q_to_k, n);
        result = mpz_cmp_ui(v, 0) == 0;
    }

    mpz_clears(d, abs_d, q, n_plus_one, odd_part,
               u, v, q_to_k, u_double, v_double, q_double,
               next_u, next_v, next_q, scratch_a, scratch_b, NULL);
    return result;
}

int baillie_psw_test(mpz_t n) {
    if (!n) return 0;
    
    /* Step 1: Miller-Rabin with base 2 */
    if (!miller_rabin_base2(n)) {
        return 0;  /* Definitely composite */
    }
    
    /* Step 2: Strong Lucas-Selfridge test. */
    if (!lucas_lehmer_test(n)) {
        return 0;  /* Definitely composite */
    }
    
    return 1;  /* Probably prime (deterministic up to 3.4×10^14) */
}

int baillie_psw_verify_gap_boundaries(mpz_t p1, mpz_t p2) {
    if (!p1 || !p2) return 0;
    
    /* Verify both boundaries are prime */
    if (!baillie_psw_test(p1)) return 0;
    if (!baillie_psw_test(p2)) return 0;
    
    return 1;  /* Both are prime */
}
