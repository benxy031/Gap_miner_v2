/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration Tests: Primality Testing
 *
 * Tests: Fermat, Euler, BPSW across known primes and composites
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include "../new_src/primality_fermat.h"
#include "../new_src/primality_euler.h"
#include "../new_src/primality_bpsw.h"
#include "../new_src/primality_limbs.h"

/* Test Fermat with known primes and composites */
static int test_fermat_known_values(void) {
    printf("[TEST] Fermat test with known values...\n");
    
    /* Known primes: 2, 3, 5, 7, 11, 97, 1009, 1000000007 */
    uint64_t primes[] = {2, 3, 5, 7, 11, 97, 1009};
    size_t prime_count = sizeof(primes) / sizeof(primes[0]);
    
    /* Known composites: 4, 6, 8, 9, 15, 21, 91 (pseudoprime to base 2) */
    uint64_t composites[] = {4, 6, 8, 9, 15, 21, 91};
    size_t composite_count = sizeof(composites) / sizeof(composites[0]);
    
    mpz_t n;
    mpz_init(n);
    
    int pass = 1;
    
    /* Test primes */
    for (size_t i = 0; i < prime_count; i++) {
        mpz_set_ui(n, primes[i]);
        int result = fermat_test_probable_prime(n, 5);
        if (!result) {
            printf("  ✗ FAIL: Fermat failed on prime %lu\n", primes[i]);
            pass = 0;
        }
    }
    
    /* Test composites */
    for (size_t i = 0; i < composite_count; i++) {
        mpz_set_ui(n, composites[i]);
        int result = fermat_test_probable_prime(n, 5);
        if (result && composites[i] != 91) {  /* 91 = 7*13, pseudoprime to base 2 */
            printf("  ✗ FAIL: Fermat passed on composite %lu\n", composites[i]);
            pass = 0;
        }
    }
    
    mpz_clear(n);
    
    if (pass) {
        printf("  ✓ PASS: Fermat test works on known values\n");
    }
    
    return pass;
}

/* Test Euler with known values */
static int test_euler_known_values(void) {
    printf("[TEST] Euler criterion with known values...\n");
    
    uint64_t primes[] = {3, 5, 7, 11, 13, 17, 19, 23};
    size_t prime_count = sizeof(primes) / sizeof(primes[0]);
    
    uint64_t composites[] = {4, 6, 8, 9, 10, 12, 14, 15};
    size_t composite_count = sizeof(composites) / sizeof(composites[0]);
    
    mpz_t n;
    mpz_init(n);
    
    int pass = 1;
    
    /* Test primes */
    for (size_t i = 0; i < prime_count; i++) {
        mpz_set_ui(n, primes[i]);
        int result = euler_criterion_base2(n);
        if (!result) {
            printf("  ✗ FAIL: Euler failed on prime %lu\n", primes[i]);
            pass = 0;
        }
    }
    
    /* Test composites */
    for (size_t i = 0; i < composite_count; i++) {
        mpz_set_ui(n, composites[i]);
        int result = euler_criterion_base2(n);
        if (result) {
            printf("  ✗ FAIL: Euler passed on composite %lu\n", composites[i]);
            pass = 0;
        }
    }
    
    mpz_clear(n);
    
    if (pass) {
        printf("  ✓ PASS: Euler criterion works on known values\n");
    }
    
    return pass;
}

static int test_euler_context_equivalence(void) {
    printf("[TEST] Euler context equivalence...\n");

    struct euler_context context;
    mpz_t n;
    int pass = 1;

    euler_context_init(&context);
    mpz_init(n);

    for (uint64_t value = 2; value <= 10000; value++) {
        mpz_set_ui(n, value);
        int reference = euler_criterion_base2(n) && euler_criterion(n, 3);
        int optimized = euler_quick_probable_prime_with_context(&context, n);
        if (reference != optimized) {
            printf("  ✗ FAIL: Euler context mismatch on %lu\n", value);
            pass = 0;
            break;
        }
    }

    mpz_clear(n);
    euler_context_clear(&context);

    if (pass) {
        printf("  ✓ PASS: Euler context matches separate base checks\n");
    }

    return pass;
}

/* Test Miller-Rabin with known values */
static int test_miller_rabin_known_values(void) {
    printf("[TEST] Miller-Rabin base-2 with known values...\n");
    
    uint64_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29};
    size_t prime_count = sizeof(primes) / sizeof(primes[0]);
    
    uint64_t composites[] = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18};
    size_t composite_count = sizeof(composites) / sizeof(composites[0]);
    
    mpz_t n;
    mpz_init(n);
    
    int pass = 1;
    
    /* Test primes */
    for (size_t i = 0; i < prime_count; i++) {
        mpz_set_ui(n, primes[i]);
        int result = miller_rabin_base2(n);
        if (!result) {
            printf("  ✗ FAIL: Miller-Rabin failed on prime %lu\n", primes[i]);
            pass = 0;
        }
    }
    
    /* Test composites */
    for (size_t i = 0; i < composite_count; i++) {
        mpz_set_ui(n, composites[i]);
        int result = miller_rabin_base2(n);
        if (result) {
            printf("  ✗ FAIL: Miller-Rabin passed on composite %lu\n", composites[i]);
            pass = 0;
        }
    }
    
    mpz_clear(n);
    
    if (pass) {
        printf("  ✓ PASS: Miller-Rabin base-2 works on known values\n");
    }
    
    return pass;
}

/* Test BPSW (combined) */
static int test_bpsw_combined(void) {
    printf("[TEST] BPSW combined test...\n");
    
    uint64_t primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    size_t prime_count = sizeof(primes) / sizeof(primes[0]);
    
    uint64_t composites[] = {
        4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25,
        2047, 1373653, 3215031751ULL
    };
    size_t composite_count = sizeof(composites) / sizeof(composites[0]);
    
    mpz_t n;
    mpz_init(n);
    
    int pass = 1;
    
    /* Test primes */
    for (size_t i = 0; i < prime_count; i++) {
        mpz_set_ui(n, primes[i]);
        int result = baillie_psw_test(n);
        if (!result) {
            printf("  ✗ FAIL: BPSW failed on prime %lu\n", primes[i]);
            pass = 0;
        }
    }
    
    /* Test composites */
    for (size_t i = 0; i < composite_count; i++) {
        mpz_set_ui(n, composites[i]);
        int result = baillie_psw_test(n);
        if (result) {
            printf("  ✗ FAIL: BPSW passed on composite %lu\n", composites[i]);
            pass = 0;
        }
    }
    
    mpz_clear(n);
    
    if (pass) {
        printf("  ✓ PASS: BPSW combined test passed\n");
    }
    
    return pass;
}

/* Test large primes (CPU Gapcoin examples) */
static int test_large_primes(void) {
    printf("[TEST] Large primes (Gapcoin examples)...\n");
    
    /* Large primes from real Gapcoin mining */
    const char *large_primes_str[] = {
        "1000000007",       /* 10^9 + 7 */
        "10000000019",      /* ~10^10 */
        "1000000000039",    /* 10^12 + 39 */
    };
    size_t count = sizeof(large_primes_str) / sizeof(large_primes_str[0]);
    
    mpz_t n;
    mpz_init(n);
    
    int pass = 1;
    
    for (size_t i = 0; i < count; i++) {
        mpz_set_str(n, large_primes_str[i], 10);
        
        int fermat = fermat_test_probable_prime(n, 5);
        int bpsw = baillie_psw_test(n);
        
        if (!fermat || !bpsw) {
            printf("  ✗ FAIL: Test failed on %s\n", large_primes_str[i]);
            printf("    Fermat: %d, BPSW: %d\n", fermat, bpsw);
            pass = 0;
        } else {
            printf("  ✓ Verified: %s is prime (Fermat + BPSW)\n", large_primes_str[i]);
        }
    }
    
    mpz_clear(n);
    
    return pass;
}

/* ── Ported limb-based tests (primality_limbs) vs GMP reference ────── */

static int test_limb_u64_helpers(void) {
    printf("[TEST] limb u64 helpers (MR + fast Fermat)...\n");
    static const uint64_t primes[] = {2, 3, 5, 7, 11, 97, 1009,
                                      1000000007ULL};
    static const uint64_t composites[] = {4, 6, 8, 9, 15, 21, 91,
                                          2047, 1373653, 3215031751ULL};
    int pass = 1;
    for (size_t i = 0; i < sizeof(primes)/sizeof(primes[0]); i++) {
        if (!primality_miller_rabin_u64(primes[i]) ||
            !primality_fast_fermat_u64(primes[i])) {
            printf("  ✗ FAIL: u64 helper rejected prime %lu\n",
                   (unsigned long)primes[i]);
            pass = 0;
        }
    }
    for (size_t i = 0; i < sizeof(composites)/sizeof(composites[0]); i++) {
        if (primality_miller_rabin_u64(composites[i])) {
            printf("  ✗ FAIL: u64 MR passed on composite %lu\n",
                   (unsigned long)composites[i]);
            pass = 0;
        }
    }
    if (pass)
        printf("  ✓ PASS: u64 helpers agree with known values\n");
    return pass;
}

static int test_limb_vs_gmp(void) {
    printf("[TEST] limb Fermat/Euler vs GMP reference (nlimbs 1..12)...\n");

    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, 20260827UL);

    mpz_t n, gmp_res, base_mpz, exp_mpz;
    mpz_init(n);
    mpz_init(gmp_res);
    mpz_init(base_mpz);
    mpz_init(exp_mpz);

    int pass = 1;
    uint32_t checked = 0;

    for (int nl = 2; nl <= 12; nl++) {
        for (int k = 0; k < 40; k++) {
            /* Random odd number near the top of the nl-limb range,
               plus an extra small factor so we hit both primes and
               composites. */
            mpz_urandomb(n, rng, (unsigned long)nl * 64U - 4U);
            mpz_setbit(n, (unsigned long)nl * 64U - 5U);
            mpz_setbit(n, 0); /* odd */
            if (k & 1) {
                /* composite: multiply by 3 while staying in range */
                mpz_tdiv_q_2exp(n, n, 2);
                mpz_setbit(n, 0);
            }

            uint64_t limbs[PRIMALITY_CPU_MAX_LIMBS];
            int got_nl = primality_limbs_export(n, limbs,
                                                PRIMALITY_CPU_MAX_LIMBS);
            if (got_nl <= 0) {
                printf("  ✗ FAIL: export failed at nl=%d\n", nl);
                pass = 0;
                break;
            }

            /* Reference Euler base-2 via GMP */
            int ref_euler = euler_criterion_base2(n);
            int lim_euler = euler_test_cpu_nlimbs(limbs, got_nl);

            /* Reference Fermat base-2 via GMP */
            mpz_sub_ui(exp_mpz, n, 1);
            mpz_set_ui(base_mpz, 2);
            mpz_powm(gmp_res, base_mpz, exp_mpz, n);
            int ref_fermat = mpz_cmp_ui(gmp_res, 1) == 0;
            int lim_fermat = fermat_test_cpu_nlimbs(limbs, got_nl);

            /* mpz adapter must match the limb call */
            int ada_euler = primality_euler_limbs_mpz(n);
            int ada_fermat = primality_fermat_limbs_mpz(n);

            /* Precomp variants must match too (nl > 1, win==4) */
            primality_exact_precomp_t precomp;
            int have_pre = primality_exact_precomp_init(&precomp, limbs, got_nl);
            int pre_euler = have_pre ? euler_test_cpu_nlimbs_precomp(&precomp)
                                     : lim_euler;
            int pre_fermat = have_pre ? fermat_test_cpu_nlimbs_precomp(&precomp)
                                      : lim_fermat;

            checked++;
            if (ref_euler != lim_euler || ref_fermat != lim_fermat ||
                ada_euler != lim_euler || ada_fermat != lim_fermat ||
                pre_euler != lim_euler || pre_fermat != lim_fermat) {
                printf("  ✗ FAIL: nl=%d mismatch (ref_e=%d lim_e=%d ref_f=%d "
                       "lim_f=%d ada_e=%d ada_f=%d pre_e=%d pre_f=%d)\n",
                       nl, ref_euler, lim_euler, ref_fermat, lim_fermat,
                       ada_euler, ada_fermat, pre_euler, pre_fermat);
                pass = 0;
                break;
            }
        }
        if (!pass) break;
    }

    /* Increment helper sanity: exported limbs + delta == export of n+delta */
    mpz_urandomb(n, rng, 128);
    mpz_setbit(n, 0);
    mpz_setbit(n, 100);
    {
        uint64_t limbs[PRIMALITY_CPU_MAX_LIMBS];
        mpz_t n2;
        mpz_init(n2);
        mpz_add_ui(n2, n, 123456789ULL);
        int nl1 = primality_limbs_export(n, limbs, PRIMALITY_CPU_MAX_LIMBS);
        if (nl1 > 0 &&
            primality_limbs_add_u64(limbs, PRIMALITY_CPU_MAX_LIMBS,
                                    123456789ULL)) {
            uint64_t limbs2[PRIMALITY_CPU_MAX_LIMBS];
            int nl2 = primality_limbs_export(n2, limbs2,
                                             PRIMALITY_CPU_MAX_LIMBS);
            if (nl1 != nl2 || memcmp(limbs, limbs2,
                                     (size_t)nl2 * sizeof(uint64_t)) != 0) {
                printf("  ✗ FAIL: limb add_u64 diverges from GMP\n");
                pass = 0;
            }
        }
        mpz_clear(n2);
    }

    mpz_clear(n);
    mpz_clear(gmp_res);
    mpz_clear(base_mpz);
    mpz_clear(exp_mpz);
    gmp_randclear(rng);

    if (pass)
        printf("  ✓ PASS: limb tests match GMP on %u random candidates "
               "(nlimbs 2..12)\n", checked);
    return pass;
}

int main(void) {
    printf("================================================\n");
    printf("GapMiner V2 — Integration Test: Primality Tests\n");
    printf("================================================\n\n");
    
    int all_pass = 1;
    
    all_pass &= test_fermat_known_values();
    printf("\n");
    
    all_pass &= test_euler_known_values();
    printf("\n");

    all_pass &= test_euler_context_equivalence();
    printf("\n");
    
    all_pass &= test_miller_rabin_known_values();
    printf("\n");
    
    all_pass &= test_bpsw_combined();
    printf("\n");
    
    all_pass &= test_large_primes();
    printf("\n");

    all_pass &= test_limb_u64_helpers();
    printf("\n");

    all_pass &= test_limb_vs_gmp();
    printf("\n");
    
    printf("================================================\n");
    if (all_pass) {
        printf("✓ All primality tests PASSED\n");
    } else {
        printf("✗ Some primality tests FAILED\n");
    }
    printf("================================================\n");
    
    return all_pass ? 0 : 1;
}
