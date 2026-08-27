/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration Test: Gap Detection
 *
 * Test consecutive prime pair detection with known candidates and primality results.
 */

#include "../new_src/gap_detection.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Test case: sample offsets + primality results */
static void test_gap_detection_basic(void) {
    printf("[TEST] Basic gap detection...\n");
    
    /* Sample offsets: [2, 5, 7, 13, 19, 23, 29, 31, 37, 41, ...] */
    uint64_t offsets[] = {2, 5, 7, 13, 19, 23, 29, 31, 37, 41};
    uint32_t count = sizeof(offsets) / sizeof(offsets[0]);
    
    /* Primality results: all offsets above are treated as probable primes for this test */
    uint8_t is_prime[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    /* Base value for merit calculation */
    mpz_t base;
    mpz_init_set_ui(base, 1000000);  /* 1 million */
    
    /* Debug: print input data */
    printf("  Input: %u candidates, base=%lu, shift=512\n", count, 1000000UL);
    printf("  Offsets: ");
    for (uint32_t i = 0; i < count; i++) printf("%llu ", offsets[i]);
    printf("\n");
    printf("  is_prime: ");
    for (uint32_t i = 0; i < count; i++) printf("%d ", is_prime[i]);
    printf("\n");
    
    /* Run gap detection with merit threshold 10.0 */
    struct gap_result *gaps = NULL;
    uint32_t gap_count = 0;
    
    int result = gap_detection_find(
        is_prime,
        offsets,
        count,
        0,  /* shift = 0 for testing (realistic shift would be 256-1024) */
        base,
        0.1,  /* Lower merit threshold for test (realistic would be 12-20) */
        UINT64_MAX,
        NULL,
        &gaps,
        &gap_count
    );
    
    printf("  Result: %d, Gap count: %u\n", result, gap_count);
    
    if (result && gap_count > 0) {
        printf("  ✓ Found %u gaps:\n", gap_count);
        for (uint32_t i = 0; i < gap_count; i++) {
            printf("    Gap #%u: offset[%llu..%llu], length=%u, merit=%.2f\n",
                   i, gaps[i].offset_p1, gaps[i].offset_p2, 
                   gaps[i].gap_length, gaps[i].merit);
        }
    } else {
        printf("  ✗ No gaps detected\n");
    }
    
    if (gap_count > 0) {
        /* Expected: (2,5), (5,7), (7,13), (13,19), (19,23), (23,29), (29,31), (31,37), (37,41) */
        /* All with gap_length=3,2,6,6,4,6,2,6,4 respectively */
        printf("  Expected 9 consecutive gaps\n");
        if (gap_count == 9) {
            printf("  ✓ PASS: Correct number of gaps\n");
        } else {
            printf("  ✗ FAIL: Expected 9, got %u\n", gap_count);
        }
    }
    
    gap_detection_free_results(gaps);
    mpz_clear(base);
}

/* Test case: sparse primality results (some composites) */
static void test_gap_detection_sparse(void) {
    printf("\n[TEST] Sparse primality (with composites)...\n");
    
    /* Offsets: [2, 5, 7, 13, 19, 23, 29, 31, 37, 41] */
    uint64_t offsets[] = {2, 5, 7, 13, 19, 23, 29, 31, 37, 41};
    uint32_t count = sizeof(offsets) / sizeof(offsets[0]);
    
    /* Primality: [1, 1, 0, 1, 1, 0, 1, 1, 0, 1] (7, 23, 37 are marked composite) */
    uint8_t is_prime[] = {1, 1, 0, 1, 1, 0, 1, 1, 0, 1};
    
    mpz_t base;
    mpz_init_set_ui(base, 1000000);
    
    struct gap_result *gaps = NULL;
    uint32_t gap_count = 0;
    
    int result = gap_detection_find(
        is_prime,
        offsets,
        count,
        0,  /* shift = 0 for testing */
        base,
        0.1,  /* low threshold */
        UINT64_MAX,
        NULL,
        &gaps,
        &gap_count
    );
    
    printf("  Result: %d, Gap count: %u\n", result, gap_count);
    
    if (result && gap_count > 0) {
        printf("  ✓ Found %u gaps:\n", gap_count);
        for (uint32_t i = 0; i < gap_count; i++) {
            printf("    Gap #%u: offset[%llu..%llu], length=%u, merit=%.2f\n",
                   i, gaps[i].offset_p1, gaps[i].offset_p2, 
                   gaps[i].gap_length, gaps[i].merit);
        }
        
        printf("  Expected 6 Euler-positive gaps, including across rejected survivors\n");
        if (gap_count == 6) {
            printf("  ✓ PASS: Correct number of gaps\n");
        } else {
            printf("  ✗ FAIL: Expected 3, got %u\n", gap_count);
        }
    } else {
        printf("  ✗ No gaps detected (unexpected)\n");
    }
    
    gap_detection_free_results(gaps);
    mpz_clear(base);
}

/* Test case: merit filtering */
static void test_gap_detection_merit_filter(void) {
    printf("\n[TEST] Merit filtering...\n");
    
    /* Larger gaps with meaningful merit values */
    uint64_t offsets[] = {1000, 1100, 1200, 1300, 1400, 1500};
    uint32_t count = sizeof(offsets) / sizeof(offsets[0]);
    uint8_t is_prime[] = {1, 1, 1, 1, 1, 1};
    
    mpz_t base;
    mpz_init_set_ui(base, 10000000);  /* 10 million */
    
    /* High threshold should filter out most gaps */
    struct gap_result *gaps = NULL;
    uint32_t gap_count = 0;
    
    int result = gap_detection_find(
        is_prime,
        offsets,
        count,
        0,  /* shift = 0 for testing */
        base,
        50.0,  /* Very high threshold */
        UINT64_MAX,
        NULL,
        &gaps,
        &gap_count
    );
    
    printf("  Result: %d, Gap count: %u (with threshold=50.0)\n", result, gap_count);
    
    if (gap_count == 0) {
        printf("  ✓ PASS: High threshold filtered all gaps\n");
    } else {
        printf("  ✗ FAIL: Expected 0 gaps with high threshold, got %u\n", gap_count);
    }
    
    if (gaps) gap_detection_free_results(gaps);
    
    /* Low threshold should find all consecutive pairs */
    gap_count = 0;
    result = gap_detection_find(
        is_prime,
        offsets,
        count,
        0,  /* shift = 0 for testing */
        base,
        0.1,  /* Very low threshold */
        UINT64_MAX,
        NULL,
        &gaps,
        &gap_count
    );
    
    printf("  Result: %d, Gap count: %u (with threshold=0.1)\n", result, gap_count);
    
    if (gap_count > 0) {
        printf("  ✓ Found %u gaps with low threshold\n", gap_count);
    } else {
        printf("  ✗ FAIL: Expected gaps with low threshold\n");
    }
    
    gap_detection_free_results(gaps);
    mpz_clear(base);
}

static void test_gap_detection_lookahead_ownership(void) {
    printf("\n[TEST] Look-ahead ownership...\n");

    uint64_t offsets[] = {100, 4300, 5000};
    uint8_t is_prime[] = {1, 1, 1};
    struct gap_result *gaps = NULL;
    struct gap_scan_stats stats;
    uint32_t gap_count = 0;
    mpz_t base;
    mpz_init_set_ui(base, 1000000);

    int result = gap_detection_find(is_prime, offsets, 3, 0, base, 0.1,
                                    4096, &stats, &gaps, &gap_count);
    if (result && gap_count == 1 && gaps[0].offset_p1 == 100 &&
        gaps[0].offset_p2 == 4300 && stats.max_gap_length == 4200) {
        printf("  ✓ PASS: Recorded owned gap across the 4096-adder boundary\n");
    } else {
        printf("  ✗ FAIL: Look-ahead ownership was not enforced\n");
    }

    gap_detection_free_results(gaps);
    mpz_clear(base);
}

static void test_gap_detection_wide_merit(void) {
    printf("\n[TEST] Wide (>1024-bit) merit computation...\n");

    /* shift 998 -> 1254-bit candidates: mpz_get_d used to overflow to +inf
       and make every merit 0 (gap shown but merit 0.00, no candidates). */
    uint64_t offsets[] = {1000, 19312};   /* gap 18312, like the report */
    uint8_t is_prime[] = {1, 1};
    struct gap_result *gaps = NULL;
    struct gap_scan_stats stats;
    uint32_t gap_count = 0;

    mpz_t base;
    mpz_init(base);
    /* ~1254-bit base: 2^1253 */
    mpz_setbit(base, 1253);

    double expected_ln = 1253.0 * log(2.0);
    double expected_merit = 18312.0 / expected_ln;

    int result = gap_detection_find(is_prime, offsets, 2, 998, base, 0.1,
                                    UINT64_MAX, &stats, &gaps, &gap_count);
    printf("  gap_count=%u max_len=%u max_merit=%.6f expected_merit=%.6f\n",
           gap_count, stats.max_gap_length, stats.max_merit, expected_merit);

    if (result && gap_count == 1 && stats.max_gap_length == 18312 &&
        fabs(stats.max_merit - expected_merit) < 0.01) {
        printf("  ✓ PASS: wide merit computed correctly (%.4f)\n",
               stats.max_merit);
    } else {
        printf("  ✗ FAIL: wide merit broken (%.4f vs expected %.4f)\n",
               stats.max_merit, expected_merit);
    }

    /* compute_merit helper with the same wide start. */
    double m2 = gap_detection_compute_merit(18312, base);
    printf("  compute_merit = %.6f (expected %.6f) %s\n", m2, expected_merit,
           fabs(m2 - expected_merit) < 0.01 ? "✓" : "✗ FAIL");

    gap_detection_free_results(gaps);
    mpz_clear(base);
}

int main(void) {
    printf("===============================================\n");
    printf("GapMiner V2 — Integration Test: Gap Detection\n");
    printf("===============================================\n\n");
    
    test_gap_detection_basic();
    test_gap_detection_sparse();
    test_gap_detection_merit_filter();
    test_gap_detection_lookahead_ownership();
    test_gap_detection_wide_merit();
    
    printf("\n===============================================\n");
    printf("Integration tests completed\n");
    printf("===============================================\n");
    
    return 0;
}
