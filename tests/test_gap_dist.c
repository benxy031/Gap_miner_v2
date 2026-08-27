/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Unit Test: Gap distribution health check (Hardy-Littlewood histogram).
 */

#include "../new_src/gap_dist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_failures = 0;

static void check(int cond, const char *what) {
    if (cond) {
        printf("  OK  %s\n", what);
    } else {
        printf("  FAIL %s\n", what);
        g_failures++;
    }
}

static void check_double(double got, double want, double tol, const char *what) {
    int ok = fabs(got - want) <= tol;
    printf("  %s  %s (got %.6f, want %.6f)\n", ok ? "OK " : "FAIL", what, got, want);
    if (!ok) g_failures++;
}

int main(void) {
    printf("[TEST] gap_dist Hardy-Littlewood factors\n");
    check_double(gap_dist_hl_factor(2), 1.0, 1e-9, "HL(2)");
    check_double(gap_dist_hl_factor(4), 1.0, 1e-9, "HL(4)");
    check_double(gap_dist_hl_factor(6), 2.0, 1e-9, "HL(6)");
    check_double(gap_dist_hl_factor(8), 1.0, 1e-9, "HL(8)");
    check_double(gap_dist_hl_factor(10), 4.0 / 3.0, 1e-9, "HL(10)");
    check_double(gap_dist_hl_factor(12), 2.0, 1e-9, "HL(12)");
    check_double(gap_dist_hl_factor(14), 6.0 / 5.0, 1e-9, "HL(14)");
    check_double(gap_dist_hl_factor(30), 8.0 / 3.0, 1e-9, "HL(30)");
    check_double(gap_dist_hl_factor(210), 3.2, 1e-9, "HL(210)");
    check_double(gap_dist_hl_factor(1), 0.0, 1e-12, "HL(1) = 0");
    check_double(gap_dist_hl_factor(3), 0.0, 1e-12, "HL(odd) = 0");

    printf("[TEST] gap_dist health roundtrip (synthetic HL-consistent histogram)\n");
    gap_dist_reset();
    gap_dist_set_logbase(530.3);   /* shift 509: ln(2^765) */
    check_double(gap_dist_get_logbase(), 530.3, 1e-9, "logbase roundtrip");

    const uint32_t gs[6] = { 2, 4, 6, 8, 10, 12 };
    uint64_t base_count = 100000;
    for (int i = 0; i < 6; i++) {
        double expected = (double)base_count * gap_dist_hl_factor(gs[i]) *
                          exp(-(double)(gs[i] - 2) / 530.3);
        uint64_t n = (uint64_t)llround(expected);
        for (uint64_t k = 0; k < n; k++) {
            gap_dist_accumulate(gs[i]);
        }
    }

    struct gap_dist_health h;
    gap_dist_health(&h);
    printf("  total=%llu enough=%d\n", (unsigned long long)h.total_gaps,
           h.enough_samples);
    check(h.enough_samples == 1, "enough_samples after 50k+ gaps");
    for (int i = 0; i < 5; i++) {
        char what[64];
        snprintf(what, sizeof(what), "dev g=%u near zero", gs[i + 1]);
        check(fabs(h.dev_pct[i]) < 0.5, what);
    }
    check(h.max_abs_dev < 0.5, "max_abs_dev near zero on clean histogram");

    printf("[TEST] gap_dist detects a skewed distribution\n");
    gap_dist_reset();
    for (uint64_t k = 0; k < 100000; k++) gap_dist_accumulate(2);
    for (uint64_t k = 0; k < 100000; k++) gap_dist_accumulate(4);
    for (uint64_t k = 0; k < 100000; k++) gap_dist_accumulate(6); /* HL wants ~2x */
    for (uint64_t k = 0; k < 100000; k++) gap_dist_accumulate(8);
    for (uint64_t k = 0; k < 133333; k++) gap_dist_accumulate(10); /* HL ~4/3x */
    for (uint64_t k = 0; k < 200000; k++) gap_dist_accumulate(12); /* HL ~2x */
    gap_dist_health(&h);
    /* All gaps except g=6 are HL-consistent (small dev); g=6 is at half its
       expected frequency, so it must be flagged as the worst deviation. */
    check(h.enough_samples == 1, "enough_samples on skewed histogram");
    check(fabs(h.dev_pct[1] + 50.0) < 2.0, "dev g=6 ~ -50% detected");
    check(h.max_abs_dev > GAP_DIST_WARN_PCT, "warning threshold crossed");
    check(h.worst_g == 6, "worst_g = 6");

    printf("[TEST] gap_dist overflow bucket\n");
    gap_dist_reset();
    gap_dist_accumulate(2);
    gap_dist_accumulate(1000000);   /* > 2*(GAP_DIST_BUCKETS-1) */
    check(gap_dist_total() == 2, "overflow gap counted in total");

    printf("[TEST] gap_dist covered-region exclusion\n");
    gap_dist_set_excluded(1000, 2000);
    check(gap_dist_offset_excluded(500) == 0, "offset below exclusion range");
    check(gap_dist_offset_excluded(1500) == 1, "offset inside exclusion range");
    check(gap_dist_offset_excluded(2000) == 0, "offset at hi boundary");
    check(gap_dist_offset_excluded(5000) == 0, "offset above exclusion range");
    gap_dist_set_excluded(0, 0);
    check(gap_dist_offset_excluded(1500) == 0, "exclusion disabled (lo>=hi)");

    if (g_failures == 0) {
        printf("\nAll gap_dist tests PASSED\n");
        return 0;
    }
    printf("\n%d gap_dist test(s) FAILED\n", g_failures);
    return 1;
}
