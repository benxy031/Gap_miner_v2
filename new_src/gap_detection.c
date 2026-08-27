/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gap Detection Implementation: Find consecutive prime pairs
 */

#include "gap_detection.h"
#include "gap_dist.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ln(x) computed WITHOUT mpz_get_d: for x > ~2^1024 (shifts >= ~770)
   mpz_get_d overflows to +inf and every merit silently becomes 0.  Compute
   log2 from the top 53 bits: log2(x) = (bits-53) + log2(top53).  Exact to
   double precision for any size. */
static double mpz_ln(const mpz_t x) {
    if (mpz_sgn(x) <= 0) return 0.0;
    size_t bits = mpz_sizeinbase(x, 2);
    if (bits <= 53) {
        return log(mpz_get_d(x));
    }
    mpz_t top;
    mpz_init(top);
    mpz_tdiv_q_2exp(top, x, (mp_bitcnt_t)(bits - 53));
    double log2v = log2(mpz_get_d(top)) + (double)(bits - 53);
    mpz_clear(top);
    return log2v * 0.69314718055994530942;
}

int gap_detection_find(
    const uint8_t *is_prime,
    const uint64_t *offsets,
    uint32_t count,
    uint32_t shift,
    mpz_t base,
    double merit_threshold,
    uint64_t owned_offset_limit,
    struct gap_scan_stats *scan_stats,
    struct gap_result **out_gaps,
    uint32_t *out_count) {
    
    if (scan_stats) memset(scan_stats, 0, sizeof(*scan_stats));
    if (!is_prime || !offsets || !out_gaps || !out_count || count == 0) {
        if (out_count) *out_count = 0;
        if (out_gaps) *out_gaps = NULL;
        return 0;
    }
    *out_gaps = NULL;
    *out_count = 0;
    if (count < 2) return 1;
    
    /* Allocate gap array (worst case: count-1 gaps) */
    struct gap_result *gaps = (struct gap_result *)malloc((count - 1) * sizeof(struct gap_result));
    if (!gaps) {
        *out_count = 0;
        *out_gaps = NULL;
        return 0;
    }
    
    uint32_t gap_count = 0;
    uint64_t previous_offset = 0;
    int have_previous = 0;
    (void)shift;
    mpz_t gap_value;
    mpz_init(gap_value);
    
    /* Consecutive Euler positives need not be adjacent sieve survivors. */
    for (uint32_t i = 0; i < count; i++) {
        if (!is_prime[i]) continue;

        uint64_t offset_p2 = offsets[i];
        if (have_previous && offset_p2 > previous_offset) {
            uint64_t offset_p1 = previous_offset;
            uint32_t gap_length = (uint32_t)(offset_p2 - offset_p1);

            /* Health check: feed the Hardy-Littlewood histogram with every
               consecutive-prime gap found in the scanned region.  Ownership
               only controls which gaps are REPORTED; the gap-length
               distribution is the same either way, so accumulating all pairs
               avoids an edge bias at owned_offset_limit.  In CRT mode both
               endpoints must sit outside the covered region: the covering
               conditions the distribution inside it, and a pair crossing the
               covered boundary (last prime before the aligned base to the
               first covering-conditioned prime after it) is biased too. */
            if (!gap_dist_offset_excluded(offset_p1) &&
                !gap_dist_offset_excluded(offset_p2)) {
                gap_dist_accumulate(gap_length);
            }

            if (previous_offset < owned_offset_limit) {
                /* base already equals (h256 << shift) + window_start. */
                mpz_set(gap_value, base);
                mpz_add_ui(gap_value, gap_value, offset_p1);

                double log_actual = mpz_ln(gap_value);
                if (log_actual <= 0.0) {
                    previous_offset = offset_p2;
                    continue;
                }

                double merit = (double)gap_length / log_actual;
                if (scan_stats) {
                    scan_stats->euler_pairs++;
                    if (gap_length > scan_stats->max_gap_length) {
                        scan_stats->max_gap_length = gap_length;
                    }
                    if (merit > scan_stats->max_merit) {
                        scan_stats->max_merit = merit;
                    }
                }
                
                /* Record if merit exceeds threshold */
                if (merit >= merit_threshold) {
                    struct gap_result *res = &gaps[gap_count];
                    res->offset_p1 = offset_p1;
                    res->offset_p2 = offset_p2;
                    res->gap_length = gap_length;
                    res->merit = merit;
                    res->verified = 0;  /* Not yet BPSW verified */
                    gap_count++;
                }
            }
        }
        previous_offset = offset_p2;
        have_previous = 1;
    }
    mpz_clear(gap_value);
    
    /* Resize results to actual count */
    if (gap_count == 0) {
        free(gaps);
        *out_gaps = NULL;
        *out_count = 0;
        return 1;  /* Success, just no gaps found */
    }
    
    *out_gaps = gaps;
    *out_count = gap_count;
    
    return 1;
}

double gap_detection_compute_merit(uint32_t gap_length, mpz_t gap_start) {
    if (gap_length == 0) return 0.0;

    double log_start = mpz_ln(gap_start);
    if (log_start <= 0.0) return 0.0;

    /* Merit = gap_length / ln(gap_start) */
    return (double)gap_length / log_start;
}

void gap_detection_free_results(struct gap_result *gaps) {
    if (gaps) {
        free(gaps);
    }
}
