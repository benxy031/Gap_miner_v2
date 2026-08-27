/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gap Detection: Find consecutive prime pairs and compute merit
 *
 * Gap = pair of consecutive Euler-positive candidates
 * with length = offset[i+1] - offset[i] and merit = length / ln(base)
 */

#ifndef GAP_DETECTION_H
#define GAP_DETECTION_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Gap descriptor: consecutive prime pair with computed merit */
struct gap_result {
    uint64_t offset_p1;          /* First prime offset from base */
    uint64_t offset_p2;          /* Second prime offset from base */
    uint32_t gap_length;         /* offset_p2 - offset_p1 */
    double merit;                /* gap_length / ln(actual_value) */
    uint8_t verified;            /* 1 if BPSW verified, 0 if tentative */
};

struct gap_scan_stats {
    uint64_t euler_pairs;
    uint32_t max_gap_length;
    double max_merit;
};

/* Gap detection: retain the previous Euler-positive offset while scanning. */
int gap_detection_find(
    const uint8_t *is_prime,     /* Array of primality results */
    const uint64_t *offsets,     /* Sieve offset array */
    uint32_t count,              /* Number of candidates */
    uint32_t shift,              /* Retained for API compatibility; base already includes it */
    mpz_t base,                  /* Fully shifted start value for this sieve window */
    double merit_threshold,      /* Minimum merit to report */
    uint64_t owned_offset_limit, /* Report only gaps whose first endpoint is owned */
    struct gap_scan_stats *scan_stats, /* Optional counters for owned Euler pairs */
    struct gap_result **out_gaps,     /* Output array (allocate) */
    uint32_t *out_count         /* Number of gaps found */
);

/* Compute merit for single gap */
double gap_detection_compute_merit(uint32_t gap_length, mpz_t gap_start);

/* Cleanup gap results array */
void gap_detection_free_results(struct gap_result *gaps);

#endif /* GAP_DETECTION_H */
