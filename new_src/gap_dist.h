/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gap distribution health check.
 *
 * Accumulates a histogram of the gaps between consecutive probable primes
 * found in every scanned window and compares the small-gap frequencies to the
 * Hardy-Littlewood k-tuple asymptotic model:
 *
 *     P(g) / P(2) = prod_{odd prime p | g} (p-1)/(p-2)  *  exp(-(g-2)/log(n))
 *
 * If the observed histogram deviates from this model, something in the
 * sieve/extract/primality pipeline is wrong (e.g. composites not marked,
 * primes falsely rejected, or candidate offsets corrupted).  This is a
 * running correctness alarm, not a performance feature.
 *
 * Thread-safety: every worker accumulates into the same padded atomic buckets
 * (one cache line per bucket); no mutex.  The health report is only read from
 * the main thread at stats time.
 */

#ifndef GAP_DIST_H
#define GAP_DIST_H

#include <stdint.h>

/* Bucket i counts gaps of length 2*i.  The last bucket is the overflow bin
   for every gap >= 2*(GAP_DIST_BUCKETS-1).  Only small gaps are used for the
   health check; the rest of the histogram exists for diagnostics. */
#define GAP_DIST_BUCKETS 512

/* Minimum number of observed gaps before deviations are considered
   statistically meaningful. */
#define GAP_DIST_MIN_SAMPLES 50000ULL

/* Minimum number of gap-2 pairs (the reference class) for the warning:
   the small-gap ratio noise is ~sqrt(2/c2). */
#define GAP_DIST_MIN_C2 1000ULL

/* Deviation threshold (percent) that triggers a console warning.
   Non-CRT scans match the Hardy-Littlewood model to ~1%, so 20% is a safe
   alarm.  CRT coverings modulate the small-gap frequencies by up to ~30%
   even in uncovered regions (empirically measured), so CRT mode uses a
   wider threshold. */
#define GAP_DIST_WARN_PCT 20.0
#define GAP_DIST_WARN_PCT_CRT 50.0

struct gap_dist_health {
    uint64_t total_gaps;        /* Total gaps accumulated (incl. overflow) */
    double   logbase;           /* ln(magnitude) used for the HL correction */
    /* Raw counts for g = 2, 4, 6, 8, 10, 12 (indices 0..5). */
    uint64_t hist_small[6];
    /* For g = 4, 6, 8, 10, 12 (indices 0..4):
       obs_ratio = hist[g]/hist[2], exp_ratio = HL prediction,
       dev_pct = 100*(obs/exp - 1).  Meaningful only when enough_samples. */
    double   obs_ratio[5];
    double   exp_ratio[5];
    double   dev_pct[5];
    double   max_abs_dev;       /* Largest |dev_pct| among g=4..12 */
    double   worst_dev;         /* Signed dev_pct of worst_g */
    int      worst_g;           /* Gap length with the largest |dev|, 0=none */
    int      enough_samples;    /* total_gaps >= MIN_SAMPLES && c2 >= MIN_C2 */
};

/* Enable/disable accumulation.  HALF_CLASS mode disables it (visible-class
   gaps are not consecutive-prime gaps, so the histogram would be garbage). */
void gap_dist_set_enabled(int enabled);

int gap_dist_enabled(void);

/* Set the magnitude for the HL density correction: ln(n) = (256+shift)*ln 2.
   Called once at startup (main thread) before workers begin. */
void gap_dist_set_logbase(double logbase);

double gap_dist_get_logbase(void);

/* Restrict accumulation to gaps whose FIRST endpoint lies outside [lo, hi).
   In CRT mode the covering conditions the gap distribution inside the covered
   region (offset range [back_limit, back_limit+gap_target) relative to the
   window base), so only gaps outside it follow the unconditioned
   Hardy-Littlewood model.  lo >= hi disables the filter (non-CRT mode). */
void gap_dist_set_excluded(uint64_t lo, uint64_t hi);

/* 1 when the given p1 offset falls inside the excluded (covered) region. */
int gap_dist_offset_excluded(uint64_t offset);

/* Accumulate one gap of the given length (must be even).  Hot path: one
   relaxed atomic increment. */
void gap_dist_accumulate(uint32_t gap_length);

uint64_t gap_dist_total(void);

/* Snapshot the histogram and compute the health report.  Main thread only. */
void gap_dist_health(struct gap_dist_health *out);

/* Hardy-Littlewood gap factor: prod over odd prime divisors p of g of
   (p-1)/(p-2).  Returns 0 for odd or tiny gaps. */
double gap_dist_hl_factor(uint32_t g);

/* Zero all buckets (startup / test use). */
void gap_dist_reset(void);

/* Copy the full bucket counts to out[0..GAP_DIST_BUCKETS-1] (diagnostics). */
void gap_dist_snapshot(uint64_t *out);

#endif /* GAP_DIST_H */
