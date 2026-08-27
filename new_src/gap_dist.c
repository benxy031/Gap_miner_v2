/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gap distribution health check implementation.
 * See gap_dist.h for the model and the failure semantics.
 */

#include "gap_dist.h"
#include <math.h>
#include <stdatomic.h>
#include <string.h>

/* One padded bucket per cache line: 8 workers increment distinct and nearby
   buckets without false sharing between buckets. */
struct gap_bucket {
    _Atomic uint64_t count;
    uint8_t pad[56];
};

static struct gap_bucket g_hist[GAP_DIST_BUCKETS];

/* Written once by main before workers start; only the main thread reads it
   (inside gap_dist_health), so no synchronization is needed. */
static double g_logbase = 0.0;

/* HALF_CLASS mode disables accumulation: visible-class gaps are not true
   consecutive-prime gaps, so the HL histogram would be garbage. */
static int g_enabled = 1;

void gap_dist_set_enabled(int enabled) {
    g_enabled = enabled ? 1 : 0;
}

int gap_dist_enabled(void) {
    return g_enabled;
}

void gap_dist_set_logbase(double logbase) {
    if (logbase > 0.0) g_logbase = logbase;
}

double gap_dist_get_logbase(void) {
    return g_logbase;
}

/* Covered-region exclusion (CRT mode).  Written once by main before workers
   start; read from the hot path, so it must be set before mining begins. */
static uint64_t g_excl_lo = 0;
static uint64_t g_excl_hi = 0;

void gap_dist_set_excluded(uint64_t lo, uint64_t hi) {
    if (hi <= lo) {
        g_excl_lo = 0;
        g_excl_hi = 0;
    } else {
        g_excl_lo = lo;
        g_excl_hi = hi;
    }
}

int gap_dist_offset_excluded(uint64_t offset) {
    return (g_excl_hi > g_excl_lo &&
            offset >= g_excl_lo && offset < g_excl_hi) ? 1 : 0;
}

void gap_dist_accumulate(uint32_t gap_length) {
    if (!g_enabled) return;
    uint64_t g = gap_length;
    if (g == 0) return;                 /* impossible for odd-prime gaps */
    uint64_t idx = g >> 1;              /* bucket i = gap 2*i */
    if (idx >= GAP_DIST_BUCKETS) idx = GAP_DIST_BUCKETS - 1;
    atomic_fetch_add_explicit(&g_hist[idx].count, 1, memory_order_relaxed);
}

uint64_t gap_dist_total(void) {
    uint64_t total = 0;
    for (int i = 0; i < GAP_DIST_BUCKETS; i++) {
        total += atomic_load_explicit(&g_hist[i].count, memory_order_relaxed);
    }
    return total;
}

double gap_dist_hl_factor(uint32_t g) {
    if (g < 2 || (g & 1U) != 0) return 0.0;  /* only even gaps have a ratio */
    double f = 1.0;
    uint32_t n = g;
    while ((n & 1U) == 0) n >>= 1;           /* drop the factor 2 */
    for (uint32_t p = 3; (uint64_t)p * p <= n; p += 2) {
        if (n % p == 0) {
            f *= (double)(p - 1) / (double)(p - 2);
            while (n % p == 0) n /= p;
        }
    }
    if (n > 1) f *= (double)(n - 1) / (double)(n - 2);
    return f;
}

void gap_dist_health(struct gap_dist_health *out) {
    memset(out, 0, sizeof(*out));
    out->logbase = g_logbase;
    out->worst_g = 0;

    uint64_t snap[GAP_DIST_BUCKETS];
    uint64_t total = 0;
    for (int i = 0; i < GAP_DIST_BUCKETS; i++) {
        snap[i] = atomic_load_explicit(&g_hist[i].count, memory_order_relaxed);
        total += snap[i];
    }
    out->total_gaps = total;

    for (int i = 0; i < 6; i++) {
        out->hist_small[i] = snap[i + 1];   /* g = 2,4,6,8,10,12 */
    }

    uint64_t c2 = snap[1];   /* gaps of length 2 = reference class */
    out->enough_samples = (total >= GAP_DIST_MIN_SAMPLES &&
                           c2 >= GAP_DIST_MIN_C2) ? 1 : 0;

    if (c2 == 0 || g_logbase <= 0.0) return;

    static const uint32_t gs[5] = { 4, 6, 8, 10, 12 };
    for (int i = 0; i < 5; i++) {
        uint32_t g = gs[i];
        double obs = (double)snap[g >> 1] / (double)c2;
        double exp_ratio = gap_dist_hl_factor(g) *
                           exp(-(double)(g - 2) / g_logbase);
        out->obs_ratio[i] = obs;
        out->exp_ratio[i] = exp_ratio;
        if (exp_ratio > 0.0) {
            double dev = 100.0 * (obs / exp_ratio - 1.0);
            out->dev_pct[i] = dev;
            if (fabs(dev) > out->max_abs_dev) {
                out->max_abs_dev = fabs(dev);
                out->worst_dev = dev;
                out->worst_g = (int)g;
            }
        }
    }
}

void gap_dist_reset(void) {
    for (int i = 0; i < GAP_DIST_BUCKETS; i++) {
        atomic_store_explicit(&g_hist[i].count, 0, memory_order_relaxed);
    }
}

void gap_dist_snapshot(uint64_t *out) {
    if (!out) return;
    for (int i = 0; i < GAP_DIST_BUCKETS; i++) {
        out[i] = atomic_load_explicit(&g_hist[i].count, memory_order_relaxed);
    }
}
