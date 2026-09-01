/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Runtime run-profile tool: load a CRT file with the SAME code the miner
 * uses (crt_runtime), then print the survivor structure of the runtime
 * template: candidate count, survivor spacing stats, and the run-length
 * profile by merit thresholds.  This measures the ACTUAL constellation the
 * miner sees (parity-adjusted odd-slot template), not the raw residue
 * bitmap of the generator.
 *
 * Usage: test_run_profile <crt-file> [logbase]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

#include "../new_src/crt_runtime.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <crt-file> [logbase]\n", argv[0]);
        return 1;
    }
    struct crt_runtime rt;
    if (!crt_runtime_load(&rt, argv[1])) {
        fprintf(stderr, "load failed: %s\n", argv[1]);
        return 1;
    }

    double logbase = argc > 2 ? atof(argv[2])
                              : (256.0 + (double)rt.shift) * log(2.0);

    uint64_t *surv = (uint64_t *)malloc((size_t)(rt.window + 8) *
                                        sizeof(uint64_t));
    if (!surv) {
        crt_runtime_free(&rt);
        return 1;
    }
    uint64_t n = crt_runtime_survivors(&rt, surv);

    /* Forward runs in slot units: slots are odd offsets 2s+1; a "run" is
       the number of consecutive marked slots after a survivor, so the
       composite run in OFFSET units is 2*slots (even offsets are trivially
       composite). */
    const uint8_t *bits = rt.template;
    uint64_t odd_slots = rt.window / 2;

    uint64_t run_min = UINT64_MAX, run_max = 0, run_sum = 0;
    uint64_t buckets[8] = {0};
    double bm[8] = {24, 26, 28, 30, 32, 35, 38, 40};
    char bname[8][8] = {"24", "26", "28", "30", "32", "35", "38", "40"};

    for (uint64_t i = 0; i < n; i++) {
        uint64_t s = (surv[i] - 1ULL) >> 1;   /* survivor slot index */
        uint64_t t = s + 1;
        uint64_t run_slots = 0;
        while (t < odd_slots &&
               ((bits[t >> 3] >> (t & 7U)) & 1U)) {
            t++;
            run_slots++;
        }
        uint64_t run_off = run_slots * 2ULL;
        if (run_off < run_min) run_min = run_off;
        if (run_off > run_max) run_max = run_off;
        run_sum += run_off;
        for (int b = 0; b < 8; b++) {
            if ((double)run_off >= bm[b] * logbase)
                buckets[b]++;
        }
    }

    printf("file=%s n_primes=%u shift=%u gap_target=%llu window=%llu "
           "logbase=%.1f\n",
           argv[1], rt.n_primes, rt.shift,
           (unsigned long long)rt.gap_target, (unsigned long long)rt.window,
           logbase);
    printf("n_survivors=%llu (%.2f%% of odd slots)\n",
           (unsigned long long)n, 100.0 * (double)n / (double)(rt.window / 2));
    if (n > 0) {
        /* spacing between consecutive survivors */
        uint64_t sp_min = UINT64_MAX, sp_max = 0, sp_sum = 0;
        for (uint64_t i = 1; i < n; i++) {
            uint64_t d = surv[i] - surv[i - 1];
            if (d < sp_min) sp_min = d;
            if (d > sp_max) sp_max = d;
            sp_sum += d;
        }
        printf("spacing: min=%llu max=%llu avg=%.1f (offsets)\n",
               (unsigned long long)sp_min, (unsigned long long)sp_max,
               (double)sp_sum / (double)(n - 1));
        printf("run (offsets): min=%llu max=%llu avg=%.1f\n",
               (unsigned long long)run_min, (unsigned long long)run_max,
               (double)run_sum / (double)n);
        printf("run histogram (survivors with run >= X-merit):\n");
        for (int b = 0; b < 8; b++) {
            printf("  >= %s-merit (%6.0f): %llu\n", bname[b],
                   bm[b] * logbase, (unsigned long long)buckets[b]);
        }
    }
    free(surv);
    crt_runtime_free(&rt);
    return 0;
}
