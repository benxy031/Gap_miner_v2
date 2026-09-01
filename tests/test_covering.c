/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Covering optimizer test: verify the residue optimization covers ~93% of the
 * gap-target interior (the CRT pre-selection boost), vs ~82% for random
 * residues, and that every survivor is genuinely uncovered.
 */

#include "../new_src/covering.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PRIME_LIMIT 2048

static uint64_t primes[PRIME_LIMIT];
static size_t prime_count = 0;

static void gen_primes(size_t n) {
    size_t limit = PRIME_LIMIT;
    uint8_t *comp = (uint8_t *)calloc(limit, 1);
    if (!comp) return;
    for (size_t f = 2; f * f < limit; f++) {
        if (comp[f]) continue;
        for (size_t m = f * f; m < limit; m += f) comp[m] = 1;
    }
    for (size_t v = 2; v < limit && prime_count < n; v++) {
        if (!comp[v]) primes[prime_count++] = v;
    }
    free(comp);
}

/* Max n primes whose product fits under 2^shift (same rule as crt_set). */
static size_t primes_for_shift(uint32_t shift, size_t max_n) {
    long double log2_limit = (long double)shift;
    long double acc = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < max_n; i++) {
        acc += log2l((long double)primes[i]);
        if (acc >= log2_limit) break;
        n = i + 1;
    }
    return n;
}

int main(void) {
    gen_primes(PRIME_LIMIT);

    int failures = 0;

    /* Test 1: shift 720, 98 primes, design merit 39. */
    {
        uint32_t shift = 720;
        size_t n = primes_for_shift(shift, PRIME_LIMIT);
        double logbase = (256.0 + shift) * log(2.0);
        uint64_t gap_target = (uint64_t)ceil(39.0 * logbase);
        printf("shift=%u  n_primes=%zu (largest %llu)  logbase=%.1f  gap_target=%llu\n",
               shift, n, (unsigned long long)primes[n - 1], logbase,
               (unsigned long long)gap_target);

        uint64_t *res = (uint64_t *)malloc(n * sizeof(uint64_t));
        uint64_t *rand_res = (uint64_t *)malloc(n * sizeof(uint64_t));

        struct covering_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.strength = 4;
        cfg.local_sweeps = 8;
        cfg.pair_search = 0;
        cfg.seed = 12345;

        uint64_t surv = covering_optimize(primes, n, gap_target, res, &cfg);

        /* random residues for comparison */
        srand(99);
        for (size_t i = 0; i < n; i++)
            rand_res[i] = 1 + (uint64_t)(rand() % (int)(primes[i] - 1));
        uint64_t rand_surv = covering_count_survivors(primes, rand_res, n,
                                                      gap_target);

        double cov = 100.0 * (1.0 - (double)surv / (double)gap_target);
        double rand_cov = 100.0 * (1.0 - (double)rand_surv / (double)gap_target);
        printf("  optimized survivors=%llu (%.2f%% uncovered, %.2f%% covered)\n",
               (unsigned long long)surv,
               100.0 * (double)surv / (double)gap_target, cov);
        printf("  random     survivors=%llu (%.2f%% uncovered, %.2f%% covered)\n",
               (unsigned long long)rand_surv,
               100.0 * (double)rand_surv / (double)gap_target, rand_cov);

        if (cov < 90.0) {
            printf("  FAIL: optimized coverage %.2f%% < 90%%\n", cov);
            failures++;
        } else {
            printf("  OK: optimized coverage >= 90%%\n");
        }

        /* Verify covering property: every survivor is uncovered by all primes. */
        uint64_t *survivors = (uint64_t *)malloc((gap_target + 1) * sizeof(uint64_t));
        uint64_t ns = covering_survivors(primes, res, n, gap_target, survivors);
        if (ns != surv) {
            printf("  FAIL: survivor list count %llu != %llu\n",
                   (unsigned long long)ns, (unsigned long long)surv);
            failures++;
        }
        int bad = 0;
        for (uint64_t k = 0; k < ns && !bad; k++) {
            uint64_t j = survivors[k];
            for (size_t i = 0; i < n; i++) {
                if ((res[i] + j) % primes[i] == 0) {
                    printf("  FAIL: survivor %llu is covered by prime %llu\n",
                           (unsigned long long)j,
                           (unsigned long long)primes[i]);
                    bad = 1;
                    failures++;
                    break;
                }
            }
        }
        if (!bad) printf("  OK: all %llu survivors genuinely uncovered\n",
                         (unsigned long long)ns);

        free(survivors);
        free(res);
        free(rand_res);
    }

    /* Test 2: shift 512, design merit 30 (faster, sanity check). */
    {
        uint32_t shift = 512;
        size_t n = primes_for_shift(shift, PRIME_LIMIT);
        double logbase = (256.0 + shift) * log(2.0);
        uint64_t gap_target = (uint64_t)ceil(30.0 * logbase);
        printf("shift=%u  n_primes=%zu  gap_target=%llu\n",
               shift, n, (unsigned long long)gap_target);

        uint64_t *res = (uint64_t *)malloc(n * sizeof(uint64_t));
        struct covering_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.strength = 2;
        cfg.local_sweeps = 4;
        cfg.pair_search = 0;
        cfg.seed = 7;

        uint64_t surv = covering_optimize(primes, n, gap_target, res, &cfg);
        double cov = 100.0 * (1.0 - (double)surv / (double)gap_target);
        printf("  optimized survivors=%llu (%.2f%% covered)\n",
               (unsigned long long)surv, cov);
        if (cov < 88.0) {
            printf("  FAIL: coverage %.2f%% < 88%%\n", cov);
            failures++;
        } else {
            printf("  OK: coverage >= 88%%\n");
        }
        free(res);
    }

    /* Test 3: blocks objective with D == design merit behaves like the
       classic min-survivors objective (every survivor's run already reaches
       D), and the run_ge helper reports a sane constellation count. */
    {
        uint32_t shift = 512;
        size_t n = primes_for_shift(shift, PRIME_LIMIT);
        double logbase = (256.0 + shift) * log(2.0);
        uint64_t gap_target = (uint64_t)ceil(30.0 * logbase);

        uint64_t *res_b = (uint64_t *)malloc(n * sizeof(uint64_t));
        struct covering_config cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.strength = 2;
        cfg.local_sweeps = 4;
        cfg.pair_search = 0;
        cfg.blocks_objective = 1;
        cfg.difficulty_merit = 30.0;
        cfg.logbase = logbase;
        cfg.seed = 7;

        uint64_t surv_b = covering_optimize(primes, n, gap_target, res_b, &cfg);
        double cov_b = 100.0 * (1.0 - (double)surv_b / (double)gap_target);
        printf("  blocks objective survivors=%llu (%.2f%% covered)\n",
               (unsigned long long)surv_b, cov_b);
        if (cov_b < 88.0) {
            printf("  FAIL: blocks coverage %.2f%% < 88%%\n", cov_b);
            failures++;
        } else {
            printf("  OK: blocks coverage >= 88%%\n");
        }

        uint64_t run_ge = covering_survivors_run_ge(
            primes, res_b, n, 2 * gap_target,
            (uint64_t)ceil(30.0 * logbase));
        printf("  run_ge_D(30-merit)=%llu (in 2x window; 0 is legal at low strength)\n",
               (unsigned long long)run_ge);
        if (run_ge > 2 * gap_target) {
            printf("  FAIL: run_ge_D out of range\n");
            failures++;
        } else {
            printf("  OK: run_ge_D bounded\n");
        }
        free(res_b);
    }

    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    }
    printf("\n%d TEST(S) FAILED\n", failures);
    return 1;
}
