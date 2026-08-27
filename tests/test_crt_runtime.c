/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CRT runtime test: generate a covering, write a CRT file, load it, verify the
 * template coverage and the CRT alignment, and confirm the aligned window is
 * prime-poor relative to the natural density (the covering effect).
 */

#include "../new_src/covering.h"
#include "../new_src/crt_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

#define MAX_PRIMES 128

static uint64_t primes[MAX_PRIMES];
static size_t prime_count = 0;

static void gen_primes(void) {
    int limit = 1200;
    char *comp = (char *)calloc((size_t)limit, 1);
    for (int f = 2; f * f < limit; f++)
        if (!comp[f])
            for (int m = f * f; m < limit; m += f)
                comp[m] = 1;
    for (int v = 2; v < limit && prime_count < MAX_PRIMES; v++)
        if (!comp[v])
            primes[prime_count++] = (uint64_t)v;
    free(comp);
}

static double primorial_log2(size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++)
        acc += log2((double)primes[i]);
    return acc;
}

int main(void) {
    gen_primes();

    int failures = 0;
    size_t n = 98;               /* 98 primes: 2..521 (shift-720 design) */
    double merit = 39.0;
    int ctr_bits = 8;
    int shift = (int)ceil(primorial_log2(n)) + ctr_bits;
    uint64_t gap_target = (uint64_t)ceil(merit * (256.0 + shift) * log(2.0));

    uint64_t residues[MAX_PRIMES];
    struct covering_config cfg;
    cfg.strength = 8;
    cfg.local_sweeps = 8;
    cfg.ils_rounds = 0;
    cfg.pair_search = 0;
    cfg.seed = 20260823;

    uint64_t candidates = covering_optimize(primes, n, gap_target, residues, &cfg);

    /* Write a temp CRT file in the gen_crt text format. */
    const char *tmp = "/tmp/test_crt_runtime_crt.txt";
    FILE *f = fopen(tmp, "w");
    if (!f) {
        printf("FAIL: cannot open temp file\n");
        return 1;
    }
    fprintf(f, "# test crt file\n");
    fprintf(f, "n_primes %zu\n", n);
    fprintf(f, "merit %.2f\n", merit);
    fprintf(f, "shift %d\n", shift);
    fprintf(f, "gap_target %llu\n", (unsigned long long)gap_target);
    fprintf(f, "n_candidates %llu\n", (unsigned long long)candidates);
    for (size_t i = 0; i < n; i++) {
        uint64_t p = primes[i];
        uint64_t off = (p - residues[i]) % p;
        fprintf(f, "%llu %llu\n", (unsigned long long)p, (unsigned long long)off);
    }
    fclose(f);

    struct crt_runtime rt;
    if (!crt_runtime_load(&rt, tmp)) {
        printf("FAIL: crt_runtime_load\n");
        return 1;
    }

    printf("n_primes=%u shift=%u gap_target=%llu window=%llu\n",
           rt.n_primes, rt.shift, (unsigned long long)rt.gap_target,
           (unsigned long long)rt.window);
    printf("file n_candidates=%llu  template survivors=%llu\n",
           (unsigned long long)candidates, (unsigned long long)rt.n_survivors);

    /* Mining scans a window of 2 * needed_gap at the MINING merit (lower than
     * the file's design merit).  The covering concentrates coverage inside
     * [1, gap_target), so the prime-poor effect only shows at that width. */
    {
        double logbase = (256.0 + shift) * log(2.0);
        uint64_t needed_gap = (uint64_t)ceil(22.0 * logbase);
        uint64_t scan_window = 2 * needed_gap;
        if (!crt_runtime_set_window(&rt, scan_window)) {
            printf("FAIL: crt_runtime_set_window\n");
            failures++;
        }
        printf("mining window=%llu (needed_gap=%llu, merit 22)\n",
               (unsigned long long)scan_window, (unsigned long long)needed_gap);
    }

    /* The template survivors in [1, gap_target) must match n_candidates. */
    {
        uint64_t *surv0 = (uint64_t *)malloc((rt.window + 1) * sizeof(uint64_t));
        uint64_t ns0 = crt_runtime_survivors(&rt, surv0);
        uint64_t in_gap = 0;
        for (uint64_t k = 0; k < ns0; k++)
            if (surv0[k] < rt.gap_target)
                in_gap++;
        free(surv0);

        printf("  survivors in [1, gap_target) = %llu (file says %llu)\n",
               (unsigned long long)in_gap, (unsigned long long)candidates);
        if (in_gap > candidates + 2 || in_gap + 2 < candidates) {
            printf("FAIL: gap-range survivors %llu vs candidates %llu\n",
                   (unsigned long long)in_gap,
                   (unsigned long long)candidates);
            failures++;
        } else {
            printf("OK: gap-range survivors match n_candidates\n");
        }
    }

    /* Align a synthetic header and verify the alignment property. */
    uint8_t h256[32];
    for (int i = 0; i < 32; i++)
        h256[i] = (uint8_t)(i * 37 + 11);

    mpz_t nadd0, base0, cand;
    mpz_inits(nadd0, base0, cand, NULL);
    if (crt_runtime_align(nadd0, h256, (uint32_t)shift, &rt) != 0) {
        printf("FAIL: crt_runtime_align\n");
        failures++;
    } else {
        mpz_set_ui(base0, 0);
        for (int i = 0; i < 32; i++) {
            mpz_mul_2exp(base0, base0, 8);
            mpz_add_ui(base0, base0, h256[i]);
        }
        mpz_mul_2exp(base0, base0, (unsigned long)shift);
        mpz_add(cand, base0, nadd0);
        if (mpz_odd_p(cand))
            mpz_sub_ui(cand, cand, 1);   /* parity adjustment (rt.adj) */

        int aligned = 1;
        for (uint32_t i = 0; i < rt.n_primes; i++) {
            if (rt.offsets[i] == 0)
                continue;
            uint64_t p = rt.primes[i];
            uint64_t expect = (p - ((rt.offsets[i] + rt.adj) % p)) % p;
            if (mpz_fdiv_ui(cand, p) != expect) {
                aligned = 0;
                break;
            }
        }
        if (!aligned) {
            printf("FAIL: alignment property violated\n");
            failures++;
        } else {
            printf("OK: CRT alignment correct (cand ≡ -(offset+adj) mod p)\n");
        }
    }

    /* Count primes in the aligned window (odd offsets) vs natural density. */
    uint64_t *surv = (uint64_t *)malloc((rt.window + 1) * sizeof(uint64_t));
    uint64_t ns = crt_runtime_survivors(&rt, surv);
    uint64_t prime_count_window = 0;
    for (uint64_t k = 0; k < ns; k++) {
        mpz_set(cand, base0);
        mpz_add(cand, cand, nadd0);
        if (mpz_odd_p(cand))
            mpz_sub_ui(cand, cand, 1);
        mpz_add_ui(cand, cand, (unsigned long)surv[k]);
        if (mpz_probab_prime_p(cand, 12) > 0)
            prime_count_window++;
    }

    double logbase = (256.0 + shift) * log(2.0);
    double natural = (double)rt.window / logbase;
    printf("window primes=%llu  natural~%.1f  ratio=%.2f\n",
           (unsigned long long)prime_count_window, natural,
           natural > 0.0 ? (double)prime_count_window / natural : 0.0);

    if (natural > 4.0 && (double)prime_count_window > natural * 0.9) {
        printf("FAIL: window is NOT prime-poor (>= 90%% of natural density)\n");
        failures++;
    } else {
        printf("OK: aligned window is prime-poor (covering effect)\n");
    }

    free(surv);
    mpz_clears(nadd0, base0, cand, NULL);
    crt_runtime_free(&rt);
    remove(tmp);

    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    }
    printf("\n%d TEST(S) FAILED\n", failures);
    return 1;
}
