/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * gen_crt — CRT (Chinese Remainder Theorem) gap-solver offset generator.
 *
 * Two-phase algorithm compatible with GapMiner --calc-ctr parameters:
 *   Phase 1 (greedy):  for each prime pick the offset covering the most
 *                       uncovered positions in [1, gap_target].
 *   Phase 2 (evolution): tournament selection + crossover + mutation +
 *                       local search on non-fixed primes (--ctr-fixed).
 *
 * Output: text CRT file (n_primes / merit / shift / gap_target /
 * n_candidates header followed by "prime offset" lines) that the miner
 * loads for CRT-aligned sieving.
 */

#include "covering.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <getopt.h>

#define MAX_PRIMES 200

static uint64_t g_primes[MAX_PRIMES];
static int g_prime_count = 0;

static void gen_primes(void) {
    const int limit = 2000;
    char *comp = (char *)calloc((size_t)limit, 1);
    if (!comp) return;
    for (int f = 2; f * f < limit; f++) {
        if (comp[f]) continue;
        for (int m = f * f; m < limit; m += f) comp[m] = 1;
    }
    for (int v = 2; v < limit && g_prime_count < MAX_PRIMES; v++) {
        if (!comp[v]) g_primes[g_prime_count++] = (uint64_t)v;
    }
    free(comp);
}

static double primorial_log2(int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++)
        acc += log2((double)g_primes[i]);
    return acc;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --calc-ctr --ctr-primes N --ctr-file FILE [options]\n"
        "\n"
        "  --calc-ctr            Calculate a CRT file (accepted for compat)\n"
        "  --ctr-primes N        Number of CRT primes (2..%d)\n"
        "  --ctr-merit  M        Target merit (default 22.0)\n"
        "  --ctr-bits   B        Extra bits: shift - log2(primorial) (default 0)\n"
        "  --ctr-strength S      Greedy restarts / quality (default 50)\n"
        "  --ctr-evolution       Enable evolutionary refinement\n"
        "  --ctr-fixed  F        Primes frozen during evolution (default 8)\n"
        "  --ctr-ivs    I        Population size for evolution (default 10)\n"
        "  --ctr-range  R        Percent deviation from --ctr-primes (default 0)\n"
        "  --ctr-file   FILE     Output CRT file (required)\n"
        "\n"
        "Minimum shift = ceil(log2(p1*p2*...*pN)) + ctr-bits.\n"
        "Tip: the original GapMiner docs recommend ctr-merit = target_merit - 1.\n",
        prog, MAX_PRIMES);
}

int main(int argc, char **argv) {
    int ctr_primes = 0;
    double ctr_merit = 22.0;
    int ctr_bits = 0;
    int ctr_strength = 50;
    int ctr_evolution = 0;
    int ctr_fixed = 8;
    int ctr_ivs = 10;
    int ctr_range = 0;
    const char *ctr_file = NULL;

    static const struct option long_opts[] = {
        {"calc-ctr",       no_argument,       NULL, 'C'},
        {"ctr-primes",     required_argument, NULL, 'p'},
        {"ctr-merit",      required_argument, NULL, 'm'},
        {"ctr-bits",       required_argument, NULL, 'b'},
        {"ctr-strength",   required_argument, NULL, 's'},
        {"ctr-evolution",  no_argument,       NULL, 'e'},
        {"ctr-fixed",      required_argument, NULL, 'f'},
        {"ctr-ivs",        required_argument, NULL, 'i'},
        {"ctr-range",      required_argument, NULL, 'r'},
        {"ctr-file",       required_argument, NULL, 'o'},
        {"help",           no_argument,       NULL, 'h'},
        {NULL, 0, NULL, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "Cp:m:b:s:ef:i:r:o:h", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'C': break;
        case 'p': ctr_primes = atoi(optarg); break;
        case 'm': ctr_merit = atof(optarg); break;
        case 'b': ctr_bits = atoi(optarg); break;
        case 's': ctr_strength = atoi(optarg); break;
        case 'e': ctr_evolution = 1; break;
        case 'f': ctr_fixed = atoi(optarg); break;
        case 'i': ctr_ivs = atoi(optarg); break;
        case 'r': ctr_range = atoi(optarg); break;
        case 'o': ctr_file = optarg; break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    if (!ctr_file) {
        fprintf(stderr, "error: --ctr-file is required\n\n");
        usage(argv[0]);
        return 1;
    }
    if (ctr_primes < 2 || ctr_primes > MAX_PRIMES) {
        fprintf(stderr, "error: --ctr-primes must be 2..%d\n", MAX_PRIMES);
        return 1;
    }
    if (ctr_merit <= 0.0) {
        fprintf(stderr, "error: --ctr-merit must be > 0\n");
        return 1;
    }
    if (ctr_fixed < 0) ctr_fixed = 0;
    if (ctr_fixed > ctr_primes) ctr_fixed = ctr_primes;
    if (ctr_ivs < 2) ctr_ivs = 2;
    if (ctr_strength < 1) ctr_strength = 1;
    if (ctr_range < 0) ctr_range = 0;

    gen_primes();

    /* Prime-count search range (--ctr-range lets the tool try nearby
     * counts and keep the one with the fewest candidates). */
    int lo = ctr_primes, hi = ctr_primes;
    if (ctr_range > 0) {
        lo = (int)((double)ctr_primes * (1.0 - (double)ctr_range / 100.0));
        hi = (int)((double)ctr_primes * (1.0 + (double)ctr_range / 100.0));
        if (lo < 2) lo = 2;
        if (hi > MAX_PRIMES) hi = MAX_PRIMES;
    }

    uint64_t best_res[MAX_PRIMES];
    uint64_t best_cand = UINT64_MAX;
    int best_n = 0, best_shift = 0;
    uint64_t best_gap = 0;

    for (int n = lo; n <= hi; n++) {
        double pb = primorial_log2(n);
        int shift = (int)ceil(pb) + ctr_bits;
        uint64_t gap_size = (uint64_t)ceil(ctr_merit * (256.0 + (double)shift) * log(2.0));

        uint64_t residues[MAX_PRIMES];
        uint64_t cand;
        if (ctr_evolution) {
            struct covering_evo_config cfg;
            cfg.strength = (uint32_t)ctr_strength;
            cfg.population = (uint32_t)ctr_ivs;
            cfg.generations = (uint32_t)(ctr_strength / 2);
            cfg.local_sweeps = 4;
            cfg.fixed = (uint32_t)ctr_fixed;
            cfg.ils_rounds = 8;
            cfg.seed = (uint64_t)time(NULL) ^ (uint64_t)n;
            cand = covering_optimize_evolution(g_primes, (size_t)n, gap_size,
                                               residues, &cfg);
        } else {
            struct covering_config cfg;
            cfg.strength = (uint32_t)ctr_strength;
            cfg.local_sweeps = 8;
            cfg.ils_rounds = 0;
            cfg.pair_search = 0;
            cfg.seed = (uint64_t)time(NULL) ^ (uint64_t)n;
            cand = covering_optimize(g_primes, (size_t)n, gap_size, residues,
                                     &cfg);
        }

        if (cand < best_cand) {
            best_cand = cand;
            best_n = n;
            best_shift = shift;
            best_gap = gap_size;
            memcpy(best_res, residues, (size_t)n * sizeof(uint64_t));
        }
    }

    FILE *f = fopen(ctr_file, "w");
    if (!f) {
        perror(ctr_file);
        return 1;
    }
    fprintf(f, "# CRT sieve file generated by gapminer gen_crt\n");
    fprintf(f, "n_primes %d\n", best_n);
    fprintf(f, "merit %.2f\n", ctr_merit);
    fprintf(f, "shift %d\n", best_shift);
    fprintf(f, "gap_target %llu\n", (unsigned long long)best_gap);
    fprintf(f, "n_candidates %llu\n", (unsigned long long)best_cand);
    for (int i = 0; i < best_n; i++) {
        uint64_t p = g_primes[i];
        uint64_t r = best_res[i];
        uint64_t off = (p - r) % p;   /* offset o_i: d ≡ o_i (mod p) covered */
        fprintf(f, "%llu %llu\n", (unsigned long long)p, (unsigned long long)off);
    }
    fclose(f);

    fprintf(stderr, "wrote %s  (%d primes, %llu candidates, shift=%d, gap_target=%llu)\n",
            ctr_file, best_n, (unsigned long long)best_cand,
            best_shift, (unsigned long long)best_gap);
    return 0;
}
