/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * CRT set generator: precompute a per-shift primorial sieve file.
 *
 * Usage:
 *   crt_gen --shift 512 [--size 25000] [--trials 1000] [--seed 0] \
 *           [--difficulty 21.92] [--out crt_512.txt]
 */

#include "../new_src/crt_set.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s --shift <N> [--size <S>] [--trials <T>] [--seed <S>]\n"
        "              [--difficulty <D>] [--out <path>]\n", prog);
}

int main(int argc, char **argv) {
    uint32_t shift = 0;
    uint64_t size = 0;
    uint64_t trials = 1000;
    uint64_t seed = 0;
    double difficulty = 21.92;
    char out_path[512] = {0};

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--shift") == 0 && i + 1 < argc) {
            shift = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--trials") == 0 && i + 1 < argc) {
            trials = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--difficulty") == 0 && i + 1 < argc) {
            difficulty = atof(argv[++i]);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            snprintf(out_path, sizeof(out_path), "%s", argv[++i]);
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (shift < 14 || shift > 1024) {
        fprintf(stderr, "error: --shift must be in [14, 1024]\n");
        return 2;
    }

    uint32_t n_primes = crt_set_primes_for_shift(shift);
    if (n_primes == 0) {
        fprintf(stderr, "error: no primes available for shift %u\n", shift);
        return 1;
    }

    if (size == 0) {
        double logbase = (256.0 + (double)shift) * log(2.0);
        size = (uint64_t)ceil(2.0 * difficulty * logbase);
    }

    struct crt_set cs;
    if (!crt_set_init(&cs, shift, size, n_primes)) {
        fprintf(stderr, "error: crt_set_init failed\n");
        return 1;
    }

    if (crt_set_search_offset(&cs, trials, seed) != 0) {
        fprintf(stderr, "error: offset search failed\n");
        crt_set_free(&cs);
        return 1;
    }

    if (out_path[0] == 0) {
        snprintf(out_path, sizeof(out_path), "crt_%u.txt", shift);
    }
    if (crt_set_save(&cs, out_path) != 0) {
        fprintf(stderr, "error: cannot write %s\n", out_path);
        crt_set_free(&cs);
        return 1;
    }

    printf("shift:          %u\n", cs.shift);
    printf("n_primes:       %u\n", cs.n_primes);
    printf("primorial bits: %llu (min shift)\n", (unsigned long long)cs.bit_size);
    printf("window size:    %llu\n", (unsigned long long)cs.size);
    printf("candidates:     %llu\n", (unsigned long long)cs.n_candidates);
    printf("avg candidates: %.1f\n", cs.avg_candidates);
    printf("ratio:          %.3f\n",
           cs.avg_candidates > 0.0
               ? (double)cs.n_candidates / cs.avg_candidates
               : 0.0);
    printf("written:        %s\n", out_path);

    crt_set_free(&cs);
    return 0;
}
