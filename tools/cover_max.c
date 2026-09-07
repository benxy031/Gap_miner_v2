/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * cover_max — covering-coverage search experiment (Ziller-Morack inspired).
 *
 * Objective: minimize survivors in [1, gap_target) (positions NOT covered by
 * any residue class j ≡ -r_p (mod p)); equivalently maximize coverage.  This
 * is the same cost function as covering.c (lex_objective primary metric) and
 * the same problem family as Jacobsthal-function computation (Ziller-Morack,
 * arXiv:1611.03310) with a partial-coverage objective.
 *
 * Modes:
 *   --file <path>          load a CRT file, sanity-check its n_candidates
 *                          against covering_count_survivors, then optimize
 *                          from scratch (prime-greedy + local sweeps + ILS)
 *                          under --seconds.
 *   --primes N --gap-target G   optimize the first N odd primes.
 *   --exact N --gap-target G    exhaustive DFS over all residue choices for
 *                          the first N odd primes (small N only) — the true
 *                          optimum, used to calibrate the greedy gap.
 *
 * Links build/new_src/covering.o for the repo's own evaluator (cross-check).
 */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "covering.h"

static volatile sig_atomic_t g_timeout = 0;
static void on_alarm(int sig) { (void)sig; g_timeout = 1; }

static uint64_t rng_state = 0x9e3779b97f4a7c15ULL;
static uint64_t rng_next(void) {
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return x;
}

static inline uint64_t first_hit(uint64_t p, uint64_t r) {
    uint64_t f = (p - (r % p)) % p;
    return f ? f : p;
}

static uint64_t marginal(const uint8_t *covered, uint64_t gap_target,
                         uint64_t p, uint64_t r) {
    uint64_t c = 0;
    for (uint64_t j = first_hit(p, r); j < gap_target; j += p)
        if (!covered[j])
            c++;
    return c;
}

static void mark_all(uint8_t *covered, uint64_t gap_target,
                     const uint64_t *primes, const uint64_t *residues,
                     size_t n_primes) {
    memset(covered, 0, gap_target);
    for (size_t i = 0; i < n_primes; i++)
        for (uint64_t j = first_hit(primes[i], residues[i]);
             j < gap_target; j += primes[i])
            covered[j] = 1;
}

static uint64_t count_uncovered(const uint8_t *covered, uint64_t gap_target) {
    uint64_t s = 0;
    for (uint64_t j = 1; j < gap_target; j++)
        if (!covered[j])
            s++;
    return s;
}

/* Prime-greedy (GPA-lite): at each step choose, over ALL unplaced primes and
   ALL their nonzero residues, the choice covering the most currently
   uncovered offsets; random tie-breaking.  Stronger than covering.c's fixed
   order greedy. */
static uint64_t prime_greedy(const uint64_t *primes, size_t n_primes,
                             uint64_t gap_target, uint64_t *residues,
                             uint64_t seed) {
    uint8_t *covered = (uint8_t *)calloc(gap_target, 1);
    uint8_t *placed = (uint8_t *)calloc(n_primes, 1);
    if (!covered || !placed) {
        free(covered);
        free(placed);
        return UINT64_MAX;
    }
    uint64_t rs = seed;
    size_t remaining = n_primes;
    while (remaining > 0) {
        uint64_t best_c = 0;
        size_t best_i = SIZE_MAX;
        uint64_t best_r = 0;
        for (size_t i = 0; i < n_primes; i++) {
            if (placed[i])
                continue;
            uint64_t p = primes[i];
            for (uint64_t r = 1; r < p; r++) {
                uint64_t c = marginal(covered, gap_target, p, r);
                if (c > best_c ||
                    (c == best_c && best_i != SIZE_MAX &&
                     ((rs = rs * 6364136223846793005ULL + 1442695040888963407ULL)
                      & 1ULL))) {
                    best_c = c;
                    best_i = i;
                    best_r = r;
                }
            }
        }
        if (best_i == SIZE_MAX)
            break;
        placed[best_i] = 1;
        residues[best_i] = best_r;
        for (uint64_t j = first_hit(primes[best_i], best_r);
             j < gap_target; j += primes[best_i])
            covered[j] = 1;
        remaining--;
    }
    uint64_t surv = count_uncovered(covered, gap_target);
    free(covered);
    free(placed);
    return surv;
}

/* Single-prime local sweep: rebuild coverage without prime i and re-pick the
   residue covering the most.  Returns updated survivor count. */
static uint64_t local_sweep(const uint64_t *primes, size_t n_primes,
                            uint64_t gap_target, uint64_t *residues) {
    uint8_t *covered = (uint8_t *)malloc(gap_target);
    uint8_t *others = (uint8_t *)malloc(gap_target);
    if (!covered || !others) {
        free(covered);
        free(others);
        return UINT64_MAX;
    }
    for (size_t i = 0; i < n_primes; i++) {
        memset(others, 0, gap_target);
        for (size_t k = 0; k < n_primes; k++) {
            if (k == i)
                continue;
            for (uint64_t j = first_hit(primes[k], residues[k]);
                 j < gap_target; j += primes[k])
                others[j] = 1;
        }
        uint64_t best_r = residues[i];
        uint64_t best_c = marginal(others, gap_target, primes[i], best_r);
        for (uint64_t r = 1; r < primes[i]; r++) {
            uint64_t c = marginal(others, gap_target, primes[i], r);
            if (c > best_c) {
                best_c = c;
                best_r = r;
            }
        }
        residues[i] = best_r;
    }
    mark_all(covered, gap_target, primes, residues, n_primes);
    uint64_t surv = count_uncovered(covered, gap_target);
    free(covered);
    free(others);
    return surv;
}

/* 2-opt pair sweep: for every prime pair, re-choose both residues jointly
   (given all others fixed) to maximize the union of new coverage.  O(m) per
   pair.  Repeats passes until no pair improves. */
static uint64_t pair_sweep(const uint64_t *primes, size_t n_primes,
                           uint64_t gap_target, uint64_t *residues) {
    uint8_t *others = (uint8_t *)malloc(gap_target);
    uint16_t *a = (uint16_t *)malloc(4096 * sizeof(uint16_t));
    uint16_t *b = (uint16_t *)malloc(4096 * sizeof(uint16_t));
    uint16_t *oc = (uint16_t *)malloc(4096 * sizeof(uint16_t));
    if (!others || !a || !b || !oc) {
        free(others); free(a); free(b); free(oc);
        return UINT64_MAX;
    }
    memset(oc, 0, 4096 * sizeof(uint16_t));
    for (int pass = 0; pass < 6; pass++) {
        size_t improved = 0;
        for (size_t i = 0; i < n_primes; i++) {
            uint64_t pi = primes[i];
            if (pi > 4096)
                continue;
            for (size_t j = i + 1; j < n_primes; j++) {
                uint64_t pj = primes[j];
                if (pj > 4096)
                    continue;
                /* bitmap of coverage without i and j */
                memset(others, 0, gap_target);
                for (size_t k = 0; k < n_primes; k++) {
                    if (k == i || k == j)
                        continue;
                    for (uint64_t x = first_hit(primes[k], residues[k]);
                         x < gap_target; x += primes[k])
                        others[x] = 1;
                }
                /* marginal counts per residue */
                memset(a, 0, pi * sizeof(uint16_t));
                memset(b, 0, pj * sizeof(uint16_t));
                for (uint64_t r = 1; r < pi; r++)
                    for (uint64_t x = first_hit(pi, r); x < gap_target;
                         x += pi)
                        if (!others[x])
                            a[r]++;
                for (uint64_t r = 1; r < pj; r++)
                    for (uint64_t x = first_hit(pj, r); x < gap_target;
                         x += pj)
                        if (!others[x])
                            b[r]++;
                /* current union of the two residues */
                uint64_t cur_ri = residues[i], cur_rj = residues[j];
                uint64_t cur_overlap = 0;
                for (uint64_t x = first_hit(pi, cur_ri); x < gap_target;
                     x += pi)
                    if (!others[x] && (x % pj) == (pj - cur_rj) % pj)
                        cur_overlap++;
                uint64_t cur_u = a[cur_ri] + b[cur_rj] - cur_overlap;
                /* best joint choice */
                uint64_t best_u = cur_u;
                uint64_t best_ri = cur_ri, best_rj = cur_rj;
                for (uint64_t ri = 1; ri < pi; ri++) {
                    if (a[ri] + (b[1] ? b[1] : 0) <= best_u)
                        continue; /* weak upper bound skip */
                    /* overlap of A(ri) with each B(rj) */
                    for (uint64_t x = first_hit(pi, ri); x < gap_target;
                         x += pi) {
                        if (!others[x]) {
                            uint64_t t = (pj - (x % pj)) % pj;
                            if (t)
                                oc[t]++;
                        }
                    }
                    for (uint64_t rj = 1; rj < pj; rj++) {
                        if (!b[rj])
                            continue;
                        uint64_t u = (uint64_t)a[ri] + (uint64_t)b[rj] -
                                     oc[rj];
                        if (u > best_u) {
                            best_u = u;
                            best_ri = ri;
                            best_rj = rj;
                        }
                    }
                    for (uint64_t x = first_hit(pi, ri); x < gap_target;
                         x += pi) {
                        if (!others[x]) {
                            uint64_t t = (pj - (x % pj)) % pj;
                            if (t)
                                oc[t]--;
                        }
                    }
                }
                if (best_u > cur_u) {
                    residues[i] = best_ri;
                    residues[j] = best_rj;
                    improved++;
                }
            }
        }
        if (!improved)
            break;
    }
    free(others); free(a); free(b); free(oc);
    /* rebuild and count */
    uint8_t *covered = (uint8_t *)malloc(gap_target);
    if (!covered)
        return UINT64_MAX;
    mark_all(covered, gap_target, primes, residues, n_primes);
    uint64_t surv = count_uncovered(covered, gap_target);
    free(covered);
    return surv;
}

static void perturb(const uint64_t *primes, size_t n_primes,
                    uint64_t *dst, const uint64_t *src, size_t k) {
    memcpy(dst, src, n_primes * sizeof(uint64_t));
    for (size_t t = 0; t < k; t++) {
        size_t idx = (size_t)(rng_next() % (uint64_t)n_primes);
        dst[idx] = 1 + (uint64_t)(rng_next() % (primes[idx] - 1));
    }
}

/* Iterated local search entry is handled in main via perturb + pair_sweep. */

/* Exhaustive DFS over all residue choices (small N only).  Bound: remaining
   primes can cover at most sum floor((gap_target-1)/p_i) uncovered slots. */
static void exact_dfs(const uint64_t *primes, size_t n_primes,
                      uint64_t gap_target, uint8_t *covered,
                      size_t k, uint64_t covered_now,
                      const uint64_t *rem_budget, uint64_t *best_surv,
                      uint64_t *best_res, uint64_t *cur, uint64_t *nodes) {
    (*nodes)++;
    if (g_timeout)
        return;
    if (k == n_primes) {
        uint64_t surv = (gap_target - 1) - covered_now;
        if (surv < *best_surv) {
            *best_surv = surv;
            memcpy(best_res, cur, n_primes * sizeof(uint64_t));
        }
        return;
    }
    uint64_t potential = covered_now + rem_budget[k];
    if ((gap_target - 1) - potential >= *best_surv)
        return; /* cannot beat current best */
    uint64_t p = primes[k];
    uint64_t marked[4096];
    for (uint64_t r = 1; r < p; r++) {
        size_t nmark = 0;
        uint64_t added = 0;
        for (uint64_t j = first_hit(p, r); j < gap_target; j += p) {
            if (!covered[j]) {
                covered[j] = 1;
                marked[nmark++] = j;
                added++;
            }
        }
        cur[k] = r;
        exact_dfs(primes, n_primes, gap_target, covered, k + 1,
                  covered_now + added, rem_budget, best_surv, best_res,
                  cur, nodes);
        for (size_t m = 0; m < nmark; m++)
            covered[marked[m]] = 0;
    }
}

/* Load a CRT file: returns n_primes in *n_out, fills primes/residues,
   gap_target, n_candidates; captures merit/shift for the writer. */
static char g_merit[64] = "30.00";
static uint64_t g_shift = 0;
static int load_file(const char *path, uint64_t *primes, uint64_t *residues,
                     size_t max_primes, size_t *n_out, uint64_t *gap_target,
                     uint64_t *n_candidates) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return -1;
    }
    char line[512];
    size_t n = 0;
    uint64_t gt = 0, nc = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        if (sscanf(line, "gap_target %llu", (unsigned long long *)&gt) == 1) {
            *gap_target = gt;
            continue;
        }
        if (sscanf(line, "n_candidates %llu", (unsigned long long *)&nc) == 1) {
            *n_candidates = nc;
            continue;
        }
        if (sscanf(line, "merit %63s", g_merit) == 1)
            continue;
        if (sscanf(line, "shift %llu", (unsigned long long *)&g_shift) == 1)
            continue;
        unsigned long long p = 0, o = 0;
        if (sscanf(line, "%llu %llu", &p, &o) == 2 && p > 1 && o > 0) {
            if (n >= max_primes) {
                fprintf(stderr, "too many primes\n");
                fclose(f);
                return -1;
            }
            primes[n] = (uint64_t)p;
            residues[n] = (uint64_t)((p - o) % p);
            n++;
        }
    }
    fclose(f);
    *n_out = n;
    return (gt && nc) ? 0 : -1;
}

/* Write a CRT sieve file in gen_crt format from a residue set. */
static int write_crt_file(const char *path, const uint64_t *primes,
                          const uint64_t *residues, size_t n,
                          uint64_t gap_target, uint64_t n_candidates) {
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", path);
        return -1;
    }
    fprintf(f, "# CRT sieve file generated by cover_max\n");
    fprintf(f, "n_primes %zu\n", n);
    fprintf(f, "merit %s\n", g_merit);
    fprintf(f, "shift %llu\n", (unsigned long long)g_shift);
    fprintf(f, "gap_target %llu\n", (unsigned long long)gap_target);
    fprintf(f, "n_candidates %llu\n", (unsigned long long)n_candidates);
    for (size_t i = 0; i < n; i++) {
        uint64_t o = (primes[i] - (residues[i] % primes[i])) % primes[i];
        if (!o)
            o = primes[i];
        fprintf(f, "%llu %llu\n", (unsigned long long)primes[i],
                (unsigned long long)o);
    }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    const char *file = NULL;
    const char *out = NULL;
    unsigned long n_primes = 0;
    uint64_t gap_target = 0;
    int exact_mode = 0;
    unsigned long seconds = 120;
    uint64_t seed = 20260907;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--file") && i + 1 < argc)
            file = argv[++i];
        else if (!strcmp(argv[i], "--primes") && i + 1 < argc)
            n_primes = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--gap-target") && i + 1 < argc)
            gap_target = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--exact") && i + 1 < argc) {
            exact_mode = 1;
            n_primes = strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out = argv[++i];
    }

    static uint64_t primes[4096], residues[4096];
    size_t n = 0;
    uint64_t file_nc = 0;

    if (file) {
        if (load_file(file, primes, residues, 4096, &n, &gap_target,
                      &file_nc) != 0) {
            fprintf(stderr, "file load failed\n");
            return 1;
        }
        uint64_t check =
            covering_count_survivors(primes, residues, n, gap_target);
        printf("[cover_max] %s: %zu primes, gap_target=%llu, "
               "file n_candidates=%llu, evaluator says %llu %s\n",
               file, n, (unsigned long long)gap_target,
               (unsigned long long)file_nc, (unsigned long long)check,
               check == file_nc ? "(MATCH)" : "(MISMATCH!)");
    } else if (n_primes >= 1 && gap_target > 1) {
        n = 0;
        for (uint64_t p = 3; n < n_primes; p += 2) {
            int isp = 1;
            for (uint64_t d = 3; d * d <= p; d += 2)
                if (p % d == 0) { isp = 0; break; }
            if (isp)
                primes[n++] = p;
        }
    } else {
        fprintf(stderr,
                "usage: cover_max --file <crt> [--seconds N] [--seed N]\n"
                "       cover_max --primes N --gap-target G [--seconds N]\n"
                "       cover_max --exact N --gap-target G\n");
        return 1;
    }

    double denom = (double)(gap_target - 1);
    uint64_t base = covering_count_survivors(primes, residues, n, gap_target);
    if (file) {
        printf("[cover_max] baseline: survivors=%llu coverage=%.4f%%\n",
               (unsigned long long)base, 100.0 * (1.0 - (double)base / denom));
    }

    if (exact_mode) {
        if (n > 10 || gap_target > 200000) {
            fprintf(stderr, "--exact supports at most 10 primes and "
                            "gap_target <= 200000\n");
            return 1;
        }
        uint8_t *covered = (uint8_t *)calloc(gap_target, 1);
        uint64_t best = UINT64_MAX;
        uint64_t best_res[4096];
        uint64_t cur[4096];
        uint64_t rem_budget[16];
        uint64_t nodes = 0;
        rem_budget[n] = 0;
        for (size_t i = n; i-- > 0;)
            rem_budget[i] = rem_budget[i + 1] + (gap_target - 1) / primes[i];
        signal(SIGALRM, on_alarm);
        alarm((unsigned int)seconds);
        exact_dfs(primes, n, gap_target, covered, 0, 0, rem_budget, &best,
                  best_res, cur, &nodes);
        printf("[cover_max] EXACT: n=%zu gap_target=%llu nodes=%llu "
               "min_survivors=%llu coverage=%.4f%%\n",
               n, (unsigned long long)gap_target, (unsigned long long)nodes,
               (unsigned long long)best,
               100.0 * (1.0 - (double)best / denom));
        /* greedy on the same config for calibration */
        uint64_t gsurv = prime_greedy(primes, n, gap_target, residues, seed);
        gsurv = local_sweep(primes, n, gap_target, residues);
        printf("[cover_max] CALIB: greedy(+sweep) survivors=%llu "
               "coverage=%.4f%%  (opt gap %lld survivors, %.4f%% coverage)\n",
               (unsigned long long)gsurv,
               100.0 * (1.0 - (double)gsurv / denom),
               (long long)((long long)gsurv - (long long)best),
               100.0 * (1.0 - (double)best / denom));
        return 0;
    }

    signal(SIGALRM, on_alarm);
    alarm((unsigned int)seconds);

    uint64_t best_surv = UINT64_MAX;
    uint64_t best_res[4096];
    uint64_t start = (uint64_t)time(NULL);
    unsigned long restarts = 0;
    uint64_t base_res[4096];
    memcpy(base_res, residues, n * sizeof(uint64_t));
    while (!g_timeout) {
        uint64_t trial[4096];
        uint64_t s;
        if (file && (restarts % 2 == 0)) {
            /* ILS starting from the file's own solution */
            perturb(primes, n, trial, base_res, 4 + (restarts / 2) % 30);
        } else {
            /* restart from prime-greedy + single sweep */
            s = prime_greedy(primes, n, gap_target, trial,
                             seed + restarts * 2654435761ULL);
            if (s == UINT64_MAX)
                break;
            local_sweep(primes, n, gap_target, trial);
        }
        s = pair_sweep(primes, n, gap_target, trial);
        if (s == UINT64_MAX)
            break;
        if (s < best_surv) {
            best_surv = s;
            memcpy(best_res, trial, n * sizeof(uint64_t));
            printf("[cover_max] restart %lu: survivors=%llu coverage=%.4f%% "
                   "(elapsed %llus)\n",
                   restarts, (unsigned long long)best_surv,
                   100.0 * (1.0 - (double)best_surv / denom),
                   (unsigned long long)((uint64_t)time(NULL) - start));
            fflush(stdout);
        }
        restarts++;
    }
    printf("[cover_max] DONE restarts=%lu best_survivors=%llu coverage=%.4f%% "
           "(baseline %llu / %.4f%%)\n",
           restarts, (unsigned long long)best_surv,
           100.0 * (1.0 - (double)best_surv / denom),
           (unsigned long long)base, 100.0 * (1.0 - (double)base / denom));
    if (file) {
        printf("[cover_max] improvement over file: %lld survivors (%.4f%% -> "
               "%.4f%%)\n",
               (long long)((long long)base - (long long)best_surv),
               100.0 * (1.0 - (double)base / denom),
               100.0 * (1.0 - (double)best_surv / denom));
    }
    if (out) {
        uint64_t check = covering_count_survivors(primes, best_res, n,
                                                  gap_target);
        if (check == best_surv && write_crt_file(out, primes, best_res, n,
                                                 gap_target, best_surv) == 0)
            printf("[cover_max] wrote %s (%llu candidates)\n", out,
                   (unsigned long long)best_surv);
        else
            fprintf(stderr, "[cover_max] NOT writing out file: evaluator "
                            "mismatch (%llu vs %llu)\n",
                    (unsigned long long)check, (unsigned long long)best_surv);
    }
    return 0;
}
