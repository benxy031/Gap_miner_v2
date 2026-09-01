/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Covering-system residue optimizer (see covering.h).
 */

#include "covering.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint64_t g_rng_state;

/* Objective mode shared by both optimizers: 0 = minimize survivors
   (classic), 1 = maximize the longest contiguous covered run, 2 =
   lexicographic (minimize survivors first, then maximize the run),
   3 = expected blocks for a difficulty D (see covering.h). */
static int g_run_mode = 0;
static int g_lex_mode = 0;
static int g_blocks_mode = 0;
static double g_blocks_D = 0.0;
static double g_blocks_logbase = 1.0;

static uint64_t rng_next(void) {
    uint64_t x = g_rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng_state = x;
    return x;
}

/* First covered offset for residue r of prime p: j ≡ -r (mod p), j in (0,p]. */
static inline uint64_t first_hit(uint64_t p, uint64_t r) {
    uint64_t f = (p - (r % p)) % p;
    return f ? f : p;
}

/* Number of currently-uncovered offsets that residue r of prime p covers. */
static uint64_t count_cover(const uint8_t *covered, uint64_t gap_target,
                            uint64_t p, uint64_t r) {
    uint64_t c = 0;
    for (uint64_t j = first_hit(p, r); j < gap_target; j += p) {
        if (!covered[j])
            c++;
    }
    return c;
}

static void mark_cover(uint8_t *covered, uint64_t gap_target,
                       uint64_t p, uint64_t r) {
    for (uint64_t j = first_hit(p, r); j < gap_target; j += p)
        covered[j] = 1;
}

static uint64_t count_survivors(const uint8_t *covered, uint64_t gap_target) {
    uint64_t s = 0;
    for (uint64_t j = 1; j < gap_target; j++)
        if (!covered[j])
            s++;
    return s;
}

/* Longest contiguous covered run + survivor count of a bitmap. */
static void covered_stats(const uint8_t *covered, uint64_t gap_target,
                          uint64_t *longest_out, uint64_t *survivors_out) {
    uint64_t longest = 0, surv = 0;
    uint64_t j = 1;
    while (j < gap_target) {
        if (!covered[j]) {
            surv++;
            j++;
            continue;
        }
        uint64_t s = j;
        while (j < gap_target && covered[j])
            j++;
        uint64_t len = j - s;
        if (len > longest)
            longest = len;
    }
    *longest_out = longest;
    *survivors_out = surv;
}

/* Encoded solution score: lower is better.  Run-mode ranks the longest
 * covered run first and survivor count second; lex-mode ranks survivors
 * first and the run second; classic ranks survivors only.
 * gap_target is < 2^32 in all supported configurations. */
static uint64_t encode_score(uint64_t longest, uint64_t survivors,
                             uint64_t gap_target) {
    if (g_run_mode)
        return ((gap_target - longest) << 32) | survivors;
    if (g_lex_mode)
        return (survivors << 32) | (gap_target - longest);
    return survivors;
}

/* Expected-blocks cost over the bitmap: for every survivor, its forward
   covered run r; cost = SUM (1 - P(gap >= D)) scaled by 2^20, where
   P ~ exp(-max(0, D - r)/logbase).  A survivor whose run already reaches D
   costs nothing.  Total O(gap_target). */
static uint64_t blocks_cost(const uint8_t *covered, uint64_t gap_target) {
    double d_abs = g_blocks_D * g_blocks_logbase;
    double scale = 1048576.0;
    uint64_t cost = 0;
    uint64_t j = 1;
    while (j < gap_target) {
        if (!covered[j]) {
            uint64_t r = 0;
            uint64_t t = j + 1;
            while (t < gap_target && covered[t]) {
                t++;
                r++;
            }
            double pen;
            if ((double)r >= d_abs)
                pen = 0.0;
            else
                pen = 1.0 - exp(-(d_abs - (double)r) / g_blocks_logbase);
            cost += (uint64_t)(pen * scale + 0.5);
            j++;
            continue;
        }
        j++;
    }
    return cost;
}

/* Full evaluation of a residue set (builds its bitmap). */
static uint64_t eval_solution(const uint64_t *primes, const uint64_t *residues,
                              size_t n_primes, uint64_t gap_target,
                              uint64_t *longest_out, uint64_t *survivors_out) {
    uint8_t *covered = (uint8_t *)calloc(gap_target, 1);
    if (!covered) {
        if (longest_out) *longest_out = 0;
        if (survivors_out) *survivors_out = UINT64_MAX;
        return UINT64_MAX;
    }
    for (size_t i = 0; i < n_primes; i++)
        mark_cover(covered, gap_target, primes[i], residues[i]);
    uint64_t longest, surv;
    covered_stats(covered, gap_target, &longest, &surv);
    uint64_t score;
    if (g_blocks_mode) {
        uint64_t cost = blocks_cost(covered, gap_target);
        score = (cost << 32) | surv;
    } else {
        score = encode_score(longest, surv, gap_target);
    }
    free(covered);
    if (longest_out) *longest_out = longest;
    if (survivors_out) *survivors_out = surv;
    return score;
}

uint64_t covering_longest_run(const uint64_t *primes, const uint64_t *residues,
                              size_t n_primes, uint64_t gap_target) {
    if (!primes || !residues || n_primes == 0 || gap_target < 2)
        return 0;
    uint64_t longest, surv;
    eval_solution(primes, residues, n_primes, gap_target, &longest, &surv);
    return longest;
}

/* One-pass per-prime refinement targeting the longest covered run: for
 * each prime, rebuild the cover without it and pick the residue whose
 * marks merge covered runs into the longest run (tie-break: most newly
 * covered positions).  O(n_primes * gap_target). */
static void run_polish(const uint64_t *primes, size_t n_primes,
                       uint64_t gap_target, uint64_t *residues) {
    uint8_t *covered = (uint8_t *)malloc(gap_target);
    uint64_t *left = (uint64_t *)malloc(gap_target * sizeof(uint64_t));
    uint64_t *right = (uint64_t *)malloc(gap_target * sizeof(uint64_t));
    if (!covered || !left || !right) {
        free(covered);
        free(left);
        free(right);
        return;
    }

    for (size_t i = 0; i < n_primes; i++) {
        uint64_t p = primes[i];
        memset(covered, 0, gap_target);
        for (size_t k = 0; k < n_primes; k++) {
            if (k != i)
                mark_cover(covered, gap_target, primes[k], residues[k]);
        }

        /* Run boundaries of the bitmap without prime i. */
        uint64_t base_run = 0;
        uint64_t base_surv = 0;
        for (uint64_t j = 1; j < gap_target; ) {
            if (!covered[j]) {
                base_surv++;
                j++;
                continue;
            }
            uint64_t s = j;
            while (j < gap_target && covered[j])
                j++;
            uint64_t e = j - 1;
            if (e - s + 1 > base_run)
                base_run = e - s + 1;
            for (uint64_t t = s; t <= e; t++) {
                left[t] = s;
                right[t] = e;
            }
        }

        uint64_t best_r = residues[i];
        uint64_t best_run = 0, best_cov = 0, best_surv = base_surv;
        int first = 1;
        for (uint64_t r = 1; r < p; r++) {
            uint64_t run = base_run;
            uint64_t ncov = 0;
            for (uint64_t j = first_hit(p, r); j < gap_target; j += p) {
                if (covered[j])
                    continue;
                ncov++;
                uint64_t len = 1;
                if (j >= 1 && covered[j - 1])
                    len += right[j - 1] - left[j - 1] + 1;
                if (j + 1 < gap_target && covered[j + 1])
                    len += right[j + 1] - left[j + 1] + 1;
                if (len > run)
                    run = len;
            }
            int better;
            if (g_lex_mode) {
                uint64_t surv = base_surv - ncov;
                better = first || surv < best_surv ||
                         (surv == best_surv && run > best_run) ||
                         (surv == best_surv && run == best_run &&
                          ncov > best_cov);
                if (surv < best_surv)
                    best_surv = surv;
            } else {
                better = first || run > best_run ||
                         (run == best_run && ncov > best_cov);
            }
            if (better) {
                best_r = r;
                best_run = run;
                best_cov = ncov;
                first = 0;
            }
        }
        residues[i] = best_r;
    }

    free(covered);
    free(left);
    free(right);
}

/* Greedy pass: process primes in the given order, picking for each the
 * residue that covers the most currently-uncovered offsets (random tie-break
 * via the RNG).  Writes residues[] at the permuted indices. */
static void greedy_pass(const uint64_t *primes, size_t n_primes,
                        uint64_t gap_target, uint64_t *residues,
                        uint8_t *covered, const int *order) {
    memset(covered, 0, gap_target);
    for (size_t k = 0; k < n_primes; k++) {
        uint64_t p = primes[order[k]];
        uint64_t best_r = 1;
        uint64_t best_c = 0;
        for (uint64_t r = 1; r < p; r++) {
            uint64_t c = count_cover(covered, gap_target, p, r);
            if (c > best_c || (c == best_c && (rng_next() & 1u))) {
                best_c = c;
                best_r = r;
            }
        }
        residues[order[k]] = best_r;
        mark_cover(covered, gap_target, p, best_r);
    }
}

/* Single-prime local search: for each prime, rebuild the coverage without it
 * and re-pick the residue that covers the most. */
static void local_sweep(const uint64_t *primes, size_t n_primes,
                        uint64_t gap_target, uint64_t *residues) {
    uint8_t *covered = (uint8_t *)malloc(gap_target);
    if (!covered)
        return;

    for (size_t i = 0; i < n_primes; i++) {
        uint64_t p = primes[i];
        memset(covered, 0, gap_target);
        for (size_t k = 0; k < n_primes; k++) {
            if (k != i)
                mark_cover(covered, gap_target, primes[k], residues[k]);
        }
        uint64_t best_r = residues[i];
        uint64_t best_c = count_cover(covered, gap_target, p, best_r);
        for (uint64_t r = 1; r < p; r++) {
            uint64_t c = count_cover(covered, gap_target, p, r);
            if (c > best_c) {
                best_c = c;
                best_r = r;
            }
        }
        residues[i] = best_r;
    }

    free(covered);
}

uint64_t covering_count_survivors(const uint64_t *primes,
                                  const uint64_t *residues,
                                  size_t n_primes, uint64_t gap_target) {
    if (!primes || !residues || n_primes == 0 || gap_target < 2)
        return 0;
    uint8_t *covered = (uint8_t *)calloc(gap_target, 1);
    if (!covered)
        return 0;
    for (size_t i = 0; i < n_primes; i++)
        mark_cover(covered, gap_target, primes[i], residues[i]);
    uint64_t s = count_survivors(covered, gap_target);
    free(covered);
    return s;
}

uint64_t covering_survivors(const uint64_t *primes, const uint64_t *residues,
                            size_t n_primes, uint64_t gap_target,
                            uint64_t *survivors_out) {
    if (!primes || !residues || n_primes == 0 || gap_target < 2 || !survivors_out)
        return 0;
    uint8_t *covered = (uint8_t *)calloc(gap_target, 1);
    if (!covered)
        return 0;
    for (size_t i = 0; i < n_primes; i++)
        mark_cover(covered, gap_target, primes[i], residues[i]);
    uint64_t n = 0;
    for (uint64_t j = 1; j < gap_target; j++) {
        if (!covered[j])
            survivors_out[n++] = j;
    }
    free(covered);
    return n;
}

uint64_t covering_optimize(const uint64_t *primes, size_t n_primes,
                           uint64_t gap_target,
                           uint64_t *residues_out,
                           const struct covering_config *cfg) {
    if (!primes || !residues_out || n_primes == 0 || gap_target < 2 || !cfg)
        return 0;

    g_run_mode = cfg->run_objective != 0;
    g_lex_mode = cfg->lex_objective != 0;
    g_blocks_mode = cfg->blocks_objective != 0;
    g_blocks_D = cfg->difficulty_merit;
    g_blocks_logbase = cfg->logbase > 0.0 ? cfg->logbase : 1.0;
    g_rng_state = cfg->seed ? cfg->seed : 0x9e3779b97f4a7c15ULL;

    uint8_t *covered = (uint8_t *)malloc(gap_target);
    uint64_t *best_res = (uint64_t *)malloc(n_primes * sizeof(uint64_t));
    int *order = (int *)malloc(n_primes * sizeof(int));
    if (!covered || !best_res || !order) {
        free(covered);
        free(best_res);
        free(order);
        return 0;
    }

    uint32_t strength = cfg->strength ? cfg->strength : 1;
    uint32_t sweeps = cfg->local_sweeps;
    uint64_t best_score = UINT64_MAX;

    for (uint32_t s = 0; s < strength; s++) {
        /* Randomize prime order (Fisher-Yates) for diverse greedy restarts. */
        for (size_t i = 0; i < n_primes; i++)
            order[i] = (int)i;
        for (size_t i = n_primes - 1; i > 0; i--) {
            size_t j = (size_t)(rng_next() % (uint64_t)(i + 1));
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }

        greedy_pass(primes, n_primes, gap_target, residues_out, covered, order);

        for (uint32_t sw = 0; sw < sweeps; sw++)
            local_sweep(primes, n_primes, gap_target, residues_out);

        uint64_t sc = eval_solution(primes, residues_out, n_primes,
                                    gap_target, NULL, NULL);
        if (sc < best_score) {
            best_score = sc;
            memcpy(best_res, residues_out, n_primes * sizeof(uint64_t));
        }
    }

    memcpy(residues_out, best_res, n_primes * sizeof(uint64_t));
    if (g_run_mode || g_lex_mode)
        run_polish(primes, n_primes, gap_target, residues_out);

    free(covered);
    free(best_res);
    free(order);
    return covering_count_survivors(primes, residues_out, n_primes,
                                    gap_target);
}

/* ── Evolutionary optimizer (GapMiner gen_crt --ctr-evolution semantics) ── */

/* Sweep primes in [fixed, n_primes), leaving the first `fixed` untouched. */
static void local_sweep_fixed(const uint64_t *primes, size_t n_primes,
                              uint64_t gap_target, uint64_t *residues,
                              size_t fixed) {
    uint8_t *covered = (uint8_t *)malloc(gap_target);
    if (!covered)
        return;
    for (size_t i = fixed; i < n_primes; i++) {
        uint64_t p = primes[i];
        memset(covered, 0, gap_target);
        for (size_t k = 0; k < n_primes; k++) {
            if (k != i)
                mark_cover(covered, gap_target, primes[k], residues[k]);
        }
        uint64_t best_r = residues[i];
        uint64_t best_c = count_cover(covered, gap_target, p, best_r);
        for (uint64_t r = 1; r < p; r++) {
            uint64_t c = count_cover(covered, gap_target, p, r);
            if (c > best_c) {
                best_c = c;
                best_r = r;
            }
        }
        residues[i] = best_r;
    }
    free(covered);
}

static size_t tournament_select(const uint64_t *score, uint32_t pop_size) {
    size_t a = (size_t)(rng_next() % pop_size);
    size_t b = (size_t)(rng_next() % pop_size);
    return (score[a] <= score[b]) ? a : b;   /* lower survivor count wins */
}

uint64_t covering_optimize_evolution(const uint64_t *primes, size_t n_primes,
                                     uint64_t gap_target,
                                     uint64_t *residues_out,
                                     const struct covering_evo_config *cfg) {
    if (!primes || !residues_out || n_primes == 0 || gap_target < 2 || !cfg)
        return 0;

    g_run_mode = cfg->run_objective != 0;
    g_lex_mode = cfg->lex_objective != 0;
    g_blocks_mode = cfg->blocks_objective != 0;
    g_blocks_D = cfg->difficulty_merit;
    g_blocks_logbase = cfg->logbase > 0.0 ? cfg->logbase : 1.0;
    g_rng_state = cfg->seed ? cfg->seed : 0x9e3779b97f4a7c15ULL;

    uint32_t pop_size = cfg->population ? cfg->population : 20;
    uint32_t strength = cfg->strength ? cfg->strength : 8;
    uint32_t sweeps = cfg->local_sweeps ? cfg->local_sweeps : 4;
    size_t fixed = cfg->fixed;
    if (fixed > n_primes)
        fixed = n_primes;

    uint8_t *covered = (uint8_t *)malloc(gap_target);
    int *order = (int *)malloc(n_primes * sizeof(int));
    uint64_t **pop = (uint64_t **)malloc((size_t)pop_size * sizeof(uint64_t *));
    uint64_t *score = (uint64_t *)malloc((size_t)pop_size * sizeof(uint64_t));
    uint64_t *child = (uint64_t *)malloc(n_primes * sizeof(uint64_t));
    if (!covered || !order || !pop || !score || !child) {
        free(covered);
        free(order);
        free(pop);
        free(score);
        free(child);
        return 0;
    }
    for (uint32_t i = 0; i < pop_size; i++)
        pop[i] = (uint64_t *)malloc(n_primes * sizeof(uint64_t));

    /* Initial population: `strength` diverse greedy solutions, the rest
     * random residues (all refined by a local sweep). */
    for (uint32_t i = 0; i < pop_size; i++) {
        if (pop[i] == NULL) {
            score[i] = UINT64_MAX;
            continue;
        }
        if (i < strength) {
            for (size_t k = 0; k < n_primes; k++)
                order[k] = (int)k;
            for (size_t k = n_primes - 1; k > 0; k--) {
                size_t j = (size_t)(rng_next() % (uint64_t)(k + 1));
                int t = order[k];
                order[k] = order[j];
                order[j] = t;
            }
            greedy_pass(primes, n_primes, gap_target, pop[i], covered, order);
        } else {
            for (size_t k = 0; k < n_primes; k++)
                pop[i][k] = 1 + (uint64_t)(rng_next() % (primes[k] - 1));
        }
        for (uint32_t s = 0; s < sweeps; s++)
            local_sweep_fixed(primes, n_primes, gap_target, pop[i], fixed);
        score[i] = eval_solution(primes, pop[i], n_primes, gap_target,
                                 NULL, NULL);
    }

    /* Evolution: tournament selection + crossover + mutation + local search,
     * replacing the worst individual when the child is better. */
    for (uint32_t gen = 0; gen < cfg->generations; gen++) {
        size_t a = tournament_select(score, pop_size);
        size_t b = tournament_select(score, pop_size);
        for (size_t k = 0; k < n_primes; k++)
            child[k] = (rng_next() & 1u) ? pop[a][k] : pop[b][k];

        /* Mutate non-fixed primes at a ~20% rate. */
        if (fixed < n_primes) {
            for (size_t k = fixed; k < n_primes; k++) {
                if ((rng_next() % 100u) < 20u)
                    child[k] = 1 + (uint64_t)(rng_next() % (primes[k] - 1));
            }
        }

        for (uint32_t s = 0; s < sweeps; s++)
            local_sweep_fixed(primes, n_primes, gap_target, child, fixed);

        uint64_t child_score = eval_solution(primes, child, n_primes,
                                             gap_target, NULL, NULL);
        size_t worst = 0;
        for (uint32_t i = 1; i < pop_size; i++)
            if (score[i] > score[worst])
                worst = i;
        if (child_score < score[worst]) {
            memcpy(pop[worst], child, n_primes * sizeof(uint64_t));
            score[worst] = child_score;
        }
    }

    size_t best = 0;
    for (uint32_t i = 1; i < pop_size; i++)
        if (score[i] < score[best])
            best = i;
    memcpy(residues_out, pop[best], n_primes * sizeof(uint64_t));

    /* Iterated local search: perturb 3-5 residues, re-sweep, keep if better. */
    for (uint32_t r = 0; r < cfg->ils_rounds; r++) {
        memcpy(child, residues_out, n_primes * sizeof(uint64_t));
        if (fixed < n_primes) {
            int k = 3 + (int)(rng_next() % 3u);
            for (int t = 0; t < k; t++) {
                size_t j = fixed + (size_t)(rng_next() % (uint64_t)(n_primes - fixed));
                child[j] = 1 + (uint64_t)(rng_next() % (primes[j] - 1));
            }
        }
        for (uint32_t s = 0; s < sweeps; s++)
            local_sweep_fixed(primes, n_primes, gap_target, child, fixed);
        uint64_t cs = eval_solution(primes, child, n_primes, gap_target,
                                    NULL, NULL);
        if (cs < score[best]) {
            memcpy(residues_out, child, n_primes * sizeof(uint64_t));
            score[best] = cs;
        }
    }

    if (g_run_mode || g_lex_mode)
        run_polish(primes, n_primes, gap_target, residues_out);

    uint64_t final = covering_count_survivors(primes, residues_out, n_primes,
                                              gap_target);

    for (uint32_t i = 0; i < pop_size; i++)
        free(pop[i]);
    free(covered);
    free(order);
    free(pop);
    free(score);
    free(child);
    return final;
}

uint64_t covering_survivors_run_ge(const uint64_t *primes,
                                  const uint64_t *residues,
                                  size_t n_primes, uint64_t window,
                                  uint64_t min_run) {
    if (!primes || !residues || n_primes == 0 || window < 2 || min_run == 0)
        return 0;
    uint8_t *covered = (uint8_t *)calloc(window, 1);
    if (!covered)
        return 0;
    for (size_t i = 0; i < n_primes; i++)
        mark_cover(covered, window, primes[i], residues[i]);

    uint64_t n = 0;
    uint64_t j = 1;
    while (j < window) {
        if (!covered[j]) {
            uint64_t r = 0;
            uint64_t t = j + 1;
            while (t < window && covered[t]) {
                t++;
                r++;
            }
            if (r >= min_run)
                n++;
            j++;
            continue;
        }
        j++;
    }
    free(covered);
    return n;
}
