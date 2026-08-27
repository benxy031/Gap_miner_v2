/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Covering-system residue optimizer (see covering.h).
 */

#include "covering.h"

#include <stdlib.h>
#include <string.h>

static uint64_t g_rng_state;

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
    uint64_t best_surv = UINT64_MAX;

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

        uint64_t surv = covering_count_survivors(primes, residues_out,
                                                 n_primes, gap_target);
        if (surv < best_surv) {
            best_surv = surv;
            memcpy(best_res, residues_out, n_primes * sizeof(uint64_t));
        }
    }

    memcpy(residues_out, best_res, n_primes * sizeof(uint64_t));

    free(covered);
    free(best_res);
    free(order);
    return best_surv;
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
        score[i] = covering_count_survivors(primes, pop[i], n_primes,
                                            gap_target);
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

        uint64_t child_score = covering_count_survivors(primes, child,
                                                        n_primes, gap_target);
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
        uint64_t cs = covering_count_survivors(primes, child, n_primes,
                                               gap_target);
        if (cs < score[best]) {
            memcpy(residues_out, child, n_primes * sizeof(uint64_t));
            score[best] = cs;
        }
    }

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
