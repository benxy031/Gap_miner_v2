/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Covering-system residue optimizer.
 *
 * The CRT offset optimization is the engine behind the "CRT pre-selection
 * boost": choose residues r_p (1 <= r_p < p) so that the interior offsets
 * j in [1, gap_target) are covered (forced composite) by as many primes as
 * possible.  Position j is covered iff (r_p + j) % p == 0 for some prime p.
 *
 * This is NOT a plain sieve: it concentrates composite coverage inside the
 * target gap window.  At a design merit well above the mining merit, the
 * covering forces long composite runs, making the mined prime sequence
 * sparser and its gap tail heavier than the Cramer e^-m model.
 */

#ifndef COVERING_H
#define COVERING_H

#include <stdint.h>
#include <stddef.h>

struct covering_config {
    uint32_t strength;      /* greedy restarts (randomized prime order) */
    uint32_t local_sweeps;  /* single-prime local-search passes after greedy */
    uint32_t ils_rounds;    /* iterated-local-search perturbation rounds */
    int pair_search;        /* reserved: pair local search (unused for now) */
    int run_objective;      /* 1 = maximize longest covered run (tie-break
                               by survivor count), 0 = minimize survivors */
    int lex_objective;      /* 1 = LEXICOGRAPHIC: minimize survivors first,
                               then maximize longest covered run.  This is
                               the mining-safe arrangement objective (the
                               run-first variant was measured worse). */
    uint64_t seed;          /* deterministic seed (0 = fixed default) */
};

/* Evolutionary covering config (GapMiner gen_crt --ctr-* semantics). */
struct covering_evo_config {
    uint32_t strength;      /* greedy restarts feeding the initial population */
    uint32_t population;    /* --ctr-ivs: individuals kept alive */
    uint32_t generations;   /* evolution rounds (tournament+crossover+mutation) */
    uint32_t local_sweeps;  /* single-prime local-search passes per refinement */
    uint32_t fixed;         /* --ctr-fixed: first N primes frozen (2,3,5,7..) */
    uint32_t ils_rounds;    /* perturbation rounds after evolution */
    int run_objective;      /* same as covering_config.run_objective */
    int lex_objective;      /* same as covering_config.lex_objective */
    uint64_t seed;
};

/* Greedy + evolutionary + ILS covering optimization.
 * Returns survivor count; fills residues_out (n_primes entries). */
uint64_t covering_optimize_evolution(const uint64_t *primes, size_t n_primes,
                                     uint64_t gap_target,
                                     uint64_t *residues_out,
                                     const struct covering_evo_config *cfg);

/* Optimize residues r_p (0 < r_p < p).  Returns survivor count
 * (uncovered offsets in [1, gap_target)), or 0 on allocation failure.
 * residues_out must hold n_primes entries. */
uint64_t covering_optimize(const uint64_t *primes, size_t n_primes,
                           uint64_t gap_target,
                           uint64_t *residues_out,
                           const struct covering_config *cfg);

/* Build the survivor offset list for a fixed residue set.
 * survivors_out must hold gap_target entries (worst case); returns the
 * number of survivors written.  Survivor offsets are ascending in [1,
 * gap_target). */
uint64_t covering_survivors(const uint64_t *primes, const uint64_t *residues,
                            size_t n_primes, uint64_t gap_target,
                            uint64_t *survivors_out);

/* Count survivors for a fixed residue set (no list output). */
uint64_t covering_count_survivors(const uint64_t *primes,
                                  const uint64_t *residues,
                                  size_t n_primes, uint64_t gap_target);

/* Length of the longest contiguous covered run in [1, gap_target) for a
 * fixed residue set (0 if allocation fails). */
uint64_t covering_longest_run(const uint64_t *primes,
                              const uint64_t *residues,
                              size_t n_primes, uint64_t gap_target);

#endif /* COVERING_H */
