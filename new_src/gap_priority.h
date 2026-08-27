/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * ═════════════════════════════════════════════════════════════════════
 * ⚠️  UNUSED / DEPRECATED (2026-08-23)
 *
 * This module is NOT wired into the miner.  It implements an A* / best-first
 * candidate REORDERING for a SEQUENTIAL tester with early exit.  The current
 * architecture is GPU-batch based (candidates are tested in parallel), where
 * reordering saves nothing, and the early-exit idea is already implemented in
 * a GPU-compatible form via the CRT smart-scan tail skip, the optimal-tail
 * scan, and the non-CRT adaptive halo.
 *
 * Kept for reference only — the sole caller is tests/test_gap_priority.c.
 * ═════════════════════════════════════════════════════════════════════
 *
 * Gap-search priority heuristic (A-star / best-first ordering).
 *
 * For an unknown position u inside a scanned window, define:
 *
 *   Sc(u) = Lc + 2 + Rc              gap length that would FORM if u is
 *                                    composite (Lc/Rc = consecutive known
 *                                    composite counts to the left/right).
 *
 *   Sp(u) = max(V_left, V_right)     gap length that would CLOSE if u is
 *                                    prime (the larger of the two gaps it
 *                                    would delimit).
 *
 *   priority(u) = max(Sc(u), Sp(u)) / cost(u)
 *
 * This is a pure search REORDERING: it chooses WHICH unknown position to
 * test next, not WHICH positions to skip.  A tester visiting candidates in
 * priority order still discovers exactly the same primes/gaps as a linear
 * scan - it just reaches large gaps earlier on average (a constant-factor
 * reduction in wasted tests for a sequential tester).  It does not change
 * the number of primes in the interval (Dirichlet equidistribution), so it
 * cannot reduce the e^merit work bound.
 */

#ifndef GAP_PRIORITY_H
#define GAP_PRIORITY_H

#include <stddef.h>
#include <stdint.h>

/* Evaluate the heuristic for a single unknown position.  cost must be > 0. */
double gap_priority_value(uint64_t Lc, uint64_t Rc,
                          uint64_t V_left, uint64_t V_right,
                          double cost);

/* Static initial priority order over a window's survivor offsets.
 *
 * offsets[] must be strictly increasing, all in [0, interval_size).
 * For survivor u_i the "would-form" gap is Sc(u_i) = offsets[i+1] - offsets[i-1]
 * (the survivor-gap spanning u_i), using 0 and interval_size as the boundary
 * positions for the first and last survivor respectively.
 *
 * Fills out_order[] (size n) with a permutation of 0..n-1 ordered by Sc
 * descending.  Ties keep the original (increasing-offset) order, so the
 * result is deterministic.
 */
void gap_priority_order(const uint64_t *offsets, size_t n,
                        uint64_t interval_size,
                        size_t *out_order);

#endif /* GAP_PRIORITY_H */
