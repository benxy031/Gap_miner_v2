/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HALF_CLASS two-pass scan (non-CRT mode).
 *
 * Idea (from briankehrig/prime-gaps-cuda, WORD_LENGTH=240 / WORD_SIEVING_LENGTH
 * =120): sieve and primality-test only 8 of the 16 residue classes coprime to
 * 60 — the "visible" classes {1,7,11,13,17,19,23,29} mod 60.  Primes are
 * equidistributed across the 16 classes, so the candidate load halves.  The
 * hidden classes {31,37,41,43,47,49,53,59} mod 60 are never scanned, except
 * that every visible gap >= threshold is a SUPERSET of any true gap >=
 * threshold (a hidden endpoint only enlarges the visible gap), so the
 * interval between its endpoints is verified on demand: mini-sieved in the
 * hidden classes only and MR-tested.  No true qualifying gap can be missed.
 */

#ifndef HALFCLASS_H
#define HALFCLASS_H

#include <stdint.h>
#include <gmp.h>
#include "gap_detection.h"

/* CRT covering template (optional prefilter for verification): odd-slot
   composite bitmap over template positions t in [1, window) relative to the
   CRT-aligned base.  Absolute scan offset o maps to t = o - base_off.
   A NULL/zero template disables the prefilter (plain mini-sieve only). */
struct halfclass_tpl {
    const uint8_t *bits;    /* odd-slot composite bitmap (byte-packed) */
    uint64_t words;         /* bitmap word count (bits[slot >> 3] layout) */
    int64_t  base_off;      /* absolute offset mapping to template t = 0 */
    uint64_t window;        /* template covers t in [1, window) */
};

/* The 60-bit mask of the visible classes {1,7,11,13,17,19,23,29} mod 60. */
uint64_t halfclass_visible_mask(void);

/* The 60-bit mask of the hidden classes {31,37,41,43,47,49,53,59} mod 60. */
uint64_t halfclass_hidden_mask(void);

/* base mod 60 for a non-CRT window: base = (h256 << shift) + window_start. */
uint32_t halfclass_base_mod60(const uint8_t h256[32], uint32_t shift,
                              uint64_t window_start);

/* 1 when (base + offset) mod 60 is one of the visible classes. */
int halfclass_offset_visible(uint32_t base_mod60, uint64_t offset);

/* In-place compaction of candidate offsets to the visible classes only.
   Returns the new count (<= old count).  Offsets stay ascending. */
uint32_t halfclass_filter_offsets(uint32_t base_mod60, uint64_t *offsets,
                                  uint32_t count);

/* Region-aware variant: offsets below region_start are never filtered
   (the CRT back-lookahead must be scanned in all classes). */
uint32_t halfclass_filter_offsets_region(uint32_t base_mod60, uint64_t *offsets,
                                         uint32_t count, uint64_t region_start);

/* Event-driven verification of a visible-gap candidate [off_a, off_b):
   mini-sieves the hidden classes in the open interval with primes <= 100k,
   MR-tests the survivors (GMP), and emits the true consecutive-prime gaps
   with merit >= merit_threshold whose FIRST endpoint is owned
   (offset < owned_offset_limit).  Returns 1 on success (out/out_count set,
   may be empty), 0 on internal failure (caller must drop the candidate). */
int halfclass_resolve_gap(mpz_t base, uint64_t off_a, uint64_t off_b,
                          uint64_t owned_offset_limit, double merit_threshold,
                          struct gap_result **out, uint32_t *out_count);

/* Template-aware variants: tpl == NULL behaves like the plain versions.
   The template prefilter is applied on top of the mini-sieve: a candidate
   proven composite by the covering is never MR-tested. */
int halfclass_resolve_gap_ex(mpz_t base, uint64_t off_a, uint64_t off_b,
                             uint64_t owned_offset_limit,
                             double merit_threshold,
                             const struct halfclass_tpl *tpl,
                             struct gap_result **out, uint32_t *out_count);

int halfclass_verify_prefix_ex(mpz_t base, uint64_t off_v0,
                               int terminal_is_prime,
                               uint64_t owned_offset_limit,
                               double merit_threshold,
                               const struct halfclass_tpl *tpl,
                               struct gap_result **out, uint32_t *out_count);

/* Emit merit-qualified, first-endpoint-owned gaps from a chain of
   consecutive probable primes given in absolute window offsets (used by the
   GPU-prefix CRT flow, which supplies the hidden prefix primes directly). */
uint32_t halfclass_emit_chain(mpz_t base, const uint64_t *chain,
                              uint32_t chain_len,
                              uint64_t owned_offset_limit,
                              double merit_threshold,
                              struct gap_result **out);

/* Prefix verification: the true primes before the first VISIBLE prime have
   no visible predecessor, so no visible superset gap can cover them.
   Verifies [0, off_v0) in the hidden classes and emits the consecutive-prime
   gaps among the hidden primes found plus off_v0 itself when
   terminal_is_prime (it is, when off_v0 is the first visible prime).
   Returns 1 on success (possibly empty), 0 on allocation failure. */
int halfclass_verify_prefix(mpz_t base, uint64_t off_v0, int terminal_is_prime,
                            uint64_t owned_offset_limit,
                            double merit_threshold,
                            struct gap_result **out, uint32_t *out_count);

#endif /* HALFCLASS_H */
