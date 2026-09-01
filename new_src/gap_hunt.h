/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GAP_HUNT — Gapcoin-independent record-hunting mode.
 *
 * The CRT cover of a design file (e.g. shift507_p74_lex_m30) is periodic
 * with period P (the product of the cover primes).  For ANY even base b0
 * with b0 ≡ -(o_i + adj) (mod p_i), every translate b_k = b0 + k*P carries
 * the IDENTICAL cover template: candidates = b_k + survivor_offsets, the
 * same sigma-conditioned gap distribution the miner exploits.  This mode
 * walks k = 0,1,2,... with the fused GPU pipeline (device sieve + CGBN MR),
 * full-class (every sieve survivor is MR-tested), chaining consecutive
 * primes across window boundaries, and reports every gap with merit >=
 * min_merit.  No PoW structure: no hashing, no headers, no difficulty,
 * no submissions.
 */

#ifndef GAP_HUNT_H
#define GAP_HUNT_H

#include <stdint.h>

struct gap_hunt_config {
    const char *crt_file;      /* design CRT file (required) */
    const char *start_hex;     /* optional base anchor (hex); NULL = 2^762 */
    double min_merit;          /* report gaps with merit >= min_merit */
    const char *state_path;    /* resume state file (optional) */
    const char *out_path;      /* results file (optional, stdout-only if NULL) */
    uint32_t sieve_primes;     /* deep-sieve limit (2M default) */
    int device;                /* CUDA device id */
};

/* Run the walk until SIGINT/SIGTERM.  Returns 0 on clean shutdown. */
int gap_hunt_run(const struct gap_hunt_config *cfg);

#endif /* GAP_HUNT_H */
