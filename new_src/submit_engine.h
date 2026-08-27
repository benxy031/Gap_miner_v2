/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Submit Engine (Phase 5)
 *
 * Payload assembly for gap submissions:
 * - getwork mode: assemble nonce-based submission
 * - getblocktemplate mode: assemble full block template
 * - CoinBase extra data: pack gap merit and verification
 */

#ifndef SUBMIT_ENGINE_H
#define SUBMIT_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Gap submission metadata */
struct gap_submission {
    uint64_t offset_p1;               /* First prime offset */
    uint64_t offset_p2;               /* Second prime offset */
    uint32_t gap_length;              /* p2 - p1 */
    double merit;                     /* Gap merit score */
    uint8_t gap_verified;             /* BPSW verification status */
};

/* Payload formats */
enum submit_mode {
    SUBMIT_GETWORK = 0,               /* Simple nonce-based (legacy) */
    SUBMIT_GETBLOCKTEMPLATE = 1,      /* Full block template (modern) */
};

/* Build submission payload */
int submit_engine_build_payload(
    enum submit_mode mode,
    const uint8_t *block_header,      /* Block header from pool */
    const struct gap_submission *gap,
    uint8_t *payload_out,
    size_t *payload_len
);

/* Validate submission before sending */
int submit_engine_validate_payload(
    const uint8_t *payload,
    size_t payload_len,
    const struct gap_submission *gap
);

#endif /* SUBMIT_ENGINE_H */
