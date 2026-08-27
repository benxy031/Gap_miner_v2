/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * RPC Interface (Phase 5)
 *
 * Wrapper for stratum.c (existing cpugapminer code):
 * - Connect to Gapcoin mining pool
 * - Receive work (getwork or getblocktemplate)
 * - Submit shares (gap solutions)
 * - Handle work updates and reconnection
 */

#ifndef RPC_INTERFACE_H
#define RPC_INTERFACE_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* RPC work context */
struct rpc_work {
    uint8_t block_header[80];         /* Block header (h256 + nonce offset) */
    uint32_t height;                  /* Block height */
    char *pool_url;                   /* Pool endpoint (e.g., "stratum.gapcoin.org") */
    char *worker_name;                /* Worker identifier */
    uint32_t shift;                   /* Shift factor from pool */
};

/* RPC submission result */
struct rpc_result {
    int accepted;                     /* 1 if share accepted, 0 if rejected */
    char *message;                    /* Pool response message */
    uint64_t difficulty;              /* Accepted difficulty level */
};

/* Initialize RPC connection */
int rpc_connect(const char *pool_url, const char *worker_name);

/* Get current work from pool */
int rpc_get_work(struct rpc_work *work);

/* Submit gap solution */
int rpc_submit_gap(struct rpc_result *result,
                  const struct rpc_work *work,
                  uint64_t offset_p1,
                  uint64_t offset_p2,
                  uint32_t gap_length);

/* Check for work update */
int rpc_work_updated(void);

/* Close RPC connection */
void rpc_disconnect(void);

#endif /* RPC_INTERFACE_H */
