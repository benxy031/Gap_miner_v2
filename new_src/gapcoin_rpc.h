/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Real RPC Client for Gapcoin (Phase 2)
 *
 * Communicates with actual Gapcoin node via JSON-RPC 2.0
 * Dependencies: libcurl, jansson (JSON)
 *
 * Usage:
 *   struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
 *                                                "benxy031", "xx");
 *   struct block_template *tmpl = gapcoin_rpc_get_block_template(rpc);
 *   gapcoin_rpc_submit_gap(rpc, &gap_info);
 */

#ifndef GAPCOIN_RPC_H
#define GAPCOIN_RPC_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Gapcoin RPC connection handle */
struct gapcoin_rpc {
    char *host;
    uint16_t port;
    char *username;
    char *password;
};

/* Block template from Gapcoin */
struct block_template {
    char *previousblockhash;      /* Hex string */
    uint32_t height;
    uint32_t curtime;
    char *bits;                   /* Difficulty bits, retained for logging */
    uint64_t difficulty;          /* Full 64-bit Gapcoin difficulty */
    uint64_t coinbasevalue;
    char *coinbaseaux;            /* Required coinbase scriptSig data */
    char **transaction_hashes;    /* Transaction IDs in template order */
    char **transaction_data;      /* Raw tx hex, required to reassemble a submittable block */
    size_t transaction_count;
    uint32_t version;
    char *longpollid;             /* For longpoll support */
};

#define GAPCOIN_WORK_HEADER_SIZE 84U

/* getwork data before the trailing shift and adder fields. */
struct gapcoin_work {
    uint8_t header[GAPCOIN_WORK_HEADER_SIZE];
    uint16_t shift;
    uint64_t target;
    uint32_t nonce;
};

/* Gap submission info */
struct gap_info {
    uint32_t height;
    uint32_t shift;
    uint64_t adder;               /* The adder used for gap */
    uint64_t offset_p1;
    uint64_t offset_p2;
    uint32_t gap_length;
    double merit;
};

/* Mining info from Gapcoin */
struct mining_info {
    uint32_t blocks;
    double difficulty;
    double networkminingpower;
    double profitability;
    struct {
        uint32_t height;
        char *bits;
        double difficulty;
    } next;
};

/* Connect to Gapcoin RPC */
struct gapcoin_rpc *gapcoin_rpc_connect(const char *host, uint16_t port,
                                        const char *username, 
                                        const char *password);

/* Get current block template */
struct block_template *gapcoin_rpc_get_block_template(struct gapcoin_rpc *rpc);

/* Get a full Gapcoin header suitable for nonce rotation and hashing. */
int gapcoin_rpc_get_work(struct gapcoin_rpc *rpc, struct gapcoin_work *work);

/* Get mining info */
struct mining_info *gapcoin_rpc_get_mining_info(struct gapcoin_rpc *rpc);

/* Submit gap solution */
int gapcoin_rpc_submit_gap(struct gapcoin_rpc *rpc, 
                          const struct gap_info *gap);

/* Submit block (raw hex) */
int gapcoin_rpc_submit_block(struct gapcoin_rpc *rpc, 
                            const char *block_hex);

/* Get block count */
uint32_t gapcoin_rpc_get_block_count(struct gapcoin_rpc *rpc);

/* 
 * Blocking longpoll call - waits for new block from wallet.
 * Pass longpoll_id from previous template to wait for next block.
 * Returns when wallet detects new block, or NULL on timeout/error.
 */
struct block_template *gapcoin_rpc_get_block_template_longpoll(
    struct gapcoin_rpc *rpc,
    const char *longpoll_id);

/* Cleanup */
void gapcoin_rpc_free(struct gapcoin_rpc *rpc);
void block_template_free(struct block_template *tmpl);
void mining_info_free(struct mining_info *info);

#endif /* GAPCOIN_RPC_H */
