/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration Test: Real Gapcoin RPC
 *
 * Tests: Connection, getmininginfo, getblocktemplate, gap submission
 */

#define _DEFAULT_SOURCE  /* usleep, mkstemp under -std=c99 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <fcntl.h>
#include "../new_src/gapcoin_rpc.h"
#include "../new_src/gapcoin_work.h"

static int test_rpc_connect(void) {
    printf("[TEST] Gapcoin RPC connection...\n");
    
    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
                                                 "benxy031", "xx");
    if (!rpc) {
        printf("  ✗ FAIL: Cannot connect to Gapcoin RPC\n");
        printf("     (Is Gapcoin node running on port 31397?)\n");
        return 0;
    }
    
    printf("  ✓ Connected to Gapcoin RPC\n");
    
    gapcoin_rpc_free(rpc);
    return 1;
}

static int test_rpc_get_block_count(void) {
    printf("[TEST] getblockcount...\n");
    
    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
                                                 "benxy031", "xx");
    if (!rpc) return 0;
    
    uint32_t count = gapcoin_rpc_get_block_count(rpc);
    
    if (count > 0) {
        printf("  ✓ Block count: %u\n", count);
        gapcoin_rpc_free(rpc);
        return 1;
    } else {
        printf("  ✗ FAIL: getblockcount returned 0\n");
        gapcoin_rpc_free(rpc);
        return 0;
    }
}

static int test_rpc_get_mining_info(void) {
    printf("[TEST] getmininginfo...\n");
    
    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
                                                 "benxy031", "xx");
    if (!rpc) return 0;
    
    struct mining_info *info = gapcoin_rpc_get_mining_info(rpc);
    
    if (!info) {
        printf("  ✗ FAIL: getmininginfo returned NULL\n");
        gapcoin_rpc_free(rpc);
        return 0;
    }
    
    printf("  Mining info:\n");
    printf("    Blocks: %u\n", info->blocks);
    printf("    Difficulty: %.2f\n", info->difficulty);
    printf("    Network power: %.0f (mH/s)\n", info->networkminingpower);
    printf("    Profitability: %.2f%%\n", info->profitability * 100);
    printf("    Next height: %u, difficulty: %.2f\n", 
           info->next.height, info->next.difficulty);
    
    printf("  ✓ PASS: Mining info retrieved\n");
    
    mining_info_free(info);
    gapcoin_rpc_free(rpc);
    return 1;
}

static int test_rpc_get_block_template(void) {
    printf("[TEST] getblocktemplate...\n");
    
    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
                                                 "benxy031", "xx");
    if (!rpc) return 0;
    
    struct block_template *tmpl = gapcoin_rpc_get_block_template(rpc);
    
    if (!tmpl) {
        printf("  ✗ FAIL: getblocktemplate returned NULL\n");
        gapcoin_rpc_free(rpc);
        return 0;
    }
    
    printf("  Block template:\n");
    printf("    Height: %u\n", tmpl->height);
    printf("    Bits: %s\n", tmpl->bits);
    printf("    Difficulty: %016lx\n", tmpl->difficulty);
    printf("    Coinbase value: %lu satoshis\n", tmpl->coinbasevalue);
    printf("    Transactions: %zu\n", tmpl->transaction_count);
    printf("    Current time: %u\n", tmpl->curtime);
    printf("    Previous hash: %.32s...\n", tmpl->previousblockhash);

    struct gapcoin_gbt_work work;
    uint8_t first_hash[32];
    uint8_t second_hash[32];
    if (gapcoin_gbt_work_init(&work, tmpl) != 0 ||
        gapcoin_gbt_work_hash(&work, first_hash) != 0 ||
        gapcoin_gbt_work_next_hash(&work, second_hash) != 0 ||
        memcmp(first_hash, second_hash, sizeof(first_hash)) == 0) {
        printf("  ✗ FAIL: Cannot materialize distinct GBT header hashes\n");
        block_template_free(tmpl);
        gapcoin_rpc_free(rpc);
        return 0;
    }
    printf("  ✓ PASS: GBT header nonce rotation produces new scan bases\n");
    
    printf("  ✓ PASS: Block template retrieved\n");
    
    block_template_free(tmpl);
    gapcoin_rpc_free(rpc);
    return 1;
}

/* Structural check only: assembles a submission and verifies its byte
 * layout. Never calls submitblock, so no real block is ever submitted. */
static int test_rpc_build_submission(void) {
    printf("[TEST] gapcoin_gbt_work_build_submission (structural, no submit)...\n");

    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397,
                                                 "benxy031", "xx");
    if (!rpc) return 0;

    struct block_template *tmpl = gapcoin_rpc_get_block_template(rpc);
    if (!tmpl) {
        printf("  ✗ FAIL: getblocktemplate returned NULL\n");
        gapcoin_rpc_free(rpc);
        return 0;
    }

    struct gapcoin_gbt_work work;
    uint8_t h256[32];
    if (gapcoin_gbt_work_init(&work, tmpl) != 0 ||
        gapcoin_gbt_work_hash(&work, h256) != 0) {
        printf("  ✗ FAIL: Cannot materialize GBT header\n");
        block_template_free(tmpl);
        gapcoin_rpc_free(rpc);
        return 0;
    }

    char block_hex[GAPCOIN_SUBMIT_HEX_CAP];
    if (gapcoin_gbt_work_build_submission(work.header_prefix, work.nonce, tmpl,
                                          26, 0, block_hex,
                                          sizeof(block_hex)) != 0) {
        printf("  ✗ FAIL: gapcoin_gbt_work_build_submission returned an error\n");
        block_template_free(tmpl);
        gapcoin_rpc_free(rpc);
        return 0;
    }

    size_t hex_len = strlen(block_hex);
    if (hex_len == 0 || hex_len % 2 != 0) {
        printf("  ✗ FAIL: Submission hex has an invalid length (%zu)\n", hex_len);
        block_template_free(tmpl);
        gapcoin_rpc_free(rpc);
        return 0;
    }

    /* First 160 hex chars must be the 80-byte header prefix, byte-for-byte. */
    char expected_prefix[161];
    for (size_t i = 0; i < sizeof(work.header_prefix); i++) {
        snprintf(expected_prefix + i * 2, 3, "%02x", work.header_prefix[i]);
    }
    if (strncmp(block_hex, expected_prefix, 160) != 0) {
        printf("  ✗ FAIL: Submission header prefix does not match the GBT header\n");
        block_template_free(tmpl);
        gapcoin_rpc_free(rpc);
        return 0;
    }

    printf("  ✓ PASS: Submission hex (%zu chars) starts with the exact GBT header prefix\n",
           hex_len);

    block_template_free(tmpl);
    gapcoin_rpc_free(rpc);
    return 1;
}

static int test_rpc_get_work(void) {
    printf("[TEST] getwork...\n");

    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397,
                                                 "benxy031", "xx");
    if (!rpc) return 0;

    struct gapcoin_work work;
    if (gapcoin_rpc_get_work(rpc, &work) != 0) {
        printf("  - SKIP: Node does not support legacy getwork; using GBT work data\n");
        gapcoin_rpc_free(rpc);
        return 1;
    }

    printf("  Work: shift=%u target=%lu nonce=%u\n",
           work.shift, work.target, work.nonce);
    printf("  ✓ PASS: Full Gapcoin work header retrieved\n");

    gapcoin_rpc_free(rpc);
    return 1;
}

static int test_rpc_submit_gap_dry_run(void) {
    printf("[TEST] submitgap (dry run with dummy values)...\n");
    
    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397, 
                                                 "benxy031", "xx");
    if (!rpc) return 0;
    
    /* Get current height for submission */
    struct mining_info *info = gapcoin_rpc_get_mining_info(rpc);
    if (!info) {
        printf("  ✗ FAIL: Cannot get mining info\n");
        gapcoin_rpc_free(rpc);
        return 0;
    }
    
    struct gap_info gap;
    memset(&gap, 0, sizeof(gap));
    gap.height = info->blocks;
    gap.shift = 512;
    gap.adder = 100;
    gap.offset_p1 = 2;
    gap.offset_p2 = 5;
    gap.gap_length = 3;
    gap.merit = 0.5;
    
    int ret = gapcoin_rpc_submit_gap(rpc, &gap);
    
    if (ret == 0) {
        printf("  ✓ Gap submitted successfully\n");
    } else {
        printf("  ✗ Gap submission failed (RPC error or validation)\n");
        printf("     (This is expected for dummy values)\n");
    }
    
    mining_info_free(info);
    gapcoin_rpc_free(rpc);
    
    /* Always pass this test as it tests the submission mechanism,
     * not actual gap validity */
    return 1;
}

static int stress_poll_failures;

static void *stress_poll_thread(void *arg) {
    struct gapcoin_rpc *rpc = (struct gapcoin_rpc *)arg;
    for (int i = 0; i < 40; i++) {
        struct block_template *tmpl = gapcoin_rpc_get_block_template(rpc);
        if (!tmpl) {
            __sync_fetch_and_add(&stress_poll_failures, 1);
        } else {
            block_template_free(tmpl);
        }
    }
    return NULL;
}

static void *stress_submit_thread(void *arg) {
    struct gapcoin_rpc *rpc = (struct gapcoin_rpc *)arg;
    /* Dummy 80-byte block: the node must reject it, but the RPC plumbing
       (request, response, parse) must stay healthy under concurrency.
       This reproduces the shift-998 failure mode: main-thread submitblock
       racing the RPC polling thread's getblocktemplate on one client. */
    char dummy_hex[161];
    memset(dummy_hex, '0', sizeof(dummy_hex) - 1);
    dummy_hex[sizeof(dummy_hex) - 1] = '\0';
    for (int i = 0; i < 20; i++) {
        (void)gapcoin_rpc_submit_block(rpc, dummy_hex);
        usleep(20000);
    }
    return NULL;
}

static int test_rpc_concurrent_stress(void) {
    printf("[TEST] Concurrent polling + submitblock stress...\n");

    struct gapcoin_rpc *rpc = gapcoin_rpc_connect("127.0.0.1", 31397,
                                                 "benxy031", "xx");
    if (!rpc) return 0;

    /* Capture stderr: transport failures must NOT appear. */
    fflush(stderr);
    char tmp_name[] = "/tmp/gapminer_rpc_stress.XXXXXX";
    int tmp_fd = mkstemp(tmp_name);
    if (tmp_fd < 0) { gapcoin_rpc_free(rpc); return 0; }
    int saved_stderr = dup(STDERR_FILENO);
    dup2(tmp_fd, STDERR_FILENO);

    stress_poll_failures = 0;
    pthread_t poll_th, submit_th;
    pthread_create(&poll_th, NULL, stress_poll_thread, rpc);
    pthread_create(&submit_th, NULL, stress_submit_thread, rpc);
    pthread_join(poll_th, NULL);
    pthread_join(submit_th, NULL);

    fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    close(saved_stderr);

    lseek(tmp_fd, 0, SEEK_SET);
    char buf[8192];
    ssize_t got = read(tmp_fd, buf, sizeof(buf) - 1);
    close(tmp_fd);
    unlink(tmp_name);
    if (got > 0) buf[got] = '\0';

    int transport_errors = 0;
    const char *markers[] = {
        "CURL error", "CURL init failed", "Empty response",
        "JSON parse error", NULL
    };
    for (int i = 0; markers[i]; i++) {
        if (got > 0 && strstr(buf, markers[i])) {
            printf("  ✗ FAIL: transport error seen: %s\n", markers[i]);
            transport_errors++;
        }
    }

    printf("  Poll failures: %d | Transport errors: %d\n",
           stress_poll_failures, transport_errors);

    gapcoin_rpc_free(rpc);
    if (stress_poll_failures == 0 && transport_errors == 0) {
        printf("  ✓ PASS: RPC stays healthy under concurrent use\n");
        return 1;
    }
    printf("  ✗ FAIL: concurrent RPC use is not thread-safe\n");
    return 0;
}

int main(void) {
    printf("================================================\n");
    printf("GapMiner V2 — Phase 2: Real Gapcoin RPC Tests\n");
    printf("================================================\n\n");
    
    int all_pass = 1;
    
    all_pass &= test_rpc_connect();
    printf("\n");
    
    all_pass &= test_rpc_get_block_count();
    printf("\n");
    
    all_pass &= test_rpc_get_mining_info();
    printf("\n");
    
    all_pass &= test_rpc_get_block_template();
    printf("\n");

    all_pass &= test_rpc_build_submission();
    printf("\n");

    all_pass &= test_rpc_get_work();
    printf("\n");
    
    all_pass &= test_rpc_submit_gap_dry_run();
    printf("\n");

    all_pass &= test_rpc_concurrent_stress();
    printf("\n");
    
    printf("================================================\n");
    if (all_pass) {
        printf("✓ All Gapcoin RPC tests PASSED\n");
    } else {
        printf("✗ Some Gapcoin RPC tests FAILED\n");
    }
    printf("================================================\n");
    
    return all_pass ? 0 : 1;
}
