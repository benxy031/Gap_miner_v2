/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Integration Tests: Worker Threading
 *
 * Tests: Atomic nonce, worker spawning, lock-free race conditions
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include "../new_src/atomic_nonce.h"
#include "../new_src/worker_gpu.h"
#include "../new_src/miner_farm.h"

/* Test atomic nonce with multiple threads */
struct nonce_test_context {
    struct atomic_nonce *nonce;
    uint32_t thread_id;
    uint32_t acquired_count;
};

static void *nonce_test_thread(void *arg) {
    struct nonce_test_context *ctx = (struct nonce_test_context *)arg;
    
    /* Each thread acquires 100 nonces */
    for (int i = 0; i < 100; i++) {
        uint32_t nonce = atomic_nonce_next(ctx->nonce);
        if (nonce < ctx->nonce->max_value) {
            ctx->acquired_count++;
        }
    }
    
    return NULL;
}

static int test_atomic_nonce_multithread(void) {
    printf("[TEST] Atomic nonce with 4 threads...\n");
    
    struct atomic_nonce nonce;
    atomic_nonce_init(&nonce, 0, 1000);  /* 1000 nonces max */
    
    struct nonce_test_context contexts[4];
    pthread_t threads[4];
    
    /* Start 4 threads, each requesting 100 nonces */
    for (int i = 0; i < 4; i++) {
        contexts[i].nonce = &nonce;
        contexts[i].thread_id = i;
        contexts[i].acquired_count = 0;
        pthread_create(&threads[i], NULL, nonce_test_thread, &contexts[i]);
    }
    
    /* Wait for all threads */
    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* Check: all 400 nonces should be distributed uniquely */
    uint32_t total_acquired = 0;
    for (int i = 0; i < 4; i++) {
        total_acquired += contexts[i].acquired_count;
        printf("  Thread %d acquired %u nonces\n", i, contexts[i].acquired_count);
    }
    
    /* Verify: current nonce should be 400 (all distributed) */
    uint32_t current = atomic_load(&nonce.current);
    
    if (total_acquired == 400 && current == 400) {
        printf("  ✓ PASS: All 400 nonces distributed uniquely\n");
        return 1;
    } else {
        printf("  ✗ FAIL: Expected 400, got %u (current=%u)\n", total_acquired, current);
        return 0;
    }
}

static int test_shift_aware_windows(void) {
    printf("[TEST] Shift-aware work window geometry...\n");

    if (non_crt_owned_window_size(26) != 1048576 ||
        non_crt_windows_per_header(26) != 64 ||
        non_crt_owned_window_size(32) != 1048576 ||
        non_crt_windows_per_header(32) != 4096 ||
        non_crt_windows_per_header(49) != 536870912ULL ||
        non_crt_windows_per_header(50) != 1073741824ULL) {
        printf("  ✗ FAIL: Unexpected window geometry\n");
        return 0;
    }

    printf("  ✓ PASS: shift 26/32 use 1048576 adders/window (2^20 cap)\n");
    return 1;
}

/* Test farm creation and worker spawning */
static int test_miner_farm_creation(void) {
    printf("[TEST] Miner farm creation with 2 workers...\n");
    
    struct miner_farm_config config;
    memset(&config, 0, sizeof(config));
    config.num_gpus = 2;
    config.merit_threshold = 100.0;
    config.sieve_prime_limit = 100;
    
    struct miner_farm *farm = miner_farm_create(&config);
    if (!farm) {
        printf("  ✗ FAIL: Farm creation failed\n");
        return 0;
    }

    if (miner_farm_set_merit_threshold(farm, 77.5) != 0 ||
        farm->config.merit_threshold != 77.5 ||
        farm->workers[0].merit_threshold != 77.5 ||
        farm->workers[1].merit_threshold != 77.5) {
        printf("  ✗ FAIL: Merit threshold update was not propagated\n");
        miner_farm_free(farm);
        return 0;
    }

    uint8_t finite_range_hash[32] = {0};
    miner_farm_update_work(farm, 1, 14, 0, finite_range_hash, 0);
    atomic_store(&farm->nonce.current, 4);
    atomic_store(&farm->completed_windows, 3);
    if (miner_farm_work_exhausted(farm, 14)) {
        printf("  ✗ FAIL: Claimed but unfinished range marked exhausted\n");
        miner_farm_free(farm);
        return 0;
    }
    atomic_store(&farm->completed_windows, 4);
    if (!miner_farm_work_exhausted(farm, 14)) {
        printf("  ✗ FAIL: Completed finite range was not marked exhausted\n");
        miner_farm_free(farm);
        return 0;
    }

    uint8_t h256[32] = {0};
    h256[31] = 1;
    miner_farm_update_work(farm, 1, 64, 0, h256, 0);
    
    /* Start workers for 2 seconds */
    int ret = miner_farm_start(farm);
    if (ret != 0) {
        printf("  ✗ FAIL: Worker start failed\n");
        miner_farm_free(farm);
        return 0;
    }
    
    /* Let workers run briefly */
    sleep(1);
    
    /* Stop workers */
    miner_farm_stop(farm);
    miner_farm_join(farm);
    
    /* Get statistics */
    struct farm_stats stats;
    miner_farm_get_stats(farm, &stats);
    
    printf("  Farm stats:\n");
    printf("    Total nonces: %lu\n", stats.total_nonces);
    printf("    Total candidates: %lu\n", stats.total_candidates);
    printf("    Total gaps: %lu\n", stats.total_gaps);
    printf("    Total submissions: %lu\n", stats.total_submissions);
    
    miner_farm_free(farm);
    
    if (stats.total_nonces > 0) {
        printf("  ✓ PASS: Farm processed nonces\n");
        return 1;
    } else {
        printf("  ✗ FAIL: Farm did not process any nonces\n");
        return 0;
    }
}

/* Test nonce exhaustion */
static int test_atomic_nonce_exhaustion(void) {
    printf("[TEST] Atomic nonce exhaustion...\n");
    
    struct atomic_nonce nonce;
    atomic_nonce_init(&nonce, 0, 10);  /* Only 10 nonces */
    
    /* Acquire all 10 nonces */
    for (uint32_t i = 0; i < 10; i++) {
        uint32_t n = atomic_nonce_next(&nonce);
        if (n != i) {
            printf("  ✗ FAIL: Expected nonce %u, got %u\n", i, n);
            return 0;
        }
    }
    
    /* Check exhausted flag */
    if (!atomic_nonce_exhausted(&nonce)) {
        printf("  ✗ FAIL: Nonce not marked as exhausted\n");
        return 0;
    }
    
    printf("  ✓ PASS: Nonce exhaustion detected correctly\n");
    return 1;
}

static int test_atomic_nonce_chunk_claim(void) {
    printf("[TEST] Atomic nonce chunk claim...\n");

    struct atomic_nonce nonce;
    uint32_t claimed_count = 0;
    atomic_nonce_init(&nonce, 0, 10);

    uint32_t first = atomic_nonce_claim(&nonce, 10, 4, &claimed_count);
    if (first != 0 || claimed_count != 4 || atomic_load(&nonce.current) != 4) {
        printf("  ✗ FAIL: First chunk claim was incorrect\n");
        return 0;
    }

    first = atomic_nonce_claim(&nonce, 10, 8, &claimed_count);
    if (first != 4 || claimed_count != 6 || atomic_load(&nonce.current) != 10) {
        printf("  ✗ FAIL: Final partial chunk claim was incorrect\n");
        return 0;
    }

    first = atomic_nonce_claim(&nonce, 10, 4, &claimed_count);
    if (first != 10 || claimed_count != 0) {
        printf("  ✗ FAIL: Exhausted chunk claim was incorrect\n");
        return 0;
    }

    printf("  ✓ PASS: Chunk claims stop exactly at the range limit\n");
    return 1;
}

int main(void) {
    printf("================================================\n");
    printf("GapMiner V2 — Integration Test: Worker Threads\n");
    printf("================================================\n\n");
    
    int all_pass = 1;
    
    all_pass &= test_atomic_nonce_multithread();
    printf("\n");

    all_pass &= test_shift_aware_windows();
    printf("\n");
    
    all_pass &= test_atomic_nonce_exhaustion();
    printf("\n");

    all_pass &= test_atomic_nonce_chunk_claim();
    printf("\n");
    
    all_pass &= test_miner_farm_creation();
    printf("\n");
    
    printf("================================================\n");
    if (all_pass) {
        printf("✓ All worker thread tests PASSED\n");
    } else {
        printf("✗ Some worker thread tests FAILED\n");
    }
    printf("================================================\n");
    
    return all_pass ? 0 : 1;
}
