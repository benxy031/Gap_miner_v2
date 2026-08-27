/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Miner Farm Implementation (Phase 3 Stub)
 *
 * Spawns per-GPU worker threads with lock-free nonce sharing.
 */

#include "miner_farm.h"
#include "gpu_adapter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct miner_farm *miner_farm_create(struct miner_farm_config *config) {
    if (!config) return NULL;
    
    struct miner_farm *farm = (struct miner_farm *)malloc(sizeof(struct miner_farm));
    if (!farm) return NULL;
    
    farm->config = *config;
    farm->stop_flag = 0;
    pthread_mutex_init(&farm->work_lock, NULL);
    atomic_init(&farm->work_generation, 0);
    atomic_init(&farm->completed_windows, 0);
    
    /* Initialize atomic nonce (start at 0, max at 2^32-1) */
    atomic_nonce_init(&farm->nonce, 0, 0xFFFFFFFFu);
    
    /* Allocate worker threads and configs */
    farm->worker_threads = (pthread_t *)malloc(config->num_gpus * sizeof(pthread_t));
    farm->workers = (struct worker_config *)calloc(config->num_gpus, sizeof(struct worker_config));
    
    if (!farm->worker_threads || !farm->workers) {
        free(farm->worker_threads);
        free(farm->workers);
        pthread_mutex_destroy(&farm->work_lock);
        free(farm);
        return NULL;
    }
    
    int gpu_count = gpu_adapter_device_count();
    if (gpu_count < 1) gpu_count = 1;

    /* Initialize per-worker configuration */
    for (uint32_t i = 0; i < config->num_gpus; i++) {
        farm->workers[i].worker_id = i;
        farm->workers[i].gpu_device = i % (uint32_t)gpu_count;
        farm->workers[i].nonce = &farm->nonce;
        farm->workers[i].crt_rt = config->crt_rt;
        farm->workers[i].nthreads = config->num_gpus;
        farm->workers[i].work_lock = &farm->work_lock;
        farm->workers[i].work_generation = &farm->work_generation;
        farm->workers[i].completed_windows = &farm->completed_windows;
        farm->workers[i].merit_threshold = config->merit_threshold;
        farm->workers[i].sieve_limit = config->sieve_prime_limit;
        
        /* TODO: Initialize farm->workers[i].base from CRT */
    }
    
    printf("[MinerFarm] Created with %u CPU worker threads\n", config->num_gpus);
    
    return farm;
}

int miner_farm_start(struct miner_farm *farm) {
    if (!farm) return -1;
    
    printf("[MinerFarm] Starting %u worker thread%s...\n",
           farm->config.num_gpus, farm->config.num_gpus == 1 ? "" : "s");
    
    void *(*entry)(void *) = farm->config.crt_rt ?
        worker_thread_run_crt : worker_thread_run;

    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        int ret = pthread_create(&farm->worker_threads[i], NULL,
                                 entry, &farm->workers[i]);
        if (ret != 0) {
            fprintf(stderr, "[MinerFarm] Failed to create worker %u: %d\n", i, ret);
            return -1;
        }
    }
    
    printf("[MinerFarm] All worker threads started\n");
    return 0;
}

void miner_farm_stop(struct miner_farm *farm) {
    if (!farm) return;
    
    printf("[MinerFarm] Stopping workers...\n");
    farm->stop_flag = 1;
    worker_stop_requested();
}

void miner_farm_join(struct miner_farm *farm) {
    if (!farm) return;
    
    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        pthread_join(farm->worker_threads[i], NULL);
    }
    
    printf("[MinerFarm] All workers joined\n");
}

void miner_farm_get_stats(struct miner_farm *farm, struct farm_stats *stats) {
    if (!farm || !stats) return;
    
    memset(stats, 0, sizeof(*stats));
    
    struct worker_stats wstats;
    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        worker_get_stats(i, &wstats);
        stats->total_nonces += wstats.nonces_processed;
        stats->total_candidates += wstats.candidates_generated;
        stats->total_euler_passes += wstats.euler_passes;
        stats->total_euler_pairs += wstats.euler_pairs;
        stats->total_merit_candidates += wstats.merit_candidates;
        stats->total_bpsw_attempts += wstats.bpsw_attempts;
        stats->total_gaps += wstats.gaps_found;
        stats->total_submissions += wstats.gaps_submitted;
        stats->total_gpu_euler_skipped += wstats.gpu_euler_skipped;
        stats->total_gpu_sieve_calls += wstats.gpu_sieve_calls;
        stats->total_gpu_sieve_windows += wstats.gpu_sieve_windows;
        stats->total_smart_tail_skipped += wstats.smart_tail_skipped;
        stats->total_gpu_accounted_us += wstats.gpu_accounted_us;
        if (wstats.max_gap_length > stats->max_gap_length) {
            stats->max_gap_length = wstats.max_gap_length;
        }
        if (wstats.max_merit > stats->max_merit) {
            stats->max_merit = wstats.max_merit;
        }
    }
}

int miner_farm_set_merit_threshold(struct miner_farm *farm,
                                   double merit_threshold) {
    if (!farm || merit_threshold <= 0.0) return -1;

    pthread_mutex_lock(&farm->work_lock);
    farm->config.merit_threshold = merit_threshold;
    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        farm->workers[i].merit_threshold = merit_threshold;
    }
    pthread_mutex_unlock(&farm->work_lock);
    return 0;
}

void miner_farm_update_work(struct miner_farm *farm,
                           uint32_t height,
                           uint32_t shift,
                           uint32_t bits,
                           const uint8_t *h256,
                           uint32_t header_nonce) {
    if (!farm || !h256) return;
    
    printf("[MinerFarm] New work: height=%u, shift=%u, bits=%08x\n",
           height, shift, bits);
    
    pthread_mutex_lock(&farm->work_lock);

    /* Update all workers with a consistent block-template snapshot. */
    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        farm->workers[i].height = height;
        farm->workers[i].shift = shift;
        farm->workers[i].bits = bits;
        farm->workers[i].header_nonce = header_nonce;
        
        /* Copy block header hash for base calculation */
        memcpy(farm->workers[i].h256, h256, 32);
        
    }

    atomic_nonce_reset(&farm->nonce, 0);
    atomic_store_explicit(&farm->completed_windows, 0, memory_order_release);
    atomic_fetch_add_explicit(&farm->work_generation, 1, memory_order_release);
    pthread_mutex_unlock(&farm->work_lock);
}

void miner_farm_update_work_crt(struct miner_farm *farm,
                                uint32_t height,
                                uint32_t shift,
                                uint32_t bits,
                                const uint8_t *hdr80,
                                uint32_t pass_nonce) {
    if (!farm || !hdr80) return;

    printf("[MinerFarm] New CRT work: height=%u, shift=%u, bits=%08x, "
           "pass_nonce=%u\n",
           height, shift, bits, pass_nonce);

    pthread_mutex_lock(&farm->work_lock);
    for (uint32_t i = 0; i < farm->config.num_gpus; i++) {
        farm->workers[i].height = height;
        farm->workers[i].shift = shift;
        farm->workers[i].bits = bits;
        farm->workers[i].pass_nonce = pass_nonce;
        memcpy(farm->workers[i].hdr80, hdr80, 80);
    }
    atomic_nonce_reset(&farm->nonce, 0);
    atomic_store_explicit(&farm->completed_windows, 0, memory_order_release);
    atomic_fetch_add_explicit(&farm->work_generation, 1, memory_order_release);
    pthread_mutex_unlock(&farm->work_lock);
}

int miner_farm_work_exhausted(struct miner_farm *farm, uint32_t shift) {
    uint64_t windows_per_header = non_crt_windows_per_header(shift);
    if (!farm || windows_per_header == 0 || windows_per_header >= UINT32_MAX) {
        return 0;
    }

    pthread_mutex_lock(&farm->work_lock);
    uint64_t windows_claimed = atomic_load_explicit(&farm->nonce.current,
                                                    memory_order_acquire);
    uint64_t windows_completed = atomic_load_explicit(&farm->completed_windows,
                                                      memory_order_acquire);
    int exhausted = windows_claimed >= windows_per_header &&
                    windows_completed >= windows_per_header;
    pthread_mutex_unlock(&farm->work_lock);
    return exhausted;
}

void miner_farm_free(struct miner_farm *farm) {
    if (!farm) return;
    
    free(farm->worker_threads);
    free(farm->workers);
    pthread_mutex_destroy(&farm->work_lock);
    free(farm);
    
    printf("[MinerFarm] Freed\n");
}

void miner_farm_reset_nonce(struct miner_farm *farm) {
    if (!farm) return;
    
    /* Reset nonce counter to 0 for new block */
    pthread_mutex_lock(&farm->work_lock);
    atomic_nonce_reset(&farm->nonce, 0);
    pthread_mutex_unlock(&farm->work_lock);
}
