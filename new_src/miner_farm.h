/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Miner Farm (Worker Management)
 *
 * Spawns per-GPU worker threads, manages lifecycle, gathers statistics.
 */

#ifndef MINER_FARM_H
#define MINER_FARM_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include "atomic_nonce.h"
#include "worker_gpu.h"

/* Farm configuration */
struct miner_farm_config {
    uint32_t num_gpus;            /* Number of GPU workers */
    struct crt_runtime *crt_rt;   /* Loaded covering CRT file (CRT mode) */
    double merit_threshold;       /* Gap submission threshold */
    uint32_t sieve_prime_limit;   /* CPU sieve limit */
};

/* Farm state */
struct miner_farm {
    struct miner_farm_config config;
    struct atomic_nonce nonce;    /* Shared lock-free nonce */
    pthread_t *worker_threads;    /* Worker thread IDs */
    struct worker_config *workers; /* Per-worker configuration */
    pthread_mutex_t work_lock;    /* Protects active work copied into workers */
    _Atomic uint64_t work_generation; /* Increments for every new template */
    _Atomic uint64_t completed_windows; /* Valid windows fully processed in this generation */
    volatile int stop_flag;       /* Graceful shutdown */
};

/* Initialize farm */
struct miner_farm *miner_farm_create(struct miner_farm_config *config);

/* Start all worker threads */
int miner_farm_start(struct miner_farm *farm);

/* Signal graceful shutdown */
void miner_farm_stop(struct miner_farm *farm);

/* Wait for all workers to finish */
void miner_farm_join(struct miner_farm *farm);

/* Gather farm statistics */
struct farm_stats {
    uint64_t total_nonces;
    uint64_t total_candidates;
    uint64_t total_euler_passes;
    uint64_t total_euler_pairs;
    uint64_t total_merit_candidates;
    uint64_t total_bpsw_attempts;
    uint64_t total_gaps;
    uint64_t total_submissions;
    uint32_t max_gap_length;
    double max_merit;
    double throughput_nonces_per_sec;
    uint64_t total_gpu_euler_skipped;
    uint64_t total_gpu_sieve_calls;
    uint64_t total_gpu_sieve_windows;
    uint64_t total_smart_tail_skipped;
    uint64_t total_gpu_accounted_us;
};

void miner_farm_get_stats(struct miner_farm *farm, struct farm_stats *stats);

/* Update the merit filter without resetting the active template or nonce range. */
int miner_farm_set_merit_threshold(struct miner_farm *farm,
                                   double merit_threshold);

/*
 * Update mining work for all workers (new block template)
 * Called when fresh block template arrives from Gapcoin node
 * 
 * height: new block height
 * shift: computed shift from block template bits
 * bits: raw bits value (for logging/validation)
 * h256: block header hash (256 bits for base calculation)
 * header_nonce: GBT header nonce that produced h256 (needed to reassemble a
 *               submittable block for any gap found under this h256)
 */
void miner_farm_update_work(struct miner_farm *farm,
                           uint32_t height,
                           uint32_t shift,
                           uint32_t bits,
                           const uint8_t *h256,
                           uint32_t header_nonce);

/* CRT-mode work update: pushes the shared 80-byte header prefix and the pass
 * nonce snapshot; each worker hashes its own strided nonces from it. */
void miner_farm_update_work_crt(struct miner_farm *farm,
                                uint32_t height,
                                uint32_t shift,
                                uint32_t bits,
                                const uint8_t *hdr80,
                                uint32_t pass_nonce);

/* Return nonzero once every valid fixed window for a finite nAdd range completed. */
int miner_farm_work_exhausted(struct miner_farm *farm, uint32_t shift);

/* Reset nonce counter for new block (Phase 6) */
void miner_farm_reset_nonce(struct miner_farm *farm);

/* Cleanup */
void miner_farm_free(struct miner_farm *farm);

#endif /* MINER_FARM_H */
