/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Worker Thread
 *
 * Per-GPU worker loop:
 * 1. Get nonce from atomic counter
 * 2. Run CPU sieve + phase 1 filtering
 * 3. Submit GPU Fermat candidates
 * 4. Check results for gap submissions
 */

#ifndef WORKER_GPU_H
#define WORKER_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <pthread.h>
#include <stdatomic.h>
#include <gmp.h>
#include "atomic_nonce.h"
#include "gap_detection.h"   /* struct gap_result */
#include "halfclass.h"       /* struct halfclass_tpl */

#define NON_CRT_MIN_WINDOW_SHIFT 12U
#define NON_CRT_MAX_WINDOW_SHIFT 20U
#define NON_CRT_LOOKAHEAD_SIZE 8192U
#define NON_CRT_WINDOW_CHUNK 8U

struct crt_runtime;

/* Owned-window sizing: window = 2^max(shift-14, MIN) capped at MAX.
   This keeps ~16384 owned windows per header through shift 32, then caps
   memory use at 2^18 adders per window for larger shifts (up to 1024). */
uint32_t non_crt_owned_window_size(uint32_t shift);
uint64_t non_crt_windows_per_header(uint32_t shift);

/* Worker thread configuration */
struct worker_config {
    uint32_t worker_id;              /* 0 = GPU 0, 1 = GPU 1, etc. */
    uint32_t gpu_device;             /* CUDA device index */
    struct atomic_nonce *nonce;      /* Shared nonce counter */
    struct crt_runtime *crt_rt;      /* Loaded covering CRT file (CRT mode) */
    pthread_mutex_t *work_lock;      /* Protects the work-template fields */
    _Atomic uint64_t *work_generation; /* Nonzero after the first template */
    _Atomic uint64_t *completed_windows; /* Completed windows in active generation */
    uint32_t height;                 /* Current block height (from block template) */
    uint32_t bits;                   /* Raw bits from block template (e.g., 0x15589358) */
    uint32_t shift;                  /* Computed shift from bits (log2 of target, ~bits-to-shift conversion) */
    uint32_t sieve_limit;            /* CPU sieve limit (e.g., 1000000) */
    double merit_threshold;          /* Minimum gap merit to submit */
    uint8_t h256[32];                /* Block header hash (256 bits = 32 bytes) for base calculation */
    uint32_t header_nonce;           /* GBT header nonce active for this h256 */
    uint8_t hdr80[80];               /* Shared block-header prefix (CRT mode) */
    uint32_t pass_nonce;             /* GBT header nonce snapshot at block start (CRT) */
    uint32_t nthreads;               /* Total CRT workers (nonce stride) */
    
    /* Merit is gap_length / ln(start); start already includes the shift. */
};

/* Worker thread entry point */
void *worker_thread_run(void *arg);

/* CRT-mode worker: one aligned window scan per header (covering boost). */
void *worker_thread_run_crt(void *arg);

/* Graceful worker shutdown */
void worker_stop_requested(void);

/* Get worker statistics */
struct worker_stats {
    uint64_t nonces_processed;
    uint64_t candidates_generated;
    uint64_t candidates_tested;
    uint64_t euler_passes;
    uint64_t euler_pairs;
    uint64_t merit_candidates;
    uint64_t bpsw_attempts;
    uint64_t gaps_found;
    uint64_t gaps_submitted;
    uint32_t max_gap_length;
    double max_merit;
    uint64_t gpu_euler_skipped; /* Euler calls skipped due to a GPU Fermat reject */
    uint64_t gpu_sieve_calls;   /* Hybrid GPU bitmap sieve batch invocations */
    uint64_t gpu_sieve_windows; /* Windows processed through batched GPU sieve */
    uint64_t smart_tail_skipped; /* CRT windows whose uncovered tail was skipped */
    uint64_t gpu_accounted_us;  /* GPU-accounted MR kernel time (CUDA events) */
};

void worker_get_stats(uint32_t worker_id, struct worker_stats *stats);

/* Gap queue entry */
struct gap_queue_entry {
    uint32_t height;
    uint32_t shift;
    uint32_t nonce;               /* Legacy field, unused by the real submission path */
    uint64_t p1;                  /* Legacy field, unused by the real submission path */
    uint32_t gap_length;
    double merit;
    uint64_t generation;           /* Work generation active when this gap was found */
    uint32_t header_nonce;         /* GBT header nonce active when this gap was found */
    uint64_t nadd;                 /* Full-width adder offset within the header's shift range */
    uint8_t nadd_bytes[128];       /* Little-endian nAdd bytes (CRT: >64-bit offsets) */
    uint32_t nadd_len;             /* 0 => use legacy `nadd`; >0 => use nadd_bytes */
};

/* Get pending gap from queue (thread-safe) */
int worker_get_pending_gap(struct gap_queue_entry *gap_out);

/* Flush all pending gaps and return count */
uint32_t worker_flush_gaps(void);

/* Enable/disable pushing BPSW-verified gaps into the submission queue (default: disabled). */
void worker_set_submission_enabled(int enabled);

/* Enable/disable GPU Fermat pre-filtering before the CPU Euler test (default:
   disabled; only takes effect in WITH_CUDA builds). Safe by construction:
   Euler's criterion at base 2 implies base-2 Fermat, so a GPU Fermat reject
   always means Euler would reject too, and is skipped rather than recomputed. */
void worker_set_gpu_fermat_enabled(int enabled);

/* Reset nonce counter for new block (Phase 6) */
void worker_reset_nonce_counter(void);

#ifdef WITH_CUDA
struct gpu_fermat_ctx;

/* GPU-side hidden-class resolution (exposed for the exactness test):
   mini-sieve + template on the host, base-2+3 MR batch on the GPU, BPSW only
   on the MR survivors.  Falls back to the CPU-only resolution on failure. */
int crt_gpu_resolve_gap_ex(struct gpu_fermat_ctx *fermat, int slot,
                           mpz_t window_base, uint64_t off_a, uint64_t off_b,
                           uint64_t owned_offset_limit, double merit_threshold,
                           const struct halfclass_tpl *tpl,
                           struct gap_result **out, uint32_t *out_count);
#endif /* WITH_CUDA */

#endif /* WORKER_GPU_H */
