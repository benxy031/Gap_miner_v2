/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gpu_sieve.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Max base limbs (64-bit) the residue kernel accepts; covers GPU_NLIMBS=20
   (1280-bit) with headroom, and the pair-batch mark uploads 2× limbs. */
#define GPU_SIEVE_MAX_BASE_LIMBS 64

/* Split (chunked) dense row-batch marking — opt-in while it earns its numbers
   in production.  GPU_MARK_SPLIT=1 enables it (default off); the chunk size in
   odd slots is GPU_MARK_SPLIT_SLOTS (default 160, clamped to [32, 4096]).
   Measured in isolation (tools/bench_mark.cu): marks exactly the same slots as
   the per-prime row walk and is 25-38x faster (21-30 us vs 700-810 us for
   W=10175, rows=8, 2M primes, RTX 3070). */
/* Split (chunked) dense row-batch marking.  Default ON (bit-exact vs the
   per-prime row walk; measured +47-55% end-to-end windows/s and 14x less mark
   kernel time in the fused chain, shift512 p75, RTX 3070 -- see README and
   docs/GPU_SCREEN_ANALYSIS.md).  GPU_MARK_SPLIT=0 disables it; the chunk size
   in odd slots is GPU_MARK_SPLIT_SLOTS (default 160, clamped to [32, 4096]).
   Isolated measurement: 25-38x faster marking (21-30 us vs 700-810 us for
   W=10175, rows=8, 2M primes) with 0 differing bitmap bits. */
static int gpu_mark_split_enabled(void) {
    const char *v = getenv("GPU_MARK_SPLIT");
    if (v && v[0] == '0') return 0;
    return 1;
}

static uint64_t gpu_mark_split_slots(void) {
    const char *v = getenv("GPU_MARK_SPLIT_SLOTS");
    uint64_t s = 160;
    if (v && v[0]) {
        unsigned long long parsed = strtoull(v, NULL, 10);
        if (parsed > 0) s = (uint64_t)parsed;
    }
    if (s < 32) s = 32;
    if (s > 4096) s = 4096;
    return s;
}

/* The split domain is "p <= odd_interval_size", and the row walk then runs on
   the remaining SUFFIX of the table -- which is only the same set if the
   table is ascending.  Verify that once per (table, count) and fail closed
   (row walk over the whole table) if it is not. */
static int gpu_mark_primes_ascending(const uint64_t *primes, size_t count) {
    static const uint64_t *cached_ptr = NULL;
    static size_t cached_count = 0;
    static int cached_ok = 0;
    if (primes == cached_ptr && count == cached_count) return cached_ok;
    int sorted = 1;
    for (size_t i = 1; i < count; i++) {
        if (primes[i] <= primes[i - 1]) { sorted = 0; break; }
    }
    cached_ptr = primes;
    cached_count = count;
    cached_ok = sorted;
    return sorted;
}

/* Floor for the right-sized candidate-buffer capacity, in slots per window
   (see gpu_sieve_ctx.cand_cap_per_window).  The measured survivor density is
   ~1,000 per 17,916-slot window at shift507/2M sieve primes, so 4096 leaves a
   4x margin before the buffer ever has to grow. */
#define GPU_EXTRACT_CAND_CAP_MIN 4096U

struct gpu_sieve_ctx {
    int device_id;
    size_t max_primes;
    uint64_t max_odd_interval;
    size_t max_bitmap_words;
    size_t batch_capacity;
    size_t bitmap_words_capacity;
    uint64_t last_elapsed_us;
    uint64_t *d_primes;
    uint64_t *d_base_mod_p;
    uint64_t *d_inv_p;
    uint64_t *d_base_offsets;
    uint64_t *d_bitmap[2];   /* ping-pong: async fused pipeline marks window
                                i into buf i&1 while window i-1 is in flight */
    cudaStream_t stream;
    uint64_t *d_base_limbs;
    int base_limbs_capacity;
    size_t primes_uploaded_count;
    char dev_name[256];

    /* Fused-pipeline extract+pack buffers (Stage 1). */
    uint64_t *d_cands_aos[2];   /* [max_candidates * active_limbs] AoS limbs,
                                   ping-pong for the async fused pipeline */
    uint64_t *d_offsets;        /* [max_candidates] full adder offsets */
    unsigned int *d_count;      /* survivor counter (written by the scan);
                                   its HIGH BIT is set when a survivor did
                                   not fit the candidate-buffer capacity */
    size_t max_candidates;
    uint32_t extract_accum;     /* K: candidate buffers sized K× per window
                                   (MR batch accumulation across windows) */
    int active_limbs_capacity;

    /* Candidate-buffer RIGHT-SIZING (see gpu_sieve_extract_pack_impl).  The
       buffer only ever holds SURVIVORS, and the survivors are a small
       fraction of the odd slots (measured 1,000 of 17,916 per window at
       shift507 / 2M sieve primes), so allocating one slot per odd position
       is a ~18x over-allocation that is what caps MINING_JUMP2_BATCH by
       VRAM.  cand_cap_per_window is the real capacity: it starts small,
       is right-sized after the first measured window, and grows (loudly)
       if a later window ever needs more.  Growth is always prefered over
       truncation: a dropped survivor would make two non-consecutive primes
       look consecutive, i.e. a FALSE GAP. */
    uint64_t cand_cap_per_window;      /* slots reserved per window */
    uint64_t cand_cap_limit;           /* caller's host allocation (slots,
                                          0 = no limit; see the setter) */
    uint64_t cand_cap_pending;         /* capacity to apply at the next
                                          slot_base == 0 (never mid-flight) */
    uint64_t cand_cap_measured;        /* max survivors seen in one window */
    int cand_cap_calibrated;           /* first window already measured */
    int cand_cap_env;                  /* GPU_EXTRACT_CAND_CAP: initial cap */
    int cand_cap_reported;             /* one-shot stderr reporting */

    /* Row-batch mark (CRT row-walk): rows × max_bitmap_words bitmaps so one
       residues pass can mark a whole row batch; d_step_mod_p[i] = P mod p. */
    uint64_t *d_row_bitmaps;
    uint32_t row_bitmap_cap;
    uint64_t *d_step_mod_p;

    /* Fused row-batch mark, split (chunked) parallelization: per-prime row
       lattice descriptors (pos0, pair decrement, first-row decrement)
       computed once per batch by gpu_sieve_rows_mark_prep_kernel, then one
       work item per (prime <= odd_interval_size, row, chunk).  Measured
       ~25-38x faster than the per-prime row walk (tools/bench_mark.cu,
       bit-exact on the same slot sets); see GPU_MARK_SPLIT. */
    uint64_t *d_mark_pos0;
    uint64_t *d_mark_pair;
    uint64_t *d_mark_firstdec;
    /* Pair-batch path (rows == 1) needs one pos0 per (window, prime); the
       window-0 values use d_mark_pos0, the window-1 values use d_mark_pos1. */
    uint64_t *d_mark_pos1;

    /* Optional per-stage kernel accounting (GPU_SIEVE_TIMING=1): CUDA events
       around the mark and extract launches.  stage 0 = none, 1 = mark,
       2 = extract; the elapsed time is drained right after the stream sync
       those paths already perform. */
    int timing_on;
    int t_inited;
    int t_stage;
    cudaEvent_t t_start;
    cudaEvent_t t_end;
    uint64_t accounted_mark_us;
    uint64_t accounted_extract_us;
};

/* ── Per-stage kernel accounting helpers (no-op unless GPU_SIEVE_TIMING=1) ── */
static void sieve_timing_begin(struct gpu_sieve_ctx *ctx, int stage) {
    if (!ctx->timing_on || !ctx->t_inited || ctx->t_stage != 0) return;
    if (cudaEventRecord(ctx->t_start, ctx->stream) != cudaSuccess) return;
    ctx->t_stage = stage;
}

static void sieve_timing_end(struct gpu_sieve_ctx *ctx) {
    if (ctx->t_stage == 0) return;
    if (cudaEventRecord(ctx->t_end, ctx->stream) != cudaSuccess) {
        ctx->t_stage = 0;
        return;
    }
}

static void sieve_timing_drain(struct gpu_sieve_ctx *ctx) {
    if (ctx->t_stage == 0) return;
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, ctx->t_start, ctx->t_end) == cudaSuccess &&
        ms > 0.0f) {
        uint64_t us = (uint64_t)(ms * 1000.0f + 0.5f);
        if (ctx->t_stage == 1)
            __atomic_fetch_add(&ctx->accounted_mark_us, us, __ATOMIC_RELAXED);
        else
            __atomic_fetch_add(&ctx->accounted_extract_us, us, __ATOMIC_RELAXED);
    }
    ctx->t_stage = 0;
}

uint64_t gpu_sieve_accounted_mark_us(gpu_sieve_ctx *ctx) {
    if (!ctx) return 0;
    return __atomic_load_n(&ctx->accounted_mark_us, __ATOMIC_RELAXED);
}

uint64_t gpu_sieve_accounted_extract_us(gpu_sieve_ctx *ctx) {
    if (!ctx) return 0;
    return __atomic_load_n(&ctx->accounted_extract_us, __ATOMIC_RELAXED);
}

static uint64_t gpu_sieve_clock_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000ULL +
           (uint64_t)ts.tv_nsec / 1000ULL;
}

static __host__ __forceinline__ cudaError_t gpu_sieve_ensure_device(int device_id) {
    int current = -1;
    cudaError_t err = cudaGetDevice(&current);
    if (err != cudaSuccess) return err;
    if (current == device_id) return cudaSuccess;
    return cudaSetDevice(device_id);
}

/* ── Composite marking for one progression ──────────────────────────────────
   Mark every q = pos + k*p < odd_interval_size in the bitmap `wb`.

   NOTE (measured, negative result, 2026-09-14): an accumulating variant of
   this loop (pack the bits of each 64-bit word in a register and flush ONE
   atomicOr per word instead of one per multiple) was implemented and measured
   as a REGRESSION: mark kernel 0.125 -> 0.178 ms/window and end-to-end
   2718 -> 2318 win/s (-15%) on the shift512 p75 fused chain, 2x120 s, order
   swapped.  Reason: the per-multiple atomicOr here is fire-and-forget (the
   return value is unused), so same-address atomics coalesce in L2 and issue in
   parallel, while the accumulating variant introduces a serial
   compare/accumulate dependency in the loop.  The mark kernel is therefore NOT
   atomic-issue bound; do not "optimize" the atomic count here again without a
   counter-example measurement. */
__device__ static void gpu_sieve_mark_progression(uint64_t *wb,
                                                  uint64_t odd_interval_size,
                                                  uint64_t p, uint64_t pos) {
    for (; pos < odd_interval_size; pos += p) {
        atomicOr((unsigned long long *)&wb[pos >> 6],
                 (unsigned long long)(1ULL << (pos & 63U)));
    }
}

__global__ static void gpu_sieve_mark_kernel_batch(uint64_t *bitmap,
                                                   uint64_t bitmap_words,
                                                   uint64_t odd_interval_size,
                                                   uint64_t first_odd_offset,
                                                   const uint64_t *base_offsets,
                                                   uint64_t batch_count,
                                                   const uint64_t *primes,
                                                   const uint64_t *base_mod_p,
                                                   uint64_t prime_count) {
    uint64_t flat_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                        (uint64_t)threadIdx.x;
    uint64_t total_threads = batch_count * prime_count;
    if (flat_idx >= total_threads) return;

    uint64_t window_idx = flat_idx / prime_count;
    uint64_t prime_idx = flat_idx - window_idx * prime_count;

    uint64_t p = primes[prime_idx];
    if (p < 3U) return;

    uint64_t base_offset = base_offsets[window_idx];
    uint64_t *window_bitmap = bitmap + window_idx * bitmap_words;

    uint64_t remainder = base_mod_p[prime_idx] + (base_offset % p);
    if (remainder >= p) remainder -= p;

    remainder += first_odd_offset % p;
    if (remainder >= p) remainder -= p;

    uint64_t inverse_two = (p + 1U) >> 1;
    uint64_t pos = (((p - remainder) % p) * inverse_two) % p;

    gpu_sieve_mark_progression(window_bitmap, odd_interval_size, p, pos);
}

static int gpu_sieve_reserve_batch(gpu_sieve_ctx *ctx,
                                   size_t batch_count,
                                   size_t total_bitmap_words) {
    if (!ctx || batch_count == 0 || total_bitmap_words == 0) return 0;
    if (batch_count <= ctx->batch_capacity &&
        total_bitmap_words <= ctx->bitmap_words_capacity) {
        return 1;
    }

    uint64_t *new_base_offsets = NULL;
    uint64_t *new_bitmap = NULL;
    cudaError_t err;

    if (batch_count > ctx->batch_capacity) {
        err = cudaMalloc(&new_base_offsets,
                         batch_count * sizeof(*ctx->d_base_offsets));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: cudaMalloc(base_offsets): %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
    }

    if (total_bitmap_words > ctx->bitmap_words_capacity) {
        err = cudaMalloc(&new_bitmap,
                         total_bitmap_words * sizeof(*ctx->d_bitmap[0]));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: cudaMalloc(bitmap reserve): %s\n",
                    cudaGetErrorString(err));
            if (new_base_offsets) cudaFree(new_base_offsets);
            return 0;
        }
    }

    if (new_base_offsets) {
        if (ctx->d_base_offsets) cudaFree(ctx->d_base_offsets);
        ctx->d_base_offsets = new_base_offsets;
        ctx->batch_capacity = batch_count;
    }
    if (new_bitmap) {
        if (ctx->d_bitmap[0]) cudaFree(ctx->d_bitmap[0]);
        ctx->d_bitmap[0] = new_bitmap;
        ctx->bitmap_words_capacity = total_bitmap_words;
    }

    return 1;
}

gpu_sieve_ctx *gpu_sieve_init(int device_id,
                              size_t max_primes,
                              uint64_t max_odd_interval) {
    if (max_primes == 0 || max_odd_interval == 0) return NULL;

    gpu_sieve_ctx *ctx = (gpu_sieve_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->device_id = device_id;
    ctx->max_primes = max_primes;
    ctx->max_odd_interval = max_odd_interval;
    ctx->max_bitmap_words = (size_t)((max_odd_interval + 63U) >> 6);
    ctx->batch_capacity = 0;
    ctx->bitmap_words_capacity = 0;
    ctx->base_limbs_capacity = 0;
    ctx->primes_uploaded_count = 0;
    ctx->extract_accum = 1;
    /* Candidate-buffer right-sizing: 0 = size it automatically from the
       first measured window (4x headroom); a positive GPU_EXTRACT_CAND_CAP
       pins the initial per-window capacity for experiments. */
    ctx->cand_cap_per_window = 0;
    ctx->cand_cap_limit = 0;
    ctx->cand_cap_pending = 0;
    ctx->cand_cap_measured = 0;
    ctx->cand_cap_calibrated = 0;
    ctx->cand_cap_reported = 0;
    {
        const char *cc = getenv("GPU_EXTRACT_CAND_CAP");
        long v = (cc && *cc) ? strtol(cc, NULL, 10) : 0;
        ctx->cand_cap_env = (v > 0) ? (int)v : 0;
    }
    (void)snprintf(ctx->dev_name, sizeof(ctx->dev_name), "cuda:%d", device_id);

    cudaError_t err = gpu_sieve_ensure_device(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaSetDevice(%d): %s\n",
                device_id, cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    err = cudaMalloc(&ctx->d_primes, max_primes * sizeof(*ctx->d_primes));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(primes): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    err = cudaMalloc(&ctx->d_base_mod_p,
                     max_primes * sizeof(*ctx->d_base_mod_p));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(base_mod_p): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    err = cudaMalloc(&ctx->d_step_mod_p,
                     max_primes * sizeof(*ctx->d_step_mod_p));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(step_mod_p): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    err = cudaMalloc(&ctx->d_inv_p, max_primes * sizeof(*ctx->d_inv_p));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(inv_p): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    /* Split row-batch mark descriptors (only filled when GPU_MARK_SPLIT). */
    err = cudaMalloc(&ctx->d_mark_pos0, max_primes * sizeof(*ctx->d_mark_pos0));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(mark_pos0): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    err = cudaMalloc(&ctx->d_mark_pair, max_primes * sizeof(*ctx->d_mark_pair));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(mark_pair): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    err = cudaMalloc(&ctx->d_mark_firstdec,
                     max_primes * sizeof(*ctx->d_mark_firstdec));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(mark_firstdec): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    err = cudaMalloc(&ctx->d_mark_pos1, max_primes * sizeof(*ctx->d_mark_pos1));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(mark_pos1): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    err = cudaMalloc(&ctx->d_base_offsets, sizeof(*ctx->d_base_offsets));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(base_offsets): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    ctx->batch_capacity = 1;

    err = cudaMalloc(&ctx->d_bitmap[0],
                     ctx->max_bitmap_words * sizeof(*ctx->d_bitmap[0]));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(bitmap[0]): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    err = cudaMalloc(&ctx->d_bitmap[1],
                     ctx->max_bitmap_words * sizeof(*ctx->d_bitmap[1]));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(bitmap[1]): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    ctx->bitmap_words_capacity = ctx->max_bitmap_words;

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaStreamCreate: %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }

    /* Optional per-stage kernel accounting (GPU_SIEVE_TIMING=1).  The timing
       events are only recorded when the env is set; their elapsed time is read
       after the stream sync that the mark/extract paths already perform, so no
       extra device synchronization is introduced. */
    {
        const char *tv = getenv("GPU_SIEVE_TIMING");
        if (tv && *tv && tv[0] != '0') {
            if (cudaEventCreate(&ctx->t_start) == cudaSuccess &&
                cudaEventCreate(&ctx->t_end) == cudaSuccess) {
                ctx->t_inited = 1;
                ctx->timing_on = 1;
            }
        }
    }

    err = cudaMalloc(&ctx->d_base_limbs,
                     GPU_SIEVE_MAX_BASE_LIMBS * sizeof(*ctx->d_base_limbs));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(base_limbs): %s\n",
                cudaGetErrorString(err));
        gpu_sieve_destroy(ctx);
        return NULL;
    }
    ctx->base_limbs_capacity = GPU_SIEVE_MAX_BASE_LIMBS;

    return ctx;
}

int gpu_sieve_mark_high_primes_batch(gpu_sieve_ctx *ctx,
                                     uint64_t odd_interval_size,
                                     uint64_t first_odd_offset,
                                     const uint64_t *base_offsets,
                                     size_t batch_count,
                                     const uint64_t *primes,
                                     const uint64_t *base_mod_p,
                                     size_t prime_count,
                                     uint64_t *host_bitmaps,
                                     size_t host_bitmaps_words) {
    uint64_t start_time = gpu_sieve_clock_us();
    if (!ctx || !host_bitmaps) return 0;
    if (odd_interval_size == 0 || batch_count == 0 || prime_count == 0) {
        return 0;
    }
    if (!base_offsets || !primes || !base_mod_p) return 0;
    if (first_odd_offset > 1U) return 0;
    if (prime_count > ctx->max_primes) return 0;

    size_t required_words = (size_t)((odd_interval_size + 63U) >> 6);
    if (required_words == 0 || required_words > ctx->max_bitmap_words) {
        return 0;
    }

    if (required_words > SIZE_MAX / batch_count) return 0;
    size_t total_words = required_words * batch_count;
    if (total_words > host_bitmaps_words) return 0;

    if (batch_count > (size_t)UINT64_MAX ||
        prime_count > (size_t)UINT64_MAX) {
        return 0;
    }
    if (batch_count > 0 && prime_count > (size_t)UINT64_MAX / batch_count) {
        return 0;
    }

    uint64_t batch_count_u64 = (uint64_t)batch_count;
    uint64_t prime_count_u64 = (uint64_t)prime_count;
    uint64_t total_threads = batch_count_u64 * prime_count_u64;
    if (total_threads == 0) return 0;

    memset(host_bitmaps, 0, total_words * sizeof(*host_bitmaps));

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: ensure_device failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    if (!gpu_sieve_reserve_batch(ctx, batch_count, total_words)) {
        return 0;
    }

    err = cudaMemcpy(ctx->d_primes, primes,
                     prime_count * sizeof(*primes), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: H2D primes failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(ctx->d_base_mod_p, base_mod_p,
                     prime_count * sizeof(*base_mod_p), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: H2D base_mod_p failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(ctx->d_base_offsets, base_offsets,
                     batch_count * sizeof(*base_offsets),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: H2D base_offsets failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemset(ctx->d_bitmap[0], 0, total_words * sizeof(*ctx->d_bitmap[0]));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMemset(bitmap) failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    const int threads_per_block = 128;
    uint64_t block_count_u64 =
        (total_threads + threads_per_block - 1U) / (uint64_t)threads_per_block;
    if (block_count_u64 > (uint64_t)UINT_MAX) {
        fprintf(stderr, "gpu_sieve: kernel grid too large for batch\n");
        return 0;
    }
    dim3 grid((unsigned int)block_count_u64);
    dim3 block(threads_per_block);
    gpu_sieve_mark_kernel_batch<<<grid, block>>>(ctx->d_bitmap[0],
                                                 (uint64_t)required_words,
                                                 odd_interval_size,
                                                 first_odd_offset,
                                                 ctx->d_base_offsets,
                                                 batch_count_u64,
                                                 ctx->d_primes,
                                                 ctx->d_base_mod_p,
                                                 prime_count_u64);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: kernel execution failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemcpy(host_bitmaps, ctx->d_bitmap[0],
                     total_words * sizeof(*host_bitmaps),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: D2H bitmap failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    uint64_t end_time = gpu_sieve_clock_us();
    ctx->last_elapsed_us = end_time >= start_time ? end_time - start_time : 0;
    return 1;
}

uint64_t gpu_sieve_last_elapsed_us(const gpu_sieve_ctx *ctx) {
    return ctx ? ctx->last_elapsed_us : 0;
}

int gpu_sieve_mark_high_primes(gpu_sieve_ctx *ctx,
                               uint64_t odd_interval_size,
                               uint64_t first_odd_offset,
                               uint64_t base_offset,
                               const uint64_t *primes,
                               const uint64_t *base_mod_p,
                               size_t prime_count,
                               uint64_t *host_bitmap,
                               size_t host_bitmap_words) {
    return gpu_sieve_mark_high_primes_batch(ctx,
                                            odd_interval_size,
                                            first_odd_offset,
                                            &base_offset,
                                            1,
                                            primes,
                                            base_mod_p,
                                            prime_count,
                                            host_bitmap,
                                            host_bitmap_words);
}

/* Compute base_mod_p[i] = base mod primes[i] on-device, one thread per prime.
   base_limbs: little-endian 64-bit limbs (Horner over 32-bit chunks).
   inv_p[i] = floor((2^64-1)/p) — Barrett reciprocal, avoids 64-bit division. */
__global__ static void gpu_sieve_residues_kernel(const uint64_t *base_limbs,
                                                 int base_limb_count,
                                                 const uint64_t *primes,
                                                 const uint64_t *inv_p,
                                                 uint64_t *base_mod_p,
                                                 uint64_t prime_count) {
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                   (uint64_t)threadIdx.x;
    if (idx >= prime_count) return;

    uint64_t p = primes[idx];
    uint64_t inv = inv_p[idx];
    const uint32_t *base32 = (const uint32_t *)base_limbs;
    int chunks = base_limb_count * 2;
    uint64_t r = 0;
    for (int i = chunks - 1; i >= 0; i--) {
        uint64_t v = (r << 32) | base32[i];
        uint64_t q = (uint64_t)(((unsigned __int128)v * inv) >> 64);
        r = v - q * p;
        if (r >= p) r -= p;
    }
    base_mod_p[idx] = r;
}

/* Row-batch fused mark (CRT row-walk): compute base mod p AND step mod p
   (step = P, the CRT row stride) in ONE sweep, then mark row_count bitmaps.
   d_base_limbs holds base limbs at [0, base_limb_count) and step limbs at
   [base_limb_count, 2*base_limb_count). */
__global__ static void gpu_sieve_rows_residues_kernel(
    const uint64_t *base_limbs,
    const uint64_t *step_limbs,
    int base_limb_count,
    const uint64_t *primes,
    const uint64_t *inv_p,
    uint64_t *base_mod_p,
    uint64_t *step_mod_p,
    uint64_t prime_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                   (uint64_t)threadIdx.x;
    if (idx >= prime_count) return;

    uint64_t p = primes[idx];
    uint64_t inv = inv_p[idx];
    int chunks = base_limb_count * 2;

    const uint32_t *base32 = (const uint32_t *)base_limbs;
    uint64_t r = 0;
    for (int i = chunks - 1; i >= 0; i--) {
        uint64_t v = (r << 32) | base32[i];
        uint64_t q = (uint64_t)(((unsigned __int128)v * inv) >> 64);
        r = v - q * p;
        if (r >= p) r -= p;
    }
    base_mod_p[idx] = r;

    const uint32_t *step32 = (const uint32_t *)step_limbs;
    uint64_t s = 0;
    for (int i = chunks - 1; i >= 0; i--) {
        uint64_t v = (s << 32) | step32[i];
        uint64_t q = (uint64_t)(((unsigned __int128)v * inv) >> 64);
        s = v - q * p;
        if (s >= p) s -= p;
    }
    step_mod_p[idx] = s;
}

/* Euclid modulo-event skip (port of Horizon
   mark_composites_combined_euclid / modulo_search_euclid_u32): returns the
   smallest k >= 1 such that (k*a mod p) lies in [l, r], for 0 < a < p and
   1 <= l <= r < p, via Euclidean continued-fraction descent.  All u32. */
__device__ __forceinline__ static uint32_t
gpu_sieve_modulo_event_euclid_u32(uint32_t p, uint32_t a,
                                  uint32_t l, uint32_t r)
{
    uint32_t stack_p[64];
    uint32_t stack_a[64];
    uint32_t stack_l[64];
    uint32_t depth = 0u;
    uint64_t result = 0u;

    while (true) {
        const uint32_t delta = r - l;
        if (a > (p >> 1u)) {
            l += delta;      /* l -> r_old */
            a = p - a;
            l = p - l;       /* -> p - r_old (negated interval lower bound) */
        }
        const uint32_t l_div = (l - 1u) / a;
        const uint32_t l_mod = l - l_div * a;
        const uint32_t r_mod = l_mod + delta;
        if (r_mod >= a) {
            result = (uint64_t)l_div + 1u;
            break;
        }
        stack_p[depth] = p;
        stack_a[depth] = a;
        stack_l[depth] = l;
        ++depth;
        const uint32_t new_a = a - (p % a);
        p = a;
        a = new_a;
        l = l_mod;
        r = r_mod;
    }

    while (depth != 0u) {
        --depth;
        result = (result * stack_p[depth] + stack_l[depth] - 1u) /
                     stack_a[depth] +
                 1u;
    }
    return (uint32_t)result;
}

/* Rows with a FIXED per-row slot decrement: mark every lattice node's row
   bitmap, skipping whole no-event stretches with one Euclid search each
   (sparse regime: p > window => at most one mark per row).  a = decrement
   mod p, 0 < a < p < 2^32; rows are m = k*row_step. */
__device__ static void gpu_sieve_rows_mark_sparse_lattice(
    uint64_t *row_bitmaps,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    uint64_t p,
    uint64_t a,
    uint64_t pos,
    uint32_t n_rows,
    uint32_t row_step)
{
    const uint32_t p32 = (uint32_t)p;
    const uint32_t a32 = (uint32_t)a;
    uint32_t k = 0;
    uint64_t *wb = row_bitmaps;
    while (k < n_rows) {
        if (pos < odd_interval_size) {
            atomicOr((unsigned long long *)&wb[pos >> 6],
                     (unsigned long long)(1ULL << (pos & 63U)));
            pos = (pos >= a) ? pos - a : pos - a + p;
            k++;
            wb += (size_t)row_step * bitmap_words;
            continue;
        }
        /* Next event lattice-node: smallest j >= 1 with
           (j*a mod p) in [pos - odd_interval_size + 1, pos]. */
        uint32_t j = gpu_sieve_modulo_event_euclid_u32(
            p32, a32, (uint32_t)(pos - odd_interval_size + 1U),
            (uint32_t)pos);
        if (j >= n_rows - k) break;
        k += j;
        wb += (size_t)j * (size_t)row_step * bitmap_words;
        uint64_t ks = ((uint64_t)(j % p32) * a) % p;
        pos = (pos >= ks) ? pos - ks : pos - ks + p;
    }
}

/* Dense rows (>= 1 mark per row) or >32-bit primes: plain row walk with
   incremental slot decrements.  dec_m[fo] = decrement when LEAVING a row
   whose current first-odd-offset grid is fo (0/1); row m's grid is
   fo0 ^ (m & 1). */
__device__ static void gpu_sieve_rows_mark_dense(
    uint64_t *row_bitmaps,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    uint64_t p,
    const uint64_t *dec_m,
    uint64_t fo0,
    uint64_t pos,
    uint32_t row_count)
{
    uint64_t *wb = row_bitmaps;
    for (uint32_t m = 0; m < row_count; m++, wb += bitmap_words) {
        gpu_sieve_mark_progression(wb, odd_interval_size, p, pos);
        uint64_t d = dec_m[(fo0 ^ (uint64_t)(m & 1U)) & 1U];
        pos = (pos >= d) ? pos - d : pos - d + p;
    }
}

/* Fixed grid: pos0 for even rows, pos1 for odd rows (pos1 == pos0 when the
   stride is even). */
__device__ static void gpu_sieve_rows_mark_constant(
    uint64_t *row_bitmaps,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    uint64_t p,
    uint64_t pos0,
    uint64_t pos1,
    int use_alt,
    uint32_t row_count)
{
    uint64_t *wb = row_bitmaps;
    for (uint32_t m = 0; m < row_count; m++, wb += bitmap_words) {
        uint64_t pos = (use_alt && (m & 1U)) ? pos1 : pos0;
        gpu_sieve_mark_progression(wb, odd_interval_size, p, pos);
    }
}

/* ── Split (chunked) dense row-batch marking ────────────────────────────────
   Cost model, measured with tools/bench_mark.cu (2026-09-14): the per-prime
   row walk below (one thread per prime, rows in the inner loop) spends ~73% of
   its time in the marking *loop*, not in the atomic store, and its cost is set
   by the warp with the longest trip count (the p=3..131 warp walks W/3 slots
   per row) -- which is why neither fewer atomics nor fewer primes helped.
   Chunking the (prime, row) lattice into slot ranges with one work item per
   (prime <= W, row, chunk) marks exactly the same slots, was verified
   bit-exact against the row walk (0 differing bits), and is 25-38x faster
   there (21-30 us vs 700-810 us for W=10175, rows=8, 2M primes).

   Semantics reproduced exactly (see gpu_sieve_rows_mark_kernel):
     pos0 = ((p - (base_mod_p + first_odd_offset) mod p) * inv2) mod p
     P even : row decrement is the constant dec; pos_m = pos0 - m*dec
     P odd  : decrements alternate dec0/dec1 (fo flips every row); the row-m
              offset is (m>>1)*(dec0+dec1) + (m&1)*first_dec with
              first_dec = first_odd_offset ? dec1 : dec0.
   Both cases collapse to pos_m = pos0 - (m>>1)*pair - (m&1)*first_dec, with
   pair = dec0+dec1 (P odd) or 2*dec (P even) and first_dec = dec (P even). */
__global__ static void gpu_sieve_rows_mark_prep_kernel(
    const uint64_t *primes,
    const uint64_t *base_mod_p,
    const uint64_t *step_mod_p,
    uint64_t first_odd_offset,
    uint32_t step_odd,
    uint64_t n_small,
    uint64_t *out_pos0,
    uint64_t *out_pair,
    uint64_t *out_first_dec)
{
    uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                 (uint64_t)threadIdx.x;
    if (i >= n_small) return;
    uint64_t p = primes[i];
    if (p < 3U) return;

    uint64_t base = base_mod_p[i];
    uint64_t step = step_mod_p[i];
    const uint64_t inv2 = (p + 1U) >> 1;

    uint64_t rem0 = base + first_odd_offset;
    if (rem0 >= p) rem0 -= p;
    uint64_t pos0 = (((p - rem0) % p) * inv2) % p;

    uint64_t pair, first_dec;
    if (!step_odd) {
        uint64_t dec = (step >> 1) + ((step & 1U) ? inv2 : 0U);
        if (dec >= p) dec -= p;
        pair = (2U * dec) % p;
        first_dec = dec;
    } else {
        uint64_t dec0 = inv2;              /* step == 0 -> (0+1)>>1 == 0 */
        uint64_t dec1 = (p >= inv2) ? (p - inv2) : inv2;   /* -inv2 mod p */
        if (step != 0U) {
            dec0 = ((step + 1U) >> 1) + (((step + 1U) & 1U) ? inv2 : 0U);
            if (dec0 >= p) dec0 -= p;
            dec1 = ((step - 1U) >> 1) + (((step - 1U) & 1U) ? inv2 : 0U);
            if (dec1 >= p) dec1 -= p;
        }
        pair = step;                       /* (dec0 + dec1) mod p */
        first_dec = first_odd_offset ? dec1 : dec0;
    }
    out_pos0[i] = pos0;
    out_pair[i] = pair % p;
    out_first_dec[i] = first_dec % p;
}

__global__ static void gpu_sieve_rows_mark_dense_split_kernel(
    uint64_t *row_bitmaps,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    const uint64_t *primes,
    const uint64_t *pos0,
    const uint64_t *pair,
    const uint64_t *first_dec,
    uint64_t n_small,
    uint32_t row_count,
    uint32_t chunks,
    uint64_t chunk_slots)
{
    uint64_t item = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                    (uint64_t)threadIdx.x;
    uint64_t total = n_small * (uint64_t)row_count * (uint64_t)chunks;
    if (item >= total) return;

    uint32_t c = (uint32_t)(item % chunks);
    uint64_t rest = item / chunks;
    uint32_t m = (uint32_t)(rest % row_count);
    uint64_t i = rest / row_count;

    uint64_t p = primes[i];
    if (p < 3U) return;

    uint64_t begin = (uint64_t)c * chunk_slots;
    if (begin >= odd_interval_size) return;
    uint64_t end = begin + chunk_slots;
    if (end > odd_interval_size) end = odd_interval_size;

    /* Row m's lattice start: pos_m = pos0 - (m>>1)*pair - (m&1)*first_dec. */
    uint64_t d = (((m >> 1) % p) * pair[i] + (uint64_t)(m & 1U) * first_dec[i])
                 % p;
    uint64_t r = pos0[i] % p;
    uint64_t rem = (r >= d) ? (r - d) : (r + p - d);

    /* First marked slot >= begin on this row's lattice. */
    uint64_t first;
    if (begin <= rem) {
        first = rem;
    } else {
        uint64_t off = begin - rem;
        uint64_t k = (off + p - 1U) / p;
        first = rem + k * p;
    }
    if (first >= odd_interval_size) return;

    uint64_t *wb = row_bitmaps + (size_t)m * bitmap_words;
    for (uint64_t q = first; q < end; q += p) {
        atomicOr((unsigned long long *)&wb[q >> 6],
                 (unsigned long long)(1ULL << (q & 63U)));
    }
}

/* Mark row_count rows: row m marks bitmap row_bitmaps + m*bitmap_words.
   offset flips with the row parity ONLY when the row stride P is odd
   (step_odd = 1); P even keeps the same grid for every row.  Sparse primes
   (p > window) use a Euclid modulo-event skip across rows instead of walking
   every row (Horizon mark_composites_combined_euclid).  Marking math matches
   gpu_sieve_mark_kernel_batch (odd-slot space). */
__global__ static void gpu_sieve_rows_mark_kernel(
    uint64_t *row_bitmaps,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    uint64_t first_odd_offset,
    uint32_t step_odd,
    const uint64_t *primes,
    const uint64_t *base_mod_p,
    const uint64_t *step_mod_p,
    uint64_t prime_count,
    uint32_t row_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                   (uint64_t)threadIdx.x;
    if (idx >= prime_count) return;

    uint64_t p = primes[idx];
    if (p < 3U) return;

    uint64_t base = base_mod_p[idx];      /* < p */
    uint64_t step = step_mod_p[idx];      /* < p */
    const uint64_t inv2 = (p + 1U) >> 1;

    uint64_t rem0 = base + first_odd_offset;
    if (rem0 >= p) rem0 -= p;
    uint64_t pos0 = (((p - rem0) % p) * inv2) % p;

    if (!step_odd) {
        /* P even: one odd grid for all rows; per-row decrement constant:
           dec = step * inv2 mod p (0 iff p | P). */
        uint64_t dec = (step >> 1) + ((step & 1U) ? inv2 : 0U);
        if (dec >= p) dec -= p;
        if (dec == 0) {
            /* p | P: every row marks the same slots. */
            gpu_sieve_rows_mark_constant(row_bitmaps, bitmap_words,
                                         odd_interval_size, p, pos0, pos0,
                                         0, row_count);
            return;
        }
        if (p <= odd_interval_size || p >= (1ULL << 32)) {
            uint64_t dec_m[2] = {dec, dec};
            gpu_sieve_rows_mark_dense(row_bitmaps, bitmap_words,
                                      odd_interval_size, p, dec_m, 0,
                                      pos0, row_count);
            return;
        }
        gpu_sieve_rows_mark_sparse_lattice(row_bitmaps, bitmap_words,
                                           odd_interval_size, p, dec, pos0,
                                           row_count, 1);
        return;
    }

    /* P odd: fo flips every row.  When p | P (step == 0) the slot value
       depends only on fo, so rows alternate between pos0 and pos_other.
       Otherwise the decrement alternates between dec0 = (step+1)*inv2
       (leaving an fo=0 row) and dec1 = (step-1)*inv2 (leaving an fo=1 row),
       and two interleaved lattices (even/odd rows) share the PAIR
       decrement dec0 + dec1 = step mod p. */
    if (step == 0) {
        uint64_t pos_other = first_odd_offset
                                 ? ((pos0 + inv2 >= p) ? pos0 + inv2 - p
                                                       : pos0 + inv2)
                                 : ((pos0 >= inv2) ? pos0 - inv2
                                                   : pos0 - inv2 + p);
        gpu_sieve_rows_mark_constant(row_bitmaps, bitmap_words,
                                     odd_interval_size, p, pos0, pos_other,
                                     1, row_count);
        return;
    }
    uint64_t dec0 = ((step + 1U) >> 1) + (((step + 1U) & 1U) ? inv2 : 0U);
    if (dec0 >= p) dec0 -= p;
    uint64_t dec1 = ((step - 1U) >> 1) + (((step - 1U) & 1U) ? inv2 : 0U);
    if (dec1 >= p) dec1 -= p;
    uint64_t dec_leave = first_odd_offset ? dec1 : dec0;
    uint64_t pos1 = (pos0 >= dec_leave) ? pos0 - dec_leave
                                        : pos0 - dec_leave + p;
    if (p <= odd_interval_size || p >= (1ULL << 32)) {
        uint64_t dec_m[2] = {dec0, dec1};
        gpu_sieve_rows_mark_dense(row_bitmaps, bitmap_words,
                                  odd_interval_size, p, dec_m,
                                  first_odd_offset, pos0, row_count);
        return;
    }
    gpu_sieve_rows_mark_sparse_lattice(row_bitmaps, bitmap_words,
                                       odd_interval_size, p, step, pos0,
                                       (row_count + 1U) >> 1, 2);
    if (row_count > 1U) {
        gpu_sieve_rows_mark_sparse_lattice(row_bitmaps + bitmap_words,
                                           bitmap_words, odd_interval_size,
                                           p, step, pos1,
                                           row_count >> 1, 2);
    }
}

/* Single-window chunked dense marking: the rows == 1 shape of the fused path
   (gpu_sieve_mark_from_base).  Same straggler argument as the row-batch and
   pair splits: one thread per prime means the p=3 thread walks W/3 slots while
   the rest of its warp is idle.  Here the residues are already on the device
   (d_base_mod_p from gpu_sieve_residues_kernel), so each work item derives its
   own lattice start and marks one chunk of the window. */
__global__ static void gpu_sieve_mark_dense_split_kernel(
    uint64_t *bitmap,
    uint64_t odd_interval_size,
    uint64_t first_odd_offset,
    const uint64_t *primes,
    const uint64_t *base_mod_p,
    uint64_t n_small,
    uint32_t chunks,
    uint64_t chunk_slots)
{
    uint64_t item = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                    (uint64_t)threadIdx.x;
    if (item >= n_small * (uint64_t)chunks) return;

    uint32_t c = (uint32_t)(item % chunks);
    uint64_t i = item / chunks;

    uint64_t p = primes[i];
    if (p < 3U) return;

    uint64_t begin = (uint64_t)c * chunk_slots;
    if (begin >= odd_interval_size) return;
    uint64_t end = begin + chunk_slots;
    if (end > odd_interval_size) end = odd_interval_size;

    uint64_t remainder = base_mod_p[i] + first_odd_offset;
    if (remainder >= p) remainder -= p;
    uint64_t inverse_two = (p + 1U) >> 1;
    uint64_t rem = (((p - remainder) % p) * inverse_two) % p;

    uint64_t first;
    if (begin <= rem) {
        first = rem;
    } else {
        uint64_t off = begin - rem;
        uint64_t k = (off + p - 1U) / p;
        first = rem + k * p;
    }
    if (first >= odd_interval_size) return;

    for (uint64_t q = first; q < end; q += p) {
        atomicOr((unsigned long long *)&bitmap[q >> 6],
                 (unsigned long long)(1ULL << (q & 63U)));
    }
}

/* Pair-batched fused mark: residues + marking in ONE kernel for TWO windows.
   Each window has its own base (different CRT alignment), so base mod p is
   reduced inline per (window, prime).  Window w ∈ {0,1} marks d_bitmap[w];
   the per-window first_odd_offset is derived from that window's own base
   parity.  Halves the per-window kernel launches and stream syncs of the
   fused pipeline (two windows per mark instead of one). */
__global__ static void gpu_sieve_residues_mark_pair_kernel(
    const uint64_t *base_limbs_pairs, /* 2 × base_limb_count */
    int base_limb_count,
    uint64_t *bitmap0,
    uint64_t *bitmap1,
    uint64_t bitmap_words,
    uint64_t odd_interval_size,
    const uint64_t *primes,
    const uint64_t *inv_p,
    uint64_t prime_count)
{
    uint64_t flat_idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                        (uint64_t)threadIdx.x;
    uint64_t total_threads = 2ULL * prime_count;
    if (flat_idx >= total_threads) return;

    uint64_t window_idx = flat_idx / prime_count;
    uint64_t prime_idx = flat_idx - window_idx * prime_count;

    uint64_t p = primes[prime_idx];
    if (p < 3U) return;

    /* Inline modular reduction base_w mod p (same math as the residues
       kernel, but per window's own base limbs). */
    const uint32_t *base32 = (const uint32_t *)(base_limbs_pairs +
        (size_t)window_idx * (size_t)base_limb_count);
    uint64_t inv = inv_p[prime_idx];
    int chunks = base_limb_count * 2;
    uint64_t r = 0;
    for (int i = chunks - 1; i >= 0; i--) {
        uint64_t v = (r << 32) | base32[i];
        uint64_t q = (uint64_t)(((unsigned __int128)v * inv) >> 64);
        r = v - q * p;
        if (r >= p) r -= p;
    }

    uint64_t *window_bitmap = (window_idx == 0) ? bitmap0 : bitmap1;
    (void)bitmap_words;

    uint64_t first_odd_offset = (base32[0] & 1U) ? 0U : 1U;
    uint64_t remainder = r + (first_odd_offset % p);
    if (remainder >= p) remainder -= p;

    uint64_t inverse_two = (p + 1U) >> 1;
    uint64_t pos = (((p - remainder) % p) * inverse_two) % p;

    gpu_sieve_mark_progression(window_bitmap, odd_interval_size, p, pos);
}

/* ── Split (chunked) pair-batch marking ─────────────────────────────────────
   Same treatment as the row-batch split above, for the rows == 1 path: the
   pair kernel is one thread per (window, prime), so the p=3 thread again walks
   W/3 slots and the kernel runs as long as that straggler.  The dense domain
   (p <= odd_interval_size) is the prefix of the ascending table, so the pair
   kernel keeps handling the suffix while these two kernels take the prefix. */
__global__ static void gpu_sieve_pair_mark_prep_kernel(
    const uint64_t *base_limbs_pairs,   /* 2 × base_limb_count */
    int base_limb_count,
    const uint64_t *primes,
    const uint64_t *inv_p,
    uint64_t n_small,
    uint64_t *out_pos_w0,
    uint64_t *out_pos_w1)
{
    uint64_t flat = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                    (uint64_t)threadIdx.x;
    if (flat >= 2ULL * n_small) return;
    uint64_t window_idx = flat / n_small;
    uint64_t i = flat - window_idx * n_small;

    uint64_t p = primes[i];
    if (p < 3U) return;

    /* Same inline reduction as gpu_sieve_residues_mark_pair_kernel. */
    const uint32_t *base32 = (const uint32_t *)(base_limbs_pairs +
        (size_t)window_idx * (size_t)base_limb_count);
    uint64_t inv = inv_p[i];
    int chunks = base_limb_count * 2;
    uint64_t r = 0;
    for (int k = chunks - 1; k >= 0; k--) {
        uint64_t v = (r << 32) | base32[k];
        uint64_t q = (uint64_t)(((unsigned __int128)v * inv) >> 64);
        r = v - q * p;
        if (r >= p) r -= p;
    }

    uint64_t first_odd_offset = (base32[0] & 1U) ? 0U : 1U;
    uint64_t remainder = r + (first_odd_offset % p);
    if (remainder >= p) remainder -= p;
    uint64_t inverse_two = (p + 1U) >> 1;
    uint64_t pos = (((p - remainder) % p) * inverse_two) % p;

    if (window_idx == 0) out_pos_w0[i] = pos;
    else out_pos_w1[i] = pos;
}

__global__ static void gpu_sieve_pair_mark_dense_split_kernel(
    uint64_t *bitmap0,
    uint64_t *bitmap1,
    uint64_t odd_interval_size,
    const uint64_t *primes,
    const uint64_t *pos_w0,
    const uint64_t *pos_w1,
    uint64_t n_small,
    uint32_t chunks,
    uint64_t chunk_slots)
{
    uint64_t item = (uint64_t)blockIdx.x * (uint64_t)blockDim.x +
                    (uint64_t)threadIdx.x;
    uint64_t total = 2ULL * n_small * (uint64_t)chunks;
    if (item >= total) return;

    uint32_t c = (uint32_t)(item % chunks);
    uint64_t rest = item / chunks;
    uint32_t w = (uint32_t)(rest % 2ULL);
    uint64_t i = rest / 2ULL;

    uint64_t p = primes[i];
    if (p < 3U) return;

    uint64_t begin = (uint64_t)c * chunk_slots;
    if (begin >= odd_interval_size) return;
    uint64_t end = begin + chunk_slots;
    if (end > odd_interval_size) end = odd_interval_size;

    uint64_t rem = (w == 0) ? pos_w0[i] : pos_w1[i];
    rem %= p;
    uint64_t first;
    if (begin <= rem) {
        first = rem;
    } else {
        uint64_t off = begin - rem;
        uint64_t k = (off + p - 1U) / p;
        first = rem + k * p;
    }
    if (first >= odd_interval_size) return;

    uint64_t *wb = (w == 0) ? bitmap0 : bitmap1;
    for (uint64_t q = first; q < end; q += p) {
        atomicOr((unsigned long long *)&wb[q >> 6],
                 (unsigned long long)(1ULL << (q & 63U)));
    }
}

/* Host side of the pair-batched fused mark.  Marks TWO windows (one into
   each ping-pong bitmap) with ONE kernel launch and ONE stream sync.
   base_limbs_pairs: 2 × base_limb_count little-endian limbs, window 0 →
   d_bitmap[0], window 1 → d_bitmap[1].  Returns 1 on success, 0 fail-closed. */
int gpu_sieve_mark_batch_from_bases(gpu_sieve_ctx *ctx,
                                    uint64_t odd_interval_size,
                                    const uint64_t *base_limbs_pairs,
                                    int base_limb_count,
                                    const uint64_t *primes,
                                    const uint64_t *inv_p,
                                    size_t prime_count)
{
    uint64_t start_time = gpu_sieve_clock_us();
    if (!ctx || !base_limbs_pairs || !primes || !inv_p) return 0;
    if (odd_interval_size == 0 || prime_count == 0) return 0;
    if (base_limb_count < 1 ||
        base_limb_count > ctx->base_limbs_capacity) return 0;
    if (prime_count > ctx->max_primes) return 0;

    size_t required_words = (size_t)((odd_interval_size + 63U) >> 6);
    if (required_words == 0 || required_words > ctx->max_bitmap_words) return 0;

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) return 0;

    /* The prime table is fixed for the sieve lifetime: upload once. */
    if (ctx->primes_uploaded_count != prime_count) {
        err = cudaMemcpyAsync(ctx->d_primes, primes,
                              prime_count * sizeof(*primes),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) return 0;
        err = cudaMemcpyAsync(ctx->d_inv_p, inv_p,
                              prime_count * sizeof(*inv_p),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) return 0;
        ctx->primes_uploaded_count = prime_count;
    }

    /* The pair upload goes through d_base_limbs (capacity is enough: the
       single-window mark uses the same buffer for base_limb_count limbs;
       reserve via d_base_offsets buffer? No — upload both bases into the
       d_base_limbs allocation only when it is large enough. */
    if ((size_t)base_limb_count * 2U >
        (size_t)ctx->base_limbs_capacity) {
        return 0;
    }
    err = cudaMemcpyAsync(ctx->d_base_limbs, base_limbs_pairs,
                          (size_t)base_limb_count * 2U * sizeof(uint64_t),
                          cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) return 0;

    err = cudaMemsetAsync(ctx->d_bitmap[0], 0,
                          required_words * sizeof(*ctx->d_bitmap[0]),
                          ctx->stream);
    if (err != cudaSuccess) return 0;
    err = cudaMemsetAsync(ctx->d_bitmap[1], 0,
                          required_words * sizeof(*ctx->d_bitmap[1]),
                          ctx->stream);
    if (err != cudaSuccess) return 0;

    const int tpb = 128;

    /* Optional split (chunked) dense marking (GPU_MARK_SPLIT, same flag as the
       row path): the dense domain (p <= odd_interval_size) is the PREFIX of
       the ascending table, so the pair kernel below can be launched on the
       remaining suffix unchanged.  Fail-closed: not-ascending table, W >= 2^32
       or grid overflow -> the whole table goes through the pair kernel. */
    uint64_t n_small = 0;
    if (gpu_mark_split_enabled() && odd_interval_size < (1ULL << 32) &&
        gpu_mark_primes_ascending(primes, prime_count)) {
        while (n_small < (uint64_t)prime_count &&
               primes[n_small] <= odd_interval_size) {
            n_small++;
        }
    }
    uint64_t rest_count = (uint64_t)prime_count - n_small;
    uint64_t total_threads = 2ULL * rest_count;
    uint64_t blocks = (total_threads + tpb - 1U) / tpb;
    if (rest_count > 0 && blocks > (uint64_t)UINT_MAX) {
        n_small = 0;   /* fail closed: pair kernel covers the whole table */
        rest_count = (uint64_t)prime_count;
        blocks = (2ULL * rest_count + tpb - 1U) / tpb;
        if (blocks > (uint64_t)UINT_MAX) return 0;
    }

    sieve_timing_begin(ctx, 1);
    if (n_small > 0) {
        uint64_t pblocks = (2ULL * n_small + tpb - 1U) / tpb;
        gpu_sieve_pair_mark_prep_kernel<<<(unsigned)pblocks, tpb, 0,
                                          ctx->stream>>>(
            ctx->d_base_limbs, base_limb_count, ctx->d_primes, ctx->d_inv_p,
            n_small, ctx->d_mark_pos0, ctx->d_mark_pos1);

        uint64_t slots = gpu_mark_split_slots();
        if (slots < 1) slots = 1;
        uint64_t chunks = (odd_interval_size + slots - 1U) / slots;
        if (chunks < 1) chunks = 1;
        if (chunks > 1024U) chunks = 1024U;
        uint64_t items = 2ULL * n_small * chunks;
        uint64_t iblocks = (items + tpb - 1U) / tpb;
        if (iblocks > 0 && iblocks <= (uint64_t)UINT_MAX) {
            gpu_sieve_pair_mark_dense_split_kernel<<<(unsigned)iblocks, tpb, 0,
                                                     ctx->stream>>>(
                ctx->d_bitmap[0], ctx->d_bitmap[1], odd_interval_size,
                ctx->d_primes, ctx->d_mark_pos0, ctx->d_mark_pos1, n_small,
                (uint32_t)chunks, slots);
        } else {
            n_small = 0;
            rest_count = (uint64_t)prime_count;
            blocks = (2ULL * rest_count + tpb - 1U) / tpb;
            if (blocks > (uint64_t)UINT_MAX) return 0;
        }
    }
    if (rest_count > 0) {
        gpu_sieve_residues_mark_pair_kernel<<<(unsigned int)blocks, tpb, 0,
                                              ctx->stream>>>(
            ctx->d_base_limbs, base_limb_count,
            ctx->d_bitmap[0], ctx->d_bitmap[1], (uint64_t)required_words,
            odd_interval_size, ctx->d_primes + n_small,
            ctx->d_inv_p + n_small, rest_count);
    }
    sieve_timing_end(ctx);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: pair mark launch failed: %s\n",
                cudaGetErrorString(err));
        ctx->t_stage = 0;
        return 0;
    }

    err = cudaStreamSynchronize(ctx->stream);
    sieve_timing_drain(ctx);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: pair mark stream sync failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    uint64_t end_time = gpu_sieve_clock_us();
    ctx->last_elapsed_us = end_time >= start_time ? end_time - start_time : 0;
    return 1;
}

int gpu_sieve_mark_from_base(gpu_sieve_ctx *ctx,
                             uint64_t odd_interval_size,
                             uint64_t first_odd_offset,
                             const uint64_t *base_limbs,
                             int base_limb_count,
                             int buf,
                             const uint64_t *primes,
                             const uint64_t *inv_p,
                             size_t prime_count,
                             uint64_t *host_bitmap,
                             size_t host_bitmap_words) {
    uint64_t start_time = gpu_sieve_clock_us();
    if (!ctx || !base_limbs || !primes || !inv_p) return 0;
    if (odd_interval_size == 0 || prime_count == 0) return 0;
    if (first_odd_offset > 1U) return 0;
    if (base_limb_count < 1 ||
        base_limb_count > ctx->base_limbs_capacity) return 0;
    if (prime_count > ctx->max_primes) return 0;

    size_t required_words = (size_t)((odd_interval_size + 63U) >> 6);
    if (required_words == 0 || required_words > ctx->max_bitmap_words) return 0;
    if (host_bitmap && required_words > host_bitmap_words) return 0;

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) return 0;

    /* The prime table is fixed for the sieve lifetime: upload once. */
    if (ctx->primes_uploaded_count != prime_count) {
        err = cudaMemcpyAsync(ctx->d_primes, primes,
                              prime_count * sizeof(*primes),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: H2D primes (from base) failed: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        err = cudaMemcpyAsync(ctx->d_inv_p, inv_p,
                              prime_count * sizeof(*inv_p),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: H2D inv_p (from base) failed: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        ctx->primes_uploaded_count = prime_count;
    }

    err = cudaMemcpyAsync(ctx->d_base_limbs, base_limbs,
                          (size_t)base_limb_count * sizeof(*base_limbs),
                          cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: H2D base_limbs failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    uint64_t zero_offset = 0;
    err = cudaMemcpyAsync(ctx->d_base_offsets, &zero_offset,
                          sizeof(zero_offset), cudaMemcpyHostToDevice,
                          ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: H2D base_offsets failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemsetAsync(ctx->d_bitmap[buf & 1], 0,
                          required_words * sizeof(*ctx->d_bitmap[0]),
                          ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMemset(bitmap) failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    const int tpb = 128;
    uint64_t prime_count_u64 = (uint64_t)prime_count;
    uint64_t blocks = (prime_count_u64 + tpb - 1U) / tpb;
    if (blocks > (uint64_t)UINT_MAX) return 0;

    gpu_sieve_residues_kernel<<<(unsigned int)blocks, tpb, 0, ctx->stream>>>(
        ctx->d_base_limbs, base_limb_count, ctx->d_primes, ctx->d_inv_p,
        ctx->d_base_mod_p, prime_count_u64);

    sieve_timing_begin(ctx, 1);

    /* Optional split (chunked) dense marking (GPU_MARK_SPLIT): the dense
       domain (p <= odd_interval_size) is the PREFIX of the ascending table and
       is marked by chunked work items; the remaining suffix still goes through
       gpu_sieve_mark_kernel_batch unchanged.  Fail-closed: non-ascending
       table, W >= 2^32 or grid overflow -> the whole table as before. */
    uint64_t n_small = 0;
    if (gpu_mark_split_enabled() && odd_interval_size < (1ULL << 32) &&
        gpu_mark_primes_ascending(primes, (size_t)prime_count)) {
        while (n_small < prime_count_u64 &&
               primes[n_small] <= odd_interval_size) {
            n_small++;
        }
    }
    uint64_t rest_count = prime_count_u64 - n_small;
    if (n_small > 0) {
        uint64_t slots = gpu_mark_split_slots();
        if (slots < 1) slots = 1;
        uint64_t chunks = (odd_interval_size + slots - 1U) / slots;
        if (chunks < 1) chunks = 1;
        if (chunks > 1024U) chunks = 1024U;
        uint64_t items = n_small * chunks;
        uint64_t iblocks = (items + tpb - 1U) / tpb;
        if (iblocks > 0 && iblocks <= (uint64_t)UINT_MAX) {
            gpu_sieve_mark_dense_split_kernel<<<(unsigned)iblocks, tpb, 0,
                                                ctx->stream>>>(
                ctx->d_bitmap[buf & 1], odd_interval_size, first_odd_offset,
                ctx->d_primes, ctx->d_base_mod_p, n_small, (uint32_t)chunks,
                slots);
        } else {
            n_small = 0;
            rest_count = prime_count_u64;
        }
    }
    if (rest_count > 0) {
        uint64_t rblocks = (rest_count + tpb - 1U) / tpb;
        if (rblocks > (uint64_t)UINT_MAX) return 0;
        gpu_sieve_mark_kernel_batch<<<(unsigned int)rblocks, tpb, 0,
                                      ctx->stream>>>(
            ctx->d_bitmap[buf & 1], (uint64_t)required_words,
            odd_interval_size, first_odd_offset, ctx->d_base_offsets, 1,
            ctx->d_primes + n_small, ctx->d_base_mod_p + n_small, rest_count);
    }
    sieve_timing_end(ctx);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: kernel launch (from base) failed: %s\n",
                cudaGetErrorString(err));
        ctx->t_stage = 0;
        return 0;
    }

    if (host_bitmap) {
        err = cudaMemcpyAsync(host_bitmap, ctx->d_bitmap[buf & 1],
                              required_words * sizeof(*host_bitmap),
                              cudaMemcpyDeviceToHost, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: D2H bitmap (from base) failed: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
    }

    err = cudaStreamSynchronize(ctx->stream);
    sieve_timing_drain(ctx);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: stream sync failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    uint64_t end_time = gpu_sieve_clock_us();
    ctx->last_elapsed_us = end_time >= start_time ? end_time - start_time : 0;
    return 1;
}

/* Host side of the row-batch fused mark: ONE residues pass for the base
   AND the row stride P, then mark row_count bitmaps (rows 0..row_count-1 of
   the row arena).  Amortizes the per-window residue cost over the batch.
   Returns 1 on success, 0 fail-closed. */
int gpu_sieve_mark_rows_from_base(gpu_sieve_ctx *ctx,
                                  uint64_t odd_interval_size,
                                  uint64_t first_odd_offset,
                                  const uint64_t *base_limbs,
                                  const uint64_t *step_limbs,
                                  int base_limb_count,
                                  uint32_t row_count,
                                  const uint64_t *primes,
                                  const uint64_t *inv_p,
                                  size_t prime_count)
{
    uint64_t start_time = gpu_sieve_clock_us();
    if (!ctx || !base_limbs || !step_limbs || !primes || !inv_p) return 0;
    if (odd_interval_size == 0 || prime_count == 0 || row_count == 0) {
        return 0;
    }
    if (first_odd_offset > 1U) return 0;
    if (base_limb_count < 1 ||
        (size_t)base_limb_count * 2U > (size_t)ctx->base_limbs_capacity) {
        return 0;
    }
    if (prime_count > ctx->max_primes) return 0;

    size_t required_words = (size_t)((odd_interval_size + 63U) >> 6);
    if (required_words == 0 || required_words > ctx->max_bitmap_words) {
        return 0;
    }

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) return 0;

    /* Grow the row bitmap arena (rows × max_bitmap_words). */
    if (row_count > ctx->row_bitmap_cap) {
        uint64_t *new_rows = NULL;
        err = cudaMalloc(&new_rows,
                         (size_t)row_count * ctx->max_bitmap_words *
                             sizeof(uint64_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: row bitmaps alloc: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        if (ctx->d_row_bitmaps) cudaFree(ctx->d_row_bitmaps);
        ctx->d_row_bitmaps = new_rows;
        ctx->row_bitmap_cap = row_count;
    }

    /* The prime table is fixed for the sieve lifetime: upload once. */
    if (ctx->primes_uploaded_count != prime_count) {
        err = cudaMemcpyAsync(ctx->d_primes, primes,
                              prime_count * sizeof(*primes),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: row H2D primes failed: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        err = cudaMemcpyAsync(ctx->d_inv_p, inv_p,
                              prime_count * sizeof(*inv_p),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: row H2D inv_p failed: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        ctx->primes_uploaded_count = prime_count;
    }

    /* base into d_base_limbs[0..n), step into d_base_limbs[n..2n). */
    err = cudaMemcpyAsync(ctx->d_base_limbs, base_limbs,
                          (size_t)base_limb_count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: row H2D base_limbs failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }
    err = cudaMemcpyAsync(ctx->d_base_limbs + base_limb_count, step_limbs,
                          (size_t)base_limb_count * sizeof(uint64_t),
                          cudaMemcpyHostToDevice, ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: row H2D step_limbs failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaMemsetAsync(ctx->d_row_bitmaps, 0,
                          (size_t)row_count * ctx->max_bitmap_words *
                              sizeof(uint64_t),
                          ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: row bitmap memset failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    const int tpb = 128;
    uint64_t prime_count_u64 = (uint64_t)prime_count;
    uint64_t blocks = (prime_count_u64 + tpb - 1U) / tpb;
    if (blocks > (uint64_t)UINT_MAX) return 0;

    gpu_sieve_rows_residues_kernel<<<(unsigned int)blocks, tpb, 0,
                                     ctx->stream>>>(
        ctx->d_base_limbs, ctx->d_base_limbs + base_limb_count,
        base_limb_count, ctx->d_primes, ctx->d_inv_p, ctx->d_base_mod_p,
        ctx->d_step_mod_p, prime_count_u64);

    /* Arena stride: rows are laid out at max_bitmap_words intervals (the
       extractor reads row r at row_bitmaps + r*max_bitmap_words). */
    sieve_timing_begin(ctx, 1);
    uint32_t step_odd = (uint32_t)(step_limbs[0] & 1ULL);

    /* Optional split (chunked) dense marking (GPU_MARK_SPLIT=1).  The dense
       domain (p <= odd_interval_size) is a PREFIX of the sorted prime table,
       so the row walk below can be launched on the remaining suffix without
       changing its semantics at all.  Default off while it earns its numbers
       in production (isolated measurement: bit-exact against the row walk and
       25-38x faster, tools/bench_mark.cu). */
    uint64_t n_small = 0;
    if (gpu_mark_split_enabled() && odd_interval_size < (1ULL << 32) &&
        gpu_mark_primes_ascending(primes, (size_t)prime_count)) {
        while (n_small < prime_count_u64 &&
               primes[n_small] <= odd_interval_size) {
            n_small++;
        }
    }
    if (n_small > 0) {
        uint64_t pblocks = (n_small + 127U) / 128U;
        gpu_sieve_rows_mark_prep_kernel<<<(unsigned)pblocks, 128, 0,
                                          ctx->stream>>>(
            ctx->d_primes, ctx->d_base_mod_p, ctx->d_step_mod_p,
            first_odd_offset, step_odd, n_small, ctx->d_mark_pos0,
            ctx->d_mark_pair, ctx->d_mark_firstdec);

        uint64_t slots = gpu_mark_split_slots();
        if (slots < 1) slots = 1;
        uint64_t chunks = (odd_interval_size + slots - 1U) / slots;
        if (chunks < 1) chunks = 1;
        if (chunks > 1024U) chunks = 1024U;
        uint64_t items = n_small * (uint64_t)row_count * chunks;
        uint64_t iblocks = (items + 127U) / 128U;
        if (iblocks > 0 && iblocks <= (uint64_t)UINT_MAX) {
            gpu_sieve_rows_mark_dense_split_kernel<<<(unsigned)iblocks, 128, 0,
                                                     ctx->stream>>>(
                ctx->d_row_bitmaps, (uint64_t)ctx->max_bitmap_words,
                odd_interval_size, ctx->d_primes, ctx->d_mark_pos0,
                ctx->d_mark_pair, ctx->d_mark_firstdec, n_small, row_count,
                (uint32_t)chunks, slots);
        } else {
            n_small = 0;   /* fail closed: row walk covers the whole table */
        }
    }

    uint64_t rest_count = prime_count_u64 - n_small;
    if (rest_count > 0) {
        uint64_t rblocks = (rest_count + tpb - 1U) / tpb;
        gpu_sieve_rows_mark_kernel<<<(unsigned int)rblocks, tpb, 0,
                                     ctx->stream>>>(
            ctx->d_row_bitmaps, (uint64_t)ctx->max_bitmap_words,
            odd_interval_size, first_odd_offset, step_odd,
            ctx->d_primes + n_small, ctx->d_base_mod_p + n_small,
            ctx->d_step_mod_p + n_small, rest_count, row_count);
    }
    sieve_timing_end(ctx);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: row mark kernel launch failed: %s\n",
                cudaGetErrorString(err));
        ctx->t_stage = 0;
        return 0;
    }

    err = cudaStreamSynchronize(ctx->stream);
    sieve_timing_drain(ctx);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: row mark stream sync failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    uint64_t end_time = gpu_sieve_clock_us();
    ctx->last_elapsed_us = end_time >= start_time ? end_time - start_time : 0;
    return 1;
}

const char *gpu_sieve_device_name(const gpu_sieve_ctx *ctx) {
    return ctx ? ctx->dev_name : "";
}

/* ═══════════════════════════════════════════════════════════════════
 * Fused pipeline Stage 1: device-side extract + pack (K2).
 *
 * Scans the marked bitmap (bit=1 composite) and for every survivor packs
 *   candidate = base + first_odd_offset + 2*odd_pos
 * into AoS little-endian limbs, recording the full adder offset.  This is the
 * kernel that removes the CPU extraction/packing round-trip.
 *
 * ORDERED stream compaction in ONE block / ONE kernel: each thread owns one
 * bitmap word, counts its survivors, runs a shared-memory exclusive scan, then
 * scatters in ascending offset order (slot = prefix + local rank).  The host
 * then needs no qsort of (offset, is_prime) pairs before gap detection, and
 * there are no extra count/scan kernel launches.  Requires <= 1023 bitmap
 * words (enforced by the host side). */
__global__ static void gpu_sieve_extract_pack_kernel(
        const uint64_t *bitmap,
        uint64_t bitmap_words,
        uint64_t odd_interval_size,
        uint64_t first_odd_offset,
        const uint64_t *base_limbs,
        int active_limbs,
        uint64_t *cands_aos,
        uint64_t *offsets,
        unsigned int *d_count,
        uint32_t max_batch,
        uint64_t lo_odd,
        uint64_t hi_odd,
        uint32_t half_filter,
        uint32_t base_mod60,
        uint64_t class_mask,
        uint64_t region_start,
        uint32_t slot_base)
{
    __shared__ uint32_t sdata[1024];

    uint64_t w_lo = lo_odd >> 6;
    uint64_t w_hi = (hi_odd + 63U) >> 6;
    if (w_hi > bitmap_words) w_hi = bitmap_words;
    uint32_t n = (w_hi > w_lo) ? (uint32_t)(w_hi - w_lo) : 0;
    uint32_t tid = threadIdx.x;

    /* Phase 1: per-word survivor counts. */
    uint32_t cnt = 0;
    if (tid < n) {
        uint64_t w = w_lo + tid;
        uint64_t s = ~bitmap[w];
        if (w == w_lo && (lo_odd & 63U) != 0)
            s &= ~((1ULL << (lo_odd & 63U)) - 1ULL);
        if (w + 1U == w_hi && (hi_odd & 63U) != 0)
            s &= (1ULL << (hi_odd & 63U)) - 1ULL;
        if (half_filter) {
            uint64_t keep = 0;
            uint64_t t = s;
            uint64_t pos = w << 6;
            while (t) {
                unsigned int b = (unsigned int)__ffsll((long long)t) - 1U;
                t &= t - 1ULL;
                uint64_t offset = first_odd_offset + ((pos + b) << 1);
                if (offset < region_start) {
                    keep |= 1ULL << b;
                } else {
                    uint32_t v = (uint32_t)((base_mod60 +
                                             (offset % 60ULL)) % 60ULL);
                    if ((class_mask >> v) & 1U) keep |= 1ULL << b;
                }
            }
            s = keep;
        }
        cnt = (uint32_t)__popcll(s);
    }
    sdata[tid] = cnt;
    __syncthreads();

    /* Phase 2: exclusive scan over the padded 1024-element array. */
    for (uint32_t s = 1; s < 1024; s <<= 1) {
        uint32_t idx = (tid + 1U) * (s << 1) - 1U;
        if (idx < 1024U) sdata[idx] += sdata[idx - s];
        __syncthreads();
    }
    if (tid == 0) sdata[1023] = 0U;
    __syncthreads();
    for (uint32_t s = 512; s > 0; s >>= 1) {
        uint32_t idx = (tid + 1U) * (s << 1) - 1U;
        if (idx < 1024U) {
            uint32_t t = sdata[idx];
            sdata[idx] += sdata[idx - s];
            sdata[idx - s] = t;
        }
        __syncthreads();
    }

    if (tid == 0 && d_count) *d_count = (n > 0) ? sdata[n] : 0;
    __syncthreads();

    /* Phase 3: ordered scatter (slot = prefix + local rank). */
    if (tid < n) {
        uint64_t w = w_lo + tid;
        uint64_t survivors = ~bitmap[w];
        if (w == w_lo && (lo_odd & 63U) != 0)
            survivors &= ~((1ULL << (lo_odd & 63U)) - 1ULL);
        if (w + 1U == w_hi && (hi_odd & 63U) != 0)
            survivors &= (1ULL << (hi_odd & 63U)) - 1ULL;
        if (half_filter) {
            uint64_t keep = 0;
            uint64_t t = survivors;
            uint64_t pos = w << 6;
            while (t) {
                unsigned int b = (unsigned int)__ffsll((long long)t) - 1U;
                t &= t - 1ULL;
                uint64_t offset = first_odd_offset + ((pos + b) << 1);
                if (offset < region_start) {
                    keep |= 1ULL << b;
                } else {
                    uint32_t v = (uint32_t)((base_mod60 +
                                             (offset % 60ULL)) % 60ULL);
                    if ((class_mask >> v) & 1U) keep |= 1ULL << b;
                }
            }
            survivors = keep;
        }

        uint32_t rank = 0;
        while (survivors) {
            unsigned int b = (unsigned int)__ffsll((long long)survivors) - 1U;
            survivors &= survivors - 1ULL;

            uint64_t odd_pos = (w << 6) + (uint64_t)b;
            if (odd_pos < lo_odd || odd_pos >= hi_odd) continue;

            uint32_t slot = sdata[tid] + rank + slot_base;
            rank++;
            /* max_batch is the candidate buffer capacity, NOT the range: a
               survivor that does not fit must never be written (overrun)
               and must never be dropped silently either (a dropped prime
               makes two non-consecutive primes look consecutive = a false
               gap).  Signal it in the HIGH BIT of the count word, which the
               host already reads, so the check costs no extra CUDA call.
               The count itself is < 2^31 (bounded by the bitmap range). */
            if (slot >= max_batch) {
                if (d_count) atomicOr(d_count, 0x80000000U);
                continue;
            }

            uint64_t offset = first_odd_offset + (odd_pos << 1);
            uint64_t *cand = cands_aos + (uint64_t)slot * (uint64_t)active_limbs;
            uint64_t carry = offset;
            for (int i = 0; i < active_limbs; i++) {
                uint64_t bi = base_limbs[i];
                uint64_t sum = bi + carry;
                cand[i] = sum;
                carry = (sum < bi) ? 1ULL : 0ULL;
            }
            offsets[slot] = offset;
        }
    }
}

/* Extract survivors + pack candidates from the currently-marked d_bitmap.
   Internal helper: copies count + offsets (and optionally packed candidates)
   back to the host.  host_cands_aos may be NULL to keep candidates on-device
   (the Stage 2/3 fused path).  Only odd positions in [lo_odd, hi_odd) are
   extracted, so the caller can slice the window into head/tail.  Output is in
   ASCENDING offset order (ordered compaction).  cand_buf selects which of the
   two device candidate buffers to write (async ping-pong). */
static int gpu_sieve_extract_pack_impl(gpu_sieve_ctx *ctx,
                                       uint64_t odd_interval_size,
                                       uint64_t first_odd_offset,
                                       uint64_t lo_odd,
                                       uint64_t hi_odd,
                                       const uint64_t *bitmap,
                                       int cand_buf,
                                       const uint64_t *base_limbs,
                                       int active_limbs,
                                       uint64_t *host_cands_aos,
                                       uint64_t *host_offsets,
                                       unsigned int *host_count,
                                       uint32_t base_mod60,
                                       uint64_t class_mask60,
                                       uint64_t region_start,
                                       uint32_t slot_base)
{
    if (!ctx || !bitmap || !base_limbs || !host_offsets || !host_count) {
        return 0;
    }
    if (odd_interval_size == 0 || active_limbs < 1) return 0;
    if (first_odd_offset > 1U) return 0;
    if (active_limbs > ctx->base_limbs_capacity) return 0;
    if (lo_odd >= hi_odd || hi_odd > odd_interval_size) return 0;

    size_t words = (size_t)((odd_interval_size + 63U) >> 6);
    if (words == 0 || words > ctx->max_bitmap_words) return 0;
    if (words > 1023) return 0;   /* single-block scan capacity */

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) return 0;

    /* ---- Candidate-buffer RIGHT-SIZING ----------------------------------
       The buffer holds SURVIVORS, a small fraction of the odd slots
       (measured 1,000 of 17,916 per window at shift507 / 2M sieve primes),
       so one slot per odd position - what this used to allocate - is a ~18x
       VRAM over-allocation, and that over-allocation is what capped
       MINING_JUMP2_BATCH (and therefore the MR batch that feeds the GPU).
       Capacity is now: a small initial guess, right-sized after the first
       measured window, and GROWN (never truncated) if a later window needs
       more.  A dropped survivor would make two non-consecutive primes look
       consecutive - i.e. a false gap - so overflow is a hard signal, not a
       silent skip. */
    size_t accum = ctx->extract_accum ? (size_t)ctx->extract_accum : 1;
    /* A pending change is only ever applied when nothing is accumulated
       (slot_base == 0): mid-flight the accumulated layout would be
       invalidated. */
    if (slot_base == 0 && ctx->cand_cap_pending) {
        ctx->cand_cap_per_window = ctx->cand_cap_pending;
        ctx->cand_cap_pending = 0;
        ctx->max_candidates = 0;            /* force the realloc below */
    }
    if (ctx->cand_cap_per_window == 0) {
        ctx->cand_cap_per_window = ctx->cand_cap_env
                                       ? (uint64_t)ctx->cand_cap_env
                                       : gpu_sieve_cand_cap_estimate(
                                             ctx->max_odd_interval);
    }
    if (ctx->cand_cap_per_window > ctx->max_odd_interval)
        ctx->cand_cap_per_window = ctx->max_odd_interval;
    /* Honour the caller's host allocation: the host arrays hold the same
       dense layout, so the device must never be able to hold more. */
    if (ctx->cand_cap_limit) {
        uint64_t max_per_win = ctx->cand_cap_limit / accum;
        if (max_per_win == 0) {
            fprintf(stderr,
                    "gpu_sieve: host candidate capacity %llu slots is smaller "
                    "than one window (K=%u)\n",
                    (unsigned long long)ctx->cand_cap_limit,
                    (unsigned)ctx->extract_accum);
            return 0;
        }
        if (ctx->cand_cap_per_window > max_per_win)
            ctx->cand_cap_per_window = max_per_win;
    }
    size_t cap = (size_t)ctx->cand_cap_per_window * accum;
    if (ctx->cand_cap_limit && cap > (size_t)ctx->cand_cap_limit)
        cap = (size_t)ctx->cand_cap_limit;
    if (cap > (size_t)0xFFFFFFFFu) cap = (size_t)0xFFFFFFFFu;

    uint64_t w_lo = lo_odd >> 6;
    uint64_t w_hi = (hi_odd + 63U) >> 6;
    if (w_hi > (uint64_t)words) w_hi = (uint64_t)words;
    if (w_lo >= w_hi) return 0;

    /* Single-block ordered compaction: one thread per bitmap word. */
    uint32_t half_filter = (class_mask60 != UINT64_MAX) ? 1U : 0U;
    unsigned int cnt = 0;
    unsigned int ovf = 0;
    int attempt = 0;

    for (;;) {
        /* (Re)allocate the extract buffers if the capacity or the limb width
           changed.  Allocating the right size ONCE and never moving it is what
           keeps async MR kernels safe: a mid-run realloc would free buffers
           referenced by in-flight CGBN kernels, hence the device sync. */
        if (ctx->max_candidates != cap ||
            ctx->active_limbs_capacity != active_limbs ||
            !ctx->d_cands_aos[0] || !ctx->d_cands_aos[1] ||
            !ctx->d_offsets || !ctx->d_count) {
            cudaDeviceSynchronize();
            if (ctx->d_cands_aos[0]) cudaFree(ctx->d_cands_aos[0]);
            if (ctx->d_cands_aos[1]) cudaFree(ctx->d_cands_aos[1]);
            if (ctx->d_offsets)   cudaFree(ctx->d_offsets);
            if (ctx->d_count)     cudaFree(ctx->d_count);
            ctx->d_cands_aos[0] = NULL;
            ctx->d_cands_aos[1] = NULL;
            ctx->d_offsets = NULL;
            ctx->d_count = NULL;
            ctx->max_candidates = 0;

            err = cudaMalloc(&ctx->d_cands_aos[0],
                             cap * (size_t)active_limbs * sizeof(uint64_t));
            if (err != cudaSuccess) {
                /* Fail-closed is correct here (the worker drops to the CPU
                   sieve), but a bare "out of memory" hides how much was needed
                   and why, and the CPU path is ~600x slower -- print the
                   numbers so an out-of-VRAM run cannot be mistaken for a slow
                   GPU. */
                size_t want = cap * (size_t)active_limbs * sizeof(uint64_t);
                size_t vfree = 0, vtotal = 0;
                cudaMemGetInfo(&vfree, &vtotal);
                fprintf(stderr,
                        "gpu_sieve: cands[0] alloc: %s\n"
                        "gpu_sieve:   needs 2 x %.0f MiB candidate buffers "
                        "(window %llu odd slots, capacity %llu slots/window "
                        "x K=%u x %d limbs); device free %.0f of %.0f MiB\n"
                        "gpu_sieve:   lower MINING_JUMP2_BATCH (now %u) or free VRAM;\n"
                        "gpu_sieve:   otherwise this worker falls back to the CPU "
                        "sieve, which is ~600x slower (measured 19 vs 11751 win/s)\n",
                        cudaGetErrorString(err), (double)want / (1024.0 * 1024.0),
                        (unsigned long long)ctx->max_odd_interval,
                        (unsigned long long)ctx->cand_cap_per_window,
                        (unsigned)ctx->extract_accum, active_limbs,
                        (double)vfree / (1024.0 * 1024.0),
                        (double)vtotal / (1024.0 * 1024.0),
                        (unsigned)ctx->extract_accum);
                return 0;
            }
            err = cudaMalloc(&ctx->d_cands_aos[1],
                             cap * (size_t)active_limbs * sizeof(uint64_t));
            if (err != cudaSuccess) {
                size_t want = cap * (size_t)active_limbs * sizeof(uint64_t);
                size_t vfree = 0, vtotal = 0;
                cudaMemGetInfo(&vfree, &vtotal);
                fprintf(stderr,
                        "gpu_sieve: cands[1] alloc: %s\n"
                        "gpu_sieve:   needs 2 x %.0f MiB candidate buffers "
                        "(window %llu odd slots, capacity %llu slots/window "
                        "x K=%u x %d limbs); device free %.0f of %.0f MiB\n"
                        "gpu_sieve:   lower MINING_JUMP2_BATCH (now %u) or free VRAM;\n"
                        "gpu_sieve:   otherwise this worker falls back to the CPU "
                        "sieve, which is ~600x slower (measured 19 vs 11751 win/s)\n",
                        cudaGetErrorString(err), (double)want / (1024.0 * 1024.0),
                        (unsigned long long)ctx->max_odd_interval,
                        (unsigned long long)ctx->cand_cap_per_window,
                        (unsigned)ctx->extract_accum, active_limbs,
                        (double)vfree / (1024.0 * 1024.0),
                        (double)vtotal / (1024.0 * 1024.0),
                        (unsigned)ctx->extract_accum);
                return 0;
            }
            err = cudaMalloc(&ctx->d_offsets, cap * sizeof(uint64_t));
            if (err != cudaSuccess) { fprintf(stderr, "gpu_sieve: offsets alloc: %s\n",
                                              cudaGetErrorString(err)); return 0; }
            err = cudaMalloc(&ctx->d_count, sizeof(unsigned int));
            if (err != cudaSuccess) { fprintf(stderr, "gpu_sieve: count alloc: %s\n",
                                              cudaGetErrorString(err)); return 0; }
            ctx->max_candidates = cap;
            ctx->active_limbs_capacity = active_limbs;
        }

        /* Upload the base (independent of whether mark_from_base already did). */
        err = cudaMemcpyAsync(ctx->d_base_limbs, base_limbs,
                              (size_t)active_limbs * sizeof(*base_limbs),
                              cudaMemcpyHostToDevice, ctx->stream);
        if (err != cudaSuccess) return 0;

        sieve_timing_begin(ctx, 2);
        gpu_sieve_extract_pack_kernel<<<1, 1024, 0, ctx->stream>>>(
            bitmap, (uint64_t)words, odd_interval_size,
            first_odd_offset, ctx->d_base_limbs, active_limbs,
            ctx->d_cands_aos[cand_buf & 1], ctx->d_offsets, ctx->d_count,
            (uint32_t)ctx->max_candidates, lo_odd, hi_odd, half_filter,
            base_mod60, class_mask60, region_start, slot_base);
        sieve_timing_end(ctx);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: extract_pack launch: %s\n",
                    cudaGetErrorString(err));
            ctx->t_stage = 0;
            return 0;
        }

        /* Read the survivor count first (async + stream sync), then copy only
           the valid entries.  Using a full-buffer copy would transfer ~1 MB of
           dead slots per window; using synchronous cudaMemcpy would insert
           DEVICE-WIDE syncs that serialize concurrent workers' streams.  The
           count's high bit is the capacity-overflow flag (see the kernel). */
        cnt = 0;
        ovf = 0;
        err = cudaMemcpyAsync(&cnt, ctx->d_count, sizeof(cnt),
                              cudaMemcpyDeviceToHost, ctx->stream);
        if (err != cudaSuccess) return 0;
        err = cudaStreamSynchronize(ctx->stream);
        sieve_timing_drain(ctx);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: extract_pack count sync: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
        if (cnt & 0x80000000U) {
            ovf = 1;
            cnt &= 0x7FFFFFFFU;
        }

        if (!ovf) break;

        /* Denser window than the reserved capacity: grow, never truncate. */
        uint64_t need = (uint64_t)cnt + 64U;
        uint64_t by_slot =
            ((uint64_t)slot_base + (uint64_t)cnt + 64U + accum - 1U) / accum;
        if (by_slot > need) need = by_slot;
        uint64_t grown = ctx->cand_cap_per_window * 2U;
        if (grown < need) grown = need;
        if (grown > ctx->max_odd_interval) grown = ctx->max_odd_interval;
        if (ctx->cand_cap_limit) {
            uint64_t max_per_win = ctx->cand_cap_limit / accum;
            if (grown > max_per_win) grown = max_per_win;
        }
        if (!ctx->cand_cap_reported) {
            ctx->cand_cap_reported = 1;
            fprintf(stderr,
                    "gpu_sieve: candidate buffer too small: %u survivors do not "
                    "fit in %llu slots/window x K=%u; raising to %llu "
                    "slots/window (this is the survivor density, not a bug)\n",
                    cnt, (unsigned long long)ctx->cand_cap_per_window,
                    (unsigned)ctx->extract_accum, (unsigned long long)grown);
        }
        if (grown <= ctx->cand_cap_per_window || attempt >= 3) {
            fprintf(stderr,
                    "gpu_sieve: candidate buffer cannot grow further "
                    "(%llu slots/window x K=%u already); failing this "
                    "extraction closed rather than dropping survivors\n",
                    (unsigned long long)ctx->cand_cap_per_window,
                    (unsigned)ctx->extract_accum);
            ctx->cand_cap_pending = 0;
            ctx->max_candidates = 0;
            return 0;
        }
        if (slot_base != 0) {
            /* The accumulated layout of this flight is invalid (a window was
               truncated); apply the new capacity at the next flight boundary
               and fail this call so the caller falls back for this window. */
            ctx->cand_cap_pending = grown;
            ctx->max_candidates = 0;
            return 0;
        }
        ctx->cand_cap_per_window = grown;
        cap = (size_t)ctx->cand_cap_per_window * accum;
        if (cap > (size_t)0xFFFFFFFFu) cap = (size_t)0xFFFFFFFFu;
        ctx->max_candidates = 0;    /* force the realloc in the next round */
        attempt++;
    }

    if ((uint64_t)cnt > (hi_odd - lo_odd)) cnt = (unsigned int)(hi_odd - lo_odd);
    *host_count = cnt;

    if (cnt > 0) {
        err = cudaMemcpyAsync(host_offsets, ctx->d_offsets + (size_t)slot_base,
                              (size_t)cnt * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost, ctx->stream);
        if (err != cudaSuccess) return 0;
        if (host_cands_aos) {
            err = cudaMemcpyAsync(host_cands_aos,
                                  ctx->d_cands_aos[cand_buf & 1] +
                                      (size_t)slot_base * (size_t)active_limbs,
                                  (size_t)cnt * (size_t)active_limbs *
                                      sizeof(uint64_t),
                                  cudaMemcpyDeviceToHost, ctx->stream);
            if (err != cudaSuccess) return 0;
        }
        err = cudaStreamSynchronize(ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpu_sieve: extract_pack sync: %s\n",
                    cudaGetErrorString(err));
            return 0;
        }
    }

    /* Right-size after the FIRST measured window: 4x the observed survivors
       plus slack.  The survivor density is a property of the cover and the
       sieve depth, so it is near-constant across windows (measured 1,000.2 +
       -0.3 per window over 380k windows), which makes 4x a wide margin.  The
       change is *pending*: it is applied at the next slot_base == 0 call, so
       the flight being accumulated now is untouched. */
    if (slot_base == 0 && !ctx->cand_cap_calibrated) {
        ctx->cand_cap_calibrated = 1;
        if ((uint64_t)cnt > ctx->cand_cap_measured)
            ctx->cand_cap_measured = cnt;
        uint64_t want = (uint64_t)cnt * 4U + 1024U;
        if (want < (uint64_t)GPU_EXTRACT_CAND_CAP_MIN)
            want = (uint64_t)GPU_EXTRACT_CAND_CAP_MIN;
        if (want < ctx->cand_cap_per_window) {
            ctx->cand_cap_pending = want;
            if (!ctx->cand_cap_reported) {
                ctx->cand_cap_reported = 1;
                fprintf(stderr,
                        "gpu_sieve: candidate buffer right-sized to %llu "
                        "slots/window (%u survivors measured in the first "
                        "window; was %llu = one slot per odd position)\n",
                        (unsigned long long)want, cnt,
                        (unsigned long long)ctx->cand_cap_per_window);
            }
        }
    }
    return 1;
}

/* Stage 1/parity-test entry: extract + pack with a host copy of candidates. */
int gpu_sieve_extract_pack(gpu_sieve_ctx *ctx,
                           uint64_t odd_interval_size,
                           uint64_t first_odd_offset,
                           const uint64_t *base_limbs,
                           int active_limbs,
                           uint64_t *host_cands_aos,
                           uint64_t *host_offsets,
                           unsigned int *host_count,
                           uint32_t base_mod60,
                           uint64_t class_mask60,
                           uint64_t region_start)
{
    return gpu_sieve_extract_pack_impl(ctx, odd_interval_size, first_odd_offset,
                                       0, odd_interval_size,
                                       ctx->d_bitmap[0], 0, base_limbs,
                                       active_limbs, host_cands_aos,
                                       host_offsets, host_count,
                                       base_mod60, class_mask60,
                                       region_start, 0);
}

/* Fused Stage 3: extract + pack keeping candidates on-device (no D→H cand
   copy).  Returns the packed AoS device pointer via *d_cands_out, and still
   copies the survivor offsets + count to the host for gap detection.  The
   output is in ascending offset order. */
int gpu_sieve_extract_pack_device(gpu_sieve_ctx *ctx,
                                  uint64_t odd_interval_size,
                                  uint64_t first_odd_offset,
                                  const uint64_t *base_limbs,
                                  int active_limbs,
                                  uint64_t **d_cands_out,
                                  uint64_t *host_offsets,
                                  unsigned int *host_count,
                                  uint32_t base_mod60,
                                  uint64_t class_mask60,
                                  uint64_t region_start)
{
    if (!d_cands_out) return 0;
    *d_cands_out = NULL;
    if (!gpu_sieve_extract_pack_impl(ctx, odd_interval_size, first_odd_offset,
                                     0, odd_interval_size,
                                     ctx->d_bitmap[0], 0, base_limbs,
                                     active_limbs, NULL, host_offsets,
                                     host_count, base_mod60, class_mask60,
                                     region_start, 0)) {
        return 0;
    }
    *d_cands_out = ctx->d_cands_aos[0];
    return 1;
}

/* Range-limited device extract (fused smart-scan tail-skip).  cand_buf
   selects the device candidate buffer (0/1) for the async ping-pong. */
int gpu_sieve_extract_pack_device_range(gpu_sieve_ctx *ctx,
                                        uint64_t odd_interval_size,
                                        uint64_t first_odd_offset,
                                        uint64_t lo_odd,
                                        uint64_t hi_odd,
                                        int cand_buf,
                                        const uint64_t *base_limbs,
                                        int active_limbs,
                                        uint64_t **d_cands_out,
                                        uint64_t *host_offsets,
                                        unsigned int *host_count,
                                        uint32_t base_mod60,
                                        uint64_t class_mask60,
                                        uint64_t region_start,
                                        uint32_t slot_base)
{
    if (!d_cands_out) return 0;
    *d_cands_out = NULL;
    if (!gpu_sieve_extract_pack_impl(ctx, odd_interval_size, first_odd_offset,
                                     lo_odd, hi_odd,
                                     ctx->d_bitmap[0], cand_buf,
                                     base_limbs,
                                     active_limbs, NULL, host_offsets,
                                     host_count, base_mod60, class_mask60,
                                     region_start, slot_base)) {
        return 0;
    }
    *d_cands_out = ctx->d_cands_aos[cand_buf & 1];
    return 1;
}

/* Accumulation variant (MR batch across K windows): reads the bitmap of
   bitmap_buf but packs candidates into candidate buffer cand_buf at
   slot_base (contiguous packing across windows). */
int gpu_sieve_extract_pack_device_range_ex(gpu_sieve_ctx *ctx,
                                           uint64_t odd_interval_size,
                                           uint64_t first_odd_offset,
                                           uint64_t lo_odd,
                                           uint64_t hi_odd,
                                           int bitmap_buf,
                                           int cand_buf,
                                           const uint64_t *base_limbs,
                                           int active_limbs,
                                           uint64_t **d_cands_out,
                                           uint64_t *host_offsets,
                                           unsigned int *host_count,
                                           uint32_t base_mod60,
                                           uint64_t class_mask60,
                                           uint64_t region_start,
                                           uint32_t slot_base)
{
    if (!d_cands_out) return 0;
    *d_cands_out = NULL;
    if (!gpu_sieve_extract_pack_impl(ctx, odd_interval_size, first_odd_offset,
                                     lo_odd, hi_odd,
                                     ctx->d_bitmap[bitmap_buf & 1], cand_buf,
                                     base_limbs,
                                     active_limbs, NULL, host_offsets,
                                     host_count, base_mod60, class_mask60,
                                     region_start, slot_base)) {
        return 0;
    }
    *d_cands_out = ctx->d_cands_aos[cand_buf & 1] +
                   (size_t)slot_base * (size_t)active_limbs;
    return 1;
}

/* Row-bitmap accumulation variant: like gpu_sieve_extract_pack_device_range_ex
   but reads an explicit device bitmap (a row of the row arena) instead of a
   ping-pong bitmap.  Used by the CRT row-walk fused path. */
int gpu_sieve_extract_pack_device_range_bitmap(
    gpu_sieve_ctx *ctx,
    const uint64_t *bitmap,
    uint64_t odd_interval_size,
    uint64_t first_odd_offset,
    uint64_t lo_odd,
    uint64_t hi_odd,
    int cand_buf,
    const uint64_t *base_limbs,
    int active_limbs,
    uint64_t **d_cands_out,
    uint64_t *host_offsets,
    unsigned int *host_count,
    uint32_t base_mod60,
    uint64_t class_mask60,
    uint64_t region_start,
    uint32_t slot_base)
{
    if (!d_cands_out || !bitmap) return 0;
    *d_cands_out = NULL;
    if (!gpu_sieve_extract_pack_impl(ctx, odd_interval_size, first_odd_offset,
                                     lo_odd, hi_odd, bitmap, cand_buf,
                                     base_limbs,
                                     active_limbs, NULL, host_offsets,
                                     host_count, base_mod60, class_mask60,
                                     region_start, slot_base)) {
        return 0;
    }
    *d_cands_out = ctx->d_cands_aos[cand_buf & 1] +
                   (size_t)slot_base * (size_t)active_limbs;
    return 1;
}

/* Device pointer of row r in the row bitmap arena (NULL when unavailable). */
uint64_t *gpu_sieve_row_bitmap(gpu_sieve_ctx *ctx, uint32_t row) {
    if (!ctx || !ctx->d_row_bitmaps || row >= ctx->row_bitmap_cap) {
        return NULL;
    }
    return ctx->d_row_bitmaps + (size_t)row * ctx->max_bitmap_words;
}

/* Size the extract candidate buffers for K-window MR batch accumulation.
   Must be called before the first extract when K > 1. */
void gpu_sieve_set_extract_accum(gpu_sieve_ctx *ctx, uint32_t k) {
    if (!ctx) return;
    ctx->extract_accum = k ? k : 1;
}

void gpu_sieve_set_cand_cap_limit(gpu_sieve_ctx *ctx, uint64_t slots) {
    if (!ctx) return;
    ctx->cand_cap_limit = slots;
}

/* The capacity the candidate buffers will START with for a given window
   geometry: the same rule gpu_sieve_extract_pack_impl applies at its first
   extract (max(floor, odd_interval/8), never more than the odd interval
   itself).  Used by callers that need to report or budget VRAM up front. */
uint64_t gpu_sieve_cand_cap_estimate(uint64_t max_odd_interval) {
    uint64_t init = max_odd_interval / 8U;
    if (init < (uint64_t)GPU_EXTRACT_CAND_CAP_MIN)
        init = (uint64_t)GPU_EXTRACT_CAND_CAP_MIN;
    if (init > max_odd_interval) init = max_odd_interval;
    return init;
}

int gpu_sieve_mem_info(gpu_sieve_ctx *ctx, size_t *free_bytes,
                       size_t *total_bytes) {
    if (!ctx) return -1;
    if (gpu_sieve_ensure_device(ctx->device_id) != cudaSuccess) return -1;
    size_t vfree = 0, vtotal = 0;
    if (cudaMemGetInfo(&vfree, &vtotal) != cudaSuccess) return -1;
    if (free_bytes) *free_bytes = vfree;
    if (total_bytes) *total_bytes = vtotal;
    return 0;
}

/* Device AoS candidate buffer of ping-pong buf (0/1). */
uint64_t *gpu_sieve_candidate_buffer(gpu_sieve_ctx *ctx, int buf) {
    return ctx ? ctx->d_cands_aos[buf & 1] : NULL;
}

/* Device ping-pong bitmap written by gpu_sieve_mark_from_base /
   gpu_sieve_mark_batch_from_bases (buf 0/1). */
uint64_t *gpu_sieve_pingpong_bitmap(gpu_sieve_ctx *ctx, int buf) {
    return ctx ? ctx->d_bitmap[buf & 1] : NULL;
}

/* Device-side survivor offsets (scratch, see gpu_sieve.h). */
const uint64_t *gpu_sieve_device_offsets(gpu_sieve_ctx *ctx) {
    return ctx ? ctx->d_offsets : NULL;
}

void gpu_sieve_destroy(gpu_sieve_ctx *ctx) {
    if (!ctx) return;

    (void)gpu_sieve_ensure_device(ctx->device_id);
    if (ctx->d_primes) cudaFree(ctx->d_primes);
    if (ctx->d_base_mod_p) cudaFree(ctx->d_base_mod_p);
    if (ctx->d_inv_p) cudaFree(ctx->d_inv_p);
    if (ctx->d_base_offsets) cudaFree(ctx->d_base_offsets);
    if (ctx->d_mark_pos0) cudaFree(ctx->d_mark_pos0);
    if (ctx->d_mark_pair) cudaFree(ctx->d_mark_pair);
    if (ctx->d_mark_firstdec) cudaFree(ctx->d_mark_firstdec);
    if (ctx->d_mark_pos1) cudaFree(ctx->d_mark_pos1);
    if (ctx->d_bitmap[0]) cudaFree(ctx->d_bitmap[0]);
    if (ctx->d_bitmap[1]) cudaFree(ctx->d_bitmap[1]);
    if (ctx->d_base_limbs) cudaFree(ctx->d_base_limbs);
    if (ctx->d_cands_aos[0]) cudaFree(ctx->d_cands_aos[0]);
    if (ctx->d_cands_aos[1]) cudaFree(ctx->d_cands_aos[1]);
    if (ctx->d_offsets) cudaFree(ctx->d_offsets);
    if (ctx->d_count) cudaFree(ctx->d_count);
    if (ctx->d_row_bitmaps) cudaFree(ctx->d_row_bitmaps);
    if (ctx->d_step_mod_p) cudaFree(ctx->d_step_mod_p);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);

    free(ctx);
}
