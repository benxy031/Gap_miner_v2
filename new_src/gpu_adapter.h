/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Adapter (CUDA/OpenCL abstraction)
 *
 * Unified interface for GPU-based primality testing.
 * Implementations: GPU Fermat (base case), GPU Sieve (pre-filtering)
 *
 * Architecture:
 * - gpu_adapter_init(): Initialize GPU context and kernel compilation
 * - gpu_adapter_test_batch(): Submit 4096 candidates for parallel Fermat test
 * - gpu_adapter_get_results(): Retrieve is_prime bitmask
 * - gpu_adapter_free(): Cleanup GPU memory
 */

#ifndef GPU_ADAPTER_H
#define GPU_ADAPTER_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Max candidates per gpu_adapter_test_batch() call; callers submitting more
   must chunk at this boundary (the CUDA context's staging buffers are sized
   to it, and any candidates beyond it would otherwise go untested).
   Raised from 8192 to 40000: the CGBN kernel is latency-bound, so larger
   batches amortize the per-launch/tail overhead (measured 1.28M/s @10k →
   1.52M/s @40k candidates, still climbing). */
#define GPU_ADAPTER_MAX_BATCH 80000U

/* GPU candidate batch (input to GPU) */
struct gpu_batch {
    uint32_t count;                    /* Number of candidates (≤4096) */
    mpz_t *candidates;                 /* Array of mpz_t numbers */
    uint8_t *is_prime;                 /* Output: primality verdicts */
};

/* GPU adapter context */
struct gpu_adapter {
    int device_id;                     /* CUDA device or 0 for OpenCL */
    void *gpu_context;                 /* GPU-specific state (opaque) */
};

/* Initialize GPU adapter for given device */
struct gpu_adapter *gpu_adapter_init(int device_id);

/* Return the number of usable CUDA devices, or 1 for CPU-only builds. */
int gpu_adapter_device_count(void);

/* Narrow the GPU arithmetic width to what candidates actually need (bits),
   instead of running every candidate at the full compiled GPU_NLIMBS width.
   Montgomery multiplication cost is O(active_limbs^2), so leaving this
   uncalled (i.e. always running at the max compiled width) can make the GPU
   path several times slower than necessary for smaller shifts. No-op
   without a WITH_CUDA build. Safe to call repeatedly (e.g. on shift change). */
void gpu_adapter_set_candidate_bits(struct gpu_adapter *adapter, uint32_t bits);

/* Active limb count currently configured for this adapter (the storage stride
   the GPU kernels read per candidate).  Returns GPU_NLIMBS without WITH_CUDA. */
int gpu_adapter_get_limbs(struct gpu_adapter *adapter);

/* Fused-pipeline accessor: expose the underlying CUDA Fermat context so the
   fused path can run gpu_fermat_submit_device() on a device-resident
   candidate buffer (no H2D).  Returns NULL when built without WITH_CUDA or
   when the adapter has no GPU context.  The returned pointer must not be
   freed; gpu_adapter_free() owns it. */
#ifdef WITH_CUDA
struct gpu_fermat_ctx *gpu_adapter_get_fermat_ctx(struct gpu_adapter *adapter);
#endif

/* Submit already-packed candidates (count × active_limbs little-endian
   64-bit limbs) to one async GPU slot.  Returns 0 on success, -1 on error
   or if the slot is busy.  The caller owns the buffer and may reuse it after
   this returns. */
int gpu_adapter_async_submit_packed(struct gpu_adapter *adapter, int slot,
                                    const uint64_t *packed, size_t count);

/* Batch Fermat test: test candidates using GPU */
int gpu_adapter_test_batch(struct gpu_adapter *adapter, 
                           struct gpu_batch *batch);

/* Submit/collect one of two asynchronous GPU Fermat slots. The submit call
    copies candidate limbs before returning; the caller may reuse its mpz array. */
int gpu_adapter_async_submit(struct gpu_adapter *adapter, int slot,
                                      struct gpu_batch *batch);
int gpu_adapter_async_collect(struct gpu_adapter *adapter, int slot,
                                        struct gpu_batch *batch);

/* Get detailed results (GPU-specific diagnostics) */
void gpu_adapter_get_results(struct gpu_adapter *adapter, 
                             struct gpu_batch *batch,
                             uint64_t *tested_count,
                             uint64_t *kernel_time_us);

/* Free GPU resources */
void gpu_adapter_free(struct gpu_adapter *adapter);

#endif /* GPU_ADAPTER_H */
