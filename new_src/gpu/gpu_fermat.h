/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

/* gpu_fermat.h — CUDA batch Fermat primality testing for Gapcoin
 *
 * Tests batches of large prime candidates using Fermat's little theorem:
 *   n is probably prime if 2^(n-1) ≡ 1 (mod n)
 *
 * Uses Montgomery multiplication on GPU for efficient modular arithmetic.
 * Each CUDA thread independently tests one candidate.
 *
 * GPU_NLIMBS controls the maximum candidate size:
 *   6 limbs = 384 bits  → shift ≤ 128
 *  12 limbs = 768 bits  → shift ≤ 512
 *  16 limbs = 1024 bits → shift ≤ 768  (default)
 *  20 limbs = 1280 bits → shift ≤ 1024
 *
 * Override at compile time: -DGPU_NLIMBS=16 or via Makefile GPU_BITS=1024
 */
#ifndef GPU_FERMAT_H
#define GPU_FERMAT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Number of 64-bit limbs per candidate.
   Default 16 limbs = 1024 bits, sufficient for hash(256) + shift(≤768).
   Override with -DGPU_NLIMBS=N at compile time for other shift ranges. */
#ifndef GPU_NLIMBS
#define GPU_NLIMBS 16
#endif

/* Opaque GPU context */
typedef struct gpu_fermat_ctx gpu_fermat_ctx;

/* Return the number of CUDA devices visible to the process, or 0 on error. */
int gpu_fermat_device_count(void);

/* Initialize GPU Fermat tester.
   device_id:  CUDA device index (usually 0).
   max_batch:  maximum candidates per batch call.
   Returns NULL on failure (no GPU, driver error, out of memory). */
gpu_fermat_ctx *gpu_fermat_init(int device_id, size_t max_batch);

/* Batch Fermat primality test (synchronous — blocks until complete).
   candidates: array of count candidates, each GPU_NLIMBS uint64_t limbs
               in little-endian limb order (limb[0] = least significant).
   results:    output array of count bytes, 1 = probably prime, 0 = composite.
   count:      number of candidates to test.
   Returns number of probable primes found, or -1 on error. */
int gpu_fermat_test_batch(gpu_fermat_ctx *ctx,
                          const uint64_t *candidates,
                          uint8_t *results,
                          size_t count);

/* Fused-pipeline Stage 2: test candidates already resident in device memory.
   d_candidates: device pointer, AoS layout, gpu_fermat_get_limbs(ctx) stride
                 per candidate (exactly the layout gpu_sieve_extract_pack()
                 writes).  No H→D transfer is performed; only the D→H results
                 copy remains.
   results: host output array of count bytes (1 = probable prime).
   count:   number of candidates (clamped to the init max_batch).
   Returns number of probable primes found, or -1 on error.  Synchronous. */
int gpu_fermat_test_device(gpu_fermat_ctx *ctx,
                           const uint64_t *d_candidates,
                           uint8_t *results,
                           size_t count);

/* Async device-pointer variant: enqueue the MR kernel + D→H results copy on
   the given slot's stream without any H→D candidate copy.  The caller owns
   d_candidates until gpu_fermat_collect() drains this slot.  Returns 0 on
   success, -1 on error (blocking until the slot is free, like submit). */
int gpu_fermat_submit_device(gpu_fermat_ctx *ctx, int slot,
                             const uint64_t *d_candidates, size_t count);

/* Asynchronous double-buffered pipeline API.
   Two slots (0 and 1) allow overlapping GPU compute with CPU work.
   Typical usage:
     gpu_fermat_submit(ctx, 0, cands_A, countA);   // returns immediately
     // ... CPU prepares next batch ...
     gpu_fermat_collect(ctx, 0, results_A, countA); // blocks until slot 0 done
     gpu_fermat_submit(ctx, 1, cands_B, countB);    // returns immediately
     // ... CPU processes results_A ...
     gpu_fermat_collect(ctx, 1, results_B, countB); // blocks until slot 1 done
*/

/* Submit candidates for async Fermat testing on the given slot (0 or 1).
   Copies candidates into pinned staging, launches async H→D + kernel + D→H.
   Returns 0 on success, -1 on error.  The candidates buffer may be reused
   immediately after this call returns. */
int gpu_fermat_submit(gpu_fermat_ctx *ctx, int slot,
                      const uint64_t *candidates, size_t count);

/* Non-blocking variant of gpu_fermat_submit.
   Returns -1 immediately (without blocking) if the slot is still busy
   with a previous submission.  Use this in per-thread contexts where the
   caller must never stall waiting for another thread's GPU work. */
int gpu_fermat_submit_try(gpu_fermat_ctx *ctx, int slot,
                          const uint64_t *candidates, size_t count);

/* Wait for async slot to complete, copy results out.
   Returns number of probable primes found, or -1 on error. */
int gpu_fermat_collect(gpu_fermat_ctx *ctx, int slot,
                       uint8_t *results, size_t count);

/* Return the CUDA device name (for logging).  Returns "" on error. */
const char *gpu_fermat_device_name(gpu_fermat_ctx *ctx);

/* Cumulative GPU-accounted MR kernel time in microseconds, measured with
   CUDA events around each batch (pure kernel time, excluding host launch and
   sync gaps).  Monotonic over the context lifetime; 0 on error.  Used to
   compute the acc/wall (GPU utilization) metric in the rolling STATS. */
uint64_t gpu_fermat_accounted_us(gpu_fermat_ctx *ctx);

/* Jump-scan walk API (GAP_HUNT jump mode): one thread block per window runs
   a serial Miller-Rabin chain (Kehrig-style), testing ~1.5 survivors per
   confirmed prime instead of all survivors, and reports gaps >= threshold
   in offset units.  Synchronous; windows are independent. */
int gpu_fermat_jump_alloc(gpu_fermat_ctx *ctx, uint32_t n_windows,
                          uint32_t max_gaps_per_window);
int gpu_fermat_jump_scan(gpu_fermat_ctx *ctx,
                         const uint64_t *d_cands,
                         const uint64_t *d_offsets,
                         const uint32_t *h_counts,
                         const uint32_t *h_cums,
                         const uint64_t *h_thresholds,
                         int active_limbs,
                         uint32_t n_windows,
                         uint64_t *h_gaps,
                         uint32_t *h_counts_out);

/* Set the arithmetic limb count for this context.
   active_limbs = ceil((256 + shift) / 64) for Gapcoin.
   Reduces Montgomery multiplication from O(GPU_NLIMBS²) to O(active²).
   At shift 43: active=5, speedup ≈ (16/5)² ≈ 10×.
   Must be called before submitting work.  Clamped to [1, GPU_NLIMBS]. */
void gpu_fermat_set_limbs(gpu_fermat_ctx *ctx, int limbs);

/* Get the current active limb count (storage stride per candidate).
   Callers should pack candidates at this width for submit(). */
int gpu_fermat_get_limbs(gpu_fermat_ctx *ctx);

/* Active limb count for a candidate bit width: ceil(bits/64), rounded up to
   an even limb count when that lands on CGBN's fast path, clamped to
   [1, GPU_NLIMBS].  Mirrors gpu_adapter_set_candidate_bits(). */
int gpu_fermat_limbs_for_bits(uint32_t bits);

/* Nonzero if the compiled build can run the CGBN kernel at this limb count
   (even widths 2,4,6,8,12,16,20 up to GPU_NLIMBS).  Zero means the scalar
   kernel is used. */
int gpu_fermat_cgbn_supports_limbs(int limbs);

/* Human-readable kernel label for a limb count, e.g. "CGBN (384-bit, TPI=4)"
   or "scalar".  Returns "" for an out-of-range limb count. */
const char *gpu_fermat_kernel_label(int limbs);

/* Free GPU resources. */
void gpu_fermat_destroy(gpu_fermat_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_FERMAT_H */
