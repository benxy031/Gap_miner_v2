/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef GPU_SIEVE_H
#define GPU_SIEVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque CUDA bitmap-sieve context. */
typedef struct gpu_sieve_ctx gpu_sieve_ctx;

/* Initialize per-worker GPU bitmap-sieve resources.
   max_primes: upper bound for prime_count passed to gpu_sieve_mark_high_primes.
   max_odd_interval: upper bound for odd interval size (number of odd slots).
   Returns NULL on any CUDA/host allocation error. */
gpu_sieve_ctx *gpu_sieve_init(int device_id,
                              size_t max_primes,
                              uint64_t max_odd_interval);

/* Mark composites for the provided high-prime slice into host_bitmap.
   odd_interval_size: number of odd slots represented by the bitmap.
   first_odd_offset: 0 or 1 offset used by the odd-only sieve representation.
   base_offset: cached-base window offset used to derive current residues.
   primes/base_mod_p: slices for the high-prime range [split_index, end).
   host_bitmap_words must be >= ceil(odd_interval_size / 64).
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
int gpu_sieve_mark_high_primes(gpu_sieve_ctx *ctx,
                               uint64_t odd_interval_size,
                               uint64_t first_odd_offset,
                               uint64_t base_offset,
                               const uint64_t *primes,
                               const uint64_t *base_mod_p,
                               size_t prime_count,
                               uint64_t *host_bitmap,
                               size_t host_bitmap_words);

/* Mark high-prime composites straight from the raw base, computing
   base mod p for every prime ON THE DEVICE (no host residue pass).  This is
   the CRT path: the base changes every nonce, so residues cannot be amortised
   on the host, but the device computes them in parallel across primes.
   base_limbs: little-endian 64-bit limbs of the window base (top limbs may be
   zero), base_limb_count limbs.  primes: high-prime slice [split_index, end);
   the prime table is uploaded once and cached by prime_count.
   buf (0/1) selects one of two ping-pong device bitmaps (async fused
   pipeline: window i marks buf i&1 while window i-1 is still in flight).
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
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
                             size_t host_bitmap_words);

/* Pair-batched fused mark: residues + marking for TWO windows in ONE kernel
   launch and ONE stream sync.  base_limbs_pairs: 2 × base_limb_count
   little-endian limbs; window 0 marks d_bitmap[0], window 1 d_bitmap[1].
   The per-window first odd offset is derived from each window's own base
   parity.  Requires the same odd_interval_size for both windows.
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
int gpu_sieve_mark_batch_from_bases(gpu_sieve_ctx *ctx,
                                    uint64_t odd_interval_size,
                                    const uint64_t *base_limbs_pairs,
                                    int base_limb_count,
                                    const uint64_t *primes,
                                    const uint64_t *inv_p,
                                    size_t prime_count);

/* HALF_CLASS extraction filter: offsets with value (base + offset) mod 60
   whose bit is set in class_mask60 are extracted.  Offsets BELOW
   region_start are never filtered (the CRT back-lookahead must be scanned
   in all classes).  class_mask60 = UINT64_MAX extracts everything. */

/* Fused-pipeline Stage 1: after gpu_sieve_mark_from_base() has marked the
   device bitmap, extract survivors and pack each candidate
   (base + first_odd_offset + 2*odd_pos) into AoS little-endian limbs.
   Copies survivor count, full adder offsets, and (optionally) packed limbs
   back to the host.  Stage 2 will keep them on-device and feed the MR kernel.
   base_mod60/class_mask60/region_start: optional HALF_CLASS residue filter
   (class_mask60 = UINT64_MAX disables it).
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
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
                           uint64_t region_start);

/* Fused Stage 3: extract survivors and pack candidates into device AoS,
   keeping the candidates ON THE DEVICE (no D→H candidate copy).  Returns the
   device candidate pointer via *d_cands_out (valid until the next
   gpu_sieve_extract_pack* / gpu_sieve_mark_from_base call on this ctx).
   host_offsets + host_count are still copied back for host gap detection.
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
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
                                  uint64_t region_start);

/* Range-limited fused extract: like gpu_sieve_extract_pack_device, but only
   extracts odd positions in [lo_odd, hi_odd) (inclusive/exclusive).  Used by
   the fused smart-scan tail-skip to extract the covered "head" first, then the
   uncovered "tail" only when no closing prime is found in the head.  Output
   offsets are ASCENDING (ordered compaction, no host qsort needed).
   cand_buf (0/1) selects one of two device candidate buffers so consecutive
   windows can be in flight without overwriting each other's candidates. */
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
                                        uint32_t slot_base);

/* Accumulation variant (MR batch across K windows): reads bitmap bitmap_buf,
   packs into candidate buffer cand_buf at slot_base (contiguous). */
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
                                           uint32_t slot_base);

/* Size the extract candidate buffers for K-window MR batch accumulation. */
void gpu_sieve_set_extract_accum(gpu_sieve_ctx *ctx, uint32_t k);

/* Device AoS candidate buffer of ping-pong buf (0/1). */
uint64_t *gpu_sieve_candidate_buffer(gpu_sieve_ctx *ctx, int buf);

/* Device-side survivor offsets buffer (all extracted windows packed
   contiguously at their slot_base).  GAP_HUNT jump mode reads it directly
   on the GPU.  Scratch semantics: valid until the next extract into the
   same buffer. */
const uint64_t *gpu_sieve_device_offsets(gpu_sieve_ctx *ctx);

/* Batched variant for multiple windows that share odd interval geometry.
   base_offsets: array of per-window base offsets (window starts), length=batch_count.
   host_bitmaps: contiguous output buffer packed as [batch_count][bitmap_words],
                 where bitmap_words = ceil(odd_interval_size / 64).
   host_bitmaps_words is total host buffer capacity in 64-bit words and must be
                 >= batch_count * bitmap_words.
   Returns 1 on success, 0 on any CUDA or validation error (fail-closed). */
int gpu_sieve_mark_high_primes_batch(gpu_sieve_ctx *ctx,
                                     uint64_t odd_interval_size,
                                     uint64_t first_odd_offset,
                                     const uint64_t *base_offsets,
                                     size_t batch_count,
                                     const uint64_t *primes,
                                     const uint64_t *base_mod_p,
                                     size_t prime_count,
                                     uint64_t *host_bitmaps,
                                     size_t host_bitmaps_words);

/* Wall time of the last completed batch, including CUDA transfers and sync. */
uint64_t gpu_sieve_last_elapsed_us(const gpu_sieve_ctx *ctx);

/* Device name for logging, empty string when unavailable. */
const char *gpu_sieve_device_name(const gpu_sieve_ctx *ctx);

/* Free GPU/host resources. Safe on NULL. */
void gpu_sieve_destroy(gpu_sieve_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GPU_SIEVE_H */
