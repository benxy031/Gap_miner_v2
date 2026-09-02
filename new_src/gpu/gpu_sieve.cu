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
    unsigned int *d_count;      /* survivor counter (written by the scan) */
    size_t max_candidates;
    uint32_t extract_accum;     /* K: candidate buffers sized K× per window
                                   (MR batch accumulation across windows) */
    int active_limbs_capacity;
};

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

    for (; pos < odd_interval_size; pos += p) {
        atomicOr((unsigned long long *)&window_bitmap[pos >> 6],
                 (unsigned long long)(1ULL << (pos & 63U)));
    }
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

    err = cudaMalloc(&ctx->d_inv_p, max_primes * sizeof(*ctx->d_inv_p));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: cudaMalloc(inv_p): %s\n",
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

    for (; pos < odd_interval_size; pos += p) {
        atomicOr((unsigned long long *)&window_bitmap[pos >> 6],
                 (unsigned long long)(1ULL << (pos & 63U)));
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
    uint64_t total_threads = 2ULL * (uint64_t)prime_count;
    uint64_t blocks = (total_threads + tpb - 1U) / tpb;
    if (blocks > (uint64_t)UINT_MAX) return 0;

    gpu_sieve_residues_mark_pair_kernel<<<(unsigned int)blocks, tpb, 0,
                                          ctx->stream>>>(
        ctx->d_base_limbs, base_limb_count,
        ctx->d_bitmap[0], ctx->d_bitmap[1], (uint64_t)required_words,
        odd_interval_size, ctx->d_primes, ctx->d_inv_p,
        (uint64_t)prime_count);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: pair mark launch failed: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    err = cudaStreamSynchronize(ctx->stream);
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

    gpu_sieve_mark_kernel_batch<<<(unsigned int)blocks, tpb, 0, ctx->stream>>>(
        ctx->d_bitmap[buf & 1], (uint64_t)required_words, odd_interval_size,
        first_odd_offset, ctx->d_base_offsets, 1, ctx->d_primes,
        ctx->d_base_mod_p, prime_count_u64);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: kernel launch (from base) failed: %s\n",
                cudaGetErrorString(err));
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
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: stream sync failed: %s\n",
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
            if (slot >= max_batch) continue; /* sized right -> unreachable */

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
                                       int bitmap_buf,
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
    if (!ctx || !base_limbs || !host_offsets || !host_count) return 0;
    if (odd_interval_size == 0 || active_limbs < 1) return 0;
    if (first_odd_offset > 1U) return 0;
    if (active_limbs > ctx->base_limbs_capacity) return 0;
    if (lo_odd >= hi_odd || hi_odd > odd_interval_size) return 0;

    size_t words = (size_t)((odd_interval_size + 63U) >> 6);
    if (words == 0 || words > ctx->max_bitmap_words) return 0;
    if (words > 1023) return 0;   /* single-block scan capacity */

    cudaError_t err = gpu_sieve_ensure_device(ctx->device_id);
    if (err != cudaSuccess) return 0;

    /* (Re)allocate the extract buffers if capacity or limb width changed.
       extract_accum > 1 sizes the candidate buffers for MR batch
       accumulation across K windows (slot_base up to K×odd_interval_size).
       Capacity is the ctx's max window size so the buffers are allocated
       ONCE and never move while async MR kernels still read them (a
       mid-run realloc would free buffers referenced by in-flight CGBN
       kernels — the K>1 accumulation widened that race window). */
    size_t accum = ctx->extract_accum ? (size_t)ctx->extract_accum : 1;
    size_t cap = ((ctx->max_odd_interval > odd_interval_size)
                      ? ctx->max_odd_interval
                      : odd_interval_size) *
                 accum;
    if (ctx->max_candidates < cap ||
        ctx->active_limbs_capacity != active_limbs ||
        !ctx->d_cands_aos[0] || !ctx->d_cands_aos[1] ||
        !ctx->d_offsets || !ctx->d_count) {
        /* Defensive: never free buffers that in-flight kernels may read.
           With the max-size preallocation this should be unreachable during
           mining, but guard it anyway. */
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
        if (err != cudaSuccess) { fprintf(stderr, "gpu_sieve: cands[0] alloc: %s\n",
                                          cudaGetErrorString(err)); return 0; }
        err = cudaMalloc(&ctx->d_cands_aos[1],
                         cap * (size_t)active_limbs * sizeof(uint64_t));
        if (err != cudaSuccess) { fprintf(stderr, "gpu_sieve: cands[1] alloc: %s\n",
                                          cudaGetErrorString(err)); return 0; }
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

    uint64_t w_lo = lo_odd >> 6;
    uint64_t w_hi = (hi_odd + 63U) >> 6;
    if (w_hi > (uint64_t)words) w_hi = (uint64_t)words;
    if (w_lo >= w_hi) return 0;

    /* Single-block ordered compaction: one thread per bitmap word. */
    uint32_t half_filter = (class_mask60 != UINT64_MAX) ? 1U : 0U;
    gpu_sieve_extract_pack_kernel<<<1, 1024, 0, ctx->stream>>>(
        ctx->d_bitmap[bitmap_buf & 1], (uint64_t)words, odd_interval_size,
        first_odd_offset, ctx->d_base_limbs, active_limbs,
        ctx->d_cands_aos[cand_buf & 1], ctx->d_offsets, ctx->d_count,
        (uint32_t)(hi_odd - lo_odd + slot_base), lo_odd, hi_odd, half_filter,
        base_mod60, class_mask60, region_start, slot_base);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: extract_pack launch: %s\n",
                cudaGetErrorString(err));
        return 0;
    }

    /* Read the survivor count first (async + stream sync), then copy only the
       valid entries.  Using a full-buffer copy would transfer ~1 MB of dead
       slots per window; using synchronous cudaMemcpy would insert DEVICE-WIDE
       syncs that serialize concurrent workers' streams. */
    unsigned int cnt = 0;
    err = cudaMemcpyAsync(&cnt, ctx->d_count, sizeof(cnt),
                          cudaMemcpyDeviceToHost, ctx->stream);
    if (err != cudaSuccess) return 0;
    err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_sieve: extract_pack count sync: %s\n",
                cudaGetErrorString(err));
        return 0;
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
                                       0, odd_interval_size, 0, 0, base_limbs,
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
                                     0, odd_interval_size, 0, 0, base_limbs,
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
                                     lo_odd, hi_odd, cand_buf, cand_buf,
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
                                     lo_odd, hi_odd, bitmap_buf, cand_buf,
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

/* Size the extract candidate buffers for K-window MR batch accumulation.
   Must be called before the first extract when K > 1. */
void gpu_sieve_set_extract_accum(gpu_sieve_ctx *ctx, uint32_t k) {
    if (!ctx) return;
    ctx->extract_accum = k ? k : 1;
}

/* Device AoS candidate buffer of ping-pong buf (0/1). */
uint64_t *gpu_sieve_candidate_buffer(gpu_sieve_ctx *ctx, int buf) {
    return ctx ? ctx->d_cands_aos[buf & 1] : NULL;
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
    if (ctx->d_bitmap[0]) cudaFree(ctx->d_bitmap[0]);
    if (ctx->d_bitmap[1]) cudaFree(ctx->d_bitmap[1]);
    if (ctx->d_base_limbs) cudaFree(ctx->d_base_limbs);
    if (ctx->d_cands_aos[0]) cudaFree(ctx->d_cands_aos[0]);
    if (ctx->d_cands_aos[1]) cudaFree(ctx->d_cands_aos[1]);
    if (ctx->d_offsets) cudaFree(ctx->d_offsets);
    if (ctx->d_count) cudaFree(ctx->d_count);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);

    free(ctx);
}
