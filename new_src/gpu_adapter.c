/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Adapter Implementation
 *
 * Without WITH_CUDA: CPU-based fallback (per-candidate GMP Fermat test).
 * With WITH_CUDA: batches candidates to the CUDA Fermat kernel in
 * new_src/gpu/gpu_fermat.cu (scalar kernel, or CGBN when built with
 * WITH_CGBN_FERMAT=1). Any candidate that doesn't fit in GPU_NLIMBS limbs,
 * or any GPU batch failure, falls back to the CPU Fermat test so a missing
 * or misbehaving GPU never silently drops candidates.
 */

#include "gpu_adapter.h"
#include "primality_fermat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WITH_CUDA
#include "gpu/gpu_fermat.h"
#endif

struct gpu_adapter *gpu_adapter_init(int device_id) {
    struct gpu_adapter *adapter = 
        (struct gpu_adapter *)malloc(sizeof(struct gpu_adapter));
    
    if (!adapter) return NULL;
    
    adapter->device_id = device_id;
    adapter->gpu_context = NULL;

#ifdef WITH_CUDA
    gpu_fermat_ctx *ctx = gpu_fermat_init(device_id, GPU_ADAPTER_MAX_BATCH);
    if (ctx) {
        adapter->gpu_context = ctx;
        printf("[GPUAdapter] Initialized CUDA device %d (%s)\n",
               device_id, gpu_fermat_device_name(ctx));
    } else {
        fprintf(stderr,
                "[GPUAdapter] CUDA init failed for device %d, "
                "falling back to CPU Fermat\n", device_id);
    }
#else
    printf("[GPUAdapter] Initialized for device %d (CPU fallback mode)\n", 
           device_id);
#endif
    
    return adapter;
}

int gpu_adapter_device_count(void) {
#ifdef WITH_CUDA
    return gpu_fermat_device_count();
#else
    return 1;
#endif
}

void gpu_adapter_set_candidate_bits(struct gpu_adapter *adapter, uint32_t bits) {
    if (!adapter) return;
#ifdef WITH_CUDA
    gpu_fermat_ctx *ctx = (gpu_fermat_ctx *)adapter->gpu_context;
    if (!ctx) return;
    int limbs = (int)((bits + 63U) / 64U);
    if (limbs < 1) limbs = 1;
#ifdef WITH_CGBN_FERMAT
    /* Round up to the nearest CGBN-supported width.  Some even widths have no
       valid CGBN instantiation: AL=10 (640-bit) is 20 32-bit limbs and TPI=4
       → 5 limbs/thread (the unimplemented dlimbs>4 path), TPI=8 → non-integer
       2.5.  The scalar fallback for those is register-heavy (~255 regs →
       ~16% occupancy) and measured ~3.6x slower per candidate than CGBN
       AL=12.  Padding with zero high limbs is arithmetic-safe
       (n < 2^bits < 2^(AL*64)) and lands on the production-verified CGBN
       paths. */
    {
        static const int cgbn_als[] = {2, 4, 6, 8, 12, 16, 20};
        int nl = (int)GPU_NLIMBS;
        if (limbs > nl) limbs = nl;
        for (size_t i = 0; i < sizeof(cgbn_als) / sizeof(cgbn_als[0]); i++) {
            if (cgbn_als[i] >= limbs && cgbn_als[i] <= nl) {
                limbs = cgbn_als[i];
                break;
            }
        }
    }
#else
    /* CGBN unavailable: round up odd widths to even (the scalar dispatch
       supports every exact width 1..GPU_NLIMBS). */
    if ((limbs & 1) != 0 && limbs < GPU_NLIMBS) limbs++;
    if (limbs > GPU_NLIMBS) limbs = GPU_NLIMBS;
#endif
    gpu_fermat_set_limbs(ctx, limbs);
#else
    (void)bits;
#endif
}

int gpu_adapter_get_limbs(struct gpu_adapter *adapter) {
    if (!adapter) return 1;
#ifdef WITH_CUDA
    gpu_fermat_ctx *ctx = (gpu_fermat_ctx *)adapter->gpu_context;
    if (ctx) return gpu_fermat_get_limbs(ctx);
#endif
    return 1;
}

#ifdef WITH_CUDA
struct gpu_fermat_ctx *gpu_adapter_get_fermat_ctx(struct gpu_adapter *adapter) {
    if (!adapter) return NULL;
    return (gpu_fermat_ctx *)adapter->gpu_context;
}
#endif

int gpu_adapter_async_submit_packed(struct gpu_adapter *adapter, int slot,
                                    const uint64_t *packed, size_t count) {
#ifdef WITH_CUDA
    if (!adapter || !packed || count == 0 || !adapter->gpu_context) return -1;
    return gpu_fermat_submit_try((gpu_fermat_ctx *)adapter->gpu_context, slot,
                                 packed, count);
#else
    (void)adapter; (void)slot; (void)packed; (void)count;
    return -1;
#endif
}

#ifdef WITH_CUDA
/* Pack n into GPU_NLIMBS little-endian 64-bit limbs, zero-padded.
   Returns 0 on success, -1 if n is negative or wider than GPU_NLIMBS*64
   bits (caller must use the CPU fallback for that candidate). */
static int pack_candidate_limbs(mpz_t n, uint64_t *limbs) {
    memset(limbs, 0, (size_t)GPU_NLIMBS * sizeof(uint64_t));
    if (mpz_sgn(n) < 0) return -1;
    if (mpz_sgn(n) == 0) return 0;
    if (mpz_sizeinbase(n, 2) > (size_t)GPU_NLIMBS * 64U) return -1;
    size_t count = 0;
    mpz_export(limbs, &count, -1, sizeof(uint64_t), 0, 0, n);
    return 0;
}

static int pack_candidate_active_limbs(mpz_t n, uint64_t *limbs, int active_limbs) {
    memset(limbs, 0, (size_t)active_limbs * sizeof(uint64_t));
    if (mpz_sgn(n) < 0 ||
        mpz_sizeinbase(n, 2) > (size_t)active_limbs * 64U) return -1;
    if (mpz_sgn(n) != 0) {
        size_t count = 0;
        mpz_export(limbs, &count, -1, sizeof(uint64_t), 0, 0, n);
    }
    return 0;
}
#endif

int gpu_adapter_test_batch(struct gpu_adapter *adapter, 
                           struct gpu_batch *batch) {
    if (!adapter || !batch || batch->count == 0) return -1;

#ifdef WITH_CUDA
    gpu_fermat_ctx *ctx = (gpu_fermat_ctx *)adapter->gpu_context;
    if (ctx) {
        uint32_t n = batch->count;
        uint64_t *cand_limbs =
            (uint64_t *)malloc((size_t)n * (size_t)GPU_NLIMBS * sizeof(uint64_t));
        uint8_t *gpu_ok = (uint8_t *)malloc(n);
        uint8_t *needs_cpu = (uint8_t *)calloc(n, 1);

        if (cand_limbs && gpu_ok && needs_cpu) {
            for (uint32_t i = 0; i < n; i++) {
                if (pack_candidate_limbs(batch->candidates[i],
                                          cand_limbs + (size_t)i * GPU_NLIMBS) != 0) {
                    needs_cpu[i] = 1;
                }
            }

            if (gpu_fermat_test_batch(ctx, cand_limbs, gpu_ok, n) < 0) {
                /* GPU batch failed outright: fail closed to CPU for all. */
                memset(needs_cpu, 1, n);
            }

            for (uint32_t i = 0; i < n; i++) {
                batch->is_prime[i] = needs_cpu[i]
                    ? (uint8_t)fermat_test_probable_prime(batch->candidates[i], 5)
                    : gpu_ok[i];
            }

            free(cand_limbs);
            free(gpu_ok);
            free(needs_cpu);
            return 0;
        }

        free(cand_limbs);
        free(gpu_ok);
        free(needs_cpu);
        /* malloc failure: fall through to the CPU-only loop below. */
    }
#endif
    
    /* CPU fallback: use Fermat test for each candidate */
    for (uint32_t i = 0; i < batch->count; i++) {
        batch->is_prime[i] = fermat_test_probable_prime(batch->candidates[i], 5);
    }
    
    return 0;
}

int gpu_adapter_async_submit(struct gpu_adapter *adapter, int slot,
                             struct gpu_batch *batch) {
#ifdef WITH_CUDA
    if (!adapter || !batch || !adapter->gpu_context || batch->count == 0) return -1;
    gpu_fermat_ctx *ctx = (gpu_fermat_ctx *)adapter->gpu_context;
    int limbs = gpu_fermat_get_limbs(ctx);
    uint64_t *packed = (uint64_t *)malloc((size_t)batch->count *
                                          (size_t)limbs * sizeof(uint64_t));
    if (!packed) return -1;
    for (uint32_t i = 0; i < batch->count; i++) {
        if (pack_candidate_active_limbs(batch->candidates[i],
                                        packed + (size_t)i * (size_t)limbs,
                                        limbs) != 0) {
            free(packed);
            return -1;
        }
    }
    int result = gpu_fermat_submit_try(ctx, slot, packed, batch->count);
    free(packed);
    return result;
#else
    (void)adapter; (void)slot; (void)batch;
    return -1;
#endif
}

int gpu_adapter_async_collect(struct gpu_adapter *adapter, int slot,
                              struct gpu_batch *batch) {
#ifdef WITH_CUDA
    if (!adapter || !batch || !adapter->gpu_context || batch->count == 0) return -1;
    return gpu_fermat_collect((gpu_fermat_ctx *)adapter->gpu_context, slot,
                              batch->is_prime, batch->count) < 0 ? -1 : 0;
#else
    (void)adapter; (void)slot; (void)batch;
    return -1;
#endif
}

void gpu_adapter_get_results(struct gpu_adapter *adapter, 
                             struct gpu_batch *batch,
                             uint64_t *tested_count,
                             uint64_t *kernel_time_us) {
    if (!adapter || !batch) return;
    
    if (tested_count) *tested_count = batch->count;
    if (kernel_time_us) *kernel_time_us = 0;  /* Not tracked (fine-grained CUDA event timing has hot-path overhead). */
}

void gpu_adapter_free(struct gpu_adapter *adapter) {
    if (!adapter) return;

#ifdef WITH_CUDA
    if (adapter->gpu_context) {
        gpu_fermat_destroy((gpu_fermat_ctx *)adapter->gpu_context);
    }
#endif
    
    free(adapter);
    printf("[GPUAdapter] Freed\n");
}
