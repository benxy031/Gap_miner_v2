/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Adapter Implementation
 *
 * Without WITH_CUDA: CPU-based fallback (per-candidate GMP Fermat test).
 * With WITH_CUDA: batches candidates to the CUDA Fermat kernel in
 * new_src/gpu/gpu_fermat.cu (scalar kernel, or CGBN when built with
 * WITH_CGBN_FERMAT=1). Any candidate that doesn't fit the active GPU width,
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
    return gpu_adapter_init_limbs(device_id, GPU_ADAPTER_MAX_BATCH, 0);
}

/* Same, with the Fermat staging sized for `limbs` limbs (0 = compiled max) and
   for `max_batch` candidates.  Each worker thread owns one adapter, so these
   two numbers multiply the fleet's memory: see the note in gpu_fermat_init_limbs. */
struct gpu_adapter *gpu_adapter_init_limbs(int device_id, size_t max_batch,
                                           int limbs) {
    struct gpu_adapter *adapter = 
        (struct gpu_adapter *)malloc(sizeof(struct gpu_adapter));
    
    if (!adapter) return NULL;
    
    adapter->device_id = device_id;
    adapter->gpu_context = NULL;

#ifdef WITH_CUDA
    if (max_batch == 0 || max_batch > GPU_ADAPTER_MAX_BATCH)
        max_batch = GPU_ADAPTER_MAX_BATCH;
    gpu_fermat_ctx *ctx = gpu_fermat_init_limbs(device_id, max_batch, limbs);
    if (ctx) {
        adapter->gpu_context = ctx;
        printf("[GPUAdapter] Initialized CUDA device %d (%s) staging: "
               "max_batch=%zu limbs=%d (~%zu MB device + ~%zu MB pinned)\n",
               device_id, gpu_fermat_device_name(ctx), max_batch,
               gpu_fermat_get_limbs(ctx),
               (size_t)(4 * 2 * max_batch *
                        (size_t)gpu_fermat_get_limbs(ctx) * 8 / (1024 * 1024)),
               (size_t)(4 * 2 * max_batch *
                        (size_t)gpu_fermat_get_limbs(ctx) * 8 / (1024 * 1024)));
    } else {
        fprintf(stderr,
                "[GPUAdapter] CUDA init failed for device %d, "
                "falling back to CPU Fermat\n", device_id);
    }
#else
    (void)max_batch;
    (void)limbs;
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
    gpu_fermat_set_limbs(ctx, gpu_fermat_limbs_for_bits(bits));
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

uint32_t gpu_adapter_crt_candidate_bits(uint32_t shift, const mpz_t period,
                                        uint64_t max_interval) {
    mpz_t bound, extra;
    mpz_inits(bound, extra, NULL);
    mpz_set_ui(bound, 1);
    mpz_mul_2exp(bound, bound, 256U + shift);
    mpz_set_ui(extra, 1);
    mpz_mul_2exp(extra, extra, shift);
    if (mpz_cmp(period, extra) > 0)
        mpz_set(extra, period);
    mpz_add(bound, bound, extra);
    mpz_import(extra, 1, -1, sizeof(max_interval), 0, 0, &max_interval);
    mpz_add(bound, bound, extra);
    size_t bits = mpz_sizeinbase(bound, 2);
    mpz_clears(bound, extra, NULL);
    if (bits > UINT32_MAX) {
        fprintf(stderr, "[GPUAdapter] CRT candidate bound exceeds uint32_t bits\n");
        return UINT32_MAX;
    }
    return (uint32_t)bits;
}

int gpu_adapter_window_fits(const mpz_t base, uint64_t interval, int limbs) {
    if (mpz_sgn(base) < 0 || interval == 0 || limbs <= 0)
        return 0;
    size_t width = (size_t)limbs * 64U;
    size_t bits = mpz_sizeinbase(base, 2);
    if (bits > width)
        return 0;
    if (bits < width && width >= 128U)
        return 1;

    mpz_t end, delta;
    uint64_t last_offset = interval - 1U;
    mpz_init_set(end, base);
    mpz_init(delta);
    mpz_import(delta, 1, -1, sizeof(last_offset), 0, 0, &last_offset);
    mpz_add(end, end, delta);
    int fits = mpz_sizeinbase(end, 2) <= width;
    mpz_clears(end, delta, NULL);
    return fits;
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
        int limbs = gpu_fermat_get_limbs(ctx);
        uint64_t *cand_limbs =
            (uint64_t *)malloc((size_t)n * (size_t)limbs * sizeof(uint64_t));
        uint8_t *gpu_ok = (uint8_t *)malloc(n);
        uint8_t *needs_cpu = (uint8_t *)calloc(n, 1);

        if (cand_limbs && gpu_ok && needs_cpu) {
            uint32_t too_wide = 0;
            for (uint32_t i = 0; i < n; i++) {
                if (pack_candidate_active_limbs(
                        batch->candidates[i],
                        cand_limbs + (size_t)i * (size_t)limbs, limbs) != 0) {
                    needs_cpu[i] = 1;
                    too_wide++;
                }
            }
            if (too_wide) {
                fprintf(stderr, "[GPUAdapter] %u candidates exceed the "
                        "%d-limb GPU width; using CPU for those candidates\n",
                        too_wide, limbs);
            }
            size_t cap = gpu_fermat_max_batch(ctx);
            for (size_t off = 0; off < n;) {
                size_t chunk = (size_t)n - off;
                if (chunk > cap) chunk = cap;
                if (chunk == 0 ||
                    gpu_fermat_test_batch(ctx, cand_limbs + off * (size_t)limbs,
                                          gpu_ok + off, chunk) < 0) {
                    fprintf(stderr, "[GPUAdapter] GPU batch failed; using CPU "
                            "for the remaining candidates\n");
                    memset(needs_cpu + off, 1, (size_t)n - off);
                    break;
                }
                off += chunk;
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
        fprintf(stderr, "[GPUAdapter] batch staging allocation failed; "
                "using CPU for this batch\n");
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
