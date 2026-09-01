/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GPU Worker Thread Implementation (Phase 4: Full Mining Pipeline)
 *
 * Per-GPU worker:
 * 1. Get nonce from atomic counter
 * 2. Run CPU sieve to extract probable prime candidates
 * 3. Apply primality filter (BPSW test)
 * 4. Detect prime gap pairs and compute merit
 * 5. Queue gaps above threshold for submission
 */

#define _POSIX_C_SOURCE 200809L

#include "worker_gpu.h"
#include "sieve_core.h"
#include "gap_detection.h"
#include "halfclass.h"
#include "primality_euler.h"
#include "primality_bpsw.h"
#include "primality_limbs.h"
#include "gpu_adapter.h"
#include "crt_runtime.h"
#include "gapcoin_work.h"
#ifdef WITH_CUDA
#include "gpu/gpu_sieve.h"
#include "gpu/gpu_fermat.h"
#endif
#include "record_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <gmp.h>
#include <math.h>
#include <time.h>

/* Editor/IntelliSense fallback: GPU_NLIMBS normally arrives from the
   Makefile (-DGPU_NLIMBS=20).  The #ifndef guard is idempotent with the
   command-line define and with gpu_fermat.h's own guard, so compiled
   behavior is unchanged. */
#ifndef GPU_NLIMBS
#define GPU_NLIMBS 16
#endif

/* CPU limb-based Euler–Plumb test with a per-window limb cache.
 *
 * Candidate = base + offset with ascending offsets inside a window, so the
 * candidate limbs are cached and advanced by 64-bit delta increments between
 * consecutive candidates instead of a GMP conversion + mpz_powm per candidate.
 * Falls back to the GMP base-2+3 Euler path when the limb count exceeds
 * PRIMALITY_CPU_MAX_LIMBS (not reachable for Gapcoin shifts ≤ 1024). */
struct worker_limb_cache {
    uint64_t limbs[PRIMALITY_CPU_MAX_LIMBS];
    int nlimbs;
    uint64_t last_offset;
    int valid;
};

static void worker_limb_cache_reset(struct worker_limb_cache *cache) {
    cache->valid = 0;
    cache->nlimbs = 0;
    cache->last_offset = 0;
}

static int worker_env_enabled(const char *value) {
    if (!value) return 0;
    while (*value == ' ' || *value == '\t' || *value == '\n' || *value == '\r') {
        value++;
    }
    if (*value == '\0' || *value == '0' || *value == 'n' || *value == 'N' ||
        *value == 'f' || *value == 'F') {
        return 0;
    }
    return 1;
}

/* GAPMINER_CPU_LIMBS=1 switches the CPU Euler filter from GMP mpz_powm to the
   ported fixed-limb CIOS Montgomery path (primality_limbs.c).  Measured on the
   dev host (GMP 6.x, modern x86-64): GMP wins at 311/765/1280-bit, so the
   default stays GMP; the limb path is kept for hosts where GMP's tuned asm is
   unavailable or slower. */
static int g_cpu_limbs_checked = 0;
static int g_cpu_limbs_enabled = 0;

static int worker_cpu_limbs_enabled(void) {
    if (!g_cpu_limbs_checked) {
        g_cpu_limbs_checked = 1;
        g_cpu_limbs_enabled = worker_env_enabled(getenv("GAPMINER_CPU_LIMBS"));
        if (g_cpu_limbs_enabled) {
            fprintf(stderr,
                    "[CPU Euler] GAPMINER_CPU_LIMBS=1: fixed-limb Montgomery "
                    "path (ADX compiled=%d, ADX enabled=%d)\n",
                    primality_cpu_adx_compiled(), primality_cpu_adx_enabled());
        }
    }
    return g_cpu_limbs_enabled;
}

static int worker_limb_cache_euler(struct worker_limb_cache *cache,
                                   struct euler_context *euler_context,
                                   mpz_t scratch, const mpz_t base,
                                   uint64_t offset) {
    if (!worker_cpu_limbs_enabled()) {
        mpz_set(scratch, base);
        mpz_add_ui(scratch, scratch, (unsigned long)offset);
        return euler_quick_probable_prime_with_context(euler_context, scratch);
    }
    if (cache->valid && offset >= cache->last_offset) {
        uint64_t delta = offset - cache->last_offset;
        if (delta == 0 ||
            primality_limbs_add_u64(cache->limbs, PRIMALITY_CPU_MAX_LIMBS,
                                    delta)) {
            cache->last_offset = offset;
            cache->nlimbs = primality_limbs_effective_nl(
                cache->limbs, PRIMALITY_CPU_MAX_LIMBS);
            return euler_test_cpu_nlimbs(cache->limbs, cache->nlimbs);
        }
        cache->valid = 0;
    }
    mpz_set(scratch, base);
    mpz_add_ui(scratch, scratch, (unsigned long)offset);
    cache->nlimbs = primality_limbs_export(scratch, cache->limbs,
                                           PRIMALITY_CPU_MAX_LIMBS);
    cache->last_offset = offset;
    cache->valid = 1;
    if (cache->nlimbs <= 0) {
        return euler_quick_probable_prime_with_context(euler_context, scratch);
    }
    return euler_test_cpu_nlimbs(cache->limbs, cache->nlimbs);
}

/* Adaptive halo look-ahead: start small and only grow (re-sieving) to the
   full NON_CRT_LOOKAHEAD_SIZE cap if no closing prime is found. Most windows
   close well within the first tier, so this usually avoids sieving/testing
   most of the fixed 8192-adder halo unconditionally on every window. */
#define NON_CRT_HALO_INITIAL 1024U

/* Global shutdown flag */
static volatile int g_stop_requested = 0;

/* Per-worker statistics */
struct worker_counter_state {
    _Atomic uint64_t nonces_processed;
    _Atomic uint64_t candidates_generated;
    _Atomic uint64_t candidates_tested;
    _Atomic uint64_t euler_passes;
    _Atomic uint64_t euler_pairs;
    _Atomic uint64_t merit_candidates;
    _Atomic uint64_t bpsw_attempts;
    _Atomic uint64_t gaps_found;
    _Atomic uint64_t gaps_submitted;
    _Atomic uint64_t max_gap_length;
    _Atomic uint64_t max_merit_scaled;
    _Atomic uint64_t gpu_euler_skipped;
    _Atomic uint64_t gpu_sieve_calls;
    _Atomic uint64_t gpu_sieve_windows;
    _Atomic uint64_t smart_tail_skipped;
    _Atomic uint64_t gpu_accounted_us;
} __attribute__((aligned(64)));

static struct worker_counter_state g_worker_stats[8] = {0};  /* Max 8 GPUs */

#define MERIT_STAT_SCALE 1000000.0

static void worker_atomic_max(_Atomic uint64_t *target, uint64_t value) {
    uint64_t current = atomic_load_explicit(target, memory_order_relaxed);
    while (current < value &&
           !atomic_compare_exchange_weak_explicit(target, &current, value,
                                                  memory_order_relaxed,
                                                  memory_order_relaxed)) {
    }
}

static uint64_t worker_scale_merit(double merit) {
    if (merit <= 0.0) return 0;
    if (merit >= (double)UINT64_MAX / MERIT_STAT_SCALE) return UINT64_MAX;
    return (uint64_t)(merit * MERIT_STAT_SCALE + 0.5);
}

/* Thread-safe gap queue for submissions. Pushes are rare (merit-qualified,
   BPSW-verified candidates only), so a plain mutex is used instead of a
   lock-free MPSC scheme to avoid a torn-write race between workers. */
#define MAX_PENDING_GAPS 256

static struct {
    struct gap_queue_entry queue[MAX_PENDING_GAPS];
    uint32_t head;
    uint32_t tail;
} g_gap_queue = {{{0}}, 0, 0};
static pthread_mutex_t g_gap_queue_lock = PTHREAD_MUTEX_INITIALIZER;

/* Disabled by default; main enables it only when the operator opts in. */
static _Atomic int g_submission_enabled = 0;

void worker_set_submission_enabled(int enabled) {
    atomic_store_explicit(&g_submission_enabled, enabled ? 1 : 0, memory_order_release);
}

/* Disabled by default; main enables it only when built WITH_CUDA and the
   operator opts in via --enable-gpu-fermat. */
static _Atomic int g_gpu_fermat_enabled = 0;

void worker_set_gpu_fermat_enabled(int enabled) {
    atomic_store_explicit(&g_gpu_fermat_enabled, enabled ? 1 : 0, memory_order_release);
}

static int worker_queue_push(const struct gap_queue_entry *entry) {
    int pushed = 0;
    pthread_mutex_lock(&g_gap_queue_lock);
    uint32_t next = (g_gap_queue.head + 1U) % MAX_PENDING_GAPS;
    if (next != g_gap_queue.tail) {
        g_gap_queue.queue[g_gap_queue.head] = *entry;
        g_gap_queue.head = next;
        pushed = 1;
    }
    pthread_mutex_unlock(&g_gap_queue_lock);
    return pushed;
}

static void worker_wait_for_work(void) {
    const struct timespec delay = {.tv_sec = 0, .tv_nsec = 10000000L};
    nanosleep(&delay, NULL);
}

static int worker_snapshot_work(const struct worker_config *config,
                                uint64_t *generation, uint32_t *height,
                                uint32_t *shift, double *merit_threshold,
                                uint8_t h256[32], uint32_t *header_nonce) {
    if (!config || !config->work_lock || !config->work_generation) return 0;

    pthread_mutex_lock(config->work_lock);
    *generation = atomic_load_explicit(config->work_generation, memory_order_acquire);
    if (*generation == 0) {
        pthread_mutex_unlock(config->work_lock);
        return 0;
    }

    *height = config->height;
    *shift = config->shift;
    *merit_threshold = config->merit_threshold;
    memcpy(h256, config->h256, 32);
    *header_nonce = config->header_nonce;
    pthread_mutex_unlock(config->work_lock);
    return 1;
}

static void worker_mark_window_complete(const struct worker_config *config,
                                        uint64_t generation) {
    if (!config || !config->work_lock || !config->work_generation ||
        !config->completed_windows) {
        return;
    }

    pthread_mutex_lock(config->work_lock);
    if (atomic_load_explicit(config->work_generation, memory_order_acquire) == generation) {
        atomic_fetch_add_explicit(config->completed_windows, 1, memory_order_release);
    }
    pthread_mutex_unlock(config->work_lock);
}

/* CRT-mode work snapshot: shared 80-byte header prefix + pass nonce. */
static int worker_snapshot_crt_work(const struct worker_config *config,
                                    uint64_t *generation, uint32_t *height,
                                    double *merit_threshold,
                                    uint8_t hdr80[80], uint32_t *pass_nonce) {
    if (!config || !config->work_lock || !config->work_generation) return 0;

    pthread_mutex_lock(config->work_lock);
    *generation = atomic_load_explicit(config->work_generation, memory_order_acquire);
    if (*generation == 0) {
        pthread_mutex_unlock(config->work_lock);
        return 0;
    }

    *height = config->height;
    *merit_threshold = config->merit_threshold;
    memcpy(hdr80, config->hdr80, 80);
    *pass_nonce = config->pass_nonce;
    pthread_mutex_unlock(config->work_lock);
    return 1;
}

static void worker_set_base(mpz_t base, const uint8_t h256[32],
                            uint32_t shift, uint64_t nadd) {
    mpz_set_ui(base, 0);
    for (size_t i = 0; i < 32; i++) {
        mpz_mul_2exp(base, base, 8);
        mpz_add_ui(base, base, h256[i]);
    }
    mpz_mul_2exp(base, base, shift);
    mpz_add_ui(base, base, (unsigned long)nadd);
}

uint32_t non_crt_owned_window_size(uint32_t shift) {
    /* Larger windows amortize the halo re-sieve and let more sieving primes
       actually land in the window (a prime p > window+halo can never mark).
       For shift <= 26 this grows the window to the 2^20 cap, which is what
       lets the CPU sieve carry more of the load when the GPU Fermat path is
       saturated. */
    uint32_t window_shift = NON_CRT_MIN_WINDOW_SHIFT;
    if (shift > 6U) {
        window_shift = shift - 6U;
    }
    if (window_shift < NON_CRT_MIN_WINDOW_SHIFT) {
        window_shift = NON_CRT_MIN_WINDOW_SHIFT;
    }
    if (window_shift > NON_CRT_MAX_WINDOW_SHIFT) {
        window_shift = NON_CRT_MAX_WINDOW_SHIFT;
    }
    return 1U << window_shift;
}

uint64_t non_crt_windows_per_header(uint32_t shift) {
    if (shift < NON_CRT_MIN_WINDOW_SHIFT || shift >= 64U) return 0;
    return (1ULL << shift) / non_crt_owned_window_size(shift);
}

/* Runs the Euler filter (optionally GPU-pre-filtered) + gap detection + BPSW
   for one window's sieve survivors, and marks the window complete. Factored
   out so the GPU path can defer this until several windows' candidates have
   been batched together (see gpu_pending_flush below). */
static void worker_process_window(uint32_t worker_id, struct worker_config *config,
                                  mpz_t candidate, mpz_t p1, mpz_t p2,
                                  struct euler_context *euler_context, uint8_t *is_prime,
                                  uint32_t height, uint32_t shift, uint32_t header_nonce,
                                  uint64_t generation, double merit_threshold,
                                  const uint8_t h256[32], uint64_t window_start,
                                  uint32_t owned_window_size,
                                  const uint64_t *candidate_offsets, uint32_t candidate_count,
                                  const uint8_t *gpu_prime /* nullable, len=candidate_count */,
                                  int half_class) {
    mpz_t base;
    mpz_init(base);
    worker_set_base(base, h256, shift, window_start);

    struct worker_limb_cache limb_cache;
    worker_limb_cache_reset(&limb_cache);

    uint64_t owned_candidate_count = 0;
    uint64_t euler_pass_count = 0;
    uint32_t euler_candidate_count = candidate_count;

    for (uint32_t i = 0; i < candidate_count; i++) {
        if (gpu_prime) {
            /* The GPU Fermat kernel now tests base 2 AND base 3, so its
               verdict is a "probably prime" result and the CPU Euler filter
               is skipped entirely.  This keeps the modular-exponentiation
               load on the (mostly idle) GPU while the CPU only sieves.
               A real prime always passes both bases, so no gap can be
               missed; a rare base-2+3 pseudoprime is only a false positive,
               which BPSW rejects for any merit-qualified gap endpoint. */
            is_prime[i] = gpu_prime[i];
            if (!gpu_prime[i]) {
                atomic_fetch_add(&g_worker_stats[worker_id].gpu_euler_skipped, 1);
            }
        } else {
            /* Fast path: cached-limb Euler–Plumb test (CIOS Montgomery),
               GMP base-2+3 Euler only as an over-limb fallback. */
            is_prime[i] = (uint8_t)worker_limb_cache_euler(
                &limb_cache, euler_context, candidate, base,
                candidate_offsets[i]);
        }
        if (candidate_offsets[i] < owned_window_size) {
            owned_candidate_count++;
            euler_pass_count += is_prime[i];
        }

        /* The first primality-positive halo endpoint closes every owned gap. */
        if (candidate_offsets[i] >= owned_window_size && is_prime[i]) {
            euler_candidate_count = i + 1;
            break;
        }
    }

    atomic_fetch_add(&g_worker_stats[worker_id].candidates_generated, owned_candidate_count);
    atomic_fetch_add(&g_worker_stats[worker_id].candidates_tested, owned_candidate_count);
    atomic_fetch_add(&g_worker_stats[worker_id].euler_passes, euler_pass_count);

    struct gap_result *gaps = NULL;
    uint32_t gap_count = 0;
    struct gap_scan_stats gap_stats = {0};

    int gap_result = gap_detection_find(
        is_prime, candidate_offsets, euler_candidate_count, shift, base,
        merit_threshold, owned_window_size, &gap_stats, &gaps, &gap_count);

    /* HALF_CLASS mode: every reported gap is a visible-class candidate, not
       necessarily a true gap.  Any true qualifying gap with a visible
       predecessor is contained in some visible qualifying gap (a hidden
       endpoint only enlarges the visible gap), so verifying each candidate's
       interior against the hidden classes and splitting it into true
       consecutive-prime gaps misses nothing.  The true primes BEFORE the
       first visible prime have no visible predecessor at all, so the prefix
       [0, first_visible_prime) is verified explicitly. */
    if (half_class && euler_candidate_count >= 1) {
        struct gap_result *resolved = NULL;
        uint32_t resolved_count = 0;
        uint32_t dropped = 0;

        uint32_t first_vis = UINT32_MAX;
        for (uint32_t i = 0; i < euler_candidate_count; i++) {
            if (is_prime[i]) {
                first_vis = i;
                break;
            }
        }
        if (first_vis != UINT32_MAX && candidate_offsets[first_vis] > 0) {
            struct gap_result *pre = NULL;
            uint32_t pre_count = 0;
            if (!halfclass_verify_prefix(base, candidate_offsets[first_vis],
                                         1, owned_window_size,
                                         merit_threshold, &pre,
                                         &pre_count)) {
                dropped++;
            } else {
                resolved = pre;
                resolved_count = pre_count;
            }
        } else if (euler_candidate_count > 0) {
            /* No visible prime in the whole window (astronomically rare):
               verify the hidden classes up to the last candidate. */
            uint32_t last = euler_candidate_count - 1;
            struct gap_result *pre = NULL;
            uint32_t pre_count = 0;
            if (!halfclass_verify_prefix(base, candidate_offsets[last],
                                         is_prime[last] ? 1 : 0,
                                         owned_window_size, merit_threshold,
                                         &pre, &pre_count)) {
                dropped++;
            } else {
                resolved = pre;
                resolved_count = pre_count;
            }
        }

        for (uint32_t i = 0; i < gap_count; i++) {
            struct gap_result *sub = NULL;
            uint32_t sub_count = 0;
            if (!halfclass_resolve_gap(base, gaps[i].offset_p1,
                                       gaps[i].offset_p2, owned_window_size,
                                       merit_threshold, &sub, &sub_count)) {
                dropped++;
                continue;
            }
            if (sub_count == 0) {
                free(sub);
                continue;
            }
            struct gap_result *grown = (struct gap_result *)realloc(
                resolved,
                (size_t)(resolved_count + sub_count) * sizeof(*grown));
            if (!grown) {
                dropped++;
                free(sub);
                continue;
            }
            resolved = grown;
            memcpy(resolved + resolved_count, sub,
                   (size_t)sub_count * sizeof(*sub));
            resolved_count += sub_count;
            free(sub);
        }
        gap_detection_free_results(gaps);
        gaps = resolved;
        gap_count = resolved_count;
        if (dropped) {
            fprintf(stderr,
                    "[Worker %u] HALF_CLASS verification: %u visible-gap candidate(s) dropped (allocation failure)\n",
                    worker_id, dropped);
        }

        /* The visible-scan maxima are NOT true consecutive-prime gaps in
           HALF_CLASS mode (a visible gap sums several true gaps between
           hidden-class primes), so reporting them would print a bogus
           "Max Euler pair" (e.g. merit 36 while Merit candidates stays 0).
           Recompute the maxima from the resolved true gaps instead; 0 when
           no visible candidate qualified this window. */
        uint32_t true_max_len = 0;
        double true_max_merit = 0.0;
        for (uint32_t i = 0; i < gap_count; i++) {
            if (gaps[i].gap_length > true_max_len) {
                true_max_len = gaps[i].gap_length;
            }
            if (gaps[i].merit > true_max_merit) {
                true_max_merit = gaps[i].merit;
            }
        }
        gap_stats.max_gap_length = true_max_len;
        gap_stats.max_merit = true_max_merit;
    }

    atomic_fetch_add(&g_worker_stats[worker_id].euler_pairs, gap_stats.euler_pairs);
    worker_atomic_max(&g_worker_stats[worker_id].max_gap_length, gap_stats.max_gap_length);
    worker_atomic_max(&g_worker_stats[worker_id].max_merit_scaled,
                      worker_scale_merit(gap_stats.max_merit));

    /* BPSW runs only for gaps that already passed the merit threshold. */
    if (gap_result && gaps && gap_count > 0) {
        atomic_fetch_add(&g_worker_stats[worker_id].merit_candidates, gap_count);
        for (uint32_t i = 0; i < gap_count; i++) {
            mpz_set(p1, base);
            mpz_add_ui(p1, p1, (unsigned long)gaps[i].offset_p1);
            mpz_set(p2, base);
            mpz_add_ui(p2, p2, (unsigned long)gaps[i].offset_p2);

            atomic_fetch_add(&g_worker_stats[worker_id].bpsw_attempts, 1);
            if (baillie_psw_verify_gap_boundaries(p1, p2)) {
                uint64_t nadd = window_start + gaps[i].offset_p1;
                atomic_fetch_add(&g_worker_stats[worker_id].gaps_found, 1);
                if (atomic_load_explicit(&g_submission_enabled, memory_order_acquire)) {
                    struct gap_queue_entry entry = {0};
                    entry.height = height;
                    entry.shift = shift;
                    entry.gap_length = gaps[i].gap_length;
                    entry.merit = gaps[i].merit;
                    entry.generation = generation;
                    entry.header_nonce = header_nonce;
                    entry.nadd = nadd;
                    int queued = worker_queue_push(&entry);
                    fprintf(stderr,
                            "[Worker %u] BPSW candidate: height=%u nAdd=%llu gap=%u merit=%.2f (%s)\n",
                            worker_id, height, (unsigned long long)nadd,
                            gaps[i].gap_length, gaps[i].merit,
                            queued ? "queued for submission" : "submission queue full, dropped");
                    record_log_write(height, shift, header_nonce, nadd, p1,
                                     gaps[i].gap_length, gaps[i].merit,
                                     queued ? "queued" : "queue-full");
                } else {
                    fprintf(stderr,
                            "[Worker %u] BPSW candidate: height=%u nAdd=%llu gap=%u merit=%.2f (dry-run)\n",
                            worker_id, height, (unsigned long long)nadd,
                            gaps[i].gap_length, gaps[i].merit);
                    record_log_write(height, shift, header_nonce, nadd, p1,
                                     gaps[i].gap_length, gaps[i].merit, "dry-run");
                }
            }
        }
    }

    gap_detection_free_results(gaps);
    mpz_clear(base);
    worker_mark_window_complete(config, generation);
}

#ifdef WITH_CUDA
/* Accumulates several windows' sieve survivors before one GPU batch call,
   instead of one tiny GPU round-trip per window: at shift 18 a window only
   has ~200-300 survivors, far too few to amortize CUDA kernel-launch/PCIe
   overhead (measured: per-window synchronous GPU calls were ~4-6x SLOWER
   overall than CPU-only Euler). Batching to GPU_ADAPTER_MAX_BATCH candidates
   amortizes that fixed overhead across thousands of candidates instead. */
#define GPU_BATCH_MAX_WINDOWS 256
#define GPU_PENDING_FLUSH_TARGET (GPU_ADAPTER_MAX_BATCH * 2U)

/* GPU bitmap-sieve window batching (separate from Fermat batching). */
#define GPU_SIEVE_WINDOW_BATCH_DEFAULT 1024U
#define GPU_SIEVE_WINDOW_BATCH_MAX 4096U

struct gpu_pending_window {
    uint64_t window_start;
    uint64_t *offsets; /* owned copy: sieve_core reuses its buffer next call */
    uint32_t count;
};

struct gpu_pending_state {
    struct gpu_pending_window windows[GPU_BATCH_MAX_WINDOWS];
    uint32_t n;
    uint64_t total;
    uint8_t h256[32];
    uint32_t shift, height, header_nonce, owned_window_size;
    uint64_t generation;
    double merit_threshold;
    int half_class;
};

struct gpu_sieve_window_state {
    uint64_t window_start;
    uint64_t scan_size;
    uint8_t h256[32];
    uint32_t shift;
    uint32_t height;
    uint32_t header_nonce;
    uint64_t generation;
    double merit_threshold;
};

struct gpu_sieve_batch_state {
    struct gpu_sieve_window_state windows[GPU_SIEVE_WINDOW_BATCH_MAX];
    uint64_t base_offsets[GPU_SIEVE_WINDOW_BATCH_MAX];
    uint64_t *bitmaps;
    size_t bitmap_words_capacity;
    uint32_t target;
    uint32_t count;
    uint64_t odd_interval_size;
    uint64_t odd_words;
    uint64_t first_odd_offset;
    int shape_valid;
};

/* Convert h256 (32 big-endian bytes) to 4 little-endian 64-bit limbs. */
static void h256_to_limbs(const uint8_t h256[32], uint64_t H[4]) {
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int b = 0; b < 8; b++)
            v |= ((uint64_t)h256[31 - (i * 8 + b)]) << (8 * b);
        H[i] = v;
    }
}

/* base = (h256 << shift) + window_start, as `limbs` little-endian 64-bit words.
   `limbs` must equal the adapter's active limb count (>= ceil((256+shift)/64)). */
static void compute_base_limbs(const uint64_t H[4], uint32_t shift,
                               uint64_t window_start, int limbs, uint64_t *base) {
    uint32_t q = shift >> 6;   /* shift / 64 */
    uint32_t r = shift & 63;   /* shift % 64 */
    memset(base, 0, (size_t)limbs * sizeof(uint64_t));
    for (int i = 0; i < 4; i++) {
        int lo = (int)q + i;
        if (lo >= limbs) break;
        base[lo] |= H[i] << r;
        if (r > 0 && lo + 1 < limbs)
            base[lo + 1] |= H[i] >> (64 - r);
    }
    uint64_t carry = window_start;
    for (int i = 0; i < limbs; i++) {
        uint64_t s = base[i] + carry;
        carry = (s < base[i]) ? 1ULL : 0ULL;
        base[i] = s;
    }
}

static void gpu_pending_flush(uint32_t worker_id, struct worker_config *config,
                              struct gpu_adapter *gpu, struct gpu_pending_state *st,
                              mpz_t candidate, mpz_t p1, mpz_t p2,
                              struct euler_context *euler_context, uint8_t *is_prime) {
    if (st->n == 0) return;

    int limbs = gpu_adapter_get_limbs(gpu);
    uint8_t *gpu_prime = (uint8_t *)malloc((size_t)st->total);
    uint64_t *packed = gpu_prime ?
        (uint64_t *)malloc((size_t)st->total * (size_t)limbs * sizeof(uint64_t)) : NULL;
    int gpu_ok = gpu_prime && packed;

    if (gpu_ok) {
        /* Pack limbs directly from h256 + shift + window offsets (no GMP). */
        uint64_t H[4];
        h256_to_limbs(st->h256, H);
        uint64_t *out = packed;
        for (uint32_t w = 0; w < st->n; w++) {
            uint64_t base[GPU_NLIMBS];
            compute_base_limbs(H, st->shift, st->windows[w].window_start,
                               limbs, base);
            for (uint32_t i = 0; i < st->windows[w].count; i++) {
                uint64_t carry = st->windows[w].offsets[i];
                for (int j = 0; j < limbs; j++) {
                    uint64_t s = base[j] + carry;
                    carry = (s < base[j]) ? 1ULL : 0ULL;
                    out[j] = s;
                }
                out += limbs;
            }
        }

        uint64_t off = 0;
        uint64_t chunk_offset[2] = {0, 0};
        uint32_t chunk_count[2] = {0, 0};
        int inflight_mask = 0;
        int next_slot = 0;
        while (off < st->total && gpu_ok) {
            uint64_t remaining = st->total - off;
            uint32_t chunk = remaining > GPU_ADAPTER_MAX_BATCH ?
                             GPU_ADAPTER_MAX_BATCH : (uint32_t)remaining;
            if (gpu_adapter_async_submit_packed(gpu, next_slot,
                                                packed + off * (size_t)limbs,
                                                chunk) != 0) {
                gpu_ok = 0;
                break;
            }
            chunk_offset[next_slot] = off;
            chunk_count[next_slot] = chunk;
            inflight_mask |= 1 << next_slot;
            off += chunk;
            next_slot ^= 1;

            /* Keep both CUDA streams full before collecting the oldest slot. */
            if (inflight_mask == 3) {
                int collect_slot = next_slot;
                struct gpu_batch done = {
                    .count = chunk_count[collect_slot],
                    .candidates = NULL,
                    .is_prime = gpu_prime + chunk_offset[collect_slot]
                };
                if (gpu_adapter_async_collect(gpu, collect_slot, &done) != 0) {
                    gpu_ok = 0;
                    break;
                }
                inflight_mask &= ~(1 << collect_slot);
            }
        }

        for (int collect_slot = 0; collect_slot < 2; collect_slot++) {
            if (!(inflight_mask & (1 << collect_slot))) continue;
            struct gpu_batch done = {
                .count = chunk_count[collect_slot],
                .candidates = NULL,
                .is_prime = gpu_prime + chunk_offset[collect_slot]
            };
            if (gpu_adapter_async_collect(gpu, collect_slot, &done) != 0) {
                gpu_ok = 0;
            }
        }

        /* Refresh GPU-accounted MR time (acc/wall metric). */
        struct gpu_fermat_ctx *fc = gpu_adapter_get_fermat_ctx(gpu);
        if (fc) {
            atomic_store(&g_worker_stats[worker_id].gpu_accounted_us,
                         gpu_fermat_accounted_us(fc));
        }
    }
    free(packed);
    if (!gpu_ok) {
        free(gpu_prime);
        gpu_prime = NULL;
    }

    uint64_t idx = 0;
    for (uint32_t w = 0; w < st->n; w++) {
        const uint8_t *slice = gpu_prime ? gpu_prime + idx : NULL;
        worker_process_window(worker_id, config, candidate, p1, p2,
                              euler_context, is_prime,
                              st->height, st->shift, st->header_nonce,
                              st->generation, st->merit_threshold, st->h256,
                              st->windows[w].window_start,
                              st->owned_window_size,
                              st->windows[w].offsets,
                              st->windows[w].count, slice, st->half_class);
        idx += st->windows[w].count;
        free(st->windows[w].offsets);
    }

    free(gpu_prime);
    st->n = 0;
    st->total = 0;
}

static void worker_dispatch_candidates(uint32_t worker_id,
                                       struct worker_config *config,
                                       struct gpu_adapter *gpu,
                                       struct gpu_pending_state *gpu_pending,
                                       mpz_t candidate, mpz_t p1, mpz_t p2,
                                       struct euler_context *euler_context,
                                       uint8_t *is_prime,
                                       uint32_t height, uint32_t shift,
                                       uint32_t header_nonce,
                                       uint64_t generation,
                                       double merit_threshold,
                                       const uint8_t h256[32],
                                       uint64_t window_start,
                                       uint32_t owned_window_size,
                                       const uint64_t *candidate_offsets,
                                       uint32_t candidate_count,
                                       int half_class) {
    if (gpu && gpu_pending) {
        uint64_t *offsets_copy =
            (uint64_t *)malloc((size_t)candidate_count * sizeof(uint64_t));
        if (offsets_copy) {
            memcpy(offsets_copy, candidate_offsets,
                   (size_t)candidate_count * sizeof(uint64_t));
            if (gpu_pending->n == 0) {
                memcpy(gpu_pending->h256, h256, 32);
                gpu_pending->shift = shift;
                gpu_pending->height = height;
                gpu_pending->header_nonce = header_nonce;
                gpu_pending->generation = generation;
                gpu_pending->merit_threshold = merit_threshold;
                gpu_pending->owned_window_size = owned_window_size;
                gpu_pending->half_class = half_class;
            }
            if (gpu_pending->n < GPU_BATCH_MAX_WINDOWS) {
                gpu_pending->windows[gpu_pending->n].window_start = window_start;
                gpu_pending->windows[gpu_pending->n].offsets = offsets_copy;
                gpu_pending->windows[gpu_pending->n].count = candidate_count;
                gpu_pending->n++;
                gpu_pending->total += candidate_count;
            } else {
                /* Buffer full: flush first so this window isn't dropped. */
                gpu_pending_flush(worker_id, config, gpu, gpu_pending,
                                  candidate, p1, p2,
                                  euler_context, is_prime);
                memcpy(gpu_pending->h256, h256, 32);
                gpu_pending->shift = shift;
                gpu_pending->height = height;
                gpu_pending->header_nonce = header_nonce;
                gpu_pending->generation = generation;
                gpu_pending->merit_threshold = merit_threshold;
                gpu_pending->owned_window_size = owned_window_size;
                gpu_pending->half_class = half_class;
                gpu_pending->windows[0].window_start = window_start;
                gpu_pending->windows[0].offsets = offsets_copy;
                gpu_pending->windows[0].count = candidate_count;
                gpu_pending->n = 1;
                gpu_pending->total = candidate_count;
            }
            if (gpu_pending->total >= GPU_PENDING_FLUSH_TARGET) {
                gpu_pending_flush(worker_id, config, gpu, gpu_pending,
                                  candidate, p1, p2,
                                  euler_context, is_prime);
            }
            return;
        }
    }

    /* Allocation failed or GPU Fermat disabled: process on CPU path now. */
    worker_process_window(worker_id, config, candidate, p1, p2, euler_context,
                          is_prime, height, shift, header_nonce, generation,
                          merit_threshold, h256, window_start,
                          owned_window_size, candidate_offsets,
                          candidate_count, NULL, half_class);
}

static void gpu_sieve_batch_reset(struct gpu_sieve_batch_state *state) {
    if (!state) return;
    state->count = 0;
    state->shape_valid = 0;
    state->odd_interval_size = 0;
    state->odd_words = 0;
    state->first_odd_offset = 0;
}

static uint32_t worker_parse_gpu_sieve_batch_size(const char *value) {
    if (!value || *value == '\0') return GPU_SIEVE_WINDOW_BATCH_DEFAULT;

    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (end == value) return GPU_SIEVE_WINDOW_BATCH_DEFAULT;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') {
        end++;
    }
    if (*end != '\0') return GPU_SIEVE_WINDOW_BATCH_DEFAULT;
    if (parsed == 0) return GPU_SIEVE_WINDOW_BATCH_DEFAULT;
    if (parsed > GPU_SIEVE_WINDOW_BATCH_MAX) {
        return GPU_SIEVE_WINDOW_BATCH_MAX;
    }
    return (uint32_t)parsed;
}

static void worker_gpu_sieve_autotune_batch(struct gpu_sieve_batch_state *state,
                                            const struct gpu_sieve_ctx *gpu_sieve) {
    if (!state || !gpu_sieve || state->count == 0) return;

    uint64_t elapsed_us = gpu_sieve_last_elapsed_us(gpu_sieve);
    if (elapsed_us == 0) return;

    /* Target a few milliseconds per launch: enough work for the GPU to be
       visible, but short enough to react quickly to new block generations. */
    if (elapsed_us < 2000U && state->target < GPU_SIEVE_WINDOW_BATCH_MAX) {
        uint32_t increase = state->target / 2U;
        if (increase == 0) increase = 1;
        state->target += increase;
        if (state->target > GPU_SIEVE_WINDOW_BATCH_MAX) {
            state->target = GPU_SIEVE_WINDOW_BATCH_MAX;
        }
    } else if (elapsed_us > 8000U && state->target > 1U) {
        state->target /= 2U;
        if (state->target == 0) state->target = 1;
    }
}

static void gpu_sieve_batch_flush(uint32_t worker_id,
                                  struct worker_config *config,
                                  struct sieve_core *sieve,
                                  struct gpu_sieve_ctx *gpu_sieve,
                                  size_t gpu_sieve_split_index,
                                  int *gpu_sieve_enabled,
                                  int *gpu_sieve_failure_reported,
                                  struct gpu_sieve_batch_state *batch_state,
                                  struct gpu_adapter *gpu,
                                  struct gpu_pending_state *gpu_pending,
                                  mpz_t base, mpz_t candidate, mpz_t p1,
                                  mpz_t p2,
                                  struct euler_context *euler_context,
                                  uint8_t *is_prime,
                                  int half_class,
                                  uint32_t owned_window_size) {
    if (!batch_state || batch_state->count == 0) return;

    int batch_ok = 0;
    size_t prime_count =
        gpu_sieve_split_index < sieve->small_primes_count ?
        sieve->small_primes_count - gpu_sieve_split_index : 0;
    size_t total_words =
        (size_t)batch_state->count * (size_t)batch_state->odd_words;

    if (gpu_sieve && gpu_sieve_enabled && *gpu_sieve_enabled &&
        batch_state->shape_valid && prime_count > 0 &&
        total_words <= batch_state->bitmap_words_capacity) {
        batch_ok = gpu_sieve_mark_high_primes_batch(
            gpu_sieve,
            batch_state->odd_interval_size,
            batch_state->first_odd_offset,
            batch_state->base_offsets,
            batch_state->count,
            sieve->small_primes + gpu_sieve_split_index,
            sieve->base_mod_p + gpu_sieve_split_index,
            prime_count,
            batch_state->bitmaps,
            total_words);
    }

    if (batch_ok) {
        atomic_fetch_add(&g_worker_stats[worker_id].gpu_sieve_calls, 1);
        atomic_fetch_add(&g_worker_stats[worker_id].gpu_sieve_windows,
                         batch_state->count);
        worker_gpu_sieve_autotune_batch(batch_state, gpu_sieve);
    } else if (gpu_sieve_enabled && *gpu_sieve_enabled) {
        if (gpu_sieve_failure_reported && !*gpu_sieve_failure_reported) {
            fprintf(stderr,
                    "[Worker %u] GPU bitmap sieve batch failed; falling back to CPU sieve\n",
                    worker_id);
            *gpu_sieve_failure_reported = 1;
        }
        *gpu_sieve_enabled = 0;
    }

    for (uint32_t i = 0; i < batch_state->count; i++) {
        struct gpu_sieve_window_state *window = &batch_state->windows[i];
        uint64_t *candidate_offsets = NULL;
        uint32_t candidate_count = 0;
        int sieve_ok = 0;

        sieve->interval_size = window->scan_size;
        sieve->bitmap_words = (window->scan_size + 63U) / 64U;

        if (batch_ok) {
            const uint64_t *window_bitmap =
                batch_state->bitmaps + ((size_t)i * (size_t)batch_state->odd_words);
            sieve_ok = sieve_core_run_from_cached_base_hybrid(
                sieve,
                window->window_start,
                window_bitmap,
                batch_state->odd_words,
                gpu_sieve_split_index,
                &candidate_offsets,
                &candidate_count);
        }

        if (!sieve_ok) {
            worker_set_base(base, window->h256, window->shift,
                            window->window_start);
            sieve_ok = sieve_core_run(sieve, base,
                                      &candidate_offsets,
                                      &candidate_count);
        }

        if (!sieve_ok || !candidate_offsets || candidate_count == 0) {
            worker_mark_window_complete(config, window->generation);
            continue;
        }

        if (half_class) {
            uint32_t bm60 = halfclass_base_mod60(
                window->h256, window->shift, window->window_start);
            candidate_count = halfclass_filter_offsets(
                bm60, candidate_offsets, candidate_count);
        }

        worker_dispatch_candidates(worker_id, config, gpu, gpu_pending,
                                   candidate, p1, p2, euler_context, is_prime,
                                   window->height, window->shift,
                                   window->header_nonce, window->generation,
                                   window->merit_threshold, window->h256,
                                   window->window_start, owned_window_size,
                                   candidate_offsets, candidate_count,
                                   half_class);
    }

    gpu_sieve_batch_reset(batch_state);
}

static size_t worker_gpu_sieve_split_index(const struct sieve_core *sieve) {
    if (!sieve || sieve->small_primes_count == 0) return 0;

    size_t by_value = sieve->small_primes_count;
    for (size_t i = 0; i < sieve->small_primes_count; i++) {
        if (sieve->small_primes[i] >= 1000U) {
            by_value = i;
            break;
        }
    }

    /* Keep only the cheap low-prime presieve on CPU. */
    return by_value;
}
#endif /* WITH_CUDA */

void *worker_thread_run(void *arg) {
    struct worker_config *config = (struct worker_config *)arg;
    if (!config) return NULL;
    
    uint32_t worker_id = config->worker_id;
    uint32_t sieve_limit = config->sieve_limit;
    uint32_t owned_window_size = non_crt_owned_window_size(config->shift);
    
    /* Scan a forward halo so gap endpoints may cross the owned range. */
    struct sieve_core sieve = {0};
    if (!sieve_core_init_window(&sieve,
                                owned_window_size + NON_CRT_LOOKAHEAD_SIZE,
                                sieve_limit)) {
        fprintf(stderr, "[Worker %u] Failed to initialize non-CRT sieve\n", worker_id);
        return NULL;
    }
    uint8_t *is_prime = malloc(sieve.interval_size * sizeof(*is_prime));
    if (!is_prime) {
        sieve_core_free(&sieve);
        fprintf(stderr, "[Worker %u] Failed to allocate Euler result buffer\n", worker_id);
        return NULL;
    }
    printf("[Worker %u] Started (CPU thread), merit_threshold=%.2f, sieve_limit=%u, window=%u, sieve=%s\n",
           worker_id, config->merit_threshold, sieve_limit, owned_window_size,
           sieve_core_simd_mode(&sieve));

    /* Initialize GMP for big number arithmetic. */
    mpz_t base, header_base, candidate, p1, p2;
    mpz_init(base);
    mpz_init(header_base);
    mpz_init(candidate);
    mpz_init(p1);
    mpz_init(p2);
    struct euler_context euler_context;
    euler_context_init(&euler_context);

#ifdef WITH_CUDA
    struct gpu_adapter *gpu = NULL;
    struct gpu_sieve_ctx *gpu_sieve = NULL;
    struct gpu_sieve_batch_state gpu_sieve_batch = {0};
    int gpu_sieve_batch_mode = 0;
    size_t gpu_sieve_split_index = 0;
    int gpu_sieve_enabled = 0;
    int gpu_sieve_failure_reported = 0;
    const char *gpu_sieve_env = getenv("GPU_SIEVE");
    const char *gpu_sieve_batch_env = getenv("GPU_SIEVE_BATCH");
    if (atomic_load_explicit(&g_gpu_fermat_enabled, memory_order_acquire)) {
        gpu = gpu_adapter_init((int)config->gpu_device);
        if (!gpu) {
            fprintf(stderr,
                    "[Worker %u] GPU Fermat init failed for device %u; "
                    "using CPU Euler filter only\n",
                    worker_id, config->gpu_device);
        }
    }

    if (gpu && worker_env_enabled(gpu_sieve_env)) {
        gpu_sieve_split_index = worker_gpu_sieve_split_index(&sieve);
        if (gpu_sieve_split_index < sieve.small_primes_count) {
            uint64_t max_odd_interval = (sieve.interval_size + 1U) >> 1;
            uint64_t max_odd_words = (max_odd_interval + 63U) >> 6;
            uint32_t batch_target =
                worker_parse_gpu_sieve_batch_size(gpu_sieve_batch_env);
            size_t bitmap_capacity_words = 0;
            const uint32_t bitmap_batch_capacity = GPU_SIEVE_WINDOW_BATCH_MAX;

            gpu_sieve = gpu_sieve_init((int)config->gpu_device,
                                       sieve.small_primes_count -
                                       gpu_sieve_split_index,
                                       max_odd_interval);

            if (gpu_sieve &&
                max_odd_words > 0 &&
                batch_target > 0 &&
                (size_t)bitmap_batch_capacity <= SIZE_MAX / (size_t)max_odd_words) {
                bitmap_capacity_words =
                    (size_t)bitmap_batch_capacity * (size_t)max_odd_words;
                gpu_sieve_batch.bitmaps =
                    (uint64_t *)calloc(bitmap_capacity_words,
                                       sizeof(*gpu_sieve_batch.bitmaps));
            }

            if (gpu_sieve && gpu_sieve_batch.bitmaps) {
                gpu_sieve_enabled = 1;
                gpu_sieve_batch_mode = 1;
                gpu_sieve_batch.target = batch_target;
                gpu_sieve_batch.bitmap_words_capacity = bitmap_capacity_words;
                gpu_sieve_batch_reset(&gpu_sieve_batch);
                fprintf(stderr,
                        "[Worker %u] GPU bitmap sieve enabled (GPU_SIEVE=%s, split_index=%zu, gpu_primes=%zu, device=%s, batch_target=%u windows)\n",
                        worker_id,
                        gpu_sieve_env,
                        gpu_sieve_split_index,
                        sieve.small_primes_count - gpu_sieve_split_index,
                        gpu_sieve_device_name(gpu_sieve),
                        batch_target);
                fprintf(stderr,
                        "[Worker %u] GPU bitmap sieve batch mode uses full halo windows (adaptive halo growth disabled in GPU_SIEVE mode)\n",
                        worker_id);
            } else {
                fprintf(stderr,
                        "[Worker %u] GPU bitmap sieve init failed; using CPU-only sieve path\n",
                        worker_id);
                if (gpu_sieve) {
                    gpu_sieve_destroy(gpu_sieve);
                    gpu_sieve = NULL;
                }
                free(gpu_sieve_batch.bitmaps);
                gpu_sieve_batch.bitmaps = NULL;
                gpu_sieve_batch.bitmap_words_capacity = 0;
                gpu_sieve_batch.target = 0;
                gpu_sieve_batch_mode = 0;
            }
        } else {
            fprintf(stderr,
                    "[Worker %u] GPU bitmap sieve not used (no eligible high-prime slice)\n",
                    worker_id);
        }
    } else if (worker_env_enabled(gpu_sieve_env) && !gpu) {
        fprintf(stderr,
                "[Worker %u] GPU bitmap sieve requested via GPU_SIEVE but GPU Fermat is unavailable; using CPU-only sieve path\n",
                worker_id);
    }
#endif
    
    /* Reset this worker's counters before it begins scanning. */
    atomic_store(&g_worker_stats[worker_id].nonces_processed, 0);
    atomic_store(&g_worker_stats[worker_id].candidates_generated, 0);
    atomic_store(&g_worker_stats[worker_id].candidates_tested, 0);
    atomic_store(&g_worker_stats[worker_id].euler_passes, 0);
    atomic_store(&g_worker_stats[worker_id].euler_pairs, 0);
    atomic_store(&g_worker_stats[worker_id].merit_candidates, 0);
    atomic_store(&g_worker_stats[worker_id].bpsw_attempts, 0);
    atomic_store(&g_worker_stats[worker_id].gaps_found, 0);
    atomic_store(&g_worker_stats[worker_id].gaps_submitted, 0);
    atomic_store(&g_worker_stats[worker_id].max_gap_length, 0);
    atomic_store(&g_worker_stats[worker_id].max_merit_scaled, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_euler_skipped, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_sieve_calls, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_sieve_windows, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_accounted_us, 0);
    uint64_t cached_generation = UINT64_MAX;
    uint64_t chunk_generation = UINT64_MAX;
    uint32_t chunk_next = 0;
    uint32_t chunk_end = 0;
#ifdef WITH_CUDA
    struct gpu_pending_state gpu_pending = {0};
#endif

    /* HALF_CLASS two-pass scan (non-CRT): only the visible classes
       {1,7,11,13,17,19,23,29} mod 60 are sieved/scanned/primality-tested;
       the hidden classes are verified on demand when a visible gap exceeds
       the threshold. */
    int half_class = 0;
    if (worker_env_enabled(getenv("HALF_CLASS")) ||
        worker_env_enabled(getenv("QUARTER_CLASS"))) {
        half_class = 1;
        fprintf(stderr,
                "[Worker %u] HALF_CLASS scan: visible classes {1,7,11,13,17,19,23,29} mod 60; hidden classes verified on demand\n",
                worker_id);
    }
    
    /* Process nonces until exhausted or stopped */
    while (!g_stop_requested) {
        uint64_t generation;
        uint32_t height;
        uint32_t shift;
        double merit_threshold;
        uint8_t h256[32];
        uint32_t header_nonce;
        if (!worker_snapshot_work(config, &generation, &height, &shift,
                      &merit_threshold, h256, &header_nonce)) {
            worker_wait_for_work();
            continue;
        }

        if (chunk_generation != generation) {
#ifdef WITH_CUDA
            if (gpu_sieve_batch_mode && gpu_sieve_batch.count > 0 &&
                gpu_sieve_batch.windows[0].generation != generation) {
                gpu_sieve_batch_flush(worker_id, config, &sieve, gpu_sieve,
                                      gpu_sieve_split_index,
                                      &gpu_sieve_enabled,
                                      &gpu_sieve_failure_reported,
                                      &gpu_sieve_batch,
                                      gpu, &gpu_pending,
                                      base, candidate, p1, p2,
                                      &euler_context, is_prime, half_class,
                                      owned_window_size);
            }
            if (gpu_pending.n > 0 && gpu_pending.generation != generation) {
                gpu_pending_flush(worker_id, config, gpu, &gpu_pending, candidate, p1, p2,
                                 &euler_context, is_prime);
            }
            if (gpu) {
                /* Narrow the GPU arithmetic width to what this shift actually
                   needs (h256 is 256 bits) instead of the full compiled
                   GPU_NLIMBS width -- O(limbs^2) cost, so this matters a lot. */
                gpu_adapter_set_candidate_bits(gpu, 256U + shift);
            }
#endif
            chunk_next = 0;
            chunk_end = 0;
            chunk_generation = generation;
        }

        if (cached_generation != generation) {
            worker_set_base(header_base, h256, shift, 0);
            if (sieve_core_prepare_base_mod_p(&sieve, header_base)) {
                cached_generation = generation;
            }
        }

        if (chunk_next == chunk_end) {
            uint64_t windows_per_header = non_crt_windows_per_header(shift);
            uint32_t window_limit = windows_per_header > 0 &&
                                    windows_per_header < UINT32_MAX ?
                                    (uint32_t)windows_per_header :
                                    config->nonce->max_value;
            uint32_t claimed_count = 0;
            chunk_next = atomic_nonce_claim(config->nonce, window_limit,
                                            NON_CRT_WINDOW_CHUNK,
                                            &claimed_count);
            chunk_end = chunk_next + claimed_count;
            if (claimed_count == 0) {
#ifdef WITH_CUDA
                if (gpu_sieve_batch_mode && gpu_sieve_batch.count > 0) {
                    gpu_sieve_batch_flush(worker_id, config, &sieve, gpu_sieve,
                                          gpu_sieve_split_index,
                                          &gpu_sieve_enabled,
                                          &gpu_sieve_failure_reported,
                                          &gpu_sieve_batch,
                                          gpu, &gpu_pending,
                                          base, candidate, p1, p2,
                                          &euler_context, is_prime, half_class,
                                          owned_window_size);
                }
                gpu_pending_flush(worker_id, config, gpu, &gpu_pending, candidate, p1, p2,
                                 &euler_context, is_prime);
#endif
                while (!g_stop_requested &&
                       atomic_load_explicit(config->work_generation,
                                            memory_order_acquire) == generation) {
                    worker_wait_for_work();
                }
                continue;
            }
        }

        uint32_t window_index = chunk_next++;

           uint64_t window_start = (uint64_t)window_index * owned_window_size;
        if (shift < 64 &&
            (window_start >= (1ULL << shift) ||
               owned_window_size > (1ULL << shift) - window_start)) {
            while (!g_stop_requested &&
                   atomic_load_explicit(config->work_generation, memory_order_acquire) == generation) {
                worker_wait_for_work();
            }
            continue;
        }

        worker_set_base(base, h256, shift, window_start);
        atomic_fetch_add(&g_worker_stats[worker_id].nonces_processed, 1);

        uint32_t base_mod60 =
            half_class ? halfclass_base_mod60(h256, shift, window_start) : 0U;

#ifdef WITH_CUDA
        if (gpu_sieve_batch_mode && gpu_sieve_enabled &&
            cached_generation == generation && sieve.base_mod_p_valid &&
            gpu_sieve_split_index < sieve.small_primes_count) {
            uint64_t nadd_remaining = UINT64_MAX;
            uint64_t scan_size = owned_window_size + NON_CRT_LOOKAHEAD_SIZE;
            if (shift < 64) {
                nadd_remaining = (1ULL << shift) - window_start;
                if (scan_size > nadd_remaining) {
                    scan_size = nadd_remaining;
                }
            }

            uint64_t first_odd_offset = (sieve.base_mod_p[0] & 1ULL) ? 0U : 1U;
            if (scan_size <= first_odd_offset) {
                worker_mark_window_complete(config, generation);
                continue;
            }

            uint64_t odd_interval_size =
                (scan_size - first_odd_offset + 1U) >> 1;
            uint64_t odd_words = (odd_interval_size + 63U) >> 6;

            if (!gpu_sieve_batch.shape_valid ||
                gpu_sieve_batch.odd_interval_size != odd_interval_size ||
                gpu_sieve_batch.first_odd_offset != first_odd_offset ||
                gpu_sieve_batch.count >= gpu_sieve_batch.target) {
                if (gpu_sieve_batch.count > 0) {
                    gpu_sieve_batch_flush(worker_id, config, &sieve, gpu_sieve,
                                          gpu_sieve_split_index,
                                          &gpu_sieve_enabled,
                                          &gpu_sieve_failure_reported,
                                          &gpu_sieve_batch,
                                          gpu, &gpu_pending,
                                          base, candidate, p1, p2,
                                          &euler_context, is_prime, half_class,
                                          owned_window_size);
                }
                if (gpu_sieve_enabled) {
                    gpu_sieve_batch.odd_interval_size = odd_interval_size;
                    gpu_sieve_batch.odd_words = odd_words;
                    gpu_sieve_batch.first_odd_offset = first_odd_offset;
                    gpu_sieve_batch.shape_valid = 1;
                }
            }

            if (gpu_sieve_enabled && gpu_sieve_batch.shape_valid &&
                gpu_sieve_batch.count < gpu_sieve_batch.target) {
                struct gpu_sieve_window_state *slot =
                    &gpu_sieve_batch.windows[gpu_sieve_batch.count];
                slot->window_start = window_start;
                slot->scan_size = scan_size;
                memcpy(slot->h256, h256, 32);
                slot->shift = shift;
                slot->height = height;
                slot->header_nonce = header_nonce;
                slot->generation = generation;
                slot->merit_threshold = merit_threshold;
                gpu_sieve_batch.base_offsets[gpu_sieve_batch.count] = window_start;
                gpu_sieve_batch.count++;

                if (gpu_sieve_batch.count >= gpu_sieve_batch.target) {
                    gpu_sieve_batch_flush(worker_id, config, &sieve, gpu_sieve,
                                          gpu_sieve_split_index,
                                          &gpu_sieve_enabled,
                                          &gpu_sieve_failure_reported,
                                          &gpu_sieve_batch,
                                          gpu, &gpu_pending,
                                          base, candidate, p1, p2,
                                          &euler_context, is_prime, half_class,
                                          owned_window_size);
                }
                continue;
            }

            if (!gpu_sieve_enabled) {
                gpu_sieve_batch_mode = 0;
            }
        }
#endif

        /* Adaptive halo: sieve+probe a small look-ahead first, and only grow
           to the full NON_CRT_LOOKAHEAD_SIZE cap if it doesn't close within
           that tier. The probe only tests the halo portion (offset >=
           owned_window_size) with the shared Euler context; it does not
           touch worker stats or gap detection -- those still run exactly
           once below, over whichever tier's final candidate list won, so
           this cannot change which gaps are found, only how much of the
           halo gets sieved/tested to find them. */
        uint64_t *candidate_offsets = NULL;
        uint32_t candidate_count = 0;
        int sieve_ok = 0;
        uint64_t halo_size = NON_CRT_HALO_INITIAL;

        for (;;) {
            uint64_t scan_size = owned_window_size + halo_size;
            uint64_t nadd_remaining = UINT64_MAX;
            if (shift < 64) {
                nadd_remaining = (1ULL << shift) - window_start;
                if (scan_size > nadd_remaining) {
                    scan_size = nadd_remaining;
                }
            }
            sieve.interval_size = scan_size;
            sieve.bitmap_words = (scan_size + 63U) / 64U;

            sieve_ok = cached_generation == generation ?
                sieve_core_run_from_cached_base(&sieve, window_start,
                                                &candidate_offsets,
                                                &candidate_count) :
                sieve_core_run(&sieve, base,
                               &candidate_offsets,
                               &candidate_count);
            if (!sieve_ok || !candidate_offsets || candidate_count == 0) break;

            /* HALF_CLASS: probe only the visible classes so the halo
               closing guarantee is a VISIBLE prime (a hidden closer must
               not keep the halo small and leave the last owned gap
               unclosed). */
            if (half_class) {
                candidate_count = halfclass_filter_offsets(
                    base_mod60, candidate_offsets, candidate_count);
            }

            int halo_closed = 0;
            {
                struct worker_limb_cache limb_cache;
                worker_limb_cache_reset(&limb_cache);
                for (uint32_t i = 0; i < candidate_count; i++) {
                    if (candidate_offsets[i] < owned_window_size) continue;
                    if (worker_limb_cache_euler(&limb_cache, &euler_context,
                                                candidate, base,
                                                candidate_offsets[i])) {
                        halo_closed = 1;
                        break;
                    }
                }
            }

            int halo_at_range_end = scan_size >= nadd_remaining;
            if (halo_closed || halo_size >= NON_CRT_LOOKAHEAD_SIZE || halo_at_range_end) {
                break;
            }
            halo_size = NON_CRT_LOOKAHEAD_SIZE; /* single growth step to the cap */
        }

        if (!sieve_ok || !candidate_offsets || candidate_count == 0) {
            worker_mark_window_complete(config, generation);
            continue;
        }

#ifdef WITH_CUDA
        worker_dispatch_candidates(worker_id, config, gpu, &gpu_pending,
                                   candidate, p1, p2, &euler_context, is_prime,
                                   height, shift, header_nonce, generation,
                                   merit_threshold, h256, window_start,
                                   owned_window_size,
                                   candidate_offsets, candidate_count,
                                   half_class);
#else
        worker_process_window(worker_id, config, candidate, p1, p2,
                              &euler_context, is_prime, height, shift,
                              header_nonce, generation, merit_threshold,
                              h256, window_start, owned_window_size,
                              candidate_offsets, candidate_count, NULL,
                              half_class);
#endif
    }
    
#ifdef WITH_CUDA
    if (gpu_sieve_batch_mode && gpu_sieve_batch.count > 0) {
        gpu_sieve_batch_flush(worker_id, config, &sieve, gpu_sieve,
                              gpu_sieve_split_index,
                              &gpu_sieve_enabled,
                              &gpu_sieve_failure_reported,
                              &gpu_sieve_batch,
                              gpu, &gpu_pending,
                              base, candidate, p1, p2,
                              &euler_context, is_prime, half_class,
                              owned_window_size);
    }
    gpu_pending_flush(worker_id, config, gpu, &gpu_pending, candidate, p1, p2,
                      &euler_context, is_prime);
#endif

    /* Cleanup */
    free(is_prime);
    sieve_core_free(&sieve);
    mpz_clear(base);
    mpz_clear(header_base);
    mpz_clear(candidate);
    mpz_clear(p1);
    mpz_clear(p2);
    euler_context_clear(&euler_context);
#ifdef WITH_CUDA
    if (gpu_sieve) gpu_sieve_destroy(gpu_sieve);
    free(gpu_sieve_batch.bitmaps);
    if (gpu) gpu_adapter_free(gpu);
#endif
    
    printf("[Worker %u] Stopped (nonces=%lu, candidates=%lu, gaps=%lu, gpu_sieve_calls=%lu, gpu_sieve_windows=%lu)\n",
           worker_id,
            atomic_load(&g_worker_stats[worker_id].nonces_processed),
            atomic_load(&g_worker_stats[worker_id].candidates_tested),
            atomic_load(&g_worker_stats[worker_id].gaps_found),
            atomic_load(&g_worker_stats[worker_id].gpu_sieve_calls),
            atomic_load(&g_worker_stats[worker_id].gpu_sieve_windows));
    
    return NULL;
}

#ifdef WITH_CUDA
/* Batch-test CRT survivors on the GPU Fermat kernel.  window_base is converted
   to `limbs` little-endian 64-bit words once, then each candidate offset is
   added with carry into the packed buffer.  Returns 0 on success, -1 if the
   GPU is unavailable (caller falls back to the CPU Euler filter). */
static int crt_gpu_batch_test(struct gpu_adapter *gpu, int limbs,
                              uint64_t *base_limbs, uint64_t *packed,
                              const mpz_t window_base,
                              const uint64_t *offsets, uint32_t count,
                              uint8_t *is_prime) {
    if (!gpu || !base_limbs || !packed || count == 0) return -1;

    size_t exported = 0;
    memset(base_limbs, 0, (size_t)limbs * sizeof(uint64_t));
    mpz_export(base_limbs, &exported, -1, sizeof(uint64_t), 0, 0, window_base);
    if (exported > (size_t)limbs) return -1;

    uint64_t *out = packed;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t carry = offsets[i];
        for (int j = 0; j < limbs; j++) {
            uint64_t s = base_limbs[j] + carry;
            carry = (s < base_limbs[j]) ? 1ULL : 0ULL;
            out[j] = s;
        }
        out += limbs;
    }

    uint64_t done = 0;
    int slot = 0;
    while (done < count) {
        uint32_t chunk = (count - done) > GPU_ADAPTER_MAX_BATCH ?
                         GPU_ADAPTER_MAX_BATCH : (uint32_t)(count - done);
        if (gpu_adapter_async_submit_packed(gpu, slot,
                                            packed + done * (size_t)limbs,
                                            chunk) != 0) {
            return -1;
        }
        struct gpu_batch batch = {
            .count = chunk,
            .candidates = NULL,
            .is_prime = is_prime + done
        };
        if (gpu_adapter_async_collect(gpu, slot, &batch) != 0) {
            return -1;
        }
        done += chunk;
    }
    return 0;
}

/* ── Fused pipeline: async double-buffered window flow ─────────────────
   Two windows are in flight: window i marks+extracts its head into device
   buffer i&1 and submits the MR asynchronously on fermat slot i&1, while the
   GPU is still running window i-1's MR.  Two windows later the host collects
   the results, applies the smart-scan tail-skip (extracting the tail from the
   still-intact bitmap only when no closing prime exists) and runs the gap
   scan.  Extraction is ordered (ascending offsets), so no host qsort is
   needed.  Fail-closed: any CUDA error disables the fused path. */
/* K-window MR batch accumulation (proposal #3): one flight = one accumulated
   MR batch of up to FUSED_MR_BATCH_MAX windows, submitted with ONE
   gpu_fermat_submit_device call. */
#define FUSED_MR_BATCH_MAX 8

struct fused_flight_window {
    uint32_t nonce;
    uint32_t count;             /* head candidates MR-tested in the batch */
    int split;
    uint32_t base_mod60;
    uint64_t base_limbs[GPU_NLIMBS];  /* snapshot: tail re-mark needs it */
    mpz_t window_base;
    mpz_t nadd0;
};

struct fused_flight {
    int active;                 /* accumulated MR batch submitted, not collected */
    int buf;                    /* accumulation candidate buffer (0/1) */
    int slot;                   /* fermat slot used (0/1) */
    int n_windows;              /* windows appended so far */
    int half_class;
    int limbs;
    uint32_t height;
    uint64_t generation;
    double merit_threshold;
    uint64_t needed_gap;        /* per generation */
    uint64_t back_limit;
    uint64_t head_end;
    uint64_t need_off;
    uint64_t interval;
    uint64_t base_window;       /* global window index of wins[0] (parity) */
    uint32_t total_count;       /* sum of head counts = MR batch size */
    struct fused_flight_window wins[FUSED_MR_BATCH_MAX];
};

/* Forward declaration: defined after this WITH_CUDA section (it is
   CUDA-independent). */
static void crt_scan_gaps(uint32_t worker_id, uint32_t height, uint32_t nonce,
                          uint64_t generation, double merit_threshold,
                          const struct crt_runtime *rt,
                          mpz_t window_base, mpz_t nadd0,
                          uint64_t back_limit, mpz_t p1, mpz_t p2,
                          mpz_t nadd_full,
                          const uint8_t *is_prime,
                          const uint64_t *candidate_offsets,
                          uint32_t tested_count, uint64_t owned_limit,
                          int half_class,
                          struct gpu_fermat_ctx *resolve_fermat,
                          int resolve_slot);

/* Stage 1 (launch side): mark all primes on-device into bitmap buf, extract
   the head (or the full window when the closing range is empty) in ascending
   offset order, and submit the MR kernel asynchronously on fermat slot. */
/* Held window for the pair-batched fused mark: window s-1 is held (mark
   data only — it was already submitted) until window s is ready, then BOTH
   are marked by one GPU kernel (gpu_sieve_mark_batch_from_bases writes the
   two ping-pong bitmaps).  Halves the mark kernel launches and stream
   syncs per window without changing the submit/collect cadence. */
struct fused_pending_state {
    int valid;
    int parity;                 /* bitmap index (0/1) of the held window */
    uint64_t interval;
    uint64_t base_limbs[GPU_NLIMBS];
};

static void fused_pending_clear(struct fused_pending_state *p) {
    p->valid = 0;
}

/* Fused window append: extract the head survivors from the window's bitmap
   (already marked) into the batch's candidate buffer at slot_base, keeping
   candidates contiguous for ONE accumulated MR submission per K windows. */
static int crt_fused_append_window(struct gpu_sieve_ctx *gpu_sieve,
                                   int bitmap_buf,
                                   int cand_buf,
                                   int split,
                                   const uint64_t *base_limbs,
                                   int limbs,
                                   uint64_t interval,
                                   uint64_t head_end,
                                   uint64_t *head_offsets,
                                   uint32_t *head_count_out,
                                   int half_class,
                                   uint32_t base_mod60,
                                   uint64_t region_start,
                                   uint32_t slot_base)
{
    uint64_t first_odd_offset = (base_limbs[0] & 1ULL) ? 0U : 1U;
    if (interval <= first_odd_offset) return 0;
    uint64_t odd_interval_size = (interval - first_odd_offset + 1U) >> 1;
    if (odd_interval_size == 0) return 0;

    /* Only split head/tail when the closing range [need_off, head_end) is
       wide enough that a closing prime is likely (>= 2*logbase): otherwise
       the tail would almost always be needed anyway and the split would just
       fragment the MR into small batches. */
    uint64_t lo = 0;
    uint64_t hi = odd_interval_size;
    if (split) {
        uint64_t head_hi_odd = 0;
        if (head_end > first_odd_offset) {
            head_hi_odd = (head_end - first_odd_offset + 1U) >> 1;
            if (head_hi_odd > odd_interval_size) {
                head_hi_odd = odd_interval_size;
            }
        }
        hi = head_hi_odd;
    }

    uint64_t *d_cands = NULL;
    unsigned int count = 0;
    uint64_t class_mask =
        half_class ? halfclass_visible_mask() : UINT64_MAX;
    if (!gpu_sieve_extract_pack_device_range_ex(
            gpu_sieve, odd_interval_size, first_odd_offset, lo, hi,
            bitmap_buf, cand_buf, base_limbs, limbs, &d_cands,
            head_offsets, &count, base_mod60, class_mask,
            region_start, slot_base)) {
        return 0;
    }
    *head_count_out = count;
    return 1;
}

/* Flush a partially-accumulated (not yet submitted) MR batch before the
   header changes or shutdown: submit whatever heads are in the buffer so
   the batch is not lost (it is collected like any active flight). */
static void fused_flush_partial(struct gpu_sieve_ctx *gpu_sieve,
                                struct gpu_fermat_ctx *fermat,
                                struct fused_flight *fl, int slot) {
    if (!gpu_sieve || !fermat || !fl || fl->active) return;
    if (fl->n_windows <= 0 || fl->total_count == 0) {
        fl->n_windows = 0;
        fl->total_count = 0;
        return;
    }
    uint64_t *d_batch = gpu_sieve_candidate_buffer(gpu_sieve, slot);
    if (gpu_fermat_submit_device(fermat, slot, d_batch,
                                 fl->total_count) == 0) {
        fl->active = 1;
    } else {
        fl->n_windows = 0;
        fl->total_count = 0;
    }
}

/* Stage 2 (collect side) for a K-window accumulated batch: collect the async
   MR results, per window apply the smart-scan tail-skip (on-demand bitmap
   re-mark + tail extract + MR only when no closing prime), update stats and
   run the gap scan per window.  Returns 1 on success. */
static int crt_fused_collect_batch(struct gpu_sieve_ctx *gpu_sieve,
                                   struct gpu_fermat_ctx *fermat,
                                   uint32_t worker_id,
                                   const struct crt_runtime *rt,
                                   struct fused_flight *fl,
                                   uint64_t *offsets,
                                   uint8_t *is_prime,
                                   mpz_t p1, mpz_t p2, mpz_t nadd_full,
                                   const uint64_t *primes,
                                   const uint64_t *inv_p,
                                   size_t prime_count)
{
    uint32_t total = fl->total_count;
    if (total > 0) {
        if (gpu_fermat_collect(fermat, fl->slot, is_prime, total) < 0) {
            return 0;
        }
    }

    uint64_t *d_batch = gpu_sieve_candidate_buffer(gpu_sieve, fl->buf);

    /* Batch arrays [0, total) hold ONLY the head slices, laid out
       window-major at append time.  Tail re-extracts must NOT be written
       into them (that would overwrite the next windows' head data and
       desynchronize offsets from MR flags — the false-gap bug).  Tails go
       into scratch buffers and each window is scanned through a spliced
       head+tail view. */
    size_t scratch_n = (size_t)fl->interval + 8U;
    uint64_t *tail_off = (uint64_t *)malloc(scratch_n * sizeof(uint64_t));
    uint8_t *tail_ip = (uint8_t *)malloc(scratch_n);
    uint64_t *win_off = (uint64_t *)malloc(scratch_n * sizeof(uint64_t));
    uint8_t *win_ip = (uint8_t *)malloc(scratch_n);
    if (!tail_off || !tail_ip || !win_off || !win_ip) {
        free(tail_off);
        free(tail_ip);
        free(win_off);
        free(win_ip);
        return 0;
    }
    uint64_t class_mask =
        fl->half_class ? halfclass_visible_mask() : UINT64_MAX;

    uint32_t head_cum = 0;
    uint32_t tail_accum = 0;

    for (int j = 0; j < fl->n_windows; j++) {
        struct fused_flight_window *wv = &fl->wins[j];
        uint32_t head_count = wv->count;
        uint32_t tested = head_count;
        int tail_skipped = 0;
        const uint8_t *win_ip_ptr = is_prime + head_cum;
        const uint64_t *win_off_ptr = offsets + head_cum;

        /* Smart-scan tail-skip (only when the head/tail split was used). */
        if (wv->split && tested > 0) {
            int have_closing = 0;
            for (uint32_t i = 0; i < tested; i++) {
                if (offsets[head_cum + i] >= fl->need_off &&
                    is_prime[head_cum + i]) {
                    have_closing = 1;
                    break;
                }
            }
            if (!have_closing) {
                uint64_t first_odd_offset =
                    (wv->base_limbs[0] & 1ULL) ? 0U : 1U;
                uint64_t odd_interval_size =
                    (fl->interval - first_odd_offset + 1U) >> 1;
                uint64_t head_hi_odd = 0;
                if (fl->head_end > first_odd_offset) {
                    head_hi_odd = (fl->head_end - first_odd_offset + 1U) >> 1;
                    if (head_hi_odd > odd_interval_size) {
                        head_hi_odd = odd_interval_size;
                    }
                }
                if (head_hi_odd < odd_interval_size) {
                    /* The window's bitmap was overwritten by later windows;
                       re-mark it on demand (deterministic: same base, same
                       primes — proven by the pair-batch experiment). */
                    int parity =
                        (int)((fl->base_window + (uint64_t)j) & 1ULL);
                    if (!gpu_sieve_mark_from_base(
                            gpu_sieve, odd_interval_size, first_odd_offset,
                            wv->base_limbs, fl->limbs, parity,
                            primes, inv_p, prime_count, NULL, 0)) {
                        free(tail_off);
                        free(tail_ip);
                        free(win_off);
                        free(win_ip);
                        return 0;
                    }
                    atomic_fetch_add(
                        &g_worker_stats[worker_id].gpu_sieve_calls, 1);
                    atomic_fetch_add(
                        &g_worker_stats[worker_id].gpu_sieve_windows, 1);

                    unsigned int tc = 0;
                    uint64_t *d_tail = NULL;
                    if (!gpu_sieve_extract_pack_device_range_ex(
                            gpu_sieve, odd_interval_size, first_odd_offset,
                            head_hi_odd, odd_interval_size, parity, fl->buf,
                            wv->base_limbs, fl->limbs, &d_tail,
                            tail_off, &tc,
                            wv->base_mod60, class_mask, fl->back_limit,
                            total + tail_accum)) {
                        free(tail_off);
                        free(tail_ip);
                        free(win_off);
                        free(win_ip);
                        return 0;
                    }
                    if (tc > 0) {
                        if (gpu_fermat_submit_device(
                                fermat, fl->slot,
                                d_batch +
                                    (size_t)(total + tail_accum) *
                                        (size_t)fl->limbs,
                                tc) < 0 ||
                            gpu_fermat_collect(fermat, fl->slot,
                                               tail_ip, tc) < 0) {
                            free(tail_off);
                            free(tail_ip);
                            free(win_off);
                            free(win_ip);
                            return 0;
                        }
                        /* Spliced ascending view: heads (all < head_end)
                           followed by the tail. */
                        memcpy(win_off, offsets + head_cum,
                               (size_t)head_count * sizeof(uint64_t));
                        memcpy(win_ip, is_prime + head_cum, head_count);
                        memcpy(win_off + head_count, tail_off,
                               (size_t)tc * sizeof(uint64_t));
                        memcpy(win_ip + head_count, tail_ip, tc);
                        win_off_ptr = win_off;
                        win_ip_ptr = win_ip;
                        tail_accum += tc;
                        tested = head_count + tc;
                    }
                }
            } else {
                tail_skipped = 1;
            }
        }

        uint64_t euler_pass_count = 0;
        for (uint32_t i = 0; i < tested; i++) {
            euler_pass_count += win_ip_ptr[i];
        }
        atomic_fetch_add(&g_worker_stats[worker_id].candidates_generated,
                         tested);
        atomic_fetch_add(&g_worker_stats[worker_id].candidates_tested,
                         tested);
        atomic_fetch_add(&g_worker_stats[worker_id].euler_passes,
                         euler_pass_count);
        if (tail_skipped) {
            atomic_fetch_add(&g_worker_stats[worker_id].smart_tail_skipped, 1);
        }

        uint64_t owned_limit = fl->back_limit + fl->needed_gap;
        crt_scan_gaps(worker_id, fl->height, wv->nonce, fl->generation,
                      fl->merit_threshold, rt, wv->window_base, wv->nadd0,
                      fl->back_limit, p1, p2, nadd_full,
                      win_ip_ptr, win_off_ptr, tested, owned_limit,
                      fl->half_class, fermat, fl->slot);
        head_cum += head_count;
    }
    free(tail_off);
    free(tail_ip);
    free(win_off);
    free(win_ip);
    return 1;
}
#endif /* WITH_CUDA */

/* GPU-side hidden-class resolution: mini-sieve + template prefilter on the
   host, base-2+3 MR batch on the GPU, BPSW only on the MR survivors.  MR has
   no false negatives (a real prime always passes), so the emitted set is
   identical to the CPU-only path; a composite survivor only wastes one BPSW
   call.  Falls back to the CPU-only resolution on GPU failure. */
#ifdef WITH_CUDA
int crt_gpu_resolve_gap_ex(struct gpu_fermat_ctx *fermat, int slot,
                                  mpz_t window_base, uint64_t off_a,
                                  uint64_t off_b, uint64_t owned_offset_limit,
                                  double merit_threshold,
                                  const struct halfclass_tpl *tpl,
                                  struct gap_result **out, uint32_t *out_count)
{
    *out = NULL;
    *out_count = 0;
    if (!fermat || off_b <= off_a || !window_base) return 1;

    uint64_t interval = off_b - off_a;
    uint32_t base_mod60 = (uint32_t)mpz_fdiv_ui(window_base, 60);
    uint32_t sub_mod60 = (uint32_t)((base_mod60 + (off_a % 60ULL)) % 60ULL);

    mpz_t bsub;
    mpz_init(bsub);
    mpz_set(bsub, window_base);
    mpz_add_ui(bsub, bsub, off_a);

    uint64_t *cand = NULL;
    int64_t nc = halfclass_collect_hidden_candidates(bsub, interval, sub_mod60,
                                                     off_a, tpl, 10000, &cand);
    if (nc < 0) {
        mpz_clear(bsub);
        return 0;
    }

    uint64_t *hid = NULL;
    uint32_t hid_count = 0;
    int ok = 1;
    /* The submit path strides candidates by the context's ACTIVE limb
       count (set per shift), NOT by GPU_NLIMBS.  Using GPU_NLIMBS here
       interleaves garbage into the MR inputs and silently drops interior
       primes — the false-gap bug. */
    int rl = gpu_fermat_get_limbs(fermat);
    if (nc > 0) {
        uint64_t *limb_buf = NULL;
        uint8_t *flags = NULL;
        if (rl >= 4) {
            limb_buf = (uint64_t *)malloc(
                (size_t)nc * (size_t)rl * sizeof(uint64_t));
            flags = (uint8_t *)malloc((size_t)nc);
        }
        hid = (uint64_t *)malloc((size_t)nc * sizeof(uint64_t));
        mpz_t c;
        mpz_init(c);
        if (!limb_buf || !flags || !hid) {
            ok = 0;
        } else {
            for (int64_t i = 0; i < nc; i++) {
                mpz_set(c, bsub);
                mpz_add_ui(c, c, cand[i]);
                uint64_t *limbs = limb_buf + (size_t)i * (size_t)rl;
                memset(limbs, 0, (size_t)rl * sizeof(uint64_t));
                size_t cnt = 0;
                mpz_export(limbs, &cnt, -1, sizeof(uint64_t), 0, 0, c);
            }
            if (gpu_fermat_submit(fermat, slot, limb_buf, (size_t)nc) != 0 ||
                gpu_fermat_collect(fermat, slot, flags, (size_t)nc) < 0) {
                ok = 0;
            } else {
                for (int64_t i = 0; i < nc; i++) {
                    if (!flags[i]) continue;
                    mpz_set(c, bsub);
                    mpz_add_ui(c, c, cand[i]);
                    if (mpz_probab_prime_p(c, 15) >= 1) {
                        hid[hid_count++] = cand[i];
                    }
                }
            }
        }
        mpz_clear(c);
        free(limb_buf);
        free(flags);
    }
    if (getenv("GAPDEBUG")) {
        fprintf(stderr,
                "[HIDDBG] gpu resolve interval=%llu abs_delta=%llu tested=%lld found=%u%s\n",
                (unsigned long long)interval,
                (unsigned long long)off_a,
                (long long)nc, hid_count, ok ? "" : " [FALLBACK]");
    }
    free(cand);
    mpz_clear(bsub);

    if (!ok) {
        free(hid);
        return halfclass_resolve_gap_ex(window_base, off_a, off_b,
                                        owned_offset_limit, merit_threshold,
                                        tpl, out, out_count);
    }
    int rc = halfclass_emit_resolved(window_base, off_a, off_b, hid,
                                     (int64_t)hid_count, owned_offset_limit,
                                     merit_threshold, out, out_count);
    free(hid);
    return rc;
}
#endif /* WITH_CUDA */

/* Gap detection + stats + BPSW verification + submission for one CRT window.
   Shared by the per-window and accumulated fused paths, and by the CPU/hybrid
   fallbacks.  candidate_offsets and is_prime must be in matching ascending
   offset order (gap_detection_find walks them in order). */
static void crt_scan_gaps(uint32_t worker_id, uint32_t height, uint32_t nonce,
                          uint64_t generation, double merit_threshold,
                          const struct crt_runtime *rt,
                          mpz_t window_base, mpz_t nadd0,
                          uint64_t back_limit, mpz_t p1, mpz_t p2,
                          mpz_t nadd_full,
                          const uint8_t *is_prime,
                          const uint64_t *candidate_offsets,
                          uint32_t tested_count, uint64_t owned_limit,
                          int half_class,
                          struct gpu_fermat_ctx *resolve_fermat,
                          int resolve_slot)
{
    struct gap_result *gaps = NULL;
    uint32_t gap_count = 0;
    struct gap_scan_stats gap_stats = {0};

    /* HALF_CLASS: the back-lookahead [0, back_limit) is scanned in ALL
       classes, so the list partitions naturally: [0, k) are the unfiltered
       back-region candidates (their primes form the prefix chain directly)
       and [k, tested_count) is the ascending visible list. */
    uint32_t k = 0;
    if (half_class) {
        while (k < tested_count && candidate_offsets[k] < back_limit) k++;
    }
    const uint8_t *pre_ip = is_prime;
    const uint64_t *pre_off = candidate_offsets;
    uint32_t pre_n = k;
    const uint8_t *vis_ip = is_prime + k;
    const uint64_t *vis_off = candidate_offsets + k;
    uint32_t vis_count = tested_count - k;

    int gap_result = gap_detection_find(
        vis_ip, vis_off, vis_count, rt->shift, window_base,
        merit_threshold, owned_limit, &gap_stats, &gaps, &gap_count);

    /* HALF_CLASS two-pass: visible-class gaps are candidates; verify their
       interiors against the hidden classes (the CRT covering template is a
       prefilter on top of the mini-sieve), split into true consecutive-prime
       gaps, and emit the prefix chain from the back-region primes. */
    if (half_class && tested_count >= 1) {
        const struct halfclass_tpl tpl = {
            rt->template, rt->template_words,
            (int64_t)back_limit, rt->window
        };
        struct gap_result *resolved = NULL;
        uint32_t resolved_count = 0;
        uint32_t dropped = 0;

        if (pre_n > 0) {
            /* (1) Gaps among the back-region primes: the back region is
               scanned in all classes, so consecutive back primes are true
               consecutive primes. */
            uint64_t *chain = (uint64_t *)malloc(
                (size_t)(pre_n + 1) * sizeof(uint64_t));
            if (!chain) {
                dropped++;
            } else {
                uint32_t cl = 0;
                for (uint32_t i = 0; i < pre_n; i++) {
                    if (pre_ip[i]) {
                        chain[cl++] = pre_off[i];
                    }
                }
                if (cl > 0) {
                    struct gap_result *pre = NULL;
                    uint32_t pre_count = halfclass_emit_chain(
                        window_base, chain, cl, owned_limit,
                        merit_threshold, &pre);
                    resolved = pre;
                    resolved_count = pre_count;
                }

                /* (2) The terminal pair (last back prime, first visible
                   prime) crosses the covered region: hidden primes may sit
                   inside, so it must be resolved like a visible gap — but
                   ONLY when its own merit can qualify (otherwise every
                   sub-gap is below threshold too, and re-verifying the whole
                   covered region on the host every window would be
                   catastrophically expensive). */
                uint32_t first_vis = UINT32_MAX;
                for (uint32_t i = 0; i < vis_count; i++) {
                    if (vis_ip[i]) {
                        first_vis = i;
                        break;
                    }
                }
                if (cl > 0 && first_vis != UINT32_MAX) {
                    mpz_t gv;
                    mpz_init(gv);
                    mpz_set(gv, window_base);
                    mpz_add_ui(gv, gv, chain[cl - 1]);
                    double term_merit = gap_detection_compute_merit(
                        (uint32_t)(vis_off[first_vis] - chain[cl - 1]), gv);
                    mpz_clear(gv);
                    if (term_merit >= merit_threshold) {
                        struct gap_result *sub = NULL;
                        uint32_t sub_count = 0;
                        int sub_rc;
#ifdef WITH_CUDA
                        if (resolve_fermat) {
                            sub_rc = crt_gpu_resolve_gap_ex(
                                resolve_fermat, resolve_slot, window_base,
                                chain[cl - 1], vis_off[first_vis],
                                owned_limit, merit_threshold, &tpl,
                                &sub, &sub_count);
                        } else
#endif
                        {
                            sub_rc = halfclass_resolve_gap_ex(
                                window_base, chain[cl - 1],
                                vis_off[first_vis],
                                owned_limit, merit_threshold, &tpl,
                                &sub, &sub_count);
                        }
                        if (!sub_rc) {
                            dropped++;
                        } else if (sub_count > 0) {
                            struct gap_result *grown =
                                (struct gap_result *)realloc(
                                    resolved,
                                    (size_t)(resolved_count + sub_count) *
                                        sizeof(*grown));
                            if (!grown) {
                                dropped++;
                                free(sub);
                            } else {
                                resolved = grown;
                                memcpy(resolved + resolved_count, sub,
                                       (size_t)sub_count * sizeof(*sub));
                                resolved_count += sub_count;
                                free(sub);
                            }
                        } else {
                            free(sub);
                        }
                    }
                }
                free(chain);
            }
        } else {
            /* No back-region candidates (astronomically rare): host
               verification of [0, first visible prime). */
            uint32_t first_vis = UINT32_MAX;
            for (uint32_t i = 0; i < vis_count; i++) {
                if (vis_ip[i]) {
                    first_vis = i;
                    break;
                }
            }
            if (first_vis != UINT32_MAX && vis_off[first_vis] > 0) {
                struct gap_result *pre = NULL;
                uint32_t pre_count = 0;
                if (!halfclass_verify_prefix_ex(window_base,
                                                vis_off[first_vis], 1,
                                                owned_limit, merit_threshold,
                                                &tpl, &pre, &pre_count)) {
                    dropped++;
                } else {
                    resolved = pre;
                    resolved_count = pre_count;
                }
            }
        }

        for (uint32_t i = 0; i < gap_count; i++) {
            struct gap_result *sub = NULL;
            uint32_t sub_count = 0;
            int sub_rc;
#ifdef WITH_CUDA
            if (resolve_fermat) {
                sub_rc = crt_gpu_resolve_gap_ex(
                    resolve_fermat, resolve_slot, window_base,
                    gaps[i].offset_p1, gaps[i].offset_p2, owned_limit,
                    merit_threshold, &tpl, &sub, &sub_count);
            } else
#endif
            {
                sub_rc = halfclass_resolve_gap_ex(
                    window_base, gaps[i].offset_p1,
                    gaps[i].offset_p2, owned_limit,
                    merit_threshold, &tpl,
                    &sub, &sub_count);
            }
            if (!sub_rc) {
                dropped++;
                continue;
            }
            if (sub_count == 0) {
                free(sub);
                continue;
            }
            struct gap_result *grown = (struct gap_result *)realloc(
                resolved,
                (size_t)(resolved_count + sub_count) * sizeof(*grown));
            if (!grown) {
                dropped++;
                free(sub);
                continue;
            }
            resolved = grown;
            memcpy(resolved + resolved_count, sub,
                   (size_t)sub_count * sizeof(*sub));
            resolved_count += sub_count;
            free(sub);
        }
        gap_detection_free_results(gaps);
        gaps = resolved;
        gap_count = resolved_count;
        if (dropped) {
            fprintf(stderr,
                    "[Worker %u] CRT HALF_CLASS verification: %u candidate(s) dropped (allocation failure)\n",
                    worker_id, dropped);
        }
        /* True maxima from the resolved gaps (visible maxima are sums of
           several true gaps and would print a bogus record-looking merit). */
        uint32_t true_max_len = 0;
        double true_max_merit = 0.0;
        for (uint32_t i = 0; i < gap_count; i++) {
            if (gaps[i].gap_length > true_max_len) {
                true_max_len = gaps[i].gap_length;
            }
            if (gaps[i].merit > true_max_merit) {
                true_max_merit = gaps[i].merit;
            }
        }
        gap_stats.max_gap_length = true_max_len;
        gap_stats.max_merit = true_max_merit;
    }

    atomic_fetch_add(&g_worker_stats[worker_id].euler_pairs, gap_stats.euler_pairs);
    worker_atomic_max(&g_worker_stats[worker_id].max_gap_length, gap_stats.max_gap_length);
    worker_atomic_max(&g_worker_stats[worker_id].max_merit_scaled,
                      worker_scale_merit(gap_stats.max_merit));

    if (gap_result && gaps && gap_count > 0) {
        static int gapdebug = -1;
        if (gapdebug < 0)
            gapdebug = getenv("GAPDEBUG") != NULL;
        if (gapdebug) {
            uint32_t wbm60 = (uint32_t)mpz_fdiv_ui(window_base, 60);
            fprintf(stderr,
                    "[GAPDEBUG] window_base mod60=%u pre_n=%u vis_count=%u\n",
                    wbm60, pre_n, vis_count);
            for (uint32_t i = 0; i < gap_count; i++) {
                uint64_t a = gaps[i].offset_p1, b = gaps[i].offset_p2;
                uint32_t ca = (uint32_t)((wbm60 + a % 60ULL) % 60ULL);
                uint32_t cb = (uint32_t)((wbm60 + b % 60ULL) % 60ULL);
                uint32_t npre = 0, nvis = 0;
                for (uint32_t q = 0; q < pre_n; q++)
                    if (pre_off[q] > a && pre_off[q] < b) npre++;
                for (uint32_t q = 0; q < vis_count; q++)
                    if (vis_off[q] > a && vis_off[q] < b) nvis++;
                char buf[512];
                size_t bl = 0;
                buf[0] = 0;
                uint32_t shown = 0;
                for (uint32_t q = 0; q < vis_count && shown < 12; q++) {
                    if (vis_off[q] > a && vis_off[q] < b) {
                        bl += (size_t)snprintf(buf + bl, sizeof(buf) - bl,
                                               "%llu:%d ",
                                               (unsigned long long)vis_off[q],
                                               vis_ip[q]);
                        shown++;
                    }
                }
                fprintf(stderr,
                        "[GAPDEBUG] gap[%u] off=%llu(cl=%u)..%llu(cl=%u) len=%llu merit=%.2f "
                        "pre_in=%u vis_in=%u vis_inside=[%s]\n",
                        i, (unsigned long long)a, ca, (unsigned long long)b, cb,
                        (unsigned long long)(b - a), gaps[i].merit,
                        npre, nvis, buf);
            }
        }
        atomic_fetch_add(&g_worker_stats[worker_id].merit_candidates, gap_count);
        for (uint32_t i = 0; i < gap_count; i++) {
            mpz_set(p1, window_base);
            mpz_add_ui(p1, p1, (unsigned long)gaps[i].offset_p1);
            mpz_set(p2, window_base);
            mpz_add_ui(p2, p2, (unsigned long)gaps[i].offset_p2);

            atomic_fetch_add(&g_worker_stats[worker_id].bpsw_attempts, 1);
            if (baillie_psw_verify_gap_boundaries(p1, p2)) {
                atomic_fetch_add(&g_worker_stats[worker_id].gaps_found, 1);

                /* Full-width adder = nadd0 - adj + position. */
                mpz_set(nadd_full, nadd0);
                if (rt->adj) mpz_sub_ui(nadd_full, nadd_full, rt->adj);
                mpz_add_ui(nadd_full, nadd_full, (unsigned long)gaps[i].offset_p1);
                mpz_sub_ui(nadd_full, nadd_full, (unsigned long)back_limit);
                char *nadd_dec = mpz_get_str(NULL, 10, nadd_full);

                if (atomic_load_explicit(&g_submission_enabled, memory_order_acquire)) {
                    struct gap_queue_entry entry = {0};
                    entry.height = height;
                    entry.shift = rt->shift;
                    entry.gap_length = gaps[i].gap_length;
                    entry.merit = gaps[i].merit;
                    entry.generation = generation;
                    entry.header_nonce = nonce;
                    entry.nadd = mpz_get_ui(nadd_full); /* legacy */
                    size_t nb = 0;
                    mpz_export(entry.nadd_bytes, &nb, -1, 1, 0, 0, nadd_full);
                    if (nb == 0) {
                        entry.nadd_bytes[0] = 0;
                        nb = 1;
                    }
                    if (nb > sizeof(entry.nadd_bytes)) {
                        nb = sizeof(entry.nadd_bytes);
                    }
                    entry.nadd_len = (uint32_t)nb;
                    int queued = worker_queue_push(&entry);
                    fprintf(stderr,
                            "[Worker %u] CRT BPSW candidate: height=%u nonce=%u nAdd=%s gap=%u merit=%.2f (%s)\n",
                            worker_id, height, nonce,
                            nadd_dec ? nadd_dec : "?",
                            gaps[i].gap_length, gaps[i].merit,
                            queued ? "queued for submission"
                                   : "submission queue full, dropped");
                    record_log_write_big(height, rt->shift, nonce,
                                         nadd_dec, p1,
                                         gaps[i].gap_length,
                                         gaps[i].merit,
                                         queued ? "queued" : "queue-full");
                } else {
                    fprintf(stderr,
                            "[Worker %u] CRT BPSW candidate: height=%u nonce=%u nAdd=%s gap=%u merit=%.2f (dry-run)\n",
                            worker_id, height, nonce,
                            nadd_dec ? nadd_dec : "?",
                            gaps[i].gap_length, gaps[i].merit);
                    record_log_write_big(height, rt->shift, nonce,
                                         nadd_dec, p1,
                                         gaps[i].gap_length,
                                         gaps[i].merit, "dry-run");
                }
                if (nadd_dec) free(nadd_dec);
            }
        }
    }

    gap_detection_free_results(gaps);
}

/* CRT-mode worker: many aligned windows in parallel (one per header nonce).
 *
 * The covering file's offsets o_i make the aligned base satisfy
 * base ≡ -(o_i + adj) (mod p_i), so the template marks the whole
 * [1, gap_target) interior composite.  Mining at a lower live merit than the
 * file's design merit scans a window 2 * needed_gap << gap_target, which sits
 * almost entirely inside the covered interior, so the window is prime-poor
 * and its prime-gap tail is far heavier than the Cramer e^-m model.
 *
 * Each worker hashes an independent strided nonce (pass_nonce+1+worker_id,
 * stride nthreads) — the cpugapminer CRT model — so N workers scan N distinct
 * headers concurrently and overlap their GPU batches with each other's CPU
 * sieves. */
void *worker_thread_run_crt(void *arg) {
    struct worker_config *config = (struct worker_config *)arg;
    if (!config || !config->crt_rt) return NULL;
    struct crt_runtime *rt = config->crt_rt;
    uint32_t worker_id = config->worker_id;

    mpz_t base, nadd0, candidate, p1, p2, nadd_full, window_base;
    mpz_init(base);
    mpz_init(nadd0);
    mpz_init(candidate);
    mpz_init(p1);
    mpz_init(p2);
    mpz_init(nadd_full);
    mpz_init(window_base);
    struct euler_context euler_context;
    euler_context_init(&euler_context);

    /* Reset this worker's counters before it begins scanning. */
    atomic_store(&g_worker_stats[worker_id].nonces_processed, 0);
    atomic_store(&g_worker_stats[worker_id].candidates_generated, 0);
    atomic_store(&g_worker_stats[worker_id].candidates_tested, 0);
    atomic_store(&g_worker_stats[worker_id].euler_passes, 0);
    atomic_store(&g_worker_stats[worker_id].euler_pairs, 0);
    atomic_store(&g_worker_stats[worker_id].merit_candidates, 0);
    atomic_store(&g_worker_stats[worker_id].bpsw_attempts, 0);
    atomic_store(&g_worker_stats[worker_id].gaps_found, 0);
    atomic_store(&g_worker_stats[worker_id].gaps_submitted, 0);
    atomic_store(&g_worker_stats[worker_id].max_gap_length, 0);
    atomic_store(&g_worker_stats[worker_id].max_merit_scaled, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_euler_skipped, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_sieve_calls, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_sieve_windows, 0);
    atomic_store(&g_worker_stats[worker_id].smart_tail_skipped, 0);
    atomic_store(&g_worker_stats[worker_id].gpu_accounted_us, 0);

    double logbase = (256.0 + (double)rt->shift) * log(2.0);

    int half_class = 0;
    if (worker_env_enabled(getenv("HALF_CLASS")) ||
        worker_env_enabled(getenv("QUARTER_CLASS"))) {
        half_class = 1;
        printf("[Worker %u] HALF_CLASS scan: visible classes "
               "{1,7,11,13,17,19,23,29} mod 60; hidden classes verified via "
               "CRT template on demand\n", worker_id);
    }
    int quarter_mode = worker_env_enabled(getenv("QUARTER_CLASS"));

    printf("[Worker %u] CRT mode: n_primes=%u shift=%u gap_target=%llu "
           "design_merit=%.2f logbase=%.1f\n",
           worker_id, rt->n_primes, rt->shift,
           (unsigned long long)rt->gap_target, rt->merit, logbase);

    /* Backward look-ahead includes the prime that precedes the covered
       interior: it starts the large gap the covering leaves open. */
    uint64_t back_limit = (uint64_t)ceil(2.0 * logbase);
    if (back_limit < 4096) back_limit = 4096;
    if (back_limit & 1ULL) back_limit++;   /* keep window_base even */

    uint64_t max_interval = rt->window + back_limit + 2;
    uint64_t sieve_limit = config->sieve_limit;
    if (sieve_limit < 2) sieve_limit = 2;

    struct sieve_core sieve = {0};
    if (!sieve_core_init_window(&sieve, max_interval, sieve_limit)) {
        fprintf(stderr, "[Worker %u] Failed to initialize CRT sieve\n", worker_id);
        mpz_clears(base, nadd0, candidate, p1, p2, nadd_full, window_base, NULL);
        euler_context_clear(&euler_context);
        return NULL;
    }
    uint8_t *is_prime = (uint8_t *)malloc(sieve.candidate_capacity * sizeof(*is_prime));
    if (!is_prime) {
        sieve_core_free(&sieve);
        mpz_clears(base, nadd0, candidate, p1, p2, nadd_full, window_base, NULL);
        euler_context_clear(&euler_context);
        return NULL;
    }

#ifdef WITH_CUDA
    struct gpu_adapter *gpu = NULL;
    int gpu_limbs = 0;
    uint64_t *gpu_base_limbs = NULL;
    uint64_t *gpu_packed = NULL;
    struct gpu_sieve_ctx *gpu_sieve = NULL;
    size_t gpu_sieve_split_index = 0;
    int gpu_sieve_enabled = 0;
    int gpu_sieve_failure_reported = 0;
    uint64_t *gpu_host_bitmap = NULL;
    if (atomic_load_explicit(&g_gpu_fermat_enabled, memory_order_acquire)) {
        gpu = gpu_adapter_init((int)config->gpu_device);
        if (gpu) {
            gpu_adapter_set_candidate_bits(gpu, 256U + rt->shift);
            gpu_limbs = gpu_adapter_get_limbs(gpu);
            gpu_base_limbs = (uint64_t *)calloc((size_t)gpu_limbs,
                                                sizeof(uint64_t));
            if (gpu_limbs > 0 && gpu_base_limbs &&
                (size_t)sieve.candidate_capacity <=
                    SIZE_MAX / ((size_t)gpu_limbs * sizeof(uint64_t))) {
                gpu_packed = (uint64_t *)malloc((size_t)sieve.candidate_capacity *
                                                (size_t)gpu_limbs *
                                                sizeof(uint64_t));
            }
            if (!gpu_base_limbs || !gpu_packed) {
                gpu_adapter_free(gpu);
                gpu = NULL;
                free(gpu_base_limbs);
                gpu_base_limbs = NULL;
                free(gpu_packed);
                gpu_packed = NULL;
                fprintf(stderr,
                        "[Worker %u] CRT GPU Fermat buffers failed; using CPU Euler filter\n",
                        worker_id);
            } else {
                fprintf(stderr,
                        "[Worker %u] CRT GPU Fermat enabled (device=%d, %d limbs, %u-bit candidates)\n",
                        worker_id, config->gpu_device, gpu_limbs,
                        256U + rt->shift);
            }
        } else {
            fprintf(stderr,
                    "[Worker %u] CRT GPU Fermat init failed for device %u; using CPU Euler filter\n",
                    worker_id, config->gpu_device);
        }
    }

    /* GPU bitmap sieve (residues computed on-device): CPU presieves primes
       < split_index, the GPU marks [split_index, sieve_limit) from the raw
       base.  Fail-closed: any CUDA error falls back to the CPU sieve.
       FUSED_GPU implies GPU_SIEVE and additionally needs the FULL prime table
       uploaded on-device (max_primes = small_primes_count). */
    const char *gpu_sieve_env = getenv("GPU_SIEVE");
    const char *fused_env = getenv("FUSED_GPU");
    if (gpu && (worker_env_enabled(gpu_sieve_env) || worker_env_enabled(fused_env))) {
        gpu_sieve_split_index = worker_gpu_sieve_split_index(&sieve);
        if (gpu_sieve_split_index < sieve.small_primes_count) {
            uint64_t max_odd_interval = (max_interval + 1U) >> 1;
            size_t gpu_sieve_max_primes = worker_env_enabled(fused_env)
                ? sieve.small_primes_count
                : sieve.small_primes_count - gpu_sieve_split_index;
            gpu_sieve = gpu_sieve_init((int)config->gpu_device,
                                       gpu_sieve_max_primes,
                                       max_odd_interval);
            if (gpu_sieve) {
                uint64_t max_odd_words = (max_odd_interval + 63U) >> 6;
                gpu_host_bitmap = (uint64_t *)calloc(max_odd_words,
                                                     sizeof(uint64_t));
                if (gpu_host_bitmap) {
                    gpu_sieve_enabled = 1;
                    fprintf(stderr,
                            "[Worker %u] CRT GPU bitmap sieve enabled (split_index=%zu, gpu_primes=%zu, device=%s)\n",
                            worker_id, gpu_sieve_split_index,
                            sieve.small_primes_count - gpu_sieve_split_index,
                            gpu_sieve_device_name(gpu_sieve));
                } else {
                    gpu_sieve_destroy(gpu_sieve);
                    gpu_sieve = NULL;
                    fprintf(stderr,
                            "[Worker %u] CRT GPU bitmap sieve alloc failed; using CPU-only sieve\n",
                            worker_id);
                }
            }
        }
    }

    /* Fused GPU pipeline (async double-buffered): device-resident sieve +
       extract + MR with two windows in flight. */
    int fused_enabled = 0;
    uint64_t fused_seq = 0;
    struct gpu_fermat_ctx *fused_fermat = NULL;
    uint64_t *fused_slot_offsets = NULL;
    uint64_t *fused_sorted_offsets = NULL;
    struct fused_flight fused_fl[2];
    uint8_t *fused_is_prime[2] = { NULL, NULL };
    uint64_t *fused_offsets[2] = { NULL, NULL };
    memset(fused_fl, 0, sizeof(fused_fl));
    for (int f = 0; f < 2; f++) {
        for (int w = 0; w < FUSED_MR_BATCH_MAX; w++) {
            mpz_init(fused_fl[f].wins[w].window_base);
            mpz_init(fused_fl[f].wins[w].nadd0);
        }
    }
    struct fused_pending_state fused_pending;
    memset(&fused_pending, 0, sizeof(fused_pending));
    int fused_pair_enabled =
        worker_env_enabled(getenv("GPU_SIEVE_PAIR"));
    int fused_mr_batch = 8;
    {
        const char *mrb = getenv("GPU_MR_BATCH");
        if (mrb && *mrb) {
            long v = strtol(mrb, NULL, 10);
            if (v >= 1 && v <= FUSED_MR_BATCH_MAX) fused_mr_batch = (int)v;
        }
    }
    if (gpu && gpu_sieve && worker_env_enabled(fused_env)) {
        fused_fermat = gpu_adapter_get_fermat_ctx(gpu);
        if (fused_fermat) {
            size_t cap = (size_t)sieve.candidate_capacity *
                         (size_t)fused_mr_batch;
            fused_slot_offsets = (uint64_t *)malloc(cap * sizeof(uint64_t));
            fused_sorted_offsets = (uint64_t *)malloc(cap * sizeof(uint64_t));
            fused_is_prime[0] = (uint8_t *)malloc(cap * sizeof(uint8_t));
            fused_is_prime[1] = (uint8_t *)malloc(cap * sizeof(uint8_t));
            if (fused_slot_offsets && fused_sorted_offsets &&
                fused_is_prime[0] && fused_is_prime[1]) {
                fused_offsets[0] = fused_slot_offsets;
                fused_offsets[1] = fused_sorted_offsets;
                fused_enabled = 1;
                gpu_sieve_set_extract_accum(gpu_sieve,
                                            (uint32_t)fused_mr_batch);
                fprintf(stderr,
                        "[Worker %u] Fused GPU pipeline enabled (FUSED_GPU=1): "
                        "async double-buffered, ordered extraction%s%s\n",
                        worker_id,
                        fused_mr_batch > 1 ? " (GPU_MR_BATCH: K-window MR accumulation)" : "",
                        fused_pair_enabled
                            ? " (GPU_SIEVE_PAIR=1: 2-window pair-batched mark)"
                            : "");
            } else {
                free(fused_slot_offsets);
                free(fused_sorted_offsets);
                free(fused_is_prime[0]);
                free(fused_is_prime[1]);
                fused_slot_offsets = NULL;
                fused_sorted_offsets = NULL;
                fused_is_prime[0] = NULL;
                fused_is_prime[1] = NULL;
                fprintf(stderr,
                        "[Worker %u] Fused GPU pipeline alloc failed; using hybrid path\n",
                        worker_id);
            }
        }
    }
#endif

    uint32_t nthreads = config->nthreads > 0 ? config->nthreads : 1;

    while (!g_stop_requested) {
        uint64_t generation;
        uint32_t height;
        double merit_threshold;
        uint8_t hdr80[80];
        uint32_t pass_nonce;
        if (!worker_snapshot_crt_work(config, &generation, &height,
                                      &merit_threshold, hdr80, &pass_nonce)) {
            worker_wait_for_work();
            continue;
        }

        /* Strided nonce space (the cpugapminer CRT model): this worker hashes
           pass_nonce+1+worker_id, then advances by nthreads each step. Every
           nonce is an independent header — its own h256, its own CRT
           alignment, its own scan window — so N workers scan N headers in
           parallel with no shared state and no contention. */
        uint32_t nonce = pass_nonce + 1U + worker_id;

        for (;;) {
            if (g_stop_requested) {
#ifdef WITH_CUDA
                fused_pending_clear(&fused_pending);
                if (fused_enabled && fused_fermat && gpu_sieve) {
                    uint64_t b0 = fused_seq / (uint64_t)fused_mr_batch;
                    fused_flush_partial(gpu_sieve, fused_fermat,
                                        &fused_fl[b0 & 1ULL],
                                        (int)(b0 & 1ULL));
                }
#endif
                break;
            }
            if (atomic_load_explicit(config->work_generation,
                                     memory_order_acquire) != generation) {
#ifdef WITH_CUDA
                fused_pending_clear(&fused_pending);
                if (fused_enabled && fused_fermat && gpu_sieve) {
                    uint64_t b0 = fused_seq / (uint64_t)fused_mr_batch;
                    fused_flush_partial(gpu_sieve, fused_fermat,
                                        &fused_fl[b0 & 1ULL],
                                        (int)(b0 & 1ULL));
                    /* Align to the next batch boundary so the new header's
                       first window starts a fresh flight. */
                    fused_seq = ((fused_seq + (uint64_t)fused_mr_batch - 1U) /
                                 (uint64_t)fused_mr_batch) *
                                (uint64_t)fused_mr_batch;
                }
#endif
                break;   /* new block: re-snapshot */
            }

            uint8_t h256[32];
            if (gapcoin_gbt_hash_nonce(hdr80, nonce, h256) == 0) {
                /* Scan window = 2 * needed_gap at the live (mining) merit. */
                uint64_t needed_gap = (uint64_t)ceil(merit_threshold * logbase);
                if (needed_gap < 2) needed_gap = 2;
                uint64_t scan_window = 2 * needed_gap;
                if (scan_window > rt->window) scan_window = rt->window;
                if (scan_window < 2) scan_window = 2;

                /* Solve the CRT alignment for this header. */
                if (crt_runtime_align(nadd0, h256, rt->shift, rt) == 0) {
                    /* base = (h256 << shift) + nadd0 - adj (even-aligned). */
                    worker_set_base(base, h256, rt->shift, 0);
                    mpz_add(base, base, nadd0);
                    if (rt->adj) mpz_sub_ui(base, base, rt->adj);

                    /* Sieve [base - back_limit, base + scan_window) so the
                       preceding prime lands inside the window and gap
                       detection sees it. */
                    mpz_sub_ui(window_base, base, (unsigned long)back_limit);
                    uint32_t base_mod60 =
                        half_class ? (uint32_t)mpz_fdiv_ui(window_base, 60)
                                   : 0U;
                    uint64_t interval = back_limit + scan_window;
                    sieve.interval_size = interval;
                    sieve.bitmap_words = (interval + 63U) / 64U;

                    uint64_t *candidate_offsets = NULL;
                    uint32_t candidate_count = 0;
                    int sieve_ok = 0;
                    int fused_async = 0;

#ifdef WITH_CUDA
                    if (fused_enabled && fused_fermat && gpu_base_limbs) {
                        /* Async fused pipeline with K-window MR batch
                           accumulation: collect the flight from two batches
                           ago, then mark + extract the head of THIS window
                           into the batch's accumulation buffer; submit ONE
                           MR per K windows. */
                        uint64_t w_idx = fused_seq;
                        uint64_t batch = w_idx / (uint64_t)fused_mr_batch;
                        int w_in = (int)(w_idx % (uint64_t)fused_mr_batch);
                        int slot = (int)(batch & 1ULL);
                        struct fused_flight *fl = &fused_fl[slot];
                        int f = (int)(w_idx & 1ULL);   /* bitmap parity */

                        if (w_in == 0 && fl->active) {
                            if (!crt_fused_collect_batch(
                                    gpu_sieve, fused_fermat, worker_id, rt,
                                    fl, fused_offsets[slot],
                                    fused_is_prime[slot],
                                    p1, p2, nadd_full,
                                    sieve.small_primes, sieve.inv_p,
                                    sieve.small_primes_count) &&
                                !gpu_sieve_failure_reported) {
                                gpu_sieve_failure_reported = 1;
                                gpu_sieve_enabled = 0;
                                fused_enabled = 0;
                                fprintf(stderr,
                                        "[Worker %u] CRT fused GPU pipeline failed; falling back to CPU sieve\n",
                                        worker_id);
                            }
                            fl->active = 0;
                            fl->n_windows = 0;
                            fl->total_count = 0;
                        }

                        if (fused_enabled) {
                            size_t exported = 0;
                            memset(gpu_base_limbs, 0,
                                   (size_t)gpu_limbs * sizeof(uint64_t));
                            mpz_export(gpu_base_limbs, &exported, -1,
                                       sizeof(uint64_t), 0, 0, window_base);
                            uint64_t head_end_v =
                                back_limit + rt->gap_target;
                            /* QUARTER_CLASS: visible primes are ~2x rarer,
                               so the closing range [need_off, head_end) must
                               be ~2.5 visible spacings longer or the tail
                               re-mark would run on ~1/3 of windows. */
                            if (quarter_mode) {
                                head_end_v += (uint64_t)(12.0 * logbase);
                            }
                            uint64_t need_off_v =
                                back_limit + needed_gap;
                            int split_v =
                                (need_off_v < head_end_v) &&
                                (head_end_v - need_off_v >=
                                 (uint64_t)(2.0 * logbase));
                            uint64_t first_odd_offset =
                                (gpu_base_limbs[0] & 1ULL) ? 0U : 1U;
                            uint64_t odd_interval_size =
                                (interval > first_odd_offset)
                                    ? (interval - first_odd_offset + 1U) >> 1
                                    : 0;

                            /* Pair-batched mark (launches halved): when the
                               previous window is held, re-mark it together
                               with THIS window in one kernel — each into its
                               own ping-pong bitmap.  Re-marking the previous
                               window is content-identical to its first mark
                               (same base, same primes), so the smart-scan
                               tail extract at its collect still reads an
                               intact bitmap.  Submission cadence stays
                               per-window and unchanged: extract + MR submit
                               for THIS window only, right after its mark. */
                            int mark_ok = 0;
                            if (exported <= (size_t)gpu_limbs &&
                                odd_interval_size > 0) {
                                if (fused_pending.valid && fused_pair_enabled) {
                                    uint64_t pair_limbs[2][GPU_NLIMBS];
                                    const uint64_t *p0, *p1;
                                    if (fused_pending.parity == 0) {
                                        p0 = fused_pending.base_limbs;
                                        p1 = gpu_base_limbs;
                                    } else {
                                        p0 = gpu_base_limbs;
                                        p1 = fused_pending.base_limbs;
                                    }
                                    memcpy(pair_limbs[0], p0,
                                           (size_t)gpu_limbs * sizeof(uint64_t));
                                    memcpy(pair_limbs[1], p1,
                                           (size_t)gpu_limbs * sizeof(uint64_t));
                                    mark_ok = gpu_sieve_mark_batch_from_bases(
                                        gpu_sieve, odd_interval_size,
                                        &pair_limbs[0][0], gpu_limbs,
                                        sieve.small_primes, sieve.inv_p,
                                        sieve.small_primes_count);
                                    if (mark_ok) {
                                        atomic_fetch_add(
                                            &g_worker_stats[worker_id].gpu_sieve_calls, 1);
                                        atomic_fetch_add(
                                            &g_worker_stats[worker_id].gpu_sieve_windows, 2);
                                    }
                                } else {
                                    mark_ok = gpu_sieve_mark_from_base(
                                        gpu_sieve, odd_interval_size,
                                        first_odd_offset, gpu_base_limbs,
                                        gpu_limbs, f,
                                        sieve.small_primes, sieve.inv_p,
                                        sieve.small_primes_count, NULL, 0);
                                    if (mark_ok) {
                                        atomic_fetch_add(
                                            &g_worker_stats[worker_id].gpu_sieve_calls, 1);
                                        atomic_fetch_add(
                                            &g_worker_stats[worker_id].gpu_sieve_windows, 1);
                                    }
                                }
                            }

                            uint32_t head_count = 0;
                            if (mark_ok) {
                                if (w_in == 0) {
                                    fl->height = height;
                                    fl->generation = generation;
                                    fl->merit_threshold = merit_threshold;
                                    fl->needed_gap = needed_gap;
                                    fl->back_limit = back_limit;
                                    fl->head_end = head_end_v;
                                    fl->need_off = need_off_v;
                                    fl->interval = interval;
                                    fl->half_class = half_class;
                                    fl->limbs = gpu_limbs;
                                    fl->buf = slot;
                                    fl->slot = slot;
                                    fl->base_window = w_idx;
                                    fl->n_windows = 0;
                                    fl->total_count = 0;
                                }
                            }
                            if (mark_ok &&
                                crt_fused_append_window(
                                    gpu_sieve, f, slot,
                                    split_v, gpu_base_limbs, gpu_limbs,
                                    interval, head_end_v,
                                    fused_offsets[slot] + fl->total_count,
                                    &head_count,
                                    half_class, base_mod60, back_limit,
                                    fl->total_count)) {
                                struct fused_flight_window *wv =
                                    &fl->wins[w_in];
                                wv->nonce = nonce;
                                wv->count = head_count;
                                wv->split = split_v;
                                wv->base_mod60 = base_mod60;
                                memcpy(wv->base_limbs, gpu_base_limbs,
                                       (size_t)gpu_limbs * sizeof(uint64_t));
                                mpz_set(wv->window_base, window_base);
                                mpz_set(wv->nadd0, nadd0);
                                fl->total_count += head_count;
                                fl->n_windows++;
                                atomic_fetch_add(
                                    &g_worker_stats[worker_id].nonces_processed, 1);
                                sieve_ok = 1;
                                fused_async = 1;
                                fused_seq++;

                                /* Submit ONE accumulated MR batch per K
                                   windows. */
                                if (w_in == fused_mr_batch - 1) {
                                    uint64_t *d_batch =
                                        gpu_sieve_candidate_buffer(gpu_sieve,
                                                                   slot);
                                    if (fl->total_count > 0 &&
                                        gpu_fermat_submit_device(
                                            fused_fermat, slot, d_batch,
                                            fl->total_count) < 0) {
                                        fl->n_windows = 0;
                                        fl->total_count = 0;
                                        if (!gpu_sieve_failure_reported) {
                                            gpu_sieve_failure_reported = 1;
                                            gpu_sieve_enabled = 0;
                                            fused_enabled = 0;
                                            fprintf(stderr,
                                                    "[Worker %u] CRT fused GPU pipeline failed; falling back to CPU sieve\n",
                                                    worker_id);
                                        }
                                    } else {
                                        fl->active = 1;
                                    }
                                    atomic_store(
                                        &g_worker_stats[worker_id].gpu_accounted_us,
                                        gpu_fermat_accounted_us(fused_fermat));
                                }

                                /* Hold this window for the next pair mark
                                   (mark only — it was already appended). */
                                if (fused_pair_enabled) {
                                    fused_pending.valid = 1;
                                    fused_pending.parity = f;
                                    fused_pending.interval = interval;
                                    memcpy(fused_pending.base_limbs,
                                           gpu_base_limbs,
                                           (size_t)gpu_limbs * sizeof(uint64_t));
                                } else {
                                    fused_pending_clear(&fused_pending);
                                }
                            } else if (!gpu_sieve_failure_reported) {
                                fused_pending_clear(&fused_pending);
                                gpu_sieve_failure_reported = 1;
                                gpu_sieve_enabled = 0;
                                fused_enabled = 0;
                                fprintf(stderr,
                                        "[Worker %u] CRT fused GPU pipeline failed; falling back to CPU sieve\n",
                                        worker_id);
                            } else {
                                fused_pending_clear(&fused_pending);
                            }
                        }
                    }

                    if (!sieve_ok && gpu_sieve_enabled && gpu_sieve &&
                        gpu_base_limbs && gpu_host_bitmap) {
                        /* Export the window base limbs for the on-device
                           residue kernel. */
                        size_t exported = 0;
                        memset(gpu_base_limbs, 0,
                               (size_t)gpu_limbs * sizeof(uint64_t));
                        mpz_export(gpu_base_limbs, &exported, -1,
                                   sizeof(uint64_t), 0, 0, window_base);
                        if (exported <= (size_t)gpu_limbs) {
                            uint64_t first_odd_offset =
                                (gpu_base_limbs[0] & 1ULL) ? 0U : 1U;
                            if (interval > first_odd_offset) {
                                uint64_t odd_interval_size =
                                    (interval - first_odd_offset + 1U) >> 1;
                                uint64_t odd_words =
                                    (odd_interval_size + 63U) >> 6;
                                if (odd_words > 0 &&
                                    gpu_sieve_mark_from_base(
                                        gpu_sieve, odd_interval_size,
                                        first_odd_offset, gpu_base_limbs,
                                        gpu_limbs, 0,
                                        sieve.small_primes +
                                            gpu_sieve_split_index,
                                        sieve.inv_p + gpu_sieve_split_index,
                                        sieve.small_primes_count -
                                            gpu_sieve_split_index,
                                        gpu_host_bitmap, odd_words) &&
                                    sieve_core_prepare_base_mod_p_range(
                                        &sieve, window_base, 0,
                                        gpu_sieve_split_index)) {
                                    atomic_fetch_add(
                                        &g_worker_stats[worker_id].gpu_sieve_calls, 1);
                                    atomic_fetch_add(
                                        &g_worker_stats[worker_id].gpu_sieve_windows, 1);
                                    sieve_ok = sieve_core_run_from_cached_base_hybrid(
                                        &sieve, 0, gpu_host_bitmap, odd_words,
                                        gpu_sieve_split_index,
                                        &candidate_offsets, &candidate_count);
                                }
                            }
                        }
                        if (!sieve_ok && !gpu_sieve_failure_reported) {
                            gpu_sieve_failure_reported = 1;
                            gpu_sieve_enabled = 0;
                            fprintf(stderr,
                                    "[Worker %u] CRT GPU sieve failed; falling back to CPU sieve\n",
                                    worker_id);
                        }
                    }
#endif

                    if (!sieve_ok) {
                        sieve_ok = sieve_core_run(&sieve, window_base,
                                                  &candidate_offsets,
                                                  &candidate_count);
                    }

                    if (sieve_ok && half_class) {
                        /* Only the covered region is class-filtered; the
                           back-lookahead [0, back_limit) is scanned in all
                           classes, so its primes are already in the list
                           and the prefix chain is built from the scan
                           itself (zero extra verification cost). */
                        candidate_count = halfclass_filter_offsets_region(
                            base_mod60, candidate_offsets, candidate_count,
                            back_limit);
                    }

                    if (sieve_ok && !fused_async) {
                        atomic_fetch_add(&g_worker_stats[worker_id].nonces_processed, 1);

                        /* Smart-scan (adaptive): the covering protects
                           [1, gap_target), so the tail [gap_target,
                           2*needed_gap) is only needed when no closing prime
                           was found inside [needed_gap, gap_target).  Test the
                           head first; skip the tail when a closing prime
                           exists (the common case). */
                        uint64_t head_end = back_limit + rt->gap_target;
                        uint32_t head_count = 0;
                        while (head_count < candidate_count &&
                               candidate_offsets[head_count] < head_end) {
                            head_count++;
                        }

                        uint32_t tested_count = head_count;
                        uint64_t euler_pass_count = 0;
                        int gpu_tested = 0;
#ifdef WITH_CUDA
                        if (gpu && gpu_packed &&
                            crt_gpu_batch_test(gpu, gpu_limbs, gpu_base_limbs,
                                               gpu_packed, window_base,
                                               candidate_offsets, head_count,
                                               is_prime) == 0) {
                            gpu_tested = 1;
                            for (uint32_t i = 0; i < head_count; i++) {
                                euler_pass_count += is_prime[i];
                            }
                        }
#else
                        (void)0;
#endif
                        if (!gpu_tested) {
                            struct worker_limb_cache limb_cache;
                            worker_limb_cache_reset(&limb_cache);
                            for (uint32_t i = 0; i < head_count; i++) {
                                is_prime[i] =
                                    (uint8_t)worker_limb_cache_euler(
                                        &limb_cache, &euler_context,
                                        candidate, window_base,
                                        candidate_offsets[i]);
                                euler_pass_count += is_prime[i];
                            }
                        }

                        /* A prime at >= needed_gap and < gap_target closes
                           every owned gap inside the covering, so the
                           uncovered tail can be skipped entirely. */
                        uint64_t need_off = back_limit + needed_gap;
                        int have_closing = 0;
                        for (uint32_t i = 0; i < head_count; i++) {
                            if (candidate_offsets[i] >= need_off && is_prime[i]) {
                                have_closing = 1;
                                break;
                            }
                        }

                        if (!have_closing && head_count < candidate_count) {
                            gpu_tested = 0;
#ifdef WITH_CUDA
                            uint32_t tail_count = candidate_count - head_count;
                            if (gpu && gpu_packed &&
                                crt_gpu_batch_test(gpu, gpu_limbs,
                                                   gpu_base_limbs, gpu_packed,
                                                   window_base,
                                                   candidate_offsets + head_count,
                                                   tail_count,
                                                   is_prime + head_count) == 0) {
                                gpu_tested = 1;
                                for (uint32_t i = 0; i < tail_count; i++) {
                                    euler_pass_count += is_prime[head_count + i];
                                }
                            }
#endif
                            if (!gpu_tested) {
                                /* Offsets continue ascending past head_count;
                                   reuse one cache reset at the tail start. */
                                struct worker_limb_cache limb_cache;
                                worker_limb_cache_reset(&limb_cache);
                                for (uint32_t i = head_count; i < candidate_count; i++) {
                                    is_prime[i] =
                                        (uint8_t)worker_limb_cache_euler(
                                            &limb_cache, &euler_context,
                                            candidate, window_base,
                                            candidate_offsets[i]);
                                    euler_pass_count += is_prime[i];
                                }
                            }
                            tested_count = candidate_count;
                        } else if (head_count < candidate_count) {
                            atomic_fetch_add(&g_worker_stats[worker_id].smart_tail_skipped, 1);
                        }

                        atomic_fetch_add(&g_worker_stats[worker_id].candidates_generated,
                                         candidate_count);
                        atomic_fetch_add(&g_worker_stats[worker_id].candidates_tested,
                                         tested_count);
                        atomic_fetch_add(&g_worker_stats[worker_id].euler_passes,
                                         euler_pass_count);
#ifdef WITH_CUDA
                        /* Refresh GPU-accounted MR time for the hybrid path
                           (crt_gpu_batch_test) as well; the fused branch
                           already updated it inline. */
                        if (fused_fermat) {
                            atomic_store(&g_worker_stats[worker_id].gpu_accounted_us,
                                         gpu_fermat_accounted_us(fused_fermat));
                        }
#endif

                        /* A gap is owned when its first prime lies below
                           needed_gap relative to the aligned base. */
                        uint64_t owned_limit = back_limit + needed_gap;

                        crt_scan_gaps(worker_id, height, nonce, generation,
                                      merit_threshold, rt, window_base, nadd0,
                                      back_limit, p1, p2, nadd_full,
                                      is_prime, candidate_offsets, tested_count,
                                      owned_limit, half_class, NULL, 0);
                    }
                }
            }

            uint32_t next = nonce + nthreads;
            if (next < nonce) break;   /* nonce space wrapped */
            nonce = next;
        }
    }

    /* Drain the async fused flights before shutdown (collects any batch
       whose MR was still in flight). */
#ifdef WITH_CUDA
    for (int f = 0; f < 2; f++) {
        if (!fused_fl[f].active) continue;
        crt_fused_collect_batch(gpu_sieve, fused_fermat, worker_id, rt,
                                &fused_fl[f], fused_offsets[f],
                                fused_is_prime[f], p1, p2, nadd_full,
                                sieve.small_primes, sieve.inv_p,
                                sieve.small_primes_count);
        fused_fl[f].active = 0;
    }
#endif

    free(is_prime);
    sieve_core_free(&sieve);
#ifdef WITH_CUDA
    if (gpu_sieve) gpu_sieve_destroy(gpu_sieve);
    free(gpu_host_bitmap);
    free(gpu_base_limbs);
    free(gpu_packed);
    free(fused_slot_offsets);
    free(fused_sorted_offsets);
    free(fused_is_prime[0]);
    free(fused_is_prime[1]);
    for (int f = 0; f < 2; f++) {
        for (int w = 0; w < FUSED_MR_BATCH_MAX; w++) {
            mpz_clear(fused_fl[f].wins[w].window_base);
            mpz_clear(fused_fl[f].wins[w].nadd0);
        }
    }
    if (gpu) gpu_adapter_free(gpu);
#endif
    mpz_clears(base, nadd0, candidate, p1, p2, nadd_full, window_base, NULL);
    euler_context_clear(&euler_context);

    printf("[Worker %u] CRT stopped (scans=%lu, candidates=%lu, gaps=%lu, gpu_sieve_windows=%lu)\n",
           worker_id,
           atomic_load(&g_worker_stats[worker_id].nonces_processed),
           atomic_load(&g_worker_stats[worker_id].candidates_tested),
           atomic_load(&g_worker_stats[worker_id].gaps_found),
           atomic_load(&g_worker_stats[worker_id].gpu_sieve_windows));
    return NULL;
}

void worker_stop_requested(void) {
    g_stop_requested = 1;
}

void worker_get_stats(uint32_t worker_id, struct worker_stats *stats) {
    if (!stats || worker_id >= 8) return;
    stats->nonces_processed = atomic_load(&g_worker_stats[worker_id].nonces_processed);
    stats->candidates_generated = atomic_load(&g_worker_stats[worker_id].candidates_generated);
    stats->candidates_tested = atomic_load(&g_worker_stats[worker_id].candidates_tested);
    stats->euler_passes = atomic_load(&g_worker_stats[worker_id].euler_passes);
    stats->euler_pairs = atomic_load(&g_worker_stats[worker_id].euler_pairs);
    stats->merit_candidates = atomic_load(&g_worker_stats[worker_id].merit_candidates);
    stats->bpsw_attempts = atomic_load(&g_worker_stats[worker_id].bpsw_attempts);
    stats->gaps_found = atomic_load(&g_worker_stats[worker_id].gaps_found);
    stats->gaps_submitted = atomic_load(&g_worker_stats[worker_id].gaps_submitted);
    stats->max_gap_length = (uint32_t)atomic_load(&g_worker_stats[worker_id].max_gap_length);
    stats->max_merit = (double)atomic_load(&g_worker_stats[worker_id].max_merit_scaled) /
                        MERIT_STAT_SCALE;
    stats->gpu_euler_skipped = atomic_load(&g_worker_stats[worker_id].gpu_euler_skipped);
    stats->gpu_sieve_calls = atomic_load(&g_worker_stats[worker_id].gpu_sieve_calls);
    stats->gpu_sieve_windows = atomic_load(&g_worker_stats[worker_id].gpu_sieve_windows);
    stats->smart_tail_skipped = atomic_load(&g_worker_stats[worker_id].smart_tail_skipped);
    stats->gpu_accounted_us = atomic_load(&g_worker_stats[worker_id].gpu_accounted_us);
}

/* Get pending gap from queue (thread-safe) */
int worker_get_pending_gap(struct gap_queue_entry *gap_out) {
    if (!gap_out) return 0;

    int found = 0;
    pthread_mutex_lock(&g_gap_queue_lock);
    if (g_gap_queue.tail != g_gap_queue.head) {
        *gap_out = g_gap_queue.queue[g_gap_queue.tail];
        g_gap_queue.tail = (g_gap_queue.tail + 1U) % MAX_PENDING_GAPS;
        found = 1;
    }
    pthread_mutex_unlock(&g_gap_queue_lock);
    return found;
}

/* Flush all gaps from queue */
uint32_t worker_flush_gaps(void) {
    uint32_t count = 0;
    struct gap_queue_entry gap;
    while (worker_get_pending_gap(&gap)) {
        count++;
    }
    return count;
}
