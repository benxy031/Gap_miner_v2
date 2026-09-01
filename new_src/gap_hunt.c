/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GAP_HUNT implementation (see gap_hunt.h for the design).
 *
 * M1.5: K-window accumulated MR batches (GAP_HUNT_BATCH env, default 8,
 * max GAP_HUNT_BATCH_MAX windows per gpu_fermat_submit_device call, two
 * alternating flights so the host processes one collected batch while the
 * GPU runs the next).  Out format: `<gap> <merit> <startprime>` per line.
 */

/* POSIX sigaction/sigemptyset under -std=c99 */
#define _POSIX_C_SOURCE 200809L

#include "gap_hunt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>
#include <gmp.h>

#include "crt_runtime.h"
#include "sieve_core.h"
#include "primality_bpsw.h"

#ifdef WITH_CUDA
#include "gpu_adapter.h"
#include "gpu/gpu_sieve.h"
#include "gpu/gpu_fermat.h"
#endif

#ifdef WITH_CUDA
static volatile sig_atomic_t g_stop = 0;

static void on_signal(int sig) {
    (void)sig;
    g_stop = 1;
}

/* Solve b0 ≡ -(o_i + adj) (mod p_i) for every cover prime with o_i != 0,
   starting from the target anchor; returns the smallest solution >= target
   in [0, P).  Mirrors crt_runtime's align_base (incremental CRT). */
static int align_anchor(mpz_t b0, const mpz_t target,
                        const struct crt_runtime *rt) {
    mpz_t nadd, M, Mmod, inv, term;
    mpz_init_set_ui(nadd, 0);
    mpz_init_set_ui(M, 1);
    mpz_inits(Mmod, inv, term, NULL);

    uint32_t adj = rt->adj;
    for (uint32_t i = 0; i < rt->n_primes; i++) {
        uint64_t p = rt->primes[i];
        uint64_t o = rt->offsets[i];
        if (o == 0)
            continue;

        uint64_t base_mod_p = mpz_fdiv_ui(target, p);
        uint64_t sum = (base_mod_p + o + adj) % p;
        uint64_t target_r = (sum == 0) ? 0 : (p - sum);

        uint64_t curr_r = mpz_fdiv_ui(nadd, p);
        uint64_t diff = (target_r + p - curr_r) % p;

        uint64_t M_mod_p = mpz_fdiv_ui(M, p);
        mpz_set_ui(Mmod, M_mod_p);
        mpz_set_ui(term, p);
        if (mpz_invert(inv, Mmod, term) == 0) {
            mpz_clears(nadd, M, Mmod, inv, term, NULL);
            return 0;
        }
        uint64_t k = (diff * mpz_get_ui(inv)) % p;

        mpz_set_ui(term, k);
        mpz_addmul(nadd, term, M);
        mpz_mul_ui(M, M, p);
    }

    mpz_add(b0, target, nadd);
    /* b0 must be EVEN (candidates are the odd offsets after it).  P is odd
       (product of odd primes), so adding one period flips the parity. */
    if (mpz_odd_p(b0))
        mpz_add(b0, b0, M);

    mpz_clears(nadd, M, Mmod, inv, term, NULL);
    return 1;
}

static int save_state(const char *path, uint64_t k, const mpz_t last_prime,
                      int have_last) {
    if (!path)
        return 0;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f)
        return -1;
    fprintf(f, "k %llu\n", (unsigned long long)k);
    if (have_last) {
        char *dec = mpz_get_str(NULL, 10, last_prime);
        if (dec) {
            fprintf(f, "last_prime %s\n", dec);
            free(dec);
        }
    }
    fclose(f);
    if (rename(tmp, path) != 0)
        return -1;
    return 0;
}

static int load_state(const char *path, uint64_t *k, mpz_t last_prime,
                      int *have_last) {
    *have_last = 0;
    if (!path)
        return 0;
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;   /* no state yet */
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        unsigned long long v;
        if (sscanf(line, "k %llu", &v) == 1)
            *k = (uint64_t)v;
        else if (strncmp(line, "last_prime ", 11) == 0) {
            char *p = line + 11;
            p[strcspn(p, "\r\n")] = 0;
            if (mpz_set_str(last_prime, p, 10) == 0)
                *have_last = 1;
        }
    }
    fclose(f);
    return 0;
}

/* ── M1.5: K-window accumulated MR batches ──────────────────────────────
   One flight = K windows marked into ping-pong bitmaps and packed
   contiguously into the slot's device candidate buffer, then ONE async MR
   submission.  Two flights alternate: while the host processes the
   collected flight, the GPU runs the other flight's MR kernel. */

#define GAP_HUNT_BATCH_MAX 16
static int g_batch = 16;       /* GAP_HUNT_BATCH env, clamped 1..MAX */

struct gh_batch {
    int active;                 /* submitted, not yet collected */
    int slot;                   /* fermat slot + candidate buffer (0/1) */
    uint32_t n_windows;
    uint32_t total;             /* MR candidates in the batch */
    uint64_t base_k[GAP_HUNT_BATCH_MAX];
    uint32_t count[GAP_HUNT_BATCH_MAX];
    uint32_t cum[GAP_HUNT_BATCH_MAX + 1];   /* prefix sums into the flags array */
    mpz_t win_base[GAP_HUNT_BATCH_MAX];
    uint64_t *win_off[GAP_HUNT_BATCH_MAX];  /* host offset slices */
};

struct gh_ctx {
    struct gpu_sieve_ctx *sieve;
    struct gpu_fermat_ctx *fermat;
    int fermat_limbs;
    int gpu_limbs;
    uint64_t *base_limbs;
    uint64_t capacity;          /* host offsets capacity per window */
    uint64_t interval;
    uint64_t odd_interval_size;
    uint64_t back_limit;
    const uint64_t *primes;
    const uint64_t *inv_p;
    size_t prime_count;
    mpz_t bk, wb;               /* scratch */
    mpz_t b0;
    mpz_t P;
};

static void gh_batch_init(struct gh_batch *b) {
    memset(b, 0, sizeof(*b));
    for (int i = 0; i < GAP_HUNT_BATCH_MAX; i++)
        mpz_init(b->win_base[i]);
}

static void gh_batch_clear(struct gh_batch *b) {
    for (int i = 0; i < GAP_HUNT_BATCH_MAX; i++)
        mpz_clear(b->win_base[i]);
}

/* Fill one flight with K windows (mark + extract into the slot's candidate
   buffer) and submit ONE async MR batch.  Returns 1 on success; on failure
   or an empty batch, active stays 0. */
static int gh_batch_fill(struct gh_batch *b, int slot, uint64_t k0,
                         struct gh_ctx *g) {
    uint32_t cum = 0;
    for (uint32_t i = 0; i < (uint32_t)g_batch; i++) {
        uint64_t k = k0 + i;
        b->base_k[i] = k;
        mpz_set(g->bk, g->b0);
        mpz_addmul_ui(g->bk, g->P, (unsigned long)k);
        mpz_sub_ui(g->wb, g->bk, g->back_limit);
        mpz_set(b->win_base[i], g->wb);

        memset(g->base_limbs, 0, (size_t)g->gpu_limbs * sizeof(uint64_t));
        size_t exported = 0;
        mpz_export(g->base_limbs, &exported, -1, sizeof(uint64_t), 0, 0,
                   g->wb);
        if (exported > (size_t)g->gpu_limbs)
            return 0;
        uint64_t first_odd_offset = (g->base_limbs[0] & 1ULL) ? 0U : 1U;
        uint64_t odd_size = 0;
        if (g->interval > first_odd_offset)
            odd_size = (g->interval - first_odd_offset + 1ULL) >> 1;

        int buf = (int)(i & 1);
        if (!gpu_sieve_mark_from_base(g->sieve, odd_size, first_odd_offset,
                                      g->base_limbs, g->gpu_limbs, buf,
                                      g->primes, g->inv_p, g->prime_count,
                                      NULL, 0))
            return 0;

        uint64_t *d_cands = NULL;
        unsigned int nc = 0;
        if (!gpu_sieve_extract_pack_device_range_ex(
                g->sieve, odd_size, first_odd_offset, 0, odd_size,
                buf, slot, g->base_limbs, g->fermat_limbs, &d_cands,
                b->win_off[i], &nc, 0, UINT64_MAX, 0, cum))
            return 0;
        if ((uint64_t)nc > g->capacity)
            return 0;
        /* gpu_fermat_submit_device silently clamps to GPU_ADAPTER_MAX_BATCH;
           exceeding it would drop candidates and fabricate false gaps. */
        if (cum + nc > GPU_ADAPTER_MAX_BATCH)
            return 0;
        b->count[i] = nc;
        b->cum[i] = cum;
        cum += nc;
    }
    b->cum[g_batch] = cum;
    b->n_windows = (uint32_t)g_batch;
    b->total = cum;
    b->slot = slot;
    if (cum == 0)
        return 1;               /* nothing to MR-test; process as empty */
    uint64_t *d_batch = gpu_sieve_candidate_buffer(g->sieve, slot);
    if (gpu_fermat_submit_device(g->fermat, slot, d_batch, (size_t)cum) != 0)
        return 0;
    b->active = 1;
    return 1;
}

/* Chain consecutive primes within each window of a collected flight,
   report every BPSW-verified gap with merit >= min_merit in the
   `<gap> <merit> <startprime>` format, and advance the counters.
   Merit is the TRUE record merit: gap / ln(start). */
static void gh_batch_process(struct gh_batch *b, const uint8_t *flags,
                             const struct gap_hunt_config *cfg,
                             uint64_t *windows,
                             uint64_t *gaps, double *best,
                             FILE *out, mpz_t p1, mpz_t p2,
                             mpz_t last_prime, int *have_last) {
    for (uint32_t i = 0; i < b->n_windows; i++) {
        /* Windows are P apart (NOT contiguous): chain only within the
           window; its first prime has an unknown predecessor. */
        *have_last = 0;
        for (uint32_t j = 0; j < b->count[i]; j++) {
            if (!flags[b->cum[i] + j])
                continue;
            mpz_set(p1, b->win_base[i]);
            mpz_add_ui(p1, p1, b->win_off[i][j]);
            if (*have_last) {
                mpz_sub(p2, p1, last_prime);
                /* True merit (record convention): gap / ln(start). */
                double merit = mpz_get_d(p2) / log(mpz_get_d(last_prime));
                if (merit >= cfg->min_merit &&
                    baillie_psw_test(p1) && baillie_psw_test(last_prime)) {
                    char *start_dec = mpz_get_str(NULL, 10, last_prime);
                    char *end_dec = mpz_get_str(NULL, 10, p1);
                    fprintf(stderr,
                            "[GAP_HUNT] gap=%llu merit=%.6f start=%s end=%s\n",
                            (unsigned long long)mpz_get_ui(p2), merit,
                            start_dec ? start_dec : "?",
                            end_dec ? end_dec : "?");
                    if (out) {
                        fprintf(out, "%llu %.6f %s\n",
                                (unsigned long long)mpz_get_ui(p2), merit,
                                start_dec ? start_dec : "?");
                        fflush(out);
                    }
                    if (merit > *best)
                        *best = merit;
                    (*gaps)++;
                    free(start_dec);
                    free(end_dec);
                }
            }
            mpz_set(last_prime, p1);
            *have_last = 1;
        }
        (*windows)++;
    }
}
#endif /* WITH_CUDA (helpers) */

int gap_hunt_run(const struct gap_hunt_config *cfg) {
#ifndef WITH_CUDA
    (void)cfg;
    fprintf(stderr, "[GAP_HUNT] requires a WITH_CUDA build\n");
    return 1;
#else
    if (!cfg || !cfg->crt_file)
        return 1;

    /* Runtime K (GAP_HUNT_BATCH env): windows per accumulated MR batch. */
    {
        const char *kb = getenv("GAP_HUNT_BATCH");
        if (kb && kb[0]) {
            int v = atoi(kb);
            if (v >= 1 && v <= GAP_HUNT_BATCH_MAX)
                g_batch = v;
        }
    }

    struct crt_runtime rt;
    if (!crt_runtime_load(&rt, cfg->crt_file)) {
        fprintf(stderr, "[GAP_HUNT] failed to load CRT file: %s\n",
                cfg->crt_file);
        return 1;
    }
    double logbase = (256.0 + (double)rt.shift) * log(2.0);
    uint64_t back_limit = (uint64_t)ceil(2.0 * logbase);
    if (back_limit < 4096)
        back_limit = 4096;
    if (back_limit & 1ULL)
        back_limit++;
    uint64_t interval = back_limit + rt.window;
    uint64_t odd_interval_size = interval >> 1;

    /* Anchor: default 2^(255+shift) — half the file's candidate width, so
       merit uses the file-consistent logbase.  Override with
       --gap-hunt-start (any hex anchor works; alignment is recomputed). */
    mpz_t target, b0, P, p1, p2, last_prime;
    mpz_init(target);
    if (cfg->start_hex && cfg->start_hex[0]) {
        if (mpz_set_str(target, cfg->start_hex, 16) != 0) {
            fprintf(stderr, "[GAP_HUNT] bad --gap-hunt-start hex\n");
            crt_runtime_free(&rt);
            return 1;
        }
    } else {
        mpz_set_ui(target, 1);
        mpz_mul_2exp(target, target, 255U + rt.shift);
    }
    mpz_inits(b0, P, p1, p2, last_prime, NULL);
    mpz_set(P, rt.primorial);
    if (!align_anchor(b0, target, &rt)) {
        fprintf(stderr, "[GAP_HUNT] CRT alignment failed\n");
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }

    /* Deep-sieve prime table (same engine the miner uses). */
    uint32_t sieve_primes = cfg->sieve_primes ? cfg->sieve_primes : 2000000U;
    struct sieve_core sieve;
    memset(&sieve, 0, sizeof(sieve));
    if (!sieve_core_init_window(&sieve, interval, sieve_primes)) {
        fprintf(stderr, "[GAP_HUNT] sieve init failed\n");
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }

    /* GPU contexts. */
    struct gpu_adapter *gpu = gpu_adapter_init(cfg->device);
    struct gpu_fermat_ctx *fermat = NULL;
    struct gpu_sieve_ctx *gpu_sieve = NULL;
    int gpu_limbs = 0;
    uint64_t *base_limbs = NULL;
    if (!gpu) {
        fprintf(stderr, "[GAP_HUNT] GPU adapter init failed\n");
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }
    gpu_adapter_set_candidate_bits(gpu, 256U + rt.shift);
    gpu_limbs = gpu_adapter_get_limbs(gpu);
    fermat = gpu_adapter_get_fermat_ctx(gpu);
    if (gpu_limbs <= 0 || !fermat) {
        fprintf(stderr, "[GAP_HUNT] GPU Fermat setup failed\n");
        gpu_adapter_free(gpu);
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }
    gpu_sieve = gpu_sieve_init(cfg->device, sieve.small_primes_count,
                               odd_interval_size);
    if (!gpu_sieve) {
        fprintf(stderr, "[GAP_HUNT] GPU sieve init failed\n");
        gpu_adapter_free(gpu);
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }
    gpu_sieve_set_extract_accum(gpu_sieve, (uint32_t)g_batch);
    base_limbs = (uint64_t *)calloc((size_t)gpu_limbs, sizeof(uint64_t));
    size_t win_cap = (size_t)sieve.candidate_capacity;
    uint64_t *offsets = (uint64_t *)malloc(
        2U * (size_t)g_batch * win_cap * sizeof(uint64_t));
    uint8_t *flags = (uint8_t *)malloc(
        (size_t)g_batch * win_cap);
    if (!base_limbs || !offsets || !flags) {
        fprintf(stderr, "[GAP_HUNT] host buffers failed\n");
        free(base_limbs);
        free(offsets);
        free(flags);
        gpu_sieve_destroy(gpu_sieve);
        gpu_adapter_free(gpu);
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }

    uint64_t k = 0;
    int have_last = 0;
    /* last_prime is saved in the state file only for diagnostics; it is
       NOT restored across runs (windows are non-contiguous). */
    load_state(cfg->state_path, &k, last_prime, &have_last);
    have_last = 0;

    FILE *out = NULL;
    if (cfg->out_path && cfg->out_path[0]) {
        out = fopen(cfg->out_path, "a");
        if (!out)
            fprintf(stderr, "[GAP_HUNT] cannot open out file %s\n",
                    cfg->out_path);
    }

    /* Persistent handlers: timeout(1) signals the child AND the process
       group, so a one-shot signal() handler (SA_RESETHAND) dies on the
       second delivery before the loop can shut down cleanly. */
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }

    fprintf(stderr,
            "[GAP_HUNT] walk: shift=%u n_primes=%u P bits=%.0f window=%llu "
            "sieve=%u batch=%d min_merit=%.6f k0=%llu device=%d\n",
            rt.shift, rt.n_primes,
            (double)mpz_sizeinbase(P, 2),
            (unsigned long long)rt.window, sieve_primes, g_batch,
            cfg->min_merit,
            (unsigned long long)k, cfg->device);

    uint64_t windows = 0, gaps_reported = 0;
    double best_merit = 0.0;

    /* Two alternating flights (fermat slots 0/1). */
    struct gh_batch A, B;
    gh_batch_init(&A);
    gh_batch_init(&B);
    struct gh_ctx g;
    memset(&g, 0, sizeof(g));
    g.sieve = gpu_sieve;
    g.fermat = fermat;
    g.fermat_limbs = gpu_fermat_get_limbs(fermat);
    g.gpu_limbs = gpu_limbs;
    g.base_limbs = base_limbs;
    g.capacity = (uint64_t)sieve.candidate_capacity;
    g.interval = interval;
    g.odd_interval_size = odd_interval_size;
    g.back_limit = back_limit;
    g.primes = sieve.small_primes;
    g.inv_p = sieve.inv_p;
    g.prime_count = sieve.small_primes_count;
    mpz_init(g.bk);
    mpz_init(g.wb);
    mpz_set(g.b0, b0);
    mpz_set(g.P, P);

    for (int i = 0; i < g_batch; i++) {
        A.win_off[i] = offsets + (size_t)i * win_cap;
        B.win_off[i] =
            offsets + ((size_t)g_batch + (size_t)i) * win_cap;
    }

    uint64_t next_k = k;
    uint64_t last_save = windows;

    while (!g_stop) {
        /* Fill A (then B while A's MR is in flight). */
        if (!A.active && A.n_windows == 0) {
            if (!gh_batch_fill(&A, 0, next_k, &g)) {
                fprintf(stderr, "[GAP_HUNT] batch fill failed (k=%llu)\n",
                        (unsigned long long)next_k);
                break;
            }
            next_k += (uint64_t)g_batch;
            if (!A.active) {
                /* Empty batch: nothing submitted; process immediately. */
                gh_batch_process(&A, flags, cfg, &windows,
                                 &gaps_reported, &best_merit, out,
                                 p1, p2, last_prime, &have_last);
                A.n_windows = 0;
            }
        }
        if (!B.active && B.n_windows == 0 && A.active) {
            if (!gh_batch_fill(&B, 1, next_k, &g)) {
                fprintf(stderr, "[GAP_HUNT] batch fill failed (k=%llu)\n",
                        (unsigned long long)next_k);
                break;
            }
            next_k += (uint64_t)g_batch;
        }

        /* Collect + process the older active flight (the GPU overlaps
           this host work with the other flight's MR kernel). */
        struct gh_batch *fl = NULL;
        if (A.active)
            fl = &A;
        else if (B.active)
            fl = &B;
        if (fl) {
            if (gpu_fermat_collect(fermat, fl->slot, flags,
                                   (size_t)fl->total) < 0) {
                fprintf(stderr, "[GAP_HUNT] GPU collect failed\n");
                break;
            }
            gh_batch_process(fl, flags, cfg, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
            fl->active = 0;
            fl->n_windows = 0;
            fl->total = 0;
        }

        if (windows - last_save >= 1024ULL) {
            save_state(cfg->state_path, next_k, last_prime, have_last);
            last_save = windows;
            fprintf(stderr,
                    "[GAP_HUNT] k=%llu windows=%llu gaps=%llu best_merit=%.6f\n",
                    (unsigned long long)next_k,
                    (unsigned long long)windows,
                    (unsigned long long)gaps_reported, best_merit);
        }
    }

    /* Drain the in-flight flights so no window is lost. */
    if (A.active) {
        if (gpu_fermat_collect(fermat, A.slot, flags, (size_t)A.total) >= 0)
            gh_batch_process(&A, flags, cfg, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
        A.active = 0;
    }
    if (B.active) {
        if (gpu_fermat_collect(fermat, B.slot, flags, (size_t)B.total) >= 0)
            gh_batch_process(&B, flags, cfg, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
        B.active = 0;
    }
    gh_batch_clear(&A);
    gh_batch_clear(&B);
    mpz_clears(g.bk, g.wb, NULL);

    save_state(cfg->state_path, next_k, last_prime, have_last);
    fprintf(stderr,
            "[GAP_HUNT] stopped: windows=%llu gaps=%llu best_merit=%.6f "
            "next_k=%llu\n",
            (unsigned long long)windows, (unsigned long long)gaps_reported,
            best_merit, (unsigned long long)next_k);

    if (out)
        fclose(out);
    free(base_limbs);
    free(offsets);
    free(flags);
    gpu_sieve_destroy(gpu_sieve);
    gpu_adapter_free(gpu);
    sieve_core_free(&sieve);
    mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
    crt_runtime_free(&rt);
    return 0;
#endif /* WITH_CUDA */
}
