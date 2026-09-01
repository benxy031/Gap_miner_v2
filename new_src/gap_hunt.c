/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GAP_HUNT implementation (see gap_hunt.h for the design).
 *
 * M1 scope: synchronous per-window device pipeline (mark -> extract ->
 * test_device -> chain).  K-window async accumulation is a follow-up.
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
#endif /* WITH_CUDA (helpers) */

int gap_hunt_run(const struct gap_hunt_config *cfg) {
#ifndef WITH_CUDA
    (void)cfg;
    fprintf(stderr, "[GAP_HUNT] requires a WITH_CUDA build\n");
    return 1;
#else
    if (!cfg || !cfg->crt_file)
        return 1;

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

    /* Anchor: 2^762 (or --gap-hunt-start), CRT-aligned. */
    mpz_t target, b0, P, bk, window_base, p1, p2, last_prime;
    mpz_init(target);
    if (cfg->start_hex && cfg->start_hex[0]) {
        if (mpz_set_str(target, cfg->start_hex, 16) != 0) {
            fprintf(stderr, "[GAP_HUNT] bad --gap-hunt-start hex\n");
            crt_runtime_free(&rt);
            return 1;
        }
    } else {
        mpz_set_ui(target, 1);
        mpz_mul_2exp(target, target, 762);
    }
    mpz_inits(b0, P, bk, window_base, p1, p2, last_prime, NULL);
    mpz_set(P, rt.primorial);
    if (!align_anchor(b0, target, &rt)) {
        fprintf(stderr, "[GAP_HUNT] CRT alignment failed\n");
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }

    /* Deep-sieve prime table (same engine the miner uses). */
    uint32_t sieve_primes = cfg->sieve_primes ? cfg->sieve_primes : 2000000U;
    struct sieve_core sieve;
    memset(&sieve, 0, sizeof(sieve));
    if (!sieve_core_init_window(&sieve, interval, sieve_primes)) {
        fprintf(stderr, "[GAP_HUNT] sieve init failed\n");
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
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
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
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
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }
    gpu_sieve = gpu_sieve_init(cfg->device, sieve.small_primes_count,
                               odd_interval_size);
    if (!gpu_sieve) {
        fprintf(stderr, "[GAP_HUNT] GPU sieve init failed\n");
        gpu_adapter_free(gpu);
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
        crt_runtime_free(&rt);
        return 1;
    }
    base_limbs = (uint64_t *)calloc((size_t)gpu_limbs, sizeof(uint64_t));
    uint64_t *offsets = (uint64_t *)malloc(
        (size_t)sieve.candidate_capacity * sizeof(uint64_t));
    uint8_t *flags = (uint8_t *)malloc((size_t)sieve.candidate_capacity);
    if (!base_limbs || !offsets || !flags) {
        fprintf(stderr, "[GAP_HUNT] host buffers failed\n");
        free(base_limbs);
        free(offsets);
        free(flags);
        gpu_sieve_destroy(gpu_sieve);
        gpu_adapter_free(gpu);
        sieve_core_free(&sieve);
        mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
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
            "sieve=%u min_merit=%.2f k0=%llu device=%d\n",
            rt.shift, rt.n_primes,
            (double)mpz_sizeinbase(P, 2),
            (unsigned long long)rt.window, sieve_primes, cfg->min_merit,
            (unsigned long long)k, cfg->device);

    uint64_t windows = 0, gaps_reported = 0;
    double best_merit = 0.0;

    while (!g_stop) {
        /* Windows are P apart (NOT contiguous), so the previous window's
           last prime is not the predecessor of this window's first prime.
           Chain only within the window; the first prime of each window has
           an unknown predecessor and is skipped for gap measurement. */
        have_last = 0;
        mpz_set(bk, b0);
        mpz_addmul_ui(bk, P, (unsigned long)k);
        mpz_sub_ui(window_base, bk, back_limit);

        /* Export base limbs (little-endian, zero-padded). */
        memset(base_limbs, 0, (size_t)gpu_limbs * sizeof(uint64_t));
        size_t exported = 0;
        mpz_export(base_limbs, &exported, -1, sizeof(uint64_t), 0, 0,
                   window_base);
        if (exported > (size_t)gpu_limbs) {
            fprintf(stderr, "[GAP_HUNT] limb overflow\n");
            break;
        }
        uint64_t first_odd_offset =
            (base_limbs[0] & 1ULL) ? 0U : 1U;
        uint64_t odd_size = odd_interval_size;
        if (interval <= first_odd_offset)
            odd_size = 0;
        else
            odd_size = (interval - first_odd_offset + 1ULL) >> 1;

        if (!gpu_sieve_mark_from_base(gpu_sieve, odd_size, first_odd_offset,
                                      base_limbs, gpu_limbs, 0,
                                      sieve.small_primes, sieve.inv_p,
                                      sieve.small_primes_count, NULL, 0)) {
            fprintf(stderr, "[GAP_HUNT] GPU mark failed (k=%llu)\n",
                    (unsigned long long)k);
            break;
        }

        uint64_t *d_cands = NULL;
        unsigned int nc = 0;
        int fermat_limbs = gpu_fermat_get_limbs(fermat);
        if (!gpu_sieve_extract_pack_device(
                gpu_sieve, odd_size, first_odd_offset, base_limbs,
                fermat_limbs, &d_cands, offsets, &nc, 0, UINT64_MAX, 0)) {
            fprintf(stderr, "[GAP_HUNT] GPU extract failed (k=%llu)\n",
                    (unsigned long long)k);
            break;
        }

        if (nc > 0 && nc <= sieve.candidate_capacity) {
            if (gpu_fermat_test_device(fermat, d_cands, flags, (size_t)nc) < 0) {
                fprintf(stderr, "[GAP_HUNT] GPU MR failed (k=%llu)\n",
                        (unsigned long long)k);
                break;
            }
            /* Chain consecutive primes: last_prime (previous window) ->
               primes of this window. */
            for (unsigned int i = 0; i < nc; i++) {
                if (!flags[i])
                    continue;
                mpz_set(p1, window_base);
                mpz_add_ui(p1, p1, offsets[i]);
                if (have_last) {
                    mpz_sub(p2, p1, last_prime);
                    double merit =
                        mpz_get_d(p2) / logbase;
                    if (merit >= cfg->min_merit) {
                        if (baillie_psw_test(p1) &&
                            baillie_psw_test(last_prime)) {
                            char *start_dec =
                                mpz_get_str(NULL, 10, last_prime);
                            char *end_dec = mpz_get_str(NULL, 10, p1);
                            fprintf(stderr,
                                    "[GAP_HUNT] k=%llu start=%s end=%s "
                                    "gap=%llu merit=%.4f\n",
                                    (unsigned long long)k,
                                    start_dec ? start_dec : "?",
                                    end_dec ? end_dec : "?",
                                    (unsigned long long)mpz_get_ui(p2),
                                    merit);
                            if (out) {
                                fprintf(out,
                                        "%llu %s %llu %.4f\n",
                                        (unsigned long long)k,
                                        start_dec ? start_dec : "?",
                                        (unsigned long long)mpz_get_ui(p2),
                                        merit);
                                fflush(out);
                            }
                            if (merit > best_merit)
                                best_merit = merit;
                            gaps_reported++;
                            free(start_dec);
                            free(end_dec);
                        }
                    }
                }
                mpz_set(last_prime, p1);
                have_last = 1;
            }
        }

        k++;
        windows++;
        if (windows % 1024ULL == 0) {
            save_state(cfg->state_path, k, last_prime, have_last);
            fprintf(stderr,
                    "[GAP_HUNT] k=%llu windows=%llu gaps=%llu best_merit=%.4f\n",
                    (unsigned long long)k, (unsigned long long)windows,
                    (unsigned long long)gaps_reported, best_merit);
        }
    }

    save_state(cfg->state_path, k, last_prime, have_last);
    fprintf(stderr,
            "[GAP_HUNT] stopped: windows=%llu gaps=%llu best_merit=%.4f "
            "next_k=%llu\n",
            (unsigned long long)windows, (unsigned long long)gaps_reported,
            best_merit, (unsigned long long)k);

    if (out)
        fclose(out);
    free(base_limbs);
    free(offsets);
    free(flags);
    gpu_sieve_destroy(gpu_sieve);
    gpu_adapter_free(gpu);
    sieve_core_free(&sieve);
    mpz_clears(target, b0, P, bk, window_base, p1, p2, last_prime, NULL);
    crt_runtime_free(&rt);
    return 0;
#endif /* WITH_CUDA */
}
