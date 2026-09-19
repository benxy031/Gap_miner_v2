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
#include <time.h>
#include <gmp.h>

#include "crt_runtime.h"
#include "sieve_core.h"
#include "primality_bpsw.h"
#include "halfclass.h"

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

/* Windows per MR batch.  Raised 32 -> 64 (2026-09-15): the chain's rounds are
   dominated by a ~4 ms per-LAUNCH floor of the CGBN MR kernel (measured with
   GAP_HUNT_TIMING=1: 512 candidates/round -> 4.1 ms, 2048 -> 5.3 ms, i.e.
   ~3.6 ms fixed + 0.76 us/test), so every window in a round pays that floor
   divided by the number of windows sharing it.  More windows per round =
   cheaper rounds; the device cost is only the gather/candidate buffers
   (K x chunk x limbs), well within the 160000-candidate adapter cap.
   Cap raised 64 -> 128 (2026-09-17) and 128 -> 512 (2026-09-18): with the
   mode-aware cum bound (see the guard in gap_hunt_run) the chain path -- the
   fleet default -- is no longer limited by the MR context cap, because it
   never submits the window-major head batch (it gathers K x chunk candidates
   per round).  The plain path keeps the MR-cap bound and is auto-reduced. */
#define GAP_HUNT_BATCH_MAX 512
/* Default 512 (was 64; the ~4 ms per-LAUNCH floor of a chain round is amortized
   over the windows sharing it).  Doubling history: 32 -> 64 was +21% (shift1017
   559 -> 679 win/s) and +28% (shift507 1876 -> 2406 win/s); re-measured
   2026-09-18 on the quiet dev GPU with the mode-aware cum bound (jump2 on,
   min-merit 19, fixed k range, emitted sets identical): shift507 2452 (K=64)
   -> 3156 (128) -> 3601 (256) -> **3824 win/s (512) = +56%**; shift1017 881
   (128) -> 1001 (256) -> **1172 win/s (512) = +33%**.
   Device cost is `2 x odd_interval_size x K x limbs x 8 B` -- the walk banner
   prints `vram_est=<MB>` for the configured K: shift507 K=512 ~2.4 GB, shift1017
   K=512 **6407 MB** by the banner and ~7.3/8 GB actually used (sieve contexts +
   adapter + display on top), i.e. at high shift K=512 is at the edge of an 8 GB
   card.  Check `vram_est` against nvidia-smi and set GAP_HUNT_BATCH=256 (~half)
   where it does not fit; a 12 GB card takes K=512 with room to spare.  An
   allocation failure is visible ("gpu_sieve: cands[0] alloc: out of memory") and
   stops the walk; it never falls back silently to a slower path. */
static int g_batch = 512;      /* GAP_HUNT_BATCH env, clamped 1..MAX */
static int g_quarter = 0;      /* GAP_HUNT_QUARTER env: 4-visible-class scan */
static uint64_t g_kmax = 0;    /* GAP_HUNT_KMAX env: stop hook (tests/bench) */
/* Per-window chain-pair capacity in jump/jump2 mode. RAISED 16 -> 64 -> 128 on
   2026-09-18 after MEASURING the real envelope (the number of pairs a window
   needs scales INVERSELY with the merit threshold: a lower threshold = shorter
   jumps = more steps). shift507 p74_lex_m30, 16k windows, quiet GPU:
     merit 10 (the fleet setting) ->  2 pairs/window   (8x margin at cap 16)
     merit  3                     -> 14 pairs/window   (1.1x margin at cap 16)
     merit  1                     -> 16 = cap 16 => the walk DIED; re-measured
                                     with cap 64 the true value is **43**, so
                                     the old ceiling was 2.7x too small
   Lower thresholds are a documented record-rate lever (merit 10 is ~14x
   records/h better than 18), so this ceiling sat directly in the way of a
   recommended configuration. 128 gives ~3x margin at merit 1 and 64x at the
   fleet setting; it costs 512 KiB of host arrays (2 x 512 x 128 x 4 B) plus the
   same on the device for the SERIAL jump path. The mid-walk guard stays
   fail-closed (truncating a chain silently would drop records) and the startup
   advisory below warns before the walk when the geometry says the ceiling is
   reachable. The walk summary reports the observed worst case as
   `jump2_pairs_max` / `jump2_cap_hits`. */
#define GAP_HUNT_JUMP_CAP 128
/* The number of chain pairs ONE window can produce is data-dependent: with a
   low merit threshold (the fleet configs run --gap-hunt-min-merit 10) and a
   large window the chain needs many steps, and GAP_HUNT_JUMP_CAP is a
   compile-time ceiling on that. Two bounded counters make the worst case a
   measurement instead of an assumption -- reported in the walk summary as
   `jump2_pairs_max` / `jump2_cap_hits`. (`cap_hits > 0` means the run STOPPED
   because a window needed more pairs than the ceiling allows.) */
static uint32_t g_j2_max_pairs = 0;
static uint64_t g_j2_cap_hits = 0;
/* Hard line cap for the GHDBG_DUMP_K diagnostics file.  The cap is the point:
   the first version of that hook could dump every candidate of every window,
   which wrote 8.1 GB and took the editor down (2026-09-17).  A diagnostic must
   never be able to flood the terminal again. */
#define GHDUMP_CAP 40000
static int g_jump = 0;         /* GAP_HUNT_JUMP env: Kehrig-style walk */
/* Per-window chunk cap for the chain.  Default 32 (was 64): with a 64-window
   batch the round floor is amortized, so the optimum moves to the smaller
   chunk that tests fewer candidates per window (measured shift1017: K=64
   chunk 32 -> 679 win/s with 268 tests/window vs K=64 chunk 64 -> 664 win/s
   with 390 tests/window; shift507: 2406 win/s @222 tests vs 2283 @369).  It is
   coupled to GAP_HUNT_BATCH: at K=32, chunk 32 is worse than 64. */
#define GAP_HUNT_JUMP2_CHUNK_DEFAULT 32
static int g_jump2 = 1;        /* GAP_HUNT_JUMP2=0 restores the full scan */
static int g_jump2_chunk = GAP_HUNT_JUMP2_CHUNK_DEFAULT;

/* Optional HOST-stage accounting (GAP_HUNT_TIMING=1, diagnostics only).  The
   walk's windows/s is only meaningful if the host side is not the wall, so the
   per-window host cost is split into the four things the host actually does:
   window setup (mpz base/wb + limb export), the mark call, the extract call,
   the MR submit (jump2 chunk scan or batched submit) and the collect+detect
   gap processing.  Printed on the periodic tick next to the GPU kernel line. */
static int g_host_timing = 0;
static uint64_t g_ht_setup, g_ht_mark, g_ht_extract, g_ht_submit, g_ht_process;

/* Chain bookkeeping for the same diagnostic line: how many MR candidates the
   jump2 chain actually submitted and in how many rounds (the two numbers that
   decide whether the walk is MR-throughput bound or round-latency bound). */
static uint64_t g_j2_tests, g_j2_rounds;

/* Round-internal split for the same diagnostic line: a chain round is
   gather -> submit -> collect, and at small chunks the FIXED part of a round
   (not the MR work) decides whether small chunks are viable. */
static uint64_t g_rt_gather_us, g_rt_submit_us, g_rt_collect_us;

static uint64_t gh_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000);
}

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
    uint8_t base_even[GAP_HUNT_BATCH_MAX];  /* template prefilter validity */
    /* jump mode results (host copies of the device walk's output) */
    uint32_t jump_n[GAP_HUNT_BATCH_MAX];
    uint32_t jump_s[GAP_HUNT_BATCH_MAX][GAP_HUNT_JUMP_CAP];
    uint32_t jump_e[GAP_HUNT_BATCH_MAX][GAP_HUNT_JUMP_CAP];
};

struct gh_ctx {
    struct gpu_sieve_ctx *sieve;
    struct gpu_fermat_ctx *fermat;
    int fermat_limbs;
    int gpu_limbs;
    uint64_t *base_limbs;
    uint64_t capacity;          /* host offsets capacity per window */
    uint64_t cap;               /* REAL per-submit candidate cap of the MR ctx */
    uint64_t interval;
    uint64_t odd_interval_size;
    uint64_t back_limit;
    int quarter;                /* GAP_HUNT_QUARTER active */
    uint64_t region_start;      /* offsets below this are all-class scanned */
    uint64_t tail_start;        /* offsets at/after this are all-class scanned */
    uint64_t owned_limit;       /* in-window high endpoint (offset rel base) */
    struct halfclass_tpl tpl;   /* CRT template prefilter for resolution */
    const uint64_t *primes;
    const uint64_t *inv_p;
    size_t prime_count;
    mpz_t bk, wb;               /* scratch */
    mpz_t b0;
    mpz_t P;
    int jump;                   /* GAP_HUNT_JUMP active */
    int jump2;                  /* GAP_HUNT_JUMP2 active (chunk-parallel) */
    int jump2_chunk;            /* per-window chunk size for jump2 rounds */
    double min_merit;           /* report threshold (jump thresholds) */
    uint64_t jump_thresh[GAP_HUNT_BATCH_MAX];
    uint64_t jump_gaps[GAP_HUNT_BATCH_MAX * GAP_HUNT_JUMP_CAP];
    uint32_t jump_n[GAP_HUNT_BATCH_MAX];
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

/* ln(n) via mantissa+exponent: mpz_get_d overflows (>2^1024) to +inf,
   which silently zeroed every merit at shift 1017 (1273-bit starts). */
static double gh_log_mpz(const mpz_t n) {
    if (mpz_sgn(n) <= 0)
        return 0.0;
    signed long int e = 0;
    double m = mpz_get_d_2exp(&e, n); /* n = m * 2^e, 0.5 <= |m| < 1 */
    return log(m) + (double)e * 0.69314718055994530942;
}

/* Fill one flight with K windows (mark + extract into the slot's candidate
   buffer) and submit ONE async MR batch.  Returns 1 on success; on failure
   or an empty batch, active stays 0. */

/* First index > p whose offset gap reaches thr (offsets are ascending). */
static uint32_t gh_jump2_find_q(struct gh_batch *b, uint32_t i, uint32_t p,
                                uint64_t thr) {
    uint32_t lo = p + 1, hi = b->count[i];
    uint64_t target = b->win_off[i][p] + thr;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (b->win_off[i][mid] >= target)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo < b->count[i] ? lo : UINT32_MAX;
}

/* ── Jump2: chunk-parallel backward search (Kehrig-exact, latency-free) ───
   Same chain semantics as cgbn_jump_scan_kernel_t (parity-exact), but every
   step tests a CHUNK of candidates batched across all windows in one tight
   MR submit, so the cooperative-test latency that killed the serial jump is
   hidden.  Fills b->jump_n/s/e for the existing jump-mode reporter. */
enum { J2_FIND_FIRST = 0, J2_JUMP_BACK = 1, J2_FIND_END = 2, J2_DONE = 3 };

static int gh_jump2_scan(struct gh_batch *b, int slot, struct gh_ctx *g,
                         uint8_t *flags) {
    const uint32_t C = (uint32_t)g->jump2_chunk;
    const uint32_t K = b->n_windows;
    if (K == 0)
        return 1;
    if (K > GAP_HUNT_BATCH_MAX)
        return 0;

    uint64_t thr[GAP_HUNT_BATCH_MAX];
    uint32_t pidx[GAP_HUNT_BATCH_MAX];
    uint32_t qidx[GAP_HUNT_BATCH_MAX];
    uint8_t  phase[GAP_HUNT_BATCH_MAX];
    uint32_t clo[GAP_HUNT_BATCH_MAX];
    uint32_t chi[GAP_HUNT_BATCH_MAX];

    for (uint32_t i = 0; i < K; i++) {
        /* Reset per-fill record slots: the flight struct is reused across
           fills, and stale jump_s/jump_e pairs from earlier windows were
           re-emitted (with the new win_base) whenever both stale offsets
           happened to be prime in a later window — false gaps whose length
           equals the original record's (the "identical gaps" bug). */
        b->jump_n[i] = 0;
        double t = g->min_merit * gh_log_mpz(b->win_base[i]);
        thr[i] = (uint64_t)ceil(t);
        if (thr[i] < 1)
            thr[i] = 1;
        pidx[i] = UINT32_MAX;
        qidx[i] = UINT32_MAX;
        phase[i] = J2_FIND_FIRST;
        clo[i] = 0;
        chi[i] = b->count[i] < C ? b->count[i] : C;
    }

    uint32_t lo[GAP_HUNT_BATCH_MAX], hi[GAP_HUNT_BATCH_MAX];
    uint32_t dcum[GAP_HUNT_BATCH_MAX];
    int any_active = 1;
    while (any_active) {
        uint32_t total = 0;
        for (uint32_t i = 0; i < K; i++) {
            int on = (phase[i] != J2_DONE && chi[i] > clo[i]);
            lo[i] = on ? clo[i] : 0;
            hi[i] = on ? chi[i] : 0;
            dcum[i] = total;
            if (on)
                total += chi[i] - clo[i];
        }
        if (total > 0) {
            uint32_t gathered = 0;
            uint64_t rt0 = g_host_timing ? gh_now_us() : 0;
            /* slot must be the slot submitted below: the gather is async on
               that slot's stream and relies on stream order, not a device
               barrier, to make the staging buffer complete before MR reads
               it (see gpu_fermat_gather_run). */
            if (gpu_fermat_gather_run(g->fermat, slot,
                    gpu_sieve_candidate_buffer(g->sieve, slot),
                    b->cum, lo, hi, dcum, K, g->fermat_limbs,
                    &gathered) != 0 || gathered != total) {
                fprintf(stderr, "[GAP_HUNT] jump2 gather fail: total=%u "
                        "gathered=%u K=%u\n", total, gathered, K);
                return 0;
            }
            uint64_t rt1 = g_host_timing ? gh_now_us() : 0;
            if (gpu_fermat_submit_device(g->fermat, slot,
                    gpu_fermat_gather_buffer(g->fermat),
                    (size_t)total) < 0) {
                fprintf(stderr, "[GAP_HUNT] jump2 submit fail: total=%u\n",
                        total);
                return 0;
            }
            uint64_t rt2 = g_host_timing ? gh_now_us() : 0;
            g_j2_tests += total;
            g_j2_rounds++;
            if (gpu_fermat_collect(g->fermat, slot, flags,
                                   (size_t)total) < 0) {
                fprintf(stderr, "[GAP_HUNT] jump2 collect fail: total=%u\n",
                        total);
                return 0;
            }
            if (g_host_timing) {
                uint64_t rt3 = gh_now_us();
                g_rt_gather_us += rt1 - rt0;
                g_rt_submit_us += rt2 - rt1;
                g_rt_collect_us += rt3 - rt2;
            }
        }

        any_active = 0;
        for (uint32_t i = 0; i < K; i++) {
            if (phase[i] == J2_DONE)
                continue;
            uint32_t n = (chi[i] > clo[i]) ? chi[i] - clo[i] : 0;
            const uint8_t *fl = flags + dcum[i];
            uint32_t base = clo[i];

            if (phase[i] == J2_FIND_FIRST) {
                uint32_t found = UINT32_MAX;
                for (uint32_t j = 0; j < n; j++)
                    if (fl[j]) { found = base + j; break; }
                if (found != UINT32_MAX) {
                    pidx[i] = found;
                    qidx[i] = gh_jump2_find_q(b, i, found, thr[i]);
                    if (qidx[i] == UINT32_MAX) {
                        phase[i] = J2_DONE;
                    } else {
                        phase[i] = J2_JUMP_BACK;
                        uint32_t cl = qidx[i] > C ? qidx[i] - C : 0;
                        clo[i] = (cl > found + 1) ? cl : found + 1;
                        chi[i] = qidx[i];
                    }
                } else {
                    clo[i] = chi[i];
                    chi[i] = (chi[i] + C < b->count[i])
                                 ? chi[i] + C : b->count[i];
                    if (clo[i] >= b->count[i])
                        phase[i] = J2_DONE;
                }
            } else if (phase[i] == J2_JUMP_BACK) {
                uint32_t found = UINT32_MAX;
                for (int32_t j = (int32_t)n - 1; j >= 0; j--)
                    if (fl[j]) { found = base + (uint32_t)j; break; }
                if (found != UINT32_MAX) {
                    pidx[i] = found;
                    qidx[i] = gh_jump2_find_q(b, i, found, thr[i]);
                    if (qidx[i] == UINT32_MAX) {
                        phase[i] = J2_DONE;
                    } else {
                        uint32_t cl = qidx[i] > C ? qidx[i] - C : 0;
                        clo[i] = (cl > found + 1) ? cl : found + 1;
                        chi[i] = qidx[i];
                    }
                } else {
                    if (base == pidx[i] + 1) {
                        /* (p, q) fully composite: find the true endpoint. */
                        phase[i] = J2_FIND_END;
                        clo[i] = qidx[i];
                        chi[i] = (qidx[i] + C < b->count[i])
                                     ? qidx[i] + C : b->count[i];
                        if (chi[i] <= clo[i])
                            phase[i] = J2_DONE;
                    } else {
                        uint32_t old_lo = base;
                        uint32_t new_lo = (old_lo > pidx[i] + 1 + C)
                                              ? old_lo - C : pidx[i] + 1;
                        chi[i] = old_lo;
                        clo[i] = new_lo;
                    }
                }
            } else { /* J2_FIND_END */
                uint32_t found = UINT32_MAX;
                for (uint32_t j = 0; j < n; j++)
                    if (fl[j]) { found = base + j; break; }
                if (found != UINT32_MAX) {
                    if (b->jump_n[i] >= GAP_HUNT_JUMP_CAP) {
                        g_j2_cap_hits++;
                        fprintf(stderr,
                                "[GAP_HUNT] jump2 pair capacity reached: k=%llu "
                                "i=%u cap=%d pairs/win (chain truncated, walk "
                                "stops; GAP_HUNT_JUMP_CAP is a compile-time "
                                "ceiling)\n",
                                (unsigned long long)b->base_k[i], i,
                                GAP_HUNT_JUMP_CAP);
                        return 0;   /* fail-closed */
                    }
                    if (b->jump_n[i] + 1U > g_j2_max_pairs)
                        g_j2_max_pairs = b->jump_n[i] + 1U;
                    b->jump_s[i][b->jump_n[i]] =
                        (uint32_t)b->win_off[i][pidx[i]];
                    b->jump_e[i][b->jump_n[i]] =
                        (uint32_t)b->win_off[i][found];
                    b->jump_n[i]++;
                    pidx[i] = found;
                    qidx[i] = gh_jump2_find_q(b, i, found, thr[i]);
                    if (qidx[i] == UINT32_MAX) {
                        phase[i] = J2_DONE;
                    } else {
                        phase[i] = J2_JUMP_BACK;
                        uint32_t cl = qidx[i] > C ? qidx[i] - C : 0;
                        clo[i] = (cl > found + 1) ? cl : found + 1;
                        chi[i] = qidx[i];
                    }
                } else {
                    clo[i] = chi[i];
                    chi[i] = (chi[i] + C < b->count[i])
                                 ? chi[i] + C : b->count[i];
                    if (clo[i] >= b->count[i])
                        phase[i] = J2_DONE;
                }
            }
            if (phase[i] != J2_DONE)
                any_active = 1;
        }
    }
    return 1;
}

static int gh_batch_fill(struct gh_batch *b, int slot, uint64_t k0,
                         struct gh_ctx *g, uint8_t *flags) {
    uint32_t cum = 0;
    for (uint32_t i = 0; i < (uint32_t)g_batch; i++) {
        uint64_t ht0 = g_host_timing ? gh_now_us() : 0;
        uint64_t k = k0 + i;
        b->base_k[i] = k;
        mpz_set(g->bk, g->b0);
        mpz_addmul_ui(g->bk, g->P, (unsigned long)k);
        mpz_sub_ui(g->wb, g->bk, g->back_limit);
        mpz_set(b->win_base[i], g->wb);
        /* The CRT template prefilter assumes an EVEN base (even t -> even
           value).  P is odd, so b_k alternates parity with k; for odd bases
           the template's shortcut would kill odd interior primes, so the
           prefilter is skipped there (mini-sieve alone, still exact). */
        b->base_even[i] = (uint8_t)(mpz_odd_p(g->wb) ? 0 : 1);

        memset(g->base_limbs, 0, (size_t)g->gpu_limbs * sizeof(uint64_t));
        size_t exported = 0;
        mpz_export(g->base_limbs, &exported, -1, sizeof(uint64_t), 0, 0,
                   g->wb);
        if (exported > (size_t)g->gpu_limbs) {
            fprintf(stderr,
                    "[GAP_HUNT] fill fail: base export %zu > %d limbs "
                    "(k=%llu i=%u)\n",
                    exported, g->gpu_limbs, (unsigned long long)k, i);
            return 0;
        }
        uint64_t first_odd_offset = (g->base_limbs[0] & 1ULL) ? 0U : 1U;
        uint64_t odd_size = 0;
        if (g->interval > first_odd_offset)
            odd_size = (g->interval - first_odd_offset + 1ULL) >> 1;

        int buf = (int)(i & 1);
        if (g_host_timing) {
            uint64_t ht1 = gh_now_us();
            g_ht_setup += ht1 - ht0;
            ht0 = ht1;
        }
        if (!gpu_sieve_mark_from_base(g->sieve, odd_size, first_odd_offset,
                                      g->base_limbs, g->gpu_limbs, buf,
                                      g->primes, g->inv_p, g->prime_count,
                                      NULL, 0)) {
            fprintf(stderr, "[GAP_HUNT] fill fail: mark "
                    "(k=%llu i=%u odd=%llu fo=%llu)\n",
                    (unsigned long long)k, i,
                    (unsigned long long)odd_size,
                    (unsigned long long)first_odd_offset);
            return 0;
        }

        if (g_host_timing) {
            uint64_t ht1 = gh_now_us();
            g_ht_mark += ht1 - ht0;
            ht0 = ht1;
        }
        uint64_t class_mask =
            g->quarter ? halfclass_visible_mask() : UINT64_MAX;
        /* The extract filter classes VALUES by (base + offset) mod 60 —
           pass the real window-base residue, never 0. */
        uint32_t base_mod60 = (uint32_t)mpz_fdiv_ui(g->wb, 60);
        /* Extension band [tail_start, interval): extracted ALL-classes so
           every prime there is a chain anchor (containers always exist). */
        uint64_t lo_tail_odd = 0;
        if (g->tail_start > first_odd_offset)
            lo_tail_odd = (g->tail_start - first_odd_offset + 1ULL) >> 1;
        if (lo_tail_odd > odd_size)
            lo_tail_odd = odd_size;
        if (!g->quarter)
            lo_tail_odd = odd_size;   /* full-class: one whole-range pass */

        uint64_t *d_cands = NULL;
        unsigned int nc = 0;
        if (!gpu_sieve_extract_pack_device_range_ex(
                g->sieve, odd_size, first_odd_offset, 0, lo_tail_odd,
                buf, slot, g->base_limbs, g->fermat_limbs, &d_cands,
                b->win_off[i], &nc, base_mod60, class_mask,
                g->region_start, cum)) {
            fprintf(stderr, "[GAP_HUNT] fill fail: extract "
                    "(k=%llu i=%u lo_tail_odd=%llu)\n",
                    (unsigned long long)k, i,
                    (unsigned long long)lo_tail_odd);
            return 0;
        }
        if ((uint64_t)nc > g->capacity) {
            fprintf(stderr, "[GAP_HUNT] fill fail: nc=%u > capacity=%llu "
                    "(k=%llu i=%u)\n", nc,
                    (unsigned long long)g->capacity,
                    (unsigned long long)k, i);
            return 0;
        }
        if (cum + nc > g->cap) {
            fprintf(stderr, "[GAP_HUNT] fill fail: cum=%u + nc=%u > %llu "
                    "(k=%llu i=%u)\n", cum, nc,
                    (unsigned long long)g->cap,
                    (unsigned long long)k, i);
            return 0;
        }
        b->count[i] = nc;
        b->cum[i] = cum;
        cum += nc;

        if (g->quarter && lo_tail_odd < odd_size) {
            unsigned int nc2 = 0;
            if (!gpu_sieve_extract_pack_device_range_ex(
                    g->sieve, odd_size, first_odd_offset, lo_tail_odd,
                    odd_size, buf, slot, g->base_limbs, g->fermat_limbs,
                    &d_cands, b->win_off[i] + nc, &nc2,
                    base_mod60, UINT64_MAX, 0, cum))
                return 0;
            if ((uint64_t)(nc + nc2) > g->capacity)
                return 0;
            if (cum + nc2 > g->cap)
                return 0;
            b->count[i] = nc + nc2;
            cum += nc2;
        }
        if (g_host_timing) {
            uint64_t ht1 = gh_now_us();
            g_ht_extract += ht1 - ht0;
            ht0 = ht1;
        }
    }
    b->cum[g_batch] = cum;
    b->n_windows = (uint32_t)g_batch;
    b->total = cum;
    b->slot = slot;
    uint64_t ht_s = g_host_timing ? gh_now_us() : 0;
    if (cum == 0)
        return 1;               /* nothing to MR-test; process as empty */
    if (g->jump2) {
        /* Chunk-parallel backward search: exact Kehrig chain semantics but
           each step is one batched MR round across all windows. */
        if (!gh_jump2_scan(b, slot, g, flags)) {
            fprintf(stderr, "[GAP_HUNT] fill fail: jump2 scan (k0=%llu)\n",
                    (unsigned long long)k0);
            return 0;
        }
        if (g_host_timing) g_ht_submit += gh_now_us() - ht_s;
        return 1;
    }
    if (g->jump) {
        /* Jump mode: per-window serial MR chain on the GPU (Kehrig-style),
           ~1.5 tests per prime instead of all survivors.  Synchronous, so
           the batch is processed immediately (active stays 0). */
        for (uint32_t i = 0; i < (uint32_t)g_batch; i++) {
            g->jump_thresh[i] = (uint64_t)ceil(
                g->min_merit * gh_log_mpz(b->win_base[i]));
            if (g->jump_thresh[i] < 1)
                g->jump_thresh[i] = 1;
        }
        uint64_t *d_batch = gpu_sieve_candidate_buffer(g->sieve, slot);
        const uint64_t *d_offs = gpu_sieve_device_offsets(g->sieve);
        if (!d_offs ||
            gpu_fermat_jump_scan(g->fermat, d_batch, d_offs,
                                 b->count, b->cum, g->jump_thresh,
                                 g->fermat_limbs, (uint32_t)g_batch,
                                 g->jump_gaps, g->jump_n) != 0) {
            fprintf(stderr, "[GAP_HUNT] fill fail: jump scan (k0=%llu)\n",
                    (unsigned long long)k0);
            return 0;
        }
        for (uint32_t i = 0; i < (uint32_t)g_batch; i++) {
            if (g->jump_n[i] > GAP_HUNT_JUMP_CAP) {
                fprintf(stderr, "[GAP_HUNT] fill fail: jump overflow "
                        "(k=%llu i=%u n=%u)\n",
                        (unsigned long long)(k0 + i), i, g->jump_n[i]);
                return 0;       /* fail-closed */
            }
            b->jump_n[i] = g->jump_n[i];
            for (uint32_t jj = 0; jj < b->jump_n[i]; jj++) {
                uint64_t pair = g->jump_gaps[
                    (size_t)i * GAP_HUNT_JUMP_CAP + jj];
                b->jump_s[i][jj] = (uint32_t)(pair >> 32);
                b->jump_e[i][jj] = (uint32_t)pair;
            }
        }
        if (g_host_timing) g_ht_submit += gh_now_us() - ht_s;
        return 1;
    }
    uint64_t *d_batch = gpu_sieve_candidate_buffer(g->sieve, slot);
    if (gpu_fermat_submit_device(g->fermat, slot, d_batch, (size_t)cum) != 0) {
        fprintf(stderr, "[GAP_HUNT] fill fail: fermat submit "
                "(slot=%d cum=%u k0=%llu)\n", slot, cum,
                (unsigned long long)k0);
        return 0;
    }
    b->active = 1;
    if (g_host_timing) g_ht_submit += gh_now_us() - ht_s;
    return 1;
}

/* Report one verified gap in `<gap> <merit> <startprime>` format. */
static void gh_report_gap(FILE *out, uint64_t *gaps, double *best,
                          uint64_t gap_u, double merit,
                          const char *start_dec, const char *end_dec) {
    fprintf(stderr,
            "[GAP_HUNT] gap=%llu merit=%.6f start=%s end=%s\n",
            (unsigned long long)gap_u, merit,
            start_dec ? start_dec : "?",
            end_dec ? end_dec : "?");
    if (out) {
        fprintf(out, "%llu %.6f %s\n",
                (unsigned long long)gap_u, merit,
                start_dec ? start_dec : "?");
        fflush(out);
    }
    if (merit > *best)
        *best = merit;
    (*gaps)++;
}

/* Chain consecutive primes within each window of a collected flight,
   report every BPSW-verified gap with merit >= min_merit in the
   `<gap> <merit> <startprime>` format, and advance the counters.
   Merit is the TRUE record merit: gap / ln(start).
   QUARTER_CLASS: only pairs inside the all-class back region are true
   consecutive pairs (reported directly); every other qualifying pair is
   resolved on demand — its interior is mini-sieved in the hidden classes
   (CRT template as prefilter) and the emitted true gaps are reported. */
static void gh_batch_process(struct gh_batch *b, const uint8_t *flags,
                             const struct gap_hunt_config *cfg,
                             const struct gh_ctx *g,
                             uint64_t *windows,
                             uint64_t *gaps, double *best,
                             FILE *out, mpz_t p1, mpz_t p2,
                             mpz_t last_prime, int *have_last) {
    if (g->jump || g->jump2) {
        /* Jump mode: the device walk already certified the chains; only
           BPSW-verify the reported endpoints and emit (the walk tests the
           same MR2 the batch path uses, so the gap set is identical). */
        for (uint32_t i = 0; i < b->n_windows; i++) {
            for (uint32_t jj = 0; jj < b->jump_n[i]; jj++) {
                uint64_t off_s = b->jump_s[i][jj];
                uint64_t off_e = b->jump_e[i][jj];
                mpz_set(p1, b->win_base[i]);
                mpz_add_ui(p1, p1, off_s);
                mpz_set(p2, b->win_base[i]);
                mpz_add_ui(p2, p2, off_e);
                double merit = (double)(off_e - off_s) / gh_log_mpz(p1);
                if (merit < cfg->min_merit)
                    continue;   /* exact gate (threshold was on wb) */
                if (baillie_psw_test(p1) && baillie_psw_test(p2)) {
                    char *sd = mpz_get_str(NULL, 10, p1);
                    char *ed = mpz_get_str(NULL, 10, p2);
                    gh_report_gap(out, gaps, best,
                                  (uint64_t)(off_e - off_s), merit,
                                  sd, ed);
                    free(sd);
                    free(ed);
                }
            }
        }
        *windows += b->n_windows;
        return;
    }
    for (uint32_t i = 0; i < b->n_windows; i++) {
        /* Windows are P apart (NOT contiguous): chain only within the
           window; its first prime has an unknown predecessor. */
        *have_last = 0;
        uint64_t prev_off = 0;
        /* Bounded dump hook (diagnostics only).  GHDBG_DUMP_K=<k> writes EVERY
           candidate of that ONE window (index, offset, MR verdict) into
           GHDBG_DUMP_FILE (default /tmp/ghdump_<k>.txt), hard-capped at
           GHDUMP_CAP lines.  That is what a missed gap must be diagnosed
           with: is the endpoint absent from the candidate list, or present
           with a wrong flag? */
        const char *dump_k_env = getenv("GHDBG_DUMP_K");
        FILE *dump_fp = NULL;
        long dump_lines = 0;
        if (dump_k_env && dump_k_env[0] &&
            (unsigned long long)b->base_k[i] ==
                strtoull(dump_k_env, NULL, 10)) {
            const char *df = getenv("GHDBG_DUMP_FILE");
            char dpath[512];
            if (!df || !df[0]) {
                snprintf(dpath, sizeof(dpath), "/tmp/ghdump_%s.txt", dump_k_env);
                df = dpath;
            }
            dump_fp = fopen(df, "w");
            if (dump_fp)
                fprintf(dump_fp, "[GHDUMP] win k=%llu count=%u\n",
                        (unsigned long long)b->base_k[i], b->count[i]);
        }
        for (uint32_t j = 0; j < b->count[i]; j++) {
            uint8_t is_prime = flags[b->cum[i] + j];
            if (dump_fp && dump_lines < GHDUMP_CAP) {
                fprintf(dump_fp, "[GHDUMP]   j=%u off=%llu prime=%u\n", j,
                        (unsigned long long)b->win_off[i][j], is_prime);
                dump_lines++;
            }
            if (!is_prime)
                continue;
            uint64_t off = b->win_off[i][j];
            if (dump_fp && dump_lines < GHDUMP_CAP) {
                fprintf(dump_fp, "[GHDUMP]   prime off=%llu\n",
                        (unsigned long long)off);
                dump_lines++;
            }
            mpz_set(p1, b->win_base[i]);
            mpz_add_ui(p1, p1, off);
            if (*have_last) {
                mpz_sub(p2, p1, last_prime);
                /* True merit (record convention): gap / ln(start). */
                double merit = mpz_get_d(p2) / gh_log_mpz(last_prime);
                if (merit >= cfg->min_merit) {
                    int direct = !g->quarter ||
                        (prev_off < g->region_start &&
                         off < g->region_start) ||
                        (prev_off >= g->tail_start &&
                         off >= g->tail_start);
                    if (direct) {
                        if (baillie_psw_test(p1) &&
                            baillie_psw_test(last_prime)) {
                            if (getenv("GAPDEBUG")) {
                                fprintf(stderr,
                                        "[GHDBG] k=%llu off_prev=%llu "
                                        "off=%llu direct=1\n",
                                        (unsigned long long)b->base_k[i],
                                        (unsigned long long)prev_off,
                                        (unsigned long long)off);
                            }
                            char *sd = mpz_get_str(NULL, 10, last_prime);
                            char *ed = mpz_get_str(NULL, 10, p1);
                            gh_report_gap(out, gaps, best,
                                          (uint64_t)mpz_get_ui(p2), merit,
                                          sd, ed);
                            free(sd);
                            free(ed);
                        }
                    } else {
                        /* Visible pair over the covered region: resolve the
                           hidden classes in the interior, BPSW the emitted
                           endpoints, report the true gaps. */
                        struct gap_result *res = NULL;
                        uint32_t rc = 0;
                        const struct halfclass_tpl *t =
                            (g->quarter && b->base_even[i]) ? &g->tpl : NULL;
                        if (halfclass_resolve_gap_ex(
                                b->win_base[i], prev_off, off,
                                g->owned_limit, cfg->min_merit,
                                t, &res, &rc) && rc > 0) {
                            if (getenv("GAPDEBUG")) {
                                fprintf(stderr,
                                        "[QHDBG] k=%llu base_even=%u "
                                        "off_a=%llu off_b=%llu emitted=%u\n",
                                        (unsigned long long)b->base_k[i],
                                        (unsigned)b->base_even[i],
                                        (unsigned long long)prev_off,
                                        (unsigned long long)off, rc);
                            }
                            mpz_t rs, re;
                            mpz_init(rs);
                            mpz_init(re);
                            for (uint32_t r = 0; r < rc; r++) {
                                if (getenv("GAPDEBUG")) {
                                    fprintf(stderr,
                                            "[GHDBG] k=%llu off_p1=%llu "
                                            "off_p2=%llu direct=0\n",
                                            (unsigned long long)b->base_k[i],
                                            (unsigned long long)res[r].offset_p1,
                                            (unsigned long long)res[r].offset_p2);
                                }
                                mpz_set(rs, b->win_base[i]);
                                mpz_add_ui(rs, rs, res[r].offset_p1);
                                mpz_set(re, b->win_base[i]);
                                mpz_add_ui(re, re, res[r].offset_p2);
                                double tm =
                                    (double)res[r].gap_length /
                                    gh_log_mpz(rs);
                                if (baillie_psw_test(rs) &&
                                    baillie_psw_test(re)) {
                                    char *sd = mpz_get_str(NULL, 10, rs);
                                    char *ed = mpz_get_str(NULL, 10, re);
                                    gh_report_gap(out, gaps, best,
                                                  res[r].gap_length, tm,
                                                  sd, ed);
                                    free(sd);
                                    free(ed);
                                }
                            }
                            mpz_clear(rs);
                            mpz_clear(re);
                            gap_detection_free_results(res);
                        }
                    }
                }
            }
            mpz_set(last_prime, p1);
            *have_last = 1;
            prev_off = off;
        }
        if (dump_fp) {
            fprintf(dump_fp, "[GHDUMP] end (lines=%ld, cap=%d)\n",
                    dump_lines, GHDUMP_CAP);
            fclose(dump_fp);
            dump_fp = NULL;
        }
        (*windows)++;
    }
}
#endif /* WITH_CUDA (helpers) */

/* Startup advisory for the jump/jump2 per-window pair ceiling.  The chain takes
   at most ~2 x odd_interval_size / thr steps (offsets are in ODD units, thr is
   in ADDERS), measured slack 2-4.6x, so this is an upper bound rather than a
   prediction.  Warning when the bound passes cap/2 turns "the walk dies after
   hours" into an announcement before the first window. */
static void gh_jump_capacity_advisory(double min_merit, double thr,
                                      double odd_span) {
    if (!(thr > 0.0) || odd_span <= 0.0)
        return;
    double est = 2.0 * odd_span / thr;
    if (est <= (double)GAP_HUNT_JUMP_CAP / 2.0)
        return;
    fprintf(stderr,
            "[GAP_HUNT] WARNING: min-merit %.4f gives a step threshold of %.0f "
            "adders over a %.0f-offset window, i.e. up to ~%.0f chain pairs per "
            "window against a capacity of %d. If any window exceeds it the walk "
            "STOPS (fail-closed). Raise --gap-hunt-min-merit, or rebuild with a "
            "larger GAP_HUNT_JUMP_CAP.\n",
            min_merit, thr, odd_span, est, GAP_HUNT_JUMP_CAP);
}
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
        const char *qv = getenv("GAP_HUNT_QUARTER");
        if (qv && qv[0] && atoi(qv) > 0)
            g_quarter = 1;
        const char *km = getenv("GAP_HUNT_KMAX");
        if (km && km[0])
            g_kmax = strtoull(km, NULL, 10);
        const char *jv = getenv("GAP_HUNT_JUMP");
        if (jv && jv[0] && atoi(jv) > 0)
            g_jump = 1;
        /* GAP_HUNT_JUMP2 (chunk-parallel Kehrig chain) is ON by default since
           2026-09-15: measured +169% at shift507 (888 -> 2386 win/s) and +300%
           at shift1017 (170 -> 677 win/s) vs the full-scan batch walk, with
           byte-identical gap sets (parity verified at 507/998/1017 and by
           test_gap_hunt).  GAP_HUNT_JUMP2=0 restores the full-scan walk. */
        const char *j2v = getenv("GAP_HUNT_JUMP2");
        if (j2v && j2v[0] && atoi(j2v) <= 0)
            g_jump2 = 0;
        else
            g_jump2 = 1;
        const char *j2c = getenv("GAP_HUNT_JUMP2_CHUNK");
        if (j2c && j2c[0]) {
            int c = atoi(j2c);
            if (c >= 8 && c <= 512)
                g_jump2_chunk = c;
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
    {
        /* Scan past the nominal window end: real gaps whose high endpoint
           lies just beyond it are still found and reported (the boundary is
           an artifact).  In quarter mode the extension also guarantees a
           visible container for every in-window gap.  24·logbase ≈ 6 visible
           spacings in the uncovered region. */
        uint64_t ext = (uint64_t)(24.0 * logbase);
        interval += ext;
    }
    if (g_quarter) {
        halfclass_set_quarter(1);
        if (!crt_runtime_build_template(&rt)) {
            fprintf(stderr, "[GAP_HUNT] template build failed\n");
            crt_runtime_free(&rt);
            return 1;
        }
    }
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
    /* Geometry dump (GAPDEBUG=1): window k covers
       [b0 + k*P - back_limit, b0 + k*P - back_limit + interval), so a
       reported start prime maps back with k = (p + back_limit - b0) / P.
       Needed to reproduce a specific window from its emitted gap. */
    if (getenv("GAPDEBUG")) {
        char *b0s = mpz_get_str(NULL, 10, b0);
        char *ps = mpz_get_str(NULL, 10, P);
        fprintf(stderr, "[GHDBG] geometry: back_limit=%llu interval=%llu "
                        "window=%llu n_primes=%u shift=%u\n",
                (unsigned long long)back_limit,
                (unsigned long long)interval,
                (unsigned long long)rt.window, rt.n_primes, rt.shift);
        fprintf(stderr, "[GHDBG] b0=%s\n", b0s);
        fprintf(stderr, "[GHDBG] P=%s\n", ps);
        free(b0s);
        free(ps);
    }

    /* GAP_HUNT_TIMING=1 must also turn on the gpu_sieve CUDA-event accounting,
       which is armed at gpu_sieve_init() below, so it has to be exported
       BEFORE that init (otherwise the kernel line reports zeros while the
       host line reports real numbers). */
    {
        const char *ht = getenv("GAP_HUNT_TIMING");
        if (ht && ht[0] && ht[0] != '0') {
            setenv("GPU_SIEVE_TIMING", "1", 1);
        }
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
    if (g_jump) {
        if (gpu_fermat_jump_alloc(fermat, (uint32_t)g_batch,
                                  GAP_HUNT_JUMP_CAP) != 0) {
            fprintf(stderr, "[GAP_HUNT] jump buffer alloc failed\n");
            free(base_limbs);
            gpu_sieve_destroy(gpu_sieve);
            gpu_adapter_free(gpu);
            sieve_core_free(&sieve);
            mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
            crt_runtime_free(&rt);
            return 1;
        }
    }
    if (g_jump2) {
        if (gpu_fermat_gather_alloc(fermat, (uint32_t)g_batch,
                                    (uint32_t)g_jump2_chunk,
                                    gpu_fermat_get_limbs(fermat)) != 0) {
            fprintf(stderr, "[GAP_HUNT] jump2 gather alloc failed\n");
            free(base_limbs);
            gpu_sieve_destroy(gpu_sieve);
            gpu_adapter_free(gpu);
            sieve_core_free(&sieve);
            mpz_clears(target, b0, P, p1, p2, last_prime, NULL);
            crt_runtime_free(&rt);
            return 1;
        }
    }
    base_limbs = (uint64_t *)calloc((size_t)gpu_limbs, sizeof(uint64_t));
    size_t win_cap = (size_t)sieve.candidate_capacity;
    /* Batch/round guard: one chain round submits up to g_batch x survivors
       candidates and the MR context must hold them -- the fill fails closed
       otherwise.  The bound comes from the CONTEXT (gpu_fermat_max_batch),
       never from the compile-time macro: on 2026-09-17 a stale gpu_adapter.o
       still carried the old 160000 while this file guarded against the newer
       320000, so every oversize batch was truncated silently and its tail
       windows lost their verdicts (the full-scan miss investigated that day).
       NOTE: win_cap is the per-window CAPACITY (an upper bound), not the
       survivor count; clamping with it would shrink the batch to a handful of
       windows (measured: batch 6 at shift507, 3 at shift1017 -> 420 and
       67 win/s).  Use the observed survivor density instead: a CRT cover
       leaves ~1/8 of the odd slots as survivors, so odd_interval_size/8 is a
       realistic estimate with margin, and the hard `cum + nc > cap` check in
       the fill remains the final guard. */
    uint64_t cap_real = 0;   /* real per-submit cap of the MR context */
    uint64_t cum_cap = 0;    /* bound on the window-major candidate index */
    {
        uint64_t est = (uint64_t)(odd_interval_size / 8U);
        cap_real = (uint64_t)gpu_fermat_max_batch(fermat);
        if (cap_real == 0)
            cap_real = GPU_ADAPTER_MAX_BATCH;
        if (est < 1) est = 1;
        /* The chain paths never submit the window-major head batch: jump2
           gathers K x chunk candidates per round and the serial jump walks one
           window at a time.  Their only bound is the capacity of the flags and
           offsets arrays (g_batch x win_cap), so the MR cap must NOT shrink
           their batch.  Only the plain path submits the head batch whole and
           is therefore auto-reduced to fit the MR context. */
        int head_batch = !(g_jump2 || g_jump);
        if (head_batch && (uint64_t)g_batch * est > cap_real) {
            int nb = (int)(cap_real / est);
            if (nb < 1) nb = 1;
            if (nb != g_batch) {
                fprintf(stderr, "[GAP_HUNT] batch reduced %d -> %d "
                                "(est survivors/window=%llu, mrcap=%llu; "
                                "chain modes avoid this bound)\n",
                        g_batch, nb, (unsigned long long)est,
                        (unsigned long long)cap_real);
                g_batch = nb;
            }
        }
        cum_cap = head_batch ? cap_real
                             : (uint64_t)g_batch * (uint64_t)win_cap;
    }
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
            "sieve=%u batch=%d mrcap=%llu cumcap=%llu vram_est=%lluMB quarter=%d "
            "jump=%d "
            "jump2=%d "
            "jump2_chunk=%d "
            "kmax=%llu min_merit=%.6f "
            "k0=%llu device=%d\n",            rt.shift, rt.n_primes,
            (double)mpz_sizeinbase(P, 2),
            (unsigned long long)rt.window, sieve_primes, g_batch,
            (unsigned long long)cap_real, (unsigned long long)cum_cap,
            (unsigned long long)(2U * odd_interval_size * (uint64_t)g_batch *
                                 (uint64_t)gpu_fermat_get_limbs(fermat) * 8U /
                                 1000000U),
            g_quarter,
            g_jump, g_jump2, g_jump2_chunk,
            (unsigned long long)g_kmax, cfg->min_merit,
            (unsigned long long)k, cfg->device);

    /* A low merit threshold is a legitimate lever (more gaps, more records per
       hour), but it also makes the per-window chain longer.  Say so BEFORE the
       walk starts rather than letting the pair ceiling kill it hours in. */
    if (g_jump2 || g_jump) {
        double ln_base = gh_log_mpz(b0);
        if (ln_base > 0.0) {
            gh_jump_capacity_advisory(cfg->min_merit,
                                      cfg->min_merit * ln_base,
                                      (double)odd_interval_size);
        }
    }

    uint64_t windows = 0, gaps_reported = 0;
    double best_merit = 0.0;

    /* Walk-rate timing (monotonic): the periodic line reports the
       instantaneous windows/s since the last save and the average since
       the walk started. */
    struct timespec t0, t_last;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    t_last = t0;
    uint64_t win_last = 0;

    /* Optional per-stage GPU accounting on the periodic line
       (GPU_SIEVE_TIMING=1).  Mark/extract come from the gpu_sieve CUDA events,
       MR from the fermat context events; all three are cumulative counters, so
       the tick prints the DELTA divided by the windows in the tick -> us/window
       and the share of the tick's wall time.  Diagnostics only: off by default
       so production walk rates are not perturbed. */
    const char *stage_timing_env = getenv("GPU_SIEVE_TIMING");
    int stage_timing = (stage_timing_env && stage_timing_env[0] &&
                        stage_timing_env[0] != '0') ? 1 : 0;
    /* GAP_HUNT_TIMING=1 turns on BOTH the GPU kernel line and the host stage
       line (host stages alone cannot explain a walk rate without them). */
    const char *host_timing_env = getenv("GAP_HUNT_TIMING");
    if (host_timing_env && host_timing_env[0] && host_timing_env[0] != '0') {
        g_host_timing = 1;
        stage_timing = 1;
    }
    uint64_t acc_mark_prev = 0, acc_ext_prev = 0, acc_mr_prev = 0;
    if (stage_timing) {
        acc_mark_prev = gpu_sieve_accounted_mark_us(gpu_sieve);
        acc_ext_prev = gpu_sieve_accounted_extract_us(gpu_sieve);
        acc_mr_prev = gpu_fermat_accounted_us(fermat);
    }

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
    g.cap = cum_cap;
    g.interval = interval;
    g.odd_interval_size = odd_interval_size;
    g.back_limit = back_limit;
    g.primes = sieve.small_primes;
    g.inv_p = sieve.inv_p;
    g.prime_count = sieve.small_primes_count;
    mpz_init(g.bk);
    mpz_init(g.wb);
    mpz_init(g.b0);
    mpz_init(g.P);
    mpz_set(g.b0, b0);
    mpz_set(g.P, P);
    g.quarter = g_quarter;
    g.jump = g_jump;
    g.jump2 = g_jump2;
    g.jump2_chunk = g_jump2_chunk;
    g.min_merit = cfg->min_merit;
    g.region_start = g_quarter ? back_limit : 0;
    g.tail_start = back_limit + rt.window;
    g.owned_limit = interval;   /* the whole scan is ours */
    if (g_quarter) {
        g.tpl.bits = rt.template;
        g.tpl.words = rt.template_words;
        g.tpl.base_off = (int64_t)back_limit;
        g.tpl.window = rt.window;
    } else {
        memset(&g.tpl, 0, sizeof(g.tpl));
    }

    for (int i = 0; i < g_batch; i++) {
        A.win_off[i] = offsets + (size_t)i * win_cap;
        B.win_off[i] =
            offsets + ((size_t)g_batch + (size_t)i) * win_cap;
    }

    uint64_t next_k = k;
    uint64_t last_save = windows;
    struct gh_batch *fls[2] = { &A, &B };
    int turn = 0;

    /* Alternate the two fermat slots so one flight's host processing
       overlaps the other flight's in-flight MR kernel. */
    while (!g_stop) {
        struct gh_batch *fl = fls[turn];

        if (fl->active) {
            if (gpu_fermat_collect(fermat, fl->slot, flags,
                                   (size_t)fl->total) < 0) {
                fprintf(stderr, "[GAP_HUNT] GPU collect failed\n");
                break;
            }
            uint64_t ht_p = g_host_timing ? gh_now_us() : 0;
            gh_batch_process(fl, flags, cfg, &g, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
            if (g_host_timing) g_ht_process += gh_now_us() - ht_p;
            fl->active = 0;
            fl->n_windows = 0;
            fl->total = 0;
        }

        /* Refill and resubmit this flight so it is in flight while the other
           is collected next; empty batches (cum==0) process inline. */
        if (!g_stop && !(g_kmax && next_k >= g_kmax)) {
            if (!gh_batch_fill(fl, turn, next_k, &g, flags)) {
                fprintf(stderr, "[GAP_HUNT] batch fill failed (k=%llu)\n",
                        (unsigned long long)next_k);
                break;
            }
            next_k += (uint64_t)g_batch;
            if (!fl->active) {
                gh_batch_process(fl, flags, cfg, &g, &windows,
                                 &gaps_reported, &best_merit, out,
                                 p1, p2, last_prime, &have_last);
                fl->n_windows = 0;
            }
        } else if (!A.active && !B.active) {
            g_stop = 1;         /* KMAX reached and both flights drained */
        }

        if (windows - last_save >= 1024ULL) {
            struct timespec tn;
            clock_gettime(CLOCK_MONOTONIC, &tn);
            double dt_avg = (double)(tn.tv_sec - t0.tv_sec) +
                            (double)(tn.tv_nsec - t0.tv_nsec) * 1e-9;
            double dt_now = (double)(tn.tv_sec - t_last.tv_sec) +
                            (double)(tn.tv_nsec - t_last.tv_nsec) * 1e-9;
            double win_s_now =
                dt_now > 0.0 ? (double)(windows - win_last) / dt_now : 0.0;
            double win_s_avg =
                dt_avg > 0.0 ? (double)windows / dt_avg : 0.0;
            uint64_t dw_tick = windows - win_last;   /* windows in THIS tick */
            t_last = tn;
            win_last = windows;
            /* CONSERVATIVE WATERMARK: a flight that is still in the air has
               been FILLED but not yet COLLECTED, so its windows are not in the
               output file yet.  Saving next_k here would claim them as done and
               a hard kill (SIGKILL/OOM/power, i.e. anything that skips the
               drain below) would then skip up to g_batch windows forever.
               Save the oldest unprocessed window instead: after a hard kill the
               walk redoes at most one batch, which is harmless (the same gaps
               are re-emitted), while a skip is a lost lottery ticket. */
            uint64_t k_safe = next_k;
            if (A.active && A.base_k[0] < k_safe)
                k_safe = A.base_k[0];
            if (B.active && B.base_k[0] < k_safe)
                k_safe = B.base_k[0];
            if (getenv("GAPDEBUG"))
                fprintf(stderr, "[GAPDEBUG] watermark next_k=%llu "
                        "A(act=%d n=%u k0=%llu) B(act=%d n=%u k0=%llu) "
                        "-> k_safe=%llu\n",
                        (unsigned long long)next_k, A.active, A.n_windows,
                        (unsigned long long)A.base_k[0],
                        B.active, B.n_windows,
                        (unsigned long long)B.base_k[0],
                        (unsigned long long)k_safe);
            save_state(cfg->state_path, k_safe, last_prime, have_last);
            last_save = windows;
            fprintf(stderr,
                    "[GAP_HUNT] k=%llu windows=%llu gaps=%llu best_merit=%.6f "
                    "win_s=%.1f win_s_avg=%.1f jump2_pairs_max=%u/%d "
                    "cap_hits=%llu\n",
                    (unsigned long long)next_k,
                    (unsigned long long)windows,
                    (unsigned long long)gaps_reported, best_merit,
                    win_s_now, win_s_avg,
                    g_j2_max_pairs, GAP_HUNT_JUMP_CAP,
                    (unsigned long long)g_j2_cap_hits);
            if (stage_timing) {
                uint64_t dw = dw_tick;
                uint64_t mark_now = gpu_sieve_accounted_mark_us(gpu_sieve);
                uint64_t ext_now = gpu_sieve_accounted_extract_us(gpu_sieve);
                uint64_t mr_now = gpu_fermat_accounted_us(fermat);
                double per_win = 1e6 / (win_s_now > 0.0 ? win_s_now : 1.0);
                double m_us = dw ? (double)(mark_now - acc_mark_prev) / (double)dw : 0.0;
                double e_us = dw ? (double)(ext_now - acc_ext_prev) / (double)dw : 0.0;
                double r_us = dw ? (double)(mr_now - acc_mr_prev) / (double)dw : 0.0;
                fprintf(stderr,
                        "[GAP_HUNT] stage us/window: total=%.0f mark=%.2f "
                        "extract=%.2f mr=%.2f (mark=%.0f%% extract=%.0f%% "
                        "mr=%.0f%%; totals mark=%llu extract=%llu mr=%llu us)\n",
                        per_win, m_us, e_us, r_us,
                        per_win > 0 ? 100.0 * m_us / per_win : 0.0,
                        per_win > 0 ? 100.0 * e_us / per_win : 0.0,
                        per_win > 0 ? 100.0 * r_us / per_win : 0.0,
                        (unsigned long long)mark_now,
                        (unsigned long long)ext_now,
                        (unsigned long long)mr_now);
                acc_mark_prev = mark_now;
                acc_ext_prev = ext_now;
                acc_mr_prev = mr_now;
            }
            if (g_host_timing) {
                uint64_t dw = dw_tick;
                double setup = dw ? (double)g_ht_setup / (double)dw : 0.0;
                double mk = dw ? (double)g_ht_mark / (double)dw : 0.0;
                double ex = dw ? (double)g_ht_extract / (double)dw : 0.0;
                double sb = dw ? (double)g_ht_submit / (double)dw : 0.0;
                double pr = dw ? (double)g_ht_process / (double)dw : 0.0;
                double per_win = 1e6 / (win_s_now > 0.0 ? win_s_now : 1.0);
                fprintf(stderr,
                        "[GAP_HUNT] host us/window: total=%.0f setup=%.1f "
                        "mark=%.1f extract=%.1f submit=%.1f detect=%.1f "
                        "(%.0f%% host; GAP_HUNT_TIMING)\n",
                        per_win, setup, mk, ex, sb, pr,
                        per_win > 0
                            ? 100.0 * (setup + mk + ex + sb + pr) / per_win
                            : 0.0);
                g_ht_setup = g_ht_mark = g_ht_extract = g_ht_submit =
                    g_ht_process = 0;
            }
            if (stage_timing) {
                uint64_t dw = dw_tick;
                uint64_t rounds_tick = g_j2_rounds;
                fprintf(stderr,
                        "[GAP_HUNT] chain: tests/window=%.1f "
                        "rounds/1k-win=%.1f (jump2 only)\n",
                        dw ? (double)g_j2_tests / (double)dw : 0.0,
                        dw ? 1000.0 * (double)rounds_tick / (double)dw : 0.0);
                if (g_host_timing && rounds_tick > 0) {
                    fprintf(stderr,
                            "[GAP_HUNT] round us (mean of %llu): "
                            "gather=%.1f submit=%.1f collect=%.1f total=%.1f\n",
                            (unsigned long long)rounds_tick,
                            (double)g_rt_gather_us / (double)rounds_tick,
                            (double)g_rt_submit_us / (double)rounds_tick,
                            (double)g_rt_collect_us / (double)rounds_tick,
                            (double)(g_rt_gather_us + g_rt_submit_us +
                                     g_rt_collect_us) / (double)rounds_tick);
                }
                g_j2_tests = 0;
                g_j2_rounds = 0;
                g_rt_gather_us = g_rt_submit_us = g_rt_collect_us = 0;
            }
        }

        turn ^= 1;
    }

    /* Drain the in-flight flights so no window is lost. */
    if (A.active) {
        if (gpu_fermat_collect(fermat, A.slot, flags, (size_t)A.total) >= 0)
            gh_batch_process(&A, flags, cfg, &g, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
        A.active = 0;
    }
    if (B.active) {
        if (gpu_fermat_collect(fermat, B.slot, flags, (size_t)B.total) >= 0)
            gh_batch_process(&B, flags, cfg, &g, &windows,
                             &gaps_reported, &best_merit, out,
                             p1, p2, last_prime, &have_last);
        B.active = 0;
    }
    gh_batch_clear(&A);
    gh_batch_clear(&B);
    mpz_clears(g.bk, g.wb, g.b0, g.P, NULL);

    save_state(cfg->state_path, next_k, last_prime, have_last);
    {
        struct timespec tn;
        clock_gettime(CLOCK_MONOTONIC, &tn);
        double dt = (double)(tn.tv_sec - t0.tv_sec) +
                    (double)(tn.tv_nsec - t0.tv_nsec) * 1e-9;
        double win_s = dt > 0.0 ? (double)windows / dt : 0.0;
        fprintf(stderr,
                "[GAP_HUNT] stopped: windows=%llu gaps=%llu best_merit=%.6f "
                "next_k=%llu win_s_avg=%.1f jump2_pairs_max=%u cap=%d "
                "jump2_cap_hits=%llu\n",
                (unsigned long long)windows, (unsigned long long)gaps_reported,
                best_merit, (unsigned long long)next_k, win_s,
                g_j2_max_pairs, GAP_HUNT_JUMP_CAP,
                (unsigned long long)g_j2_cap_hits);
    }

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
