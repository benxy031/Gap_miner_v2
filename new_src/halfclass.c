/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HALF_CLASS two-pass scan (non-CRT mode).  See halfclass.h for the design.
 */

#include "halfclass.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Residue classes coprime to 60, split into the visible (scanned) half and
   the hidden (on-demand verified) half.  All values are odd. */
static const uint8_t HID_MOD60[8] = { 31, 37, 41, 43, 47, 49, 53, 59 };

static uint64_t class_mask60(const uint8_t *vals, uint32_t n) {
    uint64_t mask = 0;
    for (uint32_t i = 0; i < n; i++) {
        mask |= 1ULL << vals[i];
    }
    return mask;
}

/* QUARTER_CLASS mode: 4 visible / 12 hidden of the 16 coprime classes. */
static int g_quarter_mode = 0;

void halfclass_set_quarter(int on) { g_quarter_mode = on ? 1 : 0; }

uint64_t halfclass_visible_mask(void) {
    if (g_quarter_mode) {
        static const uint8_t qvis[4] = { 1, 7, 11, 13 };
        return class_mask60(qvis, 4);
    }
    static const uint8_t vis[8] = { 1, 7, 11, 13, 17, 19, 23, 29 };
    return class_mask60(vis, 8);
}

uint64_t halfclass_hidden_mask(void) {
    if (g_quarter_mode) {
        static const uint8_t qhid[12] =
            { 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59 };
        return class_mask60(qhid, 12);
    }
    return class_mask60(HID_MOD60, 8);
}



uint32_t halfclass_base_mod60(const uint8_t h256[32], uint32_t shift,
                              uint64_t window_start) {
    uint32_t r = 0;
    for (int i = 0; i < 32; i++) {
        r = (uint32_t)(((uint64_t)r * 256ULL + h256[i]) % 60ULL);
    }
    /* 2^shift mod 60 by fast exponentiation. */
    uint32_t p2 = 1, b = 2, e = shift;
    while (e) {
        if (e & 1U) p2 = (uint32_t)((uint64_t)p2 * b % 60ULL);
        b = (uint32_t)((uint64_t)b * b % 60ULL);
        e >>= 1U;
    }
    return (uint32_t)(((uint64_t)r * p2 + (window_start % 60ULL)) % 60ULL);
}

int halfclass_offset_visible(uint32_t base_mod60, uint64_t offset) {
    uint32_t v = (uint32_t)((base_mod60 + (offset % 60ULL)) % 60ULL);
    return (halfclass_visible_mask() >> v) & 1U;
}

uint32_t halfclass_filter_offsets(uint32_t base_mod60, uint64_t *offsets,
                                  uint32_t count) {
    return halfclass_filter_offsets_region(base_mod60, offsets, count, 0);
}

uint32_t halfclass_filter_offsets_region(uint32_t base_mod60, uint64_t *offsets,
                                         uint32_t count, uint64_t region_start) {
    uint64_t mask = halfclass_visible_mask();
    uint32_t w = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t off = offsets[i];
        if (off < region_start) {
            offsets[w++] = off;
            continue;
        }
        uint32_t v = (uint32_t)((base_mod60 + (off % 60ULL)) % 60ULL);
        if ((mask >> v) & 1U) {
            offsets[w++] = off;
        }
    }
    return w;
}

/* Lazy one-time table of primes <= 100000 (9592 entries) for the on-demand
   hidden-class mini-sieve. */
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static uint64_t g_h_primes[10000]; /* π(100000) = 9592 */
static uint32_t g_h_prime_count = 0;

static void halfclass_init_hidden_primes(void) {
    /* Odd-only sieve of [3, 100000]. */
    uint32_t limit = 100000U;
    uint8_t *bits = (uint8_t *)calloc((size_t)(limit / 16 + 8), 1);
    if (!bits) return;
    for (uint32_t i = 3; (uint64_t)i * i <= limit; i += 2) {
        if ((bits[i / 16] >> ((i / 2) & 7)) & 1U) continue;
        for (uint64_t m = (uint64_t)i * i; m <= limit; m += 2ULL * i) {
            bits[m / 16] |= (uint8_t)(1U << ((m / 2) & 7));
        }
    }
    uint32_t n = 0;
    for (uint32_t i = 3; i <= limit && n < 10000; i += 2) {
        if (!((bits[i / 16] >> ((i / 2) & 7)) & 1U)) {
            g_h_primes[n++] = i;
        }
    }
    g_h_prime_count = n;
    free(bits);
}

/* Largest interval we are willing to bit-map for the mini-sieve (4M offsets,
   ~512 KB of bitmap).  Larger (astronomically rare) intervals skip the
   mini-sieve and MR-test every hidden-class number directly. */
#define HALFCLASS_MAX_BITMAP_INTERVAL (1ULL << 22)

/* 1 when the CRT covering template proves the candidate at absolute offset
   off_abs composite: value even inside the template range, or odd slot
   marked. */
static int halfclass_tpl_composite(const struct halfclass_tpl *tpl,
                                   uint64_t off_abs) {
    if (!tpl || !tpl->bits) return 0;
    int64_t t = (int64_t)off_abs - tpl->base_off;
    if (t <= 0 || (uint64_t)t >= tpl->window) return 0;
    if (((uint64_t)t & 1ULL) == 0) return 1;      /* even value */
    uint64_t slot = ((uint64_t)t - 1ULL) >> 1;
    if ((slot >> 3) >= tpl->words * 8U) return 0; /* byte-packed 64-bit words */
    return (int)((tpl->bits[slot >> 3] >> (slot & 7U)) & 1U);
}

/* Mini-sieve + template prefilter over the hidden classes in
   [bsub, bsub+interval); returns the surviving candidate offsets (relative
   to bsub, ascending) with NO primality testing. */
static int64_t halfclass_collect_hidden_candidates_impl(
    mpz_t bsub, uint64_t interval, uint32_t sub_mod60,
    uint64_t abs_delta, const struct halfclass_tpl *tpl,
    uint32_t sieve_cap, uint64_t **out)
{
    *out = NULL;
    if (interval == 0) return 0;

    pthread_once(&g_once, halfclass_init_hidden_primes);

    /* Mini-sieve: mark composites among ALL offsets in [0, interval). */
    uint64_t *bits = NULL;
    if (interval <= HALFCLASS_MAX_BITMAP_INTERVAL) {
        uint64_t words = (interval + 63U) >> 6;
        bits = (uint64_t *)calloc((size_t)words, sizeof(uint64_t));
    }
    if (bits) {
        uint32_t cap = g_h_prime_count < sieve_cap ? g_h_prime_count
                                                   : sieve_cap;
        for (uint32_t pi = 0; pi < cap; pi++) {
            uint64_t p = g_h_primes[pi];
            uint64_t rem = mpz_fdiv_ui(bsub, p);
            uint64_t first = rem ? (p - rem) : 0;
            if (first >= interval) continue;
            for (uint64_t m = first; m < interval; m += p) {
                bits[m >> 6] |= 1ULL << (m & 63U);
            }
        }
    }

    /* Upper bound: every hidden-class offset in the interval.  Guarantees
       no candidate is ever silently dropped (a dropped interior prime
       would fabricate a false gap). */
    uint64_t cap_off =
        (uint32_t)((interval / 60ULL) *
                       (uint64_t)__builtin_popcountll(halfclass_hidden_mask()) +
                   16ULL);
    uint64_t *cand = (uint64_t *)malloc((size_t)cap_off * sizeof(uint64_t));
    if (!cand) {
        free(bits);
        return -1;
    }
    uint32_t cnt = 0;
    uint64_t hmask = halfclass_hidden_mask();
    for (uint64_t off = 0; off < interval; off++) {
        uint32_t v = (uint32_t)((sub_mod60 + (off % 60ULL)) % 60ULL);
        if (!((hmask >> v) & 1U)) continue;
        if (bits && ((bits[off >> 6] >> (off & 63U)) & 1U)) continue;
        if (halfclass_tpl_composite(tpl, abs_delta + off)) continue;
        cand[cnt++] = off;
    }
    free(bits);
    *out = cand;
    return (int64_t)cnt;
}

int64_t halfclass_collect_hidden_candidates(mpz_t bsub, uint64_t interval,
                                            uint32_t sub_mod60, uint64_t abs_delta,
                                            const struct halfclass_tpl *tpl,
                                            uint32_t sieve_cap, uint64_t **out)
{
    return halfclass_collect_hidden_candidates_impl(
        bsub, interval, sub_mod60, abs_delta, tpl, sieve_cap, out);
}

/* Collect the hidden-class probable primes in [bsub, bsub+interval) as
   offsets relative to bsub, ascending.  Returns the count via *out (malloc'd
   array) or -1 on allocation failure.  Marks composites with the lazy prime
   table (p <= 100k) before MR-testing survivors.  abs_delta is added to the
   relative offsets to form absolute scan offsets for the optional CRT
   template prefilter. */
static int64_t halfclass_collect_hidden(mpz_t bsub, uint64_t interval,
                                        uint32_t sub_mod60,
                                        uint64_t abs_delta,
                                        const struct halfclass_tpl *tpl,
                                        int mr_reps,
                                        uint32_t sieve_cap,
                                        uint64_t **out) {
    *out = NULL;
    uint64_t *cand = NULL;
    int64_t nc = halfclass_collect_hidden_candidates_impl(
        bsub, interval, sub_mod60, abs_delta, tpl, sieve_cap, &cand);
    if (nc < 0) return -1;

    uint64_t *hid =
        (uint64_t *)malloc((size_t)(nc > 0 ? nc : 1) * sizeof(uint64_t));
    if (!hid) {
        free(cand);
        return -1;
    }
    uint32_t hid_count = 0;
    mpz_t c;
    mpz_init(c);
    for (int64_t i = 0; i < nc; i++) {
        mpz_set(c, bsub);
        mpz_add_ui(c, c, cand[i]);
        if (mpz_probab_prime_p(c, mr_reps) >= 1) {
            hid[hid_count++] = cand[i];
        }
    }
    mpz_clear(c);
    free(cand);
    if (getenv("GAPDEBUG")) {
        fprintf(stderr,
                "[HIDDBG] resolve interval=%llu abs_delta=%llu tested=%lld found=%u\n",
                (unsigned long long)interval,
                (unsigned long long)abs_delta,
                (long long)nc, hid_count);
    }
    *out = hid;
    return (int64_t)hid_count;
}

/* Emit merit-qualified, first-endpoint-owned pairs from a chain of
   consecutive probable primes given in absolute window offsets. */
static uint32_t halfclass_emit_chain_impl(mpz_t base, const uint64_t *chain,
                                          uint32_t chain_len,
                                          uint64_t owned_offset_limit,
                                          double merit_threshold,
                                          struct gap_result **out) {
    *out = NULL;
    if (chain_len < 2) return 0;
    struct gap_result *res =
        (struct gap_result *)malloc((size_t)(chain_len - 1) *
                                    sizeof(struct gap_result));
    if (!res) return 0;
    uint32_t rc = 0;
    mpz_t gv;
    mpz_init(gv);
    for (uint32_t i = 1; i < chain_len; i++) {
        uint64_t prev = chain[i - 1];
        uint64_t next = chain[i];
        if (prev >= owned_offset_limit || next <= prev) continue;
        mpz_set(gv, base);
        mpz_add_ui(gv, gv, prev);
        double merit =
            gap_detection_compute_merit((uint32_t)(next - prev), gv);
        if (merit >= merit_threshold) {
            res[rc].offset_p1 = prev;
            res[rc].offset_p2 = next;
            res[rc].gap_length = (uint32_t)(next - prev);
            res[rc].merit = merit;
            res[rc].verified = 0;
            rc++;
        }
    }
    mpz_clear(gv);
    if (rc == 0) {
        free(res);
        res = NULL;
    }
    *out = res;
    return rc;
}

uint32_t halfclass_emit_chain(mpz_t base, const uint64_t *chain,
                              uint32_t chain_len,
                              uint64_t owned_offset_limit,
                              double merit_threshold,
                              struct gap_result **out) {
    return halfclass_emit_chain_impl(base, chain, chain_len,
                                     owned_offset_limit, merit_threshold,
                                     out);
}

int halfclass_emit_resolved(mpz_t base, uint64_t off_a, uint64_t off_b,
                            const uint64_t *hid, int64_t hc,
                            uint64_t owned_offset_limit, double merit_threshold,
                            struct gap_result **out, uint32_t *out_count) {
    *out = NULL;
    *out_count = 0;
    if (off_b <= off_a || !base) return 1;

    /* Chain: off_a, hid (absolute), off_b. */
    uint64_t *chain =
        (uint64_t *)malloc((size_t)(hc + 2) * sizeof(uint64_t));
    if (!chain)
        return 0;
    chain[0] = off_a;
    for (int64_t i = 0; i < hc; i++) chain[i + 1] = off_a + hid[i];
    chain[hc + 1] = off_b;

    *out_count = halfclass_emit_chain_impl(base, chain, (uint32_t)(hc + 2),
                                           owned_offset_limit,
                                           merit_threshold, out);
    free(chain);
    return 1;
}

int halfclass_resolve_gap_ex(mpz_t base, uint64_t off_a, uint64_t off_b,
                             uint64_t owned_offset_limit,
                             double merit_threshold,
                             const struct halfclass_tpl *tpl,
                             struct gap_result **out, uint32_t *out_count) {
    *out = NULL;
    *out_count = 0;
    if (off_b <= off_a || !base) return 1;

    uint64_t interval = off_b - off_a;
    uint32_t base_mod60 = (uint32_t)(mpz_fdiv_ui(base, 60));
    uint32_t sub_mod60 = (uint32_t)((base_mod60 + (off_a % 60ULL)) % 60ULL);

    /* bsub = value of the visible prime at off_a (prime, so no sieving
       prime <= 100k divides it; offset 0 can never be marked). */
    mpz_t bsub;
    mpz_init(bsub);
    mpz_set(bsub, base);
    mpz_add_ui(bsub, bsub, off_a);

    uint64_t *hid = NULL;
    int64_t hc = halfclass_collect_hidden(bsub, interval, sub_mod60, off_a,
                                          tpl, 15, 10000, &hid);
    mpz_clear(bsub);
    if (hc < 0) return 0;

    int rc = halfclass_emit_resolved(base, off_a, off_b, hid, hc,
                                     owned_offset_limit, merit_threshold,
                                     out, out_count);
    free(hid);
    return rc;
}

int halfclass_resolve_gap(mpz_t base, uint64_t off_a, uint64_t off_b,
                          uint64_t owned_offset_limit, double merit_threshold,
                          struct gap_result **out, uint32_t *out_count) {
    return halfclass_resolve_gap_ex(base, off_a, off_b, owned_offset_limit,
                                    merit_threshold, NULL, out, out_count);
}

int halfclass_verify_prefix_ex(mpz_t base, uint64_t off_v0,
                               int terminal_is_prime,
                               uint64_t owned_offset_limit,
                               double merit_threshold,
                               const struct halfclass_tpl *tpl,
                               struct gap_result **out, uint32_t *out_count) {
    *out = NULL;
    *out_count = 0;
    if (off_v0 == 0 || !base) return 1;

    uint32_t base_mod60 = (uint32_t)(mpz_fdiv_ui(base, 60));

    /* The interval starts at the window base itself.  The prefix runs on
       EVERY window, so it uses a cheaper MR (8 reps) and a shallow mini-sieve
       (~1300 primes <= 10k): a rare false positive only splits a true gap
       (lost reward, never a false gap), and the endpoints still get
       BPSW-verified. */
    uint64_t *hid = NULL;
    int64_t hc = halfclass_collect_hidden(base, off_v0, base_mod60, 0, tpl,
                                          8, 1300, &hid);
    if (hc < 0) return 0;

    uint64_t *chain =
        (uint64_t *)malloc((size_t)(hc + 1 + (terminal_is_prime ? 1 : 0)) *
                           sizeof(uint64_t));
    if (!chain) {
        free(hid);
        return 0;
    }
    uint32_t chain_len = 0;
    for (int64_t i = 0; i < hc; i++) chain[chain_len++] = hid[i];
    if (terminal_is_prime) chain[chain_len++] = off_v0;
    free(hid);

    *out_count = halfclass_emit_chain_impl(base, chain, chain_len,
                                           owned_offset_limit,
                                           merit_threshold, out);
    free(chain);
    return 1;
}

int halfclass_verify_prefix(mpz_t base, uint64_t off_v0, int terminal_is_prime,
                            uint64_t owned_offset_limit,
                            double merit_threshold,
                            struct gap_result **out, uint32_t *out_count) {
    return halfclass_verify_prefix_ex(base, off_v0, terminal_is_prime,
                                      owned_offset_limit, merit_threshold,
                                      NULL, out, out_count);
}
