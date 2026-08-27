/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * HALF_CLASS two-pass scan tests.
 *
 * The critical property: the set of merit-qualified, first-endpoint-owned
 * consecutive-prime gaps found by the HALF pipeline (visible classes only +
 * on-demand hidden-class verification) must be IDENTICAL to the set found by
 * the full pipeline (all classes sieve + MR + gap scan).  This validates the
 * superset argument that makes the half-class trick correct.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

#include "../new_src/halfclass.h"
#include "../new_src/sieve_core.h"
#include "../new_src/gap_detection.h"
#include "../new_src/gap_dist.h"

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

struct pair {
    uint64_t a, b;
};

static int pair_cmp(const void *x, const void *y) {
    const struct pair *p = (const struct pair *)x;
    const struct pair *q = (const struct pair *)y;
    if (p->a != q->a) return p->a < q->a ? -1 : 1;
    if (p->b != q->b) return p->b < q->b ? -1 : 1;
    return 0;
}

static void collect(const struct gap_result *gaps, uint32_t n,
                    struct pair *dst, uint32_t *cnt) {
    for (uint32_t i = 0; i < n; i++) {
        dst[*cnt].a = gaps[i].offset_p1;
        dst[*cnt].b = gaps[i].offset_p2;
        (*cnt)++;
    }
}

/* Full pipeline: sieve all classes, MR every survivor, gap scan. */
static void run_full(mpz_t base, uint64_t window, uint64_t owned,
                     double thresh, struct pair *dst, uint32_t *cnt) {
    struct sieve_core sc = {0};
    CHECK(sieve_core_init_window(&sc, window, 1000) != 0);
    CHECK(sieve_core_prepare_base_mod_p(&sc, base) != 0);
    uint64_t *offs = NULL;
    uint32_t n = 0;
    CHECK(sieve_core_run_from_cached_base(&sc, 0, &offs, &n) != 0);

    uint8_t *ip = (uint8_t *)calloc((size_t)n, 1);
    CHECK(ip != NULL);
    mpz_t c;
    mpz_init(c);
    for (uint32_t i = 0; i < n; i++) {
        mpz_set(c, base);
        mpz_add_ui(c, c, offs[i]);
        ip[i] = (uint8_t)(mpz_probab_prime_p(c, 15) >= 1);
    }
    mpz_clear(c);

    /* Production mirror: the scan stops at the first halo probable prime
       (offset >= owned), which closes every owned gap. */
    for (uint32_t i = 0; i < n; i++) {
        if (ip[i] && offs[i] >= owned) {
            n = i + 1;
            break;
        }
    }

    struct gap_result *gaps = NULL;
    uint32_t gc = 0;
    struct gap_scan_stats st = {0};
    int rc = gap_detection_find(ip, offs, n, 0, base, thresh, owned, &st,
                                &gaps, &gc);
    CHECK(rc != 0);
    if (gaps) collect(gaps, gc, dst, cnt);
    gap_detection_free_results(gaps);
    free(ip);
    sieve_core_free(&sc);
}

/* HALF pipeline: visible classes only, then on-demand verification. */
static void run_half(mpz_t base, uint64_t window, uint64_t owned,
                     double thresh, struct pair *dst, uint32_t *cnt) {
    struct sieve_core sc = {0};
    CHECK(sieve_core_init_window(&sc, window, 1000) != 0);
    CHECK(sieve_core_prepare_base_mod_p(&sc, base) != 0);
    uint64_t *offs = NULL;
    uint32_t n = 0;
    CHECK(sieve_core_run_from_cached_base(&sc, 0, &offs, &n) != 0);

    uint32_t bm60 = (uint32_t)mpz_fdiv_ui(base, 60);
    n = halfclass_filter_offsets(bm60, offs, n);

    uint8_t *ip = (uint8_t *)calloc((size_t)n, 1);
    CHECK(ip != NULL);
    mpz_t c;
    mpz_init(c);
    for (uint32_t i = 0; i < n; i++) {
        mpz_set(c, base);
        mpz_add_ui(c, c, offs[i]);
        ip[i] = (uint8_t)(mpz_probab_prime_p(c, 15) >= 1);
    }
    mpz_clear(c);

    /* Production mirror: truncate at the first VISIBLE halo probable prime. */
    for (uint32_t i = 0; i < n; i++) {
        if (ip[i] && offs[i] >= owned) {
            n = i + 1;
            break;
        }
    }

    struct gap_result *gaps = NULL;
    uint32_t gc = 0;
    struct gap_scan_stats st = {0};
    int rc = gap_detection_find(ip, offs, n, 0, base, thresh, owned, &st,
                                &gaps, &gc);
    CHECK(rc != 0);

    /* Prefix: true primes before the first visible prime. */
    uint32_t first_vis = UINT32_MAX;
    for (uint32_t i = 0; i < n; i++) {
        if (ip[i]) {
            first_vis = i;
            break;
        }
    }
    if (first_vis != UINT32_MAX && offs[first_vis] > 0) {
        struct gap_result *pre = NULL;
        uint32_t pre_count = 0;
        CHECK(halfclass_verify_prefix(base, offs[first_vis], 1, owned,
                                      thresh, &pre, &pre_count) != 0);
        collect(pre, pre_count, dst, cnt);
        free(pre);
    }

    if (gaps) {
        for (uint32_t i = 0; i < gc; i++) {
            struct gap_result *sub = NULL;
            uint32_t sub_count = 0;
            CHECK(halfclass_resolve_gap(base, gaps[i].offset_p1,
                                        gaps[i].offset_p2, owned, thresh,
                                        &sub, &sub_count) != 0);
            collect(sub, sub_count, dst, cnt);
            free(sub);
        }
    }
    gap_detection_free_results(gaps);
    free(ip);
    sieve_core_free(&sc);
}

static void parity_case(mpz_t base, uint64_t window, uint64_t owned,
                        double thresh) {
    static struct pair full[200000], half[200000];
    uint32_t nf = 0, nh = 0;
    run_full(base, window, owned, thresh, full, &nf);
    run_half(base, window, owned, thresh, half, &nh);

    qsort(full, nf, sizeof(*full), pair_cmp);
    qsort(half, nh, sizeof(*half), pair_cmp);
    CHECK(nf == nh);
    uint32_t lim = nf < nh ? nf : nh;
    CHECK(memcmp(full, half, (size_t)lim * sizeof(struct pair)) == 0);
    printf("parity: full=%u half=%u %s\n", nf, nh,
           (nf == nh && memcmp(full, half, (size_t)lim * sizeof(struct pair)) == 0)
               ? "MATCH" : "MISMATCH");
}

/* CRT covering template: mark odd offsets t in [1, window) relative to the
   CRT-aligned base where base + t is divisible by a covering prime.  This is
   exactly what the production sieve reproduces via the CRT alignment. */
static void crt_template_build(mpz_t base, const uint64_t *primes, size_t np,
                               uint64_t window, uint8_t **out_bits,
                               uint64_t *out_words) {
    uint64_t odd_slots = (window + 1U) >> 1;
    uint64_t words = (odd_slots + 63U) / 64U;
    uint8_t *b = (uint8_t *)calloc((size_t)words * 8U, 1);
    CHECK(b != NULL);
    mpz_t v;
    mpz_init(v);
    for (uint64_t t = 1; t < window; t += 2) {
        mpz_set(v, base);
        mpz_add_ui(v, v, t);
        int marked = 0;
        for (size_t i = 0; i < np; i++) {
            if (mpz_fdiv_ui(v, primes[i]) == 0) {
                marked = 1;
                break;
            }
        }
        if (marked) {
            uint64_t slot = (t - 1U) >> 1;
            b[slot >> 3] |= (uint8_t)(1U << (slot & 7U));
        }
    }
    mpz_clear(v);
    *out_bits = b;
    *out_words = words;
}

/* CRT pipeline parity: the covering template is the composite prefilter and
   the owned range is [back_limit, back_limit + needed_gap). */
static void parity_case_crt(uint64_t delta) {
    static uint64_t primes[32];
    uint32_t np = 0;
    for (uint64_t v = 2; v <= 71 && np < 32; v++) {
        int isp = 1;
        for (uint32_t i = 0; i < np; i++) {
            if (v % primes[i] == 0) {
                isp = 0;
                break;
            }
        }
        if (isp) primes[np++] = v;
    }

    uint64_t needed_gap = 50;
    uint64_t back_limit = 64;
    uint64_t window = 6000;          /* scan window */
    uint64_t owned_limit = back_limit + needed_gap;
    double logbase = 64.0 * 0.6931471805599453;
    double thresh = (double)needed_gap / logbase;

    mpz_t base;   /* CRT-aligned window base (EVEN, like production) */
    mpz_init_set_ui(base, 1);
    mpz_mul_2exp(base, base, 64);
    mpz_setbit(base, 40);
    mpz_add_ui(base, base, delta);   /* even delta: several base variants */

    uint8_t *tbits = NULL;
    uint64_t twords = 0;
    crt_template_build(base, primes, np, window, &tbits, &twords);
    const struct halfclass_tpl tpl = { tbits, twords,
                                       (int64_t)back_limit, window };

    /* window_base = base - back_limit (production layout); offsets are
       relative to it, so the back-lookahead [0, back_limit) lies OUTSIDE
       the template (negative t) and the covering applies to t in [1,window). */
    mpz_t wbase;
    mpz_init(wbase);
    mpz_sub_ui(wbase, base, back_limit);

    /* FULL pipeline: template survivors -> MR -> gap scan. */
    static struct pair full[100000], half[100000];
    static uint64_t foffs[100000], hoffs[100000];
    static uint8_t fip[100000], hip[100000];
    uint32_t fn = 0, hn = 0;
    mpz_t c;
    mpz_init(c);
    for (uint64_t off = 0; off < back_limit + window; off++) {
        mpz_set(c, wbase);
        mpz_add_ui(c, c, off);
        if (mpz_even_p(c)) continue;
        /* template prefilter: t = off - back_limit */
        int64_t t = (int64_t)off - (int64_t)back_limit;
        int composite = 0;
        if (t > 0 && (uint64_t)t < window) {
            if (((uint64_t)t & 1ULL) == 0) {
                composite = 0;   /* no template slot for even t */
            } else {
                uint64_t slot = ((uint64_t)t - 1U) >> 1;
                composite = (tbits[slot >> 3] >> (slot & 7U)) & 1U;
            }
        }
        if (composite) continue;
        if (mpz_probab_prime_p(c, 15) >= 1) {
            foffs[fn++] = off;
            hoffs[hn++] = off;
        }
    }

    /* Full scan. */
    struct gap_result *gaps = NULL;
    uint32_t gc = 0, nf = 0, nh = 0;
    struct gap_scan_stats st = {0};
    for (uint32_t i = 0; i < fn; i++) fip[i] = 1;
    CHECK(gap_detection_find(fip, foffs, fn, 0, wbase, thresh, owned_limit,
                             &st, &gaps, &gc) != 0);
    collect(gaps, gc, full, &nf);
    gap_detection_free_results(gaps);

    /* HALF pipeline: back region unfiltered (region design), then prefix
       chain + terminal resolve + visible-gap resolves. */
    uint32_t bm60 = (uint32_t)mpz_fdiv_ui(wbase, 60);
    hn = halfclass_filter_offsets_region(bm60, hoffs, hn, back_limit);
    for (uint32_t i = 0; i < hn; i++) hip[i] = 1;

    uint32_t k = 0;
    while (k < hn && hoffs[k] < back_limit) k++;
    uint32_t pre_n = k, vis_n = hn - k;

    /* (1) gaps among back-region primes */
    if (pre_n > 0) {
        static uint64_t chain[1000];
        uint32_t cl = 0;
        for (uint32_t i = 0; i < pre_n; i++) {
            if (hip[i]) chain[cl++] = hoffs[i];
        }
        if (cl > 0) {
            struct gap_result *pre = NULL;
            uint32_t pre_count = halfclass_emit_chain(
                wbase, chain, cl, owned_limit, thresh, &pre);
            collect(pre, pre_count, half, &nh);
            free(pre);

            /* (2) terminal pair (last back prime, first visible prime) */
            uint32_t first_vis = UINT32_MAX;
            for (uint32_t i = 0; i < vis_n; i++) {
                if (hip[k + i]) {
                    first_vis = i;
                    break;
                }
            }
            if (first_vis != UINT32_MAX) {
                struct gap_result *sub = NULL;
                uint32_t sub_count = 0;
                CHECK(halfclass_resolve_gap_ex(
                          wbase, chain[cl - 1], hoffs[k + first_vis],
                          owned_limit, thresh, &tpl, &sub, &sub_count) != 0);
                collect(sub, sub_count, half, &nh);
                free(sub);
            }
        }
    }

    gaps = NULL;
    gc = 0;
    CHECK(gap_detection_find(hip + k, hoffs + k, vis_n, 0, wbase, thresh,
                             owned_limit, &st, &gaps, &gc) != 0);
    if (gaps) {
        for (uint32_t i = 0; i < gc; i++) {
            struct gap_result *sub = NULL;
            uint32_t sub_count = 0;
            CHECK(halfclass_resolve_gap_ex(wbase, gaps[i].offset_p1,
                                           gaps[i].offset_p2, owned_limit,
                                           thresh, &tpl, &sub,
                                           &sub_count) != 0);
            collect(sub, sub_count, half, &nh);
            free(sub);
        }
    }
    gap_detection_free_results(gaps);

    qsort(full, nf, sizeof(*full), pair_cmp);
    qsort(half, nh, sizeof(*half), pair_cmp);
    CHECK(nf == nh);
    uint32_t lim = nf < nh ? nf : nh;
    CHECK(memcmp(full, half, (size_t)lim * sizeof(struct pair)) == 0);
    printf("CRT parity: full=%u half=%u %s\n", nf, nh,
           (nf == nh && memcmp(full, half, (size_t)lim * sizeof(struct pair)) == 0)
               ? "MATCH" : "MISMATCH");

    free(tbits);
    mpz_clear(c);
    mpz_clear(wbase);
    mpz_clear(base);
}

int main(void) {
    /* 1. base_mod60 against hand-computed values. */
    {
        uint8_t h[32] = {0};
        h[31] = 1;
        CHECK(halfclass_base_mod60(h, 0, 0) == 1);
        CHECK(halfclass_base_mod60(h, 2, 0) == 4);
        CHECK(halfclass_base_mod60(h, 3, 0) == 8);
        CHECK(halfclass_base_mod60(h, 6, 0) == 4);   /* 2^6 = 64 ≡ 4 */
        CHECK(halfclass_base_mod60(h, 0, 5) == 6);
        h[31] = 2;
        CHECK(halfclass_base_mod60(h, 1, 0) == 4);
        h[31] = 1;
        printf("base_mod60 unit checks OK\n");
    }

    /* 2. filter vs brute-force residue check (all 60 base residues). */
    {
        static uint64_t arr[4096];
        for (uint32_t bm = 0; bm < 60; bm++) {
            for (uint32_t i = 0; i < 4096; i++) arr[i] = i;
            uint32_t cnt = halfclass_filter_offsets(bm, arr, 4096);
            uint32_t expect = 0;
            for (uint64_t o = 0; o < 4096; o++) {
                if (halfclass_offset_visible(bm, o)) expect++;
            }
            CHECK(cnt == expect);
            for (uint32_t i = 0; i < cnt; i++) {
                CHECK(halfclass_offset_visible(bm, arr[i]));
                if (i) CHECK(arr[i] > arr[i - 1]);
            }
        }
        printf("filter unit checks OK\n");
    }

    /* 3. End-to-end parity: HALF pipeline == full pipeline (the property
       that guards against false positives AND missed gaps). */
    {
        mpz_t base;
        mpz_init_set_str(base,
            "1684996666696914987166688442938726917102321526408785780068975640576",
            10); /* ~2^220 */
        mpz_setbit(base, 0);
        parity_case(base, 6000, 4000, 2.5);
        parity_case(base, 6000, 2000, 3.0);
        parity_case(base, 12000, 10000, 2.0);
        /* Low threshold: hundreds of visible-gap candidates, many with
           hidden-class primes inside their interior — mass-stresses the
           event-driven verification. */
        parity_case(base, 6000, 4000, 1.2);
        mpz_clear(base);
    }
    {
        mpz_t base;
        mpz_init_set_str(base,
            "13407807929942597099574024998205846127479365820592393377723561443721764030073546976801874298166903427690031858186486050853753882811946569946433649006084095",
            10); /* ~2^511, all-ones-ish */
        mpz_setbit(base, 0);
        parity_case(base, 6000, 4000, 2.5);
        mpz_clear(base);
    }
    {
        /* Production bit scale (~2^311, like shift 55): the exact-set
           equality must hold at the magnitude the miner actually scans. */
        mpz_t base;
        mpz_init_set_ui(base, 1);
        mpz_mul_2exp(base, base, 310);
        mpz_setbit(base, 200);
        mpz_setbit(base, 0);
        parity_case(base, 32768, 24576, 3.0);
        parity_case(base, 32768, 24576, 2.0);
        mpz_clear(base);
    }

    /* CRT covering-template parity: the covering is the composite prefilter
       (what the production sieve reproduces via CRT alignment).  Several
       base variants so at least some have a back-lookahead prime. */
    parity_case_crt(0);
    parity_case_crt(1ULL << 16);
    parity_case_crt(1ULL << 20);
    parity_case_crt(1ULL << 24);
    parity_case_crt(1ULL << 28);

    if (failures) {
        printf("test_halfclass FAILED (%d failures)\n", failures);
        return 1;
    }
    printf("All halfclass tests PASSED\n");
    return 0;
}
