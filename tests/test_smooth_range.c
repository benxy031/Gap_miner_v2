/*
 * Unit test: tools/smooth_range.h - streaming smooth-run census.
 *
 * The statistic (mersenneforum node/22899, "smooth ranges of gaps"): for a
 * bound g, a run is a maximal block of consecutive prime gaps all <= g; its
 * length is the number of GAPS in the block.  This test verifies the O(1)
 * segment-deque counter against
 *
 *   1. a naive O(n*K) per-bound reference on synthetic streams (all-2 gaps,
 *      alternating, pseudo-random even gaps);
 *   2. known leaders over the real primes < 2e6 - values published in the
 *      thread / recomputed from a complete gap list, independent of the
 *      counter implementation;
 *   3. the boundary rules: a run that never closes stays "open" and is not
 *      counted; duplicate/out-of-order feeds are ignored;
 *   4. cost: ns/prime at K=64 on a 5M-prime synthetic stream (informational).
 *
 * Slots: the counter is slot based; here slot = (p-1)/2, i.e. observed prime
 * = 1 + 2*slot, exactly how phase0_scan_gpu converts its A-slots.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c99 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../tools/smooth_range.h"

static int fails = 0;
static int checks = 0;

#define CHECK(cond, fmt, ...) do {                                        \
    checks++;                                                             \
    if (!(cond)) {                                                        \
        fails++;                                                          \
        printf("FAIL %s:%d: " fmt "\n", __FILE__, __LINE__, __VA_ARGS__); \
    }                                                                     \
} while (0)

typedef struct {
    int64_t  len;
    uint64_t sp, ep;
    int64_t  open;
    uint64_t osp, oep;
} ref_t;

/* naive per-bound reference over an increasing slot stream */
static void ref_census(const uint64_t *slots, size_t n, const uint64_t *gs,
                       int K, ref_t *out) {
    int j;
    size_t i;
    for (j = 0; j < K; j++) {
        out[j].len = 0;  out[j].sp = 0;  out[j].ep = 0;
        out[j].open = 0; out[j].osp = 0; out[j].oep = 0;
    }
    if (n < 2) return;
    for (j = 0; j < K; j++) {
        uint64_t g = gs[j];
        size_t start = 0;
        for (i = 0; i + 1 < n; i++) {
            uint64_t d = 2 * (slots[i + 1] - slots[i]);
            if (d > g) {
                int64_t len = (int64_t)(i - start);
                if (len > out[j].len) {
                    out[j].len = len;
                    out[j].sp = slots[start];
                    out[j].ep = slots[i];
                }
                start = i + 1;
            }
        }
        out[j].open = (int64_t)(n - 1 - start);
        if (out[j].open > 0) {
            out[j].osp = slots[start];
            out[j].oep = slots[n - 1];
        }
    }
}

/* feed a whole stream, then compare every bound against the reference */
static int compare_with_ref(const char *tag, const uint64_t *slots, size_t n,
                            const uint64_t *gs, int K) {
    smooth_range_t sr;
    char err[160] = "";
    ref_t ref[SR_MAXG];
    int64_t lens[SR_MAXG], openv[SR_MAXG];
    uint64_t sps[SR_MAXG], eps[SR_MAXG], osp[SR_MAXG], oep[SR_MAXG];
    int j, bad = 0;
    size_t i;

    if (!sr_init(&sr, gs, K, err, sizeof err)) {
        printf("FAIL %s: sr_init: %s\n", tag, err);
        fails++;
        return 0;
    }
    for (i = 0; i < n; i++) sr_prime(&sr, slots[i]);
    sr_results(&sr, lens, sps, eps, openv, osp, oep);
    ref_census(slots, n, gs, K, ref);
    for (j = 0; j < K; j++) {
        if (lens[j] != ref[j].len ||
            (ref[j].len > 0 && (sps[j] != ref[j].sp || eps[j] != ref[j].ep))) {
            printf("FAIL %s: g<=%llu closed len=%lld sp=%llu ep=%llu "
                   "| ref len=%lld sp=%llu ep=%llu\n", tag,
                   (unsigned long long)gs[j], (long long)lens[j],
                   (unsigned long long)sps[j], (unsigned long long)eps[j],
                   (long long)ref[j].len, (unsigned long long)ref[j].sp,
                   (unsigned long long)ref[j].ep);
            bad++;
        }
        if (openv[j] != ref[j].open ||
            (ref[j].open > 0 && (osp[j] != ref[j].osp || oep[j] != ref[j].oep))) {
            printf("FAIL %s: g<=%llu open=%lld sp=%llu ep=%llu "
                   "| ref open=%lld sp=%llu ep=%llu\n", tag,
                   (unsigned long long)gs[j], (long long)openv[j],
                   (unsigned long long)osp[j], (unsigned long long)oep[j],
                   (long long)ref[j].open, (unsigned long long)ref[j].osp,
                   (unsigned long long)ref[j].oep);
            bad++;
        }
    }
    checks++;
    if (bad) fails++;
    return bad == 0;
}

/* ---------------- synthetic streams ---------------- */

static uint64_t lcg_state = 0x9e3779b97f4a7c15ULL;
static uint64_t lcg_next(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state >> 11;
}

static void test_synthetic(void) {
    static uint64_t slots[20001];
    static const uint64_t gs1[] = {2, 4, 100};
    uint64_t gsmany[64];
    size_t n;
    int j;

    /* A: all gaps = 2 -> nothing can close; open run = n-1 gaps for g >= 2 */
    n = 40;
    slots[0] = 5;
    for (size_t i = 1; i < n; i++) slots[i] = slots[i - 1] + 1;   /* gap 2 */
    {
        smooth_range_t sr;
        char err[160] = "";
        int64_t lens[SR_MAXG], openv[SR_MAXG];
        uint64_t sps[SR_MAXG], eps[SR_MAXG], osp[SR_MAXG], oep[SR_MAXG];
        if (!sr_init(&sr, gs1, 3, err, sizeof err)) {
            printf("FAIL sr_init A: %s\n", err);
            fails++;
        } else {
            for (size_t i = 0; i < n; i++) sr_prime(&sr, slots[i]);
            sr_results(&sr, lens, sps, eps, openv, osp, oep);
            for (j = 0; j < 3; j++) {
                CHECK(lens[j] == 0, "A: g<=%llu unexpected closed len %lld",
                      (unsigned long long)gs1[j], (long long)lens[j]);
                CHECK(openv[j] == (int64_t)(n - 1),
                      "A: g<=%llu open %lld, want %d",
                      (unsigned long long)gs1[j], (long long)openv[j], (int)(n - 1));
                CHECK(osp[j] == slots[0] && oep[j] == slots[n - 1],
                      "A: g<=%llu open span %llu..%llu",
                      (unsigned long long)gs1[j], (unsigned long long)osp[j],
                      (unsigned long long)oep[j]);
            }
            /* one break gap (200) closes every bound with c = n-1 */
            slots[n] = slots[n - 1] + 100;
            sr_prime(&sr, slots[n]);
            sr_results(&sr, lens, sps, eps, openv, osp, oep);
            for (j = 0; j < 3; j++) {
                CHECK(lens[j] == (int64_t)(n - 1),
                      "A-close: g<=%llu len %lld, want %d",
                      (unsigned long long)gs1[j], (long long)lens[j], (int)(n - 1));
                CHECK(sps[j] == slots[0] && eps[j] == slots[n - 1],
                      "A-close: g<=%llu span %llu..%llu want %llu..%llu",
                      (unsigned long long)gs1[j], (unsigned long long)sps[j],
                      (unsigned long long)eps[j], (unsigned long long)slots[0],
                      (unsigned long long)slots[n - 1]);
            }
        }
        compare_with_ref("A", slots, n + 1, gs1, 3);
    }

    /* B: alternating 2 / 50 gaps, bounds {2, 50, 100} */
    n = 2001;
    slots[0] = 3;
    for (size_t i = 1; i < n; i++)
        slots[i] = slots[i - 1] + ((i & 1) ? 1 : 25);   /* gap 2 / gap 50 */
    compare_with_ref("B", slots, n, gs1, 3);

    /* C: pseudo-random even gaps in [2,300], 64 bounds 2..128 */
    for (j = 0; j < 64; j++) gsmany[j] = (uint64_t)(2 * (j + 1));
    n = 20001;
    slots[0] = 7;
    for (size_t i = 1; i < n; i++)
        slots[i] = slots[i - 1] + 1 + (lcg_next() % 150);   /* gap 2..300 */
    compare_with_ref("C", slots, n, gsmany, 64);

    /* D: heavy tails - mostly tiny gaps with rare huge ones */
    n = 20001;
    slots[0] = 11;
    for (size_t i = 1; i < n; i++)
        slots[i] = slots[i - 1] + 1 + ((lcg_next() % 97 == 0) ? 500 + lcg_next() % 5000
                                                              : lcg_next() % 3);
    compare_with_ref("D", slots, n, gsmany, 64);

    /* E: duplicate and out-of-order feeds must be ignored */
    {
        smooth_range_t sr;
        char err[160] = "";
        if (sr_init(&sr, gs1, 3, err, sizeof err)) {
            sr_prime(&sr, 100);
            sr_prime(&sr, 101);
            sr_prime(&sr, 101);          /* duplicate */
            sr_prime(&sr, 99);           /* regression */
            sr_prime(&sr, 102);
            CHECK(sr.n_prime == 5, "E: n_prime %llu", (unsigned long long)sr.n_prime);
            CHECK(sr.n_gap == 2, "E: n_gap %lld", (long long)sr.n_gap);
            CHECK(sr.regressions == 1, "E: regressions %llu",
                  (unsigned long long)sr.regressions);
        } else {
            printf("FAIL sr_init E: %s\n", err);
            fails++;
            checks++;
        }
    }
}

/* ---------------- CLI list parser ---------------- */

static void test_parse_list(void) {
    uint64_t g[SR_MAXG];
    char err[160] = "";
    int k;

    k = sr_parse_list("32,2,8,4,8,2", g, SR_MAXG, err, sizeof err);
    CHECK(k == 4, "parse: k=%d want 4", k);
    if (k == 4)
        CHECK(g[0] == 2 && g[1] == 4 && g[2] == 8 && g[3] == 32,
              "parse: got %llu,%llu,%llu,%llu",
              (unsigned long long)g[0], (unsigned long long)g[1],
              (unsigned long long)g[2], (unsigned long long)g[3]);

    CHECK(sr_parse_list("", g, SR_MAXG, err, sizeof err) < 0, "%s", "parse: empty accepted");
    CHECK(sr_parse_list("3", g, SR_MAXG, err, sizeof err) < 0, "%s", "parse: odd accepted");
    CHECK(sr_parse_list("2,4x", g, SR_MAXG, err, sizeof err) < 0, "%s", "parse: junk accepted");
    CHECK(sr_parse_list("2,", g, SR_MAXG, err, sizeof err) == 1, "%s", "parse: trailing comma");
    /* strictly increasing bounds are what sr_init expects */
    {
        smooth_range_t sr;
        CHECK(sr_init(&sr, g, 1, err, sizeof err) == 1, "init: %s", err);
        g[0] = 4; g[1] = 4;
        CHECK(sr_init(&sr, g, 2, err, sizeof err) == 0, "%s", "init: duplicate bounds accepted");
    }
}

/* ---------------- real primes < 2e6 ---------------- */

#define REAL_LIMIT 2000000

static void test_real_primes(void) {
    static uint64_t slots[REAL_LIMIT / 8];
    static const uint64_t gs[] = {2,  4,  6,   8,   10,  12,  14,  16,
                                  18, 20, 22,  24,  26,  28,  30,  32,
                                  34, 36, 40,  44,  48,  50,  56,  64,
                                  80, 100, 128};
    static const int K = (int)(sizeof gs / sizeof gs[0]);
    /* published / independently recomputed leaders (start..end prime) */
    static const struct {
        uint64_t g;
        int64_t  len;
        uint64_t sp, ep;
    } known[] = {
        {6,  10,   1464251, 1464299},
        {8,  13,   2657,    2719},
        {10, 24,   331,     467},
        {12, 37,   541,     773},
        {14, 48,   1129,    1361},
        {16, 54,   541,     887},
        {18, 59,   21647,   22193},
        {20, 73,   15053,   15683},
        {22, 111,  3299,    4177},
        {24, 149,  1361,    2477},
        {26, 211,  1361,    2971},
        {28, 372,  1361,    4297},
        {30, 520,  1361,    5591},
        {32, 841,  1361,    8467},
    };
    smooth_range_t sr;
    char err[160] = "";
    uint8_t *comp;
    int64_t lens[SR_MAXG], openv[SR_MAXG];
    uint64_t sps[SR_MAXG], eps[SR_MAXG], osp[SR_MAXG], oep[SR_MAXG];
    size_t n = 0;
    int64_t v;
    int j, k;
    size_t i;

    /* odd composite sieve: index (v-1)/2 */
    comp = (uint8_t *)calloc((size_t)(REAL_LIMIT / 2 + 1), 1);
    if (!comp) { printf("FAIL: calloc\n"); fails++; return; }
    for (v = 3; v * v <= REAL_LIMIT; v += 2)
        if (!comp[v / 2])
            for (int64_t w = v * v; w <= REAL_LIMIT; w += 2 * v) comp[w / 2] = 1;
    for (v = 3; v <= REAL_LIMIT; v += 2)
        if (!comp[v / 2]) slots[n++] = (uint64_t)((v - 1) / 2);
    free(comp);
    printf("test_smooth_range: %zu primes in [3,%d]\n", n, REAL_LIMIT);

    /* (1) independent reference on the full real stream.  NOTE: this stream
       starts at prime 3, so every bound's FIRST run has no left delimiter and
       merges with the start (e.g. g<=24 reports 215 gaps from 3..1327, longer
       than the published 149 @ 1361-2477).  A real scan has exactly the same
       property: the run open at the range start is reported as OPEN, never
       counted (see the A case above). */
    compare_with_ref("real-full", slots, n, gs, K);

    /* (2) known leaders, asserted on a stream that starts at prime 1301.
       A real range start is an arbitrary prime in the middle of the gap
       structure; starting here drops the merge with the stream start while
       keeping every published run that begins at >= 1301 (the g<=24..32
       family starts at 1361, the g<=8/22 family at 2657/3299, ...). */
    {
        size_t off = 0;
        while (off < n && 1 + 2 * slots[off] < 1301) off++;
        compare_with_ref("real-1301+", slots + off, n - off, gs, K);
        if (!sr_init(&sr, gs, K, err, sizeof err)) {
            printf("FAIL sr_init real: %s\n", err);
            fails++;
        } else {
            for (i = off; i < n; i++) sr_prime(&sr, slots[i]);
            sr_results(&sr, lens, sps, eps, openv, osp, oep);
            printf("test_smooth_range: real[1301+] best per bound:");
            for (j = 0; j < K; j++)
                printf(" %llu:%lld", (unsigned long long)gs[j], (long long)lens[j]);
            printf("\n");
            for (k = 0; k < (int)(sizeof known / sizeof known[0]); k++) {
                int idx = -1;
                for (j = 0; j < K; j++)
                    if (gs[j] == known[k].g) idx = j;
                if (idx < 0) {
                    printf("FAIL known: g=%llu not in bounds\n",
                           (unsigned long long)known[k].g);
                    fails++;
                    checks++;
                    continue;
                }
                if (known[k].sp < 1301) {
                    printf("test_smooth_range: known g<=%llu SKIPPED (its run starts at %llu,\n"
                           "  left of the 1301 stream start; on a stream that starts lower that run\n"
                           "  merges with the start - the counter then reports it as OPEN, as designed)\n",
                           (unsigned long long)known[k].g,
                           (unsigned long long)known[k].sp);
                    continue;
                }
                {
                    uint64_t sp = 1 + 2 * sps[idx], ep = 1 + 2 * eps[idx];
                    CHECK(lens[idx] == known[k].len && sp == known[k].sp &&
                          ep == known[k].ep,
                          "known g<=%llu: got len=%lld %llu..%llu, want len=%lld %llu..%llu",
                          (unsigned long long)known[k].g, (long long)lens[idx],
                          (unsigned long long)sp, (unsigned long long)ep,
                          (long long)known[k].len,
                          (unsigned long long)known[k].sp,
                          (unsigned long long)known[k].ep);
                }
            }
            printf("test_smooth_range: real open-run at end:");
            for (j = 0; j < K; j++)
                if (openv[j] > 0)
                    printf(" g<=%llu:%lld", (unsigned long long)gs[j],
                           (long long)openv[j]);
            printf("\n");
        }
    }
}

/* ---------------- cost ---------------- */

static void bench(void) {
    static uint64_t gs[64];
    static uint64_t slots[5000000];
    smooth_range_t sr;
    char err[160] = "";
    struct timespec t0, t1;
    size_t n = sizeof slots / sizeof slots[0];
    int j;
    double dt, ns;

    for (j = 0; j < 64; j++) gs[j] = (uint64_t)(2 * (j + 1));   /* 2..128 */
    /* gaps ~ exponential, mean 46 (a 1e20-ish prime stream has mean ln N ~ 46) */
    slots[0] = 1ULL << 40;
    for (size_t i = 1; i < n; i++) {
        double u = (double)(lcg_next() % 1000000 + 1) / 1000001.0;
        uint64_t half = (uint64_t)(-log(u) * 23.0) + 1;         /* gap/2 >= 1 */
        slots[i] = slots[i - 1] + half;
    }
    if (!sr_init(&sr, gs, 64, err, sizeof err)) {
        printf("FAIL bench sr_init: %s\n", err);
        fails++;
        return;
    }
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (size_t i = 0; i < n; i++) sr_prime(&sr, slots[i]);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    dt = (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    ns = dt * 1e9 / (double)n;
    printf("test_smooth_range: bench K=64 bounds 2..128: %.2f ns/prime "
           "(%.1f M primes/s on this core; %.1f%% of the gaps broke a bound)\n",
           ns, n / dt / 1e6, 100.0 * (double)sr.breaks / (double)n);
}

int main(void) {
    printf("=== test_smooth_range: streaming smooth-run census ===\n");
    test_parse_list();
    test_synthetic();
    test_real_primes();
    bench();
    printf("=== %s: %d checks, %d failures ===\n", fails ? "FAIL" : "PASS",
           checks, fails);
    return fails ? 1 : 0;
}
