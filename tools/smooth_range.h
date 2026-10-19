/*
 * smooth_range.h - streaming "smooth run" census for a stream of primes.
 *
 * WHAT IS COUNTED (the community's "smooth ranges of gaps" statistic,
 * mersenneforum node/22899): for a bound g, a RUN is a maximal block of
 * consecutive prime gaps whose lengths are all <= g.  Its LENGTH is the
 * number of gaps in the block; a run of length L covers L gaps / L+1 primes.
 * It is delimited on both sides by a gap > g.
 *
 * This module is fed the primes of a scan in increasing order and keeps,
 * for a small sorted set of bounds g[] (K <= SR_MAXG), the longest CLOSED
 * run seen so far (length + first/last prime slot), plus the run that is
 * still OPEN when the feed stops (reported separately, never counted).
 *
 * CONSERVATIVE COUNTING RULES
 *  - a run counts only when it is closed by a real gap > g inside the fed
 *    stream.  Runs that reach the end of the stream stay open;
 *  - the caller must start feeding at the first prime >= range start, so a
 *    run that extends left of the range start is under-counted;
 *  - a caller that splits the stream across threads/processes must merge
 *    closed runs only (runs crossing a slice boundary are never closed, so
 *    they never enter the result);
 *  - the first fed prime starts the stream: the gap from the previous prime
 *    (which the caller may know) is deliberately NOT used, because a run
 *    that "starts" with an unknown left boundary must not be counted.
 * Every one of these rules can only make the reported length shorter.
 *
 * SLOTS: the caller supplies arbitrary u64 slot indices and converts them
 * back to prime values (phase0_scan_gpu: observed prime = anchor + 2*slot).
 * Feeds must be strictly increasing; duplicates and regressions are ignored
 * (defensive, count in sr.regressions).
 *
 * COST: O(log K) amortized per gap, O(K) memory, K <= 128.  The bounds are
 * kept as a deque of "threshold-range segments": a gap d breaks exactly the
 * PREFIX of bounds whose g < d, which retires whole segments from one end
 * of the deque, so a closing run costs O(#retired segments).  Recording the
 * improvements needs no tree: best_len[] is NON-DECREASING in g (a block of
 * gaps all <= g_low is also all <= g_high, so the higher bound's best is at
 * least the lower one's), hence the bounds an improved run beats always form
 * a PREFIX of its range and one binary search finds where it ends.
 *
 * The header is standalone C99 (also C++-clean) and is unit-tested by
 * tests/test_smooth_range.c against a naive O(n*K) reference.  Measured cost
 * on one core of the dev box (pseudo-random gaps, mean 46, so 96% of the gaps
 * break at least one bound): 13 ns/prime at K=8, 23 ns at K=32, 34 ns at K=128
 * - i.e. a census over 1e12 integers (2.2e10 primes) adds ~35-60% to the
 * scanner's own wall.  Keep the bound list to what is actually compared.
 */
#ifndef SMOOTH_RANGE_H
#define SMOOTH_RANGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SR_MAXG
#define SR_MAXG 128
#endif

typedef struct {
    int      K;                       /* number of bounds */
    uint64_t g[SR_MAXG];              /* ascending, even, >= 2 */
    int      ok;                      /* 1 after a successful sr_init */

    /* stream state */
    int      have_last;
    uint64_t last_slot;
    int64_t  n_gap;                   /* gaps fed so far */
    uint64_t n_prime;                 /* primes fed so far */
    uint64_t regressions;             /* duplicate/out-of-order feeds ignored */

    /* deque of threshold-range segments over bound indices.
       Segments partition [0,K); the LAST element is the lowest-g range. */
    int      nseg;
    int      seg_lo[SR_MAXG + 1];
    int      seg_hi[SR_MAXG + 1];
    int64_t  seg_brk[SR_MAXG + 1];    /* index of the gap that broke it (-1 = none) */
    uint64_t seg_sp[SR_MAXG + 1];     /* start slot of its open run */

    /* best closed run per bound (len == 0: none yet); NON-DECREASING in g */
    int64_t  best_len[SR_MAXG];
    uint64_t best_sp[SR_MAXG];
    uint64_t best_ep[SR_MAXG];

    uint64_t breaks;                  /* diagnostics: gaps that broke any bound */
} smooth_range_t;

/* ---------------- core ---------------- */

/* Close the run of bounds [lo,hi) that ended with the gap at index `i`
   (the last gap of the run, whose upper prime is `end_slot`; `sp_slot` is
   the first prime slot of the run).  Only bounds whose stored best is below
   the closed length are touched. */
static void sr_close(smooth_range_t *sr, int lo, int hi, int64_t brk, int64_t i,
                     uint64_t end_slot, uint64_t sp_slot) {
    int64_t c = (i - 1) - brk;        /* gaps in the closed run */
    int a, b, p;
    if (hi <= lo || c <= 0) return;
    /* best_len is non-decreasing in g: the lowest-g bound in the range holds
       the smallest best, so if it is not beaten nothing is (the common case). */
    if (sr->best_len[lo] >= c) return;
    /* first index in [lo,hi) with best_len >= c; [lo,p) is improved */
    a = lo;
    b = hi;
    while (a < b) {
        int m = (a + b) >> 1;
        if (sr->best_len[m] >= c) b = m; else a = m + 1;
    }
    p = a;
    for (a = lo; a < p; a++) {
        sr->best_len[a] = c;
        sr->best_sp[a] = sp_slot;
        sr->best_ep[a] = end_slot;
    }
}

/* Bounds must be even, >= 2 and STRICTLY increasing (the caller sorts and
   de-duplicates; sr_parse_list does that for CLI strings).  Returns 1 on
   success, 0 with a message in err/n. */
static int sr_init(smooth_range_t *sr, const uint64_t *glist, int K,
                   char *err, size_t n) {
    int i;
    memset(sr, 0, sizeof *sr);
    if (K < 1 || K > SR_MAXG) {
        if (err) snprintf(err, n, "need 1..%d bounds (got %d)", SR_MAXG, K);
        return 0;
    }
    for (i = 0; i < K; i++) {
        if (glist[i] < 2 || (glist[i] & 1ULL)) {
            if (err) snprintf(err, n, "bound g=%llu must be even and >= 2",
                              (unsigned long long)glist[i]);
            return 0;
        }
        if (i > 0 && glist[i] <= glist[i - 1]) {
            if (err) snprintf(err, n, "bounds must be strictly increasing (%llu <= %llu)",
                              (unsigned long long)glist[i],
                              (unsigned long long)glist[i - 1]);
            return 0;
        }
        sr->g[i] = glist[i];
    }
    sr->K = K;
    /* one segment: every bound, no break yet, run starts at the first feed */
    sr->nseg = 1;
    sr->seg_lo[0] = 0;
    sr->seg_hi[0] = K;
    sr->seg_brk[0] = -1;
    sr->seg_sp[0] = 0;
    sr->ok = 1;
    return 1;
}

/* Feed the next prime of the stream. */
static void sr_prime(smooth_range_t *sr, uint64_t slot) {
    uint64_t d2;
    int64_t i;
    if (!sr->ok) return;
    sr->n_prime++;
    if (!sr->have_last) {
        sr->have_last = 1;
        sr->last_slot = slot;
        sr->seg_sp[sr->nseg - 1] = slot;   /* open runs start here */
        return;
    }
    if (slot <= sr->last_slot) {           /* duplicate / out of order */
        if (slot < sr->last_slot) sr->regressions++;
        return;
    }
    d2 = (slot - sr->last_slot) << 1;      /* prime gaps are even here */
    i = sr->n_gap;
    if (d2 > sr->g[0]) {
        int lo = 0, hi = sr->K, cut;
        while (lo < hi) {                  /* cut = #bounds broken by d2 */
            int mid = (lo + hi) >> 1;
            if (sr->g[mid] < d2) lo = mid + 1; else hi = mid;
        }
        cut = lo;
        sr->breaks++;
        /* retire tail segments that lie entirely in [0,cut) */
        while (sr->nseg > 0 && sr->seg_hi[sr->nseg - 1] <= cut) {
            int s = sr->nseg - 1;
            sr_close(sr, sr->seg_lo[s], sr->seg_hi[s], sr->seg_brk[s], i,
                     sr->last_slot, sr->seg_sp[s]);
            sr->nseg--;
        }
        /* split the tail segment when the cut lands inside it */
        if (sr->nseg > 0 && sr->seg_lo[sr->nseg - 1] < cut) {
            int s = sr->nseg - 1;
            sr_close(sr, sr->seg_lo[s], cut, sr->seg_brk[s], i,
                     sr->last_slot, sr->seg_sp[s]);
            sr->seg_lo[s] = cut;
        }
        /* push the merged prefix as a fresh segment: its run starts here */
        if (sr->nseg < SR_MAXG + 1) {
            sr->seg_lo[sr->nseg] = 0;
            sr->seg_hi[sr->nseg] = cut;
            sr->seg_brk[sr->nseg] = i;
            sr->seg_sp[sr->nseg] = slot;
            sr->nseg++;
        }
    }
    sr->n_gap++;
    sr->last_slot = slot;
}

/* Results: per-bound best closed run (len 0 = none), and the still-open run
   at the end of the feed (a LOWER BOUND, reported but never counted). */
static void sr_results(const smooth_range_t *sr, int64_t *lens, uint64_t *sps,
                       uint64_t *eps, int64_t *open_len, uint64_t *osp,
                       uint64_t *oep) {
    int j, s;
    for (j = 0; j < sr->K; j++) {
        lens[j] = sr->best_len[j];
        sps[j] = sr->best_sp[j];
        eps[j] = sr->best_ep[j];
        open_len[j] = 0;
        osp[j] = 0;
        oep[j] = 0;
    }
    if (!sr->have_last) return;
    for (s = 0; s < sr->nseg; s++) {
        int64_t o = sr->n_gap - 1 - sr->seg_brk[s];
        if (o <= 0) continue;
        for (j = sr->seg_lo[s]; j < sr->seg_hi[s]; j++) {
            open_len[j] = o;
            osp[j] = sr->seg_sp[s];
            oep[j] = sr->last_slot;
        }
    }
}

/* Parse a CLI list "2,6,...,128" into sorted, de-duplicated even bounds.
   Returns the count, or -1 with a message in err/n. */
static int sr_parse_list(const char *s, uint64_t *out, int max,
                         char *err, size_t n) {
    int k = 0, i, j, w;
    const char *p = s;
    if (!s || !*s) {
        if (err) snprintf(err, n, "empty bound list");
        return -1;
    }
    while (*p) {
        unsigned long long v;
        char *end = NULL;
        v = strtoull(p, &end, 10);
        if (end == p) {
            if (err) snprintf(err, n, "bad number at \"%s\"", p);
            return -1;
        }
        if (v < 2 || v > 1000000ULL || (v & 1ULL)) {
            if (err) snprintf(err, n, "bound %llu must be even and in [2,1000000]",
                              v);
            return -1;
        }
        if (k >= max) {
            if (err) snprintf(err, n, "at most %d bounds", max);
            return -1;
        }
        out[k++] = (uint64_t)v;
        p = end;
        if (*p == ',') p++;
        else if (*p) {
            if (err) snprintf(err, n, "bad separator at \"%s\"", p);
            return -1;
        }
    }
    if (k == 0) {
        if (err) snprintf(err, n, "empty bound list");
        return -1;
    }
    for (i = 1; i < k; i++) {              /* insertion sort: tiny k */
        uint64_t x = out[i];
        j = i - 1;
        while (j >= 0 && out[j] > x) { out[j + 1] = out[j]; j--; }
        out[j + 1] = x;
    }
    w = 1;
    for (i = 1; i < k; i++)
        if (out[i] != out[w - 1]) out[w++] = out[i];
    return w;
}

#endif /* SMOOTH_RANGE_H */
