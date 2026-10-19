/*
 * smooth_pattern.c - maximum ADMISSIBLE "smooth run" patterns.
 *
 * For a gap bound g, a smooth run is a block of consecutive prime gaps that are
 * all <= g.  Its length (in gaps) is bounded by a purely combinatorial object:
 * a sequence of gaps d_1..d_k, each even and in [2,g], whose offset set
 *     O = {0, d_1, d_1+d_2, ..., d_1+...+d_k}
 * is ADMISSIBLE - i.e. for every prime q <= span(O) the offsets do not cover
 * all q residue classes (otherwise one member of the tuple is divisible by q
 * whatever the starting prime, and no run of that shape can exist).
 *
 * BriK's table in mersenneforum node/22899 lists these ceilings by hand
 * (maxGapSize 2/4/6/8/10/12 -> 1/5/18/40/74/164 gaps).  This tool recomputes
 * them by exhaustive DFS inside a span cap, so a census result can be compared
 * against the ceiling instead of against other people's scans:
 *
 *     bin/smooth_pattern G SPANCAP [MAXSECONDS]
 *
 * Output: best gap count, the gap sequence (for g <= 10), the span, and whether
 * the search was exhaustive within the cap (and the node count / wall time).
 *
 * DFS pruning: incremental residue counts per prime (a prime that becomes fully
 * covered prunes the branch immediately), plus the bound "remaining gaps cannot
 * exceed (spanCap - span)/2" so branches that cannot beat the incumbent die.
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime under -std=c99 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define NPMAX 1024             /* primes considered (spans up to ~8000) */
#define KMAX 600               /* longest gap sequence we can hold */

static int primes[NPMAX], nprimes;
static int *cnt;               /* residue counters, one block per prime */
static int cnt_off[NPMAX];
static char is_full[NPMAX];
static int active_n;           /* primes[0..active_n) are <= the current span */
static int full_count;         /* primes whose residues are fully covered */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void gen_primes(int lim) {
    char *s = (char *)calloc((size_t)lim + 1, 1);
    nprimes = 0;
    for (int i = 2; i <= lim && nprimes < NPMAX; i++) {
        if (s[i]) continue;
        primes[nprimes++] = i;
        for (long j = (long)i * i; j <= lim; j += i) s[j] = 1;
    }
    free(s);
}

/* primes with p <= span contribute a coverage constraint; primes above the
   span can never be fully covered (there are at most span/2+1 offsets), so a
   prime is only tracked once the span reaches it. */
static void prime_check_full(int i) {
    int q = primes[i], *c = cnt + cnt_off[i];
    if (is_full[i]) return;
    for (int r = 0; r < q; r++) {
        if (c[r] == 0) return;
    }
    is_full[i] = 1;
    full_count++;
}

/* offsets[] holds offs[0..k] (offs[0] = 0, offs[k] = current span) */
static int offs[KMAX + 2];

/* append offset x at level k (offs[0..k] already valid); x becomes offs[k+1] */
static void push_offset(int k, int x) {
    int prev_span = offs[k];
    offs[k + 1] = x;
    /* activate primes the new span has just reached: rebuild their counters
       from every offset present right now */
    while (active_n < nprimes && primes[active_n] <= x) {
        int q = primes[active_n];
        int *c = cnt + cnt_off[active_n];
        memset(c, 0, sizeof(int) * (size_t)q);
        for (int i = 0; i <= k + 1; i++) c[offs[i] % q]++;
        is_full[active_n] = 0;
        prime_check_full(active_n);
        active_n++;
    }
    /* primes that were already active: add the new offset */
    for (int i = 0; i < active_n; i++) {
        if (primes[i] > prev_span) continue;   /* just activated above */
        int q = primes[i];
        int *c = cnt + cnt_off[i];
        c[x % q]++;
        prime_check_full(i);
    }
}

/* remove offs[k+1]; the span becomes offs[k] again */
static void pop_offset(int k) {
    int x = offs[k + 1];
    int new_span = offs[k];
    /* primes beyond the new span stop being constraints */
    while (active_n > 0 && primes[active_n - 1] > new_span) {
        active_n--;
        if (is_full[active_n]) { is_full[active_n] = 0; full_count--; }
    }
    for (int i = 0; i < active_n; i++) {
        int q = primes[i];
        if (q > x) continue;
        int *c = cnt + cnt_off[i];
        c[x % q]--;
        if (is_full[i]) { is_full[i] = 0; full_count--; }
        prime_check_full(i);
    }
}

static int g_bound, span_cap, best_k;
static int gaps[KMAX + 1], best_gaps[KMAX + 1];
static long long nodes;
static double t_start, t_cap;
static int timed_out;

static void dfs(int k, int span) {
    nodes++;
    /* bound: with the remaining span we can add at most (spanCap-span)/2 gaps */
    if (k + (span_cap - span) / 2 <= best_k) return;
    if (!timed_out && (nodes & 0x3FFFF) == 0 && now_s() - t_start > t_cap)
        timed_out = 1;
    if (timed_out) return;
    for (int d = 2; d <= g_bound; d += 2) {
        int ns = span + d;
        if (ns > span_cap) break;
        push_offset(k, ns);
        if (full_count == 0) {
            gaps[k] = d;
            if (k + 1 > best_k) {
                best_k = k + 1;
                memcpy(best_gaps, gaps, sizeof(int) * (size_t)(k + 1));
            }
            if (k + 1 < KMAX) dfs(k + 1, ns);
        }
        pop_offset(k);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s G SPANCAP [MAXSECONDS]\n", argv[0]);
        return 2;
    }
    g_bound = atoi(argv[1]);
    span_cap = atoi(argv[2]);
    t_cap = (argc > 3) ? atof(argv[3]) : 600.0;
    if (g_bound < 2 || (g_bound & 1) || span_cap < 2 || span_cap > 3600) {
        fprintf(stderr, "need even G >= 2 and 2 <= SPANCAP <= 3600\n");
        return 2;
    }
    gen_primes(span_cap + 2);
    /* residue-count blocks: q counters + 1 "is full" slot per prime */
    int total = 0;
    for (int i = 0; i < nprimes; i++) { cnt_off[i] = total; total += primes[i] + 1; }
    cnt = (int *)calloc((size_t)total, sizeof(int));
    if (!cnt) return 1;
    t_start = now_s();
    best_k = 0;
    offs[0] = 0;
    dfs(0, 0);
    double dt = now_s() - t_start;
    int span = 0;
    for (int i = 0; i < best_k; i++) span += best_gaps[i];

    printf("g=%d span_cap=%d  best=%d gaps (%d primes)  span=%d  nodes=%lld  %.2fs%s\n",
           g_bound, span_cap, best_k, best_k + 1, span, nodes, dt,
           timed_out ? "  [TIMEOUT - not exhaustive]" : "");
    if (best_k && g_bound <= 12) {
        printf("  gaps:");
        for (int i = 0; i < best_k; i++) printf(" %d", best_gaps[i]);
        printf("\n  offsets:");
        int s = 0;
        for (int i = 0; i < best_k; i++) { s += best_gaps[i]; printf(" %d", s); }
        printf("\n");
    }
    free(cnt);
    return 0;
}
