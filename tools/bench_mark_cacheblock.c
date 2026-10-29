/*
 * bench_mark_cacheblock.c - does cache-blocked DUPLICATED work beat streaming
 * one prime at a time over the whole odd-slot bitmap, at OUR sizes?
 *
 * The idea under test comes from Math::Prime::Util::GMP siqs/IMPLEMENTATION.md:
 * "Our byte sieve applies dense primes in cache-sized blocks, then applies
 * sparse primes across the full interval" and "Duplicating work, as long as it
 * keeps the working set in cache, can be much faster than monolithic work even
 * with a higher raw instruction count."
 *
 * Two layouts over the same bit-packed odd-slot bitmap (bit=1 -> composite),
 * both marking exactly the same slot set:
 *
 *   A  stream    for p ascending: mark start, start+p, ... < W.  The whole
 *                bitmap is streamed once per prime, so the working set per
 *                prime is W/8 bytes.  (Production shape on the GPU: one
 *                thread per (window, prime) walking the full window.)
 *
 *   C  blocked   for each tile of T slots: for every prime p <= T, derive the
 *                first multiple inside the tile (one reciprocal fastmod) and
 *                mark only hits; primes p > T are then streamed as in A.
 *                The working set per tile is T/8 bytes, but every (tile,
 *                dense prime) pair pays a setup and a loop test even when the
 *                prime has nothing to mark -- the DUPLICATED work.
 *
 * Both variants use the repository's reciprocal fastmod (sieve_fast_mod in
 * new_src/sieve_core.c) for the first-multiple derivation, so the duplication
 * is priced at production cost, not at a 64-bit hardware division.
 *
 * A variant cannot win by doing less work: every run hashes the whole bitmap
 * and the hash must equal A's.
 *
 * Build: make bin/bench_mark_cacheblock
 * Usage: ./bin/bench_mark_cacheblock [--primes N] [--reps R] [--one W T]
 *
 * Default run prints the geometry sweep (W from 2.2 KB to 2 MB of bitmap,
 * T = 4 KB slots) and then the tile-size sweep at the walk-engine tile.
 *
 * ---------------------------------------------------------------------------
 * MEASURED 2026-10-29, i3-10100 (L1d 32 KB / L2 256 KB / L3 6 MB), quiet box,
 * one thread, 148,932 odd primes <= 2e6, min of 7 runs after 3 warmups,
 * every run bit-identical across A and C (parity ok):
 *
 *   geometry                                 A ms     C ms   C/A   dense A->C
 *   gap_hunt window 17916 slots (2,240 B)   0.156    0.167  1.07  0.030 -> 0.039
 *   walk tile     753664 slots (94,208 B)   1.88     2.34   1.24  1.20  -> 1.66
 *   walk tile, least-duplication T=131072   1.89     2.14   1.13  1.43  -> 1.68
 *   bitmap 256 KB (2 M slots)               5.23     6.54   1.25  1.31x
 *   bitmap 512 KB (4 M slots)               10.94    12.73  1.16  1.21x
 *   bitmap 2 MB   (16 M slots)              47.44    52.39  1.10  1.16x
 *
 * This box is noisy for sub-millisecond work: A's own run-to-run spread
 * (the A%sp column) reached 100% on some rows, so no single row should be read
 * on its own.  The verdict rests on two paired statistics that are stable
 * across all 15 configurations: Cd/Ad (both phases timed in the same rep loop)
 * never went below 1.04, and C/A never went below 1.01 at --reps 7.
 *
 * VERDICT: cache-blocked duplication LOSES at every size we use.  The dense
 * phase alone is 1.04-1.51x slower when blocked; the only near-parity points
 * are T = W (no blocking at all) and the 16 KB tile, where the duplication
 * factor is lowest and the tile is still L2-resident.  Two reasons, both
 * visible in the same table:
 *
 *   1. setups/marks is 0.04-0.07 and does NOT depend on W -- the duplicated
 *      (tile, prime) setups scale exactly like the marks, so a larger interval
 *      never repays them.  Only T moves that ratio, and a larger T gives back
 *      the working-set advantage.
 *   2. our bitmaps (2.2 KB, 92 KB) are already L1/L2-resident, so streaming
 *      one prime across the whole bitmap is cheap; there is no miss traffic to
 *      recover.
 *
 * What the same numbers say the REAL lever is: at the hunt window ~82% of the
 * mark time (0.128 of 0.156 ms) goes to the 99.6% of primes that set <=1 bit
 * (148,368 primes, 23% of the marks).  The mark is ITEM-count bound, not bandwidth bound -- the
 * same conclusion the GPU side already reached (GPU_MARK_SPLIT_SLOTS: coarser
 * chunks, i.e. fewer duplicated items, won +5-6% end-to-end).  MPU's win comes
 * mostly from vectorized dense masks, not from the cache; this repo already
 * uses that where it pays (sieve_mark_low_prime_avx2, p <= 63, mask table
 * O(sum p) = 127 KB -- which is why it cannot simply be widened to p <= 4096).
 * ---------------------------------------------------------------------------
 */

#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* ---- prime table (odd primes, ascending) ------------------------------- */

static uint32_t *g_primes;
static uint64_t *g_inv;   /* floor(2^64 / p), the sieve_core reciprocal */
static uint64_t *g_pos;   /* per-prime start position in [0, p), precomputed so
                             the hardware division does not dominate both
                             layouts (production caches residues too) */
static size_t g_nprimes;

static void build_primes(uint64_t limit) {
    size_t cap = (size_t)(limit / 2 + 16);
    uint8_t *comp = calloc(cap, 1);
    if (!comp) { fprintf(stderr, "oom\n"); exit(1); }
    for (uint64_t i = 3; i * i <= limit; i += 2) {
        if (comp[i >> 1]) continue;
        for (uint64_t j = i * i; j <= limit; j += 2 * i) comp[j >> 1] = 1;
    }
    g_primes = malloc(sizeof(*g_primes) * (cap / 2 + 16));
    g_inv = malloc(sizeof(*g_inv) * (cap / 2 + 16));
    g_pos = malloc(sizeof(*g_pos) * (cap / 2 + 16));
    if (!g_primes || !g_inv || !g_pos) { fprintf(stderr, "oom\n"); exit(1); }
    size_t n = 0;
    for (uint64_t i = 3; i <= limit; i += 2) {
        if (comp[i >> 1]) continue;
        g_primes[n] = (uint32_t)i;
        g_inv[n] = UINT64_MAX / i;
        g_pos[n] = (((uint64_t)i * 0x9E3779B97F4A7C15ULL) >> 17) % i;
        n++;
    }
    free(comp);
    g_nprimes = n;
}

/* sieve_core.c's branchless x mod p: inv = floor(2^64 / p). */
static inline uint64_t fast_mod(uint64_t x, uint64_t p, uint64_t inv) {
    uint64_t q = (uint64_t)(((unsigned __int128)x * (unsigned __int128)inv) >> 64);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}

static inline void mark(uint64_t *bm, uint64_t pos) {
    bm[pos >> 6] |= 1ULL << (pos & 63);
}

static uint64_t hash_bitmap(const uint64_t *bm, uint64_t words) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint64_t i = 0; i < words; i++) {
        h ^= bm[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

/* ---- layout A: stream one prime at a time over the whole bitmap -------- */

/* Marks primes[lo, hi) in A's shape.  The [lo, hi) bounds are computed by the
   caller OUTSIDE the timer: scanning the table for the dense cutoff inside a
   timed region cost more than the marking itself and inflated every phase. */
static uint64_t stream_phase(uint64_t *bm, uint64_t W, size_t lo, size_t hi) {
    uint64_t marks = 0;
    for (size_t i = lo; i < hi; i++) {
        uint64_t p = g_primes[i];
        for (uint64_t pos = g_pos[i]; pos < W; pos += p) {
            mark(bm, pos);
            marks++;
        }
    }
    return marks;
}

static uint64_t layout_stream(uint64_t *bm, uint64_t W) {
    uint64_t marks = 0;
    for (size_t i = 0; i < g_nprimes; i++) {
        uint64_t p = g_primes[i];
        for (uint64_t pos = g_pos[i]; pos < W; pos += p) {
            mark(bm, pos);
            marks++;
        }
    }
    return marks;
}

/* ---- layout C: tile-major for dense primes, stream the sparse ones ----- */

struct blocked_stats {
    uint64_t marks;
    uint64_t dense_marks;
    uint64_t setups;       /* (tile, dense prime) pairs = duplicated setups */
};

/* Tile-major dense pass only: the cache-blocked + duplicated part. */
static uint64_t blocked_dense(uint64_t *bm, uint64_t W, uint64_t T,
                              size_t ndense, uint64_t *setups_out) {
    uint64_t marks = 0, setups = 0;
    for (uint64_t t0 = 0; t0 < W; t0 += T) {
        uint64_t t1 = t0 + T;
        if (t1 > W) t1 = W;
        for (size_t i = 0; i < ndense; i++) {
            uint64_t p = g_primes[i];
            uint64_t pos = g_pos[i];
            if (pos < t0) {
                uint64_t rem = fast_mod(t0 - pos, p, g_inv[i]);
                pos = (rem == 0) ? t0 : t0 + (p - rem);
            }
            setups++;
            for (; pos < t1; pos += p) {
                mark(bm, pos);
                marks++;
            }
        }
    }
    if (setups_out) *setups_out = setups;
    return marks;
}

static void layout_blocked(uint64_t *bm, uint64_t W, uint64_t T, size_t ndense,
                           struct blocked_stats *st) {
    st->dense_marks = blocked_dense(bm, W, T, ndense, &st->setups);
    st->marks = st->dense_marks + stream_phase(bm, W, ndense, g_nprimes);
}

static size_t dense_count(uint64_t T) {
    size_t n = 0;
    while (n < g_nprimes && (uint64_t)g_primes[n] <= T) n++;
    return n;
}

/* ---- driver ------------------------------------------------------------ */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Minimum, not median.  The kernel is deterministic and single-threaded on a
   quiet box, so every source of error only ADDS time; the min is the cleanest
   estimate.  A median lets one preempted run of A alone produce a bogus
   "C wins" row (observed: A 0.191 vs 0.154 ms while C sat at 0.169). */
static double best_ms(int reps, double *v) {
    double m = v[0];
    for (int i = 1; i < reps; i++) if (v[i] < m) m = v[i];
    return m;
}

static double worst_ms(int reps, double *v) {
    double m = v[0];
    for (int i = 1; i < reps; i++) if (v[i] > m) m = v[i];
    return m;
}

#define WARMUPS 3

struct row {
    uint64_t W, T;
    double a_ms, c_ms;
    double a_dense_ms, c_dense_ms, a_sparse_ms;
    double a_spread_pct;   /* (max-min)/min of A: the run-to-run noise, shown
                              so a 15% jump on a 0.15 ms row cannot be read as
                              a result */
    uint64_t a_marks, c_marks, dense_marks, setups;
    int identical;
};

static struct row measure(uint64_t W, uint64_t T, int reps) {
    uint64_t words = (W + 63) / 64;
    size_t bytes = (size_t)words * sizeof(uint64_t);
    uint64_t *bm = malloc(bytes);
    double *ta = malloc(sizeof(double) * (size_t)reps);
    double *tc = malloc(sizeof(double) * (size_t)reps);
    double *tad = malloc(sizeof(double) * (size_t)reps);
    double *tcd = malloc(sizeof(double) * (size_t)reps);
    double *tas = malloc(sizeof(double) * (size_t)reps);
    if (!bm || !ta || !tc || !tad || !tcd || !tas) { fprintf(stderr, "oom\n"); exit(1); }

    size_t ndense = dense_count(T);

    /* warmup: untimed, both layouts, so neither pays the first-touch cost */
    for (int w = 0; w < WARMUPS; w++) {
        memset(bm, 0, bytes);
        layout_stream(bm, W);
        memset(bm, 0, bytes);
        blocked_dense(bm, W, T, ndense, NULL);
        stream_phase(bm, W, ndense, g_nprimes);
    }

    memset(bm, 0, bytes);
    uint64_t a_marks = layout_stream(bm, W);
    uint64_t h_a = hash_bitmap(bm, words);

    struct blocked_stats st;
    memset(&st, 0, sizeof(st));
    memset(bm, 0, bytes);
    layout_blocked(bm, W, T, ndense, &st);
    uint64_t h_c = hash_bitmap(bm, words);

    for (int r = 0; r < reps; r++) {
        memset(bm, 0, bytes);
        double t0 = now_ms();
        layout_stream(bm, W);
        ta[r] = now_ms() - t0;

        memset(bm, 0, bytes);
        t0 = now_ms();
        layout_blocked(bm, W, T, ndense, &st);
        tc[r] = now_ms() - t0;

        memset(bm, 0, bytes);
        t0 = now_ms();
        stream_phase(bm, W, 0, ndense);
        tad[r] = now_ms() - t0;

        memset(bm, 0, bytes);
        t0 = now_ms();
        blocked_dense(bm, W, T, ndense, NULL);
        tcd[r] = now_ms() - t0;

        memset(bm, 0, bytes);
        t0 = now_ms();
        stream_phase(bm, W, ndense, g_nprimes);
        tas[r] = now_ms() - t0;
    }

    struct row row;
    memset(&row, 0, sizeof(row));
    row.W = W;
    row.T = T;
    row.a_ms = best_ms(reps, ta);
    row.c_ms = best_ms(reps, tc);
    row.a_dense_ms = best_ms(reps, tad);
    row.c_dense_ms = best_ms(reps, tcd);
    row.a_sparse_ms = best_ms(reps, tas);
    {
        double lo = best_ms(reps, ta), hi = worst_ms(reps, ta);
        row.a_spread_pct = lo > 0 ? 100.0 * (hi - lo) / lo : 0.0;
    }
    row.a_marks = a_marks;
    row.c_marks = st.marks;
    row.dense_marks = st.dense_marks;
    row.setups = st.setups;
    row.identical = (h_a == h_c) && (a_marks == st.marks);

    printf("%8" PRIu64 " %6" PRIu64 " %7zu %8.3f %8.3f %6.2f %5.1f %8.3f %8.3f %6.2f"
           " %8.3f %9" PRIu64 " %5.1f %6.2f %s\n",
           row.W, row.T, bytes, row.a_ms, row.c_ms,
           row.a_ms > 0 ? row.c_ms / row.a_ms : 0.0,
           row.a_spread_pct,
           row.a_dense_ms, row.c_dense_ms,
           row.a_dense_ms > 0 ? row.c_dense_ms / row.a_dense_ms : 0.0,
           row.a_sparse_ms, row.a_marks,
           row.a_marks ? 100.0 * (double)row.dense_marks / (double)row.a_marks : 0.0,
           row.a_marks ? (double)row.setups / (double)row.a_marks : 0.0,
           row.identical ? "ok" : "MISMATCH");

    free(bm); free(ta); free(tc); free(tad); free(tcd); free(tas);
    return row;
}

int main(int argc, char **argv) {
    uint64_t primes = 2000000ULL;
    int reps = 7;
    uint64_t one_w = 0, one_t = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--primes") && i + 1 < argc) primes = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--one") && i + 2 < argc) {
            one_w = strtoull(argv[i + 1], NULL, 10);
            one_t = strtoull(argv[i + 2], NULL, 10);
            i += 2;
        }
        else { fprintf(stderr, "usage: %s [--primes N] [--reps R] [--one W T]\n", argv[0]); return 2; }
    }
    if (reps < 1) reps = 1;
    if (reps % 2 == 0) reps++;

    build_primes(primes);
    printf("# primes <= %" PRIu64 " (%zu odd primes), min of %d runs, %d warmups,"
           " one thread, scalar marks\n", primes, g_nprimes, reps, WARMUPS);
    const char *hdr =
        "  W(slots)    T  bitmap_B     A ms     C ms   C/A  A%sp  A dense  C dense  Cd/Ad"
        "  A sparse    marks dense% setup/mk parity\n";

    if (one_w) {
        measure(one_w, one_t ? one_t : 4096, reps);
        free(g_primes); free(g_inv); free(g_pos);
        return 0;
    }

    printf("\n## working-set sweep, T = 4096 slots (512 B tiles)\n%s", hdr);
    static const uint64_t ws[] = { 17916, 65536, 131072, 524288, 2097152 };
    for (size_t i = 0; i < sizeof(ws) / sizeof(ws[0]); i++) measure(ws[i], 4096, reps);

    printf("\n## gap_hunt window (17916 slots = 2.2 KB), tile sweep\n%s", hdr);
    static const uint64_t ts1[] = { 1024, 2048, 4096, 8192, 17916 };
    for (size_t i = 0; i < sizeof(ts1) / sizeof(ts1[0]); i++) measure(17916, ts1[i], reps);

    printf("\n## walk-engine tile (753664 slots = 92 KB), tile sweep\n%s", hdr);
    static const uint64_t ts2[] = { 4096, 16384, 32768, 65536, 131072 };
    for (size_t i = 0; i < sizeof(ts2) / sizeof(ts2[0]); i++) measure(753664, ts2[i], reps);

    free(g_primes);
    free(g_inv);
    free(g_pos);
    return 0;
}
