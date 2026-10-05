/*
 * phase0_scan_gpu.cu - live Phase-0 exhaustive prime-gap scanner, GPU test stage.
 *
 * Pipeline (per thread slice): bucketed segmented sieve (ported from
 * tools/phase0_scan.c) -> survivor candidates buffered in ascending order ->
 * batched base-2 Miller-Rabin on the GPU (tools/mr68_kernel.cuh, the kernel
 * validated by bin/mr68_gpu --validate) -> consecutive-prime gap detection
 * (merit = gap / ln(lower prime), mirroring phase0_scan) -> GMP verification
 * of every reported gap -> automatic record check against the best-known-merit
 * table (criterion identical to the gap-hunt watcher and new_src/record_log.c)
 * -> text log.
 *
 * Live features: progress lines with ETA, checkpoint file (resume after
 * Ctrl-C / crash / reboot), append-only gap log in real time, graceful
 * SIGINT / SIGTERM, saturated GPU with a fail-closed CUDA path.
 *
 * Gates (run before trusting timings or results):
 *   --check N : the FULL production path is walked over [start, start+N) and
 *               its prime sequence compared against a pure GMP mpz_nextprime
 *               walk; any mismatch aborts.
 *   gap log   : every logged gap carries verified=1 from
 *               mpz_probab_prime_p(lower, 25) >= 1  AND
 *               mpz_nextprime(lower) == upper.
 *
 * Test strength: the kernel is a base-2 strong probable prime test. Composite
 * verdicts are conclusive; "prime" verdicts carry the usual base-2 caveat
 * (base-2 strong pseudoprimes: density ~1.8e-15 near 2^64, none known near
 * 2e20). A pseudoprime could only SPLIT a real gap (making us miss it); it
 * cannot create a false gap, and every reported gap is GMP-re-verified below.
 *
 * Limits: start > sieve-limit (primes in range must exceed the sieve), and
 * start + length <= 2^96 (the kernel container is 3 x 32-bit limbs);
 * length < 2^63.
 *
 * Live-tuned defaults (re-measured 2026-09-30 AFTER removing the per-candidate
 * cross-core atomics, which had dominated walk+proc): --threads 8 and
 * --sieve-limit 30000.  The optimum MOVED when the atomics went away: bucket
 * probe count now costs more than survivor count, so a smaller prime table
 * (32k vs 1m primes) wins 2.22e9 vs 1.63e9 ints/s.  See
 * docs/PHASE0_scan_bench.md section 10.
 *
 * Build:  make bin/phase0_scan_gpu WITH_CUDA=1
 * Usage:  see usage()
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <pthread.h>
#include <cuda_runtime.h>
#include <gmp.h>

/* Host-safe Phase-0 types (P0Item/P0_CHUNK/w_gaprec_t/p0_vis_mask30) are
   needed in BOTH build modes: the host builds the item tables itself. */
#include "p0_types.h"

#ifdef PHASE0_KERNEL_DLL
/* Windows kernel-DLL split (README_WINDOWS.md): the kernels live in
   phase0gpu.dll (nvcc + MSVC); this host code is built by MinGW and calls the
   wrappers in phase0gpu_api.h.  The .cuh kernel headers must NOT be included
   here - MSVC cannot compile their device code. */
#include "phase0gpu_api.h"
#else
#include "mr68_kernel.cuh"
#include "p0_mark.cuh"
#include "perig.cuh"
#endif

typedef unsigned __int128 u128;

#define N_BANDS 7
static const double k_bands[N_BANDS] = {10, 15, 20, 25, 30, 35, 40};

/* ================= walk engine (--engine walk) =============================
   class-30 bitmap sieve + batched jump-walk, ported from tools/walk_engine.cu
   (Phase-0 parity prototype; gates: sieve VERIFY arbiter false=0 miss=0 of
   35.8M slots, 22/22 set-gate vs the odd-slot engine, NB=1 identical).
   block = 30*2^25 = 1,006,632,960 numbers = 2^28 slots (32 MB bitmap),
   tile = 2^19 slots (64 KB shared), wheel {7,11,13} as a precomputed group
   pattern, item table = (prime, 64-mark chunk) rows with per-prime setup
   reuse in contiguous per-thread runs.                                      */
#ifndef PHASE0_KERNEL_DLL
#include "p0_walk_kern.cuh"   /* class-30 sieve + walk kernels (moved here
                                  2026-10-02 for the Windows kernel-DLL split;
                                  also defines w_gaprec_t via p0_types.h) */
#endif

#define MAX_THREADS 64

/* ---------------- configuration / globals ---------------- */

static u128 g_start = 0;                 /* range start (absolute) */
static uint64_t g_len = 0;               /* range length */
static int g_nthreads = 4;

static uint32_t *g_primes;
static size_t    g_nprimes;

static int      g_device = 0;
static uint32_t g_batch = 1024u * 1024u;
static double   g_merit_min = 20.0;
static int      g_merit_explicit = 0;   /* --merit-min given on the command line */
static double   g_progress = 10.0;       /* seconds between progress lines (0 = off) */
static double   g_state_every = 30.0;    /* seconds between checkpoint writes */
static const char *g_log_path = NULL;
static const char *g_state_path = NULL;
static const char *g_records_path = NULL;   /* NULL = default path if present */
static int         g_no_records = 0;
static FILE *g_log = NULL;

/* best-known-merit table: dense array indexed by gap length, 0 = absent */
static double     *g_table = NULL;
static uint64_t    g_table_len = 0;
static char        g_table_stamp[192] = "off";

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_mon_stop = 0;

static uint32_t *g_d_steps = NULL;
static uint32_t *g_d_base3 = NULL;
static uint8_t  *g_d_res = NULL;
static pthread_mutex_t g_gpu_mtx = PTHREAD_MUTEX_INITIALIZER;

/* --gpu-sieve: GPU bitmap marking (shared p0_mark kernel; device-wide table,
   per-thread double-buffered bitmaps).  Default ON since 2026-09-30: measured
   6.9e9 vs 3.6e9 ints/s (see docs/PHASE0_scan_bench.md section 10). */
static int       g_use_gpu_sieve = 1;
static P0Item   *g_d_items = NULL;
static uint64_t *g_d_primes64 = NULL;
static uint64_t *g_d_r64 = NULL;
static uint64_t *g_d_invp = NULL;     /* (2^64-1)/p per prime */
static uint32_t  g_nitems = 0;
static uint64_t  g_bm_bytes = 0;      /* bitmap bytes per (full) segment */
static uint32_t  g_mark_tpb = 256;
static uint64_t *g_d_wpat = NULL;     /* doubled wheel pattern on device */
static uint64_t  g_wheel_r64p = 0;    /* 2^64 mod WHEEL_P */

/* progress: per-thread counters live in job_t and are updated with plain
   stores.  A cross-core atomic per candidate serialises six threads on one
   cache line (measured 2026-09-30: it dominated the walk and proc terms).
   The monitor reads them with relaxed loads; final sums are taken after join. */
typedef struct job_s job_t;   /* full definition below */
static job_t *g_jobs = NULL;
static uint64_t g_pre_ints = 0;     /* ints done before this session (resume) */
static uint64_t g_total_ints = 0;
static uint64_t g_gap_gate = 0;     /* min gap for merit work; see flush_batch */
static int g_fast_mr = 1;           /* bitmap-verdict MR pipeline (--legacy-mr disables) */
/* --engine walk (class-30 sieve + batched jump-walk) */
static int      g_walk_engine = 0;
static uint64_t g_walk_gapmin = 0;      /* 0 = derive from merit_min */
static uint32_t g_walk_primes = 15000;  /* sieve/item primes 17..P */
static uint32_t g_walk_batch = 64;      /* blocks per super-batch (4..96);
                                           default 64 since 2026-10-03 (+2.45%
                                           ABBA vs 32; K=96 no better) */
static int g_fast_dbg = 0;          /* P0_FAST_DBG=N: dump N fast-walk summaries */

/* wheel {3,5,7,11,13}: one bit per odd-value slot, period WHEEL_P slots.
   Bit i of the un-rotated pattern is set iff slot c is a multiple of a wheel
   prime; the pattern is rotated per segment by shift = v0 * inv2 (mod P) so
   that bit i means "value v0 + 2i is divisible by a wheel prime".  This
   replaces ~40 % of the marking writes (and the per-segment memset) by a
   tiled copy of a 3.8 KB doubled pattern. */
#define WHEEL_P 15015u           /* 3*5*7*11*13, period in slots */
#define WHEEL_INV2 7508u         /* inverse of 2 mod WHEEL_P */
static uint64_t g_wpat2[(2 * 15015 + 63) / 64];
static size_t   g_bucket0 = 0;       /* first prime index above the wheel */

static void wheel_init(void) {
    static const uint32_t qs[5] = {3u, 5u, 7u, 11u, 13u};
    memset(g_wpat2, 0, sizeof g_wpat2);
    for (int k = 0; k < 5; k++) {
        uint32_t q = qs[k];
        for (uint32_t c = 0; c < WHEEL_P; c += q) {
            g_wpat2[c >> 6] |= 1ULL << (c & 63u);
            uint32_t c2 = c + WHEEL_P;
            g_wpat2[c2 >> 6] |= 1ULL << (c2 & 63u);
        }
    }
}

/* per-thread resume offset: first UNprocessed candidate, relative to g_start.
   Written by the owning worker immediately after each completed GPU batch
   (single aligned store after a barrier), read by the monitor for the
   checkpoint file.  Guarantee: everything below the offset is fully processed
   (tested + gap-tracked), so resume can never re-emit a gap. */
static volatile uint64_t g_resume_off[MAX_THREADS];

static double g_t0 = 0.0;

static void on_signal(int sig) { (void)sig; g_stop = 1; }

static void die(const char *msg) {
    fprintf(stderr, "[phase0-gpu] FATAL: %s\n", msg);
    exit(1);
}

static void die_cuda(cudaError_t e, const char *what) {
    fprintf(stderr, "[phase0-gpu] FATAL: %s: %s\n", what, cudaGetErrorString(e));
    exit(1);
}

/* ---------------- small helpers (ported from tools/phase0_scan.c) ---------------- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double u128_ln(u128 x) { return (double)logl((long double)x); }

static u128 parse_u128(const char *s) {
    u128 v = 0;
    for (; *s; s++) {
        if (*s < '0' || *s > '9') break;
        v = v * 10 + (unsigned)(*s - '0');
    }
    return v;
}

static void u128_str(u128 v, char *out, size_t n) {
    char tmp[48];
    int i = 0;
    if (v == 0) { snprintf(out, n, "0"); return; }
    while (v != 0 && i < 47) { tmp[i++] = (char)('0' + (int)(v % 10)); v /= 10; }
    size_t j = 0;
    while (i > 0 && j + 1 < n) out[j++] = tmp[--i];
    out[j] = 0;
}

static void u128_to_mpz(mpz_t out, u128 x) {
    uint64_t w[2] = { (uint64_t)x, (uint64_t)(x >> 64) };
    mpz_import(out, 2, -1, 8, 0, 0, w);
}

static u128 mpz_to_u128(const mpz_t in) {
    uint64_t w[2] = { 0, 0 };
    size_t cnt = 0;
    mpz_export(w, &cnt, -1, 8, 0, 0, in);
    return ((u128)w[1] << 64) | (u128)w[0];
}

static int u128_is_probable(u128 x, mpz_t scratch) {
    u128_to_mpz(scratch, x);
    return mpz_probab_prime_p(scratch, 30) >= 1;
}

/* ---------------- odd-prime table (3,5,7,...) ---------------- */

static void gen_primes(uint32_t limit) {
    size_t half = (size_t)(limit / 2) + 1;
    uint8_t *comp = (uint8_t *)calloc(half, 1);
    if (!comp) die("primes alloc failed");
    for (size_t k = 1; (uint64_t)(2 * k + 1) * (2 * k + 1) <= (uint64_t)limit; k++) {
        if (comp[k]) continue;
        uint64_t p = 2 * k + 1;
        for (uint64_t m = p * p; m <= (uint64_t)limit; m += 2 * p)
            comp[m / 2] = 1;
    }
    size_t cnt = 0;
    for (size_t k = 1; k < half; k++) if (!comp[k]) cnt++;
    g_primes = (uint32_t *)malloc((cnt ? cnt : 1) * sizeof(uint32_t));
    if (!g_primes) die("primes alloc failed");
    size_t j = 0;
    for (size_t k = 1; k < half; k++) if (!comp[k]) g_primes[j++] = (uint32_t)(2 * k + 1);
    g_nprimes = j;
    free(comp);
    while (g_bucket0 < g_nprimes && g_primes[g_bucket0] <= 13u) g_bucket0++;
    fprintf(stderr, "[phase0-gpu] sieve primes up to %u: %zu odd primes (%zu in buckets after the wheel)\n",
            limit, g_nprimes, g_nprimes - g_bucket0);
}

/* ---------------- job state ---------------- */

struct job_s {
    int tid;
    u128 start;                /* first candidate to scan (absolute; resume point) */
    u128 end;                  /* slice end (absolute, exclusive) */
    uint32_t seg_bits;
    int do_test;
    int completed;
    /* collection (--check) */
    u128 *collect;
    size_t collect_cap, collect_n;
    /* candidate batch transport: first candidate (base3/batch_base) + a u32
       step per candidate, n = batch_base + 2*step (3x less host traffic and
       3x less H2D than shipping 96-bit values) */
    uint32_t *steps;           /* cap u32 steps */
    uint32_t  base3[3];        /* first candidate of the current batch */
    u128      batch_base;
    uint64_t  bi;              /* slot index of the batch base, relative to v_anchor */
    u128      v_anchor;        /* slot-0 value of this thread's scan (fixed anchor;
                                  all candidates are v_anchor + 2*A for A >= 0) */
    uint8_t  *res;             /* cap verdict bytes */
    uint32_t  cap, n;
    /* --gpu-sieve resources: per-thread stream + double-buffered bitmaps */
    cudaStream_t sstream;
    cudaStream_t cstream;      /* fast path: D2H copies on their own stream */
    cudaEvent_t  sev[2];
    cudaEvent_t  mev[2][4];    /* fast path: mark/mr/copy split, per buffer */
    double       t_mm, t_mr, t_cp;
    uint64_t    *d_bm[2];
    uint64_t    *h_bm[2];
    uint64_t    *d_rbm[2];     /* MR verdict bitmaps (fast path) */
    uint32_t    *d_offs[2];    /* compacted candidate offsets (fast path) */
    uint32_t    *d_cnt[2];     /* candidate counter per buffer */
    uint32_t    *h_cnt[2];     /* pinned candidate counter per buffer */
    uint64_t     pend_len[2];
    u128         pend_v0[2];
    int          gs_ready;
    int          fast;         /* this job uses the bitmap-verdict MR path */
    /* running prime walk state */
    u128 prev_prime;
    uint64_t prev_A;           /* slot index of the previous prime (v_anchor + 2*A) */
    int  prev_A_valid;
    int  have_prev;
    mpz_t scratch;
    /* results */
    uint64_t ints, survivors, tests, primes_found;
    uint64_t batches, gaps_reported, gaps_bad, records_new;
    uint64_t band_cnt[N_BANDS];
    double t_mark, t_walk, t_wait, t_gpu, t_proc, t_boot;
    /* top-4 gaps */
    int      ntop;
    u128     top_lower[4], top_upper[4];
    uint64_t top_gap[4];
    double   top_merit[4];
};

static void top_insert(job_t *j, u128 lower, u128 upper, uint64_t gap, double merit) {
    int pos = -1;
    for (int t = 0; t < 4; t++) {
        if (merit > j->top_merit[t]) { pos = t; break; }
    }
    if (pos < 0 && j->ntop < 4) pos = j->ntop;
    if (pos < 0) return;
    for (int t = 3; t > pos; t--) {
        j->top_merit[t] = j->top_merit[t - 1];
        j->top_lower[t] = j->top_lower[t - 1];
        j->top_upper[t] = j->top_upper[t - 1];
        j->top_gap[t]   = j->top_gap[t - 1];
    }
    j->top_merit[pos] = merit;
    j->top_lower[pos] = lower;
    j->top_upper[pos] = upper;
    j->top_gap[pos]   = gap;
    if (j->ntop < 4) j->ntop++;
}

/* ---------------- gap verification / reporting ---------------- */

/* exact-or-nothing check: lower must be a (probable) prime whose NEXT prime is
   exactly upper (this also proves there is no prime strictly in between) */
static int verify_gap(u128 lower, u128 upper) {
    mpz_t a, b;
    mpz_init(a);
    mpz_init(b);
    u128_to_mpz(a, lower);
    int ok = 0;
    if (mpz_probab_prime_p(a, 25) >= 1) {
        mpz_nextprime(b, a);
        ok = (mpz_to_u128(b) == upper);
    }
    mpz_clear(a);
    mpz_clear(b);
    return ok;
}

/* ---------------- records table (hunt criterion) ---------------- */

/* Returns -1 = check off, 0 = gap absent from the table, 1 = not a record,
   2 = NEW record.  Criterion (identical to scripts/watch_gap_hunt_records.py
   and new_src/record_log.c): gap present in the table AND merit > best known
   merit for that gap length; *best_out receives the table merit. */
static int record_check(uint64_t gap, double merit, double *best_out) {
    *best_out = 0.0;
    if (!g_table) return -1;
    if (gap >= g_table_len || g_table[gap] <= 0.0) return 0;
    *best_out = g_table[gap];
    return (merit > g_table[gap]) ? 2 : 1;
}

static int load_records(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    uint64_t maxg = 0, cnt = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        unsigned long long g = 0;
        double m = 0.0;
        if (sscanf(line, "%llu %lf", &g, &m) == 2 && g > 0 &&
            g <= (1ULL << 24) && m > 0.0) {
            cnt++;
            if (g > maxg) maxg = g;
        }
    }
    if (cnt == 0 || maxg == 0) { fclose(f); return 0; }
    g_table = (double *)calloc((size_t)maxg + 1, sizeof(double));
    if (!g_table) { fclose(f); return 0; }
    rewind(f);
    uint64_t filled = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        unsigned long long g = 0;
        double m = 0.0;
        if (sscanf(line, "%llu %lf", &g, &m) == 2 && g > 0 && g <= maxg && m > 0.0) {
            if (g_table[g] <= 0.0) filled++;
            g_table[g] = m;
        }
    }
    fclose(f);
    g_table_len = maxg + 1;
    struct stat st;
    if (stat(path, &st) == 0) {
        struct tm tmv;
        time_t mt = st.st_mtime;
        localtime_r(&mt, &tmv);
        snprintf(g_table_stamp, sizeof g_table_stamp,
                 "%s (%llu lengths, %llu B, mtime %04d-%02d-%02d)",
                 path, (unsigned long long)filled, (unsigned long long)st.st_size,
                 tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday);
    } else {
        snprintf(g_table_stamp, sizeof g_table_stamp, "%s (%llu lengths)",
                 path, (unsigned long long)filled);
    }
    return 1;
}

static int report_gap(u128 lower, u128 upper, uint64_t gap, double merit,
                      int ok, int rec, double best) {
    char s_low[48], s_up[48];
    u128_str(lower, s_low, sizeof s_low);
    u128_str(upper, s_up, sizeof s_up);
    const char *recstr = (rec == 2) ? "NEW" : (rec == 1) ? "no" :
                         (rec == 0) ? "absent" : "off";
    char beststr[32];
    if (rec >= 1) snprintf(beststr, sizeof beststr, "%.4f", best);
    else snprintf(beststr, sizeof beststr, "unknown");
    if (g_log) {
        fprintf(g_log, "%lld %s %llu %.4f %s verified=%d record=%s table=%s\n",
                (long long)time(NULL), s_low, (unsigned long long)gap, merit, s_up,
                ok, recstr, beststr);
        fflush(g_log);
    }
    printf("[phase0-gpu] GAP merit=%.4f gap=%llu lower=%s upper=%s verified=%d"
           " record=%s table=%s\n",
           merit, (unsigned long long)gap, s_low, s_up, ok, recstr, beststr);
    fflush(stdout);
    int isrec = 0;
    if (rec == 2) {
        printf("[phase0-gpu] *** RECORD: gap=%llu merit=%.4f best_known=%.4f "
               "(table: %s) ***\n",
               (unsigned long long)gap, merit, best, g_table_stamp);
        fflush(stdout);
        isrec = 1;
    }
    return isrec;
}

/* ------------- shared per-gap reporting (both MR pipeline paths) -------------
   Called only with gap >= g_gap_gate; xp/x are the two prime values (96-bit
   candidates), gap = x - xp in ints.  Keep the semantics identical in both
   callers: bands, top-4, record check, and verify/report beyond merit_min. */
static void process_prime_gap(job_t *j, u128 xp, u128 x, uint64_t gap) {
    double merit = (double)gap / u128_ln(xp);
    if (merit >= k_bands[0]) {
        int nb = (int)((merit - k_bands[0]) / 5.0) + 1;
        if (nb > N_BANDS) nb = N_BANDS;
        for (int b = 0; b < nb; b++) j->band_cnt[b]++;
    }
    if (j->ntop < 4 || merit > j->top_merit[3])
        top_insert(j, xp, x, gap, merit);
    double best = 0.0;
    int rec = record_check(gap, merit, &best);
    if (merit >= g_merit_min || rec == 2) {
        int ok = verify_gap(xp, x);
        int isrec = report_gap(xp, x, gap, merit, ok, rec, best);
        j->gaps_reported++;
        if (!ok) j->gaps_bad++;
        if (isrec) j->records_new++;
    }
}

/* slot-index variant: both endpoints on the thread's v_anchor grid */
static void process_gap_A(job_t *j, uint64_t Aprev, uint64_t A) {
    uint64_t gap = 2 * (A - Aprev);
    if (gap < g_gap_gate) return;
    u128 x  = j->v_anchor + (u128)2 * (u128)A;
    u128 xp = j->v_anchor + (u128)2 * (u128)Aprev;
    process_prime_gap(j, xp, x, gap);
}

/* ---------------- GPU batch: test + gap detection ---------------- */

static void flush_batch(job_t *j) {
    if (j->n == 0) return;
    uint32_t cnt = j->n;

    double f0 = now_s();
    pthread_mutex_lock(&g_gpu_mtx);
    double f1 = now_s();
    cudaError_t e = cudaMemcpy(g_d_base3, j->base3, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { pthread_mutex_unlock(&g_gpu_mtx); die_cuda(e, "H2D base"); }
    e = cudaMemcpy(g_d_steps, j->steps, (size_t)cnt * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { pthread_mutex_unlock(&g_gpu_mtx); die_cuda(e, "H2D steps"); }
#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_mr68_packed(g_d_base3, g_d_steps, g_d_res, cnt,
                          (cnt + 127u) / 128u, 128u, NULL) != 0)
        die("p0gpu_mr68_packed failed");
#else
    mr68_kernel_packed<<<(cnt + 127) / 128, 128>>>(g_d_base3, g_d_steps, g_d_res, cnt);
#endif
    e = cudaGetLastError();
    if (e != cudaSuccess) { pthread_mutex_unlock(&g_gpu_mtx); die_cuda(e, "kernel launch"); }
    e = cudaMemcpy(j->res, g_d_res, cnt, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { pthread_mutex_unlock(&g_gpu_mtx); die_cuda(e, "D2H results"); }
    pthread_mutex_unlock(&g_gpu_mtx);
    double f2 = now_s();
    j->t_wait += f1 - f0;
    j->t_gpu += f2 - f1;

    j->batches++;

    for (uint32_t i = 0; i < cnt; i++) {
        if (!j->res[i]) continue;
        /* 64-bit slot chain: candidate = v_anchor + 2*A; the previous prime
           carries its own slot index, so the gap is 2*(A - prev_A) and the
           u128 value is only built on the rare reporting path. */
        uint64_t A = j->bi + j->steps[i];
        u128 x = 0;                        /* lazily built (0 = not built yet) */

        j->primes_found++;
        if (j->collect) {
            if (j->collect_n >= j->collect_cap)
                die("--check collect overflow (raise cap)");
            x = j->v_anchor + (u128)2 * (u128)A;
            j->collect[j->collect_n++] = x;
        }
        if (j->have_prev) {
            uint64_t gap;
            if (j->prev_A_valid) {
                gap = 2 * (A - j->prev_A);
            } else {
                x = j->v_anchor + (u128)2 * (u128)A;
                gap = (uint64_t)(x - j->prev_prime);
            }
            /* Only gaps above the integer gate need the per-prime log+divide:
               g_gap_gate = min(6, merit_min) * ln(start) is a strict lower
               bound for "merit >= min(6, merit_min)" across the whole run, so
               every band-10 gap and every gap at/above --merit-min still
               passes.  Below the gate no band, top-list entry or record is
               possible (the record table has no sub-6 merits in this range). */
            if (gap >= g_gap_gate) {
                if (j->prev_A_valid) {
                    process_gap_A(j, j->prev_A, A);
                } else {
                    if (!x) x = j->v_anchor + (u128)2 * (u128)A;
                    process_prime_gap(j, j->prev_prime, x, gap);
                }
            }
        }
        j->prev_A = A;
        j->prev_A_valid = 1;
        j->have_prev = 1;
    }

    j->tests += cnt;
    j->n = 0;
    {   /* resume point = next candidate after the last one processed */
        u128 lastn = j->batch_base + (u128)2 * (u128)j->steps[cnt - 1];
        u128 next = lastn + 2;
        __sync_synchronize();
        g_resume_off[j->tid] = (uint64_t)(next - g_start);
    }
    j->t_proc += now_s() - f2;
}

/* ---------------- GPU sieve integration (--gpu-sieve) ---------------- */

static void job_gs_alloc(job_t *j) {
    cudaError_t e;
    e = cudaStreamCreate(&j->sstream);
    if (e != cudaSuccess) die_cuda(e, "sieve stream create");
    e = cudaStreamCreate(&j->cstream);
    if (e != cudaSuccess) die_cuda(e, "copy stream create");
    e = cudaEventCreateWithFlags(&j->sev[0], cudaEventDisableTiming);
    if (e != cudaSuccess) die_cuda(e, "sieve event create");
    e = cudaEventCreateWithFlags(&j->sev[1], cudaEventDisableTiming);
    if (e != cudaSuccess) die_cuda(e, "sieve event create");
    for (int q = 0; q < 4; q++) {
        /* timing enabled: cudaEventElapsedTime refuses cudaEventDisableTiming */
        e = cudaEventCreateWithFlags(&j->mev[0][q], cudaEventDefault);
        if (e != cudaSuccess) die_cuda(e, "split event create");
        e = cudaEventCreateWithFlags(&j->mev[1][q], cudaEventDefault);
        if (e != cudaSuccess) die_cuda(e, "split event create");
    }
    for (int k = 0; k < 2; k++) {
        e = cudaMalloc(&j->d_bm[k], g_bm_bytes);
        if (e != cudaSuccess) die_cuda(e, "sieve bitmap malloc");
        /* pinned host buffer: pageable async D2H is NOT asynchronous from the
           device's point of view and serialises thousands of small copies */
        e = cudaHostAlloc((void **)&j->h_bm[k], g_bm_bytes, cudaHostAllocDefault);
        if (e != cudaSuccess) die_cuda(e, "sieve bitmap pinned host alloc");
        e = cudaMalloc(&j->d_rbm[k], g_bm_bytes);
        if (e != cudaSuccess) die_cuda(e, "verdict bitmap malloc");
        e = cudaMalloc(&j->d_offs[k], (g_bm_bytes * 8 + 64) * sizeof(uint32_t));
        if (e != cudaSuccess) die_cuda(e, "offset list malloc");
        e = cudaMalloc(&j->d_cnt[k], sizeof(uint32_t));
        if (e != cudaSuccess) die_cuda(e, "candidate counter malloc");
        e = cudaHostAlloc((void **)&j->h_cnt[k], sizeof(uint32_t), cudaHostAllocDefault);
        if (e != cudaSuccess) die_cuda(e, "candidate counter pinned alloc");
    }
    j->gs_ready = 1;
}

static void job_gs_free(job_t *j) {
    if (!j->gs_ready) return;
    cudaStreamSynchronize(j->sstream);
    cudaStreamSynchronize(j->cstream);
    for (int k = 0; k < 2; k++) {
        cudaFree(j->d_bm[k]);
        cudaFreeHost(j->h_bm[k]);
        cudaFree(j->d_rbm[k]);
        cudaFree(j->d_offs[k]);
        cudaFree(j->d_cnt[k]);
        cudaFreeHost(j->h_cnt[k]);
    }
    cudaEventDestroy(j->sev[0]);
    cudaEventDestroy(j->sev[1]);
    for (int q = 0; q < 4; q++) {
        cudaEventDestroy(j->mev[0][q]);
        cudaEventDestroy(j->mev[1][q]);
    }
    cudaStreamDestroy(j->sstream);
    cudaStreamDestroy(j->cstream);
    j->gs_ready = 0;
}

/* build the (prime, P0_CHUNK-hit) item table for the fixed segment size and
   upload it once; device-wide, shared by all threads */
static void gpu_sieve_init(int seg_bits) {
    uint64_t half_max = ((uint64_t)1 << seg_bits) >> 1;
    uint32_t cap = 0;
    for (size_t i = g_bucket0; i < g_nprimes; i++) {
        uint64_t hits = half_max / g_primes[i] + 2;
        cap += (uint32_t)((hits + P0_CHUNK - 1) / P0_CHUNK);
    }
    P0Item  *items = (P0Item *)malloc((size_t)cap * sizeof(P0Item));
    uint64_t *h_p  = (uint64_t *)malloc(g_nprimes * sizeof(uint64_t));
    uint64_t *h_r  = (uint64_t *)malloc(g_nprimes * sizeof(uint64_t));
    uint64_t *h_iv = (uint64_t *)malloc(g_nprimes * sizeof(uint64_t));
    if (!items || !h_p || !h_r || !h_iv) die("gpu-sieve item table alloc failed");
    uint32_t ni = 0;
    for (size_t i = g_bucket0; i < g_nprimes; i++) {
        uint64_t p = g_primes[i];
        uint64_t hits = half_max / p + 2;
        uint32_t chunks = (uint32_t)((hits + P0_CHUNK - 1) / P0_CHUNK);
        for (uint32_t c = 0; c < chunks; c++) {
            items[ni].pidx = (uint32_t)i;
            items[ni].k0 = c * P0_CHUNK;
            ni++;
        }
        h_p[i] = p;
        h_r[i] = (uint64_t)(((__uint128_t)1 << 64) % p);
        h_iv[i] = ~0ULL / p;        /* floor((2^64-1)/p) for fastmod */
    }
    g_nitems = ni;
    g_bm_bytes = ((half_max + 63) >> 6) * 8;
    cudaError_t e;
    e = cudaMalloc(&g_d_items, (size_t)ni * sizeof(P0Item));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc items");
    e = cudaMalloc(&g_d_primes64, g_nprimes * sizeof(uint64_t));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc primes64");
    e = cudaMalloc(&g_d_r64, g_nprimes * sizeof(uint64_t));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc r64");
    e = cudaMalloc(&g_d_invp, g_nprimes * sizeof(uint64_t));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc invp");
    e = cudaMalloc(&g_d_wpat, sizeof g_wpat2);
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc wheel pattern");
    e = cudaMemcpy(g_d_items, items, (size_t)ni * sizeof(P0Item), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) die_cuda(e, "H2D items");
    e = cudaMemcpy(g_d_primes64, h_p, g_nprimes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) die_cuda(e, "H2D primes64");
    e = cudaMemcpy(g_d_r64, h_r, g_nprimes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) die_cuda(e, "H2D r64");
    e = cudaMemcpy(g_d_invp, h_iv, g_nprimes * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) die_cuda(e, "H2D invp");
    e = cudaMemcpy(g_d_wpat, g_wpat2, sizeof g_wpat2, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) die_cuda(e, "H2D wheel pattern");
    g_wheel_r64p = (uint64_t)(((__uint128_t)1 << 64) % WHEEL_P);
    free(items); free(h_p); free(h_r); free(h_iv);
    printf("[phase0-gpu] gpu-sieve: GPU marking, %u items/segment (wheel 3..13 tiled "
           "on device), %.1f KB bitmap/segment\n",
           g_nitems, g_bm_bytes / 1024.0);
}

/* segment s -> memset + mark kernel + D2H on the thread stream, slot k */
static void submit_segment(job_t *j, uint64_t s, int k) {
    const uint64_t seg = 1ULL << j->seg_bits;
    const uint64_t span = (uint64_t)(j->end - j->start);
    const uint64_t seg_lo_off = s * seg;
    const uint64_t seg_len = (seg_lo_off + seg <= span) ? seg : (span - seg_lo_off);
    const u128 base = j->start + seg_lo_off;
    const uint64_t kpar = ((uint64_t)base & 1ULL) ? 0 : 1;
    const u128 v0 = base + kpar;
    const uint64_t half = (seg_len + 1) >> 1;
    j->pend_len[k] = seg_len;
    j->pend_v0[k] = v0;
    /* wheel fill replaces the memset: it writes every word the walk reads */
    if (j->fast) {
        cudaError_t er = cudaEventRecord(j->mev[k][0], j->sstream);
        if (er != cudaSuccess) die_cuda(er, "mev0 record");
    }
    const uint32_t nwords = (uint32_t)((half + 63) >> 6);
    const uint32_t wgrid = (nwords + g_mark_tpb - 1) / g_mark_tpb;
#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_p0_wheel(g_d_wpat, WHEEL_P, WHEEL_INV2, g_wheel_r64p,
                       (uint64_t)v0, (uint64_t)(v0 >> 64), nwords, j->d_bm[k],
                       wgrid, g_mark_tpb, j->sstream) != 0)
        die("p0gpu_p0_wheel failed");
#else
    p0_wheel_kernel<<<wgrid, g_mark_tpb, 0, j->sstream>>>(
        g_d_wpat, WHEEL_P, WHEEL_INV2, g_wheel_r64p,
        (uint64_t)v0, (uint64_t)(v0 >> 64), nwords, j->d_bm[k]);
#endif
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "[phase0-gpu] wheel kernel fail at tid=%d seg=%llu k=%d "
                        "wgrid=%u tpb=%u\n",
                j->tid, (unsigned long long)s, k, wgrid, g_mark_tpb);
        die_cuda(e, "wheel kernel");
    }
    const uint32_t grid = (g_nitems + g_mark_tpb - 1) / g_mark_tpb;
#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_p0_mark(g_d_items, g_nitems, g_d_primes64, g_d_r64, g_d_invp, half,
                      (uint64_t)v0, (uint64_t)(v0 >> 64),
                      (uint32_t)(v0 % 30u), j->d_bm[k],
                      grid, g_mark_tpb, j->sstream) != 0)
        die("p0gpu_p0_mark failed");
#else
    p0_mark_kernel<<<grid, g_mark_tpb, 0, j->sstream>>>(
        g_d_items, g_nitems, g_d_primes64, g_d_r64, g_d_invp, half,
        (uint64_t)v0, (uint64_t)(v0 >> 64),
        p0_vis_mask30((uint32_t)(v0 % 30u)), j->d_bm[k]);
#endif
    e = cudaGetLastError();
    if (e != cudaSuccess) die_cuda(e, "sieve mark kernel");
    if (j->fast) {
        cudaEventRecord(j->mev[k][1], j->sstream);
        /* bitmap-verdict pipeline: one extra kernel tests every candidate
           straight from the survivor bitmap, so the CPU only ever sees the
           verdict bitmap - no enumeration, no steps H2D, no per-candidate
           host round trip. */
        const uint64_t valid_slots =
            half - ((kpar && (seg_len & 1ULL)) ? 1ULL : 0ULL);
        const uint32_t nwords = (uint32_t)((valid_slots + 63) >> 6);
        uint64_t last_mask = 0;
        if (nwords) {
            uint64_t lb = valid_slots - (uint64_t)(nwords - 1) * 64;
            last_mask = (lb >= 64) ? ~0ULL : ((1ULL << lb) - 1ULL);
        }
        /* the buffers of slot k are reused two segments later: only then must
           the kernel stream wait for the copies of the previous verdict */
        e = cudaStreamWaitEvent(j->sstream, j->mev[k][3]);
        if (e != cudaSuccess) die_cuda(e, "copy-stream reuse wait");
        e = cudaMemsetAsync(j->d_cnt[k], 0, sizeof(uint32_t), j->sstream);
        if (e != cudaSuccess) die_cuda(e, "candidate counter reset");
        e = cudaMemsetAsync(j->d_rbm[k], 0, g_bm_bytes, j->sstream);
        if (e != cudaSuccess) die_cuda(e, "verdict bitmap reset");
        if (nwords) {
            /* 1) ordered-free compaction of the survivor set to offsets,
                  2) one thread per candidate for the Montgomery chains,
                  3) verdict bits written straight into the output bitmap */
            const uint32_t cgrid = (nwords + 127u) / 128u;
#ifdef PHASE0_KERNEL_DLL
            if (p0gpu_p0_compact(j->d_bm[k], nwords, last_mask, j->d_offs[k],
                                 j->d_cnt[k], cgrid, j->sstream) != 0)
                die("p0gpu_p0_compact failed");
#else
            p0_compact_kernel<<<cgrid, 128, 0, j->sstream>>>(
                j->d_bm[k], nwords, last_mask, j->d_offs[k], j->d_cnt[k]);
#endif
            e = cudaGetLastError();
            if (e != cudaSuccess) die_cuda(e, "compact kernel");
            /* grid sized for the typical survivor density; a grid-stride loop
               inside the kernel keeps it correct for any count */
            const uint32_t mgrid = ((uint32_t)(nwords * 6u) + 127u) / 128u;
#ifdef PHASE0_KERNEL_DLL
            if (p0gpu_mr68_from_offsets((uint64_t)v0, (uint64_t)(v0 >> 64),
                                        j->d_offs[k], j->d_cnt[k], j->d_rbm[k],
                                        mgrid, j->sstream) != 0)
                die("p0gpu_mr68_from_offsets failed");
#else
            mr68_from_offsets<<<mgrid, 128, 0, j->sstream>>>(
                (uint64_t)v0, (uint64_t)(v0 >> 64),
                j->d_offs[k], j->d_cnt[k], j->d_rbm[k]);
#endif
            e = cudaGetLastError();
            if (e != cudaSuccess) die_cuda(e, "mr from offsets kernel");
        }
        cudaEventRecord(j->mev[k][2], j->sstream);
        /* D2H on the copy stream: the next segment's kernels queue behind the
           SM work, never behind the copy engine */
        e = cudaStreamWaitEvent(j->cstream, j->mev[k][2]);
        if (e != cudaSuccess) die_cuda(e, "copy-stream wait");
        e = cudaMemcpyAsync(j->h_bm[k], j->d_rbm[k], g_bm_bytes,
                            cudaMemcpyDeviceToHost, j->cstream);
        if (e != cudaSuccess) die_cuda(e, "verdict bitmap D2H");
        e = cudaMemcpyAsync(j->h_cnt[k], j->d_cnt[k], sizeof(uint32_t),
                            cudaMemcpyDeviceToHost, j->cstream);
        if (e != cudaSuccess) die_cuda(e, "candidate counter D2H");
        cudaEventRecord(j->mev[k][3], j->cstream);
    } else {
        e = cudaMemcpyAsync(j->h_bm[k], j->d_bm[k], g_bm_bytes,
                            cudaMemcpyDeviceToHost, j->sstream);
        if (e != cudaSuccess) die_cuda(e, "sieve bitmap D2H");
    }
    if (j->fast) {
        e = cudaEventRecord(j->sev[k], j->cstream);
        if (e != cudaSuccess) die_cuda(e, "sieve event record (copy stream)");
    } else {
        e = cudaEventRecord(j->sev[k], j->sstream);
        if (e != cudaSuccess) die_cuda(e, "sieve event record");
    }
}

/* survivor walk over one segment bitmap: word scan + ctz.  Candidates are
   v0 + 2*idx; kpar only selects the parity class for the tail guard. */
static void walk_segment(job_t *j, const uint64_t *bm, uint64_t seg_len,
                         u128 v0, uint64_t kpar) {
    double wk0 = now_s();
    const uint64_t half = (seg_len + 1) >> 1;
    const uint64_t wn = (half + 63) >> 6;
    const uint64_t nvalid = half - (wn - 1) * 64;
    /* 64-bit slot space: candidate value = v_anchor + 2*A.  A batch step is
       A - bi (<= u32), so the hot path never touches u128. */
    const uint64_t seg_A0 = (uint64_t)((v0 - j->v_anchor) >> 1);
    for (uint64_t wi = 0; wi < wn; wi++) {
        uint64_t w = ~bm[wi];
        if (wi == wn - 1) {
            uint64_t m = (nvalid == 64) ? ~0ULL : ((1ULL << nvalid) - 1ULL);
            w &= m;
        }
        while (w) {
            int b = __builtin_ctzll(w);
            w &= w - 1;
            uint64_t k = 2 * (wi * 64 + (uint64_t)b) + kpar;
            if (k >= seg_len) continue;
            j->survivors++;
            if (!j->do_test) continue;
            uint64_t A = seg_A0 + wi * 64 + (uint64_t)b;
            if (j->n == 0) {
                j->bi = A;
                j->batch_base = j->v_anchor + (u128)2 * (u128)A;
                j->base3[0] = (uint32_t)j->batch_base;
                j->base3[1] = (uint32_t)(j->batch_base >> 32);
                j->base3[2] = (uint32_t)(j->batch_base >> 64);
                j->steps[0] = 0;
                j->n = 1;
            } else {
                uint64_t d = A - j->bi;
                if (d > 0xFFFFFFFFu) {
                    flush_batch(j);
                    j->bi = A;
                    j->batch_base = j->v_anchor + (u128)2 * (u128)A;
                    j->base3[0] = (uint32_t)j->batch_base;
                    j->base3[1] = (uint32_t)(j->batch_base >> 32);
                    j->base3[2] = (uint32_t)(j->batch_base >> 64);
                    j->steps[0] = 0;
                    j->n = 1;
                } else {
                    j->steps[j->n++] = (uint32_t)d;
                }
            }
            if (j->n == j->cap) flush_batch(j);
        }
    }
    j->t_walk += now_s() - wk0;
}

/* fast-path walk: consume the MR verdict bitmap directly.  Set bits are
   primes; a gap between two consecutive primes is materialised only when its
   zero-run can carry gap >= g_gap_gate (2*run >= gate, i.e. run >= ceil).
   The gap from the previous segment (or the boot prime) is closed through
   the A chain when the first set bit of the segment is seen. */
static void walk_result_bitmap(job_t *j, const uint64_t *rb, uint64_t seg_len,
                               u128 v0, uint64_t kpar, const uint32_t *cnt) {
    double wk0 = now_s();
    const uint64_t half = (seg_len + 1) >> 1;
    const uint64_t valid_slots = half - ((kpar && (seg_len & 1ULL)) ? 1ULL : 0ULL);
    const uint64_t wn = (valid_slots + 63) >> 6;
    const uint64_t seg_A0 = (uint64_t)((v0 - j->v_anchor) >> 1);
    /* zeros between two primes = distance - 1, so a gap g = 2*(zeros+1)
       reaches the integer gate at zeros >= ((g_gap_gate+1)>>1) - 1 */
    const uint64_t run_gate = (((g_gap_gate + 1) >> 1) > 0)
                                  ? (((g_gap_gate + 1) >> 1) - 1) : 0;
    j->survivors += *cnt;
    j->tests += *cnt;
    uint64_t run = 0;                  /* zero slots since the last set bit */
    int first_done = 0;
    uint64_t nzw = 0, pf0 = j->primes_found;
    uint64_t fset = ~0ULL, lset = 0;
    for (uint64_t wi = 0; wi < wn; wi++) {
        uint64_t w = rb[wi];
        if (wi == wn - 1) {
            uint64_t lb = valid_slots - (wn - 1) * 64;
            if (lb < 64) w &= (1ULL << lb) - 1ULL;
        }
        if (!w) {
            if (first_done) run += 64;
            continue;
        }
        nzw++;
        j->primes_found += (uint64_t)__builtin_popcountll(w);
        int b = __builtin_ctzll(w);
        if (!first_done) {
            /* first prime of the segment: closes the gap from the previous
               segment (or the boot prime) through the A chain */
            uint64_t A = seg_A0 + wi * 64 + (uint64_t)b;
            if (j->prev_A_valid) {
                process_gap_A(j, j->prev_A, A);
            } else {
                u128 x = j->v_anchor + (u128)2 * (u128)A;
                uint64_t gap = (uint64_t)(x - j->prev_prime);
                if (gap >= g_gap_gate) process_prime_gap(j, j->prev_prime, x, gap);
            }
            j->prev_A = A;
            j->prev_A_valid = 1;
            j->have_prev = 1;
            first_done = 1;
            run = 0;                    /* the cross gap is already consumed */
        } else {
            /* zeros since the last set bit = carried run + in-word offset */
            run += (uint64_t)b;
            uint64_t A = seg_A0 + wi * 64 + (uint64_t)b;
            if (run >= run_gate) process_gap_A(j, A - run - 1, A);
            run = 0;
        }
        j->prev_A = seg_A0 + wi * 64 + (uint64_t)b;   /* chain for the next segment */
        w &= w - 1;                     /* consume the first set bit */
        if (fset == ~0ULL) fset = wi * 64 + (uint64_t)b;
        int lastb = b;
        while (w) {
            int b2 = __builtin_ctzll(w);
            w &= w - 1;
            uint64_t zeros = (uint64_t)(b2 - lastb - 1);
            uint64_t A = seg_A0 + wi * 64 + (uint64_t)b2;
            if (zeros >= run_gate) process_gap_A(j, A - zeros - 1, A);
            j->prev_A = A;                    /* chain for the next segment */
            lastb = b2;
        }
        lset = wi * 64 + (uint64_t)lastb;
        run = (uint64_t)(63 - lastb);        /* trailing zeros carry over */
    }
    if (g_fast_dbg > 0 && j->tid == 0 && g_fast_dbg-- > 0)
        fprintf(stderr, "[fast-dbg] seg_A0=%llu wn=%llu nzw=%llu primes=%llu "
                        "first=%llu last=%llu valid=%llu\n",
                (unsigned long long)seg_A0, (unsigned long long)wn,
                (unsigned long long)nzw,
                (unsigned long long)(j->primes_found - pf0),
                (unsigned long long)fset, (unsigned long long)lset,
                (unsigned long long)valid_slots);
    j->t_walk += now_s() - wk0;
}

/* ---------------- worker: sieve slice -> batches -> GPU ---------------- */

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    double w0 = now_s();

    const uint64_t seg = 1ULL << j->seg_bits;
    const u128 lo = j->start, hi = j->end;
    const uint64_t span = (uint64_t)(hi - lo);
    if (span == 0) {
        j->completed = 1;
        __sync_synchronize();
        g_resume_off[j->tid] = (uint64_t)(hi - g_start);
        return NULL;
    }
    const uint64_t nseg = (span + seg - 1) / seg;
    if (nseg > (1ULL << 26)) {
        fprintf(stderr, "[phase0-gpu] range/segment-ratio too large (%llu segments)"
                        " - raise --seg-bits\n", (unsigned long long)nseg);
        exit(1);
    }

    int32_t  *heads   = NULL;
    int32_t  *nexti   = NULL;
    uint64_t *nextoff = NULL;
    uint64_t *bitw    = NULL;
    const uint64_t seg_halfmax = (seg + 1) >> 1;
    const uint64_t seg_words   = (seg_halfmax + 63) >> 6;
    if (!g_use_gpu_sieve) {
        heads   = (int32_t *)malloc((size_t)nseg * sizeof(int32_t));
        nexti   = (int32_t *)malloc(g_nprimes * sizeof(int32_t));
        nextoff = (uint64_t *)malloc(g_nprimes * sizeof(uint64_t));
        /* odd-value bitmap: ONE BIT per odd value (the survivor walk scans it
           64 candidates per u64 word with ctz) */
        bitw = (uint64_t *)malloc((size_t)seg_words * 8);
        if (!heads || !nexti || !nextoff || !bitw) die("worker alloc failed");
        for (uint64_t s = 0; s < nseg; s++) heads[s] = -1;

        /* Bucket init: for every prime above the wheel, the offset (relative
           to lo) of its first odd multiple >= lo; primes whose next multiple
           is beyond the slice are parked with nexti = -1. */
        double ts0 = now_s();
        for (size_t i = g_bucket0; i < g_nprimes; i++) {
            uint64_t p = g_primes[i];
            uint64_t r = (uint64_t)(lo % p);
            uint64_t s = (p - r) % p;
            u128 f = lo + s;
            if (((uint64_t)f & 1ULL) == 0) f += p;
            if (f < hi) {
                uint64_t off = (uint64_t)(f - lo);
                uint64_t sg = off >> j->seg_bits;
                nextoff[i] = off;
                nexti[i] = heads[sg];
                heads[sg] = (int32_t)i;
            } else {
                nexti[i] = -1;
            }
        }
        j->t_mark += now_s() - ts0;
    } else if (!j->gs_ready) {
        die("--gpu-sieve: per-thread GPU resources missing");
    }

    /* previous prime before lo (for the gap entering the slice) */
    if (j->do_test) {
        double b0 = now_s();
        u128 pv = lo - 1;
        if (((uint64_t)pv & 1ULL) == 0) pv -= 1;
        while (pv > 2 && !u128_is_probable(pv, j->scratch)) pv -= 2;
        j->prev_prime = pv;
        j->have_prev = 1;
        j->prev_A_valid = 0;   /* first gap uses the u128 path once */
        j->t_boot += now_s() - b0;
    }

    int completed = 1;
    const uint64_t kpar = ((uint64_t)lo & 1ULL) ? 0 : 1;
    j->v_anchor = lo + kpar;
    const int fast = (g_fast_mr && g_use_gpu_sieve && j->do_test && !j->collect);
    j->fast = fast;
    if (g_use_gpu_sieve) {
        /* double-buffered GPU sieve: queue marking for segment s+1 while the
           survivor walk of segment s runs on the CPU (bitmap D2H overlaps) */
        double gs0 = now_s();
        int k = 0;
        submit_segment(j, 0, 0);
        for (uint64_t s = 0; s < nseg; s++) {
            if (g_stop) { completed = 0; break; }
            if (s + 1 < nseg) submit_segment(j, s + 1, k ^ 1);
            cudaError_t e = cudaEventSynchronize(j->sev[k]);
            if (e != cudaSuccess) die_cuda(e, "gpu-sieve event sync");
            j->ints += j->pend_len[k];
            if (fast) {
                walk_result_bitmap(j, j->h_bm[k], j->pend_len[k], j->pend_v0[k],
                                   kpar, j->h_cnt[k]);
                {
                    float ms;
                    cudaError_t ee = cudaEventElapsedTime(&ms, j->mev[k][0], j->mev[k][1]);
                    if (ee == cudaSuccess)
                        j->t_mm += ms / 1000.0;
                    ee = cudaEventElapsedTime(&ms, j->mev[k][1], j->mev[k][2]);
                    if (ee == cudaSuccess)
                        j->t_mr += ms / 1000.0;
                    ee = cudaEventElapsedTime(&ms, j->mev[k][2], j->mev[k][3]);
                    if (ee == cudaSuccess)
                        j->t_cp += ms / 1000.0;
                }
                /* segment-level resume point: everything up to the segment
                   end is fully processed (boot rebuilds the prime chain) */
                __sync_synchronize();
                g_resume_off[j->tid] = (uint64_t)((j->start +
                        (u128)(s * seg + j->pend_len[k])) - g_start);
            } else {
                walk_segment(j, j->h_bm[k], j->pend_len[k], j->pend_v0[k], kpar);
            }
            k ^= 1;
        }
        if (!completed) {
            cudaError_t e = cudaStreamSynchronize(j->sstream);
            if (e != cudaSuccess) die_cuda(e, "gpu-sieve drain");
        }
        j->t_mark += now_s() - gs0;
    } else {
        for (uint64_t s = 0; s < nseg; s++) {
            if (g_stop) { completed = 0; break; }
            const uint64_t seg_lo_off = s * seg;
            const uint64_t seg_len = (seg_lo_off + seg <= span) ? seg : (span - seg_lo_off);
            const uint64_t seg_end = seg_lo_off + seg_len;
            if (seg_len == 0) continue;
            j->ints += seg_len;
            const u128 base = lo + seg_lo_off;
            /* wheel fill (see WHEEL_P comment): replaces memset + marking of
               3,5,7,11,13; bit i = 1 means value v0+2i is divisible by a
               wheel prime.  The walk masks off the tail bits beyond half. */
            {
                const u128 wv0 = base + kpar;
                uint32_t wob = (uint32_t)(((uint64_t)(wv0 % WHEEL_P) * WHEEL_INV2) % WHEEL_P);
                uint64_t nw = ((((seg_len + 1) >> 1) + 63) >> 6);
                for (uint64_t wi = 0; wi < nw; wi++) {
                    uint32_t b = wob & 63u, src = wob >> 6;
                    uint64_t w = g_wpat2[src] >> b;
                    if (b) w |= g_wpat2[src + 1] << (64 - b);
                    bitw[wi] = w;
                    wob += 64u;
                    if (wob >= WHEEL_P) wob -= WHEEL_P;
                }
            }

            /* --- marking --- */
            double a0 = now_s();
            for (int32_t i = heads[s]; i != -1; ) {
                int32_t ni = nexti[i];
                uint64_t p = g_primes[i];
                uint64_t off = nextoff[i];
                while (off < seg_end) {
                    uint64_t bidx = (off - seg_lo_off) >> 1;
                    bitw[bidx >> 6] |= (1ULL << (bidx & 63));
                    off += 2 * p;
                }
                nextoff[i] = off;
                if (off < span) {
                    uint64_t sg = off >> j->seg_bits;
                    if (sg < nseg) { nexti[i] = heads[sg]; heads[sg] = i; }
                    else nexti[i] = -1;
                } else {
                    nexti[i] = -1;
                }
                i = ni;
            }
            heads[s] = -1;
            j->t_mark += now_s() - a0;

            walk_segment(j, bitw, seg_len, base + kpar, kpar);
        }
    }

    if (j->do_test) flush_batch(j);
    if (completed) {
        __sync_synchronize();
        g_resume_off[j->tid] = (uint64_t)(hi - g_start);
    }
    j->completed = completed;
    free(heads); free(nexti); free(nextoff); free(bitw);
    (void)w0;
    return NULL;
}

/* ---------------- checkpoint file ---------------- */

/* format:
     phase0gpu-state v1
     start <DEC>          (absolute range start)
     length <DEC>
     threads <T>
     off<i> <DEC>         (resume offset of thread i, relative to start)
*/
static void write_state(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    if (!f) {
        fprintf(stderr, "[phase0-gpu] WARNING: cannot write state file %s\n", tmp);
        return;
    }
    char s0[48];
    u128_str(g_start, s0, sizeof s0);
    fprintf(f, "phase0gpu-state v1\nstart %s\nlength %llu\nthreads %d\n",
            s0, (unsigned long long)g_len, g_nthreads);
    for (int t = 0; t < g_nthreads; t++)
        fprintf(f, "off%d %llu\n", t, (unsigned long long)g_resume_off[t]);
    fclose(f);
    if (rename(tmp, path) != 0)
        fprintf(stderr, "[phase0-gpu] WARNING: cannot rename state file into place\n");
}

/* returns 1 if a state file was loaded, 0 if absent */
static int load_state(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char tag[64], val[64];
    if (fscanf(f, "%63s %63s", tag, val) != 2 || strcmp(tag, "phase0gpu-state") != 0) {
        fprintf(stderr, "[phase0-gpu] FATAL: state file %s is not a phase0gpu-state file\n", path);
        exit(3);
    }
    uint64_t saved[MAX_THREADS];
    int have_off[MAX_THREADS];
    for (int t = 0; t < MAX_THREADS; t++) { saved[t] = 0; have_off[t] = 0; }
    while (fscanf(f, "%63s %63s", tag, val) == 2) {
        if (!strcmp(tag, "start")) {
            if (parse_u128(val) != g_start) {
                fprintf(stderr, "[phase0-gpu] FATAL: state file range start does not match --start\n");
                exit(3);
            }
        } else if (!strcmp(tag, "length")) {
            if (strtoull(val, NULL, 10) != g_len) {
                fprintf(stderr, "[phase0-gpu] FATAL: state file length does not match --length\n");
                exit(3);
            }
        } else if (!strcmp(tag, "threads")) {
            if (atoi(val) != g_nthreads) {
                fprintf(stderr, "[phase0-gpu] FATAL: state file threads=%s does not match --threads=%d"
                                " (slice geometry must match to resume)\n", val, g_nthreads);
                exit(3);
            }
        } else {
            int idx = -1;
            if (sscanf(tag, "off%d", &idx) == 1 && idx >= 0 && idx < MAX_THREADS) {
                saved[idx] = strtoull(val, NULL, 10);
                have_off[idx] = 1;
            }
        }
    }
    fclose(f);
    for (int t = 0; t < g_nthreads; t++) {
        if (!have_off[t]) {
            fprintf(stderr, "[phase0-gpu] FATAL: state file missing off%d\n", t);
            exit(3);
        }
        g_resume_off[t] = saved[t];
    }
    return 1;
}

/* ---------------- monitor thread: progress + checkpoints ---------------- */

static void fmt_eta(double secs, char *out, size_t n) {
    if (secs < 0 || secs > 3600.0 * 24 * 365) { snprintf(out, n, "?"); return; }
    long s = (long)secs;
    long h = s / 3600, m = (s % 3600) / 60;
    if (h > 0) snprintf(out, n, "%ldh%02ldm", h, m);
    else if (m > 0) snprintf(out, n, "%ldm%02lds", m, (long)(s % 60));
    else snprintf(out, n, "%lds", s);
}

static void *monitor_thread(void *arg) {
    (void)arg;
    double last_print = now_s();
    double last_state = last_print;
    while (!g_stop && !g_mon_stop) {
        struct timespec ts = {0, 200 * 1000 * 1000};
        nanosleep(&ts, NULL);
        double now = now_s();
        if (g_progress > 0 && now - last_print >= g_progress) {
            uint64_t done = g_pre_ints, surv = 0, tests = 0, primes = 0, batches = 0, gaps = 0;
            for (int t = 0; t < g_nthreads; t++) {
                done   += __atomic_load_n(&g_jobs[t].ints, __ATOMIC_RELAXED);
                surv   += __atomic_load_n(&g_jobs[t].survivors, __ATOMIC_RELAXED);
                tests  += __atomic_load_n(&g_jobs[t].tests, __ATOMIC_RELAXED);
                primes += __atomic_load_n(&g_jobs[t].primes_found, __ATOMIC_RELAXED);
                batches+= __atomic_load_n(&g_jobs[t].batches, __ATOMIC_RELAXED);
                gaps   += __atomic_load_n(&g_jobs[t].gaps_reported, __ATOMIC_RELAXED);
            }
            double rate = (now > g_t0) ? (double)done / (now - g_t0) : 0.0;
            char eta[32];
            double remain = (g_total_ints > done) ? (double)(g_total_ints - done) : 0.0;
            fmt_eta((rate > 0) ? remain / rate : -1.0, eta, sizeof eta);
            printf("[phase0-gpu] %6.2f%%  ints=%.3e  %.3e ints/s  u=%.2f%%  tests=%.3e  "
                   "primes=%.3e  gaps=%llu  batches=%llu  eta=%s\n",
                   g_total_ints ? 100.0 * (double)done / (double)g_total_ints : 0.0,
                   (double)done, rate,
                   done ? 100.0 * (double)surv / (double)done : 0.0,
                   (double)tests, (double)primes,
                   (unsigned long long)gaps,
                   (unsigned long long)batches, eta);
            fflush(stdout);
            last_print = now;
        }
        if (g_state_path && now - last_state >= g_state_every) {
            write_state(g_state_path);
            last_state = now;
        }
    }
    return NULL;
}


/* ============ --engine walk: class-30 sieve + batched jump-walk ============
   Fast threshold gap scanner (ported from tools/walk_engine.cu; exactness
   gates there: siege arbiter false=0 miss=0, 22/22 set-gate vs the odd-slot
   engine, NB=1 identical).  Reports every prime gap >= --gap-min (default
   ceil(merit-min * ln(start))) and feeds each through the same
   process_prime_gap path as the sieve engine (bands, top-4, record check,
   GMP verify, --log).  A gap is exact for gap lengths up to one block
   (1e9 values) - beyond every realistic gap at these sizes.

   Geometry: block = 30*2^25 numbers (2^28 slots, 32 MB bitmap), tile 2^19
   slots (64 KB shared), wheel {7,11,13} pattern OR, item table of 64-mark
   chunks.  Super-batches of --walk-batch blocks; two bitmap regions
   alternate; the walk of batch j is collected one iteration late (P0_PIPE,
   below) so the sieve of batch j+1 really overlaps it.  Measured BEFORE the
   P0_PIPE change (2026-10-05): mark 34% + walk 66% = 99.5% of wall, i.e. the
   two stages were strictly serial.  Each batch carries one lookahead block
   (re-sieved by the next batch) so a gap straddling a batch boundary is found
   exactly once (by the batch that owns its lower prime).  Constraint: the
   whole range must fit one 64-bit window of the block base A (checked;
   aborts otherwise).                                                       */
static int run_walk_engine(void) {
    const uint64_t blockNum30 = 30ull * (1ull << 25);
    const uint64_t cWords = (1ull << 28) / 64;
    const uint64_t tileSlots = 1ull << 19;
    const uint32_t tileWords = (uint32_t)(tileSlots >> 6);
    const uint64_t tilesPerBlockC = (1ull << 28) / tileSlots;
    uint32_t sg_grid = 92, sg_block = 512;
    {   /* P0_SG_GRID: mark-kernel CTA count.  92 (the pre-pipeline value) is a
           full-width wave; under P0_PIPE the marks only need to fill the walk's
           latency shadow, so a smaller grid can be cheaper (contention knob). */
        const char *ev = getenv("P0_SG_GRID");
        if (ev) { int v = atoi(ev); if (v >= 1 && v <= 4096) sg_grid = (uint32_t)v; }
    }
    const uint32_t wk_grid = 46, wk_block = 512;

    u128 A = ((g_start | 1) / 30u) * 30u;
    uint64_t Alo = (uint64_t)A, Ahi = (uint64_t)(A >> 64);
    u128 range_end = g_start + (u128)g_len;
    if (range_end <= A) { fprintf(stderr, "[phase0-gpu] empty range\n"); return 2; }
    uint64_t b_total = (uint64_t)(((range_end - A) + blockNum30 - 1) / blockNum30);
    if ((u128)Alo + (u128)(b_total + 2) * (u128)blockNum30 >= ((u128)1 << 64)) {
        fprintf(stderr, "[phase0-gpu] --engine walk: range crosses the 64-bit block window;"
                        " use --engine sieve\n");
        return 2;
    }
    if (g_start <= (u128)g_walk_primes) {
        fprintf(stderr, "[phase0-gpu] --engine walk: --start must exceed --walk-primes\n");
        return 2;
    }
    uint64_t gapmin = g_walk_gapmin;
    if (!gapmin) {
        double gmn = ceil(g_merit_min * u128_ln(g_start));
        if (gmn < 100.0) gmn = 100.0;
        gapmin = (uint64_t)gmn;
    }
    /* walk reports exactly the gaps >= gapmin through the shared log path:
       clear the merit gate unless --merit-min was given explicitly (then it
       acts as an additional bar). */
    if (!g_merit_explicit) g_merit_min = 0.0;
    else if (g_merit_min * u128_ln(g_start) < (double)gapmin)
        g_merit_min = (double)gapmin / u128_ln(g_start);
    if (gapmin < 100ull || gapmin > 10000000ull) {
        fprintf(stderr, "[phase0-gpu] --engine walk: --gap-min must be in [100, 1e7]\n");
        return 2;
    }
    if (gapmin < 300ull)
        fprintf(stderr, "[phase0-gpu] WARNING: --gap-min %llu is small; walk cost scales ~1/gap-min\n",
                (unsigned long long)gapmin);
    uint32_t K = g_walk_batch ? g_walk_batch : 16;
    if (K < 4) K = 4;
    if (K > 96) K = 96;   /* region = (K+1) x 33.6 MB, x2; K=96 = 6.5 GB */

    /* ---- primes 7..P, per-prime tables, item table, wtab ---- */
    uint32_t P = g_walk_primes < 30u ? 30u : g_walk_primes;
    uint8_t *sv = (uint8_t *)calloc((size_t)P + 1, 1);
    if (!sv) die("walk prime alloc failed");
    for (uint64_t i = 4; i <= P; i += 2) sv[i] = 1;
    for (uint64_t i = 3; i * i <= (uint64_t)P; i += 2)
        if (!sv[i]) for (uint64_t j = i * i; j <= P; j += i) sv[j] = 1;
    size_t np = 0;
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)P / 2 + 64));
    if (!hp) die("walk prime table alloc failed");
    for (uint64_t i = 7; i <= P; i += 2) if (!sv[i]) hp[np++] = i;
    free(sv);
    uint64_t *h_iv = (uint64_t *)malloc(np * 8), *h_r = (uint64_t *)malloc(np * 8);
    if (!h_iv || !h_r) die("walk prime aux alloc failed");
    for (size_t i = 0; i < np; i++) {
        h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]);
        h_iv[i] = ~0ULL / hp[i];
    }
    const uint32_t CLS8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    int8_t clsidx[30];
    uint32_t perm[30] = {0}, pinv[30] = {0};
    for (int i = 0; i < 30; i++) clsidx[i] = -1;
    for (int i = 0; i < 8; i++) clsidx[CLS8[i]] = (int8_t)i;
    for (int pm = 0; pm < 30; pm++) {
        if (clsidx[pm] < 0) continue;
        uint32_t m = 0;
        for (int j = 0; j < 8; j++) m |= 1u << ((pm * (int)CLS8[j]) % 30);
        perm[pm] = m;
        for (int x = 1; x < 30; x++) if ((pm * x) % 30 == 1) pinv[pm] = (uint32_t)x;
    }
    size_t icap2 = 0;
    for (size_t i = 0; i < np; i++) {
        if (hp[i] <= 13u) continue;
        icap2 += (size_t)((tileSlots / hp[i] + 2 + 63) / 64);
    }
    uint32_t *h_ip2 = (uint32_t *)malloc(icap2 * 4), *h_ik2 = (uint32_t *)malloc(icap2 * 4);
    if (!h_ip2 || !h_ik2) die("walk item alloc failed");
    size_t nii2 = 0;
    for (size_t i = 0; i < np; i++) {
        if (hp[i] <= 13u) continue;
        uint64_t marks = tileSlots / hp[i] + 2;
        uint32_t chunks = (uint32_t)((marks + 63) / 64);
        for (uint32_t c = 0; c < chunks; c++) { h_ip2[nii2] = (uint32_t)i; h_ik2[nii2] = c; nii2++; }
    }
    uint32_t h_wpidx[3] = {0, 0, 0};
    for (size_t i = 0; i < np; i++) {
        if (hp[i] == 7u) h_wpidx[0] = (uint32_t)i;
        if (hp[i] == 11u) h_wpidx[1] = (uint32_t)i;
        if (hp[i] == 13u) h_wpidx[2] = (uint32_t)i;
    }
    uint32_t wtab[8];
    {
        uint8_t cp[30];
        for (int i = 0; i < 30; i++) cp[i] = 0;
        for (int i = 0; i < 8; i++) cp[CLS8[i]] = 1;
        for (int c = 0; c < 8; c++) {
            uint32_t w = 0;
            for (uint32_t t = 1; t <= (uint32_t)gapmin - 1u; t++)
                if (cp[(CLS8[c] + t) % 30u]) w++;
            wtab[c] = w;
        }
    }

    /* ---- device ---- */
    cudaError_t e;
    e = cudaSetDevice(g_device);
    if (e != cudaSuccess) die_cuda(e, "cudaSetDevice");
    uint64_t *d_p = NULL, *d_iv = NULL, *d_r = NULL, *d_bm = NULL;
    uint32_t *d_ip2 = NULL, *d_ik2 = NULL, *d_wtab = NULL, *d_wpidx = NULL;
    w_gaprec_t *d_out = NULL;
    unsigned long long *d_stats = NULL;
    size_t region_words = (size_t)(K + 1) * cWords;
    e = cudaMalloc(&d_bm, 2 * region_words * 8); if (e != cudaSuccess) die_cuda(e, "bitmap alloc");
    e = cudaMalloc(&d_p, np * 8); if (e != cudaSuccess) die_cuda(e, "primes alloc");
    e = cudaMalloc(&d_iv, np * 8); if (e != cudaSuccess) die_cuda(e, "invp alloc");
    e = cudaMalloc(&d_r, np * 8); if (e != cudaSuccess) die_cuda(e, "r64 alloc");
    e = cudaMalloc(&d_ip2, nii2 * 4); if (e != cudaSuccess) die_cuda(e, "items alloc");
    e = cudaMalloc(&d_ik2, nii2 * 4); if (e != cudaSuccess) die_cuda(e, "items alloc");
    e = cudaMalloc(&d_wtab, 8 * 4); if (e != cudaSuccess) die_cuda(e, "wtab alloc");
    e = cudaMalloc(&d_wpidx, 3 * 4); if (e != cudaSuccess) die_cuda(e, "wpidx alloc");
    /* P0_PIPE: two result/stat slots, one per alternating bitmap region, so a
       batch's walk can stay in flight while the previous batch is collected. */
    e = cudaMalloc(&d_out, 2 * (size_t)(1u << 20) * sizeof(w_gaprec_t)); if (e != cudaSuccess) die_cuda(e, "gap buffer alloc");
    e = cudaMalloc(&d_stats, 4 * sizeof(unsigned long long)); if (e != cudaSuccess) die_cuda(e, "stats alloc");
    cudaMemcpy(d_p, hp, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip2, h_ip2, nii2 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ik2, h_ik2, nii2 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wtab, wtab, 8 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wpidx, h_wpidx, 3 * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(w_gaprec_t));
    cudaMemset(d_stats, 0, 2 * sizeof(unsigned long long));
#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_c30_tables(clsidx, perm, pinv) != 0)
        die("p0gpu_c30_tables failed");
    if (p0gpu_set_sieve_shared((size_t)tileWords * 8) != 0)
        die("p0gpu_set_sieve_shared failed");
#else
    cudaMemcpyToSymbol(c30_clsidx, clsidx, sizeof(clsidx));
    cudaMemcpyToSymbol(c30_perm, perm, sizeof(perm));
    cudaMemcpyToSymbol(c30_pinv, pinv, sizeof(pinv));
    cudaFuncSetAttribute(class30_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(tileWords * 8));
#endif
    cudaStream_t sA, sB, sC;
    cudaEvent_t evS[2], evW[2], evM[2], evX[2];
    /* P0_PIPE priorities: the walk is the critical path, the mark must fill its
       leftover latency (W1: the walk is latency/stream-limited, ~87% of its own
       pure-MR ceiling), so sB runs at the highest and sA at the lowest stream
       priority and the scheduler prefers walk CTAs.  Non-blocking streams: no
       implicit legacy-stream coupling (the collect copies use their own sC). */
    int prio_lo = 0, prio_hi = 0;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);   /* (least, greatest) */
    e = cudaStreamCreateWithPriority(&sA, cudaStreamNonBlocking, prio_lo); if (e != cudaSuccess) die_cuda(e, "stream create");
    e = cudaStreamCreateWithPriority(&sB, cudaStreamNonBlocking, prio_hi); if (e != cudaSuccess) die_cuda(e, "stream create");
    e = cudaStreamCreateWithPriority(&sC, cudaStreamNonBlocking, prio_hi); if (e != cudaSuccess) die_cuda(e, "stream create");
    for (int i = 0; i < 2; i++) {
        /* P0_SPLIT: timing-enabled - the three events measure the mark stage
           (evM->evS, stream A) and the walk stage (evS->evW, stream B); the
           sums are printed as the stage-split line in the summary. */
        e = cudaEventCreate(&evS[i]); if (e != cudaSuccess) die_cuda(e, "event create");
        e = cudaEventCreate(&evW[i]); if (e != cudaSuccess) die_cuda(e, "event create");
        e = cudaEventCreate(&evM[i]); if (e != cudaSuccess) die_cuda(e, "event create");
        e = cudaEventCreate(&evX[i]); if (e != cudaSuccess) die_cuda(e, "event create");
        cudaEventRecord(evW[i], sB);   /* pre-recorded: first waits are no-ops */
    }
    w_gaprec_t *hgaps = (w_gaprec_t *)malloc((size_t)(1u << 20) * sizeof(w_gaprec_t));
    if (!hgaps) die("gap host buffer alloc failed");

    /* ---- job bookkeeping, resume, monitor ---- */
    static job_t wj;
    memset(&wj, 0, sizeof wj);
    mpz_init(wj.scratch);
    wj.tid = 0;
    g_jobs = &wj;
    g_nthreads = 1;
    uint64_t b0 = 0;
    if (g_state_path) {
        if (load_state(g_state_path)) {
            uint64_t off = g_resume_off[0];
            b0 = (off + (uint64_t)(g_start - A)) / blockNum30;
            if (b0 > b_total) b0 = b_total;
            g_pre_ints = off > g_len ? g_len : off;
            printf("[phase0-gpu] resumed from %s: %llu ints already done before this session (%.2f%%)\n",
                   g_state_path, (unsigned long long)g_pre_ints,
                   100.0 * (double)g_pre_ints / (double)g_len);
        } else {
            printf("[phase0-gpu] no state file at %s; starting fresh\n", g_state_path);
        }
    }
    char s0[48];
    u128_str(g_start, s0, sizeof s0);
    printf("[phase0-gpu] engine=walk start=%s length=%llu ln(start)=%.4f\n",
           s0, (unsigned long long)g_len, u128_ln(g_start));
    printf("[phase0-gpu] walk: primes<=%u items=%zu gap_min=%llu walk_batch=%u "
           "block=%llu bitmap_region=%.2f GB x2\n",
           P, nii2, (unsigned long long)gapmin, K,
           (unsigned long long)blockNum30, (double)region_words * 8.0 / 1e9);
    if (g_log_path) printf("[phase0-gpu] gap log: %s\n", g_log_path);
    if (g_state_path) printf("[phase0-gpu] state: %s\n", g_state_path);
    if (g_log) {
        fprintf(g_log, "# phase0-gpu session %lld engine=walk start=%s length=%llu "
                       "gap_min=%llu walk_primes=%u walk_batch=%u merit_min=%.1f table=%s\n",
                (long long)time(NULL), s0, (unsigned long long)g_len,
                (unsigned long long)gapmin, P, K, g_merit_min, g_table_stamp);
        fflush(g_log);
    }
    pthread_t mon;
    int have_mon = 0;
    if (g_progress > 0 || g_state_path)
        if (pthread_create(&mon, NULL, monitor_thread, NULL) == 0) have_mon = 1;

    /* ---- main loop: super-batches with lookahead + 2 alternating regions ---- */
    int sb = 0;
    uint64_t b_next = b0;
    unsigned long long jumps_total = 0;
    double t_mark_gpu = 0.0, t_walk_gpu = 0.0;   /* P0_SPLIT stage sums (seconds) */
    /* P0_PIPE (2026-10-05): the host used to block on evW[r] right after launching
       batch b's walk, so the mark of batch b+1 could never overlap it (measured:
       mark 34% + walk 66% = 99.5% of wall, i.e. strictly serial).  Now the walk of
       batch b is collected one iteration LATER, while the next batch's marks are
       already queued on sA: steady state = sB runs walks back-to-back with the mark
       stage hidden behind them.  Safety: every batch owns one of the two per-region
       result slots (sB is stream-ordered, so slot reuse is safe), and a slot is read
       from the host only after cudaEventSynchronize of that batch's own evW. */
    int prev_r = -1;                     /* region of the launched, not-yet-collected batch */
    uint64_t prev_b = 0, prev_k = 0;     /* its first block and its reported block count */
    for (uint64_t b = b0; ; b += K) {
        int have_new = 0, nr = 0;
        uint64_t nb = 0, nk = 0;
        if (b < b_total && !g_stop) {
            int r = (sb++) & 1;
            uint64_t k = b_total - b; if (k > K) k = K;             /* blocks reported here */
            uint64_t cnb = k + ((b + k < b_total) ? 1 : 0);         /* coverage (+1 lookahead) */
            uint64_t *bm = d_bm + (size_t)r * region_words;
            u128 Ab = A + (u128)b * (u128)blockNum30;
            w_gaprec_t *out_r = d_out + (size_t)r * (1u << 20);
            unsigned long long *st_r = d_stats + 2 * (size_t)r;
            e = cudaStreamWaitEvent(sA, evW[r], 0);
            if (e != cudaSuccess) die_cuda(e, "wait evW");
            e = cudaEventRecord(evM[r], sA); if (e != cudaSuccess) die_cuda(e, "record evM");
            for (uint64_t j = 0; j < cnb; j++) {
                u128 av = A + (u128)(b + j) * (u128)blockNum30;
                u128 Aj = (av / 30u) * 30u;
#ifdef PHASE0_KERNEL_DLL
                if (p0gpu_class30_sieve(d_p, d_iv, d_r, d_ip2, d_ik2, (uint32_t)nii2,
                                       (uint64_t)Aj, (uint64_t)(Aj >> 64),
                                       tileSlots, tilesPerBlockC, bm + j * cWords,
                                       tileWords, d_wpidx, 1,
                                       sg_grid, sg_block, (size_t)tileWords * 8,
                                       sA) != 0)
                    die("p0gpu_class30_sieve failed");
#else
                class30_sieve_kernel<<<sg_grid, sg_block, (size_t)tileWords * 8, sA>>>(
                    d_p, d_iv, d_r, d_ip2, d_ik2, (uint32_t)nii2,
                    (uint64_t)Aj, (uint64_t)(Aj >> 64), tileSlots, tilesPerBlockC,
                    bm + j * cWords, tileWords, d_wpidx, 1);
#endif
                e = cudaGetLastError();
                if (e != cudaSuccess) die_cuda(e, "walk sieve launch");
            }
            e = cudaEventRecord(evS[r], sA); if (e != cudaSuccess) die_cuda(e, "record evS");
            e = cudaStreamWaitEvent(sB, evS[r], 0); if (e != cudaSuccess) die_cuda(e, "wait evS");
            e = cudaMemsetAsync(out_r, 0, sizeof(w_gaprec_t), sB); if (e != cudaSuccess) die_cuda(e, "gap reset");
            /* d_stats must be zeroed per batch as well: the kernel accumulates with
               atomicAdd and the host sums the per-batch values (without this the
               cumulative counts are summed again, over-reporting tests/jumps). */
            e = cudaMemsetAsync(st_r, 0, 2 * sizeof(unsigned long long), sB); if (e != cudaSuccess) die_cuda(e, "stats reset");
            e = cudaEventRecord(evX[r], sB); if (e != cudaSuccess) die_cuda(e, "record evX");
#ifdef PHASE0_KERNEL_DLL
            if (p0gpu_walk(bm, cnb * cWords, (uint64_t)Ab, (uint64_t)(Ab >> 64),
                           d_wtab, (uint32_t)gapmin, out_r, 1u << 20, st_r,
                           wk_grid, wk_block, sB) != 0)
                die("p0gpu_walk failed");
#else
            walk_kernel7<<<wk_grid, wk_block, 0, sB>>>(bm, cnb * cWords,
                (uint64_t)Ab, (uint64_t)(Ab >> 64), d_wtab, (uint32_t)gapmin,
                out_r, 1u << 20, st_r);
#endif
            e = cudaGetLastError(); if (e != cudaSuccess) die_cuda(e, "walk launch");
            e = cudaEventRecord(evW[r], sB); if (e != cudaSuccess) die_cuda(e, "record evW");
            have_new = 1; nr = r; nb = b; nk = k;
        }
        /* ---- collect the batch launched in the PREVIOUS iteration (P0_PIPE) ---- */
        if (prev_r >= 0) {
            w_gaprec_t *out_p = d_out + (size_t)prev_r * (1u << 20);
            unsigned long long *st_p = d_stats + 2 * (size_t)prev_r;
            e = cudaEventSynchronize(evW[prev_r]); if (e != cudaSuccess) die_cuda(e, "walk sync");
            {   /* P0_SPLIT: accumulate this batch's GPU-timeline stage durations.
                   mark = evM->evS (pure mark-kernel time, stream A); walk =
                   evX->evW (pure walk-kernel time, stream B - evX is recorded
                   after the memsets, so the memsets stay out of the walk
                   number).  When P0_PIPE overlaps the stages, mark+walk EXCEEDS
                   the wall and "other" goes negative - that is the overlap. */
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, evM[prev_r], evS[prev_r]) == cudaSuccess) t_mark_gpu += (double)ms * 1e-3;
                if (cudaEventElapsedTime(&ms, evX[prev_r], evW[prev_r]) == cudaSuccess) t_walk_gpu += (double)ms * 1e-3;
            }
            uint32_t ng = 0;
            unsigned long long st2[2] = {0, 0};
            /* The collect copies MUST NOT use the legacy default stream: a
               synchronous cudaMemcpy there serializes against EVERY blocking
               stream (implicit sync), i.e. it waits for the next batch's marks
               and walk that were just queued on sA/sB and re-serializes the
               pipeline (measured 2026-10-05: v1 of P0_PIPE still showed
               mark 34% + walk 66% = 100% of wall for exactly this reason).
               sC is a private stream and the source data is already final
               (the host synchronized evW[prev_r] above). */
            e = cudaMemcpyAsync(&ng, (char *)out_p + 8, 4, cudaMemcpyDeviceToHost, sC); if (e != cudaSuccess) die_cuda(e, "gap count copy");
            e = cudaMemcpyAsync(st2, st_p, 16, cudaMemcpyDeviceToHost, sC); if (e != cudaSuccess) die_cuda(e, "stats copy");
            e = cudaStreamSynchronize(sC); if (e != cudaSuccess) die_cuda(e, "collect sync");
            if (ng > (1u << 20) - 1) {
                fprintf(stderr, "[phase0-gpu] FATAL: walk gap buffer overflow (ng=%u)\n", ng);
                return 1;
            }
            wj.tests += st2[0];
            jumps_total += st2[1];
            wj.batches++;
            if (ng) {
                e = cudaMemcpyAsync(hgaps, out_p + 1, (size_t)ng * sizeof(w_gaprec_t), cudaMemcpyDeviceToHost, sC);
                if (e != cudaSuccess) die_cuda(e, "gap copy");
                e = cudaStreamSynchronize(sC); if (e != cudaSuccess) die_cuda(e, "gap copy sync");
                u128 rep_lo = A + (u128)prev_b * (u128)blockNum30;
                u128 rep_hi = A + (u128)(prev_b + prev_k) * (u128)blockNum30;
                for (uint32_t i = 0; i < ng; i++) {
                    u128 lower = ((u128)Ahi << 64) | (u128)hgaps[i].slot;
                    uint64_t gap = hgaps[i].gap;
                    if (lower < rep_lo || lower >= rep_hi) continue;  /* other batch's range */
                    if (lower < g_start || lower >= range_end) continue;
                    process_prime_gap(&wj, lower, lower + (u128)gap, gap);
                }
            }
            b_next = prev_b + prev_k;
            {   /* progress/checkpoint accounting (report-end of this batch) */
                uint64_t done = (uint64_t)((uint64_t)((A + (u128)(prev_b + prev_k) * (u128)blockNum30) - g_start));
                if (done > g_len) done = g_len;
                __sync_synchronize();
                g_resume_off[0] = done;
                wj.ints = done - g_pre_ints;
            }
            prev_r = -1;
        }
        if (have_new) { prev_r = nr; prev_b = nb; prev_k = nk; }
        else break;
    }
    int all_done = (b_next >= b_total) ? 1 : 0;
    g_mon_stop = 1;
    if (have_mon) pthread_join(mon, NULL);
    double wall = now_s() - g_t0;
    if (g_state_path) {
        if (all_done) {
            remove(g_state_path);
            printf("[phase0-gpu] range complete; state file removed\n");
        } else {
            write_state(g_state_path);
            printf("[phase0-gpu] STOPPED early; checkpoint saved to %s - rerun the same "
                   "command to resume\n", g_state_path);
        }
    }
    wj.completed = all_done;

    /* ---- summary (same shape as the sieve engine's) ---- */
    {
        uint64_t ints = g_pre_ints + wj.ints;
        if (ints > g_len) ints = g_len;
        double per14 = (ints > 0) ? (1e14 / (double)ints) : 0.0;
        printf("[phase0-gpu] --- results ---\n");
        printf("[phase0-gpu] wall=%.2f s  ints=%llu  end_to_end=%.3e ints/s%s\n",
               wall, (unsigned long long)ints, (double)ints / wall,
               all_done ? "" : "  (STOPPED early)");
        printf("[phase0-gpu] gpu batches=%llu  tests=%llu  test_rate=%.3e tests/s  "
               "tests/jump=%.2f\n",
               (unsigned long long)wj.batches, (unsigned long long)wj.tests,
               wall > 0 ? (double)wj.tests / wall : 0.0,
               jumps_total > 0 ? (double)wj.tests / (double)jumps_total : 0.0);
        if (t_mark_gpu + t_walk_gpu > 0.0) {
            double other = wall - t_mark_gpu - t_walk_gpu;
            printf("[phase0-gpu] stage split (GPU events): mark=%.1f s (%.1f%%)  "
                   "walk=%.1f s (%.1f%%)  other=%.1f s (%.1f%%) "
                   "[other = host/sync/launch+boot; negative = mark/walk overlap]\n",
                   t_mark_gpu, 100.0 * t_mark_gpu / wall,
                   t_walk_gpu, 100.0 * t_walk_gpu / wall,
                   other, 100.0 * other / wall);
        }
        printf("[phase0-gpu] gaps per merit threshold (this session -> per 1e14 ints):\n");
        for (int bnd = 0; bnd < N_BANDS; bnd++)
            printf("[phase0-gpu]   m>=%2.0f : %8llu  ->  %.4g\n",
                   k_bands[bnd], (unsigned long long)wj.band_cnt[bnd],
                   (double)wj.band_cnt[bnd] * per14);
        for (int r2 = 0; r2 < wj.ntop; r2++) {
            char sl[48], su[48];
            u128_str(wj.top_lower[r2], sl, sizeof sl);
            u128_str(wj.top_upper[r2], su, sizeof su);
            int ok = verify_gap(wj.top_lower[r2], wj.top_upper[r2]);
            printf("[phase0-gpu]   top#%d gap=%llu merit=%.4f lower=%s upper=%s verified=%d\n",
                   r2 + 1, (unsigned long long)wj.top_gap[r2], wj.top_merit[r2], sl, su, ok);
        }
        printf("[phase0-gpu] gaps reported=%llu  verification_failures=%llu  records_new=%llu%s\n",
               (unsigned long long)wj.gaps_reported, (unsigned long long)wj.gaps_bad,
               (unsigned long long)wj.records_new, g_log_path ? "" : "  (no --log file)");
        if (g_table) printf("[phase0-gpu] records table: %s\n", g_table_stamp);
        printf("[phase0-gpu] --- end ---\n");
    }

    cudaStreamDestroy(sA); cudaStreamDestroy(sB); cudaStreamDestroy(sC);
    for (int i = 0; i < 2; i++) { cudaEventDestroy(evS[i]); cudaEventDestroy(evW[i]); cudaEventDestroy(evM[i]); cudaEventDestroy(evX[i]); }
    cudaFree(d_bm); cudaFree(d_p); cudaFree(d_iv); cudaFree(d_r);
    cudaFree(d_ip2); cudaFree(d_ik2); cudaFree(d_wtab); cudaFree(d_wpidx);
    cudaFree(d_out); cudaFree(d_stats);
    free(hgaps); free(hp); free(h_iv); free(h_r); free(h_ip2); free(h_ik2);
    mpz_clear(wj.scratch);
    return 0;
}

/* ---------------- --check: full-sequence comparison vs GMP ---------------- */

static int run_check(u128 start, uint64_t len) {
    if (len == 0) len = 2000000;
    printf("[phase0-gpu] --check %llu candidates from start (full production path + GMP walk)\n",
           (unsigned long long)len);

    job_t j;
    memset(&j, 0, sizeof j);
    j.tid = 0;
    j.start = start;
    j.end = start + len;
    j.seg_bits = 20;
    j.do_test = 1;
    j.cap = g_batch;
    {
        cudaError_t e2 = cudaHostAlloc((void **)&j.steps, (size_t)j.cap * sizeof(uint32_t),
                                       cudaHostAllocDefault);
        if (e2 != cudaSuccess) die_cuda(e2, "pinned steps alloc");
        e2 = cudaHostAlloc((void **)&j.res, j.cap, cudaHostAllocDefault);
        if (e2 != cudaSuccess) die_cuda(e2, "pinned res alloc");
    }
    mpz_init(j.scratch);
    if (g_use_gpu_sieve) job_gs_alloc(&j);

    size_t cap = (size_t)(len / 10) + 4096;
    j.collect = (u128 *)malloc(cap * sizeof(u128));
    if (!j.collect) die("check collect alloc failed");
    j.collect_cap = cap;

    worker(&j);

    /* GMP reference walk from the last prime < start (mirrors phase0_scan) */
    mpz_t gp, t;
    mpz_init(gp);
    mpz_init(t);
    u128 pv = start - 1;
    if (((uint64_t)pv & 1ULL) == 0) pv -= 1;
    while (pv > 2 && !u128_is_probable(pv, t)) pv -= 2;
    u128_to_mpz(gp, pv);

    size_t mismatch = 0;
    for (size_t i = 0; i < j.collect_n && mismatch <= 5; i++) {
        mpz_nextprime(gp, gp);
        u128 g = mpz_to_u128(gp);
        if (g != j.collect[i]) {
            char a[48], b[48];
            u128_str(g, a, sizeof a);
            u128_str(j.collect[i], b, sizeof b);
            fprintf(stderr, "[phase0-gpu] CHECK MISMATCH #%zu: gmp=%s gpu-path=%s\n", i, a, b);
            mismatch++;
        }
    }
    char s0[48];
    u128_str(start, s0, sizeof s0);
    int rc = 0;
    if (mismatch) {
        fprintf(stderr, "[phase0-gpu] CHECK FAILED (%zu mismatches)\n", mismatch);
        rc = 1;
    } else {
        printf("[phase0-gpu] CHECK PASS: %zu primes in [%s, %s+%llu) identical between "
               "bucketed sieve + GPU base-2 MR + GMP verify path and a pure GMP "
               "mpz_nextprime walk\n",
               j.collect_n, s0, s0, (unsigned long long)len);
    }
    if (g_use_gpu_sieve) job_gs_free(&j);
    cudaFreeHost(j.steps); cudaFreeHost(j.res); free(j.collect);
    mpz_clear(j.scratch); mpz_clear(gp); mpz_clear(t);
    return rc;
}

/* ---------------- GPU init ---------------- */

static void gpu_init(void) {
    cudaError_t e = cudaSetDevice(g_device);
    if (e != cudaSuccess) die_cuda(e, "cudaSetDevice");
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, g_device) == cudaSuccess)
        printf("[phase0-gpu] device %d: %s (sm_%d%d, %d SMs), batch=%u candidates\n",
               g_device, prop.name, prop.major, prop.minor, prop.multiProcessorCount, g_batch);
    e = cudaMalloc(&g_d_steps, (size_t)g_batch * sizeof(uint32_t));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc steps");
    e = cudaMalloc(&g_d_base3, 3 * sizeof(uint32_t));
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc base");
    e = cudaMalloc(&g_d_res, g_batch);
    if (e != cudaSuccess) die_cuda(e, "cudaMalloc results");
}

/* ---------------- usage / main ---------------- */

static void usage(const char *p) {
    fprintf(stderr,
        "usage: %s [--start DEC] [--length DEC] [--sieve-limit P] [--threads T]\n"
        "          [--seg-bits B] [--batch K] [--merit-min M] [--log FILE]\n"
        "          [--state FILE] [--state-every S] [--progress S] [--device D]\n"
        "          [--records FILE] [--no-records] [--gpu-sieve] [--check [N]]\n"
        "          [--legacy-mr] [--engine sieve|walk] [--gap-min G]\n"
        "          [--walk-primes P] [--walk-batch B]\n"
        "\n"
        "  sieve: default is CPU marking (wheel tile + bucket marking of\n"
        "         17..sieve-limit).  --gpu-sieve moves the whole mark stage\n"
        "         to the GPU (p0_mark kernel, per-thread double-buffered\n"
        "         bitmaps); it requires the test stage (no --no-test).\n"
        "  --engine walk selects the class-30 bitmap sieve + batched jump-walk\n"
        "         engine (approx. 10x faster than the exhaustive sieve path):\n"
        "         it reports every gap >= --gap-min (default ceil(merit-min *\n"
        "         ln(start))) through the same log/records/verify path, with\n"
        "         --walk-primes (sieve depth, default 15000) and --walk-batch\n"
        "         (blocks per super-batch, default 64, range 4..96; VRAM =\n"
        "         2 x (K+1) x 33.6 MB: K=32 2.2 GB, K=64 4.4 GB, K=96 6.5 GB).  The range\n"
        "         must fit one 64-bit window of the block base (true for all\n"
        "         realistic ranges; the check aborts otherwise).  --check,\n"
        "         --gpu-sieve/--cpu-sieve, --legacy-mr and --sieve-limit apply\n"
        "         to the sieve engine only.\n"
        "  MR pipeline (--gpu-sieve, test stage, no --check): default is the\n"
        "         bitmap-verdict pass (GPU tests every candidate straight from\n"
        "         the survivor bitmap; host sees one verdict bitmap per\n"
        "         segment).  --legacy-mr selects the batched steps/res path\n"
        "         (4 M candidates per flush; reference implementation for A/B).\n"
        "  defaults: start=200000000000000000000 (2e20) length=10000000000 (1e10)\n"
        "            threads=8 merit-min=20 progress=10s state-every=30s device=0\n"
        "            --gpu-sieve (default): sieve-limit=1e7 batch=4194304 seg-bits=24\n"
        "            --cpu-sieve:           sieve-limit=30000 batch=1048576 seg-bits=20\n"
        "  records: --records FILE (default: data/prime_gap_merits.txt if present);\n"
        "           --no-records disables the check.  Criterion identical to the\n"
        "           gap-hunt watcher and new_src/record_log.c: gap in the table AND\n"
        "           merit > best known; a beat is logged even below --merit-min.\n"
        "  gap log format (append): <unix_ts> <lower_prime> <gap> <merit> <upper_prime>\n"
        "           verified=<0|1> record=<NEW|no|absent|off> table=<best|unknown>,\n"
        "           plus one '# phase0-gpu session ...' header line per run.\n"
        "\n", p);
}

int main(int argc, char **argv) {
    u128 start = (u128)200000000000000000ULL * 1000ULL;  /* 2e20 */
    uint64_t length = 10000000000ULL;
    uint32_t sieve_limit = 0;          /* mode-dependent default below */
    int threads = 8, do_test = 1, seg_bits = 0;
    int sl_set = 0, seg_set = 0, batch_set = 0;
    int do_check = 0;
    uint64_t check_len = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_u128(argv[++i]);
        else if (!strcmp(argv[i], "--length") && i + 1 < argc) length = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--sieve-limit") && i + 1 < argc) { sieve_limit = (uint32_t)strtoul(argv[++i], NULL, 10); sl_set = 1; }
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seg-bits") && i + 1 < argc) { seg_bits = atoi(argv[++i]); seg_set = 1; }
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc) { g_batch = (uint32_t)strtoul(argv[++i], NULL, 10); batch_set = 1; }
        else if (!strcmp(argv[i], "--merit-min") && i + 1 < argc) { g_merit_min = atof(argv[++i]); g_merit_explicit = 1; }
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) g_log_path = argv[++i];
        else if (!strcmp(argv[i], "--records") && i + 1 < argc) g_records_path = argv[++i];
        else if (!strcmp(argv[i], "--no-records")) g_no_records = 1;
        else if (!strcmp(argv[i], "--gpu-sieve")) g_use_gpu_sieve = 1;
        else if (!strcmp(argv[i], "--cpu-sieve")) g_use_gpu_sieve = 0;
        else if (!strcmp(argv[i], "--state") && i + 1 < argc) g_state_path = argv[++i];
        else if (!strcmp(argv[i], "--state-every") && i + 1 < argc) g_state_every = atof(argv[++i]);
        else if (!strcmp(argv[i], "--progress") && i + 1 < argc) g_progress = atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) g_device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-test")) do_test = 0;
        else if (!strcmp(argv[i], "--legacy-mr")) g_fast_mr = 0;
        else if (!strcmp(argv[i], "--engine") && i + 1 < argc) {
            const char *en = argv[++i];
            if (!strcmp(en, "walk")) g_walk_engine = 1;
            else if (!strcmp(en, "sieve")) g_walk_engine = 0;
            else { fprintf(stderr, "[phase0-gpu] --engine must be sieve|walk\n"); return 2; }
        }
        else if (!strcmp(argv[i], "--gap-min") && i + 1 < argc) g_walk_gapmin = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--walk-primes") && i + 1 < argc) g_walk_primes = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--walk-batch") && i + 1 < argc) g_walk_batch = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--check")) {
            do_check = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') check_len = strtoull(argv[++i], NULL, 10);
        } else { usage(argv[0]); return 2; }
    }

    if (!sl_set) sieve_limit = g_use_gpu_sieve ? 10000000u : 30000u;
    if (!batch_set) g_batch = g_use_gpu_sieve ? 4194304u : 1048576u;
    if (!seg_set) seg_bits = g_use_gpu_sieve ? 24 : 20;
    if (seg_bits < 16 || seg_bits > 27) { fprintf(stderr, "[phase0-gpu] --seg-bits out of range [16,27]\n"); return 2; }
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    if (g_batch < 1024 || g_batch > (1u << 24)) { fprintf(stderr, "[phase0-gpu] --batch out of range [1024, 16777216]\n"); return 2; }
    {
        const char *dbg = getenv("P0_FAST_DBG");
        if (dbg)
            g_fast_dbg = atoi(dbg);
    }
    if (g_device < 0 || g_device > 15) { fprintf(stderr, "[phase0-gpu] --device out of range\n"); return 2; }
    if (!g_walk_engine && g_use_gpu_sieve && !do_test) { fprintf(stderr, "[phase0-gpu] --gpu-sieve requires the test stage\n"); return 2; }
    if (length >= ((uint64_t)1 << 63)) { fprintf(stderr, "[phase0-gpu] --length must be < 2^63\n"); return 2; }
    if (!g_walk_engine && start <= (u128)sieve_limit) { fprintf(stderr, "[phase0-gpu] --start must exceed --sieve-limit\n"); return 2; }
    if (start + (u128)length > ((u128)1 << 96)) { fprintf(stderr, "[phase0-gpu] range exceeds the 96-bit kernel container\n"); return 2; }

    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    g_start = start;
    g_len = length;
    g_nthreads = threads;
    g_total_ints = length;
    g_t0 = now_s();
    {
        double gf = (g_merit_min < 6.0) ? g_merit_min : 6.0;
        if (gf < 0.0) gf = 0.0;
        g_gap_gate = (uint64_t)(gf * u128_ln(start));
    }

    gen_primes(sieve_limit);
    wheel_init();

    if (do_check) {
        if (g_walk_engine) { fprintf(stderr, "[phase0-gpu] --check is sieve-engine only\n"); return 2; }
        gpu_init();
        if (g_use_gpu_sieve) gpu_sieve_init(seg_bits);
        return run_check(start, check_len);
    }

    if (g_log_path) {
        g_log = fopen(g_log_path, "a");
        if (!g_log) { fprintf(stderr, "[phase0-gpu] cannot open log %s\n", g_log_path); return 1; }
    }

    /* records table (best-known merits, hunt criterion) */
    if (!g_no_records) {
        const char *rp = g_records_path ? g_records_path : "data/prime_gap_merits.txt";
        if (load_records(rp)) {
            printf("[phase0-gpu] records table: %s\n", g_table_stamp);
        } else if (g_records_path) {
            fprintf(stderr, "[phase0-gpu] WARNING: cannot read records table %s"
                            " (record check off)\n", rp);
        } else {
            printf("[phase0-gpu] records table data/prime_gap_merits.txt not found"
                   " (record check off; use --records FILE)\n");
        }
    }

    if (g_walk_engine) return run_walk_engine();

    char s0[48];
    u128_str(start, s0, sizeof s0);
    printf("[phase0-gpu] start=%s  length=%llu  ln(start)=%.4f\n",
           s0, (unsigned long long)length, u128_ln(start));
    printf("[phase0-gpu] sieve_limit=%u  threads=%d  seg_bits=%d  batch=%u  test=%s  merit_min=%.1f\n",
           sieve_limit, threads, seg_bits, g_batch, do_test ? "on" : "off", g_merit_min);
    if (g_log_path) printf("[phase0-gpu] gap log: %s\n", g_log_path);
    if (g_state_path) printf("[phase0-gpu] state: %s\n", g_state_path);
    if (g_log) {
        fprintf(g_log, "# phase0-gpu session %lld start=%s length=%llu threads=%d "
                       "sieve_limit=%u seg_bits=%d batch=%u merit_min=%.1f table=%s\n",
                (long long)time(NULL), s0, (unsigned long long)length, threads,
                sieve_limit, seg_bits, g_batch, g_merit_min, g_table_stamp);
        fflush(g_log);
    }

    if (do_test) gpu_init();
    if (g_use_gpu_sieve) gpu_sieve_init(seg_bits);

    /* slice geometry + resume offsets */
    u128 slice_lo[MAX_THREADS], slice_hi[MAX_THREADS];
    u128 lo_abs[MAX_THREADS];
    for (int t = 0; t < threads; t++) {
        slice_lo[t] = start + (u128)length * (u128)t / (u128)threads;
        slice_hi[t] = start + (u128)length * (u128)(t + 1) / (u128)threads;
        g_resume_off[t] = (uint64_t)(slice_lo[t] - start);
    }
    if (g_state_path) {
        if (load_state(g_state_path)) {
            uint64_t pre = 0;
            for (int t = 0; t < threads; t++) {
                u128 lo = start + (u128)g_resume_off[t];
                if (lo < slice_lo[t]) lo = slice_lo[t];
                if (lo > slice_hi[t]) lo = slice_hi[t];
                g_resume_off[t] = (uint64_t)(lo - start);
                pre += (uint64_t)(lo - slice_lo[t]);
            }
            g_pre_ints = pre;
            printf("[phase0-gpu] resumed from %s: %llu ints already done before this session (%.2f%%)\n",
                   g_state_path, (unsigned long long)pre,
                   100.0 * (double)pre / (double)length);
        } else {
            printf("[phase0-gpu] no state file at %s; starting fresh\n", g_state_path);
        }
    }
    for (int t = 0; t < threads; t++) {
        u128 lo = start + (u128)g_resume_off[t];
        if (lo < slice_lo[t]) lo = slice_lo[t];
        if (lo > slice_hi[t]) lo = slice_hi[t];
        lo_abs[t] = lo;
    }

    job_t *jobs = (job_t *)calloc((size_t)threads, sizeof(job_t));
    pthread_t *tid = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!jobs || !tid) die("job alloc failed");
    g_jobs = jobs;

    for (int t = 0; t < threads; t++) {
        jobs[t].tid = t;
        jobs[t].start = lo_abs[t];
        jobs[t].end = slice_hi[t];
        jobs[t].seg_bits = (uint32_t)seg_bits;
        jobs[t].do_test = do_test;
        jobs[t].cap = g_batch;
        mpz_init(jobs[t].scratch);
        if (do_test) {
            cudaError_t e2;
            e2 = cudaHostAlloc((void **)&jobs[t].steps, (size_t)g_batch * sizeof(uint32_t),
                               cudaHostAllocDefault);
            if (e2 != cudaSuccess) die_cuda(e2, "pinned steps alloc");
            e2 = cudaHostAlloc((void **)&jobs[t].res, g_batch, cudaHostAllocDefault);
            if (e2 != cudaSuccess) die_cuda(e2, "pinned res alloc");
        }
        if (g_use_gpu_sieve) job_gs_alloc(&jobs[t]);
    }

    pthread_t mon;
    int have_mon = 0;
    if (g_progress > 0 || g_state_path) {
        if (pthread_create(&mon, NULL, monitor_thread, NULL) == 0) have_mon = 1;
    }

    if (threads == 1) {
        worker(&jobs[0]);
    } else {
        for (int t = 0; t < threads; t++)
            if (pthread_create(&tid[t], NULL, worker, &jobs[t]) != 0)
                die("pthread_create failed");
        for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    }

    g_mon_stop = 1;
    if (have_mon) pthread_join(mon, NULL);

    double wall = now_s() - g_t0;
    int all_done = 1;
    for (int t = 0; t < threads; t++) if (!jobs[t].completed) all_done = 0;

    if (g_state_path) {
        if (all_done) {
            remove(g_state_path);
            printf("[phase0-gpu] range complete; state file removed\n");
        } else {
            write_state(g_state_path);
            printf("[phase0-gpu] STOPPED early; checkpoint saved to %s - rerun the same "
                   "command to resume\n", g_state_path);
        }
    }

    /* aggregate */
    uint64_t ints = 0, surv = 0, tests = 0, primes = 0;
    uint64_t tot_batches = 0, tot_gaps = 0, tot_bad = 0, tot_rec = 0;
    uint64_t bands[N_BANDS] = {0};
    double t_mark = 0, t_walk = 0, t_wait = 0, t_gpu = 0, t_proc = 0, t_boot = 0;
    u128 top_lower[4] = {0, 0, 0, 0}, top_upper[4] = {0, 0, 0, 0};
    uint64_t top_gap[4] = {0};
    double top_merit[4] = {0};
    int ntop = 0;
    for (int t = 0; t < threads; t++) {
        ints += jobs[t].ints;
        surv += jobs[t].survivors;
        tests += jobs[t].tests;
        primes += jobs[t].primes_found;
        tot_batches += jobs[t].batches;
        tot_gaps += jobs[t].gaps_reported;
        tot_bad += jobs[t].gaps_bad;
        tot_rec += jobs[t].records_new;
        t_mark += jobs[t].t_mark;
        t_gpu += jobs[t].t_gpu;
        t_walk += jobs[t].t_walk;
        t_wait += jobs[t].t_wait;
        t_proc += jobs[t].t_proc;
        t_boot += jobs[t].t_boot;
        for (int b = 0; b < N_BANDS; b++) bands[b] += jobs[t].band_cnt[b];
        for (int r = 0; r < jobs[t].ntop; r++) {
            int pos = -1;
            for (int q = 0; q < 4; q++) if (jobs[t].top_merit[r] > top_merit[q]) { pos = q; break; }
            if (pos < 0 && ntop < 4) pos = ntop;
            if (pos < 0) continue;
            for (int q = 3; q > pos; q--) {
                top_merit[q] = top_merit[q - 1];
                top_lower[q] = top_lower[q - 1];
                top_upper[q] = top_upper[q - 1];
                top_gap[q]   = top_gap[q - 1];
            }
            top_merit[pos] = jobs[t].top_merit[r];
            top_lower[pos] = jobs[t].top_lower[r];
            top_upper[pos] = jobs[t].top_upper[r];
            top_gap[pos]   = jobs[t].top_gap[r];
            if (ntop < 4) ntop++;
        }
    }

    double per14 = (ints > 0) ? (1e14 / (double)ints) : 0.0;
    printf("[phase0-gpu] --- results ---\n");
    printf("[phase0-gpu] wall=%.2f s  ints=%llu  end_to_end=%.3e ints/s%s\n",
           wall, (unsigned long long)ints, (double)ints / wall,
           all_done ? "" : "  (STOPPED early)");
    printf("[phase0-gpu] sieve marking=%.2f s  marking_rate=%.3e ints/s\n",
           t_mark, t_mark > 0 ? (double)ints / t_mark : 0.0);
    printf("[phase0-gpu] thread accounting (CPU-seconds per thread):\n");
    for (int t = 0; t < threads; t++)
        printf("[phase0-gpu]   t%d: mark=%.2f walk=%.2f wait=%.2f gpu=%.2f proc=%.2f boot=%.2f\n",
               t, jobs[t].t_mark, jobs[t].t_walk, jobs[t].t_wait,
               jobs[t].t_gpu, jobs[t].t_proc, jobs[t].t_boot);
    printf("[phase0-gpu] totals (CPU-s sums): mark=%.2f walk=%.2f wait=%.2f gpu=%.2f "
           "proc=%.2f boot=%.2f\n",
           t_mark, t_walk, t_wait, t_gpu, t_proc, t_boot);
    if (g_fast_mr) {
        double smm = 0, smr = 0, scp = 0;
        for (int t = 0; t < threads; t++) {
            smm += jobs[t].t_mm;
            smr += jobs[t].t_mr;
            scp += jobs[t].t_cp;
        }
        printf("[phase0-gpu] fast split (GPU-s sums): wheel+mark=%.3f MR=%.3f "
               "copies=%.3f\n", smm, smr, scp);
    }
    printf("[phase0-gpu] wall-share estimate (sum/threads vs wall %.2f s): mark=%.2f "
           "walk=%.2f wait=%.2f gpu=%.2f proc=%.2f boot=%.2f\n",
           wall, t_mark / threads, t_walk / threads, t_wait / threads,
           t_gpu / threads, t_proc / threads, t_boot / threads);
    printf("[phase0-gpu] survivors=%llu  u=%.3f%% of ints (%.3f%% of odds)\n",
           (unsigned long long)surv,
           100.0 * (double)surv / (double)ints,
           100.0 * (double)surv / ((double)ints / 2.0));
    if (do_test) {
        printf("[phase0-gpu] gpu batches=%llu  tests=%llu  gpu-busy=%.2f s (%.1f%% of wall; "
               "per-thread cost %.2f s)  gpu_rate=%.3e tests/s\n",
               (unsigned long long)tot_batches, (unsigned long long)tests, t_gpu,
               wall > 0 ? 100.0 * t_gpu / wall : 0.0,
               t_gpu / threads,
               t_gpu > 0 ? (double)tests / t_gpu : 0.0);
        printf("[phase0-gpu] primes=%llu\n", (unsigned long long)primes);
        printf("[phase0-gpu] gaps per merit threshold (this session -> per 1e14 ints):\n");
        for (int b = 0; b < N_BANDS; b++)
            printf("[phase0-gpu]   m>=%2.0f : %8llu  ->  %.4g\n",
                   k_bands[b], (unsigned long long)bands[b], (double)bands[b] * per14);
        for (int r = 0; r < ntop; r++) {
            char sl[48], su[48];
            u128_str(top_lower[r], sl, sizeof sl);
            u128_str(top_upper[r], su, sizeof su);
            int ok = verify_gap(top_lower[r], top_upper[r]);
            printf("[phase0-gpu]   top#%d gap=%llu merit=%.4f lower=%s upper=%s verified=%d\n",
                   r + 1, (unsigned long long)top_gap[r], top_merit[r], sl, su, ok);
        }
        printf("[phase0-gpu] gaps reported=%llu  verification_failures=%llu  records_new=%llu%s\n",
               (unsigned long long)tot_gaps, (unsigned long long)tot_bad,
               (unsigned long long)tot_rec,
               g_log_path ? "" : "  (no --log file)");
        if (g_table) printf("[phase0-gpu] records table: %s\n", g_table_stamp);
        int bb = 0;
        for (int b = 0; b < N_BANDS; b++) if (k_bands[b] <= g_merit_min) bb = b;
        printf("[phase0-gpu] list_estimate: ~%.1f MB per 1e14 ints at m>=%.0f (12 B/record)\n",
               (double)bands[bb] * per14 * 12.0 / 1e6, k_bands[bb]);
    }
    printf("[phase0-gpu] --- end ---\n");

    for (int t = 0; t < threads; t++) {
        if (g_use_gpu_sieve) job_gs_free(&jobs[t]);
        cudaFreeHost(jobs[t].steps);
        cudaFreeHost(jobs[t].res);
        mpz_clear(jobs[t].scratch);
    }
    free(jobs); free(tid);
    if (g_log) fclose(g_log);
    if (g_d_steps) cudaFree(g_d_steps);
    if (g_d_base3) cudaFree(g_d_base3);
    if (g_d_res) cudaFree(g_d_res);
    if (g_d_items) cudaFree(g_d_items);
    if (g_d_primes64) cudaFree(g_d_primes64);
    if (g_d_r64) cudaFree(g_d_r64);
    if (g_d_invp) cudaFree(g_d_invp);
    if (g_d_wpat) cudaFree(g_d_wpat);
    free(g_primes);
    return 0;
}
