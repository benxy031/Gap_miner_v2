/*
 * phase0_scan.c — Phase-0 exhaustive prime-gap scan prototype (CPU reference).
 *
 * For a contiguous integer range [start, start+length) near a chosen magnitude
 * (default 2e20, the white paper's target class) this tool measures:
 *   - sieve marking rate (integers/s) with a bucketed segmented sieve
 *   - survivor fraction after sieving to --sieve-limit
 *   - primality-test rate (GMP mpz_probab_prime_p) and the end-to-end rate
 *   - gaps above merit thresholds and the implied count per 1e14 integers
 *   - a result-list size estimate (bytes per 1e14 integers)
 *
 * Correctness gate: `--check N` re-walks [start, start+N) with GMP
 * mpz_nextprime and compares the COMPLETE prime sequence against the primes
 * it found (catches sieve over-marking / any missed prime).  Run it before
 * trusting any timing run.
 *
 * Usage:
 *   phase0_scan [--start DEC] [--length DEC] [--sieve-limit P] [--threads T]
 *               [--merit-min M] [--seg-bits B] [--no-test] [--check [N]]
 * Defaults: start=200000000000000000000 (2e20)  length=10000000000 (1e10)
 *           sieve-limit=100000000 (1e8)  threads=4  merit-min=20  seg-bits=20
 *
 * Notes:
 *   - Numbers are 128-bit (__int128); the default start (2e20) is ~68 bits,
 *     i.e. above 2^64, so the 64-bit deterministic MR shortcut does not apply
 *     and GMP is used as the (slow but authoritative) reference test.
 *   - This is a CPU reference for the white paper's Phase 0: the GPU MR rate
 *     is measured separately with `bin/bench_fermat 2 <batch> <iters> 68`.
 *   - The bucketed sieve keeps one "next multiple" probe per prime, so the
 *     per-segment overhead is O(multiples), not O(primes x segments).
 *
 * Build: make bin/phase0_scan
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gmp.h>

typedef unsigned __int128 u128;

#define N_BANDS 7
static const double k_bands[N_BANDS] = {10, 15, 20, 25, 30, 35, 40};

/* the merit computation uses the gap's LOWER prime, matching "merit = g / ln(p)" */
static double u128_ln(u128 x) { return (double)logl((long double)x); }

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

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

static uint32_t *g_primes;
static size_t    g_nprimes;

static void gen_primes(uint32_t limit) {
    size_t half = (size_t)(limit / 2) + 1;
    uint8_t *comp = (uint8_t *)calloc(half, 1);
    if (!comp) { fprintf(stderr, "[phase0] primes alloc failed\n"); exit(1); }
    for (size_t k = 1; (uint64_t)(2 * k + 1) * (2 * k + 1) <= (uint64_t)limit; k++) {
        if (comp[k]) continue;
        uint64_t p = 2 * k + 1;
        for (uint64_t m = p * p; m <= (uint64_t)limit; m += 2 * p)
            comp[m / 2] = 1;
    }
    size_t cnt = 0;
    for (size_t k = 1; k < half; k++) if (!comp[k]) cnt++;
    g_primes = (uint32_t *)malloc((cnt ? cnt : 1) * sizeof(uint32_t));
    if (!g_primes) { fprintf(stderr, "[phase0] primes alloc failed\n"); exit(1); }
    size_t j = 0;
    for (size_t k = 1; k < half; k++) if (!comp[k]) g_primes[j++] = (uint32_t)(2 * k + 1);
    g_nprimes = j;
    free(comp);
    fprintf(stderr, "[phase0] sieve primes up to %u: %zu odd primes\n", limit, g_nprimes);
}

/* ---------------- job + worker ---------------- */

typedef struct {
    u128 start, end;
    uint32_t seg_bits;
    int do_test;
    /* optional collection (for --check) */
    u128 *collect;
    size_t collect_cap, collect_n;
    /* results */
    uint64_t ints, survivors, tests, primes_found;
    uint64_t band_cnt[N_BANDS];
    double t_sieve, t_test, wall;
    int      ntop;
    u128     top_start[4];
    uint64_t top_gap[4];
    double   top_merit[4];
} job_t;

static void top_insert(job_t *j, u128 start, uint64_t gap, double merit) {
    int pos = -1;
    for (int t = 0; t < 4; t++) {
        if (merit > j->top_merit[t]) { pos = t; break; }
    }
    if (pos < 0 && j->ntop < 4) pos = j->ntop;
    if (pos < 0) return;
    for (int t = 3; t > pos; t--) {
        j->top_merit[t] = j->top_merit[t - 1];
        j->top_start[t] = j->top_start[t - 1];
        j->top_gap[t]   = j->top_gap[t - 1];
    }
    j->top_merit[pos] = merit;
    j->top_start[pos] = start;
    j->top_gap[pos]   = gap;
    if (j->ntop < 4) j->ntop++;
}

static void *worker(void *arg) {
    job_t *j = (job_t *)arg;
    double w0 = now_s();

    const uint64_t seg = 1ULL << j->seg_bits;
    const u128 lo = j->start, hi = j->end;
    const uint64_t span = (uint64_t)(hi - lo);
    const uint64_t nseg = (span + seg - 1) / seg;
    if (nseg > (1ULL << 26)) {
        fprintf(stderr, "[phase0] range/segment-ratio too large (%llu segments) — raise --seg-bits\n",
                (unsigned long long)nseg);
        exit(1);
    }

    int32_t  *heads   = (int32_t *)malloc((size_t)nseg * sizeof(int32_t));
    int32_t  *nexti   = (int32_t *)malloc(g_nprimes * sizeof(int32_t));
    uint64_t *nextoff = (uint64_t *)malloc(g_nprimes * sizeof(uint64_t));
    uint8_t  *bits    = (uint8_t *)malloc(seg);
    if (!heads || !nexti || !nextoff || !bits) {
        fprintf(stderr, "[phase0] worker alloc failed\n");
        exit(1);
    }
    for (uint64_t s = 0; s < nseg; s++) heads[s] = -1;

    /* Bucket init: for every prime, the offset (relative to lo) of its first
       odd multiple >= lo; primes whose next multiple is beyond the chunk are
       parked with nexti = -1. */
    double ts0 = now_s();
    for (size_t i = 0; i < g_nprimes; i++) {
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
    j->t_sieve += now_s() - ts0;

    /* Previous prime before lo (for the first gap) */
    mpz_t n;
    u128 prev_prime = 0;
    int have_prev = 0;
    if (j->do_test) {
        mpz_init(n);
        u128 pv = lo - 1;
        if (((uint64_t)pv & 1ULL) == 0) pv -= 1;
        while (pv > 2 && !u128_is_probable(pv, n)) pv -= 2;
        prev_prime = pv;
        have_prev = 1;
    }

    for (uint64_t s = 0; s < nseg; s++) {
        const uint64_t seg_lo_off = s * seg;
        const uint64_t seg_len = (seg_lo_off + seg <= span) ? seg : (span - seg_lo_off);
        const uint64_t seg_end = seg_lo_off + seg_len;

        if (seg_len == 0) continue;
        memset(bits, 0, seg_len);
        j->ints += seg_len;

        /* --- sieve marking --- */
        double a0 = now_s();
        for (int32_t i = heads[s]; i != -1; ) {
            int32_t ni = nexti[i];
            uint64_t p = g_primes[i];
            uint64_t off = nextoff[i];
            while (off < seg_end) {
                bits[off - seg_lo_off] = 1;
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
        j->t_sieve += now_s() - a0;

        /* --- survivor walk + primality test --- */
        double b0 = now_s();
        const u128 base = lo + seg_lo_off;
        const uint64_t k0 = ((uint64_t)base & 1ULL) ? 0 : 1;
        for (uint64_t k = k0; k < seg_len; k += 2) {
            if (bits[k]) continue;
            j->survivors++;
            if (!j->do_test) continue;
            u128 x = base + k;
            j->tests++;
            if (u128_is_probable(x, n)) {
                j->primes_found++;
                if (j->collect) {
                    if (j->collect_n >= j->collect_cap) {
                        fprintf(stderr, "[phase0] collect overflow (raise cap)\n");
                        exit(1);
                    }
                    j->collect[j->collect_n++] = x;
                }
                if (have_prev) {
                    uint64_t gap = (uint64_t)(x - prev_prime);
                    double merit = (double)gap / u128_ln(prev_prime);
                    for (int b = 0; b < N_BANDS; b++)
                        if (merit >= k_bands[b]) j->band_cnt[b]++;
                    top_insert(j, x, gap, merit);
                }
                prev_prime = x;
                have_prev = 1;
            }
        }
        j->t_test += now_s() - b0;
    }

    if (j->do_test) mpz_clear(n);
    j->wall = now_s() - w0;
    free(heads); free(nexti); free(nextoff); free(bits);
    return NULL;
}

/* ---------------- --check: full-sequence comparison vs GMP ---------------- */

static int run_check(u128 start, uint64_t len) {
    job_t j;
    memset(&j, 0, sizeof j);
    j.start = start;
    j.end = start + len;
    j.seg_bits = 20;
    j.do_test = 1;

    size_t cap = (size_t)(len / 10) + 4096;
    u128 *collect = (u128 *)malloc(cap * sizeof(u128));
    if (!collect) { fprintf(stderr, "[phase0] check alloc failed\n"); return 1; }
    j.collect = collect;
    j.collect_cap = cap;

    worker(&j);

    /* GMP reference walk from the last prime < start */
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
        if (g != collect[i]) {
            char a[48], b[48];
            u128_str(g, a, sizeof a);
            u128_str(collect[i], b, sizeof b);
            fprintf(stderr, "[phase0] CHECK MISMATCH #%zu: gmp=%s sieve=%s\n", i, a, b);
            mismatch++;
        }
    }

    char s0[48];
    u128_str(start, s0, sizeof s0);
    if (mismatch) {
        fprintf(stderr, "[phase0] CHECK FAILED (%zu mismatches)\n", mismatch);
        free(collect); mpz_clear(gp); mpz_clear(t);
        return 1;
    }
    printf("[phase0] CHECK PASS: %zu primes in [%s, %s+%llu) identical between "
           "bucketed sieve + GMP MR and a pure GMP mpz_nextprime walk\n",
           j.collect_n, s0, s0, (unsigned long long)len);
    free(collect);
    mpz_clear(gp);
    mpz_clear(t);
    return 0;
}

/* ---------------- main ---------------- */

static void usage(const char *p) {
    fprintf(stderr,
        "usage: %s [--start DEC] [--length DEC] [--sieve-limit P] [--threads T]\n"
        "          [--merit-min M] [--seg-bits B] [--no-test] [--check [N]]\n", p);
}

int main(int argc, char **argv) {
    u128 start = (u128)200000000000000000ULL * 1000ULL;  /* 2e20 */
    uint64_t length = 10000000000ULL;
    uint32_t sieve_limit = 100000000u;
    int threads = 4, do_test = 1, seg_bits = 20;
    double merit_min = 20.0;
    int do_check = 0;
    uint64_t check_len = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--start") && i + 1 < argc) start = parse_u128(argv[++i]);
        else if (!strcmp(argv[i], "--length") && i + 1 < argc) length = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--sieve-limit") && i + 1 < argc) sieve_limit = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--merit-min") && i + 1 < argc) merit_min = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seg-bits") && i + 1 < argc) seg_bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-test")) do_test = 0;
        else if (!strcmp(argv[i], "--check")) {
            do_check = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') check_len = strtoull(argv[++i], NULL, 10);
        } else { usage(argv[0]); return 2; }
    }

    if (seg_bits < 16 || seg_bits > 27) { fprintf(stderr, "[phase0] --seg-bits out of range [16,27]\n"); return 2; }
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;

    gen_primes(sieve_limit);

    if (do_check) {
        if (check_len == 0) check_len = 2000000;
        return run_check(start, check_len) ? 1 : 0;
    }

    char s0[48];
    u128_str(start, s0, sizeof s0);
    printf("[phase0] start=%s  length=%llu  ln(start)=%.4f\n",
           s0, (unsigned long long)length, u128_ln(start));
    printf("[phase0] sieve_limit=%u  threads=%d  seg_bits=%d  test=%s  merit_min=%.1f\n",
           sieve_limit, threads, seg_bits, do_test ? "on" : "off", merit_min);

    job_t *jobs = (job_t *)calloc((size_t)threads, sizeof(job_t));
    pthread_t *tid = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    if (!jobs || !tid) { fprintf(stderr, "[phase0] alloc failed\n"); return 1; }

    double t0 = now_s();
    for (int t = 0; t < threads; t++) {
        jobs[t].start = start + (u128)length * (u128)t / (u128)threads;
        jobs[t].end   = start + (u128)length * (u128)(t + 1) / (u128)threads;
        jobs[t].seg_bits = (uint32_t)seg_bits;
        jobs[t].do_test = do_test;
    }
    if (threads == 1) {
        worker(&jobs[0]);
    } else {
        for (int t = 0; t < threads; t++)
            if (pthread_create(&tid[t], NULL, worker, &jobs[t]) != 0) {
                fprintf(stderr, "[phase0] pthread_create failed\n");
                return 1;
            }
        for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    }
    double wall = now_s() - t0;

    /* aggregate */
    uint64_t ints = 0, surv = 0, tests = 0, primes = 0;
    uint64_t bands[N_BANDS] = {0};
    double t_sieve = 0, t_test = 0;
    u128 top_start[4] = {0, 0, 0, 0};
    uint64_t top_gap[4] = {0};
    double top_merit[4] = {0};
    int ntop = 0;
    for (int t = 0; t < threads; t++) {
        ints += jobs[t].ints; surv += jobs[t].survivors;
        tests += jobs[t].tests; primes += jobs[t].primes_found;
        t_sieve += jobs[t].t_sieve; t_test += jobs[t].t_test;
        for (int b = 0; b < N_BANDS; b++) bands[b] += jobs[t].band_cnt[b];
        for (int r = 0; r < jobs[t].ntop; r++) {
            int pos = -1;
            for (int q = 0; q < 4; q++) if (jobs[t].top_merit[r] > top_merit[q]) { pos = q; break; }
            if (pos < 0 && ntop < 4) pos = ntop;
            if (pos < 0) continue;
            for (int q = 3; q > pos; q--) {
                top_merit[q] = top_merit[q - 1];
                top_start[q] = top_start[q - 1];
                top_gap[q]   = top_gap[q - 1];
            }
            top_merit[pos] = jobs[t].top_merit[r];
            top_start[pos] = jobs[t].top_start[r];
            top_gap[pos]   = jobs[t].top_gap[r];
            if (ntop < 4) ntop++;
        }
    }

    double per14 = (ints > 0) ? (1e14 / (double)ints) : 0.0;
    printf("[phase0] --- results ---\n");
    printf("[phase0] wall=%.2f s  ints=%llu  end_to_end=%.3e ints/s\n",
           wall, (unsigned long long)ints, (double)ints / wall);
    printf("[phase0] sieve=%.2f s  sieve_rate=%.3e ints/s  (marking only)\n",
           t_sieve, t_sieve > 0 ? (double)ints / t_sieve : 0.0);
    printf("[phase0] survivors=%llu  u=%.3f%% of ints (%.3f%% of odds)  tests=%llu\n",
           (unsigned long long)surv,
           100.0 * (double)surv / (double)ints,
           100.0 * (double)surv / ((double)ints / 2.0),
           (unsigned long long)tests);
    if (do_test) {
        printf("[phase0] test_time=%.2f s  test_rate=%.3e tests/s  primes=%llu\n",
               t_test, t_test > 0 ? (double)tests / t_test : 0.0,
               (unsigned long long)primes);
        printf("[phase0] gaps per merit threshold (measured slice -> per 1e14 ints):\n");
        for (int b = 0; b < N_BANDS; b++)
            printf("[phase0]   m>=%2.0f : %8llu  ->  %.4g\n",
                   k_bands[b], (unsigned long long)bands[b], (double)bands[b] * per14);
        for (int r = 0; r < ntop; r++) {
            char st[48];
            u128_str(top_start[r], st, sizeof st);
            printf("[phase0]   top#%d gap=%llu merit=%.4f start=%s\n",
                   r + 1, (unsigned long long)top_gap[r], top_merit[r], st);
        }
        /* result-list size estimate: 12 B per record (offset delta + length + class) */
        int bb = 0;
        for (int b = 0; b < N_BANDS; b++) if (k_bands[b] <= merit_min) bb = b;
        printf("[phase0] list_estimate: ~%.1f MB per 1e14 ints at m>=%.0f (12 B/record)\n",
               (double)bands[bb] * per14 * 12.0 / 1e6, k_bands[bb]);
    }
    printf("[phase0] --- end ---\n");

    free(jobs); free(tid); free(g_primes);
    return 0;
}
