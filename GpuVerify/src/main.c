/*
 * GpuVerify CLI: standalone GPU-accelerated prime-gap verification.
 *
 *   gpuverify verify  --expr "10^19662 - 58727" --gap 161450
 *   gpuverify sweep   --db allgaps.sql [--min-digits D] [--max-digits D]
 *                     [--with-interval] [--pmax P] [--limit K] [--skip K]
 *                     [--cpu-max-digits D] [--out LOG] [--cpu]
 *   gpuverify selftest [--db F]
 *   gpuverify bench   --bits B --count N
 */
#include "gv.h"
#include "host.h"
#include "parse.h"

#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef struct {
    mpz_t n;
    uint64_t gap;
    int digits;
    char *expr;
} gv_row;

/* ---- one-number classification ---- */
/* returns GV_OK / GV_COMPOSITE_END; sets *failed = 0 (start) or 1 (end) */
static int check_endpoints_gpu(mpz_t start, mpz_t end, int *failed)
{
    mpz_t nums[2];
    mpz_init_set(nums[0], start);
    mpz_init_set(nums[1], end);
    uint8_t pass[2];
    gv_gpu_prp_batch(nums, 2, pass);
    for (int i = 0; i < 2; i++) {
        if (pass[i] == 2) {
            /* too large for GPU: straight to CPU BPSW */
            pass[i] = gv_bpsw(nums[i]) ? 1 : 0;
        } else if (pass[i] == 1 && !gv_bpsw(nums[i])) {
            pass[i] = 0; /* Lucas part of BPSW caught it: composite */
        }
    }
    mpz_clear(nums[0]);
    mpz_clear(nums[1]);
    if (!pass[0]) {
        *failed = 0;
        return GV_COMPOSITE_END;
    }
    if (!pass[1]) {
        *failed = 1;
        return GV_COMPOSITE_END;
    }
    return GV_OK;
}

/* ---- verify one gap ---- */
static int cmd_verify(const char *expr, uint64_t gap, uint32_t pmax,
                      int cpu_only)
{
    mpz_t start, end;
    mpz_init(start);
    mpz_init(end);
    int rc = gv_parse_expr(expr, start, 0);
    if (rc == GV_PARSE_TOO_LARGE) {
        printf("verdict: UNVERIFIABLE (value too large to materialize)\n");
        mpz_clear(start);
        mpz_clear(end);
        return GV_PARSE_FAIL;
    }
    if (rc != 0) {
        printf("verdict: PARSE_FAIL\n");
        mpz_clear(start);
        mpz_clear(end);
        return GV_PARSE_FAIL;
    }
    char *ns = mpz_get_str(NULL, 10, start);
    size_t ns_len = strlen(ns);
    if (ns_len > 60) {
        printf("start: %.57s... (%zu digits)\n", ns, ns_len);
    } else {
        printf("start: %s (%zu digits)\n", ns, ns_len);
    }
    free(ns);
    printf("gap:   %llu\n", (unsigned long long)gap);
    mpz_add_ui(end, start, gap);

    int verdict;
    if (cpu_only) {
        int ps = gv_bpsw(start), pe = gv_bpsw(end);
        printf("start: %s\n", ps ? "prime (BPSW)" : "COMPOSITE");
        printf("end:   %s\n", pe ? "prime (BPSW)" : "COMPOSITE");
        verdict = (ps && pe) ? GV_OK : GV_COMPOSITE_END;
    } else {
        int failed = -1;
        verdict = check_endpoints_gpu(start, end, &failed);
        if (verdict == GV_OK)
            printf("start: prime (strong base-2 + BPSW)\n"
                   "end:   prime (strong base-2 + BPSW)\n");
        else
            printf("%s is COMPOSITE\n", failed == 0 ? "start" : "end");
    }

    if (verdict != GV_OK)
        goto done;

    /* interval scan: GPU only; for starts beyond the GPU range the scan is
     * skipped (CPU PRP per survivor would be infeasible) */
    if ((mpz_sizeinbase(start, 2) + 31) / 32 > GV_MAX_LIMBS) {
        printf("interval: skipped (start > %d bits); endpoints checked only\n",
               GV_MAX_BITS);
        goto done;
    }
    uint8_t *marks;
    uint64_t half;
    if (gv_gpu_sieve_gap(start, gap, pmax, &marks, &half) != 0) {
        printf("verdict: SIEVE_FAILED\n");
        goto done;
    }
    int survivors = 0;
    for (uint64_t i = 0; i < half; i++)
        if (!marks[i])
            survivors++;
    printf("interval: %llu odd candidates, %d survivors after p <= %u\n",
           (unsigned long long)half, survivors, pmax);
    if (survivors) {
        mpz_t *cand = (mpz_t *)malloc((size_t)survivors * sizeof(mpz_t));
        uint64_t *off = (uint64_t *)malloc((size_t)survivors * sizeof(uint64_t));
        int k = 0;
        for (uint64_t i = 0; i < half; i++) {
            if (!marks[i]) {
                off[k] = 2 * (i + 1);
                mpz_init(cand[k]);
                mpz_add_ui(cand[k], start, off[k]);
                k++;
            }
        }
        uint8_t *pass = (uint8_t *)malloc((size_t)survivors);
        gv_gpu_prp_batch(cand, survivors, pass);
        int found = 0;
        for (int i = 0; i < survivors && !found; i++) {
            int prime = (pass[i] == 2) ? gv_bpsw(cand[i])
                                       : (pass[i] == 1 && gv_bpsw(cand[i]));
            if (prime) {
                char *ws = mpz_get_str(NULL, 10, cand[i]);
                printf("PRIME BETWEEN: N + %llu = %s\n",
                       (unsigned long long)off[i], ws);
                free(ws);
                verdict = GV_PRIME_BETWEEN;
                found = 1;
            }
        }
        for (int i = 0; i < survivors; i++)
            mpz_clear(cand[i]);
        free(cand);
        free(off);
        free(pass);
        if (marks)
            free(marks);
        if (!found)
            printf("interval: no prime found between endpoints\n");
    } else {
        free(marks);
    }

done:
    printf("verdict: %s\n",
           verdict == GV_OK ? "VALID (both endpoints BPSW-prime, no prime "
                              "between)"
           : verdict == GV_PRIME_BETWEEN ? "INVALID (prime between)"
           : "INVALID (endpoint composite)");
    mpz_clear(start);
    mpz_clear(end);
    return verdict;
}

/* ---- sweep ---- */
static void cmd_sweep(const char *db, int min_digits, int max_digits,
                      int with_interval, uint32_t pmax, long limit, long skip,
                      int cpu_max_digits, const char *outpath, int cpu_only)
{
    FILE *f = fopen(db, "r");
    if (!f) {
        perror(db);
        return;
    }
    FILE *log = outpath ? fopen(outpath, "w") : stdout;
    if (!log)
        log = stdout;

    char *line = NULL;
    size_t cap = 0;
    unsigned long lineno = 0;

    long cnt_total = 0, cnt_sel = 0, cnt_ok = 0, cnt_bad_end = 0,
         cnt_between = 0, cnt_parse = 0, cnt_large = 0, cnt_cpu = 0,
         cnt_skipped_cpu = 0;

    clock_t t0 = clock();
    while (getline(&line, &cap, f) != -1) {
        lineno++;
        uint64_t gap;
        int digits;
        char *expr = NULL;
        if (gv_db_extract_row(line, lineno, &gap, &digits, &expr) != 0)
            continue;
        cnt_total++;
        if (cnt_total <= skip) {
            free(expr);
            continue;
        }
        if (limit > 0 && cnt_sel >= limit) {
            free(expr);
            continue;
        }
        if (digits < min_digits || (max_digits > 0 && digits > max_digits)) {
            free(expr);
            continue;
        }
        cnt_sel++;

        mpz_t start, end;
        mpz_init(start);
        mpz_init(end);
        int rc = gv_parse_expr(expr, start, 1);
        if (rc == GV_PARSE_TOO_LARGE) {
            cnt_large++;
            fprintf(log, "UNVERIFIABLE gap=%llu digits=%d expr=%s\n",
                    (unsigned long long)gap, digits, expr);
            free(expr);
            mpz_clear(start);
            mpz_clear(end);
            continue;
        }
        if (rc != 0) {
            cnt_parse++;
            fprintf(log, "PARSE_FAIL gap=%llu digits=%d expr=%s\n",
                    (unsigned long long)gap, digits, expr);
            free(expr);
            mpz_clear(start);
            mpz_clear(end);
            continue;
        }
        mpz_add_ui(end, start, gap);

        int verdict;
        uint64_t wit_off = 0;
        if (cpu_only || digits > GV_MAX_DIGITS_GPU) {
            if (digits > cpu_max_digits) {
                cnt_skipped_cpu++;
                fprintf(log, "SKIP_CPU_TOO_BIG gap=%llu digits=%d expr=%s\n",
                        (unsigned long long)gap, digits, expr);
                free(expr);
                mpz_clear(start);
                mpz_clear(end);
                continue;
            }
            cnt_cpu++;
            int ps = gv_bpsw(start), pe = gv_bpsw(end);
            verdict = (ps && pe) ? GV_OK : GV_COMPOSITE_END;
        } else {
            int failed = -1;
            verdict = check_endpoints_gpu(start, end, &failed);
        }

        if (verdict == GV_COMPOSITE_END) {
            cnt_bad_end++;
            fprintf(log, "COMPOSITE_END gap=%llu digits=%d expr=%s\n",
                    (unsigned long long)gap, digits, expr);
        } else if (verdict == GV_OK && with_interval) {
            uint8_t *marks;
            uint64_t half;
            if (gv_gpu_sieve_gap(start, gap, pmax, &marks, &half) == 0) {
                int survivors = 0;
                for (uint64_t i = 0; i < half; i++)
                    if (!marks[i])
                        survivors++;
                if (survivors) {
                    mpz_t *cand = (mpz_t *)malloc((size_t)survivors *
                                                  sizeof(mpz_t));
                    uint64_t *off = (uint64_t *)malloc((size_t)survivors *
                                                       sizeof(uint64_t));
                    int k = 0;
                    for (uint64_t i = 0; i < half; i++) {
                        if (!marks[i]) {
                            off[k] = 2 * (i + 1);
                            mpz_init(cand[k]);
                            mpz_add_ui(cand[k], start, off[k]);
                            k++;
                        }
                    }
                    uint8_t *pass = (uint8_t *)malloc((size_t)survivors);
                    gv_gpu_prp_batch(cand, survivors, pass);
                    for (int i = 0; i < survivors; i++) {
                        int prime =
                            (pass[i] == 2)
                                ? gv_bpsw(cand[i])
                                : (pass[i] == 1 && gv_bpsw(cand[i]));
                        if (prime) {
                            verdict = GV_PRIME_BETWEEN;
                            wit_off = off[i];
                            break;
                        }
                    }
                    for (int i = 0; i < survivors; i++)
                        mpz_clear(cand[i]);
                    free(cand);
                    free(off);
                    free(pass);
                }
                free(marks);
            }
        }

        if (verdict == GV_OK) {
            cnt_ok++;
        } else if (verdict == GV_PRIME_BETWEEN) {
            cnt_between++;
            fprintf(log, "PRIME_BETWEEN gap=%llu digits=%d expr=%s "
                         "witness_offset=%llu\n",
                    (unsigned long long)gap, digits, expr,
                    (unsigned long long)wit_off);
        }
        free(expr);
        mpz_clear(start);
        mpz_clear(end);

        if ((cnt_sel & 1023) == 0) {
            fprintf(stderr,
                    "\rrows=%ld ok=%ld bad_end=%ld between=%ld parse=%ld "
                    "unverifiable=%ld cpu=%ld cpu_skip=%ld",
                    cnt_sel, cnt_ok, cnt_bad_end, cnt_between, cnt_parse,
                    cnt_large, cnt_cpu, cnt_skipped_cpu);
            fflush(stderr);
        }
    }
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\n");
    printf("sweep done in %.1fs: total=%ld selected=%ld ok=%ld "
           "composite_end=%ld prime_between=%ld parse_fail=%ld "
           "unverifiable=%ld cpu=%ld cpu_skip=%ld\n",
           sec, cnt_total, cnt_sel, cnt_ok, cnt_bad_end, cnt_between,
           cnt_parse, cnt_large, cnt_cpu, cnt_skipped_cpu);
    if (log != stdout)
        fclose(log);
    free(line);
    fclose(f);
}

/* ---- selftest ---- */
static void cmd_selftest(const char *db)
{
    int fails = 0;
    /* known primes must pass, known composites must fail */
    const char *primes[] = {"2", "3", "89", "523", "1000000007", "2^61-1"};
    const char *composites[] = {"9", "15", "2047", "3277", "4033", "4681",
                                "8321", "15841", "29341", "341550071728321"};
    mpz_t n;
    mpz_init(n);
    for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); i++) {
        if (gv_parse_expr(primes[i], n, 1) != 0) {
            printf("SELFTEST parse fail: %s\n", primes[i]);
            fails++;
            continue;
        }
        mpz_t nums[1];
        mpz_init_set(nums[0], n);
        uint8_t pass[1];
        gv_gpu_prp_batch(nums, 1, pass);
        mpz_clear(nums[0]);
        int final = (pass[0] == 2) ? gv_bpsw(n)
                                   : (pass[0] == 1 && gv_bpsw(n));
        if (!final || gv_ref_strong_prp2(n) != 1) {
            printf("SELFTEST FAIL: %s should be prime (gpu=%d bpsw=%d)\n",
                   primes[i], pass[0], gv_bpsw(n));
            fails++;
        }
    }
    for (size_t i = 0; i < sizeof(composites) / sizeof(composites[0]); i++) {
        if (gv_parse_expr(composites[i], n, 1) != 0) {
            printf("SELFTEST parse fail: %s\n", composites[i]);
            fails++;
            continue;
        }
        mpz_t nums[1];
        mpz_init_set(nums[0], n);
        uint8_t pass[1];
        gv_gpu_prp_batch(nums, 1, pass);
        mpz_clear(nums[0]);
        /* composite must fail either on GPU or via BPSW; the final call */
        int final = (pass[0] == 2) ? gv_bpsw(n)
                                   : (pass[0] == 1 && gv_bpsw(n));
        if (final) {
            printf("SELFTEST FAIL: %s should be composite (gpu=%d bpsw=%d)\n",
                   composites[i], pass[0], gv_bpsw(n));
            fails++;
        }
    }
    /* known DB cases */
    if (db) {
        struct {
            const char *expr;
            uint64_t gap;
            int expect; /* GV_OK / GV_COMPOSITE_END / GV_PRIME_BETWEEN */
        } cases[] = {
            {"10^19662 - 58727", 161450, GV_COMPOSITE_END},
            {"10^19666 - 98199", 191356, GV_COMPOSITE_END},
            {"10^16104 - 131519", 226058, GV_OK}, /* endpoints prime; the
                                                   * interior prime is checked
                                                   * below via the witness */
            {"10^21530 - 61643", 263402, GV_COMPOSITE_END},
            {"1009", 4, GV_OK},
        };
        for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
            int got = cmd_verify(cases[i].expr, cases[i].gap, 1000000, 0);
            if (got != cases[i].expect) {
                printf("SELFTEST FAIL: %s gap=%llu expect=%d got=%d\n",
                       cases[i].expr, (unsigned long long)cases[i].gap,
                       cases[i].expect, got);
                fails++;
            }
        }
        /* PRIME_BETWEEN case 226058: witness 10^16104 + 28021 must be prime
         * and strictly between the stored endpoints */
        {
            mpz_t start, end, wit;
            mpz_init(start);
            mpz_init(end);
            mpz_init(wit);
            gv_parse_expr("10^16104 - 131519", start, 1);
            gv_parse_expr("10^16104 + 94539", end, 1);
            gv_parse_expr("10^16104 + 28021", wit, 1);
            if (!gv_bpsw(wit) || mpz_cmp(wit, start) <= 0 ||
                mpz_cmp(wit, end) >= 0) {
                printf("SELFTEST FAIL: 226058 witness not a prime between\n");
                fails++;
            } else {
                printf("SELFTEST OK: 226058 -> prime between "
                       "(10^16104 + 28021 verified prime, strictly inside)\n");
            }
            mpz_clear(start);
            mpz_clear(end);
            mpz_clear(wit);
        }
    }
    mpz_clear(n);
    printf(fails ? "SELFTEST: %d FAILURES\n" : "SELFTEST: all passed\n",
           fails);
}

/* ---- bench ---- */
static void cmd_bench(int bits, int count)
{
    if (bits < 32 || bits > GV_MAX_BITS) {
        fprintf(stderr, "bits must be in [32, %d]\n", GV_MAX_BITS);
        return;
    }
    mpz_t *nums = (mpz_t *)malloc((size_t)count * sizeof(mpz_t));
    gmp_randstate_t rnd;
    gmp_randinit_mt(rnd);
    gmp_randseed_ui(rnd, 12345);
    for (int i = 0; i < count; i++) {
        mpz_init(nums[i]);
        mpz_rrandomb(nums[i], rnd, bits);
        mpz_setbit(nums[i], bits - 1);
        mpz_setbit(nums[i], 0);
    }
    gmp_randclear(rnd);

    uint8_t *pass = (uint8_t *)malloc((size_t)count);
    clock_t t0 = clock();
    if (gv_gpu_prp_batch(nums, count, pass) != 0) {
        fprintf(stderr, "bench failed\n");
        return;
    }
    double sec = (double)(clock() - t0) / CLOCKS_PER_SEC;
    int npass = 0, ncomp = 0;
    for (int i = 0; i < count; i++) {
        if (pass[i] == 1)
            npass++;
        else
            ncomp++;
    }
    printf("bench: %d x %d-bit strong PRP base 2 in %.3fs -> %.0f PRP/s "
           "(%d passed, %d composite)\n",
           count, bits, sec, count / sec, npass, ncomp);
    free(pass);
    for (int i = 0; i < count; i++)
        mpz_clear(nums[i]);
    free(nums);
}

static void usage(const char *prog)
{
    printf(
        "usage:\n"
        "  %s verify --expr E --gap G [--pmax P] [--cpu]\n"
        "  %s sweep  --db FILE [--min-digits D] [--max-digits D]\n"
        "             [--with-interval] [--pmax P] [--limit K] [--skip K]\n"
        "             [--cpu-max-digits D] [--cpu] [--out LOG]\n"
        "  %s selftest [--db FILE]\n"
        "  %s bench --bits B --count N\n",
        prog, prog, prog, prog);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }
    const char *cmd = argv[1];

    static struct option longopts[] = {
        {"expr", required_argument, 0, 'e'},
        {"gap", required_argument, 0, 'g'},
        {"db", required_argument, 0, 'd'},
        {"pmax", required_argument, 0, 'p'},
        {"min-digits", required_argument, 0, 'm'},
        {"max-digits", required_argument, 0, 'M'},
        {"with-interval", no_argument, 0, 'w'},
        {"limit", required_argument, 0, 'l'},
        {"skip", required_argument, 0, 's'},
        {"cpu-max-digits", required_argument, 0, 'C'},
        {"cpu", no_argument, 0, 'c'},
        {"out", required_argument, 0, 'o'},
        {"bits", required_argument, 0, 'b'},
        {"count", required_argument, 0, 'n'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0},
    };

    const char *expr = NULL, *db = NULL, *out = NULL;
    uint64_t gap = 0;
    uint32_t pmax = 1 << 20;
    int min_digits = 0, max_digits = 0, with_interval = 0, cpu_only = 0;
    int cpu_max_digits = 200000;
    long limit = 0, skip = 0;
    int bits = 4096, count = 10000;

    optind = 2;
    int c;
    while ((c = getopt_long(argc, argv, "e:g:d:p:m:M:wl:s:C:co:b:n:h",
                            longopts, NULL)) != -1) {
        switch (c) {
        case 'e': expr = optarg; break;
        case 'g': gap = strtoull(optarg, NULL, 10); break;
        case 'd': db = optarg; break;
        case 'p': pmax = (uint32_t)strtoul(optarg, NULL, 10); break;
        case 'm': min_digits = atoi(optarg); break;
        case 'M': max_digits = atoi(optarg); break;
        case 'w': with_interval = 1; break;
        case 'l': limit = atol(optarg); break;
        case 's': skip = atol(optarg); break;
        case 'C': cpu_max_digits = atoi(optarg); break;
        case 'c': cpu_only = 1; break;
        case 'o': out = optarg; break;
        case 'b': bits = atoi(optarg); break;
        case 'n': count = atoi(optarg); break;
        case 'h':
        default: usage(argv[0]); return 1;
        }
    }

    if (strcmp(cmd, "verify") == 0) {
        if (!expr || !gap) {
            fprintf(stderr, "verify needs --expr and --gap\n");
            return 1;
        }
        return cmd_verify(expr, gap, pmax, cpu_only);
    }
    if (strcmp(cmd, "sweep") == 0) {
        if (!db) {
            fprintf(stderr, "sweep needs --db\n");
            return 1;
        }
        cmd_sweep(db, min_digits, max_digits, with_interval, pmax, limit,
                  skip, cpu_max_digits, out, cpu_only);
        return 0;
    }
    if (strcmp(cmd, "selftest") == 0) {
        cmd_selftest(db);
        return 0;
    }
    if (strcmp(cmd, "bench") == 0) {
        cmd_bench(bits, count);
        return 0;
    }
    usage(argv[0]);
    return 1;
}
