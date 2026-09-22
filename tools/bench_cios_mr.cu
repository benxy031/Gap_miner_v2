/* tools/bench_cios_mr.cu - can a non-shuffle 32-bit CIOS kernel beat CGBN?
 *
 * WHY THIS EXISTS
 *   CGBN TPI=8 (what we run) spreads ONE candidate across 8 lanes, which makes
 *   every step a warp-cooperative shuffle chain with barriers.  Measured baseline
 *   on this box: bin/bench_fermat 12 40000 20 -> 1,387,762 candidates/s
 *   (0.721 us per 768-bit base-2 Miller-Rabin candidate).
 *   The measured instruction budget says that is NOT a compute limit: 768
 *   squarings x 576 32-bit products = ~442k products per candidate, i.e. ~12% of
 *   the SM's IMAD rate and ~3.5% of its issue slots.  So the wall is the *shape*:
 *   ~8 lanes cooperate per candidate (few independent chains per SM) and every
 *   squaring is a barrier-separated shuffle chain.
 *
 * THIS KERNEL tests the opposite shape: 1 thread = 1 candidate, all 24 limbs in
 * registers, no shuffles, no barriers, no shared memory.  A warp then holds 32
 * independent candidates in lockstep instead of 4, and the SM's occupancy
 * (limited only by registers) sets how many chains are in flight.
 *   This is the design mr_blackwell uses for Blackwell; it was never measured on
 *   Ampere against CGBN.  It replaces an assumption with a number.
 *
 * HONESTY RULES FOLLOWED HERE
 *   - Correctness FIRST: the MR verdict is compared against GMP on the host for
 *     20 real primes (mpz_nextprime) and 44 random composites.  Timing is only
 *     reported if the verdicts match 100%; a fast wrong Montgomery is worthless.
 *   - Same workload as bench_fermat: one base-2 MR per candidate, random odd
 *     768-bit inputs, so cand/s is directly comparable.
 *   - Roofline printed: achieved IMAD/s vs the SM's issue capability, so the
 *     reader can see how far from the limit either kernel actually is.
 *
 * Build: make bin/bench_cios_mr WITH_CUDA=1
 * Run:   ./bin/bench_cios_mr [candidates=200000] [threads=128]
 *        (needs -lgmp; the Makefile's tools/%.cu rule may need -lgmp added)
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <gmp.h>
#include <cuda_runtime.h>

#define NL 24                 /* 768-bit = 24 x 32-bit limbs */
#define MAXCAND 400000

static void ck(const char *what, cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

/* ---------------------------------------------------------------- device */
/* CIOS Montgomery multiplication, 24 limbs, all in registers.
   n0inv = -n^{-1} mod 2^32.  r = a*b*R^{-1} mod n  (result < n). */
__device__ __forceinline__ static void mont_mul(uint32_t *r, const uint32_t *a,
                                                const uint32_t *b,
                                                const uint32_t *n, uint32_t n0inv) {
    uint32_t t[NL + 2];
#pragma unroll
    for (int i = 0; i < NL + 2; i++) t[i] = 0U;

#pragma unroll
    for (int i = 0; i < NL; i++) {
        uint64_t c = 0;
        const uint64_t bi = (uint64_t)b[i];
#pragma unroll
        for (int j = 0; j < NL; j++) {
            uint64_t p = (uint64_t)a[j] * bi + (uint64_t)t[j] + c;
            t[j] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s = (uint64_t)t[NL] + c;
        uint64_t s2 = (uint64_t)t[NL + 1] + (s >> 32);
        t[NL] = (uint32_t)s;
        t[NL + 1] = (uint32_t)s2;

        uint32_t m = t[0] * n0inv;                  /* mod 2^32 by truncation */
        c = ((uint64_t)t[0] + (uint64_t)m * (uint64_t)n[0]) >> 32;
#pragma unroll
        for (int j = 1; j < NL; j++) {
            uint64_t p = (uint64_t)m * (uint64_t)n[j] + (uint64_t)t[j] + c;
            t[j - 1] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s3 = (uint64_t)t[NL] + c;
        t[NL - 1] = (uint32_t)s3;
        t[NL] = (uint32_t)((uint64_t)t[NL + 1] + (s3 >> 32));
        t[NL + 1] = 0U;
    }

    /* conditional subtract: result may be in [n, 2n) */
    uint32_t borrow = 0;
    uint32_t sub[NL];
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (t[NL] != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < NL; j++) r[j] = need ? sub[j] : t[j];
}

/* r = 2*a mod n  (expensive-free: one add + conditional subtract) */
__device__ __forceinline__ static void dbl_mod(uint32_t *r, const uint32_t *a,
                                               const uint32_t *n) {
    uint32_t c = 0, t[NL];
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint64_t s = (uint64_t)a[j] * 2ULL + (uint64_t)c;
        t[j] = (uint32_t)s;
        c = (uint32_t)(s >> 32);
    }
    uint32_t borrow = 0, sub[NL];
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (c != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < NL; j++) r[j] = need ? sub[j] : t[j];
}

/* base-2 Miller-Rabin, one candidate per thread.  0 = composite, 1 = probable
   prime (this is a strong-pseudoprime test to base 2; the production pipeline
   runs BPSW afterwards, exactly as here). 
   n-1 = 2^s * d.  y = 2^d (Montgomery form), then up to s-1 squarings looking
   for -1.  ONE = R mod n = 2^768 - n (n has the top bit set, so 2^768 < 2n).
   NEGONE = n - ONE.  TWO = 2*ONE mod n, so "multiply by base 2" == doubling. */
__device__ static int mr_base2(const uint32_t *nin) {
    uint32_t n[NL], one[NL], neg[NL], two[NL], y[NL], acc[NL], tmp[NL];
    uint32_t n0inv = 1U;                  /* Newton: n0inv = -n^-1 mod 2^32 */
#pragma unroll
    for (int k = 0; k < 5; k++) n0inv = n0inv * (2U - nin[0] * n0inv);
    n0inv = (uint32_t)(0U - n0inv);
#pragma unroll
    for (int j = 0; j < NL; j++) n[j] = nin[j];

    /* ONE = R mod n = 2^768 mod n, correct for EVERY n < 2^768.
       Fast exact route: with b = bitlength(n) we have 2^(b-1) <= n < 2^b, so
       R0 = 2^b mod n = 2^b - n (one 24-limb subtract, no borrow ambiguity), and
       then only (768 - b) doublings are needed.  Production candidates are
       763-bit -> 5 doublings instead of 768 (the naive version cost ~7%).
       b == 768 falls out of the same code path as "2^768 - n" with 0 doublings. */
    int b = 0;
#pragma unroll
    for (int j = NL - 1; j >= 0; j--) {
        if (b == 0 && n[j] != 0U) b = j * 32 + (32 - __clz(n[j]));
    }
    {
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < NL; j++) {
            uint32_t v = (b < 768 && j == (b >> 5)) ? (1U << (b & 31)) : 0U;
            uint64_t d = (uint64_t)v - (uint64_t)n[j] - (uint64_t)borrow;
            one[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
#pragma unroll 1
    for (int i = b; i < 768; i++) dbl_mod(one, one, n);
    {   /* neg = n - one */
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < NL; j++) {
            uint64_t d = (uint64_t)n[j] - (uint64_t)one[j] - (uint64_t)borrow;
            neg[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
    dbl_mod(two, one, n);

    /* exponent d = (n-1) >> s ; s = ctz(n-1) = ctz(n[0]-1 low bits) */
    uint32_t nm1[NL];
    {
        uint32_t borrow = 0;
        uint64_t d = (uint64_t)n[0] - 1ULL;
        nm1[0] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
#pragma unroll
        for (int j = 1; j < NL; j++) {
            uint64_t dd = (uint64_t)n[j] - (uint64_t)borrow;
            nm1[j] = (uint32_t)dd;
            borrow = (uint32_t)((dd >> 32) & 1ULL);
        }
    }
    int s = 0;
    {
        int j = 0;
        while (j < NL && nm1[j] == 0U) { s += 32; j++; }
        if (j < NL) {
            uint32_t w = nm1[j];
            while ((w & 1U) == 0U) { w >>= 1; s++; }
        }
    }
    uint32_t dsh[NL];
    {
        int sh = s & 31, wi = s >> 5;
#pragma unroll
        for (int j = 0; j < NL; j++) {
            uint32_t lo = (j + wi < NL) ? nm1[j + wi] : 0U;
            uint32_t hi = (j + wi + 1 < NL) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint32_t)((lo >> sh) | (hi << (32 - sh)));
        }
    }
    int dbits = 768 - s;                    /* bit length of d */
    while (dbits > 0 && ((dsh[(dbits - 1) >> 5] >> ((dbits - 1) & 31)) & 1U) == 0U)
        dbits--;

#pragma unroll 1
    for (int j = 0; j < NL; j++) y[j] = two[j];
    for (int b = dbits - 2; b >= 0; b--) {
        mont_mul(tmp, y, y, n, n0inv);
#pragma unroll
        for (int j = 0; j < NL; j++) y[j] = tmp[j];
        if ((dsh[b >> 5] >> (b & 31)) & 1U) dbl_mod(y, y, n);
    }
    /* y == one  -> pass ; y == neg -> pass ; else square and look for neg */
    int is_one = 1, is_neg = 1;
#pragma unroll
    for (int j = 0; j < NL; j++) {
        if (y[j] != one[j]) is_one = 0;
        if (y[j] != neg[j]) is_neg = 0;
    }
    if (is_one || is_neg) return 1;
    for (int r = 0; r < s - 1; r++) {
        mont_mul(tmp, y, y, n, n0inv);
#pragma unroll
        for (int j = 0; j < NL; j++) y[j] = tmp[j];
#pragma unroll
        for (int j = 0; j < NL; j++) acc[j] = y[j];
        int got_neg = 1, got_one = 1;
#pragma unroll
        for (int j = 0; j < NL; j++) {
            if (acc[j] != neg[j]) got_neg = 0;
            if (acc[j] != one[j]) got_one = 0;
        }
        if (got_neg) return 1;
        if (got_one) return 0;
    }
    return 0;
}

__global__ void mr_kernel(const uint32_t *cands, int n_cand, uint8_t *res) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < n_cand; i += stride) res[i] = (uint8_t)mr_base2(cands + (size_t)i * NL);
}

/* ---------------------------------------------------------------- host */
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void mpz_to_limbs(uint32_t *out, const mpz_t x) {
    memset(out, 0, NL * sizeof(uint32_t));
    size_t cnt = 0;
    mpz_export(out, &cnt, -1, sizeof(uint32_t), 0, 0, x);   /* little endian */
}

#define CGBN_BASELINE 1387762.0    /* measured: bin/bench_fermat 12 40000 20 */

int main(int argc, char **argv) {
    int n_cand = (argc > 1) ? atoi(argv[1]) : 200000;
    int tpb = (argc > 2) ? atoi(argv[2]) : 128;
    if (n_cand > MAXCAND) n_cand = MAXCAND;

    printf("32-bit CIOS Miller-Rabin, 1 thread = 1 candidate, %d limbs (768-bit)\n",
           NL);
    printf("  baseline to beat: CGBN TPI=8 = %.0f cand/s (measured today)\n",
           CGBN_BASELINE);

    uint32_t *h_cand = (uint32_t *)malloc((size_t)n_cand * NL * sizeof(uint32_t));
    uint8_t *h_res = (uint8_t *)malloc(n_cand);
    uint8_t *h_ref = (uint8_t *)malloc(n_cand);
    if (!h_cand || !h_res || !h_ref) { fprintf(stderr, "host OOM\n"); return 1; }

    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 20260922u);
    mpz_t x;
    mpz_init(x);
    const int N_PRIME = 20, N_COMP = 44;
    for (int i = 0; i < n_cand; i++) {
        if (i < N_PRIME) {                       /* real primes (GMP nextprime) */
            mpz_urandomb(x, rs, 767);
            mpz_setbit(x, 766);
            mpz_nextprime(x, x);
            if (mpz_sizeinbase(x, 2) < 768) mpz_setbit(x, 767);
        } else if (i < N_PRIME + N_COMP) {
            /* odd composites, HALF of them with the top bit CLEAR: that is the
               production shape (763-bit candidates in a 768-bit stride), and
               it is exactly what a wrong ONE = 2^768 - n assumption breaks. */
            if ((i & 1) == 0) {
                mpz_urandomb(x, rs, 768);
                mpz_setbit(x, 767);
                mpz_setbit(x, 0);
            } else {
                mpz_urandomb(x, rs, 762);
                mpz_setbit(x, 761);
                mpz_setbit(x, 0);
            }
        } else {                                 /* benchmark population */
            mpz_urandomb(x, rs, 768);
            mpz_setbit(x, 767);
            mpz_setbit(x, 0);
        }
        mpz_to_limbs(h_cand + (size_t)i * NL, x);
        h_ref[i] = (uint8_t)(mpz_probab_prime_p(x, 20) > 0 ? 1 : 0);
    }
    mpz_clear(x);
    gmp_randclear(rs);

    uint32_t *d_cand;
    uint8_t *d_res;
    ck("alloc", cudaMalloc(&d_cand, (size_t)n_cand * NL * sizeof(uint32_t)));
    ck("alloc2", cudaMalloc(&d_res, n_cand));
    ck("copy", cudaMemcpy(d_cand, h_cand, (size_t)n_cand * NL * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int blocks = 0;
    ck("occ", cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, mr_kernel,
                                                            tpb, 0));
    cudaDeviceProp p;
    ck("prop", cudaGetDeviceProperties(&p, 0));
    int grid = p.multiProcessorCount * blocks;
    printf("  launch: %d blocks x %d threads, %d blocks/SM (%d warps/SM), grid=%d\n",
           grid, tpb, blocks, blocks * tpb / 32, grid);

    mr_kernel<<<grid, tpb>>>(d_cand, n_cand, d_res);
    ck("warmup", cudaDeviceSynchronize());
    ck("copyback", cudaMemcpy(h_res, d_res, n_cand, cudaMemcpyDeviceToHost));

    int mismatch = 0;
    for (int i = 0; i < N_PRIME + N_COMP; i++) {
        if (h_res[i] != h_ref[i]) {
            if (mismatch < 5)
                printf("  MISMATCH cand %d: kernel=%u gmp=%u (%s)\n", i, h_res[i],
                       h_ref[i], i < N_PRIME ? "prime" : "composite");
            mismatch++;
        }
    }
    printf("  correctness: %d primes + %d composites, %d mismatches -> %s\n",
           N_PRIME, N_COMP, mismatch, mismatch ? "FAILED (no timing)" : "PASSED");
    if (mismatch) return 1;

    /* batch <= 0 : one grid-stride launch over everything (max throughput)
       batch  > 0 : repeated small launches of `batch` candidates, which is the
                    production regime (the chain runs MR batches of 1-4k, not
                    40k) -> this is the number that decides the real win.
       streams > 1: mirror the 8-worker production shape: `streams` concurrent
                    small-batch launch chains in ONE CUDA context (separate
                    PROCESSES are NOT a valid model - 8 contexts thrash). */
    int batch = (argc > 3) ? atoi(argv[3]) : 0;
    int streams = (argc > 4) ? atoi(argv[4]) : 1;
    if (streams > 16) streams = 16;
    if (streams < 1) streams = 1;
    int n_timed = n_cand;
    double t0 = now_s();
    if (batch <= 0) {
        mr_kernel<<<grid, tpb>>>(d_cand, n_cand, d_res);
        ck("run", cudaDeviceSynchronize());
    } else if (streams == 1) {
        int iters = n_cand / batch;
        int bg = (batch + tpb - 1) / tpb;
        for (int it = 0; it < iters; it++) {
            mr_kernel<<<bg, tpb>>>(d_cand + (size_t)it * batch * NL, batch,
                                   d_res + (size_t)it * batch);
        }
        ck("run", cudaDeviceSynchronize());
        n_timed = iters * batch;
    } else {
        cudaStream_t st[16];
        for (int s = 0; s < streams; s++) ck("stream", cudaStreamCreate(&st[s]));
        int per_stream = n_cand / streams;
        int iters = per_stream / batch;
        int bg = (batch + tpb - 1) / tpb;
        for (int s = 0; s < streams; s++)
            for (int it = 0; it < iters; it++) {
                size_t off = ((size_t)s * iters + it) * batch;
                mr_kernel<<<bg, tpb, 0, st[s]>>>(d_cand + off * NL, batch,
                                                 d_res + off);
            }
        for (int s = 0; s < streams; s++) {
            ck("sync", cudaStreamSynchronize(st[s]));
            cudaStreamDestroy(st[s]);
        }
        n_timed = streams * iters * batch;
    }
    double wall = now_s() - t0;

    double cand_per_s = (double)n_timed / wall;
    double us_per_cand = wall * 1e6 / (double)n_timed;
    printf("  timing: %d candidates in %.1f ms -> %.0f cand/s (%.3f us/cand)%s\n",
           n_timed, wall * 1e3, cand_per_s, us_per_cand,
           batch > 0 ? "  [small-batch mode]" : "");
    printf("  vs CGBN %.0f cand/s -> %.2fx %s\n", CGBN_BASELINE,
           cand_per_s / CGBN_BASELINE,
           cand_per_s > CGBN_BASELINE ? "FASTER" : "slower");

    /* roofline: how full is the machine, honestly */
    double imad_per_s = cand_per_s * 768.0 * 576.0;         /* squarings x products */
    double sm_imad = 64.0 * (double)p.multiProcessorCount;
    double ghz = (double)p.clockRate * 1e-6;
    double peak_imad = sm_imad * ghz * 1e9;
    printf("  roofline: %.1f G product/s vs ~%.0f G IMAD/s peak (%.1f%% of IMAD,\n",
           imad_per_s / 1e9, peak_imad / 1e9, 100.0 * imad_per_s / peak_imad);
    printf("            CGBN baseline is %.1f%% of the same peak)\n",
           100.0 * (CGBN_BASELINE * 768.0 * 576.0) / peak_imad);
    return 0;
}
