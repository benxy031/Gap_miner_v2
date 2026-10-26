/* bench_perig_setup.cu - HOW MUCH of perig's per-test cost is the SETUP?
 *
 * Motivation: PGN's engine (docs/PGN_SNAPSHOT_REVIEW_2026-10-20.md section 2)
 * removes the per-candidate reduction setup by carrying an exact reciprocal
 * between NEIGHBOURING candidates and reusing its remainder R = 2^160 mod n as
 * the exponentiation seed.  Our production test pays that setup on EVERY test:
 *
 *     mmagic = montgomeryInverse64(n_lo);           (Newton, cheap)
 *     ciosConstants128(n_lo, n_hi, &one_lo, &one_hi);   == (-m) % m, i.e. one
 *                                                   128/128 modulo per candidate
 *     ... then ~bitlength(n) CIOS squarings.
 *
 * This aborts the exponentiation and keeps ONLY the setup, on an identical
 * candidate stream and launch geometry.  It is NOT a primality test; its only
 * job is to bound the prize:
 *
 *     share_setup = t_setup / t_full = rate_full / rate_setup
 *     best-case win from eliminating the setup entirely = 1 / (1 - share_setup)
 *
 * If that ceiling is small, the whole stateful-reciprocal avenue is dead ON
 * OUR SHAPE and there is nothing worth implementing.  If it is large, the
 * stateful test (exact certificate 0 <= R < n + exact fallback) is the lever.
 *
 * Build:  nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *             tools/bench_perig_setup.cu -o bin/bench_perig_setup
 * Run:    bin/bench_perig_setup [N=1048576] [iters=50] [reps=3]
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "perig.cuh"            /* ciosFermatTest128_hi, montgomeryInverse64,
                                   ciosConstants128 (production definitions) */

__global__ static void k_full(const uint64_t *lo, const uint64_t *hi,
                              uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], hi[i]) ? 1 : 0;
}

/* setup-only arm: exactly the two per-candidate constants the exponentiation
   starts from, nothing else.  res is written so the compiler cannot drop it. */
__global__ static void k_setup(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t mm = montgomeryInverse64(lo[i]);
    uint64_t a, b;
    ciosConstants128(lo[i], hi[i], &a, &b);
    res[i] = (uint8_t)((mm ^ a ^ b) & 1ULL);
}

/* mechanism (3): the same production test, but with HIGH_64 (= 10 here) known
   at COMPILE time instead of passed as a runtime argument. */
__global__ static void k_const(const uint64_t *lo, const uint64_t *hi,
                               uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    (void)hi;
    if (i >= n) return;
    res[i] = ciosFermatTest128(lo[i]) ? 1 : 0;
}

/* dump the setup constants so the HOST can prove the arm did the work */
__global__ static void k_setup_dump(const uint64_t *lo, const uint64_t *hi,
                                    uint32_t n, uint64_t *mm,
                                    uint64_t *one_lo, uint64_t *one_hi) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t a, b;
    mm[i] = montgomeryInverse64(lo[i]);
    ciosConstants128(lo[i], hi[i], &a, &b);
    one_lo[i] = a;
    one_hi[i] = b;
}

typedef void (*kfn)(const uint64_t *, const uint64_t *, uint8_t *, uint32_t);

/* decimal print for a 128-bit value (host only; sanity print of the stream) */
static void u128_str_host(unsigned __int128 v, char *out, size_t n) {
    char tmp[48];
    int i = 0;
    if (v == 0) { snprintf(out, n, "0"); return; }
    while (v != 0 && i < 47) { tmp[i++] = (char)('0' + (int)(v % 10)); v /= 10; }
    size_t j = 0;
    while (i > 0 && j + 1 < n) out[j++] = tmp[--i];
    out[j] = 0;
}

static double bench(kfn k, const uint64_t *dlo, const uint64_t *dhi,
                    uint8_t *dres, uint32_t n, int iters) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);   /* warm */
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) k<<<(n + 255) / 256, 256>>>(dlo, dhi, dres, n);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms * 1e-3 / (double)iters;   /* seconds per launch */
}

int main(int argc, char **argv) {
    uint32_t N = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    int iters  = (argc > 2) ? atoi(argv[2]) : 50;
    int reps   = (argc > 3) ? atoi(argv[3]) : 3;

    /* candidate stream: consecutive ODD numbers around 2e20 (high word 10),
       i.e. the same band PGN searches and the band our walk runs at. */
    uint64_t *hlo = (uint64_t *)malloc((size_t)N * 8);
    uint64_t *hhi = (uint64_t *)malloc((size_t)N * 8);
    if (!hlo || !hhi) { fprintf(stderr, "alloc failed\n"); return 1; }
    const uint64_t HI = 10;                    /* 2e20 = 10 * 2^64 + lo */
    /* 2e20 does NOT fit in 64 bits - build it in 128-bit arithmetic or the
       literal silently wraps (caught by the compiler warning, not by a test). */
    unsigned __int128 two_e20 = (unsigned __int128)2000000000000000000ULL * 100u;
    uint64_t base_lo = (uint64_t)(two_e20 - ((unsigned __int128)HI << 64));
    for (uint32_t i = 0; i < N; i++) {
        hlo[i] = base_lo + 2ull * i + 1ull;    /* odd */
        hhi[i] = HI;
    }
    {
        unsigned __int128 first = ((unsigned __int128)HI << 64) + hlo[0];
        unsigned __int128 last  = ((unsigned __int128)HI << 64) + hlo[N - 1];
        char sf[48], sl[48];
        u128_str_host(first, sf, sizeof sf);
        u128_str_host(last, sl, sizeof sl);
        printf("candidates: %s .. %s (all odd, step 2)\n", sf, sl);
    }

    uint64_t *dlo, *dhi;
    uint8_t  *dres;
    cudaMalloc(&dlo, (size_t)N * 8);
    cudaMalloc(&dhi, (size_t)N * 8);
    cudaMalloc(&dres, (size_t)N);
    cudaMemcpy(dlo, hlo, (size_t)N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dhi, hhi, (size_t)N * 8, cudaMemcpyHostToDevice);

    char name[128];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    strncpy(name, prop.name, sizeof name - 1);
    name[sizeof name - 1] = 0;
    printf("device: %s (%d SMs)  N=%u  iters=%d  reps=%d  band=2e20 (hi=%llu)\n",
           name, prop.multiProcessorCount, N, iters, reps,
           (unsigned long long)HI);

    /* ---- falsification 1: the setup arm must reproduce the HOST constants --- */
    {
        const uint32_t K = 8;
        uint64_t *dmm, *dol, *doh;
        cudaMalloc(&dmm, K * 8); cudaMalloc(&dol, K * 8); cudaMalloc(&doh, K * 8);
        k_setup_dump<<<1, 32>>>(dlo, dhi, K, dmm, dol, doh);
        cudaDeviceSynchronize();
        uint64_t hmm[8], hol[8], hoh[8];
        cudaMemcpy(hmm, dmm, K * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hol, dol, K * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hoh, doh, K * 8, cudaMemcpyDeviceToHost);
        int bad = 0;
        for (uint32_t i = 0; i < K; i++) {
            uint64_t emm = montgomeryInverse64(hlo[i]);
            uint64_t eol, eoh;
            ciosConstants128(hlo[i], hhi[i], &eol, &eoh);
            if (emm != hmm[i] || eol != hol[i] || eoh != hoh[i]) bad++;
        }
        printf("setup arm verification (device vs host, %u candidates): %s\n",
               K, bad ? "MISMATCH - timing NOT trustworthy" : "all match");
        cudaFree(dmm); cudaFree(dol); cudaFree(doh);
    }

    /* ---- falsification 2: full and const_hi must agree on every verdict ---- */
    {
        uint8_t *hr1 = (uint8_t *)malloc(N), *hr2 = (uint8_t *)malloc(N);
        k_full<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaDeviceSynchronize();
        cudaMemcpy(hr1, dres, N, cudaMemcpyDeviceToHost);
        k_const<<<(N + 255) / 256, 256>>>(dlo, dhi, dres, N);
        cudaDeviceSynchronize();
        cudaMemcpy(hr2, dres, N, cudaMemcpyDeviceToHost);
        uint32_t primes = 0, diff = 0;
        for (uint32_t i = 0; i < N; i++) { if (hr1[i] != hr2[i]) diff++; primes += hr1[i]; }
        printf("verdict parity full vs const_hi: %u/%u probable primes, %u difference%s\n",
               primes, N, diff, diff ? "s  <-- BUG" : "");
        free(hr1); free(hr2);
    }

    double bf_full = 1e9, bf_const = 1e9, bf_setup = 1e9;
    for (int r = 0; r < reps; r++) {
        double t = bench(k_full,  dlo, dhi, dres, N, iters);
        if (t < bf_full)  bf_full  = t;
        t = bench(k_const, dlo, dhi, dres, N, iters);
        if (t < bf_const) bf_const = t;
        t = bench(k_setup, dlo, dhi, dres, N, iters);
        if (t < bf_setup) bf_setup = t;
    }

    double rf = (double)N / bf_full;    /* candidates per second */
    double rc = (double)N / bf_const;
    double rs = (double)N / bf_setup;
    double share = bf_setup / bf_full;  /* fraction of the full test spent on setup */

    printf("\narm      ms/launch   tests/s      ns/test\n");
    printf("full     %9.3f   %.4e   %.3f\n", bf_full * 1e3, rf, 1e9 / rf);
    printf("const_hi %9.3f   %.4e   %.3f   (%.3fx vs full)\n",
           bf_const * 1e3, rc, 1e9 / rc, rc / rf);
    printf("setup    %9.3f   %.4e   %.3f   (%.1f%% of full)\n",
           bf_setup * 1e3, rs, 1e9 / rs, 100.0 * share);
    printf("\nmechanism (2) ceiling - eliminate the setup entirely: %.2fx\n",
           1.0 / (1.0 - share));
    printf("mechanism (3) - compile-time high word:              %.3fx\n", rc / rf);

    cudaFree(dlo); cudaFree(dhi); cudaFree(dres);
    free(hlo); free(hhi);
    return 0;
}
