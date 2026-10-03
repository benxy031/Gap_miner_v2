/*
 * mr68_gpu.cu — dedicated 68-bit-class Miller-Rabin kernel for the Prime-Gap
 * Network Phase-0 exhaustive scan (standalone tool; no gapminer/gap_hunt code).
 *
 * WHY A DEDICATED KERNEL: the existing CGBN AL=2 (128-bit) MR path delivers
 * 31-33M tests/s on a 3070 regardless of candidate magnitude (overhead-bound
 * at 2 limbs, measured 2026-09-30).  The Phase-0 workload needs ~3.06e12 MR
 * tests per 1e14 integers at 2e20, and a purpose-built kernel should come
 * within a small factor of the instruction ceiling instead.
 *
 * DESIGN
 *   - candidates are consecutive odd integers n = BASE + 2*i + 1 starting at
 *     2e20 (exactly the scan workload's shape), carried as 3 x 32-bit limbs
 *     (96-bit container; n < 2^68 so the top word is < 16);
 *   - ONE THREAD PER CANDIDATE: 3-word CIOS Montgomery, all state in
 *     registers (~3x fewer words than the mining kernels), no shuffles, no
 *     shared memory, no barriers;
 *   - base-2 strong-probable-prime test (MR base 2) — the composite filter
 *     the scan pipeline runs on every sieve survivor.  The second-stage
 *     confirmation (extra bases / BPSW on the ~2% survivors) is deliberately
 *     NOT part of this kernel: it is a ~2% add-on, not the throughput lever.
 *   - the CIOS core, the Montgomery-constant construction (ONE = 2^96 mod n
 *     via 2^b - n then (96-b) doublings) and the MR tail are ported from the
 *     verified tools/bench_cios_mr.cu math (GMP-validated there at 24 limbs);
 *     the --validate mode below re-validates THIS kernel end-to-end against
 *     GMP before any benchmark is trusted.
 *
 * Build: make bin/mr68_gpu WITH_CUDA=1
 * Run:
 *   ./bin/mr68_gpu --validate 200000          # GMP cross-check, must be 0 bad
 *   ./bin/mr68_gpu 1048576 20                 # batch x iters benchmark
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <gmp.h>

#ifdef PHASE0_KERNEL_DLL
/* Windows kernel-DLL split: kernels live in phase0gpu.dll (see
   phase0gpu_api.h / README_WINDOWS.md); NW is mirrored in the API header. */
#include "phase0gpu_api.h"
#undef NW
#define NW P0GPU_NW
#else
#include "mr68_kernel.cuh"
#endif

/* ---------------- host ---------------- */

typedef unsigned __int128 u128;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* candidates: odd n = BASE + 2*i + 1 with BASE = 2e20 (the Phase-0 start) */
static void fill_candidates(uint32_t *cands, uint32_t count) {
    const u128 base = (u128)20 * (u128)10000000000000000000ULL;  /* 2e20 */
    for (uint32_t i = 0; i < count; i++) {
        u128 n = base + (u128)2 * (u128)i + 1;
        cands[(size_t)i * NW + 0] = (uint32_t)n;
        cands[(size_t)i * NW + 1] = (uint32_t)(n >> 32);
        cands[(size_t)i * NW + 2] = (uint32_t)(n >> 64);
    }
}

static int check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[mr68] %s: %s\n", what, cudaGetErrorString(e));
        return 1;
    }
    return 0;
}

static int run_validate(uint32_t count) {
    printf("[mr68] --validate %u candidates starting at 2e20 ...\n", count);
    uint32_t *d_cands = NULL;
    uint8_t *d_res = NULL, *h_res = (uint8_t *)malloc(count);
    uint32_t *h_cands = (uint32_t *)malloc((size_t)count * NW * sizeof(uint32_t));
    if (!h_res || !h_cands) { fprintf(stderr, "[mr68] alloc failed\n"); return 1; }
    fill_candidates(h_cands, count);
    if (check_cuda(cudaMalloc(&d_cands, (size_t)count * NW * sizeof(uint32_t)), "cudaMalloc cands")) return 1;
    if (check_cuda(cudaMalloc(&d_res, count), "cudaMalloc results")) return 1;
    if (check_cuda(cudaMemcpy(d_cands, h_cands, (size_t)count * NW * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D")) return 1;

#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_mr68_kernel(d_cands, d_res, count, (count + 127) / 128, 128,
                          NULL) != 0)
        return 1;
#else
    mr68_kernel<<<(count + 127) / 128, 128>>>(d_cands, d_res, count);
#endif
    if (check_cuda(cudaGetLastError(), "kernel launch")) return 1;
    if (check_cuda(cudaDeviceSynchronize(), "sync")) return 1;
    if (check_cuda(cudaMemcpy(h_res, d_res, count, cudaMemcpyDeviceToHost), "D2H")) return 1;

    mpz_t n;
    mpz_init(n);
    size_t gpu_primes = 0, gmp_primes = 0, bad = 0, bad_composite = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t w[2] = {
            (uint64_t)h_cands[(size_t)i * NW + 0] | ((uint64_t)h_cands[(size_t)i * NW + 1] << 32),
            (uint64_t)h_cands[(size_t)i * NW + 2]
        };
        mpz_import(n, 2, -1, 8, 0, 0, w);
        int gmp = mpz_probab_prime_p(n, 30) >= 1;
        if (h_res[i]) gpu_primes++;
        if (gmp) gmp_primes++;
        if ((int)h_res[i] != gmp) {
            bad++;
            if (h_res[i] && !gmp) bad_composite++;
        }
    }
    printf("[mr68] kernel primes=%zu  GMP primes=%zu  mismatches=%zu (kernel-prime-but-composite=%zu)\n",
           gpu_primes, gmp_primes, bad, bad_composite);
    if (bad) {
        printf("[mr68] VALIDATE FAILED\n");
        mpz_clear(n); free(h_res); free(h_cands);
        cudaFree(d_cands); cudaFree(d_res);
        return 1;
    }
    printf("[mr68] VALIDATE PASS (0 mismatches vs GMP over %u consecutive candidates)\n", count);
    mpz_clear(n); free(h_res); free(h_cands);
    cudaFree(d_cands); cudaFree(d_res);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && !strcmp(argv[1], "--validate")) {
        uint32_t count = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 200000u;
        cudaFree(0);
        return run_validate(count);
    }

    uint32_t batch = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : 1048576u;
    int iters = (argc > 2) ? atoi(argv[2]) : 20;
    if (batch == 0 || iters < 1) return 2;

    cudaFree(0);
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("[mr68] device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
               prop.multiProcessorCount);

    uint32_t *d_cands = NULL;
    uint8_t *d_res = NULL;
    uint32_t *h_cands = (uint32_t *)malloc((size_t)batch * NW * sizeof(uint32_t));
    if (!h_cands) { fprintf(stderr, "[mr68] alloc failed\n"); return 1; }
    fill_candidates(h_cands, batch);
    if (check_cuda(cudaMalloc(&d_cands, (size_t)batch * NW * sizeof(uint32_t)), "cudaMalloc cands")) return 1;
    if (check_cuda(cudaMalloc(&d_res, batch), "cudaMalloc results")) return 1;
    if (check_cuda(cudaMemcpy(d_cands, h_cands, (size_t)batch * NW * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D")) return 1;

    const int tpb = 128;
    const uint32_t grid = (batch + tpb - 1) / tpb;

    /* warm up */
#ifdef PHASE0_KERNEL_DLL
    if (p0gpu_mr68_kernel(d_cands, d_res, batch, grid, tpb, NULL) != 0)
        return 1;
#else
    mr68_kernel<<<grid, tpb>>>(d_cands, d_res, batch);
#endif
    if (check_cuda(cudaDeviceSynchronize(), "warmup sync")) return 1;

    double t0 = now_s();
    for (int k = 0; k < iters; k++) {
#ifdef PHASE0_KERNEL_DLL
        if (p0gpu_mr68_kernel(d_cands, d_res, batch, grid, tpb, NULL) != 0)
            return 1;
#else
        mr68_kernel<<<grid, tpb>>>(d_cands, d_res, batch);
#endif
        if (check_cuda(cudaPeekAtLastError(), "launch")) return 1;
    }
    if (check_cuda(cudaDeviceSynchronize(), "sync")) return 1;
    double dt = now_s() - t0;

    double total = (double)batch * (double)iters;
    double rate = total / dt;
    printf("[mr68] NW=%d (%d-bit container, 68-bit class)  batch=%u  iters=%d\n",
           NW, NW * 32, batch, iters);
    printf("[mr68] wall=%.4f s  throughput=%.4e tests/s  (%.1f ns/test)\n",
           dt, rate, dt * 1e9 / total);
    /* projection: Phase-0 needs 3.06e12 MR tests per 1e14 integers at 2e20 */
    printf("[mr68] projection: 1e14 integers (3.06e12 tests) = %.1f min at this rate\n",
           3.06e12 / rate / 60.0);

    free(h_cands);
    cudaFree(d_cands);
    cudaFree(d_res);
    return 0;
}
