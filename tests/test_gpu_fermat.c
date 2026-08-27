/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Correctness test for the CUDA/CGBN GPU Fermat kernel.
 *
 * The GPU kernel computes the exact same deterministic function as the CPU
 * (base-2 Miller-Rabin: 2^d mod n with n-1 = d·2^s, plus s squarings), so
 * every candidate's GPU result must match the CPU reference exactly (not just
 * "usually agree"). GMP's own independent Miller-Rabin (mpz_probab_prime_p)
 * is used as a trusted ground truth to confirm the invariant that every real
 * prime always passes (no base-2 Miller-Rabin false negatives are possible
 * for odd n).
 *
 * When built without WITH_CUDA=1, this test builds and links but skips at
 * runtime (there is no GPU kernel to exercise).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include "../new_src/primality_fermat.h"

#ifdef WITH_CUDA
#include "../new_src/gpu_adapter.h"
#include "../new_src/gpu/gpu_fermat.h"
#include <cuda_runtime.h>

#define TEST_CANDIDATE_COUNT 4000U
/* Keep comfortably under GPU_NLIMBS*64 bits (default build: 384-bit). */
#define TEST_CANDIDATE_BITS 300U

/* CPU reference: base-2 Miller-Rabin (strong probable prime) test.
   The GPU kernel now runs this exact test, so results must match exactly. */
static int cpu_mr_base2(const mpz_t n) {
    if (mpz_cmp_ui(n, 2) == 0) return 1;
    if (mpz_cmp_ui(n, 2) < 0 || mpz_even_p(n)) return 0;

    mpz_t d, x, n1;
    mpz_init_set(n1, n);
    mpz_sub_ui(n1, n1, 1);          /* n1 = n-1 */
    mpz_init_set(d, n1);

    unsigned long s = 0;
    while (mpz_even_p(d)) {
        mpz_fdiv_q_2exp(d, d, 1);   /* d >>= 1 */
        s++;
    }

    mpz_init_set_ui(x, 2);
    mpz_powm(x, x, d, n);           /* x = 2^d mod n */

    int ok = 0;
    if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, n1) == 0) {
        ok = 1;
    } else {
        for (unsigned long i = 1; i < s; i++) {
            mpz_powm_ui(x, x, 2, n);    /* x = x^2 mod n */
            if (mpz_cmp(x, n1) == 0) { ok = 1; break; }
            if (mpz_cmp_ui(x, 1) == 0) break;  /* hit 1 before n-1 => composite */
        }
    }

    mpz_clear(d);
    mpz_clear(x);
    mpz_clear(n1);
    return ok;
}

static int run_gpu_fermat_correctness_test(void) {
    printf("[TEST] GPU Fermat batch vs. CPU Fermat + GMP ground truth...\n");

    struct gpu_adapter *adapter = gpu_adapter_init(0);
    if (!adapter) {
        fprintf(stderr, "  SKIP: no CUDA device available\n");
        return 1;
    }

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, 0xC0FFEEu);

    mpz_t *candidates = (mpz_t *)malloc(TEST_CANDIDATE_COUNT * sizeof(mpz_t));
    uint8_t *gpu_is_prime = (uint8_t *)calloc(TEST_CANDIDATE_COUNT, 1);
    if (!candidates || !gpu_is_prime) {
        fprintf(stderr, "  FAIL: out of memory\n");
        free(candidates);
        free(gpu_is_prime);
        gpu_adapter_free(adapter);
        return 0;
    }

    for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) {
        mpz_init(candidates[i]);
        mpz_urandomb(candidates[i], rng, TEST_CANDIDATE_BITS);
        mpz_setbit(candidates[i], TEST_CANDIDATE_BITS - 1); /* fixed bit width */
        mpz_setbit(candidates[i], 0);                       /* force odd */
    }

    struct gpu_batch batch;
    batch.count = TEST_CANDIDATE_COUNT;
    batch.candidates = candidates;
    batch.is_prime = gpu_is_prime;

    int rc = gpu_adapter_test_batch(adapter, &batch);
    if (rc != 0) {
        fprintf(stderr, "  FAIL: gpu_adapter_test_batch returned %d\n", rc);
        for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) mpz_clear(candidates[i]);
        free(candidates);
        free(gpu_is_prime);
        gpu_adapter_free(adapter);
        gmp_randclear(rng);
        return 0;
    }

    uint64_t mismatches = 0;
    uint64_t false_negatives_vs_gmp = 0;
    uint64_t gpu_primes = 0;

    for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) {
        /* Same deterministic base-2 Miller-Rabin function: must match exactly. */
        int cpu_mr_base2_result = cpu_mr_base2(candidates[i]);
        if ((int)gpu_is_prime[i] != cpu_mr_base2_result) {
            mismatches++;
            gmp_fprintf(stderr,
                        "  MISMATCH at %u: gpu=%d cpu_mr_base2=%d n=%Zd\n",
                        i, gpu_is_prime[i], cpu_mr_base2_result, candidates[i]);
        }

        /* Independent ground truth: a real prime must never be flagged
           composite by base-2 Miller-Rabin (no false negatives are possible). */
        if (mpz_probab_prime_p(candidates[i], 25) > 0 && !gpu_is_prime[i]) {
            false_negatives_vs_gmp++;
        }

        gpu_primes += gpu_is_prime[i] ? 1 : 0;
    }

    for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) mpz_clear(candidates[i]);
    free(candidates);
    free(gpu_is_prime);
    gpu_adapter_free(adapter);
    gmp_randclear(rng);

    printf("  Candidates=%u GPU-probable-primes=%llu mismatches=%llu false_negatives=%llu\n",
           TEST_CANDIDATE_COUNT, (unsigned long long)gpu_primes,
           (unsigned long long)mismatches, (unsigned long long)false_negatives_vs_gmp);

    if (mismatches != 0 || false_negatives_vs_gmp != 0) {
        fprintf(stderr, "  FAIL: GPU Fermat kernel disagrees with CPU/GMP\n");
        return 0;
    }

    printf("  PASS: GPU Miller-Rabin base-2 kernel exactly matches CPU reference "
           "(0 mismatches, 0 false negatives vs. GMP)\n");
    return 1;
}

/* Fused-pipeline Stage 2: the device-pointer entry point (gpu_fermat_test_device)
   must produce bit-identical verdicts to the H2D path (gpu_fermat_test_batch),
   whose math is already validated against CPU/GMP above.  Covers both the
   AoS CGBN kernel (even AL) and the AoS scalar kernel (odd AL). */
static int run_gpu_fermat_device_path_test(void) {
    printf("[TEST] GPU Fermat device-pointer path vs H2D path...\n");

    gpu_fermat_ctx *ctx = gpu_fermat_init(0, TEST_CANDIDATE_COUNT);
    if (!ctx) {
        fprintf(stderr, "  SKIP: no CUDA device available\n");
        return 1;
    }

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, 0xD00D1234u);

    static const int limb_cases[] = {5, 10, 12, 20};
    int all_ok = 1;

    for (size_t ci = 0; ci < sizeof(limb_cases) / sizeof(limb_cases[0]); ci++) {
        int AL = limb_cases[ci];
        gpu_fermat_set_limbs(ctx, AL);

        uint64_t *h_cands =
            (uint64_t *)calloc(TEST_CANDIDATE_COUNT * (size_t)AL, sizeof(uint64_t));
        uint8_t *host_results = (uint8_t *)calloc(TEST_CANDIDATE_COUNT, 1);
        uint8_t *dev_results = (uint8_t *)calloc(TEST_CANDIDATE_COUNT, 1);
        uint64_t *d_cands = NULL;
        if (!h_cands || !host_results || !dev_results) {
            fprintf(stderr, "  FAIL: out of memory\n");
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }

        /* Random odd candidates, full AL-limb width, packed at AL stride. */
        mpz_t n;
        mpz_init(n);
        for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) {
            mpz_urandomb(n, rng, (unsigned long)AL * 64UL);
            mpz_setbit(n, (unsigned long)AL * 64UL - 1UL);
            mpz_setbit(n, 0);
            size_t written = 0;
            mpz_export(h_cands + (size_t)i * (size_t)AL, &written, -1,
                       sizeof(uint64_t), 0, 0, n);
            /* remaining high limbs stay zero from calloc */
        }
        mpz_clear(n);

        /* H2D reference path. */
        int host_primes =
            gpu_fermat_test_batch(ctx, h_cands, host_results, TEST_CANDIDATE_COUNT);
        if (host_primes < 0) {
            fprintf(stderr, "  FAIL: H2D path returned -1 (AL=%d)\n", AL);
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }

        /* Upload candidates to a device AoS buffer, then test in-place. */
        cudaError_t err = cudaMalloc((void **)&d_cands,
                                     TEST_CANDIDATE_COUNT * (size_t)AL *
                                         sizeof(uint64_t));
        if (err != cudaSuccess) {
            fprintf(stderr, "  FAIL: cudaMalloc (AL=%d): %s\n", AL,
                    cudaGetErrorString(err));
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }
        err = cudaMemcpy(d_cands, h_cands,
                         TEST_CANDIDATE_COUNT * (size_t)AL * sizeof(uint64_t),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "  FAIL: cudaMemcpy H2D (AL=%d): %s\n", AL,
                    cudaGetErrorString(err));
            cudaFree(d_cands);
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }

        int dev_primes =
            gpu_fermat_test_device(ctx, d_cands, dev_results, TEST_CANDIDATE_COUNT);
        cudaFree(d_cands);

        if (dev_primes < 0) {
            fprintf(stderr, "  FAIL: device path returned -1 (AL=%d)\n", AL);
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }

        uint32_t mismatches = 0;
        for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++)
            if (host_results[i] != dev_results[i]) mismatches++;

        if (mismatches != 0 || host_primes != dev_primes) {
            fprintf(stderr,
                    "  FAIL AL=%d: host_primes=%d dev_primes=%d mismatches=%u\n",
                    AL, host_primes, dev_primes, mismatches);
            all_ok = 0;
        } else {
            printf("  OK  AL=%d: %d primes, device path == H2D path "
                   "(0 mismatches)\n", AL, dev_primes);
        }

        free(h_cands);
        free(host_results);
        free(dev_results);
    }

    gmp_randclear(rng);
    gpu_fermat_destroy(ctx);
    return all_ok;
}
#endif /* WITH_CUDA */

int main(void) {
#ifndef WITH_CUDA
    printf("test_gpu_fermat: SKIPPED (built without WITH_CUDA=1)\n");
    return 0;
#else
    printf("========== GPU Fermat Kernel Tests ==========\n\n");

    int ok = run_gpu_fermat_correctness_test();
    if (ok)
        ok = run_gpu_fermat_device_path_test();

    printf("\n==============================================\n");
    if (!ok) {
        printf("SOME GPU FERMAT TESTS FAILED\n");
        return 1;
    }
    printf("All GPU Fermat tests PASSED\n");
    return 0;
#endif
}
