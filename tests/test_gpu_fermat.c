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
 *
 * A third pass adds "Schryer-style" boundary bit-pattern operands (all ones,
 * (1<<i)+/-(1<<k), contiguous masks, repeated 1-blocks, values around real
 * primes).  On the NVIDIA forums (thread 384294, 2026) these patterns exposed
 * a dropped-carry bug with ~10^8x higher hit rate than uniform-random
 * operands -- the same class of carry/range edge case as our CGBN
 * lazy-reduction fix, so random-only coverage is not enough for this math.
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
   whose math is already validated against CPU/GMP above. Covers both CGBN and
   scalar AoS kernels at their supported widths. */
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

    /* AL=24/28 use scalar; AL=32 is the widest CGBN instance. AL=16 is the
       hunt's 1024-bit width. Both 16 and 32 also have CIOS implementations. */
    static const int limb_cases[] = {5, 10, 12, 16, 20, 24, 28, 32};
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

        /* Random odd candidates packed at AL stride.  Three widths matter:
             - full width (top bit at AL*64-1): the classic case;
             - partial top limb (production shape: e.g. a 976-bit candidate
               inside a 1024-bit stride) -- this is exactly where the
               "Mont(1) = 2^(32*N32) - n" shortcut is WRONG, so it is the only
               shape that validates cios_mont_one's bitlength + doublings;
             - half width: a longer doubling run and exponent bits above the
               top word must read as zero.
           Remaining high limbs stay zero from calloc. */
        mpz_t n;
        mpz_init(n);
        for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) {
            unsigned long bits = (unsigned long)AL * 64UL;
            if ((i & 3U) == 2U) {
                bits = (unsigned long)AL * 64UL - 1UL - (unsigned long)(i % 56U);
            } else if ((i & 3U) == 3U) {
                bits = (unsigned long)AL * 32UL;      /* ~half width */
            }
            mpz_urandomb(n, rng, bits);
            mpz_setbit(n, bits - 1UL);
            mpz_setbit(n, 0);
            if ((AL == 5 || AL == 10) && i < 16U) {
                mpz_clrbit(n, bits - 2UL);
                mpz_nextprime(n, n);
            }
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

        if (dev_primes < 0) {
            fprintf(stderr, "  FAIL: device path returned -1 (AL=%d)\n", AL);
            cudaFree(d_cands);
            free(h_cands); free(host_results); free(dev_results);
            all_ok = 0;
            break;
        }

        uint32_t mismatches = 0, cpu_mismatches = 0;
        mpz_init(n);
        for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT; i++) {
            if (host_results[i] != dev_results[i]) mismatches++;
            if (AL == 5 || AL == 10) {
                mpz_import(n, (size_t)AL, -1, sizeof(uint64_t), 0, 0,
                           h_cands + (size_t)i * (size_t)AL);
                int expected = cpu_mr_base2(n);
                if (host_results[i] != expected || dev_results[i] != expected)
                    cpu_mismatches++;
            }
        }
        mpz_clear(n);

        if (AL == 5 || AL == 10) {
            static const uint32_t counts[] = {1, 31, 32, 33, 63, 64, 65};
            for (size_t j = 0; j < sizeof(counts) / sizeof(counts[0]); j++) {
                uint32_t count = counts[j];
                int h = gpu_fermat_test_batch(ctx, h_cands, host_results, count);
                int d = gpu_fermat_test_device(ctx, d_cands, dev_results, count);
                if (h < 0 || d < 0 || h != d) {
                    fprintf(stderr, "  FAIL AL=%d batch=%u: H2D=%d device=%d\n",
                            AL, count, h, d);
                    all_ok = 0;
                    continue;
                }
                for (uint32_t i = 0; i < count; i++) {
                    mpz_t candidate;
                    mpz_init(candidate);
                    mpz_import(candidate, (size_t)AL, -1, sizeof(uint64_t), 0,
                               0, h_cands + (size_t)i * (size_t)AL);
                    int expected = cpu_mr_base2(candidate);
                    if (host_results[i] != expected || dev_results[i] != expected)
                        cpu_mismatches++;
                    mpz_clear(candidate);
                }
            }
        }
        cudaFree(d_cands);

        /* Bounded diagnostics (GAPDBG_FERMAT_MISMATCH=1): which candidates
           disagree, and their top-limb shape.  The mismatch count alone cannot
           tell a layout bug from a verdict bug. */
        if (mismatches != 0 && getenv("GAPDBG_FERMAT_MISMATCH")) {
            uint32_t shown = 0;
            for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT && shown < 12; i++) {
                if (host_results[i] == dev_results[i]) continue;
                unsigned long bits = (unsigned long)AL * 64UL;
                if ((i & 3U) == 2U)
                    bits = (unsigned long)AL * 64UL - 1UL -
                           (unsigned long)(i % 56U);
                else if ((i & 3U) == 3U)
                    bits = (unsigned long)AL * 32UL;
                unsigned top = (unsigned)(bits & 63UL);
                fprintf(stderr,
                        "  [mismatch] AL=%d i=%u shape=%u bits=%lu topbits=%u "
                        "host=%d dev=%d\n",
                        AL, i, i & 3U, bits, top ? top : 64,
                        (int)host_results[i], (int)dev_results[i]);
                shown++;
            }
        }

        /* Isolated retest of the first few mismatches: one candidate per call
           through BOTH paths.  If a candidate that disagreed inside the batch
           agrees on its own, the bug is in the batch/grid path, not in the
           arithmetic or in the AoS/SoA conversion. */
        if (mismatches != 0 && getenv("GAPDBG_FERMAT_MISMATCH")) {
            uint32_t shown = 0;
            for (uint32_t i = 0; i < TEST_CANDIDATE_COUNT && shown < 4; i++) {
                if (host_results[i] == dev_results[i]) continue;
                uint8_t r1 = 0xFF, r2 = 0xFF;
                uint64_t *d_one = NULL;
                int p1 = gpu_fermat_test_batch(ctx, h_cands + (size_t)i * (size_t)AL,
                                               &r1, 1);
                int p2 = -1;
                if (cudaMalloc((void **)&d_one, (size_t)AL * sizeof(uint64_t)) ==
                    cudaSuccess) {
                    if (cudaMemcpy(d_one, h_cands + (size_t)i * (size_t)AL,
                                   (size_t)AL * sizeof(uint64_t),
                                   cudaMemcpyHostToDevice) == cudaSuccess)
                        p2 = gpu_fermat_test_device(ctx, d_one, &r2, 1);
                    cudaFree(d_one);
                }
                fprintf(stderr,
                        "  [isolated] AL=%d i=%u single: batch=%d(r=%d) "
                        "device=%d(r=%d) in_batch: host=%d dev=%d\n",
                        AL, i, p1, (int)r1, p2, (int)r2,
                        (int)host_results[i], (int)dev_results[i]);
                shown++;
            }
        }

        if (mismatches != 0 || cpu_mismatches != 0 || host_primes != dev_primes) {
            fprintf(stderr,
                    "  FAIL AL=%d: host_primes=%d dev_primes=%d "
                    "mismatches=%u cpu_mismatches=%u\n",
                    AL, host_primes, dev_primes, mismatches, cpu_mismatches);
            all_ok = 0;
        } else {
            printf("  OK  AL=%d: %d primes, device path == H2D path "
                   "(0 mismatches%s)\n", AL, dev_primes,
                   (AL == 5 || AL == 10) ? ", 0 CPU mismatches" : "");
        }

        free(h_cands);
        free(host_results);
        free(dev_results);
    }

    gmp_randclear(rng);
    gpu_fermat_destroy(ctx);
    return all_ok;
}

/* ---------------- Schryer-style boundary bit-patterns ---------------- */

#define PATTERN_MAX 256U

/* Append odd, width-b (bit b-1 set) boundary patterns of one bit width.
 * Families: all ones; 2^b - c (modulus just below the width boundary);
 * 1...10...01 (2^b - 2^j + 1); left-contiguous masks + low 1; bottom masks +
 * top + 1; sparse two/three-bit values; repeated 1-blocks (2^b-1)/(2j+1);
 * 32-bit limb patterns (0xFFFF.../0x7FFF.../0xFFFE...); and values around
 * real primes (p-2, p, p+2) built with mpz_nextprime. */
static uint32_t build_pattern_candidates(mpz_t *arr, uint32_t cap,
                                         unsigned long bits) {
    uint32_t n = 0;
    unsigned long top = bits - 1UL;
    unsigned long half = bits / 2UL;
    mpz_t x, t;
    mpz_inits(x, t, NULL);

#define PUSH() do { if (n < cap) mpz_set(arr[n++], x); } while (0)

    /* A: all ones */
    mpz_set_ui(x, 1);
    mpz_mul_2exp(x, x, bits);
    mpz_sub_ui(x, x, 1);
    PUSH();

    /* B: 2^b - c for small odd c */
    static const unsigned long cs[] = {1, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65,
                                       127, 129, 255, 257, 511, 513, 1023, 1025};
    for (size_t i = 0; i < sizeof(cs) / sizeof(cs[0]); i++) {
        mpz_set_ui(x, 1);
        mpz_mul_2exp(x, x, bits);
        mpz_sub_ui(x, x, cs[i]);
        PUSH();
    }

    /* C: 1...10...01 -> 2^b - 2^j + 1 */
    unsigned long js_c[] = {1, 2, 3, 8, 16, 32, 64, half, bits - 2};
    for (size_t i = 0; i < sizeof(js_c) / sizeof(js_c[0]); i++) {
        unsigned long j = js_c[i];
        if (j >= bits - 1) continue;
        mpz_set_ui(x, 1);
        mpz_mul_2exp(x, x, bits);
        mpz_sub_ui(x, x, 1);
        mpz_set_ui(t, 1);
        mpz_mul_2exp(t, t, j);
        mpz_sub_ui(t, t, 1);
        mpz_sub(x, x, t);
        mpz_add_ui(x, x, 1);
        PUSH();
    }

    /* D: left-contiguous mask + low 1 -> ((2^(b-j) - 1) << j) | 1 */
    unsigned long js_d[] = {1, 2, 3, 8, 16, 32, 64, half, bits / 4};
    for (size_t i = 0; i < sizeof(js_d) / sizeof(js_d[0]); i++) {
        unsigned long j = js_d[i];
        if (j >= bits - 1) continue;
        mpz_set_ui(x, 1);
        mpz_mul_2exp(x, x, bits - j);
        mpz_sub_ui(x, x, 1);
        mpz_mul_2exp(x, x, j);
        mpz_setbit(x, 0);
        PUSH();
    }

    /* E: bottom mask + top bit + low 1 */
    unsigned long ms[] = {2, 3, 7, 8, 15, 16, 31, 32, 63, 64, half};
    for (size_t i = 0; i < sizeof(ms) / sizeof(ms[0]); i++) {
        unsigned long m = ms[i];
        if (m >= bits - 1) continue;
        mpz_set_ui(x, 1);
        mpz_mul_2exp(x, x, m);
        mpz_sub_ui(x, x, 1);
        mpz_setbit(x, top);
        mpz_setbit(x, 0);
        PUSH();
    }

    /* F: sparse three-bit: top + 2^i + 1 */
    unsigned long is_f[] = {0, 1, 2, 3, 8, 16, 31, 32, 63, 64, half, top - 1};
    for (size_t i = 0; i < sizeof(is_f) / sizeof(is_f[0]); i++) {
        unsigned long b_i = is_f[i];
        if (b_i >= top) continue;
        mpz_set_ui(x, 0);
        mpz_setbit(x, top);
        mpz_setbit(x, b_i);
        mpz_setbit(x, 0);
        PUSH();
    }

    /* G: repeated 1-blocks floor((2^b - 1)/(2j+1)), plus top bit and low 1 */
    static const unsigned long js_g[] = {2, 4, 8, 16, 32, 64};
    for (size_t i = 0; i < sizeof(js_g) / sizeof(js_g[0]); i++) {
        mpz_set_ui(x, 1);
        mpz_mul_2exp(x, x, bits);
        mpz_sub_ui(x, x, 1);
        mpz_fdiv_q_ui(x, x, 2UL * js_g[i] + 1UL);
        mpz_setbit(x, top);
        mpz_setbit(x, 0);
        PUSH();
    }

    /* H: 32-bit limb patterns (0xFFFF.../0x7FFF.../0xFFFE.../masks), both
     * orders; top bit and low bit pinned afterwards. */
    static const uint32_t pat[8] = {0x00000001u, 0x7fffffffu, 0xffffffffu,
                                    0xfffffffeu, 0x80000000u, 0x0000ffffu,
                                    0xffff0000u, 0x00010000u};
    for (unsigned rev = 0; rev < 2U; rev++) {
        unsigned long fields = bits / 32UL;
        mpz_set_ui(x, 0);
        for (unsigned long k = 0; k < fields; k++) {
            unsigned long idx = rev ? (fields - 1UL - k) : k;
            mpz_set_ui(t, pat[idx % 8U]);
            mpz_mul_2exp(t, t, 32UL * k);
            mpz_add(x, x, t);
        }
        mpz_setbit(x, top);
        mpz_setbit(x, 0);
        PUSH();
    }

    /* I: values around real primes: p-2, p, p+2 for two seeds */
    for (int k = 0; k < 2; k++) {
        mpz_set_ui(t, 1);
        mpz_mul_2exp(t, t, top);
        if (k == 0) mpz_add_ui(t, t, 5);
        else { mpz_setbit(t, half); mpz_setbit(t, 0); }
        mpz_nextprime(x, t);          /* x = p */
        mpz_sub_ui(t, x, 2);
        if (n < cap) mpz_set(arr[n++], t);       /* p-2 */
        if (n < cap) mpz_set(arr[n++], x);       /* p   */
        mpz_add_ui(t, x, 2);
        if (n < cap) mpz_set(arr[n++], t);       /* p+2 */
    }

#undef PUSH
    mpz_clears(x, t, NULL);
    return n;
}

/* Every pattern candidate must produce the same base-2 Miller-Rabin verdict
 * as the GMP-based CPU reference, at every production limb width, and no real
 * prime may be flagged composite.
 *
 * Several families DETERMINISTICALLY construct base-2 strong pseudoprimes:
 *   2^307 - 1             (partial 307-bit width: 307 is prime, so the
 *                          Mersenne number is a strong psp(2) -- produced by
 *                          families A/B1/C1/D1), and
 *   2^b - 2^(b/2) + 1 = Phi_6(2^(b/2))   (families C/D at j = b/2).
 * Base-2 MR MUST accept these exactly like the CPU reference does (they are
 * the accept-path boundary cases); GMP's full BPSW (random bases + Lucas)
 * correctly rejects them -- which is why gpu-primes may exceed gmp-primes,
 * and exactly why the miner's merit-qualified BPSW layer exists. */
static int run_gpu_fermat_pattern_test(void) {
    printf("[TEST] GPU Fermat boundary bit-patterns (Schryer-style) vs CPU reference...\n");

    gpu_fermat_ctx *ctx = gpu_fermat_init(0, PATTERN_MAX);
    if (!ctx) {
        fprintf(stderr, "  SKIP: no CUDA device available\n");
        return 1;
    }

    static const int limb_cases[] = {5, 10, 12, 16, 20, 24, 28, 32};
    int all_ok = 1;
    uint32_t total_psp2 = 0;

    for (size_t ci = 0; ci < sizeof(limb_cases) / sizeof(limb_cases[0]); ci++) {
        int AL = limb_cases[ci];
        gpu_fermat_set_limbs(ctx, AL);

        mpz_t pats[PATTERN_MAX];
        for (uint32_t i = 0; i < PATTERN_MAX; i++) mpz_init(pats[i]);

        unsigned long full = (unsigned long)AL * 64UL;
        unsigned long partial = full - 13UL;   /* production partial-top-limb shape */
        uint32_t count = 0;
        count += build_pattern_candidates(pats + count, PATTERN_MAX - count, full);
        count += build_pattern_candidates(pats + count, PATTERN_MAX - count, partial);

        uint64_t *h_cands =
            (uint64_t *)calloc((size_t)count * (size_t)AL, sizeof(uint64_t));
        uint8_t *results = (uint8_t *)calloc(count, 1);
        uint8_t *dev_results = (uint8_t *)calloc(count, 1);
        uint64_t *d_cands = NULL;
        if (!h_cands || !results || !dev_results) {
            fprintf(stderr, "  FAIL: out of memory (AL=%d)\n", AL);
            free(h_cands); free(results); free(dev_results);
            for (uint32_t i = 0; i < PATTERN_MAX; i++) mpz_clear(pats[i]);
            all_ok = 0;
            break;
        }

        for (uint32_t i = 0; i < count; i++) {
            size_t written = 0;
            mpz_export(h_cands + (size_t)i * (size_t)AL, &written, -1,
                       sizeof(uint64_t), 0, 0, pats[i]);
        }

        int primes = gpu_fermat_test_batch(ctx, h_cands, results, count);
        if (primes < 0) {
            fprintf(stderr, "  FAIL: H2D path returned -1 (AL=%d)\n", AL);
            free(h_cands); free(results); free(dev_results);
            for (uint32_t i = 0; i < PATTERN_MAX; i++) mpz_clear(pats[i]);
            all_ok = 0;
            break;
        }
        size_t bytes = (size_t)count * (size_t)AL * sizeof(uint64_t);
        cudaError_t err = cudaMalloc((void **)&d_cands, bytes);
        if (err == cudaSuccess)
            err = cudaMemcpy(d_cands, h_cands, bytes, cudaMemcpyHostToDevice);
        int dev_primes = err == cudaSuccess
            ? gpu_fermat_test_device(ctx, d_cands, dev_results, count) : -1;
        if (err != cudaSuccess || dev_primes < 0) {
            fprintf(stderr, "  FAIL: device pattern path (AL=%d): %s\n",
                    AL, cudaGetErrorString(err));
            if (d_cands) cudaFree(d_cands);
            free(h_cands); free(results); free(dev_results);
            for (uint32_t i = 0; i < PATTERN_MAX; i++) mpz_clear(pats[i]);
            all_ok = 0;
            break;
        }
        cudaFree(d_cands);

        uint32_t mismatches = 0, gpu_primes = 0, gmp_primes = 0, false_neg = 0;
        uint32_t psp2 = 0;
        for (uint32_t i = 0; i < count; i++) {
            int cpu = cpu_mr_base2(pats[i]);
            if ((int)results[i] != cpu || (int)dev_results[i] != cpu) {
                mismatches++;
                if (mismatches <= 8) {
                    gmp_fprintf(stderr,
                                "  MISMATCH AL=%d i=%u H2D=%d device=%d "
                                "cpu=%d n=%Zd\n",
                                AL, i, results[i], dev_results[i], cpu, pats[i]);
                }
            }
            int gmp_p = mpz_probab_prime_p(pats[i], 25) > 0;
            gmp_primes += (gmp_p != 0);
            if (gmp_p && (!results[i] || !dev_results[i])) false_neg++;
            if (results[i] && !gmp_p) psp2++;
            gpu_primes += results[i] ? 1 : 0;
        }

        printf("  %s AL=%d: patterns=%u gpu-primes=%u gmp-primes=%u "
               "psp2=%u mismatches=%u false_negatives=%u\n",
               (mismatches == 0 && false_neg == 0) ? "OK " : "FAIL",
               AL, count, gpu_primes, gmp_primes, psp2, mismatches, false_neg);
        if (mismatches != 0 || false_neg != 0) all_ok = 0;
        total_psp2 += psp2;

        free(h_cands);
        free(results);
        free(dev_results);
        for (uint32_t i = 0; i < PATTERN_MAX; i++) mpz_clear(pats[i]);
    }

    if (all_ok && total_psp2) {
        printf("  note: %u hits are constructed base-2 strong pseudoprimes "
               "(2^307-1 and Phi_6(2^(bits/2))); base-2 MR must accept them "
               "exactly as the CPU reference does.\n", total_psp2);
    }
    gpu_fermat_destroy(ctx);
    return all_ok;
}

static int run_width_and_adapter_test(void) {
    printf("[TEST] width selection, window bounds and adapter batching\n");
    static const uint32_t bit_counts[] = {256, 257, 320, 321,
                                          512, 513, 640, 641,
                                          1792, 1793, 2048};
    int ok = 1;
    struct gpu_adapter *adapter = gpu_adapter_init_limbs(0, 17, 0);
    if (!adapter) {
        fprintf(stderr, "  FAIL: could not initialize adapter\n");
        return 0;
    }

#ifdef WITH_CGBN_FERMAT
    const char *narrow_env = getenv("GPU_FERMAT_NARROW_WIDTHS");
    const char *scalar_env = getenv("GPU_FERMAT_NO_CGBN");
    const char *kernel_env = getenv("GPU_MR_KERNEL");
    int scalar = scalar_env && *scalar_env && *scalar_env != '0' &&
                 *scalar_env != 'n' && *scalar_env != 'N' &&
                 *scalar_env != 'f' && *scalar_env != 'F';
    int cios = kernel_env && (kernel_env[0] == 'c' || kernel_env[0] == 'C') &&
               (kernel_env[1] == 'i' || kernel_env[1] == 'I');
    int narrow = !(narrow_env && narrow_env[0] == '0') && !scalar && !cios;
    int expected[] = {4, narrow ? 5 : 6, narrow ? 5 : 6, 6,
                      8, narrow ? 10 : 12, narrow ? 10 : 12, 12,
                      28, 32, 32};
#else
    int expected[] = {4, 6, 6, 6, 8, 10, 10, 12, 28, 30, 32};
#endif
    for (size_t i = 0; i < sizeof(bit_counts) / sizeof(bit_counts[0]); i++) {
        int want = expected[i] < GPU_NLIMBS ? expected[i] : GPU_NLIMBS;
        int actual = gpu_fermat_limbs_for_bits(bit_counts[i]);
        gpu_adapter_set_candidate_bits(adapter, bit_counts[i]);
        int adapter_limbs = gpu_adapter_get_limbs(adapter);
        if (actual != want || adapter_limbs != want) {
            fprintf(stderr, "  FAIL: %u bits: dispatch=%d adapter=%d expected=%d\n",
                    bit_counts[i], actual, adapter_limbs, want);
            ok = 0;
        }
    }

#ifdef WITH_CGBN_FERMAT
    if (!scalar && !cios && GPU_NLIMBS >= 10 &&
        (strcmp(gpu_fermat_kernel_label(5), "CGBN (320-bit, TPI=2)") != 0 ||
         strcmp(gpu_fermat_kernel_label(10), "CGBN (640-bit, TPI=4)") != 0)) {
        fprintf(stderr, "  FAIL: narrow CGBN widths use an invalid TPI\n");
        ok = 0;
    }
    if (scalar && !cios && GPU_NLIMBS >= 6 &&
        strcmp(gpu_fermat_kernel_label(6), "scalar (384-bit)") != 0) {
        fprintf(stderr, "  FAIL: disabled CGBN reported as active\n");
        ok = 0;
    }
    if (GPU_NLIMBS >= 32) {
        char expected_label[96];
        const char *tpi_env = getenv("GPU_FERMAT_TPI");
        int tpi = tpi_env ? atoi(tpi_env) : 0;
        if (tpi != 4 && tpi != 8 && tpi != 16 && tpi != 32) tpi = 8;
        if (cios) {
            snprintf(expected_label, sizeof(expected_label),
                     "CIOS 32-bit (2048-bit, 64 x 32-bit limbs, 1 thread/cand)");
        } else if (scalar) {
            snprintf(expected_label, sizeof(expected_label), "scalar (2048-bit)");
        } else {
            snprintf(expected_label, sizeof(expected_label),
                     "CGBN (2048-bit, TPI=%d)", tpi);
        }
        if (!gpu_fermat_cgbn_supports_limbs(32) ||
            strcmp(gpu_fermat_kernel_label(32), expected_label) != 0) {
            fprintf(stderr, "  FAIL: AL=32 capability/label does not match %s\n",
                    expected_label);
            ok = 0;
        }
    }
#endif

    static const int test_limbs[] = {5, 10, 32};
    mpz_t bound, period;
    mpz_inits(bound, period, NULL);
    for (size_t j = 0; j < sizeof(test_limbs) / sizeof(test_limbs[0]); j++) {
        int al = test_limbs[j];
        if (al > GPU_NLIMBS) continue;
        mpz_set_ui(bound, 1);
        mpz_mul_2exp(bound, bound, (unsigned long)al * 64UL);
        mpz_sub_ui(bound, bound, 6);
        if (!gpu_adapter_window_fits(bound, 6, al) ||
            gpu_adapter_window_fits(bound, 7, al)) {
            fprintf(stderr, "  FAIL: %d-limb window carry boundary\n", al);
            ok = 0;
        }
    }
    mpz_set_si(bound, -1);
    if (gpu_adapter_window_fits(bound, 1, 5)) {
        fprintf(stderr, "  FAIL: negative window accepted\n");
        ok = 0;
    }
    mpz_set_ui(period, 1);
    mpz_mul_2exp(period, period, 400);
    if (gpu_adapter_crt_candidate_bits(55, period, 4096) != 401) {
        fprintf(stderr, "  FAIL: CRT period omitted from candidate bound\n");
        ok = 0;
    }
    mpz_clears(bound, period, NULL);

    for (size_t j = 0; j < sizeof(test_limbs) / sizeof(test_limbs[0]); j++) {
        int al = test_limbs[j];
        if (al > GPU_NLIMBS) continue;
        enum { COUNT = 35 };
        mpz_t candidates[COUNT];
        uint8_t results[COUNT];
        for (int i = 0; i < COUNT; i++) {
            mpz_init_set_ui(candidates[i], 1);
            mpz_mul_2exp(candidates[i], candidates[i],
                         (unsigned long)al * 64UL - 2UL);
            mpz_add_ui(candidates[i], candidates[i], (unsigned long)(2 * i + 1));
            if (i % 7 == 0) mpz_nextprime(candidates[i], candidates[i]);
        }
        mpz_set_ui(candidates[COUNT - 1], 1);
        mpz_mul_2exp(candidates[COUNT - 1], candidates[COUNT - 1],
                     (unsigned long)al * 64UL);
        mpz_nextprime(candidates[COUNT - 1], candidates[COUNT - 1]);
        gpu_fermat_set_limbs(gpu_adapter_get_fermat_ctx(adapter), al);
        struct gpu_batch batch = {COUNT, candidates, results};
        if (gpu_adapter_test_batch(adapter, &batch) < 0) {
            fprintf(stderr, "  FAIL: %d-limb adapter batch\n", al);
            ok = 0;
        } else {
            for (int i = 0; i < COUNT; i++) {
                if (results[i] != cpu_mr_base2(candidates[i])) {
                    fprintf(stderr, "  FAIL: %d-limb adapter candidate %d\n",
                            al, i);
                    ok = 0;
                }
            }
        }
        for (int i = 0; i < COUNT; i++) mpz_clear(candidates[i]);
    }
    gpu_adapter_free(adapter);
    if (ok) printf("  OK: width boundaries, carry, fallback and batches\n");
    return ok;
}
#endif /* WITH_CUDA */

int main(int argc, char **argv) {
    int width_only = argc == 2 && strcmp(argv[1], "--width-only") == 0;
    if (argc > 1 && !width_only) {
        fprintf(stderr, "usage: %s [--width-only]\n", argv[0]);
        return 2;
    }
#ifndef WITH_CUDA
    (void)width_only;
    printf("test_gpu_fermat: SKIPPED (built without WITH_CUDA=1)\n");
    return 0;
#else
    printf("========== GPU Fermat Kernel Tests ==========\n\n");

    int ok = run_width_and_adapter_test();
    if (ok && !width_only)
        ok = run_gpu_fermat_correctness_test();
    /* Reference FIRST, then the internal H2D-vs-device consistency check: the
       device-path test aborts the chain on failure, and a failing consistency
       check used to hide whether the KERNEL or one of the two PATHS is wrong. */
    if (ok && !width_only)
        ok = run_gpu_fermat_pattern_test();
    if (ok && !width_only)
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
