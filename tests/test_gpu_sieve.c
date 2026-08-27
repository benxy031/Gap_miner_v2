/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Parity test for the fused-pipeline Stage 1 GPU extract+pack kernel.
 *
 * The GPU path (gpu_sieve_mark_from_base + gpu_sieve_extract_pack) must
 * produce the exact same survivor set as a CPU reference that uses the
 * identical arithmetic (GMP for base mod p, then odd-slot marking and
 * extraction).  Three levels are compared per window:
 *
 *   1. the marked bitmap itself (fail-closed intermediate check),
 *   2. the multiset of survivor offsets (sorted memcmp),
 *   3. every packed candidate limb == (base + offset) computed with GMP.
 *
 * Any extra survivor (spurious candidate) is the dangerous direction: the
 * trailing-word mask in both the CPU and GPU extractors must agree exactly,
 * otherwise the GPU would feed composites to the Miller-Rabin kernel.
 *
 * Built and linked unconditionally, but self-skips at runtime when built
 * without WITH_CUDA=1 (there is no GPU kernel to exercise).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

#include "../new_src/gpu/gpu_sieve.h"

#ifdef WITH_CUDA

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

/* Eratosthenes sieve, includes prime 2 at index 0 (mirrors sieve_core). */
static size_t gen_primes(uint64_t limit, uint64_t *primes, size_t cap) {
    if (limit < 2U) return 0;
    unsigned char *is = (unsigned char *)calloc((size_t)limit + 1U, 1);
    if (!is) return 0;
    size_t count = 0;
    for (uint64_t i = 2; i <= limit && count < cap; i++) {
        if (!is[i]) {
            primes[count++] = i;
            for (uint64_t j = i * i; j <= limit; j += i) is[j] = 1U;
        }
    }
    free(is);
    return count;
}

/* Convert mpz to little-endian 64-bit limbs; zero-fills up to max_limbs.
   Returns the number of non-zero limbs written. */
static size_t mpz_to_limbs(const mpz_t v, uint64_t *limbs, size_t max_limbs) {
    memset(limbs, 0, max_limbs * sizeof(*limbs));
    size_t n = 0;
    mpz_export(limbs, &n, -1, sizeof(uint64_t), 0, 0, v);
    return n;
}

/* CPU reference: mark odd-slot composites exactly like the GPU mark kernel.
   base_mod_p[i] is already base mod p; base_offset is always 0 here. */
static void cpu_mark_odd(const uint64_t *primes, size_t prime_count,
                         const uint64_t *base_mod_p,
                         uint64_t odd_count, uint64_t first_odd_offset,
                         uint64_t *bitmap) {
    for (size_t i = 0; i < prime_count; i++) {
        uint64_t p = primes[i];
        if (p < 3U) continue;

        uint64_t remainder = base_mod_p[i];
        remainder += first_odd_offset % p;
        if (remainder >= p) remainder -= p;

        uint64_t inverse_two = (p + 1U) >> 1;
        uint64_t t = (remainder == 0U) ? 0U : (p - remainder);
        uint64_t pos = (t * inverse_two) % p; /* product < 2^63, no overflow */

        for (; pos < odd_count; pos += p) {
            bitmap[pos >> 6] |= 1ULL << (pos & 63U);
        }
    }
}

/* CPU reference: extract survivors as full adder offsets. */
static size_t cpu_extract(const uint64_t *bitmap, uint64_t bitmap_words,
                          uint64_t odd_count, uint64_t first_odd_offset,
                          uint64_t *offsets) {
    size_t count = 0;
    for (uint64_t w = 0; w < bitmap_words; w++) {
        uint64_t survivors = ~bitmap[w];
        if (w + 1U == bitmap_words && (odd_count & 63U) != 0) {
            survivors &= (1ULL << (odd_count & 63U)) - 1ULL;
        }
        while (survivors) {
            unsigned int b = (unsigned int)__builtin_ctzll(survivors);
            offsets[count++] = first_odd_offset + (((w << 6) + b) << 1);
            survivors &= survivors - 1ULL;
        }
    }
    return count;
}

/* Verify every packed AoS candidate equals base + offset (GMP ground truth). */
static int verify_candidates(const mpz_t base, int active_limbs,
                             const uint64_t *cands_aos,
                             const uint64_t *offsets, unsigned int count) {
    mpz_t c, off;
    uint64_t limb[32];
    mpz_init(c);
    mpz_init(off);

    for (unsigned int s = 0; s < count; s++) {
        mpz_set_ui(off, offsets[s]);
        mpz_add(c, base, off);

        size_t n = 0;
        mpz_export(limb, &n, -1, sizeof(uint64_t), 0, 0, c);
        if (n > (size_t)active_limbs) n = (size_t)active_limbs;

        for (int i = 0; i < active_limbs; i++) {
            uint64_t expected = (i < (int)n) ? limb[i] : 0ULL;
            if (cands_aos[(size_t)s * (size_t)active_limbs + (size_t)i] !=
                expected) {
                fprintf(stderr,
                        "  FAIL: candidate slot %u limb %d mismatch: "
                        "got 0x%016llx expected 0x%016llx (base + offset %llu)\n",
                        s, i,
                        (unsigned long long)cands_aos[(size_t)s *
                                                          (size_t)active_limbs +
                                                      (size_t)i],
                        (unsigned long long)expected,
                        (unsigned long long)offsets[s]);
                mpz_clear(c);
                mpz_clear(off);
                return 0;
            }
        }
    }

    mpz_clear(c);
    mpz_clear(off);
    return 1;
}

static int run_one_window(gpu_sieve_ctx *ctx, const mpz_t base, int limbs,
                          uint64_t interval_size, uint64_t first_odd_offset,
                          const uint64_t *primes, size_t prime_count,
                          const uint64_t *inv_p, const char *label) {
    uint64_t odd_count = (interval_size - first_odd_offset + 1U) >> 1;
    uint64_t bitmap_words = (odd_count + 63U) >> 6;

    uint64_t base_limbs[32];
    mpz_to_limbs(base, base_limbs, 32);

    uint64_t *base_mod_p = (uint64_t *)malloc(prime_count * sizeof(uint64_t));
    uint64_t *cpu_bitmap =
        (uint64_t *)calloc(bitmap_words ? bitmap_words : 1, sizeof(uint64_t));
    uint64_t *gpu_bitmap =
        (uint64_t *)calloc(bitmap_words ? bitmap_words : 1, sizeof(uint64_t));
    uint64_t *cpu_offsets = (uint64_t *)malloc(odd_count * sizeof(uint64_t));
    uint64_t *gpu_offsets = (uint64_t *)malloc(odd_count * sizeof(uint64_t));
    uint64_t *gpu_cands =
        (uint64_t *)malloc((size_t)odd_count * (size_t)limbs * sizeof(uint64_t));
    if (!base_mod_p || !cpu_bitmap || !gpu_bitmap || !cpu_offsets ||
        !gpu_offsets || !gpu_cands) {
        fprintf(stderr, "  FAIL: out of memory (%s)\n", label);
        free(base_mod_p); free(cpu_bitmap); free(gpu_bitmap);
        free(cpu_offsets); free(gpu_offsets); free(gpu_cands);
        return 0;
    }

    for (size_t i = 0; i < prime_count; i++) {
        base_mod_p[i] = mpz_fdiv_ui(base, primes[i]);
    }

    /* CPU reference. */
    cpu_mark_odd(primes, prime_count, base_mod_p, odd_count, first_odd_offset,
                 cpu_bitmap);
    size_t cpu_count = cpu_extract(cpu_bitmap, bitmap_words, odd_count,
                                   first_odd_offset, cpu_offsets);

    /* GPU: mark from base (device bitmap), then extract+pack. */
    int ok = gpu_sieve_mark_from_base(ctx, odd_count, first_odd_offset,
                                      base_limbs, limbs, 0, primes, inv_p,
                                      prime_count, gpu_bitmap,
                                      (size_t)bitmap_words);
    if (!ok) {
        fprintf(stderr, "  FAIL: gpu_sieve_mark_from_base returned 0 (%s)\n",
                label);
        goto fail;
    }

    unsigned int gpu_count = 0;
    ok = gpu_sieve_extract_pack(ctx, odd_count, first_odd_offset, base_limbs,
                                limbs, gpu_cands, gpu_offsets, &gpu_count,
                                0, UINT64_MAX, 0);
    if (!ok) {
        fprintf(stderr, "  FAIL: gpu_sieve_extract_pack returned 0 (%s)\n",
                label);
        goto fail;
    }

    /* Level 1: bitmap equality. */
    if (memcmp(cpu_bitmap, gpu_bitmap,
               (size_t)bitmap_words * sizeof(uint64_t)) != 0) {
        fprintf(stderr, "  FAIL: bitmap mismatch (%s)\n", label);
        for (uint64_t w = 0; w < bitmap_words; w++) {
            if (cpu_bitmap[w] != gpu_bitmap[w]) {
                fprintf(stderr, "        first diff word %llu: cpu=0x%016llx "
                                "gpu=0x%016llx\n",
                        (unsigned long long)w,
                        (unsigned long long)cpu_bitmap[w],
                        (unsigned long long)gpu_bitmap[w]);
                break;
            }
        }
        goto fail;
    }

    /* Level 2: survivor count + multiset equality. */
    if (gpu_count != (unsigned int)cpu_count) {
        fprintf(stderr, "  FAIL: survivor count mismatch (%s): gpu=%u cpu=%zu\n",
                label, gpu_count, cpu_count);
        goto fail;
    }

    /* Level 3: packed candidates == base + offset.  Must run BEFORE gpu_offsets
       is sorted, because candidate slot s corresponds to gpu_offsets[s] in the
       original (atomicAdd) slot order. */
    if (!verify_candidates(base, limbs, gpu_cands, gpu_offsets, gpu_count)) {
        fprintf(stderr, "        (candidate packing mismatch, %s)\n", label);
        goto fail;
    }

    qsort(cpu_offsets, cpu_count, sizeof(uint64_t), cmp_u64);
    qsort(gpu_offsets, gpu_count, sizeof(uint64_t), cmp_u64);
    if (memcmp(cpu_offsets, gpu_offsets,
               cpu_count * sizeof(uint64_t)) != 0) {
        fprintf(stderr, "  FAIL: survivor offset mismatch (%s)\n", label);
        goto fail;
    }

    printf("  OK  %s: %zu survivors, bitmap+offsets+candidates match\n", label,
           cpu_count);

    free(base_mod_p); free(cpu_bitmap); free(gpu_bitmap);
    free(cpu_offsets); free(gpu_offsets); free(gpu_cands);
    return 1;

fail:
    free(base_mod_p); free(cpu_bitmap); free(gpu_bitmap);
    free(cpu_offsets); free(gpu_offsets); free(gpu_cands);
    return 0;
}

static int run_gpu_sieve_parity_test(void) {
    printf("[TEST] GPU sieve extract+pack parity vs CPU reference...\n");

    int device_id = 0;
    gpu_sieve_ctx *ctx = gpu_sieve_init(device_id, 4096, 1U << 16);
    if (!ctx) {
        fprintf(stderr, "  SKIP: no CUDA device available or init failed\n");
        return 1;
    }
    printf("  using device: %s\n", gpu_sieve_device_name(ctx));

    /* Fixed prime table shared by all windows (uploaded once on-device). */
    const uint64_t prime_limit = 20000U;
    const size_t prime_cap = 4096;
    uint64_t *primes = (uint64_t *)malloc(prime_cap * sizeof(uint64_t));
    uint64_t *inv_p = (uint64_t *)malloc(prime_cap * sizeof(uint64_t));
    if (!primes || !inv_p) {
        fprintf(stderr, "  FAIL: out of memory\n");
        gpu_sieve_destroy(ctx);
        free(primes); free(inv_p);
        return 0;
    }
    size_t prime_count = gen_primes(prime_limit, primes, prime_cap);
    for (size_t i = 0; i < prime_count; i++) {
        inv_p[i] = (primes[i] >= 3U) ? (UINT64_MAX / primes[i]) : 0U;
    }
    printf("  prime table: %zu primes up to %llu\n", prime_count,
           (unsigned long long)prime_limit);

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, 0x5EED5EEDu);

    mpz_t base;
    mpz_init(base);

    int pass = 1;
    char label[64];

    static const int limb_cases[] = {1, 2, 6, 12, 20};
    static const uint64_t interval_cases[] = {16384U, 1U << 16};

    for (size_t li = 0; li < sizeof(limb_cases) / sizeof(limb_cases[0]); li++) {
        int limbs = limb_cases[li];
        for (int parity = 0; parity < 2; parity++) {
            uint64_t interval = interval_cases[li % 2];
            /* Random base with headroom: top limb limited to 32 bits so
               base + offset never overflows `limbs` limbs. */
            mpz_urandomb(base, rng, (unsigned long)64 * (unsigned long)limbs - 32UL);
            if (limbs > 1) mpz_setbit(base, (unsigned long)64 * (unsigned long)(limbs - 1) - 1UL);
            if (parity) mpz_setbit(base, 0); else mpz_clrbit(base, 0);

            uint64_t first_odd_offset = mpz_tstbit(base, 0) ? 0U : 1U;
            snprintf(label, sizeof(label),
                     "limbs=%d parity=%s window=%llu",
                     limbs, parity ? "odd" : "even",
                     (unsigned long long)interval);

            if (!run_one_window(ctx, base, limbs, interval, first_odd_offset,
                                primes, prime_count, inv_p, label)) {
                pass = 0;
                break;
            }
        }
        if (!pass) break;
    }

    mpz_clear(base);
    gmp_randclear(rng);
    gpu_sieve_destroy(ctx);
    free(primes);
    free(inv_p);

    return pass;
}

#endif /* WITH_CUDA */

int main(void) {
#ifndef WITH_CUDA
    printf("[TEST] GPU sieve extract+pack parity: SKIP (built without WITH_CUDA)\n");
    return 0;
#else
    return run_gpu_sieve_parity_test() ? 0 : 1;
#endif
}
