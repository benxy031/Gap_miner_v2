/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Exactness test for the GPU-side hidden-class resolution
 * (crt_gpu_resolve_gap_ex vs the CPU-only halfclass_resolve_gap_ex).
 *
 * The property: on an identical synthetic gap the two paths must emit the
 * IDENTICAL set of consecutive-prime pairs.  The GPU path pre-filters the
 * hidden candidates with a base-2+3 MR batch; MR has no false negatives, so
 * a real interior hidden prime must always survive — if a bug (e.g. a limb
 * stride mismatch in the MR input buffer) makes the GPU miss an interior
 * prime, the emitted set differs and this test fails.
 *
 * Self-skips at runtime when built without WITH_CUDA or when no CUDA device
 * is present.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

#include "../new_src/halfclass.h"
#include "../new_src/gap_detection.h"

#ifdef WITH_CUDA
#include "../new_src/gpu/gpu_fermat.h"
#include "../new_src/worker_gpu.h"

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

struct pair {
    uint64_t a, b;
};

static int pair_cmp(const void *x, const void *y) {
    const struct pair *p = (const struct pair *)x;
    const struct pair *q = (const struct pair *)y;
    if (p->a != q->a) return p->a < q->a ? -1 : 1;
    if (p->b != q->b) return p->b < q->b ? -1 : 1;
    return 0;
}

static void collect(const struct gap_result *gaps, uint32_t n,
                    struct pair *dst, uint32_t *cnt) {
    for (uint32_t i = 0; i < n; i++) {
        dst[*cnt].a = gaps[i].offset_p1;
        dst[*cnt].b = gaps[i].offset_p2;
        (*cnt)++;
    }
}

static void run_case(struct gpu_fermat_ctx *ctx, mpz_t base) {
    static struct pair cpu_pairs[256], gpu_pairs[256];
    uint32_t nc = 0, ng = 0;

    /* A chain of 30 consecutive primes from base+20000: the interior
       primes split into visible and hidden classes; the resolve must find
       the hidden ones. */
    mpz_t p, nxt;
    mpz_init(p);
    mpz_init(nxt);
    mpz_add_ui(p, base, 20000);
    mpz_nextprime(p, p);
    uint64_t off_a = 0, off_b = 0;
    mpz_t t;
    mpz_init(t);
    mpz_sub(t, p, base);
    off_a = mpz_get_ui(t);
    for (int i = 0; i < 29; i++) {
        mpz_set(nxt, p);
        mpz_add_ui(nxt, nxt, 1);
        mpz_nextprime(nxt, nxt);
        mpz_set(p, nxt);
    }
    mpz_sub(t, p, base);
    off_b = mpz_get_ui(t);
    uint64_t owned = off_b + 1000;   /* every endpoint owned */
    mpz_clear(t);

    struct gap_result *gcpu = NULL, *ggpu = NULL;
    uint32_t c1 = 0, c2 = 0;

    CHECK(halfclass_resolve_gap_ex(base, off_a, off_b, owned, 0.0, NULL,
                                   &gcpu, &c1) != 0);
    CHECK(crt_gpu_resolve_gap_ex(ctx, 0, base, off_a, off_b, owned, 0.0,
                                 NULL, &ggpu, &c2) != 0);
    collect(gcpu, c1, cpu_pairs, &nc);
    collect(ggpu, c2, gpu_pairs, &ng);

    qsort(cpu_pairs, nc, sizeof(*cpu_pairs), pair_cmp);
    qsort(gpu_pairs, ng, sizeof(*gpu_pairs), pair_cmp);

    CHECK(nc == ng);
    uint32_t lim = nc < ng ? nc : ng;
    CHECK(nc > 0);
    CHECK(memcmp(cpu_pairs, gpu_pairs,
                 (size_t)lim * sizeof(struct pair)) == 0);
    printf("gpu-resolve parity: cpu=%u gpu=%u %s\n", nc, ng,
           (nc == ng &&
            memcmp(cpu_pairs, gpu_pairs, (size_t)lim * sizeof(struct pair)) == 0)
               ? "MATCH"
               : "MISMATCH");

    gap_detection_free_results(gcpu);
    gap_detection_free_results(ggpu);
    mpz_clear(p);
    mpz_clear(nxt);
}

int main(void) {
#ifndef WITH_CUDA
    printf("test_gpu_resolve: SKIP (built without WITH_CUDA)\n");
    return 0;
#else
    if (gpu_fermat_device_count() <= 0) {
        printf("test_gpu_resolve: SKIP (no CUDA device)\n");
        return 0;
    }
    struct gpu_fermat_ctx *ctx = gpu_fermat_init(0, 8192);
    if (!ctx) {
        printf("test_gpu_resolve: SKIP (GPU init failed)\n");
        return 0;
    }
    /* Match the production shift-507 configuration: 12 active limbs. */
    gpu_fermat_set_limbs(ctx, 12);

    mpz_t base;
    mpz_init(base);
    for (uint64_t variant = 0; variant < 3; variant++) {
        mpz_set_ui(base, 1);
        mpz_mul_2exp(base, base, 751);   /* ~763-bit, production scale */
        mpz_add_ui(base, base, variant * 1000000ULL);
        if (mpz_odd_p(base)) mpz_add_ui(base, base, 1);
        run_case(ctx, base);
    }
    mpz_clear(base);
    gpu_fermat_destroy(ctx);

    if (failures) {
        printf("test_gpu_resolve FAILED (%d failures)\n", failures);
        return 1;
    }
    printf("All gpu-resolve tests PASSED\n");
    return 0;
#endif
}
#endif /* WITH_CUDA */
