/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "crt_runtime.h"

#include <stdlib.h>
#include <string.h>

static void mark_slot(uint8_t *bits, uint64_t slot) {
    bits[slot >> 3] |= (uint8_t)(1u << (slot & 7u));
}

static int slot_is_survivor(const uint8_t *bits, uint64_t slot) {
    return !(bits[slot >> 3] & (uint8_t)(1u << (slot & 7u)));
}

static int align_base(mpz_t nadd0, const mpz_t base0,
                      const struct crt_runtime *rt);

int crt_runtime_load(struct crt_runtime *rt, const char *path) {
    if (!rt || !path)
        return 0;
    memset(rt, 0, sizeof(*rt));

    FILE *f = fopen(path, "r");
    if (!f)
        return 0;

    char line[512];
    int n_primes = 0;
    double merit = 0.0;
    int shift = 0;
    uint64_t gap_target = 0;
    uint64_t n_candidates = 0;

    /* First pass: header fields. */
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "n_primes %d", &n_primes) == 1)
            continue;
        if (sscanf(line, "merit %lf", &merit) == 1)
            continue;
        if (sscanf(line, "shift %d", &shift) == 1)
            continue;
        if (sscanf(line, "gap_target %llu", (unsigned long long *)&gap_target) == 1)
            continue;
        if (sscanf(line, "n_candidates %llu", (unsigned long long *)&n_candidates) == 1)
            continue;
    }

    if (n_primes < 2 || gap_target < 2 || shift < 14) {
        fclose(f);
        return 0;
    }

    rt->n_primes = (uint32_t)n_primes;
    rt->merit = merit;
    rt->shift = (uint32_t)shift;
    rt->gap_target = gap_target;
    rt->n_candidates = n_candidates;
    rt->window = 2 * gap_target;
    if (rt->window < 10000)
        rt->window = 10000;

    rt->primes = (uint64_t *)calloc((size_t)n_primes, sizeof(uint64_t));
    rt->offsets = (uint64_t *)calloc((size_t)n_primes, sizeof(uint64_t));
    if (!rt->primes || !rt->offsets) {
        crt_runtime_free(rt);
        fclose(f);
        return 0;
    }

    /* Second pass: prime offset pairs. */
    rewind(f);
    size_t got = 0;
    while (fgets(line, sizeof(line), f) && got < (size_t)n_primes) {
        unsigned long long p = 0, o = 0;
        if (sscanf(line, "%llu %llu", &p, &o) == 2) {
            rt->primes[got] = (uint64_t)p;
            rt->offsets[got] = (uint64_t)(o % p);
            got++;
        }
    }
    fclose(f);

    if (got != (size_t)n_primes) {
        crt_runtime_free(rt);
        return 0;
    }

    mpz_init_set_ui(rt->primorial, 1);
    for (uint32_t i = 0; i < rt->n_primes; i++) {
        if (rt->offsets[i] != 0)
            mpz_mul_ui(rt->primorial, rt->primorial, rt->primes[i]);
    }

    return crt_runtime_build_template(rt);
}

void crt_runtime_free(struct crt_runtime *rt) {
    if (!rt)
        return;
    free(rt->primes);
    free(rt->offsets);
    free(rt->template);
    mpz_clear(rt->primorial);
    rt->primes = NULL;
    rt->offsets = NULL;
    rt->template = NULL;
}

int crt_runtime_set_window(struct crt_runtime *rt, uint64_t window) {
    if (!rt || window < 2)
        return 0;
    rt->window = window;
    return crt_runtime_build_template(rt);
}

int crt_runtime_build_template(struct crt_runtime *rt) {
    if (!rt || !rt->primes || !rt->offsets || rt->window < 2)
        return 0;

    /* Base parity: candidate = base0 + nadd0 - adj, where adj = nadd0(base0=0)
     * mod 2.  The parity is independent of the block header (base0 is even),
     * so the static template can bake it in once. */
    mpz_t n0;
    mpz_init(n0);
    {
        mpz_t zero;
        mpz_init_set_ui(zero, 0);
        align_base(n0, zero, rt);
        mpz_clear(zero);
    }
    rt->adj = (uint32_t)(mpz_fdiv_ui(n0, 2));
    mpz_clear(n0);

    uint64_t odd_slots = rt->window / 2;   /* odd offsets 1,3,5,...,window-1 */
    rt->template_words = (odd_slots + 63) / 64;
    uint8_t *bits = (uint8_t *)calloc((size_t)rt->template_words * 8, 1);
    if (!bits)
        return 0;
    free(rt->template);
    rt->template = bits;

    uint64_t adj = rt->adj;
    for (uint32_t i = 0; i < rt->n_primes; i++) {
        uint64_t p = rt->primes[i];
        uint64_t o = rt->offsets[i];
        if (p == 2) {
            /* candidate is even; odd offset t keeps candidate+t odd. */
            continue;
        }
        /* candidate + t ≡ 0 (mod p)  <=>  t ≡ (o + adj) (mod p).
           candidate ≡ -(o + adj) (mod p) by construction. */
        uint64_t t = (o + adj) % p;
        if (t == 0)
            t = p;
        if ((t & 1u) == 0)
            t += p;   /* p odd => flips parity */
        for (; t < rt->window; t += 2 * p)
            mark_slot(bits, (t - 1) / 2);
    }

    uint64_t survivors = 0;
    for (uint64_t s = 0; s < odd_slots; s++)
        if (slot_is_survivor(bits, s))
            survivors++;
    rt->n_survivors = survivors;
    return 1;
}

/* Core incremental CRT: solve nadd0 ≡ -(base0 + offsets[i]) (mod primes[i])
 * for every prime with offset != 0. */
static int align_base(mpz_t nadd0, const mpz_t base0,
                      const struct crt_runtime *rt) {
    mpz_set_ui(nadd0, 0);
    mpz_t M, Mmod, inv, term;
    mpz_init_set_ui(M, 1);
    mpz_inits(Mmod, inv, term, NULL);

    for (uint32_t i = 0; i < rt->n_primes; i++) {
        uint64_t p = rt->primes[i];
        uint64_t o = rt->offsets[i];
        if (o == 0)
            continue;   /* offset 0 primes are excluded from the primorial */

        /* nAdd0 ≡ -(base0 + o) (mod p) */
        uint64_t base_mod_p = mpz_fdiv_ui(base0, p);
        uint64_t sum = (base_mod_p + o) % p;
        uint64_t target_r = (sum == 0) ? 0 : (p - sum);

        uint64_t curr_r = mpz_fdiv_ui(nadd0, p);
        uint64_t diff = (target_r + p - curr_r) % p;

        uint64_t M_mod_p = mpz_fdiv_ui(M, p);
        mpz_set_ui(Mmod, M_mod_p);
        mpz_set_ui(term, p);
        mpz_invert(inv, Mmod, term);        /* inv = (M mod p)^-1 mod p */
        uint64_t k = (diff * mpz_get_ui(inv)) % p;

        mpz_set_ui(term, k);
        mpz_addmul(nadd0, term, M);         /* nadd0 += k * M */
        mpz_mul_ui(M, M, p);
    }

    mpz_clears(M, Mmod, inv, term, NULL);
    return 0;
}

int crt_runtime_align(mpz_t nadd0, const uint8_t h256[32], uint32_t shift,
                      const struct crt_runtime *rt) {
    if (!rt || !h256 || !nadd0)
        return -1;

    /* base0 = h256 << shift (256-bit big-endian into mpz). */
    mpz_t base0;
    mpz_init(base0);
    mpz_set_ui(base0, 0);
    for (size_t i = 0; i < 32; i++) {
        mpz_mul_2exp(base0, base0, 8);
        mpz_add_ui(base0, base0, h256[i]);
    }
    mpz_mul_2exp(base0, base0, shift);

    int rc = align_base(nadd0, base0, rt);
    mpz_clear(base0);
    return rc;
}

uint64_t crt_runtime_survivors(const struct crt_runtime *rt,
                               uint64_t *survivor_offsets) {
    if (!rt || !rt->template || !survivor_offsets)
        return 0;
    uint64_t odd_slots = rt->window / 2;
    uint64_t n = 0;
    for (uint64_t s = 0; s < odd_slots; s++) {
        if (slot_is_survivor(rt->template, s))
            survivor_offsets[n++] = 2 * s + 1;
    }
    return n;
}
