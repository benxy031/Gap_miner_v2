/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "crt_set.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CRT_PRIME_LIMIT 10000U   /* enough for shifts far beyond 1024 */

static uint64_t s_prime_cache[2048];
static uint32_t s_prime_count = 0;

/* Generate the first primes once (sieve of Eratosthenes). */
static void crt_generate_primes(void) {
    if (s_prime_count != 0) {
        return;
    }
    uint8_t *composite = (uint8_t *)calloc(CRT_PRIME_LIMIT, 1);
    if (!composite) {
        return;
    }
    for (uint64_t f = 2; f * f < CRT_PRIME_LIMIT; f++) {
        if (composite[f]) continue;
        for (uint64_t m = f * f; m < CRT_PRIME_LIMIT; m += f) {
            composite[m] = 1;
        }
    }
    for (uint64_t v = 2; v < CRT_PRIME_LIMIT && s_prime_count < 2048; v++) {
        if (!composite[v]) {
            s_prime_cache[s_prime_count++] = v;
        }
    }
    free(composite);
}

uint32_t crt_set_primes_for_shift(uint32_t shift) {
    crt_generate_primes();
    if (s_prime_count == 0) {
        return 0;
    }

    mpz_t P, limit;
    mpz_init_set_ui(P, 1);
    mpz_init_set_ui(limit, 1);
    mpz_mul_2exp(limit, limit, shift);   /* 2^shift */

    uint32_t n = 0;
    while (n < s_prime_count) {
        mpz_mul_ui(P, P, s_prime_cache[n]);
        if (mpz_cmp(P, limit) >= 0) {
            break;
        }
        n++;
    }

    mpz_clear(P);
    mpz_clear(limit);
    return n;
}

int crt_set_init(struct crt_set *cs, uint32_t shift, uint64_t size,
                 uint32_t n_primes) {
    if (!cs || shift < 14 || size < 2 || n_primes == 0) {
        return 0;
    }
    crt_generate_primes();
    if (n_primes > s_prime_count) {
        return 0;
    }

    memset(cs, 0, sizeof(*cs));
    cs->shift = shift;
    cs->size = size;
    cs->n_primes = n_primes;

    cs->primes = (uint64_t *)malloc((size_t)n_primes * sizeof(uint64_t));
    if (!cs->primes) {
        return 0;
    }
    memcpy(cs->primes, s_prime_cache, (size_t)n_primes * sizeof(uint64_t));

    mpz_init_set_ui(cs->primorial, 1);
    for (uint32_t i = 0; i < n_primes; i++) {
        mpz_mul_ui(cs->primorial, cs->primorial, cs->primes[i]);
    }
    mpz_init_set_ui(cs->offset, 0);

    cs->bit_size = (uint64_t)mpz_sizeinbase(cs->primorial, 2);
    cs->bitmap_words = (size + 63U) >> 6;
    cs->bitmap = (uint64_t *)calloc(cs->bitmap_words, sizeof(uint64_t));
    if (!cs->bitmap) {
        free(cs->primes);
        cs->primes = NULL;
        mpz_clear(cs->primorial);
        mpz_clear(cs->offset);
        return 0;
    }

    long double survival = 1.0L;
    for (uint32_t i = 0; i < n_primes; i++) {
        survival *= (1.0L - 1.0L / (long double)cs->primes[i]);
    }
    cs->avg_candidates = (double)((long double)size * survival);

    crt_set_resieve(cs);
    return 1;
}

void crt_set_free(struct crt_set *cs) {
    if (!cs) return;
    free(cs->primes);
    cs->primes = NULL;
    mpz_clear(cs->primorial);
    mpz_clear(cs->offset);
    free(cs->bitmap);
    cs->bitmap = NULL;
}

uint64_t crt_set_resieve(struct crt_set *cs) {
    if (!cs || !cs->bitmap) {
        return 0;
    }
    memset(cs->bitmap, 0, (size_t)cs->bitmap_words * sizeof(uint64_t));

    for (uint32_t i = 0; i < cs->n_primes; i++) {
        uint64_t p = cs->primes[i];
        uint64_t off_mod = mpz_fdiv_ui(cs->offset, p);
        uint64_t start = (p - off_mod) % p;
        for (uint64_t t = start; t < cs->size; t += p) {
            cs->bitmap[t >> 6] |= 1ULL << (t & 63U);
        }
    }

    uint64_t survivors = 0;
    for (uint64_t t = 0; t < cs->size; t++) {
        if (!(cs->bitmap[t >> 6] & (1ULL << (t & 63U)))) {
            survivors++;
        }
    }
    cs->n_candidates = survivors;
    return survivors;
}

int crt_set_search_offset(struct crt_set *cs, uint64_t trials, uint64_t seed) {
    if (!cs || trials == 0) {
        return -1;
    }

    gmp_randstate_t state;
    gmp_randinit_default(state);
    gmp_randseed_ui(state, seed);

    mpz_t candidate, best;
    mpz_init(candidate);
    mpz_init_set(best, cs->offset);

    uint64_t best_survivors = UINT64_MAX;
    for (uint64_t i = 0; i < trials; i++) {
        mpz_urandomm(candidate, state, cs->primorial);
        mpz_set(cs->offset, candidate);
        uint64_t s = crt_set_resieve(cs);
        if (s < best_survivors) {
            best_survivors = s;
            mpz_set(best, candidate);
        }
    }

    mpz_set(cs->offset, best);
    crt_set_resieve(cs);

    mpz_clear(candidate);
    mpz_clear(best);
    gmp_randclear(state);
    return 0;
}

int crt_set_save(const struct crt_set *cs, const char *path) {
    if (!cs || !path) return -1;
    FILE *f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "|== crt_set ==|\n");
    fprintf(f, "shift: %u\n", cs->shift);
    fprintf(f, "n_primes: %u\n", cs->n_primes);
    fprintf(f, "size: %llu\n", (unsigned long long)cs->size);
    fprintf(f, "offset: ");
    mpz_out_str(f, 10, cs->offset);
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

int crt_set_load(struct crt_set *cs, const char *path) {
    if (!cs || !path) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[256];
    uint32_t shift = 0, n_primes = 0;
    uint64_t size = 0;
    char offset_str[2048] = {0};

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "shift: ", 7) == 0) {
            shift = (uint32_t)strtoul(line + 7, NULL, 10);
        } else if (strncmp(line, "n_primes: ", 10) == 0) {
            n_primes = (uint32_t)strtoul(line + 10, NULL, 10);
        } else if (strncmp(line, "size: ", 6) == 0) {
            size = (uint64_t)strtoull(line + 6, NULL, 10);
        } else if (strncmp(line, "offset: ", 8) == 0) {
            char *s = line + 8;
            s[strcspn(s, "\r\n")] = 0;
            snprintf(offset_str, sizeof(offset_str), "%s", s);
        }
    }
    fclose(f);

    if (shift == 0 || n_primes == 0 || size == 0 || offset_str[0] == 0) {
        return -1;
    }
    if (!crt_set_init(cs, shift, size, n_primes)) {
        return -1;
    }
    if (mpz_set_str(cs->offset, offset_str, 10) != 0) {
        crt_set_free(cs);
        return -1;
    }
    crt_set_resieve(cs);
    return 0;
}
