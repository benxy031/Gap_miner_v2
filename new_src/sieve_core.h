/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Sieve Core: Segmented Sieve of Eratosthenes
 *
 * Bitmap-based sieve with:
 * - 64-bit word packing (8x memory reduction)
 * - Trial division up to sqrt(interval)
 * - Candidate offset extraction
 */

#ifndef SIEVE_CORE_H
#define SIEVE_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Sieve configuration and state */
struct sieve_core {
    /* Interval bounds */
    mpz_t primorial;             /* Legacy name for the configured interval length */
    uint64_t interval_size;      /* Candidate offsets in [base, base + interval_size) */

    /* Sieve bitmap (bit-packed, 1 word = 64 bits) */
    uint64_t *bitmap;            /* bit=1 → composite, bit=0 → candidate prime */
    uint64_t bitmap_words;       /* Number of 64-bit words needed */
    uint64_t *candidate_buffer;  /* Reusable extracted candidate offsets */
    uint64_t candidate_capacity; /* Capacity of candidate_buffer in offsets */

    /* Prime sieve factors */
    const uint64_t *small_primes;     /* Primes to sieve with */
    size_t small_primes_count;        /* Number of sieving primes */
    uint64_t prime_limit;             /* Upper bound for sieving */
    int owns_small_primes;            /* 1 when init_window allocated the table */

    /* Optional: base_mod_p cache for repeated nonces */
    uint64_t *base_mod_p;        /* base mod p for each sieving prime */
    int base_mod_p_valid;        /* 1 if cache is current */

    /* Reciprocal table for branchless mod (inv_p[i] = floor(2^64 / p)).
       Replaces the per-prime 64-bit hardware division in the hot marking
       setup with a 128-bit multiply + shift (~4-5x faster on x86-64). */
    uint64_t *inv_p;             /* reciprocal for each sieving prime */
    int avx2_enabled;            /* Runtime-selected AVX2 marking/extraction path */
};

/* Bitmap access macros (64-bit word, LSB = index 0) */
#define SIEVE_IS_CANDIDATE(bitmap, idx) (!((bitmap)[(idx) >> 6] & (1ULL << ((idx) & 0x3f))))
#define SIEVE_MARK_COMPOSITE(bitmap, idx) ((bitmap)[(idx) >> 6] |= (1ULL << ((idx) & 0x3f)))
#define SIEVE_CLEAR(bitmap, idx) ((bitmap)[(idx) >> 6] &= ~(1ULL << ((idx) & 0x3f)))

/* Initialize sieve with primorial and prime table */
int sieve_core_init(struct sieve_core *sc, mpz_t primorial, 
                    const uint64_t *small_primes, size_t count, 
                    uint64_t prime_limit);

/* Initialize a fixed segmented-sieve window without CRT metadata. */
int sieve_core_init_window(struct sieve_core *sc, uint64_t interval_size,
                           uint64_t prime_limit);

/* Sieve interval [base, base+primorial) and return borrowed candidate offsets. */
int sieve_core_run(struct sieve_core *sc, mpz_t base, 
                   uint64_t **out_candidates, uint32_t *out_count);

/* Cache header-base residues for subsequent non-CRT windows. */
int sieve_core_prepare_base_mod_p(struct sieve_core *sc, const mpz_t base);

/* Cache header-base residues only for the prime range [start, end).
   Used by the GPU-sieve path, where high-prime residues are computed on the
   GPU and only the low-prime slice is needed on the host. */
int sieve_core_prepare_base_mod_p_range(struct sieve_core *sc, const mpz_t base,
                                        size_t start, size_t end);

/* Sieve a window at base_offset from the residue-cached header base. */
int sieve_core_run_from_cached_base(struct sieve_core *sc, uint64_t base_offset,
                                    uint64_t **out_candidates,
                                    uint32_t *out_count);

/* Hybrid cached-base sieve path:
     - premarked_bitmap contains high-prime composite marks for odd slots
     - CPU marks low-prime range [0, split_index)
     - candidate extraction remains odd-only
     premarked_words must be at least ceil(odd_slot_count / 64).
 */
int sieve_core_run_from_cached_base_hybrid(struct sieve_core *sc,
                                                                                     uint64_t base_offset,
                                                                                     const uint64_t *premarked_bitmap,
                                                                                     uint64_t premarked_words,
                                                                                     size_t split_index,
                                                                                     uint64_t **out_candidates,
                                                                                     uint32_t *out_count);

/* Get candidate at index (offset from base) */
uint64_t sieve_core_candidate(const struct sieve_core *sc, uint32_t idx);

/* Return the active CPU path for startup telemetry. */
const char *sieve_core_simd_mode(const struct sieve_core *sc);

/* Cleanup */
void sieve_core_free(struct sieve_core *sc);

#endif /* SIEVE_CORE_H */
