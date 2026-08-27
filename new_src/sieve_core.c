/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Sieve Core Implementation: Segmented Sieve of Eratosthenes
 */

#include "sieve_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
#include <immintrin.h>
#include <pthread.h>
#define SIEVE_CAN_USE_AVX2 1
#define SIEVE_AVX2_PRIME_MAX 63U
#define SIEVE_AVX2_BLOCK_OFFSETS 256U

static uint64_t g_avx2_mark_masks[SIEVE_AVX2_PRIME_MAX + 1U]
                                  [SIEVE_AVX2_PRIME_MAX][4];
static pthread_once_t g_avx2_mark_masks_once = PTHREAD_ONCE_INIT;

static void sieve_init_avx2_mark_masks(void) {
    for (uint32_t prime = 2; prime <= SIEVE_AVX2_PRIME_MAX; prime++) {
        for (uint32_t phase = 0; phase < prime; phase++) {
            for (uint32_t offset = 0; offset < SIEVE_AVX2_BLOCK_OFFSETS; offset++) {
                if ((phase + offset) % prime == 0) {
                    g_avx2_mark_masks[prime][phase][offset >> 6] |=
                        1ULL << (offset & 63U);
                }
            }
        }
    }
}

static int sieve_cpu_has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

__attribute__((target("avx2")))
static void sieve_mark_low_prime_avx2(uint64_t *bitmap, uint64_t block_count,
                                      uint64_t prime, uint64_t base_remainder) {
    uint64_t phase = base_remainder;
    uint64_t phase_step = SIEVE_AVX2_BLOCK_OFFSETS % prime;

    for (uint64_t block = 0; block < block_count; block++) {
        __m256i marked = _mm256_loadu_si256(
            (const __m256i *)g_avx2_mark_masks[prime][phase]);
        __m256i current = _mm256_loadu_si256(
            (const __m256i *)(bitmap + (block * 4U)));
        _mm256_storeu_si256((__m256i *)(bitmap + (block * 4U)),
                            _mm256_or_si256(current, marked));

        phase += phase_step;
        if (phase >= prime) phase -= prime;
    }
}
#else
#define SIEVE_CAN_USE_AVX2 0
#endif

/* Branchless x mod p for p < 2^32, using a precomputed reciprocal
   inv_p = floor(2^64 / p). Valid for any odd p >= 3: the product
   x * inv_p overflows 128 bits only when x >= 2^64, which never happens
   for our offsets (window_start < 2^shift, shift < 64). */
static inline uint64_t sieve_fast_mod(uint64_t x, uint64_t p, uint64_t inv_p) {
    uint64_t q = (uint64_t)(((unsigned __int128)x * (unsigned __int128)inv_p) >> 64);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}

static uint32_t sieve_extract_odd_candidates_scalar(const uint64_t *bitmap,
                                                   uint64_t bitmap_words,
                                                   uint64_t odd_count,
                                                   uint64_t first_odd_offset,
                                                   uint64_t *candidates) {
    uint32_t count = 0;
    for (uint64_t word = 0; word < bitmap_words; word++) {
        uint64_t survivors = ~bitmap[word];
        if (word + 1U == bitmap_words && (odd_count & 63U) != 0) {
            survivors &= (1ULL << (odd_count & 63U)) - 1ULL;
        }
        while (survivors) {
            unsigned int bit = (unsigned int)__builtin_ctzll(survivors);
            candidates[count++] = first_odd_offset +
                                  (((word << 6) + bit) << 1);
            survivors &= survivors - 1ULL;
        }
    }
    return count;
}

#if SIEVE_CAN_USE_AVX2
__attribute__((target("avx2")))
static uint32_t sieve_extract_odd_candidates_avx2(const uint64_t *bitmap,
                                                  uint64_t bitmap_words,
                                                  uint64_t odd_count,
                                                  uint64_t first_odd_offset,
                                                  uint64_t *candidates) {
    const __m256i all_composite = _mm256_set1_epi64x(-1LL);
    uint32_t count = 0;
    uint64_t word = 0;

    for (; word + 4U <= bitmap_words; word += 4U) {
        __m256i values = _mm256_loadu_si256((const __m256i *)(bitmap + word));
        __m256i full = _mm256_cmpeq_epi64(values, all_composite);
        if (_mm256_movemask_epi8(full) == -1) continue;

        for (uint64_t lane = 0; lane < 4U; lane++) {
            uint64_t survivors = ~bitmap[word + lane];
            /* Mask the trailing partial word so positions >= odd_count are
               never emitted (matches the scalar extractor; without this the
               AVX2 path returns spurious candidates whenever bitmap_words is
               a multiple of 4 and odd_count is not a multiple of 64). */
            if (word + lane + 1U == bitmap_words && (odd_count & 63U) != 0) {
                survivors &= (1ULL << (odd_count & 63U)) - 1ULL;
            }
            while (survivors) {
                unsigned int bit = (unsigned int)__builtin_ctzll(survivors);
                candidates[count++] = first_odd_offset +
                                      ((((word + lane) << 6) + bit) << 1);
                survivors &= survivors - 1ULL;
            }
        }
    }

    for (; word < bitmap_words; word++) {
        uint64_t survivors = ~bitmap[word];
        while (survivors) {
            unsigned int bit = (unsigned int)__builtin_ctzll(survivors);
            uint64_t index = (word << 6) + bit;
            if (index < odd_count) {
                candidates[count++] = first_odd_offset + (index << 1);
            }
            survivors &= survivors - 1ULL;
        }
    }
    return count;
}
#endif

static int sieve_core_run_odd_only(struct sieve_core *sc,
                                   const mpz_t base,
                                   uint64_t base_offset,
                                   int use_cache,
                                   uint64_t **out_candidates,
                                   uint32_t *out_count) {
    if (!sc || !out_candidates || !out_count ||
        (use_cache && !sc->base_mod_p_valid)) {
        return 0;
    }

    uint64_t base_parity = use_cache ? (sc->base_mod_p[0] & 1ULL) : 0;
    uint64_t first_odd_offset = base_parity ? 0U : 1U;
    if (sc->interval_size <= first_odd_offset) return 0;

    uint64_t odd_count = (sc->interval_size - first_odd_offset + 1U) >> 1;
    uint64_t bitmap_words = (odd_count + 63U) >> 6;
    memset(sc->bitmap, 0, (size_t)bitmap_words * sizeof(*sc->bitmap));

    for (size_t i = 0; i < sc->small_primes_count; i++) {
        uint64_t p = sc->small_primes[i];
        if (p < 3U || p > sc->prime_limit) continue;

        uint64_t base_mod = use_cache ? sc->base_mod_p[i] :
            mpz_fdiv_ui(base, p);
        uint64_t remainder = base_mod + sieve_fast_mod(base_offset, p, sc->inv_p[i]);
        if (remainder >= p) remainder -= p;
        remainder += first_odd_offset % p;
        if (remainder >= p) remainder -= p;

        uint64_t inverse_two = (p + 1U) >> 1;
        uint64_t t = (remainder == 0U) ? 0U : (p - remainder);
        uint64_t pos = sieve_fast_mod(t * inverse_two, p, sc->inv_p[i]);
        uint64_t vector_end = (sc->avx2_enabled && p <= SIEVE_AVX2_PRIME_MAX) ?
                              (odd_count / SIEVE_AVX2_BLOCK_OFFSETS) *
                              SIEVE_AVX2_BLOCK_OFFSETS : 0;

#if SIEVE_CAN_USE_AVX2
        if (vector_end > 0) {
            uint64_t phase = (p - (pos % p)) % p;
            sieve_mark_low_prime_avx2(sc->bitmap,
                                      vector_end / SIEVE_AVX2_BLOCK_OFFSETS,
                                      p, phase);
            if (pos < vector_end) {
                pos += ((vector_end - pos + p - 1U) / p) * p;
            }
        }
#endif
        for (; pos < odd_count; pos += p) {
            sc->bitmap[pos >> 6] |= 1ULL << (pos & 63U);
        }
    }

    uint32_t count;
#if SIEVE_CAN_USE_AVX2
    if (sc->avx2_enabled) {
        count = sieve_extract_odd_candidates_avx2(
            sc->bitmap, bitmap_words, odd_count, first_odd_offset,
            sc->candidate_buffer);
    } else {
        count = sieve_extract_odd_candidates_scalar(
            sc->bitmap, bitmap_words, odd_count, first_odd_offset,
            sc->candidate_buffer);
    }
#else
    count = sieve_extract_odd_candidates_scalar(
        sc->bitmap, bitmap_words, odd_count, first_odd_offset,
        sc->candidate_buffer);
#endif

    if (count == 0) return 0;
    *out_candidates = sc->candidate_buffer;
    *out_count = count;
    return 1;
}

static uint32_t sieve_extract_candidate_word(uint64_t bitmap_word,
                                             uint64_t word_start,
                                             uint64_t interval_size,
                                             uint64_t *candidates,
                                             uint32_t count) {
    uint64_t candidate_bits = ~bitmap_word;
    if (word_start + 64U > interval_size) {
        uint64_t valid_bits = interval_size - word_start;
        candidate_bits &= (1ULL << valid_bits) - 1ULL;
    }

    while (candidate_bits) {
        unsigned int bit = (unsigned int)__builtin_ctzll(candidate_bits);
        candidates[count++] = word_start + bit;
        candidate_bits &= candidate_bits - 1ULL;
    }
    return count;
}

static uint32_t sieve_extract_candidates_scalar(const struct sieve_core *sc,
                                                uint64_t *candidates) {
    uint32_t count = 0;
    for (uint64_t word = 0; word < sc->bitmap_words; word++) {
        count = sieve_extract_candidate_word(sc->bitmap[word], word * 64U,
                                             sc->interval_size, candidates, count);
    }
    return count;
}

#if SIEVE_CAN_USE_AVX2
__attribute__((target("avx2")))
static uint32_t sieve_extract_candidates_avx2(const struct sieve_core *sc,
                                              uint64_t *candidates) {
    const __m256i all_composite = _mm256_set1_epi64x(-1LL);
    uint32_t count = 0;
    uint64_t word = 0;

    for (; word + 4U <= sc->bitmap_words; word += 4U) {
        __m256i bitmap = _mm256_loadu_si256((const __m256i *)(sc->bitmap + word));
        __m256i full = _mm256_cmpeq_epi64(bitmap, all_composite);
        if (_mm256_movemask_epi8(full) == -1) continue;

        for (uint64_t lane = 0; lane < 4U; lane++) {
            count = sieve_extract_candidate_word(sc->bitmap[word + lane],
                                                 (word + lane) * 64U,
                                                 sc->interval_size, candidates, count);
        }
    }

    for (; word < sc->bitmap_words; word++) {
        count = sieve_extract_candidate_word(sc->bitmap[word], word * 64U,
                                             sc->interval_size, candidates, count);
    }
    return count;
}
#endif

int sieve_core_init(struct sieve_core *sc, mpz_t primorial, 
                    const uint64_t *small_primes, size_t count, 
                    uint64_t prime_limit) {
    if (!sc || !small_primes || count == 0) return 0;
    
    mpz_init_set(sc->primorial, primorial);
    sc->interval_size = mpz_get_ui(primorial);  /* Simplified: assume fits in uint64_t */
    
    /* Allocate bitmap: need (interval_size + 63) / 64 words */
    sc->bitmap_words = (sc->interval_size + 63) / 64;
    sc->bitmap = (uint64_t *)calloc(sc->bitmap_words, sizeof(uint64_t));
    if (!sc->bitmap) {
        mpz_clear(sc->primorial);
        return 0;
    }

    sc->candidate_buffer = malloc(sc->interval_size * sizeof(*sc->candidate_buffer));
    if (!sc->candidate_buffer) {
        free(sc->bitmap);
        sc->bitmap = NULL;
        mpz_clear(sc->primorial);
        return 0;
    }
    sc->candidate_capacity = sc->interval_size;
    
    sc->small_primes = small_primes;
    sc->small_primes_count = count;
    sc->prime_limit = prime_limit;
    sc->owns_small_primes = 0;
    
    /* Reciprocal table for branchless mod: inv_p[i] = floor(2^64 / p). */
    sc->inv_p = malloc(sc->small_primes_count * sizeof(*sc->inv_p));
    if (!sc->inv_p) {
        free(sc->bitmap);
        sc->bitmap = NULL;
        free(sc->candidate_buffer);
        sc->candidate_buffer = NULL;
        mpz_clear(sc->primorial);
        return 0;
    }
    for (size_t i = 0; i < sc->small_primes_count; i++) {
        uint64_t p = sc->small_primes[i];
        sc->inv_p[i] = (p >= 3U) ? (UINT64_MAX / p) : 0U;
    }
    
    /* No base_mod_p cache initially */
    sc->base_mod_p = NULL;
    sc->base_mod_p_valid = 0;
    sc->avx2_enabled = 0;
#if SIEVE_CAN_USE_AVX2
    if (sieve_cpu_has_avx2()) {
        pthread_once(&g_avx2_mark_masks_once, sieve_init_avx2_mark_masks);
        sc->avx2_enabled = 1;
    }
#endif
    
    return 1;
}

int sieve_core_init_window(struct sieve_core *sc, uint64_t interval_size,
                           uint64_t prime_limit) {
    if (!sc || interval_size < 2 || prime_limit < 2 ||
        prime_limit > UINT32_MAX) {
        return 0;
    }

    size_t table_size = (size_t)prime_limit + 1;
    uint8_t *composite = calloc(table_size, sizeof(*composite));
    if (!composite) return 0;

    for (uint64_t factor = 2; factor <= prime_limit / factor; factor++) {
        if (composite[factor]) continue;
        for (uint64_t multiple = factor * factor;
             multiple <= prime_limit;
             multiple += factor) {
            composite[multiple] = 1;
        }
    }

    size_t count = 0;
    for (uint64_t value = 2; value <= prime_limit; value++) {
        if (!composite[value]) count++;
    }

    uint64_t *small_primes = malloc(count * sizeof(*small_primes));
    if (!small_primes) {
        free(composite);
        return 0;
    }

    size_t index = 0;
    for (uint64_t value = 2; value <= prime_limit; value++) {
        if (!composite[value]) {
            small_primes[index++] = value;
        }
    }
    free(composite);

    mpz_t window;
    mpz_init_set_ui(window, interval_size);
    int initialized = sieve_core_init(sc, window, small_primes, count, prime_limit);
    mpz_clear(window);

    if (!initialized) {
        free(small_primes);
        return 0;
    }

    sc->owns_small_primes = 1;
    return 1;
}

static int sieve_core_run_impl(struct sieve_core *sc, const mpz_t base,
                               uint64_t base_offset, int use_cache,
                               uint64_t **out_candidates, uint32_t *out_count) {
    if (!sc || !out_candidates || !out_count ||
        (use_cache && !sc->base_mod_p_valid)) {
        return 0;
    }

    if (use_cache || (base && mpz_cmp_ui(base, 3) >= 0)) {
        if (!use_cache) {
            if (sc->small_primes_count == 0 || sc->small_primes[0] != 2U) {
                return 0;
            }
        }
        if (use_cache || mpz_cmp_ui(base, 3) >= 0) {
            return sieve_core_run_odd_only(sc, base, base_offset, use_cache,
                                           out_candidates, out_count);
        }
    }
    
    /* Clear bitmap (all bits = 0 initially, means candidates) */
    memset(sc->bitmap, 0, sc->bitmap_words * sizeof(uint64_t));
    
    uint64_t vector_end = sc->avx2_enabled ?
        (sc->interval_size / SIEVE_AVX2_BLOCK_OFFSETS) * SIEVE_AVX2_BLOCK_OFFSETS : 0;

    /* Sieve: mark multiples of each small prime as composite */
    for (size_t i = 0; i < sc->small_primes_count && sc->small_primes[i] <= sc->prime_limit; i++) {
        uint64_t p = sc->small_primes[i];
        
        uint64_t remainder;
        if (use_cache) {
            remainder = sc->base_mod_p[i] +
                sieve_fast_mod(base_offset, p, sc->inv_p[i]);
            if (remainder >= p) remainder -= p;
        } else {
            remainder = mpz_fdiv_ui(base, p);
        }
        
        /* First multiple in range: base + (p - remainder) mod p */
        uint64_t start = (remainder == 0) ? 0 : (p - remainder);

#if SIEVE_CAN_USE_AVX2
        if (sc->avx2_enabled && p <= SIEVE_AVX2_PRIME_MAX && vector_end > 0) {
            sieve_mark_low_prime_avx2(sc->bitmap,
                                      vector_end / SIEVE_AVX2_BLOCK_OFFSETS,
                                      p, remainder);
            if (start < vector_end) {
                start += ((vector_end - start + p - 1U) / p) * p;
            }
        }
#endif
        
        /* Mark all multiples: base + start, base + start + p, ... */
        for (uint64_t offset = start; offset < sc->interval_size; offset += p) {
            SIEVE_MARK_COMPOSITE(sc->bitmap, offset);
        }
    }
    
    /* Extract candidates (offsets where bit = 0) */
    if (sc->interval_size > sc->candidate_capacity) return 0;

    uint32_t count;
#if SIEVE_CAN_USE_AVX2
    if (sc->avx2_enabled) {
        count = sieve_extract_candidates_avx2(sc, sc->candidate_buffer);
    } else {
        count = sieve_extract_candidates_scalar(sc, sc->candidate_buffer);
    }
#else
    count = sieve_extract_candidates_scalar(sc, sc->candidate_buffer);
#endif
    
    if (count == 0) {
        return 0;
    }
    
    *out_candidates = sc->candidate_buffer;
    *out_count = count;
    
    return 1;
}

/* Simple trial division sieve: mark multiples of primes as composite */
int sieve_core_run(struct sieve_core *sc, mpz_t base,
                   uint64_t **out_candidates, uint32_t *out_count) {
    return sieve_core_run_impl(sc, base, 0, 0, out_candidates, out_count);
}

int sieve_core_prepare_base_mod_p(struct sieve_core *sc, const mpz_t base) {
    if (!sc || !base) return 0;

    if (!sc->base_mod_p) {
        sc->base_mod_p = malloc(sc->small_primes_count * sizeof(*sc->base_mod_p));
        if (!sc->base_mod_p) return 0;
    }

    for (size_t i = 0; i < sc->small_primes_count; i++) {
        sc->base_mod_p[i] = mpz_fdiv_ui(base, sc->small_primes[i]);
    }
    sc->base_mod_p_valid = 1;
    return 1;
}

int sieve_core_prepare_base_mod_p_range(struct sieve_core *sc, const mpz_t base,
                                        size_t start, size_t end) {
    if (!sc || !base) return 0;
    if (start >= end || end > sc->small_primes_count) return 0;

    if (!sc->base_mod_p) {
        sc->base_mod_p = malloc(sc->small_primes_count * sizeof(*sc->base_mod_p));
        if (!sc->base_mod_p) return 0;
    }

    for (size_t i = start; i < end; i++) {
        sc->base_mod_p[i] = mpz_fdiv_ui(base, sc->small_primes[i]);
    }
    sc->base_mod_p_valid = 1;
    return 1;
}

int sieve_core_run_from_cached_base(struct sieve_core *sc, uint64_t base_offset,
                                    uint64_t **out_candidates,
                                    uint32_t *out_count) {
    return sieve_core_run_impl(sc, NULL, base_offset, 1,
                               out_candidates, out_count);
}

int sieve_core_run_from_cached_base_hybrid(struct sieve_core *sc,
                                           uint64_t base_offset,
                                           const uint64_t *premarked_bitmap,
                                           uint64_t premarked_words,
                                           size_t split_index,
                                           uint64_t **out_candidates,
                                           uint32_t *out_count) {
    if (!sc || !out_candidates || !out_count || !premarked_bitmap ||
        !sc->base_mod_p_valid) {
        return 0;
    }

    if (split_index > sc->small_primes_count) {
        split_index = sc->small_primes_count;
    }

    uint64_t base_parity = sc->base_mod_p[0] & 1ULL;
    uint64_t first_odd_offset = base_parity ? 0U : 1U;
    if (sc->interval_size <= first_odd_offset) return 0;

    uint64_t odd_count = (sc->interval_size - first_odd_offset + 1U) >> 1;
    uint64_t bitmap_words = (odd_count + 63U) >> 6;
    if (bitmap_words == 0 || bitmap_words > premarked_words ||
        bitmap_words > sc->bitmap_words) {
        return 0;
    }
    if (sc->interval_size > sc->candidate_capacity) return 0;

    memcpy(sc->bitmap, premarked_bitmap,
           (size_t)bitmap_words * sizeof(*sc->bitmap));

    for (size_t i = 0; i < split_index; i++) {
        uint64_t p = sc->small_primes[i];
        if (p < 3U || p > sc->prime_limit) continue;

        uint64_t remainder = sc->base_mod_p[i] +
            sieve_fast_mod(base_offset, p, sc->inv_p[i]);
        if (remainder >= p) remainder -= p;
        remainder += first_odd_offset % p;
        if (remainder >= p) remainder -= p;

        uint64_t inverse_two = (p + 1U) >> 1;
        uint64_t t = (remainder == 0U) ? 0U : (p - remainder);
        uint64_t pos = sieve_fast_mod(t * inverse_two, p, sc->inv_p[i]);

        for (; pos < odd_count; pos += p) {
            sc->bitmap[pos >> 6] |= 1ULL << (pos & 63U);
        }
    }

    uint32_t count;
#if SIEVE_CAN_USE_AVX2
    if (sc->avx2_enabled) {
        count = sieve_extract_odd_candidates_avx2(
            sc->bitmap, bitmap_words, odd_count, first_odd_offset,
            sc->candidate_buffer);
    } else {
        count = sieve_extract_odd_candidates_scalar(
            sc->bitmap, bitmap_words, odd_count, first_odd_offset,
            sc->candidate_buffer);
    }
#else
    count = sieve_extract_odd_candidates_scalar(
        sc->bitmap, bitmap_words, odd_count, first_odd_offset,
        sc->candidate_buffer);
#endif

    if (count == 0) return 0;

    *out_candidates = sc->candidate_buffer;
    *out_count = count;
    return 1;
}

uint64_t sieve_core_candidate(const struct sieve_core *sc, uint32_t idx) {
    if (!sc || idx >= sc->interval_size) return 0;
    return idx;  /* Direct index mapping (simplified) */
}

const char *sieve_core_simd_mode(const struct sieve_core *sc) {
    return sc && sc->avx2_enabled ? "AVX2" : "scalar";
}

void sieve_core_free(struct sieve_core *sc) {
    if (!sc) return;
    
    mpz_clear(sc->primorial);
    
    if (sc->bitmap) {
        free(sc->bitmap);
        sc->bitmap = NULL;
    }

    if (sc->candidate_buffer) {
        free(sc->candidate_buffer);
        sc->candidate_buffer = NULL;
        sc->candidate_capacity = 0;
    }
    
    if (sc->base_mod_p) {
        free(sc->base_mod_p);
        sc->base_mod_p = NULL;
    }

    if (sc->inv_p) {
        free(sc->inv_p);
        sc->inv_p = NULL;
    }

    if (sc->owns_small_primes) {
        free((void *)sc->small_primes);
        sc->small_primes = NULL;
        sc->small_primes_count = 0;
        sc->owns_small_primes = 0;
    }
}
