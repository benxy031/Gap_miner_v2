/*
 * Integration Test: scalar and AVX2 sieve equivalence
 */

#include "../new_src/sieve_core.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static size_t test_hybrid_split_index(const struct sieve_core *sc) {
    size_t by_value = sc->small_primes_count;
    for (size_t i = 0; i < sc->small_primes_count; i++) {
        if (sc->small_primes[i] >= 1000U) {
            by_value = i;
            break;
        }
    }
    return by_value;
}

static int test_build_high_prime_bitmap(const struct sieve_core *sc,
                                        uint64_t base_offset,
                                        size_t split_index,
                                        uint64_t *bitmap,
                                        uint64_t bitmap_words) {
    if (!sc || !sc->base_mod_p_valid || !bitmap) return 0;

    uint64_t first_odd_offset = (sc->base_mod_p[0] & 1ULL) ? 0U : 1U;
    if (sc->interval_size <= first_odd_offset) return 0;

    uint64_t odd_count = (sc->interval_size - first_odd_offset + 1U) >> 1;
    uint64_t required_words = (odd_count + 63U) >> 6;
    if (bitmap_words < required_words) return 0;

    memset(bitmap, 0, (size_t)required_words * sizeof(*bitmap));

    for (size_t i = split_index; i < sc->small_primes_count; i++) {
        uint64_t p = sc->small_primes[i];
        if (p < 3U || p > sc->prime_limit) continue;

        uint64_t remainder = sc->base_mod_p[i] + base_offset % p;
        if (remainder >= p) remainder -= p;
        remainder += first_odd_offset % p;
        if (remainder >= p) remainder -= p;

        uint64_t inverse_two = (p + 1U) >> 1;
        uint64_t pos = ((p - remainder) % p * inverse_two) % p;
        for (; pos < odd_count; pos += p) {
            bitmap[pos >> 6] |= 1ULL << (pos & 63U);
        }
    }

    return 1;
}

static int test_build_high_prime_bitmap_batch(const struct sieve_core *sc,
                                              const uint64_t *base_offsets,
                                              size_t batch_count,
                                              size_t split_index,
                                              uint64_t *bitmaps,
                                              uint64_t bitmap_words_per_window) {
    if (!sc || !base_offsets || !bitmaps || batch_count == 0) return 0;

    for (size_t i = 0; i < batch_count; i++) {
        uint64_t *window_bitmap =
            bitmaps + (i * (size_t)bitmap_words_per_window);
        if (!test_build_high_prime_bitmap(sc, base_offsets[i], split_index,
                                          window_bitmap,
                                          bitmap_words_per_window)) {
            return 0;
        }
    }

    return 1;
}

int main(void) {
    struct sieve_core scalar = {0};
    struct sieve_core accelerated = {0};
    struct sieve_core hybrid = {0};
    uint64_t *scalar_candidates = NULL;
    uint64_t *accelerated_candidates = NULL;
    uint32_t scalar_count = 0;
    uint32_t accelerated_count = 0;
    uint64_t *shifted_candidates = NULL;
    uint32_t shifted_count = 0;
    uint64_t *cached_candidates = NULL;
    uint32_t cached_count = 0;
    uint64_t *hybrid_candidates = NULL;
    uint32_t hybrid_count = 0;
    uint64_t *hybrid_batch_bitmaps = NULL;
    uint64_t hybrid_bitmap_words = 0;
    size_t hybrid_split_index = 0;
    mpz_t base;
    mpz_t shifted_base;
    mpz_init_set_str(base, "1267650600228229401496703205376", 10);
    mpz_init(shifted_base);

    if (!sieve_core_init_window(&scalar, 12288, 50000) ||
        !sieve_core_init_window(&accelerated, 12288, 50000) ||
        !sieve_core_init_window(&hybrid, 12288, 50000)) {
        fprintf(stderr, "FAIL: Cannot initialize sieve cores\n");
        return 1;
    }
    scalar.avx2_enabled = 0;

    hybrid_split_index = test_hybrid_split_index(&hybrid);
    if (hybrid_split_index > hybrid.small_primes_count) {
        hybrid_split_index = hybrid.small_primes_count;
    }
    hybrid_bitmap_words = (((hybrid.interval_size + 1U) >> 1) + 63U) >> 6;

    int matched = sieve_core_run(&scalar, base, &scalar_candidates, &scalar_count) &&
                  sieve_core_run(&accelerated, base, &accelerated_candidates,
                                 &accelerated_count) &&
                  scalar_count == accelerated_count &&
                  memcmp(scalar_candidates, accelerated_candidates,
                         scalar_count * sizeof(*scalar_candidates)) == 0;

    const uint64_t cached_offsets[] = {0, 1, 4096, 8192, 12288, 65535};
    const size_t cached_offset_count =
        sizeof(cached_offsets) / sizeof(cached_offsets[0]);
    int cache_matched = sieve_core_prepare_base_mod_p(&accelerated, base) &&
                        sieve_core_prepare_base_mod_p(&hybrid, base);
    int hybrid_matched = cache_matched;
    if (cache_matched) {
        if (cached_offset_count > 0 &&
            cached_offset_count <= SIZE_MAX / (size_t)hybrid_bitmap_words) {
            hybrid_batch_bitmaps = calloc(cached_offset_count *
                                          (size_t)hybrid_bitmap_words,
                                          sizeof(*hybrid_batch_bitmaps));
        }
        if (!hybrid_batch_bitmaps) {
            hybrid_matched = 0;
        }
    }
    if (hybrid_matched) {
        hybrid_matched = test_build_high_prime_bitmap_batch(
            &hybrid,
            cached_offsets,
            cached_offset_count,
            hybrid_split_index,
            hybrid_batch_bitmaps,
            hybrid_bitmap_words);
    }
    for (size_t i = 0;
         cache_matched && hybrid_matched &&
         i < cached_offset_count;
         i++) {
        mpz_add_ui(shifted_base, base, cached_offsets[i]);

        cache_matched = sieve_core_run(&scalar, shifted_base, &shifted_candidates,
                                       &shifted_count) &&
                        sieve_core_run_from_cached_base(
                            &accelerated, cached_offsets[i],
                            &cached_candidates, &cached_count) &&
                        shifted_count == cached_count &&
                        memcmp(shifted_candidates, cached_candidates,
                               shifted_count * sizeof(*shifted_candidates)) == 0;

        hybrid_matched = sieve_core_run_from_cached_base_hybrid(
                             &hybrid,
                             cached_offsets[i],
                             hybrid_batch_bitmaps +
                                 (i * (size_t)hybrid_bitmap_words),
                             hybrid_bitmap_words,
                             hybrid_split_index,
                             &hybrid_candidates,
                             &hybrid_count) &&
                         shifted_count == hybrid_count &&
                         memcmp(shifted_candidates, hybrid_candidates,
                                shifted_count * sizeof(*shifted_candidates)) == 0;

        shifted_candidates = NULL;
        cached_candidates = NULL;
        hybrid_candidates = NULL;
    }

    printf("Scalar/AVX2 sieve comparison (%s): %u candidates\n",
           sieve_core_simd_mode(&accelerated), accelerated_count);
    printf("Cached-base sieve comparison: %u candidates\n", cached_count);
    printf("Hybrid GPU-pre-marked sieve comparison (packed batch): %u candidates\n",
           hybrid_count);
    printf("%s\n", matched && cache_matched && hybrid_matched ? "PASS" : "FAIL");

    free(hybrid_batch_bitmaps);
    sieve_core_free(&scalar);
    sieve_core_free(&accelerated);
    sieve_core_free(&hybrid);
    mpz_clear(base);
    mpz_clear(shifted_base);
    return matched && cache_matched && hybrid_matched ? 0 : 1;
}