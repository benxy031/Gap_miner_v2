/*
 * Integration Test: scalar and AVX2 sieve equivalence
 */

#include "../new_src/sieve_core.h"
#include "../new_src/halfclass.h"
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

static int test_odd_positions_against_division(void) {
    struct sieve_core sc = {0};
    mpz_t base, shifted;
    mpz_init_set_str(base, "1267650600228229401496703205376", 10);
    mpz_init(shifted);
    if (!sieve_core_init_window(&sc, 513, 50000)) {
        mpz_clear(base);
        mpz_clear(shifted);
        return 0;
    }

    int ok = 1;
    for (unsigned parity = 0; parity < 2 && ok; parity++) {
        mpz_add_ui(shifted, base, parity);
        if (!sieve_core_prepare_base_mod_p(&sc, shifted)) { ok = 0; break; }
        for (uint64_t offset = 0; offset <= 512 && ok; offset += 256) {
            uint64_t *candidates;
            uint32_t count;
            if (!sieve_core_run_from_cached_base(&sc, offset, &candidates, &count)) {
                ok = 0;
                break;
            }
            uint32_t index = 0;
            for (uint64_t j = 0; j < sc.interval_size; j++) {
                int survives = 1;
                for (size_t i = 0; i < sc.small_primes_count; i++) {
                    uint64_t p = sc.small_primes[i];
                    if ((sc.base_mod_p[i] + offset + j) % p == 0) {
                        survives = 0;
                        break;
                    }
                }
                if (survives && (index >= count || candidates[index++] != j)) {
                    ok = 0;
                    break;
                }
            }
            if (index != count) ok = 0;
        }
    }
    sieve_core_free(&sc);
    mpz_clear(base);
    mpz_clear(shifted);
    return ok;
}

static int test_visible_extraction(void) {
    const uint64_t lengths[] = {507, 513, 2049, 2050, 1049600, 1056768};
    mpz_t base;
    mpz_init_set_str(base, "1267650600228229401496703205376", 10);
    int ok = 1;
    for (size_t len = 0; len < sizeof(lengths) / sizeof(lengths[0]) && ok; len++) {
        struct sieve_core full = {0}, visible = {0};
        mpz_set_str(base, "1267650600228229401496703205376", 10);
        uint64_t prime_limit = lengths[len] > 1000000 ? 2200000 : 50000;
        if (!sieve_core_init_window(&full, lengths[len], prime_limit)) {
            fprintf(stderr, "FAIL: visible sieve initialization\n");
            ok = 0;
            break;
        }
        if (!sieve_core_init_window(&visible, lengths[len], prime_limit)) {
            fprintf(stderr, "FAIL: visible sieve initialization\n");
            sieve_core_free(&full);
            ok = 0;
            break;
        }
        int avx2_enabled = visible.avx2_enabled;
        for (unsigned parity = 0; parity < 2 && ok; parity++) {
            if (parity) mpz_add_ui(base, base, 1);
            if (!sieve_core_prepare_base_mod_p(&full, base) ||
                !sieve_core_prepare_base_mod_p(&visible, base)) {
                ok = 0;
                break;
            }
            for (int scalar = 0; scalar < 2 && ok; scalar++) {
                visible.avx2_enabled = scalar ? 0 : avx2_enabled;
                uint64_t *empty;
                uint32_t empty_count;
                if (!sieve_core_run_from_cached_base_visible(
                        &visible, 0, (uint32_t)mpz_fdiv_ui(base, 60), 0,
                        &empty, &empty_count) || empty_count != 0) {
                    ok = 0;
                    break;
                }
                uint64_t step = lengths[len] > 1000000 ? 1048576 : 2;
                uint64_t last = lengths[len] > 1000000 ? 2 * step : 28;
                for (uint64_t off = 0; off <= last && ok; off += step) {
                    uint32_t mod60 = (uint32_t)((mpz_fdiv_ui(base, 60) + off) % 60);
                    for (int quarter = 0; quarter < 2; quarter++) {
                        halfclass_set_quarter(quarter);
                        uint64_t *all, *got;
                        uint32_t all_count, got_count;
                        if (!sieve_core_run_from_cached_base(&full, off, &all, &all_count) ||
                            !sieve_core_run_from_cached_base_visible(
                                &visible, off, mod60, halfclass_visible_mask(), &got, &got_count)) {
                            ok = 0;
                            break;
                        }
                        all_count = halfclass_filter_offsets(mod60, all, all_count);
                        if (all_count != got_count ||
                            memcmp(all, got, all_count * sizeof(*all)) != 0) ok = 0;
                    }
                    halfclass_set_quarter(0);
                }
            }
        }
        sieve_core_free(&full);
        sieve_core_free(&visible);
    }
    mpz_clear(base);
    return ok;
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
    int division_matched = test_odd_positions_against_division();
    int visible_matched = test_visible_extraction();
    printf("Odd-position direct division comparison: %s\n", division_matched ? "PASS" : "FAIL");
    printf("Visible extraction comparison: %s\n", visible_matched ? "PASS" : "FAIL");
    printf("%s\n", matched && cache_matched && hybrid_matched && division_matched && visible_matched ? "PASS" : "FAIL");

    free(hybrid_batch_bitmaps);
    sieve_core_free(&scalar);
    sieve_core_free(&accelerated);
    sieve_core_free(&hybrid);
    mpz_clear(base);
    mpz_clear(shifted_base);
    return matched && cache_matched && hybrid_matched && division_matched && visible_matched ? 0 : 1;
}