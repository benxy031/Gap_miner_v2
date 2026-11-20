/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "crt_row_select.h"

#include <math.h>

int crt_row_score_prepare(double *powers, size_t count, double rate) {
    if (!powers || count == 0 || !isfinite(rate) || rate <= 0.0 || rate >= 1.0)
        return 0;
    powers[0] = 1.0;
    for (size_t i = 1; i < count; i++)
        powers[i] = powers[i - 1] * (1.0 - rate);
    return 1;
}

int crt_row_score_bitmap(const uint64_t *bitmap, uint64_t odd_slots,
                         uint64_t first_odd_offset, uint64_t gap,
                         uint64_t owned_limit, double rate,
                         const double *powers, size_t powers_count,
                         uint32_t *prefix, size_t prefix_count,
                         double *score) {
    if (!bitmap || !powers || !prefix || !score || odd_slots == 0 ||
        odd_slots > UINT32_MAX || first_odd_offset > 1 || gap == 0 ||
        !isfinite(rate) || rate <= 0.0 || rate >= 1.0)
        return 0;
    size_t words = (size_t)((odd_slots + 63U) / 64U);
    if (prefix_count < words + 1U || powers_count <= odd_slots)
        return 0;

    prefix[0] = 0;
    for (size_t w = 0; w < words; w++) {
        uint64_t alive = ~bitmap[w];
        if (w + 1 == words && (odd_slots & 63U))
            alive &= (1ULL << (odd_slots & 63U)) - 1U;
        prefix[w + 1] = prefix[w] + (uint32_t)__builtin_popcountll(alive);
    }

    double sum = 0.0;
    uint64_t gap_slots = gap / 2U + (gap & 1U);
    for (size_t w = 0; w < words; w++) {
        uint64_t alive = ~bitmap[w];
        if (w + 1 == words && (odd_slots & 63U))
            alive &= (1ULL << (odd_slots & 63U)) - 1U;
        uint32_t seen = prefix[w];
        while (alive) {
            unsigned bit = (unsigned)__builtin_ctzll(alive);
            uint64_t pos = (uint64_t)w * 64U + bit;
            alive &= alive - 1U;
            seen++;
            if (first_odd_offset + 2U * pos >= owned_limit)
                break;
            if (gap_slots >= odd_slots - pos)
                continue;
            uint64_t right = pos + gap_slots;
            size_t right_word = (size_t)(right >> 6);
            unsigned right_bit = (unsigned)(right & 63U);
            uint64_t lower = right_bit ? (1ULL << right_bit) - 1U : 0U;
            uint32_t before_right = prefix[right_word] +
                (uint32_t)__builtin_popcountll(~bitmap[right_word] & lower);
            uint32_t between = before_right - seen;
            uint32_t after = prefix[words] - before_right;
            sum += rate * powers[between] * (1.0 - powers[after]);
        }
    }
    *score = sum;
    return 1;
}

int crt_row_select_best(const double *scores, uint32_t rows, uint32_t keep,
                        uint64_t *mask) {
    if (!scores || !mask || rows == 0 || rows > 64 || keep == 0 || keep > rows)
        return 0;
    uint32_t order[64];
    for (uint32_t i = 0; i < rows; i++) {
        if (!isfinite(scores[i]) || scores[i] < 0.0)
            return 0;
        uint32_t j = i;
        while (j > 0 && scores[order[j - 1]] < scores[i]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = i;
    }
    uint64_t result = 0;
    for (uint32_t i = 0; i < keep; i++)
        result |= 1ULL << order[i];
    *mask = result;
    return 1;
}
