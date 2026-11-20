/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../new_src/crt_row_select.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static double brute_score(const uint64_t *bitmap, uint64_t odd_slots,
                          uint64_t first, uint64_t gap, uint64_t owned,
                          double rate) {
    double sum = 0.0;
    for (uint64_t left = 0; left < odd_slots; left++) {
        uint64_t left_offset = first + 2U * left;
        if (left_offset >= owned || (bitmap[left >> 6] & (1ULL << (left & 63U))))
            continue;
        uint64_t between = 0, after = 0;
        for (uint64_t right = left + 1U; right < odd_slots; right++) {
            if (bitmap[right >> 6] & (1ULL << (right & 63U)))
                continue;
            if (2U * (right - left) < gap) between++;
            else after++;
        }
        sum += rate * pow(1.0 - rate, (double)between) *
               (1.0 - pow(1.0 - rate, (double)after));
    }
    return sum;
}

int main(void) {
    static const uint64_t lengths[] = {1, 2, 63, 64, 65, 127, 128, 129, 257};
    static const uint64_t gaps[] = {1, 2, 3, 7, 64, 129, 400};
    const double rate = 0.031;
    double powers[258];
    uint32_t prefix[6];
    uint64_t bitmap[5];
    if (!crt_row_score_prepare(powers, 258, rate))
        return 1;
    for (size_t n = 0; n < sizeof(lengths) / sizeof(lengths[0]); n++) {
        uint64_t odd_slots = lengths[n];
        for (unsigned pattern = 0; pattern < 3; pattern++) {
            for (size_t i = 0; i < 5; i++)
                bitmap[i] = pattern == 0 ? 0U : pattern == 1 ? UINT64_MAX :
                    (0x9521e48b71c0a63dULL ^ (0x9e3779b97f4a7c15ULL * i));
            for (uint64_t first = 0; first <= 1; first++) {
                for (size_t gi = 0; gi < sizeof(gaps) / sizeof(gaps[0]); gi++) {
                    for (unsigned own = 0; own < 3; own++) {
                        uint64_t owned = own == 0 ? 0 : own == 1 ? odd_slots :
                                         2U * odd_slots + first;
                        double actual;
                        if (!crt_row_score_bitmap(bitmap, odd_slots, first,
                                                  gaps[gi], owned, rate,
                                                  powers, 258, prefix, 6, &actual)) {
                            fprintf(stderr, "score rejected valid input\n");
                            return 1;
                        }
                        double expected = brute_score(bitmap, odd_slots, first,
                                                      gaps[gi], owned, rate);
                        if (fabs(actual - expected) > 1e-12 * (1.0 + expected)) {
                            fprintf(stderr,
                                    "score mismatch n=%llu gap=%llu first=%llu "
                                    "owned=%llu got=%.17g expected=%.17g\n",
                                    (unsigned long long)odd_slots,
                                    (unsigned long long)gaps[gi],
                                    (unsigned long long)first,
                                    (unsigned long long)owned, actual, expected);
                            return 1;
                        }
                    }
                }
            }
        }
    }

    double scores[64];
    for (unsigned i = 0; i < 64; i++) scores[i] = 1.0;
    uint64_t mask = 0;
    if (!crt_row_select_best(scores, 64, 32, &mask) || mask != 0xffffffffULL)
        return 1;
    for (unsigned i = 0; i < 64; i++) scores[i] = (double)i;
    if (!crt_row_select_best(scores, 64, 32, &mask) ||
        mask != 0xffffffff00000000ULL)
        return 1;
    scores[0] = NAN;
    if (crt_row_select_best(scores, 64, 32, &mask) ||
        crt_row_score_prepare(powers, 258, 1.0) ||
        crt_row_score_bitmap(bitmap, 65, 1, 2, 65, rate, powers, 65,
                             prefix, 6, &scores[0]))
        return 1;
    puts("CRT row scores: exact brute-force agreement and stable top-half selection");
    return 0;
}
