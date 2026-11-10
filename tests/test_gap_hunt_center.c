/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../new_src/gap_hunt_center.h"

#define MAX_CANDIDATES 4096
#define RANDOM_CANDIDATES 128

static uint64_t cases;

static void fail(const char *message) {
    fprintf(stderr, "center test failed: %s (case %llu)\n", message,
            (unsigned long long)cases);
    exit(1);
}

static void check_case(const uint64_t *offsets, const uint8_t *flags,
                       uint32_t count, uint64_t center, uint64_t threshold,
                       uint32_t chunk, uint32_t wave) {
    uint32_t starts[MAX_CANDIDATES], ends[MAX_CANDIDATES];
    uint32_t expected_starts[MAX_CANDIDATES], expected_ends[MAX_CANDIDATES];
    uint32_t expected = 0, previous = UINT32_MAX;
    for (uint32_t i = 0; i < count; i++) {
        if (!flags[i])
            continue;
        if (previous != UINT32_MAX &&
            offsets[i] - offsets[previous] >= threshold) {
            expected_starts[expected] = (uint32_t)offsets[previous];
            expected_ends[expected++] = (uint32_t)offsets[i];
        }
        previous = i;
    }

    struct gh_center_state s;
    if (gh_center_init(&s, offsets, count, center, threshold, chunk, wave,
                       starts, ends, MAX_CANDIDATES) != GH_CENTER_OK)
        fail("initialization");
    uint32_t rounds = 0;
    while (s.phase != GH_CENTER_DONE) {
        if (s.lo >= s.hi || s.hi > count || s.hi - s.lo > chunk)
            fail("invalid candidate range");
        if (++rounds > 10 * MAX_CANDIDATES)
            fail("scan did not terminate");
        uint32_t lo = s.lo, size = s.hi - s.lo;
        if (gh_center_accept(&s, flags + lo, size) != GH_CENTER_OK)
            fail("accepting candidate flags");
    }
    if (s.n_pairs != expected)
        fail("pair count differs from full scan");
    for (uint32_t i = 0; i < expected; i++)
        if (starts[i] != expected_starts[i] || ends[i] != expected_ends[i])
            fail("pair or output order differs from full scan");
    cases++;
}

static uint32_t random_value(void) {
    static uint32_t state = 0x6f35a1c9U;
    state = state * 1664525U + 1013904223U;
    return state;
}

static void check_capacity_and_errors(void) {
    const uint64_t offsets[] = {0, 10, 20, 30, 40};
    const uint8_t flags[] = {1, 1, 1, 1, 1};
    uint32_t starts[2] = {0, 0x12345678U};
    uint32_t ends[2] = {0, 0x87654321U};
    struct gh_center_state s;
    const uint64_t centers[] = {0, 10, 20, 40};
    for (size_t i = 0; i < sizeof(centers) / sizeof(centers[0]); i++) {
        if (gh_center_init(&s, offsets, 5, centers[i], 5, 2, 1,
                           starts, ends, 1) != GH_CENTER_OK)
            fail("capacity test initialization");
        int rc = GH_CENTER_OK;
        while (s.phase != GH_CENTER_DONE && rc == GH_CENTER_OK) {
            uint32_t lo = s.lo, size = s.hi - s.lo;
            rc = gh_center_accept(&s, flags + lo, size);
        }
        if (rc != GH_CENTER_CAPACITY || starts[1] != 0x12345678U ||
            ends[1] != 0x87654321U)
            fail("pair-capacity failure did not stop safely");
    }
    if (gh_center_init(&s, offsets, 5, 10, 5, 2, 1, NULL, NULL, 0) !=
        GH_CENTER_OK)
        fail("zero-capacity initialization");
    int rc = GH_CENTER_OK;
    while (s.phase != GH_CENTER_DONE && rc == GH_CENTER_OK) {
        uint32_t lo = s.lo, size = s.hi - s.lo;
        rc = gh_center_accept(&s, flags + lo, size);
    }
    if (rc != GH_CENTER_CAPACITY)
        fail("zero-capacity scan did not fail");

    if (gh_center_init(&s, offsets, 5, 20, 0, 2, 1, starts, ends, 1) !=
        GH_CENTER_INVALID)
        fail("zero threshold accepted");
    if (gh_center_init(&s, offsets, 5, 20, 5, 2, 3, starts, ends, 1) !=
        GH_CENTER_INVALID)
        fail("oversized wave accepted");
    const uint64_t too_wide[] = {(uint64_t)UINT32_MAX + 1};
    if (gh_center_init(&s, too_wide, 1, 0, 5, 2, 1, starts, ends, 1) !=
        GH_CENTER_INVALID)
        fail("unrepresentable offset accepted");
    if (gh_center_init(&s, offsets, 5, 20, 5, 2, 1, starts, ends, 1) !=
        GH_CENTER_OK)
        fail("invalid-accept test initialization");
    if (gh_center_accept(&s, NULL, s.hi - s.lo) != GH_CENTER_INVALID ||
        gh_center_accept(&s, flags + s.lo, 0) != GH_CENTER_INVALID)
        fail("invalid flags or range accepted");
    s.phase = (enum gh_center_phase)99;
    if (gh_center_accept(&s, flags + s.lo, s.hi - s.lo) != GH_CENTER_INVALID)
        fail("invalid scan phase accepted");
}

int main(void) {
    const uint64_t offsets[] = {3, 5, 8, 10, 13, 17, 18, 23};
    const uint64_t centers[] = {0, 3, 9, 17, 25};
    const uint64_t thresholds[] = {1, 3, 7, 20, 40};
    const uint32_t chunks[] = {1, 3, 8};
    uint8_t flags[MAX_CANDIDATES];
    for (uint32_t n = 0; n <= 8; n++) {
        for (uint32_t mask = 0; mask < (1U << n); mask++) {
            for (uint32_t i = 0; i < n; i++)
                flags[i] = (uint8_t)((mask >> i) & 1U);
            for (size_t c = 0; c < sizeof(centers) / sizeof(centers[0]); c++)
                for (size_t t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); t++)
                    for (size_t w = 0; w < sizeof(chunks) / sizeof(chunks[0]); w++)
                        check_case(offsets, flags, n, centers[c], thresholds[t],
                                   chunks[w], 1);
        }
    }

    for (unsigned trial = 0; trial < 20000; trial++) {
        uint64_t random_offsets[MAX_CANDIDATES];
        uint32_t n = random_value() % (RANDOM_CANDIDATES + 1);
        uint64_t at = random_value() % 10;
        for (uint32_t i = 0; i < n; i++) {
            at += 1 + random_value() % 13;
            random_offsets[i] = at;
            flags[i] = (uint8_t)((random_value() >> 24) < 70);
        }
        uint64_t center = random_value() % (at + 20);
        uint64_t threshold = 1 + random_value() % 200;
        uint32_t chunk = 1 + random_value() % 32;
        uint32_t wave = 1 + random_value() % chunk;
        check_case(random_offsets, flags, n, center, threshold, chunk, wave);
    }

    uint64_t many_offsets[MAX_CANDIDATES];
    uint64_t at = 0;
    for (uint32_t i = 0; i < MAX_CANDIDATES; i++) {
        at += 1 + random_value() % 13;
        many_offsets[i] = at;
        flags[i] = (uint8_t)((random_value() >> 24) < 70);
    }
    const uint64_t large_thresholds[] = {1, 25, 100, 1000, UINT64_MAX};
    for (size_t t = 0; t < sizeof(large_thresholds) / sizeof(large_thresholds[0]); t++) {
        check_case(many_offsets, flags, MAX_CANDIDATES, 0,
                   large_thresholds[t], 16, 8);
        check_case(many_offsets, flags, MAX_CANDIDATES,
                   many_offsets[MAX_CANDIDATES / 2], large_thresholds[t], 128, 8);
        check_case(many_offsets, flags, MAX_CANDIDATES, UINT64_MAX,
                   large_thresholds[t], 8, 8);
    }

    const uint64_t large[] = {UINT32_MAX - 10U, UINT32_MAX - 5U, UINT32_MAX};
    const uint8_t large_flags[] = {1, 1, 1};
    check_case(large, large_flags, 3, UINT32_MAX - 7U, 5, 2, 1);
    check_case(large, large_flags, 3, UINT64_MAX, UINT64_MAX, 2, 1);
    check_capacity_and_errors();
    printf("center state machine: %llu full-scan parity cases passed\n",
           (unsigned long long)cases);
    return 0;
}
