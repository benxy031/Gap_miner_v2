/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef GAP_HUNT_CENTER_H
#define GAP_HUNT_CENTER_H

#include <stdint.h>

enum gh_center_phase {
    GH_CENTER_FIND_LEFT,
    GH_CENTER_FIND_RIGHT,
    GH_CENTER_LEFT_JUMP,
    GH_CENTER_LEFT_END,
    GH_CENTER_RIGHT_JUMP,
    GH_CENTER_RIGHT_END,
    GH_CENTER_DONE
};

enum gh_center_result {
    GH_CENTER_OK = 0,
    GH_CENTER_INVALID = -1,
    GH_CENTER_CAPACITY = -2
};

struct gh_center_state {
    const uint64_t *offsets;
    uint32_t count;
    uint64_t threshold;
    uint32_t chunk, first_wave, wave;
    uint32_t split, left, right, anchor, target;
    uint32_t lo, hi;
    uint32_t *starts, *ends;
    uint32_t capacity, n_pairs;
    uint64_t hops, extensions;
    enum gh_center_phase phase;
};

/*
 * Offsets must be strictly increasing and fit in uint32_t. Each non-DONE
 * state requests flags for [lo, hi); accept them in ascending offset order.
 * The result is every consecutive positive pair at least threshold apart,
 * ordered by start offset.
 */
int gh_center_init(struct gh_center_state *s, const uint64_t *offsets,
                   uint32_t count, uint64_t center, uint64_t threshold,
                   uint32_t chunk, uint32_t first_wave,
                   uint32_t *starts, uint32_t *ends, uint32_t capacity);
int gh_center_accept(struct gh_center_state *s, const uint8_t *flags,
                     uint32_t count);

#endif
