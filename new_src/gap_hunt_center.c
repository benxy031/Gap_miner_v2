/*
 * Copyright (C) 2026 GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "gap_hunt_center.h"

#include <stddef.h>
#include <string.h>

#define NO_INDEX UINT32_MAX

static uint32_t grow_wave(const struct gh_center_state *s, uint32_t wave) {
    return wave > s->chunk / 2 ? s->chunk : wave * 2;
}

static uint32_t bounded_add(uint32_t start, uint32_t width, uint32_t end) {
    return width < end - start ? start + width : end;
}

static uint32_t find_right_target(const struct gh_center_state *s,
                                  uint32_t anchor) {
    if (s->offsets[anchor] > UINT64_MAX - s->threshold)
        return NO_INDEX;
    uint64_t target = s->offsets[anchor] + s->threshold;
    uint32_t lo = anchor + 1, hi = s->count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (s->offsets[mid] >= target)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo < s->count ? lo : NO_INDEX;
}

static uint32_t find_left_target(const struct gh_center_state *s,
                                 uint32_t anchor) {
    if (s->offsets[anchor] < s->threshold)
        return NO_INDEX;
    uint64_t target = s->offsets[anchor] - s->threshold;
    uint32_t lo = 0, hi = anchor;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (s->offsets[mid] <= target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo ? lo - 1 : NO_INDEX;
}

static int append_pair(struct gh_center_state *s, uint32_t left,
                       uint32_t right) {
    if (s->n_pairs >= s->capacity)
        return GH_CENTER_CAPACITY;
    s->starts[s->n_pairs] = (uint32_t)s->offsets[left];
    s->ends[s->n_pairs] = (uint32_t)s->offsets[right];
    s->n_pairs++;
    return GH_CENTER_OK;
}

static void start_right_end(struct gh_center_state *s) {
    s->phase = GH_CENTER_RIGHT_END;
    s->lo = s->target;
    s->hi = bounded_add(s->lo, s->first_wave, s->count);
    s->wave = grow_wave(s, s->first_wave);
}

static void start_right_jump(struct gh_center_state *s) {
    s->target = find_right_target(s, s->anchor);
    if (s->target == NO_INDEX) {
        s->phase = GH_CENTER_DONE;
        return;
    }
    s->phase = GH_CENTER_RIGHT_JUMP;
    s->hi = s->target;
    s->lo = s->hi > s->first_wave ? s->hi - s->first_wave : 0;
    if (s->lo < s->anchor + 1)
        s->lo = s->anchor + 1;
    s->wave = grow_wave(s, s->first_wave);
    if (s->lo == s->hi)
        start_right_end(s);
}

static int finish_left(struct gh_center_state *s) {
    for (uint32_t i = 0; i < s->n_pairs / 2; i++) {
        uint32_t j = s->n_pairs - 1 - i;
        uint32_t start = s->starts[i], end = s->ends[i];
        s->starts[i] = s->starts[j];
        s->ends[i] = s->ends[j];
        s->starts[j] = start;
        s->ends[j] = end;
    }
    if (s->left != NO_INDEX && s->right != NO_INDEX &&
        s->offsets[s->right] - s->offsets[s->left] >= s->threshold) {
        int rc = append_pair(s, s->left, s->right);
        if (rc != GH_CENTER_OK)
            return rc;
    }
    if (s->right == NO_INDEX) {
        s->phase = GH_CENTER_DONE;
    } else {
        s->anchor = s->right;
        start_right_jump(s);
    }
    return GH_CENTER_OK;
}

static void start_left_end(struct gh_center_state *s) {
    s->phase = GH_CENTER_LEFT_END;
    s->hi = s->target + 1;
    s->lo = s->hi > s->first_wave ? s->hi - s->first_wave : 0;
    s->wave = grow_wave(s, s->first_wave);
}

static int start_left_jump(struct gh_center_state *s) {
    s->target = find_left_target(s, s->anchor);
    if (s->target == NO_INDEX)
        return finish_left(s);
    s->phase = GH_CENTER_LEFT_JUMP;
    s->lo = s->target + 1;
    s->hi = bounded_add(s->lo, s->first_wave, s->anchor);
    s->wave = grow_wave(s, s->first_wave);
    if (s->lo == s->hi)
        start_left_end(s);
    return GH_CENTER_OK;
}

static int start_outer_search(struct gh_center_state *s) {
    if (s->left == NO_INDEX)
        return finish_left(s);
    s->anchor = s->left;
    return start_left_jump(s);
}

static int start_center_right(struct gh_center_state *s) {
    if (s->split == s->count)
        return start_outer_search(s);
    s->phase = GH_CENTER_FIND_RIGHT;
    s->lo = s->split;
    s->hi = bounded_add(s->lo, s->first_wave, s->count);
    s->wave = grow_wave(s, s->first_wave);
    return GH_CENTER_OK;
}

int gh_center_init(struct gh_center_state *s, const uint64_t *offsets,
                   uint32_t count, uint64_t center, uint64_t threshold,
                   uint32_t chunk, uint32_t first_wave,
                   uint32_t *starts, uint32_t *ends, uint32_t capacity) {
    if (!s || (count && !offsets) || (capacity && (!starts || !ends)) ||
        !threshold || !chunk || chunk > 512 || !first_wave ||
        first_wave > chunk || (count && offsets[count - 1] > UINT32_MAX))
        return GH_CENTER_INVALID;
    memset(s, 0, sizeof(*s));
    s->offsets = offsets;
    s->count = count;
    s->threshold = threshold;
    s->chunk = chunk;
    s->first_wave = first_wave;
    s->starts = starts;
    s->ends = ends;
    s->capacity = capacity;
    s->left = s->right = NO_INDEX;

    uint32_t lo = 0, hi = count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (offsets[mid] <= center)
            lo = mid + 1;
        else
            hi = mid;
    }
    s->split = lo;
    if (s->split == 0) {
        if (count == 0) {
            s->phase = GH_CENTER_DONE;
            return GH_CENTER_OK;
        }
        return start_center_right(s);
    }
    s->phase = GH_CENTER_FIND_LEFT;
    s->hi = s->split;
    s->lo = s->hi > first_wave ? s->hi - first_wave : 0;
    s->wave = grow_wave(s, first_wave);
    return GH_CENTER_OK;
}

static uint32_t first_positive(const uint8_t *flags, uint32_t count,
                               uint32_t base) {
    for (uint32_t i = 0; i < count; i++)
        if (flags[i])
            return base + i;
    return NO_INDEX;
}

static uint32_t last_positive(const uint8_t *flags, uint32_t count,
                              uint32_t base) {
    while (count) {
        count--;
        if (flags[count])
            return base + count;
    }
    return NO_INDEX;
}

int gh_center_accept(struct gh_center_state *s, const uint8_t *flags,
                     uint32_t count) {
    if (!s || s->phase == GH_CENTER_DONE || !flags || s->hi <= s->lo ||
        count != s->hi - s->lo)
        return GH_CENTER_INVALID;

    uint32_t found;
    switch (s->phase) {
    case GH_CENTER_FIND_LEFT:
        found = last_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            s->left = found;
            return start_center_right(s);
        }
        if (s->lo == 0)
            return start_center_right(s);
        s->hi = s->lo;
        s->lo = s->hi > s->wave ? s->hi - s->wave : 0;
        s->wave = grow_wave(s, s->wave);
        break;

    case GH_CENTER_FIND_RIGHT:
        found = first_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            s->right = found;
            return start_outer_search(s);
        }
        if (s->hi == s->count)
            return start_outer_search(s);
        s->lo = s->hi;
        s->hi = bounded_add(s->lo, s->wave, s->count);
        s->wave = grow_wave(s, s->wave);
        break;

    case GH_CENTER_LEFT_JUMP:
        found = first_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            s->hops++;
            s->anchor = found;
            return start_left_jump(s);
        }
        if (s->hi == s->anchor) {
            start_left_end(s);
        } else {
            s->lo = s->hi;
            s->hi = bounded_add(s->lo, s->wave, s->anchor);
            s->wave = grow_wave(s, s->wave);
            s->extensions++;
        }
        break;

    case GH_CENTER_LEFT_END:
        found = last_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            int rc = append_pair(s, found, s->anchor);
            if (rc != GH_CENTER_OK)
                return rc;
            s->anchor = found;
            return start_left_jump(s);
        }
        if (s->lo == 0)
            return finish_left(s);
        s->hi = s->lo;
        s->lo = s->hi > s->wave ? s->hi - s->wave : 0;
        s->wave = grow_wave(s, s->wave);
        break;

    case GH_CENTER_RIGHT_JUMP:
        found = last_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            s->hops++;
            s->anchor = found;
            start_right_jump(s);
        } else if (s->lo == s->anchor + 1) {
            start_right_end(s);
        } else {
            s->hi = s->lo;
            s->lo = s->hi > s->wave ? s->hi - s->wave : 0;
            if (s->lo < s->anchor + 1)
                s->lo = s->anchor + 1;
            s->wave = grow_wave(s, s->wave);
            s->extensions++;
        }
        break;

    case GH_CENTER_RIGHT_END:
        found = first_positive(flags, count, s->lo);
        if (found != NO_INDEX) {
            int rc = append_pair(s, s->anchor, found);
            if (rc != GH_CENTER_OK)
                return rc;
            s->anchor = found;
            start_right_jump(s);
        } else if (s->hi == s->count) {
            s->phase = GH_CENTER_DONE;
        } else {
            s->lo = s->hi;
            s->hi = bounded_add(s->lo, s->wave, s->count);
            s->wave = grow_wave(s, s->wave);
        }
        break;

    case GH_CENTER_DONE:
        return GH_CENTER_INVALID;
    default:
        return GH_CENTER_INVALID;
    }
    return GH_CENTER_OK;
}
