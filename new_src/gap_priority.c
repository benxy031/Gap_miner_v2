/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * ⚠️  UNUSED / DEPRECATED (2026-08-23): see gap_priority.h for why.
 */

#include "gap_priority.h"

#include <stdlib.h>

double gap_priority_value(uint64_t Lc, uint64_t Rc,
                          uint64_t V_left, uint64_t V_right,
                          double cost) {
    uint64_t sc = Lc + 2ULL + Rc;
    uint64_t sp = (V_left > V_right) ? V_left : V_right;
    uint64_t benefit = (sc > sp) ? sc : sp;

    if (cost <= 0.0) {
        return 0.0;
    }
    return (double)benefit / cost;
}

struct indexed_sc {
    size_t idx;
    uint64_t sc;
};

static int compare_indexed_sc_desc(const void *a, const void *b) {
    const struct indexed_sc *x = (const struct indexed_sc *)a;
    const struct indexed_sc *y = (const struct indexed_sc *)b;

    if (x->sc != y->sc) {
        return (x->sc < y->sc) ? 1 : -1;  /* descending by sc */
    }
    /* stable tie-break: keep increasing index order */
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

void gap_priority_order(const uint64_t *offsets, size_t n,
                        uint64_t interval_size,
                        size_t *out_order) {
    if (!offsets || !out_order || n == 0) {
        return;
    }

    struct indexed_sc *items =
        (struct indexed_sc *)malloc(n * sizeof(*items));
    if (!items) {
        /* Allocation failure: fall back to the identity order. */
        for (size_t i = 0; i < n; i++) {
            out_order[i] = i;
        }
        return;
    }

    for (size_t i = 0; i < n; i++) {
        uint64_t prev = (i == 0) ? 0ULL : offsets[i - 1];
        uint64_t next = (i + 1 == n) ? interval_size : offsets[i + 1];
        items[i].idx = i;
        items[i].sc = next - prev;
    }

    qsort(items, n, sizeof(*items), compare_indexed_sc_desc);

    for (size_t i = 0; i < n; i++) {
        out_order[i] = items[i].idx;
    }

    free(items);
}
