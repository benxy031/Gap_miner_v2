/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#define _POSIX_C_SOURCE 200809L

#include "merit_records.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *g_merits = NULL;      /* index i -> merit for gap length 2*i */
static uint32_t g_merits_count = 0;  /* number of entries allocated */

int merit_records_load(const char *path) {
    merit_records_free();

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[MeritRecords] Could not open %s; record checks disabled\n", path);
        return -1;
    }

    /* Two passes: first find the max gap length to size the table, then fill it. */
    uint64_t gap = 0;
    double merit = 0.0;
    char name[128];
    uint64_t max_gap = 0;
    while (fscanf(f, "%llu %lf %127s", (unsigned long long *)&gap, &merit, name) == 3) {
        if (gap > max_gap) max_gap = gap;
    }
    if (max_gap == 0) {
        fclose(f);
        fprintf(stderr, "[MeritRecords] %s has no usable entries; record checks disabled\n", path);
        return -1;
    }

    uint32_t count = (uint32_t)(max_gap / 2U) + 1U;
    double *table = (double *)calloc(count, sizeof(double));
    if (!table) {
        fclose(f);
        return -1;
    }

    rewind(f);
    uint32_t loaded = 0;
    while (fscanf(f, "%llu %lf %127s", (unsigned long long *)&gap, &merit, name) == 3) {
        if (gap % 2U != 0U) continue;
        uint32_t idx = (uint32_t)(gap / 2U);
        if (idx < count) {
            table[idx] = merit;
            loaded++;
        }
    }
    fclose(f);

    g_merits = table;
    g_merits_count = count;
    printf("[MeritRecords] Loaded %u gap-length records (max gap %llu) from %s\n",
           loaded, (unsigned long long)max_gap, path);
    return 0;
}

int merit_records_lookup(uint32_t gap_length, double *out_best_merit) {
    if (!g_merits || gap_length % 2U != 0U) return 0;
    uint32_t idx = gap_length / 2U;
    if (idx >= g_merits_count || g_merits[idx] <= 0.0) return 0;
    if (out_best_merit) *out_best_merit = g_merits[idx];
    return 1;
}

void merit_records_free(void) {
    free(g_merits);
    g_merits = NULL;
    g_merits_count = 0;
}
