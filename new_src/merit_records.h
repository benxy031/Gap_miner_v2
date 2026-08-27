/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Merit Records: best-known merit per gap length
 *
 * Loads a local copy of the record table published at
 * https://primegaps.cloudygo.com/merits.txt (one "<gap_length> <merit> <name>"
 * line per even gap length) so a found gap's merit can be compared against
 * the best merit ever recorded for that exact gap length.
 */

#ifndef MERIT_RECORDS_H
#define MERIT_RECORDS_H

#include <stdint.h>

/* Load the reference table from `path` (see data/prime_gap_merits.txt).
 * Safe to call once at startup. Returns 0 on success, -1 if the file could
 * not be read (record lookups then always report "unknown", never block
 * mining). */
int merit_records_load(const char *path);

/* Look up the best known merit for an even gap length.
 * Returns 1 and sets *out_best_merit when the length is within the loaded
 * table's range; returns 0 (out_best_merit untouched) if the table isn't
 * loaded, the length is odd, or it exceeds the table's max known length. */
int merit_records_lookup(uint32_t gap_length, double *out_best_merit);

/* Free the loaded table. */
void merit_records_free(void);

#endif /* MERIT_RECORDS_H */
