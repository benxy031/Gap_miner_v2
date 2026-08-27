/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Record Log: append-only log of every BPSW-verified gap candidate
 * (dry-run, queued, or actually submitted), with full parameters, so
 * potential record-worthy gaps can be reviewed/reported later.
 */

#ifndef RECORD_LOG_H
#define RECORD_LOG_H

#include <stdint.h>
#include <gmp.h>

/* Open (create/append) the record log file. Safe to call once at startup.
 * Returns 0 on success, -1 if the file could not be opened (writes then
 * silently no-op rather than blocking mining). */
int record_log_init(const char *path);

/* Append one line for a BPSW-verified candidate. `status` is a short word
 * such as "dry-run", "queued", "submission-queue-full", "accepted",
 * "rejected", or "stale". `start` is the gap's first prime (p1). */
void record_log_write(uint32_t height, uint32_t shift, uint32_t header_nonce,
                      uint64_t nadd, const mpz_t start, uint32_t gap_length,
                      double merit, const char *status);

/* CRT-mode variant: the adder offset can exceed 64 bits (up to ~1024 bits),
 * so it is logged as a decimal string rather than a uint64. */
void record_log_write_big(uint32_t height, uint32_t shift, uint32_t header_nonce,
                          const char *nadd_dec, const mpz_t start,
                          uint32_t gap_length, double merit, const char *status);

/* Append a follow-up line for the actual submitblock RPC outcome (accepted,
 * rejected, stale) using only the fields available at that point (no `start`
 * decimal -- cross-reference by height/nAdd with the discovery-time line). */
void record_log_write_outcome(uint32_t height, uint32_t shift, uint32_t header_nonce,
                              uint64_t nadd, uint32_t gap_length, double merit,
                              const char *status);

/* CRT-mode outcome variant (decimal nAdd string for >64-bit offsets). */
void record_log_write_outcome_big(uint32_t height, uint32_t shift, uint32_t header_nonce,
                                  const char *nadd_dec, uint32_t gap_length,
                                  double merit, const char *status);

void record_log_close(void);

#endif /* RECORD_LOG_H */
