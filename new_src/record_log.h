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

/* Record WHICH template a submitted solution belongs to.  Called from BOTH
 * work sources: the pool path (`stratum_submit_share`) and the node path
 * (`--enable-submission`, immediately before `submitblock`).
 *
 * A block is valid only for the template it was mined on -- its parent must
 * still be the chain tip when the solution is submitted -- so when a
 * block-level solution is reported accepted and yet never appears on the chain,
 * there are exactly two explanations: the work source handed out a stale
 * template, or it never submitted the block.  The template identity is what
 * separates them, and it can only be captured AT SUBMIT TIME: the verdict
 * arrives later and carries no template information at all.
 *
 * `template_prevhash_hex` is the header's 32-byte prevhash in DISPLAY order (the
 * internal little-endian bytes reversed), so it compares byte for byte with
 * `getblockhash` and with explorers.  `template_ndiff` is the header's 8-byte
 * nDifficulty (bytes 72..79): the node path fills it from the GBT template's
 * difficulty (`gapcoin_work.c`), the pool path from the pool's own header, so
 * both paths log the same quantity.  The line also prints its merit
 * (ndiff / 2^48) so it can be read next to the candidate's own merit.
 *
 * Writes, next to the usual candidate fields:
 *   status=submitted template_prevhash=<64 hex> template_time=<unix>
 *   template_ndiff=<n> template_merit=<m>
 */
void record_log_write_submit_ctx(uint32_t height, uint32_t shift,
                                 uint32_t header_nonce, const char *nadd_dec,
                                 uint32_t gap_length, double merit,
                                 const char *template_prevhash_hex,
                                 uint32_t template_time,
                                 uint64_t template_ndiff);

/* Append one line for a BPSW-verified candidate. `status` is a short word
 * such as "dry-run", "queued", "submission-queue-full", "accepted",
 * "rejected", "stale", or "assemble-failed" (the local block assembly
 * failed, so the gap was never offered to the node -- distinct from "stale",
 * which means the header rotated before submission). `start` is the gap's
 * first prime (p1). */
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
