/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Gapcoin pool client: the LEGACY stratum protocol (suprnova port 2434).
 *
 * This is NOT Bitcoin stratum v1.  Gapcoin's pool protocol is getwork over a
 * newline-delimited JSON-RPC stream, and it is the protocol the official
 * Gapcoin miners speak ("old miners" on gap.suprnova.cc:2434); the "new
 * stratum" on port 2433 is a private dialect that only suprnova's own miners
 * implement.
 *
 *   client -> pool   {"id":N,"method":"mining.request","params":["user","pass"]}\n
 *   pool   -> client {"id":N,"result":{"data":"<160 hex>","difficulty":<ndiff>}}\n
 *   pool   -> client {"id":null,"method":"blockchain.block.new",
 *                     "params":{"data":"<160 hex>","difficulty":<ndiff>}}\n
 *   client -> pool   {"id":N,"method":"mining.submit","params":["user","pass","<hex>"]}\n
 *   pool   -> client {"id":N,"result":true|false}\n
 *   pool   -> client {"id":null,"method":"mining.set_difficulty","params":[...]}\n
 *
 * "data" is the 80-byte Gapcoin header prefix, NOT a full block:
 *
 *   version(4) + prevhash(32) + merkleroot(32) + time(4) + nDifficulty(8, LE)
 *
 * The 4-byte nNonce is NOT in it -- the miner picks it and appends it, and the
 * hashed header is therefore SHA256d(hdr80 || nonce) over 84 bytes, which must
 * have its top bit set (Gapcoin requires mpz_sizeinbase(hash,2) == 256).
 *
 * TWO DIFFICULTIES, and they are different things:
 *   - "difficulty" in the JSON  = the pool's SHARE target (ndiff, merit =
 *     ndiff / 2^48).
 *   - hdr80[72..79]             = the NETWORK nDifficulty of the block the
 *     pool is trying to find.  A solution above the share target earns pool
 *     credit; a solution above the NETWORK difficulty is a real block.
 *
 * There is NO share-vs-block flag in the protocol: the envelope is identical
 * and the pool classifies by merit.  What the miner sends is the PoW solution
 * (never a full block, because the pool holds the template -- its merkle root
 * is already inside the 80 bytes it handed us):
 *
 *   hdr80(80) + nNonce(4, LE) + nShift(2, LE) + nAdd(LE, >= 1 byte)
 *
 * i.e. > 86 bytes total, the same shape gapcoind's legacy getwork submit
 * expects.  nShift is chosen by the miner (our CRT cover shift) and is what
 * makes the base of the searched range h256 << shift.
 */

#ifndef STRATUM_H
#define STRATUM_H

#include <stddef.h>
#include <stdint.h>

/* 80-byte header prefix, 160 hex chars + NUL. */
#define STRATUM_HDR80_SIZE 80U
#define STRATUM_DATA_HEX_SIZE 161U

/* Smallest payload the pool/node accepts (gapcoind: total must be > 86 B). */
#define STRATUM_POW_MIN_BYTES 87U

/* Longest nAdd the pool payload carries (a CRT alignment offset is well
   inside this; the legacy uint64 path uses 8). */
#define STRATUM_NADD_MAX 64U

typedef struct stratum_ctx stratum_ctx;

/*
 * Metadata carried with a queued share so the pool's VERDICT can be logged
 * against the gap it belongs to (the verdict arrives asynchronously, keyed
 * only by JSON message id).  height is 0 in pool mode: the legacy Gapcoin pool
 * protocol does not hand out a block height.
 */
struct stratum_share_meta {
    uint32_t header_nonce;
    uint16_t shift;
    uint32_t height;
    uint32_t gap_length;
    double merit;
    size_t nadd_len;
    uint8_t nadd[STRATUM_NADD_MAX];
};

/*
 * Verdict callback, invoked from the receive thread.
 *   accepted =  1  the pool accepted the share
 *   accepted =  0  the pool rejected it (message explains why, if it did)
 *   accepted = -1  the connection dropped with the share in flight, so no
 *                  verdict ever arrived (logged as unresolved, not as reject)
 */
typedef void (*stratum_verdict_cb)(void *user, int accepted,
                                   const struct stratum_share_meta *meta,
                                   const char *message);

/* Connect to host:port, send the first mining.request and start the receive
 * thread.  Returns NULL if the socket cannot be established. */
stratum_ctx *stratum_connect(const char *host, const char *port,
                             const char *user, const char *pass);

/* Stop the receive thread and free everything.  Safe on NULL. */
void stratum_disconnect(stratum_ctx *ctx);

/* Blocking: wait (up to timeout_ms) for work and copy it out.
 * Returns 1 on success, 0 on timeout or shutdown. */
int stratum_wait_work(stratum_ctx *ctx, char data_hex[STRATUM_DATA_HEX_SIZE],
                      uint64_t *share_ndiff, uint64_t *net_ndiff,
                      unsigned timeout_ms);

/* Non-blocking: 1 if work arrived since the previous call, else 0. */
int stratum_poll_work(stratum_ctx *ctx, char data_hex[STRATUM_DATA_HEX_SIZE],
                      uint64_t *share_ndiff, uint64_t *net_ndiff);

/* Current share target / network difficulty as merit (ndiff / 2^48).
 * 0.0 when no work has been seen yet. */
double stratum_share_merit(stratum_ctx *ctx);
double stratum_network_merit(stratum_ctx *ctx);

/*
 * Build the PoW solution hex: hdr80 + nNonce(4,LE) + nShift(2,LE) + nAdd(LE).
 * nadd_len is normalized to a minimum of one byte.  Returns the hex length
 * (strlen) on success, 0 on bad arguments or insufficient out_cap.
 */
size_t stratum_pow_hex(const uint8_t hdr80[STRATUM_HDR80_SIZE], uint32_t nonce,
                       uint16_t shift, const uint8_t *nadd, size_t nadd_len,
                       char *out, size_t out_cap);

/* Convenience: assemble and queue the share.  Duplicates (same payload since
 * the last new work) are dropped locally and counted, because the pool rejects
 * them.  Returns 1 if queued, 0 otherwise. */
int stratum_submit_pow(stratum_ctx *ctx, const uint8_t hdr80[STRATUM_HDR80_SIZE],
                       uint32_t nonce, uint16_t shift, const uint8_t *nadd,
                       size_t nadd_len);

/* Same, but carries the gap metadata that the verdict callback reports back.
 * `meta` may be NULL (equivalent to stratum_submit_pow()). */
int stratum_submit_share(stratum_ctx *ctx, const uint8_t hdr80[STRATUM_HDR80_SIZE],
                         uint32_t nonce, uint16_t shift, const uint8_t *nadd,
                         size_t nadd_len,
                         const struct stratum_share_meta *meta);

/* Register the verdict callback (single slot, replaces any previous one). */
void stratum_set_verdict_callback(stratum_ctx *ctx, stratum_verdict_cb cb,
                                  void *user);

/*
 * Classic getwork semantics: the pool may rotate its block template WITHOUT
 * pushing anything, and then rejects every share built on the old header --
 * suprnova does exactly this, with no error message, which is what makes it
 * hard to attribute.  The miner therefore re-requests work on a timer
 * (`seconds`, 0 disables) and only republishes work that actually CHANGED, so
 * an unchanged answer costs nothing (no chain restart, no dropped shares).
 * The default is 10 s; override with the STRATUM_REFRESH_S environment
 * variable when a pool rotates faster or slower.
 */
void stratum_set_refresh_seconds(stratum_ctx *ctx, unsigned seconds);

/* Send a work request right now (used on reconnect and after rejections). */
int stratum_request_work(stratum_ctx *ctx);

/* Cumulative counters.  Any pointer may be NULL.  `reconnects` counts
   successful reconnections after the first connect, `connect_failures` the
   attempts that never established a session, and `unresolved` the shares
   whose verdict never arrived because the connection dropped. */
void stratum_get_stats(stratum_ctx *ctx, uint64_t *accepted,
                       uint64_t *rejected, uint64_t *duplicates,
                       uint64_t *send_failures, uint64_t *reconnects,
                       uint64_t *connect_failures, uint64_t *unresolved);

int stratum_is_connected(stratum_ctx *ctx);

/* Pure helpers, exposed for tests. */
uint64_t stratum_net_ndiff_from_header(const uint8_t hdr80[STRATUM_HDR80_SIZE]);
double stratum_ndiff_to_merit(uint64_t ndiff);

#endif /* STRATUM_H */
