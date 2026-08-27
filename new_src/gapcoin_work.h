/*
 * Gapcoin GBT Work Materialization
 *
 * Constructs the hashed portion of a Gapcoin block header from a modern
 * getblocktemplate response. The nShift and nAdd fields are not hashed.
 */

#ifndef GAPCOIN_WORK_H
#define GAPCOIN_WORK_H

#include <stdint.h>
#include "gapcoin_rpc.h"

#define GAPCOIN_HASHED_HEADER_SIZE 84U

struct gapcoin_gbt_work {
    uint8_t header_prefix[80];
    uint32_t nonce;
};

/* Build a dry-run GBT header that pays its coinbase to an OP_TRUE output. */
int gapcoin_gbt_work_init(struct gapcoin_gbt_work *work,
                          const struct block_template *tmpl);

/* Hash the current nonce, advancing it until the hash has 256 bits. */
int gapcoin_gbt_work_hash(struct gapcoin_gbt_work *work, uint8_t h256[32]);

/* Stateless per-nonce hash: h256 = byte-reverse(SHA256(hdr80 || nonce)).
   Returns 0 and fills h256 when the raw hash's top bit is set (the Gapcoin
   256-bit PoW requirement); returns -1 for a nonce whose hash is invalid.
   Used by the CRT multi-header worker to hash independently-claimed nonces. */
int gapcoin_gbt_hash_nonce(const uint8_t hdr80[80], uint32_t nonce,
                           uint8_t h256[32]);

/* Advance to the next valid header nonce and return its 256-bit hash. */
int gapcoin_gbt_work_next_hash(struct gapcoin_gbt_work *work,
                               uint8_t h256[32]);

/*
 * Override the coinbase payout scriptPubKey (hex) used by future
 * gapcoin_gbt_work_init()/gapcoin_gbt_work_build_submission() calls. Pass
 * NULL or an empty string to fall back to the anyone-can-spend OP_TRUE
 * placeholder.
 */
void gapcoin_work_set_payout_script_hex(const char *script_hex);

/* Maximum hex-encoded size of a block assembled by gapcoin_gbt_work_build_submission. */
#define GAPCOIN_SUBMIT_HEX_CAP (256U * 1024U)

/*
 * Assemble a full, submittable Gapcoin block: header_prefix(80) + nonce(4) +
 * shift(2) + CompactSize(nAdd)+nAdd(LE bytes) + txcount + coinbase tx +
 * template transactions, hex-encoded into out_hex. header_nonce is the GBT
 * header nonce active when nadd was found (not necessarily work->nonce,
 * which may have advanced since). Returns 0 on success, -1 on failure
 * (e.g. a template transaction is missing raw data or the buffer is full).
 */
int gapcoin_gbt_work_build_submission(const uint8_t header_prefix[80],
                                      uint32_t header_nonce,
                                      const struct block_template *tmpl,
                                      uint32_t shift, uint64_t nadd,
                                      char *out_hex, size_t out_hex_cap);

/*
 * Variant that serializes an arbitrary-length nAdd (little-endian bytes,
 * e.g. the >64-bit CRT alignment offset) instead of a uint64.  nadd_len is
 * normalized to a minimum of one byte. */
int gapcoin_gbt_work_build_submission_bytes(const uint8_t header_prefix[80],
                                            uint32_t header_nonce,
                                            const struct block_template *tmpl,
                                            uint32_t shift,
                                            const uint8_t *nadd,
                                            size_t nadd_len,
                                            char *out_hex, size_t out_hex_cap);

#endif /* GAPCOIN_WORK_H */