/*
 * Gapcoin GBT Work Materialization
 */

#define _POSIX_C_SOURCE 200809L

#include "gapcoin_work.h"
#include <openssl/sha.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Real payout scriptPubKey (hex); NULL falls back to OP_TRUE. */
static char *g_payout_script_hex = NULL;

void gapcoin_work_set_payout_script_hex(const char *script_hex) {
    free(g_payout_script_hex);
    g_payout_script_hex = (script_hex && script_hex[0]) ? strdup(script_hex) : NULL;
}

static void write_le32(uint8_t *output, uint32_t value) {
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8);
    output[2] = (uint8_t)(value >> 16);
    output[3] = (uint8_t)(value >> 24);
}

static void write_le64(uint8_t *output, uint64_t value) {
    for (size_t index = 0; index < 8; index++) {
        output[index] = (uint8_t)(value >> (index * 8));
    }
}

static int hex_nibble(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static int decode_hex(const char *hex, uint8_t *output, size_t output_size,
                      int reverse) {
    if (!hex || strlen(hex) != output_size * 2U) return -1;

    for (size_t index = 0; index < output_size; index++) {
        int high = hex_nibble(hex[index * 2U]);
        int low = hex_nibble(hex[index * 2U + 1U]);
        if (high < 0 || low < 0) return -1;
        size_t output_index = reverse ? output_size - 1U - index : index;
        output[output_index] = (uint8_t)((high << 4) | low);
    }
    return 0;
}

static void double_sha256(const uint8_t *data, size_t data_size,
                          uint8_t hash[32]) {
    uint8_t intermediate[SHA256_DIGEST_LENGTH];
    SHA256(data, data_size, intermediate);
    SHA256(intermediate, sizeof(intermediate), hash);
}

static size_t encode_height(uint32_t height, uint8_t output[5]) {
    size_t size = 0;
    do {
        output[size++] = (uint8_t)height;
        height >>= 8;
    } while (height != 0);

    if (output[size - 1U] & 0x80U) {
        output[size++] = 0;
    }
    return size;
}

static size_t write_compact_size(uint8_t *out, uint64_t value) {
    if (value < 0xfdULL) {
        out[0] = (uint8_t)value;
        return 1;
    } else if (value <= 0xffffULL) {
        out[0] = 0xfd;
        out[1] = (uint8_t)value;
        out[2] = (uint8_t)(value >> 8);
        return 3;
    } else if (value <= 0xffffffffULL) {
        out[0] = 0xfe;
        write_le32(out + 1, (uint32_t)value);
        return 5;
    }
    out[0] = 0xff;
    write_le64(out + 1, value);
    return 9;
}

static void bytes_to_hex(const uint8_t *data, size_t len, char *out_hex) {
    static const char digits[] = "0123456789abcdef";
    for (size_t index = 0; index < len; index++) {
        out_hex[index * 2U] = digits[data[index] >> 4];
        out_hex[index * 2U + 1U] = digits[data[index] & 0x0FU];
    }
    out_hex[len * 2U] = '\0';
}

/* Raw coinbase tx bytes (caller frees); reused for both hashing and final
   block serialization so the merkle root always matches the submitted tx. */
static uint8_t *build_coinbase_tx(const struct block_template *tmpl,
                                  const char *payout_script_hex,
                                  size_t *out_len) {
    uint8_t script[128];
    uint8_t height[5];
    size_t script_size = 0;
    size_t height_size = encode_height(tmpl->height, height);

    script[script_size++] = (uint8_t)height_size;
    memcpy(script + script_size, height, height_size);
    script_size += height_size;
    script[script_size++] = 0x00; /* OP_0, matching CreateNewBlock(). */

    if (tmpl->coinbaseaux) {
        size_t flags_size = strlen(tmpl->coinbaseaux) / 2U;
        if (strlen(tmpl->coinbaseaux) % 2U != 0 ||
            script_size + flags_size > sizeof(script) ||
            decode_hex(tmpl->coinbaseaux, script + script_size,
                       flags_size, 0) != 0) {
            return NULL;
        }
        script_size += flags_size;
    }

    if (script_size < 2 || script_size > 100) return NULL;

    uint8_t payout[512];
    size_t payout_size;
    if (payout_script_hex && payout_script_hex[0]) {
        size_t hex_len = strlen(payout_script_hex);
        payout_size = hex_len / 2U;
        if (hex_len % 2U != 0 || payout_size == 0 ||
            payout_size > sizeof(payout) ||
            decode_hex(payout_script_hex, payout, payout_size, 0) != 0) {
            return NULL;
        }
    } else {
        payout[0] = 0x51; /* OP_TRUE anyone-can-spend placeholder. */
        payout_size = 1;
    }

    uint8_t *transaction = malloc(256U + script_size + payout_size);
    if (!transaction) return NULL;
    size_t transaction_size = 0;

    write_le32(transaction + transaction_size, 1);
    transaction_size += 4;
    transaction[transaction_size++] = 1; /* One coinbase input. */
    memset(transaction + transaction_size, 0, 32);
    transaction_size += 32;
    write_le32(transaction + transaction_size, UINT32_MAX);
    transaction_size += 4;
    transaction[transaction_size++] = (uint8_t)script_size;
    memcpy(transaction + transaction_size, script, script_size);
    transaction_size += script_size;
    write_le32(transaction + transaction_size, UINT32_MAX);
    transaction_size += 4;
    transaction[transaction_size++] = 1; /* One payout output. */
    write_le64(transaction + transaction_size, tmpl->coinbasevalue);
    transaction_size += 8;
    transaction_size += write_compact_size(transaction + transaction_size,
                                           (uint64_t)payout_size);
    memcpy(transaction + transaction_size, payout, payout_size);
    transaction_size += payout_size;
    write_le32(transaction + transaction_size, 0);
    transaction_size += 4;

    *out_len = transaction_size;
    return transaction;
}

static int build_coinbase_hash(const struct block_template *tmpl,
                               uint8_t hash[32]) {
    size_t tx_len = 0;
    uint8_t *tx = build_coinbase_tx(tmpl, g_payout_script_hex, &tx_len);
    if (!tx) return -1;
    double_sha256(tx, tx_len, hash);
    free(tx);
    return 0;
}

static int build_merkle_root(const struct block_template *tmpl,
                             uint8_t merkle_root[32]) {
    size_t hash_count = tmpl->transaction_count + 1U;
    uint8_t *hashes = calloc(hash_count, 32);
    if (!hashes) return -1;

    if (build_coinbase_hash(tmpl, hashes) != 0) {
        free(hashes);
        return -1;
    }

    for (size_t index = 0; index < tmpl->transaction_count; index++) {
           /* Compute double-SHA256 from raw transaction data (not pre-computed txid) */
           const char *tx_hex = tmpl->transaction_data[index];
           if (!tx_hex || strlen(tx_hex) == 0 || strlen(tx_hex) % 2 != 0) {
               fprintf(stderr, "[build_merkle_root] TX[%zu] has invalid hex data\n", index);
               free(hashes);
               return -1;
           }
           size_t tx_len = strlen(tx_hex) / 2;
           uint8_t *tx_bytes = malloc(tx_len);
           if (!tx_bytes) {
               free(hashes);
               return -1;
           }
           if (decode_hex(tx_hex, tx_bytes, tx_len, 0) != 0) {
               fprintf(stderr, "[build_merkle_root] Failed to decode TX[%zu] hex\n", index);
               free(tx_bytes);
               free(hashes);
               return -1;
           }
           double_sha256(tx_bytes, tx_len, hashes + ((index + 1U) * 32U));
           free(tx_bytes);
    }

    while (hash_count > 1U) {
        size_t next_count = (hash_count + 1U) / 2U;
        for (size_t index = 0; index < hash_count; index += 2U) {
            size_t right = index + 1U < hash_count ? index + 1U : index;
            uint8_t pair[64];
            memcpy(pair, hashes + (index * 32U), 32);
            memcpy(pair + 32, hashes + (right * 32U), 32);
            double_sha256(pair, sizeof(pair), hashes + ((index / 2U) * 32U));
        }
        hash_count = next_count;
    }

    memcpy(merkle_root, hashes, 32);
    free(hashes);
    return 0;
}

int gapcoin_gbt_work_init(struct gapcoin_gbt_work *work,
                          const struct block_template *tmpl) {
    uint8_t merkle_root[32];
    if (!work || !tmpl || !tmpl->previousblockhash || tmpl->difficulty == 0 ||
        build_merkle_root(tmpl, merkle_root) != 0) {
        return -1;
    }

    memset(work, 0, sizeof(*work));
    write_le32(work->header_prefix, tmpl->version);
    if (decode_hex(tmpl->previousblockhash, work->header_prefix + 4, 32, 1) != 0) {
        return -1;
    }
    memcpy(work->header_prefix + 36, merkle_root, sizeof(merkle_root));
    write_le32(work->header_prefix + 68, tmpl->curtime);
    write_le64(work->header_prefix + 72, tmpl->difficulty);
    work->nonce = 0;
    return 0;
}

int gapcoin_gbt_work_hash(struct gapcoin_gbt_work *work, uint8_t h256[32]) {
    uint8_t header[GAPCOIN_HASHED_HEADER_SIZE];
    uint8_t raw_hash[32];
    if (!work || !h256) return -1;

    for (;;) {
        memcpy(header, work->header_prefix, sizeof(work->header_prefix));
        write_le32(header + 80, work->nonce);
        double_sha256(header, sizeof(header), raw_hash);

        /* Core's PoW imports uint256 little-endian and requires 256 bits. */
        if (raw_hash[31] & 0x80U) {
            for (size_t index = 0; index < sizeof(raw_hash); index++) {
                h256[index] = raw_hash[sizeof(raw_hash) - 1U - index];
            }
            return 0;
        }

        if (work->nonce == UINT32_MAX) return -1;
        work->nonce++;
    }
}

int gapcoin_gbt_hash_nonce(const uint8_t hdr80[80], uint32_t nonce,
                           uint8_t h256[32]) {
    uint8_t header[GAPCOIN_HASHED_HEADER_SIZE];
    uint8_t raw_hash[32];
    if (!hdr80 || !h256) return -1;

    memcpy(header, hdr80, 80);
    write_le32(header + 80, nonce);
    double_sha256(header, sizeof(header), raw_hash);

    /* Core's PoW imports uint256 little-endian and requires 256 bits. */
    if (!(raw_hash[31] & 0x80U)) {
        return -1;
    }
    for (size_t index = 0; index < sizeof(raw_hash); index++) {
        h256[index] = raw_hash[sizeof(raw_hash) - 1U - index];
    }
    return 0;
}

int gapcoin_gbt_work_next_hash(struct gapcoin_gbt_work *work,
                               uint8_t h256[32]) {
    if (!work || work->nonce == UINT32_MAX) return -1;
    work->nonce++;
    return gapcoin_gbt_work_hash(work, h256);
}

int gapcoin_gbt_work_build_submission_bytes(const uint8_t header_prefix[80],
                                            uint32_t header_nonce,
                                            const struct block_template *tmpl,
                                            uint32_t shift,
                                            const uint8_t *nadd,
                                            size_t nadd_len,
                                            char *out_hex, size_t out_hex_cap) {
    if (!header_prefix || !tmpl || !out_hex || out_hex_cap == 0) return -1;

    /* nAdd: raw little-endian bytes, minimum one zero byte. */
    if (!nadd || nadd_len == 0) {
        static const uint8_t zero = 0;
        nadd = &zero;
        nadd_len = 1;
    }
    /* Strip redundant leading (most-significant) zero bytes, keep >= 1. */
    while (nadd_len > 1 && nadd[nadd_len - 1] == 0) {
        nadd_len--;
    }

    uint8_t *header = (uint8_t *)malloc(80U + 4U + 2U + 9U + nadd_len);
    if (!header) return -1;
    size_t header_len = 0;
    memcpy(header, header_prefix, 80);
    header_len += 80;
    write_le32(header + header_len, header_nonce);
    header_len += 4;
    header[header_len] = (uint8_t)shift;
    header[header_len + 1U] = (uint8_t)(shift >> 8);
    header_len += 2;
    header_len += write_compact_size(header + header_len, (uint64_t)nadd_len);
    memcpy(header + header_len, nadd, nadd_len);
    header_len += nadd_len;

    size_t coinbase_len = 0;
    uint8_t *coinbase_tx = build_coinbase_tx(tmpl, g_payout_script_hex, &coinbase_len);
    if (!coinbase_tx) {
        free(header);
        return -1;
    }

    size_t total_txs = 1U + tmpl->transaction_count;
    uint8_t **tx_bytes = calloc(total_txs, sizeof(*tx_bytes));
    size_t *tx_lens = calloc(total_txs, sizeof(*tx_lens));
    if (!tx_bytes || !tx_lens) {
        free(coinbase_tx);
        free(tx_bytes);
        free(tx_lens);
        free(header);
        return -1;
    }
    tx_bytes[0] = coinbase_tx;
    tx_lens[0] = coinbase_len;

    int ok = 1;
    for (size_t index = 0; ok && index < tmpl->transaction_count; index++) {
        const char *hex = tmpl->transaction_data ? tmpl->transaction_data[index] : NULL;
        size_t len = hex ? strlen(hex) / 2U : 0U;
        uint8_t *bytes = (hex && len > 0U) ? malloc(len) : NULL;
        if (!hex || len == 0U || strlen(hex) % 2U != 0U || !bytes ||
            decode_hex(hex, bytes, len, 0) != 0) {
            free(bytes);
            ok = 0;
            break;
        }
        tx_bytes[index + 1U] = bytes;
        tx_lens[index + 1U] = len;
    }

    size_t total_len = 0;
    if (ok) {
        total_len = header_len + 9U;
        for (size_t index = 0; index < total_txs; index++) {
            total_len += tx_lens[index];
        }
        if (total_len * 2U + 1U > out_hex_cap) ok = 0;
    }

    if (ok) {
        uint8_t *block = malloc(total_len);
        if (!block) {
            ok = 0;
        } else {
            size_t offset = 0;
            memcpy(block + offset, header, header_len);
            offset += header_len;
            offset += write_compact_size(block + offset, (uint64_t)total_txs);
            for (size_t index = 0; index < total_txs; index++) {
                memcpy(block + offset, tx_bytes[index], tx_lens[index]);
                offset += tx_lens[index];
            }
            bytes_to_hex(block, offset, out_hex);
            free(block);
        }
    }

    for (size_t index = 0; index < total_txs; index++) {
        free(tx_bytes[index]);
    }
    free(tx_bytes);
    free(tx_lens);
    free(header);
    return ok ? 0 : -1;
}

int gapcoin_gbt_work_build_submission(const uint8_t header_prefix[80],
                                      uint32_t header_nonce,
                                      const struct block_template *tmpl,
                                      uint32_t shift, uint64_t nadd,
                                      char *out_hex, size_t out_hex_cap) {
    if (!header_prefix || !tmpl || !out_hex || out_hex_cap == 0) return -1;

    fprintf(stderr, "[gapcoin_work] build_submission: height=%u shift=%u nadd=%llu tx_count=%zu\n",
            tmpl->height, shift, (unsigned long long)nadd, tmpl->transaction_count);

    uint8_t nadd_bytes[8];
    size_t nadd_len = 0;
    uint64_t remaining = nadd;
    if (remaining == 0) {
        nadd_bytes[0] = 0;
        nadd_len = 1;
    } else {
        while (remaining > 0) {
            nadd_bytes[nadd_len++] = (uint8_t)(remaining & 0xffU);
            remaining >>= 8;
        }
    }

    return gapcoin_gbt_work_build_submission_bytes(
        header_prefix, header_nonce, tmpl, shift, nadd_bytes, nadd_len,
        out_hex, out_hex_cap);
}