/* Block Assembly Implementation */
#include "block_assembly.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <openssl/sha.h>
#include <arpa/inet.h>

/* For little-endian byte order helpers */
#ifndef htole16
#define htole16(x) (x)  /* Assume little-endian system */
#endif
#ifndef htole32
#define htole32(x) (x)
#endif
#ifndef htole64
#define htole64(x) (x)
#endif
#ifndef le16toh
#define le16toh(x) (x)
#endif
#ifndef le32toh
#define le32toh(x) (x)
#endif
#ifndef le64toh
#define le64toh(x) (x)
#endif

/* Helper: write variable-length integer (Bitcoin VarInt) */
static size_t write_varint(uint8_t *buf, uint64_t value) {
    if (value < 0xfd) {
        buf[0] = (uint8_t)value;
        return 1;
    } else if (value <= 0xffff) {
        buf[0] = 0xfd;
        *(uint16_t *)(buf + 1) = htole16((uint16_t)value);
        return 3;
    } else if (value <= 0xffffffff) {
        buf[0] = 0xfe;
        *(uint32_t *)(buf + 1) = htole32((uint32_t)value);
        return 5;
    } else {
        buf[0] = 0xff;
        *(uint64_t *)(buf + 1) = htole64(value);
        return 9;
    }
}

/* Helper: double SHA256 hash */
static void double_sha256(const uint8_t *data, size_t len, uint8_t *hash) {
    unsigned char temp[SHA256_DIGEST_LENGTH];
    SHA256(data, len, temp);
    SHA256(temp, SHA256_DIGEST_LENGTH, hash);
}

/* Create empty block */
AssembledBlock *block_assembly_create(void) {
    AssembledBlock *block = malloc(sizeof(AssembledBlock));
    if (!block) return NULL;
    
    memset(block, 0, sizeof(AssembledBlock));
    block->header.version = 1;  /* Block version */
    return block;
}

/* Free block */
void block_assembly_free(AssembledBlock *block) {
    if (!block) return;
    if (block->coinbase_tx) free(block->coinbase_tx);
    if (block->block_data) free(block->block_data);
    free(block);
}

/* Set block header fields */
void block_assembly_set_header(
    AssembledBlock *block,
    uint32_t version,
    const uint8_t *prev_block_hash,
    uint32_t time,
    uint32_t bits,
    uint32_t nonce)
{
    block->header.version = version;
    memcpy(block->header.prev_block_hash, prev_block_hash, 32);
    block->header.time = time;
    block->header.bits = bits;
    block->header.nonce = nonce;
}

/* Build coinbase transaction */
int block_assembly_build_coinbase(
    AssembledBlock *block,
    const CoinbaseInfo *info)
{
    if (!block || !info) return -1;
    
    /* Maximum coinbase size: 1KB should be plenty */
    uint8_t *tx = malloc(1024);
    if (!tx) return -1;
    
    size_t offset = 0;
    
    /* Version (4 bytes, little-endian) */
    *(uint32_t *)(tx + offset) = 1;
    offset += 4;
    
    /* Input count (varint, 1 input) */
    offset += write_varint(tx + offset, 1);
    
    /* Input 0: previous output (null input for coinbase) */
    /* Previous TXID (32 bytes, all zeros) */
    memset(tx + offset, 0, 32);
    offset += 32;
    
    /* Previous output index (4 bytes, 0xffffffff for coinbase) */
    *(uint32_t *)(tx + offset) = 0xffffffff;
    offset += 4;
    
    /* Script length (varint) - will fill gap encoding here */
    /* Reserve space for script length and height encoding */
    uint8_t *script_len_offset = tx + offset;
    offset += 1;  /* Assume script < 253 bytes */
    
    uint8_t *script_data = tx + offset;
    size_t script_len = 0;
    
    /* Encode block height in scriptSig (BIP141 requirement) */
    /* Height as variable-length integer */
    script_len += write_varint(script_data + script_len, info->height);
    
    /* Encode gap information (simplified for now) */
    /* Format: gap_length (4) | first_prime (8) | shift (4) | nonce (4) | merit (4 as fixed-point) */
    *(uint32_t *)(script_data + script_len) = htole32(info->gap_length);
    script_len += 4;
    
    *(uint64_t *)(script_data + script_len) = htole64(info->first_prime);
    script_len += 8;
    
    *(uint32_t *)(script_data + script_len) = htole32(info->shift);
    script_len += 4;
    
    *(uint32_t *)(script_data + script_len) = htole32(info->nonce_val);
    script_len += 4;
    
    /* Merit as fixed-point (multiply by 1000 for precision) */
    uint32_t merit_fixed = (uint32_t)(info->merit_score * 1000.0);
    *(uint32_t *)(script_data + script_len) = htole32(merit_fixed);
    script_len += 4;
    
    /* Update script length */
    *script_len_offset = (uint8_t)script_len;
    offset += script_len;
    
    /* Sequence number (4 bytes) */
    *(uint32_t *)(tx + offset) = 0;
    offset += 4;
    
    /* Output count (varint, 1 output) */
    offset += write_varint(tx + offset, 1);
    
    /* Output 0: reward to miner */
    /* Value (8 bytes, little-endian satoshis) */
    *(uint64_t *)(tx + offset) = htole64(info->coinbase_reward);
    offset += 8;
    
    /* Script PubKey (simplified: P2PKH format would be complex without privkey)
       For now, use simple OP_TRUE script: just OP_1 (0x51) */
    offset += write_varint(tx + offset, 1);  /* Script length = 1 byte */
    tx[offset++] = 0x51;  /* OP_1 (placeholder - would need real pubkey script) */
    
    /* Locktime (4 bytes) */
    *(uint32_t *)(tx + offset) = 0;
    offset += 4;
    
    /* Store coinbase transaction */
    block->coinbase_tx = malloc(offset);
    if (!block->coinbase_tx) {
        free(tx);
        return -1;
    }
    memcpy(block->coinbase_tx, tx, offset);
    block->coinbase_tx_len = offset;
    free(tx);
    
    /* Calculate coinbase TX hash (double SHA256) */
    double_sha256(block->coinbase_tx, block->coinbase_tx_len, block->coinbase_tx_hash);
    
    return 0;
}

/* Calculate merkle root (simplified: only one tx = coinbase) */
int block_assembly_calculate_merkle_root(
    AssembledBlock *block,
    uint8_t *merkle_root)
{
    if (!block || !block->coinbase_tx || !merkle_root) return -1;
    
    /* For single transaction, merkle root = hash of that transaction */
    memcpy(merkle_root, block->coinbase_tx_hash, 32);
    return 0;
}

/* Finalize block */
int block_assembly_finalize(AssembledBlock *block) {
    if (!block || !block->coinbase_tx) return -1;
    
    /* Calculate merkle root */
    uint8_t merkle_root[32];
    if (block_assembly_calculate_merkle_root(block, merkle_root) < 0) return -1;
    
    /* Set merkle root in header (little-endian) */
    memcpy(block->header.merkle_root, merkle_root, 32);
    
    /* Build complete block: header (80 bytes) + transaction count + transactions */
    size_t block_size = 80 + 1 + block->coinbase_tx_len;  /* header + varint(1) + coinbase */
    
    block->block_data = malloc(block_size);
    if (!block->block_data) return -1;
    
    size_t offset = 0;
    
    /* Copy header (80 bytes) */
    memcpy(block->block_data + offset, &block->header, 80);
    offset += 80;
    
    /* Transaction count (varint = 1) */
    block->block_data[offset++] = 1;
    
    /* Copy coinbase transaction */
    memcpy(block->block_data + offset, block->coinbase_tx, block->coinbase_tx_len);
    offset += block->coinbase_tx_len;
    
    block->block_data_len = offset;
    return 0;
}

/* Convert block to hex string */
char *block_assembly_to_hex(const AssembledBlock *block) {
    if (!block || !block->block_data) return NULL;
    
    /* Each byte = 2 hex chars + null terminator */
    char *hex = malloc(block->block_data_len * 2 + 1);
    if (!hex) return NULL;
    
    for (size_t i = 0; i < block->block_data_len; i++) {
        sprintf(hex + i * 2, "%02x", block->block_data[i]);
    }
    hex[block->block_data_len * 2] = '\0';
    
    return hex;
}

/* Verify block hash meets target (simplified) */
int block_assembly_verify_hash(
    const AssembledBlock *block,
    uint32_t bits)
{
    (void)bits;  /* Simplified check below does not decode the target yet */
    if (!block) return -1;
    
    /* Calculate block header hash (double SHA256) */
    uint8_t block_hash[32];
    double_sha256((uint8_t *)&block->header, 80, block_hash);
    
    /* Convert bits to target (simplified: bits format is complex)
       For now, just check block hash is not zero */
    if (memcmp(block_hash, "\x00\x00\x00\x00\x00\x00\x00\x00"
                           "\x00\x00\x00\x00\x00\x00\x00\x00"
                           "\x00\x00\x00\x00\x00\x00\x00\x00"
                           "\x00\x00\x00\x00\x00\x00\x00\x00", 32) == 0) {
        return -1;  /* Hash is zero, which is not valid */
    }
    
    return 0;  /* Hash verification passed (simplified) */
}
