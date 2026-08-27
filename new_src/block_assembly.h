/* Block Assembly - Construct Gapcoin blocks for submission */
#ifndef BLOCK_ASSEMBLY_H
#define BLOCK_ASSEMBLY_H

#include <stdint.h>
#include <stddef.h>
#include <gmp.h>

/* Block header structure (80 bytes) */
typedef struct {
    uint32_t version;
    uint8_t prev_block_hash[32];  /* little-endian */
    uint8_t merkle_root[32];      /* little-endian */
    uint32_t time;
    uint32_t bits;
    uint32_t nonce;
} BlockHeader;

/* Coinbase transaction (simplified) */
typedef struct {
    uint32_t height;              /* Block height to encode */
    uint32_t gap_length;          /* Gap size found */
    uint64_t first_prime;         /* p1 of the gap (simplified: uint64) */
    uint32_t shift;               /* Difficulty shift value */
    uint32_t nonce_val;           /* Miner nonce */
    double merit_score;           /* Merit value */
    uint64_t coinbase_reward;     /* Block reward in satoshis */
    const char *miner_address;    /* Address to send reward to */
} CoinbaseInfo;

/* Assembled block (for submission) */
typedef struct {
    BlockHeader header;
    uint8_t *coinbase_tx;         /* Raw coinbase transaction bytes */
    size_t coinbase_tx_len;
    uint8_t coinbase_tx_hash[32]; /* TX hash (double SHA256) */
    uint8_t *block_data;          /* Full block: header + transactions */
    size_t block_data_len;
} AssembledBlock;

/* Create empty block structure */
AssembledBlock *block_assembly_create(void);

/* Free allocated block */
void block_assembly_free(AssembledBlock *block);

/* Assemble block header from template parameters */
void block_assembly_set_header(
    AssembledBlock *block,
    uint32_t version,
    const uint8_t *prev_block_hash,  /* 32 bytes, little-endian */
    uint32_t time,
    uint32_t bits,
    uint32_t nonce
);

/* Build coinbase transaction with gap encoding */
int block_assembly_build_coinbase(
    AssembledBlock *block,
    const CoinbaseInfo *info
);

/* Calculate merkle root from transactions (simplified: only coinbase) */
int block_assembly_calculate_merkle_root(
    AssembledBlock *block,
    uint8_t *merkle_root  /* Output: 32 bytes */
);

/* Finalize block: set merkle root in header, calculate full block */
int block_assembly_finalize(AssembledBlock *block);

/* Get block as hex string for submitblock RPC */
char *block_assembly_to_hex(const AssembledBlock *block);

/* Verify block hash meets difficulty target */
int block_assembly_verify_hash(
    const AssembledBlock *block,
    uint32_t bits  /* Difficulty target in bits format */
);

#endif /* BLOCK_ASSEMBLY_H */
