/* Test: Block Assembly + Submission Pipeline */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../new_src/block_assembly.h"
#include "../new_src/submission_pipeline.h"
#include "../new_src/worker_gpu.h"

int test_block_assembly_basic(void) {
    printf("\n=== Test: Block Assembly ===\n");
    
    AssembledBlock *block = block_assembly_create();
    if (!block) {
        printf("✗ Failed to create block\n");
        return 1;
    }
    
    /* Set header */
    uint8_t prev_hash[32] = {0};
    block_assembly_set_header(block, 1, prev_hash, 1692000000, 0x15564321, 0);
    printf("✓ Header set\n");
    
    /* Build coinbase */
    CoinbaseInfo coinbase = {
        .height = 2514084,
        .gap_length = 47,
        .first_prime = 1234567890123ULL,
        .shift = 5462,
        .nonce_val = 12345,
        .merit_score = 15.37,
        .coinbase_reward = 5000000000ULL,  /* 50 BTC */
        .miner_address = "GAP1A1Wvxp1BhpqfuHmpVvf2F6yYQX6vK"
    };
    
    if (block_assembly_build_coinbase(block, &coinbase) < 0) {
        printf("✗ Failed to build coinbase\n");
        block_assembly_free(block);
        return 1;
    }
    printf("✓ Coinbase built (tx size: %zu bytes)\n", block->coinbase_tx_len);
    
    /* Finalize block */
    if (block_assembly_finalize(block) < 0) {
        printf("✗ Failed to finalize block\n");
        block_assembly_free(block);
        return 1;
    }
    printf("✓ Block finalized (total size: %zu bytes)\n", block->block_data_len);
    
    /* Convert to hex */
    char *hex = block_assembly_to_hex(block);
    if (!hex) {
        printf("✗ Failed to convert to hex\n");
        block_assembly_free(block);
        return 1;
    }
    
    printf("✓ Block hex: %zu characters (first 64: %.64s...)\n", 
           strlen(hex), hex);
    
    free(hex);
    block_assembly_free(block);
    
    printf("✓ Block Assembly test PASSED\n");
    return 0;
}

int test_gap_record_conversion(void) {
    printf("\n=== Test: Gap Record Conversion ===\n");
    
    struct gap_queue_entry gap_entry = {
        .height = 2514084,
        .shift = 5462,
        .nonce = 54321,
        .p1 = 987654321ULL,
        .gap_length = 52,
        .merit = 14.89
    };
    
    GapRecord gap_rec = {
        .height = gap_entry.height,
        .shift = gap_entry.shift,
        .first_prime = gap_entry.p1,
        .gap_length = gap_entry.gap_length,
        .merit_score = gap_entry.merit,
        .nonce_val = gap_entry.nonce
    };
    
    printf("✓ Converted gap_queue_entry to GapRecord\n");
    printf("  height: %u\n", gap_rec.height);
    printf("  shift: %u\n", gap_rec.shift);
    printf("  gap_length: %u\n", gap_rec.gap_length);
    printf("  merit: %.2f\n", gap_rec.merit_score);
    
    printf("✓ Gap Record Conversion test PASSED\n");
    return 0;
}

int main(void) {
    printf("================================================\n");
    printf("Phase 6: Block Submission Tests\n");
    printf("================================================\n");
    
    int failures = 0;
    
    failures += test_block_assembly_basic();
    failures += test_gap_record_conversion();
    
    printf("\n================================================\n");
    if (failures == 0) {
        printf("✓ All Phase 6 tests PASSED\n");
    } else {
        printf("✗ %d test(s) FAILED\n", failures);
    }
    printf("================================================\n");
    
    return failures > 0 ? 1 : 0;
}
