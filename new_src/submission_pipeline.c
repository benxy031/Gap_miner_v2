/* Gap Submission Pipeline Implementation */
#include "submission_pipeline.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

SubmissionPipeline *submission_pipeline_create(
    struct gapcoin_rpc *rpc,
    uint64_t coinbase_reward,
    const char *miner_address)
{
    if (!rpc || !miner_address) return NULL;
    
    SubmissionPipeline *pipeline = malloc(sizeof(SubmissionPipeline));
    if (!pipeline) return NULL;
    
    pipeline->rpc = rpc;
    pipeline->coinbase_reward = coinbase_reward;
    pipeline->miner_address = miner_address;
    
    return pipeline;
}

int submission_pipeline_submit_gap(
    SubmissionPipeline *pipeline,
    const GapRecord *gap,
    const struct block_template *template)
{
    if (!pipeline || !gap || !template) return -1;
    
    printf("[Submission] Processing gap: length=%u, merit=%.2f\n", 
           gap->gap_length, gap->merit_score);
    
    /* Create block assembly */
    AssembledBlock *block = block_assembly_create();
    if (!block) {
        fprintf(stderr, "[Submission] Failed to create block\n");
        return -1;
    }
    
    /* Parse previous block hash from template (hex string to bytes) */
    uint8_t prev_hash[32];
    if (strlen(template->previousblockhash) != 64) {
        fprintf(stderr, "[Submission] Invalid previousblockhash length\n");
        block_assembly_free(block);
        return -1;
    }
    
    for (int i = 0; i < 32; i++) {
        sscanf(template->previousblockhash + i * 2, "%2hhx", &prev_hash[i]);
    }
    
    /* Set block header */
    uint32_t bits_val = 0;  /* Would need to parse from template->bits */
    block_assembly_set_header(
        block,
        template->version,
        prev_hash,
        template->curtime,
        bits_val,
        0  /* nonce = 0, will be searched */
    );
    
    /* Build coinbase transaction with gap data */
    CoinbaseInfo coinbase = {
        .height = gap->height,
        .gap_length = gap->gap_length,
        .first_prime = gap->first_prime,
        .shift = gap->shift,
        .nonce_val = gap->nonce_val,
        .merit_score = gap->merit_score,
        .coinbase_reward = pipeline->coinbase_reward,
        .miner_address = pipeline->miner_address
    };
    
    if (block_assembly_build_coinbase(block, &coinbase) < 0) {
        fprintf(stderr, "[Submission] Failed to build coinbase\n");
        block_assembly_free(block);
        return -1;
    }
    
    /* Finalize block */
    if (block_assembly_finalize(block) < 0) {
        fprintf(stderr, "[Submission] Failed to finalize block\n");
        block_assembly_free(block);
        return -1;
    }
    
    /* Convert to hex */
    char *block_hex = block_assembly_to_hex(block);
    if (!block_hex) {
        fprintf(stderr, "[Submission] Failed to convert block to hex\n");
        block_assembly_free(block);
        return -1;
    }
    
    printf("[Submission] Block hex length: %zu bytes\n", strlen(block_hex) / 2);
    
    /* Submit via RPC */
    int result = gapcoin_rpc_submit_block(pipeline->rpc, block_hex);
    if (result == 0) {
        printf("[Submission] ✓ Block submitted successfully!\n");
        printf("[Submission] Gap: length=%u, merit=%.2f, reward=%.8f BTC\n",
               gap->gap_length, gap->merit_score, 
               (double)pipeline->coinbase_reward / 100000000.0);
    } else {
        fprintf(stderr, "[Submission] ✗ Block submission failed\n");
    }
    
    free(block_hex);
    block_assembly_free(block);
    
    return result;
}

void submission_pipeline_free(SubmissionPipeline *pipeline) {
    if (pipeline) {
        free(pipeline);
    }
}
