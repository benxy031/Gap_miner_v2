/* Gap Submission Pipeline - Integrate block assembly + RPC */
#ifndef SUBMISSION_PIPELINE_H
#define SUBMISSION_PIPELINE_H

#include "block_assembly.h"
#include "gapcoin_rpc.h"
#include <stdint.h>

/* Gap record from worker queue */
typedef struct {
    uint32_t height;
    uint32_t shift;
    uint64_t first_prime;      /* Simplified: can only store first 64 bits */
    uint32_t gap_length;
    double merit_score;
    uint32_t nonce_val;
} GapRecord;

/* Submission pipeline state */
typedef struct {
    struct gapcoin_rpc *rpc;   /* RPC connection */
    uint64_t coinbase_reward;  /* Block reward */
    const char *miner_address; /* Mining address */
} SubmissionPipeline;

/* Create submission pipeline */
SubmissionPipeline *submission_pipeline_create(
    struct gapcoin_rpc *rpc,
    uint64_t coinbase_reward,
    const char *miner_address
);

/* Process and submit a gap */
int submission_pipeline_submit_gap(
    SubmissionPipeline *pipeline,
    const GapRecord *gap,
    const struct block_template *template
);

/* Free pipeline */
void submission_pipeline_free(SubmissionPipeline *pipeline);

#endif /* SUBMISSION_PIPELINE_H */
