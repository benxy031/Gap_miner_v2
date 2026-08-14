# Gap Scanning Detailed Implementation (from Unified GapMiner)

**Date**: 2026-08-14  
**Source**: lentmaistrouve GapMiner GPU-CRT source (20260803)  
**Purpose**: Document proven gap detection patterns to integrate into clean rewrite

---

## 1. Gap Candidate Model

### 1.1 Candidate Start Context (Immutable)

```c
struct gap_candidate_start {
    mpz_t base;          /* base = hash_h256 mod shift */
    mpz_t primorial;     /* primorial = 2*3*5*7*...*p */
    
    /* Lazy materialization: don't compute full start until used */
};

void materialize_candidate_start(mpz_t dest, const struct gap_candidate_start *ctx, uint64_t row) {
    /* start(row) = base + row * primorial */
    mpz_addmul_ui(dest, ctx->primorial, row);
    mpz_add(dest, dest, ctx->base);
}
```

**Insight**: Store base + primorial separately, compute `base + row*primorial` lazily only when Fermat worker needs it. Reduces memory footprint for candidate queues.

### 1.2 Gap Candidate Descriptor

```c
struct gap_candidate {
    uint32_t nonce;              /* Nonce for this candidate batch */
    uint64_t target;             /* Minimum gap length required (difficulty) */
    
    mpz_t gap_start;             /* Full multiprecision start for this candidate */
    uint32_t n_candidates;       /* Number of offsets in this batch */
    uint32_t interval_size;      /* Total integer offsets in sieve window */
    
    /* Merit and probability stats (for scheduling/ordering) */
    double interval_merit;       /* L / ln(base) for full interval */
    double empty_probability;    /* Prob(no prime in interval) */
    double valid_probability;    /* Prob(satisfies target) */
    double equivalent_speed_factor;  /* Speedup vs. unsieved */
    double expected_fermat_tests; /* Work before result */
    double priority;             /* Scheduling objective */
    
    /* Candidate offsets within sieve window */
    vector<uint32_t> candidates; /* Offset of each candidate within interval */
};
```

---

## 2. Sieve Output: Bit-Packed Candidates

### 2.1 Sieve Bitmap Encoding

```c
/* From Sieve.h: is_prime and set_composite macros */

#define is_prime(ary, i) !bit_at(ary, i)           /* bit=0 → prime candidate */
#define set_composite(ary, i) set_bit(ary, i)      /* mark as composite */

/* Example: sieve array with 1000 candidates */
uint64_t sieve_bitmap[16];  /* 1000 bits = 16 * 64-bit words */

/* After sieving, extract candidate offsets: */
vector<uint32_t> candidates;
for (uint32_t i = 0; i < interval_size; i++) {
    if (is_prime(sieve_bitmap, i)) {
        candidates.push_back(i);  /* Offset i is candidate prime */
    }
}
```

**Optimization**: Bitmap packing reduces memory → faster GPU transfer

### 2.2 Sieve Window → Candidates

```c
void sieve_and_extract_candidates(
    struct gap_candidate *out,
    const uint8_t *h256,
    uint32_t nonce,
    const struct chinese_set *crt,
    uint64_t base_nonce_multiplier
) {
    /* Step 1: Compute base = (h256 XOR nonce*base_mult) mod primorial */
    mpz_t base;
    mpz_init(base);
    hash_to_mpz(base, h256, nonce, base_nonce_multiplier);
    
    /* Step 2: Sieve entire interval [base, base + interval_size*shift) */
    uint64_t *bitmap = (uint64_t *)malloc(crt->byte_size);
    sieve_interval(bitmap, base, crt, nonce);
    
    /* Step 3: Extract candidate offsets where is_prime(bitmap, i) */
    out->candidates.clear();
    for (uint32_t i = 0; i < crt->size; i++) {
        if (is_prime(bitmap, i)) {
            out->candidates.push_back(i);
        }
    }
    
    out->n_candidates = out->candidates.size();
    out->interval_merit = (double)crt->size / log((double)base);
    
    free(bitmap);
    mpz_clear(base);
}
```

---

## 3. Gap Detection: Consecutive Primes

### 3.1 Core Algorithm: Fermat + Gap Hunt

```c
struct gap_result {
    uint64_t gap_base;      /* Start of prime gap */
    uint32_t gap_length;    /* End - Start of gap */
    double merit;           /* gap_length / ln(gap_base) */
    uint8_t valid;          /* BPSW verified */
};

/**
 * Process Fermat results and detect gaps
 * gpu_is_prime[i] = 1 if candidate[i] is probable prime, 0 if composite
 */
vector<struct gap_result> detect_gaps(
    const struct gap_candidate *cand,
    const uint8_t *gpu_is_prime  /* 1 = probable prime, 0 = composite */
) {
    vector<struct gap_result> gaps;
    
    /* Consecutive prime pairs form gap */
    for (int i = 0; i < cand->n_candidates - 1; i++) {
        if (gpu_is_prime[i] && gpu_is_prime[i+1]) {
            
            /* Found two consecutive probable primes */
            uint32_t offset_p1 = cand->candidates[i];     /* First prime offset */
            uint32_t offset_p2 = cand->candidates[i+1];   /* Second prime offset */
            
            /* Gap length = (offset_p2 - offset_p1) * shift */
            uint32_t gap_len = (offset_p2 - offset_p1) * cand->shift;
            
            /* Gap base = base + offset_p1 * shift */
            mpz_t gap_base;
            mpz_init(gap_base);
            mpz_addmul_ui(gap_base, cand->interval_start, offset_p1);
            
            /* Merit = ln(N) = gap_length / ln(N) */
            double merit = (double)gap_len / log(mpz_get_d(gap_base));
            
            struct gap_result result = {
                .gap_base = mpz_get_ui(gap_base),
                .gap_length = gap_len,
                .merit = merit,
                .valid = 0,  /* Will BPSW verify */
            };
            gaps.push_back(result);
            mpz_clear(gap_base);
        }
    }
    
    return gaps;
}
```

**Critical insight**: Gap exists between candidates[i] and candidates[i+1], **not** between consecutive integers. The offset difference multiplied by shift gives true gap length.

### 3.2 False Gap Prevention: BPSW

```c
/**
 * Baillie-PSW: Deterministic primality for final submission
 * Combines Miller-Rabin (base 2) + Lucas test
 */
int baillie_psw_test(const mpz_t n) {
    /* Miller-Rabin (base 2) */
    if (!miller_rabin(n, 2)) return 0;  /* Composite */
    
    /* Lucas-Lehmer test */
    if (!lucas_lehmer(n)) return 0;
    
    /* Both pass → BPSW verified (no known pseudoprimes) */
    return 1;
}

/**
 * Verify gap boundaries: both endpoints must be prime
 */
int verify_gap_boundaries(mpz_t gap_base, uint32_t gap_length, uint32_t shift) {
    mpz_t p1, p2;
    mpz_init(p1);
    mpz_init(p2);
    
    /* p1 = base * 2^shift */
    mpz_mul_2exp(p1, gap_base, shift);
    
    /* p2 = (base + gap_length) * 2^shift */
    mpz_set(p2, gap_base);
    mpz_add_ui(p2, p2, gap_length);
    mpz_mul_2exp(p2, p2, shift);
    
    int valid = baillie_psw_test(p1) && baillie_psw_test(p2);
    
    mpz_clear(p1);
    mpz_clear(p2);
    
    return valid;
}
```

---

## 4. Worker Thread Architecture (Multithreaded)

### 4.1 Per-GPU Worker

```c
struct gpu_worker {
    int worker_id;
    int gpu_id;
    
    /* CRT engine (shared across workers) */
    struct chinese_set *crt;
    
    /* Sieve state (per-thread) */
    uint64_t *sieve_bitmap;
    
    /* GPU resources */
    struct cuda_context *cuda;
    struct cuda_fermat_batch *gpu_batch;  /* 4096 candidates */
    
    /* Primality */
    struct primality_ctx *prim;
    
    /* RPC and submit */
    struct rpc_interface *rpc;
    struct submit_engine *submit;
    
    /* Statistics */
    uint64_t stats_candidates_sieved;
    uint64_t stats_fermat_tests;
    uint64_t stats_gaps_found;
    uint64_t stats_gaps_submitted;
};

void *gpu_worker_thread(void *arg) {
    struct gpu_worker *w = (struct gpu_worker *)arg;
    
    cuda_set_device(w->gpu_id);
    
    while (!global_abort) {
        /* 1. Get current block from RPC */
        struct mining_pass pass = rpc_get_current_pass();
        
        /* 2. Sieve for candidates */
        struct gap_candidate cand;
        sieve_and_extract_candidates(
            &cand, pass.h256, pass.nonce, w->crt, 1ULL << 32
        );
        w->stats_candidates_sieved += cand.n_candidates;
        
        /* 3. GPU batch Fermat test */
        uint8_t gpu_results[MAX_CANDIDATES];
        cuda_fermat_batch_test(
            w->gpu_batch,
            &cand,
            16,  /* Fermat rounds */
            gpu_results
        );
        w->stats_fermat_tests += cand.n_candidates;
        
        /* 4. Detect gaps */
        vector<struct gap_result> gaps = detect_gaps(&cand, gpu_results);
        
        for (const struct gap_result &gap : gaps) {
            if (gap.merit >= global_scan_target) {
                /* BPSW final check */
                if (verify_gap_boundaries(gap.gap_base, gap.gap_length, cand.shift)) {
                    /* SUBMIT */
                    submit_engine_submit(w->submit, 
                                        pass.nonce, 
                                        gap.gap_base, 
                                        gap.gap_length);
                    w->stats_gaps_submitted++;
                }
            }
            w->stats_gaps_found++;
        }
        
        /* 5. Advance nonce (atomic) */
        advance_nonce_atomic();
    }
    
    log_msg("[worker-%d] GPU%d: %.0f candidates/s, %.0f gaps/s\n",
            w->worker_id, w->gpu_id,
            w->stats_candidates_sieved / elapsed_seconds,
            w->stats_gaps_found / elapsed_seconds);
    
    return NULL;
}
```

### 4.2 Multi-Worker Coordination

```c
struct miner_farm {
    int n_workers;
    struct gpu_worker workers[MAX_GPUS];
    pthread_t threads[MAX_GPUS];
    
    /* Shared state */
    _Atomic uint32_t current_nonce;
    pthread_mutex_t nonce_lock;
};

int miner_farm_launch(int gpu_count, const struct chinese_set *crt) {
    struct miner_farm farm;
    farm.n_workers = gpu_count;
    
    for (int i = 0; i < gpu_count; i++) {
        farm.workers[i].worker_id = i;
        farm.workers[i].gpu_id = i;
        farm.workers[i].crt = crt;  /* Shared */
        farm.workers[i].gpu_batch = cuda_batch_alloc(4096);
        farm.workers[i].cuda = cuda_init(i);
        
        pthread_create(&farm.threads[i], NULL, 
                      gpu_worker_thread, &farm.workers[i]);
    }
    
    /* Wait for all workers */
    for (int i = 0; i < gpu_count; i++) {
        pthread_join(farm.threads[i], NULL);
    }
    
    return 0;
}
```

---

## 5. CRT File-Driven Configuration

### 5.1 ChineseSet Loading

```c
struct chinese_set {
    uint32_t n_primes;        /* Number of primes in primorial */
    uint32_t size;            /* Sieve interval size (offsets) */
    uint32_t byte_size;       /* Bitmap size in bytes */
    
    mpz_t primorial;          /* 2*3*5*7*... */
    mpz_t offset;             /* Offset for this set */
    uint32_t n_candidates;    /* Expected candidates in sieve */
    
    uint32_t bit_size;        /* Minimum shift for this set */
    uint64_t *sieve;          /* Pre-sieved bitmap */
    
    double avg_candidates;    /* Average candidates per nonce */
    double max_merit;         /* Max theoretical merit */
};

struct chinese_set *chinese_set_load(const char *fname) {
    FILE *f = fopen(fname, "rb");
    struct chinese_set *cs = malloc(sizeof(*cs));
    
    /* Binary format (from GapMiner-Unified) */
    fread(&cs->n_primes, 4, 1, f);
    fread(&cs->size, 4, 1, f);
    fread(&cs->n_candidates, 4, 1, f);
    
    /* Load primorial and offset from GMP serialization */
    mpz_init(cs->primorial);
    mpz_init(cs->offset);
    mpz_inp_raw(cs->primorial, f);
    mpz_inp_raw(cs->offset, f);
    
    /* Load pre-sieved bitmap */
    cs->byte_size = (cs->size + 7) / 8;
    cs->sieve = malloc(cs->byte_size);
    fread(cs->sieve, 1, cs->byte_size, f);
    
    fclose(f);
    return cs;
}
```

### 5.2 Runtime: Nonce Loop with CRT

```c
void mine_with_crt(struct chinese_set *crt) {
    for (uint32_t nonce = 0; nonce < UINT32_MAX && !global_abort; nonce++) {
        /* Spawn GPU workers for this nonce */
        struct gap_candidate cand;
        sieve_and_extract_candidates(
            &cand,
            current_block_hash,
            nonce,
            crt,
            1ULL << 32
        );
        
        /* All GPU workers share same nonce's candidates */
        /* They test Fermat in parallel, find gaps independently */
        
        if (new_block_detected()) {
            nonce = 0;  /* Reset on new block */
        }
    }
}
```

---

## 6. Key Differences from Old Miner

| Aspect | Old (monolithic) | New (Unified/Clean) |
|--------|-----------------|-------------------|
| **Candidates** | Scattered in main | Explicit `gap_candidate` struct |
| **Sieve output** | Bit array processed inline | Extracted to offset vector |
| **Gap detection** | Merged with Fermat loop | Separate detect_gaps() function |
| **Candidate start** | Full mpz_t for each | Base+row factorization, lazy materialize |
| **Multi-GPU** | Queue-based sync | Atomic nonce, workers race |
| **Fermat results** | Processed immediately | Batch 4096, then gap hunt |
| **BPSW** | Only on submit | On every high-merit gap |
| **Worker model** | Producer/Consumer | Independent per-GPU |

---

## 7. Integration Checklist for new_src/

- [ ] `gap_candidate.h` — Candidate descriptor struct
- [ ] `sieve_extraction.c` — Bitmap → offset vector
- [ ] `gap_detection.c` — Consecutive prime pair detection
- [ ] `primality_bpsw.c` — BPSW verification
- [ ] `chinese_set.c` — CRT file loading
- [ ] `worker_gpu.c` — Per-GPU worker thread
- [ ] `miner_farm.c` — Multi-worker coordination
- [ ] `atomic_nonce.c` — Lock-free nonce advancement

---

## References

- `ChineseSet.h/cpp` — CRT file format and loading
- `GapCandidate.h/cpp` — Candidate data structures
- `Sieve.h` — Bitmap encoding (is_prime, set_composite macros)
- `PoW.h/cpp` — Gap verification and merit calculation
- `Miner.cpp` — Worker thread loop (lines ~200–400)
- `ShareProcessor.h/cpp` — Result queue and submission
