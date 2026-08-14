# New Miner Architecture — Clean Redesign Plan

**Status**: Planning phase  
**Date**: 2026-08-14  
**Scope**: Rewrite miner core while preserving CRT, RPC, and submit logic

---

## 📚 Reference Documents

- **[GAP_SCANNING_FLOW.md](GAP_SCANNING_FLOW.md)** — Complete visual flow diagrams: RPC → Sieve → Fermat → Gap Detection → Submit
- **[GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md)** — Proven implementation patterns from unified GapMiner (candidate management, bitmap encoding, BPSW)
- **[UNIFIED_GAPMINER_REFERENCE.md](UNIFIED_GAPMINER_REFERENCE.md)** — Source code mapping: which files to study for each module

**Quick Links**:
- Gap detection algorithm: [GAP_SCANNING_DETAILED.md §3](GAP_SCANNING_DETAILED.md#3-gap-detection-consecutive-primes)
- Worker thread architecture: [GAP_SCANNING_DETAILED.md §4](GAP_SCANNING_DETAILED.md#4-worker-thread-architecture-multithreaded)
- CRT file loading: [GAP_SCANNING_DETAILED.md §5](GAP_SCANNING_DETAILED.md#5-crt-file-driven-configuration)
- Flow diagrams: [GAP_SCANNING_FLOW.md](GAP_SCANNING_FLOW.md)
- Implementation checklist: [UNIFIED_GAPMINER_REFERENCE.md §7](UNIFIED_GAPMINER_REFERENCE.md#new-architecture-checklist-files-to-create-from-unified-patterns)

---

## 1. Current State: What's Broken

The existing codebase (`src/main.c` — 13,137 lines) is monolithic and difficult to maintain:

- **main.c**: Everything crammed together — CLI parsing, RPC, sieve, CRT, GPU, stats, mining loop
- **Sieve logic** is tangled with window processing
- **GPU/CPU path** is mixed in the same flow
- **RPC poll + submit** is tightly coupled to worker threads
- **No clean separation** between discovery, validation, and submission

**Result**: Hard to fix bugs, hard to add features, hard to reason about behavior.

---

## 2. Preserve: Core Components That Stay

### 2.1 CRT Runtime Pipeline
From `src/crt_runtime_cpu.c` and related:
- `compute_cramer_score()` — scoring function (from main.c)
- `crt_runtime_prepare_solver_nonce()` — nonce setup
- `crt_runtime_process_solver_window()` — gap search logic
- `crt_gap_scan_for_nonce()` — window sizing and modes
- CRT file loader with shift alignment checks

**Action**: Extract into `new_src/crt_engine.h` and `new_src/crt_engine.c` (no rewrites, direct copy + minimal interface cleanup)

### 2.2 RPC + Stratum Protocol
From `src/stratum.c` and RPC wrappers:
- Stratum polling and message handling
- getwork/getblocktemplate support
- Header parsing and block template handling
- NTime update logic

**Action**: Keep `src/stratum.c` as-is. Provide clean interface in `new_src/rpc_interface.h` that wraps existing RPC calls.

### 2.3 Submit Logic: Both getwork and getblocktemplate Paths

From `src/main.c` (submit path):
- `assemble_mining_block()` — builds submit payload for both modes
- PoW verification logic
- JSON encoding for both getwork and submitblock
- Stale-header guard (sequence checking)

**Two submit methods preserved**:

1. **Legacy getwork** (`RPC_MINING_GETWORK = 1`):
   - Input: 80-byte header from getwork (no nonce)
   - Output: header + nonce + shift + nAdd (raw LE)
   - Submit via: `getwork([data])`
   - Best for: older wallets, backward compatibility

2. **Modern getblocktemplate + submitblock** (`RPC_MINING_GBT = 2`):
   - Input: full block template from getblocktemplate
   - Output: full serialized block hex (coinbase tx + block header)
   - Submit via: `submitblock([blockhex])`
   - Best for: modern nodes, visible coinbase outputs, deterministic rewards
   - Optional: `--coinbase-script-hex` for custom script (override wallet output)

**Action**: Extract to `new_src/submit_engine.h` + `new_src/submit_engine.c`:
```c
enum submit_method {
    SUBMIT_GETWORK = 1,
    SUBMIT_GBT = 2,
};

struct submit_engine {
    enum submit_method mode;
    const char *rpc_url;
    const char *rpc_user, *rpc_pass;
    
    /* Cached templates */
    struct gbt_template *gbt_cache;       /* For GBT mode */
    uint8_t coinbase_script_hex[256];     /* Override script */
};

int submit_gap(
    struct submit_engine *submit,
    const uint8_t h256[32],               /* Current block hash */
    uint32_t nonce,
    uint64_t nAdd,
    uint32_t shift,
    uint64_t pass_id                      /* Sequence guard */
);
```

**Initialization** (in new `new_src/rpc_interface.c`):
```c
int rpc_detect_submit_mode(struct rpc_interface *rpc) {
    /* Try getblocktemplate first (modern node) */
    if (rpc_getblocktemplate(rpc->url, rpc->user, rpc->pass)) {
        rpc->submit_mode = SUBMIT_GBT;
        log_msg("Using getblocktemplate + submitblock\n");
        return 1;
    }
    
    /* Fallback to getwork (legacy node) */
    if (rpc_getwork(rpc->url, rpc->user, rpc->pass)) {
        rpc->submit_mode = SUBMIT_GETWORK;
        log_msg("Using getwork (legacy)\n");
        return 1;
    }
    
    return 0;  /* Neither available */
}
```

**Payload building** (in new `new_src/submit_engine.c`):
```c
int submit_gap(...) {
    char blockhex[SUBMIT_BLOCK_HEX_CAP];
    
    if (submit->mode == SUBMIT_GETWORK) {
        /* Build 80-byte header + nonce + shift + nAdd */
        if (!assemble_getwork_payload(h256, nonce, shift, nAdd, blockhex)) {
            return 0;  /* Stale or invalid */
        }
        return rpc_submit(submit->rpc_url, submit->rpc_user, submit->rpc_pass, 
                         "getwork", blockhex);
    } else if (submit->mode == SUBMIT_GBT) {
        /* Build full block (coinbase tx + header) from cached template */
        if (!assemble_gbt_payload(submit->gbt_cache, h256, nonce, shift, nAdd,
                                 submit->coinbase_script_hex, blockhex)) {
            return 0;  /* Stale or invalid */
        }
        return rpc_submit(submit->rpc_url, submit->rpc_user, submit->rpc_pass,
                         "submitblock", blockhex);
    }
    return 0;
}
```

### 2.4 CRT File Generation and GPU Execution

**CRT file generation** (`new_src/crt_generator.c`):

From `tools/gen_crt.c` — extract generation logic for on-the-fly CRT file creation:

```c
struct crt_generator_cfg {
    uint32_t shift;
    uint32_t gap_target;
    uint32_t merit;
    
    /* Auto-tune (optional) */
    int auto_tune_primes;       /* Auto-select prime count */
    int auto_tune_merit;        /* Auto-adjust merit based on shift */
};

struct crt_file *crt_generate(struct crt_generator_cfg *cfg) {
    /* Calls existing gen_crt logic:
       1. Generate prime/offset pairs via sieve
       2. Build CRT system (Chinese Remainder Theorem)
       3. Compute candidate count per nonce
       4. Validate coverage (no drift)
       5. Return in-memory CRT file struct
    */
}

int crt_save_to_file(struct crt_file *crt, const char *filename);
```

**Auto-tuning** (optional, based on shift):
```c
void crt_auto_tune_for_shift(struct crt_generator_cfg *cfg, uint32_t shift) {
    cfg->shift = shift;
    
    if (shift <= 300) {
        cfg->gap_target = 50;
        cfg->merit = 12;
        cfg->auto_tune_primes = 300;  /* Smaller CRT */
    } else if (shift <= 512) {
        cfg->gap_target = 100;
        cfg->merit = 15;
        cfg->auto_tune_primes = 1000;
    } else if (shift <= 768) {
        cfg->gap_target = 150;
        cfg->merit = 18;
        cfg->auto_tune_primes = 3000;
    } else {
        cfg->gap_target = 200;
        cfg->merit = 20;
        cfg->auto_tune_primes = 5000;
    }
}
```

**CLI auto-generation**:
```bash
# Auto-generate CRT for shift 512, save to file
./gap_miner -s 512 --crt-generate --crt-file crt_s512_m20.txt

# Or load from file if exists
./gap_miner -s 512 --crt-file crt_s512_m20.txt

# Or inline: on startup, generate if missing
./gap_miner -s 512 --crt-auto-generate
```

---

### 2.5 CRT GPU Execution: Multithreaded with GPU Backend

**Architecture**: N worker threads, each with GPU affinity:
- **CPU workers** (shift < 512): Standard sieve + Fermat/Euler
- **GPU workers** (shift ≥ 512): Sieve on CPU + batch Fermat on GPU
- All workers race to find gaps, submit independently
- No producer/consumer bottleneck

**Worker thread per GPU** (`new_src/worker_gpu.c`):

```c
struct worker_gpu_ctx {
    int worker_id;
    int gpu_id;                  /* CUDA/OpenCL device */
    
    crt_engine *crt;
    struct sieve_state *sieve;   /* Per-thread sieve */
    struct primality_ctx *prim;  /* CPU primality (trial div) */
    struct cuda_fermat_batch *gpu_batch;  /* GPU batch processor */
    
    struct rpc_interface *rpc;
    struct submit_engine *submit;
    
    uint64_t stats_nonces_processed;
    uint64_t stats_gaps_found;
};

void *worker_gpu_main(void *arg) {
    struct worker_gpu_ctx *ctx = (struct worker_gpu_ctx *)arg;
    struct mining_pass pass;
    
    cuda_set_device(ctx->gpu_id);  /* GPU thread affinity */
    
    while (!global_abort) {
        /* Get current pass from RPC */
        pass = *rpc_get_current_pass(ctx->rpc);
        
        /* Mine this nonce: sieve + GPU batch Fermat */
        uint64_t candidates[MAX_CANDIDATES];
        int n_cand = crt_sieve_nonce(ctx->crt, pass.h256, pass.nonce, candidates);
        
        if (n_cand > 0) {
            /* GPU: batch Fermat test */
            uint8_t gpu_results[n_cand];
            cuda_fermat_batch_test(
                ctx->gpu_batch,
                candidates, n_cand,
                16,  /* Fermat rounds */
                gpu_results
            );
            
            /* CPU: post-process, check gaps, submit */
            for (int i = 0; i < n_cand - 1; i++) {
                if (gpu_results[i] && gpu_results[i+1]) {
                    uint64_t gap_base = candidates[i];
                    uint32_t gap_len = candidates[i+1] - candidates[i];
                    double merit = (double)gap_len / log((double)gap_base);
                    
                    if (merit >= global_scan_target) {
                        /* BPSW final verification before submit */
                        if (baillie_psw_verify_boundaries(gap_base, gap_len)) {
                            submit_discovered_gap(ctx->submit, pass.nonce, 
                                                 gap_base, gap_len);
                            ctx->stats_gaps_found++;
                        }
                    }
                }
            }
        }
        
        ctx->stats_nonces_processed++;
        
        /* Advance to next nonce (shared nonce state) */
        if (!advance_nonce_atomic(&pass.nonce)) {
            global_abort = 1;  /* 2^32 nonce wrap */
        }
        
        /* Check for new block (rare) */
        if (new_block_detected()) break;
    }
    
    log_msg("[worker-%d] GPU%d: processed %llu nonces, found %llu gaps\n",
            ctx->worker_id, ctx->gpu_id,
            (unsigned long long)ctx->stats_nonces_processed,
            (unsigned long long)ctx->stats_gaps_found);
    
    return NULL;
}
```

**Main thread spawning** (in `new_src/miner_loop.c`):

```c
int launch_multithreaded_mining(int shift, int gpu_count, int cpu_count, struct rpc_interface *rpc) {
    struct worker_gpu_ctx workers[MAX_WORKERS];
    pthread_t threads[MAX_WORKERS];
    int total_workers = 0;
    
    /* GPU workers (one per GPU) */
    for (int i = 0; i < gpu_count && i < MAX_WORKERS; i++) {
        workers[total_workers].worker_id = total_workers;
        workers[total_workers].gpu_id = i;
        workers[total_workers].crt = load_crt_engine(shift);
        workers[total_workers].sieve = sieve_init(SIEVE_SIZE, SIEVE_LIMIT);
        workers[total_workers].gpu_batch = cuda_batch_alloc(4096, shift);
        workers[total_workers].rpc = rpc;
        workers[total_workers].submit = submit_engine_init(rpc);
        
        pthread_create(&threads[total_workers], NULL, 
                      worker_gpu_main, &workers[total_workers]);
        log_msg("[init] Spawned GPU worker %d on GPU %d\n", total_workers, i);
        total_workers++;
    }
    
    /* CPU workers (if shift < 512 or no GPU) */
    int cpu_workers = (shift < 512 && gpu_count == 0) ? cpu_count : 0;
    for (int i = 0; i < cpu_workers && total_workers < MAX_WORKERS; i++) {
        /* ... similar to GPU, but without cuda_batch, using CPU primality ... */
        pthread_create(&threads[total_workers], NULL,
                      worker_cpu_main, &workers[total_workers]);
        total_workers++;
    }
    
    log_msg("[init] Mining with %d total workers (%d GPU, %d CPU)\n",
            total_workers, gpu_count, cpu_workers);
    
    /* Wait for all workers to complete (until abort or nonce wrap) */
    for (int i = 0; i < total_workers; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}
```

**Atomic nonce advancement** (safe for all workers):

```c
static _Atomic uint32_t g_current_nonce = 0;

int advance_nonce_atomic(uint32_t *out_nonce) {
    uint32_t old, new;
    do {
        old = atomic_load(&g_current_nonce);
        if (old == UINT32_MAX) return 0;  /* Wrap */
        new = old + 1;
    } while (!atomic_compare_exchange_weak(&g_current_nonce, &old, new));
    
    *out_nonce = new;
    return 1;
}
```

**Worker scaling** (automatic):

| Shift | GPUs | CPU workers | Total | Throughput |
|-------|------|-------------|-------|------------|
| 256 | 0 | 8 (8-core) | 8 | ~800 gaps/sec |
| 512 | 1 RTX | 0 | 1 | ~20k gaps/sec |
| 768 | 1 A100 | 0 | 1 | ~5k gaps/sec |
| 1024 | 2 A100 | 0 | 2 | ~2k gaps/sec each |

**No consumer thread**, no queue overhead, all workers directly submit findings via RPC.

---

### Appendix: Proven Gap Scanning Architecture Reference

See [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) for detailed implementation patterns derived from lentmaistrouve's unified GapMiner (20260803):

- **Gap Candidate Model** (section 1): Immutable base+primorial, lazy materialization reduces memory
- **Sieve Output** (section 2): Bitmap encoding, extracting candidate offsets, reducing transfer overhead
- **Gap Detection** (section 3): Consecutive prime pair algorithm, proper gap length calculation
- **Multi-Worker Coordination** (section 4): Per-GPU worker design, atomic nonce advancement
- **CRT File Loading** (section 5): Binary format, primorial/offset handling, pre-sieved bitmap
- **Integration Checklist** (section 7): All new files needed for clean implementation



---

### 2.6 Automatic Parameter Tuning (No Manual Config)

**Zero-config startup**:

```c
int miner_auto_init(int shift, const char *rpc_url, const char *rpc_user, const char *rpc_pass) {
    /* 1. Auto-detect hardware */
    int gpu_count = detect_gpu_count();  /* CUDA / OpenCL */
    int cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    
    /* 2. Auto-tune CRT if not provided */
    struct crt_generator_cfg crt_cfg;
    crt_auto_tune_for_shift(&crt_cfg, shift);
    struct crt_file *crt = crt_generate(&crt_cfg);
    
    /* 3. Auto-select execution mode */
    int use_gpu = (gpu_count > 0 && shift >= 512);  /* GPU only for shift ≥ 512 */
    int num_workers = use_gpu ? gpu_count : cpu_count;
    
    /* 4. Auto-set primality strategy */
    enum primality_strategy {
        STRAT_FERMAT,      /* shift < 512 */
        STRAT_EULER,       /* shift 256-512, fast discovery */
        STRAT_HYBRID,      /* shift ≥ 512, trial div + GPU Fermat */
        STRAT_BPSW,        /* Final submit check */
    } strat;
    
    if (shift < 256)
        strat = STRAT_FERMAT;
    else if (shift < 512)
        strat = STRAT_EULER;
    else if (use_gpu)
        strat = STRAT_HYBRID;
    
    log_msg("[auto-init] Shift=%d, GPUs=%d, Workers=%d, Strategy=%s\n",
            shift, gpu_count, num_workers, strategy_name[strat]);
    
    /* 5. Auto-set sieve parameters */
    uint32_t sieve_size = (shift < 512) ? 32768 : 65536;
    uint32_t sieve_limit = (shift < 512) ? 1000000 : 10000000;
    
    log_msg("[auto-init] Sieve: size=%u, prime_limit=%u\n", sieve_size, sieve_limit);
    
    return 1;
}
```

**CLI simplified**:
```bash
# Old (manual tuning):
./gap_miner -s 512 --sieve-size 32768 --sieve-primes 1000 \
            --crt-file crt_s512_m20.txt --target 16.5 \
            -o localhost -u user -p pass

# New (automatic):
./gap_miner -s 512 -o localhost -u user -p pass
# Everything auto-detected and optimized!

# Optional: override if needed
./gap_miner -s 512 -o localhost -u user -p pass --force-cpu  # No GPU
./gap_miner -s 512 -o localhost -u user -p pass --crt-file custom.txt
```

---

## 3. Rewrite: Everything Else

### 3.1 Main Mining Loop

**Old** (`src/main.c` lines ~9000–10800): Worker threads, window loop, abort logic intertwined  
**New** (`new_src/miner_loop.c`):

```c
struct miner_worker_ctx {
    int         worker_id;
    uint32_t    shift;
    crt_engine  *crt;           /* CRT state */
    sieve_state *sieve;         /* Sieve (new, clean) */
    gap_search  *searcher;      /* Gap search engine */
};

void *worker_main(void *arg) {
    struct miner_worker_ctx *ctx = arg;
    while (!global_abort) {
        struct pass_header *pass = get_current_pass();  /* RPC layer */
        
        /* Mine this pass: generate nonces, check adder range */
        while (ctx->adder < ctx->adder_max && !global_new_block) {
            struct gap_result result = gap_search_one_nonce(
                ctx->searcher, pass->h256, ctx->nonce, ctx->adder);
            
            if (result.merit >= global_scan_target) {
                submit_gap(result);  /* Submit layer */
            }
            ctx->adder++;
        }
        
        advance_to_next_nonce();
    }
}
```

**Key differences**:
- One worker loop per thread, not a complex state machine
- Clear pass-in/pass-out semantics
- RPC polling happens in separate thread (not mixed with workers)

### 3.2 Sieve Engine (New Design)

**Old**: `src/wheel_sieve.c`, `src/presieve_utils.c`, tightly coupled to main loop  
**New** (`new_src/sieve_core.c`, `new_src/sieve_core.h`):

```c
struct sieve_state {
    uint32_t     sieve_size;
    uint32_t     *bitmap;      /* Sieve output */
    primes_list  *primes;
};

struct sieve_state *sieve_init(uint32_t size, uint32_t prime_limit);
void sieve_range(struct sieve_state *s, uint64_t base, uint64_t range);
int  sieve_next_candidate(struct sieve_state *s, uint64_t *pos);
void sieve_clear(struct sieve_state *s);
```

**Design principles**:
- Stateless per-call, not monolithic
- Simple input: `(base, range)` → output: candidate list
- No tightly coupled state machines

### 3.3 Gap Search Engine (New)

**Old**: Scattered in main.c (candidate check, Fermat, merit calc)  
**New** (`new_src/gap_search.c`, `new_src/gap_search.h`):

```c
struct gap_search {
    crt_engine *crt;
    sieve_state *sieve;
    primality_ctx *primality;    /* Fermat + trial division */
    gap_distribution *dist;       /* Merit calibration */
};

struct gap_result {
    uint64_t  base;
    uint32_t  gap_len;
    double    merit;
    uint8_t   valid;
};

struct gap_result gap_search_one_nonce(
    struct gap_search *g,
    const uint8_t h256[32],
    uint32_t nonce,
    uint64_t adder
);
```

**What it does**:
1. Calls sieve for candidate generation
2. For each candidate: trial division + Fermat test
3. Compute gap length and merit
4. Return result

**Detailed implementation patterns**: See [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) for proven techniques from unified GapMiner:
- Section 1: Gap candidate data structures and lazy materialization
- Section 2: Sieve bitmap encoding and candidate extraction
- Section 3: Gap detection algorithm (consecutive primes)
- Section 4: Worker thread architecture
- Section 5: CRT file-driven configuration

### 3.4 Primality Testing (Rewrite)

**Old**: `src/primality_utils.c` — tied to GMP init/clear everywhere  
**New** (`new_src/primality.c`, `new_src/primality.h`):

```c
struct primality_ctx {
    gmp_randstate_t rng;
    mpz_t scratch[4];
};

struct primality_ctx *primality_init(void);
int  fermat_test_ui(struct primality_ctx *p, uint64_t n, int rounds);
int  trial_division_ui(uint64_t n, uint32_t limit);
void primality_clear(struct primality_ctx *p);
```

**Changes**:
- One init/clear per thread, not per function call
- Scratch GMP objects reused, not allocated per call
- Trial division separated from Fermat

### 3.5 RPC + Pass Management (New Wrapper, Dual Submit Support)

**Old**: RPC polling mixed with worker threads  
**New** (`new_src/rpc_interface.c` + `new_src/submit_engine.c`):

```c
struct mining_pass {
    uint32_t    nonce;
    uint8_t     h256[32];
    uint64_t    ndiff;           /* difficulty */
    uint64_t    pass_id;         /* for ordering checks */
};

enum submit_method {
    SUBMIT_GETWORK = 1,          /* Legacy: 80-byte header + nonce + shift + nAdd */
    SUBMIT_GBT = 2,              /* Modern: full block template + submitblock */
};

struct rpc_interface {
    const char *url, *user, *pass;
    int stratum_mode;
    
    struct mining_pass current_pass;
    pthread_mutex_t pass_lock;
    
    /* Submit path selection */
    enum submit_method submit_method;
    struct gbt_template *gbt_cache;       /* For GBT mode only */
    uint8_t coinbase_script_hex[256];     /* Custom coinbase script (GBT) */
};

struct rpc_interface *rpc_init(const char *url, const char *user, const char *pass);
int rpc_auto_detect_submit(struct rpc_interface *rpc);  /* Try GBT first, fallback to getwork */
int rpc_poll_new_pass(struct rpc_interface *rpc, struct mining_pass *out);
struct mining_pass *rpc_get_current_pass(struct rpc_interface *rpc);
void rpc_clear(struct rpc_interface *rpc);
```

**Auto-detection on startup**:
```c
int rpc_auto_detect_submit(struct rpc_interface *rpc) {
    /* Try getblocktemplate (modern nodes: gapcoind, bitcoin-abc) */
    char *gbt = rpc_getblocktemplate(rpc->url, rpc->user, rpc->pass);
    if (gbt) {
        rpc->submit_method = SUBMIT_GBT;
        rpc->gbt_cache = parse_gbt(gbt);
        log_msg("[rpc] getblocktemplate available → using submitblock\n");
        free(gbt);
        return 1;
    }
    
    /* Fallback to getwork (older wallets: gapcoin, legacy gapminer) */
    char data[161];
    uint64_t ndiff = 0;
    if (rpc_getwork_data(rpc->url, rpc->user, rpc->pass, data, &ndiff)) {
        rpc->submit_method = SUBMIT_GETWORK;
        log_msg("[rpc] getblocktemplate unavailable → using legacy getwork\n");
        return 1;
    }
    
    log_msg("[rpc] ERROR: neither getblocktemplate nor getwork available\n");
    return 0;
}
```

**RPC Poll Thread** (separate, unified):
```c
void *rpc_poll_thread(void *arg) {
    struct rpc_interface *rpc = arg;
    
    while (!global_abort) {
        struct mining_pass new_pass;
        
        if (rpc->submit_method == SUBMIT_GBT) {
            /* Poll getblocktemplate for new block */
            char *gbt = rpc_getblocktemplate(rpc->url, rpc->user, rpc->pass);
            if (gbt) {
                struct gbt_template *new_tpl = parse_gbt(gbt);
                if (gbt_extract_pass(new_tpl, &new_pass)) {
                    pthread_mutex_lock(&rpc->pass_lock);
                    rpc->current_pass = new_pass;
                    if (rpc->gbt_cache) free_gbt(rpc->gbt_cache);
                    rpc->gbt_cache = new_tpl;
                    pthread_mutex_unlock(&rpc->pass_lock);
                    
                    signal_workers_new_block();
                    log_msg("[rpc] new block (gbt mode): nonce=%u\n", new_pass.nonce);
                } else {
                    free_gbt(new_tpl);
                }
                free(gbt);
            }
        } else {
            /* Poll getwork for new block */
            char data[161];
            uint64_t ndiff = 0;
            if (rpc_getwork_data(rpc->url, rpc->user, rpc->pass, data, &ndiff)) {
                if (getwork_extract_pass(data, ndiff, &new_pass)) {
                    pthread_mutex_lock(&rpc->pass_lock);
                    rpc->current_pass = new_pass;
                    pthread_mutex_unlock(&rpc->pass_lock);
                    
                    signal_workers_new_block();
                    log_msg("[rpc] new block (getwork mode): nonce=%u\n", new_pass.nonce);
                }
            }
        }
        
        sleep(0.1);  /* Poll interval: 100ms */
    }
}
```

**Submit integration** (in worker/gap_search):
```c
int submit_discovered_gap(
    struct rpc_interface *rpc,
    uint32_t nonce,
    uint64_t nAdd,
    uint32_t shift
) {
    struct mining_pass pass = rpc_get_current_pass(rpc);
    
    char blockhex[SUBMIT_BLOCK_HEX_CAP];
    int ok = 0;
    
    if (rpc->submit_method == SUBMIT_GETWORK) {
        ok = assemble_getwork_payload(
            pass.h256, nonce, shift, nAdd, blockhex);
        if (ok) {
            ok = rpc_submit(rpc->url, rpc->user, rpc->pass,
                           "getwork", blockhex);
        }
    } else if (rpc->submit_method == SUBMIT_GBT) {
        ok = assemble_gbt_payload(
            rpc->gbt_cache, pass.h256, nonce, shift, nAdd,
            rpc->coinbase_script_hex, blockhex);
        if (ok) {
            ok = rpc_submit(rpc->url, rpc->user, rpc->pass,
                           "submitblock", blockhex);
        }
    }
    
    return ok;
}
```

**CLI options**:
```bash
# Auto-detect submit method at startup
./gap_miner -s 512 -o localhost -u user -p pass

# Force getwork (for compatibility)
./gap_miner -s 512 -o localhost -u user -p pass --force-getwork

# GBT with custom coinbase
./gap_miner -s 512 -o localhost -u user -p pass \
            --coinbase-script-hex "0123456789abcdef"
```

**Why separate**: No race conditions, clear semantics, easy to debug, unified polling for both methods

### 3.6 Submit Engine

**Old**: `assemble_mining_block()` in main.c  
**New** (`new_src/submit.c`):

```c
struct submit_result {
    int success;
    const char *submit_id;
    const char *error_msg;
};

struct submit_result submit_gap(
    struct rpc_interface *rpc,
    const uint8_t h256[32],
    uint32_t nonce,
    uint64_t adder,
    uint32_t shift
);
```

**Actions**:
- Call existing `assemble_mining_block()` logic
- Format JSON payload
- Send via RPC/stratum
- Log result

---

## 4. File Structure (New)

```
src/
    main.c                      (thin entry point only)
    stratum.c / stratum.h       (keep as-is)
    rpc_json.c / rpc_json.h     (keep as-is)

new_src/
    crt_engine.c/.h             (copy from crt_runtime_*.c, no changes)
    rpc_interface.c/.h          (wrapper around existing RPC)
    submit_engine.c/.h          (extract from main.c submit path)
    miner_loop.c/.h             (new worker main loop)
    sieve_core.c/.h             (new sieve, clean API)
    gap_search.c/.h             (new gap discovery + merit)
    primality.c/.h              (new primality testing)
    gap_distribution.c/.h       (new merit calibration, if needed)
```

---

## 5. Phased Build

### Phase 1: Preserve Existing
- Extract CRT logic → `new_src/crt_engine.*`
- Wrap RPC → `new_src/rpc_interface.*`
- Extract submit → `new_src/submit_engine.*`
- **Checkpoint**: All three components link, existing RPC path still works

### Phase 2: New Sieve + Gap Search
- Write `new_src/sieve_core.*` (clean, stateless)
- Write `new_src/primality.*` (thread-local context)
- Write `new_src/gap_search.*` (ties them together)
- **Checkpoint**: Single-threaded gap search works

### Phase 3: New Mining Loop
- Write `new_src/miner_loop.c` with worker threads
- Integrate RPC polling thread
- Switch main entry point
- **Checkpoint**: Full multi-threaded mining works

### Phase 4: Feature Migration (Optional)
- GPU support (new design)
- Stats collection (new design)
- Advanced sieve modes

---

## 6. Testing Strategy

### Unit Tests
- CRT engine: gap scoring, file loading
- Sieve: candidate generation for known ranges
- Primality: known test vectors
- RPC: mock responses, pass sequencing
- Submit: payload encoding

### Integration Tests
- Full worker loop with RPC mock
- Offline header mode (no RPC)
- CRT mode with varying shifts
- Nonce advancement

### Acceptance Tests
- Compare output (gaps found, merit) vs. existing miner
- Performance comparison (single-threaded baseline)

---

## 7. Expected Outcomes

| Aspect | Old | New |
|--------|-----|-----|
| **main.c** | 13,137 lines | ~500 lines (entry + config) |
| **Worker logic** | Mixed with RPC + stats | Clear, testable `miner_loop.c` |
| **Sieve** | Coupled to window loop | Standalone, reusable |
| **Debugging** | Hard to trace data flow | Each component has clear API |
| **New features** | Risky, require main.c edits | Add new modules, not edit core |
| **Performance** | Unmeasured | Profiled per component |

---

## 8. Timeline

- **Week 1**: Phase 1 (preserve CRT, RPC, submit) — ensure compilation
- **Week 2**: Phase 2 (sieve + gap search) — validate correctness
- **Week 3**: Phase 3 (mining loop) — full miner functional
- **Week 4**: Phase 4 (polish, tests, docs)

---

## 9. Git Strategy

1. Create branch: `git checkout -b rewrite/clean-architecture`
2. Add `new_src/` directory (new files only)
3. Keep `src/` as-is (can delete gradually after validation)
4. Build: compile both `src/` and `new_src/`, link new miner as `gap_miner_v2`
5. Validate: old miner vs. new miner on same hardware
6. Merge when passes acceptance tests

---

## 10. Gapcoin PoW Verification: No False Gaps

### 10.1 What is a Valid Gap in Gapcoin PoW?

A **gap** is a run of consecutive composite numbers between two probable primes. The PoW is:

```
Given: block hash h256, nonce N, shift S, adder A
Compute: n = hash(h256 || N || A) × primorial(S) + offset
Prove: p - n is a gap (first probable prime before n, second probable prime after n)
```

**Validation criteria**:
1. **Primality of boundary**: n - gaplen and n + gaplen must both be **probable primes** (not composite)
2. **Compositeness between**: All numbers n - gaplen + 1 ... n + gaplen - 1 must be composite
3. **Maximality**: Gap must be maximal (no larger gap at that n) — verified by checking n ± gaplen ± 1
4. **Merit threshold**: Gap merit = gaplen / ln(n) must exceed scan target to submit

### 10.2 False Gap Detection: What Can Go Wrong

**False positive risks** (gaps that appear valid but aren't):

| Risk | Cause | Detection |
|------|-------|-----------|
| Fermat fool | Carmichael number fools Fermat test | Use BPSW (Baillie-PSW) — deterministic |
| Partial gap | Missed composite near boundary (trial div limit too low) | Extend trial division check beyond limit |
| Merit drift | Miscalculated ln(n) or gap length | Re-verify merit = gaplen / ln(n) on submit |
| Hash collision | Two different nonces map to same n | Use full 256-bit hash, no truncation |
| Primorial error | Wrong primorial or offset encoding | Validate primorial matches CRT file shift |

### 10.3 Multi-Layer Verification Pipeline

**Layer 1: Discovery (sieve + candidate check)**

```c
struct gap_result {
    uint64_t base;              /* The start of gap region */
    uint32_t gaplen;            /* Gap length (candidate distance) */
    double merit;               /* gaplen / ln(base) */
    uint8_t validated;          /* Flags below */
    
    #define GAP_TRIAL_DIV_OK     0x01  /* Passed trial div at limit */
    #define GAP_FERMAT_OK        0x02  /* Passed Fermat test */
    #define GAP_BPSW_OK          0x04  /* Passed BPSW (high confidence) */
    #define GAP_BOUNDS_OK        0x08  /* No gap boundary errors */
    #define GAP_MERIT_OK         0x10  /* Merit >= scan target */
};

int gap_discover_and_validate(
    struct gap_search *gs,
    uint64_t *candidates,       /* Sieve output: sorted offsets */
    uint32_t n_candidates,
    double scan_target,
    struct gap_result *results  /* Out: array of gaps found */
) {
    int gaps_found = 0;
    
    for (uint32_t i = 1; i < n_candidates - 1; i++) {
        uint64_t cand_a = candidates[i-1];
        uint64_t cand_b = candidates[i];
        uint64_t cand_c = candidates[i+1];
        
        uint32_t gap1 = cand_b - cand_a;  /* Gap before candidate i */
        uint32_t gap2 = cand_c - cand_b;  /* Gap after candidate i */
        
        /* Larger gap is potential PoW candidate */
        uint32_t gaplen = (gap1 > gap2) ? gap1 : gap2;
        uint64_t base   = (gap1 > gap2) ? cand_a : cand_b;
        
        if (gaplen < 6) continue;  /* Too small */
        
        /* === Layer 1: Trial Division === */
        if (!trial_div_check_extended(base, 10000000)) {  /* 10M primes */
            results[gaps_found].validated &= ~GAP_TRIAL_DIV_OK;
            continue;
        }
        results[gaps_found].validated |= GAP_TRIAL_DIV_OK;
        
        /* === Layer 2: Fermat Test === */
        if (!fermat_multi_base(base, 16)) {  /* 16 bases */
            results[gaps_found].validated &= ~GAP_FERMAT_OK;
            continue;
        }
        results[gaps_found].validated |= GAP_FERMAT_OK;
        
        /* === Layer 3: BPSW (for high confidence) === */
        if (!baillie_psw_test(base)) {
            results[gaps_found].validated &= ~GAP_BPSW_OK;
            continue;  /* Optional: can proceed without BPSW */
        }
        results[gaps_found].validated |= GAP_BPSW_OK;
        
        /* === Layer 4: Boundary Check === */
        if (!verify_gap_bounds(base, gaplen)) {
            results[gaps_found].validated &= ~GAP_BOUNDS_OK;
            continue;
        }
        results[gaps_found].validated |= GAP_BOUNDS_OK;
        
        /* === Layer 5: Merit Threshold === */
        double merit = (double)gaplen / log((double)base);
        if (merit < scan_target) {
            results[gaps_found].validated &= ~GAP_MERIT_OK;
            continue;
        }
        results[gaps_found].validated |= GAP_MERIT_OK;
        
        /* Valid gap found */
        results[gaps_found].base = base;
        results[gaps_found].gaplen = gaplen;
        results[gaps_found].merit = merit;
        gaps_found++;
    }
    
    return gaps_found;
}
```

### 10.4 Boundary Verification: No Compositeness Gaps

**Key check**: All numbers between (n - gaplen) and (n + gaplen) must be composite.

```c
int verify_gap_bounds(uint64_t n, uint32_t gaplen) {
    /* Check that n - gaplen is probable prime */
    if (!fermat_test_ui64(n - gaplen, 16)) {
        return 0;  /* False prime boundary */
    }
    
    /* Check that n + gaplen is probable prime */
    if (!fermat_test_ui64(n + gaplen, 16)) {
        return 0;  /* False prime boundary */
    }
    
    /* Check that no number in between is prime (via trial div) */
    for (uint64_t i = n - gaplen + 1; i < n + gaplen; i++) {
        if (i % 2 == 0 && i > 2) continue;  /* Even, skipped */
        if (i % 3 == 0 && i > 3) continue;  /* Divisible by 3, skipped */
        
        /* Quick check: if passes trial div at small limit, might be prime */
        if (is_probable_prime(i)) {
            return 0;  /* Gap is not maximal, contains prime */
        }
    }
    
    return 1;  /* Gap is valid and maximal */
}
```

### 10.5 BPSW: Deterministic Primality for Gapcoin (Shift ≥ 512)

**Why BPSW over Fermat for PoW**:
- Fermat: base^(p-1) ≡ 1 (mod p) — can be fooled by Carmichael numbers
- BPSW: Fermat + Lucas + trial div — **no known composite BPSW pseudoprimes** (proven up to 2^64)

**Implementation** (`new_src/bpsw.c`):

```c
int baillie_psw_test_ui64(uint64_t n) {
    /* Stage 1: Trial division up to small primes (2, 3, 5, ..., 30) */
    if (n == 2) return 1;
    if (n < 2 || n % 2 == 0) return 0;
    for (uint32_t p : small_primes) {
        if (n == p) return 1;
        if (n % p == 0) return 0;
    }
    
    /* Stage 2: Miller-Rabin with base 2 (Fermat pretest) */
    if (!miller_rabin_ui64(n, 2)) return 0;
    
    /* Stage 3: Lucas test (catches most Carmichael escapees) */
    uint32_t D = find_lucas_D(n);  /* Find first D where (D/n) = -1 */
    if (!lucas_lehmer_ui64(n, D)) return 0;
    
    return 1;  /* Deterministically prime (no known counterexample) */
}
```

**Gapcoin compliance**:
- Node validates gap using Fermat (fast, good enough for blockchain)
- Miner verifies using BPSW (bulletproof, prevents false gaps)

### 10.6 On-Submit Verification (Final Check Before RPC)

Before calling `rpc_submit()`, final re-validation:

```c
int gap_verify_before_submit(
    const uint8_t h256[32],
    uint32_t nonce,
    uint64_t adder,
    uint32_t shift,
    struct gap_result *gap
) {
    /* Reconstruct n from hash + nonce + adder + shift */
    uint64_t n = hash_to_n(h256, nonce, adder, shift);
    
    /* Final BPSW check on both boundaries */
    if (!baillie_psw_test_ui64(n - gap->gaplen)) {
        log_msg("[submit] REJECTED: n - gaplen failed BPSW\n");
        return 0;
    }
    
    if (!baillie_psw_test_ui64(n + gap->gaplen)) {
        log_msg("[submit] REJECTED: n + gaplen failed BPSW\n");
        return 0;
    }
    
    /* Verify merit one more time */
    double merit = (double)gap->gaplen / log((double)n);
    if (merit < global_scan_target - 0.001) {  /* Small epsilon for rounding */
        log_msg("[submit] REJECTED: merit %.2f < target %.2f\n", merit, global_scan_target);
        return 0;
    }
    
    return 1;
}
```

### 10.7 Anti-Cheating: No Gap Inflation

**False merit tricks** (what we prevent):

| Trick | How It Works | Prevention |
|-------|--------------|-----------|
| Truncated hash | Use short hash instead of full 256-bit | Always use full h256 |
| Offset wrap | Encode adder outside [0, 2^shift) | Validate adder bounds |
| Primorial skip | Use wrong primorial for shift | CRT file loader checks shift |
| Merging gaps | Count disjoint gaps as one | Verify boundary primes independently |
| Over-counting | Count composite as prime (Fermat fool) | BPSW deterministically rules out |

---

## 11. GPU Strategy: CUDA + OpenCL for Fermat Testing (Shift up to 1024)

### 11.1 Problem: CPU Fermat Becomes Bottleneck

For shift ≥ 512, Fermat round complexity is O(k × log³n):
- shift 512: ~512-bit numbers, ~20ms per Fermat round (GMP, serial)
- shift 768: ~768-bit numbers, ~100ms per Fermat round
- shift 1024: ~1024-bit numbers, ~200ms+ per Fermat round
- **CPU throughput at shift 1024**: ~5 candidates/sec (unacceptable)

GPU solution: Parallel Fermat on 1000s of candidates simultaneously.

### 11.2 CUDA Path: CGBN-Based Fermat Batching

**Component**: `new_src/gpu/fermat_cuda.cu` + `fermat_cuda.h`

**Design**:
```c
struct cuda_fermat_batch {
    uint32_t     capacity;           /* 1024 – 8192 candidates per batch */
    uint32_t     candidate_bits;     /* typically 512–1024 bits */
    
    /* GPU memory */
    uint32_t    *gpu_candidates;     /* CGBN format: 32-bit chunks */
    uint8_t     *gpu_results;        /* 1 byte per candidate: 0 or 1 */
    
    /* Host staging */
    uint64_t    *cpu_candidates;     /* Staging buffer before transfer */
    uint8_t     *cpu_results;
};

struct cuda_fermat_batch *cuda_batch_alloc(uint32_t capacity, uint32_t bits);
int cuda_fermat_batch_test(
    struct cuda_fermat_batch *batch,
    uint64_t *candidates,           /* Input: candidate offsets or full 64-bit vals */
    uint32_t n_candidates,
    uint64_t base,                  /* Base for candidate = base + candidate[i] */
    uint32_t rounds,                /* Fermat rounds (typically 5–10 for 64-bit error prob) */
    uint8_t  *results_out           /* Output: 1 = probably prime, 0 = composite */
);
void cuda_batch_free(struct cuda_fermat_batch *batch);
```

**CGBN Usage** (CUDA C++ Big Number library):
- CGBN is NVIDIA's official library for large-integer arithmetic on GPU
- Directly supports modular exponentiation (`cgbn_modexp_...`)
- Optimal for 256-bit to 1024-bit numbers
- Built-in Barrett reduction for fast modular multiplication

**Kernel skeleton** (`fermat_cuda.cu`):
```cuda
__global__ void fermat_test_kernel(
    uint32_t *candidates,    /* CGBN format */
    uint8_t *results,
    uint32_t n,
    uint32_t rounds,
    cgbn_error_handler_t *error_handler
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    cgbn_context_t<CGBN_BITS> context(cgbn_error_handler_t);
    cgbn_local_t<CGBN_BITS> a, p_minus_1, result;
    
    /* Load candidate from global memory (coalesced) */
    cgbn_load(context, a, (uint32_t*)&candidates[idx * stride]);
    
    /* Compute p - 1 */
    cgbn_sub_ui32(context, p_minus_1, a, 1);
    
    /* Fermat: result = base^(p-1) mod p, with different bases per round */
    for (int round = 0; round < rounds; round++) {
        uint32_t base = 2 + round;  /* Try base 2, 3, 4, ... */
        cgbn_modexp_ui32(context, result, base, p_minus_1, a);
        if (cgbn_compare_ui32(context, result, 1) != 0) {
            results[idx] = 0;  /* Composite */
            return;
        }
    }
    results[idx] = 1;  /* Probably prime */
}
```

**Batch parameters**:
- **Capacity**: 4096 candidates per batch (shift 1024: ~4 GB GPU memory)
- **Transfer overhead**: PCI-e 4.0 = ~30 GB/s → 4096 × 128 bytes = ~512 us
- **Kernel execution**: ~10–50 ms for 4096 × 10 rounds (dominates, good)
- **Throughput**: ~400k candidates/sec at shift 1024 (vs. 5 on CPU) — **80x speedup**

**Integration with gap search**:
```c
struct gap_search_gpu {
    crt_engine *crt;
    sieve_state *sieve;
    struct cuda_fermat_batch *fermat_batch;
    primality_ctx *cpu_primality;      /* For small composites, trial div */
};

struct gap_result gap_search_one_nonce_gpu(
    struct gap_search_gpu *g,
    const uint8_t h256[32],
    uint32_t nonce,
    uint64_t adder
) {
    /* Generate sieve candidates (CPU) */
    uint64_t candidates[MAX_CANDIDATES];
    int n_cand = sieve_batch(g->sieve, &candidates[0]);
    
    /* Ship to GPU for Fermat (batched) */
    cuda_batch_add_candidates(g->fermat_batch, candidates, n_cand, ...);
    
    if (g->fermat_batch->count >= 1024) {
        cuda_fermat_batch_test(g->fermat_batch, ...);
        process_gpu_results(g, ...);  /* Merit calc on CPU for survivors */
    }
    
    return best_result;
}
```

### 11.3 OpenCL Path: Portable Fermat (AMD/Intel/Apple)

**Component**: `new_src/gpu/fermat_opencl.c` + `fermat_opencl.h`

**Why OpenCL**: NVIDIA cuFFT/CGBN not available; need vendor-neutral option.

**Design** (similar to CUDA but using OpenCL 2.0):
```c
struct opencl_fermat {
    cl_context ctx;
    cl_device_id device;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    
    cl_mem gpu_candidates;
    cl_mem gpu_results;
    
    size_t max_batch;
};

int opencl_fermat_test(
    struct opencl_fermat *ocl,
    uint64_t *candidates,
    uint32_t n,
    uint32_t rounds,
    uint8_t *results_out
);
```

**OpenCL Kernel** (`fermat_opencl.cl`):
```opencl
/* Manual 1024-bit big-int using local memory + barriers */
#define BIGINT_WORDS (1024 / 32)

__kernel void fermat_test_batch(
    __global uint32_t *candidates,
    __global uint8_t *results,
    uint32_t n,
    uint32_t rounds
) {
    uint32_t idx = get_global_id(0);
    if (idx >= n) return;
    
    __local uint32_t a[BIGINT_WORDS];
    __local uint32_t p_minus_1[BIGINT_WORDS];
    __local uint32_t result[BIGINT_WORDS];
    
    /* Load candidate (32 threads cooperatively) */
    for (int i = get_local_id(0); i < BIGINT_WORDS; i += get_local_size(0)) {
        a[i] = candidates[idx * BIGINT_WORDS + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* Compute p - 1 */
    bigint_sub_1(p_minus_1, a);
    
    /* Fermat rounds */
    for (int round = 0; round < rounds; round++) {
        bigint_modexp(result, 2 + round, p_minus_1, a);
        if (bigint_compare_1(result) != 0) {
            results[idx] = 0;
            return;
        }
    }
    results[idx] = 1;
}

void bigint_modexp(__local uint32_t *result, uint32_t base,
                   __local uint32_t *exp, __local uint32_t *mod) {
    /* Binary exponentiation with Barrett reduction */
    /* ... */
}
```

**OpenCL vs. CUDA**:
| Aspect | CUDA (CGBN) | OpenCL (Manual) |
|--------|-------------|-----------------|
| Speed | ~10–50ms for 4k×10 rounds | ~50–150ms (no optimized lib) |
| Portability | NVIDIA only | AMD, Intel, Apple |
| Dev cost | Low (use CGBN) | High (manual bigint) |

**Recommendation**: CUDA for RTX/A100 (prod), OpenCL as fallback.

---

## 12. Primality Strategy: Shift 1024 Support

### 12.1 CPU Fermat + Trial Division (Baseline)

For shift < 512, CPU is sufficient:
```c
struct primality_ctx {
    gmp_randstate_t rng;
    mpz_t scratch[4];
    
    uint32_t trial_div_limit;        /* Typically 1M–10M */
};

int primality_test_cpu(
    struct primality_ctx *p,
    uint64_t candidate,
    uint32_t shift,
    uint8_t *is_prime
) {
    /* Trial division up to limit */
    if (!trial_div_check(candidate, p->trial_div_limit)) {
        *is_prime = 0;
        return 1;
    }
    
    /* Fermat: rounds = 16 (64-bit error prob ≤ 2^-32) */
    int rounds = (shift < 512) ? 10 : 16;
    if (!fermat_multi_base_ui64(p, candidate, rounds)) {
        *is_prime = 0;
        return 1;
    }
    
    *is_prime = 1;
    return 1;
}
```

### 12.1.5 Fast-Euler Optimization: Euler's Criterion

**Faster alternative to Fermat test**:

Euler's criterion: `a^((p-1)/2) ≡ Legendre(a, p) (mod p)`

Why faster than Fermat:
- Fermat: compute `a^(p-1) mod p` (full exponent)
- Euler: compute `a^((p-1)/2) mod p` (half exponent) — **2x fewer multiplications**
- Same error probability for pseudoprime detection

**Implementation** (`new_src/primality.c`):

```c
int euler_criterion_test_ui64(uint64_t n, uint32_t base) {
    /* Compute Legendre symbol (base/n) using Euler criterion:
       base^((n-1)/2) mod n
       Should be 1 (if base is QR) or n-1 ≡ -1 (if NQR) mod n
       If result is anything else, n is composite.
    */
    
    mpz_t base_z, n_z, exp_z, result;
    mpz_init_set_ui(base_z, base);
    mpz_init_set_ui(n_z, n);
    mpz_init_set_ui(exp_z, (n - 1) / 2);
    mpz_init(result);
    
    mpz_powm(result, base_z, exp_z, n_z);
    
    int is_prime = (mpz_cmp_ui(result, 1) == 0 || 
                    mpz_cmp_ui(result, n - 1) == 0);
    
    mpz_clear(base_z);
    mpz_clear(n_z);
    mpz_clear(exp_z);
    mpz_clear(result);
    
    return is_prime;
}

/* Fast path: trial div + Euler (no Fermat) */
int primality_test_fast_euler(
    struct primality_ctx *p,
    uint64_t candidate,
    uint32_t shift
) {
    /* Trial division (eliminates 99%) */
    if (!trial_div_check(candidate, p->trial_div_limit)) {
        return 0;
    }
    
    /* Euler's criterion with 5–8 different bases (faster than Fermat) */
    int rounds = (shift < 512) ? 5 : 8;
    for (int i = 0; i < rounds; i++) {
        uint32_t base = small_primes[i];  /* 2, 3, 5, 7, 11, ... */
        if (!euler_criterion_test_ui64(candidate, base)) {
            return 0;  /* Composite */
        }
    }
    
    return 1;  /* Probably prime */
}
```

**Performance**: 
- Fermat (10 rounds): ~100 µs per candidate
- Euler (8 rounds): ~50 µs per candidate
- **2x speedup** for discovery phase

**When to use**:
- **Discovery (fast path)**: Use Euler for quick filtering
- **Submit verification**: Use BPSW (deterministic)
- **GPU Fermat**: Can't optimize much (already batched)

**Hybrid strategy**:
```
Discovery:
  1. Trial div (10M) → eliminates 99%
  2. Euler (5–8 bases) → fast, 2x faster than Fermat
  3. If merit high enough, queue for final BPSW

Submit:
  BPSW (deterministic, bulletproof)
```

---

### 12.2 Hybrid: CPU Trial Div + GPU Fermat (Shift ≥ 512)

For shift ≥ 512, split work:
- **CPU**: Trial division (fast, eliminates 99% of composites)
- **GPU**: Fermat for survivors (expensive, batched)

```c
struct primality_hybrid {
    struct primality_ctx cpu;
    struct cuda_fermat_batch *gpu_batch;    /* or opencl_fermat */
};

int primality_test_hybrid(
    struct primality_hybrid *ph,
    uint64_t *candidates,
    uint32_t n,
    uint32_t shift,
    uint8_t *results_out
) {
    /* Phase 1: CPU trial division (eliminates ~99%) */
    uint32_t trial_div_survivors = 0;
    uint64_t survivors[n];
    for (uint32_t i = 0; i < n; i++) {
        if (trial_div_check(candidates[i], 1000000)) {
            survivors[trial_div_survivors++] = candidates[i];
        }
    }
    
    /* Phase 2: GPU Fermat (on ~1% of input) */
    uint8_t gpu_results[trial_div_survivors];
    cuda_batch_add_candidates(ph->gpu_batch, survivors, trial_div_survivors, ...);
    cuda_fermat_batch_test(ph->gpu_batch, ..., gpu_results);
    
    /* Phase 3: Merge results */
    uint32_t idx = 0;
    for (uint32_t i = 0; i < n; i++) {
        results_out[i] = 0;
    }
    for (uint32_t i = 0; i < trial_div_survivors; i++) {
        for (uint32_t j = 0; j < n; j++) {
            if (candidates[j] == survivors[i]) {
                results_out[j] = gpu_results[i];
                break;
            }
        }
    }
    
    return 1;
}
```

### 12.3 Lucas-Lehmer and BPSW for Shift 1024+

For ultra-high security (shift 1024+, rare merits):
```c
int primality_test_advanced(
    struct primality_hybrid *ph,
    uint64_t candidate,
    uint32_t shift
) {
    /* Baillie–PSW test = Miller-Rabin + Lucas + trial div */
    if (!baillie_psw_test_ui64(candidate)) {
        return 0;  /* Composite */
    }
    
    /* For gap PoW submission, additionally check Lucas-Lehmer chain */
    if (shift >= 768) {
        if (!lucas_lehmer_chain_ui64(candidate, 10)) {  /* 10 random bases */
            return 0;
        }
    }
    
    return 1;
}
```

**Why BPSW over Fermat**:
- Deterministic for 64-bit (no false primes known)
- Fermat can be fooled by Carmichael numbers (rare but possible)
- BPSW = Fermat + Lucas + trial div = bulletproof

---

## 13. Performance Targets: GPU vs. CPU

### shift 512
| Method | Throughput | Time/candidate |
|--------|-----------|-----------------|
| CPU (trial + Fermat) | 100 cand/sec | 10ms |
| GPU (CUDA) | 20k cand/sec | 50 µs |
| **Speedup** | **200x** | — |

### shift 768
| Method | Throughput | Time/candidate |
|--------|-----------|-----------------|
| CPU (trial + Fermat) | 20 cand/sec | 50ms |
| GPU (CUDA) | 5k cand/sec | 200 µs |
| **Speedup** | **250x** | — |

### shift 1024
| Method | Throughput | Time/candidate |
|--------|-----------|-----------------|
| CPU (trial + Fermat) | 5 cand/sec | 200ms |
| GPU (CUDA) | 1k cand/sec | 1ms |
| **Speedup** | **200x** | — |

**Conclusion**: GPU mandatory for shift ≥ 768 (otherwise mining is impractical).

---

## 14. Implementation Roadmap

### Phase 3a (optional, after Phase 3): Add GPU Support

**Timeline: Week 5–8** (after phase 1–3 complete)

| Week | Task |
|------|------|
| 5 | CUDA/CGBN setup, `fermat_cuda.cu` skeleton, unit tests |
| 6 | OpenCL fallback, portable bigint lib |
| 7 | Integrate GPU batching into `gap_search_gpu.c` |
| 8 | Hybrid trial-div + GPU, performance tuning, BAPSsw option |

**Build flags**:
```makefile
# CPU only (fast to compile, limited to shift < 512)
make clean && make

# CUDA support (requires CUDA toolkit)
make clean && make WITH_CUDA=1

# OpenCL support (requires OpenCL SDK)
make clean && make WITH_OPENCL=1

# Both GPU backends
make clean && make WITH_CUDA=1 WITH_OPENCL=1
```

---

## 15. Key Decisions for GPU Path

1. **CGBN vs. Manual**: Use CGBN for CUDA (battle-tested, 2–3x faster than manual). Manual for OpenCL (no choice).

2. **Batch size**: 4096 candidates per batch (balance: GPU memory, transfer overhead, kernel occupancy).

3. **Trial div limit**: 1M primes (eliminates ~99.9% of composites, cost < 1% of GPU time).

4. **Fermat rounds**: 
   - shift < 512: 10 rounds (64-bit error prob ~2^-20, acceptable for gaps)
   - shift ≥ 512: 16 rounds (64-bit error prob ~2^-32, high confidence)

5. **BPSW**: Add only for shift ≥ 768 (rare, extra ~5% cost, bulletproof results).

6. **Fallback**: If GPU unavailable or fails, seamlessly fall back to CPU (slower but safe).

---

## 16. Non-Goals (Phase 1–4)

- GPU optimization per-device (cuDNN, ROCm tuning) — later
- Advanced stats collection — later
- Complex sieve tuning modes — later
- Config file reloading — later
- Multisig or advanced coinbase scripting — keep simple

---

## 17. Questions for Validation

1. Is CRT engine the only "must preserve" component, or are there others?
2. Should primality testing stay with GMP, or prefer different library?
3. CRT file format — any changes to loader?
4. Submit payload format — any changes expected?
5. RPC polling interval — any specific requirements?
6. **GPU**: RTX 4090 / A100 primary target, or broad compatibility (OpenCL first)?
7. **Shift range**: Prioritize shift 512–768, or go straight to 1024?
8. **PoW verification**: Is BPSW mandatory, or is Fermat sufficient for all shifts? Should BPSW apply to all gaps or only shift ≥ 768?
9. **False positives**: Are there known Carmichael numbers in the shift ≤ 512 range that Fermat could fool?
10. **Merit precision**: Should merit be verified at double precision or higher (mpfr) before submit?
