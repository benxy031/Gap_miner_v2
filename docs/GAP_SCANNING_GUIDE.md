# Gap Scanning Implementation Guide

**Prepared**: 2026-08-14  
**Source**: Analysis of lentmaistrouve's unified GapMiner + existing cpugapminer  
**Purpose**: Implement gap discovery pipeline for new clean architecture

---

## Summary: How Gap Scanning Will Work

### The Complete Pipeline

```
1. RPC Thread → Poll new block, update atomic nonce
2. Worker Threads (N per GPU) → Race on shared nonce
   a. Sieve: Generate candidate offsets (~1000-5000 per nonce)
   b. Primality: GPU batch Fermat or CPU trial+Euler
   c. Gap Detection: Find consecutive prime pairs
   d. Merit Filter: Only test high-merit gaps with BPSW
   e. Submit: Direct RPC call (no queue)
3. Repeat for next nonce (atomic advancement)
```

### Key Design Decisions from Unified GapMiner

| Decision | Reasoning | Impact |
|----------|-----------|--------|
| **Atomic nonce** | Lock-free advancement | No synchronization overhead |
| **Per-GPU worker** | True parallelism | Each GPU works independently |
| **Direct submission** | No result queue | Workers unblocked, submit async |
| **Candidate lazy-load** | base + row factorization | 8x memory reduction |
| **GPU batch Fermat** | 4096 candidates/batch | 50ms kernel ≈ 400k candidates/sec |
| **BPSW only on submit** | Expensive verification | ~1-5 gaps/sec submitted vs. 100-400 found |
| **Separate RPC thread** | Non-blocking polling | Workers never stall on network |

---

## Gap Scanning: Step-by-Step

### 1️⃣ **Candidate Generation (Sieve)**

**Input**: Block hash + nonce  
**Output**: Offsets of probable primes in interval

```c
/* Example: CRT sieve output */
uint64_t candidates[] = {2, 5, 7, 13, 19, 23, 29, 31, ...};  /* ~1000-5000 */
uint32_t n = sizeof(candidates) / sizeof(candidates[0]);
```

**Algorithm**:
1. Compute base = (h256 ⊕ nonce) mod primorial
2. Create bitmap for interval [base, base + primorial]
3. Sieve of Eratosthenes (trial division up to √interval)
4. Extract offsets where bitmap[i] = 0 (is prime)

**Performance**: 1ms per nonce on CPU

---

### 2️⃣ **Primality Testing (GPU for shift≥512, CPU for shift<512)**

**Input**: Candidate offsets  
**Output**: is_prime[i] = 1 (probable prime) or 0 (composite)

#### GPU Path (shift ≥ 512):
```c
/* CGBN batch Fermat kernel */
for i in candidates:
    p = (base + offset[i]) × 2^shift
    Test p with Fermat (16 rounds)
    is_prime[i] = pass all 16 rounds ? 1 : 0
```

**Performance**: 50ms for 4096 candidates on RTX 4090

#### CPU Path (shift < 512):
```c
/* Hybrid trial div + Euler + Fermat */
for i in candidates:
    p = (base + offset[i]) × 2^shift
    
    if (trial_division(p, limit=2^24)):      /* ~99% filtered */
        is_prime[i] = 0; continue;
    
    if (!euler_criterion(p)):                  /* 2x faster */
        is_prime[i] = 0; continue;
    
    if (fermat_test(p, bases=[2,3,5,7])):
        is_prime[i] = 1;
    else:
        is_prime[i] = 0;
```

**Performance**: 0.1-1.0 sec for 4000 candidates

---

### 3️⃣ **Gap Detection (Consecutive Primes)**

**Input**: is_prime[] array  
**Output**: List of gap structures {base, length, merit}

```c
/* Scan for consecutive probable primes */
gaps[] = {};
for i in [0..n-2]:
    if (is_prime[i] AND is_prime[i+1]):
        gap_base = candidates[i]
        gap_len = candidates[i+1] - candidates[i]
        merit = gap_len / ln(base + gap_base)
        
        gaps.push({gap_base, gap_len, merit});
```

**Example**:
```
is_prime:    [0, 1, 1, 0, 0, 1, 1, 1, 0, ...]
offsets:     [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]
             ┬  ┬ ┬        ┬ ┬ ┬
             gap#1       gap#2
             
gap#1: base=candidates[1], len=candidates[2]-candidates[1]=1
gap#2: base=candidates[5], len=candidates[6]-candidates[5]=1
```

**Performance**: ~10-50 gaps per nonce (merit ≥ 14)

---

### 4️⃣ **Merit Filtering + BPSW Verification**

**Input**: Gap list  
**Output**: Gaps ready for RPC submission

```c
for gap in gaps:
    if (gap.merit < scan_target):  /* e.g., 16.0 */
        skip;  /* Vast majority filtered here */
    else:
        /* Deterministic primality check */
        p1 = (base + gap.base) × 2^shift
        p2 = (base + gap.base + gap.len) × 2^shift
        
        if (baillie_psw_verify(p1) AND baillie_psw_verify(p2)):
            submit_gap(nonce, gap.base, gap.len);
        else:
            log_false_positive(gap);
```

**Performance**: 
- Finding: ~20-100 gaps/sec (merit ≥ 14)
- Submitting: ~1-5 gaps/sec (merit ≥ 16, BPSW-verified)

---

### 5️⃣ **RPC Submission (Direct, Async)**

**Key insight**: Workers don't wait for RPC response.

```c
/* Worker thread: submit and continue */
submit_discovered_gap(nonce, gap.base, gap.len);
/* No blocking, worker already processing next nonce */

/* Separate RPC thread (or async callback) */
rpc_async_submitblock(payload, callback);
callback(accept/reject):
    stats.gaps_submitted++;  /* or rejected */
    log_submission(...);
```

**Protocol**: getwork (legacy) or getblocktemplate (modern)

---

## Multithreaded Coordination

### Atomic Nonce Pool

```c
_Atomic uint32_t g_current_nonce = 0;

/* All N workers race here, lock-free */
uint32_t worker_get_and_advance_nonce(void) {
    uint32_t old, new;
    do {
        old = atomic_load(&g_current_nonce);
        if (old == UINT32_MAX) return 0;  /* Done */
        new = old + 1;
    } while (!atomic_compare_exchange_weak(&g_current_nonce, &old, &new));
    
    return new;
}
```

**Race safety**: Only one worker gets each nonce value.

### Per-GPU Worker Independence

```c
struct gpu_worker {
    int gpu_id;                    /* Assigned GPU */
    struct chinese_set *crt;       /* Shared read-only */
    struct sieve_state *sieve;     /* Thread-local */
    struct cuda_fermat *gpu_batch; /* Thread-local GPU resource */
    struct rpc_interface *rpc;     /* Shared with other workers */
};

void *worker_main(void *arg) {
    struct gpu_worker *w = arg;
    while (running) {
        nonce = get_and_advance_nonce();
        candidates = crt_sieve(w->crt, w->sieve, hash, nonce);
        results = gpu_batch_fermat(w->gpu_batch, candidates);
        gaps = detect_gaps(candidates, results);
        for (gap in gaps):
            if (gap.merit >= target):
                submit(gap);
    }
}
```

**No queues, no producer/consumer sync** — each worker is self-contained.

---

## Memory Footprint

**Per GPU worker**:
- CRT shared state: ~100 KB
- Sieve bitmap (thread): ~64 KB
- Candidate buffer: ~32 KB
- GPU batch (16MB VRAM): ~16 MB
- GMP scratch objects: ~16 KB

**Total per worker**: ~16-17 MB (mostly on GPU)

**Multi-GPU (8 A100s)**: ~136 MB

---

## Performance Targets

| Hardware | Shift | Throughput |
|----------|-------|-----------|
| CPU (8-core) | 256 | ~800 gaps/sec |
| RTX 4090 | 512 | ~20k gaps/sec |
| A100 | 768 | ~5k gaps/sec |
| A100 | 1024 | ~1k gaps/sec |

---

## Documentation Structure

### For Implementation
- **[GAP_SCANNING_FLOW.md](../GAP_SCANNING_FLOW.md)** — Visual pipeline, detailed steps
- **[GAP_SCANNING_DETAILED.md](../GAP_SCANNING_DETAILED.md)** — Code patterns from proven implementation

### For Architecture
- **[NEW_MINER_ARCHITECTURE_PLAN.md](../NEW_MINER_ARCHITECTURE_PLAN.md)** — Full redesign plan
- **[UNIFIED_GAPMINER_REFERENCE.md](../UNIFIED_GAPMINER_REFERENCE.md)** — Source file mapping

### For Debugging
- [PoW verification strategy](../NEW_MINER_ARCHITECTURE_PLAN.md#10-gapcoin-pow-verification-no-false-gaps)
- [Primality testing strategies](../NEW_MINER_ARCHITECTURE_PLAN.md#12-primality-strategies-deep-dive)
- [GPU batching performance](../NEW_MINER_ARCHITECTURE_PLAN.md#11-gpu-strategy-cuda--opencl)

---

## Quick Start: Implement Gap Scanning

### Phase 1: Core Modules
1. `new_src/crt_engine.c` — Extract from src/crt_runtime_cpu.c
2. `new_src/sieve_core.c` — Port from Sieve.cpp (section 2 of detailed guide)
3. `new_src/gap_detection.c` — Implement section 3 algorithm
4. `new_src/gap_candidate.h` — Define structures (section 1)

### Phase 2: Primality + Validation
5. `new_src/primality_fermat.c` — CPU Fermat tests
6. `new_src/primality_bpsw.c` — BPSW verification
7. `new_src/gap_validation.c` — Merit filter + boundary checks

### Phase 3: Worker Threads
8. `new_src/worker_gpu.c` — Main worker loop (section 4)
9. `new_src/miner_farm.c` — Worker spawning + atomic nonce

### Phase 4: GPU Integration
10. `new_src/gpu/fermat_cuda.cu` — CGBN batch kernel
11. `new_src/gpu/gpu_adapter.h` — GPU abstraction

### Phase 5: RPC + Finish
12. `new_src/rpc_interface.c` — Wrap stratum.c
13. `new_src/submit_engine.c` — Payload assembly
14. `new_src/main.c` — Entry point, config, worker spawn

---

## Testing Strategy

1. **Unit tests**:
   - Sieve: verify bitmap → candidates extraction
   - Gap detection: verify consecutive pair identification
   - BPSW: verify on Carmichael numbers (known to fool Fermat)
   - Merit: verify calculation matches on-chain

2. **Integration tests**:
   - Mine 1000 nonces, verify no false gaps
   - Test RPC mode switching
   - Verify atomic nonce advancement (no skips, no double-tests)

3. **Performance validation**:
   - Measure gaps/sec for each shift
   - Profile GPU utilization
   - Check memory footprint per worker

---

## Key Takeaways

✅ **Gap scanning = sieve + Fermat + gap detection**
- Sieve eliminates composites (bitmap-based, fast)
- Fermat tests probable primes (GPU-batched for shift≥512)
- Gap detection finds consecutive prime pairs (linear scan)

✅ **Multithreading = atomic nonce + independent workers**
- All workers race on same nonce pool (lock-free)
- Each worker has thread-local sieve + GPU batch
- No queues, no producer/consumer sync

✅ **BPSW prevents false gaps**
- Fermat can fool Carmichael numbers
- BPSW = Miller-Rabin(2) + Lucas-Lehmer (deterministic)
- Only used on high-merit gaps (expensive)

✅ **GPU batching = 4-8x speedup**
- Batch 4096 candidates per GPU call
- 50ms kernel → 400k candidates/sec at shift 512
- Overhead negligible vs. kernel time

✅ **Direct submission = unblocked workers**
- No result queue, no wait for RPC response
- Workers continue mining while submitting
- Async callback updates stats

---

## References

See `docs/` directory:
- `NEW_MINER_ARCHITECTURE_PLAN.md` — Full plan (1633 lines)
- `GAP_SCANNING_FLOW.md` — Visual flows + detailed steps (394 lines)
- `GAP_SCANNING_DETAILED.md` — Implementation patterns (457 lines)
- `UNIFIED_GAPMINER_REFERENCE.md` — Source mapping (303 lines)
