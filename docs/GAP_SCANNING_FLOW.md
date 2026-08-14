# Gap Scanning Flow Diagram

## Full Pipeline: From RPC Block to Submitted Gap

```
┌─────────────────────────────────────────────────────────────────┐
│                     RPC POLLING THREAD                           │
│  (separate, listens for new blocks, updates atomic pass state)   │
└────────┬──────────────────────────────────────────────┬──────────┘
         │                                              │
         │ new block detected                          │
         │ (getblocktemplate or getwork)               │
         │                                              │
         ▼                                              ▼
    ┌──────────┐                               ┌─────────────────┐
    │ Parse    │                               │ Update atomically│
    │ new pass │                               │ current_pass    │
    │ (nonce)  │                               └─────────────────┘
    └──────────┘
         │
         │ Signal workers: new_block_detected()
         │
         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │             WORKER THREADS (N workers, one per GPU)           │
    └──────────────────────────────────────────────────────────────┘
         │
    ┌────┴────┬────────┬────────┬──────────┐
    │          │        │        │          │
    ▼          ▼        ▼        ▼          ▼
  GPU#0      GPU#1    GPU#2    GPU#3     CPU-worker
  Worker     Worker   Worker   Worker    (if shift<512)
    │          │        │        │          │
    └────────┬─┴────────┴────────┴──────────┘
             │
             │ All workers share same atomic nonce
             │ (race: each worker reads nonce, processes, advances)
             │
             ▼
    ┌─────────────────────────────────────────┐
    │  ATOMIC NONCE READ (lock-free)          │
    │  nonce = atomic_load(&g_current_nonce)  │
    │  (advances via atomic_compare_exchange)  │
    └─────────────────────────────────────────┘
             │
             ▼ (Each worker independently processes same nonce)
    ┌──────────────────────────────────────────────┐
    │     SIEVE: Generate Candidates (CPU)         │
    │  crt_sieve_nonce(hash, nonce) → candidates[]│
    │  Output: ~1000-5000 candidate offsets        │
    │  Time: ~1ms (parallel across workers,        │
    │        but on CPU, may be cache-contended)   │
    └──────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             │                                         │
             ▼ (for CPU-only path)                    ▼ (for GPU path)
    ┌─────────────────────────┐      ┌─────────────────────────────┐
    │  CPU Primality Test     │      │  GPU Batch Fermat Test      │
    │  (Trial div + Fermat)   │      │  (CGBN, 4096 candidates/    │
    │                         │      │   batch, 16 rounds)         │
    │ 1. Trial division       │      │                             │
    │    (eliminates ~99%)    │      │ 1. Accumulate candidates    │
    │ 2. Euler's criterion    │      │    (up to 4096)             │
    │    (2x faster)          │      │ 2. Transfer to GPU          │
    │ 3. CPU Fermat (16 rnd)  │      │ 3. Run kernel (50ms)        │
    │                         │      │ 4. Copy results back        │
    │ Results: is_prime[i]    │      │                             │
    │ (~0.1s for 5k cands)    │      │ Results: is_prime[i]        │
    └─────────────────────────┘      │ (~0.05s for 4096 cands)     │
             │                       └─────────────────────────────┘
             │                                      │
             └──────────────────┬───────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────┐
                    │   GAP DETECTION                 │
                    │  (on any primality results)     │
                    │                                 │
                    │ for i in [0..n_cand-2]:         │
                    │   if (is_prime[i] &&            │
                    │       is_prime[i+1]) {          │
                    │     gap_base = cand[i]          │
                    │     gap_len = cand[i+1] - cand[i]│
                    │     merit = gap_len / ln(base)  │
                    │                                 │
                    │     if (merit >= scan_target) { │
                    │       FOUND GAP ✓               │
                    │     }                           │
                    │   }                             │
                    └─────────────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────┐
                    │   MERIT FILTERING               │
                    │  merit >= scan_target?          │
                    │ (e.g., 16.0)                    │
                    └─────────────────────────────────┘
                         No │    Yes │
                            ▼        ▼
                        (skip)   ┌──────────────────────┐
                                 │  BPSW VERIFICATION  │
                                 │  (Deterministic)     │
                                 │  - Miller-Rabin(2)   │
                                 │  - Lucas test        │
                                 │ (no Carmichael fools)│
                                 └──────────────────────┘
                                   No │    Yes │
                                      ▼        ▼
                                  (reject) ┌──────────────────┐
                                           │  SUBMIT DIRECTLY │
                                           │  (via RPC)       │
                                           │  No queue, no    │
                                           │  ShareProcessor  │
                                           └──────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────┐
                    │  RPC SUBMISSION (Async)         │
                    │  - getwork or getblocktemplate  │
                    │  - submitblock RPC call         │
                    │  - Worker continues mining      │
                    │    (doesn't wait for accept/    │
                    │     reject from pool)           │
                    └─────────────────────────────────┘
                            │
                       ┌────┴────┐
                       │          │
                    Accept    Reject
                    Stale gap or invalid
                       │
                  (logged, stats)
                       │
             ┌─────────┴──────────┐
             │                    │
             ▼                    ▼
        Update stats         Log error
        gaps_submitted       (merit drift,
        (increment)          false positive)
```

---

## Single Nonce Processing: Detailed

```
┌─────────────────────────────────────────────────────────────────┐
│                   Worker Processing One Nonce                    │
│         (multiple workers race on same nonce)                    │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Read Current Pass (Atomic)
    pass = rpc_get_current_pass()
    ├─ h256:     [32 bytes] block hash
    ├─ nonce:    [4 bytes] current nonce (atomic shared)
    └─ target:   [8 bytes] minimum merit to submit

STEP 2: Sieve Candidates (CPU, Per-Thread)
    
    Input:  h256 (32B), nonce (4B), CRT metadata
    Output: candidates[] = sorted offsets of probable primes
    
    Algorithm:
    1. Compute base = hash_to_mpz(h256, nonce) mod primorial
    2. Create sieve bitmap [interval_size bits]
    3. Sieve of Eratosthenes: mark multiples of 2,3,5,...,p as composite
    4. Extract offsets where bitmap[i] = 0 (is_prime)
    
    Example:
    ┌───────────────────────────────────────────────┐
    │ Base: 0x123456... (from h256 XOR nonce)       │
    │ Primorial: 2*3*5*7*11*13*...*p                │
    │ Interval: [base, base + primorial]            │
    │                                               │
    │ Sieve marks composites:                       │
    │ base+0:    composite (even)                   │
    │ base+2:    PRIME ← candidate[0]               │
    │ base+3:    composite (div 3)                  │
    │ base+5:    PRIME ← candidate[1]               │
    │ base+7:    PRIME ← candidate[2]               │
    │ base+12:   composite (div 2,3)                │
    │ base+13:   PRIME ← candidate[3]               │
    │ ...                                           │
    │ Total: ~1000-5000 candidates per nonce        │
    └───────────────────────────────────────────────┘
    
    Output: candidates = [2, 5, 7, 13, 19, 23, ...]  (offsets)

STEP 3: Primality Testing (GPU Batch for shift≥512)
    
    Input:  candidates[], n_candidates (~4000)
    Output: is_prime[] = array of 0/1 (1=probable prime)
    
    GPU Path (shift ≥ 512):
    ┌──────────────────────────────────────────────────┐
    │ GPU Fermat Batch Kernel (CGBN)                   │
    │                                                  │
    │ For each candidate c in candidates[]:           │
    │   1. Compute p = (base + c) * 2^shift           │
    │   2. Test p for primality:                      │
    │      - Fermat: p^(p-2) mod p ≟ 1                │
    │      - 16 rounds (2, 3, 5, 7, ...)              │
    │   3. Set is_prime[i] = 1 if all pass            │
    │                                                  │
    │ Kernel execution: ~50ms on RTX 4090             │
    │ Throughput: 4096 candidates / 0.05s = 81k/s    │
    │ Per-candidate Fermat: 16 multiplications        │
    └──────────────────────────────────────────────────┘
    
    CPU Path (shift < 512):
    ┌──────────────────────────────────────────────────┐
    │ CPU Fermat + Trial Division                      │
    │                                                  │
    │ For each candidate c in candidates[]:           │
    │   1. Trial division (up to limit, e.g., 2^24)   │
    │      → eliminates ~99% as composite              │
    │   2. Euler criterion (fast):                     │
    │      (2^((p-1)/2) mod p) ≟ ±1                   │
    │   3. Fermat (if both pass):                      │
    │      (2^(p-1) mod p) ≟ 1                        │
    │                                                  │
    │ Total: ~0.1-1.0 sec for 4000 candidates         │
    └──────────────────────────────────────────────────┘
    
    Output: is_prime = [0, 1, 1, 0, 1, 0, 1, ...]

STEP 4: Gap Detection (Consecutive Pairs)
    
    ┌─────────────────────────────────────────────────┐
    │  is_prime:  [0, 1, 1, 0, 1, 0, 1, 1, 0, ...]    │
    │  offset #:  [0, 1, 2, 3, 4, 5, 6, 7, 8, ...]    │
    │                                                  │
    │ Scan for consecutive 1's:                       │
    │   is_prime[1]=1, is_prime[2]=1  ← GAP!          │
    │     gap_base = candidates[1]                     │
    │     gap_len = candidates[2] - candidates[1]      │
    │     merit = gap_len / ln(base + candidates[1])   │
    │                                                  │
    │   is_prime[4]=1, is_prime[5]=0  ← NO gap        │
    │   is_prime[6]=1, is_prime[7]=1  ← GAP!          │
    │     gap_base = candidates[6]                     │
    │     gap_len = candidates[7] - candidates[6]      │
    │     merit = gap_len / ln(base + candidates[6])   │
    │                                                  │
    │ Total: ~10-50 gaps per nonce (merit ≥ 14)       │
    └─────────────────────────────────────────────────┘
    
    Output: gaps[] = array of {base, len, merit}

STEP 5: Merit Filter + BPSW (High-Merit Gaps Only)
    
    For each gap in gaps[]:
        if (merit < scan_target):  /* e.g., 16.0 */
            skip  ← vast majority filtered here
        else:
            ┌─────────────────────────────────┐
            │ BPSW Verification (Deterministic)│
            │                                 │
            │ p1 = (base + gap_base) * 2^shift│
            │ p2 = (base + gap_base + gap_len)│
            │      * 2^shift                  │
            │                                 │
            │ Both must satisfy:              │
            │ - Miller-Rabin(base 2)          │
            │ - Lucas-Lehmer test             │
            │                                 │
            │ Result: Deterministic primality │
            │ (no Carmichael number fools)    │
            └─────────────────────────────────┘
            
            if (valid):
                ✓ SUBMIT via RPC
            else:
                ✗ Log as false positive
                  (merit drift or rounding error)

STEP 6: Advance Nonce (Atomic, Lock-Free)
    
    do {
        old = atomic_load(&g_current_nonce)
        if (old == UINT32_MAX):
            ABORT  ← nonce wrap, mining complete
        new = old + 1
    } while (!atomic_compare_exchange_weak(&g_current_nonce, &old, &new))
    
    Race condition safe:
    - If worker A and B both attempt to advance nonce 12345:
    - One succeeds (gets 12346), other retries and gets 12347
    - No gap is missed, no gap is double-tested
    
    ┌──────────────────────────────────┐
    │ Nonce Pool (Atomic Counter)       │
    │  [12345] ← current value          │
    │            (all workers read)     │
    │                                  │
    │ Worker#0: reads 12345, advances  │
    │            → CAS 12345 → 12346 ✓ │
    │                                  │
    │ Worker#1: reads 12345, tries CAS │
    │            → fails (already 12346)│
    │            → retry, reads 12346   │
    │            → CAS 12346 → 12347 ✓ │
    │                                  │
    │ New value: [12347]               │
    └──────────────────────────────────┘

STEP 7: Check for New Block (Rare)
    
    if (rpc_new_block_detected()):
        break  ← exit worker loop
        (main thread spawns new workers for new block)
    else:
        continue to STEP 1 (read next nonce)
```

---

## Data Structure Sizes (Memory Footprint)

```
Per-GPU Worker:
├─ CRT state (shared):     ~100 KB (primorial, offset, sieve bitmap)
├─ Sieve bitmap (thread):  ~64 KB (for 512K interval)
├─ Candidate buffer:       ~32 KB (4000 candidates × 8 bytes)
├─ GPU Fermat batch:       ~16 MB (4096 × 1024-bit numbers)
├─ GPU results:            ~4 KB (4096 × 1 byte)
├─ Primality context:      ~16 KB (GMP mpz_t scratch objects)
└─ Statistics:             ~1 KB
                          ─────────
                          ≈ 16-17 MB per GPU worker

Total (1 A100 + CPU):     ≈ 17 MB
Total (8 A100s):          ≈ 136 MB

Memory bandwidth:
├─ Host→GPU (candidates): 4000 × 128 bytes = 512 KB/transfer
├─ GPU→Host (results):    4096 × 1 byte = 4 KB/transfer
├─ PCIe x16 (≈14 GB/s):   ~36 µs per batch
└─ GPU execution:         50 ms >> transfer time (negligible overhead)
```

---

## Performance Expectations

```
Shift 512 + RTX 4090:
├─ Sieve/nonce:     ~1 ms
├─ GPU Fermat/4k:   ~50 ms
├─ Gap detection:   ~1 ms
├─ Total/nonce:     ~52 ms
├─ Nonces/sec:      19 nonces/sec
├─ Candidates/nonce:4000
├─ Candidates/sec:  76k/sec
├─ Gaps/nonce:      ~20 (merit ≥ 14)
├─ Gaps/sec:        380 gaps/sec
└─ Submitted:       ~1-5 gaps/sec (merit ≥ 16)

Shift 768 + A100:
├─ GPU Fermat/4k:   ~200 ms (larger integers)
├─ Total/nonce:     ~210 ms
├─ Nonces/sec:      4.8 nonces/sec
├─ Gaps/sec:        ~5 gaps/sec (merit ≥ 16)

Shift 1024 + A100:
├─ GPU Fermat/4k:   ~500 ms (1024-bit)
├─ Total/nonce:     ~510 ms
├─ Nonces/sec:      2 nonces/sec
├─ Gaps/sec:        ~1 gap/sec (merit ≥ 16)
```

---

## Multithreading Benefits

| Aspect | Monolithic (Old) | Multithreaded (New) |
|--------|-----------------|-------------------|
| Sieve overlap | None | All workers simultaneous (CPU parallel) |
| GPU utilization | Single stream | All GPUs busy |
| Cache locality | High (single thread) | Medium (per-thread state, shared CRT) |
| Synchronization | Simple | Atomic nonce, minimal locking |
| Scalability | 1 GPU → rebuild | N GPUs → N workers |
| False sharing | N/A | Possible (mitigated by per-thread locals) |
| RPC latency impact | Blocks mining | Workers continue (separate thread) |

---

## References

- Gap Candidate struct: [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) §1
- Sieve implementation: [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) §2
- Gap detection algorithm: [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) §3
- Worker thread: [GAP_SCANNING_DETAILED.md](GAP_SCANNING_DETAILED.md) §4
- CRT file format: [UNIFIED_GAPMINER_REFERENCE.md](UNIFIED_GAPMINER_REFERENCE.md) §5
