# GapMiner V2 — Clean Architecture Redesign

**Status**: Architecture planning + gap scanning documentation  
**Date Started**: 2026-08-14  
**Base**: Copied from cpugapminer (full history)  

---

## What is GapMiner V2?

Clean rewrite of gap mining miner for Gapcoin, based on comprehensive analysis of:
- **Existing cpugapminer** (13,137 lines of monolithic C)
- **Unified GapMiner** (lentmaistrouve's GPU-accelerated version, Windows/Linux)
- **Proven patterns** from GPU+CRT gap discovery

### Key Improvements Over V1

| Aspect | V1 (cpugapminer) | V2 (gapminer_v2) |
|--------|-----------------|-----------------|
| **Architecture** | Monolithic (main.c 13K lines) | Modular 20+ clean C files |
| **Threading** | Producer/Consumer queue | N workers, atomic nonce (lock-free) |
| **GPU Support** | Single GPU, mixed with CPU | Per-GPU worker, batch Fermat (4096 cand/batch) |
| **RPC** | Interleaved with mining | Separate polling thread |
| **Submission** | Queued, async | Direct, immediate (no queue) |
| **CRT** | File-only | File + auto-generation |
| **False Gaps** | Fermat can fool | BPSW deterministic verification |
| **Memory** | ~500MB per worker | ~16MB per worker (+ 16MB GPU VRAM) |
| **Scaling** | 1 GPU max | N GPUs → N independent workers |

---

## Documentation Structure

### 📚 Core Architecture
- **[NEW_MINER_ARCHITECTURE_PLAN.md](docs/NEW_MINER_ARCHITECTURE_PLAN.md)** (1633 lines)
  - What to preserve (CRT engine, RPC, submit logic)
  - What to rewrite (sieve, worker loop, primality testing)
  - Zero-config auto-tuning
  - Performance targets

### 🔄 Gap Scanning Pipeline
- **[GAP_SCANNING_FLOW.md](docs/GAP_SCANNING_FLOW.md)** (394 lines)
  - Visual flow diagrams (RPC → Sieve → Fermat → Gap → Submit)
  - Step-by-step detailed execution
  - Data structure sizes
  - Performance expectations

- **[GAP_SCANNING_DETAILED.md](docs/GAP_SCANNING_DETAILED.md)** (457 lines)
  - Proven patterns from Unified GapMiner
  - Gap candidate model (lazy materialization)
  - Sieve bitmap encoding
  - Gap detection algorithm
  - BPSW verification strategy
  - Worker thread coordination

- **[GAP_SCANNING_GUIDE.md](docs/GAP_SCANNING_GUIDE.md)** (354 lines)
  - Quickstart implementation guide
  - 5-phase development roadmap
  - Testing strategy
  - Key takeaways

### 🔗 Source Reference
- **[UNIFIED_GAPMINER_REFERENCE.md](docs/UNIFIED_GAPMINER_REFERENCE.md)** (303 lines)
  - File mapping: Unified → new_src/
  - Key insights from proven implementation
  - External library dependencies
  - Testing strategy

---

## Quick Start: Reading Order

1. **Planning phase**: [NEW_MINER_ARCHITECTURE_PLAN.md](docs/NEW_MINER_ARCHITECTURE_PLAN.md) (15 min read)
2. **How it works**: [GAP_SCANNING_GUIDE.md](docs/GAP_SCANNING_GUIDE.md) (10 min)
3. **Visual flow**: [GAP_SCANNING_FLOW.md](docs/GAP_SCANNING_FLOW.md) (15 min)
4. **Implementation details**: [GAP_SCANNING_DETAILED.md](docs/GAP_SCANNING_DETAILED.md) (20 min)
5. **Source mapping**: [UNIFIED_GAPMINER_REFERENCE.md](docs/UNIFIED_GAPMINER_REFERENCE.md) (for coding)

---

## 5-Phase Implementation Roadmap

### Phase 1: Core CRT + Sieve (Week 1)
- [ ] Extract `new_src/crt_engine.c` from src/crt_runtime_cpu.c
- [ ] Create `new_src/sieve_core.c` (port from Sieve.cpp)
- [ ] Create `new_src/gap_candidate.h` (data structures)
- [ ] Create `new_src/gap_detection.c` (consecutive prime detection)

### Phase 2: Primality Testing (Week 1-2)
- [ ] `new_src/primality_fermat.c` — CPU Fermat
- [ ] `new_src/primality_euler.c` — Euler's criterion (2x speedup)
- [ ] `new_src/primality_bpsw.c` — BPSW verification (deterministic)

### Phase 3: Worker Threads (Week 2)
- [ ] `new_src/worker_gpu.c` — Per-GPU worker loop
- [ ] `new_src/miner_farm.c` — Worker spawning + atomic nonce
- [ ] Implement atomic nonce advancement (lock-free)

### Phase 4: GPU Integration (Week 2-3)
- [ ] `new_src/gpu/fermat_cuda.cu` — CGBN batch kernel
- [ ] `new_src/gpu/gpu_adapter.h` — CUDA/OpenCL abstraction
- [ ] Performance tuning (4096 candidates/batch)

### Phase 5: RPC + Finish (Week 3)
- [ ] `new_src/rpc_interface.c` — Wrap stratum.c
- [ ] `new_src/submit_engine.c` — Payload assembly (both modes)
- [ ] `new_src/main.c` — Entry point, CLI, worker spawn
- [ ] `new_src/crt_generator.c` — Auto-generate CRT files

---

## Key Design Decisions

### ✅ Atomic Nonce Pool (Lock-Free)
All N workers race on single `_Atomic uint32_t` counter.
- **Benefit**: No mutexes, no false sharing
- **Trade-off**: All workers access same nonce (can use different shifts via separate passes)

### ✅ Per-GPU Worker (Independent)
Each GPU gets own worker thread, no producer/consumer queue.
- **Benefit**: True parallelism, each GPU runs full pipeline
- **Trade-off**: More memory per worker (but only 16MB main + 16MB GPU VRAM)

### ✅ Direct RPC Submission (No Queue)
Workers submit directly to RPC without waiting for response.
- **Benefit**: Workers unblocked, async callbacks update stats
- **Trade-off**: Must handle RPC failures/rejections asynchronously

### ✅ BPSW Only on Submit (Expensive Check)
Use fast Fermat for discovery (~100-400 gaps/sec), BPSW only for high-merit (~1-5 gaps/sec).
- **Benefit**: 100x fewer expensive checks
- **Trade-off**: Some false positives (caught by BPSW before submit)

### ✅ Separate RPC Thread (Non-Blocking)
Dedicated thread polls getblocktemplate/getwork, updates atomic pass state.
- **Benefit**: Mining never stalls on network
- **Trade-off**: Additional thread coordination

---

## Performance Targets

| Hardware | Shift | CPU Path | GPU Path | Gaps/sec |
|----------|-------|----------|----------|----------|
| 8-core CPU | 256 | Trial+Euler | — | ~800 |
| RTX 4090 | 512 | Sieve | GPU Fermat | ~20k |
| A100 | 768 | Sieve | GPU Fermat | ~5k |
| A100 | 1024 | Sieve | GPU Fermat | ~1k |

**Discovery rate** (merit ≥ 14): 100-400 gaps/sec  
**Submit rate** (merit ≥ 16, BPSW): 1-5 gaps/sec

---

## Memory Footprint

**Per GPU worker**:
- CRT shared state: ~100 KB
- Sieve bitmap: ~64 KB
- Candidate buffer: ~32 KB
- GMP scratch: ~16 KB
- GPU batch (VRAM): ~16 MB

**Total per worker**: ~16-17 MB (mostly GPU VRAM)

**Multi-GPU (8 A100s)**: ~136 MB total

---

## Testing Strategy

### Unit Tests
- Sieve: verify bitmap → candidate extraction
- Gap detection: verify consecutive prime pair identification
- BPSW: verify on Carmichael numbers (5, 561, 1105, ...)
- Merit: verify calculation vs. reference implementation

### Integration Tests
- Mine 1000 nonces, verify no false gaps
- Test RPC mode switching (getwork ↔ getblocktemplate)
- Verify atomic nonce advancement (no skips, no double-tests)

### Performance Validation
- Measure gaps/sec for each shift
- Profile GPU utilization vs. memory bandwidth
- Memory footprint per worker
- Cache behavior (L1/L2/L3 misses)

---

## File Structure (Post-Rewrite)

```
gapminer_v2/
├── docs/
│   ├── NEW_MINER_ARCHITECTURE_PLAN.md      ← Master plan
│   ├── GAP_SCANNING_FLOW.md                ← Visual pipeline
│   ├── GAP_SCANNING_DETAILED.md            ← Implementation patterns
│   ├── GAP_SCANNING_GUIDE.md               ← Quickstart
│   └── UNIFIED_GAPMINER_REFERENCE.md       ← Source mapping
│
├── new_src/                                 ← Clean rewrite (Phase 1-5)
│   ├── crt_engine.c/h                      ← Preserve from src/crt_runtime_cpu.c
│   ├── rpc_interface.c/h                   ← Wrap src/stratum.c
│   ├── submit_engine.c/h                   ← Submit payloads (both modes)
│   │
│   ├── sieve_core.c/h                      ← Port from Sieve.cpp
│   ├── gap_candidate.h                     ← Data structures
│   ├── gap_detection.c                     ← Consecutive prime detection
│   ├── gap_validation.c                    ← Merit + boundary checks
│   │
│   ├── primality_fermat.c/h                ← CPU Fermat testing
│   ├── primality_euler.c/h                 ← Euler's criterion (2x faster)
│   ├── primality_bpsw.c/h                  ← BPSW verification
│   │
│   ├── worker_gpu.c/h                      ← Per-GPU worker thread
│   ├── miner_farm.c/h                      ← Worker coordination
│   ├── atomic_nonce.c/h                    ← Lock-free advancement
│   │
│   ├── crt_generator.c/h                   ← Auto-generate CRT files
│   ├── config.c/h                          ← CLI parsing + auto-tuning
│   │
│   ├── gpu/
│   │   ├── fermat_cuda.cu                  ← CGBN batch kernel
│   │   ├── fermat_opencl.c                 ← OpenCL fallback
│   │   └── gpu_adapter.h                   ← CUDA/OpenCL abstraction
│   │
│   └── main.c                              ← Entry point
│
├── src/                                    ← Keep as reference/backup
│   ├── main.c                              ← Old monolithic (13K lines)
│   ├── crt_runtime_cpu.c                   ← CRT engine (preserve logic)
│   ├── stratum.c                           ← RPC protocol (keep as-is)
│   └── ...
│
├── Makefile                                ← Build new_src/
├── README.md                               ← This file
└── ...
```

---

## Migration Path (Old → New)

### Preserve (Extract Directly)
- `src/crt_runtime_cpu.c` → `new_src/crt_engine.c` (no rewrites, minimal API cleanup)
- `src/stratum.c` → wrap in `new_src/rpc_interface.c` (keep as-is)
- Payload assembly → `new_src/submit_engine.c` (from main.c ~400 lines)

### Rewrite (From Scratch)
- Main loop (old ~10K lines) → `new_src/miner_loop.c` (~300 lines, cleaner)
- Sieve (wheel+presieve) → `new_src/sieve_core.c` (from Sieve.cpp)
- Primality (scattered) → `new_src/primality_*.c` (modular, BPSW + Euler)
- Worker threads → `new_src/worker_gpu.c` (per-GPU, atomic nonce)
- RPC polling → `new_src/rpc_interface.c` (separate thread)

### New Capabilities
- `new_src/crt_generator.c` — Auto-generate CRT files
- `new_src/gpu/fermat_cuda.cu` — CGBN GPU kernel
- `new_src/atomic_nonce.c` — Lock-free nonce advancement
- `new_src/gap_candidate.h` — Explicit candidate structures

---

## References

### Gapcoin PoW
- Merit formula: `merit = gap_length / ln(gap_base)`
- Validation: Gap between two probable primes, maximal at that base
- Target: Only submit gaps with merit ≥ scan_target (e.g., 16.0)

### GPU Acceleration
- CGBN (CUDA C++ Big Number library) for 1024-bit arithmetic
- Fermat testing: (a^(p-1) mod p) ≟ 1 (16 rounds, 2,3,5,7,...)
- Batch size: 4096 candidates per kernel call (~50ms RTX 4090)

### CRT (Chinese Remainder Theorem)
- Primorial = 2 × 3 × 5 × 7 × ... × p (product of first n primes)
- CRT file stores primorial + offset + pre-sieved bitmap
- Reduces candidate search space dramatically (~99% composite elimination)

### BPSW (Baillie-PSW Primality Test)
- Combines Miller-Rabin (base 2) + Lucas-Lehmer test
- Deterministic for n < 3,317,044,064,679,887,385,961,981
- No known pseudoprimes (Carmichael numbers don't fool it)
- Slow (~1ms per number) — only for final submission

---

## Next Steps

1. **Read docs** — Start with [NEW_MINER_ARCHITECTURE_PLAN.md](docs/NEW_MINER_ARCHITECTURE_PLAN.md)
2. **Understand gap scanning** — [GAP_SCANNING_GUIDE.md](docs/GAP_SCANNING_GUIDE.md)
3. **Phase 1 implementation** — Combine CRT + sieve + gap detection
4. **Performance tuning** — Profile against targets
5. **Multi-GPU validation** — Test atomic nonce + worker coordination

---

**Status**: Ready for Phase 1 implementation (Core CRT + Sieve)  
**Contact**: See AGENTS.md for project rules and conceptual escape doctrine
