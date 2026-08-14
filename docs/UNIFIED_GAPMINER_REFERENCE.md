# Unified GapMiner Source Reference Mapping

**Source**: `/home/dejan/Downloads/lentmaistrouve_GapMiner-GPU-CRT-Windows-x64-20260803/SOURCE-SNAPSHOT/GapMiner-Unified-source-20260803`

**Purpose**: Cross-reference between old architecture (existing cpugapminer), Unified GapMiner, and new clean architecture

---

## Key Source Files Analysis

### CRT + Candidate Management

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/ChineseSet.h/cpp` | CRT file storage + loading | `new_src/crt_engine.h` | `primorial`, `offset`, pre-sieved bitmap |
| `src/ChineseSieve.h/cpp` | CRT-based sieve integration | `new_src/sieve_crt.c` | Applies CRT to reduce candidate space |
| `src/GapCandidate.h/cpp` | Candidate descriptor struct | `new_src/gap_candidate.h` | **Lazy materialization**: base+row factorization |
| `src/BestChinese.h/cpp` | CRT optimization utils | `new_src/crt_autotuner.c` | Finding best primes/offsets for shift |
| `src/CrtOffsetOptimizer.h/cpp` | Offset search + tuning | `new_src/crt_generator.c` | Auto-generate CRT for given shift |

### Sieve + Candidate Extraction

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/PoWCore/src/Sieve.h` | Bit-packed candidate bitmap | `new_src/sieve_core.h` | Macros: `is_prime()`, `set_composite()` |
| `src/PoWCore/src/Sieve.cpp` | Sieve of Eratosthenes engine | `new_src/sieve_core.c` | Trial division + wheel sieve |
| `src/OnePrimeSieve.h/cpp` | Single-prime sieve | `new_src/sieve_single.c` | Optimized for small primes |
| `src/HybridSieve.h/cpp` | CPU + GPU sieve coordination | `new_src/sieve_hybrid.c` | Delegates heavy lifting to GPU |

### Primality Testing

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/PoWCore/src/PoWUtils.h` | GMP utilities + Fermat test | `new_src/primality_fermat.c` | Base-2 Fermat rounds |
| `src/PoWCore/src/PoW.h/cpp` | PoW descriptor + validation | `new_src/gap_validation.c` | Gap boundaries, merit, BPSW |
| `src/CUDAFermat.h/cu` | CGBN GPU Fermat batching | `new_src/gpu/fermat_cuda.cu` | 4096-candidate batch, 16 rounds |
| `src/GPUFermat.h/cpp` | GPU Fermat wrapper/scheduler | `new_src/gpu/fermat_interface.h` | Batch coordination + results |

### Worker Thread + Mining Loop

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/Miner.h/cpp` | Main miner orchestration | `new_src/miner_loop.c` | Worker spawning, coordination |
| `src/ShareProcessor.h/cpp` | Result queue + submission | `new_src/submit_engine.c` | Queues gaps, checks staleness |
| `src/BlockHeader.h/cpp` | Block header + nonce tracking | `new_src/block_header.h` | Pass header struct |
| `src/MiningConfig.h/cpp` | Configuration parsing | `new_src/config.c` | CLI args, auto-tuning |

### RPC + Network

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/Rpc.h/cpp` | HTTP RPC client (curl-based) | `new_src/rpc_interface.c` | getwork + getblocktemplate |
| `src/Stratum.h/cpp` | Stratum protocol handler | `new_src/stratum.c` (keep as-is) | Mining notifications |
| `src/ResultJournal.h/cpp` | Log accepted/rejected shares | `new_src/result_journal.c` | Submission history |

### GPU Integration

| Unified File | Purpose | New new_src/ Equivalent | Key Pattern |
|--------------|---------|------------------------|-------------|
| `src/gpu/integration/GpuMiningAdapter.h` | GPU abstraction layer | `new_src/gpu/gpu_adapter.h` | CUDA vs OpenCL dispatch |
| `src/gpu/cuda/CudaContext.h` | CUDA device management | `new_src/gpu/cuda_context.c` | Device init, stream management |
| `src/gpu/opencl/OpenCLContext.h` | OpenCL fallback | `new_src/gpu/opencl_context.c` | Portable GPU fallback |

---

## Architecture Mapping: Old → Unified → New

### Single-Threaded Discovery (Old cpugapminer)

```
main.c (13K lines)
├── Sieve (wheels, presieve)
├── CRT processing (monolithic)
├── GPU vs CPU path (merged)
├── Worker threads (producer/consumer)
└── RPC polling (interleaved)
```

### Multi-Threaded with CRT (Unified GapMiner)

```
Miner (class-based orchestration)
├── Worker threads (N per GPU)
├── Per-thread:
│   ├── ChineseSieve + CRT
│   ├── GapCandidate queue
│   ├── Fermat (CPU or GPU)
│   └── Gap detection
├── Separate RPC poll thread
└── ShareProcessor (result queue)
```

### Clean Multithreaded Rewrite (New)

```
new_src/miner_loop.c
├── GPU worker threads (N)
│   ├── Per-thread:
│   │   ├── crt_engine (shared read-only)
│   │   ├── sieve_state (thread-local)
│   │   ├── primality_ctx (thread-local)
│   │   ├── gpu_batch (thread-local GPU)
│   │   └── Atomic gap detection
│   └── Direct submission (no queue)
├── RPC poll thread (separate)
│   └── Atomic pass updates
└── Main thread (spawn/join)
```

---

## Key Insights from Unified GapMiner

### 1. Candidate Management (GapCandidate.h)

**Observation**: Unified uses `GapCandidateStartContext` with immutable base+primorial pair to avoid recomputing per-candidate.

**Implementation idea**:
```c
/* Unified approach */
struct gap_candidate_start_ctx {
    mpz_t base;
    mpz_t primorial;
};

void materialize(mpz_t dest, const struct gap_candidate_start_ctx *ctx, uint64_t row) {
    /* start = base + row * primorial */
    mpz_addmul_ui(dest, ctx->primorial, row);
}

/* Vs. old approach: store full mpz_t for each candidate → more memory */
```

### 2. Bitmap Encoding Efficiency (Sieve.h)

**Observation**: Use 64-bit (or 32-bit) words, pack is_prime() as bit_at macro.

**Performance**: 8-64x memory reduction vs. byte-per-candidate.

**Macros to adopt**:
```c
#define is_prime(ary, i) !bit_at(ary, i)
#define set_composite(ary, i) set_bit(ary, i)
#define bit_at(ary, i) (ary[(i) >> 6] & (1ULL << ((i) & 0x3f)))
```

### 3. GPU Fermat Batching (CUDAFermat.cu)

**Observation**: Batch 4096 candidates, ship once to GPU, 50ms kernel execution.

**Performance**: ~400k candidates/sec on RTX 4090, shift 512.

**Key pattern**:
```c
struct cuda_fermat_batch {
    uint32_t batch_size;  /* e.g., 4096 */
    uint8_t results[];    /* 1 = probable prime */
};

void cuda_fermat_batch_test(
    struct cuda_fermat_batch *batch,
    const uint64_t *candidates,  /* Array of candidates */
    uint32_t n,
    uint32_t fermat_rounds,
    uint8_t *results
);
```

### 4. RPC + ShareProcessor (Unified pattern)

**Observation**: ShareProcessor is separate class with its own queue + thread.

**Design benefit**: Workers never block on RPC, results are asynchronously processed.

**Pattern**:
```c
struct share_processor {
    queue<struct gap_result> results;
    pthread_t processor_thread;
    pthread_mutex_t queue_lock;
};

/* Worker: enqueue and continue */
share_processor_enqueue(processor, gap_result);

/* Processor thread: dequeue, validate, submit */
void *share_processor_main(void *arg) {
    struct share_processor *proc = arg;
    while (running) {
        struct gap_result res = share_processor_dequeue(proc);
        if (bpsw_verify(res)) {
            rpc_submit(res);
        }
    }
}
```

### 5. Worker Thread Independence (Miner.cpp)

**Observation**: Each GPU worker is completely independent; no producer/consumer sync.

**Design benefit**: No false sharing, no cache bouncing, true parallelism.

**Pattern**:
```c
struct gpu_worker {
    int gpu_id;
    struct cuda_context *cuda;
    struct chinese_set *crt;  /* Shared read-only */
};

void *gpu_worker_main(void *arg) {
    struct gpu_worker *w = arg;
    while (running) {
        /* 1. Sieve (CPU, fast) */
        /* 2. GPU Fermat (GPU, batch) */
        /* 3. Gap detection (CPU, instant) */
        /* 4. Submit (RPC, async) */
        /* No sync except atomic nonce */
    }
}
```

---

## New Architecture Checklist: Files to Create from Unified Patterns

### Phase 1: Core CRT + Sieve

- [ ] `new_src/crt_engine.c` — Extract/port from src/crt_runtime_cpu.c
- [ ] `new_src/chinese_set.c` — Port ChineseSet loading from Unified
- [ ] `new_src/sieve_core.c` — Adapt Sieve.cpp to C, add Eratosthenes
- [ ] `new_src/gap_candidate.h` — Define candidate struct (from GapCandidate.h pattern)
- [ ] `new_src/gap_candidate_extract.c` — Extract candidates from sieve bitmap

### Phase 2: Worker + Primality

- [ ] `new_src/worker_gpu.c` — Per-GPU worker thread (from Miner::worker_thread pattern)
- [ ] `new_src/primality_fermat.c` — CPU Fermat testing
- [ ] `new_src/primality_euler.c` — Euler's criterion optimization
- [ ] `new_src/primality_bpsw.c` — BPSW (Miller-Rabin + Lucas)
- [ ] `new_src/gap_detection.c` — Consecutive prime pair hunting (from PoW gap detection)

### Phase 3: GPU Integration

- [ ] `new_src/gpu/fermat_cuda.cu` — CGBN batch Fermat (from CUDAFermat.cu)
- [ ] `new_src/gpu/fermat_opencl.c` — OpenCL fallback
- [ ] `new_src/gpu/gpu_adapter.h` — Abstraction layer

### Phase 4: RPC + Submission

- [ ] `new_src/rpc_interface.c` — Wrap stratum.c, auto-detect mode
- [ ] `new_src/submit_engine.c` — Payload assembly (both modes)
- [ ] `new_src/share_processor.c` — Result queue (from ShareProcessor pattern)

### Phase 5: Configuration + CRT Generation

- [ ] `new_src/crt_generator.c` — Generate CRT files on-the-fly
- [ ] `new_src/config.c` — CLI parsing, auto-tuning (from MiningConfig.cpp)

---

## Testing Strategy

1. **Unit tests** (for each module):
   - Sieve: verify bitmap encoding, candidate extraction
   - CRT: verify file loading, primorial encoding
   - Primality: verify Fermat/Euler/BPSW on known Carmichael numbers
   - Gap detection: verify consecutive pair detection

2. **Integration tests**:
   - Spawn N workers, mine 1 block, check no false gaps
   - Verify merit calculation matches on-chain verification
   - Test RPC mode switching (getwork ↔ GBT)

3. **Performance validation**:
   - CPU throughput: ~800 gaps/sec (shift 256, 8-core)
   - GPU throughput: ~20k gaps/sec (shift 512, RTX 4090)
   - Memory footprint: <1GB per GPU worker

---

## External Library Dependencies (from Unified)

- **GMP** (libgmp) — Multi-precision arithmetic (primality testing)
- **MPFR** (libmpfr) — Floating-point multi-precision (merit calculation)
- **CUDA** (libcuda, libcudart) — GPU Fermat kernel
- **OpenCL** (libOpenCL) — GPU fallback
- **curl** (libcurl) — HTTP RPC client
- **OpenSSL** (libcrypto) — SHA256, TLS for RPC
- **Jansson** (libjansson) — JSON parsing for RPC
- **pthreads** (libpthread) — Worker thread management

---

## References in Unified Source

- Performance targets documented in `SMOKE-TEST.txt`
- Configuration examples in `MINER-CONFIG.cmd`
- CRT generation algorithm in `CrtOffsetOptimizer.h`
- BPSW implementation in `PoWCore/src/PoW.cpp`
- GPU batch strategy in `CUDAFermat.cu`
- Worker spawning in `Miner.cpp` (lines ~150–250)
