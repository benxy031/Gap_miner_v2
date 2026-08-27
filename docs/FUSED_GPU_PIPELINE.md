# Fused GPU-Resident Sieve + Miller-Rabin Pipeline

**Status:** design (implementation staged). Target: eliminate CPU sieve/residue
work and CPU↔GPU round-trips so the GPU does sieve → extract → pack → MR in
device memory, and a deeper sieve becomes free.

---

## 1. Motivation — where the time actually goes

In CRT + GPU mode (`--crt-file … --enable-gpu-fermat`), measured on RTX 3070:

| Stage | Where | Cost |
|---|---|---|
| Miller-Rabin (CGBN TPI=8) | GPU | **bottleneck** (~1.55M cand/s ceiling) |
| Sieve residue compute (`base mod p`) | CPU | mpz_fdiv_ui × N primes; saturates 4C/8T if N > ~100k |
| Sieve mark + extract | CPU | bitmap marking + ctz extraction |
| Candidate limb packing | CPU | base + offset → limbs, no GMP |
| H2D packed candidates | PCIe | ~91 KB/window @ 948 cand × 12 limbs |
| D2H results | PCIe | ~1 KB/window |

The GPU MR is the throughput ceiling. The only way to test **fewer**
candidates per window is a **deeper sieve** (more primes), but today that
costs CPU residue work that saturates the weak host — so `sieve_primes` is
stuck at 100k (948 survivors/window). Raising it to 1M would give ~790
survivors (−16%) but the CPU residue pass makes it a **net loss** (measured).

**The fix:** move residue + mark + extract + pack **onto the GPU**, where
parallel residue computation is nearly free (1M primes across thousands of
threads). Then `sieve_primes = 1M` costs ~nothing and cuts MR work ~16% →
**~+16–20% windows/s at the same MR ceiling.**

A previous prototype (`GPU_SIEVE=1`, `gpu_sieve_mark_from_base`) already
computes residues on-device, but it was a **net loss** because every window
still did `H2D base → GPU mark → D2H bitmap → CPU extract+pack → H2D cands →
GPU MR`, i.e. **two extra round-trips + CPU extract/pack per window**.
Fusion removes those round-trips: the bitmap, extracted candidates and packed
limbs never leave the GPU.

---

## 2. Current vs fused data flow

### Current (per CRT window)

```
CPU: base (mpz) ──residues──▶ CPU sieve mark/extract ──▶ offsets[]
CPU: pack offsets → limb array (SoA)
H2D: limb array ─────────────────────────────────────▶ GPU MR kernel
D2H: is_prime[] ◀─────────────────────────────────────
CPU: gap_detection_find(offsets, is_prime)
```

Round-trips per window: **1× H2D (cands) + 1× D2H (results)**. With the
`GPU_SIEVE=1` prototype: **+1 H2D (base) +1 D2H (bitmap)** on top.

### Fused (per CRT window)

```
CPU: base limbs (12×u64, mpz_export once)
H2D: base limbs ──────────────────────────────────────▶ GPU
      GPU: residues ─▶ mark bitmap ─▶ extract survivors ─▶ pack SoA limbs
      GPU: append survivors to persistent device batch
      (every ~40k candidates) GPU MR kernel on the batch
D2H: is_prime[] + offsets[]  ◀──────────────────────────
CPU: gap_detection_find(offsets, is_prime)
```

Round-trips per window: **1× H2D (96 B base) + 1× D2H (~9 KB results)**.
The bitmap (~64 KB), the extracted offsets (~8 KB) and the packed candidates
(~91 KB) stay in device memory.

---

## 3. Kernel design

Four device stages, the first three fused into a single kernel launch chain
(no host sync between them). Primes/inv_p are uploaded **once** at init and
cached on device.

### K1 — residue + mark (fused)
- One thread per sieve prime `p`.
- Compute `r = base mod p` via Barrett: `q = (base_lo * inv_p) >> 64; r = base_lo − q*p`
  (for the low limb; higher primes only need the low-limb residue because the
  window is < 2^20 and p < 2^32, see existing `gpu_sieve_residues_kernel`).
- Mark bitmap bits `(r + k·p) mod odd_slots` for `k = 0…`.
- *Already exists* as `gpu_sieve_mark_kernel_batch` + `gpu_sieve_residues_kernel`;
  we fuse them so the residues buffer never round-trips.

### K2 — extract + pack (NEW, the core of this change)
- Grid-stride scan over bitmap words. For each word with a clear bit, use
  `__ffsll` to pop survivor positions.
- For each survivor at odd offset `t`: candidate = `base + first_odd + 2t`.
  Pack into `GPU_NLIMBS` little-endian limbs:
  - low limb = `base[0] + (first_odd + 2t)` with carry ripple into `base[1]`.
  - high limbs = copy of `base[1..AL-1]`.
- Write limbs to a **persistent SoA device buffer** at the atomic-incremented
  survivor slot (SoA matches `fermat_kernel_soa_t`/CGBN input layout).
- Write the survivor **offset** to a parallel `uint64_t` buffer for host
  gap detection.
- Survivor count via `atomicAdd` on a device counter.

### K3 — Miller-Rabin (existing)
- The existing CGBN `cgbn_fermat_kernel_soa_t<BITS,TPI>` / scalar kernel reads
  the device SoA buffer directly (a new device-pointer entry point; the
  arithmetic is unchanged).

### Host
- `cudaMemcpy` only the base (96 B) in, and `count + offsets + is_prime` out
  (~9 KB/window).
- No per-window `cudaDeviceSynchronize` on the sieve path; streams overlap
  window *i+1* sieve with window *i* MR.

---

## 4. Memory layout

- `d_primes[N]`, `d_inv_p[N]` — uploaded once (8 MB each at N=1M).
- `d_bitmap_pool[P][odd_words]` — pool of P window bitmaps (~64 KB each),
  reused round-robin; K1/K2 write one, K3 is done with a previous one.
- `d_cands_soa[AL][MAX_BATCH]` — persistent packed candidates (SoA).
- `d_offsets[MAX_BATCH]` — survivor offsets (for host gap detection).
- `d_count` — atomic survivor counter (one `u32`; read back for MR launch size).

---

## 5. Batching

The MR kernel is latency-bound and best at ~40k candidates (the batch-size
work we just did). One CRT window yields ~790–950 survivors, so:

- Windows are processed **independently** by K1/K2, appending survivors into
  the persistent buffers.
- When `d_count ≥ 40000` (or 256 windows), host launches K3 on the
  accumulated candidates, then D2H results + offsets and runs
  `gap_detection_find` per window (the offsets map each survivor back to its
  window via a per-window `[start, end)` offset range bookkept on the host).

This keeps the GPU MR saturated at the 40k batch optimum while the sieve runs
in the same stream without host round-trips.

---

## 6. Correctness

- **Parity gate:** K2 output must exactly match the CPU
  `sieve_core_run_from_cached_base_hybrid` survivor set for the same
  (base, primes, window). Add a test mode that runs both and `memcmp`s the
  offsets; any divergence = fail-closed to the CPU path.
- **Fail-closed:** any CUDA error disables the fused path and falls back to
  the existing CPU-sieve + GPU-MR path (which remains in the tree).
- **No false gaps:** a missed survivor only means a candidate isn't tested
  (missed gap, caught by the RGM validator in cpugapminer; here we gate on
  parity before enabling). An extra survivor (spurious) is the dangerous
  direction — the parity gate and the exact bitmask handling in K2 prevent it.

---

## 7. Staged implementation plan

1. **Stage 1 — device extract+pack kernel** (`gpu_sieve.cu`): add `K2`
   (bitmap → SoA limbs + offsets + count on-device). Unit-test it by
   launching K1+K2 for one window, D2H offsets, and `memcmp` against the CPU
   sieve. *(This is the only genuinely new arithmetic; the rest is wiring.)*
2. **Stage 2 — device→MR wiring:** add a `gpu_fermat` entry point that runs
   the MR kernel on a device SoA buffer + count (no H2D). Verify `is_prime`
   matches the current H2D path.
3. **Stage 3 — cross-window accumulation + streams:** persistent buffers,
   per-window offset-range bookkeeping, MR launch at 40k. A/B vs the current
   pipeline at `sieve_primes=100k` (expect ~0%, this is the parity milestone).
4. **Stage 4 — deep sieve:** raise `sieve_primes` to 1M now that residue cost
   is gone. A/B expect **~+16–20% windows/s**.

---

## 8. Expected outcome

| Milestone | Expected |
|---|---|
| Stage 1–3 (parity, 100k primes) | ~0% (just re-plumbed) |
| Stage 4 (1M primes, free residues) | **+16–20%** windows/s |
| CPU freed | host can run more workers or lower power |

The MR kernel itself is unchanged — this is a **data-flow** win (fewer
candidates reach the already-optimal kernel), not a kernel win.

---

## 9. Risks / open questions

- **Stream contention** on the 8-worker → device round-robin: each worker has
  its own `gpu_fermat_ctx`/`gpu_sieve_ctx` on its device. The persistent
  buffers must be per-(worker,device), or shared with a lock-free allocator.
- **D2H of offsets** adds ~8 KB/window back to PCIe; unavoidable for host
  gap detection. Could be avoided later by doing gap detection on-device too
  (future work).
- **Bitmap pool sizing**: P must cover (sieve in flight + MR in flight) ×
  workers per device; 8–16 per device is ample.
