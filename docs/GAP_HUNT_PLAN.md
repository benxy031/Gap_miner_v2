# GAP_HUNT — standalone record-hunting mode (plan)

Status: M1+M2 IMPLEMENTED and validated (2026-09-01). M3 (A/B vs prime-gap) pending.

## 1. Goal

Add a Gapcoin-independent record-hunting mode to GapMiner V2, focused on the
measured best cover (`shift507_p74_lex_m30`), that must beat
`/home/dejan/Git/prime-gap` (classic `m*P#/d` combined_sieve + gap_test flow)
in gaps-found-per-hour above a merit threshold.

The mode is a **separate CLI flag** (`--gap-hunt`) — the PoW pipeline stays
untouched. Existing engines (fused GPU sieve, CGBN MR, CRT runtime) are reused;
new code lives in dedicated files.

## 2. Core concept — the CRT-periodic walk (the new part)

Gapcoin structure (base = h256 << shift) exists only to satisfy PoW. Without
the protocol we are free to pick the bases ourselves:

- The m30 cover is **periodic with period P** (product of the 74 cover primes,
  P ~ 2^538). For ANY base `b0 ≡ -o_i (mod p_i)` (one CRT alignment), every
  translate `b_k = b0 + k*P` carries the **identical** cover template:
  candidates = `b_k + survivor_offsets`, same σ-conditioned gap distribution.
- **CRT walk**: align once, then step `k = 0, 1, 2, ...`. Each step is an
  independent 763-bit sample of the σ≈1.38-tail distribution (measured on
  today's quarter+fused pipeline: m30 dominates every threshold).

This is exactly the per-header mining loop **minus**: SHA256, GBT polling,
CRT alignment per window (done once), difficulty filtering, owned-window
limits, submission, stale handling. All of that overhead disappears.

## 3. Why it should beat prime-gap

| axis | prime-gap (m*P#/d) | GAP_HUNT (this plan) |
|---|---|---|
| cover | single primorial center | lex-optimized 74-prime cover, measured tail σ≈1.38, 26× prime enrichment |
| sieve | CPU `modulo_search_*` (80%+ of time) | device-resident fused GPU sieve (2M primes) |
| primality | PRP only on top-% of stats | CGBN 12-limb MR on **100%** of survivors |
| gap capture | stats-selected subset | **every** true gap ≥ threshold (no selection loss) |
| boundaries | interval-by-interval | cross-window chaining (reuses back_limit machinery) |

Measured production throughput (quarter, m30, one 3070): ~0.108 gaps ≥ 16-merit
**per second** already inside the PoW harness. The walk mode removes per-header
overhead and adds full-class scanning (no hidden-class split needed for
records), so the same metric should improve further; the acceptance criterion
is a direct A/B against prime-gap on this machine.

## 4. Flags & files

New CLI flags (main.c):

- `--gap-hunt`                enable the standalone mode (mutually exclusive
                              with live mining/submission)
- `--gap-hunt-start HEX`      base anchor `b0` (hex; default: 2^762, CRT-aligned
                              automatically)
- `--gap-hunt-min-merit M`    log/keep gaps with merit >= M (default 15.0)
- `--gap-hunt-state FILE`     resume state (k position + last chain prime)
- `--gap-hunt-out FILE`       results file, prime-gap-compatible columns
                              (`start gap merit`), plus our record_log

Reused (no changes): `--crt-file data/crt/m23/shift507_p74_lex_m30.txt`,
`FUSED_GPU=1 GPU_SIEVE=1`, `--threads`, `--sieve-primes`.

New files:

- `new_src/gap_hunt.c` / `gap_hunt.h`  — walk loop, state, results
- `tests/test_gap_hunt.c`              — exactness + resume tests
- (no new data files; the m30 CRT file is the configuration)

## 5. Architecture per window (reuses the fused pipeline 1:1)

1. `b_k = b0 + k*P` (one GMP add per window — replaces SHA256+CRT-align).
2. `window_base = b_k - back_limit`; GPU mark (2M primes), extract survivors
   **full-class** (mask = all classes — records need no visible/hidden split,
   and exactness is then trivial: every prime is directly MR-tested).
3. CGBN MR batch on survivors (K-window accumulation as today).
4. Chain consecutive primes across window boundaries (carry the last prime;
   the existing back-lookahead scan already proves this boundary logic).
5. For every gap with merit >= M: BPSW both endpoints, write result + state.
   No owned-window filter, no hidden resolution.

## 6. Validation (before any claim of "better")

1. **Exactness**: for every reported gap, `p2 == nextprime(p1)` via GMP;
   unit tests with synthetic windows + a 10-minute live run audited against
   `gap_test.py` from prime-gap on the same interval.
2. **A/B**: same machine, same hours: gaps ≥ 16 / ≥ 18 / ≥ 20 merit per hour —
   GAP_HUNT vs `combined_sieve --save` + `gap_stats` + `gap_test.py`. Win
   condition: higher merit-rate at every threshold with equal wall time.
3. **Resume**: kill/restart mid-walk reproduces identical gap stream (state
   file round-trip test).
4. **No interference**: PoW mode binaries/config unchanged; `--gap-hunt` is
   additive (CI builds green on WITH_CUDA and non-CUDA).

## 7. Risks

- Boundary-chain bugs → mitigated by reusing the proven back_limit design +
  explicit nextprime audit in validation.
- BPSW load on huge gap counts at low M → cap with `--gap-hunt-min-merit`
  (BPSW only on qualifying endpoints).
- Nothing here changes σ itself — the win is structural (full coverage, no PoW
  overhead), not a new mathematical constant. Expected gain vs prime-gap is
  throughput/coverage-driven, to be measured in §6.2, not assumed.

## 8. Milestones

1. M1: `gap_hunt.c` walk + full-class fused pipeline + results file
   (reuse: mark/extract/MR from worker_gpu.c; ~1 day).
2. M2: exactness tests + resume state (tests/test_gap_hunt.c).
3. M3: A/B vs prime-gap (24h) — go/no-go on the "better" claim.
4. M4: optional prime-gap DB interop (`--gap-hunt-out` format + RECORD checks).

## 9. Status log

### 2026-09-01 — M1 + M2 DONE (validated)

Implemented `new_src/gap_hunt.{h,c}`, CLI flags (`--gap-hunt`,
`--gap-hunt-start`, `--gap-hunt-min-merit`, `--gap-hunt-state`,
`--gap-hunt-out`), Makefile wiring, `bin/test_gap_hunt` validator, README
section.  Validated on the dev host (RTX 3070, shift507_p74_lex_m30):

- ~480 windows/s single-threaded synchronous pipeline; 319 exact gaps ≥ 8
  merit in 10 s; best merit 16.25 in 5 s, 18.25 in 26 s.
- `test_gap_hunt` exactness: `mpz_nextprime(start) == start + gap` for every
  record, plus BPSW on both endpoints — **319/319 PASS**.
- Resume round-trip verified (k resumes from state; state saved every 1024
  windows and on clean shutdown).

Two lessons recorded (see AGENTS-adjacent memory):

1. **Cross-window chaining is invalid.** Consecutive windows are `P ≈ 2^501`
   apart (not contiguous), so chaining the previous window's last prime into
   the next window produced bogus "gap ≈ P" records (merit ≈ 2^491).  Fixed
   by resetting the chain at each window start; the first prime of each
   window is skipped for gap measurement (unknown predecessor).  Boundary
   loss is 1 prime per ~1300 — negligible.
2. **`signal()` is one-shot on glibc (SA_RESETHAND).** `timeout(1)` sends the
   signal to the child AND to the process group; the second delivery killed
   the process with the default action before the loop could save state.
   Fixed with persistent `sigaction(SA_RESTART)` handlers.  Clean shutdown
   now saves final state under `timeout -s TERM`.

Next: M3 A/B vs prime-gap (24h, acceptance criteria in §6.2).

### 2026-09-01 (later) — M1.5 DONE (K-window async MR batches)

Replaced the synchronous per-window pipeline with K=8 accumulated MR batches:
windows pack contiguously into the slot's device candidate buffer
(`gpu_sieve_extract_pack_device_range_ex` with `slot_base` prefix sums) and
one `gpu_fermat_submit_device` per flight; two flights alternate so host gap
scanning overlaps the other flight's MR kernel.  Out format changed to
`<gap> <merit> <startprime>` (k lives in state + progress lines; validator
updated).  Measured on the dev host (RTX 3070, shared with live mining):
**802 win/s vs 480 synchronous (+67%)**; 10 s run: 501 gaps ≥ 8 merit,
`test_gap_hunt` 501/501 exact (nextprime + BPSW); clean shutdown under
`timeout -s TERM` with both flights drained.

### 2026-09-01 (evening) — K=32 batching; QUARTER_CLASS falsified

- K=32 (GAP_HUNT_BATCH, max raised to 32 with the MR limit at 80000
  candidates, fail-closed guard): **893 win/s vs 706 at K=16 (+27%)** on the
  dev host while mining runs; default.
- QUARTER_CLASS (4-visible-class scan + on-demand hidden resolution) was
  implemented and is **exact** (parity test: identical records to full-class
  on the same k-range, after three real bugs were found and fixed: extract
  `base_mod60=0`, template prefilter assuming an even base while b_k
  alternates parity, and all-class extension band for containers).  But it is
  **falsified for record hunting**: visible gaps inherit the σ-tail (mean
  merit ≈ 8), so the containment trigger fires ~1.3×/window at merit 15 and
  CPU resolution (~40 ms, mini-sieve 10k + GMP MR) costs more than the MR
  savings — 46 win/s vs 893.  Kept behind `GAP_HUNT_QUARTER` (off) with the
  parity evidence; a GPU-resolution variant would need the trigger to be
  ~300× cheaper, which is not reachable.

### 2026-09-01 (later) — true record merit

GAP_HUNT now reports the **true record merit** `gap / ln(start)` instead of
the miner's nominal protocol merit `gap / ln(2^(256+shift))` (they differ by
~0.13% at shift507 — an external verifier flagged gap 11182 as
21.1709 vs our 21.1431; gap and endpoints were correct).  The validator now
also re-derives merit from each record's start and fails on any mismatch.

### 2026-09-02 — shift 1017 CRT support (MR batch limit 80k -> 160k)

First GAP_HUNT run at shift > 507 exposed the 80k MR batch cap: the 1017
window is ~65% longer (interval 78,215 vs 48,521), giving ~2,675
survivors/window x 32 = ~85,600 > 80,000, so the fail-closed guard rejected
the whole batch at i=29 (`cum=77586 + nc=2619 > 80000`).  Fixed by raising
`GPU_ADAPTER_MAX_BATCH` to 160,000 (uint8 result buffers: +80 KB/slot, no
mining impact).  Smoke test: CGBN kernel active at AL=20 (1280-bit) and 96
windows processed cleanly on shift1017_p130_lex_m30.  Note: the Makefile has
no header dependencies — a header-only change needs `make clean` before the
rebuild (stale `gpu_adapter.o` masked the fix initially).  For shift >=
~2000 use `GAP_HUNT_BATCH=16` (per-window survivors scale with logbase).

### 2026-09-02 (later) — double-overflow merit bug: shift >= ~770 lost ALL gaps

`mpz_get_d` overflows to +inf for values > 2^1024, so `merit =
gap / log(mpz_get_d(start))` silently became 0 at shift 1017 (1273-bit
starts): every gap fell at the threshold gate — 0 gaps over 468k windows at
merit 20, and 0 over 8,608 windows even at merit 8 (expected ~570).  Fixed
with an overflow-safe `gh_log_mpz` (mantissa + exponent via mpz_get_d_2exp).
Same class found and fixed in `gap_candidate.c` (density heuristic, no
readers yet) and in `test_gap_hunt.c` (the validator's own merit arithmetic
— it flagged all 837 post-fix records as failures).  The MINING scan
(`gap_detection.c`) already had `mpz_ln` + a 1254-bit regression test from
an earlier session, so mining was NOT losing gaps; record_log uses the
`_big` (full-decimal) variants in the CRT path.  `test_crt_runtime` now
accepts file arguments and loads every fleet CRT file (139/139 OK).
Validation after fix: 60s at shift1017/min-merit 8 -> 837 gaps, best
15.58; 837/837 exact (nextprime + merit arithmetic).
