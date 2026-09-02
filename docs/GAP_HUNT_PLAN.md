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

### 2026-09-02 (evening) — external review: briankehrig/prime-gaps-cuda

Studied Kehrig's exhaustive gap hunter (2^64..2^78, driver of the search that
found maximal gaps 1676/1724/1854).  Key transferable idea: his jump-by-
minGap + backward-search gap finder tests ~1.5 survivors per confirmed prime
instead of all survivors — for GAP_HUNT at shift1017 that is ~70 CGBN tests
per window instead of 2,675 (~38x less MR work); windows are independent so
one CGBN thread-block per window running the serial chain in parallel across
windows is the natural mapping (candidate next milestone, needs a parity
test).  Dormant idea: fixed-high-limb Montgomery (his HIGH_64 magic/
derivative trick) — all candidates in a window share the top ~19 limbs.
Not transferable: pseudoprime sieving (we BPSW + make no completeness
claim), 1.4 GB wheel bitmaps, exhaustive labeling.  Confirms our choices:
skip-half = QUARTER_CLASS economics (viable there, falsified here), A/B
flight overlap, process-per-GPU fleets, FIRST_KNOWN_OCCURRENCE claims.

Follow-up applicability verdict: jump-strategy -> GAP_HUNT only (windows are
independent, parallel across SMs, MR dominates at high shifts).  NOT for
gap_miner: one window per nonce => latency-bound serial chain, MR is not the
dominant cost, and the jump misses small gaps that RGM/gap_dist accumulate
in stride=1 mode.  For mining the transferable candidate would be
fixed-high-limb Montgomery instead.

### 2026-09-02 (night) — jump-strategy IMPLEMENTED and FALSIFIED for throughput

Implemented the Kehrig-style walk for GAP_HUNT behind `GAP_HUNT_JUMP`: the
CGBN Miller-Rabin test was factored into a shared `__device__` function and
a new `cgbn_jump_scan_kernel_t` runs one serial chain per window (one CGBN
block per window, 32 windows in parallel), reporting gaps >= threshold in
offset units; `gpu_fermat_jump_scan()` API + `gpu_sieve_device_offsets()`
accessor.  CORRECTNESS: parity-exact — identical gap sets to the batch path
on shift507 (299/299) and shift1017 (411/411), byte-for-byte after sorting.
THROUGHPUT: 6.3x SLOWER at shift1017 (23.5 vs 147 win/s) — the cooperative
CGBN test has ~10 ms latency per test under load, which 32-way window
parallelism cannot hide (Kehrig's win comes from per-thread 128-bit scalar
tests with ~100x lower latency).  Also diagnostic: batch throughput is the
same at merit 8 and 18 (143 vs 147 win/s), so shift1017 is MARK/EXTRACT-
bound, not MR-bound — the jump was attacking a non-bottleneck.  Reopen
trigger: a per-thread low-latency MR at AL=20, or >500 concurrent windows.

### 2026-09-02 (late night) — bottleneck CORRECTED; deep-sieve default 2M -> 10M

Diagnosis fix: shift1017 is CGBN-**MR**-bound, not mark/extract-bound.
Evidence: 147 win/s x 2,675 survivors = 393k MR tests/s vs CGBN AL=20
capacity scaled from the AL=12 measurement (907k x (12/20)^2 = 326k/s,
within noise).  The m=8-vs-m=18 throughput equality only excludes
BPSW/reporting.  GPU_FERMAT_TPI knob CLOSED at AL=20: TPI=8 (66 win/s) >
16 (44) > 32 (24).

Deep-sieve lever (survivors ~ 1/ln(limit), mark is cheap): 2M -> 10M gives
shift1017 91 -> 154 win/s (+70%; 20M -> 161), shift507 667 -> 711 (+6.5%).
Parity identical (411/411 gaps).  Default changed to 10M in main.c and
gap_hunt.c; README updated.

NOVELTY REGISTER (dormant, this milestone):
- N1 chunk-parallel jump: GAP_HUNT_JUMP reopen refinement — test the
  backward pass in CGBN-instance-parallel chunks (16-way) instead of one
  candidate at a time; chain stays serial but each link costs ONE test
  latency.  Needs a quiet dedicated GPU to measure (dev box is shared with
  mining, which inflates chain latency ~10x).
- N2 factor-first certification: for survivors whose smallest factor is
  < 2^40 (~10-20% of survivor composites), bounded rho/ECM beats one CGBN
  MR; hybrid could shave ~5-15% of MR at comparable kernel cost.  Untested,
  marginal, falsifiable.
- N3 first-ever distribution dataset at 10^229-10^383: at min-merit 8 the
  hunt emits ~800+ exact gaps/hour at sizes with ZERO prior data.  Archiving
  m>=8 results is a by-product nobody else can produce today; costs one
  log file.
