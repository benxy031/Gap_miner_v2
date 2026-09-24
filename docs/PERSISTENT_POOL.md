# Persistent window pool for the mining chain — design + measurements

Status: **design + evidence (2026-09-24). Implementation NOT started.**

## The measurement that motivates it

Production (3 workers, FUSED_GPU=1, shift507_p74_lex_m30) delivers **1.05M MR
tests/s** (73.76M candidates / 70.4 s). CGBN's standalone bench curve on this box
(`bin/bench_fermat 12 BATCH 10`, AL=12/768-bit):

| candidates in flight | tests/s |
|---|---|
| 1,200 | 593,284 |
| 2,400 | 872,813 |
| 4,800 | 1,100,330 |
| 9,600 | 1,200,138 |
| 19,200 | 1,365,678 |
| 40,000 | 1,425,686 |

The curve has **no plateau** — the kernel is latency-bound, so throughput is set by
how many candidates are in flight. Production's mean MR batch is **2,407**
(`tests/window 78.0 / rounds/window 0.0324`, both printed by the existing
`FUSED_STAGE_TIMING` + worker counters), i.e. ~7.2k in flight across 3 workers,
and 1.05M sits exactly on the curve for that number (4,800 -> 1.10M,
9,600 -> 1.20M). **Nothing is lost to GPU idle or launch overhead.**

So the whole remaining lever is the in-flight count, and its ceiling on this
card is **+36%** (1.05M -> 1.43M).

## Why the count is what it is

In `crt_fused_chain_flight` (`new_src/worker_gpu.c`) one round submits
`rtotal = SUM over active windows of (chi[i] - clo[i])` — with the chunk default
`C=12` that is **12 x (active windows)**.

* a window lives in the flight from its first slice until its walk terminates:
  mean ~6.5 slices, but the *slowest* window needs up to ~67 (a long prime desert
  in FIND_END) -> the flight runs ~67 rounds;
* therefore the mean number of ACTIVE windows is only ~199 of 2048
  (6.5 / 67 x K), and the batch decays 24,576 -> ~200 over the flight's life.

Two knobs were already measured against this, and BOTH are dead ends:

* **bigger K** (2048 -> 4096): flat (12,398 -> 11,963 win/s). More windows make
  the *tail* longer, which cancels the larger first round.
* **bigger chunk C** (12 -> 24): tests/window +29% for tests/s +22% -> net
  -5.9%. A larger slice overshoots the first-prime search by up to C per slice,
  and the waste is proportional to C (`~6.5 x C/2` extra tests per window), so
  the batch gain and the waste cancel. **C=12 is the measured optimum.**
* the 2026-09-19 **round-admission pool** was the wrong direction: it *throttled*
  the number of windows advanced per round (P=128..448 < K), which shrinks the
  batch below the natural first round — hence the monotone regression. Its
  recorded "mechanism" (a pool extends the tail) describes that throttled
  version only.

## The design: a refilled pool instead of a flight

Keep the round barrier (it is what makes batching possible) and keep the CUDA
API unchanged (`gpu_fermat_gather_run` / `submit_device` / `collect`). Change
what a round's active set is:

1. **Window slots**, not a flight: `K` slots whose lifetime is independent. A
   slot holds {nonce, base, survivor list, phase, clo/chi, counters}.
2. **Refill on retire**: when a slot's phase becomes `MJ2_DONE`, its work is
   gap-scanned (as today, per window) and the slot is immediately refilled from a
   **queue of already-marked windows**.
3. **Continuous supply**: the mark+extract pipeline keeps the queue non-empty
   (it already exists for the 2-flight ping-pong; it must run *while* the chain
   runs, not before).
4. Steady state: every round carries `12 x K` candidates (24,576 at K=2048)
   instead of decaying to ~200, so the kernel runs at the 24k-batch point of the
   curve -> **+36% ceiling**, realistically +20-30% after the extra bookkeeping.

Speculative deepening is NOT part of this design (it trades batch for wasted
tests, the same trade the C knob already measured as break-even).

## Implementation stages

* **Stage 1 — two concurrent flights per worker** (`MINING_JUMP2_FLIGHTS=2`,
  env, default 1). Round-robins the round loop between two flights, each with its
  own K windows / candidate buffer / chain state. Keeps ~2 x 2.4k = 4.8k in
  flight -> bench says **+26%** (873k -> 1.10M). Smallest change that tests the
  mechanism; cost is 2x candidate VRAM (12 GB fleet cards yes, 8 GB dev card only
  at reduced K).
* **Stage 2 — the refilled pool** (steps 1-4 above). Needs the queue + on-the-fly
  mark/extract scheduling; this is the full +36% design.

## Hazards (from this repo's own history — read before coding)

* **Stale per-slot state**: the jump2 false-gap bug (2026-09-04) was exactly this
  — per-flight accumulators not reset per fill. A refilled slot MUST reset every
  field (`jump_n`, `pidx`, `qidx`, `anchors`, `wtests`, `phase`, `clo/chi`) or
  the reporter will re-emit stale pairs against a later window's base.
* **Batch dispatch when a batch is partially filled**: the 2026-09-25
  "fused tail false gap" bug — data laid out window-major at append time must
  never be rewritten at different positions later; splice tails via scratch.
* **Never truncate a candidate**: the candidate buffer overflow guard must
  fail closed (loud), never `continue` silently (that created false gaps).
* **`any_active` invariant**: unstarted windows must keep `any_active=1` or the
  round loop exits and silently drops those windows' gaps (learned in the pool
  experiment).
* Verification is non-negotiable: `MINING_JUMP2_VERIFY=1` parity vs a full scan
  (`bad_windows=0 bad_pairs=0`), plus `bin/test_gap_hunt` on a hunt output, plus
  the yield counters compared at a fixed threshold.

## Acceptance criteria

1. Parity clean at the new default (verify flights, not just win/s).
2. `tests/window` unchanged within 1% (the pool must NOT raise the test cost —
   that is what separates it from the C knob).
3. Tests/s up; the honest expectation is the bench curve for the new sustained
   in-flight count.
4. No new `candidates`/OOM/fallback messages; fail-closed to the CPU path if the
   queue starves.
