# GapMiner V2 — Gapcoin Prime-Gap Miner

GapMiner V2 mines Gapcoin blocks by searching for **prime gaps**. Gapcoin's
proof-of-work is: find two consecutive primes `p1 < p2` whose gap

```
merit = (p2 - p1) / ln(p1)
```

meets the live network difficulty, then submit that gap as the block.

## How it works

Every candidate window runs the same four-stage pipeline:

```
sieve → probable-prime filter → gap/merit filter → BPSW verify → submit
```

1. **Sieve** — a segmented sieve removes numbers with small prime factors
   (64-bit bitmap, AVX2-marking/extraction on x86-64).
2. **Probable-prime filter** — the CPU Euler test by default, or — with
   `--enable-gpu-fermat` — a CUDA base-2 Miller-Rabin test that *replaces* it
   (a real prime never fails, so no gap is missed).
3. **Gap/merit filter** — among probable primes, find consecutive pairs whose
   gap merit clears the live node difficulty (or the `--merit` override).
4. **BPSW verify** — only gaps above the threshold get the full Baillie–PSW
   check. By default the result is logged (`dry-run`); `--enable-submission`
   submits it as a real `submitblock` RPC call.

## Two search modes

- **Non-CRT** (default) — workers scan fixed windows of adders materialized
  from the live block template. `--shift` sets the candidate width
  (`256 + shift` bits); `--threads` runs N parallel workers.
- **CRT covering** (`--crt-file <file>`) — a precomputed covering system makes
  the whole interior of `[1, gap_target)` composite, so the scan window is
  prime-poor and the gap tail stretches ~1.27× (`P(gap ≥ m) ≈ exp(-m/1.27)`
  instead of Cramér `exp(-m)`). This is the fast path for finding
  merit-qualified gaps. Prebuilt merit-23 files ship in `data/crt/m23/`; batch
  generate your own with `scripts/gen_crt_batch.sh` (see `gen_crt.md`).

## Quick start

```bash
# 1. Install dependencies
sudo apt-get install -y libgmp-dev libcurl4-openssl-dev libjansson-dev libssl-dev \
                        build-essential pkg-config

# 2. Build (CPU only)
cd /home/dejan/Git/gapminer_v2 && make clean && make

# 3. Run (dry-run against a local Gapcoin node, shift 26, 4 threads)
./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
  --threads 4 --shift 26
```

Nothing is submitted unless you add `--enable-submission` (with your payout
script):

```bash
./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
  --threads 4 --shift 26 \
  --enable-submission --coinbase-script-hex 76a914<20-byte-hash160>88ac
```

## GPU acceleration (CUDA)

Optional. Build with the fast [CGBN](https://github.com/NVlabs/CGBN) kernel
(auto-cloned into `tools/cgbn/` on first build):

```bash
make clean && make WITH_CUDA=1 WITH_CGBN_FERMAT=1
# verify the kernel against CPU/GMP ground truth (must show 0 mismatches):
./bin/test_gpu_fermat
```

Then enable it in the live pipeline with `--enable-gpu-fermat`; the CUDA
base-2 Miller-Rabin test becomes the primality filter and the CPU Euler test is
skipped. Candidates are batched across windows before each GPU call to amortize
the kernel launch (a single window is far too small).

The active limb count is rounded **up to the nearest CGBN-supported width**
(`2, 4, 6, 8, 12, 16, 20` limbs) so widths without a valid CGBN instantiation
— e.g. 514-bit candidates (AL=10: TPI=4 needs 5 limbs/thread, TPI=8 needs a
non-integer 2.5) — are zero-padded to the next CGBN width instead of falling
back to the register-heavy scalar kernel (measured on shift258: 610 → 2786
win/s, ~4.6×).

```bash
# GPU-accelerated dry-run
./bin/gapminer --threads 1 --shift 26 --enable-gpu-fermat
```

Two environment variables control an *experimental* GPU bitmap sieve (the
residue-marking step also moves to the GPU). It is **off by default** and is
only worth trying on multi-GPU hosts:

```bash
GPU_SIEVE=1 GPU_SIEVE_BATCH=16 ./bin/gapminer --threads 1 --shift 26 --enable-gpu-fermat
```

A third variable, `FUSED_GPU`, runs the whole CRT window pipeline on the GPU:
residue computation, bitmap marking, survivor extraction/packing and the
Miller-Rabin kernel all stay in device memory (no bitmap/candidate H2D/D2H
round-trips). The pipeline is **asynchronous and double-buffered**: two
windows are in flight (ping-pong bitmaps/candidate buffers and the two
Fermat slots), the MR kernel overlaps the next window's mark+extract, and
extraction is an ordered single-block stream compaction so the host never
sorts `(offset, is_prime)` pairs. It implies `GPU_SIEVE`, is **off by
default**, and falls back to the hybrid sieve + H2D path on any CUDA error:

```bash
FUSED_GPU=1 GPU_SIEVE=1 ./bin/gapminer --crt-file data/crt/m23/shift450_p67_strong_m23.txt --threads 8 --enable-gpu-fermat
```

An experimental `GPU_SIEVE_PAIR=1` variant batch-marks **two** windows with
one kernel (each into its own ping-pong bitmap), halving mark kernel
launches and stream syncs per window. Measured on the dev host (RTX 3070,
8 workers, one GPU): **-58% throughput** (shift258: 1262 vs 3022 win/s) —
the monolithic 2-window kernel starves the small extract/MR kernels at the
GPU scheduler, producing 2-50 ms stalls in `cudaEventSynchronize`. It is
**off by default** and kept as a benchmark-gated experiment (see
`GPU_SIEVE_PAIR` in the environment table).

A fourth variable, `HALF_CLASS`, enables the *two-pass half-class scan*: only
the 8 visible residue classes coprime to 60 (`1,7,11,13,17,19,23,29 mod 60`)
are sieved and primality-tested, which halves the Miller-Rabin candidate
load.  The hidden classes (`31,37,41,43,47,49,53,59 mod 60`) are never
scanned normally; instead, every visible gap that meets the merit threshold
is verified on demand (its interior is mini-sieved in the hidden classes and
MR-tested), so no true qualifying gap can be missed — a hidden endpoint only
enlarges the visible gap, which then still exceeds the threshold.
Verification events are rare (rate ≈ qualifying-gap rate), so the cost is
negligible.  The gap-distribution health histogram is disabled in this mode
(visible gaps are not consecutive-prime gaps), and
`candidates_generated`/`candidates_tested` in the rolling stats count
visible-class candidates only.

In CRT mode `HALF_CLASS` also works: the covering template pre-filters the
on-demand verification (measured on shift509: fused 1647 → 1888 win/s,
+14.6%; CPU-only 20 → 28 win/s, +40%).  The back-lookahead region is scanned
in ALL classes so its primes anchor the prefix chain directly; the terminal
pair (last back prime → first visible prime) is resolved only when its merit
can qualify:

```bash
HALF_CLASS=1 ./bin/gapminer --threads 8 --shift 55 --enable-gpu-fermat
HALF_CLASS=1 FUSED_GPU=1 GPU_SIEVE=1 ./bin/gapminer --crt-file data/crt/m23/shift509_p74_strong_m32.txt --threads 8 --enable-gpu-fermat
```

### CPU primality filter (ported fixed-limb Montgomery path)

The CPU-only primality filter was ported from cpugapminer: `primality_limbs`
implements base-2 Fermat and Euler–Plumb tests over fixed-limb little-endian
CIOS Montgomery arithmetic with an ADX/BMI2 (MULX + ADCX/ADOX) inner loop,
runtime CPUID detection and a portable fallback (`new_src/primality_limbs.{c,h}`,
ported and refactored — dead bucket-layer variants dropped, 4-bit fixed-window
exponentiation, exact-per-limb-count specializations for 2..20 limbs).

Measured on the dev host (GMP 6.x, modern x86-64) GMP's hand-tuned assembly is
faster at every width (311/765/1280-bit), so **GMP stays the default**. The
limb path is opt-in via `GAPMINER_CPU_LIMBS=1` for hosts where GMP's tuned asm
is unavailable or slower (its original cpugapminer target). The path is
verified against GMP by `./bin/test_primality` (limb-vs-GMP cross-check on
random candidates for limb counts 2..12).

```bash
GAPMINER_CPU_LIMBS=1 ./bin/gapminer --crt-file data/crt/m23/shift509_p74_strong_m32.txt --threads 2
```

The RPC thread polls the node for a new block template every **500 ms**
(average new-block detection latency ≈ 0.25 s, ~0.2% of the ~118 s block
interval).

## A/B shift comparison

`scripts/ab_shift_compare.sh` runs two CRT shifts back-to-back (default:
`shift258_p43_strong_m40.txt` vs `shift998_p128_strong_m40.txt` — same
covering generation, merit-40 "strong" files), parses the final
`ROLLING STATS` block of each run, and prints `accepted/hour` and
`candidates/hour` for both. Runs are **dry-run by default** — set
`COINBASE_HEX` to your payout script to enable real submissions:

```bash
DURATION=3600 ITERATIONS=2 COINBASE_HEX=76a914<20-byte-hash160>88ac \
  ./scripts/ab_shift_compare.sh
```

Duration is per single-shift run (`3600` s default); iteration order
alternates to cancel difficulty drift. Results land in
`/tmp/ab_shift_results.tsv` (`OUT=`), logs in `/tmp/ab_shift_logs`
(`LOGDIR=`). Treat runs with <5 merit candidates as noisy (Poisson).

## GAP_HUNT — standalone record-hunting mode

`--gap-hunt` decouples gap discovery from Gapcoin's PoW: no headers, no
difficulty, no submissions.  The CRT cover of a design file is periodic with
period `P` (product of the cover primes), so every translate
`b_k = b0 + k·P` of an aligned base carries the identical cover template and
the same σ-conditioned gap distribution the miner exploits.  The walk uses
the fused GPU pipeline (device sieve + CGBN MR, full-class) and reports every
gap whose both endpoints are BPSW-verified and whose merit is at least
`--gap-hunt-min-merit`.  Merit is the **true record merit** `gap / ln(start)`
(the prime-gap community convention, matching external verifiers) — NOT the
miner's nominal `gap / ln(2^(256+shift))` protocol merit.

Windows are `P` apart (not contiguous), so gaps are chained only within a
window; the first prime of each window is skipped for gap measurement
(unknown predecessor).  Windows accumulate into **K=16 async MR batches**
(tunable via `GAP_HUNT_BATCH`, 1..16) with two alternating flights: the host
processes one collected batch while the GPU runs the next flight's MR kernel
(measured on the dev host while mining runs: **842 win/s at K=16 vs 808 at
K=8**; the M1 synchronous baseline was 480).  The walk is single-threaded by
design — `--threads` and the other miner flags are ignored in this mode; GPU
concurrency comes from the K flights.  State is written to
`--gap-hunt-state` every 1024 windows and on `SIGINT`/`SIGTERM`; `k` resumes
from the state file.

```bash
# Requires WITH_CUDA=1 build (the same binary as the miner)
./bin/gapminer --gap-hunt \
    --crt-file data/crt/m23/shift507_p74_lex_m30.txt \
    --gap-hunt-min-merit 15 \
    --gap-hunt-state data/gap_hunt_state.txt \
    --gap-hunt-out data/gap_hunt_records.txt
```

Each out record is `<gap> <merit> <startprime>` (one per line; merit printed
with 6 decimals, the record-submission precision).  Validate with
`bin/test_gap_hunt <out-file>` (checks `nextprime(start) == start + gap` for
every record).

## CLI reference

| Option | Default | Description |
|---|---|---|
| `--host <addr>` | `127.0.0.1` | Gapcoin node RPC host |
| `--port <port>` | `31397` | Gapcoin node RPC port |
| `--user <name>` | `benxy031` | RPC username |
| `--pass <pass>` | `xx` | RPC password |
| `--threads <n>` | `1` | Worker threads |
| `--shift <v>` | `26` | Non-CRT shift (`20..1024`); candidate = `256+shift` bits |
| `--crt-file <path>` | none | Enable CRT covering mode; the file's shift overrides `--shift` |
| `--sieve-primes <n>` | `50000` | Small-prime sieve limit. Non-CRT + `--enable-gpu-fermat`: adaptive bit-scaled default (`log2(depth)` interpolated between 282-bit → window+halo cover and 311-bit → 20M, clamped to `[cover, 20M]`; measured +14% win/s at shift 55, deeper than 20M makes the CPU sieve the bottleneck). With `HALF_CLASS` the 311-bit anchor drops to 5M (measured peak at shift 55: 5M = 609 win/s vs 20M = 391). CRT mode: `10000000` on CPU, `100000` on GPU, `2000000` on the fused GPU path (`FUSED_GPU=1`; measured on the production host at shift475 live merit: 500K=3183, 1M=3197, 2M=3324, 5M=3205 win/s — 2M is the optimum; older dev-host runs: shift258 3106 win/s at 1M vs 2814 at 5M, shift509 1971 vs 1905) |
| `--merit <v>` | node difficulty | Merit threshold override (lower = more BPSW work) |
| `--enable-submission` | off | Submit BPSW-verified gaps via `submitblock` |
| `--coinbase-script-hex <hex>` | none (`OP_TRUE`) | Payout scriptPubKey for submitted blocks |
| `--enable-gpu-fermat` | off | Use the CUDA base-2 MR kernel as the primality filter (requires `WITH_CUDA=1`; falls back to CPU on failure) |
| `--record-log <path>` | `gapminer_records.log` | Log every BPSW-verified candidate with full parameters |
| `--merit-records <path>` | `data/prime_gap_merits.txt` | Best-known-merit table used to flag `new_record=yes` |
| `--gap-hunt` | off | Standalone record-hunting walk (requires `--crt-file` and a `WITH_CUDA=1` build; runs the walk and exits instead of starting the miner) |
| `--gap-hunt-start <hex>` | `2^(255+shift)` | Base anchor for the walk (hex); default follows the CRT file's shift; CRT-aligned internally |
| `--gap-hunt-min-merit <m>` | `15` | Report gaps with merit ≥ m (true record merit `gap/ln(start)`) |
| `--gap-hunt-state <path>` | none | Resume state file (k and diagnostic last prime) |
| `--gap-hunt-out <path>` | none | Results file (`<gap> <merit> <startprime>` per line; stdout-only if unset) |
| `--help` | — | Print the help message |

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `GPU_SIEVE` | off | Experimental GPU bitmap sieve (`1` enables; multi-GPU only) |
| `GPU_SIEVE_BATCH` | `1024` | Windows per GPU bitmap-sieve batch (`1..4096`, autotuned) |
| `FUSED_GPU` | off | Full GPU-resident CRT pipeline (sieve+extract+MR on-device; implies `GPU_SIEVE`, defaults to a 2M deep sieve) |
| `GPU_MR_BATCH` | `8` | Windows per accumulated MR batch on the fused path (`1..8`; `1` = per-window, the old behavior). One `gpu_fermat_submit_device` per K windows instead of K small-batch launches. Measured on the dev host (RTX 3070, 8 workers, 20s runs, 0 failures): shift258 K=1 3047 → K=8 **5715 (+88%)**; shift509 K=1 1971 → K=8 **3211 (+63%)**. K=8 vs K=4 (all wins, no regressions): +6.5% (258), +2.7% (450), +4.5% (509), +1.4% (657), +0.7% (720), +1.9% (1008). Candidate counts per window are identical across K (verified per shift). Memory note: K=8 sizes the device candidate buffers at 8 windows; on ≤4 GB cards with 8 workers at shifts ≥ 1008 this may OOM and fail-closed to the CPU path |
| `GPU_SIEVE_PAIR` | off | Experimental 2-window pair-batched fused mark (one kernel writes both ping-pong bitmaps). Measured **-58%** win/s on the dev host (8 workers / 1 GPU, shift258: 1262 vs 3022); the monolithic kernel starves extract/MR kernels at the GPU scheduler. Benchmark-gated — do not enable in production |
| `HALF_CLASS` | off | Two-pass scan: sieve/test only residues `{1,7,11,13,17,19,23,29} mod 60`, verify the hidden classes on demand (~2× fewer MR candidates; non-CRT +87% at shift 55, CRT fused +14.6% at shift509); disables the gap-dist health histogram. In CRT mode the covering template pre-filters the verification and the back-lookahead stays unfiltered |
| `QUARTER_CLASS` | off | Generalizes `HALF_CLASS` to 4 visible / 12 hidden coprime classes (`{1,7,11,13} mod 60` visible). The containment lemma (every true qualifying gap is contained in a visible qualifying gap) guarantees no blocks are lost while the GPU MR load halves. Hidden resolution runs on the GPU MR pipeline (base-2+3 batch + BPSW only on MR survivors; falls back to the CPU path without CUDA), and the fused head is extended ~12·logbase in this mode to keep tail re-marks rare. Measured on the dev host (RTX 3070, fused path, live difficulty): **3515 win/s vs 3176 for `HALF_CLASS` (+10.7%)**, tails skipped 99.4%, GPU-bound (acc/wall 2.8). Parity-exactness unit tests green; production A/B on the dual-3060 box pending |
| `GAPMINER_CPU_LIMBS` | off | Use the ported fixed-limb Montgomery Fermat/Euler path instead of GMP `mpz_powm` for the CPU-only Euler filter (slower than GMP 6 on the dev host; opt-in for hosts without GMP's tuned asm) |
| `GAPMINER_CPU_WINDOW_OVERRIDE` | `4` | Force the CPU limb path's exponentiation window width (`3`, `4` or `5`; `4` is the specialized default) |
| `GAPMINER_CPU_WINDOW_LOG` | off | Log the selected window width once per limb count (diagnostic) |
| `GAPDEBUG` | off | CRT gap diagnostics (HALF_CLASS and full-class modes): for every emitted gap, log to stderr the window class, gap class endpoints, and the interior candidates with their MR flags; in HALF_CLASS mode also logs `[HIDDBG]` hidden-class resolution counters (candidates tested / primes found per resolved interval). Used to trace false-gap regressions; verbose — development only |
| `GAP_HUNT_BATCH` | `16` | GAP_HUNT windows per accumulated MR batch (`1..16`). Measured on the dev host (RTX 3070, mining running): 842 win/s at 16 vs 808 at 8 vs 778 at K=8+1M sieve — K=16 is the default optimum; batches are guarded against silent truncation at the 40000-candidate MR limit (fail-closed) |

## Testing

```bash
make clean && make test      # all suites
./bin/test_sieve_core        # scalar vs AVX2 sieve equivalence
./bin/test_gpu_fermat        # GPU kernel vs GMP ground truth + device-pointer path (WITH_CUDA build)
./bin/test_gpu_sieve         # fused extract+pack kernel parity vs CPU sieve (WITH_CUDA build)
./bin/test_halfclass         # HALF_CLASS two-pass pipeline parity vs full-class pipeline
```

## Rolling stats (`acc/wall`)

Every 30 seconds the miner prints a `ROLLING STATS` block. With GPU Miller-Rabin
active (`--enable-gpu-fermat`), the `GPU MR acc/wall` line reports the fraction
of wall-clock time the GPU actually spent executing the MR kernel:

```
GPU MR acc/wall: 3.792 (113.780 s GPU-accounted of 30.0 s wall) [<1 host-bound, >1 GPU-bound]
```

- **`acc` (GPU-accounted)** — the sum of pure MR-kernel execution time over all
  workers, measured with CUDA events around each batch (`cudaEventElapsedTime`).
  It excludes host launch/sync gaps, H2D/D2H copies, and idle time.
- **`wall`** — the elapsed wall-clock interval of the stats window (≈30 s).

Because the workers share one GPU, their kernels serialize on the device, so the
summed `acc` equals the GPU's total busy time. Interpreting the ratio:

| `acc/wall` | Meaning |
|---|---|
| `< 1.0` | Host-bound: the GPU is idle part of the time (launch/sync gaps or CPU work starving it). This is the invisible host-contention signal the metric exists to surface. |
| `≈ 1.0` | Balanced: the GPU is nearly fully busy. |
| `> 1.0` | GPU-bound: workers demand more GPU time than wall clock provides (the healthy, saturated mining case). |

The metric is only meaningful with GPU MR active and stays `0.000` in pure-CPU
runs or when the GPU path is disabled.

In `HALF_CLASS` mode the `Max Euler pair` and `Merit candidates` lines reflect
**true** consecutive-prime gaps (resolved by the on-demand hidden-class
verification), never the raw visible-class gaps: a visible gap sums several
true gaps between hidden-class primes, and reporting it would print a bogus
record-looking merit (e.g. `merit=36` while `Merit candidates` stays 0).
`Max Euler pair` is `gap=0` in intervals where no visible candidate exceeded
the merit threshold (every true gap was then below it).

### Gap distribution health check

The same `ROLLING STATS` block also carries a running **correctness alarm**:

```
Gap-dist health: 312456 gaps | dev g=4:+0.4% g=6:-1.1% g=8:+0.9% g=10:-0.3% g=12:+1.2% [OK]
```

Every consecutive probable-prime pair found by the scanner feeds a global gap
histogram. The frequencies of the smallest gaps (`g = 4, 6, 8, 10, 12`, relative
to `g = 2`) are compared with the Hardy–Littlewood k-tuple asymptotic model

$$\frac{P(g)}{P(2)} = \prod_{\substack{p\ \text{odd prime}\\ p \mid g}} \frac{p-1}{p-2}\;\cdot\;e^{-(g-2)/\ln n}.$$

In CRT mode only *honest* gaps are used: both endpoints must lie outside the
covering's region (the covering conditions the gap distribution inside it, so
covered and boundary-crossing pairs are skipped). Even so, the covering's
periodic structure modulates the small-gap frequencies by up to ~30% in
uncovered regions (empirically measured, file-dependent), so the warning
threshold is **20%** in non-CRT mode and **50%** in CRT mode. Once 50 000
gaps and 1 000 gap-2 pairs have accumulated, a small-gap deviation beyond the
threshold prints a warning:

```
⚠ GAP-DIST DEVIATION: g=6 deviates -51.2% from Hardy-Littlewood (|dev| > 20%) — possible sieve/primality bug
```

That signals a real bug (composites not marked, primes falsely rejected, or
candidate offsets corrupted) — not a performance issue. The check adds one
relaxed atomic increment per honest gap pair and has no measurable mining
overhead.

## Project structure

```
new_src/          Core miner (sieve_core, worker_gpu, miner_farm, crt_*, …)
new_src/gpu/      CUDA kernels (gpu_fermat.cu, gpu_sieve.cu)
tests/            Unit/integration tests
data/crt/m23/     Prebuilt merit-23 CRT covering files (shift 450..1017)
data/prime_gap_merits.txt  Best-known-merit reference table
scripts/          gen_crt_batch.sh, update_merits.sh, ab_shift_compare.sh
gen_crt.md        CRT covering-file generator guide
docs/             Architecture references
```

## License

GPL-3.0-or-later. See `LICENSE.md`.
