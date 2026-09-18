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

## Recommended commands

All GPU commands assume a `WITH_CUDA=1` build
(`make clean && make WITH_CUDA=1 WITH_CGBN_FERMAT=1 -j4`). `MINING_JUMP2` and
`GPU_MARK_SPLIT` are already on by default; only `FUSED_GPU=1` (which implies
`GPU_SIEVE=1`) has to be asked for. Use **one worker thread per GPU**
(`--threads` = number of GPUs) — the device is assigned round-robin, and each
worker owns its own GPU contexts, so one chain runs per card.

```bash
# ── Mining: fused GPU pipeline + no-test chain (fastest CRT mining mode) ─────
# 1 thread per GPU; needs a CRT cover file; submits only with --enable-submission.
FUSED_GPU=1 ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
  --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat \
  --enable-submission --coinbase-script-hex 76a914<20-byte-hash160>88ac

# Two GPUs: two workers, one chain each (device 0 and 1).
FUSED_GPU=1 ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
  --crt-file data/crt/m23/shift1017_p130_lex_m30.txt --threads 2 --enable-gpu-fermat

# ── Mining: non-CRT (no cover file), high shift + GPU Fermat ────────────────
FUSED_GPU=1 ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
  --shift 96 --threads 8 --enable-gpu-fermat --sieve-primes 7000000

# ── Record hunting (GAP_HUNT): single walker ────────────────────────────────
./bin/gapminer --gap-hunt --crt-file data/crt/m23/shift507_p74_lex_m30.txt \
  --gap-hunt-device 0 --gap-hunt-min-merit 19 \
  --gap-hunt-state data/gap_hunt_state_f1.txt \
  --gap-hunt-out data/gap_hunt_records_f1.txt

# ── Record hunting: fleet (one walker per conf line) ────────────────────────
./scripts/gap_hunt_fleet.sh                              # uses gap_hunt_fleet.conf
CONF=gap_hunt_fleet_dual3060.conf ./scripts/gap_hunt_fleet.sh   # dual-3060 config
# conf line format:  <crt-file> <gpu-device> <min-merit>

# ── Diagnostics / validation (slower; do not leave on in production) ────────
GPU_SIEVE_TIMING=1 FUSED_STAGE_TIMING=1 FUSED_GPU=1 ./bin/gapminer \
  --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat   # stage + kernel split lines
MINING_JUMP2_VERIFY=1 FUSED_GPU=1 ./bin/gapminer \
  --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat   # chain vs full-scan parity
MINING_JUMP2=0 FUSED_GPU=1 ./bin/gapminer \
  --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat   # restore the full scan
GPU_MARK_SPLIT=0 FUSED_GPU=1 ./bin/gapminer \
  --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat   # restore the per-prime mark walk

# ── Benchmarks / analysis tools ─────────────────────────────────────────────
make bin/bench_mark WITH_CUDA=1      # GPU mark-kernel cost model (see docs/MARK_SPLIT_RESULT.md)
./bin/bench_mark --primes 2000000 --rows 8 --window 10175
scripts/ab_shift_compare.sh          # accepted blocks / qualifying gaps per hour, two shifts

# ── Record-rate model: ranking without GPU time (docs/RECORD_RATE_MODEL.md) ─
scripts/record_rate_model.py gap_hunt_records_f1.txt gap_hunt_records_f2.txt
                                     # predicted vs observed records per file;
                                     # the `band` column is the +-5% sigma error
                                     # - a ranking is only real above it (§8)
scripts/record_rate_model.py --targets 6 gap_hunt_records_f2.txt   # nearest record targets
scripts/record_rate_model.py --shift-scan 900,1000,1017,1050       # shift/size alignment
scripts/tail_compare.py data/gap_hunt_records_f1.txt data/gap_hunt_records_f2.txt
                                     # cover A/B tail comparison; the verdict M0
                                     # auto-clamps to the data's own report
                                     # threshold (docs/RECORD_RATE_MODEL.md §8)
scripts/tail_shape.py --selftest    # verify the exp/stretched/GPD estimators first
scripts/tail_shape.py gap_hunt_records_f1.txt --u-fit 16 --u-test 18,20,22
                                     # is the tail exponential at depth? (S9)
scripts/tail_shape.py FILE ... --pool --u-fit 21
                                     # merge same-config files (same L, and fit
                                     # above the highest report threshold)
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
the same σ-conditioned gap distribution the miner exploits.  The walk runs the
**chunk-parallel backward chain** by default (`GAP_HUNT_JUMP2=1` since
2026-09-15; `GAP_HUNT_JUMP2=0` restores the full-scan batch walk, ~2.9–4×
slower) on the fused GPU pipeline (device sieve + CGBN MR) and reports every
gap whose both endpoints are BPSW-verified and whose merit is at least
`--gap-hunt-min-merit`.  Merit is the **true record merit** `gap / ln(start)`
(the prime-gap community convention, matching external verifiers) — NOT the
miner's nominal `gap / ln(2^(256+shift))` protocol merit.

Windows are `P` apart (not contiguous), so gaps are chained only within a
window; the first prime of each window is skipped for gap measurement
(unknown predecessor).  Windows accumulate into **K=64 async MR batches**
(tunable via `GAP_HUNT_BATCH`, 1..512) with two alternating flights: the host
processes one collected batch while the GPU runs the next flight's MR kernel.
The batch is large because a chain round pays a **~4 ms per-LAUNCH floor** of the
CGBN MR kernel (measured with `GAP_HUNT_TIMING=1`), which every window sharing
the round amortizes; going 32 → 64 windows per round measured **+27% at shift507
(1876 → 2386 win/s) and +21% at shift1017 (559 → 677 win/s)**. Cost is staging
memory: the MR staging buffers are sized by `GPU_ADAPTER_MAX_BATCH` (320000
candidates ≈ 200 MB per adapter at 20 limbs), and the walk auto-reduces the
batch with a warning if a cover's windows hold so many survivors that a round
would not fit.  The walk is single-threaded by
design — `--threads` is **inert** in this mode (no worker farm is created and
`gap_hunt.c` uses no threads), as are the other miner flags; GPU concurrency
comes from the K flights and the 2 alternating batches inside one walker.  The
way to add walkers is one PROCESS per GPU (the fleet script): measured at
shift258, min-merit 18 — **1 walker 3858 win/s vs 2 walkers on the SAME GPU
1144 + 1144 = 2288 win/s (−41%)**, because the walk is GPU-bound and the two
processes split the device.  Never run two walkers per card; use one per GPU.
State is written to
`--gap-hunt-state` every 1024 windows and on `SIGINT`/`SIGTERM`; `k` resumes
from the state file.  The same 1024-window tick prints the walk rate:
`[GAP_HUNT] k=… windows=… gaps=… best_merit=… win_s=… win_s_avg=…` —
`win_s` is the instantaneous windows/s since the previous tick and
`win_s_avg` the average since the walk started (the `stopped:` line also
carries `win_s_avg`).

```bash
# Requires WITH_CUDA=1 build (the same binary as the miner)
# The chunk-parallel chain walk is the default (GAP_HUNT_JUMP2=1 since
# 2026-09-15); it is 2.9-4x faster than the full-scan batch walk and finds the
# same records.  Add GAP_HUNT_JUMP2=0 to fall back to the full scan.
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

Automatic record checking while the hunt runs (follows the results file,
compares each gap against `data/prime_gap_merits.txt`, appends new records to
`gap_hunt_records_found.txt` with a versioned comparison snapshot and the
strongest claim label):

```bash
scripts/watch_gap_hunt_records.py data/gap_hunt_records.txt &
```

Record lines carry `claim=FIRST_KNOWN_OCCURRENCE` and
`coverage=known_table_<sha256[:12]>` — the taxonomy labels are defined in
`docs/CLOSED_FINGERPRINTS.md` (the watcher never claims `FIRST_OCCURRENCE`,
`MAXIMAL_GAP`, `MERIT_RECORD`, or `NETWORK_ACCEPTED`, which require coverage
or network evidence it does not have).

Multi-size / multi-GPU fleets: `scripts/gap_hunt_fleet.sh` reads
`gap_hunt_fleet.conf` (one `<crt-file> <device> <min-merit>` line per walker),
launches one walker + one record watcher per line with separate
state/out/log files, and stops all of them losslessly on `SIGINT`/`SIGTERM`
(each walker resumes from its own state file).  The record landscape is
nearly flat across sizes (easiest record merit ≈ 24.35 at shift 1017 vs
26.44 at shift 507), so several sizes in parallel sum independent record
lotteries.  Example config:

```bash
# gap_hunt_fleet.conf
# crt-file                                device  min-merit
data/crt/m23/shift507_p74_lex_m30.txt     0       18
data/crt/m23/shift998_p128_m23.txt        1       18
# coverage-certified p98 walker (952 survivors / 93.88%; cert:
# docs/COVERAGE_CERT.md):
# data/crt/m23/shift720_p98_covermax_m23.txt  0  8
```

CRT covering quality is certified in `docs/COVERAGE_CERT.md` (search tool:
`bin/cover_max`, report: `docs/COVERAGE_MAX_EXPERIMENT.md`). The p74
production cover is at its search optimum (1141 survivors, 92.81% coverage);
the p98 cover was improved from 987 to 952 survivors (93.66% → 93.88%) and
the p67 cover from 1108 to 1101 survivors (92.45% → 92.50%); both improved
covers are deployed in `gap_hunt_fleet.conf`.

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
| `--sieve-primes <n>` | `50000` | Small-prime sieve limit. Non-CRT + `--enable-gpu-fermat`: adaptive bit-scaled default (`log2(depth)` interpolated between 282-bit → window+halo cover and 311-bit → 20M, clamped to `[cover, 20M]`; measured +14% win/s at shift 55, deeper than 20M makes the CPU sieve the bottleneck). With `HALF_CLASS` the 311-bit anchor drops to 5M (measured peak at shift 55: 5M = 609 win/s vs 20M = 391). CRT mode: `10000000` on CPU, `100000` on GPU, `2000000` on the fused GPU path (`FUSED_GPU=1`; measured on the production host at shift475 live merit: 500K=3183, 1M=3197, 2M=3324, 5M=3205 win/s — 2M is the optimum; older dev-host runs: shift258 3106 win/s at 1M vs 2814 at 5M, shift509 1971 vs 1905). GAP_HUNT: default **`2000000`** (lowered from 10M on 2026-09-15; re-measured with the chunked mark split and the jump2 chain on, where the curve is flat-then-falling: shift507 0.5M=1893, 1M=1908, 2M=1897, 5M=1819, 10M=1736, 20M=1558 win/s; shift1017 2M=599, 5M=596, 10M=593, 20M=538 — 2M is equal-best at both shifts with 5× less prime table and faster startup; record parity at 2M vs 10M verified: identical gap/merit/start over the same k range. The older "deeper sieving trims survivors directly" advice (2M=91, 10M=154, 20M=161 win/s at shift1017) was measured in the full-scan path *before* the mark split, when the host sieve dominated; it no longer holds in chain mode) |
| `--merit <v>` | node difficulty | Merit threshold override (lower = more BPSW work) |
| `--enable-submission` | off | Submit BPSW-verified gaps via `submitblock` |
| `--coinbase-script-hex <hex>` | none (`OP_TRUE`) | Payout scriptPubKey for submitted blocks |
| `--enable-gpu-fermat` | off | Use the CUDA base-2 MR kernel as the primality filter (requires `WITH_CUDA=1`; falls back to CPU on failure) |
| `--record-log <path>` | `gapminer_records.log` | Log every BPSW-verified candidate with full parameters; each line carries `new_record=yes/no/unknown` and the strongest claim label `claim=FIRST_KNOWN_OCCURRENCE` (known-corpus first occurrence) or `none` — never a bare "record" |
| `--merit-records <path>` | `data/prime_gap_merits.txt` | Best-known-merit table used to flag `new_record=yes` |
| `--gap-hunt` | off | Standalone record-hunting walk (requires `--crt-file` and a `WITH_CUDA=1` build; runs the walk and exits instead of starting the miner) |
| `--gap-hunt-start <hex>` | `2^(255+shift)` | Base anchor for the walk (hex); default follows the CRT file's shift; CRT-aligned internally |
| `--gap-hunt-min-merit <m>` | `15` | Report gaps with merit ≥ m (true record merit `gap/ln(start)`) |
| `--gap-hunt-state <path>` | none | Resume state file (k and diagnostic last prime) |
| `--gap-hunt-out <path>` | none | Results file (`<gap> <merit> <startprime>` per line; stdout-only if unset) |
| `--gap-hunt-device <n>` | `0` | CUDA device id for the walk (multi-GPU fleets); `--threads` is ignored in `--gap-hunt` mode (one walker per process/GPU) |
| `--help` | — | Print the help message |

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `GPU_SIEVE` | off | Experimental GPU bitmap sieve (`1` enables; multi-GPU only) |
| `GPU_SIEVE_TIMING` | off | CUDA-event accounting for the GPU sieve kernels: adds a `GPU kernel split` line to `ROLLING STATS` with the pure kernel time of the MR, mark and extract launches as a share of uptime (`gpu_sieve_accounted_mark_us` / `_extract_us`). In `--gap-hunt` it additionally prints a `stage us/window` line on the periodic tick (`mark=`/`extract=`/`mr=` per window and as a share of the window). Events are recorded around the launches and read after the stream sync those paths already perform, so no extra synchronization is added. Measured (RTX 3070, shift512 `p75_lex_m30`, difficulty ≈ 23.9, 1 thread, fused chain, chunk 32): **MR 45.6% + mark 31.2% + extract 5.6% = 82.4% of wall is GPU kernel execution** (nvidia-smi utilization 93%, the difference being gather/memset kernels and copies) — that was BEFORE the `GPU_MARK_SPLIT` rewrite; after it the mark share drops to 3.5% and the MR kernel becomes the wall (see `GPU_MARK_SPLIT`). Hunt (jump2, 2M hunt default, per window): shift1017 1683 µs total — mark 59 µs (3.5%) + extract 60 µs (3.6%) + **MR 1421 µs (84.5%)**; shift507 576 µs total — mark 46 µs (8.0%) + extract 30 µs (5.1%) + **MR 387 µs (67%)**, the rest host/gather overhead |
| `GPU_SIEVE_BATCH` | `1024` | Windows per GPU bitmap-sieve batch (`1..4096`, autotuned) |
| `FUSED_GPU` | off | Full GPU-resident CRT pipeline (sieve+extract+MR on-device; implies `GPU_SIEVE`, defaults to a 2M deep sieve) |
| `FUSED_STAGE_TIMING` | off | Adds a `Fused stage split` line to `ROLLING STATS` with the host wall time of each fused stage (mark / extract / chain, plus the chain's own gather and MR round trips) as a percentage of uptime, and the chain round count. Costs four `clock_gettime` calls per window plus four per chain round — diagnostics only. Measured (RTX 3070, shift512 `p75_lex_m30`, live difficulty ≈ 23.9, 1 thread, fused chain): at chunk 32 **mark 20.8% + extract 36.2% (host side 57%) vs chain 42.0%** (gather 14.7%, MR round trip 27.0%) with ~16.6k chain rounds per 120 s ≈ 3 ms per round — i.e. after the chunk fix the host sieve side is the bottleneck, not the MR kernel. In the clean chunk-sweep runs the split was mark 35.3% / extract 11.1% / chain 51.8% (rounds 35479) at the default chunk 32 and 36.1% / 11.1% / 51.0% (rounds 23106) at chunk 64; with `GPU MR acc/wall = 0.492` and external `nvidia-smi` utilization **93% (50/50 samples ≥ 90%)** the fused path is GPU-saturated, so the non-MR ~50% of wall is the mark/extract kernels, **not idle time** — hiding the chain round trip (second in-flight context) buys nothing |
| `GPU_MR_BATCH` | `8` | Windows per accumulated MR batch on the fused path (`1..8`; `1` = per-window, the old behavior). One `gpu_fermat_submit_device` per K windows instead of K small-batch launches. Measured on the dev host (RTX 3070, 8 workers, 20s runs, 0 failures): shift258 K=1 3047 → K=8 **5715 (+88%)**; shift509 K=1 1971 → K=8 **3211 (+63%)**. K=8 vs K=4 (all wins, no regressions): +6.5% (258), +2.7% (450), +4.5% (509), +1.4% (657), +0.7% (720), +1.9% (1008). Candidate counts per window are identical across K (verified per shift). Memory note: K=8 sizes the device candidate buffers at 8 windows; on ≤4 GB cards with 8 workers at shifts ≥ 1008 this may OOM and fail-closed to the CPU path |
| `GPU_SIEVE_PAIR` | off | Experimental 2-window pair-batched fused mark (one kernel writes both ping-pong bitmaps). Measured **-58%** win/s on the dev host (8 workers / 1 GPU, shift258: 1262 vs 3022); the monolithic kernel starves extract/MR kernels at the GPU scheduler. Benchmark-gated — do not enable in production |
| `CRT_ROWS_BATCH` | `64` | CRT row-walk (Horizon-style): one header nonce yields several aligned bases `base + m·P` (P = the cover primorial), amortizing the per-window SHA256 + CRT alignment over a batch of rows (`0` disables, `1..1024`; the remaining nonce space `(2^shift − nadd0)/P` also bounds the rows — ~64 rows at shift507 with the p74 cover (log2 P = 500.5), only ~8 at shift512 with the p75 cover (log2 P = 509.0)). That starvation is measurable and matters: at shift512, ctr_bits=3 (7 rows) costs ~9% of win/s versus a ctr_bits≥6 cover at the same shift (measured 2026-09-17, quiet GPU, K=128 chain, identical 120.3 MR tests/window: 4809/5047 win/s for `shift512_p75_lex_m30` vs 5288 for `shift509_p74_strong_m32` and 5229/5464 for a freshly generated 73-prime `--ctr-bits 20` file, so **any shift512 benchmark number taken with the p75 file is ~9% low**). Rule of thumb: keep `ctr_bits ≥ 6` (≥ 64 rows, the measured saturation point); the shipped m23 table uses 8, and the production fleet configs are all in that regime. On the fused GPU path (`FUSED_GPU=1`) rows > 1 additionally switch to the **row-batch mark**: one residue sweep computes `base mod p` AND `P mod p` on-device, then one kernel marks all row bitmaps (rank-major: thread = prime, rows in the inner loop). The per-row first-odd-offset flips with the row parity only when P is odd (some covers have an even primorial P — the even-P grid bug was caught by a false-gap flood and fixed; `test_gpu_sieve` covers both step parities). Measured at shift507 with kernel accounting (`GPU_SIEVE_TIMING=1`, fused chain, chunk 32, 1 thread, 120 s): **rows=64 → 2861 win/s vs 2428 with the row walk off (+17.8%)**; mark cost only drops 0.138 → 0.121 ms/window while the per-window SHA256/alignment/launch bucket drops 0.077 → 0.032 ms/window (this supersedes the older "statistically neutral at 4 threads" note). Correctness: parity-verified per row vs the CPU sieve. With `GPU_MARK_SPLIT` (default on) the dense part (`p ≤ window`) of that marking is chunked over (prime, row, chunk) work items instead of one thread per prime — see `GPU_MARK_SPLIT` below (that is where the row batch's mark cost went: 32% → 3.5% of wall). Example: `FUSED_GPU=1 CRT_ROWS_BATCH=64 ./bin/gapminer --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 4 --enable-gpu-fermat` |
| `GPU_MARK_SPLIT` | on | Dense marking for primes `p ≤ window` is done by **chunked work items** — one per (prime, row, odd-slot chunk) — instead of the per-prime walk (rows in the inner loop), on **all three** fused marking shapes: the CRT row batch, the single-window mark (`rows == 1`, e.g. `CRT_ROWS_BATCH=0`) and the 2-window pair batch (`GPU_SIEVE_PAIR=1`). Same slots marked (bit-exact: `test_gpu_sieve` `split-rows` + `pair rows1` cases cover both stride parities, the split+sparse mixed table, forced chunk sizes and both ping-pong windows) at **~9–14× less mark kernel time**: **+47–55%** windows/s on the row batch (shift512 and shift507), **+43%** single-window (`CRT_ROWS_BATCH=0`: 2399 → 3439 win/s, mark 33.7% → 5.3% of wall), **+40%** pair batch (2400 → 3354, mark 33.7% → 5.2%), and **+16%** on the standalone `--gap-hunt` walk used by the fleet (shift507, min-merit 19, default 10M hunt depth: 724/726 → 839/845 win/s, order swapped — smaller because the walk is CGBN-MR-bound). Cost model behind it (`tools/bench_mark.cu`, `make bin/bench_mark WITH_CUDA=1`): the old walk's time is set by the warp with the longest trip count (the `p = 3..131` warp walks `W/3` slots per row) and 73% of it is the marking loop, **not** the atomic store — which is why cutting atomics by 4.5× was a −15% regression and why the 100k→2M sieve-depth sweep was flat. `0` disables (falls back to the per-prime walks, byte-for-byte the old behavior); it also fails closed to them if the prime table is not ascending (the split domain is the `p ≤ window` prefix of the table). Expensive MR tests per window unchanged in all A/Bs and `MINING_JUMP2_VERIFY` parity clean (1173 flights at rows=64, 1138 at rows=1). Example: `GPU_MARK_SPLIT=0 FUSED_GPU=1 ./bin/gapminer --crt-file data/crt/m23/shift512_p75_lex_m30.txt --threads 1 --enable-gpu-fermat` |
| `GAP_HUNT_TIMING` | off | Hunt diagnostics: adds three lines to the periodic tick — **GPU kernel split** (mark/extract/MR per window and as a share of the window; arms the same CUDA-event accounting as `GPU_SIEVE_TIMING`, which it enables automatically), **host stage split** (setup / mark / extract / submit / detect per window) and **chain bookkeeping** (MR candidates tested per window and chain rounds per 1000 windows). Diagnostics only — leave it off in production. Measured at shift1017 `p130_lex_m30`, jump2, chunk 64, 2M depth (RTX 3070): window 1748 µs — **MR 1538 µs (88%)**, extract 68 µs (4%), mark 32 µs (2%), with ~65 µs (3.5%) of host-side call time outside the kernels → the walk is MR-throughput bound and the host sieve side is no longer a lever. At shift507 the same split is MR 387 µs (67%) of a 576 µs window (mark 8%, extract 5%) |
| `GPU_MARK_SPLIT_SLOTS` | `160` | Odd slots per marking chunk when `GPU_MARK_SPLIT` is active (clamped to `32..4096`; chunks per row = `ceil(window / slots)`, capped at 1024). The optimum is broad: 32–128 chunks per row all land within noise at shift512 (isolated bench: 21 µs at 64 chunks vs 42 µs at 8 chunks vs 757 µs for the per-prime row walk). Example: `GPU_MARK_SPLIT_SLOTS=64 FUSED_GPU=1 ./bin/gapminer --crt-file data/crt/m23/shift512_p75_lex_m30.txt --threads 1 --enable-gpu-fermat` |
| `MINING_JUMP2` | **on** | **No-test chain** (Horizon M19-style certificates) on the fused CRT mining path: per-window backward-search chain in chunked MR rounds shared across K flight windows (~3× fewer MR tests per window; stops at the merit frontier — every skipped interior gap is provably below threshold). **Default on since 2026-09-15** (measured +70–76% windows/s, see below); `MINING_JUMP2=0` restores the full scan. **Inert without `FUSED_GPU=1`** (the chain lives on the fused pipeline; without that gate `chain_on` stays 0 and the run is a plain full scan). Works with `HALF_CLASS`/`QUARTER_CLASS` (the chain walks the class-filtered survivor list; reported visible pairs are identical to the full scan's and hidden interiors are resolved by the existing on-demand machinery). Each worker owns its GPU contexts (gather staging is per-worker), so the chain runs on every worker independently; with the round-robin device assignment (`gpu_device = i % gpu_count`) set `--threads` to the number of GPUs for one chain per card. Fail-closed: any chain error falls back to a full scan of the same flight, then to the CPU path. Disables the gap-dist health histogram (the frontier cut distorts small-gap frequencies, same policy as `HALF_CLASS`). Parity-verified via `MINING_JUMP2_VERIFY` vs the full scan: **0 mismatched merit-candidate sets** (2370 flights at shift507, 1173 at shift512 rows=64, 1138 at `CRT_ROWS_BATCH=0`). A/B on the dev host (RTX 3070, fused chain, 1 thread, default chunk 32, `GPU_MARK_SPLIT` on, 90 s per point, order swapped): shift512 `p75_lex_m30` **2353 / 2300 win/s full-scan → 4002 / 3994 with the chain (+70%)**, shift507 `p74_lex_m30` **2416 / 2423 → 4275 / 4250 (+76%)**, with expensive MR tests per window **620.7 / 622.0 → 120.1 / 120.5 (5.2× fewer)** at the same settings. (An earlier 300 s A/B measured +41%/+51% — the gain is larger now because the mark kernel no longer dominates the window.) Class modes (parity-verified, 0 mismatches): `HALF_CLASS` 1973 → **2630 (+33%)**, `QUARTER_CLASS` 928 → **1018 (+9.7%)** (quarter stays host-bound at 1 thread). 2 chain workers on one 3070: `3639 win/s` vs `2953` for one. Example: `MINING_JUMP2=0 FUSED_GPU=1 ./bin/gapminer --crt-file data/crt/m23/shift507_p74_lex_m30.txt --threads 1 --enable-gpu-fermat` |
| `MINING_JUMP2_BATCH` | `512` | Windows per chain flight (`1..512`; the compile-time cap was raised 128 → 256 → 512 on 2026-09-18 after the 2026-09-17 "K=256 corrupts the emitted set" verdict was traced to the **verifier**, not the chain: `MINING_JUMP2_VERIFY` replayed the whole flight's full scan in ONE submit (256 x ~1800 = 455k candidates at shift512, 256 x ~2676 = 685k at shift1017) and the MR staging cap (320000) truncated it, so the verifier reported the untestable tail as `bad`; the replay is now sliced at the context cap and every size verifies clean — K=256: 214 flights at shift512 and 67 at shift1017, K=512: 123 flights, all `bad_windows=0 bad_pairs=0`, while the chain's own submits are K x chunk = 16k candidates, far below the cap). **This batch is a lever that has not saturated:** shift512 5264 (K=128) → 6113 (K=256) → **6725 win/s (K=512)**; shift1017 1498 → 1788 → **2014 win/s**. The cost is device memory — the sieve sizes its candidate buffers as `2 x odd_interval_size x K x limbs x 8 B` (shift512 K=512 ≈ 1.6 GB, shift1017 K=512 ≈ 4.6 GB; measured 5.6/8 GB used) — so lower K on cards with less free VRAM; an allocation failure fails closed to the CPU path, never to a wrong result. The per-round gather/submit/collect latency is amortized over K windows. **Default raised 64 → 128 (2026-09-17, +35% measured** on the quiet dev GPU, shift512 `p75_lex_m30`, 1 thread, fused chain, chunk 32, 45 s runs, order swapped: K=64 3902/3780 → K=128 5218/5175 win/s, identical 120.3 MR tests/window), parity-verified with the new default: 377 clean `windows=128 bad_windows=0 bad_pairs=0` flights, 0 bad). **Raised 128 → 256 (2026-09-18): +17% at shift512** (`p75_lex_m30`, 1 thread: 5264 → 6113 win/s, 119.7/122.6 → 141.4/143.3/141.5 mH/s) **and +19.4% at shift1017** (`p130_lex_m30`: 1498 → 1788 win/s), both parity-verified with the sliced replay) |
| `MINING_JUMP2_CHUNK` | `32` | Survivor chunk size for one backward-search step (`8..512`). Measured (RTX 3070, shift512 `p75_lex_m30`, live difficulty ≈ 23.9, 1 thread, fused path): chunk 32 and 64 give the same throughput (2736 vs 2751 win/s, within noise) but 32 needs **1.66× fewer expensive MR tests per window** (120.5 vs 200.2) — same speed with real MR headroom. Chunk 16 is latency-bound (2069 win/s, 87.4 tests/window); chunk 128 wastes tests (384.5/window) and is slower (2163 win/s) |
| `MINING_JUMP2_VERIFY` | off | Dev-only parity check: full-scans every flight and compares the emitted merit-candidate sets with the chain (doubles MR work — benchmark/validation only) |
| `HALF_CLASS` | off | Two-pass scan: sieve/test only residues `{1,7,11,13,17,19,23,29} mod 60`, verify the hidden classes on demand (~2× fewer MR candidates; non-CRT +87% at shift 55, CRT fused +14.6% at shift509); disables the gap-dist health histogram. In CRT mode the covering template pre-filters the verification and the back-lookahead stays unfiltered |
| `QUARTER_CLASS` | off | Generalizes `HALF_CLASS` to 4 visible / 12 hidden coprime classes (`{1,7,11,13} mod 60` visible). The containment lemma (every true qualifying gap is contained in a visible qualifying gap) guarantees no blocks are lost while the GPU MR load halves. Hidden resolution runs on the GPU MR pipeline (base-2+3 batch + BPSW only on MR survivors; falls back to the CPU path without CUDA), and the fused head is extended ~12·logbase in this mode to keep tail re-marks rare. Measured on the dev host (RTX 3070, fused path, live difficulty): **3515 win/s vs 3176 for `HALF_CLASS` (+10.7%)**, tails skipped 99.4%, GPU-bound (acc/wall 2.8). Parity-exactness unit tests green; production A/B on the dual-3060 box pending |
| `GAPMINER_CPU_LIMBS` | off | Use the ported fixed-limb Montgomery Fermat/Euler path instead of GMP `mpz_powm` for the CPU-only Euler filter (slower than GMP 6 on the dev host; opt-in for hosts without GMP's tuned asm) |
| `GAPMINER_CPU_WINDOW_OVERRIDE` | `4` | Force the CPU limb path's exponentiation window width (`3`, `4` or `5`; `4` is the specialized default) |
| `GAPMINER_CPU_WINDOW_LOG` | off | Log the selected window width once per limb count (diagnostic) |
| `GHDBG_DUMP_K` | none | Bounded per-window dump for hunting a lost record: writes EVERY candidate of window `<k>` (index, offset, MR verdict) plus the prime offsets to `GHDBG_DUMP_FILE` (default `/tmp/ghdump_<k>.txt`), hard-capped at 40000 lines. It is how a missing gap is diagnosed (endpoint absent vs present-with-wrong-flag vs present-but-unpaired) |
| `GHDBG_DUMP_FILE` | `/tmp/ghdump_<k>.txt` | Destination of the `GHDBG_DUMP_K` dump. There is deliberately **no** "dump every window" mode: an earlier version of it wrote 8.1 GB in one run and took the editor down (`docs/BUILD_HYGIENE_INCIDENT.md`) |
| `GHDBG_SUBMIT` | off | Prints, once per process, the first `gpu_fermat_submit*` call with `count > 100000` together with the context's real `max_batch`. Use it to catch a stale-object/macro mismatch that would silently drop candidates |
| `GAPDEBUG` | off | CRT gap diagnostics (HALF_CLASS and full-class modes): for every emitted gap, log to stderr the window class, gap class endpoints, and the interior candidates with their MR flags; in HALF_CLASS mode also logs `[HIDDBG]` hidden-class resolution counters (candidates tested / primes found per resolved interval). Used to trace false-gap regressions; verbose — development only |
| `GAP_HUNT_BATCH` | `512` | GAP_HUNT windows per accumulated MR batch (`1..512`; raised 32 → 64 on 2026-09-15, 64 → 128 on 2026-09-17, **64 → 512 on 2026-09-18** with `GAP_HUNT_BATCH_MAX` = 512). Full sweep on the quiet dev GPU with the mode-aware bound below (jump2 on, min-merit 19, fixed k range 0..65535, emitted sets identical): shift507 **2452 (K=64) → 3156 (128) → 3601 (256) → 3824 win/s (512), +56%**; shift1017 **881 (128) → 1001 (256) → 1172 win/s (512), +33%**. Non-vacuous parity at min-merit 16 over k∈[0,131072): **15 gaps, set-identical** at K=512 vs K=64, 3434.6 vs 2291.8 win/s (**+49.9%**). A chain round pays a ~4 ms per-launch floor in the CGBN MR kernel, so more windows per round = cheaper rounds. **Device-memory cost is `2 × odd_interval_size × K × limbs × 8 B`**, printed by the walk banner as `vram_est=<MB>` for the configured K: shift507 K=512 = **2384 MB**, shift1017 K=512 = **6407 MB** (and ≈ 7.3/8 GB actually used once the sieve contexts + adapter + display are added, i.e. at high shift K=512 sits at the edge of an 8 GB card). Check it against `nvidia-smi`: on an 8 GB card at shift ≥ 1000 use `GAP_HUNT_BATCH=256` (≈ half); a 12 GB card takes K=512 with room to spare. An allocation failure is visible (`gpu_sieve: cands[0] alloc: out of memory`) and stops the walk; it never falls back silently. The batch is bounded by the mode: with the full-scan head (`GAP_HUNT_JUMP`/`GAP_HUNT_JUMP2` off) it is clamped per window to the MR context's REAL per-call cap. Batch 128 measures **+19.6%** (shift507 `p74_lex_m30`, min-merit 19, fixed k range 0..65536, quiet GPU: 64 → 2435.1/2450.5 win/s, 128 → 2917.3/2924.8 win/s) and the emitted gap set is batch-independent at that setting, but **the default stays 64** because the larger setting only pays off when the batch fits the MR context's real cap: at min-merit 16 over k∈[0,131145) the K=128 and K=64 chains both emitted the same 15 gaps while the full scan emitted 11 — that discrepancy turned out to be **build hygiene, not gap logic** (a stale `gpu_adapter.o` carried the old 160000 candidate cap and every candidate past it was dropped silently; see `docs/BUILD_HYGIENE_INCIDENT.md`), and after the clean rebuild the same plain run gives 15/15 byte-identical to the chain. Batches are sized from the MR context's REAL per-call cap (`gpu_fermat_max_batch()`, printed as `mrcap=` in the walk banner), never from a compile-time macro: a stale object can carry an older cap and an oversize batch is then truncated silently (see `docs/BUILD_HYGIENE_INCIDENT.md`). The walk auto-reduces the batch when the density estimate does not fit and the fill fails closed |
| `GAP_HUNT_QUARTER` | off | **FALSIFIED experiment** — 4-visible-class scan + on-demand hidden resolution (containment lemma, exact — parity-tested identical to full-class). At record thresholds (merit ≥ 15) the visible-gap trigger fires ~1.3×/window (visible gaps inherit the σ-tail with mean merit ≈ 8) and CPU resolution costs ~40 ms each → 46 win/s vs 893 full-class. Kept off; exact but not profitable |
| `GAP_HUNT_KMAX` | none | Stop the walk at this k (tests/benchmarks) |
| `GAP_HUNT_JUMP` | off | **FALSIFIED experiment** — Kehrig-style per-window serial CGBN walk (jump-by-threshold + backward search, ~1.5 MR tests per prime instead of all survivors). Exact: parity-identical gap sets on shift507 (299/299) and shift1017 (411/411). But 6.3× slower at shift1017 (23.5 vs 147 win/s) — CGBN cooperative-test latency (~10 ms/test under load) cannot be hidden by 32-way window parallelism. Follow-up diagnosis: the 1017 batch path is **MR-bound** (147 win/s × 2,675 survivors = 393k tests/s ≈ CGBN AL=20 capacity scaled from the 907k/s AL=12 measurement); `GPU_FERMAT_TPI` 16/32 are slower than 8 at AL=20 (66 vs 44 vs 24 win/s). Kept off; reopen trigger: per-thread low-latency MR or a quiet dedicated GPU |
| `GAP_HUNT_JUMP2` | **on** | **Chunk-parallel backward search (Kehrig-exact, latency-free)** — the same jump-by-threshold chain semantics as `GAP_HUNT_JUMP`, but each step tests a CHUNK of survivors batched across all K windows (default 512) in one tight MR submit (gather kernel → `gpu_fermat_submit_device`), so the cooperative-test latency that killed the serial jump is hidden. Parity-verified: gap sets byte-identical to the full-scan batch path on shift507 (KMAX=128), shift1017 (KMAX=64), and the full 188k-window production range on shift998 (k 12819456..13007648, `test_gap_hunt` bad=0). A production false-gap bug (stale per-flight `jump_s/jump_e` pairs re-emitted against later windows) was fixed by resetting the per-fill record slots; `verify_gap_candidate.py` confirms the strict interior scan. Measured on the dev host (RTX 3070, record walker sharing the GPU, shift1017 merit 20): **~486 win/s vs ~159 full-scan = 3.0×** (stable across runs). Fail-closed: any CUDA error falls back to the batch path. `GAP_HUNT_JUMP2_CHUNK` (default **32** since 2026-09-15, 8..512) sets the per-window chunk size; it is coupled to `GAP_HUNT_BATCH`: with a 64-window batch the optimum is the smaller chunk (shift1017: chunk 32 → 679 win/s at 268 tests/window vs chunk 64 → 664 win/s at 390 tests/window; shift507: 2406 win/s at 222 tests vs 2283 at 369), while at K=32 the old default of 64 was better (425 vs 559 win/s) — a chain round pays a fixed per-launch floor, so the two knobs must be tuned together. **Since 2026-09-18 the batch default is 512; the chunk was NOT re-swept at K=512 (32 was optimal at K=64), so re-measure with `GAP_HUNT_TIMING=1` before changing `GAP_HUNT_JUMP2_CHUNK`.** **Re-measured 2026-09-15 with the current defaults** (`GPU_MARK_SPLIT` on, 2M hunt depth, fixed k ranges of 16384 windows, no competing GPU work): shift507 **888 win/s full-scan → 1799 win/s with jump2 (+103%)**, shift1017 **170 → 579 win/s (3.4×)** at the then-default K=32/chunk 64; with the 2026-09-15 defaults (K=64, chunk 32) the same fixed k ranges measure **2386 win/s (shift507, +27% vs K=32) and 677 win/s (shift1017, +21%)**. Chunk re-swept at K=64 (shift1017): 8=262, 16=456, 24=577, **32=679**, 64=664 win/s — with the bigger batch the optimum moves to the smaller chunk. With the mark split in place the walk is now **MR-throughput bound**: `GPU_SIEVE_TIMING=1` reports mark 3.5% + extract 3.6% + **MR 84.5%** of the window at shift1017 (shift507: 8.0% + 5.1% + 67%), so the remaining levers are the MR tier itself (per-test cost/rejected-candidate count), not the host sieve side. Chain trade-off measured with `GAP_HUNT_TIMING=1` at shift1017 (tests the chain submits vs the rounds it needs): chunk 16 → **216 tests/window, 879 rounds/1k-windows, 283 win/s**; chunk 32 → 266 tests, 490 rounds, **455 win/s**; chunk 64 → **386 tests, 300 rounds, 563 win/s**; chunk 128 → 673 tests, 222 rounds, 437 win/s. Smaller chunks really do test fewer candidates (44% fewer at chunk 16), but every round costs a gather+submit+collect+sync, so the optimum sits where that fixed cost balances the candidate count — and it MOVES with the batch (K=32 → chunk 64; K=64 → chunk 32), because the fixed cost is per LAUNCH, not per candidate. Host-side batching was evaluated and rejected: the host contributes only ~3.5% of the window, so batching the per-window D2H/state work has no measurable headroom |

## Build hygiene

The Makefile generates and includes header dependencies (`-MMD -MP` + the
`.d` files), so changing a header rebuilds every user of it. Keep it that way: a
header-only change (for example raising a batch-cap constant) that does NOT
recompile its users leaves objects built from different revisions linked
together, and constants can then disagree silently. If a measurement contradicts
itself by *configuration* rather than by data — the 2026-09-17 case was "the same
window reports its gap with `GAP_HUNT_BATCH=64` but not with 128" — run
`make clean` before chasing the logic. Full write-up:
`docs/BUILD_HYGIENE_INCIDENT.md` (stale-object silent truncation + the bounded
diagnostics that localized it).

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

The same block reports the expensive-test density on a `GPU MR tests` line:

```
GPU MR tests: 47185920 (90.0/window, 78.8 per 1000 survivors)
```

`GPU MR tests` is the number of candidates actually submitted to the GPU MR
kernel (not the number of sieve survivors). In full-scan mode it tracks the
survivor count; in `MINING_JUMP2` chain mode it drops to the frontier candidates
only. Both ratios are geometry-independent and are the metric to compare against
other miners' per-row test counts (e.g. a 74-prime/S512 cover with 761
candidates/row and ~19 expensive tests/row).

In `MINING_JUMP2` mode the `Max Euler pair` line reports the largest **chain certificate span** (≈ active difficulty by construction: the frontier jump pair's gap is `ceil(difficulty·logbase) − ε`, always just below the threshold), not a true consecutive-prime pair — the true per-window max pair (full-scan) is the cross-cover gap ≈ 9-10k (merit ~17-19). `Merit candidates`/BPSW/submission are unaffected (parity-verified, 0 mismatches).

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
data/prime_gap_merits.txt  Best-known-merit reference table (local only,
                  generated by scripts/update_merits.sh; not in git)
scripts/          gen_crt_batch.sh, update_merits.sh, ab_shift_compare.sh,
                  watch_gap_hunt_records.py, gap_hunt_stats.py, analyze_n3.py,
                  tail_compare.py, tail_shape.py, record_rate_model.py
gen_crt.md        CRT covering-file generator guide
docs/             Architecture references, GAP_HUNT plan, closed-fingerprints registry (dead routes and their reopen triggers)
```

## License

GPL-3.0-or-later. See `LICENSE.md`.
