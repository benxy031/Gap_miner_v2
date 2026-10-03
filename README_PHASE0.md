# Phase-0 Exhaustive Prime-Gap Scanner

Standalone toolkit for the white-paper Phase-0 workload: **exhaustive** scanning
of a decimal range for large prime gaps, with every reported gap re-verified by
GMP.  It is independent of the Gapcoin miner in this repo (no gapminer/
gap_hunt objects; its own kernels and tools).  Measurement history, theory and
model comparisons live in [`docs/PHASE0_scan_bench.md`](docs/PHASE0_scan_bench.md)
- this file is the operator manual (tools, flags, defaults, gates).

Current performance (RTX 3070, i3-10100, 8 threads):

| range | wall | rate |
|---|---|---|
| 1e10 from 2e20 | 0.84-0.88 s | 1.14-1.19e10 ints/s |
| 1e12 from 2e20 | 56.6-58.4 s | 1.71-1.77e10 ints/s |
| 1e14 from 2e20 | **1.545 h** (measured 2026-10-01: wall 5561.24 s) | 1.86x the old 2.87 h campaign |

---

## 1. The toolkit

| binary | source | purpose |
|---|---|---|
| `bin/phase0_scan_gpu` | `tools/phase0_scan_gpu.cu` | production scanner (GPU sieve + GPU MR, GMP gap verification, records, resume) |
| `bin/phase0_scan` | `tools/phase0_scan.c` | CPU-only reference scanner (same gates, slow; no GPU) |
| `bin/bench_p0sieve` | `tools/bench_p0sieve.cu` | GPU marking probe: rate + **bit-exact check** vs a naive CPU marker |
| `bin/mr68_gpu` | `tools/mr68_gpu.cu` | 96-bit Miller-Rabin kernel validator (`--validate`, GMP cross-check) + benchmark |
| `scripts/phase0_hl_compare.py` | | merit-tail comparison vs HL-1t / HL-4p models (figure) |

Shared device code: `tools/mr68_kernel.cuh` (CIOS Montgomery + base-2 MR +
bitmap-fed stages), `tools/p0_mark.cuh` (wheel fill + marking kernel).

Build:

```bash
make bin/phase0_scan_gpu bin/bench_p0sieve bin/mr68_gpu WITH_CUDA=1
make bin/phase0_scan            # CPU reference (needs GMP, no CUDA)

# Windows (MSYS2 MINGW64, see README_WINDOWS.md): the kernels are split into
# bin\phase0gpu.dll (nvcc + MSVC) and the tools are MinGW g++ host programs.
windows\build_phase0.bat        # 1x: kernel DLL, cached in build-win\gpu-phase0-*
make -f Makefile.win phase0 tools   # the five tools (also built by build_all.bat)
```

`bin/phase0_scan_gpu` includes `tools/mr68_kernel.cuh`; after editing that
header rebuild the scanner, the probe and the validator together
(`make ... WITH_CUDA=1`) and re-run the gates in section 5.

---

## 2. `bin/phase0_scan_gpu` - production scanner

```
bin/phase0_scan_gpu [--start DEC] [--length DEC] [--sieve-limit P] [--threads T]
                    [--seg-bits B] [--batch K] [--merit-min M] [--log FILE]
                    [--state FILE] [--state-every S] [--progress S] [--device D]
                    [--records FILE] [--no-records] [--gpu-sieve|--cpu-sieve]
                    [--check [N]] [--legacy-mr] [--no-test]
                    [--engine sieve|walk] [--gap-min G] [--walk-primes P]
                    [--walk-batch B]
```

Defaults:

| flag | default | notes |
|---|---|---|
| `--start DEC` | `200000000000000000000` (2e20) | any integer; range must stay < 2^96 |
| `--length DEC` | `10000000000` (1e10) | |
| `--sieve-limit P` | `1e7` (GPU sieve) / `3e4` (CPU sieve) | marking primes 17..P; 1e7 is the measured optimum (deeper sieving loses: item count scales with pi(P)) |
| `--threads T` | `8` | one CUDA stream + one slice per thread |
| `--seg-bits B` | `24` (GPU) / `20` (CPU) | segment = 2^B integers; range 16..27 |
| `--batch K` | `4194304` (GPU) / `1048576` (CPU) | only used by `--legacy-mr`; range 1024..16777216 |
| `--merit-min M` | `20.0` | gaps >= M are written to the log and GMP-verified; bands/top-4/records still see every gap with merit >= 6 (internal gate) |
| `--engine sieve\|walk` | `sieve` | `walk` selects the class-30 bitmap sieve + batched jump-walk engine (~10.6x the sieve engine's throughput at the production default: measured 5.2 s vs 55.8 s for 1e12 integers, identical 5-gap output, 2026-10-02; same log/records/verify/top-4/resume path). `--check`, `--gpu-sieve`/`--cpu-sieve`, `--legacy-mr` and `--sieve-limit` apply to the sieve engine only; combining them with `--engine walk` is an error (or ignored where noted) |
| `--gap-min G` | `ceil(--merit-min * ln(start))` | walk engine only: report every gap >= G (GMP-verified). G in [100, 1e7]; values < 300 print a warning (walk cost scales ~1/G). Passing `--gap-min` explicitly removes the merit gate; an explicit `--merit-min` stays as an additional bar |
| `--walk-primes P` | `15000` | walk engine: sieve/item primes 17..P (measured optimum at this geometry) |
| `--walk-batch B` | `64` | walk engine: super-batch = B x 30-blocks (**default 64 since 2026-10-03**, was 32; range **4..96**, raised from 4..32 the same day). ABBA on 1e13 at `--gap-min 702` (dev 3070, 4 arms): K32 63.68 s vs **K64 62.12 s = +2.45 %**, K96 62.26 s (+2.23 %, no better than 64 but 6.5 GB); both K64 arms beat both K32 arms and the emitted sets are identical (parity K32==K64==K96 exact, 1624/1624 on 1e12). VRAM = 2 regions x (B+1) x 33.6 MB + gap buffer: default K=64 -> 4.4 GB, K=32 -> 2.2 GB, K=96 -> 6.5 GB (an over-large K fails the bitmap alloc loudly, never silently) |
| `--log FILE` | none | appended; one `# phase0-gpu session ...` header per run |
| `--state FILE` / `--state-every S` | none / `30` s | checkpoint for resume; **deleted when the range completes** |
| `--progress S` | `10` s | progress + ETA line every S seconds (`0` = off) |
| `--device D` | `0` | CUDA device |
| `--records FILE` / `--no-records` | `data/prime_gap_merits.txt` if present | record check: gap length in table AND merit > best known; a beat is logged even below `--merit-min` |
| sieve mode | `--gpu-sieve` | `--cpu-sieve` selects the (slow) CPU marking path |
| MR pipeline | bitmap-fed (fast) | `--legacy-mr` selects the batched steps/res path (A/B reference; `--check` always uses it) |
| `--check [N]` | off | sieve+MR `[start, start+N)` then re-walk the range with `mpz_nextprime`; N default 2e6. Forces the legacy MR path (needs the collect list) |
| `--no-test` | off | sieve + walk only (no MR, no gap detection); not allowed with `--gpu-sieve` |

Environment: `P0_FAST_DBG=N` dumps the first N fast-walk summaries
(word counts, prime count, first/last set slot) to stderr - the tool that
isolated the cross-segment chain bug in the bitmap walk.

### Pipelines

`--gpu-sieve` (default), per segment on the thread's stream:

```
wheel fill (3..13) -> p0_mark (primes 17..1e7; slots already covered by the
wheel are skipped) -> p0_compact (GPU-side survivor compaction to a dense
offset list) -> mr68_from_offsets (one thread per candidate, verdict bits)
-> D2H verdict bitmap (1 MB) on a per-job copy stream
```

The host only walks the verdict bitmap (rare gap events), chains the
segment-boundary gaps and GMP-verifies everything that passes the gate.
`--cpu-sieve` marks on the host instead (measured ~12 CPU-seconds of marking
per 1e10 vs 0.14 s on the GPU; end-to-end several times slower - used for
cross-checks, not production).

`--legacy-mr` = the earlier batched path: host enumerates survivors, ships
`(base3, u32 step)` batches to `mr68_kernel_packed`, scans a 1-byte-per-
candidate verdict array.  Same results, ~1.3x slower at 1e10.

### Walk engine (`--engine walk`)

One sieve kernel marks a 2^28-slot class-30 bitmap (one 30-block of
1,006,632,960 numbers = 32 MB); one fused jump-walk kernel per super-batch
of `--walk-batch` blocks scans the compositeness map, tests candidates with
the device base-2 MR (perig port) and reports every gap >= `--gap-min`.
Reported gaps
enter the same log/records/top-4/band/GMP-verification path as the sieve
engine, and state files are format-identical (`off0` in integers from
`--start`), so a killed walk run resumes exactly with the same command.

Minimal usage (production defaults at the default threshold):

```
./bin/phase0_scan_gpu --engine walk --start 200000000000000000000 \
    --length 1000000000000 --log data/p0_walk.log
```

Higher-yield sweep (report every gap >= 500, ~10.7 merit at 2e20; expect
~13000 rows per 6.4e10 integers and ~2x the sieve engine's ints/s):

```
./bin/phase0_scan_gpu --engine walk --gap-min 500 \
    --start 200000000000000000000 --length 64426429440 \
    --log data/p0_walk_g500.log
```

Constraints (checked at startup, both abort with a message): the whole
range must stay inside one 64-bit window of the aligned block base (always
true for campaign ranges spanning < 1.8e19 integers), and `--start` must
exceed `--walk-primes`.  One worker only (the sieve engine's `--threads`
does not apply); `--state`/`--state-every`/`--progress` work as usual.
Larger super-batches (fewer event syncs, more VRAM) and a deeper sieve:

```
./bin/phase0_scan_gpu --engine walk --walk-batch 32 --walk-primes 40000 \
    --start 200000000000000000000 --length 64426429440 --log data/p0_walk_b32.log
```

### Output

Progress/console (stdout, line-buffered): banner, then every `--progress` seconds
one status line and, for each gap above `--merit-min`, a `GAP ...` line:

```
[phase0-gpu]  42.15%  ints=1.234e+12  1.75e+10 ints/s  u=3.51%  tests=...  primes=...  gaps=...  batches=...  eta=0:52:34
[phase0-gpu] GAP merit=16.0445 gap=750 lower=... upper=... verified=1 record=no table=22.9982
```

The monitor thread prints exactly this status line; end of run:

```
[phase0-gpu] wall=... end_to_end=... ints/s
[phase0-gpu] survivors=... u=...% of ints
[phase0-gpu] fast split (GPU-s sums): wheel+mark=... MR=... copies=...
[phase0-gpu] gaps per merit threshold (this session -> per 1e14 ints):
[phase0-gpu]   m>=10 :  ...  ->  ...
[phase0-gpu]   top#1..4 gap/merit/lower/upper/verified
[phase0-gpu] gaps reported=...  verification_failures=0  records_new=...
```

Note: the `fast split` sums are per-kernel wall times on 8 shared streams and
inflate with stream overlap; run `--threads 1` (or use the 1-thread numbers in
`docs/PHASE0_scan_bench.md` section 13) for honest per-stage costs.

Gap log row format (append; columns space separated):

```
<unix_ts> <lower_prime> <gap> <merit> <upper_prime> verified=<0|1> record=<NEW|no|absent|off> table=<best|unknown>
```

`verification_failures=0` and `verified=0` rows are both expected to be zero at
the end of any run; any `verified=0` row is a reportable bug.

### Resume semantics

* `--state FILE` checkpoint is written every `--state-every` seconds, per
  thread, as the absolute offset of the last fully processed segment.
* Re-running the SAME command resumes automatically (prints
  `resumed from ...: N ints already done before this session (...%)`).
* A completed range deletes the state file (`range complete; state file
  removed`), so a re-run starts fresh.
* The boot prime before each thread's slice start is recomputed on resume, so
  boundary gaps are correct; segmented resume is duplicate-free (content-keyed
  check: 0 duplicates across an interrupted + resumed 4e10 test).

---

## 3. `bin/phase0_scan` - CPU reference

```
bin/phase0_scan [--start DEC] [--length DEC] [--sieve-limit P] [--threads T]
                [--merit-min M] [--seg-bits B] [--no-test] [--check [N]]
```

Defaults: start 2e20, length 1e10, `--sieve-limit 1e8`, threads 4, seg-bits 20,
merit-min 20.0, `--check` N 2e6.  Pure CPU bucketed sieve + GMP MR; used as an
independent reference for the GPU path (same `--check` gate).  No `--log`,
`--state`, `--records` support.

---

## 4. Probes

`bin/bench_p0sieve` - GPU marking rate + bit-exactness:

```
bin/bench_p0sieve [--primes P] [--segs N] [--verify K] [--tpb T]
```
Defaults: P=30000, segs=2000, verify=1, tpb=256.  `--verify K` rebuilds K
segments with a naive CPU marker and compares every bit ("bit-exact vs naive
CPU marker" must be printed after any marking-kernel change).

`bin/mr68_gpu` - MR kernel gate + rate:

```
bin/mr68_gpu --validate N        # kernel verdicts vs GMP over N candidates, 0 mismatches required
bin/mr68_gpu BATCH ITERS         # benchmark: reports tests/s and ns/test
```
Validated configurations on record: `--validate 200000` (8562 primes) and
`--validate 2000000` (85563 primes), both 0 mismatches; bit lengths 63-91 are
covered by `phase0_scan_gpu --check` at varied starts.

`scripts/phase0_hl_compare.py` - measured merit tail vs models:

```
scripts/phase0_hl_compare.py [log] [--out PNG] [--dark] [--m-min 15] [--m-max 30]
```
Default log `data/p0_campaign_2e20.log`; `--dark` also writes
`*_dark.png`.  The script asserts a gate on the HL-4p table
(`sum rho_g = 1.0001 x (1/L)`) and prints per-band measured/model ratios.
`--m-max` default raised 27 -> 30 on 2026-10-03 (the 2e15 walk campaign has
events up to m>=30; the old 27 cut dated from the 1e14 campaign where m>=27
was empty).  Rows with single-digit measured counts are Poisson noise — the
model test lives where the counts are >= ~10 (m <= ~25 at 2e15).

---

## 5. Acceptance gates (run after ANY kernel/pipeline change)

| gate | reference |
|---|---|
| `./bin/phase0_scan_gpu --check 200000` | `CHECK PASS: 4293 primes` |
| `./bin/phase0_scan_gpu --check 2000000` | `CHECK PASS: 42725 primes` |
| `./bin/mr68_gpu --validate 2000000` | `0 mismatches` (85563 primes) |
| `./bin/bench_p0sieve --verify 3` | `3 segment(s) bit-exact vs naive CPU marker` |
| `./bin/phase0_scan_gpu --length 500000000 --merit-min 10 --log L` | `204` rows, top-4 `724/690/684/682` (merits `15.4883/14.7610/14.6326/14.5898`), 0 unverified |
| fast vs legacy equivalence | `--merit-min 10` on 1e10: 4439 rows, byte-identical between default and `--legacy-mr` (sorted, timestamp stripped) |
| PSP fixture | `--start 18447233110000000000 --length 5000000000 --merit-min 15` contains `gap=710 merit=16.0049 verified=1` (the 2-PSP 18447233112263860537 must never appear), 0 unverified |
| walk engine regression | `--engine walk --start 200000000000000000000 --length 18119393280 --gap-min 702 --log L`: exactly 22 rows, identical to the standalone engine fixture `(lower mod 2^64, gap)` pairs, all `verified=1` |
| engine cross-check | same range, `--merit-min 10.65` (sieve) vs `--engine walk --gap-min 500`: after filtering gap >= 500 the gap sets must be identical (measured 1696/1696 on 8 blocks); gaps 498..499 appear only in the sieve log |
| walk resume | kill a `--state` run mid-flight, rerun the same command: checkpoint `off0` printed as `resumed from ... (X%)`, final log set identical to an uninterrupted run, no duplicates (measured 12969 = 12969 exact) |
| resume | interrupt a state-file run, resume, `0` unverified in the combined log |
| split-mode parity (Windows architecture gate) | `make bin/phase0_scan_gpu_split WITH_CUDA=1`, then run it and `bin/phase0_scan_gpu` on the same fixed range with the same flags (`--start 200000000000000000000 --length 10000000000 --engine walk --gap-min 500`, and the sieve engine) and diff the logs sorted with the timestamp field stripped: **0 diff lines** (measured 2026-10-02: 2093 walk rows identical, sieve `tests=351259011` identical). The split compiles the kernels from `tools/phase0gpu_dll.cu` behind `tools/phase0gpu_api.h` - the exact source `windows\build_phase0.bat` builds with nvcc+MSVC |
| 128-bit portability shim | rebuild the split DLL object with `nvcc -DP0_PERIG_U128_SHIM -Itools -c tools/phase0gpu_dll.cu` and repeat the parity gate: the software \_\_int128 substitute (the MSVC path) must give 0 diff lines too (measured 2026-10-02: identical on both engines) |

---

## 6. Campaign recipes

Fresh 1e14 (the campaign range), detached + resumable:

```bash
cd /home/dejan/Git/gapminer_v2
nohup setsid ./bin/phase0_scan_gpu --length 100000000000000 --threads 8 \
  --merit-min 15 --state data/p0_state_2e20_v2.txt --state-every 30 \
  --progress 60 --log data/p0_campaign_2e20_v2.log \
  > data/p0_campaign_2e20_v2.out 2>&1 &
tail -f data/p0_campaign_2e20_v2.out
```

Interrupted?  Run the same command again - it resumes.  Completed?  The state
file disappears; check `verification_failures=0` and the log row count in the
`.out`.

Next slice beyond the completed range (2e20+1e14 .. 2e20+2e14):

```bash
nohup setsid ./bin/phase0_scan_gpu --start 200000000000100000000000 \
  --length 100000000000000 --threads 8 --merit-min 15 \
  --state data/p0_state_2e20b.txt --state-every 30 --progress 60 \
  --log data/p0_campaign_2e20b.log > data/p0_campaign_2e20b.out 2>&1 &
```

**Walk-engine campaign (the fast path, 2026-10-03).**  For a slice, the walk
engine with `--gap-min G`, `G = ceil(15 x ln(start))`, is EXACTLY equivalent
to the sieve engine's `--merit-min 15` on that slice (compute the REAL ln:
`ln(2e20) = 46.7448`, so G = **702** at 2e20 and **696** at 1.33e20; a wrong
ln costs a probe above/below the gate, see `docs/PHASE0_scan_bench.md` §18).
Verified 2026-10-03 on a completed 1e14 slice: **653.5 s wall** (1.53e11
ints/s, 167,789 gaps, 0 verification failures, 8.5x the sieve engine's
1.545 h) with a full cross-engine audit - every one of the 16,213 rows of
the co-located sieve run is contained in the walk log and per-segment set
equality is exact.  The walk engine scans the range CONTIGUOUSLY - its
state file carries a single `off0`, so a resumed walk never leaves slice
holes, unlike the sieve engine, whose per-thread slices (`off0..offN`)
mean a partially done range is a set of short scanned segments:

```bash
nohup setsid ./bin/phase0_scan_gpu --engine walk \
  --start 133000000000000000000 --length 100000000000000 --gap-min 696 \
  --state data/p0_state_1p33e20_walk.txt --state-every 30 \
  --progress 60 --log data/p0_campaign_1p33e20_walk.log \
  > data/p0_campaign_1p33e20_walk.out 2>&1 &
```

Measured on the dev box: 1e12 at `gap-min 696` = 6.67 s (1.50e11 ints/s,
1688 gaps), a full 1e14 slice = **653.5 s** (10.9 min); VRAM = 2 x (K+1) x
33.6 MB -> **4.4 GB at the default K=64** (2.2 GB at K=32), no `--walk-batch`
needed in the recipe.  Audit an engine switch with `p0_setdiff`-style set
checks: every row of the older engine's log must be contained in the walk
log, and per-segment set equality must hold wherever both engines covered
the same integers; and cross-check the derived `--gap-min` against the
sibling engine's minimum logged gap before comparing sets.

Hygiene: stop any miner on the GPU before a timed run; use a fresh `--log`
per campaign (the tool appends headers, but per-file datasets keep the
analysis simple); keep `--state` file names unique per campaign geometry.

---

## 7. Data products

| path | contents |
|---|---|
| `data/p0_campaign_2e20.log` | completed 1e14 m>=15 dataset (169,276 rows; 0 verification failures) |
| `data/p0_campaign_2e20.out` | console transcript of the same campaign (bands, top-4, split) |
| `data/p0_hl_compare.png`, `data/p0_hl_compare_dark.png` | merit-tail vs HL-1t / HL-4p figure |
| `data/prime_gap_merits.txt` | best-known gap-length merit table used by the record check (123,436 lengths) |
| `p0.state` | leftover checkpoint of an aborted 6-thread attempt (~1-3% done, done Sep 30); not used by production runs |
| `docs/PHASE0_scan_bench.md` | full measurement history (sections 1-13), gates provenance, HL comparison, rejected experiments |

The white-paper Phase-0 corpus is the pair
`data/p0_campaign_2e20.log` + `.out`; anything else in `data/` belongs to the
miner.
