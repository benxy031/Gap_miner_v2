# Phase-0 Exhaustive Prime-Gap Scanner

Standalone toolkit for the white-paper Phase-0 workload: **exhaustive** scanning
of a decimal range for large prime gaps, with every reported gap re-verified by
GMP.  It is independent of the Gapcoin miner in this repo (no gapminer/
gap_hunt objects; its own kernels and tools).  Measurement history, theory and
model comparisons live in [`docs/PHASE0_scan_bench.md`](docs/PHASE0_scan_bench.md)
- this file is the operator manual (tools, flags, defaults, gates).

A self-contained, shareable copy of this toolkit (same layout, its own Makefile
and prose README) lives in [`phase0/`](phase0/README.md) - build and run from
there when the rest of the repo should not be distributed.

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
| `tools/mr128_kernel.cuh`, `tools/mr128_64_kernel.cuh` | | base-2 CIOS tests for 2^96 < n < 2^128 (4 x 32-bit and 2 x 64-bit limbs); the walk engine uses the 2 x 64-bit one at and above 2^120 |
| `bin/mr128_bench`, `tools/mr128_bench.cu` | | 128-bit test validator (`--validate`, GMP cross-check) + benchmark of the production kernel and its faster experimental variants |
| `tools/mr128_64_range.cpp`, `tools/mr128_64_sqr_range.cpp` | | host-only GMP harnesses for the 2 x 64-bit test and its square (the second one unit-tests the square against the validated generic multiply) |
| `tools/mr128_range.cpp`, `tools/perig_range.cpp` | | host-only GMP harnesses: MR verdicts vs `mpz_probab_prime_p` per bit length (`g++ -O2 -I tools <harness>.cpp -lgmp`; the headers define the CUDA qualifiers away when not compiled by nvcc) |
| `tools/p0_test_u96.cuh` | | OUR wall-band primality test (walk `P0_WALK_TESTMODE=4`): the same base-2 Euler/Jacobi predicate as perig for `2^95 <= n < 2^96 - 2^79`, as one fused column-wise 3 x 32-bit Montgomery square+reduce per step (`mad.lo.cc`/`madc.hi.cc` PTX); 1.70x perig standalone, 1.45x end to end at the wall band (§2) |
| `bin/bench_u96`, `tools/bench_u96.cu` | | gate + A/B of that test against perig and GMP: `gate` (exact `2^e mod n` vs `mpz_powm`, GPU and host), `verd` (verdict parity on consecutive odd n), `sweep` (parity across the whole domain incl. both limits), `time` (ns/test).  Build: `make bin/bench_u96 WITH_CUDA=1` (needs `-lgmp`); gates and example commands: §5 |
| `tools/p0_test_u108.cuh` | | Four-limb fused Montgomery Euler/Jacobi test, selected automatically for `2^105 <= n < 2^108` (mode 5); the existing test remains the reference (§2) |
| `bin/bench_u108`, `tools/bench_u108.cu` | | Dispatch boundary checks, GMP step/residue gates and perig verdict parity/timing for mode 5. Build: `make bin/bench_u108 WITH_CUDA=1` (needs `-lgmp`); acceptance commands: §5 |
| `scripts/p0_bench.py` | | fixed-range walk-engine bench: runs arms, parses wall/e2e/tests + the GPU-event stage split; `--bin-b` gives the ABBA order; results append to `data/p0_bench_results.txt` |
| `scripts/phase0_hl_compare.py` | | merit-tail comparison vs HL-1t / HL-4p models (figure) |
| `scripts/p0_walk_hl_compare.py` | | walk-scan census + first-occurrence comparison vs titanV's HL tables (figure) |
| `scripts/p0_rec_hl_compare.py` | | record-campaign census + record-yield comparison vs titanV's HL tables at x = 1e32 (figure) |
| `scripts/hl_validate_tables.py` | | out-of-sample test of titanV's HL tables at three scan scales, and what our data can/cannot improve in them (figure + CSV) |
| `scripts/psp_probe.py` | | pseudoprime risk of the census: Euler/Jacobi vs strong test on Carmichael numbers, and the surviving-pseudoprime density at a scan scale |
| `scripts/psp_hunt.py`, `scripts/psp_math.py` | | independent, resumable base-2 Fermat pseudoprime search and its shared exact test predicates |
| `scripts/p0_record_forecast.py` | | what records are still available at a fixed scan scale: deterministic winnable set, expected yield, durability |
| `scripts/p0_pgs_bench.py` | | head-to-head vs PGS (`briankehrig/prime-gaps-cuda`): same box, same start, identical ranges; steady + startup rates and a cross-tool gap-set audit |
| `scripts/p0_deep_sweep.py` | | walk-engine stage split vs sieve depth at a fixed band: e2e, mark/walk spans and the pacer per depth, two arms (production + mark-only), with a gap-set identity check; needs the GPU to itself, results append to `data/p0_deep_sweep_results.txt` |

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
                    [--walk-batch B] [--best N] [--recent N] [--smooth G,..]
```

Defaults:

| flag | default | notes |
|---|---|---|
| `--start DEC` | `200000000000000000000` (2e20) | any integer; the walk engine accepts up to 2^128 (auto-nudge over the 64-bit block window), the sieve engine up to 2^96 and - since the small-n MR branch (2026-10-18) - down to at least 100 (validated with `--check`, which now separates real defects from the base-2 strong pseudoprimes the MR kernel legitimately reports: 49 of them in `[100, 2e6)`, 2 in `[1000, 2e6)`, 0 at 1e7) |
| `--length DEC` | `10000000000` (1e10) | |
| `--sieve-limit P` | `1e7` (`--gpu-sieve`) / `3e4` (`--cpu-sieve`) | Inclusive upper bound for the sieve-prime table in `phase0_scan_gpu`; the standalone CPU scanner defaults to `1e8`. Startup reports odd primes through P and those above 13 after the 3..13 wheel. At P=1e7 these counts are 664,578 and 664,573. In walk mode the table is unused and the line says so; the separate walk table uses `--walk-primes`. Sieve marking uses primes 17..P; 1e7 is the measured optimum (deeper sieving loses: item count scales with pi(P)) |
| `--threads T` | `8` | one CUDA stream + one slice per thread |
| `--seg-bits B` | `24` (GPU) / `20` (CPU) | segment = 2^B integers; range 16..27 |
| `--batch K` | `4194304` (GPU) / `1048576` (CPU) | only used by `--legacy-mr`; range 1024..16777216 |
| `--merit-min M` | `20.0` | gaps >= M are written to the log and GMP-verified; bands/top-4/records still see every gap with merit >= 6 (internal gate) |
| `--engine sieve\|walk` | `sieve` | `walk` selects the class-30 bitmap sieve + batched jump-walk engine (~10.6x the sieve engine's throughput at the production default: measured 5.2 s vs 55.8 s for 1e12 integers, identical 5-gap output, 2026-10-02; same log/records/verify/top-4/resume path). `--check`, `--gpu-sieve`/`--cpu-sieve`, `--legacy-mr` and `--sieve-limit` apply to the sieve engine only; combining them with `--engine walk` is an error (or ignored where noted) |
| `--gap-min G` | `ceil(--merit-min * ln(start))` | walk engine only: report every gap >= G (GMP-verified). G in [100, 1e7]; values < 300 print a warning (walk cost scales ~1/G). Passing `--gap-min` explicitly removes the merit gate; an explicit `--merit-min` stays as an additional bar |
| `--walk-primes P` | `150000` | walk engine: sieve/item primes 17..P.  Deeper sieving cuts the number of primality tests (measured tests follow 1/ln P exactly) but grows the marking, so the optimum is **tile-dependent and band-dependent**.  With the shipped 92 KB tile the 7.9e28 campaign band measures **2.39 / 2.45-2.48 / 2.33 / 2.30e11 ints/s** for P = 120000 / **150000** / 180000 / 210000 (4e12 ints, `--gap-min 1664`, ABBA), i.e. the optimum moved 100000 -> 150000 when the mark's per-item cost fell (§31).  With the old 64 KB tile the same band measured 2.059 / 2.237 / **2.397** / 2.247 / 1.926 / 1.497e11 for P = 15000 / 40000 / **100000** / 250000 / 500000 / 1e6 (§28).  At shallow starts a smaller depth wins: at `--start 1e20` 40000 gives 4.03e11 vs 3.87-3.95e11 at 150000, and 15000 is the optimum at the 1e13 campaign geometry, so pass `--walk-primes 40000` (or 15000) for shallow bands.  Gap sets are identical at every depth (verified 87034/87034 gaps at `--gap-min 800`, P = 40000..1e6, and 1696/1696 walk-vs-sieve at P = 150000).  **Re-swept 2026-10-30** with the wall-band test (`P0_WALK_TESTMODE=4`, below) and the chunk-major mark order (`P0_ITEM_ORDER`, below), `--start 7.9e28 --length 4.8e12 --gap-min 1664`, dev 3070, 3 interleaved reps: P = 70000 / 85000 / **100000** / 120000 / 150000 -> **13.30 / 12.85 / 12.70 / 12.85 / 12.99 s**.  A cheaper test moves the optimum back toward a shallower sieve: 100000 is -2.2 % against the default and the curve is flat within 1.5 % over 85000..150000 (earlier 2-rep sweep: 40000 13.9 s, 60000 13.6 s, 80000 13.31/12.27 s - an outlier pair, hence the 3-rep round above - 220000 13.3 s, 320000 13.9 s).  **Fleet RTX 3060s, measured 2026-10-30** (same band, `--length 4.8e12 --gap-min 1664 --walk-batch 96`, both cards at once as in production, 4 reps ordered F R R F, mean wall s GPU 0 / GPU 1): P = **85000 21.79 / 21.35**, 100000 21.94 / 21.49, 120000 22.04 / 21.55, 150000 22.34 / 21.89, 200000 22.79 / 22.31, 250000 23.36 / 22.91, i.e. **-2.5 % at 85000** against 150000 on both cards (100000 -1.8 %, 120000 -1.5 %, 200000 +2.0 %, 250000 +4.6 %; the two cards agree within 0.3 points).  85000 is the lowest depth tried there, so the 3060's optimum may lie below it.  Gap sets stay identical (428/428 gaps at `--length 1.2e12 --gap-min 1100`, 85000 vs 150000, both cards).  **The engine default is unchanged** (150000: the old binary and other bands prefer it); the fleet wall chain passes `--walk-primes 85000` explicitly (see the chain recipe in §6; a slice resumes from its checkpoint at the new depth) |
| `--walk-batch B` | `64` | walk engine: super-batch = B x 30-blocks (**default 64 since 2026-10-03**, was 32; range **4..96**, raised from 4..32 the same day). ABBA on 1e13 at `--gap-min 702` (dev 3070, 4 arms): K32 63.68 s vs **K64 62.12 s = +2.45 %**, K96 62.26 s (+2.23 %, no better than 64 but 6.5 GB) - at the 7.9e28 wall with the 92 KB tile K96 measured 16.12/16.33 s vs 16.36/16.94 s for K64 (**+2.5 %**, both pairs), still opt-in because of the VRAM; both K64 arms beat both K32 arms and the emitted sets are identical (parity K32==K64==K96 exact, 1624/1624 on 1e12). VRAM = 2 regions x (B+1) x 33.6 MB + gap buffer: default K=64 -> 4.4 GB, K=32 -> 2.2 GB, K=96 -> 6.5 GB (an over-large K fails the bitmap alloc loudly, never silently) |
| `--log FILE` | none | appended; one `# phase0-gpu session ...` header per run |
| `--state FILE` / `--state-every S` | none / `30` s | checkpoint for resume; **deleted when the range completes** |
| `--progress S` | `10` s | progress + ETA line every S seconds (`0` = off); after a `--state` resume the line's `ints/s` and the final `end_to_end` cover only the current session (the `%` and `ints=` stay whole-slice) |
| `--device D` | `0` | CUDA device |
| `--records FILE` / `--no-records` | `data/prime_gap_merits.txt` if present | record check: gap length in table AND merit > best known; a beat is logged even below `--merit-min` |
| `--best N` | `10` | how many of the STRONGEST gaps the run found are listed as `top#k` rows in the results (merit order; range 1..32). Report-only: no band, log line, test or record decision depends on it. `--best 12` lists 12 |
| `--recent N` | `10` | how many of the LAST REPORTED gaps (the ones that passed `--merit-min`) are listed as `recent finds`, oldest first; range 0..64, `0` disables the section. Costs one mutex per reported gap (~30 locks/s at `--merit-min 15`, i.e. invisible). `--recent 20` lists 20 |
| `--smooth G,..` | off | sieve engine only: smooth-run census (longest run of consecutive gaps `<= g`) for each bound; bounds even, 2..1e6, at most 128 entries. Rejected with `--engine walk` (it only reports gaps `>= --gap-min`, so it cannot see the small gaps a census needs), with `--no-test`, and with `--check`. With the flag off the output is byte-identical to a build without it (verified: same 21,232-row gap log; docs §30) |
| sieve mode | `--gpu-sieve` | `--cpu-sieve` selects the (slow) CPU marking path |
| MR pipeline | bitmap-fed (fast) | `--legacy-mr` selects the batched steps/res path (A/B reference; `--check` always uses it) |
| `--check [N]` | off | sieve+MR `[start, start+N)` then re-walk the range with `mpz_nextprime`; N default 2e6. Forces the legacy MR path (needs the collect list) |
| `--no-test` | off | sieve + walk only (no MR, no gap detection); not allowed with `--gpu-sieve` |

Environment: `P0_FAST_DBG=N` dumps the first N fast-walk summaries
(word counts, prime count, first/last set slot) to stderr - the tool that
isolated the cross-segment chain bug in the bitmap walk.  `P0_SG_GRID=N`
(walk engine) sets the mark-kernel CTA count under the P0_PIPE overlap,
default `92` (= the pre-pipeline full wave; range 1..4096).  Measured on the
2e13 arm at `--gap-min 1586`: 92 -> 58.9 s, 46 -> 58.6 s, 23 -> 75.6 s (too few
CTAs - the mark stage becomes the pacer).  Keep the default.
`P0_TILE_SLOTS=N` (walk engine) overrides the mark tile in bitmap slots
(64 slots = 512 B; range 131072..753664, i.e. 8..92 KB of dynamic shared).  The
default is **the largest tile the device can opt into** (92 KB on sm_86, capped
by `cudaDevAttrMaxSharedMemoryPerBlockOptin` minus 4 KB of static shared; 64 KB
on devices that cannot opt in past it), because the item table holds one row per
(prime, 256-mark chunk) per tile and that row count is the mark's dominant cost:
92 KB vs 64 KB cut the isolated mark 5.02 -> 4.58 s and 6.58 -> 5.98 s per 4e12
ints (P = 100000 / 250000) with an identical gap set (8753/8753 rows,
`tests=896651987` identical).  A partial last tile is handled by the kernel's
`lastSlots` clamp (block = 2^28 slots does not divide by 753664).
`P0_ITEM_CHUNK=N` (walk engine) sets how many marks one item-table row covers,
default `256` (rounded down to a multiple of 8; range 8..4096).  Together with
`P0_TILE_SLOTS` it fixes the row count, which is the mark's dominant cost.
Re-swept 2026-10-27 at the campaign band (`--start 1e32 --length 2e12`,
`--gap-min 2002`, dev 3070, wall per arm): chunk 64/128/**256**/512/1024/2048 ->
8.80 / 8.63 / **8.66** / 8.80 / 9.04 / 8.82 s, and tile
131072/262144/524288/**753664** -> 10.81 / 9.60 / 9.20 / **8.61** s, so both
defaults still sit at the optimum.
`P0_ITEM_ORDER=0|1` (walk engine) is the row order of the mark kernel's item
table when it has more than `P0_ITEM_STRIDE_CUT` = 5000 rows (`tools/p0_types.h`;
the warp-strided path - every production depth: 16599 rows at P = 150000, 12343
at 100000), default **`1`** = chunk-major (every prime's chunk-0 row, then every
prime's chunk-1 row, ...); `0` = the historical prime-major order.  A warp takes
32 CONSECUTIVE rows.  In prime-major order those are consecutive chunks of ONE
prime, whose first marks sit `256 * p` bit positions = `8 * p` words apart, so
they fall on only 4 of the 32 shared-memory banks and every `atomicOr` of a
small-prime row (the primes with many chunks per tile) is an up-to-8-way bank
conflict; chunk-major puts 32 different primes in a warp.  The kernel is
unchanged (the OR-union of the marks does not depend on the row order), so the
bitmap, the `tests` count and the gap set are identical.  Measured 2026-10-30
(dev 3070, `--start 79000000000010000000000000000 --length 4.8e12`,
`--walk-primes 150000`): mark alone (`P0_WALK_OFF=1`, ABBA) **5.69 -> 4.71 s
(-17 %)**; end to end 13.35 -> 12.98 s (**-2.8 %**, 3 interleaved reps, with
`P0_WALK_TESTMODE=4`) - the overlap hides most of the mark, so only about a third
of the gain reaches the wall; `tests=9505435886` in every arm.  At the dev 1e32
campaign geometry (perig test, `--gap-min 2002`, `--length 4.8e12`, ABBA) it is
neutral, 19.45 -> 19.28 s (-0.9 %, within noise), with identical `tests` and gap
sets (185/185 at `--length 2e12 --gap-min 1300`).  Example, A/B
against the old order:

    P0_ITEM_ORDER=0 ./bin/phase0_scan_gpu --engine walk \
        --start 79000000000010000000000000000 --length 4800000000000 \
        --gap-min 1664 --walk-primes 150000 --walk-batch 96 --no-records

An out-of-range value prints `P0_ITEM_ORDER N out of range [0,1]; using 1` on
stderr; the table builder aborts with `walk item table size mismatch` if the row
count and the table ever disagree.
`P0_WK_GRID=N` / `P0_WK_BLOCK=N` (walk engine) are the walk kernel's CTA count and
threads per CTA, defaults `46` / `512` (ranges 1..4096 and 32..1024).  They are a
measured optimum: same arm, 46x512 -> 8.67 s against 92x512 9.40 s, 138x512
9.60 s, 46x256 9.61 s and 46x1024 9.80 s.  The mark CTA count `P0_SG_GRID` is
flat there (46/92/138/184/276 -> 8.64..8.84 s), so under `P0_PIPE` the mark is no
longer the knob it was pre-pipeline - keep its default too.
`P0_WALK_PROF=1` (walk engine) adds per-thread `clock64` counters and prints the
walk's scan/test split.  At the campaign band it reads **scan 1.8 % / test
98.2 %**: the walk is arithmetic, not bitmap scanning.

**Ceiling of the arithmetic avenue, measured 2026-10-27 - CORRECTED 2026-10-30**
(start = 10^32, length = 2e12): the walk is 91.8 % of the wall and 98 % of the
walk is the test, and the MARK stage SPAN is 98.5 % of the wall because the two
streams overlap 1.8x.  The 2026-10-27 reading of that - "`wall/mark` = 1.5 % is
the whole prize for making the walk free" - was **wrong**: the span of the mark
inside the overlapped run is stretched by the walk that shares the SMs, it is not
what the mark costs.  Measured standalone (`P0_WALK_OFF=1`) the mark is **2.11 s
of the 8.5 s wall** at this band (full wall 8.54 / 8.51 s, `tests=3634782921`,
`--gap-min 2002`) and 4.71 s of 12.98 s at the 7.9e28 wall band, so the walk's
test IS the critical path and a cheaper test pays: the wall-band test (mode 4,
below) took 19.36 -> 12.98 s per 4.8e12 (**1.49x** with the mark order).  At
1e32 (n = 2^106) mode 4 does not apply, but the four-limb mode 5 below now
addresses that band. What stands: the device-resident CUDA-graph rewrite of the scan loop
(a scheduling change) was specified, measured and **rejected** (an nsys trace at
2e28 shows no GPU idle time for a graph to remove: the kernels cover 7.581 s of
their 7.582 s span), and PGN68's 2.18x
is not a scheduler win - it is per-bit-width arithmetic: see
`docs/PHASE0_pgn_lever_verdict_2026-10-26.md` (§42, corrected there) and
`docs/PGN_SNAPSHOT_REVIEW_2026-10-20.md` (§12).

Diagnostics for cost isolation (walk engine, docs §32) - **never production
settings**:

* `P0_WALK_OFF=1` skips the walk launches, so the run measures the mark stage
  alone (the gap log stays empty; `tests=0`).
* `P0_MARK_NOSTORE=0|1|2|3` picks the mark's tile-store shape at compile time:
  `0` = production (per-tile 32 MB/block DRAM stream), `1` = no store,
  `2` = every store on one address, `3` = every tile into the same L2-resident
  region.  Modes 1..3 do not write the bitmap, so the binary refuses them
  without `P0_WALK_OFF=1`.  Measured (4e12 ints, P = 150000, mark-only):
  0 -> 4.91 s, 3 -> 4.73 s, 2 -> 4.52 s, 1 -> 4.51 s, i.e. the tile stream costs
  8.1 % of the mark.
* `P0_WALK_TESTMODE=0|1|2|3|4|5` forces the walk's primality variant instead of the
  magnitude dispatch: `0` = perig (2x64 CIOS Montgomery), `1` = exact lazy 2x64,
  `2` = canonical 2x64, `3` = our three-limb lazy-Barrett test, `4` = our fused
  column-wise three-limb Montgomery test (`tools/p0_test_u96.cuh`, exact only for
  `2^95 <= n < 2^96 - 2^79`; forced outside that range it prints a one-time
  `P0_WALK_TESTMODE=4 only covers Ahi in [2^31, 2^32-2^15); Ahi=N uses the
  magnitude dispatch` on stderr and falls back to the dispatch); `5` selects the
  four-limb test (`tools/p0_test_u108.cuh`) only for
  `2^105 <= n < 2^108`, with the same guarded fallback outside that domain.
  The default magnitude dispatch selects mode 5 within that domain.
  Modes 0..2 live
  in the same 128-bit container (perig is exact to 2^120, the lazy and canonical
  variants past it), so their band boundaries are COST choices, not correctness
  ones - the three agree on the emitted gap set at 1e32 (`--gap-min 1300`:
  185/185 gaps in every arm).  Measured at start = 10^29 for 0/1/2: perig
  16.36/17.18 s vs lazy
  19.32/21.56 s vs canonical 24.75/25.14 s with `tests` and the gap set
  identical, so perig is the fastest of those three; re-measured at the campaign
  band (start = 10^32, length = 2e12, `--gap-min 1300`): perig **12.86 s** < lazy
  16.00 s < canonical 18.40 s.  Modes 3 and 4 at
  start = 10^22 (hi = 542) and at the 7.9e28 wall band: see the dispatch
  default below.
* Walk test dispatch (default, no environment variable): `n` under 2^70 ->
  perig, `2^70 .. 2^75` -> mode 3, `2^75 .. 2^95` -> perig, `2^95 ..
  2^96 - 2^79` -> **mode 4**, `2^96 - 2^79 .. 2^105` -> perig,
  `2^105 .. 2^108` -> **mode 5**, `2^108 .. 2^120` -> perig, `2^120 .. 2^126`
  -> mode 1, above -> mode 2 (all upper limits exclusive). The boundaries
  protect correctness assumptions (mode 3's five-word square needs a < 2^80,
  mode 4's lazy representative must fit three limbs, mode 5's proven bound is
  stated below, perig is exact to 2^120, mode 1 to 2^126). The mode-5 range is
  the verified domain, not a claim that extension is impossible. Lower bounds
  include cost and schedule choices: 2^70 for mode 3 is a measured crossover (our
  test's reciprocal costs more up front than perig's constants, so it only pays
  once the exponent is long enough) and 2^95 for mode 4 is the range that was
  gated and timed (its fixed 95-bit exponent schedule is exactly full there;
  below it the verdicts still matched perig - 2M consecutive odd n at each of 8
  high words from 2^70 to 2^95, 0 differences - but squarings are wasted, so it
  is not selected there), and 2^105 for mode 5 bounds its 104-squaring schedule
  and lazy representative. Example:
  `./bin/phase0_scan_gpu --engine walk --start 10000000000000000000000 --length 2000000000000 --merit-min 20`
  runs mode 3 without any variable set.
  **Measured 2026-10-26** (dev 3070, start = 10^22, length = 2e12, `--merit-min 20`,
  2 interleaved reps): perig 9.07/9.00 s vs mode 3 8.44/8.40 s = **+7 % end to
  end**, with a **bit-identical gap set** (22,939 gaps at `--merit-min 12` and
  1,826,878 at `--merit-min 8` both diff to zero on (lower, gap, upper)).  The
  +7 % is smaller than the test's own 1.23x because the test is only part of the
  walk: at this band mark = 64.7 % and walk = 61.7 % of the wall (they overlap).
  **Mode 4, measured 2026-10-30** (dev 3070, `--start 79000000000010000000000000000
  --length 4800000000000 --gap-min 1664 --walk-primes 150000 --walk-batch 96`,
  3 interleaved reps, same binary): perig (`P0_WALK_TESTMODE=0`, `P0_ITEM_ORDER=0`)
  **19.36 s** -> mode 4 **13.35 s** (**1.45x**) -> mode 4 + the chunk-major mark
  order (the new default) **12.98 s** (**1.49x** in total); `tests=9505435886`
  and zero gaps in all nine runs.  At `--gap-min 1100` (same range, one run per
  arm) 27.88 -> 18.28 s (1.52x) with 1694 gaps and `tests=14681132483`
  identical.  Standalone (`bin/bench_u96 time 1048576 200`, 2^20 consecutive odd
  n at Ahi = 4282402316): **1.70 - 1.72x** (0.91 vs 1.57 ns/test at 1920 MHz; the
  absolute ns drift with the clock/thermal state, the ratio does not), 26 vs 38
  registers, 43 vs ~96 SASS instructions per modular squaring.  Why it is faster:
  one fused column-wise square + Montgomery
  reduction per step, every product accumulated with the PTX triple
  `mad.lo.cc / madc.hi.cc / addc` that ptxas fuses into a single in-place
  `IMAD.WIDE.U32`, and the last squaring of each 4-bit exponent window folds the
  window's multiply-by-2^v into the Montgomery constant, so 92 squarings and no
  multiplications.  The code is ours (the PGN snapshot has no license and none of
  it is used).  A/B on any range in the band (both print the same `tests=`):

      ./bin/phase0_scan_gpu --engine walk --start 79000000000010000000000000000 \
          --length 4800000000000 --gap-min 1664 --walk-primes 150000 \
          --walk-batch 96 --no-records                      # mode 4 (default)
      P0_WALK_TESTMODE=0 P0_ITEM_ORDER=0 ./bin/phase0_scan_gpu --engine walk \
          --start 79000000000010000000000000000 --length 4800000000000 \
          --gap-min 1664 --walk-primes 150000 --walk-batch 96 --no-records   # old

  Verification (2026-10-30): `bin/bench_u96 gate 2000000` - the GPU residue
  `2^((n-1)/2) mod n` equals GMP `mpz_powm` on 2,000,000 random odd n of the
  domain plus 14 hand-picked edge cases (host path and verdicts: the first
  200,000), 0 mismatches; verdict parity with perig on 4,294,967,296 candidates
  over 4096 random high words (`sweep 4096 1048576`: 129,712,962 primes in both,
  0 differences) plus, in a second run, 1e9 consecutive odd n at the wall band
  and 3e8 at each domain limit (Ahi = 2^31 and 2^32 - 2^15 - 1), all identical;
  identical gap sets/test counts end to end; the split build
  (`phase0_scan_gpu_split`, the Windows-DLL source) gives the same 428 gaps /
  `tests=3671852807` as the monolithic one at `--gap-min 1100 --length 1.2e12`.
  Mode 4 itself does not cover the 1e32 campaign band.
  Fleet re-validation (2026-10-30, 2x RTX 3060 sm_86, binary rebuilt there with
  plain `make`, GCC 13): `bin/bench_u96` `gate`/`verd`/`sweep` 0 mismatches and
  the same prime counts as on the dev box; `--length 1.2e12 --gap-min 1100`: old
  (`P0_WALK_TESTMODE=0 P0_ITEM_ORDER=0`) and new both `tests=3671852807`, 428 gaps,
  identical gap sets on both cards, wall 13.57 -> 8.75 s (GPU 0) and 13.61 -> 8.57 s
  (GPU 1), 1.55-1.59x.  Resumed from the real wall checkpoints (c34 / c35) the
  end-to-end rate went from 1.429 / 1.457e11 ints/s (old binary) to 2.179 / 2.214e11
  at the old depth (**1.52x**) and to 2.243 / 2.276e11 at `--walk-primes 85000`
  (**1.57x**, first progress line, start-up included), ETA 11 h -> 7 h.

  **Mode 5, measured on the dev RTX 3070** (`--start
  100000000000000000000000000000000 --length 2000000000000 --gap-min 2002
  --walk-primes 150000 --walk-batch 96 --no-records`, interleaved ABBA):
  perig **8.27/8.22 s** vs mode 5 **6.74/6.69 s**, 18.6% less wall time
  (22.8% greater throughput), with `tests=3634780774` in all arms. At
  `--gap-min 1300` the reference and default took 12.46 and 9.97 s; default,
  forced mode 5, the split build and the profiled build all matched the
  reference's **185 GMP-verified gaps** and `tests=5711269146`. Standalone,
  the warmed test was about 1.29-1.32x perig at both ends of the band and at
  1e32. These are dev-box results, not fleet validation. The same 1e32
  `gap-min=2002` depth sweep gave mean 6.705 s at 100000 and 6.745 s at 150000
  (two forward/reverse repetitions), too close to change the default.

  The new test uses four 32-bit limbs and `R=2^128`; its 104-squaring window
  schedule implements the same base-2 Euler/Jacobi predicate as perig, not a
  deterministic primality proof. For `2^105 <= n < 2^108`, a step with
  `x < 2^109`, window value `v <= 15` and `0 <= Q < 2^(128-v)` gives
  `(x^2 + Q*n)/2^(128-v) < n + 2^105 <= 2n`, so its lazy representative
  stays within the stated bound and needs at most one final subtraction.
  `bin/bench_u108 step 200000` checked both step variants and their bounds
  against GMP; `gate 2000000` checked exact GPU/host residues and verdicts
  against GMP; `sweep 1024 1048576` plus `verd 300000000` at each domain
  boundary and at the campaign high word compared **1,973,741,824** verdicts
  against perig. All differences were zero. End-to-end boundary runs also
  checked the guarded fallback and the prior mode-4 regression.
* Kernel-DLL builds (`phase0_scan_gpu_split`, and on Windows `phase0gpu.dll` +
  the MinGW host tools) pick the test from the MAGNITUDE only:
  `P0_WALK_TESTMODE` is a monolithic-build variable.  `p0gpu_walk` must carry a
  `case` for every mode `w_test_mode()` can return - mode 3 was missing there
  while the monolithic path had it, so the split build silently ran perig below
  2^75 (4.19 s vs 3.66 s at start = 10^22, length = 5e11, and the gap set is
  identical either way, so only the timing exposed it).  Both dispatch sites are
  now in sync (including cases 4 and 5); `bin/phase0_scan_gpu` and
  `bin/phase0_scan_gpu_split` agree to
  the bit (22,939 gaps, 3.66 / 3.64 s).

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

Overlap (P0_PIPE, 2026-10-05): the walk result of each super-batch is
collected one iteration late, so the sieve of batch j+1 really runs while the
walk of batch j is in flight (before the change the host synchronized right
after launching the walk and the stages were strictly serial: measured
mark 34 % + walk 66 % = 99.5 % of the wall).  Measured at the campaign
geometry (start 1.3300107e20, 2e13 ints, `--gap-min 1586`, K=64):
65.1 s -> 58.1-58.9 s = **+9..11 %** (3.06e11 -> 3.38-3.44e11 ints/s; the
drift-immune ABBA run reads +8.5 %, `docs/PHASE0_scan_bench.md` §20), and the
emitted sets and `tests` count are identical to the pre-P0_PIPE binary (182,535 rows
and tests=6,029,529,566 on the 1e12 `--gap-min 500` gate).  The collect copies
run on a private stream: a synchronous `cudaMemcpy` in the legacy default
stream implicitly synchronizes with every blocking stream and would wait for
the next batch's marks and walk, re-serializing the pipeline (measured: the
first P0_PIPE version still showed 100 % serial for exactly this reason).
Stop/resume is unchanged - state is written from the deferred collect, and a
stop drains the in-flight batch first.

Mark item-loop (2026-10-05, later): two changes in the class-30 sieve's item
loop, both bit-identical in output - an **early break** when a mark chunk
passes the tile end (positions are monotone; the loop used to walk all 64
guarded iterations) and **item chunks of 256 marks** instead of 64 (4x fewer
item rows; `offset = C*p*k0` in builder and kernel together).  Measured at
the campaign geometry (2e13, `--gap-min 1586`, K=64): 57.5 s -> 50.9 s =
**another -11.5 %** (3.48e11 -> 3.93e11 ints/s; ABBA reps=2, mark span
58.1 -> 50.0 s, walk unchanged, `tests` identical 35,646,342,874; sorted
gap-set parity on 5 geometries + P-invariance + split parity all green).
`--walk-primes 15000` is the optimum at THIS campaign geometry within noise
(60000 -> 29.4 s vs 25.9 s at 1e13, old binary); at the 2^96 container wall the
optimum moved to 40000 after the 2026-10-08 audit - see § 26.5 and the
`--walk-primes` default row above - see `docs/PHASE0_scan_bench.md` §22.

Item loop (2026-10-06): two changes in the class-30 sieve's item loop, both
bit-identical in output.  (a) The item table is read UNCOALESCED once it
outgrows the L1: with per-thread contiguous slices a warp's 32 lanes sit
`nitems/nthreads` items apart (157 rows = 628 B at `--walk-primes 1e6`), so
every read was a 32-sector fetch with L2 latency behind it.  Above
`P0_ITEM_STRIDE_CUT` (5000 rows, ~40 KB) the loop now switches to a
**warp-strided** walk that reads 32 consecutive items per load: mark at
P=1e6 251.6 s -> **81.8 s**, P=150000 50.8 -> 25.7 s.  (b) A prime's eight
step deltas are **table-driven** (an 8x8x8 bracket table built in shared
memory once per CTA) instead of ~24 arithmetic ops per step, which also pays
for the production depth: mark solo 15.4 -> **14.6 s** at the default
`--walk-primes 15000`.  The wall is unchanged at the default (ABBA reps=2,
2e13, g1586: 51.28 s vs 51.00 s = -0.56 %, inside the +-2.3 % spread), and
the *cost of depth* collapsed - the depth curve is flat 15000..30000 and the
walk side's 1/ln P tail (~40 s of walk span at 60000) is now reachable.
Gates after the change: walk fixture 22/22 rows, 1e12 `--gap-min 500` sorted
gap-set parity 201,575 rows, split-vs-main parity 201,575 rows, `--check
200000`/`2000000` and `bench_p0sieve --verify 3` all green.  See
`docs/PHASE0_scan_bench.md` §23.

Reproduce: `python3 scripts/p0_bench.py --bin-b <pre-P0_PIPE binary>
--length 2e13 --gap-min 1586 --label P0_PIPE` (~8 min ABBA; raw lines in
`data/p0_bench_results.txt`).

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

Container and constraints.  The walk path is exact up to
`start + length <= 2^128` (its perig Fermat test covers <= 2^120, the 4 x 32-bit
`tools/mr128_kernel.cuh` covers 2^120..2^128; see `docs/PHASE0_scan_bench.md`
§ 25).  The kernel builds each candidate as a 64-bit low word plus a constant
high word, so the aligned block base must keep `(blocks + 2) * 1.006e9`
numbers inside one 64-bit window; a start that violates it - every power of two
does, e.g. 2^100 -> base 2^100-16 -> low word 2^64-16 - is advanced to the next
class-30 base past the wrap and reported on stdout (shift < (blocks + 2) *
1.006e9 numbers, < 0.02 s of scan time; the effective start is what the log
header and the state file carry).  Within ~3.1e9 numbers of 2^128 the shift
cannot fit and the tool says so.  `--start` must exceed `--walk-primes`.  One worker only (the sieve engine's `--threads`
does not apply); `--state`/`--state-every`/`--progress` work as usual.
Larger super-batches (fewer event syncs, more VRAM) and a deeper sieve:

```
./bin/phase0_scan_gpu --engine walk --walk-batch 96 --walk-primes 150000 \
    --start 200000000000000000000 --length 64426429440 --log data/p0_walk_b96.log
```

VRAM = 2 x (B+1) x 33.6 MB: B=96 -> 6.5 GB, B=64 -> 4.4 GB, B=32 -> 2.2 GB
(an over-large B fails the bitmap alloc loudly).  At the 7.9e28 wall B=96 is
+2.5 % over 64; the shipped `scripts/p0_walk_chain.py` therefore defaults to 96
and accepts `--walk-batch 64` for cards with less than ~7 GB free.

### Output

Progress/console (stdout, line-buffered): banner, then every `--progress` seconds
one status line and, for each gap above `--merit-min`, a `GAP ...` line:

```
[phase0-gpu]  42.15%  ints=1.234e+12  1.75e+10 ints/s  u=3.51%  tests=...  primes=...  gaps=...  batches=...  eta=0:52:34
[phase0-gpu] GAP merit=16.0445 gap=750 lower=... upper=... verified=1 record=no table=22.9982
```

and, once in the `--- results ---` block, a fixed-shape report (also printed by
`--engine walk`):

```
[phase0-gpu] target: merit >= 20.00  (gap >= 936 at ln(start)=46.7448)
[phase0-gpu]   top#1 gap=724 merit=15.4883 lower=... upper=... verified=1   (x --best)
[phase0-gpu] recent finds (last 10 reported gaps, oldest first):              (x --recent)
[phase0-gpu]   gap=630 merit=13.4774 lower=... upper=...
```

The `target:` line restates the merit gate as the smallest **even** gap that can
meet it at this start (gaps between odd primes are even).  `ln(start)` is
constant over any realistic range (1e14 at 2e20 moves it 5e-7 relative), so this
is one scalar per run, not a per-gap cost.  Under `--engine walk` it is printed
from `--gap-min` instead (e.g. `--gap-min 700` -> `merit >= 14.97`).

The monitor thread prints exactly this status line; end of run:

```
[phase0-gpu] wall=... end_to_end=... ints/s
[phase0-gpu]   (session start offset <N> ints = <X>% of the slice; the rate
                above covers only this session's <M> ints)
[phase0-gpu] survivors=... u=...% of ints
[phase0-gpu] fast split (GPU-s sums): wheel+mark=... MR=... copies=...
[phase0-gpu] gaps per merit threshold (this session -> per 1e14 ints):
[phase0-gpu]   m>=10 :  ...  ->  ...
[phase0-gpu]   top#1..4 gap/merit/lower/upper/verified
[phase0-gpu] gaps reported=...  verification_failures=0  records_new=...
```

Walk runs print one extra summary line before the band table:
```
[phase0-gpu] gpu batches=...  tests=...  tests_per_wall=... tests/s  tests/jump=...
[phase0-gpu] stage SPANS (GPU events): mark=... (..%)  walk=... (..%)  other=...
```

Two readings that are easy to get wrong, and how to get them right:

- **`tests_per_wall` is `tests / TOTAL wall`**, so the mark time is inside it. It
  is *not* the walk kernel's own test rate. For the walk's own rate, measure the
  mark-only wall (`P0_WALK_OFF=1`) and use
  `tests / (wall - mark_only_wall)`. At 1e32 / `--walk-primes 150000` the two
  differ by ~1.3x (docs/PHASE0_scan_bench.md section 43).
- **`stage SPANS` are spans, not cost shares.** Under `P0_PIPE` both stages run
  concurrently, so each span also covers the other stage's time: both can read
  ~100 % of the wall and `other` goes negative by exactly the overlap. Never
  read those percentages as where the time goes. The honest split comes from
  two independent measurements that agree to 3 %: the mark-only wall
  (`P0_WALK_OFF=1`) and the intercept of a `wall = a + b/gapmin` fit over
  several `--gap-min` values, where `a` is the mark and `b/gapmin` the walk
  (docs/PHASE0_scan_bench.md section 42).

Both lines were relabelled on 2026-10-27 after the span numbers had been read as
shares and produced a wrong conclusion ("the mark is 94-98 % of the wall") that
stood for a month; the true split at 1e32 is mark ~24 %, walk ~76 %.

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

### Pseudoprimes and Carmichael numbers (completeness doctrine)

```
scripts/psp_probe.py                     # all three blocks, defaults (~5 s)
scripts/psp_probe.py --fixtures          # only the two known gaps (0.2 s)
scripts/psp_probe.py --carmichael        # only the test-strength block
scripts/psp_probe.py --density [--x 1e32] [--sieve 150000] [--odds 2000000]
scripts/psp_probe.py --brik          # score tests vs BrianK's published list
scripts/psp_probe.py --sieve-class   # what a plain sieve of depth S leaves behind
```

`psp_hunt.py` is a separate PSP search, not a Phase-0 campaign. It requires
Python 3 and `gmpy2` on a POSIX system and records results/checkpoints in SQLite:

```
python3 scripts/psp_hunt.py pairs --p-start 10000000001 --p-end 10000003001 \
    --db data/psp_pairs.sqlite3
python3 scripts/psp_hunt.py pairs --p-start 10000003001 --p-end 10000006001 \
    --db data/psp_pairs_next.sqlite3  # adjacent factor slice, new database
python3 scripts/psp_hunt.py show --db data/psp_pairs.sqlite3 --min-gap 100
python3 scripts/psp_hunt.py chernick --k-end 1670000 \
    --db data/psp_chernick.sqlite3
python3 scripts/psp_hunt.py scan --start 160696461657079906273 \
    --end 160696461657079906274 --db data/psp_control.sqlite3
```

| option | effective default | meaning |
|---|---|---|
| subcommand | required | `pairs`: semiprime-factor slice; `chernick`: a three-factor family; `scan`: generic interval; `show`: stored results |
| `--db FILE` | required | SQLite results/checkpoint; resume with identical search settings |
| `--start N`, `--end N` | `1e21`, `2e21` (`pairs`); `1e21`, required (`scan`) | inclusive/exclusive number range; decimal and decimal scientific notation are exact |
| `--p-start P`, `--p-end P` | required (`pairs`) | inclusive/exclusive range for the smaller prime factor |
| `--k-start K`, `--k-end K` | `1666667`, required (`chernick`) | inclusive/exclusive range of family parameters |
| `--factor-floor D` | `10000000` | confirmed hits have no prime factor <= D |
| `--batch-size N` | `10000` (`pairs`, `chernick`); `100000` (`scan`) | work per committed checkpoint |
| `--max-batches N` | `0` | no extra limit; a positive N pauses after N committed batches |
| `--min-gap G` (`pairs`) | `0` | retain/print only PSPs in gaps >= G; positive G is part of the database identity |
| `--min-gap G`, `--limit N`, `--status S` | `0`, `20`, `confirmed` (`show`) | filter displayed gaps, row limit (`0` = all), and `confirmed`, `pending` or `all` |

`pairs` exhausts only semiprimes (including squares) in its factor slice and
product interval; the example is only a small slice. Complete semiprime
coverage in `[1e21, 2e21)` with the default floor requires contiguous
smaller-factor slices covering `[10000001, 44721359550)`. `--p-start` must
exceed `--factor-floor`, and each new slice uses a new database. `chernick`
covers only the specified family. Neither covers all composites above `1e21`,
where three or more factors above the floor are possible. The generic `scan`
checks all eligible odd integers in its interval but is much slower.

The prime-factor table is capped at `25000000`, limiting `scan`'s factor floor
and the table needed for `pairs`. The `pairs` and `chernick` modes reject
factor ranges outside the exclusive deterministic MR12 bound
`318665857834031151167461`
([Sorenson and Webster](https://arxiv.org/abs/1509.00864)).
`scan` retains unresolved Fermat passers at or above that bound as `pending`.
Gaps entirely below it are checked across the interior and labelled
`proven_mr12`; those beyond it are labelled `probable_gmp`. Confirmed JSON
records include the test outcomes, proven factors or a compositeness witness,
and the surrounding gap.

Each checkpoint atomically commits progress and results; Ctrl-C or SIGTERM
stops safely, and the same search settings resume from the checkpoint. Batch
size and batch limit may change on resume. Incomplete work or unresolved
candidates returns exit 4; invalid input or database errors return 2, and an
interrupt returns 130 or 143. A hit is not claimed to be novel, and none of
these searches touches the dev or fleet campaign.

`pairs --backend cuda` (default backend `cpu`) uses a standalone CUDA
progression filter and additionally requires NumPy.
It accelerates the base-2 congruence for `q`, **not** Phase-0 discovery or
gapminer's mining code; CPU verification is unchanged. Build and check it with
`make test-psp-gpu WITH_CUDA=1`, then, with the GPU free:

```bash
python3 scripts/psp_hunt.py pairs --backend cuda --device 0 \
    --batch-size 1048576 --p-start 10002000001 --p-end 10003000001 \
    --db data/psp_pairs_gpu_next.sqlite3
```

`--device` defaults to `0`, `--gpu-library` to the repository's
`bin/libpsp_pairs_gpu.so`, and the batch default remains `10000` (CUDA maximum
`16777216`). The GPU backend requires `p,q < 2^63` and rejects unsupported
ranges or CUDA failures without fallback or committing unfinished work.
CPU/GPU checkpoints are interchangeable. CUDA batch telemetry reports
`q_tested`, `filter_passers`, `kernel_ms` and `batch_seconds`; Brian's published
61 PSPs with gaps >=800 are the regression controls.

The CUDA scheduler reuses at most one in-memory prime-order sieve segment
across checkpoint batches. This cache is discarded on exit and reconstructed
from the checkpoint on resume; it changes neither candidate coverage nor the
default batch size. Its benefit depends on how many batches share a segment.
On the dev RTX 3070, a 17-batch slice near `p=2.6e10` fell from 1.51 s to
0.61 s (median of three end-to-end runs); this is not a whole-campaign estimate.
The slice used `--start 1e20 --end 1e21 --factor-floor 10000000 --min-gap 800
--p-start 26000000001 --p-end 26000500001 --batch-size 10000`.
The sieve also handles the smallest valid factor `p=3` without sieving primes.

For a Brian comparison, use a new database:

```bash
python3 -u scripts/psp_hunt.py pairs --backend cuda --device 0 \
    --start 1e20 --end 1e21 --factor-floor 10000000 \
    --p-start 10000001 --p-end 31622776602 --min-gap 800 \
    --batch-size 16777216 --db data/psp_brian_1e20_1e21_g800.sqlite3 \
    >> data/psp_brian_1e20_1e21_g800.out 2>&1
```

The threshold is inclusive; strictly greater than 800 means `--min-gap 802`.
CPU and CUDA use the same lossless short-gap check: two proven bracketing
primes less than G apart exclude a containing gap >= G. Only retained hits
receive full interior verification and are stored/printed. Factor coverage
and GPU work are unchanged. Positive thresholds require product ranges and
bracketing endpoints below the deterministic MR12 bound; proof failures abort
without committing the unfinished batch.

`confirmed` counts all proven PSPs, `gap_filtered` counts discarded short-gap
hits, and `retained = confirmed - gap_filtered`. Counters and progress are
committed even when no hits are retained. Repeat the threshold on resume;
changing it requires a new database. Unfiltered older databases are unchanged.
CUDA telemetry adds batch `gap_filtered`; progress and `show` add cumulative
`retained` and `gap_filtered`. The maximum CUDA batch above reduces checkpoint
overhead; the same command resumes after Ctrl-C or SIGTERM. Keep the session
open while it runs. The tool does not launch a campaign itself.

Pass `--sieve` to match the campaign you are asking about - the engine default
is 150000, the dev record campaign runs 150000, the fleet wall runs 85000, and
the 2e20 corpus used `sieve_limit 1e7`.  Comparing 85000 with 150000 at 7.9e28
is itself informative: same Euler-pass count, 1885 extra survivors that all fail
the test.
`--fixtures` walks the interior of each known pseudoprime gap with both tests and
prints which one is lost; `--carmichael` compares the two tests on Carmichael
numbers; `--density` sieves `--odds` odd numbers at `--x` to depth `--sieve` and
runs the Euler/Jacobi criterion on the survivors, printing the 95 % upper bound
per odd number (3/odds) and per Euler pass.  `--x` is parsed **exactly** -
`int(1e32)` is smaller than `10**32` by 5.4e15, which would silently move the
window.  Reaching a tighter bound costs linear time: `--odds 2e7` is ~10x
slower and ~3x tighter.

A high-merit hunt only needs *some* gap above a threshold: a pseudoprime there
costs a candidate, and the next candidate is minutes away.  An exhaustive census
has no second chance - a gap it misses is missing from the record forever - and
the distinction was put sharply on Discord:

> For the normal high merit search with hundred+ digit numbers, we don't care
> about PSPs since all we want is to find high-merit gaps, but with this search,
> we want to find ALL gaps.  If we treat a pseudoprime as prime, then we could
> miss a large gap, which is not acceptable if we want the search to be
> comprehensive (unless you can find all missed pseudoprimes and deal with them
> another way).

The same thread (2026-11-03, the PGN / BrianK channel) settled which test each
program actually runs, and produced the counter-example this section now
carries.  Wizz confirmed that PGN's "Fermat" test is Euler/Jacobi underneath -
an improved version of Perig's test that kept the misleading name - and BrianK
named the risk exactly: *it sounds like it knows it can miss a real gap if it
happens to include a pseudoprime that cuts it in two*.  That is precisely the
failure mode described below, and it is not hypothetical for us (see the two
fixtures).

The mechanism is asymmetric and worth stating once:

* a **prime never fails** a Fermat/Euler/MR test, so a *failed* test really is a
  proof of compositeness, and an interior position that fails is provably not a
  prime.  The census therefore **cannot invent** a gap;
* a *passing* test is only a probable-prime verdict.  A composite that passes
  **splits** one real gap into two, and if both halves fall below the reporting
  threshold the real gap is **lost silently**.  `verified=1` on every logged row
  does not help: `mpz_probab_prime_p` + `mpz_nextprime` re-proves a *reported*
  gap and can never recover one that was never reported.

### The three tests, and why they are not interchangeable

For odd composite `n` with `gcd(2,n) = 1`:

| test | condition | what it inspects |
|---|---|---|
| **Fermat** base 2 | `2^(n-1) = 1 (mod n)` | only the endpoint of the squaring chain |
| **Euler/Jacobi** base 2 | `2^((n-1)/2) = (2/n) (mod n)`, `(2/n)` the Jacobi symbol | the half-power itself, against a sign predicted by cheap arithmetic (`(2/p) = +1` iff `p = +-1 mod 8`) |
| **strong** base 2 (Miller-Rabin) | `n-1 = 2^s d`, `d` odd: `2^d = 1`, or `2^(2^r d) = -1` for some `0 <= r < s` | every rung of the chain |

The nesting is `strong` `subset` `Euler/Jacobi` `subset` `Fermat`.  Euler implies
Fermat by squaring (`(+-1)^2 = 1`); strong implies Euler is the classical
theorem, and the first 51 Carmichael numbers confirm it empirically - Fermat
accepts 51, Euler/Jacobi 41, strong 5, with `strong subset Euler subset Fermat`
holding on the nose.

Neither strengthening is cosmetic, and each is illustrated by a small number:

* **Fermat but not Euler: `341 = 11*31`.**  `2^170 = 1`, but `(2/341) = -1`
  (341 = 5 mod 8), so the Euler test demands `340` and fails.  A composite `n`
  has up to `2^k` square roots of 1 (`k` = number of prime factors); Fermat only
  asks for *some* root, Euler pins the sign from outside.
* **Euler but not strong: our gap-956 pseudoprime.**  `n-1 = 2^5 d`, and the
  chain runs `2^d = 18297422721374251885`, then `103980063429580871173`, then
  `1`.  The last value is exactly what the Euler test wants, so Euler passes -
  but `103980063429580871173^2 = 1 (mod n)` with the value neither `1` nor `-1`
  is a **nontrivial square root of 1**, and the strong test looks for the `-1`
  that must precede a `1` and does not find it.  Euler inspects the endpoint;
  strong inspects the chain.

That is why "the kernel is a probable-prime test" is not a specification: the
three variants differ by which composites they let through, and the difference
is exactly the class that can swallow a gap.

So the only question that matters for completeness is what the **discovery**
test is, and every design pays for it with speed.  The three programs:

| | discovery test | guard against a split | what happens to a lost gap |
|---|---|---|---|
| **this repo** (`--engine walk`) | base-2 **Euler/Jacobi** (`perig` / `p0u96` / `p0u108`, `w_test_mode` modes 0/3/4/5). The sieve engine uses strong `mr_base2` instead | none in-tree. Every *reported* gap is GMP-verified (`mpz_probab_prime_p(lower,25)` + `mpz_nextprime(lower) == upper`) | silent: both halves below `--gap-min` -> nothing logged |
| **PGS** (`briankehrig/prime-gaps-cuda`) | Perig base-2 Fermat, jump-by-`minGap` walk | (a) **2-PSP sieve**: marks `n = 0 mod p` with `n/p = 1 mod ord_p(2)` for every sieved prime below the 4096th prime, 38873 (7..83 by wheel, 89..38873 by the medium sieve) - the exact subset of multiples of `p` that the base-2 Fermat test would wrongly accept, so it closes every PSP whose factors are inside the sieve at ~1 mark per `p^2` instead of 1 per `p`; (b) CPU re-walk of each reported gap with `isPrime84` = strong MR to bases 2..37, deterministic below 318665857834031151167461 | documented and *bounded*, in the repo's own words: the author's separate search for base-2 PSPs with **no** factor `< 1e7` is "now complete up to 1e21.  The largest prime gap that contains one of them is of length 1194.  Thus, this code will not miss any gaps of size >= 1200 under 1e21", while `minGap < 960` "is not guaranteed to be 100 % exhaustive". The declared domain stays inside the deterministic MR12 band, so the CPU re-walk is a proof, not a probable-prime verdict |
| **PGN / Prime Gap Scout** (`wizz13150/prime-gap-scout`) | PGN's own fused base-2 Euler/Jacobi - the headers are literally `namespace specialized_euler` and the criterion is `terminalEquation`, an Euler congruence with a Jacobi-derived sign; Wizz has said so himself ("an undercover Euler/Jacobi test").  Same class as ours, so it **splits at exactly the same pseudoprimes** | mandatory `PGN_VERIFY_GAPS` GPU certificate - the runtime will not build without it (`"Production source adapter requires GPU gap certificates"`): both endpoints must be **proven** prime by 12-base MR and every interior candidate that passes Euler must be **proven composite** by the same test.  Inside the qualified window that is a real proof (MR12 is deterministic below 3.19e23).  But `candidate_walk.cuh` emits a gap only `if (gap >= minGapVal)`, and the verifier only inspects emitted gaps - see the emission analysis below | the strictest verifier of the three, and it does not claim more: `"prime_certificate":"MR12 below318665857834031151167461"`, `"coverage":"SOURCE_PRP_INDEPENDENT_PROOF_PENDING"`, README says "baseline independent exhaustive coverage remains pending".  Window is one high word, `supported_high_word: 10` = 1.8446744e20..2.0291418e20 - which contains only **5 of BrianK's 61** published gaps |

### Two fixtures, and a correction to one of them

`scripts/psp_probe.py --fixtures` reproduces all of this and checks it:

| gap | interior number | factorisation | Fermat | Euler/Jacobi | strong | engines |
|---|---|---|---|---|---|---|
| 156 | 108081990803 | 42443 x 2546521 | **yes** | no | no | both FOUND |
| 710 | 18447233112263860537 | 4015729 x 4593744526153 | no | no | no | both FOUND |
| 956 | 160696461657079906273 | 2142738313 x 74995840921 | yes | **YES** | no | walk MISS, sieve FOUND |

The three rows bracket our test exactly: 156 is the Fermat-but-not-Euler class
and our Euler/Jacobi test catches it; 710 is not a pseudoprime at all; 956 is
the Euler/Jacobi class the walk cannot see.  `--gap-min` 155 / 709 / 955
respectively, so each row is "just under the true gap".

All three *gaps* pass the scanner's own `verify_gap` (`mpz_probab_prime_p(lower,
25)` + `mpz_nextprime(lower) == upper`, the check `tools/phase0_scan_gpu.cu`
runs on every reported gap), and GMP's `mpz_probab_prime_p(interior, 25)` calls
all three interior numbers composite - so the pseudoprime label is decided by
the Fermat/Euler/strong columns alone, never by a disagreement about primality
itself.

**The 156 row is constructed, not mined.**  No public example of that class was
available at a usable scale (the canonical small one is `341 = 11 x 31`, whose
enclosing gap is 2 and therefore below the walk's `--gap-min >= 100` floor), so
it was built with the classical sufficient condition: for primes `p`, `q`,
`p*q` is a base-2 Fermat pseudoprime whenever `q = 1 (mod ord_p(2))` and
`ord_q(2) | p-1`.  Taking `p = 42443`, `q = 2546521` gives
`n = 108081990803`, and it lands inside the prime gap of 156 from
108081990733 to 108081990889 with both factors above the 30000 sieve.  Running
the same construction over 300 primes with `p` in `(42000, 60000)` and
`q < 8e6` produced 180 such numbers, 20 of them inside a gap `>= 100` - so the
recipe is cheap to repeat if this fixture ever needs replacing.

**Correction (2026-11-04).** PGS's `worktodo.txt` calls the 710 number "a base-2
PSP that is contained in a gap of size 710", `docs/PGS_GPU_TOOL_REVIEW.md`
repeated that, and this document repeated it too until it was checked:
`2^(n-1) mod n = 16649844496336080870`, not 1.  It is **not** a pseudoprime of
any kind - it is a semiprime with two large factors, and no number in that gap's
interior passes Fermat or Euler/Jacobi at all.  So the 710 gap is found by *any*
test, it cannot bracket anything, and the "PSP fixture" gate below was never
testing a pseudoprime.  Verified three ways: `gmpy2.powmod`, Python's builtin
`pow`, and the factorisation `4015729 x 4593744526153 = n`.

Only the 956 entry is real, and it is the Euler/Jacobi one:

* gap 956 at `--gap-min 955`: **the strong test reports it and our engine does
  not** - the number passes Fermat *and* Euler/Jacobi base 2, splitting the gap
  into 410 + 546, both below the threshold.  Verified against the engine's own
  predicate, not a re-implementation: `ciosFermatTest128_hi(13122509067403493345,
  8)` returns "probable prime" for it, and it is
  `2142738313 x 74995840921` - both factors are far above both depths we run
  (85000 on the fleet wall, 150000 on the dev box) and above PGS's 38873, so
  **no sieve of either program can remove it**.

BrianK's 1194 and 1044 gaps are presumably the *natural* members of that third
class below 1e21, but reproducing them means enumerating base-2 pseudoprimes
with no factor `< 1e7` up to 1e21 - his "separate search".  Following that
constraint: a number below 1e21 with all prime factors above 1e7 has at most
two of them, since `(1e7)^3 = 1e21` exactly, so every member of the class is a
**semiprime** and no Carmichael number can appear.  That makes his enumeration a
semiprime search over `q = 1 (mod ord_p(2))`, the same progression used above -
a research-scale run, and one we do not need, because the constructed fixture
exercises the identical code path at four orders of magnitude less cost.

### The statistics a miner actually needs

Two numbers decide whether pseudoprime recovery is worth building into a mining
program, and both are measurements from BrianK's published enumeration rather
than estimates:

* **How many pseudoprimes can share one gap?**  2 789 289 gaps `>= 300` in
  `1e20..1e21` contain a 1e7-rough base-2 PSP, i.e. a density of `3.1e-15` per
  position.  In a gap of 1200 that is an expectation of `3.7e-12`, so
  `P(two or more) = 7e-24`.  **Gaps with two pseudoprimes do not occur at this
  scale**, which is why the cheap `nextprime(p1)` repair (re-walk from the lower
  prime of a suspicious gap) is sufficient with an enormous margin.
* **What is the at-risk set at a given merit?**  Merit 20.5 at 2e20 is gap 958.
  Of BrianK's 61 published gaps `>= 800`, only **three** pass Euler/Jacobi at
  `>= 900` (956 and two of 902) and exactly **one** exceeds 958 - the 956 gap.
  So for an Euler/Jacobi engine the risk in that decade is one gap, not 61, and
  the 1194/1044 often quoted are Fermat-only and are found normally.

For scale, base-2 Fermat pseudoprimes below `1e19` number 91,210,364
(OEIS A055550) and grow 2.6-2.7x per decade, so the `2e20..3e20` window holds
about `6.1e7` of them; the 1e7-rough subset is ~`1.4e6` per half decade.  Work
for an enumeration is linear in range length while the yield grows only 2.6x per
decade, so each further decade costs ~10x the time for ~2.6x the pseudoprimes.

### Does this project need to fix it?  The exposure number

The two halves of the problem have different answers, and only one of them is
open.

**Publishing is already closed.**  `verify_gap()` (`tools/phase0_scan_gpu.cu`,
"gap verification / reporting") demands `mpz_probab_prime_p(lower, 25) >= 1`
**and** `mpz_nextprime(lower) == upper` for every row it reports, and
`process_prime_gap()` logs the outcome as `verified=`.  A pseudoprime therefore
cannot become a published endpoint, and it cannot sit unrecorded in a published
interior: `nextprime` establishes that no prime exists strictly between the two
endpoints.  The guard is exercised, not assumed: walking the 956 window at
`--gap-min 400` makes the walk *accept* the pseudoprime and emit two fragments
(410 and 546), and both come out `verified=0` - see the gate row
"pseudoprime endpoint rejected at emission".  Measured on every campaign log in
this tree - **10 622 wall rows and 700 record rows, 10 622/10 622 and 700/700
`verified=1`, zero unverified**, 18.3 M and 1.4 M interior positions
respectively.

**Discovery is open, and its size is measurable.**  A pseudoprime can only lose
a gap, and only if it lands inside one we would have reported.  So the expected
loss over a finished campaign is

    E = d_psp(x) * sum(gap - 1)          over the gaps that campaign reported

with `d_psp` the 1e7-rough base-2 PSP density per position.  Using the measured
`d_psp(1e20) = 3.1e-15` and the `x^-0.585` fall implied by A055550 (the PSP
count grows 2.6x per decade while positions grow 10x, so the density divides by
3.8 per decade; extrapolated over 9 and 12 decades, which is the one soft number
here):

| campaign | x | reported gaps | `gap-1` summed | `d_psp` | E\[lost\] |
|---|---|---|---|---|---|
| wall (`g >= 1664`) | 7.9e28 | 10 622 | 1.83e7 | 1.9e-20 | **3.5e-13** |
| record (`g >= 2002`) | 1e32 | 700 | 1.45e6 | 3.0e-22 | **4.3e-16** |

Both are far below one lost gap per campaign, and the reason is not luck: the
two thresholds are merit 25.0 and merit 27.2, while BrianK's entire published
list - the only complete measurement that exists - tops out at gap 1100 at
`1e20`, i.e. **merit 23.9**.  We are asking a question the data cannot answer
directly, at a PSP density five to seven orders of magnitude below his.

The rule that follows is a function of threshold and height, not a constant:

* at `1e20` with `--gap-min 700`, one decade (9e20 positions) holds **559**
  PSP-containing gaps (`>= 300`: 2.79e6) - a program with that threshold over a
  whole decade has hundreds of candidate silent losses, which is why "check for
  pseudoprimes in parallel" is the right advice for PGN/Scout and not a
  theoretical worry;
* at `1e28..1e32` with `--gap-min >= 1664` the figure is `1e-13..1e-16`.

**Conclusion: no hot-path change.**  If a campaign ever drops below ~`1e22`, or
to a `--gap-min` under ~1000, the discovery test itself has to change, and the
costs are already measured: strong base-2 is `1.81x` the wall and does **not**
close the class (26 of BrianK's 61 are genuine strong pseudoprimes to base 2),
bases 2+3 score 0/61 but `spsp(2,3)` numbers exist and are infinite, so 0/61 is
an empirical result and must never be written up as a proof, and PGN's
complementary 2-PSP sieve closes the `(sieve depth, 1e7]` factor class exactly at
a GPU rate cost inside the noise (431.7 -> 433.9 Gpos/s).  None of the three
touches the 1e7-rough class, which no in-program test can enumerate.


### PGN's own attempt at this (report of 4 November 2026)

PGN published a private engineering report implementing the complementary sieve
BrianK described, and it agrees with the analysis above in every particular:

* the sieve covers `(80 000, 1e7]` by marking `N = p (mod p*ord_p(2))`, which is
  the same construction as PGS's and is provably safe (every mark is divisible
  by `p` and larger than it, so composite; it over-marks, which is harmless);
* **the 956 example is not recovered by that sieve** - both of its factors
  exceed `1e7` - it is recovered by a separate "large-factor witness" family,
  which they state is **bounded and not exhaustive**;
* "Complete pseudoprime coverage over 67-95 bits remains open", i.e. they
  explicitly do not claim the census property we are discussing;
* the cost is what makes the sieve attractive for us: GPU rate unchanged within
  noise (431.693 -> 433.853 Gpos/s), `5 544 292` small-factor marks in a `1e12`
  cohort, a 656 742-row canonical table verified by a pinned SHA-256 and loaded
  in 0.165 s; the `+24.19 %` cold-worker time in that build is 1.014 s of
  CPU-side preparation, which they identify as a prototype artifact and expect a
  resident pipeline to remove;
* 19 native executions, thresholds 954/956/960, a shifted origin, a descriptor
  boundary and a 1554-gap positive control - all certified reference gaps
  conserved, with exactly the 956 gap added.

Their numbers were not reproduced here (no build on this box, and their `high =
10` backend is absent from the public tree: `bits_69` accepts `high` in
`[16, 32)`), so treat them as their measurement, quoted, not ours.

### Does a certifying verifier (PGN Scout) find those gaps?  No.

A verifier is not a detector.  `candidate_walk.cuh` emits a gap only
`if (gap >= minGapVal)`, and `PGN_VERIFY_GAPS` only inspects emitted gaps; PGS's
`checkGapAndPrint` works the same way.  A pseudoprime splits `G` at offset `a`
into `a` and `G-a`, so:

* `minGap <= max(a, b)` - a half is emitted, the verifier sees the pseudoprime,
  and the gap is **rejected**.  Loud, correct - and the gap is still not found.
* `minGap > max(a, b)` - nothing is emitted, the verifier never runs, and the
  gap is **silently lost**.

Scored against the 61 published gaps (`scripts/psp_probe.py --brik`, last block):

```
minGap   loud      silent
300      61 / 61     0 / 61
500      49 / 61    12 / 61
700      28 / 61    33 / 61
800      11 / 61    50 / 61
1000      0 / 61    61 / 61
```

So at the thresholds a real hunt uses most of those gaps are lost silently
rather than caught - and even the "loud" cases end with the gap discarded, never
reported.  **Scout cannot find them: its search layer is Euler/Jacobi, the same
class as ours, so it splits identically.**  What it adds over us is that it
refuses to *publish* an uncertifiable gap; what it lacks, exactly like us, is
any way to *see* one that never became a candidate at all.

### Ground truth: how leaky is each test, measured

BrianK published the **complete** list of gaps >= 800 in `1e20..1e21` whose
interior holds a base-2 Fermat pseudoprime with no prime factor `< 1e7` - 61
gaps (out of 2 789 289 gaps `>= 300` in that range), i.e. exactly the gaps his
program would miss without its separate PSP search.  That is the only public
ground truth for "how leaky is test X", so it is now a regression set in the
tool: `scripts/psp_probe.py --brik`.

| discovery test | gaps missed | note |
|---|---|---|
| Fermat base 2 | **61 / 61** | useless here, as expected |
| Euler/Jacobi base 2 (our walk) | **37 / 61** | the engine we run |
| strong base 2 (`mr_base2`) | **26 / 61** | **still not sound** |
| strong base 2 **and** base 3 | 0 / 61 | sufficient for this list, not proven in general |
| strong bases 2..37 (12 bases) | 0 / 61 | deterministic below 3.19e23, so a *proof* here |

Two corrections follow, and the first one retracts something this document said
earlier.

1. **"Switch to the sieve engine" is not a fix.**  Its `mr_base2` is a strong
   base-2 test, and 26 of the 61 published gaps contain numbers that **pass** a
   strong base-2 test - they are genuine strong pseudoprimes to base 2 (largest
   in a gap of 902).  The sieve engine is *less* leaky than the walk, not sound.
   No single-witness base-2 test of any strength closes this range; only a second
   base (3 suffices for this list) or a PSP sieve does.
2. **The relevant constant is different in each range.**  For `1e20..1e21` a
   12-base MR is a *deterministic* primality proof (the Sorenson-Webster bound
   is 3.19e23), which is what PGS's `isPrime84` exploits - that is why his CPU
   re-walk of reported gaps is a proof and why his PSP sieve is the piece that
   makes the *search* complete.  At our wall band (2^96 = 7.9e28) that bound is
   far behind us: no small set of bases is a proof, so there the only sound
   options are a PSP sieve or a census claim that is explicitly probabilistic.

The 956 fixture is in that published list exactly as we factored it
(`160696461657079906273 2142738313 74995840921 956`), and the 1194 and 1044
entries confirm the Euler/Jacobi refinement numerically: both are Fermat-only
(`eul()` returns False for both), so **an Euler/Jacobi engine finds those two
gaps** - 956 really is the worst case for our engine class below 1e21.

**Our own 2e20 campaign is clear.**  Its range `[2e20, 2.0001e20]` contains
**zero** of the 61 published >= 800 pseudoprime gaps: the nearest below is at
1.982154e20 (gap 826) and the nearest above at 2.005452e20 (gap 936).  So the
HL validation anchor loses nothing to this class at `>= 800`; the `[701, 800)`
slice is not covered by the published list (he kept the `>= 300` data but posted
only the `>= 800` extract), and would need the full 2.79M-row set to settle.

**Cross-engine check, run end to end (2026-11-04).**  The same binary, the same
3000-number window, the same thresholds - only `--engine` differs:

| fixture | `--engine sieve` (strong base-2 `mr_base2`) | `--engine walk` (Euler/Jacobi) |
|---|---|---|
| gap 956 (Euler/Jacobi pseudoprime) | **FOUND** `956 20.5476 ... verified=1` | **MISSED**, log has zero rows |
| gap 710 (semiprime, not a pseudoprime - see above) | FOUND | FOUND |

```
./bin/phase0_scan_gpu --engine sieve --start 160696461657079904863 \
    --length 3000 --merit-min 20 --sieve-limit 30000 --no-records --log /tmp/s.log
./bin/phase0_scan_gpu --engine walk  --start 160696461657079904863 \
    --length 3000 --gap-min 900 --merit-min 0 --walk-primes 30000 --walk-batch 8 \
    --no-records --log /tmp/w.log
```

So the two engines are not interchangeable for a completeness claim.  Note the
scope carefully: the sieve engine is the *less leaky* of the two here, but the
ground-truth block below shows it is **not sound either** - 26 of BrianK's 61
published gaps contain numbers that pass a strong base-2 test.  A trap worth recording
because it cost a run: **`--gap-min` is a walk-only flag** (the binary rejects it
with "`--engine walk`: `--gap-min` must be in [100, 1e7]" outside that path);
the sieve engine filters gaps by **`--merit-min`**, and its default of 20 is
high enough to hide a merit-20.55 gap if you pass a gap threshold and assume it
applied.  The 956 gap is merit 20.5476.

A refinement that follows, and that is worth more than the fixture itself:
Brian's public claim is "the largest prime gap that contains one of them is of
length 1194, so no gap >= 1200 is missed under 1e21" - but that is stated for
**Fermat** pseudoprimes.  In the thread he adds the decisive detail: of the
three Fermat-PSP-carrying gaps >= 960 below 1e21 (sizes 1194, 1044, 956), the
two larger ones are **not** Euler-Jacobi pseudoprimes.  So for an engine that
runs Euler/Jacobi - ours, PGN's, and PGS's own `FERMAT_TEST`, which is Euler/
Jacobi despite the name - the relevant bound below 1e21 is **956, not 1194**,
and only that one gap is at risk.  Smaller bound, same conclusion, but it is
the correct number to quote for our engine class.

Two facts make the table sharper than it looks, and a third retracts one of
them.

**1. A base-2 Euler/Jacobi test is much weaker against Carmichael numbers than
a strong test.**  Reproduce with `scripts/psp_probe.py --carmichael`.  Measured
2026-11-03 on the first 51 Carmichael numbers:
**41 pass** the Euler/Jacobi congruence that `perig`/`p0u96`/`p0u108` and PGS's
`FERMAT_TEST` implement, but only **5 pass** the strong base-2 MR that our sieve
engine uses.  So "Euler/Jacobi base-2" and "strong base-2" are not
interchangeable for a completeness claim, and both our production campaigns sit
on the weak side (wall 7.9e28 -> `w_test_mode` 4, record 1e32 -> mode 5) purely
for speed.  The Chernick family `(6k+1)(12k+1)(18k+1)` makes it concrete: for
`k = 25025` the factors 150151, 300301, 450451 are all prime, so
`n = 150151*300301*450451 = 2.03e16` has **no factor below either sieve depth
we run (85000 / 150000)**,
survives it, passes Euler/Jacobi base-2, and is rejected by the strong test -
and no PSP sieve can remove it either, because by construction none of its
factors is small enough (that point is developed below).

da li wizzov program nalazi te pseudpprimes**2. (retracts part of 1.) "Strong is stricter" is true about Carmichael
numbers and false about pseudoprimes as a class.**  Point 1 compares the tests on
*Carmichael* numbers, where strong wins 5-to-41.  On the actual pseudoprimes that
occur in our kind of range the ordering is very different: of BrianK's 61
published `>= 800` pseudoprime gaps in `1e20..1e21`, Euler/Jacobi misses 37 and
**strong base 2 misses 26** - those 26 are genuine strong pseudoprimes to base 2.
A second base (3) catches all 61 (`scripts/psp_probe.py --brik`).  So a
base-2-only test, at any strength, cannot support a completeness claim in that
range; and the deterministic MR bound (3.19e23) that makes a 12-base check a
*proof* there does not reach our wall band at 7.9e28.

**3. At our scale the risk is nonetheless vanishingly small, and the bound is
measured, not assumed.**  Reproduce with `scripts/psp_probe.py --density`.
Direct probe at the production wall (x = 1e32, sieve depth 150000, the
`--walk-primes` default):

```
odd numbers sampled           2,000,000
survived the sieve              188,405   (9.4 % of odds)
passed base-2 Euler/Jacobi       54,536   of which 100.0 % are primes
COMPOSITE that passed                 0   (95 % CL < 1.5e-6 per odd number)
```

Zero in 54536 Euler passes.  That is far below the small-x literature because
the pseudoprime count grows much slower than `x`: over `10^12..10^19` the
count of base-2 pseudoprimes grows 2.60-2.70x per decade (OEIS A055550:
`psi(1e10) = 14884`, `psi(1e11) = 38975`, `psi(1e19) = 91210364`), i.e. a local
exponent of only ~0.42-0.43 in `log-log`.  Carrying that exponent to 1e32 puts
the count of base-2 pseudoprimes anywhere in a whole campaign's range
(`R ~ 1.6e17`) at order 0.02-2, so pseudoprimes are not impossible in the range
- they simply cannot do damage, for a second reason:

**a pseudoprime can only lose a gap of size `< 2 x gap-min`.**  Split `G` at
offset `a` into `a` and `G-a`; both fall below the threshold only if
`G < 2*gap_min`.  A split of a longer gap always leaves one half `>= gap_min`,
that half IS reported, and its GMP check fails - so it surfaces as
`verification_failures`, not as a silent loss.  That single rule sorts our
campaigns, and the sort is the useful result:

```
campaign            gap_min   sieve   at-risk window    #gaps   frac of R
2e20 (HL anchor)        701   1.0e7    [ 701, 1402)    169276   1.26e-6
wall (fleet)           1664   8.5e4    [1664, 3328)     10962   6.25e-11
record (dev)           2002   1.5e5    [2002, 4004)       697   9.19e-12
```

The wall's sieve is **85000**, not the engine default 150000: the fleet's chain
runs `--walk-primes 85000`, the depth measured best on its RTX 3060s with the
u96 binary (AGENTS.md).  That difference was measured too, at 7.9e28 over
2 000 000 odd numbers:

```
sieve 85000 : survived 197916 (9.90 % of odds)   passed Euler/Jacobi 60429   PSP 0
sieve 150000: survived 188481 (9.42 % of odds)   passed Euler/Jacobi 60429   PSP 0
```

**The verdict set is identical** - 60429 Euler passes either way.  The shallower
sieve leaves 9435 more survivors and every one of them fails the Euler/Jacobi
test, so it costs tests, not correctness, at this scale.  The trade is also why a
shallower depth can win at all: 85000 marks with 8277 primes instead of 13848
(-40 % of the marking work) to buy +5.1 % more tests.  §26.5 has the sweep on the
dev box (40000 best there, flat to 250000, collapsing at 1e6) and the direct
check that settles the correctness half: at `--gap-min 400` over 1e10 integers at
the wall both depths reported **277746 gaps and the sets are equal, 0
differences** - as they must be, since the sieve only removes numbers that have a
prime factor `<= P`, and those are certainly composite.

It does widen the surviving *class* in principle: a Chernick number whose
smallest factor lies in `(85000, 150000]`, e.g. `k = 14206` giving
`85237 x 170473 x 255709 = 3.7e15`, is sieved out on the dev box and survives on
the fleet, and it passes Euler/Jacobi.  At 7.9e28 the equivalent would need all
three factors near 1e9, which is far outside anything we have found (or Brian's
enumeration, whose cases are products of two primes); the measurement above is
the reason to expect none, not the sieve depth.  Note also that the identical
verdict count is a measurement, not a theorem - the Chernick family is a valid
counter-class - so it is worth re-running this probe after any change to the
sieve depth or the test.

Three things follow.

* **The high-threshold campaigns are structurally the safest**, and for a
  reason that has nothing to do with density: every pseudoprime-in-a-gap found
  so far is in a gap of 956-1194, which is *below* both thresholds.  A gap that
  small cannot be reported at `--gap-min 1664` in the first place, so it cannot
  be lost either.  Only the `[1664, 3328)` / `[2002, 4004)` windows matter, and
  they occupy `6.3e-11` / `9.2e-12` of the scanned range.
* **The 2e20 campaign is the exposed one** (`gap_min 701 < 956`, and
  `1.26e-6` of its range is in the at-risk window) **- and it is also the one
  that can be settled rigorously.**  Its sieve depth is **1e7**
  (`sieve_limit=10000000` in its session header), which is exactly the class
  Brian's enumeration covers ("base-2 pseudoprimes with no prime factor
  `< 1e7`, complete up to 1e21"), and its range `[2e20, 2.0001e20]` lies inside
  that.  Of the three PSP-carrying gaps `>= 960` below 1e21, only the 956 one is
  an Euler-Jacobi case, and it sits at 1.6069e20 - **below our start**.  So the
  HL validation anchor is complete with respect to that class, by a published
  enumeration rather than by extrapolation.  (The enumeration is quoted for gaps
  `>= 960`; the `[701, 960)` slice of our window is not covered by it.)
* **The wall and record campaigns cannot use that argument**, because they sieve
  only to 150000 and scan at 7.9e28 / 1e32, ten to eleven decades above where
  any enumeration exists.  For them the bound is the measurement above:

    E[lost gaps] ~ E[pseudoprimes in range] x (fraction of R in the window)
                 ~ O(1) x 9.2e-12  ~  1e-12 .. 1e-11   (record campaign)

  and the census agreement with the HL tables (597 found vs 591 predicted at
  the time of measurement) is consistent with zero loss.  Nothing at this scale
  can distinguish the two.

**The 2-PSP sieve DOES transfer to us - an earlier version of this document
said it did not, and that was wrong.**  The sieve marks the exact subset of
multiples of `p` that pass the Fermat test (`n = 0 mod p` with
`n/p = 1 mod ord_p(2)`) for every prime `p` in `(sieve_depth, 1e7]`.  The
mistaken argument was "our sieve to 150000 already removes every PSP with a
small factor".  It removes every composite with a factor `<= 150000` - but a
pseudoprime whose factors all lie in `(150000, 1e7]` is untouched by that and
still passes Euler/Jacobi.  The Chernick number
`n = 150151*300301*450451 = 2.03e16` is exactly such a case: smallest factor
150151, so it survives BOTH of our depths (85000 and 150000), and
`2^((n-1)/2) = 1` with `jacobi(2/n) = +1`, so our pipeline calls it **prime**.
The class is open for us, and the 2-PSP sieve is the cheap way to close it.

What the sieve cannot close, for us or for PGN, is the class where **all**
factors exceed `1e7` - BrianK's semiprime class.  PGN's own report is explicit:
their complementary sieve covers `(80 000, 1e7]`, and the 956 recovery comes
from a *separate* "large-factor witness" family they describe as **bounded and
not exhaustive** ("Complete pseudoprime coverage over 67-95 bits remains open").
So the sieve closes one of the two classes; the other is exactly the 2.79M-entry
object of BrianK's enumeration, and no cheap construction currently covers it.

**What a comprehensive claim would still need.**  The measurement bounds the
risk, it does not prove absence.  Three guards were on the table; the thread and
the sieve-depth argument above remove one of them:

| guard | effect | cost | verdict |
|---|---|---|---|
| sample slices with the strong test (`P0_WALK_TESTMODE=2`) and diff the gap sets | turns the *claim* into a *check*: any gap in one set and not the other exposes a pseudoprime | one slice, no code change | **recommended** - the only cheap one, and the walk already supports it |
| the complementary 2-PSP sieve in the marking | closes the whole `(sieve_depth, 1e7]` factor class, including Chernick-type numbers our sieve leaves behind | Brian: sub-1 % of runtime.  PGN's qualified build: GPU rate unchanged within noise (431.7 -> 433.9 Gpos/s), **+24 % cold-worker time from 1.014 s of CPU table prep** they expect a resident pipeline to remove; table is 656 742 rows, 0.165 s verified load | **recommended, and nearly free for us** - a segment is 8.5e14 positions (~850 PGN cohorts), so 1 s of table load amortises to 0.04 %, and ~5.5e6 marks per 1e12 positions is noise |
| a strong discovery test | closes Carmichael numbers *and* the Euler/Jacobi gap exemplified by the 956 fixture | **measured 2026-11-04: 1.81x wall, -45 % throughput** (below) | **not sufficient on its own** - 26/61 of the published PSP gaps pass strong base 2; needs a second base |
| a second base (3) on top of strong | closes all 61 published gaps | ~2x the strong test, i.e. ~3.6x the wall | the cheapest option that is complete *for the published list*; still not a proof above 3.19e23 |

**The cost, measured rather than estimated** (dev RTX 3070, wall slice of 1e12
integers at 7.9e28, `--walk-primes 85000`, `--walk-batch 16`, two runs each,
`P0_WALK_TESTMODE` forced):

```
mode 4  p0u96  Euler/Jacobi (production)   3.44 s / 3.14 s   3.04e11 ints/s
mode 1  mr128_64_base2_lazy  strong base-2 5.95 s / 5.99 s   1.68e11 ints/s
                                                        -> 1.81x wall, -45 %
```

Per test alone the strong variant is much closer: `bin/bench_walk_test` on
~96-bit candidates gives 1.699 ns for perig and 2.151 ns for the strong 2x64
kernel (+26.6 %); the wall's 1.81x is larger because the production path is the
fused u96 kernel, which is already 1.70x faster than perig - the two multiply to
1.82x, matching the measurement.  So the honest price of strictness here is
**1.81x campaign time**, not the +26 % a per-test reading suggests.

Two cheaper options were checked and rejected: a second witness (base-3
Fermat/Euler) costs about one more full exponentiation, ~+100 % of test time,
i.e. worse than strong MR; and a *conditional* strong test only for candidates
near a prime buys nothing, because the condition "previous prime within
`2*gap_min`" is almost always true (3328 positions against a mean gap of 66).
The remaining path to strictness without the 1.81x would be a *fused* strong
kernel for the u96/u108 containers - it does not exist, and building it is a
hypothesis, not a measurement.

An **audit slice** is the middle ground: a contiguous sub-range of a campaign
scanned **twice**, once with the production discovery test and once with the
stricter one, whose two gap sets are then compared.  It is a *detector*, not a
proof - it says nothing about the integers outside the slice - and it is only as
sensitive as the number of gaps the slice contains.  The recipe, with the trap:

```
S=79000000000000000000000000000
for m in 4 1; do                     # 4 = p0u96 (production), 1 = strong base-2
  P0_WALK_TESTMODE=$m ./bin/phase0_scan_gpu --engine walk --start $S \
      --length 10000000000 --gap-min 400 --walk-primes 85000 --walk-batch 16 \
      --no-records --log /tmp/aud$m.log
done
# compare (lower, gap) pairs, AFTER sorting: the two modes emit in a
# different order, so a raw diff reports hundreds of spurious differences
diff <(grep -v '^#' /tmp/aud4.log | awk '{print $2,$3}' | sort) \
     <(grep -v '^#' /tmp/aud1.log | awk '{print $2,$3}' | sort)
```

Measured at the wall (1e10 integers, `--gap-min 400`, 277 707 gaps):
**the gap sets are identical**, which is the expected result and the reason the
recipe is worth keeping as a check rather than as a campaign.  Its statistics
are what limit it: gaps are rare, so a 1 % slice of the wall campaign (~110 gaps)
only detects a loss rate of ~3 % and costs +0.8 %, while bounding the rate below
0.3 % takes ~9 % of the range (+7 %).  The density probe above is the tighter
instrument per unit cost, because Euler passes are far more numerous than gaps.
Note also that at `--gap-min 400` the walk is dominated by the jump-walk itself
(3.6e8 ints/s, three orders below the production rate), so an audit slice must
be run at the campaign's own `--gap-min` if its *timing* is to mean anything -
here only its *gap set* does.

Brian's framing in the thread is the right default and it matches ours: a
deterministic test "will completely kill the speed; it's much better to stick
with a probabilistic one and then separately deal with all the pseudoprimes that
the code would have missed".  The difference is that his separate search can be
run to 1e21 because his domain stops there, while ours does not - which is
exactly why the two fixtures and the measurement matter more to us than to him.

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

`scripts/p0_walk_hl_compare.py` - the 128-bit walk scan vs titanV's HL gap
tables, in the two framings the tables support:

```
scripts/p0_walk_hl_compare.py [--out PNG] [--dark] [--glob 'PATTERN']
```
Reads every walk log matched by `--glob` (default
`data/p0_walk_wall*_g*.log`, one contiguous range), sums their coverage into a
single `R` and takes `L` at the range midpoint, then draws four panels:

* **A/B census** - `N(g >= G)` measured vs `R * SUM_g rho_g(L)` from the
  order-4 coefficients, with Poisson error bars and the measured/model ratio.
  The tables are calibrated on the published record curve, so the agreement is
  an out-of-sample test of the model and a closure test of the scanner.
* **C first-occurrence map** - `log10 x` vs gap size: every published record in
  the band (from `data/prime_gap_merits.txt`, `x = exp(gap/merit)`), the model's
  predicted first occurrence (root of `Y_g(L) = e^L`), and our records with the
  move down from the previous best.
* **D decades closed** - per length, `log10(x_prev/x_ours)` (green) against
  `log10(x_ours/x_hl)` (grey): what one campaign closes versus what remains to
  the predicted first occurrence.

Coverage per slice is taken from the session summary line when the run
finished, else from the `off0` of the matching `data/p0_state_*.txt`, else from
the last monitor line (3 significant digits - a 0.1 % effect on `R`, far below
the Poisson error).  Slice logs are appended across resume, so the event counts
and the covered integers stay disjoint.  Prints the coefficient gate
(`sum rho_g / (1/L)`), the per-length table and the mean decades closed.
`--dark` also writes `*_dark.png`.

`scripts/p0_rec_hl_compare.py` - the dev record campaign (x = 1e32) vs the same
HL tables, asked one octave higher:

```
scripts/p0_rec_hl_compare.py [--log LOG] [--out PNG] [--dark]
                             [--band-hi G] [--emin E]
```
Default log `data/p0_rec_1e32.log` (the `scripts/p0_rec_campaign.sh` campaign),
default figure `data/p0_rec_1e32_vs_hl.png`; `--dark` also writes
`*_dark.png`.  The campaign logs EVERY gap above a hard threshold, so one band
of it is an exhaustive census just like the wall walk - but at
`L = ln 1e32 = 73.68`, ~9 above the `L ~ 30..64` curve the tables were
calibrated on, which makes the agreement an out-of-sample test.  Four panels:

* **A/B census** - `N(g >= G)` measured vs `R * SUM_g rho_g(L)`, then the same
  comparison bin by bin (`rho_g(L) R` vs the count of exactly `g`) with a
  chi-square over the lengths that have `E >= 1`.
* **C merit** - everything on ONE scale, `m = gap / ln x`: the record table for
  every length in the band (`data/prime_gap_merits.txt`), **split into external
  holders and our own wall campaign** (see below), the external bests this
  campaign beat, our records (`m = gap/L`, fixed by the range) and the model's
  merit at the predicted first occurrence (`gap/ln x_hl`, 39.5-45.1 here).
  Merit units matter: drawn as `log10(x)` the external bests reach 1e40 and the
  axis ticks (37.5, 40) read like merits the campaign never earned - the
  highest merit actually found is 33.63, and the title says so.  The shaded top
  is where a length would have to appear BEFORE its expected first occurrence.
  It is empty: the model's first occurrence is far beyond anything either
  campaign has produced (see "why the model's first occurrences are so high").
* **D record yield** - cumulative `#winnable -> E[records] -> recorded`.  A
  length is winnable iff `table_merit(g) < g/L` (deterministic at a fixed
  range), but it only becomes a record if a gap of that exact length appears,
  at `1 - exp(-rho_g R)`; `E[records]` is therefore a prediction the campaign
  can be scored against.

Coverage is read differently from the walk case: the driver appends segment
after segment to ONE log/out pair, so `R = (frontier - first start) + off0` of
the live state file (the `next start=` line of the driver's segment summary),
not a sum over per-slice logs.  A segment killed by SIGTERM is re-run from its
own state file, so its `engine=walk start=` line appears twice while the range
is still scanned once - summing sessions would double-count.  Prints the
density gate, the census ratio with its Poisson sigma, the chi-square, the
yield funnel and the per-length record table (previous best, ours, gain,
decades closed / still to go).

Measured 2026-11-03 (183 completed segments + the live one, `R = 1.57e17`, 597
events): census 597 vs 591 predicted (+1.0 %, +0.2 sigma), chi-square 90.5 over
95 lengths, record lengths 63 vs 61.1 expected (+0.2 sigma), median 2.40 of
11.47 decades closed.  Merits: highest found 33.63; the 63 bests it beat were
all external and sat at 25.99-29.86; the model's first occurrence is 39.5-45.1,
i.e. every record is still ~5.9 merit (9.1 decades of x) below where its length
is expected to begin.
`R` grows while the campaign runs, so the predicted count drifts by < 0.1 % per
8.5e14 segment - a re-run never reproduces the exact digits, only the verdict.

**Two campaigns, do not mix them.**  The WALL campaign runs - and keeps running
- on the fleet host (`ssh gap20`, `/home/deki/git/Gap_miner_v2`): one walker per
GPU under `scripts/p0_walk_chain.py`, start 7.9e28, gap_min 1664, one log per
slice under `data/p0_walk_wall*`.  Its logs live there, so
`scripts/p0_walk_hl_compare.py` is run there and that figure is produced there.
The RECORD campaign is this box's (`scripts/p0_rec_campaign.sh`, start 1e32,
gap_min 2002, ONE appended log/out pair, charted here).  Neither script will
accept the other's logs any more - `p0_rec_hl_compare.py --log
data/p0_walk_wall_c1_g1664.log` and `p0_walk_hl_compare.py --glob
'data/p0_rec*.log'` both stop with a pointer to the right tool, because a wall
slice has no campaign frontier and a record log has no slices.  They meet in exactly one place: the live record
table already contains the wall campaign's own records, so in our band 22 of
the 269 lengths are held by this account at x = 7.9e28.  Panel C draws those as
separate diamonds and panel D splits the 53 seen-but-not-recorded lengths into
`blocked by our own wall record: 20` / `blocked by an external record: 33`;
treating them as third-party results would make the record campaign look like
it competes with others on lengths where it is in fact competing with our own
fleet.  The 63 records it did set were all against external holders.

`scripts/hl_validate_tables.py` - out-of-sample test of titanV's tables, and
the honest answer to "can our data improve them":

```
scripts/hl_validate_tables.py [--csv OUT.csv] [--plot [PNG]] [--first-occ-only]
```
Every one of our campaigns is an exhaustive census at ONE `L`, so the model's
free content there is `rho_g(L)` and every measured count is a Poisson draw
from `rho_g(L)*R`.  The script fits ONE parameter per campaign,
`rho_true = rho_model * exp(-eps/L)` (a uniform shift of `c1`), which is far
tighter than the median of the per-length ratios.  Three scans, all outside or
at the edge of the tables' `L ~ 30..64` calibration band:

| campaign | L | R | census | eps | +/- 1 sigma | eps/g (error of c1) |
|---|---|---|---|---|---|---|
| `p0_campaign_2e20` | 46.74 | 1.0e14 | 0.9982 (-0.2 %) | +0.08 | 0.11 | +0.012 % +/- 0.016 % |
| wall (fleet) | 66.54 | 3.0e17 | 1.0101 (+1.0 %) | -0.67 | 0.64 | -0.040 % +/- 0.038 % |
| record (dev) | 73.68 | 1.6e17 | 1.0127 (+1.3 %) | -0.93 | 3.01 | -0.046 % +/- 0.150 % |

**Verdict: the tables need no correction here - they are validated, out of
sample, to 0.05 % in `c1` and 1.5 % in the census count up to `L = 73.7`.**  That
is the contribution worth handing over, not a refit.  Two details in the method
are load-bearing and were both measured, not assumed:

* the expected census must be summed over EVERY tabulated length above the
  cut, not only the ones that were seen - otherwise the data chooses its own
  support;
* the cut must be the device threshold (`--gap-min`, or `merit_min*L`), rounded
  up to an even number.  At 2e20 the single unloggable length `g = 700` alone
  adds 24k of the 178k expected, and the census reads 0.949 instead of 0.998.

Fitting the residual drift, `eps(L) = a + b/L` gives `delta_c2 = -118 +/- 98`
(about -1.3 % of `c2`) - a 1.2 sigma hint, explicitly not a correction.

The second block tests the **first-occurrence** column, which is defined as the
root of `rho_g(L)*x = 1`, i.e. `E[count] = 1`.  Two different things can be
wrong with that, and they have very different sizes:

* the STATISTICS: the first occurrence of a Poisson process is a random
  variable - conditional on it happening at `x`, `rho*x` is Exponential(1), so
  a single observed first occurrence scatters by a factor `e`.  The root of
  `E[count] = 1` is the MEAN; the MEDIAN is the root of `rho*x = ln 2`, a
  fixed `-0.19` in `ln x` (= -0.083 decades) earlier for every length.  That is
  a free, exact improvement to the table.
* the PHYSICS: for `g <= 1476` the record in `data/prime_gap_merits.txt` IS the
  true first occurrence (the searches below 4e18 are exhaustive), so
  `E[count]` at it can be measured against Exp(1).  Over 693 such lengths the
  model gives median 1.3-1.8 and mean ~2.0 against Exp(1)'s 0.69 / 1.0, i.e.
  its `rho` is ~1.5-2x too large and `x*` ~20 % too small for `g <= 1400`;
  above `g ~ 1400` the sign flips (2x too late, n = 7).  That last band is the
  only genuinely wrong physics, and our campaigns do not reach it (their
  thresholds sit above it), so we cannot fix it - the constant-gap-law data can.

`--plot` writes `data/hl_validate_tables.png` (the `eps(L)` points with error
bars, and the first-occurrence histogram vs Exp(1)); `--csv` writes the
per-length measurements (`data/hl_validate_points.csv`, 514 points: campaign,
`L`, `R`, gap, observed, expected, `rho_meas`, `c1_eff`, `c1_table`) as
validation points the table's author can check directly.

`scripts/p0_record_forecast.py` - which records are still available at a fixed
scan scale (and how many the rest of a campaign will produce):

```
scripts/p0_record_forecast.py [--r-total 2e17] [--x X | --L L] [--glob GLOB]
                              [--gmin 1664] [--rate 2.254e11] [--top 12]
                              [--gain-min 1000] [--emin 1e-4] [--compact]
                              [--selftest]
```
`--compact` prints only the two summary lines the walk chain uses (see the chain
section above); everything else is unchanged.  The computation lives in
`analyze()`, which is pure (no printing, no `sys.exit`) so
`scripts/p0_walk_chain.py` can import it directly - `open_targets()` and
`fragility()` are shared the same way.
On an EXHAUSTIVE scan the range fixes the merit of every gap it finds:
`m(g) = g/L` (constant to ~1e-12 across the campaign), so "can this length still
be recorded?" is DETERMINISTIC - `win(g) <=> table_merit(g) < g/L`.  The only
randomness left is which winnable length shows up, at the HL rate `rho_g(L)`, so
`E[new records] = SUM_win (1 - exp(-rho_g * R))`.  The report prints

* the scan state (`R`, per-slice coverage, won lengths) and the density gate;
* the winnable / not-winnable / reachable counts, and `E[records]` for the
  scanned `R` **against the actual count** (a live model check);
* `E[NEW records]` for `R_remaining = --r-total - R`, with h/record;
* **open targets** ranked by `E[count]` (gap, our merit, table merit, margin,
  x gain, E[count]);
* **jackpots** above `--gain-min`, with an explicit frontier ("best gain at
  `P(hit) >= 1e-3`") because raw gain is meaningless without its reachability;
* **durability** of the records held: margin `< 0.05` / `< 0.13` marks a record
  that any small improvement breaks (same convention as `record_planner.py`).

Complementary to `scripts/record_planner.py`: that tool plans a COVERED hunt
(you choose L, merit is a design variable, the cost is `exp(E(m))` primality
tests, one target at a time).  This one reads an EXHAUSTIVE scan (L is given by
the range, merit comes free with every gap, the currency is integers scanned,
all reachable lengths are collected at once).  Their merit/table/margin triples
agree exactly (verified 2026-10-10 on 40 lengths, max diff 0.0).

`--selftest` re-checks nine invariants: the density gate, the winnable-set
partition, `E[records]` reproducibility, model-vs-actual within 3 sigma, the
monotonicity of the winnable set in the merit, and that the windowed mean of
`rho_g` falls with `g` (`rho_g` itself is NOT monotone - `S_g` oscillates with
`g` mod small primes).

`scripts/p0_pgs_bench.py` - head-to-head against PGS
(`briankehrig/prime-gaps-cuda`), the threshold-search tool class that drives the
Mersenne-forum distributed search:

```
scripts/p0_pgs_bench.py [--pgs-dir /tmp/pgcuda] [--start 2e20]
                        [--thresholds 900,1260] [--lengths 1e12,1e13]
                        [--out FILE] [--no-pgs | --no-ours] [--no-audit]
                        [--no-warmup] [--with-sieve] [--tmp DIR]
```
Protocol is the one in `docs/PHASE0_scan_bench.md` sections 16-17: same box,
same start, identical ranges for both tools.  Wall for us = process wall; wall
for PGS = the interval between the "Start time" and "End time" lines of their
`GapReport` file - their printed "Speed" meter runs 1.5-2x above that and is not
comparable.  Both walls include their own setup.  The script solves `wall = N x
steady + S` from the two `--lengths`, prints steady rates in ints/s and in B/s
(= 1e9 ints/s, PGS's own unit), the startup constants, and then the cross-tool
audit: the `(gap, lower prime)` sets of both tools must be identical.

PGS needs a tty (the wrapper draws progress, so it is run under `script(1)`) and
a build (`nvcc` compiles `prime_gaps.cu` on the first run; the default warm-up
spends one 1e12 run on it, `--no-warmup` skips that).  Their `settings.json` is
edited in place and restored on exit.  Per-threshold kernel settings are their
recommendation plus the values that ran clean on an 8 GB card - minGap 900 ->
`WORD_LENGTH 120`, `BLOCK_SIZE 60e9`; minGap 1260 -> `240` / `140e9` (their auto
value OOMs at 8 GB).  Thresholds missing from `PGS_PROFILE` use 240/140e9; add a
row there to tune another one.  Run it on an idle GPU only: the script warns via
`pgrep` if another scanner is alive, and a contended run reads low.

Scope: our walk engine and PGS are both threshold searches here ("every gap >=
minGap, no completeness claim below it"), which is NOT what the sieve engine does
(full enumeration: every prime, every gap >= merit 15).  Comparing the sieve
engine against PGS compares two different tasks; the walk engine is the
like-for-like one.  Our walk additionally runs MR and GMP-verifies every reported
gap, where PGS uses a Perig Fermat test (their PSP caveat starts below gap
1200).  `--with-sieve` also times the sieve engine at the same threshold (the
merit that lands exactly on `minGap` is `(minGap - 0.5)/ln(start)`, since its
threshold is `ceil(merit_min * ln(start))`); it is printed on its own line
because its wall includes work a threshold search never does - the enumeration
itself.  Measured results: `docs/PHASE0_scan_bench.md` sections 16-17 and 27.

`scripts/p0_deep_sweep.py` - is the sieve depth right for this band?

    python3 scripts/p0_deep_sweep.py --start 79000000000050000000000000000 \
        --length 4e12 --gap-min 1664 --depths 15000,40000,100000,250000,500000,1000000

The walk engine runs two pipelined stages (GPU mark into the class-30 bitmap,
then the batched jump-walk with the primality tests) and reports both stage
spans.  Both stages use the whole GPU, so what has to be minimized is the sum,
not the longer one: the tool therefore runs two arms per depth - the production
geometry, plus `--gap-min 10000000`, which makes the walk trivial so the
reported mark span is the mark alone (isolated), uncontaminated by pipeline
overlap.  Same start, same `--walk-batch`, same binary for every depth; the
reported gap set must be identical to the reference depth's (a deeper sieve
only removes tests, never changes which numbers are reported) and the exit
status is non-zero if it is not.  It stops the GPU from being shared: run it
when no slice is active, and it passes `--no-records` so the campaign's records
file is never touched.  `--length` accepts `4e12` (the engine itself does not).
Measured depths for the 7.9e28 band, and why 100000 won there at the old 64 KB
tile: section 28 of `docs/PHASE0_scan_bench.md` (the 92 KB tile moved the optimum
to 150000, and the 2026-10-30 wall-band test + chunk-major mark order moved it
back to 100000 on the dev 3070 and to 85000 or lower on the fleet's 3060s - see
the `--walk-primes` row in §2).

---

## 5. Acceptance gates (run after ANY kernel/pipeline change)

| gate | reference |
|---|---|
| `./bin/phase0_scan_gpu --check 200000` | `CHECK PASS: 4293 primes` |
| `./bin/phase0_scan_gpu --check 2000000` | `CHECK PASS: 42725 primes` |
| `./bin/mr68_gpu --validate 2000000` | `0 mismatches` (85563 primes) |
| `./bin/mr128_bench --validate 3000 0 121` | `VALIDATE PASS (0 mismatches vs GMP)` - the 2 x 64-bit walk kernel above 2^120; bits 121/125/130 must pass, 96 is outside its `2^96 < n < 2^128` domain and fails by design (§29) |
| `./bin/bench_p0sieve --verify 3` | `3 segment(s) bit-exact vs naive CPU marker` |
| `make test-phase0-sieve-counts WITH_CUDA=1` | CPU, monolithic and split scanner startup counts match 28 inclusive limits, including 0, even/odd boundaries, 10,000,000 and 10,000,001; the runs use CPU marking and do not require a GPU device. |
| `./bin/bench_u96 gate 200000`, `./bin/bench_u96 sweep 64 1048576` | the walk's mode-4 test (build line: §1) is exact: `gate N=200000 : gpu residue mismatches=0  host mismatches=0  verdict mismatches=0` and `sweep chunks=64 N=1048576: primes u96=2027625 perig=2027625 diff=0` (measured 2026-10-30; the larger verdict runs are in `docs/PGN_SNAPSHOT_REVIEW_2026-10-20.md` §12).  Run it after ANY change to `tools/p0_test_u96.cuh` |
| `./bin/bench_u108 dispatch`, `./bin/bench_u108 step 200000`, `./bin/bench_u108 gate 200000`, `./bin/bench_u108 sweep 64 1048576` | Dispatch boundaries, step bounds, GMP residues and perig verdicts must have zero failures/differences. Run after changes to the mode-5 test or walk dispatch. |
| walk test-mode parity (1e32) | `--engine walk --start 100000000000000000000000000000000 --length 2000000000000 --gap-min 1300 --walk-primes 150000 --walk-batch 96 --no-records --log L`: default, `P0_WALK_TESTMODE=0`, forced mode 5, profiled default and `bin/phase0_scan_gpu_split` each have **185 identical verified gaps and `tests=5711269146`**. |
| walk test-mode parity (wall band) | `--engine walk --start 79000000000010000000000000000 --length 1200000000000 --gap-min 1100 --walk-primes 150000 --walk-batch 96 --no-records --log L` with the default (mode 4) and with `P0_WALK_TESTMODE=0` (perig), and with `bin/phase0_scan_gpu_split`: **428 gaps and `tests=3671852807` in all three**, gap sets identical after stripping the session-id column (measured 2026-10-30; the fleet's two RTX 3060s give the same 428 gaps and `tests=3671852807` with the default and with `P0_WALK_TESTMODE=0`, and at `--walk-primes 85000` the same 428 gaps with `tests=3855397099`) |
| `./bin/phase0_scan_gpu --length 500000000 --merit-min 10 --log L` | `204` rows, top-4 `724/690/684/682` (merits `15.4883/14.7610/14.6326/14.5898`), 0 unverified |
| fast vs legacy equivalence | `--merit-min 10` on 1e10: 4439 rows, byte-identical between default and `--legacy-mr` (sorted, timestamp stripped) |
| hard-composite fixture | `--start 18447233110000000000 --length 5000000000 --merit-min 15` contains `gap=710 merit=16.0049 verified=1`, 0 unverified, and the semiprime `18447233112263860537 = 4015729 x 4593744526153` never appears as a prime.  **Not** a pseudoprime despite the old label - see the correction in §2 (measured 2026-11-04: `2^(n-1) mod n != 1`).  Exercises "two factors above any sieve depth must not be prime", nothing about test strength.  CPU-side, no GPU: `scripts/psp_probe.py --fixtures` |
| ground truth vs BrianK's published list | `scripts/psp_probe.py --brik` must print the misses as **61 / 37 / 26 / 0 / 0** for Fermat / Euler/Jacobi / strong base 2 / strong 2+3 / strong 2..37 over the 61 published `>= 800` pseudoprime gaps in `1e20..1e21`.  Any change to a discovery test or to `w_test_mode` that moves these numbers is a correctness event. |
| pseudoprime fixtures (all three classes) | three small windows, no GPU needed for the first: `scripts/psp_probe.py --fixtures` prints the class of each interior number and whether each engine finds the gap.  Against the binary: **(a) gap 156, Fermat-but-not-Euler** `108081990803 = 42443 x 2546521` - `--engine walk --start 108081990600 --length 600 --gap-min 150 --walk-primes 30000 --walk-batch 8` must log `108081990733 156 6.1402 ... verified=1` (our Euler/Jacobi test DOES catch this class); **(b) gap 956, Euler/Jacobi** - `--engine sieve --start 160696461657079904863 --length 3000 --merit-min 20 --sieve-limit 30000` must log `160696461657079905863 956 20.5476 ... verified=1` while `--engine walk` (`--gap-min 900 --walk-primes 30000 --walk-batch 8`) prints **0 rows** (the documented limitation, pinned on purpose: if the walk starts reporting it, the test selection changed and §2 must be re-read); **(c) gap 710** is a hard composite, not a pseudoprime, and both engines find it. |
| pseudoprime endpoint rejected at emission | the publish-side guard, and the only thing that stops a pseudoprime from becoming a *claim*.  `--engine walk --start 160696461657079904863 --length 3000 --gap-min 400 --walk-primes 30000 --walk-batch 8 --log L` must log **exactly two rows and both `verified=0`**: `160696461657079905863 410 8.8123 160696461657079906273` and `160696461657079906273 546 11.7354 160696461657079906819`.  The walk accepts the Euler/Jacobi pseudoprime `...906273` and splits the 956 gap there, so the *emission* happens - `verify_gap()` (`mpz_probab_prime_p(lower,25)` **and** `mpz_nextprime(lower) == upper`) is what turns the flag off, and `process_prime_gap()` counts it in `gaps_bad`.  A `verified=1` on either row means the guard is off and every published gap is suspect.  Reproducible (2026-11-05: two runs, data rows identical). |
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

**Continuum chain (`scripts/p0_walk_chain.py`, 2026-10-03).**  Runs walk
slices back-to-back over an open-ended range: after every completed slice
`start += length`, with per-slice `--state/--log/--out`
(`p0_walk_<tag>_c<k>_g<G>.*`).  `--devices 0,1` runs one scan PER GPU, each
taking the next slice from a shared queue (the ledger records the device).
Existing slices are detected at start: completed ones are skipped,
unfinished ones (e.g. after a kill) are requeued first, so rerunning the
same command restarts the continuum (`data/p0_walk_<tag>_chain.state` keeps
the fixed `start0/index0`).  A slice that does not complete drains the
chain (no new launches, exit 1); a lock file and a live-scanner check
refuse a second chain on the same tag.  Restart through the chain (rerun its
command), never with hand-started `phase0_scan_gpu` processes: a bare scanner
completes only its own slice and exits, and with the chain gone nothing
launches the next one.  Before every slice it re-derives
the shortest WINNABLE length at that start (shortest even `g` with
`g/ln(start) > table(g)` - the shortest length that would beat the record
table) and warns in BOTH directions: `--gap-min` below it only adds
non-record rows, above it MISSES the winnable band `[g, gap-min)`.  Note
`--gap-min` must be a winnable length, not an arbitrary round number: at
1.33e20 the shortest is 1586 (1576 is *not* winnable - its table entry is
merit 35.08 vs 34.01 achievable).  Per-slice ledger:
`data/p0_walk_<tag>_chain.log` (ts, slice, dev, start, len, gap_min, rc,
wall, gaps, walk_batch, walk_primes, complete; `walk_primes=0` means the
engine default was used, so slices scanned at different depths stay
distinguishable).

After every COMPLETED slice the chain prints a two-line record forecast from
`scripts/p0_record_forecast.py` (imported, not shelled out), which answers "what
can this campaign still win, and how fast":

```
[chain] forecast: won=14  model=14.7 (-0.2 sigma)  E[new in the remaining 1.511e+17]=21.3 -> 8.7 h/record
[chain] forecast: top open: 1906 (107.3x, E=2.26)  1930 (1.187x, E=2.08)  1956 (29.18x, E=2.00) | fragile 7/14
```
`won` counts distinct lengths with a `record=NEW` event in the tag's logs,
`model` is the HL expectation for the same scanned `R` (a live check of the
model against reality), `fragile` counts held records whose merit margin is
below 0.13.  `--no-forecast` disables the lines; `--forecast-top N` (default 3)
sets how many open targets are listed; `--rate` (default 2.254e11 ints/s) is the
rate used for the h/record figures.  The forecast never takes the chain down: a
missing table or an unparseable log degrades to a single `forecast: unavailable`
line.  The "remaining" branch needs the WHOLE campaign's integers: that is
`--r-total` (default 0 = `--slices * --length`, which is only right on a chain's
first run).  Since `--slices` is per-invocation, a restart would otherwise
understate the remaining work - pass the campaign total in `--r-total` and let
`--slices` set how many slices THIS run adds (e.g. restart with `--slices 16
--r-total 2e17` for slices 5..20 of a 2e17 campaign).  A `--slices 0` (forever)
chain without `--r-total` reports the per-1e16 yield instead.

`--walk-primes P` (default: the engine's 150000) is passed straight to every
slice, so the whole campaign can run at one sieve depth.  The depth is a
property of the band, of the mark tile AND of the primality test: with the old
binary at the 7.9e28 wall and the shipped 92 KB tile the measured optimum was
**150000** (2.45-2.48e11 ints/s, vs 2.39 at 120000 and 2.33 at 180000), while
250000 and above lose because the marking grows faster than the tests fall
(`docs/PHASE0_scan_bench.md` §28, `scripts/p0_deep_sweep.py`).  With the
2026-10-30 binary (TESTMODE 4 + chunk-major mark) a test is cheaper and the
optimum moved to a shallower sieve on both cards that were measured: **100000**
on the dev RTX 3070 (12.70 s vs 12.99 s at 150000 per 4.8e12, -2.2 %) and
**85000** (the lowest depth tried) on the fleet's RTX 3060s (21.57 s vs 22.12 s,
-2.5 %, both cards within 0.3 points; 100000 is -1.8 %).  The fleet wall has
therefore run with `--walk-primes 85000` since 2026-10-30 17:15.  The depth can
be changed between restarts - a slice resumes from its checkpoint at the new
depth (the state file does not store it), and the reported gaps are identical at
every depth, so the census stays consistent across slices taken at different
depths (the chain writes `walk_primes=` into its ledger).  With the old binary
keep 150000.  Example: the fleet's restart at the measured depth, which also
passes the campaign total:

```bash
nohup setsid python3 scripts/p0_walk_chain.py \
    --start 79000000000010000000000000000 --length 10000000000000000 \
    --gap-min 1664 --tag wall --walk-batch 96 --walk-primes 85000 \
    --slices 11 --r-total 2e17 > /tmp/wall_chain_stdout.txt 2>&1 &
```

```bash
# one GPU / two GPUs (fleet box):  add --devices 0,1
nohup setsid python3 scripts/p0_walk_chain.py \
    --start 133001070000720000000 --length 1070000000000000 \
    --gap-min 1586 --tag 1p33e20 --index-start 3 --devices 0,1 \
    > data/p0_walk_1p33e20_chain.out 2>&1 &
```

**Record harvest watcher (`scripts/p0_records_to_submit.py`, 2026-10-08).**
Turns any walk log into a submit-ready batch file while the chain runs.  The
walk engine flushes one line per logged gap
(`<epoch> <lower> <gap> <merit> <upper> verified=<0|1> record=<NEW|no|absent|off> table=<best>`),
so the watcher harvests the `record=NEW` lines, keeps the strongest start prime
per gap length, recomputes `merit = gap/ln(lower)` (the site's convention) and
regenerates `records_to_submit_phase0.txt` (`gap merit prime_start`, comments
ignored by `scripts/submit_records.py`).  Every row is re-verified independently
with `scripts/verify_gap_candidate.py` (OpenSSL endpoints + strict scan of all
odd intermediates, ~2.5 s per row) before it enters the data section; failed
rows, rows already POSTed and rows the table has caught are listed as comments
only.  The file is a pure function of (logs, sent-file, verify-cache), so
restarting the watcher is harmless.

| flag | default | effect |
|---|---|---|
| `--logs GLOB...` | `data/p0_*.log` | gap logs to harvest |
| `--max-log-mb N` | `64` | skip logs larger than N MB (0 = no limit); oversized files are named in a warning |
| `--out FILE` | `records_to_submit_phase0.txt` | generated batch |
| `--interval S` | `30` | watch poll interval (>= 5) |
| `--once` | off | one pass, then exit (cron/systemd friendly) |
| `--discoverer NAME` | `D.Benko` | name written into the suggested send command |
| `--sent-file FILE` | `data/p0_records_sent.txt` | POSTed rows, excluded from the batch |
| `--state-file FILE` | `data/p0_records_watch.state` | liveness line (`pass=`, `rows=`, `time=`) |
| `--verify-cache FILE` | `data/p0_records_verify.json` | per-row verification verdicts |
| `--no-verify` | verification ON | write rows without independent verification |
| `--reverify` | off | ignore cached verdicts and re-run them |
| `--check-live` / `--live-interval S` | off / `1800` | fetch merits.txt and exclude rows the live table has caught |
| `--mark-sent FILE` | — | append FILE's rows to the sent-file, then do one pass |
| `--dry-run` / `--quiet` | off | print the batch instead of writing / no heartbeat |

```bash
nohup setsid python3 -u scripts/p0_records_to_submit.py --check-live \
    >> data/p0_records_watch.out 2>&1 &
tail -f data/p0_records_watch.out          # progress; data/p0_records_watch.state = liveness

python3 scripts/p0_records_to_submit.py --once --dry-run    # inspect, write nothing
python3 scripts/p0_records_to_submit.py --mark-sent records_to_submit_phase0.txt
```

Submit the batch only while the site's commit limiter is idle - a refused POST
shadows those numbers for ~9 h (the site then answers `Already processed`) - and
mark it afterwards so the watcher stops offering it.

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
| `data/p0_rec_1e32_vs_hl.png`, `data/p0_rec_1e32_vs_hl_dark.png` | dev record campaign (x = 1e32) vs the HL tables: census, per-length census, first occurrence, record yield |
| `data/hl_validate_tables.png` | out-of-sample test of the HL tables at L = 46.7 / 66.5 / 73.7, and the first-occurrence diagnostic |
| `data/hl_validate_points.csv` | 514 per-length validation points (campaign, L, R, gap, observed, expected, rho_meas, c1_eff, c1_table) - hand-off for the table's author |
| `data/prime_gap_merits.txt` | best-known gap-length merit table used by the record check (123,436 lengths) |
| `records_to_submit_phase0.txt` | GENERATED submit batch (own records, `D.Benko`); see the harvest watcher in § 6 |
| `data/p0_records_sent.txt` | rows already POSTed - excluded from the batch (written by `--mark-sent`) |
| `data/p0_records_watch.state`, `data/p0_records_watch.out` | watcher liveness line + progress log |
| `data/p0_records_verify.json` | per-row independent-verification verdicts |
| `p0.state` | leftover checkpoint of an aborted 6-thread attempt (~1-3% done, done Sep 30); not used by production runs |
| `docs/PHASE0_scan_bench.md` | full measurement history (sections 1-13), gates provenance, HL comparison, rejected experiments |

The white-paper Phase-0 corpus is the pair
`data/p0_campaign_2e20.log` + `.out`; anything else in `data/` belongs to the
miner.
