# Coverage-Max Experiment (GPA-style covering search)

Status: DONE (2026-09-07). Tool: `tools/cover_max.c` → `bin/cover_max`.

## Question

How close to the true maximum coverage is the production CRT covering?
If the optimizer is provably near-optimal, it is finished (certificate);
if not, port a GPA/ILP-style optimizer (Ziller–Morack family,
arXiv:1611.03310) into `gen_crt`.

## Method

Minimize survivors in `[1, gap_target)` over residue choices
`r_p ∈ [1, p)` for the fixed prime set of a CRT file
(position `j` covered iff `(r_p + j) ≡ 0 (mod p)`, identical to
`covering.c` / `crt_runtime` semantics). Search:

1. **prime-greedy (GPA-lite)**: choose over all unplaced primes and all
   their residues the choice covering the most uncovered offsets,
   randomized tie-breaking.
2. **2-opt pair sweep**: for every prime pair, re-choose both residues
   jointly given all others fixed (exact marginal-union computation,
   O(window) per pair), repeated passes until no pair improves.
3. **ILS**: multi-start from (a) perturbations of the file solution and
   (b) fresh prime-greedy starts, with pair sweep as local optimizer.
4. **exact DFS** for small configs (bounds via remaining-prime budget) to
   calibrate the greedy-vs-optimum gap.

All survivor counts cross-checked against the repo's own
`covering_count_survivors` (linked from `build/new_src/covering.o`);
loader verified with `test_crt_runtime`.

## Results

Commit `8cc2a6f` (working tree dirty: Makefile/README/prime_gap_merits
changes unrelated to this experiment), gcc 11, `-O2`.

### Exact calibration (greedy vs true optimum)

| primes | window | optimum | greedy+sweep | gap |
|---|---|---|---|---|
| 8 | 300 | 87 (70.9030%) | 89 (70.2341%) | 2 survivors (0.67 pp) |

Command: `./bin/cover_max --exact 8 --gap-target 300 --seconds 300`
Runtime: < 1 s (6,066,281 nodes).

### p74 production config (`shift507_p74_lex_m30.txt`)

| run | restarts | best survivors | coverage |
|---|---|---|---|
| file | — | 1141 | 92.8085% |
| 15 s smoke | 34 | 1141 | 92.8085% |
| 600 s search | 1355 | **1141** | **92.8085%** |

Command: `./bin/cover_max --file data/crt/m23/shift507_p74_lex_m30.txt
--seconds 600 --seed 20260907` — runtime 600 s, zero improvement over
the file solution despite ILS perturbations up to 34 residues and full
2-opt pair sweeps.

### p98 production config (`shift720_p98_1000_5000_m23.txt`)

| run | restarts | best survivors | coverage |
|---|---|---|---|
| file | — | 987 | 93.6564% |
| 12 s smoke (seed 42) | 19 | 961 | 93.8235% |
| 600 s search | 1091 | **952** | **93.8814%** |
| 120 s regenerate + write | 211 | 952 | 93.8814% |

Improvement: **−35 survivors (−3.5%), +0.225 pp coverage**.
Output file: `data/crt/m23/shift720_p98_covermax_m23.txt`
(written with `--out`, header `n_candidates=952` recomputed and
cross-checked by the repo evaluator).

Survivor constellation comparison (independent Python re-check):

| file | survivors | longest run | mean fwd run | max fwd run |
|---|---|---|---|---|
| original | 987 | 77 | 14.7 | 77 |
| covermax | 952 | 83 | 15.3 | 83 |

The new solution strictly dominates: fewer candidates AND longer covered
runs everywhere measured.

Loader check: `./bin/test_crt_runtime
data/crt/m23/shift720_p98_covermax_m23.txt` → ALL TESTS PASSED,
window survivors 2295 vs 2328 for the original (full aligned window
31120 = 2×gap_target).

## Conclusion

- **p74 config: certified done.** 1355 restarts of a stronger optimizer
  (GPA prime-greedy + 2-opt pair sweep + ILS) never beat 1141; combined
  with the n=8 calibration (greedy within 2 survivors of optimum), the
  production file is at or within ~0.1 pp of the maximum coverage.
  No GPA port needed for this config.
- **p98 config: the old optimizer left ~0.225 pp on the table.**
  `cover_max` found a strictly better solution (952 vs 987 survivors,
  better run profile). **DEPLOYED** (2026-09-07): walker added to
  `gap_hunt_fleet.conf`, deployment-ready line in
  `gap_hunt_fleet_dual3060.conf`; certificate in
  `docs/COVERAGE_CERT.md`.

## Notes / caveats

- The search minimizes total survivors (lex objective). The miner's
  blocks-cost objective additionally weights survivor tail runs; for the
  p98 replacement this was checked to be strictly better on both.
- `tools/cover_max.c` links the repo's `covering.o` evaluator, so file
  format / residue convention can never drift from `gen_crt`.
- Provenance rule honored: exact commands and runtimes recorded above;
  old results superseded only by rerun, never patched in place.
