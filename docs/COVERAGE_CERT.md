# Coverage Certificates (CRT covering optimality + deployment)

Status: ISSUED 2026-09-07; Certificates E and F added 2026-09-20. Companion to
`docs/COVERAGE_MAX_EXPERIMENT.md`
(full method, provenance, runtimes). Tool: `tools/cover_max.c` →
`bin/cover_max`, evaluator linked from the repo's own `covering.o`.

## Certificate A — p74 config optimality (issued)

- File: `data/crt/m23/shift507_p74_lex_m30.txt`
- Config: 74 primes, `gap_target = 15867`, merit 30, shift 507
- **Certified value: 1141 survivors = 92.8085% coverage.**
- Evidence: 1355 restarts / 600 s of prime-greedy (GPA) + 2-opt pair sweep
  + ILS (perturbations up to 34 residues) never found fewer survivors.
  Calibration on exactly-solved small configs (n=8, window 300: optimum 87
  vs greedy 89) shows greedy-class search lands within ~2 survivors of the
  true optimum.
- Layer: *derived inference* — "at or within ~0.1 pp of the maximum
  coverage" follows from the calibration plus the search evidence; the
  exact global optimum for 74 primes is not computed (exhaustive search
  is astronomically infeasible).

## Certificate B — p98 config improvement + deployment (issued)

- Old file: `data/crt/m23/shift720_p98_1000_5000_m23.txt`
  → 987 survivors = 93.6564% coverage.
- **Deployed file: `data/crt/m23/shift720_p98_covermax_m23.txt`
  → 952 survivors = 93.8814% coverage** (−35 survivors, −3.5%,
  +0.225 pp coverage).
- Evidence: 1091 restarts / 600 s; best found at elapsed 35 s, stable for
  the remaining ~1000 restarts. Independent Python re-check of the
  constellation: longest covered run 83 vs 77, mean forward run 15.3 vs
  14.7 — strictly better on every measured statistic, no regression.
- Validation: `bin/test_crt_runtime data/crt/m23/shift720_p98_covermax_m23.txt`
  → **ALL TESTS PASSED** (window 31120 survivors: 2295 vs 2328 for the old
  file). Header `n_candidates=952` recomputed and cross-checked against the
  repo evaluator before writing.
- Layer: *exact fact* (the two survivor counts and test outcomes above).

## Certificate C — p67 config improvement + deployment (issued 2026-09-07)

- Old file: `data/crt/m23/shift450_p67_m30.txt`
  → 1108 survivors = 92.4523% coverage.
- **Deployed file: `data/crt/m23/shift450_p67_covermax_m30.txt`
  → 1101 survivors = 92.5000% coverage** (−7 survivors, +0.048 pp).
- Evidence: 300 s / 1385 restarts; best found at elapsed 1 s, stable for the
  remaining restarts. Constellation: longest run 65 vs 69 (slightly shorter),
  mean forward run 12.3 vs 12.2 (slightly better) — near-neutral; the win is
  the candidate reduction (−0.6% GPU tests).
- Validation: `bin/test_crt_runtime data/crt/m23/shift450_p67_covermax_m30.txt`
  → **ALL TESTS PASSED** (window 29362 survivors: 2524 vs 2541 for the old
  file).
- Layer: *exact fact* (the two survivor counts and test outcomes above).

## Certificate D — p130 config improvement (issued 2026-09-12)

- Old file: `data/crt/m23/shift1017_p130_lex_m30.txt`
  → 1563 survivors = 94.0954% coverage.
- **Certified file: `data/crt/m23/shift1017_p130_covermax_m30.txt`
  → 1544 survivors = 94.1672% coverage** (−19 survivors, +0.072 pp).
- Evidence: 600 s / 397 restarts; best found at elapsed 180 s (restart
  114), stable for the remaining ~280 restarts. Independent Python re-check
  of the constellation: longest covered run 123 vs 117, mean forward run
  16.1 vs 15.9 — strictly better, no regression.
- Validation: `bin/test_crt_runtime data/crt/m23/shift1017_p130_covermax_m30.txt`
  → **ALL TESTS PASSED** (window 52944 survivors: 3739).
- Layer: *exact fact* (the two survivor counts and test outcomes above).
- Deployment: fleet lines pending user decision (merit-30+ hunt setup).

## Certificate E — p128 m40 config, new best of the shift998 family (issued 2026-09-20)

Same geometry across the whole family (128 primes, `shift 998`,
`gap_target 34769`, merit 40.00) — so the comparison is exact. Survivors were
recomputed independently from the `(p, r)` rows for every file; **all 7 headers
matched the recomputation**:

| file | survivors | coverage |
|---|---|---|
| `shift998_p128_strong_m40_run.txt` | 2857 | 91.78 % |
| `shift998_p128_weaklex_m40.txt` | 2357 | 93.22 % |
| `shift998_p128_strong_m40_lex.txt` | 2350 | 93.24 % |
| `shift998_p128_strong_m40_s300.txt` | 2344 | 93.26 % |
| `shift998_p128_strong_m40.txt` | 2191 | 93.70 % |
| Old file: `shift998_p128_lex_m40.txt` | 2186 | 93.71 % |
| **Certified: `shift998_p128_covermax_m40.txt`** | **2177** | **93.7385 %** |

- **Certified value: 2177 survivors = 93.7385 % coverage** (−9 survivors,
  −0.41 %, +0.03 pp over the old `lex_m40` file; −680 over the worst sibling).
- Evidence: **337 restarts / 600 s** (`--seed 20260920`), best found at restart 0
  (elapsed 2 s) and no improvement in the remaining 336 restarts; the search's
  `--out` file came back **byte-identical** to the input, i.e. the constellation
  is a fixed point of the search. `cover_max --file` reports
  `file n_candidates=2177, evaluator says 2177 (MATCH)`.
- Independent Python re-check of the constellation (covermax vs lex): longest
  covered run **111 vs 111** (tie), mean forward run **14.96 vs 14.90** (better),
  survivors **2177 vs 2186** (better) — no regression on any measured statistic.
- Validation: `bin/test_crt_runtime data/crt/m23/shift998_p128_covermax_m40.txt`
  → **ALL TESTS PASSED** (`window primes=39  natural~44.0  ratio=0.89`, i.e. the
  aligned window is demonstrably prime-poor; window 69 538, survivors 5 088).
- Geometry check: `log2(prod p) = 989.92 < 998` → **8.08 bits of headroom**, so
  the primorial still fits inside the shift (the `ctr_bits = 8` convention).
- Note on the target: `gap_target 34769` is **odd**, so the table-credited length
  is the even **34 768** (merit 39.9998 ≈ 40.00); the covering marks offsets
  34 763–34 769, the first uncovered offset is 16 and the last 34 762. A future
  generator revision should print the even target.
- Layer: *exact fact* (every survivor count, coverage figure, test outcome and
  search datum above). "Cannot be improved within the search class of
  `tools/cover_max.c`" is *derived inference* from 337 restarts — no exact
  global optimum is computed.
- Deployment: **certified, not deployed** — fleet lines pending user decision.
  For a merit-40 hunt the line is this file, one walker per GPU,
  `--gap-hunt-min-merit 20` (a low threshold reports more gaps per hour at the
  same record rate, see `docs/RECORD_RATE_MODEL.md` §4).

## Certificate F — p43 m40 config improvement (issued 2026-09-20)

- Old file: `data/crt/m23/shift258_p43_strong_m40.txt`
  → 1311 survivors = 90.8006 % coverage.
- **Certified file: `data/crt/m23/shift258_p43_covermax_m40.txt`
  → 1294 survivors = 90.9199 % coverage** (−17 survivors, −1.30 %,
  +0.12 pp).
- Evidence: **6129 restarts / 600 s** (`--seed 20260920`), best found at
  **restart 36 (elapsed 4 s)** and no improvement in the remaining ~6090
  restarts. Unlike Certificate E, the input file was **not** a local optimum:
  the search beat it within seconds, so this is an *improvement* certificate
  (the B/C/D shape); the optimality layer applies to the NEW file only.
- Independent Python re-check of the constellation (new vs old): longest
  covered run **107 vs 67**, mean forward run **10.01 vs 9.87**, survivors
  **1294 vs 1311** — strictly better on every measured statistic, no
  regression.
- Validation: `bin/test_crt_runtime data/crt/m23/shift258_p43_covermax_m40.txt`
  → **ALL TESTS PASSED** (window 28504 survivors **2794**, vs 2814 for the old
  file; `window primes=39  natural~44.0  ratio=0.89`). Header recomputed
  independently and cross-checked against the repo evaluator:
  `cover_max --file` → `evaluator says 1294 (MATCH)`.
- Geometry unchanged: 43 primes (2..191), `shift 258`, `gap_target 14252`,
  merit 40.00; `log2(prod p) = 249.19 < 258` → 8.81 bits of headroom.
- Target context (why this cover exists): gap 14 252 at shift 258 is merit
  **40.0025**, and the table requires **28.4224** at that length (neighbours
  27.97 / 29.44 / 31.65) — a merit-40 gap here is a **+11.6 record at 155
digits**. The rate is the reason to run this geometry at all: 514-bit
  candidates (CGBN AL=9) cost ~5x less per test than the 1273-bit ones at
  shift 998 (AL=20), against the weaker cover that 43 primes can buy
  (9.20 % vs 6.26 % pre-sieve survivors).
- Layer: *exact fact* (every count and test outcome above). "Improved within
  the search class of `tools/cover_max.c`" is *derived inference* from 6129
  restarts; no exact global optimum is computed.
- Deployment: **certified, not deployed** — fleet lines pending user decision.

## Deployment record

| Change | File |
|---|---|
| Active walker added (device 0, min-merit 8) | `gap_hunt_fleet.conf` |
| Deployment-ready walker commented (activate on a free GPU) | `gap_hunt_fleet_dual3060.conf` |
| Certificate + experiment report | `docs/COVERAGE_CERT.md`, `docs/COVERAGE_MAX_EXPERIMENT.md` |
| Search tool + Makefile target | `tools/cover_max.c`, `Makefile` (`bin/cover_max`) |

Deployed covers: p98 `shift720_p98_covermax_m23.txt` (952 survivors) and p67
`shift450_p67_covermax_m30.txt` (1101 survivors). Certified, deployment-ready:
p130 `shift1017_p130_covermax_m30.txt` (1544 survivors), p128
`shift998_p128_covermax_m40.txt` (2177 survivors, Certificate E) and p43
`shift258_p43_covermax_m40.txt` (1294 survivors, Certificate F).

Commit at deployment: `8cc2a6f` (working tree also carries unrelated
README/merits changes).

## Re-certification procedure

To re-certify any CRT file:

```bash
make bin/cover_max
./bin/cover_max --file <crt-file> --seconds 600 --seed 20260907 \
    --out <new-crt-file>
./bin/test_crt_runtime <new-crt-file>   # must print ALL TESTS PASSED
```

Only a rerun that beats the certified value supersedes it; certified
numbers are never patched in place.
