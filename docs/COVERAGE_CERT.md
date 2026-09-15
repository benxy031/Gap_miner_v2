# Coverage Certificates (CRT covering optimality + deployment)

Status: ISSUED 2026-09-07. Companion to `docs/COVERAGE_MAX_EXPERIMENT.md`
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

## Deployment record

| Change | File |
|---|---|
| Active walker added (device 0, min-merit 8) | `gap_hunt_fleet.conf` |
| Deployment-ready walker commented (activate on a free GPU) | `gap_hunt_fleet_dual3060.conf` |
| Certificate + experiment report | `docs/COVERAGE_CERT.md`, `docs/COVERAGE_MAX_EXPERIMENT.md` |
| Search tool + Makefile target | `tools/cover_max.c`, `Makefile` (`bin/cover_max`) |

Deployed covers: p98 `shift720_p98_covermax_m23.txt` (952 survivors) and p67
`shift450_p67_covermax_m30.txt` (1101 survivors). Certified, deployment-ready:
p130 `shift1017_p130_covermax_m30.txt` (1544 survivors).

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
