# Record-rate model — validated on collected hunt data

Status: **validated for ranking** (not for absolute projections).
Cost: zero GPU time — everything below comes from data already on disk.
Tool: `scripts/record_rate_model.py` (pure Python 3, stdlib only).

Provenance: commit `141809a`, 2026-09-17, table
`data/prime_gap_merits.txt` sha256[:12] `32c910d92e49` (122 611 entries),
data `gap_hunt_records_f1.txt` (755 202 gaps) and `gap_hunt_records_f2.txt`
(233 656 gaps).

Canonical commands:

```
scripts/record_rate_model.py gap_hunt_records_f1.txt gap_hunt_records_f2.txt
scripts/record_rate_model.py --targets 6 gap_hunt_records_f2.txt
scripts/record_rate_model.py --sweep-max 28 gap_hunt_records_f2.txt
scripts/record_rate_model.py --shift-scan 200:1050:25
scripts/record_rate_model.py --shift-scan 900,925,950,975,1000,1017,1025,1050 \
    --sigma-fixed 1.3716
```

## 1. Method

**Record criterion** (identical to `scripts/watch_gap_hunt_records.py` and
`new_src/record_log.c`): a gap is a record iff

```
gap size g is present in the table AND merit > best_known_merit[g]
```

FACT: all 988 858 gaps in f1 and f2 have their size present in the table, so
the predicate is well defined for every collected gap (no "unknown size"
escape hatch).

**Sizes are quantised by the walk's log-base.** A hunt anchor is
`2^(255+shift)`, so `L = (255+shift)*ln2` and a gap with merit `m` has size
`g ≈ m*L`; measured medians reproduce this exactly (`L = 528.178` for
shift507, `881.683` for shift1017). One merit unit therefore spans `L` gap
sizes (528 and 882 respectively), and the bin of one integer size is
`1/(L*sigma)` wide in merit.

**Model.** With `S(x) = exp(-(x-m0)/sigma)` for the merit tail above the
report threshold `m0`:

```
p_record(L, sigma, m0) = SUM over table sizes g of
    S(max(lo_g, mu(g), m0)) - S(max(hi_g, m0)),
lo_g = (g-0.5)/L, hi_g = (g+0.5)/L, mu(g) = best_known_merit[g]
```

The clamp at `m0` is essential and not cosmetic: only gaps with merit >= `m0`
are ever reported, so bins entirely below `m0` contribute exactly zero (a
size-2 gap cannot be reported however easy its record is). `sigma` is the
truncated-exponential MLE `mean(m - m0 | m >= m0)`.

**Observable.** `d_best = max over gaps of (merit - mu(size))` is the walk's
closest approach to any record. The model predicts its distribution through

```
P(d_best <= x) = (1 - p_record(margin=x))^n
```

which is what makes the validation testable without a single GPU-second.

## 2. Validation

| | f1 (shift507, 763-bit) | f2 (shift1017, 1273-bit) |
|---|---|---|
| reported gaps | 755 202 | 233 656 |
| gap size range | 4226..12504 | 7054..21690 |
| max merit | 23.6738 | 24.6007 |
| tail sigma (M0=8) | 1.2618 ± 0.0015 | 1.3716 ± 0.0028 |
| observed records | **0** | **0** |
| observed `d_best` | −6.7532 (−5.35 σ) | −3.9694 (−2.89 σ) |
| predicted `d_best` median | −4.58 | **−4.00** |
| observed CDF position | 0.112 | **0.514** |
| E[records], P(observe 0) | 0.035, 0.965 | 0.087, 0.916 |
| gaps per record | 2.141e7 | **2.677e6** |

Out-of-sample (fit sigma on the first half by anchor order, predict the rest):

| | f1 | f2 |
|---|---|---|
| E[records] in holdout, observed | 0.018, 0 | 0.043, 0 |
| P(observe 0) | 0.983 | 0.958 |
| holdout `d_best` vs predicted median | −6.7532 vs −5.9065 | −3.9694 vs −5.2287 |
| holdout CDF position | 0.334 | 0.720 |

Verdict (INFERENCE): the model is adequate for ranking configurations. f2's
closest approach lands on the model median to 0.03 merit units; both holdout
counts are consistent with zero observed records; all four closest-approach
positions sit inside the central 90 % of the predicted distribution, with f1
in-sample the lowest at the 11th percentile. The model is *not* claimed to
predict absolute record times better than a factor of ~2 (sigma ± 5 % moves
gaps/record by 2–3×, see §5 of the tool output).

## 3. Finding: the frontier binds on SIZE, not on merit height

FACT — the nearest targets for f2, with the walk's own best merit at that size:

| target size | required merit | our best at that size | margin |
|---|---|---|---|
| 21 488 | 24.2381 | not hit | — |
| 22 054 | 24.8767 | not hit | — |
| 22 112 | 24.6888 | not hit | — |
| 22 478 | 24.7360 | not hit | — |

The walk's maximum merit is 24.6007 (at size 21 690). So **f2 has already
produced merits above what the nearest target requires — just never at the
required size.** The binding constraint is landing on a specific integer size,
not producing a taller merit.

INFERENCE: the practical lever is therefore the *alignment* between the size
band the walk can reach (`g ≈ m*L`, set by the shift) and the cheap part of
the external frontier `mu(g)`. This replaces the earlier framing ("fatten the
merit tail") for the record-rate objective: a heavier tail only helps if it
lands on beatable sizes.

HYPOTHESIS (needs a GPU A/B): a cover that maximises `p_record` against the
live frontier table would beat `lex-m30` at equal survivor count, because the
objective would be the actual record predicate instead of a generic tail
proxy. Reopen trigger: the dual-3060 same-size cover test currently running
(`shift1017_p130_strong_m30` vs `shift1017_p130_lex_m30`) answers whether
sigma is cover-driven or size-driven; if cover-driven, the cover and the tail
are the same lever and this becomes the main one.

## 4. Negative result: report-threshold tuning is a no-op

The sweep (`--sweep-max`) shows E[records] for a fixed walk time **flat at
0.087 from threshold 8 to threshold 24**, falling only past 25 where the rows
become model extrapolations.

This is **not an observation — it is a tautology of the model** and is
documented here to stop it being rediscovered as a "finding". For a pure
exponential tail with the frontier well above the threshold,
`n(m0) ~ exp(-m0/sigma)` and `p_record(m0) ~ exp(+m0/sigma)`, so the product
is invariant by construction. The sweep exists to make that visible.

The only real coupling between threshold and record rate is the **walk rate**,
which this model does not contain: in the JUMP2 chain a higher threshold makes
the chain walk further per window before it can emit, hence slower (measured
on this host: 3968 win/s at m8 vs 486 win/s in the m20 chain configuration).
INFERENCE: keep `--gap-hunt-min-merit` low. It buys N3 science data, better
sigma estimates, and an unchanged record rate.

## 5. Shift alignment landscape (offline)

`--shift-scan` slides the size band against the fixed frontier. Two columns
have different epistemic status, and the tool prints them separately:

* **needs merit** — model-free (table + L only). This is the alignment fact.
* **gaps/record** — depends on `sigma(L)`, measured only at `L = 528.178`
  (shift507) and `L = 881.683` (shift1017); rows outside that range are
  marked `*` and are HYPOTHESIS.

| shift | L | easiest size | needs merit | gaps/record (σ(L)) | gaps/record (σ pinned 1.3716) |
|---|---|---|---|---|---|
| 900 | 800.6 | 20 950 | 26.0364 | 7.819e6 | 5.961e6 |
| 925 | 817.9 | 21 124 | 25.7611 | 5.056e6 | 4.109e6 |
| 950 | 835.2 | 21 298 | 24.6400 | 3.969e6 | 3.422e6 |
| 975 | 852.6 | 21 298 | 24.6400 | 3.415e6 | 3.115e6 |
| 1000 | 869.9 | **21 488** | **24.2381** | 2.845e6 | 2.743e6 |
| **1017** | 881.7 | **21 488** | **24.2381** | **2.677e6** | **2.677e6** |
| 1025 | 887.2 | 22 112 | 24.6888 | 3.154e6 * | 3.209e6 |
| 1050 | 904.6 | 22 478 | 24.7360 | 2.323e6 * | 2.492e6 |

FACT: the cheapest target in the whole scanned range 200..1050 is
`size 21 488, needs merit 24.2381`, and it is reachable at shift 1000 and
shift 1017. The shipped fleet runs 1017 — i.e. it is already at the alignment
optimum, within ~7 % of the best model value (1050 wins on `sigma(L)` but
pays 0.5 merit of alignment).

FACT: alignment is worth ~4× across 900..1050 (26.04 vs 24.24 required merit,
`exp(1.8/1.37) ≈ 3.6`), so shift choice is not a rounding error — but the
landscape is broad and flat near the optimum, so there is no big win hiding in
a shift change.

## 6. Claim ladder

| Layer | Statement |
|---|---|
| FACT | 988 858 collected gaps contain 0 records; best margins −6.75 (f1) and −3.97 (f2). |
| FACT | f2 reaches merit 24.60 at size 21 690 while the nearest target (21 488) needs 24.2381. |
| FACT | The cheapest reachable target over shifts 200..1050 needs merit 24.2381, at shifts 1000 and 1017. |
| DERIVED | `p_record` is a sum over size-exact targets; the walk is a lottery on landing on a size, not just on merit height. |
| DERIVED | Threshold invariance is a property of the exponential tail, not a measurement. |
| INFERENCE | The model is adequate for ranking configurations; f2 ≈ 8.0× f1 per reported gap (2.677e6 vs 2.141e7). |
| INFERENCE | The fleet shift is within ~7 % of the model optimum for record rate per gap. |
| HYPOTHESIS | A frontier-targeted cover objective (maximise `p_record`, not a generic tail) beats lex-m30 at equal survivor count. |
| HYPOTHESIS | The strong/lex cover tails cross: near-threshold sigma favours lex, the record depth (and the record count) favour strong. ~2 sigma only. |
| SPECULATION | Records/hour might improve from a per-target "watch list" scheduler that reallocates walkers as the frontier moves. Untested; the table snapshot is a moving target. |

## 7. Reopen triggers

* **ANSWERED 2026-09-17** — see §8: the dual-3060 same-size cover test ran, the
  near-threshold sigma IS cover-driven, but it does NOT reach the record depth.
* A new table snapshot changes the frontier → rerun §5; the alignment optimum
  may move (the table is refreshed by `scripts/update_merits.sh`).
* If a per-shift measured gaps/hour is recorded for the fleet, replace the
  per-gap ranking with records/hour; the tool already accepts
  `--gph NAME=VALUE`.
* Write-back: `docs/CLOSED_FINGERPRINTS.md` gained the threshold-tuning entry.

## 8. Fleet data: the cover question, answered (2026-09-17)

The dual-3060 fleet (`gap_hunt_fleet_dual3060.conf`, merit 18) ran the
same-size / different-cover pair at shift1017: file `f1` =
`shift1017_p130_strong_m30.txt`, file `f2` = `shift1017_p130_lex_m30.txt`
(`scripts/gap_hunt_fleet.sh` names outputs `gap_hunt_records_f${ID}.txt` by
conf line order). Both walkers report from merit 18, so `L = 881.7` and the
reachable size band is identical — the only difference is the cover.

### 8.1 The near-threshold tail IS cover-driven

| M0 | sigma A (strong) | sigma B (lex) | sep | nA / nB |
|---|---|---|---|---|
| 18 | 1.2741 ± 0.0098 | 1.3621 ± 0.0104 | **+6.2** | 16831 / 17207 |
| 19 | 1.2351 ± 0.0139 | 1.3408 ± 0.0147 | +5.2 | 7885 / 8322 |
| 20 | 1.2171 ± 0.0205 | 1.3100 ± 0.0207 | +3.2 | 3542 / 4011 |
| 21 | 1.3018 ± 0.0339 | 1.2916 ± 0.0298 | −0.2 | 1476 / 1873 |
| 22 | 1.2544 ± 0.0472 | 1.2575 ± 0.0429 | 0.0 | 705 / 859 |
| 23 | 1.1992 ± 0.0665 | 1.2545 ± 0.0639 | +0.6 | 325 / 385 |
| 24 | 1.3154 ± 0.1172 | 0.9376 ± 0.0649 | −2.8 | 126 / 209 |

FACT: at the report threshold the lex cover has a heavier tail at 6.2 sigma.
INFERENCE: sigma is therefore cover-dependent, not purely size-driven — the
question recorded as pending in the repo memory is answered YES for the band
18–20.

### 8.2 ... but it does NOT reach the record depth

FACT: the separation collapses to ≤0.6 sigma at M0 = 21–23 and inverts to
−2.8 sigma at M0 = 24 (n = 126/209, not significant).

FACT: the observed record counts agree with the deep band, not with the
threshold band: **13 records in strong vs 10 in lex** (16831 / 17207 gaps),
and the single best gaps are 29.389 (strong) vs 29.038 (lex). Both differences
are ~0.6 sigma — i.e. the covers are indistinguishable at the record depth.

DERIVED: the two cover tails **cross**. Near the threshold lex is heavier; at
the record depth strong is at least as heavy. `tail_compare.py` now prints an
explicit TAIL CROSSING warning when the two verdicts differ in sign.

### 8.3 Model check on data that actually contains records

Using `p(threshold) = p(8) * exp((threshold−8)/sigma)` with the shift1017
N3 anchor `p(8) = 3.735e-07` (§2):

| file | sigma near threshold | E[records] | observed |
|---|---|---|---|
| f2 lex | 1.3621 | 9.9 | **10** |
| f1 strong | 1.2741 | 6.4 | **13** |

INFERENCE: for lex the model is exact to 1 %; for strong it under-predicts by
2×, in the same direction as the tail crossing (a cover whose near-threshold
sigma is low but whose deep sigma is high). Poisson P(N≥13 | 6.4) ≈ 2 %, so
the crossing has independent support at the ~2 sigma level — a HYPOTHESIS, not
a result. This is the first validation of the model against a non-zero record
count, and it is much stronger evidence than §2's 0-in-989k test.

### 8.4 What this closes

The hoped-for lever "optimise covers for sigma, not for survivor count"
(recorded 2026-09-03 as the dual-3060 experiment) is **not supported**: a
cover optimised for the near-threshold sigma would not raise the record rate.
Closed in `docs/CLOSED_FINGERPRINTS.md` with its reopen trigger.

### 8.5 Tooling defect found and fixed

`scripts/tail_compare.py` computed its VERDICT at the default `M0 = 10`, which
for this data lies *below* the report threshold (18): the fit then measures
`(threshold − M0) + sigma` and inflates the standard error by the same factor,
turning the real 6.2 sigma into 0.9 sigma ("no significant tail difference").
Fixed: the verdict threshold auto-clamps to the highest minimum merit of the
two files, a NOTE explains the clamp, and a second verdict is printed at the
deepest threshold with n ≥ 100 in both files (plus the crossing warning).
Regression-checked: the N3 size-comparison default pair still reports
`M0=10, +0.0992, 14.6 sigma` exactly as before.

## 9. What this does not claim

* Not a record claim: the tool emits no `FIRST_KNOWN_OCCURRENCE`; it only
  evaluates the same predicate the watcher uses.
* Not an absolute-time predictor: sigma uncertainty alone moves the numbers by
  a factor of ~2.
* Not a cover result: nothing here was walked; no cover was regenerated.
* Not a shift recommendation to move the fleet: the measured 1017 is already
  at the optimum within model error.
