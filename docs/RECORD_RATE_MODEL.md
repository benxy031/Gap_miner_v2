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
| E[records], P(observe 0) | 0.071, 0.932 | 0.175, 0.840 |
| gaps per record | 1.071e7 | **1.339e6** |

(values corrected by §10; the pre-correction figures were 0.035/0.965,
0.087/0.916, 2.141e7 and 2.677e6)

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
| INFERENCE | The model is adequate for ranking configurations **only where the predicted ratio exceeds the +-5 % sigma band**; the N3 pair (8.0x vs bands 3.9-4.9x) qualifies, the fleet strong/lex pair (1.56x vs 1.9x) does not. |
| INFERENCE | Calibrated on the lex cover (1.10x in-sample, 1.27x out-of-sample) and pessimistic on the strong cover (2.3x / 4.1x under) at threshold 18. |
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

`scripts/record_rate_model.py data/gap_hunt_records_f1.txt \
  data/gap_hunt_records_f2.txt --m0 18 --targets 10` on the live fleet files:

| | f1 strong | f2 lex |
|---|---|---|
| reported gaps | 16866 | 17240 |
| observed records | **13** | **10** |
| sigma at M0=18 | 1.2744 ± 0.0098 | 1.3627 ± 0.0104 |
| predicted E[records] | 5.69 (2.3x under) | 9.06 (1.10x under) |
| predicted gaps/record | 2.96e3 | 1.90e3 |
| **observed gaps/record** | **1.30e3** | **1.72e3** |
| holdout: E vs observed | 1.94 vs 8 (4.1x under) | 5.52 vs 7 (1.27x under) |
| holdout `d_best` CDF position | 0.944 | **0.489** |

INFERENCE: the model is calibrated on the lex cover (1.10x in-sample, 1.27x
out-of-sample, holdout closest-approach sitting on the model median) and
under-predicts the strong cover by 2.3x (4.1x out-of-sample). The residual is
cover-specific, which is the same phenomenon as the tail crossing in §8.2:
a single exponential fitted at the report threshold cannot describe both
covers at the record depth. The under-prediction is the *safe* direction for
allocation, but it is a real bias and it must be quoted with the numbers.

**The ranking is NOT resolved for this pair.** The predicted ratio is
1.56x (2.96e3 / 1.90e3) while a ±5 % sigma error alone moves each file by
1.90–1.98x — so the model ranks the files with less confidence than its own
uncertainty, and the observed counts rank them the *other* way (13 vs 10, a
0.6 sigma difference, i.e. indistinguishable). The tool now prints the band
factor per file and emits `WARNING: ... RANKING NOT RESOLVED` whenever the
predicted ratio is smaller than the band. Ranking is only meaningful where
that warning does not fire — e.g. the N3 pair (§2) at 8.0x against bands of
3.9–4.9x.

### 8.4 What this closes

The hoped-for lever "optimise covers for sigma, not for survivor count"
(recorded 2026-09-03 as the dual-3060 experiment) is **not supported**: a
cover optimised for the near-threshold sigma would not raise the record rate.
Closed in `docs/CLOSED_FINGERPRINTS.md` with its reopen trigger.

### 8.5 Tooling defects found and fixed

Two defects surfaced while interpreting this run.

**(a) `tail_compare.py` verdict below the report threshold.** The VERDICT was
computed at the default `M0 = 10`, which for this data lies *below* the report
threshold (18): the fit then measures `(threshold − M0) + sigma` and inflates
the standard error by the same factor, turning the real 6.2 sigma into
0.9 sigma ("no significant tail difference"). Fixed: the verdict threshold
auto-clamps to the highest minimum merit of the two files, a NOTE explains the
clamp, a second verdict is printed at the deepest threshold with n ≥ 100 in
both files, and a TAIL CROSSING warning fires when the two verdicts differ in
sign. Regression-checked: the N3 default pair still reports
`M0=10, +0.0992, 14.6 sigma` exactly as before.

**(b) The frontier-slope sigma rescale (`sigma_eff`) was wrong twice.**
(i) The form was inverted: for `mu(g) = a − b*g` the margin grows with merit
at rate `(1 + bL) > 1`, so `sigma_D = sigma * (1 − L*dmu/dg)`, not
`sigma / (1 − L*dmu/dg)`. (ii) Even corrected it double-counts, because
`p_record` already evaluates `mu(g)` at every size. The fleet data settles it:
no single rescale fits both covers — strong moves 5.7 → 4.9 (divide) or
→ 12.8 (multiply) against an observed 13, lex moves 9.1 → 6.3 or → 19.3
against an observed 10. The slope is now reported as a **DIAG** line only and
is never applied to sigma. Registered in `docs/CLOSED_FINGERPRINTS.md`.

## 9. Tail shape: is the deep tail exponential? (§9.1-9.6 = small-sample pass, superseded by the powered result in §9.7)

Tool: `scripts/tail_shape.py` (stdlib-only; `--selftest` verifies every
estimator on synthetic samples before use: exp 0.03 % off, stretched
beta 1.3485 vs 1.3500, GPD xi 0.2456 vs 0.2500, and GPD on exponential data
returns xi = +0.0006, i.e. no false heavy tail).

### 9.1 The measured signature (shift507 lex walk, 755 202 gaps)

Mean excess E[m - M0 | m >= M0] must be CONSTANT for an exponential:

| M0 | 8 | 10 | 12 | 14 | 16 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|
| measured sigma | 1.261 | 1.292 | 1.283 | 1.293 | 1.186 | 0.980 | 0.858 | 0.770 |
| n | 755202 | 153383 | 32622 | 6864 | 1509 | 295 | 119 | 42 |

FACT: the mean excess falls by ~35 % between M0 = 12 and M0 = 19. The
shift1017 lex walk shows the opposite-or-flat behaviour (1.372 at 8, 1.300 at
16, 1.489 at 18 with n=142, 1.295 at 20 with n=39). So the deep-tail
behaviour is a property of (size, cover), exactly like sigma.

### 9.2 Family fits (held out, never in-sample)

Fitted above u_fit, scored on the exceedances above u_test:

| fit above | family | params | held-out LL | sum abs(z) |
|---|---|---|---|---|
| u=12, n=32622 | exp | sigma=1.2840 | -10739.94 | 5.04 |
| u=12 | stretched | beta=1.0062 lam=1.2873 | -10737.33 | 4.64 |
| u=12 | gpd | xi=-0.0001 scale=1.2841 | -10739.79 | 5.03 |
| u=16, n=1509 | exp | sigma=1.1868 | -328.44 | 2.28 |
| u=16 | stretched | beta=1.0519 lam=1.2104 | -323.77 | 1.52 |
| u=16 | **gpd** | **xi=-0.0843 scale=1.2869** | **-320.67** | **0.73** |

FACT: fitted at u=12 BOTH two-parameter families collapse onto the
exponential (beta=1.006, xi=-0.0001). The deep behaviour only appears when
the fit moves to u=16, where the GPD wins on both criteria (Delta-LL = 7.8
vs exp, i.e. 2Delta-LL = 15.5 for one extra parameter).
FACT: the exponential over-predicts the held-out deep count: at M0=20 it
predicts 64.2 gaps where 42 were observed (z = -2.8).

### 9.3 The replacement is NOT determined

FACT: xi is stably negative for the shift507 walk but its magnitude drifts
with the fit threshold: -0.0043 (u=13), -0.0562 (14), -0.0701 (15),
-0.0843 (16), -0.1269 (17), -0.0431 (18). The implied finite upper endpoint
(u + scale/|xi|) therefore spans roughly **merit 26 to 45** — and the record
rate is exponentially sensitive to it (the endpoint is a hard cutoff).
FACT: for the shift1017 lex walk the same scan gives xi = -0.0151, -0.0215,
-0.0071, **+0.0828, +0.0972** — the sign flips to positive (heavy) at u >= 16.

INFERENCE: the exponential merit tail is **rejected** at the record depth
(both by the mean-excess signature and by the held-out Poisson z), but no
single replacement family is established, and the shape cannot be transferred
between covers. Consequences:

* Absolute record-rate projections must quote the **family spread**, not a
  sigma perturbation. On the shift507 lex file the sigma +-5 % band was 4.86x
  while the exp/stretched/gpd spread is ~5 orders of magnitude
  (9.9e7 / 4.3e8 / 1.1e13 gaps per record from u_rec=17.5), because the GPD
  endpoint (~27.5) sits just above the cheapest record target (~26.4).
* The pairwise **ranking** must be judged against that same spread; the
  `band` column now reports whichever spread is active (`band_kind`).
* `record_rate_model.py --tail exp|stretched|gpd --u-rec auto|m0|X` selects the
  extrapolating family; the default (`exp`, `u_rec=m0`) reproduces the earlier
  numbers exactly (regression-checked: 2.677e6 and 2.141e7 gaps/record).

### 9.4 Implementation trap found while wiring this up

GPD with `xi < 0` has a finite upper endpoint, and beyond it the standard
formula `(1 + xi*x/B) ** (-1/xi)` raises a NEGATIVE base to a fractional
exponent, which in Python returns a **complex number** instead of failing.
Every downstream sum then silently becomes complex (`TypeError` only much
later, if at all). Both `survival` and `logpdf` now return 0 / -inf past the
endpoint, and `--selftest` covers the three families.

### 9.5 Honest bounds on this section

* The u=16 fit rests on 1509 exceedances and the M0=20 check on 42; the
  endpoint estimate is not resolved, only the *rejection* of the pure
  exponential at depth is.
* Two-parameter families were compared at several thresholds, so the reported
  p-value (2Delta-LL = 15.5, ~8e-5 before any multiple-comparison penalty)
  should be treated as suggestive, not definitive.
* Nothing here was walked: no GPU time, no cover regenerated, no record made.
* The shape is measured on the N3 (threshold 8) and fleet-snapshot
  (threshold 18) files; the live fleet files at 16866/17240 gaps are 10x
  larger and will settle part of this — rerun `tail_shape.py` on them with
  `--u-fit 18 --u-test 21,23,25`.

### 9.6 Pooling rules (and why the local files cannot settle this)

`tail_shape.py --pool` merges files into one fit and enforces two conditions,
both of which caught a real error in the first attempt:

1. **Same `L = ln(anchor)` within 0.1 %.** Different `L` means a different
   size, so the merit axes are not comparable at all — hard refusal.
   Caught: `data/gap_hunt_records_f2.txt` is shift450 (L = 488.669), NOT the
   shift507 file it looks like next to `f1` (L = 528.178, spread 7.8 %), so
   pooling it with `f1` was refused.
2. **`u_fit` >= the highest member report threshold.** A walker only reports
   gaps above its own `--gap-hunt-min-merit`, so between a low member threshold
   and a high one the high-threshold walker contributes nothing and the pooled
   density in that band is under-sampled. The first pooled run used u_fit=19
   across members with thresholds 8 / 19 / 21 and produced a spurious
   sigma(19) = 1.579 — a mixture artifact, since 107 of the 327 gaps above 19
   came from the threshold-21 member — together with a +3.4 sigma excess at
   M0=21. With the rule enforced the fit correctly starts at 21.0118.

The **anchor-digit prefix** printed by the tool is informational only: offsets
are tiny next to the base, so any cover at the same shift shares a long
prefix. It does not prove the same cover; the fleet conf does.

RESULT (FACT): the valid pooled sample holds only n_fit = 123 above 21 with a
single usable held-out point (n = 16). The three families differ by
Delta-LL = 0.16 and all |z| < 1 — **no discrimination**. Locally the deep tail
cannot be settled; the N3 u=16 fit (n = 1509, Delta-LL = 7.8) stays the only
discriminating measurement, and the fleet's LIVE files (16 866 / 17 240 gaps at
threshold 18, ~10-50x the local snapshots) are the ones that can settle it —
**done in §9.7, which supersedes §9.1-9.6.**
Command for the fleet box (pooling is not needed there — each file is already
big):

```
scripts/tail_shape.py data/gap_hunt_records_f1.txt \
    data/gap_hunt_records_f2.txt --u-fit 18 --u-test 21,23,25
```

### 9.7 Powered fleet result (2026-09-17): the ceiling was a small-sample artifact

Command on the live fleet files (16 893 / 17 275 gaps at threshold 18):

```
scripts/tail_shape.py data/gap_hunt_records_f1.txt \
    data/gap_hunt_records_f2.txt --u-fit 18 --u-test 21,23,25
```

| file | measured mean excess @21 / 23 / 25 | best held-out LL | xi(gpd) |
|---|---|---|---|
| f1 strong | 1.2997 (n=1482) / 1.1992 (325) / 1.3130 (58) | **exp** -2329.20 | -0.0179 |
| f2 lex | 1.2933 (1888) / 1.2493 (390) / 1.2123 (59) | **gpd** -2916.46 | -0.0275 |

FACT (f1 strong): the exponential has the best held-out log-likelihood;
Delta-LL(exp - gpd) = 1.78, i.e. 2Delta-LL = 3.6 for one extra parameter ->
**not significant**. The mean excess is FLAT within errors (the 23-point dips
1.4 sigma and reverses at 25). The exponential is ADEQUATE here.
FACT (f2 lex): the GPD wins by Delta-LL = 8.77 (2Delta-LL = 17.5, p ~ 3e-5) and
the exponential over-predicts the held-out deep counts - at M0=23, 441.5
predicted vs 390 observed (z = -2.45); at M0=25, 101.9 vs 59 (z = **-4.25**).
The mean excess falls 1.293 -> 1.212. The exponential is REJECTED here.
FACT: both shape parameters are MILD (-0.018, -0.0275) and the implied
endpoints (~69 for lex, effectively unbounded for strong) sit far above the
operating range (merit <= 30).

DERIVED - this refutes the headline of §9.3:

* The "finite merit ceiling at 26-45" came from a fit at u=16 on n=1509. With
  10x more data at the record-relevant threshold the ceiling moves to ~69 and
  the shape parameter shrinks by ~3x. That claim is downgraded from
  HYPOTHESIS-with-numbers to **refuted as stated**; the u=16 endpoint was a
  small-sample artifact.
* The exponential is therefore NOT the problem for the strong cover - its tail
  IS exponential. The strong-cover record surplus (13 observed vs 5.69
  predicted, §8.3) is **not a tail-shape effect**; the remaining candidate is
  the record probability GIVEN a deep gap, i.e. size-exact landing and/or the
  provenance of the frontier table (§3).
* For the lex cover the better-fitting family predicts ~2x FEWER records, i.e.
  it makes the model *worse* (exp E = 9.06 vs observed 10). The exponential's
  agreement with the lex record count is therefore not evidence FOR the
  exponential - it may be a cancellation.

BONUS - model-free confirmation of the tail crossing (§8.2):
`tail_compare.py` measured sigma at the report threshold: strong 1.2741 < lex
1.3621. The mean-excess curves here say the opposite at depth: strong is flat
(1.30 / 1.20 / 1.31) while lex falls (1.29 / 1.25 / 1.21). Two independent,
model-free statistics on the same files agree, so the crossing no longer rests
on a single sigma comparison.

The tool's own NOTE was correct in advance: for the strong cover it printed
"xi is indistinguishable from 0 at this threshold ... refit higher". Refitting
higher costs sample size faster than it buys shape resolution (n=325 at M0=23,
n=58 at M0=25), so the honest statement is: **as far as 17k gaps at threshold
18 can see, the strong cover's tail is exponential and the lex cover's is very
slightly lighter.**

## 10. Correction (2026-09-17): the size bin must be 2/L wide

**The bug.** Gaps between odd primes are EVEN, so the walk can only produce
sizes on the even lattice. A gap of size `g` therefore occupies merit in
`[(g-1)/L, (g+1)/L)` — width `2/L` — and the next table entry `g+2` starts
exactly at `(g+1)/L`, so those bins tile the merit axis. The code used a
half-width of 0.5, leaving **half the merit axis uncovered**, which made
`p_record` exactly 2x too small.

**How it was found, and the trap on the way.** A synthetic self-test (data
generated exactly per the model's own assumptions: exponential merit tail,
size = round(m*L), a known frontier) reported the model 10x low — which is what
made me look. The first synthetic table was *derived from the sample itself*,
so it was sparse and the arithmetic of that test was misleading in its own
right. The decisive evidence is a direct **Monte-Carlo on the REAL table**:
draw merits from the tail, map to the even-gap lattice, look up the table.

| half-width | bin-sum vs Monte-Carlo (strong / lex) |
|---|---|
| 0.5 (old) | 0.507 / 0.510 |
| 1.0 (fixed) | **1.014 / 1.019** |

Supporting fact: the real table is **100 % dense** in the record-relevant band
(2501 entries for 2500 possible even sizes in [21000, 26000], 5501 for 5500 in
[21000, 32000]) and contains **no odd gap above 1000**, so the lattice argument
applies cleanly.

**Impact.** `p_record` x2, `E[records]` x2, gaps-per-record /2. Crucially the
**band factors and every inter-file ratio are unchanged** (both scale by 2), so
the ranking verdicts in §2, §5 and §8 still stand as written.
`--bin-half 0.5` reproduces the pre-correction numbers exactly
(f1 2.141e7, f2 2.677e6), which is the proof that the change is isolated.

| quantity | f1 (shift507 lex) | f2 (shift1017 lex) |
|---|---|---|
| E[records], corrected | 0.071 (P(obs 0)=0.932) | 0.175 (P(obs 0)=0.840) |
| gaps per record, corrected | 1.071e7 | 1.339e6 |
| superseded values | 0.035 / 2.141e7 | 0.087 / 2.677e6 |

The `--mu-shift` grid and calibration solve added for the frontier audit are
kept as diagnostics, but they are **no longer needed to explain the
strong-cover surplus**: that surplus was this bug.

## 11. The fleet arithmetic closes

Correcting the bin (x2) and using the independently measured tail shape per
cover:

| | observed | old model | corrected bin | + measured tail |
|---|---|---|---|---|
| f1 strong, in-sample | 13 | 5.69 (2.3x low) | 11.38 (**1.14x low**) | 11.38 (exponential tail verified §9.7) |
| f1 strong, holdout | 8 | 1.94 (4.1x low) | 3.89 (2.1x low) | 3.89 |
| f2 lex, in-sample | 10 | 9.06 (1.10x low) | 18.12 (1.81x high) | **~10.6 ✓** |
| f2 lex, holdout | 7 | 5.52 | 11.04 (1.58x high) | **~6.5 ✓** |

The lex rows use the measured over-prediction of the exponential at depth
(§9.7: 101.9 predicted vs 59 observed at M0=25 = 1.73x) as the correction
factor. Both covers therefore reconcile to within ~15 % once each correction is
applied on its own evidence. **The 2.3x/4.1x discrepancy that motivated the
whole frontier audit and the tail-shape investigation was a factor-2 bin bug in
my own code**, not the tail and not the table.

Consequences worth stating plainly:

* Numbers reported earlier today for `E[records]` and gaps-per-record
  (5.69 / 9.06 / 2.96e3 / 1.90e3 / 2.677e6 / 2.141e7) are **superseded** by a
  factor 2. Use `--bin-half 0.5` only to reproduce them deliberately.
* The *qualitative* results survive untouched: the ranking and band rule, the
  threshold-invariance tautology, the size-exact frontier finding, the tail
  crossing, the rejection of `sigma_eff`, and the tail-shape verdicts.
* Any future probability kernel of the form "a lattice quantity lands in a
  bin" must be validated against a Monte-Carlo on the real table BEFORE it is
  used to score coverings.

## 12. What this does not claim
* Not a record claim: the tool emits no `FIRST_KNOWN_OCCURRENCE`; it only
  evaluates the same predicate the watcher uses.
* Not an absolute-time predictor: sigma uncertainty alone moves the numbers by
  a factor of ~2.
* Not a cover result: nothing here was walked; no cover was regenerated.
* Not a shift recommendation to move the fleet: the measured 1017 is already
  at the optimum within model error.
