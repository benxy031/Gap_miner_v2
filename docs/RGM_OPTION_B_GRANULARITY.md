# Why RGM "Option B" region scoring cannot prune anything (GPU smart-scan path)

## Summary

Both `rgm_score_regions()` (the always-on, default-shipped scorer) and
`rgm_score_regions_adaptive()` (a self-calibrated variant built and unit
tested during this investigation, **not** wired into `main.c`) are
functionally *no-ops* in the non-CRT GPU smart-scan path in
[`src/main.c`](../src/main.c). This was previously attributed to a vague
"statistical population mismatch." Deeper investigation found two concrete,
fixable/documentable causes — one a real bug (fixed), one a structural
incompatibility (not fixed; requires a redesign out of scope here).

## Cause 1 (fixed): vacuous boundary-override check

Both functions contained a "boundary override" rule intended to auto-keep
(and exclude from calibration) regions whose raw endpoint-to-endpoint span
already exceeds the merit target, on the theory that such a region is
"obviously good enough" without needing interior-spread analysis:

```c
if (hi != (uint64_t)-1 && hi > lo && (hi - lo) >= (uint64_t)target_gap)
    continue;
```

In `main.c`, every GPU-smart-scan region is constructed using **exactly**
this same `target_gap` (`gpu_rth = needed_gap`) as the *promotion*
criterion:

```c
for (size_t i = 0; i + 1 < sp_cnt; i++) {
    if (sampled_primes[i+1] - sampled_primes[i] >= gpu_rth) {
        gap_reg_lo[n_gap_regions] = sampled_primes[i];
        gap_reg_hi[n_gap_regions] = sampled_primes[i+1];
        n_gap_regions++;
    }
}
```

So `(hi - lo) >= target_gap` is true for **every** region by construction —
the boundary check always fired, permanently short-circuiting both scoring
functions before they ever reached the interior-gap analysis or the
self-referential baseline accumulator. This was confirmed empirically: live
runs showed `rgm_self_cal_cnt=0` after 800+ telemetry samples covering
17,000+ regions.

**Fix applied**: the boundary check now only excludes genuinely one-sided /
unbounded regions (the two possible window-edge regions per window, where
`lo == 0` or `hi == UINT64_MAX`), which is what the rule was actually meant
to identify. See `rgm_score_regions()` and `rgm_score_regions_adaptive()` in
[`src/rgm_check.c`](../src/rgm_check.c).

## Cause 2 (not fixed — structural): regions have zero interior degrees of freedom

After fixing Cause 1, live testing (`--threads 4`, `--rgm-self-calibrate`,
30s run, 17,000+ regions observed) still showed `rgm_self_cal_cnt=0`
throughout. Root cause: in `main.c`, a region's `lo`/`hi` are two **adjacent**
entries of the same sparse `sampled_primes` array (phase-1 "every Kth
survivor" subsample):

```c
gap_reg_lo[n] = sampled_primes[i];
gap_reg_hi[n] = sampled_primes[i+1];
```

Both scoring functions look for *other* entries of `sampled_primes` strictly
between `lo` and `hi` to compute an interior max-gap/min-gap ratio, requiring
at least 2 interior gaps (3 points). Since `lo` and `hi` are **immediate
neighbors** in the same sorted array, there is — by construction — never
another sampled point between them. `n_gaps_in_region` is therefore always
0, and the existing `n_gaps_in_region < 2` guard causes every single
non-boundary region to be skipped (left alive, unscored, unaccumulated).

This is not a statistical/calibration issue and not fixable by tuning
`skip_thresh`, `cal_min_samples`, or the boundary rule — it is a hard
mismatch between the granularity Option B was designed for (a region
containing multiple interior sample points to derive a spread statistic
from) and the granularity `main.c` actually uses (one gap = one region, with
no interior members). Making this work would require changing what a
"region" is (e.g. grouping several consecutive sub-threshold sample-gaps
into one wider candidate cluster before promotion, or scoring at the
whole-window level the way the pre-existing `rgm_accumulate_window()` /
mean-gap health check already does) — a materially different design, not a
parameter fix, and out of scope for this investigation.

## Disposition

- `rgm_score_regions()`: left wired in `main.c` (always-on, as before) since
  changing shipped default behavior wasn't the scope of this session — but
  it is now understood, and documented in code comments, to be a harmless
  no-op at this call site.
- `rgm_score_regions_adaptive()` + `rgm_region_baseline_snapshot()`: kept in
  [`src/rgm_check.c`](../src/rgm_check.c) / [`src/rgm_check.h`](../src/rgm_check.h),
  fully unit tested ([`tests/test_rgm_score_regions_adaptive.c`](../tests/test_rgm_score_regions_adaptive.c)),
  but **not** wired into `main.c` / the CLI. Shipping a `--rgm-self-calibrate`
  flag that can never prune anything at this granularity would be
  misleading. The functions remain available as validated building blocks
  for a future redesign that scores at a granularity with real interior
  structure.

## Takeaway for future "smart protocol" work

The earlier conclusion that RGM Option B is "empirically ineffective" was
correct in outcome but imprecise in explanation. The real reason smart-scan
region pruning based on per-region interior gap spread cannot work today is
architectural: individual promoted regions are singleton gaps with no
interior sample points. Any future non-CRT pruning heuristic needs to either
(a) operate on multi-gap groups/windows (real interior structure available),
or (b) use a signal that doesn't require interior structure within a single
promoted region at all (e.g. a purely a-priori model of gap-size
probability given the *sizes* of neighboring already-computed gaps, not
points strictly inside this one).
