#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""record_rate_model.py - record-rate model validation for GAP_HUNT data.

ZERO GPU COST.  Everything below is derived from already-collected hunt gap
files (`<gap> <merit> <startprime>` per line, the GAP_HUNT output format) plus
the known-record table (data/prime_gap_merits.txt, same table the watcher and
record_log.c use).

What it answers
---------------
1. How many of the collected gaps would have been records?  (exact predicate)
2. Does the Cramer/exponential-tail model predict that number?  (model check)
3. How close is the walk to a record, in sigma units, and what does the model
   say the closest-approach distribution should be?  (discriminating test)
4. Which configuration buys records sooner, per gap?  (scheduler input)

Definitions
-----------
Record criterion (identical to scripts/watch_gap_hunt_records.py and
new_src/record_log.c):

    gap size g present in the table AND merit > best_known_merit[g]

The merit of a gap of size g at a walk with L = ln(start) is m = g / L.  For a
walk L is nearly constant (start = b0 + k*P + offset), so "size exactly g" and
"merit in the bin [(g-0.5)/L, (g+0.5)/L)" are the same event.  Hence

    p_record(L, sigma, m0) = SUM over table entries g of
        S(max(lo, mu(g), m0)) - S(max(hi, m0)),
    with lo = (g-0.5)/L, hi = (g+0.5)/L, S(x) = exp(-(x-m0)/sigma).

The clamp at m0 is essential: only gaps with merit >= m0 are ever reported, so
bins entirely below m0 contribute exactly zero (a size-2 gap cannot be
reported, no matter how easy its record is).

The margin (how far a gap is from beating its own size's record) is

    d(g, m) = m - best_known_merit[g]        (d > 0  =>  new record)

and the walk's closest approach d_min = min over all reported gaps is the
observable that the model predicts via

    P(d_min > x) = (1 - p_record(x))^n,      n = number of reported gaps.

Note on merit width: very large integers (10^383) overflow double, so ln(start)
is computed from the bit length (the Python analogue of the mpz_get_d_2exp fix
in gap_hunt.c); float(start) would silently be +inf.

Claim layers (see docs/RECORD_RATE_MODEL.md):
    FACT       - measured in this run
    INFERENCE  - follows from the fit under stated assumptions
    HYPOTHESIS - needs a discriminating test

Usage
-----
    scripts/record_rate_model.py gap_hunt_records_f1.txt gap_hunt_records_f2.txt
    scripts/record_rate_model.py --m0 8 --holdout 0.5 --gph f1=50000 f2=40000 \\
        gap_hunt_records_f1.txt gap_hunt_records_f2.txt
"""

import argparse
import bisect
import math
import os
import sys

LN2 = math.log(2.0)
DEFAULT_TABLE = "data/prime_gap_merits.txt"

# Measured tail scales (docs: gap_hunt_records_f1/f2, scripts/analyze_n3.py).
# The hunt anchor is 2^(255+shift), so L = (255+shift)*ln2 reproduces the
# measured L exactly (shift507 -> 528.18, shift1017 -> 881.68).
# Half-width, in gap-size units, of the merit bin that one table entry stands
# for.  Gaps between odd primes are EVEN, so the walk can only produce sizes on
# the even lattice: a gap of size g occupies merit in [(g-1)/L, (g+1)/L) and
# the neighbouring table entry g+2 starts exactly at (g+1)/L.  Half-width 1.0
# therefore tiles the merit axis without gaps; half-width 0.5 leaves half of it
# uncovered and under-counts p_record by exactly 2x.
# VERIFIED against a direct Monte-Carlo on the real table (2026-09-17):
#   half-width 0.5 -> bin/MC = 0.505 (strong), 0.497 (lex)
#   half-width 1.0 -> bin/MC = 1.00
BIN_HALF = 1.0

# Tail-shape machinery lives in tail_shape.py (same directory).  It is only
# needed for --tail / --u-rec; without it the tool still runs exponential-only.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import tail_shape as ts
    HAVE_TAIL_SHAPE = True
except ImportError:                      # pragma: no cover - fleet boxes
    ts = None
    HAVE_TAIL_SHAPE = False


class _PureExp:
    """Exponential survival from the report threshold: S(x) = exp(-(x-m0)/s)."""

    def __init__(self, sigma, m0):
        self.sigma = sigma
        self.u = m0

    def s(self, x):
        return math.exp(-max(0.0, x - self.u) / self.sigma)


class Surv:
    """HYBRID survival: empirical below u_rec, fitted family above it.

    This is the honest arrangement: where the walk has data we use the data,
    and the exp/stretched/GPD families then differ ONLY in the extrapolation
    beyond u_rec - which is exactly the question being asked.  With u_rec = m0
    the object reproduces the exponential-from-the-threshold behaviour used
    before, so --u-rec is a strictly additive knob.
    """

    def __init__(self, kind, params, u_rec, merits_sorted, sigma_equiv):
        self.kind, self.params, self.u = kind, params, u_rec
        self.ms = sorted(merits_sorted)
        self.n = len(self.ms)
        self.sigma = sigma_equiv
        self.s_u = (self.n - bisect.bisect_right(self.ms, u_rec)) / self.n

    def s(self, x):
        if x <= self.u:
            return (self.n - bisect.bisect_right(self.ms, x)) / self.n
        return self.s_u * ts.survival(self.kind, self.params, x - self.u)

    def label(self):
        return f"{self.kind} above u_rec={self.u:g}"


def auto_u_rec(merits, min_n=300.0):
    """Deepest 0.5-step threshold that still has >= min_n exceedances."""
    ms = sorted(merits)
    n = len(ms)
    u = ms[0]
    best = u
    while u <= ms[-1]:
        if (n - bisect.bisect_right(ms, u)) >= min_n:
            best = u
        u += 0.5
    return best


def build_family_surv(merits, u_rec, kind, min_fit=60):
    """Fit `kind` above u_rec; return a Surv or None if the data is too thin."""
    if not HAVE_TAIL_SHAPE:
        return None
    xs = [m - u_rec for m in merits if m > u_rec]
    if len(xs) < min_fit:
        return None
    params = ts.fit(kind, xs)
    sig = ts.mean_excess(kind, params, 0.0)
    if not (sig and sig == sig and sig > 1e-6):      # NaN / degenerate guard
        sig = sum(xs) / len(xs)
    return Surv(kind, params, u_rec, merits, sig), params


# --------------------------------------------------------------------------
# exact ln() for arbitrarily large positive integers
# --------------------------------------------------------------------------
def ln_int(n):
    """Natural log of a big Python int without overflow (mpz_get_d_2exp port).

    float(n) raises OverflowError (or returns inf via float('1e400') parsing)
    for n > 2^1024, which silently turns merit into 0 - the exact bug fixed in
    gap_hunt.c on 2026-09-02.  Split n = top53 * 2^(b-53) instead.
    """
    if n <= 0:
        return float("-inf")
    b = n.bit_length()
    if b <= 53:
        return math.log(n)
    top = n >> (b - 53)          # 53-bit mantissa, exact
    return math.log(top) + (b - 53) * LN2


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def load_table(path):
    """{gap: best_known_merit}.  Semantics match watch_gap_hunt_records.py."""
    table = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                table[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return table


def load_gaps(path):
    """[(gap, merit, ln_start)] for a `<gap> <merit> <startprime>` file."""
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                gap = int(parts[0])
                merit = float(parts[1])
            except ValueError:
                continue
            if gap <= 0 or merit <= 0:
                continue
            ln_start = float("nan")
            if len(parts) == 3:
                try:
                    ln_start = ln_int(int(parts[2]))
                except ValueError:
                    pass
            rows.append((gap, merit, ln_start))
    return rows


# --------------------------------------------------------------------------
# tail fit
# --------------------------------------------------------------------------
def fit_sigma(merits, m0):
    """Truncated-exponential MLE: sigma = mean(m - m0 | m >= m0).

    Returns (sigma, standard_error, n_tail).  The MLE is unbiased-corrected by
    n/(n-1); with n > 1000 the correction is irrelevant but harmless.
    """
    tail = [m for m in merits if m >= m0]
    n = len(tail)
    if n < 2:
        return float("nan"), float("nan"), n
    mean = sum(tail) / n
    sigma = mean - m0
    if n > 2:
        sigma *= n / (n - 1.0)
    return sigma, sigma / math.sqrt(n), n


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
def reachable_entries(table, L, m0, sigma, margin, span_sigmas=26.0):
    """Table entries whose bin lower edge can still carry tail mass.

    exp(-26) = 5e-12 is far below anything measurable, so pruning here is
    lossless for the reported precision (and turns a 122k-entry sum into a
    few-hundred-entry sum that can go inside a bisection).
    """
    inv = 1.0 / L
    sg = sigma.sigma if isinstance(sigma, (_PureExp, Surv)) else sigma
    limit = m0 + span_sigmas * sg + margin
    out = []
    for g, mu in table.items():
        lo = (g - BIN_HALF) * inv
        if lo < limit:
            out.append((lo, (g + BIN_HALF) * inv, mu))
    return out


def p_record(entries, L, sigma, m0, margin=0.0, mu_shift=0.0):
    """P(one reported gap is a record) - or beats by `margin` merit units.

    `sigma` may be a float (exponential from m0) or a Surv (hybrid empirical +
    fitted family), which is what `--tail`/`--u-rec` select.
    `mu_shift` moves EVERY table requirement by that many merit units, which is
    how the frontier-table uncertainty is probed (negative = easier frontier).
    """
    S = sigma if isinstance(sigma, (_PureExp, Surv)) else _PureExp(sigma, m0)
    total = 0.0
    for lo, hi, mu in entries:
        a = max(lo, mu + mu_shift + margin, m0)
        b = max(hi, m0)
        if a < b:
            total += S.s(a) - S.s(b)
    return total


def calibrate_mu_shift(entries, L, sigma, m0, n, n_obs):
    """Uniform frontier shift that would make E[records] match the observation.

    Returns (shift, reachable).  Everything else in the model is already
    verified (exponential tail, constant L, size = round(m*L)), so this is the
    single free input: if the observed record count needs a large negative
    shift, the frontier TABLE disagrees with the one used here.
    """
    def gap(x):
        return n * p_record(entries, L, sigma, m0, 0.0, x) - n_obs

    lo, hi = -8.0, 8.0
    f_lo, f_hi = gap(lo), gap(hi)
    if f_lo < 0.0 or f_hi > 0.0:
        return float("nan"), False
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), True


def dbest_quantile(entries, L, sigma, m0, n, q):
    """Quantile x of d_best (the walk's best margin) with P(d_best <= x) = q.

    d_best = max over gaps of (merit - best_known_merit[size]); a gap beats
    its own size's record by more than x with probability p_record(margin=x),
    so P(d_best <= x) = (1 - p_record(x))^n (independent gaps, justified by
    the Poisson-like walk).  p_record is monotone decreasing in x, so invert
    p_record(x) = 1 - q^(1/n) by bisection.
    """
    if n <= 0:
        return float("nan")
    target = 1.0 - q ** (1.0 / n)
    lo_x, hi_x = -25.0, 15.0
    if p_record(entries, L, sigma, m0, lo_x) < target:
        return lo_x
    if p_record(entries, L, sigma, m0, hi_x) >= target:
        return hi_x
    for _ in range(80):
        mid = 0.5 * (lo_x + hi_x)
        if p_record(entries, L, sigma, m0, mid) >= target:
            lo_x = mid
        else:
            hi_x = mid
    return 0.5 * (lo_x + hi_x)


def frontier_slope(table, g_lo, g_hi):
    """Least-squares slope d(mu)/dg over table entries in [g_lo, g_hi]."""
    pts = [(g, mu) for g, mu in table.items() if g_lo <= g <= g_hi]
    n = len(pts)
    if n < 3:
        return float("nan"), n
    sg = sum(p[0] for p in pts)
    sm = sum(p[1] for p in pts)
    sgg = sum(p[0] * p[0] for p in pts)
    sgm = sum(p[0] * p[1] for p in pts)
    den = n * sgg - sg * sg
    if den == 0:
        return float("nan"), n
    return (n * sgm - sg * sm) / den, n


def easiest_target(entries, L, sigma, m0):
    """Table entry contributing most to p_record = the nearest record.

    Returns (gap, required_merit, bin_lower_edge, contribution).
    """
    S = sigma if isinstance(sigma, (_PureExp, Surv)) else _PureExp(sigma, m0)
    best = (0, float("nan"), float("nan"), 0.0)
    for lo, hi, mu in entries:
        a = max(lo, mu, m0)
        b = max(hi, m0)
        if a < b:
            c = S.s(a) - S.s(b)
            if c > best[3]:
                best = (int(round(0.5 * (lo + hi) * L)), mu, lo, c)
    return best


# --------------------------------------------------------------------------
# per-file report
# --------------------------------------------------------------------------
def analyse(path, rows, table, m0, args):
    res = {"path": path, "name": os.path.basename(path)}
    merits = [r[1] for r in rows]
    gaps = [r[0] for r in rows]
    ln_starts = [r[2] for r in rows if not math.isnan(r[2])]

    res["n"] = len(rows)
    if not rows:
        return res

    # --- configuration constants -----------------------------------------
    L = sorted(ln_starts)[len(ln_starts) // 2] if ln_starts else float("nan")
    res["L"] = L
    res["L_spread_pct"] = (
        100.0 * (max(ln_starts) - min(ln_starts)) / L if ln_starts else float("nan")
    )
    res["min_gap"] = min(gaps)
    res["max_gap"] = max(gaps)
    res["max_merit"] = max(merits)

    # --- observed record count and closest approach -----------------------
    margins = []
    unknown = 0
    for g, m, _ in rows:
        mu = table.get(g)
        if mu is None:
            unknown += 1
            continue
        margins.append(m - mu)
    res["in_table"] = len(margins)
    res["unknown"] = unknown
    res["n_records"] = sum(1 for d in margins if d > 0.0)
    # d_best = the walk's closest approach to a record = LARGEST margin
    res["d_best"] = max(margins) if margins else float("nan")

    # --- tail fit ---------------------------------------------------------
    sigma, se, n_tail = fit_sigma(merits, m0)
    res["sigma"], res["sigma_se"], res["n_tail"] = sigma, se, n_tail

    # --- tail-shape families (extrapolation above u_rec) ------------------
    # Empirical survival below u_rec, fitted family above it, so exp /
    # stretched / gpd differ ONLY in the extrapolation.  u_rec = m0 reproduces
    # the pure-exponential-from-the-threshold behaviour exactly.
    u_rec = m0
    if HAVE_TAIL_SHAPE:
        if args.u_rec == "auto":
            u_rec = auto_u_rec(merits, 300.0)
        elif args.u_rec != "m0":
            u_rec = float(args.u_rec)
    res["u_rec"] = u_rec
    fams = {}
    n_exc = sum(1 for m in merits if m > u_rec)
    for fam in (ts.FAMILIES if HAVE_TAIL_SHAPE else ()):
        got = build_family_surv(merits, u_rec, fam)
        if got is None:
            continue
        surv, params = got
        p_f = p_record(reachable_entries(table, L, u_rec, surv, 0.0),
                       L, surv, u_rec, 0.0)
        fams[fam] = {"params": params, "p": p_f,
                     "gpr": (1.0 / p_f) if p_f > 0 else float("inf"),
                     "E": p_f * res["n"]}
        if fam == "gpd" and params.get("xi", 0.0) < -0.02:
            fams[fam]["endpoint"] = u_rec + params["beta_scale"] / abs(params["xi"])
    res["families"] = fams
    res["n_exceedances"] = n_exc
    res["families"] = fams
    sel = None
    if HAVE_TAIL_SHAPE and args.tail != "exp" and args.tail in fams:
        sel = build_family_surv(merits, u_rec, args.tail)
    t_obj = sel[0] if sel else sigma
    t_m0 = u_rec if sel else m0
    res["tail_used"] = args.tail if sel else "exp"
    res["u_rec_used"] = t_m0

    # --- split model: fit on the first part, predict the second -----------
    if args.holdout and args.holdout > 0.0 and ln_starts:
        order = sorted(range(len(rows)), key=lambda i: rows[i][2])
        cut = int(len(order) * (1.0 - args.holdout))
        fit_idx, test_idx = order[:cut], order[cut:]
        sig_a, _, n_a = fit_sigma([rows[i][1] for i in fit_idx], m0)
        res["ho_sigma"] = sig_a
        res["ho_n_fit"] = n_a
        res["ho_n_test"] = len(test_idx)
        ent = reachable_entries(table, L, m0, sig_a, 0.0)
        p = p_record(ent, L, sig_a, m0)
        res["ho_p"] = p
        res["ho_expected"] = p * len(test_idx)
        res["ho_observed"] = sum(
            1
            for i in test_idx
            if table.get(rows[i][0]) is not None
            and rows[i][1] > table[rows[i][0]]
        )
        # Poisson consistency of "observed 0" against the prediction
        res["ho_p_zero"] = (
            math.exp(-res["ho_expected"]) if res["ho_expected"] < 700 else 0.0
        )
        # out-of-sample closest-approach test (sharper than the 0-vs-E count)
        ho_margins = [
            rows[i][1] - table[rows[i][0]]
            for i in test_idx
            if table.get(rows[i][0]) is not None
        ]
        if ho_margins:
            res["ho_d_best"] = max(ho_margins)
            res["ho_n_margins"] = len(ho_margins)
            res["ho_d_best_cdf"] = (
                1.0 - p_record(ent, L, sig_a, m0, res["ho_d_best"])
            ) ** len(ho_margins)
            res["ho_d_best_q50"] = dbest_quantile(
                ent, L, sig_a, m0, len(ho_margins), 0.5
            )

    # --- full-sample prediction + closest-approach quantiles --------------
    ent = reachable_entries(table, L, t_m0, t_obj, 0.0)
    res["n_entries"] = len(ent)
    p = p_record(ent, L, t_obj, t_m0)
    res["p_record"] = p
    res["expected"] = p * res["n"]
    res["p_zero"] = math.exp(-res["expected"]) if res["expected"] < 700 else 0.0
    res["gaps_per_record"] = (1.0 / p) if p > 0 else float("inf")

    qs = {}
    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        qs[q] = dbest_quantile(ent, L, t_obj, t_m0, res["n"], q)
    res["dbest_q"] = qs
    if not math.isnan(res["d_best"]):
        res["d_best_sigmas"] = res["d_best"] / sigma
        # where does the observation sit in the predicted d_best CDF?
        res["d_best_cdf"] = (
            1.0 - p_record(ent, L, t_obj, t_m0, res["d_best"])
        ) ** res["n"]

    # --- nearest target + uncertainty band --------------------------------
    g_star, mu_star, lo_star, c_star = easiest_target(ent, L, t_obj, t_m0)
    res["t_easy"] = (g_star, mu_star, lo_star, c_star)
    if sel is None:
        band = {}
        for f in (0.95, 1.0, 1.05):
            sg = sigma * f
            p_f = p_record(reachable_entries(table, L, m0, sg, 0.0), L, sg, m0)
            band[f] = (1.0 / p_f) if p_f > 0 else float("inf")
        res["band_kind"] = "sigma +-5%"
    else:
        # with a fitted family the honest uncertainty is the FAMILY spread,
        # not a sigma perturbation: the shape is what we do not know.
        band = {k: v["gpr"] for k, v in fams.items()}
        res["band_kind"] = "tail-family spread"
    res["gpr_band"] = band
    finite = [v for v in band.values() if v > 0 and v != float("inf")]
    res["band_factor"] = (max(finite) / min(finite)) if len(finite) > 1 else float("nan")
    res["obs_gaps_per_record"] = (
        res["n"] / res["n_records"] if res.get("n_records") else float("nan")
    )

    # --- frontier-table probe ---------------------------------------------
    # Everything else in the pipeline is verified (exponential tail for this
    # cover, L constant to 0.000%, size = the even-gap lattice, bin verified
    # against Monte-Carlo), so the table is the only remaining free input.
    # These two probes quantify it: a sensitivity grid (what a uniform table
    # error would do) and a calibration solve (how large it would have to be).
    res["mu_grid"] = {}
    if not math.isnan(L):
        ent_x = reachable_entries(table, L, t_m0, t_obj, 1.5)
        for x in (-1.5, -1.0, -0.5, 0.5):
            p_x = p_record(ent_x, L, t_obj, t_m0, 0.0, x)
            res["mu_grid"][x] = p_x * res["n"]
        if res.get("n_records"):
            sh, ok = calibrate_mu_shift(ent, L, t_obj, t_m0, res["n"],
                                       res["n_records"])
            res["mu_needed"] = sh
            res["mu_needed_ok"] = ok

    # --- frontier slope in the target region: DIAGNOSTIC ONLY ------------
    # Do NOT rescale sigma by it.  The pointwise sum in p_record already
    # evaluates mu(g) at EVERY size, so folding the frontier slope into sigma
    # would double-count it.  Measured 2026-09-17 on the fleet pair (shift1017
    # strong vs lex, threshold 18): with the slope folded in either direction
    # the strong-cover prediction moves 5.7 -> 4.9 (divide) or -> 12.8
    # (multiply) against an observed 13, while the lex prediction moves
    # 9.1 -> 6.3 or -> 19.3 against an observed 10 - no single rescale is
    # right for both covers, and the un-rescaled form is the one that
    # reproduces lex (1.10x).  The earlier divide-by-(1-L*dmu/dg) form was
    # also inverted: for mu(g) = a - b*g the margin grows with merit at rate
    # (1+bL) > 1, so sigma_D = sigma*(1-L*dmu/dg), not sigma/(...).
    if g_star > 0:
        slope, npts = frontier_slope(table, int(g_star * 0.95), int(g_star * 1.05))
        res["frontier_slope"] = slope
        res["frontier_pts"] = npts

    if args.gph:
        for key, val in args.gph:
            if key in res["name"] or key in path:
                res["gph"] = val
                break

    # --- threshold sweep --------------------------------------------------
    if args.sweep_max > m0:
        res["sweep"] = threshold_sweep(rows, table, L, sigma, m0, args.sweep_max)
    # --- nearest-target watch list ----------------------------------------
    if args.targets > 0:
        res["targets"] = near_targets(rows, table, L, sigma, m0, args.targets)
    return res


def threshold_sweep(rows, table, L, sigma, m0_base, m0_max):
    """Expected records for the SAME observation period at several thresholds.

    WARNING - this curve is FLAT BY CONSTRUCTION, not by measurement.  For a
    pure exponential tail with the frontier well above the threshold,
    n(m0) = n_base*exp(-(m0-m0_base)/sigma) and p_record(m0) ~ exp(+m0/sigma),
    so the product is invariant.  Do NOT read the flatness as evidence; it is a
    property of the model.  The real threshold coupling is the WALK RATE, which
    this model does not contain (in the JUMP2 chain a higher threshold makes
    the chain walk further per window, hence slower).  The sweep is here to
    make that tautology visible instead of accidentally rediscovering it.
    """
    merits = [r[1] for r in rows]
    n_base = sum(1 for m in merits if m >= m0_base)
    out = []
    m0 = int(math.floor(m0_base))
    while m0 <= int(m0_max):
        n_model = n_base * math.exp(-(m0 - m0_base) / sigma)
        n_obs = sum(1 for m in merits if m >= m0)
        ent = reachable_entries(table, L, m0, sigma, 0.0)
        p = p_record(ent, L, sigma, m0)
        out.append((m0, n_obs, n_model, p, n_model * p))
        m0 += 1
    return out


def near_targets(rows, table, L, sigma, m0, top_n):
    """Watch list: targets ranked by the chance of beating them next.

    For each target size g the relevant facts are its required merit mu(g),
    the best merit this walk already produced at that exact size, and the
    margin between them (positive = beaten, i.e. a record).
    """
    S = sigma if isinstance(sigma, (_PureExp, Surv)) else _PureExp(sigma, m0)
    ent = reachable_entries(table, L, m0, sigma, 0.0)
    scored = []
    for lo, hi, mu in ent:
        a = max(lo, mu, m0)
        b = max(hi, m0)
        if a < b:
            c = S.s(a) - S.s(b)
            scored.append((c, int(round(0.5 * (lo + hi) * L)), mu))
    scored.sort(reverse=True)
    best_at = {}
    for g, m, _ in rows:
        if m > best_at.get(g, -1e18):
            best_at[g] = m
    out = []
    for c, g, mu in scored[:top_n]:
        ours = best_at.get(g)
        margin = (ours - mu) if ours is not None else float("nan")
        out.append((g, mu, ours, margin, c))
    return out


def sigma_of_L(L, anchors=None, fixed=None):
    """HYPOTHESIS: sigma grows linearly with the anchor size (unless fixed).

    Anchored on two measured points (shift507 sigma=1.2618, shift1017
    sigma=1.3716).  Interpolation inside that range is supported by the N3
    fit; anything outside is an extrapolation and must be labelled as such.
    A caller-supplied `fixed` value removes the extrapolation entirely, which
    is how the model-free part of the ranking is separated out.
    """
    if fixed is not None:
        return fixed
    a = anchors or SIGMA_ANCHORS
    (L1, s1), (L2, s2) = a[0], a[-1]
    if L2 == L1:
        return s1
    return s1 + (s2 - s1) * (L - L1) / (L2 - L1)


def shift_scan(shifts, table, m0, anchors=None, sigma_fixed=None):
    """Rank shifts by expected gaps per record, using the frontier table.

    For each shift the hunt anchor is 2^(255+shift), so L is fixed and the
    reachable size band g ~ m*L slides against the FIXED external frontier
    mu(g).  The output is therefore an alignment landscape: which shift puts
    the top of its merit tail onto the cheapest part of the frontier.

    Two columns have different epistemic status:
      needs merit  - model-free (table + L only).  THIS IS THE ALIGNMENT FACT.
      gaps/record  - depends on sigma(L), which is measured only at L=528 and
                     L=882 and extrapolated elsewhere (HYPOTHESIS).
    """
    out = []
    for sh in shifts:
        L = (255.0 + sh) * LN2
        sig = sigma_of_L(L, anchors, sigma_fixed)
        ent = reachable_entries(table, L, m0, sig, 0.0)
        p = p_record(ent, L, sig, m0)
        g_star, mu_star, lo_star, c_star = easiest_target(ent, L, sig, m0)
        out.append({
            "shift": sh,
            "L": L,
            "sigma": sig,
            "g_star": g_star,
            "mu_star": mu_star,
            "p": p,
            "gaps_per_record": (1.0 / p) if p > 0 else float("inf"),
            "extrapolated": sigma_fixed is None and not (528.0 <= L <= 882.0),
        })
    return out


def parse_shifts(text):
    """'258,450,507' or '200:1050:50' -> [shifts]."""
    if ":" in text:
        parts = text.split(":")
        lo = int(parts[0])
        hi = int(parts[1])
        step = int(parts[2]) if len(parts) > 2 else 25
        return list(range(lo, hi + 1, step))
    return [int(x) for x in text.split(",") if x.strip()]


def fmt(x, w=12, prec=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--".rjust(w)
    if isinstance(x, float) and math.isinf(x):
        return "inf".rjust(w)
    return f"{x:{w}.{prec}f}"


def report(res):
    if not res.get("n"):
        print(f"\n== {res['name']}: no usable rows ==")
        return
    print(f"\n=== {res['name']} ===")
    print(f"  FACT  reported gaps              : {res['n']}")
    print(f"  FACT  gap size range             : {res['min_gap']}..{res['max_gap']}"
          f"   max merit {res['max_merit']:.4f}")
    print(f"  FACT  ln(start) median L         : {res['L']:.3f}"
          f"   (spread over the file {res['L_spread_pct']:.3f}%)")
    print(f"  FACT  sizes present in table     : {res['in_table']}/{res['n']}"
          f"   (unknown {res['unknown']})")
    print(f"  FACT  OBSERVED records           : {res['n_records']}")
    print(f"  FACT  best margin d_best          : {res['d_best']:+.4f} merit"
          f"  ({res.get('d_best_sigmas', float('nan')):+.2f} sigma)")
    print(f"  FACT  tail fit sigma (M0={res['m0_used']:.1f})   :"
          f" {res['sigma']:.4f} +- {res['sigma_se']:.4f}  (n={res['n_tail']})")
    print(f"  INFER p_record per gap           : {res['p_record']:.3e}"
          f"   ({res['n_entries']} usable table targets)")
    print(f"  INFER expected records           : {res['expected']:.3f}"
          f"   -> P(observe 0) = {res['p_zero']:.3f}")
    print(f"  INFER gaps per record            : {res['gaps_per_record']:.3e}")
    q = res.get("dbest_q", {})
    if q:
        print("  INFER predicted d_best quantiles : "
              + "  ".join(f"q{int(k*100)}={v:+.2f}" for k, v in sorted(q.items())))
        if "d_best_cdf" in res:
            print(f"  INFER observed d_best CDF pos    : {res['d_best_cdf']:.3f}"
                  "   (0.5 = right on the model median)")
    if "t_easy" in res:
        g_star, mu_star, lo_star, c_star = res["t_easy"]
        print(f"  INFER nearest record target      : size {g_star} needs merit"
              f" {mu_star:.4f}  (bin edge {lo_star:.3f}, p-contrib {c_star:.2e})")
    band = res.get("gpr_band", {})
    if band:
        print(f"  INFER {res.get('band_kind', 'band')} (gaps/rec): "
              + "  ".join(f"{k if isinstance(k, str) else format(k, '.2f')}"
                          f"={v:.2e}"
                          for k, v in sorted(band.items(),
                                             key=lambda kv: str(kv[0]))))
        print(f"  INFER band factor                : {res['band_factor']:.2f}x"
              f"  (ranking vs another file is only resolved if the ratio"
              f" exceeds this)")
    fams = res.get("families", {})
    if fams:
        print(f"  INFER tail families above u_rec={res['u_rec']:g}"
              f" (empirical below, fitted above; n_exc={res.get('n_exceedances', 0)}):")
        for k, v in fams.items():
            ep = (f"  endpoint~{v['endpoint']:.1f}" if "endpoint" in v else "")
            print(f"        {k:10} {ts.describe(k, v['params']):38}"
                  f" gaps/rec={v['gpr']:.3e}  E={v['E']:.2f}{ep}")
        if res.get("n_exceedances", 0) < 300:
            print("        WARNING: fewer than 300 exceedances above u_rec ->"
                  " the SHAPE is not resolved; the spread below is an upper"
                  " bound on the ignorance, not a measurement")
        print(f"        headline uses '{res.get('tail_used', 'exp')}'"
              f" (u_rec={res.get('u_rec_used', float('nan')):g})")
    if "frontier_slope" in res:
        print(f"  DIAG  frontier slope dmu/dg      : {res['frontier_slope']:+.3e}"
              f"   L*dmu/dg = {res['frontier_slope'] * res['L']:+.4f}"
              f"  ({res['frontier_pts']} entries near the target; diagnostic only,"
              f" NOT applied to sigma)")
    if "ho_observed" in res:
        print(f"  HOLD  fit sigma on first half    : {res['ho_sigma']:.4f}"
              f"   (n_fit={res['ho_n_fit']})")
        print(f"  HOLD  predict on last {res['ho_n_test']} gaps : E[records]"
              f" = {res['ho_expected']:.3f}   observed = {res['ho_observed']}"
              f"   P(obs 0) = {res['ho_p_zero']:.3f}")
        if "ho_d_best" in res:
            print(f"  HOLD  out-of-sample best margin  : {res['ho_d_best']:+.4f}"
                  f"   predicted median {res['ho_d_best_q50']:+.4f}"
                  f"   CDF pos {res['ho_d_best_cdf']:.3f}"
                  f"   (n={res['ho_n_margins']})")
    if "gph" in res:
        gph = res["gph"]
        print(f"  INFER gaps/hour supplied         : {gph:.0f}"
              f"   -> hours per record = {res['gaps_per_record'] / gph:.1f}")
    grid = res.get("mu_grid", {})
    if grid:
        base = res["expected"]
        print("  PROBE uniform frontier shift (all table requirements moved):"
              " E[records]")
        for x in sorted(grid):
            ratio = f"   (x{grid[x]/base:.2f} vs nominal)" if base > 0 else ""
            print(f"        mu {x:+.2f}  ->  E = {grid[x]:8.3f}{ratio}")
        if "mu_needed" in res:
            if res["mu_needed_ok"]:
                print(f"  PROBE required shift to reproduce the {res['n_records']}"
                      f" observed record(s): mu {res['mu_needed']:+.3f}"
                      f"  (i.e. the frontier table would have to be"
                      f" {abs(res['mu_needed']):.2f} merit EASIER)")
            else:
                print(f"  PROBE no uniform shift in [-8,+8] reproduces the"
                      f" observed {res['n_records']} record(s) -> the gap is NOT"
                      f" a uniform table offset")
    sw = res.get("sweep")
    if sw:
        best = max(sw, key=lambda r: r[4])
        print(f"  SWEEP expected records for the SAME walk time"
              f"  (flat by construction - see --help)")
        print(f"        {'thr':>4} {'n_obs':>9} {'n_model':>10} {'p_rec':>10}"
              f" {'E[rec]':>9}")
        for m0, n_obs, n_model, p, e in sw:
            star = "" if n_obs > 0 else "*"
            print(f"        {m0:4d} {n_obs:9d} {n_model:10.0f} {p:10.3e}"
                  f" {e:9.3f}{star}")
        print("        * = model-extrapolated tail (no observed gaps that high)")
        print(f"        spread over thr 8..24: {min(r[4] for r in sw if r[0] <= 24):.3f}"
              f" .. {max(r[4] for r in sw if r[0] <= 24):.3f}  (invariance is a"
              f" MODEL property, not an observation)")
    tg = res.get("targets")
    if tg:
        print("  INFER nearest targets (size, required merit, our best at that"
              " size, margin):")
        for g, mu, ours, margin, c in tg:
            if ours is None:
                print(f"        size {g:>7}  needs {mu:8.4f}   not hit yet"
                      f"              p={c:.2e}")
            else:
                flag = "  RECORD" if margin > 0 else ""
                print(f"        size {g:>7}  needs {mu:8.4f}   ours {ours:8.4f}"
                      f"  margin {margin:+7.4f}{flag}   p={c:.2e}")


def main():
    ap = argparse.ArgumentParser(
        description="Record-rate model validation on collected GAP_HUNT data."
    )
    ap.add_argument("files", nargs="*")
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--m0", type=float, default=8.0,
                    help="report threshold used when the data was collected"
                         " (AUTO-CLAMPED up to the smallest merit in the data:"
                         " a walker only reports gaps above its own"
                         " --gap-hunt-min-merit, and fitting below it measures"
                         " (threshold-M0)+sigma instead of sigma)")
    ap.add_argument("--holdout", type=float, default=0.5,
                    help="fraction reserved for prediction (0 disables)")
    ap.add_argument("--gph", action="append", default=None,
                    help="NAME=VALUE gaps per hour for files matching NAME")
    ap.add_argument("--sweep-max", type=float, default=0.0,
                    help="sweep --gap-hunt-min-merit up to this value"
                         " (0 disables; typical 28; the curve is flat BY"
                         " CONSTRUCTION in the exponential-tail regime)")
    ap.add_argument("--targets", type=int, default=0,
                    help="print the N nearest record targets (0 disables)")
    ap.add_argument("--shift-scan", default=None,
                    help="rank shifts by gaps/record, e.g. '258,450,507' or"
                         " '200:1050:50' (offline, model-based)")
    ap.add_argument("--sigma-fixed", type=float, default=None,
                    help="use one sigma for every shift (removes the sigma(L)"
                         " extrapolation from the scan)")
    ap.add_argument("--tail", choices=("exp", "stretched", "gpd"),
                    default="exp",
                    help="extrapolating family above --u-rec (default exp)")
    ap.add_argument("--u-rec", default="auto",
                    help="threshold above which the fitted family takes over:"
                         " 'auto' (deepest with n>=300), 'm0' (report"
                         " threshold = the old behaviour), or a number")
    ap.add_argument("--bin-half", type=float, default=None,
                    help="override the merit-bin half-width in gap-size units"
                         " (default 1.0 = the even-gap lattice; 0.5 reproduces"
                         " the pre-2026-09-17 numbers, which were 2x low)")
    ap.add_argument("--sigma-band", action="store_true",
                    help="force the +-5%% sigma band even with --tail != exp")
    args = ap.parse_args()
    if args.bin_half is not None:
        global BIN_HALF
        BIN_HALF = args.bin_half
        print(f"NOTE: bin half-width overridden to {BIN_HALF:g} gap units "
              f"(1.0 = correct for the even-gap lattice)")

    if not args.files and not args.shift_scan:
        ap.error("give at least one gap file or --shift-scan")

    if args.gph:
        parsed = []
        for item in args.gph:
            if "=" in item:
                k, v = item.split("=", 1)
                parsed.append((k, float(v)))
        args.gph = parsed
    else:
        args.gph = []

    table = load_table(args.table)
    print(f"table: {args.table}  entries={len(table)}  "
          f"M0={args.m0:g}  holdout={args.holdout:g}")

    if args.shift_scan:
        scan = shift_scan(parse_shifts(args.shift_scan), table, args.m0,
                          sigma_fixed=args.sigma_fixed)
        print("\n=== shift alignment scan (offline, model-based) ===")
        print("  anchor = 2^(255+shift)  =>  L = (255+shift)*ln2 ;"
              " sigma(L) from the two measured anchors"
              f" {SIGMA_ANCHORS[0][0]:.0f}->{SIGMA_ANCHORS[0][1]:.4f},"
              f" {SIGMA_ANCHORS[1][0]:.0f}->{SIGMA_ANCHORS[1][1]:.4f}")
        print(f"  {'shift':>5} {'L':>8} {'sigma*':>7} {'easiest size':>13}"
              f" {'needs merit':>12} {'gaps/record':>12}")
        for r in sorted(scan, key=lambda x: x["gaps_per_record"]):
            mark = "*" if r["extrapolated"] else " "
            print(f"  {r['shift']:5d} {r['L']:8.1f} {r['sigma']:7.4f}"
                  f" {r['g_star']:13d} {r['mu_star']:12.4f}"
                  f" {r['gaps_per_record']:12.3e}{mark}")
        print("  * = sigma(L) extrapolated outside the measured anchor range"
              " (HYPOTHESIS, not measurement)")
        print("  gaps/record is per reported gap; multiply by a measured"
              " gaps/hour to get hours/record.")
        if not args.files:
            return

    results = []
    loaded = []
    for path in args.files:
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        rows = load_gaps(path)
        if rows:
            loaded.append((path, rows))

    # A walker only reports gaps with merit >= its own --gap-hunt-min-merit, so
    # the smallest merit in a file IS its report threshold.  Fitting below it
    # measures (threshold - M0) + sigma: a diluted OFFSET that inflates sigma
    # and E[records] and can make a healthy corpus look broken.  Seen live
    # 2026-09-20 on the fleet's threshold-20 corpora: the old default M0=8 read
    # sigma=13.3 and E[records]=66.9 with "P(observe 0)=0.000", while the SAME
    # data fitted from its own threshold gives sigma=1.28 and E=0.97 (0 records
    # observed, P=0.38).  Clamp to the highest per-file threshold - the same
    # rule tail_compare.py applies - and say so.
    data_thr = max(min(r[1] for r in rows) for _, rows in loaded) if loaded \
        else 0.0
    if loaded and args.m0 < data_thr - 1e-9:
        print(f"NOTE: --m0 {args.m0:g} lies BELOW the data's own report"
              f" threshold {data_thr:.4f}; using M0={data_thr:.4f} instead"
              f" (a fit below it measures (threshold-M0)+sigma, not sigma)")
        args.m0 = data_thr

    for path, rows in loaded:
        res = analyse(path, rows, table, args.m0, args)
        res["m0_used"] = args.m0
        report(res)
        results.append(res)

    if len(results) > 1:
        print("\n=== ranking (records first) ===")
        print(f"{'file':34} {'n':>8} {'sigma':>7} {'pred g/rec':>11}"
              f" {'obs g/rec':>10} {'band':>6} {'recs':>5} {'pred E':>7}")
        for r in sorted(results, key=lambda x: x.get("gaps_per_record", 1e18)):
            print(f"{r['name']:34} {r['n']:8d} {r['sigma']:7.4f}"
                  f" {r['gaps_per_record']:11.3e}"
                  f" {r.get('obs_gaps_per_record', float('nan')):10.3e}"
                  f" {r.get('band_factor', float('nan')):5.2f}x"
                  f" {r['n_records']:5d} {r['expected']:7.2f}")
        ranks = sorted(results, key=lambda x: x.get("gaps_per_record", 1e18))
        for i in range(len(ranks) - 1):
            a, b = ranks[i], ranks[i + 1]
            ratio = b["gaps_per_record"] / a["gaps_per_record"]
            need = max(a.get("band_factor", 1.0), b.get("band_factor", 1.0))
            if ratio < need:
                print(f"WARNING: {a['name']} vs {b['name']}: predicted ratio"
                      f" {ratio:.2f}x < band factor {need:.2f}x -> RANKING NOT"
                      f" RESOLVED by this data (a 5% sigma error swaps them)")
        print("\nNOTE: gaps/record is a per-gap lottery ratio; hours/record needs a"
              "\n      measured gaps/hour per configuration (--gph NAME=VALUE).")
        print("NOTE: 'band' is the spread from a +-5% sigma error; a ranking is"
              "\n      only real when the predicted ratio exceeds it.  Check"
              " 'obs g/rec' too:")
        print("      when the model ranks two configs opposite to the observed"
              "\n      counts, the model is under-determined and the data wins.")


if __name__ == "__main__":
    main()
