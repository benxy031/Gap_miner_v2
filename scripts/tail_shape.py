#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tail_shape.py - what shape IS the gap-merit tail above the report threshold?

Why this exists
---------------
`record_rate_model.py` sums record probabilities using an exponential merit
tail S(x) = exp(-(x-m0)/sigma).  On the fleet data that model is calibrated on
the lex cover (1.10x in-sample) but 2.3x pessimistic on the strong cover
(docs/RECORD_RATE_MODEL.md §8.3), and the same threshold sigma cannot describe
both covers.  Before adding machinery, this tool asks the prior question:

    is the tail exponential at all, above the threshold?

The mean-excess function E[m - M0 | m >= M0] answers it directly.  For an
exponential it is CONSTANT (= sigma) at every M0.  It RISES with M0 for a
heavier-than-exponential tail and FALLS for a lighter one.  Measured on the
shift507 lex walk (same cover, same anchor, 755k gaps):

    M0:      8      10     12     14     16     18     19
    sigma: 1.261  1.292  1.283  1.293  1.186  0.980  0.858
                                          +-0.031 +-0.057 +-0.079

i.e. the mean excess falls by ~35 % over 12 -> 19, so the deep tail of that
walk is LIGHTER than exponential.  That single fact invalidates the
exponential extrapolation at the record depth and is what this tool measures
properly, with three nested families and held-out validation.

Families (all fitted above a threshold u_fit, on exceedances x = m - u_fit)
-------------------------------------------------------------------------
    exp       S(x) = exp(-x/s)                       1 param, s = mean(x)
    stretched S(x) = exp(-(x/l)^b)                   2 params; b<1 heavier,
                                                     b=1 exponential,
                                                     b>1 lighter
    gpd       S(x) = (1 + xi*x/B)^(-1/xi)            2 params (peaks over
                                                     threshold law); xi>0
                                                     heavy, xi=0 exponential,
                                                     xi<0 LIGHT with a finite
                                                     upper endpoint
                                                     u_fit + B/|xi|

Fitting is stdlib-only (scipy is not available on the fleet boxes):
    exp       closed form MLE
    stretched profile MLE over b (closed-form l per b via mean(x^b))
    gpd       probability-weighted moments (Hosking & Wallis),
              xi = (4-3r)/(2-r), B = b0*(1-xi), r = b0/b1,
              b1 = (1/n) SUM p_i x_(i), p_i = (i-0.35)/n

Model selection is HELD OUT, never in-sample: each family is fitted above
u_fit and then scored on the exceedances above u_test > u_fit by
    predicted count   N_pred = N(>= u_fit) * S(u_test - u_fit)
    Poisson z         (N_obs - N_pred) / sqrt(N_pred)
    conditional log-likelihood of the held-out exceedances
    the predicted mean-excess curve, to compare with the measured one

`--selftest` fits synthetic samples with known parameters and fails loudly if
an estimator is biased - run it before trusting any number on real data.

Usage
-----
    scripts/tail_shape.py --selftest
    scripts/tail_shape.py gap_hunt_records_f1.txt
    scripts/tail_shape.py gap_hunt_records_f1.txt --u-fit 12 --u-test 16,18,20
    scripts/tail_shape.py data/gap_hunt_records_f1.txt --min-n 50 --u-test 21,23
"""

import argparse
import math
import os
import random
import sys

FAMILIES = ("exp", "stretched", "gpd")


# --------------------------------------------------------------------------
# survival / log-density / mean excess
# --------------------------------------------------------------------------
def survival(fam, p, x):
    """P(X > x) for exceedances x >= 0 (normalised so S(0) = 1)."""
    if x <= 0.0:
        return 1.0
    if fam == "exp":
        return math.exp(-x / p["sigma"])
    if fam == "stretched":
        return math.exp(-(x / p["lam"]) ** p["beta"])
    xi, B = p["xi"], p["beta_scale"]
    if abs(xi) < 1e-9:
        return math.exp(-x / B)
    arg = 1.0 + xi * x / B
    if arg <= 0.0:
        # xi < 0 has a finite upper endpoint: past it the survival is 0.
        # (Without this guard a negative base ** fractional exponent returns a
        # COMPLEX number in Python and silently poisons every downstream sum.)
        return 0.0
    return arg ** (-1.0 / xi)


def logpdf(fam, p, x):
    """log density of the exceedance distribution at x > 0."""
    if fam == "exp":
        return -math.log(p["sigma"]) - x / p["sigma"]
    if fam == "stretched":
        b, l = p["beta"], p["lam"]
        return (math.log(b) + (b - 1.0) * math.log(x)
                - b * math.log(l) - (x / l) ** b)
    xi, B = p["xi"], p["beta_scale"]
    if abs(xi) < 1e-9:
        return -math.log(B) - x / B
    arg = 1.0 + xi * x / B
    if arg <= 0.0:
        return float("-inf")          # outside the support (xi < 0)
    return -math.log(B) - (1.0 + 1.0 / xi) * math.log(arg)


def _simpson(f, a, b, n=2000):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return s * h / 3.0


def mean_excess(fam, p, x):
    """E[X - x | X > x]: constant for exp, falling = lighter tail."""
    if fam == "exp":
        return p["sigma"]
    if fam == "gpd":
        xi, B = p["xi"], p["beta_scale"]
        if xi >= 1.0:
            return float("inf")
        if abs(xi) < 1e-9:
            return B
        return max(0.0, (B + xi * x) / (1.0 - xi))
    s_x = survival(fam, p, x)
    if s_x <= 0.0:
        return float("nan")
    hi = x + 60.0 * p["lam"]          # S decays far faster than this
    return _simpson(lambda t: survival(fam, p, t), x, hi) / s_x


# --------------------------------------------------------------------------
# fitters (all above threshold; xs = exceedances, strictly positive)
# --------------------------------------------------------------------------
def fit_exp(xs):
    return {"sigma": sum(xs) / len(xs)}


def fit_stretched(xs):
    """Profile MLE: for fixed beta the MLE of lam is mean(x^beta)^(1/beta)."""
    n = len(xs)
    sx = sum(math.log(x) for x in xs)

    def prof_ll(b):
        mb = sum(x ** b for x in xs) / n
        lam = mb ** (1.0 / b)
        return n * math.log(b) - n * b * math.log(lam) + (b - 1.0) * sx - n

    lo, hi = 0.25, 4.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = prof_ll(c), prof_ll(d)
    for _ in range(300):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = prof_ll(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = prof_ll(d)
        if b - a < 1e-6:
            break
    beta = 0.5 * (a + b)
    lam = (sum(x ** beta for x in xs) / n) ** (1.0 / beta)
    return {"beta": beta, "lam": lam}


def fit_gpd(xs):
    """Probability-weighted moments (Hosking & Wallis)."""
    n = len(xs)
    xo = sorted(xs)
    b0 = sum(xo) / n
    b1 = sum(((i + 1 - 0.35) / n) * xo[i] for i in range(n)) / n
    if b1 <= 0:
        return {"xi": 0.0, "beta_scale": b0}
    r = b0 / b1
    den = 2.0 - r
    if abs(den) < 1e-12:
        return {"xi": 0.0, "beta_scale": b0}
    xi = (4.0 - 3.0 * r) / den
    scale = b0 * (1.0 - xi)
    # degenerate fits (tiny samples, a few outliers) -> fall back to exp
    if not (0.05 < scale < 1e12) or abs(xi) > 2.0:
        return {"xi": 0.0, "beta_scale": b0}
    return {"xi": xi, "beta_scale": scale}


def fit(fam, xs):
    return {"exp": fit_exp, "stretched": fit_stretched,
            "gpd": fit_gpd}[fam](xs)


def describe(fam, p):
    if fam == "exp":
        return f"sigma={p['sigma']:.4f}"
    if fam == "stretched":
        return f"beta={p['beta']:.4f} lam={p['lam']:.4f}"
    return f"xi={p['xi']:+.4f} scale={p['beta_scale']:.4f}"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_merits(path):
    ms = []
    with open(path, errors="ignore") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                m = float(parts[1])
            except ValueError:
                continue
            if m > 0:
                ms.append(m)
    return ms


def ln_int(n):
    """ln() of a big int without overflow (float() dies above 2^1024)."""
    if n <= 0:
        return float("-inf")
    b = n.bit_length()
    if b <= 53:
        return math.log(n)
    return math.log(n >> (b - 53)) + (b - 53) * math.log(2.0)


def load_full(path):
    """(merits, median ln(start), first start-digit string).

    L = ln(anchor) is the size scale that fixes the merit-to-gap mapping, so
    two files may only be pooled when their L agree; the raw start string lets
    the caller show a common anchor prefix, which is evidence (not proof) that
    the same CRT cover and base produced both files.
    """
    ms, lns, first = [], [], None
    with open(path, errors="ignore") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                m = float(parts[1])
            except ValueError:
                continue
            if m > 0:
                ms.append(m)
            if len(parts) == 3 and first is None:
                first = parts[2].strip()
                try:
                    lns.append(ln_int(int(parts[2])))
                except ValueError:
                    pass
    L = sorted(lns)[len(lns) // 2] if lns else float("nan")
    return ms, L, first


def pool_files(paths):
    """Merge same-config files; refuse if the size scale L does not match.

    Two independent consistency conditions, both mandatory:

    1. L = ln(anchor).  Different L means different size, so the merit axes are
       not comparable at all - hard refusal.
    2. The fit threshold must be >= the HIGHEST member report threshold.  A
       file only reports gaps above its own --gap-hunt-min-merit, so between a
       low member threshold and a high one the high-threshold walker
       contributes nothing and the pooled density in that band is
       under-represented.  Fitting below max(min merit) therefore measures a
       sampling artifact, not the tail.
    """
    infos, all_ms = [], []
    for p in paths:
        ms, L, first = load_full(p)
        infos.append((p, len(ms), L, first, min(ms) if ms else float("nan")))
        all_ms += ms
    print("pool members:")
    for p, n, L, _, mn in infos:
        print(f"  {os.path.basename(p):38} n={n:8d}  L={L:.3f}"
              f"  min_merit={mn:.4f}")
    Ls = [i[2] for i in infos if i[2] == i[2]]
    if Ls and len(Ls) > 1:
        spread = (max(Ls) - min(Ls)) / (sum(Ls) / len(Ls))
        print(f"  L spread = {100*spread:.4f}%")
        if spread > 0.001:
            print("  REFUSING to pool: L differs by more than 0.1%, so the files"
                  " are different sizes and their merit axes are not the same")
            return None, None
    print(f"  highest member report threshold = {max(i[4] for i in infos):.4f}"
          f"  (the pool may only be FITTED above this value)")
    pref = os.path.commonprefix([i[3] for i in infos if i[3]]) if infos else ""
    print(f"  common anchor-digit prefix = {len(pref)} digits (informational"
          f" ONLY: offsets are tiny next to the base, so any cover at the same"
          f" shift shares a long prefix - it does not prove the same cover;"
          f" check the fleet conf that produced the files)")
    print(f"  pooled n = {len(all_ms)}")
    return all_ms, infos


def auto_u_fit(merits, min_n):
    """Smallest 0.5-step threshold with at least min_n merits above it."""
    ms = sorted(merits)
    n = len(ms)
    u = ms[0]
    while u <= ms[-1]:
        k = n - bisect_right(ms, u)
        if k >= min_n:
            return math.ceil(u * 2) / 2.0
        u += 0.5
    return None


def bisect_right(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------
# per-file analysis
# --------------------------------------------------------------------------
def analyse(path, merits, u_fit, u_tests, min_n):
    out = {"path": path, "n": len(merits)}
    if u_fit is None:
        u_fit = auto_u_fit(merits, min_n)
    if u_fit is None:
        out["error"] = f"no threshold with >= {min_n} exceedances"
        return out
    xs = [m - u_fit for m in merits if m > u_fit]
    out["u_fit"] = u_fit
    out["n_fit"] = len(xs)
    if len(xs) < 50:
        out["error"] = f"only {len(xs)} exceedances above {u_fit:g}"
        return out

    # measured mean-excess curve (keep the tail lists for the held-out LL)
    meas = []
    for t in u_tests:
        x = [m - t for m in merits if m >= t]
        if len(x) >= 10:
            meas.append((t, sum(x) / len(x), len(x), [m for m in merits
                                                      if m >= t]))
    out["measured"] = meas

    models = {}
    for fam in FAMILIES:
        p = fit(fam, xs)
        rec = {"params": p, "u_tests": []}
        for t, emp_sig, n_obs, tail_m in meas:
            x0 = t - u_fit
            s = survival(fam, p, x0)
            n_pred = out["n_fit"] * s
            z = ((n_obs - n_pred) / math.sqrt(n_pred)) if n_pred > 0 else 0.0
            ll = 0.0
            if x0 > 0:
                log_s = math.log(s)
                for m in tail_m:
                    ll += logpdf(fam, p, m - u_fit) - log_s
            rec["u_tests"].append({
                "t": t, "n_obs": n_obs, "n_pred": n_pred, "z": z,
                "pred_sigma": mean_excess(fam, p, x0), "emp_sigma": emp_sig,
                "ll": ll,
            })
        rec["ll_total"] = sum(r["ll"] for r in rec["u_tests"])
        rec["z_abs_sum"] = sum(abs(r["z"]) for r in rec["u_tests"])
        models[fam] = rec
    out["models"] = models
    out["best_ll"] = max(FAMILIES, key=lambda f: models[f]["ll_total"])
    out["best_z"] = min(FAMILIES, key=lambda f: models[f]["z_abs_sum"])
    return out


def report(res):
    if "error" in res:
        print(f"\n=== {os.path.basename(res['path'])}: {res['error']} ===")
        return
    print(f"\n=== {os.path.basename(res['path'])} ===")
    print(f"  FACT  gaps={res['n']}  fit above u_fit={res['u_fit']:g}"
          f"  (n_fit={res['n_fit']})")
    if res["measured"]:
        print("  FACT  measured mean excess: "
              + "  ".join(f"M0={t:g}: {s:.4f} (n={n})"
                          for t, s, n, _ in res["measured"]))
    print(f"  {'family':10} {'params':38} {'held-out LL':>12} {'sum|z|':>8}")
    for fam in FAMILIES:
        rec = res["models"][fam]
        tag = ""
        if fam == res["best_ll"]:
            tag += " <-best LL"
        if fam == res["best_z"]:
            tag += " <-best z"
        print(f"  {fam:10} {describe(fam, rec['params']):38}"
              f" {rec['ll_total']:12.2f} {rec['z_abs_sum']:8.2f}{tag}")
    for fam in FAMILIES:
        rec = res["models"][fam]
        for r in rec["u_tests"]:
            print(f"    {fam:10} M0={r['t']:5.1f}  n_obs={r['n_obs']:5d}"
                  f"  n_pred={r['n_pred']:8.2f}  z={r['z']:+6.2f}"
                  f"  sigma_pred={r['pred_sigma']:.4f}"
                  f"  sigma_emp={r['emp_sigma']:.4f}")
    p = res["models"]["gpd"]["params"]
    if p["xi"] < -0.02:
        end = res["u_fit"] + p["beta_scale"] / abs(p["xi"])
        print(f"  NOTE  GPD xi<0 => finite upper endpoint at merit"
              f" ~{end:.2f} (no record is possible beyond it)")
    elif abs(p["xi"]) <= 0.02:
        print(f"  NOTE  GPD xi={p['xi']:+.4f} is indistinguishable from 0 at"
              f" this threshold => no finite endpoint resolved here;"
              f" refit higher (--u-fit) to probe the deep curvature")


# --------------------------------------------------------------------------
# self-test: recover known parameters from synthetic samples
# --------------------------------------------------------------------------
def selftest():
    rng = random.Random(20260917)
    n = 200000
    ok = True

    xs = [-math.log(rng.random()) * 1.30 for _ in range(n)]
    s = fit_exp(xs)["sigma"]
    print(f"  exp       sigma_true=1.3000  sigma_hat={s:.4f}  "
          f"({100*abs(s-1.3)/1.3:.2f}% off)")
    ok &= abs(s - 1.3) / 1.3 < 0.02

    b_true, l_true = 1.35, 3.0
    xs = [l_true * (-math.log(rng.random())) ** (1.0 / b_true)
          for _ in range(n)]
    p = fit_stretched(xs)
    print(f"  stretched beta_true={b_true:.4f} beta_hat={p['beta']:.4f}  "
          f"lam_true={l_true:.4f} lam_hat={p['lam']:.4f}")
    ok &= abs(p["beta"] - b_true) / b_true < 0.02

    # GPD via inverse transform: X = (B/xi)((1-U)^-xi - 1)
    xi_true, B_true = 0.25, 1.5
    xs = []
    for _ in range(n):
        u = rng.random()
        xs.append((B_true / xi_true) * ((1.0 - u) ** (-xi_true) - 1.0))
    p = fit_gpd(xs)
    print(f"  gpd       xi_true={xi_true:+.4f} xi_hat={p['xi']:+.4f}  "
          f"B_true={B_true:.4f} B_hat={p['beta_scale']:.4f}")
    ok &= abs(p["xi"] - xi_true) < 0.02

    # exponential data must give xi ~ 0 through the GPD path
    xs = [-math.log(rng.random()) * 1.3 for _ in range(n)]
    p = fit_gpd(xs)
    print(f"  gpd(exp)  xi_true= 0.0000 xi_hat={p['xi']:+.4f}  "
          f"B_hat={p['beta_scale']:.4f} (sigma_true=1.3000)")
    ok &= abs(p["xi"]) < 0.02

    print(f"  mean-excess exp(+0)={mean_excess('exp', {'sigma': 1.3}, 0.0):.4f}"
          f"  exp(+5)={mean_excess('exp', {'sigma': 1.3}, 5.0):.4f}  (constant)")
    print(f"  mean-excess gpd(xi=-0.2) at x=0:"
          f" {mean_excess('gpd', {'xi': -0.2, 'beta_scale': 1.5}, 0.0):.4f}"
          f"  at x=5:"
          f" {mean_excess('gpd', {'xi': -0.2, 'beta_scale': 1.5}, 5.0):.4f}"
          f"  (must fall)")
    print("  SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="Fit the gap-merit tail shape above the report threshold.")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--u-fit", default="auto",
                    help="fit threshold ('auto' = smallest with --min-n)")
    ap.add_argument("--min-n", type=int, default=2000,
                    help="exceedances required when --u-fit auto")
    ap.add_argument("--u-test", default="",
                    help="comma list of held-out thresholds (default: "
                         "u_fit+2,4,6,8)")
    ap.add_argument("--pool", action="store_true",
                    help="merge all input files into ONE fit (requires matching"
                         " L; use for the same cover+shift measured at"
                         " different report thresholds)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.files:
        ap.error("give at least one results file (or --selftest)")

    u_fit = None if args.u_fit == "auto" else float(args.u_fit)
    print(f"gate: {len(FAMILIES)} families, held-out scoring, stdlib only")

    if args.pool and len(args.files) > 1:
        all_ms, infos = pool_files(args.files)
        if all_ms is None:
            return 2
        thr_max = max(i[4] for i in infos)
        m0 = u_fit if u_fit is not None else auto_u_fit(all_ms, args.min_n)
        if m0 < thr_max:
            print(f"  NOTE: requested u_fit={m0:g} is below the highest member"
                  f" threshold {thr_max:.4f}; raising u_fit to {thr_max:.4f}"
                  f" (below that, the high-threshold walker contributes"
                  f" nothing and the density there is under-sampled)")
            m0 = math.ceil(thr_max * 100) / 100.0
        tests = ([float(x) for x in args.u_test.split(",")] if args.u_test
                 else [m0 + d for d in (2, 4, 6, 8)])
        label = "+".join(os.path.basename(p) for p in args.files)
        tests = [t for t in tests if t > m0]
        if not tests:
            print("  no held-out threshold above u_fit; pass --u-test")
            return 2
        report(analyse(label, all_ms, m0, tests, args.min_n))
        return 0

    for path in args.files:
        if not os.path.exists(path):
            print(f"missing: {path}", file=sys.stderr)
            continue
        merits = load_merits(path)
        if not merits:
            print(f"no merits in {path}", file=sys.stderr)
            continue
        m0 = u_fit if u_fit is not None else auto_u_fit(merits, args.min_n)
        if m0 is None:
            print(f"\n=== {os.path.basename(path)}: no threshold with"
                  f" >= {args.min_n} exceedances ===")
            continue
        if args.u_test:
            tests = [float(x) for x in args.u_test.split(",")]
        else:
            tests = [m0 + d for d in (2, 4, 6, 8)]
        tests = [t for t in tests if t > m0]
        res = analyse(path, merits, m0, tests, args.min_n)
        report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
