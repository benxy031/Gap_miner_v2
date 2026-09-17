#!/usr/bin/env python3
"""tail_compare.py — compare gap-tail sigma between two walker output files.

The same-size / different-cover discriminating test: both files should be
from walkers at the SAME size (shift) with DIFFERENT covers (e.g. strong
vs lex).  If their tail sigmas differ significantly, the cover shapes the
tail (a new hunt-quality lever); if they agree, the tail is size-driven.

Usage:
    tail_compare.py [FILE_A FILE_B] [M0] [--plot PREFIX]

  FILE_A/B   gap-hunt results files (`<gap> <merit> <startprime>` lines).
             Default: gap_hunt_records_f1.txt (763-bit) vs
             gap_hunt_records_f2.txt (1273-bit) — the N3 size comparison.
  M0         primary tail-fit threshold for the VERDICT (default 10).
             AUTO-CLAMPED to the data's own report threshold: a walker only
             writes gaps with merit >= its --gap-hunt-min-merit, so fitting
             below the smallest merit in the file measures an offset rather
             than a tail.  A second verdict is printed at the deepest
             threshold that still has n>=100 in both files (the record-
             relevant depth); if the two verdicts disagree in sign, the
             covers have crossing tails and the threshold sigma is not the
             record-rate sigma.
  The sigma table sweeps the fixed record-merit grid 16, 18, 19, 20, ..., 35;
  thresholds with fewer than MIN_N records in either file are printed as
  skipped (no fit).
  --plot     also write PREFIX_cdf.png (empirical + exp-fit survival at M0)
             and PREFIX_sigma.png (tail sigma vs threshold over the sweep);
             requires numpy/matplotlib, skipped gracefully if unavailable

Runs without numpy/matplotlib for the text part (fleet boxes).
"""
import sys
import math
import os


def resolve(path):
    """Fall back to data/<path> so fleet boxes (results in data/) work."""
    if os.path.exists(path):
        return path
    alt = os.path.join("data", os.path.basename(path))
    return alt if os.path.exists(alt) else path


def load(path):
    merits = []
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
                merits.append(m)
    return merits


def fit(merits, m0):
    excess = [m - m0 for m in merits if m >= m0]
    if not excess:
        return 0.0, 0.0, 0
    s = sum(excess) / len(excess)
    return s, s / math.sqrt(len(excess)), len(excess)


# Record-merit grid for the sigma table: 16, 18, 19, 20, ..., 35.
SWEEP = [16.0] + [float(t) for t in range(18, 36)]
MIN_N = 5    # fewer records per file than this -> no meaningful exp-fit
MIN_N_DEEP = 100   # depth needed for the second (record-relevant) verdict


def main():
    args = sys.argv[1:]
    plot = None
    if "--plot" in args:
        i = args.index("--plot")
        plot = args[i + 1] if i + 1 < len(args) else "tail_compare"
        del args[i:i + 2]
    pos = [a for a in args if not a.startswith("--")]
    if len(pos) == 0:
        pos = ["gap_hunt_records_f1.txt", "gap_hunt_records_f2.txt"]
    elif len(pos) == 1:
        pos.append("gap_hunt_records_f2.txt")
    fa, fb = resolve(pos[0]), resolve(pos[1])
    m0_req = float(pos[2]) if len(pos) > 2 else 10.0

    ma = load(fa)
    mb = load(fb)
    if not ma or not mb:
        print("one or both files have no parseable records", file=sys.stderr)
        return 2

    print(f"A: {fa}  n={len(ma)}  best={max(ma):.4f}")
    print(f"B: {fb}  n={len(mb)}  best={max(mb):.4f}")

    # A walker only writes gaps with merit >= its --gap-hunt-min-merit, so the
    # smallest merit in a file IS the report threshold.  Fitting below it does
    # not measure a tail: mean(m - M0 | m >= M0) becomes
    # (threshold - M0) + sigma, i.e. an arbitrary offset that also inflates the
    # standard error by the same factor and can hide a real difference.
    thr_a, thr_b = min(ma), min(mb)
    thr_eff = max(thr_a, thr_b)
    m0 = m0_req
    forced = False
    if m0 < thr_eff:
        m0 = thr_eff
        forced = True

    # Verdict fit at M0 (may lie outside the table sweep).
    sa, sea, na0 = fit(ma, m0)
    sb, seb, nb0 = fit(mb, m0)
    if forced:
        print(f"NOTE: requested M0={m0_req:.1f} lies BELOW the data's own report "
              f"threshold (min merit A={thr_a:.4f}, B={thr_b:.4f}); the verdict "
              f"uses M0={m0:.4f} instead")
        print(f"      (fitting below the threshold measures "
              f"(threshold-M0)+sigma and dilutes the SE the same way)")
    z_main = None
    if na0 >= MIN_N and nb0 >= MIN_N:
        z_main = (sb - sa) / math.sqrt(sea**2 + seb**2)

    print(f"{'m':>5} {'sigmaA':>8} {'+/-':>8} {'sigmaB':>8} {'+/-':>8} "
          f"{'sep':>7}   (nA, nB)")
    sweep_rows = []   # (t, a1, e1, a2, e2, z, n1, n2) for the sigma plot
    for t in SWEEP:
        a1, e1, n1 = fit(ma, t)
        a2, e2, n2 = fit(mb, t)
        ok = n1 >= MIN_N and n2 >= MIN_N
        z = (a2 - a1) / math.sqrt(e1**2 + e2**2) if ok else None
        if ok:
            sweep_rows.append((t, a1, e1, a2, e2, z, n1, n2))
            print(f"{t:5.1f} {a1:8.4f} {e1:8.4f} {a2:8.4f} {e2:8.4f} "
                  f"{z:7.1f}   (nA={n1}, nB={n2})")
        else:
            print(f"{t:5.1f} {'--':>8} {'--':>8} {'--':>8} {'--':>8} "
                  f"{'--':>7}   (nA={n1}, nB={n2} <{MIN_N}: fit skipped)")

    if z_main is None:
        print(f"\nVERDICT at M0={m0:.1f}: too few records for a fit "
              f"(nA={na0}, nB={nb0}); pass a lower M0")
        return 2
    print(f"\nVERDICT at M0={m0:.1f}: sigmaB - sigmaA = {sb-sa:+.4f} "
          f"= {z_main:.1f} sigma "
          f"-> {'SIGNIFICANT tail difference' if abs(z_main) > 5 else 'no significant tail difference'}"
          f" (interpret as cover effect ONLY if the two files are same-size)")

    # The verdict above sits at the report threshold, which is where most of
    # the data is but NOT where records happen: record targets need merits far
    # above it.  Report the deepest threshold that still has enough points in
    # both files, so a tail crossing (heavier near the threshold, lighter at
    # the record depth, or the reverse) cannot hide behind a single number.
    deep = None
    for row in sweep_rows:
        if row[6] >= MIN_N_DEEP and row[7] >= MIN_N_DEEP:
            deep = row
    if deep is not None and abs(deep[0] - m0) > 0.5:
        t, a1, e1, a2, e2, z, n1, n2 = deep
        print(f"VERDICT at M0={t:.1f} (deepest with n>={MIN_N_DEEP} in both):"
              f" sigmaB - sigmaA = {a2-a1:+.4f} = {z:.1f} sigma"
              f" -> {'SIGNIFICANT' if abs(z) > 5 else 'not significant'}"
              f"  (nA={n1}, nB={n2})")
        if (a2 - a1) * (sb - sa) < 0:
            print("      TAIL CROSSING: the two covers rank differently near the"
                  " threshold and at the record depth — the threshold sigma is"
                  " NOT the record-rate sigma")

    if plot:
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(numpy/matplotlib unavailable — skipping plot)")
            return 0
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m, lab, c, s in ((ma, "A " + fa, "tab:blue", sa),
                             (mb, "B " + fb, "tab:orange", sb)):
            xs = np.arange(m0, max(m) + 0.5, 0.25)
            surv = [float(sum(1 for x in m if x >= v) / len(m)) for v in xs]
            ax.plot(xs, surv, color=c, label=f"{lab} (empirical)")
            ax.plot(xs, [math.exp(-(v - m0) / s) for v in xs], "--",
                    color=c, label=f"fit exp(-(m-{m0:.0f})/{s:.3f})")
        ax.set_yscale("log")
        ax.set_xlabel("merit threshold m")
        ax.set_ylabel("P(merit >= m)")
        ax.set_title(f"cover A/B at the same size: "
                     f"sigmaA={sa:.4f} sigmaB={sb:.4f} ({z_main:.1f} sigma)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{plot}_cdf.png", dpi=130)
        print(f"wrote {plot}_cdf.png")

        if sweep_rows:
            fig2, ax2 = plt.subplots(figsize=(10, 5.5))
            xs2 = [r[0] for r in sweep_rows]
            ya = [r[1] for r in sweep_rows]
            ea = [r[2] for r in sweep_rows]
            yb = [r[3] for r in sweep_rows]
            eb = [r[4] for r in sweep_rows]
            ax2.errorbar(xs2, ya, yerr=ea, fmt="o-", color="tab:blue",
                         capsize=3, markersize=4, label="A " + fa)
            ax2.errorbar(xs2, yb, yerr=eb, fmt="o-", color="tab:orange",
                         capsize=3, markersize=4, label="B " + fb)
            ax2.set_xlabel("merit threshold m")
            ax2.set_ylabel("tail sigma (mean excess)")
            ax2.set_title("tail sigma vs threshold (sweep 16..35)")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(f"{plot}_sigma.png", dpi=130)
            print(f"wrote {plot}_sigma.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
