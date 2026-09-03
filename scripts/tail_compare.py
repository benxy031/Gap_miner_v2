#!/usr/bin/env python3
"""tail_compare.py — compare gap-tail sigma between two walker output files.

The same-size / different-cover discriminating test: both files should be
from walkers at the SAME size (shift) with DIFFERENT covers (e.g. strong
vs lex).  If their tail sigmas differ significantly, the cover shapes the
tail (a new hunt-quality lever); if they agree, the tail is size-driven.

Usage:
    tail_compare.py FILE_A FILE_B [M0] [--plot PREFIX]

  FILE_A/B   gap-hunt results files (`<gap> <merit> <startprime>` lines)
  M0         primary tail-fit threshold (default 10; a sweep M0..M0+4 is
             always printed)
  --plot     also write PREFIX_cdf.png (requires numpy/matplotlib; skipped
             gracefully if unavailable)

Runs without numpy/matplotlib for the text part (fleet boxes).
"""
import sys
import math


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


def main():
    args = sys.argv[1:]
    plot = None
    if "--plot" in args:
        i = args.index("--plot")
        plot = args[i + 1] if i + 1 < len(args) else "tail_compare"
        del args[i:i + 2]
    if len(args) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    fa, fb = args[0], args[1]
    m0 = float(args[2]) if len(args) > 2 else 10.0

    ma = load(fa)
    mb = load(fb)
    if not ma or not mb:
        print("one or both files have no parseable records", file=sys.stderr)
        return 2

    print(f"A: {fa}  n={len(ma)}  best={max(ma):.4f}")
    print(f"B: {fb}  n={len(mb)}  best={max(mb):.4f}")
    print(f"{'M0':>5} {'sigmaA':>8} {'+/-':>8} {'sigmaB':>8} {'+/-':>8} "
          f"{'sep':>7}")
    sa, sea, _ = fit(ma, m0)
    sb, seb, _ = fit(mb, m0)
    z_main = None
    for t in (m0, m0 + 1, m0 + 2, m0 + 3, m0 + 4):
        a1, e1, n1 = fit(ma, t)
        a2, e2, n2 = fit(mb, t)
        z = (a2 - a1) / math.sqrt(e1**2 + e2**2) if e1 and e2 else 0.0
        if t == m0:
            sa, sea, sb, seb, z_main = a1, e1, a2, e2, z
        print(f"{t:5.1f} {a1:8.4f} {e1:8.4f} {a2:8.4f} {e2:8.4f} "
              f"{z:7.1f} sigma   (nA={n1}, nB={n2})")

    print(f"\nVERDICT at M0={m0:.1f}: sigmaB - sigmaA = {sb-sa:+.4f} "
          f"= {z_main:.1f} sigma "
          f"-> {'SIGNIFICANT tail difference' if abs(z_main) > 5 else 'no significant tail difference'}"
          f" (interpret as cover effect ONLY if the two files are same-size)" )

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
