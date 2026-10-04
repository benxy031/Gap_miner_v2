#!/usr/bin/env python3
"""cover_gain_plot.py — a CRT-cover hunt's merit tail vs the exact HL natural law.

Usage:
    scripts/cover_gain_plot.py [--log PATH] [--out PATH]

Reads a gapminer records log (grammar of scripts/records_report.py) and draws
two panels:

  left   P(merit >= m): the empirical covered tail (step) against the
         Hardy-Littlewood natural law at the corpus's own L, evaluated with the
         EXACT c1,c2 of the distributed coefficient database (every gap <=
         90090; c3 sampled power law, c4 fitted) and normalized at the log's
         threshold m0 = min merit.  A dashed exponential of the fitted
         mean-excess scale is drawn for contrast.
  right  observed/natural at the report's fixed sigma-spaced thresholds, with
         Poisson error bars and the fitted per-merit gain, so the cover's
         multiplier is readable directly.

The natural reference is NOT fitted to this corpus: it is the external law
(scripts/hl_natural.py).  Defaults: --log /tmp/fleet_s720.log,
--out data/fleet_s720_cover_vs_nature.png.
"""
import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import records_report as rr           # noqa: E402
import hl_natural as hn               # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/fleet_s720.log")
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/fleet_s720_cover_vs_nature.png"))
    a = ap.parse_args(argv)

    cands = rr.build_candidates(rr.load_file(a.log)[0])[0]
    merits = sorted(c.merit for c in cands if c.merit is not None)
    n = len(merits)
    if not n:
        raise SystemExit("no candidates in %s" % a.log)
    m0 = merits[0]
    L = rr.group_L(cands)
    sigma, _ = rr.fit_sigma(merits)
    nat = hn.load()
    if nat is None:
        raise SystemExit("HL baseline unavailable (forum/ missing)")

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Natural survival, normalized at the log's own threshold.
    e0 = nat.E(L, m0)
    grid = np.arange(m0, merits[-1] + 0.25, 0.1)
    nat_srv = np.array([math.exp(-(nat.E(L, m) - e0)) for m in grid])
    emp = np.array([(n - np.searchsorted(merits, m, side="left")) / n
                    for m in grid])
    sigma_nat = nat.sigma(L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))

    ax1.step(grid, emp, where="post", color="tab:blue", lw=2.0,
             label=f"covered hunt (empirical, n={n})")
    ax1.plot(grid, nat_srv, "k:", lw=2.0,
             label=f"HL natural, exact c1,c2 (sigma_nat={sigma_nat:.3f})")
    ax1.plot(grid, np.exp(-(grid - m0) / sigma), "--", color="gray", lw=1.4,
             label=f"exp fit to the hunt (sigma_eff={sigma:.3f})")
    ax1.set_yscale("log")
    ax1.set_ylim(0.6 / n, 1.6)
    ax1.set_xlabel("merit threshold m")
    ax1.set_ylabel("P(merit >= m)")
    ax1.set_title("Covered tail vs the natural law at the same L\n"
                  "(dotted: no cover; flatter than dotted = cover working)")
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(alpha=0.3)

    # Right: observed/natural at the report's sigma-spaced thresholds.
    thr, obs, en, rat, err = [], [], [], [], []
    for k in range(1, 6):
        t = m0 + k * sigma
        c = sum(1 for m in merits if m >= t)
        e = n * math.exp(-(nat.E(L, t) - e0))
        thr.append(t); obs.append(c); en.append(e)
        rat.append(c / e if e >= 5 else float("nan"))
        err.append((c / e) / math.sqrt(c) if (e >= 5 and c > 0)
                   else float("nan"))
    ax2.errorbar(thr, rat, yerr=err, fmt="o", color="tab:red", capsize=4,
                 label="observed / natural (Poisson 2 sigma)")
    pts = [(t, math.log(r)) for t, r, e in zip(thr, rat, en)
           if r == r and e >= 5]
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        b = np.polyfit(xs, ys, 1)[0]
        m_lo, m_hi = xs[0], xs[-1]
        fit_x = np.linspace(m_lo, m_hi, 50)
        # anchor the fitted line at the first valid point
        y0 = ys[0] - b * xs[0]
        fit_y = np.exp(y0 + b * fit_x)
        ax2.plot(fit_x, fit_y, "-", color="darkred", lw=1.2, alpha=0.8,
                 label="fit: x%.2f per merit unit (%.1f-%.1f)"
                       % (math.exp(b), m_lo, m_hi))
        ax2.annotate("cover gain grows with depth",
                     xy=(xs[-1], ys[-1]), xytext=(xs[0] + 0.4, ys[-1]),
                     fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="darkred"))
    ax2.axhline(1.0, color="k", lw=1.0, ls=":")
    ax2.text(m0 + 0.15, 1.05, "1.0 = no cover effect", fontsize=8, color="k")
    ax2.set_xlabel("merit threshold m")
    ax2.set_ylabel("observed / HL natural")
    ax2.set_title("Cover multiplier vs the natural law\n"
                  "(external reference; bars are the counting error)")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(alpha=0.3)

    bits = L / math.log(2.0)
    fig.suptitle("Gapcoin CRT-cover hunt (shift 720, ~%.0f-bit, L=ln x=%.1f) "
                 "vs the Hardy-Littlewood natural law" % (bits, L),
                 fontsize=12)
    fig.text(0.5, 0.066,
             "Covered hunt: fleet walker shift720_p98_lex_m45 (Gapcoin); log "
             "2026-09-25 17:46 .. 2026-10-04 07:33 UTC; n=%d; m0=%.4f; "
             "sigma_eff=%.3f vs sigma_nat=%.3f." % (n, m0, sigma, sigma_nat),
             ha="center", fontsize=8.4)
    fig.text(0.5, 0.042,
             "Natural law: HL 4-parameter (P. Williams), EXACT c1,c2 from the "
             "distributed database (B1,B2 to g=90090), c3 sampled power law, "
             "c4 fitted.", ha="center", fontsize=8.4)
    fig.text(0.5, 0.018,
             "Right-panel gains: %s (x%.2f per merit unit)."
             % (", ".join("x%.2f @ m%.2f" % (r, t)
                          for t, r in zip(thr, rat) if r == r),
                math.exp(b) if len(pts) >= 2 else float("nan")),
             ha="center", fontsize=8.4)
    fig.tight_layout(rect=(0, 0.085, 1, 0.96))
    fig.savefig(a.out, dpi=130)
    print("saved:", a.out)
    print("n=%d  L=%.2f  m0=%.4f  sigma_eff=%.3f  sigma_nat=%.3f"
          % (n, L, m0, sigma, sigma_nat))
    for t, c, e, r in zip(thr, obs, en, rat):
        print("  m>=%.4f  obs=%4d  nat_exp=%7.1f  ratio=%s"
              % (t, c, e, ("%.2f" % r) if r == r else "-"))


if __name__ == "__main__":
    sys.exit(main())
