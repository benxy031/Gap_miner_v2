#!/usr/bin/env python3
"""phase0_hl_compare.py — measured Phase-0 merit tail vs Hardy-Littlewood models.

WHY
---
The Phase-0 scanner measures the UNBIASED (no covering) natural merit tail of
consecutive prime gaps at L = ln(2e20) = 46.744.  To read a number like
"N(merit>=20) = 667 per 1e14" you need a reference for what nature should give
at that prime scale.  This script overlays two Hardy-Littlewood baselines:

  HL-1t  (one-term HL / Cramer):
      N(m) = 2*C2 * R/L * exp(-m)
      (the "1.32 * e^-m / ln N" curve used elsewhere in this repo)

  HL-4p  (P. Williams' four-parameter consecutive-gap model, forum/):
      rho_g(L) = (S_g / L^2) * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4))
      N(m) = R * sum over even g >= m*L of rho_g(L)

  Coefficients come straight from
      forum/hl_gap_4param_exact_parameters_with_first_occurrence.csv   (g <= 3600)
      forum/hl_gap_3param_exact_parameters_with_first_occurrence.csv   (g <= 9990)
  Our relevant gaps are g <= ~1400 (m <= 30), i.e. INSIDE the order-4 table, so
  table rows are used exactly (no power-law extrapolation of the c's).
  Cross-verified 2026-10-04 against the community exact database
  (forum/hl_gap_distributed/data/hl_gap_cumulants.csv): for g = 700..1500 the
  two sources agree to <= 1e-9 relative on every coefficient (c1 to 1e-15), so
  this tool needs no coefficient update for the Phase-0 range.
  The g-tail beyond 9990 contributes < 1e-30 at this L (e^{-(9990-12)/46.7}).

CAVEAT / PROVENANCE (travels with the figure)
--------------------------------------------
The 4-param coefficients are L-independent (exact inclusion-exclusion over the
internal offsets + singular series), and the author demonstrates histogram
reconstructions at e^32, 10^25, 2^92, e^62 (L ~ 32..64) to +0.003..0.03 %.
Our L = 46.74 AND our g <= ~1400 sit INSIDE that demonstrated band and inside
the tabulated g-range, so this overlay is an interpolation in both axes - the
agreement below is therefore a genuine independent test of the model against
an UNBIASED 1e14 corpus.

GATES
-----
* sum_g rho_g over all tabulated g must reproduce the gap density 1/L
  (one gap per prime); printed as a ratio and expected within a few %.
* The measured cumulative counts are read directly from the gap log; the
  session header supplies start/length so N is per the actual range.

Usage:
    scripts/phase0_hl_compare.py [log] [--out PNG] [--dark] [--m-min 15] [--m-max 30]
"""
import argparse
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_table(path):
    """{gap: {S_g, c1..c4}} from an HL parameter CSV."""
    rows = {}
    with open(path, errors="ignore") as fh:
        head = fh.readline().rstrip("\n").split(",")
        idx = {k: i for i, k in enumerate(head)}
        need = ("gap", "S_g", "c1", "c2", "c3", "c4")
        for k in need:
            if k not in idx:
                raise ValueError("%s: no column %s" % (path, k))
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < len(head):
                p += [""] * (len(head) - len(p))
            try:
                g = int(p[idx["gap"]])
            except ValueError:
                continue
            row = {}
            for c in need[1:]:
                v = p[idx[c]]
                try:
                    row[c] = float(v)
                except ValueError:
                    row[c] = 0.0
            rows[g] = row
    return rows


def read_log(path):
    """(start, length, {m: cumulative_count}) for the first session in the log."""
    start = length = None
    merits = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#"):
                if start is None and "session" in line:
                    for tok in line.split():
                        if tok.startswith("start="):
                            start = int(tok.split("=", 1)[1])
                        elif tok.startswith("length="):
                            length = int(tok.split("=", 1)[1])
                continue
            p = line.split()
            if len(p) >= 4:
                try:
                    merits.append(float(p[3]))
                except ValueError:
                    continue
    if start is None or length is None:
        raise SystemExit("no session header with start=/length= in %s" % path)
    return start, length, merits


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=os.path.join(REPO, "data/p0_campaign_2e20.log"))
    ap.add_argument("--out", default=os.path.join(REPO, "data/p0_hl_compare.png"))
    ap.add_argument("--dark", action="store_true", help="also save *_dark.png")
    ap.add_argument("--m-min", type=int, default=15)
    ap.add_argument("--m-max", type=int, default=30, help="top merit bin; rows with single-digit counts are Poisson noise, not a model test (raised 27 -> 30 when the 2e15 campaign produced events at m28-30)")
    args = ap.parse_args(argv)

    start, length, merits = read_log(args.log)
    L = math.log(start)
    R = float(length)
    ms = list(range(args.m_min, args.m_max + 1))
    meas = {m: sum(1 for x in merits if x >= m) for m in ms}

    # ---- HL tables ------------------------------------------------------
    p4 = os.path.join(REPO, "forum/hl_gap_4param_exact_parameters_with_first_occurrence.csv")
    p3 = os.path.join(REPO, "forum/hl_gap_3param_exact_parameters_with_first_occurrence.csv")
    rows = {}
    rows.update(read_table(p4))
    for g, r in read_table(p3).items():
        rows.setdefault(g, r)          # order-4 wins where both exist
    S2 = rows[2]["S_g"]                # 2*C2 (twin constant, model normalization)

    def rho(g):
        r = rows[g]
        e = r["c1"] / L + r["c2"] / L ** 2 + r["c3"] / L ** 3 + r["c4"] / L ** 4
        return r["S_g"] / L ** 2 * math.exp(-e)

    # Gate: sum over all tabulated even g should equal the gap density 1/L.
    gmax = max(rows)
    total_rho = sum(rho(g) for g in sorted(rows) if g % 2 == 0)
    density_ratio = total_rho / (1.0 / L)

    # ---- model curves ---------------------------------------------------
    def n_hl4(m):
        g_min = int(math.ceil(m * L))
        if g_min % 2:
            g_min += 1
        return R * sum(rho(g) for g in rows if g >= g_min and g % 2 == 0)

    def n_hl1(m):
        return S2 * R / L * math.exp(-m)

    grid = [args.m_min - 0.5 + 0.1 * i for i in range(int((args.m_max + 1.0 - (args.m_min - 0.5)) / 0.1) + 1)]
    hl4 = [n_hl4(m) for m in grid]
    hl1 = [n_hl1(m) for m in grid]

    # local slopes (in merit units) at m=20, from the curves themselves
    def slope(f, m=20.0, d=0.25):
        return (math.log(f(m - d)) - math.log(f(m + d))) / (2 * d)

    s_hl1 = slope(n_hl1)
    s_hl4 = slope(n_hl4)

    # weighted fit of the measured points (weight = count, Poisson for ln n)
    pts = [(m, meas[m]) for m in ms if meas[m] >= 4]
    sw = sum(n for _, n in pts)
    sx = sum(m * n for m, n in pts) / sw
    sy = sum(math.log(n) * n for _, n in pts) / sw
    sxx = sum((m - sx) ** 2 * n for m, n in pts) / sw
    sxy = sum((m - sx) * (math.log(n) - sy) * n for m, n in pts) / sw
    b = sxy / sxx                    # d ln N / dm  (negative)
    decay_meas = -b                  # positive decay rate per merit
    a = math.exp(sy - b * sx)
    fit = lambda m: a * math.exp(b * m)

    # ---- report ---------------------------------------------------------
    print("measured: %s  start=%d  length=%.0e  L=%.4f  gaps(m>=%d)=%d"
          % (os.path.basename(args.log), start, R, L, args.m_min, meas[args.m_min]))
    print("gate: sum rho_g / (1/L) = %.4f  (over g=2..%d)" % (density_ratio, gmax))
    print("decay per merit (dlnN/dm):  measured fit %.4f   HL-4p %.4f   HL-1t %.4f"
          % (decay_meas, s_hl4, s_hl1))
    print("%4s %10s %12s %12s %8s %8s" % ("m", "measured", "HL-4p", "HL-1t", "me/4p", "me/1t"))
    for m in ms:
        print("%4d %10d %12.1f %12.0f %8.2f %8.2f"
              % (m, meas[m], n_hl4(m), n_hl1(m),
                 meas[m] / n_hl4(m) if n_hl4(m) > 0 else float("nan"),
                 meas[m] / n_hl1(m) if n_hl1(m) > 0 else float("nan")))

    # ---- plot -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(dark=False):
        if dark:
            plt.rcParams.update({
                "figure.facecolor": "#101014", "axes.facecolor": "#101014",
                "axes.edgecolor": "#888", "text.color": "#e8e8e8",
                "axes.labelcolor": "#e8e8e8", "xtick.color": "#c8c8c8",
                "ytick.color": "#c8c8c8", "grid.color": "#2a2a30",
                "savefig.facecolor": "#101014"})
        fg = "#e8e8e8" if dark else "#111111"
        fig = plt.figure(figsize=(12.6, 9.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.0], hspace=0.10,
                              left=0.09, right=0.98, top=0.93, bottom=0.235)
        ax = fig.add_subplot(gs[0])
        axr = fig.add_subplot(gs[1])

        # measured points with Poisson error bars
        xs = [m for m in ms if meas[m] > 0]
        ys = [meas[m] for m in xs]
        el = [min(math.sqrt(n), n * 0.9) for n in ys]
        eh = [math.sqrt(n) for n in ys]
        ax.errorbar(xs, ys, yerr=[el, eh], fmt="o", ms=6.5, lw=1.6,
                    color=fg, capsize=3, zorder=5,
                    label="measured (Phase-0, %.1e range)" % R)

        ax.plot(grid, hl4, "-", color="#d62728", lw=2.2, label="HL-4p (P. Williams model)")
        ax.plot(grid, hl1, "-", color="#1f77b4", lw=2.2, label="HL-1t (2C2 e^-m R/L)")
        ax.plot([m for m in grid if args.m_min <= m <= 24], 
                [fit(m) for m in grid if args.m_min <= m <= 24],
                "--", color="#2ca02c", lw=1.6,
                label="measured fit: decay %.3f/merit (m=%d..24)" % (decay_meas, args.m_min))

        # zero events at the top bin -> 95% upper limit arrow (the old 1e14
        # campaign hit this at m>=27; the 2e15 campaign at m>=31)
        if args.m_max in meas and meas[args.m_max] == 0:
            ax.annotate("0 events at m>=%d (95%% UL ~3.0)" % args.m_max,
                        xy=(args.m_max, 3.0), xytext=(args.m_max - 2.6, 0.045),
                        arrowprops=dict(arrowstyle="->", color=fg, lw=1.2),
                        fontsize=9, color=fg)

        ax.set_yscale("log")
        ax.set_xlim(args.m_min - 0.6, args.m_max + 0.4)
        ylo = 0.01
        yhi = 4 * meas[args.m_min]
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=0.25)
        ax.set_ylabel("gaps with merit >= m  (per %.0e range)" % R)
        ax.set_title("Phase-0 scan at 2e20 (L=%.3f): measured merit tail vs Hardy-Littlewood"
                     % L, fontsize=12.5, color=fg)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
        txt = ("decay per merit (dlnN/dm):  measured %.3f   HL-4p %.3f   HL-1t %.3f\n"
               "at m=15:  measured/HL-4p = %.2fx   measured/HL-1t = %.2fx\n"
               "gate: sum rho_g = %.3f x (1/L)   [should be ~1]"
               % (decay_meas, s_hl4, s_hl1, meas[args.m_min] / n_hl4(args.m_min),
                  meas[args.m_min] / n_hl1(args.m_min), density_ratio))
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=9.5,
                va="bottom", color=fg, family="monospace")

        # ratio panel
        rm4 = [meas[m] / n_hl4(m) for m in xs]
        rm1 = [meas[m] / n_hl1(m) for m in xs]
        rel = [math.sqrt(n) / n for n in ys]
        axr.errorbar(xs, rm4, yerr=[r * e for r, e in zip(rm4, rel)], fmt="s", ms=5,
                     lw=1.4, color="#d62728", capsize=3, label="measured / HL-4p")
        axr.errorbar([m + 0.12 for m in xs], rm1, yerr=[r * e for r, e in zip(rm1, rel)],
                     fmt="^", ms=5, lw=1.4, color="#1f77b4", capsize=3,
                     label="measured / HL-1t")
        axr.axhline(1.0, color=fg, lw=1.0, alpha=0.6)
        axr.set_yscale("log")
        axr.set_xlim(args.m_min - 0.6, args.m_max + 0.4)
        axr.grid(alpha=0.25)
        axr.set_xlabel("merit m")
        axr.set_ylabel("ratio to model")
        axr.legend(loc="center right", fontsize=9)
        axr.text(0.02, 0.08, "above 1 = model under-predicts nature; below 1 = model over-predicts",
                 transform=axr.transAxes, fontsize=8.5, color=fg, alpha=0.85)

        explainer = (
"WHY: the Phase-0 scan is the UNBIASED (no covering) natural tail at L=ln(2e20)=46.744, so it is the control\n"
"     experiment for every covered hunt.  Two Hardy-Littlewood baselines are overlaid.\n"
"HL-1t: N(m) = 2*C2 * R/L * exp(-m)  (one-term HL/Cramer; the '1.32 e^-m/lnN' curve used elsewhere).\n"
"HL-4p: N(m) = R * SUM_{g even >= m*L} (S_g/L^2) * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4));  coefficients:\n"
"       P. Williams, forum/hl_gap_4param_exact_parameters...csv.  Our gaps (g <= ~1400) sit INSIDE the table\n"
"       (g <= 3600), so rows are used exactly; the g>9990 tail is < 1e-30 at this L.\n"
"PROVENANCE: the 4-param coefficients are L-independent (exact inclusion-exclusion + singular series); the author\n"
"       demonstrates histogram reconstructions at e^32, 10^25, 2^92, e^62 (L ~ 32..64) to +0.003..0.03 %.  Our\n"
"       L=46.74 and g range sit inside that band, so this is an interpolation in both axes - the agreement is a\n"
"       genuine independent test of the model against an unbiased corpus.\n"
"GATES: sum_g rho_g must equal the gap density 1/L (one gap per prime) - printed; error bars are Poisson.\n"
"READING: where the measured points sit relative to the curves is the statement 'the natural tail at this\n"
"       scale is thinner/thicker than the model says', with the ratio panel making the size explicit."
        )
        fig.text(0.5, 0.015, explainer, fontsize=7.6, family="monospace",
                 color=fg, ha="center", va="bottom")

        out = args.out if not dark else args.out.replace(".png", "_dark.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved: %s" % out)

    draw(dark=False)
    if args.dark:
        draw(dark=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
