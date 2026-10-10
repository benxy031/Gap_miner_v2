#!/usr/bin/env python3
"""p0_walk_hl_compare.py — our 128-bit walk scan vs the Hardy-Littlewood tables.

WHY
---
The walk engine scans a natural (uncovered) range far past the published record
scale and asks two independent questions of P. Williams' (titanV) HL gap tables
in forum/:

  (1) CENSUS — how many gaps of each size SHOULD the range contain?
        rho_g(L) = (S_g / L^2) * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4))
        N(g >= G) = R * SUM_{g >= G} rho_g(L),   R = integers actually scanned
      The coefficients are calibrated on the published record curve (L ~ 30..64);
      our L = 66.54 is ABOVE that band, so this is an out-of-sample test of the
      model AND a closure test of the scanner (we must find what nature owes).

  (2) FIRST OCCURRENCE — where should each length FIRST appear?
      First occurrence = the L at which the expected count of that exact gap
      size reaches 1, i.e. the root of  2 ln L - ln S_g + SUM c_k/L^k - L = 0.
      Plotted against (a) the best published record for the length and (b) our
      new record, it shows how far the whole community still is from the model's
      predicted first occurrence, and what fraction one campaign closes.

The range spans dL/L = 5.7e-13, so the density is effectively constant and one
L (the range midpoint) is used throughout.

Usage:
    scripts/p0_walk_hl_compare.py [--out PNG] [--dark] [--glob PATTERN]
    scripts/p0_walk_hl_compare.py --glob 'data/p0_walk_wall*_g*.log'
"""
import argparse
import glob
import math
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import hl_model as hm                                                   # noqa: E402

# All matched logs must belong to ONE contiguous scan range (they are summed
# into a single R at a single L).  The default is the 7.9e28 wall chain.
GLOBS = [os.path.join(REPO, "data/p0_walk_wall*_g*.log")]
MERITS = os.path.join(REPO, "data/prime_gap_merits.txt")

RE_GAP = re.compile(
    r"^(?P<ts>\d+)\s+(?P<lower>\d+)\s+(?P<gap>\d+)\s+(?P<merit>[\d.]+)\s+"
    r"(?P<upper>\d+)\s+verified=(?P<ver>[01])\s+record=(?P<rec>\w+)(?:\s+table=(?P<tab>[\d.]+))?")
RE_INT = re.compile(r"ints=(\d+)")
RE_INT_SCI = re.compile(r"ints=([\d.]+e[+-]\d+)")


def slice_tag(logpath):
    """data/p0_walk_wall_c3_g1664.log -> ('wall_c3', 1664)."""
    base = os.path.basename(logpath)
    m = re.match(r"p0_walk_(?P<tag>.+)_g(?P<gmin>\d+)\.log$", base)
    return (m.group("tag"), int(m.group("gmin"))) if m else (base, 0)


def coverage(logpath):
    """(ints_covered, how) for one slice: exact if the run finished, else the
    state file offset, else the last monitor line (3 significant digits)."""
    out = logpath[:-4] + ".out"
    tag, _ = slice_tag(logpath)
    length = None
    with open(logpath, errors="ignore") as fh:
        head = fh.readline()
    for tok in head.split():
        if tok.startswith("length="):
            length = int(tok.split("=", 1)[1])
    if length is None:
        raise SystemExit("no length= in the session header of %s" % logpath)

    txt = open(out, errors="ignore").read() if os.path.exists(out) else ""
    if "range complete" in txt:
        return length, "complete"
    state = os.path.join(REPO, "data", "p0_state_%s.txt" % tag)
    if os.path.exists(state):
        with open(state, errors="ignore") as fh:
            for line in fh:
                if line.startswith("off0"):
                    return int(line.split()[1]), "state off0"
    m = RE_INT.findall(txt)
    if m:
        return int(m[-1]), "monitor (3 s.f.)"
    m = RE_INT_SCI.findall(txt)
    if m:
        return int(float(m[-1])), "monitor (3 s.f.)"
    raise SystemExit("cannot determine coverage of %s" % logpath)


def read_slice(logpath):
    """(header, gaps[]) for one slice.  gaps = dicts per logged gap event."""
    start = gmin = None
    with open(logpath, errors="ignore") as fh:
        head = fh.readline()
        for tok in head.split():
            if tok.startswith("start="):
                start = int(tok.split("=", 1)[1])
            elif tok.startswith("gap_min="):
                gmin = int(tok.split("=", 1)[1])
        gaps = []
        for line in fh:
            if line.startswith("#"):
                continue
            m = RE_GAP.match(line.strip())
            if not m:
                continue
            d = m.groupdict()
            gaps.append({
                "lower": int(d["lower"]),
                "gap": int(d["gap"]),
                "merit": float(d["merit"]),
                "record": d["rec"] == "NEW",
                "verified": d["ver"] == "1",
                "table_merit": float(d["tab"]) if d["tab"] else None,
            })
    if start is None:
        raise SystemExit("no start= in the session header of %s" % logpath)
    return {"start": start, "gap_min": gmin}, gaps


def load_published():
    """{gap: (merit, discoverer)} from the live merits snapshot."""
    out = {}
    if not os.path.exists(MERITS):
        return out
    with open(MERITS, errors="ignore") as fh:
        for line in fh:
            p = line.split(None, 2)
            if len(p) < 2:
                continue
            try:
                g = int(p[0])
                m = float(p[1])
            except ValueError:
                continue
            if m > 0:
                out.setdefault(g, (m, p[2].strip() if len(p) > 2 else "?"))
    return out


def density(gap, L, rec):
    s = math.log(rec["S_g"]) - 2.0 * math.log(L)
    for k in (1, 2, 3, 4):
        c = rec.get("c%d" % k)
        if c:
            s -= c / L ** k
    return math.exp(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        REPO, "data/p0_walk_vs_hl.png"))
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--glob", action="append", default=None,
                    help="walk log glob (repeatable); default covers the wall chain")
    a = ap.parse_args(argv)

    pats = a.glob or GLOBS
    logs = []
    for p in pats:
        logs += glob.glob(p)
    logs = sorted(set(logs))
    if not logs:
        raise SystemExit("no walk logs matched %s" % pats)

    slices, gaps, records = [], [], {}
    for log in logs:
        head, gs = read_slice(log)
        cov, how = coverage(log)
        tag, _ = slice_tag(log)
        slices.append({"log": log, "tag": tag, "start": head["start"],
                       "cov": cov, "how": how, "n": len(gs), "gmin": head["gap_min"]})
        gaps += gs
        for g in gs:
            if g["record"] and g["verified"] and g["table_merit"]:
                cur = records.get(g["gap"])
                if cur is None or g["lower"] < cur["lower"]:
                    records[g["gap"]] = g
        print("[slice] %-28s start=%.3e  cov=%.4e (%s)  events=%d  gap_min=%d"
              % (tag, head["start"], cov, how, len(gs), head["gap_min"]))

    R = sum(s["cov"] for s in slices)
    x_lo = min(s["start"] for s in slices)
    x_hi = max(s["start"] + s["cov"] for s in slices)
    L = math.log(0.5 * (x_lo + x_hi))
    tab = hm.table()
    print("[scan] R = %.6e integers,  x = [%.4e, %.4e],  L = %.4f,  %d gap events"
          % (R, x_lo, x_hi, L, len(gaps)))

    # ---- gate: the tabulated rho_g must reproduce the gap density 1/L --------
    all_g = sorted(g for g in tab if g % 2 == 0)
    s_all = sum(density(g, L, tab[g]) for g in all_g)
    print("[gate] sum_g rho_g / (1/L) = %.5f   (over %d tabulated even gaps <= %d)"
          % (s_all * L, len(all_g), all_g[-1]))

    # ---- census -------------------------------------------------------------
    gmax = int(max((g["gap"] for g in gaps), default=2000))
    gs_grid = list(range(1660, min(gmax + 20, 3000) + 2, 2))
    n_meas, n_pred = [], []
    for G in gs_grid:
        n_meas.append(sum(1 for g in gaps if g["gap"] >= G))
        n_pred.append(R * sum(density(g, L, tab[g]) for g in all_g if g >= G))

    # ---- first occurrence ---------------------------------------------------
    pub = load_published()
    fo = {g: math.exp(hm.first_occurrence(g)) for g in all_g
          if hm.first_occurrence(g) is not None}
    # panel C band: the neighbourhood of the records we set (auto-widened as the
    # chain sets records at larger lengths)
    CB_LO = 1660
    CB_HI = max(2060, int(math.ceil((max(records) + 60) / 50.0)) * 50) if records else 2060
    cloud = [(g, math.exp(g / m)) for g, (m, _) in pub.items()
             if CB_LO <= g <= CB_HI]
    cloud = [(g, x) for g, x in cloud if 1e14 < x < 1e40]
    band_ratio = sorted(x / fo[g] for g, x in cloud if g in fo and fo[g])
    band_n_below = sum(1 for r in band_ratio if r < 1)

    rec_rows = []
    for g, ev in sorted(records.items()):
        x_hl = fo.get(g, float("nan"))
        x_prev = math.exp(g / ev["table_merit"])
        x_our = ev["lower"]
        rec_rows.append({
            "gap": g, "x_hl": x_hl, "x_prev": x_prev, "x_our": x_our,
            "gain": x_prev / x_our, "above": x_our / x_hl if x_hl else float("nan"),
            "closed": math.log10(x_prev / x_our),
            "left": math.log10(x_our / x_hl) if x_hl else float("nan"),
            "total": math.log10(x_prev / x_hl) if x_hl else float("nan"),
            "merit_hl": g / math.log(x_hl) if x_hl else float("nan"),
        })
    rec_rows.sort(key=lambda r: r["gap"])

    # panel-C y-window from the data actually plotted (cloud + model + records)
    _ys = [math.log10(x) for _, x in cloud]
    _ys += [math.log10(fo[g]) for g in fo if CB_LO <= g <= CB_HI]
    _ys += [math.log10(r[k]) for r in rec_rows for k in ("x_prev", "x_our")]
    C_YLO = (min(_ys) if _ys else 18.0) - 0.85
    C_YHI = (max(_ys) if _ys else 35.0) + 0.95

    # ---- plot ---------------------------------------------------------------
    import numpy as np
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
        C_HL, C_MEAS, C_OLD, C_NEW = "#1f77b4", "#d62728", "#ff7f0e", "#d62728"

        fig = plt.figure(figsize=(14.4, 11.6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                              hspace=0.30, wspace=0.22,
                              left=0.065, right=0.985, top=0.945, bottom=0.26)
        axA = fig.add_subplot(gs[0, 0])
        axB = fig.add_subplot(gs[1, 0])
        axC = fig.add_subplot(gs[0, 1])
        axD = fig.add_subplot(gs[1, 1])

        # --- A: census -------------------------------------------------------
        meas = np.array(n_meas, dtype=float)
        pred = np.array(n_pred, dtype=float)
        keep = meas >= 1
        axA.errorbar(np.array(gs_grid)[keep], meas[keep],
                     yerr=np.sqrt(meas[keep]), fmt="o", ms=5.5, lw=1.5,
                     mfc=C_MEAS, mec=("#101014" if dark else "white"), mew=0.8,
                     color=C_MEAS, capsize=3, zorder=5,
                     label="measured (walk scan, R=%.2e)" % R)
        axA.plot(gs_grid, pred, "-", color=C_HL, lw=2.2,
                 label="HL 4-param model at L=%.2f" % L)
        axA.fill_between(gs_grid, np.maximum(pred - np.sqrt(np.maximum(pred, 1)), 0.3),
                         pred + np.sqrt(np.maximum(pred, 1)),
                         color=C_HL, alpha=0.13, lw=0, label=r"model $\pm$1$\sigma$ Poisson")
        axA.set_yscale("log")
        axA.set_xlim(1650, gs_grid[-1] + 10)
        axA.set_ylim(0.5, 3 * max(meas[0], 1.0))
        axA.grid(alpha=0.25)
        axA.set_xlabel("gap size G")
        axA.set_ylabel("gaps with size $\\geq$ G   in R = %.2e integers" % R)
        axA.set_title("A.  Census: the scan must find what the model owes\n"
                      "(out-of-sample: tables calibrated at L~30-64, we scan L=%.2f)" % L,
                      fontsize=11.5, color=fg)
        axA.legend(loc="upper right", fontsize=9)

        # --- B: census ratio -------------------------------------------------
        rel = 1.0 / np.sqrt(np.maximum(meas[keep], 1))
        axB.errorbar(np.array(gs_grid)[keep], (meas / pred)[keep], yerr=rel,
                     fmt="o", ms=5, lw=1.4, color=C_MEAS, capsize=3)
        axB.axhline(1.0, color=fg, lw=1.1, alpha=0.65)
        axB.fill_between([1640, gs_grid[-1] + 20], 0.9, 1.1, color=C_HL, alpha=0.14, lw=0)
        axB.set_ylim(0.0, 2.0)
        axB.set_xlim(1650, gs_grid[-1] + 10)
        axB.grid(alpha=0.25)
        axB.set_xlabel("gap size G")
        axB.set_ylabel("measured / model")
        axB.set_title("B.  Ratio (shaded: $\\pm$10%%) — the model and the scanner agree",
                      fontsize=11.5, color=fg)
        axB.text(0.015, 0.06,
                 "gate  $\\sum_g \\rho_g\\,/\\,(1/L)$ = %.4f  (1.0 = one gap per prime)\n"
                 "band  $\\pm$1$\\sigma$ Poisson on the model in panel A"
                 % (s_all * L),
                 transform=axB.transAxes, fontsize=9, va="bottom", color=fg,
                 family="monospace")

        # --- C: first-occurrence map -----------------------------------------
        if cloud:
            axC.scatter([g for g, _ in cloud], np.log10([x for _, x in cloud]),
                        s=15, color="#8a8a8a", alpha=0.5, lw=0,
                        label="best published record per length\n(band %d-%d, n=%d)"
                              % (CB_LO, CB_HI, len(cloud)),
                        zorder=2)
        gg = [g for g in all_g if CB_LO <= g <= CB_HI]
        axC.plot(gg, np.log10([fo[g] for g in gg]), "-", color=C_HL, lw=2.4,
                 label="HL predicted FIRST occurrence", zorder=3)
        axC.scatter([r["gap"] for r in rec_rows],
                    [math.log10(r["x_prev"]) for r in rec_rows],
                    s=52, facecolor="none", edgecolor=C_OLD, lw=1.8, zorder=4,
                    label="published best for our lengths (n=%d)" % len(rec_rows))
        for r in rec_rows:
            axC.annotate("", xy=(r["gap"], math.log10(r["x_our"])),
                         xytext=(r["gap"], math.log10(r["x_prev"])),
                         arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.4,
                                         shrinkA=3, shrinkB=3), zorder=5)
        axC.scatter([r["gap"] for r in rec_rows],
                    [math.log10(r["x_our"]) for r in rec_rows],
                    s=46, marker="s", color=C_NEW, lw=0, zorder=6,
                    label="our records (walk scan, x$\\approx$7.9e28)")
        axC.axhline(L / math.log(10), color="#2ca02c", ls="--", lw=1.4, zorder=1)
        axC.text(CB_LO + 6, L / math.log(10) + 0.35, "our scan scale  x = 7.90e28",
                 fontsize=8.5, color="#2ca02c", va="bottom",
                 bbox=dict(fc=("#101014" if dark else "white"), ec="none", alpha=0.75))
        if 1854 in pub:
            x1854 = math.exp(1854 / pub[1854][0])
            axC.scatter([1854], [math.log10(x1854)], s=85, marker="*",
                        color="#9467bd", lw=0, zorder=7,
                        label="1854 (merit %.2f): below prediction"
                              % pub[1854][0])
            axC.annotate("%d of the %d lengths in this\nband already sit BELOW the\n"
                         "prediction (the 6 smallest\n+ 1854)"
                         % (band_n_below, len(cloud)),
                         xy=(1862, math.log10(x1854)), xytext=(1672, 21.3),
                         fontsize=8.5, color="#9467bd",
                         bbox=dict(fc=("#101014" if dark else "white"), ec="none",
                                   alpha=0.82, pad=1.0),
                         arrowprops=dict(arrowstyle="->", color="#9467bd", lw=1.1))
        axC.set_xlim(CB_LO - 10, CB_HI + 10)
        axC.set_ylim(C_YLO, C_YHI)
        axC.grid(alpha=0.25)
        axC.set_xlabel("gap size")
        axC.set_ylabel("$\\log_{10}$(x)   at which the length is first found")
        if rec_rows:
            _ab = [r["above"] for r in rec_rows]
            _cl = [r["closed"] for r in rec_rows]
            axC.set_title("C.  First-occurrence map: our lengths' records sat "
                          "10^%.1f-10^%.1f above\n"
                          "the predicted first occurrence; we closed %.2f-%.2f decades"
                          % (math.log10(min(_ab)), math.log10(max(_ab)),
                             min(_cl), max(_cl)), fontsize=11.5, color=fg)
        else:
            axC.set_title("C.  First-occurrence map: published records vs the "
                          "predicted first occurrence", fontsize=11.5, color=fg)
        axC.text(0.015, 0.955,
                 "band %d-%d: published x / predicted x\n  min %.1e  median %.1e  max %.1e\n"
                 "green arrows: previous record $\\rightarrow$ ours\n(gain per length in panel D)"
                 % (CB_LO, CB_HI, band_ratio[0] if band_ratio else float("nan"),
                    band_ratio[len(band_ratio) // 2] if band_ratio else float("nan"),
                    band_ratio[-1] if band_ratio else float("nan")),
                 transform=axC.transAxes, fontsize=8.4, va="top", color=fg,
                 family="monospace",
                 bbox=dict(fc=("#101014" if dark else "white"), ec="#888", lw=0.6,
                           alpha=0.8))
        axC.legend(loc="lower right", fontsize=8.4)

        # --- D: decades closed ------------------------------------------------
        mid = np.arange(len(rec_rows))
        axD.barh(mid, [r["closed"] for r in rec_rows], color="#2ca02c",
                 height=0.66, label="closed by this campaign")
        axD.barh(mid, [r["left"] for r in rec_rows],
                 left=[r["closed"] for r in rec_rows], color="#9a9a9a",
                 height=0.66, label="still to the predicted first occurrence")
        for i, r in enumerate(rec_rows):
            axD.text(r["total"] + 0.12, i, ("%.2fx" % r["gain"]) if r["gain"] < 2
                     else ("%.1fx" % r["gain"]), va="center", fontsize=8.5, color=fg)
        axD.set_yticks(mid)
        axD.set_yticklabels([str(r["gap"]) for r in rec_rows], fontsize=9)
        axD.set_ylim(-0.7, len(rec_rows) - 0.3)
        axD.set_xlim(0, (max((r["total"] for r in rec_rows), default=1.0)) * 1.14)
        axD.grid(axis="x", alpha=0.25)
        axD.set_xlabel("decades of $x$  ($\\log_{10}$)")
        axD.set_ylabel("gap length")
        cl = sum(r["closed"] for r in rec_rows) / max(len(rec_rows), 1)
        tl = sum(r["total"] for r in rec_rows) / max(len(rec_rows), 1)
        axD.set_title("D.  What one campaign closes (green) vs what remains (grey):\n"
                      "mean %.2f of the %.2f decades to the predicted first occurrence (%.0f%%)"
                      % (cl, tl, 100.0 * cl / tl if tl else 0.0),
                      fontsize=11.5, color=fg)

        yrs_lo = min(r["x_hl"] for r in rec_rows) / (2.25e11 * 3.156e7) if rec_rows else 0
        yrs_hi = max(r["x_hl"] for r in rec_rows) / (2.25e11 * 3.156e7) if rec_rows else 0
        txt = (
"WHY: two independent tests of P. Williams' (titanV) Hardy-Littlewood gap tables against the 128-bit walk scan.\n"
"  A/B  CENSUS  rho_g(L) = (S_g/L^2)*exp(-(c1/L+c2/L^2+c3/L^3+c4/L^4));  N(g>=G) = R * SUM_{g>=G} rho_g(L).\n"
"       The tables are calibrated on the published record curve; our L is above that band, so the agreement shown is\n"
"       an out-of-sample test of the model and a closure test of the scanner (it must find what nature owes).\n"
"  C/D  FIRST OCCURRENCE: root of  2 ln L - ln S_g + SUM c_k/L^k - L = 0  (= where the expected count reaches 1).\n"
"       Published record x read from data/prime_gap_merits.txt as x = exp(gap/merit) (merit stored to 4 dp -> ~1e-4 on x).\n"
"       Our x = the lower prime of each verified record=NEW event in the walk logs.\n"
"PROVENANCE: coefficients are the EXACT inclusion-exclusion rows of forum/hl_gap_4param_exact_parameters_with_first_\n"
"       occurrence.csv (g <= 3600, order 4) and forum/hl_gap_3param_...csv (g <= 9990, order 3); order 4 wins where both\n"
"       exist.  The extrapolated B8 table (c1..c8) agrees with these order-4 first occurrences to <0.02%% in ln x.\n"
"GATES: sum_g rho_g must equal the gap density 1/L (one gap per prime) - printed in panel B; error bars are Poisson.\n"
"READING: A/B say 'the scan behaves exactly as nature's law requires'.  C/D say 'for these lengths the published record\n"
"       was still 10^6.8-10^7.5 above where the length should first appear, and one campaign closes only a thin sliver of\n"
"       that distance'.  The floor of the grey cloud (the best-covered lengths) is itself 10^4-10^5 above the curve and\n"
"       only %d of the %d lengths in the band have been pushed below it - so these records are still common objects at\n"
"       our x (the model expects ~1e12 of size 1864 below 7.9e28), and the remaining distance cannot be scanned away:\n"
"       covering [0, x_hl] for these lengths would take %.0f-%.0f years at our rate (2.25e11 ints/s).\n"
        ) % (band_n_below, len(cloud), yrs_lo, yrs_hi)
        fig.text(0.5, 0.012, txt, fontsize=7.5, family="monospace", color=fg,
                 ha="center", va="bottom")
        out = a.out if not dark else a.out.replace(".png", "_dark.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("saved: %s" % out)

    print("\n %-6s %-10s %-10s %-10s %-8s %-9s %-8s %-8s" %
          ("gap", "x_hl", "x_prev", "x_ours", "gain", "ours/x_hl", "closed", "to go"))
    for r in rec_rows:
        print(" %-6d %.3e %.3e %.3e %7.2fx %8.1e %8.2f %8.2f"
              % (r["gap"], r["x_hl"], r["x_prev"], r["x_our"], r["gain"],
                 r["above"], r["closed"], r["left"]))
    if rec_rows:
        print(" mean: %.2f of %.2f decades closed (%.0f%%), ours/x_hl median %.1e"
              % (sum(r["closed"] for r in rec_rows) / len(rec_rows),
                 sum(r["total"] for r in rec_rows) / len(rec_rows),
                 100 * sum(r["closed"] for r in rec_rows) / sum(r["total"] for r in rec_rows),
                 float(np.median([r["above"] for r in rec_rows]))))

    draw(dark=False)
    if a.dark:
        draw(dark=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
