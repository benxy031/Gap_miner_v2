#!/usr/bin/env python3
"""Visual summary of data/hl_order3_targets.csv (order-3 HL coefficients).

Four panels:
  A  c1 vs gap            (+ secondary axis: delta = g - c1)
  B  ln x at first occurrence vs gap   (root of Y_g(L) = e^L, c4 unknown)
  C  merit at first occurrence vs gap  (gap / ln x)
  D  coefficient ratios c2/c1 and c3/c1 vs gap (log scale)

Usage:
  python3 scripts/plot_hl_order3_targets.py            # -> data/hl_order3_targets.png
  python3 scripts/plot_hl_order3_targets.py --dark     # -> data/hl_order3_targets_dark.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hl_model  # noqa: E402

CSV = ROOT / "data/hl_order3_targets.csv"

# two hunt bands: < 25k (short walks: shift 720/1017 classes) and >= 25k (shift 1784)
COL_LOW, COL_HIGH = "#2E86AB", "#D1495B"
ANN_B = (18084, 18684, 33628, 34860, 40462)
ANN_C = (17924, 21224, 40462)
OFF_B = {18084: (8, 8), 18684: (8, -17), 33628: (10, -18),
         34860: (10, 8), 40462: (-50, -3)}
OFF_C = {17924: (6, -16), 21224: (8, 8), 40462: (-58, -14)}


def load_rows():
    rows = []
    with CSV.open() as fh:
        for r in csv.DictReader(fh):
            g = int(r["gap"])
            rows.append(dict(
                gap=g,
                c1=float(r["c1"]), c2=float(r["c2"]), c3=float(r["c3"]),
                S_g=float(r["S_g"]) if r["S_g"] else None,
            ))
    rows.sort(key=lambda x: x["gap"])
    missing = []
    for r in rows:
        lnx = hl_model.first_occurrence(r["gap"])
        if lnx is None:
            missing.append(r["gap"])
        r["lnx"] = lnx
        r["merit_fo"] = r["gap"] / lnx if lnx else None
    if missing:
        print(f"warning: no first occurrence for {missing}", file=sys.stderr)
    return rows


def style(ax, dark):
    grid = "#ffffff" if dark else "#444444"
    ax.grid(True, which="major", alpha=0.28 if not dark else 0.22,
            lw=0.8, color=grid)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.12 if not dark else 0.10,
            lw=0.5, color=grid)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def color(gap):
    return COL_LOW if gap < 25000 else COL_HIGH


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dark", action="store_true", help="dark theme variant")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [r for r in load_rows() if r["lnx"]]
    g = [r["gap"] for r in rows]
    c1 = [r["c1"] for r in rows]
    delta = [r["gap"] - r["c1"] for r in rows]
    lnx = [r["lnx"] for r in rows]
    merit = [r["merit_fo"] for r in rows]
    r21 = [r["c2"] / r["c1"] for r in rows]
    r31 = [r["c3"] / r["c1"] for r in rows]

    if args.dark:
        plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "figure.facecolor": "#0f0f13" if args.dark else "white",
        "axes.facecolor": "#14141a" if args.dark else "white",
    })
    fg = "#e8e8ee" if args.dark else "#1a1a1a"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.6))
    fig.suptitle("HL 4-parameter model — order-3 targets  (data/hl_order3_targets.csv)",
                 fontsize=15.5, fontweight="bold", color=fg, y=0.975)
    fig.text(0.5, 0.935,
             f"{len(rows)} consecutive gap lengths · coefficients computed 2026-09-27 · "
             "c₁ independently verified (dev 1.17e-8)",
             ha="center", fontsize=10.5, color=fg, alpha=0.75)

    # ---- A: delta = g - c1
    # delta is a PER-GAP quantity and is NOT smooth: neighbouring lengths differ
    # by up to ~2.2 (g=33628: 14.35 vs g=33650: 12.26).  It gets its own single
    # axis -- mixing it with c1 on a twin axis made the per-gap cloud read like
    # the identity "jumping" across the panel.
    ax = axes[0][0]
    style(ax, args.dark)
    for lo in (True, False):
        xs = [gi for gi in g if (gi < 25000) == lo]
        ys = [d for gi, d in zip(g, delta) if (gi < 25000) == lo]
        ax.plot(xs, ys, "o", ms=5.5, color=COL_LOW if lo else COL_HIGH,
                zorder=3,
                label="g < 25k  (720 / 1017 walks)" if lo
                else "g ≥ 25k  (shift-1784 hunt)")
    dmin, gmin = min(zip(delta, g))
    dmax, gmax = max(zip(delta, g))
    ax.annotate(f"δ_max {dmax:.2f}", (gmax, dmax), textcoords="offset points",
                xytext=(-68, -8), fontsize=8.5, color=fg, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.18", ec="none",
                          fc="#14141a" if args.dark else "white", alpha=0.78))
    ax.annotate(f"δ_min {dmin:.2f}", (gmin, dmin), textcoords="offset points",
                xytext=(8, 3), fontsize=8.5, color=fg, alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.18", ec="none",
                          fc="#14141a" if args.dark else "white", alpha=0.78))
    ax.set_ylim(10.8, 15.2)
    ax.set_title("δ = g − c₁   (c₁ tracks the identity to ≤ 0.08 %)")
    ax.set_xlabel("gap length g")
    ax.set_ylabel("δ = g − c₁")
    step = max(abs(delta[i + 1] - delta[i]) for i in range(len(delta) - 1))
    ax.text(0.03, 0.97,
            f"per-gap deficit, NOT smooth:\nmax step {step:.2f} between "
            "neighbouring lengths",
            transform=ax.transAxes, fontsize=8.5, color=fg, alpha=0.7,
            va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    # ---- B: first occurrence
    ax = axes[0][1]
    style(ax, args.dark)
    ax.plot(g, lnx, "-", lw=1.0, color=fg, alpha=0.25, zorder=1)
    for lo in (True, False):
        xs = [gi for gi in g if (gi < 25000) == lo]
        ys = [lx for gi, lx in zip(g, lnx) if (gi < 25000) == lo]
        ax.plot(xs, ys, "o", ms=5.5, color=COL_LOW if lo else COL_HIGH,
                zorder=3, label="g < 25k  (720 / 1017 walks)" if lo
                else "g ≥ 25k  (shift-1784 hunt)")
    for gi, lx in zip(g, lnx):
        if gi in ANN_B:
            ax.annotate(str(gi), (gi, lx), textcoords="offset points",
                        xytext=OFF_B.get(gi, (6, 5)), fontsize=8.5,
                        color=fg, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.18", ec="none",
                                  fc="#14141a" if args.dark else "white",
                                  alpha=0.78))
    ax.set_title("Model first occurrence — ln x, root of Y_g(L) = e^L")
    ax.set_xlabel("gap length g")
    ax.set_ylabel("ln x at first occurrence")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)

    # ---- C: merit at first occurrence
    ax = axes[1][0]
    style(ax, args.dark)
    ax.plot(g, merit, "-", lw=1.0, color=fg, alpha=0.25, zorder=1)
    for lo in (True, False):
        xs = [gi for gi in g if (gi < 25000) == lo]
        ys = [m for gi, m in zip(g, merit) if (gi < 25000) == lo]
        ax.plot(xs, ys, "o", ms=5.5, color=COL_LOW if lo else COL_HIGH, zorder=3)
    for gi, m in zip(g, merit):
        if gi in ANN_C:
            ax.annotate(f"{gi}\n{m:.1f}", (gi, m), textcoords="offset points",
                        xytext=OFF_C.get(gi, (4, 7)), fontsize=8.5,
                        color=fg, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.18", ec="none",
                                  fc="#14141a" if args.dark else "white",
                                  alpha=0.78))
    ax.set_title("Merit at first occurrence  (g / ln x)")
    ax.set_xlabel("gap length g")
    ax.set_ylabel("merit@first-occurrence")
    ax.text(0.03, 0.95,
            "true first occurrences sit at x ≈ 10⁶¹–10⁹¹ —\nhunt records are "
            "records, not true firsts",
            transform=ax.transAxes, fontsize=9, color=fg, alpha=0.7,
            va="top")

    # ---- D: coefficient ratios
    ax = axes[1][1]
    style(ax, args.dark)
    ax.semilogy(g, r21, "-o", ms=5, lw=1.4, color=COL_LOW,
                mfc="white", mew=1.2, label="c₂/c₁")
    ax.semilogy(g, r31, "-s", ms=4.5, lw=1.4, color=COL_HIGH,
                mfc="white", mew=1.2, label="c₃/c₁")
    ax.set_title("Coefficient ratios  (log scale)")
    ax.set_xlabel("gap length g")
    ax.set_ylabel("ratio")
    ax.set_yticks([6, 8, 10, 20, 30, 50])
    ax.set_yticklabels(["6", "8", "10", "20", "30", "50"])
    ax.tick_params(axis="y", which="minor", labelleft=False)
    ax.set_ylim(5.2, 58)
    ax.legend(loc="center left", frameon=False, fontsize=10)
    ax.text(0.97, 0.08,
            f"c₂/c₁: {min(r21):.2f} → {max(r21):.2f}\n"
            f"c₃/c₁: {min(r31):.1f} → {max(r31):.1f}",
            transform=ax.transAxes, fontsize=9, color=fg, alpha=0.75,
            ha="right")

    fig.text(0.5, 0.012,
             "source: data/hl_order3_targets.csv (order 3; c₄ unknown, c₄/L⁴ ≲ 10⁻² at these L) — "
             "Peter Williams HL 4-parameter model; c₁ cross-checked to 1.17e-8",
             ha="center", fontsize=9, color=fg, alpha=0.6)

    # ---- detailed explainer (keeps the figure self-documenting) ----
    fig.text(0.05, 0.155,
             "A — δ = g − c₁: deficit of the leading coefficient below g.\n"
             "      PER-GAP value, not smooth: each c₁ is its own inclusion–\n"
             "      exclusion sum over the divisors of g, so neighbouring\n"
             "      lengths differ by > 2 (δ(33628)=14.35 vs δ(33650)=12.26;\n"
             "      same in Williams' table: δ(3598)=11.82 vs δ(3600)=10.47).\n"
             "      c₁ tracks the identity to ≤ 0.08 % — so panel A shows δ\n"
             "      alone: scatter, no line interpolated between lengths.",
             ha="left", va="top", fontsize=8.2, color=fg, alpha=0.85,
             family="DejaVu Sans Mono")
    fig.text(0.53, 0.155,
             "B — model first occurrence: ln x where the expected count of\n"
             "      g-gaps reaches 1 (root of Y_g(L)=e^L, c₄ ≈ 0).  Hunt targets\n"
             "      sit orders of magnitude below their true first occurrence —\n"
             "      records are 'smallest known', not true firsts.\n"
             "C — merit at that first occurrence (g / ln x).\n"
             "D — leading ratios c₂/c₁ ≈ 5.6–6.0, c₃/c₁ ≈ 47–54; c₄ unknown\n"
             "      at order 3, c₄/L⁴ ≲ 10⁻² — negligible at these L.",
             ha="left", va="top", fontsize=8.2, color=fg, alpha=0.85,
             family="DejaVu Sans Mono")

    fig.tight_layout(rect=(0, 0.175, 1, 0.925))
    out = args.out or (ROOT / "data" / (
        "hl_order3_targets_dark.png" if args.dark else "hl_order3_targets.png"))
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"  gaps={len(rows)}  delta=[{min(delta):.2f},{max(delta):.2f}]  "
          f"lnx=[{min(lnx):.1f},{max(lnx):.1f}]  "
          f"merit@fo=[{min(merit):.1f},{max(merit):.1f}]")


if __name__ == "__main__":
    main()
