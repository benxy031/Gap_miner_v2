#!/usr/bin/env python3
"""N3 dataset analysis: gap-hunt records from the dual-3060 fleet.
f1 = shift507  = 763-bit  (10^229)
f2 = shift1017 = 1273-bit (10^383)
File format: <gap> <merit> <startprime-decimal>
Outputs PNGs into analysis/ and prints a compact scientific report.

WHAT THE TWO CORPORA DO *NOT* PROVE
------------------------------------
The two files differ in size AND in cover (and, in the tail fit, in how much
of the data sits just above the report threshold).  A difference between them
is therefore a fact about the two WALKS, not about size.  The natural
(Hardy-Littlewood) reference drawn into the figures is what makes that
separation visible: scripts/hl_natural.py reads the coefficient tables in
forum/ and reports a natural sigma of 0.961 (L=528) vs 0.960 (L=882), i.e.
SIZE-FLAT to 0.15 %, while our corpora sit 31-45 % above it and differ from
each other by ~8 %.  Two consequences for the captions below: the natural
baseline explains neither corpus's distance from it nor the gap between them,
and the same-size cover A/B (shifts 998 and 1017) shows a cover effect that
CHANGES SIGN (-2.5 % and +5.6 %), so the cover is not a settled explanation
either.  The open attribution is recorded in docs/RECORD_RATE_MODEL.md.
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import hl_natural
    NATURAL = hl_natural.load()
except ImportError:                      # pragma: no cover - fleet boxes
    NATURAL = None

F1 = "gap_hunt_records_f1.txt"
F2 = "gap_hunt_records_f2.txt"
OUTDIR = "analysis"
M0 = 8.0   # primary tail-fit threshold (= report cutoff, max samples)
THRESH = [8.0, 9.0, 10.0, 11.0, 12.0]  # robustness sweep


def load(path):
    gaps, merits = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                g = float(parts[0])
                m = float(parts[1])
            except ValueError:
                continue
            if g <= 0 or m <= 0:
                continue
            gaps.append(g)
            merits.append(m)
    return np.asarray(gaps), np.asarray(merits)


def tail_fit(merits, m0):
    tail = merits[merits >= m0]
    n = len(tail)
    if n == 0:
        return 0.0, 0.0, 0
    excess = tail - m0
    sigma = float(np.mean(excess))
    se = sigma / np.sqrt(n)
    return sigma, se, n


def corpus_L(gaps, merits):
    """L = ln x of a corpus: mean(gap/merit), constant for a fixed size."""
    m = merits[merits > 0]
    g = gaps[merits > 0]
    if len(m) == 0:
        return 0.0
    return float(np.mean(g / m))


def local_sigma_bands(merits, lo=8, hi=None):
    """Depth-resolved sigma per unit merit band, with Poisson errors.

    sigma = (m2-m1)/ln(N1/N2) from the CCDF ratio of the band edges; this is
    the estimator that exposes a drifting slope, which one number per file
    hides.  Returns [(band_lo, sigma, err, n1, n2), ...].
    """
    if hi is None:
        hi = int(np.ceil(float(merits.max()))) + 1
    out = []
    for a in range(lo, hi):
        n1 = int(np.sum(merits >= a))
        n2 = int(np.sum(merits >= a + 1))
        if n1 < 2 or n2 < 1 or n1 <= n2:
            out.append((a, None, None, n1, n2))
            continue
        d = np.log(n1 / n2)
        s = 1.0 / d
        out.append((a, s, s * np.sqrt(1.0 / n1 + 1.0 / n2) / d, n1, n2))
    return out


def caption(fig, text):
    """English explanation under each graph, with room reserved for it.

    The text is wrapped by matplotlib, so the caller's tight_layout cannot know
    how tall it will be: estimate the wrapped line count (about 105 characters
    per line at fontsize 8.5 on a 10-inch figure) and take that much bottom
    margin.  Without this, a caption longer than ~3 lines collides with the
    x-axis label.
    """
    lines = max(1, int(len(text) / 105) + 1)
    fig.subplots_adjust(bottom=min(0.32, 0.055 + 0.030 * lines))
    fig.text(0.02, 0.012, text, fontsize=8.5, style="italic",
             color="dimgray", wrap=True)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"loading {F1} ...", flush=True)
    g1, m1 = load(F1)
    print(f"loading {F2} ...", flush=True)
    g2, m2 = load(F2)

    s1, se1, n1t = tail_fit(m1, M0)
    s2, se2, n2t = tail_fit(m2, M0)
    z = (s2 - s1) / np.sqrt(se1**2 + se2**2)

    print("sigma threshold robustness:")
    for m0 in THRESH:
        a1, e1, _ = tail_fit(m1, m0)
        a2, e2, _ = tail_fit(m2, m0)
        zz = (a2 - a1) / np.sqrt(e1**2 + e2**2)
        print(f"  M0={m0:4.1f}  sigma1={a1:.4f}+-{e1:.4f}  "
              f"sigma2={a2:.4f}+-{e2:.4f}  sep={zz:.1f} sigma")

    b1 = float(m1.max())
    b2 = float(m2.max())

    lab = f"tail n (m>={M0:.0f})"
    print(f"\n{'':24} {'f1 (763-bit)':>14} {'f2 (1273-bit)':>14}")
    print(f"{'records':24} {len(m1):14d} {len(m2):14d}")
    print(f"{'min merit':24} {m1.min():14.3f} {m2.min():14.3f}")
    print(f"{'mean merit':24} {m1.mean():14.3f} {m2.mean():14.3f}")
    print(f"{'median merit':24} {np.median(m1):14.3f} {np.median(m2):14.3f}")
    print(f"{'max merit':24} {b1:14.3f} {b2:14.3f}")
    print(f"{'max gap':24} {g1.max():14.0f} {g2.max():14.0f}")
    print(f"{lab:24} {n1t:14d} {n2t:14d}")
    print(f"{'tail sigma':24} {s1:14.4f} {s2:14.4f}")
    print(f"{'sigma SE':24} {se1:14.4f} {se2:14.4f}")
    print(f"\nsize separation: sigma(f2)-sigma(f1) = {s2-s1:+.4f} "
          f"= {z:.1f} sigma  (p ~ {10**(-z*z/2/2.3026):.1e})\n")

    L1 = corpus_L(g1, m1)
    L2 = corpus_L(g2, m2)
    if NATURAL is not None:
        print("HL natural baseline (forum/ coefficient tables; the natural "
              "tail is essentially size-flat):")
        for lab, L, s_eff in (("f1", L1, s1), ("f2", L2, s2)):
            print("  %s: %s" % (lab, NATURAL.describe(L, s_eff)))
        nat = 100.0 * (NATURAL.sigma(L2) / NATURAL.sigma(L1) - 1.0)
        print("  -> natural predicts %+.2f %% between the two sizes; the "
              "corpora differ by %+.2f %%." % (nat, 100.0 * (s2 / s1 - 1.0)))
        print("  -> size alone does NOT explain it, and the same-size cover "
              "A/B changes sign (-2.5 % at shift 998, +5.6 % at 1017): "
              "the cause is still open (docs/RECORD_RATE_MODEL.md).")
    else:
        print("(no HL baseline: forum/ coefficient CSVs not found)")

    # ---- G1: merit histogram (log y) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(8.0, max(b1, b2) + 0.4, 0.2)
    ax.hist(m1, bins=bins, alpha=0.55, label=f"f1 763-bit (n={len(m1)})",
            color="tab:blue")
    ax.hist(m2, bins=bins, alpha=0.55, label=f"f2 1273-bit (n={len(m2)})",
            color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("merit = gap / ln(start)")
    ax.set_ylabel("count per 0.2-merit bin")
    ax.set_title(f"N3: gap merit distribution, two sizes (tail fits: "
                 f"$\\sigma_1$={s1:.3f}, $\\sigma_2$={s2:.3f}, {z:.1f}$\\sigma$ apart)")
    ax.legend()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Gap merits from the N3 fleet (dual RTX 3060), report "
                "threshold m>=8. The 1273-bit histogram is shifted right: "
                f"sigma=1.372 vs 1.262 at 763-bit ({z:.1f} sigma on the global "
                "MLE). READ THE ATTRIBUTION: the two corpora differ in size "
                "AND cover (and in the share of data just above the "
                "threshold), so this is a difference between two WALKS. The "
                "natural HL tail is size-flat to 0.15 % over this range, and "
                "the same-size cover A/B changes sign, so neither size nor "
                "cover is established as the cause.")
    fig.savefig(f"{OUTDIR}/n3_merit_hist.png", dpi=130)
    plt.close(fig)

    # ---- G2: survival tail with exponential fits ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, lab, c, s, L in ((m1, "f1 763-bit", "tab:blue", s1, L1),
                            (m2, "f2 1273-bit", "tab:orange", s2, L2)):
        xs = np.arange(M0, m.max(), 0.25)
        surv = [float(np.mean(m >= x)) for x in xs]
        ax.plot(xs, surv, color=c, label=f"{lab} (empirical)")
        ax.plot(xs, np.exp(-(xs - M0) / s), "--", color=c,
                label=f"{lab} fit exp(-(m-{M0:.0f})/{s:.3f})")
        if NATURAL is not None and L > 0:
            ax.plot(xs, NATURAL.survival(L, M0, xs), ":", lw=1.6, color=c,
                    label=f"{lab} HL natural (sigma_nat "
                          f"{NATURAL.sigma(L):.3f}, no cover)")
    ax.set_yscale("log")
    ax.set_xlabel("merit threshold m")
    ax.set_ylabel("P(merit >= m)")
    ax.set_title("N3: two corpora differ, but size is not established as the "
                 "cause\n(dotted = Hardy-Littlewood natural, which is "
                 "size-flat to 0.15 % here)")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Survival probability P(merit >= m), log scale. Dashed = "
                "maximum-likelihood exponential fits; dotted = the natural HL "
                "law with NO cover, at each corpus's own size. Reading: both "
                "hunts sit ~1.3-1.5x above the natural decay (the covering's "
                "gain), the two corpora separate over merit 8-14 and CROSS "
                "again near 15-17, and the natural law predicts only 0.15 % "
                "between the two sizes. So the separation is real for these "
                "walks but is NOT attributable to size alone; the same-size "
                "cover A/B changes sign.")
    fig.savefig(f"{OUTDIR}/n3_tail_cdf.png", dpi=130)
    plt.close(fig)

    # ---- G3: sigma convergence (rolling MLE + 95% band) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, lab, c in ((m1, "f1 763-bit", "tab:blue"),
                      (m2, "f2 1273-bit", "tab:orange")):
        mask = m >= M0
        idx = np.arange(1, len(m) + 1)
        cnt = np.cumsum(mask)
        csum = np.cumsum(np.where(mask, m - M0, 0.0))
        sig = np.where(cnt > 0, csum / np.maximum(cnt, 1), np.nan)
        se = np.where(cnt > 0, sig / np.sqrt(np.maximum(cnt, 1)), np.nan)
        ax.plot(idx[mask], sig[mask], color=c, label=f"{lab} rolling sigma")
        ax.fill_between(idx[mask], (sig - 1.96 * se)[mask],
                        (sig + 1.96 * se)[mask], color=c, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("records processed")
    ax.set_ylabel("tail sigma (MLE, m>=10)")
    ax.set_title("N3: sigma estimates are stable — the separation is not "
                 "noise (but is not attributed to size)")
    ax.legend()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Running maximum-likelihood estimate of the tail slope sigma "
                "as records accumulate, with 95% confidence bands. Both curves "
                "converge early and never overlap: the difference between "
                "these two WALKS is not sampling noise. It is also not a "
                "statement about size: the corpora differ in cover as well, "
                "and the natural HL tail is size-flat over this range.")
    fig.savefig(f"{OUTDIR}/n3_sigma_convergence.png", dpi=130)
    plt.close(fig)

    # ---- G4: best-merit progression ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, lab, c in ((m1, "f1 763-bit", "tab:blue"),
                      (m2, "f2 1273-bit", "tab:orange")):
        ax.plot(np.arange(1, len(m) + 1), np.maximum.accumulate(m),
                color=c, label=lab)
    ax.set_xlabel("records processed (chronological)")
    ax.set_ylabel("best merit so far")
    ax.set_title(f"N3: exploration progress (f1 best {b1:.3f}, f2 best {b2:.3f})")
    ax.legend()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Best merit found so far vs records processed (chronological). "
                "Each step is a new personal record. The 1273-bit walker is "
                "currently ahead despite fewer records — consistent with its "
                "fatter tail (graph 2).")
    fig.savefig(f"{OUTDIR}/n3_best_progression.png", dpi=130)
    plt.close(fig)

    # ---- G5: gap-size histogram (log bins) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    gbins = np.geomspace(2e3, max(g1.max(), g2.max()) * 1.05, 30)
    ax.hist(g1, bins=gbins, alpha=0.55, label=f"f1 763-bit", color="tab:blue")
    ax.hist(g2, bins=gbins, alpha=0.55, label=f"f2 1273-bit", color="tab:orange")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gap (log scale)")
    ax.set_ylabel("count per log bin")
    ax.set_title("N3: raw gap sizes (log-binned)")
    ax.legend()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Raw gap lengths on log-log axes. The 1273-bit distribution "
                "extends to larger absolute gaps because gaps scale with "
                "log(start); merit (gap / ln start) in graphs 1-2 is the "
                "size-normalized comparison where the real difference shows.")
    fig.savefig(f"{OUTDIR}/n3_gap_hist.png", dpi=130)
    plt.close(fig)

    # ---- G6: sigma vs fit threshold (robustness) ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    s1s, s2s, zs = [], [], []
    for m0 in THRESH:
        a1, e1, _ = tail_fit(m1, m0)
        a2, e2, _ = tail_fit(m2, m0)
        s1s.append((a1, e1))
        s2s.append((a2, e2))
        zs.append((a2 - a1) / np.sqrt(e1**2 + e2**2))
    s1v = np.array([s[0] for s in s1s]); s2v = np.array([s[0] for s in s2s])
    e1v = np.array([s[1] for s in s1s]); e2v = np.array([s[1] for s in s2s])
    ax.errorbar(THRESH, s1v, yerr=1.96 * e1v, fmt="o-", color="tab:blue",
                label="f1 763-bit")
    ax.errorbar(THRESH, s2v, yerr=1.96 * e2v, fmt="s-", color="tab:orange",
                label="f2 1273-bit")
    for x, zz in zip(THRESH, zs):
        ax.text(x, s2v[list(THRESH).index(x)] + 0.008, f"{zz:.0f}$\\sigma$",
                fontsize=8, ha="center", color="dimgray")
    if NATURAL is not None:
        for L, c, lab in ((L1, "tab:blue", "f1"), (L2, "tab:orange", "f2")):
            if L > 0:
                ax.axhline(NATURAL.sigma(L), ls=":", lw=1.6, color=c,
                           label=f"HL natural {lab} "
                                 f"({NATURAL.sigma(L):.3f}, no cover)")
    ax.set_xlabel("tail-fit threshold M0 (merit)")
    ax.set_ylabel("fitted sigma (MLE excess over M0)")
    ax.set_title("N3: the separation survives every threshold — and so does "
                 "the distance to the natural law\n(neither is attributable "
                 "to size alone)")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "The 1273-bit sigma exceeds the 763-bit sigma by 7.4 to "
                "34.4 sigma depending on the fit threshold, and BOTH sit "
                "30-45 % above the natural HL line (dotted, no cover). Two "
                "honest readings: the offset from nature is the covering's "
                "gain; the gap between the files is a property of these two "
                "walks, because size predicts only 0.15 % and the same-size "
                "cover A/B changes sign.")
    fig.savefig(f"{OUTDIR}/n3_sigma_threshold.png", dpi=130)
    plt.close(fig)

    # ---- G7: band-resolved local sigma (the profile, not one number) ----
    # A single sigma per file is an average over the whole depth range and it
    # moves with the fit threshold; the profile below is what a record-rate
    # extrapolation actually needs, because records live at the deep end.
    fig, ax = plt.subplots(figsize=(10.5, 6))
    profiles = []
    for m, lab, c, s, L in ((m1, "f1 763-bit", "tab:blue", s1, L1),
                            (m2, "f2 1273-bit", "tab:orange", s2, L2)):
        rows = local_sigma_bands(m)
        profiles.append((lab, rows))
        pts = [(a + 0.5, sv, ev) for a, sv, ev, _n1, _n2 in rows
               if sv is not None and ev is not None]
        if pts:
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                        yerr=[p[2] for p in pts], fmt="o-", ms=4, capsize=3,
                        color=c, label=f"{lab} local sigma")
        ax.axhline(s, ls="--", lw=1, color=c, alpha=0.8,
                   label=f"{lab} single sigma {s:.3f}")
        if NATURAL is not None and L > 0:
            ax.axhline(NATURAL.sigma(L), ls=":", lw=1.6, color=c,
                       label=f"{lab} HL natural {NATURAL.sigma(L):.3f}")
    ax.set_ylim(0.0, 2.6)
    ax.set_xlabel("merit band (width 1)")
    ax.set_ylabel("local sigma")
    ax.set_title("N3: sigma is a depth PROFILE, not a number"
                 "\nthe 763-bit tail steepens with depth while the 1273-bit "
                 "tail does not")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    caption(fig, "Local sigma per unit merit band (CCDF ratio of the band "
                "edges, Poisson errors). f1 steepens toward the record depth "
                "(1.29 -> ~1.10 by merit 18-19 -> 0.96 by 19-20), so its "
                "single sigma OVERSTATES deep-tail rates; f2 stays flat/near "
                "the natural line. This is why one sigma per file must not be "
                "extrapolated to the record depth, and why the record-rate "
                "model keeps a stretched/GPD option.")
    fig.savefig(f"{OUTDIR}/n3_bands.png", dpi=130)
    plt.close(fig)

    print(f"graphs written to {OUTDIR}/: "
          "n3_merit_hist.png, n3_tail_cdf.png, n3_sigma_convergence.png, "
          "n3_best_progression.png, n3_gap_hist.png, "
          "n3_sigma_threshold.png, n3_bands.png")


if __name__ == "__main__":
    main()
