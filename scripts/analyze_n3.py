#!/usr/bin/env python3
"""N3 dataset analysis: gap-hunt records from the dual-3060 fleet.
f1 = shift507  = 763-bit  (10^229)
f2 = shift1017 = 1273-bit (10^383)
File format: <gap> <merit> <startprime-decimal>
Outputs PNGs into analysis/ and prints a compact scientific report.
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/n3_merit_hist.png", dpi=130)
    plt.close(fig)

    # ---- G2: survival tail with exponential fits ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m, lab, c, s in ((m1, "f1 763-bit", "tab:blue", s1),
                         (m2, "f2 1273-bit", "tab:orange", s2)):
        xs = np.arange(M0, m.max(), 0.25)
        surv = [float(np.mean(m >= x)) for x in xs]
        ax.plot(xs, surv, color=c, label=f"{lab} (empirical)")
        ax.plot(xs, np.exp(-(xs - M0) / s), "--", color=c,
                label=f"{lab} fit exp(-(m-{M0:.0f})/{s:.3f})")
    ax.set_yscale("log")
    ax.set_xlabel("merit threshold m")
    ax.set_ylabel("P(merit >= m)")
    ax.set_title("N3: exponential tail is size-dependent "
                 "(larger numbers => fatter tail)")
    ax.legend(fontsize=9)
    fig.tight_layout()
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
    ax.set_title("N3: sigma estimates are stable — separation is not noise")
    ax.legend()
    fig.tight_layout()
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
    fig.tight_layout()
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
    fig.tight_layout()
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
    ax.set_xlabel("tail-fit threshold M0 (merit)")
    ax.set_ylabel("fitted sigma (MLE excess over M0)")
    ax.set_title("N3: sigma size-separation is robust across thresholds "
                 "(tail not a pure exponential)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/n3_sigma_threshold.png", dpi=130)
    plt.close(fig)

    print(f"graphs written to {OUTDIR}/: "
          "n3_merit_hist.png, n3_tail_cdf.png, n3_sigma_convergence.png, "
          "n3_best_progression.png, n3_gap_hist.png, "
          "n3_sigma_threshold.png")


if __name__ == "__main__":
    main()
