#!/usr/bin/env python3
"""tail_compare.py — compare gap-tail sigma between two walker output files.

The same-size / different-cover discriminating test: both files should be
from walkers at the SAME size (shift) with DIFFERENT covers (e.g. strong
vs lex).  If their tail sigmas differ significantly, the cover shapes the
tail (a new hunt-quality lever); if they agree, the tail is size-driven.

Usage:
    tail_compare.py [FILE_A FILE_B] [M0] [--plot [PREFIX]]

  --plot's PREFIX is optional (default `tail_compare`); the token after
  `--plot` is used as the prefix ONLY when it resolves to no existing file, so
  `--plot f1.txt f2.txt` cannot silently eat f1.txt as the output prefix.

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
  --plot     also write three figures (numpy/matplotlib required, skipped
             gracefully if unavailable):
               PREFIX_cdf.png    empirical vs exp-fit survival at M0
               PREFIX_sigma.png  tail sigma vs threshold over the sweep
               PREFIX_panels.png the 8-panel diagnostic set that
                                 records_report.py draws for miner logs,
                                 adapted to hunt logs
             Hunt-specific caveats, printed on the figure: these logs carry NO
             timestamp (so order-based panels use the candidate INDEX, drawn as
             a fraction of the file) and NO denominator (only the finds are
             written, so a finds-per-hour panel is impossible -- the merit
             sequence replaces it, and a STEP in its running minimum is what a
             mid-file --gap-hunt-min-merit change looks like).
             A depth-resolved local-sigma table (per merit band, with Poisson
             errors) is printed whether or not --plot is given.

Runs without numpy/matplotlib for the text part (fleet boxes).
"""
import sys
import math
import os
import bisect


def resolve(path):
    """Fall back to data/<path> so fleet boxes (results in data/) work."""
    if os.path.exists(path):
        return path
    alt = os.path.join("data", os.path.basename(path))
    return alt if os.path.exists(alt) else path


def exists_as_data(path):
    """True when `path` resolves to a real file (directly or under data/).

    The same two layouts `resolve()` accepts: the dev box keeps the corpora in
    the repo root, the fleet keeps them in data/.  Used only to decide whether
    the token after `--plot` is a prefix or a positional file.
    """
    return os.path.exists(resolve(path))


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


def load_full(path):
    """Like load(), but also keeps the gap and the DISCOVERY ORDER.

    Hunt logs are `<gap> <merit> <startprime>` with NO timestamp, so file
    order (the order the walker emitted the records) is the only time-like
    axis a hunt log has.  Callers subsample for scatter panels rather than
    shrinking this list: 755k-line corpora are normal.
    """
    gaps, merits = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            try:
                g = int(parts[0])
                m = float(parts[1])
            except ValueError:
                continue
            gaps.append(g)
            merits.append(m)
    return gaps, merits


def load_table(path):
    """Record table (`<gap> <merit> <name>`) -> {gap: merit}, or None."""
    table = {}
    try:
        fh = open(path, "r")
    except OSError:
        return None
    with fh:
        for line in fh:
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
    return table or None


def local_sigma_bands(sorted_merits, edges):
    """Depth-resolved local sigma from the CCDF ratio of the band edges.

    P(merit >= m) ~ exp(-m/sigma) => sigma = (m2-m1) / ln(N1/N2) with
    N = count(merit >= edge); Poisson errors give
    err = sigma * sqrt(1/N1 + 1/N2) / ln(N1/N2).  This is the measurement
    that exposes the DEPTH DEPENDENCE a single fit hides: on shift 507 the
    local sigma falls 1.29 (merit 10-12) -> 1.03 (18-20) -> 0.75 (20-21),
    which is why a threshold sigma must never be extrapolated to depth.
    `sorted_merits` must be sorted ascending (bisect assumes it).
    """
    rows = []
    n = len(sorted_merits)
    for m1, m2 in zip(edges, edges[1:]):
        n1 = n - bisect.bisect_left(sorted_merits, m1)
        n2 = n - bisect.bisect_left(sorted_merits, m2)
        if n1 < 2 or n2 < 1 or n1 == n2:
            rows.append((m1, m2, n1, n2, None, None))
            continue
        d = math.log(n1 / n2)
        s = (m2 - m1) / d
        rows.append((m1, m2, n1, n2, s,
                     s * math.sqrt(1.0 / n1 + 1.0 / n2) / d))
    return rows


def load_cover_targets(dirpath="data/crt/m23", pattern="*covermax_m40*.txt"):
    """Certified cover design points as (L, gap_target, label).

    A cover file states its own geometry: `shift S` (so L = (256+S)*ln 2) and
    `gap_target G` (the gap length the covering was built for, whose merit at
    that L is the design merit G/L).  Used to mark the aiming points on the
    frontier figure; no matching files simply yields no points.
    """
    import glob
    out = []
    for path in sorted(glob.glob(os.path.join(dirpath, pattern))):
        L = g = None
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    if parts[0] == "shift":
                        L = (256.0 + int(parts[1])) * math.log(2.0)
                    elif parts[0] == "gap_target":
                        g = int(parts[1])
                    if L and g:
                        break
        except (OSError, ValueError):
            continue
        if L and g:
            out.append((L, g, os.path.basename(path)))
    return out


# Record-merit grid for the sigma table: 16, 18, 19, 20, ..., 35.
SWEEP = [16.0] + [float(t) for t in range(18, 36)]
MIN_N = 5    # fewer records per file than this -> no meaningful exp-fit
MIN_N_DEEP = 100   # depth needed for the second (record-relevant) verdict


def main():
    args = sys.argv[1:]
    plot = None
    if "--plot" in args:
        i = args.index("--plot")
        plot = "tail_compare"
        # Only treat the next token as a PREFIX when it cannot be a file:
        # `--plot f1.txt f2.txt` must not eat f1.txt as the output prefix.
        if i + 1 < len(args) and not args[i + 1].startswith("-") \
                and not exists_as_data(args[i + 1]):
            plot = args[i + 1]
            del args[i:i + 2]
        else:
            del args[i]
    pos = [a for a in args if not a.startswith("--")]
    if len(pos) == 0:
        pos = ["gap_hunt_records_f1.txt", "gap_hunt_records_f2.txt"]
    elif len(pos) == 1:
        pos.append("gap_hunt_records_f2.txt")
    fa, fb = resolve(pos[0]), resolve(pos[1])
    m0_req = float(pos[2]) if len(pos) > 2 else 10.0

    for p in (fa, fb):
        if not os.path.exists(p):
            print(f"file not found: {p}", file=sys.stderr)
            print("usage: tail_compare.py [FILE_A FILE_B] [M0] [--plot [PREFIX]]",
                  file=sys.stderr)
            return 2

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
        ok = n1 >= MIN_N and n2 >= MIN_N and t >= thr_eff - 1e-9
        z = (a2 - a1) / math.sqrt(e1**2 + e2**2) if ok else None
        if ok:
            sweep_rows.append((t, a1, e1, a2, e2, z, n1, n2))
            print(f"{t:5.1f} {a1:8.4f} {e1:8.4f} {a2:8.4f} {e2:8.4f} "
                  f"{z:7.1f}   (nA={n1}, nB={n2})")
        elif t < thr_eff - 1e-9 and n1 >= MIN_N and n2 >= MIN_N:
            # Below the files' own report threshold every row is
            # (threshold - t) + sigma, i.e. an OFFSET, not a sigma: it is also
            # excluded from the sigma figure so a 5.2 cannot be read as one.
            print(f"{t:5.1f} {a1:8.4f} {e1:8.4f} {a2:8.4f} {e2:8.4f} "
                  f"{'--':>7}   (nA={n1}, nB={n2}) below report "
                  f"threshold {thr_eff:.3f}: OFFSET artifact, not a sigma")
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

    # Depth-resolved local sigma over a fixed merit grid.  The sweep above fits
    # ONE exponent per threshold; this shows whether that exponent is stable
    # with depth, and it is not (see local_sigma_bands' docstring).
    sort_a, sort_b = sorted(ma), sorted(mb)
    edges = [float(e) for e in range(
        int(math.floor(min(min(ma), min(mb)))),
        int(math.ceil(max(max(ma), max(mb)))) + 1)]
    rows_a = local_sigma_bands(sort_a, edges)
    rows_b = local_sigma_bands(sort_b, edges)
    print("\nlocal sigma per merit band (from the CCDF ratio of its two edges):")
    print(f"{'band':>11} {'nA>=':>9} {'sigA':>7} {'+/-':>7} "
          f"{'nB>=':>9} {'sigB':>7} {'+/-':>7}")
    for ra, rb in zip(rows_a, rows_b):
        f_s = lambda v: f"{v:.3f}" if v is not None else "--"
        print(f"{ra[0]:5.0f}-{ra[1]:<5.0f} {ra[2]:9d} {f_s(ra[4]):>7} "
              f"{f_s(ra[5]):>7} {rb[2]:9d} {f_s(rb[4]):>7} {f_s(rb[5]):>7}")

    if plot:
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(numpy/matplotlib unavailable — skipping plot)")
            return 0

        # CCDF through a sorted array + binary search: the SAME numbers as the
        # previous `sum(1 for x in m if x >= v) / len(m)` loop, but one
        # O(n log n) sort instead of O(len(xs) * n) per figure — on a 755k-line
        # corpus that loop alone cost tens of seconds per panel.
        def ccdf(sorted_arr, xs):
            return 1.0 - np.searchsorted(sorted_arr, xs, side="left") / len(sorted_arr)

        def subsample(seq, cap=20000):
            """Stride a long series down to <= cap points for scatter panels."""
            n = len(seq)
            if n <= cap:
                return np.asarray(seq), 1
            k = int(math.ceil(n / float(cap)))
            return np.asarray(seq[::k]), k

        srt_a = np.sort(np.asarray(ma, dtype=float))
        srt_b = np.sort(np.asarray(mb, dtype=float))

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for m, srt, lab, c, s in ((ma, srt_a, "A " + fa, "tab:blue", sa),
                                  (mb, srt_b, "B " + fb, "tab:orange", sb)):
            xs = np.arange(m0, max(m) + 0.5, 0.25)
            ax.plot(xs, ccdf(srt, xs), color=c, label=f"{lab} (empirical)")
            ax.plot(xs, np.exp(-(xs - m0) / s), "--",
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

        # ── diagnostic panel set: the same panels the miner log gets in
        # scripts/records_report.py, adapted to hunt corpora.  Two structural
        # differences drive the adaptation:
        #   * a hunt log carries NO timestamp (`<gap> <merit> <start>`), so
        #     every order-based panel uses the CANDIDATE INDEX (file order =
        #     discovery order), and a flat run = the walk was stopped;
        #   * a hunt log carries no DENOMINATOR (only the finds are written),
        #     so a "finds per hour" panel is impossible — the merit sequence
        #     replaces it, whose running MINIMUM exposes a mid-file threshold
        #     change (a step) exactly the way the rate panel does for miners.
        gaps_a, ord_a = load_full(fa)
        gaps_b, ord_b = load_full(fb)
        series = [(os.path.basename(fa), gaps_a, ord_a, sa, "tab:blue"),
                  (os.path.basename(fb), gaps_b, ord_b, sb, "tab:orange")]
        # Both corpora are perfectly steady here, so the cumulative lines
        # coincide exactly: draw the first one thick so both stay visible.
        series_lw = [series[0] + (3.0,), series[1] + (1.2,)]
        fig3, ax3 = plt.subplots(4, 2, figsize=(15, 16))
        ax = ax3.ravel()

        # (0) cumulative finds, normalized per file.
        for name, gs, ms, sg, col, lw in series_lw:
            x = np.arange(1, len(ms) + 1) / float(len(ms))
            ax[0].step(x, x, where="post", color=col, lw=lw,
                       label=f"{name} (n={len(ms)})")
        ax[0].set_title("cumulative finds, normalized per file\n"
                        "x = candidate order as a fraction of the file\n"
                        "straight = steady walk; flat run = pause/restart; "
                        "slope change = new session/cover")
        ax[0].set_xlabel("candidate order (0 = first find, 1 = last)")
        ax[0].set_ylabel("share of that file's finds")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        # (1) merit vs order, with running minimum and mean: the session /
        # threshold-mixture detector (a STEP in the minimum is a different
        # --gap-hunt-min-merit, a drift is load or a cover change).  The axis
        # is the fraction of the file so two corpora of different length
        # cannot be misread as a mid-file session boundary.
        for name, gs, ms, sg, col in series:
            ys, stride = subsample(ms)
            ax[1].scatter(np.arange(0, len(ms), stride)[:len(ys)]
                          / float(len(ms)), ys, s=4, alpha=0.35, color=col,
                          label=f"{name} (every {stride})")
            blk = max(1, len(ms) // 200)
            rmin = [min(sl) for sl in
                    (ms[i:i + blk] for i in range(0, len(ms), blk))]
            rmean = [sum(sl) / float(len(sl)) for sl in
                     (ms[i:i + blk] for i in range(0, len(ms), blk))]
            xs_b = np.arange(0, len(ms), blk)[:len(rmin)] / float(len(ms))
            ax[1].plot(xs_b, rmin, lw=1.6, color=col,
                       label=f"{name} running min")
            ax[1].plot(xs_b, rmean, lw=1.2, ls=":", color=col,
                       label=f"{name} running mean")
        ax[1].set_title("merit vs candidate order\n"
                        "a STEP in the running minimum = the file mixes two "
                        "--gap-hunt-min-merit sessions")
        ax[1].set_xlabel("candidate order (0 = first find, 1 = last)")
        ax[1].set_ylabel("merit")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)

        # (2) merit histogram vs the exponential fitted at that file's OWN
        # report threshold (the smallest merit in it).  One file is drawn as an
        # outline so the two shapes stay readable when they overlap.
        for idx, (name, gs, ms, sg, col) in enumerate(series):
            lo = math.floor(min(ms) * 2.0) / 2.0
            hi = math.ceil(max(ms) * 2.0) / 2.0
            ax[2].hist(ms, bins=np.arange(lo, hi + 0.5, 0.5),
                       histtype="step" if idx == 0 else "stepfilled",
                       alpha=0.5, lw=1.2 if idx == 0 else 1.0, color=col,
                       label=f"{name} (n={len(ms)})")
            own0 = min(ms)
            s_own, _, _ = fit(ms, own0)
            xs = np.arange(lo, hi, 0.05)
            if s_own > 0:
                ax[2].plot(xs, len(ms) * 0.5 / s_own
                           * np.exp(-(xs - own0) / s_own), ls="--", lw=1.5,
                           color=col,
                           label=f"{name} fit sigma={s_own:.3f} (own threshold)")
        ax[2].set_yscale("log")
        ax[2].set_title("merit distribution vs the fitted exponential\n"
                        "(each fit is anchored at that file's own "
                        "report threshold)")
        ax[2].set_xlabel("merit")
        ax[2].set_ylabel("finds (log)")
        ax[2].legend(fontsize=7)
        ax[2].grid(alpha=0.3)

        # (3) tail survival vs the fit — the records_report.py reading, with a
        # hunt-specific addition in the title: a hunt file may legally
        # concatenate several walk sessions, so a bend is not automatically a
        # light tail (cross-check panel 1).
        for name, gs, ms, sg, col in series:
            srt = np.sort(np.asarray(ms, dtype=float))
            own0 = min(ms)
            s_own, _, _ = fit(ms, own0)
            xs = np.arange(own0, max(ms) + 0.2, 0.05)
            ax[3].step(xs, ccdf(srt, xs), where="post", color=col,
                       label=f"{name} (n={len(ms)})")
            if s_own > 0:
                ax[3].plot(xs, np.exp(-(xs - own0) / s_own), "--", color=col,
                           lw=1.2, label=f"{name} fit sigma={s_own:.3f}")
        ax[3].set_yscale("log")
        ax[3].set_title("tail: observed P(merit >= m) vs exponential fit\n"
                        "falling FASTER than its dash line = fewer deep finds "
                        "than the fit predicts:\nlight tail, mid-file "
                        "threshold mix, or missing records")
        ax[3].set_xlabel("merit")
        ax[3].set_ylabel("P(merit >= m)")
        ax[3].legend(fontsize=7)
        ax[3].grid(alpha=0.3)

        # (4) depth-resolved local sigma: the single exponent is a summary,
        # this is the shape (and it is why the summary must not be
        # extrapolated to the record depth).
        for rows, name, col, sg in ((rows_a, os.path.basename(fa),
                                     "tab:blue", sa),
                                    (rows_b, os.path.basename(fb),
                                     "tab:orange", sb)):
            pts = [(r[0], r[4], r[5]) for r in rows if r[4] is not None]
            if not pts:
                continue
            ax[4].errorbar([p[0] + 0.5 for p in pts], [p[1] for p in pts],
                           yerr=[p[2] for p in pts], fmt="o-", ms=4, capsize=3,
                           color=col, label=f"{name} local sigma")
            ax[4].axhline(sg, ls="--", lw=1, color=col,
                          label=f"{name} global fit {sg:.3f}")
        ax[4].set_title("local sigma per merit band\n"
                        "sigma falling with depth = lighter-than-exponential "
                        "tail (do not extrapolate the global fit)")
        ax[4].set_xlabel("merit band (band width 1.0)")
        ax[4].set_ylabel("local sigma")
        ax[4].legend(fontsize=7)
        ax[4].grid(alpha=0.3)

        # (5) record proximity: gap vs merit against the table's requirement
        # (merit = gap/ln(start) is linear at a fixed size, so each file is a
        # straight line and the table envelope is what decides a record).
        table = load_table(resolve("prime_gap_merits.txt"))
        gmin = min(min(gaps_a), min(gaps_b))
        gmax = max(max(gaps_a), max(gaps_b))
        if table:
            pts = sorted((g, v) for g, v in table.items()
                         if gmin * 0.9 <= g <= gmax * 1.1)
            if pts:
                ax[5].plot([p[0] for p in pts], [p[1] for p in pts], lw=1,
                           color="k", alpha=0.5,
                           label="needed for a record (table)")
        for name, gs, ms, sg, col in series:
            ys, stride = subsample(ms)
            ax[5].scatter(np.asarray(gs[::stride])[:len(ys)], ys, s=5,
                          alpha=0.35, color=col,
                          label=f"{name} (every {stride})")
            if table:
                rec = [(g, m) for g, m in zip(gs, ms)
                       if g in table and m > table[g]]
                if rec:
                    ax[5].scatter([p[0] for p in rec], [p[1] for p in rec],
                                  s=90, marker="*", edgecolor="k", zorder=5,
                                  color=col,
                                  label=f"{name}: ABOVE known best ({len(rec)})")
                # The table envelope sits far above every find here, so the
                # zoom keeps the candidates readable and the CLOSEST APPROACH
                # (the number that actually matters) is written on the axes.
                near = [(table[g] - m, g, m) for g, m in zip(gs, ms)
                        if g in table]
                if near:
                    d, gb, mb_ = min(near)
                    ax[5].text(0.02, 0.04 + 0.05 * (name == series[-1][0]),
                               f"{name}: closest {d:+.3f} merit "
                               f"(gap {gb}: {mb_:.3f} vs needed {table[gb]:.3f})",
                               transform=ax[5].transAxes, fontsize=7, color=col)
        ax[5].set_title("gap vs merit against the merit a record needs there\n"
                        "stars = above the table; y zoomed to the finds, "
                        "closest approach printed on the axes")
        ax[5].set_xlabel("gap length")
        ax[5].set_ylabel("merit")
        cand_merits = list(ma) + list(mb)
        ax[5].set_ylim(min(cand_merits) - 1.0, max(cand_merits) + 2.0)
        ax[5].legend(fontsize=7)
        ax[5].grid(alpha=0.3)

        # (6) the A/B comparison itself, laid on top: subtract each file's own
        # report threshold so different shifts/covers are comparable — this is
        # what the sigma verdict measures, drawn.
        for name, gs, ms, sg, col in series:
            srt = np.sort(np.asarray(ms, dtype=float))
            own0 = min(ms)
            xs = np.arange(0.0, max(ms) - own0, 0.05)
            ax[6].plot(xs, ccdf(srt, xs + own0), color=col,
                       label=f"{name} (threshold {own0:.3f}, n={len(ms)})")
        ax[6].set_yscale("log")
        ax[6].set_title("tails superposed: P(merit - own threshold >= x)\n"
                        "the curve that stays higher at large x is the heavier "
                        "tail")
        ax[6].set_xlabel("merit excess over the file's own report threshold")
        ax[6].set_ylabel("P(merit - threshold >= x)")
        ax[6].legend(fontsize=8)
        ax[6].grid(alpha=0.3)

        # (7) the verdict as a curve: CCDF ratio B/A with Poisson errors.
        # >1 = B keeps more of its records at that depth.  Curves heading in
        # opposite directions = a tail CROSSING, the case in which one
        # threshold sigma cannot rank the two files.
        lo = max(min(ma), min(mb))
        hi = min(max(ma), max(mb))
        if hi > lo + 0.2:
            xs = np.arange(lo, hi, 0.1)
            pa = ccdf(srt_a, xs)
            pb = ccdf(srt_b, xs)
            na = np.round(pa * len(ma))
            nb = np.round(pb * len(mb))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where((na > 0) & (nb > 0), pb / np.maximum(pa, 1e-12),
                                 np.nan)
                rel = np.sqrt(1.0 / np.maximum(na, 1) + 1.0 / np.maximum(nb, 1))
            ax[7].errorbar(xs, ratio, yerr=ratio * rel, fmt="-", lw=1,
                           color="tab:purple", ecolor="0.7", elinewidth=1,
                           label="B/A (CCDF ratio)")
            ax[7].axhline(1.0, color="k", lw=1, ls=":")
        else:
            ax[7].axis("off")
        ax[7].set_yscale("log")
        ax[7].set_title("CCDF ratio B/A (Poisson errors)\n"
                        ">1 = B keeps more records at that depth", fontsize=11)
        ax[7].set_xlabel("merit")
        ax[7].set_ylabel("P_B(merit >= m) / P_A(merit >= m)")
        ax[7].legend(fontsize=8)
        ax[7].grid(alpha=0.3)

        fig3.suptitle("gap_hunt corpus diagnostics — "
                      f"A={os.path.basename(fa)}  B={os.path.basename(fb)}\n"
                      "NOTE: hunt logs carry no timestamps, so order-based "
                      "panels use the candidate index (discovery order)",
                      fontsize=12)
        fig3.tight_layout(rect=(0, 0, 1, 0.97))
        fig3.savefig(f"{plot}_panels.png", dpi=120)
        print(f"wrote {plot}_panels.png")

        # ── frontier figure, in the style of the forum FO plots
        # ("FO Image Aug2026.png" / "TotalFO Aug2026.png" in the repo root):
        # bold title + subtitle stating the axes and conventions, the frontier
        # as data (not only as a model line), the observed range distinguished
        # from the extrapolation, an inset zoom, annotated key points and a
        # provenance footer.
        #
        # WHY this is the same object as those plots: the record table IS a
        # first-occurrence frontier.  Each row is `<gap> <merit> <who>`, and
        # merit = gap/ln(x) => ln(x) = gap/merit, so a row fixes the prime
        # scale L = ln x at which that gap length was FIRST reached.  Our
        # corpora sit at a single L each (L = gap/merit per line, constant by
        # construction), so they appear as vertical bands, and the distance
        # from a band to the grey frontier at the same gap is exactly the
        # merit we still have to find for a record.
        if table:
            import hashlib
            tpath = resolve("prime_gap_merits.txt")
            try:
                with open(tpath, "rb") as fh:
                    tsha = hashlib.sha256(fh.read()).hexdigest()[:12]
            except OSError:
                tsha = "unavailable"
            tg = sorted(g for g in table if 1 <= g <= 60000)
            tl = [g / table[g] for g in tg]
            fig4, axf = plt.subplots(figsize=(13, 7))
            axf.scatter(tl, tg, s=3, c="0.6",
                        label=f"record table frontier ({len(tg):,} gaps)")
            L_ours = []
            for name, gs, ms, sg, col in series:
                Ls = sorted(g / m for g, m in zip(gs, ms) if m > 0)
                if not Ls:
                    continue
                Lbar = sum(Ls) / len(Ls)
                L_ours.append(Lbar)
                axf.plot([Lbar, Lbar], [min(gs), max(gs)], lw=2.2, color=col,
                         label=f"{name}: L={Lbar:.1f}, gaps "
                               f"{min(gs)}..{max(gs)}")
                gb = max(gs)
                axf.scatter([Lbar], [gb], s=70, marker="o", color=col,
                            edgecolor="k", zorder=5)
                need = table.get(gb)
                if need is None:      # odd length -> the even lattice
                    need = table.get(gb - 1) if gb % 2 else table.get(gb + 1)
                # The actionable number: the smallest gap length at which a
                # merit we can plausibly reach at THIS L beats the table.  In
                # the figure that is "our band sits LEFT of the frontier at
                # that gap", and it is the size-exact landing condition.
                reach = [g for g in sorted(table)
                         if 1000 <= g <= 60000 and g / Lbar > table[g] + 1.0]
                if need is not None:
                    axf.annotate(f"max gap {gb} (merit {gb/Lbar:.2f}; table "
                                 f"needs {need:.2f})",
                                 xy=(Lbar, gb), xytext=(16, -24),
                                 textcoords="offset points", fontsize=8,
                                 color=col)
                if reach:
                    # `reach` demands a +1.0 merit margin; the margin-free
                    # boundary is the STRICT structural limit: below it the
                    # table's merit exceeds what this L can produce at all, so
                    # no find of that length can ever be a record here.
                    strict = [g for g in sorted(table)
                              if 1000 <= g <= 60000 and g / Lbar > table[g]]
                    print(f"{name}: easiest reachable gap at L={Lbar:.1f} = "
                          f"{reach[0]} (table needs {table[reach[0]]:.3f}, "
                          f"our L gives {reach[0]/Lbar:.3f} merit); strict "
                          f"boundary (any margin) = "
                          f"{strict[0] if strict else 'none'}")
            cover_first = True
            for Lc, gc, labname in load_cover_targets():
                axf.scatter([Lc], [gc], s=170, marker="*", color="crimson",
                            edgecolor="k", zorder=6,
                            label="cover target (certified)" if cover_first
                            else None)
                cover_first = False
                need = table.get(gc)
                if need is None:   # odd target: the even lattice cannot gap
                    need = table.get(gc - 1) if gc % 2 else table.get(gc + 1)
                dsn = gc / Lc
                txt = (f"{labname}: gap {gc}, design merit {dsn:.2f}"
                       + (f"\ntable needs {need:.2f} -> {dsn - need:+.2f}"
                          + (" (credited even length)" if gc % 2 else "")
                          if need else "\n(gap length not in table)"))
                # Keep the top-of-plot target (the 34,769 cover) from running
                # into the title: aim its label down-left, the rest up-right.
                if gc > 30000:
                    axf.annotate(txt, xy=(Lc, gc), xytext=(-12, -40),
                                 textcoords="offset points", fontsize=8,
                                 color="crimson", ha="right")
                else:
                    axf.annotate(txt, xy=(Lc, gc), xytext=(8, 26),
                                 textcoords="offset points", fontsize=8,
                                 color="crimson")
            axf.set_yscale("log")
            axf.set_xlabel("prime scale L = ln(x)   [record rows: L = gap/merit]")
            axf.set_ylabel("gap length")
            axf.set_title("Record-table frontier vs our corpora\n"
                          "grey = the world frontier (first occurrence per gap "
                          "length, as L = gap/merit);\n"
                          "coloured bars = our walkers (one L each, span = "
                          "gap range observed); stars = certified cover targets",
                          fontsize=11)
            axf.legend(fontsize=8, loc="upper left")
            axf.grid(alpha=0.3, which="both")
            # Forum convention ("Observed gap range" inset): the zoomed main
            # axes carry our range and the frontier crossing, the inset keeps
            # the global picture (the frontier starts at L~0 for gap 1 and
            # runs past L=3000 for the longest recorded gaps).
            axf.set_ylim(1500.0, 45000.0)
            axi = axf.inset_axes([0.615, 0.50, 0.36, 0.44])
            axi.scatter(tl, tg, s=1.5, c="0.6")
            for name, gs, ms, sg, col in series:
                Ls = sorted(g / m for g, m in zip(gs, ms) if m > 0)
                if Ls:
                    axi.plot([sum(Ls) / len(Ls)] * 2, [min(gs), max(gs)],
                             lw=1.5, color=col)
            axi.set_yscale("log")
            axi.set_title("full table range", fontsize=8)
            axi.tick_params(labelsize=7)
            axi.grid(alpha=0.25, which="both")
            fig4.text(0.01, 0.015,
                      f"Source: {tpath} sha256[:12]={tsha}; "
                      f"{os.path.basename(fa)} (n={len(ma)}), "
                      f"{os.path.basename(fb)} (n={len(mb)}). "
                      "L per corpus is constant by construction (merit = "
                      "gap/L), so a corpus is a vertical band, not a curve.",
                      fontsize=8, color="0.35")
            fig4.tight_layout(rect=(0, 0.03, 1, 0.93))
            fig4.savefig(f"{plot}_frontier.png", dpi=130)
            print(f"wrote {plot}_frontier.png")
            if L_ours:
                print(f"our prime scales: L = "
                      f"{', '.join(f'{v:.1f}' for v in L_ours)} "
                      f"(table frontier reaches L = {max(tl):.1f})")
            # Did a corpus actually BEAT the table?  The panels mark stars,
            # but the text has to say it: a hit here is the whole point of the
            # run, and it is also what decides whether to walk longer.
            for name, gs, ms, sg, col in series:
                hits, nearest = [], None
                for g, m in zip(gs, ms):
                    shown = g
                    need = table.get(g)
                    if need is None:            # odd length -> even lattice
                        shown = g - 1 if g % 2 else g + 1
                        need = table.get(shown)
                    if need is None:
                        hits.append((float("inf"), g, m, None))
                    elif m > need:
                        hits.append((m - need, g, m, need))
                cand = [(table[gg] - mm, gg, mm) for gg, mm in zip(gs, ms)
                        if gg in table]
                if cand:
                    nearest = min(cand)
                if hits:
                    hits.sort(key=lambda h: -h[0])
                    for d, g, m, need in hits[:3]:
                        if need is None:
                            print(f"{name}: *** gap {g} merit {m:.4f} - length "
                                  f"ABSENT from the table (new length?)")
                        else:
                            print(f"{name}: *** ABOVE TABLE: gap {g} merit "
                                  f"{m:.4f} vs needed {need:.4f} "
                                  f"(margin {d:+.4f}) - verify before "
                                  f"claiming")
                    if len(hits) > 1:
                        print(f"{name}:     ({len(hits)} hit(s) total)")
                elif nearest is not None:
                    print(f"{name}: no records; closest = gap {nearest[1]} "
                          f"(merit {nearest[2]:.4f} vs needed "
                          f"{table[nearest[1]]:.4f}, margin {nearest[0]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
