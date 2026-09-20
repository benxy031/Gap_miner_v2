#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap_hunt_stats.py — live statistics and record-proximity for GAP_HUNT fleets.

For every results file (`<gap> <merit> <startprime>` lines; old 4-field lines
are skipped) it reports:

  * record count, threshold, best merit;
  * logbase estimate from the starts (ln(start));
  * sigma estimates of the tail: mean-excess over the minimum (exponential
    MLE) and a two-quantile rate estimate;
  * record proximity: the closest records to FIRST_KNOWN_OCCURRENCE
    (per-gap comparison against the merits table) and any already-found
    records;
  * P(next reported gap is a record): under the discrete exponential
    model, P(gap size = g) is the tail probability times the per-unit-gap
    interval factor (1 - exp(-1/(L*sigma))).  The headline is the SUM of
    that size density over EVERY recordable table target (each reported
    gap has one size, so per-target events are disjoint); the bare
    exp(-(m_easy - m_min)/sigma) is shown only as the P(merit >= easiest
    target) upper bound.  Targets whose needed merit is already below the
    threshold are excluded from the sum and counted separately.  With
    fewer than 30 records the fitted sigma is noise, so the projection is
    also shown for the prior sigma ~= 1.29 (three independent rate
    anchors from the shift-507 corpus);
  * the easiest recordable targets at this size.

Usage:
    scripts/gap_hunt_stats.py [records-file ...] [--table PATH] [--bin-half X]

With no arguments it globs data/gap_hunt_records*.txt (excluding the
watcher's *_found_* files).  `--bin-half` exists only to reproduce pre-fix
numbers deliberately; leave it alone for real estimates.
"""

import sys
import glob
import math
import os

# Half-width, in GAP UNITS, of the merit bin a table entry occupies.  Gaps
# between odd primes are EVEN, so a size g lives in the merit bin [g-1, g+1)
# (width 2/L) and the next table entry g+2 starts exactly where it ends: the
# entries tile the merit axis.  A half-width of 0.5 leaves half the axis
# uncovered and makes p_record exactly 2x too small.  This is the same bug that
# `record_rate_model.py` carried, found there by a Monte-Carlo on the real table
# and documented in docs/RECORD_RATE_MODEL.md §10; it was still present here, so
# the two tools disagreed by 2x.  Any lattice-bin kernel must be validated
# against a Monte-Carlo on the real table before it is used to score anything.
# `--bin-half 0.5` reproduces the pre-fix (pessimistic) numbers deliberately.
BIN_HALF = 1.0


def load_table(path):
    table = {}
    with open(path, "r") as f:
        for line in f:
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
    return table


def parse_file(path):
    recs = []  # (gap, merit, ln_start)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                gap = int(parts[0])
                merit = float(parts[1])
                ln = math.log(int(parts[2]))
            except ValueError:
                continue
            recs.append((gap, merit, ln))
    return recs


def file_integrity(path):
    """(lines, distinct_starts, duplicate_lines, size_hist).

    A "duplicate" in a report is ambiguous because merit is a FUNCTION of the
    gap size when L = ln(anchor) is constant (measured spread 0.000%): two
    genuinely different gaps of the same size print identically.  Only the
    start prime distinguishes them, so integrity must be checked on the raw
    line.  Locally 2453 sizes repeat in 755202 gaps, i.e. repeated
    (gap, merit) pairs are EXPECTED and are not evidence of a re-emission.
    """
    lines = 0
    starts = set()
    seen = set()
    dup = 0
    hist = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lines += 1
            starts.add(parts[2])
            if line in seen:
                dup += 1
            else:
                seen.add(line)
            try:
                hist[int(parts[0])] = hist.get(int(parts[0]), 0) + 1
            except ValueError:
                pass
    return lines, len(starts), dup, hist


def main():
    args = sys.argv[1:]
    table_path = "data/prime_gap_merits.txt"
    files = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in ("--table", "--bin-half"):
            skip_next = True          # its value is not a file
            continue
        if a.startswith("--"):
            print(f"unknown: {a}", file=sys.stderr)
            return 2
        files.append(a)
    if "--table" in args:
        table_path = args[args.index("--table") + 1]
    if "--bin-half" in args:
        global BIN_HALF
        BIN_HALF = float(args[args.index("--bin-half") + 1])
        print(f"NOTE: bin half-width overridden to {BIN_HALF:g} gap units "
              f"(default 1.0; 0.5 reproduces the pre-2026-09-17 numbers, "
              f"which are 2x pessimistic)", file=sys.stderr)
    if not files:
        files = sorted(
            f for f in glob.glob("data/gap_hunt_records*.txt")
            if "_found_" not in os.path.basename(f))

    table = load_table(table_path)
    if not files:
        print("no results files found", file=sys.stderr)
        return 2
    l_info = []

    for path in files:
        recs = parse_file(path)
        if not recs:
            print(f"== {path}: no new-format records")
            continue
        merits = sorted(m for _, m, _ in recs)
        m_min = merits[0]
        best = merits[-1]
        best_gap = max((g for g, m, _ in recs if m == best), default=0)
        L = sum(ln for _, _, ln in recs) / len(recs)
        l_info.append((path, L))

        # sigma: mean excess over threshold (exponential MLE)
        sig_me = sum(m - m_min for m in merits) / len(merits)
        # sigma: two-quantile rate estimate
        n = len(merits)
        q25 = merits[max(0, n // 4)]
        q75 = merits[min(n - 1, (3 * n) // 4)]
        c25 = sum(1 for m in merits if m >= q25)
        c75 = sum(1 for m in merits if m >= q75)
        sig_q = None
        if q75 > q25 and c75 > 0:
            sig_q = (q75 - q25) / math.log(c25 / c75)

        # record proximity
        deltas = []
        for g, m, _ in recs:
            if g in table:
                d = table[g] - m
                if d < 0:
                    deltas.append((d, g, m, "RECORD!"))
                else:
                    deltas.append((d, g, m, ""))
        deltas.sort()
        found = [d for d in deltas if d[0] < 0]
        near = [d for d in deltas if d[0] >= 0][:5]

        # P(next reported gap is a record) = sum over all recordable
        # targets of P(gap size = g | merit >= threshold).  For the
        # discrete exponential model, merit m maps to gap g = m*L, so the
        # size density is the tail probability times the per-unit-gap
        # interval factor (1 - exp(-1/(L*sigma))) — summing TAIL
        # probabilities directly would overcount (each reported gap has ONE
        # size) and can exceed 1.  Targets whose needed merit is already
        # below the threshold are "free records" (not summable).
        tgt = sorted(
            ((g / L, g, bm) for g, bm in table.items()
             if 8000 <= g <= 70000 and g / bm >= L))
        m_easy = tgt[0][0] if tgt else None
        below = [t for t in tgt if t[0] <= m_min]   # free records, not summable
        tgt_above = [t for t in tgt if t[0] > m_min]
        small = n < 30
        sig_fit = sig_me if sig_me > 0 else 1.29
        p_fit = math.exp(-(m_easy - m_min) / sig_fit) if m_easy else 0.0
        unit = (1.0 - math.exp(-(2.0 * BIN_HALF) / (sig_fit * L))
                if L > 0 and sig_fit > 0 else 0.0)
        p_sum_fit = (unit * sum(math.exp(-(m_t - m_min) / sig_fit)
                                for m_t, _, _ in tgt_above)
                     if m_easy else 0.0)
        p_prior = math.exp(-(m_easy - m_min) / 1.29) if m_easy else 0.0
        unit_p = (1.0 - math.exp(-(2.0 * BIN_HALF) / (1.29 * L))
                  if L > 0 else 0.0)
        p_sum_prior = (unit_p * sum(math.exp(-(m_t - m_min) / 1.29)
                                    for m_t, _, _ in tgt_above)
                       if m_easy else 0.0)

        print(f"== {path}")
        print(f"   records={n} threshold={m_min:.4f} best={best:.6f} "
              f"(gap {best_gap}) logbase~{L:.1f}")
        lines, dstarts, dup, size_hist = file_integrity(path)
        print(f"   integrity: lines={lines} distinct_starts={dstarts} "
              f"duplicate_lines={dup}"
              + ("  [OK: no line appears twice]" if dup == 0 else
                 f"  [WARNING: {dup} identical line(s) -> re-emitted gap]"))
        if dstarts < lines and dup == 0:
            print(f"   NOTE: {lines - dstarts} repeated start(s) that are not"
                  f" identical lines -> same size, different gaps (normal;"
                  f" merit is a function of the gap size at constant L)")
        # sizes carried by more than one gap are printed with an xN suffix so
        # the lists below can never be misread as duplicates
        def szn(g):
            c = size_hist.get(g, 1)
            return f" x{c}" if c > 1 else ""
        print(f"   sigma: mean-excess={sig_me:.3f}"
              + (f"  quantile={sig_q:.3f}" if sig_q else "")
              + ("  [n<30: NOISY — prior 1.29 used below]" if small else ""))
        if found:
            print(f"   *** {len(found)} RECORD(S) ALREADY FOUND ***")
            for d, g, m, _ in found:
                print(f"      gap={g}{szn(g)} merit={m:.6f} delta={d:.6f}")
        print("   closest to a record:")
        for d, g, m, tag in near:
            print(f"      gap={g}{szn(g)} merit={m:.6f} "
                  f"needed={table[g]:.6f} delta={d:.6f}")
        print(f"   P(next reported gap is a record): easiest target "
              f"merit={m_easy:.3f} (gap {tgt[0][1]})")
        if small:
            print(f"      SUM over {len(tgt_above)} recordable targets "
                  f"(size density): fitted sigma {sig_fit:.2f} -> "
                  f"{p_sum_fit:.3e} -> expected reported gaps "
                  f"~{1.0 / p_sum_fit:.0f} (unreliable, n<30)"
                  if p_sum_fit > 0 else
                  "      (no summable targets)")
            print(f"      prior sigma 1.29 -> sum {p_sum_prior:.3e} -> "
                  f"expected reported gaps ~{1.0 / p_sum_prior:.0f}")
        else:
            print(f"      SUM over {len(tgt_above)} recordable targets "
                  f"(size density): fitted sigma {sig_fit:.2f} -> "
                  f"{p_sum_fit:.3e} -> expected reported gaps "
                  f"~{1.0 / p_sum_fit:.0f}")
        print(f"      (P(merit >= easiest target) = {p_fit:.3e} — tail "
              f"upper bound, not size-matched)")
        if below:
            print(f"      NOTE: {len(below)} recordable target(s) already "
                  f"below the threshold — any such exact gap is a free record")
        print("   easiest recordable targets (needed merit, gap):")
        for m, g, bm in tgt[:8]:
            print(f"      {m:.3f}  {g}  (table {bm:.4f})")
        print()

    # Cross-file guard (AFTER the loop): the default glob is
    # data/gap_hunt_records*.txt, which can mix DIFFERENT shifts/covers
    # (locally f1 = shift507 L=528.178 while f2 = shift450 L=488.669).  A report
    # that silently merges them invites wrong conclusions, so say it out loud.
    if len(l_info) > 1:
        lo = min(x[1] for x in l_info)
        hi = max(x[1] for x in l_info)
        if lo > 0.0 and (hi - lo) / hi > 0.001:
            print("\nWARNING: this file set mixes different size scales "
                  "(L = ln anchor):")
            for p, l in l_info:
                print(f"          {os.path.basename(p):38} L={l:.3f}"
                      f"  shift~{l / 0.693147 - 255:.0f}")
            print("         Different configurations: compare them per file, "
                  "never as one pooled sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
