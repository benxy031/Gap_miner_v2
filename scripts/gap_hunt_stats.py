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
  * P(next reported gap is a record): exact tail probability under the
    exponential model, P(merit >= easiest_target | merit >= threshold) =
    exp(-(m_easy - m_min)/sigma).  With fewer than 30 records the fitted
    sigma is noise, so the projection is also shown for the prior
    sigma ~= 1.29 (three independent rate anchors from the shift-507
    corpus);
  * the easiest recordable targets at this size.

Usage:
    scripts/gap_hunt_stats.py [records-file ...] [--table PATH]

With no arguments it globs data/gap_hunt_records*.txt (excluding the
watcher's *_found_* files).
"""

import sys
import glob
import math
import os


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


def main():
    args = sys.argv[1:]
    table_path = "data/prime_gap_merits.txt"
    files = []
    for a in args:
        if a == "--table":
            continue
        if a.startswith("--"):
            print(f"unknown: {a}", file=sys.stderr)
            return 2
        files.append(a)
    if "--table" in args:
        table_path = args[args.index("--table") + 1]
    if not files:
        files = sorted(
            f for f in glob.glob("data/gap_hunt_records*.txt")
            if "_found_" not in os.path.basename(f))

    table = load_table(table_path)
    if not files:
        print("no results files found", file=sys.stderr)
        return 2

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

        # P(next reported gap is a record) = P(merit >= easiest target
        # merit | merit >= threshold) — exact for the exponential tail.
        tgt = sorted(
            ((g / L, g, bm) for g, bm in table.items()
             if 8000 <= g <= 70000 and g / bm >= L))
        m_easy = tgt[0][0] if tgt else None
        small = n < 30
        sig_fit = sig_me if sig_me > 0 else 1.29
        p_fit = math.exp(-(m_easy - m_min) / sig_fit) if m_easy else 0.0
        p_prior = math.exp(-(m_easy - m_min) / 1.29) if m_easy else 0.0

        print(f"== {path}")
        print(f"   records={n} threshold={m_min:.4f} best={best:.6f} "
              f"(gap {best_gap}) logbase~{L:.1f}")
        print(f"   sigma: mean-excess={sig_me:.3f}"
              + (f"  quantile={sig_q:.3f}" if sig_q else "")
              + ("  [n<30: NOISY — prior 1.29 used below]" if small else ""))
        if found:
            print(f"   *** {len(found)} RECORD(S) ALREADY FOUND ***")
            for d, g, m, _ in found:
                print(f"      gap={g} merit={m:.6f} delta={d:.6f}")
        print("   closest to a record:")
        for d, g, m, tag in near:
            print(f"      gap={g} merit={m:.6f} "
                  f"needed={table[g]:.6f} delta={d:.6f}")
        print(f"   P(next reported gap is a record): easiest target "
              f"merit={m_easy:.3f} (gap {tgt[0][1]})")
        if small:
            print(f"      fitted sigma {sig_fit:.2f} -> {p_fit:.3f} "
                  f"(unreliable, n<30)")
            print(f"      prior  sigma 1.29 -> {p_prior:.3f} "
                  f"-> expected reported gaps ~{1.0/p_prior:.0f}")
        else:
            print(f"      fitted sigma {sig_fit:.2f} -> {p_fit:.3f} "
                  f"-> expected reported gaps ~{1.0/p_fit:.0f}")
        print("   easiest recordable targets (needed merit, gap):")
        for m, g, bm in tgt[:8]:
            print(f"      {m:.3f}  {g}  (table {bm:.4f})")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
