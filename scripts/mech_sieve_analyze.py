#!/usr/bin/env python3
"""mech_sieve_analyze.py - compare the arms of scripts/mech_sieve_ab.sh.

For each arm (differing only in --sieve-primes, hence in the survivor density u)
it reports the merit distribution's SHAPE, because the shape is the measurement:
  * the mode (which merit bin is the peak), to test mode ~ 1/u;
  * the fitted slope above the floor (sigma_MLE), to test whether the tails agree;
  * the yield per window (gaps/window), to see whether only the RATE moved;
  * a normalised histogram so arms with different n are comparable.

Usage: mech_sieve_analyze.py [arm-dump ...]      (defaults to /tmp/mech_P*.txt)
"""
import glob
import math
import os
import re
import statistics
import sys

RE_STATS = re.compile(r"\[GAP_HUNT\]\s+k=(\d+)\s+windows=(\d+)\s+gaps=(\d+)"
                      r"\s+best_merit=([0-9.]+)\s+win_s=([0-9.]+)")
RE_BANNER = re.compile(r"sieve=(\d+)")


def load(path):
    merits = []
    with open(path, errors="ignore") as f:
        for line in f:
            p = line.split(None, 2)
            if len(p) < 2:
                continue
            try:
                merits.append(float(p[1]))
            except ValueError:
                continue
    return merits


def log_stats(path):
    """(windows, win_s_avg, sieve limit) from the arm's log, or Nones."""
    win = wins = sieve = None
    if not os.path.exists(path):
        return win, wins, sieve
    last = None
    for line in open(path, errors="ignore"):
        m = RE_STATS.search(line)
        if m:
            last = m
        b = RE_BANNER.search(line)
        if b and sieve is None:
            sieve = int(b.group(1))
    if last:
        win = float(last.group(2))
        wins = float(last.group(5))
    return win, wins, sieve


def u_model(P):
    """Survivor density of a sieve to P: prod (1-1/p) ~ e^-(ln ln P + 0.2615)."""
    if not P:
        return None
    return math.exp(-(math.log(math.log(P)) + 0.2615))


def main(argv):
    paths = argv or sorted(glob.glob("/tmp/mech_P*.txt"))
    paths = [p for p in paths if os.path.getsize(p) > 0]
    if not paths:
        print("no arm dumps found (expected /tmp/mech_P*.txt)", file=sys.stderr)
        return 2
    arms = []
    for p in paths:
        m = load(p)
        win, win_s, sieve = log_stats(p[:-4] + ".log")
        floors = 10.0
        arms.append(dict(path=p, merits=m, n=len(m), win=win, win_s=win_s,
                         sieve=sieve, floor=floors))
    print("=== arm summary ===")
    print("%-26s %7s %8s %8s %8s %9s %9s %10s" %
          ("arm", "n", "min", "median", "mean", "sigma_MLE", "u(model)",
           "gaps/1kw"))
    for a in arms:
        m = a["merits"]
        if not m:
            print("%-26s %7d   (empty)" % (os.path.basename(a["path"]), 0))
            continue
        sig = statistics.mean(m) - a["floor"]
        u = u_model(a["sieve"])
        per_kw = (a["n"] / a["win"] * 1000.0) if a["win"] else float("nan")
        print("%-26s %7d %8.3f %8.3f %8.3f %9.4f %9.4f %10.2f" %
              (os.path.basename(a["path"]), a["n"], min(m),
               statistics.median(m), statistics.mean(m), sig,
               u if u else float("nan"), per_kw))

    print("\n=== normalised merit histogram (share of that arm's gaps per bin) ===")
    hi = int(max(max(a["merits"]) for a in arms if a["merits"])) + 1
    bins = list(range(10, hi + 1))
    print("%-8s" % "bin" + "".join("%18s" % os.path.basename(a["path"])
                                   for a in arms))
    for lo in bins:
        row = ""
        for a in arms:
            m = a["merits"]
            if not m:
                row += "%18s" % "-"
                continue
            c = sum(1 for x in m if lo <= x < lo + 1)
            share = 100.0 * c / len(m)
            row += "%18s" % ("%6d  %5.2f%%" % (c, share))
        print("%-8s" % ("[%d,%d)" % (lo, lo + 1)) + row)

    print("\n=== mode and local slope ===")
    for a in arms:
        m = a["merits"]
        if not m:
            continue
        counts = [(lo, sum(1 for x in m if lo <= x < lo + 1))
                  for lo in range(10, hi + 1)]
        peak = max(counts, key=lambda kv: kv[1]) if counts else (None, 0)
        print("%-26s mode bin = [%s,%s)  n_peak=%d   sigma_MLE=%.3f   "
              "u_model=%.3f   1/u=%.1f" %
              (os.path.basename(a["path"]), peak[0], (peak[0] or 0) + 1,
               peak[1], statistics.mean(m) - a["floor"],
               u_model(a["sieve"]) or float("nan"),
               1.0 / (u_model(a["sieve"]) or float("nan"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
