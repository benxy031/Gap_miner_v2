#!/usr/bin/env python3
"""record_planner.py - which record is actually the CHEAPEST at a given size?

At a fixed hunt size, merit = g/L, so the frontier table and L alone decide
whether a gap of length g would be a record.  The CHEAP label is our merit m(g)
= g/L, because the cost of finding that exact length is

    E[primes] ~ exp(E_ours(m))          (E_ours = our measured tail exponent)

so targets are ranked by m, not by margin.  The margin only says how safe the
record is (a margin of 0.13 breaks on the next table refresh, 5.0 does not).

Usage:
    scripts/record_planner.py --L 881.7 [--L 528.9 ...] [--top 10]
                              [--lam20 126.1 --sigma 1.3272] [--ref-L 881.7]
Options:
    --lam20/--sigma   measured hunt rate at merit 20 and its tail sigma, both
                      at --ref-L; the ETA column is only printed for --ref-L
                      (other sizes are ranked in cost units m/exp).
    --gmin/--gmax     reachable gap-length window (default 1000..40000)
"""
import argparse
import math
import os


def load_table(path):
    tab = {}
    with open(path, errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                try:
                    tab[int(p[0])] = float(p[1])
                except ValueError:
                    continue
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/prime_gap_merits.txt")
    ap.add_argument("--L", type=float, action="append", default=None)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--gmin", type=int, default=1000)
    ap.add_argument("--gmax", type=int, default=40000)
    ap.add_argument("--lam20", type=float, default=126.1)
    ap.add_argument("--sigma", type=float, default=1.3272)
    ap.add_argument("--m0", type=float, default=20.0)
    ap.add_argument("--ref-L", type=float, default=881.7)
    args = ap.parse_args()

    tab = load_table(args.table)
    Ls = args.L or [528.9, 881.7]
    print("table %s: %d entries\n" % (args.table, len(tab)))

    for L in Ls:
        cand = []
        for g, need in tab.items():
            if g < args.gmin or g > args.gmax or g % 2:
                continue
            m = g / L
            if m > need:                      # would be a record
                cand.append((m, g, need, m - need))
        cand.sort()
        print("=== L = %.1f  (%.0f-bit numbers) : %d reachable record lengths"
              % (L, L / math.log(2.0), len(cand)))
        if not cand:
            print("   none\n")
            continue
        print("   %-8s %8s %9s %9s %9s %10s" %
              ("gap", "our mer", "frontier", "margin", "L_front", "ETA (at L=%.0f)"
               % args.ref_L))
        for m, g, need, marg in cand[:args.top]:
            eta = ""
            if abs(L - args.ref_L) < 1e-6:
                # rate of the exact length g: lambda(m)/L, from the measured
                # exponential tail anchored at merit m0
                lam = args.lam20 * math.exp(-(m - args.m0) / args.sigma) / L
                h = 1.0 / lam if lam > 0 else float("inf")
                eta = ("%.1f h" % h) if h < 48 else ("%.1f d" % (h / 24.0))
            print("   %-8d %8.4f %9.4f %9.4f %9.0f %10s"
                  % (g, m, need, marg, g / need, eta if eta else "(rank only)"))
        # how much of the reachable set is SAFE (margin >= 1 merit)?
        safe = [c for c in cand if c[3] >= 1.0]
        cheap_safe = safe[0] if safe else None
        print("   margin >= 1.0 (not broken by the next table refresh): %d"
              % len(safe))
        if cheap_safe:
            print("   cheapest SAFE target: gap %d  (our merit %.4f, frontier "
                  "%.4f, margin %.4f)" % (cheap_safe[1], cheap_safe[0],
                                          cheap_safe[2], cheap_safe[3]))
        print()


if __name__ == "__main__":
    main()
