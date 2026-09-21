#!/usr/bin/env python3
"""convert_horizon_crt.py -- convert a Horizon/Golden "ChineseSet" CRT file into
gapminer_v2's text CRT format, so their covers can be driven by our miner and by
our tools (test_crt_runtime / cover_max / gap_hunt).

Horizon format (4-5 lines):
    |== ChineseSet ==|
    n_primes:     74
    size:         11319
    n_candidates: 761
    offset:       <one ~500-bit integer X, digits wrapped over several lines>

Reverse-engineered convention (VERIFIED: reproduces their n_candidates exactly for
all three shipped m22/m30 files): the per-prime residue is
    o_p = (-X) mod p        for the first n_primes primes,
a position j is covered iff (j - o_p) % p == 0 for some p, and THEIR n_candidates
counts uncovered j over [0, size-1] -- i.e. they include the anchor at j=0, while
our convention counts over [1, gap_target] and excludes it.

Our format:
    # comment
    n_primes 74
    merit 22.00
    shift 512
    gap_target 11712
    n_candidates 846
    2 1
    3 2
    ...

Usage: convert_horizon_crt.py IN.txt OUT.txt --shift 512 [--gap-target N]
       (defaults: shift 512, gap_target = their size, merit = size / ((256+shift)*ln2))
"""
import argparse
import math
import re
import sys


def first_primes(n):
    ps = []
    x = 2
    while len(ps) < n:
        if all(x % q for q in ps if q * q <= x):
            ps.append(x)
        x += 1
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--shift", type=int, default=512)
    ap.add_argument("--gap-target", type=int, default=None,
                    help="default: their `size`")
    args = ap.parse_args()

    txt = open(args.infile).read()
    n_primes = int(re.search(r"n_primes:\s*(\d+)", txt).group(1))
    size = int(re.search(r"size:\s*(\d+)", txt).group(1))
    their_nc = int(re.search(r"n_candidates:\s*(\d+)", txt).group(1))
    if "offset:" not in txt:
        sys.exit("no `offset:` field -- not a Horizon ChineseSet file")
    # all digits after the offset: label, joined (the number wraps over lines)
    X = int("".join(re.findall(r"\d", txt.split("offset:")[1])))
    if X <= 0:
        sys.exit("bad offset")

    primes = first_primes(n_primes)
    off = {p: (-X) % p for p in primes}

    gap_target = args.gap_target or size
    # our convention: uncovered over [1, gap_target]
    uncovered = 0
    for j in range(1, gap_target + 1):
        for p in primes:
            if (j - off[p]) % p == 0:
                break
        else:
            uncovered += 1
    # round-trip check in THEIR convention: uncovered over [0, size-1] MUST equal
    # their n_candidates, otherwise the residue convention was mis-read.
    theirs_recomputed = 0
    for j in range(0, size):
        for p in primes:
            if (j - off[p]) % p == 0:
                break
        else:
            theirs_recomputed += 1

    L = (256.0 + args.shift) * math.log(2.0)
    merit = gap_target / L

    with open(args.outfile, "w") as f:
        f.write("# converted from %s by convert_horizon_crt.py\n" % args.infile)
        f.write("n_primes %d\n" % n_primes)
        f.write("merit %.2f\n" % merit)
        f.write("shift %d\n" % args.shift)
        f.write("gap_target %d\n" % gap_target)
        f.write("n_candidates %d\n" % uncovered)
        for p in primes:
            f.write("%d %d\n" % (p, off[p]))

    print("converted %s -> %s" % (args.infile, args.outfile))
    print("  n_primes %d, shift %d, gap_target %d (their size %s), merit %.4f" %
          (n_primes, args.shift, gap_target,
           size if size == gap_target else "%d -> overridden" % size, merit))
    print("  ROUND-TRIP (their convention, [0,%d]) = %d vs their n_candidates %d -> %s"
          % (size - 1, theirs_recomputed, their_nc,
             "OK" if theirs_recomputed == their_nc else "MISMATCH (residue convention wrong)"))
    if theirs_recomputed != their_nc:
        sys.exit(2)
    print("  our convention, [1,%d] = %d%s" %
          (gap_target, uncovered,
           "" if size == gap_target else " (gap_target overridden, not comparable)"))
    print("  note: their window is [0,size-1] and includes the anchor at j=0;"
          " ours is [1,gap_target] and excludes it, so the two counts may differ"
          " by +-1 even when the cover is identical.")


if __name__ == "__main__":
    main()
