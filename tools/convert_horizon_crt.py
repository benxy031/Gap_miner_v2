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
ours counts over [1, gap_target) (covering.c's range: endpoint and anchor both
excluded), so the two counts differ by one exactly when the anchor is uncovered.

Residue 0 means "prime excluded" in OUR format (crt_runtime.c skips such rows),
but o_p = 0 is a REAL class in their convention (j ≡ 0 mod p).  Those rows are
therefore written as r = p, which expresses the same class; writing 0 would
silently drop the prime and weaken the cover (measured on their m22 file:
3,079 survivors at gap_target 12,644 instead of 876).

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


def count_uncovered(primes, residues, lo, hi, zero_excludes):
    """Uncovered j in [lo, hi] where a residue r covers j = r (mod p).

    With zero_excludes the rule our loader uses (crt_runtime.c / cover_max.c:
    residue 0 means "prime excluded") is reproduced, so the same helper can
    count both the source convention and what our tools will see in the file.
    """
    n = 0
    for j in range(lo, hi + 1):
        for p, r in zip(primes, residues):
            if zero_excludes and r == 0:
                continue
            if (j - r) % p == 0:
                break
        else:
            n += 1
    return n


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
    # our convention: uncovered over [1, gap_target) -- the same range covering.c
    # counts in count_survivors() (our generated files' headers use it too).
    src_res = [off[p] for p in primes]
    uncovered = count_uncovered(primes, src_res, 1, gap_target - 1, False)
    # round-trip check in THEIR convention: uncovered over [0, size-1] MUST equal
    # their n_candidates, otherwise the residue convention was mis-read.
    theirs_recomputed = count_uncovered(primes, src_res, 0, size - 1, False)

    L = (256.0 + args.shift) * math.log(2.0)
    merit = gap_target / L

    zero_rows = [p for p in primes if off[p] == 0]
    with open(args.outfile, "w") as f:
        f.write("# converted from %s by convert_horizon_crt.py\n" % args.infile)
        f.write("n_primes %d\n" % n_primes)
        f.write("merit %.2f\n" % merit)
        f.write("shift %d\n" % args.shift)
        f.write("gap_target %d\n" % gap_target)
        f.write("n_candidates %d\n" % uncovered)
        for p in primes:
            # our format reserves 0 for "prime excluded"; their class 0
            # (j == 0 mod p) is the same class as residue p.
            f.write("%d %d\n" % (p, off[p] or p))

    print("converted %s -> %s" % (args.infile, args.outfile))
    print("  n_primes %d, shift %d, gap_target %d (their size %s), merit %.4f" %
          (n_primes, args.shift, gap_target,
           size if size == gap_target else "%d -> overridden" % size, merit))
    print("  ROUND-TRIP (their convention, [0,%d]) = %d vs their n_candidates %d -> %s"
          % (size - 1, theirs_recomputed, their_nc,
             "OK" if theirs_recomputed == their_nc else "MISMATCH (residue convention wrong)"))
    if theirs_recomputed != their_nc:
        sys.exit(2)
    print("  our convention, [1,%d) = %d%s" %
          (gap_target, uncovered,
           "" if size == gap_target else " (gap_target overridden, not comparable)"))
    if zero_rows:
        print("  residue-0 rows kept via r=p (0 would EXCLUDE the prime): %s" %
              ",".join(str(p) for p in zero_rows))
    # Regression guard for the residue-0 class of bug: re-read the written file
    # exactly as our loader does (residue 0 means "prime excluded") and require
    # the same survivor count.
    f_primes, f_res = [], []
    for line in open(args.outfile):
        m = re.match(r"^(\d+)\s+(\d+)\s*$", line)
        if m:
            f_primes.append(int(m.group(1)))
            f_res.append(int(m.group(2)))
    if f_primes != primes:
        sys.exit("SELF-CHECK FAILED: written prime rows do not match the input")
    loaded = count_uncovered(f_primes, f_res, 1, gap_target - 1, True)
    if loaded != uncovered:
        sys.exit("SELF-CHECK FAILED: loader-side count %d != %d "
                 "(residue-0 handling in the written file is wrong)"
                 % (loaded, uncovered))
    print("  note: their window is [0,size-1] and includes the anchor at j=0;"
          " ours is [1,gap_target) and excludes the anchor and the endpoint,"
          " so the two counts may differ by +-1 even when the cover is identical.")


if __name__ == "__main__":
    main()
