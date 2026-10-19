#!/usr/bin/env python3
"""Verify a claimed smooth run: `verify_smooth.py G START END`.

A run for bound G is maximal when every gap between consecutive primes in
[START, END] is <= G, the prime before START is further than G below START
and the prime after END is further than G above END.  Prints the gap count,
the delimiters, and PASS/FAIL for each condition (exact, using GMP via
gmpy2 when available, otherwise sympy).
"""
import sys

try:
    import gmpy2 as gp
    isprime = lambda n: bool(gp.is_prime(int(n)))
    nextprime = lambda n: int(gp.next_prime(int(n)))
    prevprime = lambda n: int(gp.prev_prime(int(n)))
    ENGINE = "gmpy2"
except ImportError:                                    # fallback
    import sympy
    isprime = lambda n: bool(sympy.isprime(int(n)))
    nextprime = lambda n: int(sympy.nextprime(int(n)))
    prevprime = lambda n: int(sympy.prevprime(int(n)))
    ENGINE = "sympy"


def primes_between(lo, hi):
    """Consecutive primes in [lo, hi] (both ends included when prime)."""
    out = []
    p = nextprime(lo - 1)
    while p <= hi:
        out.append(p)
        p = nextprime(p)
    return out


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    g, start, end = (int(x) for x in sys.argv[1:4])
    if start % 2 == 0 or end % 2 == 0:
        print("note: start/end are even; a run's endpoints are primes, so one "
              "of the two will fail - pass the prime endpoints")
    p = primes_between(start, end)
    bad = 0
    if not p or p[0] != start:
        print("FAIL: START is not prime")
        bad += 1
    if not p or p[-1] != end:
        print("FAIL: END is not prime")
        bad += 1
    gaps = [p[i + 1] - p[i] for i in range(len(p) - 1)]
    big = [(p[i], gaps[i]) for i in range(len(gaps)) if gaps[i] > g]
    if big:
        print("FAIL: %d interior gap(s) > %d, first at %d (gap %d)"
              % (len(big), g, big[0][0], big[0][1]))
        bad += 1
    before = start - prevprime(start)
    after = nextprime(end) - end
    if before <= g:
        print("FAIL: left delimiter %d <= %d (run is not maximal on the left)"
              % (before, g))
        bad += 1
    if after <= g:
        print("FAIL: right delimiter %d <= %d (run is not maximal on the right)"
              % (after, g))
        bad += 1
    print("g<=%d  gaps=%d  primes=%d  span=%d..%d  delimiters %d / %d  engine=%s"
          % (g, len(gaps), len(p), start, end, before, after, ENGINE))
    print("VERDICT:", "PASS" if bad == 0 else "FAIL")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
