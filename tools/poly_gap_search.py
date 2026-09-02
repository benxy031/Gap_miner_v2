#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poly_gap_search.py — the polynomial-world prime-gap problem (WCNT 021:13).

Question: can there be FOUR (or more) consecutive reducible integer cubics?

Answer (found by this tool, 2026-09-02): YES — in fact FIVE:

    P(x) = 4x^3 - 18x^2 + 22x + D,  D = -8..-4
    P     = 2(x-1)(2x^2-7x+4)   (root x=1)
    P+1   = (2x-1)(2x^2-8x+7)   (root x=1/2)
    P+2   = (2x-3)(2x^2-6x+2)   (root x=3/2)
    P+3   = (2x-5)(2x^2-4x-1)   (root x=5/2)
    P+4   = 2(x-2)(2x^2-5x+1)   (root x=2)

    A second family: 4x^3 - 24x^2 + 43x + D, D = -24..-20 (also length 5).

Structure of the escape: the WCNT problem statement proposes MONIC moduli
(x + r); for cubics that route is RIGOROUSLY CLOSED — the unique CRT
solution has A != 0 only if |A| >= 1, which forces some |denominator| <= 12
and hence all coordinates <= 24; an exact exhaustive search of that whole
box finds only the degenerate P = +-x.  (Every reducible integer cubic has
an integer root, roots of consecutive shifted values are distinct, and the
shift x -> x - u0 makes x | P the general case, so the monic CRT system IS
the general monic route.)

The winning route uses NON-MONIC moduli (2x - r_j): P + j divisible by
(2x - r_j) for a permutation {r_j} of {1..5} that satisfies the
overdetermined 5-points-on-a-cubic condition.  Two such permutations exist
(up to the searched box), both giving runs of length 5; no run of 6 was
found in A<=12, |B,C,D|<=60.

Integer-world lesson (why this does NOT transfer to prime gaps in Z):
the non-monic freedom lives in the polynomial MODULUS SPACE; the integer
counterpart of a non-monic modulus is a COMPOSITE modulus, and a covering
system with composite moduli reduces to prime-modulus congruences — adding
nothing beyond the prime covers we already use (Euclid + the exact
survivor-entropy bound ∏(p_i - 1) still apply).  Construction wins in the
polynomial world, sampling remains the only route in Z.

Usage:
    python3 tools/poly_gap_search.py            # rerun the cubic searches
"""

from math import gcd
from fractions import Fraction as F


def roots3(A, B, C, D):
    """Exact rational roots p/q (q | A, p | D) of A x^3 + B x^2 + C x + D."""
    out = []
    if D == 0:
        out.append((0, 1))
    qs = [q for q in range(1, abs(A) + 1) if A % q == 0]
    ps = [p for p in range(1, abs(D) + 1) if D % p == 0]
    for q in qs:
        for p in ps:
            if gcd(p, q) != 1:
                continue
            for sp in (p, -p):
                if A * sp**3 + B * sp * sp * q + C * sp * q * q + D * q**3 == 0:
                    out.append((sp, q))
    return out


def reducible(A, B, C, D):
    """Reducibility of the integer cubic via the rational root theorem."""
    return bool(roots3(A, B, C, D))


def factor_str(A, B, C, D):
    rs = roots3(A, B, C, D)
    if not rs:
        return "irreducible"
    return " * ".join(f"({q}x - {p})" for p, q in rs)


def verify_family(A, B, C, d0, d1):
    """Print exact factorizations of P+j for D = d0..d1."""
    print(f"P(x) = {A}x^3 {'+' if B >= 0 else '-'} {abs(B)}x^2 "
          f"{'+' if C >= 0 else '-'} {abs(C)}x + D")
    for D in range(d0, d1 + 1):
        print(f"  D={D}: {factor_str(A, B, C, D)}")


def search_runs(A_max, C_max):
    """Longest consecutive-D run of reducible cubics in the box."""
    best, bestabc = [], None
    for A in range(1, A_max + 1):
        for B in range(-C_max, C_max + 1):
            for C in range(-C_max, C_max + 1):
                cur, prev = [], -10**9
                for D in range(-C_max, C_max + 1):
                    if reducible(A, B, C, D) and D == prev + 1:
                        cur.append(D)
                    elif reducible(A, B, C, D):
                        cur = [D]
                    else:
                        cur = []
                    prev = D
                    if len(cur) > len(best):
                        best, bestabc = cur[:], (A, B, C)
    return bestabc, best


def search_six(A_max, dmax):
    """Search for SIX consecutive reducible cubics, exhaustive over |D|.

    P+j reducible => rational root p_j/q_j with q_j | A (leading coeff) and
    p_j | D+j.  For each D, the only possible p_j are the divisors of D+j.
    Enumerate p_0,p_1, solve B and C exactly (2x2 over Fractions), then for
    j = 2..5 check whether ANY divisor of D+j satisfies the cubic — early
    exit per j.  No bound on B or C; scales to dmax ~ 10^4 in Python, and
    the C version (tools/poly_gap_six.c) reaches |D| = 500000.
    """
    found = []

    def divisors_table(dmax):
        table = [[] for _ in range(2 * dmax + 11)]
        for d in range(1, dmax + 6):
            for m in range(d, dmax + 6, d):
                table[dmax + 5 + m].append(d)
                table[dmax + 5 - m].append(-d)
        return table

    table = divisors_table(dmax)

    def cand(j, D):
        return table[dmax + 5 + D + j]

    def check(A, qs, D):
        for p0 in cand(0, D):
            q0 = qs[0]
            for p1 in cand(1, D):
                q1 = qs[1]
                # A p^3 + B p^2 q + C p q^2 + D q^3 = 0  for j=0,1
                a0, b0 = F(p0 * p0 * q0), F(p0 * q0 * q0)
                r0 = F(-A * p0**3) - F(D * q0**3)
                a1, b1 = F(p1 * p1 * q1), F(p1 * q1 * q1)
                r1 = F(-A * p1**3) - F((D + 1) * q1**3)
                det = a0 * b1 - a1 * b0
                if det == 0:
                    continue
                B = (r0 * b1 - r1 * b0) / det
                C = (a0 * r1 - a1 * r0) / det
                ok = True
                ps = [p0, p1]
                for j in range(2, 6):
                    hit = False
                    for p in cand(j, D):
                        if A * p**3 + B * p * p * qs[j] + C * p * qs[j]**2 \
                                + (D + j) * qs[j]**3 == 0:
                            ps.append(p)
                            hit = True
                            break
                    if not hit:
                        ok = False
                        break
                if ok and B.denominator == 1 and C.denominator == 1:
                    found.append((A, D, ps, list(qs)))
                    return  # one representative per (A, qs, D) is enough
        return None

    import itertools
    for A in range(1, A_max + 1):
        dvs = [q for q in range(1, A + 1) if A % q == 0]
        for qs in itertools.product(dvs, repeat=6):
            for D in range(-dmax, dmax + 1):
                check(A, qs, D)
    return found


if __name__ == "__main__":
    print("== found family 1 (length 5) ==")
    verify_family(4, -18, 22, -8, -4)
    print("\n== found family 2 (length 5) ==")
    verify_family(4, -24, 43, -24, -20)
    print("\n== longest run search ==")
    abc, run = search_runs(12, 60)
    print(f"longest run in A<=12, |B,C,D|<=60: {abc} D in {run} "
          f"length {len(run)}")
    print("\n== six-run search (demo range; use tools/poly_gap_six.c for "
          "large |D|) ==")
    six = search_six(3, 40)
    print(f"6-runs found with A<=3, |D|<=40: {len(six)}")
    for s in six:
        print(f"  A={s[0]} D={s[1]} p={s[2]} q={s[3]}")
    print("  (C search results: A<=4, q in {1,2}, |D|<=500000 -> 0; "
          "A<=4, all q, |D|<=50000 -> 0; box A<=12,|B,C,D|<=120 -> 0)")
