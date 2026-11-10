#!/usr/bin/env python3
"""psp_probe.py — how much pseudoprime risk our exhaustive census actually carries.

WHY
---
An exhaustive gap scan can only lose a gap one way: a composite that PASSES the
discovery test splits one real gap into two, and if both halves fall below the
reporting threshold the real gap is never logged (so the GMP re-verification of
every reported gap cannot help - it never sees it).  Everything else is safe,
because a prime never fails a Fermat/Euler/MR test, so a failed test is a real
proof of compositeness.

Two questions follow, and this script answers both with measurements:

  (1) STRENGTH - how much weaker is the test we actually run?  Our walk engine
      selects a base-2 EULER/JACOBI congruence (w_test_mode modes 0/3/4/5:
      perig, p0lazy, p0u96, p0u108) for every production band; only modes 1/2
      (n >= 2^120) are strong.  Carmichael numbers are the class where that
      difference shows, so they are the natural yardstick: a Carmichael number
      passes the FERMAT test to every base coprime to it, while a strong test
      rejects almost all of them.

  (2) DENSITY - how often does a base-2 Euler/Jacobi pseudoprime that SURVIVED
      our sieve appear at the scale we scan?  Measured by sieving a sample to
      --walk-primes and running the exact Euler/Jacobi criterion on survivors.

The answer for the production wall (measured 2026-11-03, x = 1e32, sieve 150000)
is 0 pseudoprimes in 54536 Euler passes, and the campaign-wide expected loss is
~1e-13..1e-11 gaps, dominated not by pseudoprime rarity but by the fraction of
the range that lies inside a gap of size >= gap_min.  See README_PHASE0.md,
"Pseudoprimes and Carmichael numbers (completeness doctrine)".

Usage:
    scripts/psp_probe.py                 # both blocks, default sample
    scripts/psp_probe.py --carmichael    # only the test-strength block
    scripts/psp_probe.py --density --odds 5000000 --x 1e32 --sieve 150000
"""
import argparse
import math
import sys

try:
    import gmpy2
except ImportError:                                        # pragma: no cover
    raise SystemExit("psp_probe needs gmpy2 (pip install gmpy2)")

from psp_math import euler_base2, strong_base2, strong_base as _strong


def carmichael_block(limit=51, sieve_depth=150000):
    """(a) Euler vs strong on Carmichael numbers, (b) a Chernick case that
    survives any realistic sieve."""
    # Korselt's criterion, generated rather than hard-coded.
    carm, n = [], 3
    while len(carm) < limit:
        n += 2
        if gmpy2.is_prime(n):
            continue
        f = {}
        m = n
        d = 2
        while d * d <= m:
            while m % d == 0:
                f[d] = f.get(d, 0) + 1
                m //= d
            d += 1 if d == 2 else 2
        if m > 1:
            f[m] = f.get(m, 0) + 1
        if len(f) > 1 and all(e == 1 for e in f.values()) and \
                all((n - 1) % (p - 1) == 0 for p in f):
            carm.append(n)

    e_ok = sum(1 for c in carm if euler_base2(c))
    s_ok = sum(1 for c in carm if strong_base2(c))
    print("(1) TEST STRENGTH on the first %d Carmichael numbers" % len(carm))
    print("    base-2 Euler/Jacobi accepts : %d  (%.0f %%)   <- perig / p0u96 / p0u108"
          % (e_ok, 100.0 * e_ok / len(carm)))
    print("    base-2 strong MR accepts    : %d  (%.0f %%)   <- mr_base2, modes 1/2"
          % (s_ok, 100.0 * s_ok / len(carm)))

    # Chernick family: (6k+1)(12k+1)(18k+1) is Carmichael when all three are prime.
    k, shown = 9000, 0
    print("\n    Carmichael numbers whose SMALLEST factor is above the %d sieve"
          % sieve_depth)
    print("    (so no sieve and no 2-PSP sieve can remove them):")
    print("      k       n          digits  euler  strong")
    while shown < 5 and k < 400000:
        a, b, c = 6 * k + 1, 12 * k + 1, 18 * k + 1
        if min(a, b, c) > sieve_depth and gmpy2.is_prime(a) \
                and gmpy2.is_prime(b) and gmpy2.is_prime(c):
            nn = a * b * c
            print("      %-7d %-11.3e %4d    %-5s  %s"
                  % (k, nn, len(str(nn)), euler_base2(nn), strong_base2(nn)))
            shown += 1
        k += 1
    print("    -> every one passes the Euler/Jacobi test we run and is caught")
    print("       only by a strong test: this is why Euler/Jacobi and strong are")
    print("       not interchangeable in a completeness claim.")


# Two real gaps with a hard composite in the interior.  Both are public: the
# 710 one is PGS's worktodo fixture, the 956 one is BrianK's counter-example
# from Discord (2026-11-03).
#
# CORRECTION (measured 2026-11-04): the 710 number is NOT a base-2 pseudoprime,
# although PGS's worktodo.txt calls it one.  n-1 = 18447233112263860536 and
# 2^(n-1) mod n = 16649844496336080870, not 1 (checked with gmpy2.powmod, with
# Python's built-in pow and independently by factorising 4015729 x 4593744526153
# = n).  No number in that gap's interior passes Fermat or Euler/Jacobi at all,
# so the 710 gap is found by ANY test and it does not bracket the boundary -
# it only exercises "a semiprime with both factors above any sieve is not
# prime".  Kept because the value is quoted in three places, with the correct
# label.  Only the 956 entry is a pseudoprime, and it is the Euler/Jacobi one.
#
# The third class - Fermat but NOT Euler/Jacobi - had no public example at any
# convenient scale, so it is constructed here rather than searched for.  With
# p = 42443 (ord_p(2) = d) and q = 2546521 satisfying q = 1 (mod d) and
# ord_q(2) | p-1, the product p*q is guaranteed to be a base-2 Fermat
# pseudoprime (measured 2026-11-04: 2^(n-1) mod n = 1, 2^((n-1)/2) mod n =
# 29310456711 with Jacobi(2/n) = -1, so Euler needs n-1 and rejects).  It sits
# inside a prime gap of 156, both factors are above the 30000 sieve used in the
# gate, and both engines report the gap - which is the point: our Euler/Jacobi
# test DOES catch this class.
FIXTURES = [
    # (gap, lower prime, upper prime, interior number, expected label)
    (156, 108081990733, 108081990889, 108081990803,
     "Fermat pseudoprime, NOT Euler/Jacobi (constructed; both engines find it)"),
    (710, 18447233112263859833, 18447233112263860543,
     18447233112263860537, "not a pseudoprime (PGS labels it 2-PSP: wrong)"),
    (956, 160696461657079905863, 160696461657079906819,
     160696461657079906273, "Euler/Jacobi pseudoprime (walk is blind to it)"),
]


def fixture_block(sieve_depth=150000):
    """Walk each fixture's interior with both tests and report what is logged."""
    print("(0) FIXTURES - what the walk logs around a gap with a hard composite")
    print("    in the interior (gap_min set just under the true gap):")
    print("\n    %-4s %-21s %-23s %-6s %-6s %-6s" %
          ("gap", "interior number", "factorisation", "ferm", "euler", "strong"))
    for gap, lo, hi, psp, _ in FIXTURES:
        assert lo + gap == hi, "fixture %d: delimiters do not match" % gap
        assert gmpy2.is_prime(lo) and gmpy2.is_prime(hi), "fixture %d: delimiters" % gap
        assert (not gmpy2.is_prime(psp)), "fixture %d: interior is PRIME" % gap
        print("    %-4d %-21d %-23s %-6s %-6s %-6s"
              % (gap, psp, _factor_str(psp),
                 gmpy2.powmod(2, psp - 1, psp) == 1,
                 euler_base2(psp), strong_base2(psp)))
    print("\n    (ferm = 2^(n-1) == 1, euler = the perig congruence,"
          " strong = Miller-Rabin base 2)")
    for gap, lo, hi, psp, label in FIXTURES:
        print("    gap %d: %s" % (gap, label))
    print()
    for gap, lo, hi, psp, _ in FIXTURES:
        gm = gap - 1
        e = _walk_logs(lo, hi, euler_base2, gm)
        st = _walk_logs(lo, hi, strong_base2, gm)
        print("    gap %d at gap_min %d:  euler -> %s   strong -> %s"
              % (gap, gm,
                 "FOUND" if any(l == lo and g == gap for l, g in e) else "LOST",
                 "FOUND" if any(l == lo and g == gap for l, g in st) else "LOST"))
    print("    -> the three classes, bracketing our test exactly:")
    print("       156: Fermat only  -> Euler rejects -> BOTH engines find it")
    print("       710: fails everything -> BOTH find it (says nothing about tests)")
    print("       956: Fermat + Euler -> walk blind, only strong finds it")
    print("       None is removable by sieving: all are products of two primes")
    print("       above any depth we or PGS run.")


def _pollard(n):
    """Smallest non-trivial factor by Pollard rho (fixtures are ~2^68 semiprimes)."""
    if n % 2 == 0:
        return 2
    import random
    rng = random.Random(1)
    while True:
        c, x, y, d = rng.randrange(1, n), 2, 2, 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d


def _factor_str(n):
    f = []
    while not gmpy2.is_prime(n):
        d = _pollard(n)
        while n % d == 0:
            f.append(d)
            n //= d
    f.append(n)
    return "x".join(str(x) for x in sorted(f))


def _walk_logs(lo, hi, test, gap_min):
    """The scanner's own rule: consecutive 'primes' whose gap reaches gap_min."""
    prev, out = None, []
    n = lo | 1
    while n <= hi:
        if test(n):
            if prev is not None and n - prev >= gap_min:
                out.append((prev, n - prev))
            prev = n
        n += 2
    return out


# BrianK's COMPLETE list of gaps >= 800 in 1e20..1e21 whose interior contains a
# base-2 Fermat pseudoprime with no prime factor < 1e7, i.e. the gaps his program
# would miss without its separate PSP search (mersenneforum, 2026-05; he reports
# 2 789 289 gaps >= 300 in that range and publishes these 61).  Columns:
# pseudoprime, p, q, gap.
#
# This is the only public ground truth for the question "how leaky is test X",
# so it is a regression set: `--brik` scores any candidate discovery test
# against it.  Measured 2026-11-05:
#     Euler/Jacobi (our walk)      misses 37 / 61
#     strong base 2                misses 26 / 61
#     strong base 2 AND base 3     misses  0 / 61
# No single base-2 test of any strength is sufficient here - the 26 that pass
# strong base 2 are strong pseudoprimes to base 2.  That is why PGS keeps a PSP
# sieve instead of strengthening the kernel, and why a base-2-only completeness
# claim cannot be made in this range.
BRIK_800 = """
841879998525731583133 9710332633 86699398501 1194
398647440525424234057 2495769673 159729259009 1044
160696461657079906273 2142738313 74995840921 956
200545176165814333153 10013620129 20027240257 936
315354790006745996473 3551646323 88791158051 902
754534114930208272001 1171060001 644317212001 902
124033519487731048681 791474521 156711954961 898
954572281011252505561 2497805941 382164308821 894
735360763592344584979 3623733187 202929058417 890
552976230137418164029 1777600679 311080118651 882
797152441425886414081 24077940737 33107168513 882
928300146896217155713 11515828529 80610799697 880
867732283554462810553 8169983917 106209790909 874
290928636023534740711 6030429463 48243435697 870
293532632155199372797 4230641941 69382527817 864
996795869426073331561 4218995881 236263769281 862
738866765366986551461 11770196197 62774379713 860
472861430000504169841 2739660013 172598580757 852
544096484726919042869 728222813 747156605113 852
876245868100295987149 7076100089 123831751541 852
188116284432498156901 1231692661 152729889841 846
613397761893113872501 5681909251 107956275751 840
868025079805685283091 10416483811 83331870481 838
238552724838230287097 9969802841 23927526817 836
253407781849973160901 650967613 389278631977 834
321090982975979133613 10645479289 30162191317 834
496019873290215784007 4546151639 109107639313 832
162820330115941291861 2853246661 57064933201 830
664727601657218754401 19041499201 34909415201 830
321688290707703091201 6779046401 47453324801 828
198215372262987469501 8904277001 22260692501 826
224988094770040002601 5303160553 42425284417 826
877922197729627989751 14814875951 59259503801 826
835980173013742950913 18286390273 45715975681 824
909193640320286854849 9091422733 100005650053 824
977562559561483555873 30802771297 31736188609 820
786671643085960263629 3554897393 221292362653 818
166917924562296044029 726788333 229665112913 816
232294158920175822469 6222193597 37333161577 816
586865683197146069111 7929594319 74009546969 816
993695686248725428993 7799904577 127398441409 816
192252899840188018393 3969689461 48430211413 814
541329083087000415643 8225942827 65807542609 814
161565864543667901353 4641348433 34810113241 812
800755465348217726701 10695496151 74868473051 812
272478457761341369401 11672156137 23344312273 810
553743242358817685869 14715557641 37629783109 810
772745204927162635829 9572860373 80722498273 810
861871006844189168821 651267317 1323375186113 810
270815735394092979781 1615232621 167663611961 806
549944922293957092921 4999749641 109994492081 806
722971312890359831453 6586218853 109770314201 806
286285300784795129047 8459983759 33839935033 804
417040380904354258721 11790396953 35371190857 804
538804220290054899797 4738161653 113715879649 804
894926051706648901633 17271614977 51814844929 804
126791676545849688049 2022388213 62694034573 802
353385293427745017001 6266181661 56395634941 802
632286480495076609141 23280030937 27160036093 802
186493152660063926561 7052056961 26445213601 800
617384291046359813377 1470103813 419959655629 800
"""


def sieve_class_block():
    """Which pseudoprimes does a PLAIN sieve of depth S leave behind?

    A plain sieve removes every n with a prime factor <= S.  So the class it
    cannot touch is "all prime factors > S", and a pseudoprime in that class
    still passes the Euler/Jacobi test.  This block demonstrates the boundary
    with Chernick numbers (6k+1)(12k+1)(18k+1), whose smallest factor can be
    placed just above any depth we run.
    """
    print("(4) WHAT A PLAIN SIEVE LEAVES BEHIND")
    print("    A plain sieve to depth S removes every n with a factor <= S; the")
    print("    pseudoprimes it CANNOT remove are those whose factors are all > S.")
    print("    Chernick numbers put the smallest factor exactly where we choose:")
    print()
    print("    %-10s %-24s %-11s %-9s %-9s %s"
          % ("k", "n = p*q*r", "smallest p", "sieve 85k", "sieve 150k", "euler"))
    shown = 0
    k = 9000
    while shown < 4:
        a, b, c = 6 * k + 1, 12 * k + 1, 18 * k + 1
        if min(a, b, c) > 85000 and gmpy2.is_prime(a) \
                and gmpy2.is_prime(b) and gmpy2.is_prime(c):
            n = a * b * c
            print("    %-10d %-24.4e %-11d %-9s %-9s %s"
                  % (k, n, a,
                     "SURVIVES" if a > 85000 else "removed",
                     "SURVIVES" if a > 150000 else "removed",
                     euler_base2(n)))
            shown += 1
        k += 1
    print("\n    -> such an n passes our discovery test, so a real gap containing")
    print("       it would be split and lost.  A plain sieve cannot close the")
    print("       class; the complementary 2-PSP sieve can, at ~1 mark per p^2:")
    print("       mark n = 0 (mod p) with n/p = 1 (mod ord_p(2)) for every prime")
    print("       p in (S, 1e7].  PGN measures that build at GPU rate unchanged")
    print("       within noise (431.7 -> 433.9 Gpos/s) and +24 % cold-worker")
    print("       time from 1.014 s of CPU table prep (656 742 rows).")
    print("       Factors ABOVE 1e7 are a different class and stay open - that")
    print("       is BrianK's 2.79M-entry semiprime enumeration.")


def brik_block():
    """Score candidate discovery tests against BrianK's complete >=800 list."""
    items = []
    for line in BRIK_800.strip().splitlines():
        n, p, q, g = (int(t) for t in line.split())
        assert p * q == n, "transcription error at %d" % n
        items.append((g, n))
    print("(3) GROUND TRUTH - BrianK's complete list of gaps >= 800 in 1e20..1e21")
    print("    whose interior holds a base-2 Fermat pseudoprime with no factor < 1e7")
    print("    (%d entries; the full >= 300 count in that range is 2,789,289)." % len(items))
    print("\n    %-34s %-10s %s" % ("discovery test", "misses", "note"))
    tests = [
        ("Fermat base 2", lambda n: gmpy2.powmod(2, n - 1, n) == 1),
        ("Euler/Jacobi base 2 (our walk)", euler_base2),
        ("strong base 2 (mr_base2)", strong_base2),
        ("strong base 2 AND base 3", lambda n: strong_base2(n) and _strong(n, 3)),
        ("strong bases 2,3,5,7,11,13,17,19,23,29,31,37",
         lambda n: all(_strong(n, b) for b in
                       (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37))),
    ]
    for name, passes in tests:
        miss = [g for g, n in items if passes(n)]
        note = ""
        if name.startswith("Euler"):
            note = "the engine we run"
        elif name.startswith("strong base 2 ("):
            note = "still not sound - see below"
        elif "AND base 3" in name:
            note = "sufficient for THIS list, not proven in general"
        elif name.startswith("strong bases"):
            note = "deterministic below 3.19e23, so a proof here"
        print("    %-34s %2d / %-5d %s" % (name, len(miss), len(items), note))
    worst = [g for g, n in items if strong_base2(n)]
    print("\n    The %d that pass strong base 2 are strong pseudoprimes to base 2"
          % len(worst))
    print("    (smallest gap %d, largest %d), so no single-witness base-2 test of"
          % (min(worst), max(worst)))
    print("    any strength closes this range - the second base does.")

    # --- when does a CERTIFIER help, and when can it not see the loss? ------
    # Any threshold-search program (PGN Scout, PGS) emits a gap only when it is
    # >= minGap, and its verifier only inspects emitted gaps.  A pseudoprime
    # splits G into a and b; a half is emitted iff it reaches minGap, so the
    # verifier can act iff max(a,b) >= minGap.  Below that, nothing is emitted
    # and the psEudoprime is invisible to the verifier entirely.
    print("\n    --- what a CERTIFYING verifier can and cannot see ---")
    print("    A split gap G becomes a and b; a half is emitted only if it")
    print("    reaches minGap, and a verifier only inspects emitted gaps.  So:")
    print("      minGap <= max(a,b)  -> verifier sees the pseudoprime -> gap")
    print("                             REJECTED (loud, but still not found)")
    print("      minGap >  max(a,b)  -> nothing emitted -> SILENT loss")
    spans = []
    for g, n in items:
        lo = int(gmpy2.prev_prime(n))
        hi = int(gmpy2.next_prime(n))
        spans.append((g, max(n - lo, hi - n)))
    print("\n      %-8s %-10s %-10s" % ("minGap", "loud", "silent"))
    for mg in (300, 400, 500, 600, 700, 800, 1000):
        loud = sum(1 for _, mh in spans if mg <= mh)
        print("      %-8d %3d / %-4d %3d / %d" % (mg, loud, len(spans),
                                                  len(spans) - loud, len(spans)))
    print("    At any threshold a real hunt uses (>= 700) the great majority of")
    print("    these gaps is silently lost, not caught - so 'PGN certifies gaps'")
    print("    does not extend to finding them.")


def parse_x(text):
    """Exact power-of-ten parser: int(1e32) != 10**32 by 5.4e15, which would
    silently move the sample window.  Accepts '1e32', '1E32', '3.5e20', '1234'."""
    t = str(text).strip().lower()
    if "e" in t:
        mant, exp = t.split("e")
        sign = -1 if mant.startswith("-") else 1
        mant = mant.lstrip("+-")
        frac = len(mant.split(".")[1]) if "." in mant else 0
        num = int(mant.replace(".", ""))
        return sign * num * 10 ** (int(exp) - frac)
    return int(t)


def density_block(x="1e32", sieve=150000, odds=2_000_000, chunk=200_000):
    """How often a base-2 Euler/Jacobi pseudoprime survives the production sieve."""
    X = parse_x(x)

    def primes_upto(n):
        s = bytearray([1]) * (n + 1)
        s[0] = s[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if s[i]:
                s[i * i::i] = bytearray(len(s[i * i::i]))
        return [i for i in range(3, n + 1) if s[i]]

    P = primes_upto(sieve)
    print("(2) DENSITY at x = %s (exact), sieve depth %d, %d odd numbers"
          % (x, sieve, odds))
    surv = passed = 0
    psp = []
    done, a = 0, X | 1
    while done < odds:
        cnt = min(chunk, odds - done)
        alive = bytearray([1]) * cnt                  # index j <-> value a + 2j
        for p in P:
            j = (-a * ((p + 1) // 2)) % p              # a + 2j == 0 (mod p)
            alive[j::p] = bytearray(len(alive[j::p]))
        keep = [a + 2 * j for j in range(cnt) if alive[j]]
        surv += len(keep)
        for n in keep:
            if euler_base2(n):
                passed += 1
                if not gmpy2.is_prime(n, 40):
                    psp.append(n)
        done += cnt
        a += 2 * cnt
        print("    %8d/%d  survivors=%d  euler=%d  PSP=%d"
              % (done, odds, surv, passed, len(psp)), flush=True)

    print("    survived the sieve          : %d  (%.4f of odds)"
          % (surv, surv / max(odds, 1)))
    print("    passed base-2 Euler/Jacobi  : %d" % passed)
    print("    COMPOSITE that passed (PSP) : %d" % len(psp))
    print("    95 %% upper bound                       : %.2e per odd number"
          % (3.0 / odds))
    if passed:
        print("    95 %% upper bound per Euler pass        : %.2e"
              % (3.0 / passed))
    print("    (the per-odd bound is the one that feeds E[lost gaps]: a")
    print("     pseudoprime damages the census if it lands inside a gap of")
    print("     size < 2*gap_min, not merely if it appears)")
    for n in psp[:5]:
        print("      PSP %d" % n)
    return psp


def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples
  scripts/psp_probe.py                     all three blocks, defaults (~5 s)
  scripts/psp_probe.py --fixtures          0.2 s - the two known gaps, only
  scripts/psp_probe.py --density --odds 20000000
                                           raise the upper bound ~3x (10x slower)
  scripts/psp_probe.py --density --x 7.9e28 --sieve 85000
                                           probe the FLEET wall - its chain runs
                                           --walk-primes 85000, not the engine
                                           default 150000
  scripts/psp_probe.py --density --x 2e20 --sieve 1e7
                                           probe the 2e20 HL anchor (its sieve
                                           depth is 1e7, not 150000)

reading the output
  (0) FIXTURES   which gaps the walk loses, and why: a pseudoprime that passes
                 the discovery test splits its gap into two halves.
  (1) CARMICHAEL how much weaker Euler/Jacobi (our production test) is than a
                 strong test on the class where it matters most.
  (2) DENSITY    how often a pseudoprime that SURVIVED the sieve appears at the
                 chosen scale.  The 95 % upper bound is the deliverable: with 0
                 found in N Euler passes it is 3/N.
""")
    ap.add_argument("--fixtures", action="store_true",
                    help="walk the interiors of the two known pseudoprime gaps "
                         "(710 and 956) with both tests and report which is lost "
                         "[included by default; this runs only the block]")
    ap.add_argument("--brik", action="store_true",
                    help="score candidate discovery tests against BrianK's "
                         "published COMPLETE list of >=800 pseudoprime gaps in "
                         "1e20..1e21 [included by default; runs only this block]")
    ap.add_argument("--sieve-class", action="store_true",
                    help="show which pseudoprimes a plain sieve of depth S "
                         "leaves behind (Chernick family) [included by default]")
    ap.add_argument("--carmichael", action="store_true",
                    help="Euler/Jacobi vs strong on Carmichael numbers, plus "
                         "Chernick cases that no sieve can remove "
                         "[included by default; this runs only the block]")
    ap.add_argument("--density", action="store_true",
                    help="measure how many base-2 Euler/Jacobi pseudoprimes "
                         "survive the sieve at --x [included by default; this "
                         "runs only the block]")
    ap.add_argument("--x", default="1e32",
                    help="start of the sampled window, parsed EXACTLY "
                         "(default 1e32; int(1e32) is NOT 10**32)")
    ap.add_argument("--sieve", type=int, default=150000,
                    help="sieve depth to simulate (default 150000 = the walk "
                         "engine's --walk-primes; the fleet wall runs 85000 and "
                         "the 2e20 corpus used sieve_limit 1e7)")
    ap.add_argument("--odds", type=int, default=2_000_000,
                    help="how many odd numbers to sample (default 2e6, ~4 s; "
                         "the 95 %% bound scales as 3/odds)")
    a = ap.parse_args(argv)
    both = not (a.fixtures or a.brik or a.sieve_class
                or a.carmichael or a.density)
    if both or a.fixtures:
        fixture_block(sieve_depth=a.sieve)
    if both:
        print()
    if both or a.brik:
        brik_block()
    if both or a.carmichael:
        carmichael_block(sieve_depth=a.sieve)
    if both:
        print()
    if both:
        print()
    if both or a.sieve_class:
        sieve_class_block()
    if both:
        print()
    if both or a.density:
        density_block(a.x, a.sieve, a.odds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
