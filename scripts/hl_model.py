#!/usr/bin/env python3
"""HL four-parameter consecutive-gap model - parameter access + calculators.

Reads the coefficient tables shipped in forum/ (Peter Williams, "Hardy Littlewood
Four Parameter Consecutive Gap Model", Rev 20260927):

    forum/hl_gap_4param_exact_parameters_with_first_occurrence.csv   g <= 3600, c1..c4 (+ first occurrence)
    forum/hl_gap_3param_exact_parameters_with_first_occurrence.csv   g <= 9990, c1..c3
    data/hl_order3_targets.csv                                       local order-3 extensions (optional;
                                                                     produced by scripts/hl_order3_targets.py)

Model (verified 2026-09-27 against an independent recomputation, ~1e-6 % on c1):

    rho_g(x) ~ S_g / L^2 * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4)),   L = ln x
    Y_g(L)   = 1 / rho_g ;  first occurrence = root of Y_g(L) = e^L
    natural local tail slope (sigma): 1 / (c1' + c2'/L + c3'/L^2 + c4'/L^3)

Relation to this repo's model: S_g = 2*C2 * h(g), i.e. the model's arithmetic
factor IS the h(D) modulation used by winnability_map.py; the c-series supplies
the natural exponential part (natural sigma ~ 0.982-0.989 at our L values, vs
the measured random-cover sigma 0.978).  A cover raises sigma above this
natural value (measured 1.26-1.37) - that lift is the cover's own lever.

CLI:
    python3 scripts/hl_model.py --selftest
    python3 scripts/hl_model.py --first-occ 6,64,3600,4224,9990
    python3 scripts/hl_model.py --sigma 527.2 --gap 3600
    python3 scripts/hl_model.py --verify-c1 6,30,18084,40462
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [
    ROOT / "forum" / "hl_gap_3param_exact_parameters_with_first_occurrence.csv",   # order 3, g <= 9990
    ROOT / "data" / "hl_order3_targets.csv",                                       # local order-3 extensions
    ROOT / "forum" / "hl_gap_4param_exact_parameters_with_first_occurrence.csv",   # order 4, g <= 3600 (wins on overlap)
]

_TAB = None


def _f(row, key):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def table(verbose=False):
    """Load and merge the tables -> dict gap -> record (higher order wins)."""
    global _TAB
    if _TAB is not None:
        return _TAB
    tab = {}
    for path in FILES:
        if not path.exists():
            if verbose:
                print(f"[hl] missing (skipped): {path}")
            continue
        n = 0
        for row in csv.DictReader(open(path, errors="ignore")):
            try:
                g = int(row["gap"])
            except (KeyError, ValueError):
                continue
            rec = {
                "S_g": _f(row, "S_g"),
                "c1": _f(row, "c1"), "c2": _f(row, "c2"),
                "c3": _f(row, "c3"), "c4": _f(row, "c4"),
                "first_occ_lnx": _f(row, "first_occurrence_ln_x"),
                "order": int(_f(row, "order") or 0),
                "source": path.name,
            }
            prev = tab.get(g)
            if prev is None or rec["order"] >= prev["order"]:
                tab[g] = rec
            n += 1
        if verbose:
            print(f"[hl] loaded {n} rows from {path.name}")
    _TAB = tab
    return tab


def params(gap):
    return table().get(gap)


def _solve_root(S, cs):
    """Root of phi(L) = 2 ln L - ln S + sum ck/L^k - L = 0 (phi decreasing for L >= 3)."""
    def phi(L):
        s = 2.0 * math.log(L) - math.log(S)
        for k, c in enumerate(cs, 1):
            if c:
                s += c / L ** k
        return s - L

    lo = 1.0
    hi = 10.0
    while phi(hi) > 0 and hi < 1e6:
        hi *= 2.0
    if phi(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def first_occurrence(gap):
    """ln x of the model first occurrence: stored column if present, else root solve."""
    rec = params(gap)
    if rec is None:
        return None
    if rec["first_occ_lnx"]:
        return rec["first_occ_lnx"]
    cs = [rec.get("c1") or 0.0, rec.get("c2") or 0.0, rec.get("c3") or 0.0, rec.get("c4") or 0.0]
    return _solve_root(rec["S_g"], cs)


def _slopes_at(gap, step=40):
    """Finite-difference slopes of c1..c4 around `gap`.

    A LARGE step is essential: c1(g) = g - delta(g) fluctuates between adjacent
    even gaps (delta jumps ~+-1.4), so 2-gap differences are arithmetic noise;
    a 40-gap backward difference averages it out (validated: sigma(527.2, 3600)
    = 0.982 with step 40 vs 0.63 with step 2).  Backward difference preferred
    (covers the order-4 table edge), forward fallback.
    """
    tab = table()
    rec = tab.get(gap)
    if rec is None:
        return None
    other = None
    d = 0
    for cand in (gap - step, gap + step):
        o = tab.get(cand)
        if o is not None and o["order"] == rec["order"]:
            other, d = o, cand - gap
            break
    if other is None:
        # sparse large-gap tables: fall back to the NEAREST same-order gap
        # (the delta-drift is slow, so a long-range difference still estimates
        # the local slope well enough for sigma)
        near = [(abs(g2 - gap), g2) for g2, r2 in tab.items()
                if r2["order"] == rec["order"] and g2 != gap]
        if not near:
            return None
        _, g2 = min(near)
        other, d = tab[g2], g2 - gap
    out = []
    for k in ("c1", "c2", "c3", "c4"):
        a, b = rec.get(k), other.get(k)
        out.append((b - a) / d if (a is not None and b is not None) else None)
    return out


def natural_sigma(L, gap):
    """Natural (uncovered) tail slope at scale L, from local c-slopes at `gap`."""
    sl = _slopes_at(gap)
    if sl is None:
        return None
    s = 0.0
    for k, v in enumerate(sl, 0):
        if v:
            s += v / L ** k
    return 1.0 / s if s else None


def natural_sigma_avg(L, gap_target, k=900, min_base=40, max_base=1200):
    """Natural sigma at L from a LONG-BASELINE difference of same-order rows.

    Stable reading near `gap_target`: among the k table rows nearest the
    target, take the pair (same order as the nearest row) with the LARGEST
    separation <= max_base (and >= min_base).  The reason is delta(g) noise:
    neighbouring rows differ by ~+-1.5 in delta, so a short baseline (the
    +-40 step inside `_slopes_at`, or the ~22-gap spacing of the local
    extension rows) reads wobble, not the law.  The long baseline makes both
    the wobble and the slow delta drift negligible: concrete numbers at our
    scales are 0.996 at L=1413 (target 30966; the local method read 1.054)
    and 0.984 at L=528.  Falls back to averaging the local slopes of the k
    nearest rows when the table around the target cannot give a baseline
    >= min_base (e.g. near the table's low edge).
    """
    tab = table()
    if not tab:
        return None
    near = sorted(tab, key=lambda g: abs(g - gap_target))[:max(2, k)]
    ref = tab[min(near, key=lambda g: abs(g - gap_target))]
    same = sorted(g for g in near if tab[g]["order"] == ref["order"])
    best = None
    for i, a in enumerate(same):
        for b in same[i + 1:]:
            sep = b - a
            if min_base <= sep <= max_base and (best is None or sep > best[0]):
                best = (sep, a, b)
    if best is not None:
        _, a, b = best
        s = 0.0
        for kk in ("c1", "c2", "c3", "c4"):
            va, vb = tab[a].get(kk), tab[b].get(kk)
            if va is not None and vb is not None:
                # same term order as natural_sigma: (dc_k/dg) / L^(k-1),
                # because dE/dm = L * dE/dg with E = SUM c_k/L^k
                s += ((vb - va) / (b - a)) / L ** (int(kk[1]) - 1)
        return 1.0 / s if s else None
    vals = [v for v in (natural_sigma(L, g) for g in near) if v]
    return (sum(vals) / len(vals)) if vals else None


def ln_weight(gap, L):
    """ln of the model density up to terms common to all gaps at the same L:
       ln S_g - (c1/L + c2/L^2 + c3/L^3 + c4/L^4).  Use for ranking."""
    rec = params(gap)
    if rec is None or rec["S_g"] is None:
        return None
    s = math.log(rec["S_g"])
    for k in ("c1", "c2", "c3", "c4"):
        c = rec.get(k)
        if c:
            s -= c / L ** int(k[1])
    return s


def density_ratio(g1, g2, L):
    w1, w2 = ln_weight(g1, L), ln_weight(g2, L)
    if w1 is None or w2 is None:
        return None
    return math.exp(w1 - w2)


# ----------------------------------------------------------------- independent c1 check

_PRIME_CACHE = {}


def _sieve_primes(pmax):
    s = bytearray([1]) * (pmax + 1)
    s[0] = s[1] = 0
    for i in range(2, int(pmax ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = b"\x00" * ((pmax - i * i) // i + 1)
    return [i for i in range(pmax + 1) if s[i]]


def _spf_sieve(n):
    spf = list(range(n + 1))
    i = 2
    while i * i <= n:
        if spf[i] == i:
            for j in range(i * i, n + 1, i):
                if spf[j] == j:
                    spf[j] = i
        i += 1
    return spf


def _odd_prime_divisors(x, spf):
    out = set()
    while x > 1:
        p = spf[x]
        if p != 2:
            out.add(p)
        while x % p == 0:
            x //= p
    return out


def _log_base(p):
    # f_base(p) = ((p-3)/(p-2)) * (p/(p-1)), valid whenever the offset set has
    # nu_T = 3 and nu_E = 2 (i.e. p divides none of h, g, g-h)
    return math.log((p - 3) / (p - 2)) + math.log(p / (p - 1))


def _log_ftrue(p, g, h):
    nuE = 1 if g % p == 0 else 2
    nuT = len({0, h % p, g % p})
    if nuT == p:
        return None                      # inadmissible tuple -> ratio 0
    return math.log((1 - nuT / p) / (1 - nuE / p)) + math.log(p / (p - 1))


def _prime_ctx(pmax):
    if pmax not in _PRIME_CACHE:
        primes = _sieve_primes(pmax)
        sb = 0.0
        for p in primes:
            if p >= 5:
                sb += _log_base(p)
        _PRIME_CACHE[pmax] = (primes, sb)
    return _PRIME_CACHE[pmax]


def verify_c1(gap, pmax=10 ** 7):
    """Independent c1 (= B1) recomputation via the correction-set method.

        ratio(g,h) = 2 * exp(SB + log f_true(3) + sum_{p | h or g or (g-h), p >= 5}
                                   (log f_true(p) - log_base(p)))

    SB = sum of log f_base over odd primes p >= 5, valid for every prime that is
    not a divisor of h, g or g-h (then nu_T = 3, nu_E = 2).  Omitting the p > pmax
    tail costs < 2e-7 relative at pmax = 1e7.  Validated against the slow direct
    implementation and the shipped tables (machine precision).
    """
    _, sb = _prime_ctx(pmax)
    spf = _spf_sieve(gap)
    gdivs = _odd_prime_divisors(gap, spf)
    B1 = 0.0
    for h in range(2, gap, 2):
        lf3 = _log_ftrue(3, gap, h)
        if lf3 is None:
            continue
        divs = gdivs | _odd_prime_divisors(h, spf) | _odd_prime_divisors(gap - h, spf)
        s = sb + lf3
        bad = False
        for p in divs:
            if p == 3:
                continue
            lf = _log_ftrue(p, gap, h)
            if lf is None:
                bad = True
                break
            s += lf - _log_base(p)
        if bad:
            continue
        B1 += 2.0 * math.exp(s)
    return B1


# ----------------------------------------------------------------- selftest

def selftest():
    ok = True
    tab = table(verbose=True)
    print(f"[hl] merged table: {len(tab)} gaps, range {min(tab)}..{max(tab)}")

    checks = [
        # (gap, c1 expected, first occurrence ln x expected)
        (6,    2.1648090870361826887,  2.469892468182313),
        (3600, 3589.5331160010124698, 65.973954868428905),
    ]
    for g, c1, lnx in checks:
        rec = params(g)
        if rec is None:
            print(f"  FAIL g={g}: missing"); ok = False; continue
        d1 = abs(rec["c1"] - c1) / c1
        r = first_occurrence(g)
        dr = abs(r - lnx)
        good = d1 < 1e-9 and dr < 0.02
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} g={g}: c1 dev {d1:.1e}  "
              f"first-occ {r:.6f} vs {lnx:.6f} (d={dr:.2e})")

    s = natural_sigma(527.2, 3600)
    good = s is not None and 0.975 < s < 0.995
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} natural_sigma(527.2, g=3600) = {s:.5f} "
          f"(expect ~0.982)")

    # root solver: recompute stored first occurrences from the stored c-series
    for g, lnx in ((64, 11.339), (210, 18.259), (3600, 65.974)):
        rec = params(g)
        cs = [rec.get(k) or 0.0 for k in ("c1", "c2", "c3", "c4")]
        r = _solve_root(rec["S_g"], cs)
        good = r is not None and abs(r - lnx) < 0.05
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} root-solve g={g}: {r:.4f} vs {lnx} "
              f"(d={abs(r - lnx):.2e})")

    # cross-check the two tables on an overlapping gap: order-3 vs order-4 c1 fields
    rec3 = params(4224)  # only in the 3-param table
    good = rec3 is not None and abs(rec3["c1"] - 4213.24) < 0.1
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} order-3 table g=4224 c1 = "
          f"{rec3['c1'] if rec3 else None} (expect ~4213.24)")

    # independent fast recomputation (correction-set method) of two known gaps
    for g in (6, 30):
        rec = params(g)
        v = verify_c1(g, pmax=10 ** 6)
        dev = abs(v - rec["c1"]) / rec["c1"]
        good = dev < 1e-6
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} verify_c1({g}) = {v:.9f} vs table "
              f"{rec['c1']:.9f} (dev {dev:.1e})")

    print("[hl] selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="HL 4-parameter model access",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--first-occ", dest="first_occ",
                    help="comma list of even gaps; print first-occurrence ln x / x / merit")
    ap.add_argument("--sigma", type=float, help="scale L (ln x) for the natural sigma")
    ap.add_argument("--gap", type=int, default=0,
                    help="reference gap for the finite-difference slopes (default: max in table)")
    ap.add_argument("--verify-c1", dest="verify_c1",
                    help="independent c1 recomputation (correction-set method) for a comma list of gaps")
    ap.add_argument("--pmax", type=int, default=10 ** 7,
                    help="prime bound for --verify-c1 (tail beyond costs <2e-7 rel at 1e7)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.verify_c1:
        for tok in args.verify_c1.split(","):
            g = int(tok)
            v = verify_c1(g, pmax=args.pmax)
            rec = params(g)
            if rec and rec.get("c1"):
                dev = abs(v - rec["c1"]) / rec["c1"]
                tag = "OK" if dev < 1e-6 else "CHECK"
                print(f"g={g:6d}: c1 = {v:.6f}   table {rec['c1']:.6f}   dev {dev:.2e}   [{tag}]")
            else:
                print(f"g={g:6d}: c1 = {v:.6f}   (not in table)")
        return
    if args.first_occ:
        for tok in args.first_occ.split(","):
            g = int(tok)
            rec = params(g)
            if rec is None:
                print(f"g={g}: not in table")
                continue
            L = first_occurrence(g)
            if L is None:
                print(f"g={g}: no root (c-series degenerate)")
                continue
            print(f"g={g:6d}: ln x = {L:10.4f}   x = {math.exp(min(L, 700)):.4e}   "
                  f"merit = {g / L:8.4f}   order {rec['order']} ({rec['source']})")
        return
    if args.sigma:
        g = args.gap or max(table())
        s = natural_sigma(args.sigma, g)
        print(f"natural sigma at L={args.sigma} (g={g}): {s:.5f}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
