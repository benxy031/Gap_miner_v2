#!/usr/bin/env python3
"""p0_record_forecast.py — which records are still available at a fixed scan scale.

WHY
---
On an EXHAUSTIVE (uncovered) walk of a range around x = e^L, every gap of length
g the scanner finds has merit

    m(g) = g / L

fixed by the RANGE, not by us — and constant to ~1e-12 over a whole campaign
(the range spans dL/L ~ 1e-12, so the merit does not move).  Therefore "can this
length still be recorded?" is DETERMINISTIC:

    win(g)  <=>  table_merit(g) < g / L

The only randomness left is WHICH winnable length actually shows up in the range,
at the Hardy-Littlewood rate of that exact length,

    rho_g(L) = (S_g / L^2) * exp(-(c1/L + c2/L^2 + c3/L^3 + c4/L^4))
    E[gaps of exactly g in R integers] = rho_g(L) * R

so the expected number of NEW records over a remaining stretch R is

    E[new(R)] = SUM_g win(g) * (1 - exp(-rho_g(L) * R))

This is the complement of scripts/record_planner.py: that tool plans a COVERED
hunt, where L is a DESIGN CHOICE (merit is free, the currency is exp(E(m))
primality tests per candidate, one target at a time).  This tool reads an
EXHAUSTIVE scan, where L is GIVEN by the range (merit comes free with every gap,
the currency is integers scanned, and all reachable lengths are collected at
once).  Run both before choosing a campaign shape.

The report also carries the MARGIN (our merit - table merit), i.e. how durable a
record is: below ~0.13 it is expected to be broken by the next table refresh
(same convention as record_planner.py), so a large count of low-margin records
buys count, not permanence.

Usage:
    scripts/p0_record_forecast.py                     # live wall chain
    scripts/p0_record_forecast.py --r-total 2e17      # forecast the rest of the chain
    scripts/p0_record_forecast.py --x 7.9e28 --top 20
    scripts/p0_record_forecast.py --selftest
"""
import argparse
import glob
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import hl_model as hm                                                    # noqa: E402
import p0_walk_hl_compare as pwc                                         # noqa: E402

DEFAULT_GLOB = os.path.join(REPO, "data/p0_walk_wall*_g*.log")
DEFAULT_TABLE = os.path.join(REPO, "data/prime_gap_merits.txt")
FRAGILE, HYPER_FRAGILE = 0.13, 0.05        # merit margins, see record_planner.py


def load_table(path):
    """{gap: best published merit}."""
    tab = {}
    with open(path, errors="ignore") as fh:
        for line in fh:
            p = line.split(None, 2)
            if len(p) < 2:
                continue
            try:
                g, m = int(p[0]), float(p[1])
            except ValueError:
                continue
            if m > 0:
                tab.setdefault(g, m)
    return tab


def rho(gap, L, rec):
    """Hardy-Littlewood density of gaps of exactly `gap` at scale L."""
    if rec is None:
        return 0.0
    s = math.log(rec["S_g"]) - 2.0 * math.log(L)
    for k in (1, 2, 3, 4):
        c = rec.get("c%d" % k)
        if c:
            s -= c / L ** k
    return math.exp(s)


def scan_state(patterns):
    """(R, x_lo, x_hi, won, slices) from the walk logs; won = {gap: (x, merit)}."""
    logs = []
    for pat in patterns:
        logs += glob.glob(pat)
    logs = sorted(set(logs))
    R = 0
    x_lo = None
    x_hi = None
    won = {}
    slices = []
    for log in logs:
        head, gaps = pwc.read_slice(log)
        cov, how = pwc.coverage(log)
        tag, _ = pwc.slice_tag(log)
        a, b = head["start"], head["start"] + cov
        x_lo = a if x_lo is None else min(x_lo, a)
        x_hi = b if x_hi is None else max(x_hi, b)
        R += cov
        for g in gaps:
            if g["record"] and g["verified"]:
                prev = won.get(g["gap"])
                if prev is None or g["lower"] < prev[0]:
                    won[g["gap"]] = (g["lower"], g["merit"])
        slices.append((tag, a, b, cov, how, len(gaps)))
    return R, x_lo, x_hi, won, slices


def analyze(patterns, table_path=DEFAULT_TABLE, L=None, x=None, r_total=None,
            gmin=1664, emin=1.0e-4):
    """Everything the report and the compact line derive from (L, table, R).

    Pure - no printing - so the walk chain can import it and print its own
    summary without parsing the report text.
    """
    tab = load_table(table_path)
    if not tab:
        raise ValueError("no usable rows in %s" % table_path)
    R, x_lo, x_hi, won, slices = scan_state(patterns)
    # the scale can come from --L, from --x, or from the scanned logs - in that
    # order; --L alone must work even when no log matches
    if L is None:
        if x is not None:
            L = math.log(x)
        elif x_lo is not None:
            L = math.log(0.5 * (x_lo + x_hi))
        else:
            raise ValueError("cannot infer the scan scale: pass L= (or x=) or a "
                             "matching log glob")
    if x is not None and abs(math.log(x) - L) > 1e-6:
        raise ValueError("x and L disagree (ln x = %.9f vs L = %.9f)"
                         % (math.log(x), L))

    hltab = hm.table()
    allg = sorted(g for g in hltab if g % 2 == 0)
    gate = sum(rho(g, L, hltab[g]) for g in allg) * L
    tabbed = sorted(g for g in tab if g % 2 == 0 and g >= gmin)
    RH = {g: rho(g, L, hltab.get(g)) for g in tabbed}    # rho once; it dominates
    win = [g for g in tabbed if tab[g] < g / L]
    winset = set(win)
    R_rem = (r_total - R) if r_total else 0.0

    def e_cnt(rows, Rr):
        return sum(-math.expm1(-RH[g] * Rr) for g in rows)

    open_win = [g for g in win if g not in won]
    return dict(tab=tab, hltab=hltab, allg=allg, gate=gate, L=L, x=math.exp(L),
                R=R, x_lo=x_lo, x_hi=x_hi, won=won, slices=slices, gmin=gmin,
                tabbed=tabbed, RH=RH, winnable=win, winset=winset,
                lose=[g for g in tabbed if g not in winset],
                reach=[g for g in win if RH[g] * max(R_rem, 1.0e16) > emin],
                R_rem=R_rem, emin=emin, e_cnt=e_cnt, open_win=open_win,
                e_scanned=e_cnt(win, R), e_open=e_cnt(open_win, R_rem),
                per1e16=e_cnt(win, 1.0e16))


def open_targets(an):
    """(gap, our merit, table merit, margin, ln x-gain, E[count]) per reachable
    length we have NOT won, best yield first."""
    L, tab, RH = an["L"], an["tab"], an["RH"]
    Rref = max(an["R_rem"], 1.0e16)
    rows = []
    for g in an["reach"]:
        if g in an["won"]:
            continue
        mo = g / L
        mt = tab[g]
        rows.append((g, mo, mt, mo - mt, g / mt - L, RH[g] * Rref))
    rows.sort(key=lambda r: -r[5])
    return rows


def fragility(an):
    """([(gap, margin, verdict)] for the records held, how many are fragile)."""
    L, tab = an["L"], an["tab"]
    out = []
    for g in sorted(an["won"]):
        mt = tab.get(g)
        if mt is None:
            out.append((g, float("nan"), "not in table"))
            continue
        marg = g / L - mt
        out.append((g, marg, "permanent" if marg >= 1.0 else
                    "durable" if marg >= FRAGILE else
                    "FRAGILE" if marg >= HYPER_FRAGILE else "HYPER-FRAGILE"))
    nf = sum(1 for _, m, _ in out if m == m and 0 <= m < FRAGILE)
    return out, nf


def compact(an, top=3, rate=2.254e11):
    """The summary the walk chain prints after every finished slice."""
    won, es, er = len(an["won"]), an["e_scanned"], an["e_open"]
    sd = math.sqrt(max(es, 1e-9))
    line = ("[chain] forecast: won=%d  model=%.1f (%+.1f sigma)"
            % (won, es, (won - es) / sd))
    if an["R_rem"] > 0:
        line += ("  E[new in the remaining %.3e]=%.1f -> %.1f h/record"
                 % (an["R_rem"], er, an["R_rem"] / rate / er / 3600.0))
    else:
        line += ("  E[new per 1e16]=%.2f -> %.1f h/record"
                 % (an["per1e16"], 1.0e16 / rate / max(an["per1e16"], 1e-9) / 3600.0))
    out = [line]
    rows = open_targets(an)
    if rows:
        _, nf = fragility(an)
        out.append("[chain] forecast: top open: %s | fragile %d/%d"
                   % ("  ".join("%d (%s, E=%.2f)" % (g, gain_str(lng), e)
                                for g, _, _, _, lng, e in rows[:top]), nf, won))
    return out


def report(args):
    an = analyze(args.glob, table_path=args.table, L=args.L, x=args.x,
                 r_total=args.r_total, gmin=args.gmin, emin=args.emin)
    tab, hltab, allg = an["tab"], an["hltab"], an["allg"]
    R, won, L, slices = an["R"], an["won"], an["L"], an["slices"]

    print("scale      : x = %.6e   L = ln x = %.4f" % (an["x"], L))
    print("scan       : R = %.6e integers over %d slice(s), %d won length(s)"
          % (R, len(slices), len(won)))
    for tag, a, b, cov, how, n in slices:
        print("   %-14s [%.4e .. +%.4e]  cov=%.3e (%s)  events=%d"
              % (tag, a, b - a, cov, how, n))

    # ---- gate: the tabulated rows must reproduce the gap density 1/L ----
    print("gate       : sum_g rho_g / (1/L) = %.5f  (1.0 = one gap per prime, "
          "%d tabulated even gaps)" % (an["gate"], len(allg)))

    # ---- the deterministic winnable set --------------------------------
    gmin, tabbed, winset, win = an["gmin"], an["tabbed"], an["winset"], an["winnable"]
    lose, RH, reach, R_rem = an["lose"], an["RH"], an["reach"], an["R_rem"]
    print("\nDETERMINISTIC  win(g) <=> table_merit(g) < g/L =", end=" ")
    print("merit %.6f at g = %d,%.6f at g = %d"
          % (gmin / L, gmin, (gmin + 2000) / L, gmin + 2000))
    print("  tabulated even lengths >= %d : %d" % (gmin, len(tabbed)))
    print("  winnable                      : %d" % len(win))
    print("  NOT winnable (table >= g/L)   : %d" % len(lose))
    print("  reachable (E[count] > %.0e in max(R_rem,1e16)): %d"
          % (args.emin, len(reach)))

    e_new = an["e_cnt"]
    open_win = an["open_win"]
    print("\nRECORD FORECAST")
    print("  won so far (record=NEW in the logs)     : %d" % len(won))
    print("  E[records] in the scanned R             : %.1f   (model vs actual %d)"
          % (e_new(win, R), len(won)))
    per1e16 = an["per1e16"]
    print("  E[records] per 1e16 integers scanned    : %.2f   -> %.1f h/record at "
          "%.3e ints/s" % (per1e16, 1.0e16 / args.rate / per1e16 / 3600.0, args.rate))
    if R_rem > 0:
        en = an["e_open"]
        print("  E[NEW records] in the remaining %.3e : %.1f" % (R_rem, en))
        print("     -> %.1f h/record over that stretch (vs %.1f h/record so far)"
              % (R_rem / args.rate / en / 3600.0,
                 R / args.rate / max(len(won), 1) / 3600.0))
        print("     -> the rate FALLS because the densest winnable lengths are won first")

    # ---- open targets ---------------------------------------------------
    rows = open_targets(an)
    print("\nOPEN TARGETS (reachable, not yet won) — sorted by expected count%s"
          % (" in R_remaining" if R_rem > 0 else " in 1e16"))
    print("   %-7s %9s %9s %8s %12s %8s" %
          ("gap", "our merit", "table", "margin", "x gain", "E[count]"))
    for g, mo, mt, marg, lng, e in rows[:args.top]:
        print("   %-7d %9.4f %9.4f %8.4f %12s %8.3f"
              % (g, mo, mt, marg, gain_str(lng), e))
    tot = sum(1.0 - math.exp(-an["RH"][g] * max(R_rem, 1.0e16)) for g, *_ in rows)
    print("   ... %d open reachable lengths, E[open records] = %.1f" % (len(rows), tot))

    # ---- jackpots --------------------------------------------------------
    jack = [r for r in rows if r[4] > math.log(args.gain_min)]
    jack.sort(key=lambda r: -r[4])
    print("\nJACKPOTS (x gain > %.0fx, still open): %d lengths" % (args.gain_min, len(jack)))
    if jack:
        def p_hit(g):
            return 1.0 - math.exp(-RH[g] * max(R_rem, 1.0e16))
        print("   available frontier (best gain you can realistically land):")
        for pmin in (1e-2, 1e-3, 1e-4):
            cand = [r for r in jack if p_hit(r[0]) >= pmin]
            if not cand:
                continue
            g, mo, mt, marg, lng, e = max(cand, key=lambda r: r[4])
            print("     P(hit) >= %-6.0e -> %-10s  gap %-5d (P=%.1e, %d lengths)"
                  % (pmin, gain_str(lng), g, p_hit(g), len(cand)))
        print("   top by gain:")
        print("   %-7s %9s %9s %14s %11s" %
              ("gap", "our merit", "table", "x gain", "P(hit)"))
        for g, mo, mt, marg, lng, e in jack[:args.top]:
            print("   %-7d %9.4f %9.4f %14s %11.2e"
                  % (g, mo, mt, gain_str(lng), p_hit(g)))
        ej = sum(p_hit(g) for g, *_ in jack)
        print("   E[jackpot records] = %.2f  (of E[open] = %.1f)" % (ej, tot))
    else:
        print("   none")

    # ---- durability of what we hold --------------------------------------
    if won:
        frag, nf = fragility(an)
        print("\nDURABILITY of the %d records held (margin = our merit - table)"
              % len(won))
        print("   %-7s %9s %9s %8s %-14s" %
              ("gap", "our merit", "table", "margin", "verdict"))
        for g, marg, verdict in frag:
            print("   %-7d %9.4f %9s %8s %s"
                  % (g, g / L, ("%.4f" % tab[g]) if g in tab else "-",
                     ("%.4f" % marg) if marg == marg else "-", verdict))
        print("   %d of %d break on any small improvement (margin < %.2f)"
              % (nf, len(won), FRAGILE))

    if args.selftest:
        return selftest(L, tab, hltab, allg, win, RH, won, R, e_new(win, R),
                        an["gate"] / L)
    return 0


def gain_str(log_gain):
    """x multiplier from its natural log, always marked as a gain.

    Every value ends in 'x' or starts with '10^', so a column of these can never
    be mistaken for a margin (both are bare numbers otherwise, and a parser -
    or a reader - cannot tell 48.2 from 0.0482-style margins).
    """
    d = log_gain / math.log(10.0)
    if d > 6:
        return "10^%.1f" % d
    return "%.4gx" % math.exp(log_gain)


def selftest(L, tab, hltab, allg, win, RH, won, R, e_pred, s_all):
    print("\nSELFTEST")
    bad = 0

    def chk(name, ok, detail=""):
        nonlocal bad
        bad += 0 if ok else 1
        print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))

    winset = set(win)
    chk("hl_model table non-empty", len(allg) > 1000, "%d even gaps" % len(allg))
    chk("density gate  sum rho_g ~= 1/L", abs(s_all * L - 1.0) < 1e-3,
        "%.5f" % (s_all * L))
    # every length the report calls winnable really is one, and nothing else is
    mism = [g for g in tab
            if g % 2 == 0 and g >= 1664 and ((tab[g] < g / L) != (g in winset))]
    chk("win(g) partition of the table", not mism,
        "" if not mism else "%d mismatches (e.g. %s)" % (len(mism), mism[:3]))
    # the forecast must be the sum of per-length Poisson probabilities
    s_direct = sum(RH[g] for g in win)
    chk("sum rho_g(winnable) > 0", s_direct > 0, "%.6e" % s_direct)
    e_check = sum(-math.expm1(-RH[g] * R) for g in win)
    chk("E[records] reproducible", abs(e_check - e_pred) < 1e-9 * max(e_pred, 1),
        "%.4f vs %.4f" % (e_check, e_pred))
    # model vs actual, Poisson
    n = len(won)
    sd = math.sqrt(max(e_pred, 1e-9))
    chk("model vs actual records within 3 sigma",
        abs(n - e_pred) < 3 * sd + 1.0,
        "predicted %.1f, actual %d (%.1f sigma)" % (e_pred, n, (n - e_pred) / sd))
    # E[new] must fall as already-won lengths are removed, and be <= E[all]
    e_open = sum(-math.expm1(-RH[g] * R) for g in win if g not in won)
    chk("E[open] <= E[all]", e_open <= e_pred + 1e-9,
        "%.4f <= %.4f" % (e_open, e_pred))
    # win(g) is "table_merit < our merit", so a strictly LARGER our-merit can only
    # add winners.  Scale the merit explicitly (not 1/L, which is easy to flip).
    win_bigger = set(g for g in tab if g % 2 == 0 and g >= 1664
                     and tab[g] < 1.001 * g / L)
    chk("winnable set monotone in the merit", winset <= win_bigger,
        "|merit|=%d, |1.001*merit|=%d" % (len(winset), len(win_bigger)))
    # rho_g is NOT monotone in g (S_g oscillates with g mod small primes), but
    # its windowed mean must fall - that is the tail the census uses
    ok = True
    prev = None
    for G in range(1700, 2600, 100):
        win_rho = [rho(g, L, hltab[g]) for g in allg if G <= g < G + 100]
        m = sum(win_rho) / max(len(win_rho), 1)
        if prev is not None and m > prev:
            ok = False
        prev = m
    chk("windowed mean rho_g falls with g", ok)
    print("  %s" % ("all checks passed" if not bad else "%d FAILED" % bad))
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--glob", action="append", default=None,
                    help="walk log glob (repeatable); default = the wall chain")
    ap.add_argument("--L", type=float, default=None, help="ln x of the scan")
    ap.add_argument("--x", type=float, default=None, help="scan scale (e^L)")
    ap.add_argument("--r-total", type=float, default=None,
                    help="total integers the campaign plans to scan; the forecast "
                         "for R_remaining = R_total - R_scanned is then printed "
                         "(the wall chain is 20 slices x 1e16 = 2e17)")
    ap.add_argument("--rate", type=float, default=2.254e11,
                    help="scan rate in ints/s (default: measured 2.254e11)")
    ap.add_argument("--gmin", type=int, default=1664,
                    help="smallest length the scanner logs (default 1664)")
    ap.add_argument("--emin", type=float, default=1e-4,
                    help="reachability floor on E[count] (default 1e-4)")
    ap.add_argument("--gain-min", type=float, default=1000.0,
                    help="x-multiplier that counts as a jackpot (default 1000)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--compact", action="store_true",
                    help="print only the two summary lines the walk chain uses")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    a.glob = a.glob or [DEFAULT_GLOB]
    try:
        if a.compact:
            an = analyze(a.glob, table_path=a.table, L=a.L, x=a.x,
                         r_total=a.r_total, gmin=a.gmin, emin=a.emin)
            for line in compact(an, top=a.top, rate=a.rate):
                print(line)
            return 0
        return report(a)
    except ValueError as exc:                    # analyze() reports this way
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
