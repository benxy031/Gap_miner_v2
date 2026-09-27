#!/usr/bin/env python3
"""Order-3 HL coefficient extension for arbitrary gap lengths (our hunt targets).

The shipped Peter Williams tables stop at g = 3600 (order 4) / g = 9990 (order 3);
our record-target lengths (18k-40k) are beyond both.  His derivation driver,
forum/derive_hl_gap_4param_exact_90090.py, computes exact c1..c3 for ANY gap with
--order 3 (cost Theta(n^3), n = g/2 - 1; the author's notes say "feasible to
90090").  This wrapper:

  * runs the derivation for a list of gaps (default: our current target lengths),
  * appends the rows to data/hl_order3_targets.csv,
  * summarizes FIRST OCCURRENCES (root of Y_g(L) = e^L) for those lengths.

data/hl_order3_targets.csv is auto-merged by scripts/hl_model.py, so the results
become visible to `winnability_map.py --hl` (annotation mode).

USAGE
    python3 scripts/hl_order3_targets.py                      # default target list
    python3 scripts/hl_order3_targets.py --gaps 18084,40462
    python3 scripts/hl_order3_targets.py --dry-run            # plan + cost only
    python3 scripts/hl_order3_targets.py --summarize          # table only
    python3 scripts/hl_order3_targets.py --selftest           # g=4224 smoke + table cross-check

The derivation runs niceness 15 by default so an active GPU hunt on the same box
keeps its host thread.

Dependencies: numba, mpmath (pip install --user --break-system-packages numba mpmath).
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DERIVE = ROOT / "forum" / "derive_hl_gap_4param_exact_90090.py"
OUT = ROOT / "data" / "hl_order3_targets.csv"
SELFTEST_OUT = ROOT / "data" / "hl_order3_selftest.csv"

# default target list = our own record-relevant lengths:
#   18084 = shift-720 rate-ranked top target (TARGET_LENGTHS)
#   21224 = the fleet m45 block length (2026-09-25)
#   34596 = shift-1792 rate-ranked top target
#   40462 = the shift-1784 hunt record (2026-09-26, merit 28.6289)
DEFAULT_GAPS = "18084,21224,34596,40462"


def parse_gaps(spec):
    out = []
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1, 2))
        else:
            out.append(int(tok))
    return sorted(set(g for g in out if g >= 4 and g % 2 == 0))


def triple_cost(gaps):
    """Order-3 cost units: sum n^3/6 with n = g/2 - 1 (the triples enumerated)."""
    return sum(((g // 2 - 1) ** 3) / 6.0 for g in gaps)


def check_deps(python):
    r = subprocess.run([python, "-c", "import numba, mpmath; print('deps ok')"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[hl3] FATAL: numba/mpmath missing for", python)
        print("      install: python3 -m pip install --user --break-system-packages numba mpmath")
        return False
    print("[hl3]", r.stdout.strip())
    return True


def run_derive(python, gaps, output, threads, nice, extra=()):
    cmd = []
    if nice is not None:
        cmd += ["nice", "-n", str(nice)]
    cmd += [python, str(DERIVE),
            "--gaps", ",".join(str(g) for g in gaps),
            "--order", "3",
            "--output", str(output),
            "--threads", str(threads),
            "--yes", *extra]
    print("[hl3] running:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    dt = time.time() - t0
    print(f"[hl3] derivation finished rc={rc} in {dt/60:.1f} min", flush=True)
    return rc, dt


def _load_rows(path):
    rows = {}
    if not Path(path).exists():
        return rows
    for r in csv.DictReader(open(path, errors="ignore")):
        try:
            rows[int(r["gap"])] = r
        except (KeyError, ValueError):
            continue
    return rows


def summarize(output=OUT, gaps=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hl_model

    ours = _load_rows(output)
    if not ours:
        print(f"[hl3] no rows in {output} yet")
        return 1
    ref3 = _load_rows(ROOT / "forum" / "hl_gap_3param_exact_parameters_with_first_occurrence.csv")
    ref4 = _load_rows(ROOT / "forum" / "hl_gap_4param_exact_parameters_with_first_occurrence.csv")

    print(f"[hl3] {len(ours)} gap(s) in {Path(output).name}")
    print(f"  {'gap':>9} {'c1':>13} {'c2':>14} {'c3':>15} {'first-occ ln x':>14} "
          f"{'merit@fo':>9} {'ref-check':>10}")
    for g in sorted(ours):
        r = ours[g]
        c1, c2, c3 = (float(r["c1"]), float(r["c2"]), float(r["c3"]))
        Lg = hl_model.first_occurrence(g)
        mer = g / Lg if Lg else float("nan")
        ref = ""
        for tag, ref_rows in (("4p", ref4), ("3p", ref3)):
            rr = ref_rows.get(g)
            if rr:
                dev = abs(c1 - float(rr["c1"])) / float(rr["c1"])
                ref = f"{tag} c1dev {dev:.1e}"
                break
        Ls = f"{Lg:14.3f}" if Lg else f"{'-':>14}"
        print(f"  {g:>9} {c1:>13.3f} {c2:>14.3f} {c3:>15.3f} {Ls} {mer:>9.3f} {ref:>10}")
    print("  note: first occurrence = root of Y_g(L)=e^L from c1..c3 (c4 unknown at order 3;")
    print("        c4/L^4 ~ 1e-2 at these L, negligible).  True first occurrences sit FAR")
    print("        below the hunt scale - records are 'smallest known', not true firsts.")
    return 0


def selftest(python, threads, nice):
    """Run g=4224 order 3 and cross-check against the shipped 3-param table."""
    rc, _ = run_derive(python, [4224], SELFTEST_OUT, threads, nice, extra=("--recompute",))
    if rc != 0:
        print("[hl3] selftest FAILED (derivation rc != 0)")
        return 1
    rows = _load_rows(SELFTEST_OUT)
    ref = _load_rows(ROOT / "forum" / "hl_gap_3param_exact_parameters_with_first_occurrence.csv")
    if 4224 not in rows or 4224 not in ref:
        print("[hl3] selftest FAILED (missing rows)")
        return 1
    g = 4224
    ok = True
    for k in ("c1", "c2", "c3"):
        a, b = float(rows[g][k]), float(ref[g][k])
        dev = abs(a - b) / abs(b) if b else abs(a)
        good = dev < 1e-8
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'} {k}: {a:.6f} vs shipped {b:.6f} (dev {dev:.1e})")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hl_model
    Lg = hl_model.first_occurrence(g)
    Lref = float(ref[g]["first_occurrence_ln_x"])
    good = abs(Lg - Lref) < 0.05
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'} first-occ: {Lg:.6f} vs shipped {Lref:.6f}")
    print("[hl3] selftest", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gaps", default=DEFAULT_GAPS,
                    help=f"comma list and/or ranges (default: {DEFAULT_GAPS})")
    ap.add_argument("--output", default=str(OUT))
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that has numba+mpmath (default: current)")
    ap.add_argument("--nice", type=int, default=15, help="nice level (default 15; -1 to disable)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--assume-tput", type=float, default=9.0e8, dest="tput",
                    help="assumed triples/s for the dry-run wall estimate "
                         "(measured 9.2e8 on g=4224 smoke, 8 threads, nice 15)")
    args = ap.parse_args()
    nice = None if args.nice < 0 else args.nice

    if args.summarize:
        sys.exit(summarize(args.output))

    gaps = parse_gaps(args.gaps)
    if not gaps:
        ap.error("no even gaps parsed")

    if args.dry_run:
        print(f"[hl3] derive script: {DERIVE}  (exists: {DERIVE.exists()})")
        print(f"[hl3] target gaps: {gaps}")
        tc = triple_cost(gaps)
        print(f"[hl3] order-3 cost: {tc:.3e} triples  ->  est. {tc/args.tput/60:.1f} min "
              f"at {args.tput:.1e} triples/s (nice {nice}, {args.threads} threads)")
        check_deps(args.python)
        return
    if args.selftest:
        sys.exit(selftest(args.python, args.threads, nice))

    if not check_deps(args.python):
        sys.exit(2)
    rc, _ = run_derive(args.python, gaps, Path(args.output), args.threads, nice)
    if rc == 0:
        summarize(args.output)
    sys.exit(rc)


if __name__ == "__main__":
    main()
