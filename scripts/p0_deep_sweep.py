#!/usr/bin/env python3
"""p0_deep_sweep.py - walk-engine stage split vs sieve depth P, at a band.

Answers "can the sieve go deeper?" with measurements instead of a model: the
walk engine runs two pipelined stages (GPU mark into the class-30 bitmap, then
the batched jump-walk with the primality tests) and reports both stage spans
plus end-to-end ints/s.  The stage that is the longer one is the pacer, so:

  mark span < walk span -> deeper P (more primes, fewer tests) is nearly free
  mark span > walk span -> deeper P is a loss; the mark has to get cheaper first

Two arms per depth:

  prod      the production geometry (--gap-min G): both stages live
  markonly  --gap-min 10000000: the walk finds nothing, so the reported mark
            span is the mark alone (uncontaminated by pipeline overlap)

Same range, same --walk-batch, same binary for every depth, so the numbers are
comparable; the gap set reported at each depth must be IDENTICAL to the
reference depth's (a deeper sieve only removes tests, never changes which
numbers are reported), and the script fails if it is not.

Usage:
    python3 scripts/p0_deep_sweep.py --start 79000000000050000000000000000 \
        --length 4e12 --gap-min 1664

Note: this needs the GPU to itself - stop any running slice first.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p0_bench import RX, md5, parse_len          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKONLY_GAPMIN = 10000000      # walk cannot report anything at this gap


def gapset(log_path):
    """(lower, gap) pairs from a walk log; same two line forms as p0_pgs_bench."""
    out = set()
    if not os.path.exists(log_path):
        return out
    with open(log_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) >= 4 and f[0].isdigit():          # ts lower gap merit ...
                out.add((int(f[1]), int(f[2])))
            elif "gap=" in line:                        # GAP merit=... gap=...
                d = dict(x.split("=", 1) for x in f if "=" in x)
                try:
                    out.add((int(d["lower"]), int(d["gap"])))
                except (KeyError, ValueError):
                    pass
    return out


def run_depth(binary, a, depth, arm, log_path):
    gap_min = MARKONLY_GAPMIN if arm == "markonly" else a.gap_min
    cmd = [binary, "--engine", "walk", "--start", str(a.start),
           "--length", str(a.length), "--gap-min", str(gap_min),
           "--walk-batch", str(a.walk_batch), "--walk-primes", str(depth),
           "--device", str(a.device), "--progress", "0", "--log", log_path,
           "--no-records"]
    t0 = datetime.datetime.now()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") + (e.stderr or ""), 99
    r = {"rc": rc, "depth": depth, "arm": arm, "gap_min": gap_min,
         "secs": (datetime.datetime.now() - t0).total_seconds(),
         "cmd": " ".join(cmd)}
    for k, rx in RX.items():
        m = rx.search(out)
        if m:
            r[k] = float(m.group(1))
            if k in ("mark", "walk"):
                r[k + "_pct"] = float(rx.search(out).group(2))
    if rc not in (0, 99) and "wall" not in r:
        r["err"] = out.strip().splitlines()[-1][:200] if out.strip() else "no output"
    return r


def fmt(r, base_e2e=None):
    if "wall" not in r:
        return f"  {r['depth']:>9} {r['arm']:<8} FAILED rc={r['rc']} {r.get('err', '')}"
    s = (f"  {r['depth']:>9} {r['arm']:<8} wall={r['wall']:7.2f} s"
         f"  e2e={r.get('e2e', 0):.4e} ints/s")
    if base_e2e:
        s += f"  ({100 * (r['e2e'] / base_e2e - 1):+5.1f}%)"
    if "mark" in r:
        pacer = "mark" if r["mark"] > r.get("walk", 0) else "walk"
        s += (f"\n  {'':>9} {'':<8} mark={r['mark']:7.2f} s ({r.get('mark_pct', 0):4.1f}%)"
              f"  walk={r.get('walk', 0):7.2f} s ({r.get('walk_pct', 0):4.1f}%)"
              f"  other={r.get('other', 0):6.2f} s  pacer={pacer}")
    if "tests" in r and r["tests"]:
        s += (f"\n  {'':>9} {'':<8} tests={r['tests']:.4g}  tests/jump={r.get('tjump', 0):.3f}"
              f"  gaps={r.get('gaps', 0):.0f}  vfail={r.get('vfail', 0):.0f}")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(ROOT, "bin", "phase0_scan_gpu"))
    ap.add_argument("--start", default="79000000000050000000000000000")
    ap.add_argument("--length", default="4e12")
    ap.add_argument("--gap-min", type=int, default=1664)
    ap.add_argument("--walk-batch", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--depths", default="15000,40000,100000,250000,500000,1000000")
    ap.add_argument("--ref-depth", type=int, default=None,
                    help="depth whose gap set is the reference (default: the "
                         "depth closest to the production default 40000)")
    ap.add_argument("--arms", default="prod,markonly")
    ap.add_argument("--logs-dir", default="/tmp/p0_deep_sweep")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "p0_deep_sweep_results.txt"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    args.length = parse_len(args.length)
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    arms = [x for x in args.arms.split(",") if x.strip()]
    ref = args.ref_depth or min(depths, key=lambda d: abs(d - 40000))
    os.makedirs(args.logs_dir, exist_ok=True)

    print(f"[sweep] start={args.start} length={args.length} gap_min={args.gap_min} "
          f"walk_batch={args.walk_batch} ref_depth={ref}")
    print(f"[sweep] binary={args.bin} md5={md5(args.bin)}")
    runs, sets = [], {}
    for arm in arms:
        for d in depths:
            log = os.path.join(args.logs_dir, f"{arm}_d{d}.log")
            r = run_depth(args.bin, args, d, arm, log)
            r["gapset_n"] = len(gapset(log))
            sets[(arm, d)] = gapset(log)
            runs.append(r)
            print(fmt(r), flush=True)

    print("\n[sweep] gap-set check (deeper sieve = fewer tests, same results):")
    ok = True
    for arm in arms:
        refset = sets.get((arm, ref), set())
        for d in depths:
            s = sets.get((arm, d), set())
            same = s == refset
            ok &= same
            print(f"  {arm:<8} P={d:>9}: {len(s):5d} gaps  "
                  f"{'identical to ref' if same else 'DIFFERENT from ref'}"
                  f"{'' if d == ref else ''}"
                  + ("" if same or d == ref else
                     f"  missing={sorted(refset - s)[:3]} extra={sorted(s - refset)[:3]}"))

    base = next((r for r in runs if r["arm"] == "prod" and r["depth"] == ref
                 and "e2e" in r), None)
    print(f"\n[sweep] speed vs P={ref} (production arm):")
    for r in runs:
        if r["arm"] == "prod" and "e2e" in r and base and r is not base:
            print(f"  P={r['depth']:>9}: {100 * (r['e2e'] / base['e2e'] - 1):+6.2f}%"
                  f"   (mark {r['mark']:.2f} s vs walk {r['walk']:.2f} s)"
                  f"   {'MARK is the pacer' if r['mark'] > r['walk'] else 'walk (tests) is the pacer'}")

    if not args.no_write:
        with open(args.out, "a") as fh:
            fh.write(f"# p0_deep_sweep {datetime.datetime.now().isoformat(timespec='seconds')}"
                     f"  bin={args.bin} md5={md5(args.bin)}\n")
            fh.write(f"# start={args.start} length={args.length} gap_min={args.gap_min} "
                     f"walk_batch={args.walk_batch} arms={','.join(arms)} "
                     f"gap_set_identical={'yes' if ok else 'NO'}\n")
            for r in runs:
                fh.write(f"{r['arm']} P={r['depth']} rc={r['rc']} gapset={r['gapset_n']}"
                         + "".join(f" {k}={r[k]:.6g}" for k in
                                   ("wall", "e2e", "tests", "tjump", "mark", "mark_pct",
                                    "walk", "walk_pct", "other") if k in r) + "\n")
            fh.write("\n")
        print(f"[sweep] appended to {args.out}")
    print(f"[sweep] JSON: {json.dumps([{k: v for k, v in r.items() if k != 'cmd'} for r in runs])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
