#!/usr/bin/env python3
"""p0_bench.py - fixed-range benchmark harness for the Phase-0 walk engine.

Runs `bin/phase0_scan_gpu --engine walk` on a fixed range and parses the
summary line block (wall, end_to_end, tests, tests/jump, the GPU-event stage
split mark/walk/other, gaps reported, verification_failures).  With --bin-b
the arms alternate A B B A per repeat, so machine drift cannot fake an A/B
delta (the repo's ABBA convention).

The script is location independent: it resolves the repo/package root from
its own path, so it works from gapminer_v2/ and from the phase0/ share
package alike.

Examples:
    # fresh numbers at the campaign geometry (2e13, g1586, K=64):
    python3 scripts/p0_bench.py --length 2e13 --gap-min 1586 --reps 2

    # ABBA comparison of two builds (serial baseline vs P0_PIPE):
    python3 scripts/p0_bench.py --bin-b /tmp/p0_prepipe.bin \
        --length 2e13 --gap-min 1586 --label P0_PIPE

Raw results are appended to data/p0_bench_results.txt (use --no-out to skip).

Options:
    --bin PATH       binary A (default <root>/bin/phase0_scan_gpu)
    --bin-b PATH     optional binary B; arms run A B B A per repeat
    --start S        range start (default 133001070000720000000 = campaign)
    --length L       range length; accepts 2e13 or 20000000000000 (default 2e13)
    --blocks N       alternative: length = N * 1006632960 (one 30-block)
    --gap-min G      walk threshold (default 1586 = campaign)
    --walk-batch B   super-batch in blocks (default 64)
    --walk-primes P  optional sieve/item prime limit (tool default when omitted)
    --device D       CUDA device (default 0)
    --reps R         repeats per binary (default 1; ABBA = 2 runs per repeat)
    --label TAG      tag written into the results file
    --out FILE       results file (default <root>/data/p0_bench_results.txt)
    --no-out         print only, write nothing
    --timeout S      per-run timeout in seconds (default 7200)
"""
import argparse
import datetime
import hashlib
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 1006632960                      # 30 * 2^25 numbers per walk block

RX = {
    "wall":      re.compile(r"wall=([0-9.]+) s"),
    "ints":      re.compile(r"ints=([0-9]+)"),
    "e2e":       re.compile(r"end_to_end=([0-9.eE+-]+) ints/s"),
    "batches":   re.compile(r"gpu batches=([0-9]+)"),
    "tests":     re.compile(r"tests=([0-9]+)"),
    "tjump":     re.compile(r"tests/jump=([0-9.]+)"),
    "mark":      re.compile(r"mark=([0-9.]+) s \(([0-9.]+)%\)"),
    "walk":      re.compile(r"walk=([0-9.]+) s \(([0-9.]+)%\)"),
    "other":     re.compile(r"other=(-?[0-9.]+) s"),
    "gaps":      re.compile(r"gaps reported=([0-9]+)"),
    "vfail":     re.compile(r"verification_failures=([0-9]+)"),
}


def parse_len(s):
    try:
        return int(s)
    except ValueError:
        v = float(s)
        if abs(v - round(v)) > 1e-6 * max(1.0, abs(v)):
            sys.exit(f"--length {s}: non-integer")
        return int(round(v))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one(binary, args, no):
    cmd = [binary, "--engine", "walk",
           "--start", str(args.start), "--length", str(args.length),
           "--gap-min", str(args.gap_min), "--walk-batch", str(args.walk_batch),
           "--device", str(args.device), "--progress", "0"]
    if args.walk_primes:
        cmd += ["--walk-primes", str(args.walk_primes)]
    t0 = datetime.datetime.now()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    out = p.stdout + p.stderr
    r = {"rc": p.returncode}
    for k, rx in RX.items():
        m = rx.search(out)
        if m:
            r[k] = float(m.group(1))
    if "mark" in r and "walk" in r:
        r["mark_pct"] = float(RX["mark"].search(out).group(2))
        r["walk_pct"] = float(RX["walk"].search(out).group(2))
    r["secs"] = (datetime.datetime.now() - t0).total_seconds()
    return r


def fmt_run(tag, r):
    if r.get("rc", 1) != 0 or "wall" not in r:
        return f"  {tag:<22} FAILED rc={r.get('rc')} ({r.get('secs', 0):.1f} s)"
    s = (f"  {tag:<22} wall={r['wall']:8.2f} s  e2e={r['e2e']:.4e} ints/s"
         f"  tests={r.get('tests', 0):.0f}  tests/jump={r.get('tjump', 0):.2f}")
    if "mark" in r:
        s += (f"\n  {'':<22} stage: mark={r['mark']:.1f} s ({r.get('mark_pct', 0):.1f}%)"
              f"  walk={r['walk']:.1f} s ({r.get('walk_pct', 0):.1f}%)"
              f"  other={r.get('other', 0):.1f} s")
    if r.get("vfail", 0) or r.get("gaps", 0):
        s += f"\n  {'':<22} gaps={r.get('gaps', 0):.0f}  verification_failures={r.get('vfail', 0):.0f}"
    return s


def aggregate(runs):
    walls = [r["wall"] for r in runs if "wall" in r]
    e2es = [r["e2e"] for r in runs if "e2e" in r]
    if not walls:
        return None
    return {"n": len(walls), "wall_mean": statistics.mean(walls),
            "wall_min": min(walls), "wall_max": max(walls),
            "e2e_mean": statistics.mean(e2es) if e2es else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default=os.path.join(ROOT, "bin", "phase0_scan_gpu"))
    ap.add_argument("--bin-b", default=None)
    ap.add_argument("--start", default="133001070000720000000")
    ap.add_argument("--length", default="2e13")
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--gap-min", type=int, default=1586)
    ap.add_argument("--walk-batch", type=int, default=64)
    ap.add_argument("--walk-primes", type=int, default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "p0_bench_results.txt"))
    ap.add_argument("--no-out", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args(argv)

    args.length = args.blocks * BLOCK if args.blocks else parse_len(args.length)
    for b in (args.bin, args.bin_b):
        if b and not os.path.exists(b):
            sys.exit(f"binary not found: {b}")
    if args.reps < 1:
        sys.exit("--reps must be >= 1")

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"[p0_bench] {stamp}")
    print(f"[p0_bench] start={args.start} length={args.length} gap_min={args.gap_min} "
          f"walk_batch={args.walk_batch} reps={args.reps}")
    print(f"[p0_bench] A: {args.bin}  md5={md5(args.bin)}")
    if args.bin_b:
        print(f"[p0_bench] B: {args.bin_b}  md5={md5(args.bin_b)}")

    bins = [("A", args.bin)] + ([("B", args.bin_b)] if args.bin_b else [])
    runs = {t: [] for t, _ in bins}
    order = ["A", "B", "B", "A"] if args.bin_b else ["A"]
    for rep in range(args.reps):
        for tag in order:
            path = dict(bins)[tag]
            r = run_one(path, args, rep)
            runs[tag].append(r)
            print(fmt_run(f"[{tag} rep{rep + 1}.{len(runs[tag])}]", r))

    print("[p0_bench] summary")
    agg = {}
    for tag, _ in bins:
        a = aggregate(runs[tag])
        agg[tag] = a
        if a:
            print(f"  {tag}: n={a['n']}  wall mean={a['wall_mean']:.2f} s "
                  f"(min {a['wall_min']:.2f}, max {a['wall_max']:.2f})"
                  + (f"  e2e mean={a['e2e_mean']:.4e} ints/s" if a["e2e_mean"] else ""))
    if args.bin_b and agg.get("A") and agg.get("B") and agg["A"]["e2e_mean"] and agg["B"]["e2e_mean"]:
        d = 100.0 * (agg["B"]["e2e_mean"] / agg["A"]["e2e_mean"] - 1.0)
        print(f"  B vs A: e2e delta = {d:+.2f}%  (wall {agg['B']['wall_mean']:.2f} vs "
              f"{agg['A']['wall_mean']:.2f} s)")

    if not args.no_out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(f"# p0_bench {stamp}"
                     + (f" label={args.label}" if args.label else "") + "\n")
            fh.write(f"# start={args.start} length={args.length} gap_min={args.gap_min} "
                     f"walk_batch={args.walk_batch} device={args.device} reps={args.reps}\n")
            for tag, path in bins:
                fh.write(f"# bin_{tag}={path} md5={md5(path)}\n")
            for tag, _ in bins:
                for i, r in enumerate(runs[tag]):
                    fh.write(f"{tag} run{i + 1} rc={r.get('rc')}"
                             + "".join(f" {k}={r[k]:.6g}" for k in
                                       ("wall", "e2e", "tests", "tjump", "mark", "walk",
                                        "other", "gaps", "vfail") if k in r)
                             + "\n")
                a = agg[tag]
                if a:
                    fh.write(f"{tag} mean wall={a['wall_mean']:.3f}"
                             + (f" e2e={a['e2e_mean']:.6g}" if a["e2e_mean"] else "") + "\n")
            if args.bin_b and agg.get("A") and agg.get("B") and agg["A"]["e2e_mean"] and agg["B"]["e2e_mean"]:
                d = 100.0 * (agg["B"]["e2e_mean"] / agg["A"]["e2e_mean"] - 1.0)
                fh.write(f"delta_B_vs_A_e2e_pct={d:+.2f}\n")
            fh.write("\n")
        print(f"[p0_bench] appended to {args.out}")


if __name__ == "__main__":
    main()
