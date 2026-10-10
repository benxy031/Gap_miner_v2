#!/usr/bin/env python3
"""p0_pgs_bench.py - head-to-head: our walk engine vs PGS (briankehrig/prime-gaps-cuda).

Protocol is the one used in docs/PHASE0_scan_bench.md section 17: same box, same
start, identical ranges for both tools.  Wall for us = process wall; wall for PGS
= the interval between the "Start time" and "End time" lines of their report
(their printed "Speed" meter runs 1.5-2x above that, so only report times are
comparable).  Both include their own setup.

    python3 scripts/p0_pgs_bench.py [--pgs-dir /tmp/pgcuda] [--start 2e20]
        [--thresholds 900,1260] [--lengths 1e12,1e13] [--out data/p0_pgs_bench_results.txt]
        [--no-pgs | --no-ours] [--no-audit] [--warmup/--no-warmup]

Both tools are threshold searches here: they report every gap >= minGap with no
completeness claim below it.  Our walk engine tests with MR + GMP-verifies every
reported gap; PGS uses a Perig Fermat test (their PSP caveat starts below 1200).

PGS settings per threshold are their own recommendation plus the values that ran
clean on this 8 GB card (their auto BLOCK_SIZE OOMs):
    minGap  900 -> WORD_LENGTH 120, BLOCK_SIZE 60e9
    minGap 1260 -> WORD_LENGTH 240, BLOCK_SIZE 140e9
settings.json is edited in place and restored at exit.
"""
import argparse
import datetime as dt
import math
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "bin", "phase0_scan_gpu")
PGS_PROFILE = {900: (120, 60000000000), 1260: (240, 140000000000)}
PGS_FALLBACK = (240, 140000000000)


def secs(text):
    """'2e20' / '1e13' / '10000000000000' -> int."""
    return int(float(text.replace("_", "")))


def e12(n):
    assert n % 10 ** 12 == 0, "%d is not a multiple of 1e12" % n
    return n // 10 ** 12


def run_ours(start, length, min_gap, tmp, timeout=1800):
    """(wall seconds, {(gap, lower prime)}, log path)."""
    log = os.path.join(tmp, "ours_%d_%d.log" % (min_gap, length))
    state = log + ".state"
    for f in (log, log + ".out", state):
        if os.path.exists(f):
            os.remove(f)
    cmd = [BIN, "--engine", "walk", "--start", str(start), "--length", str(length),
           "--gap-min", str(min_gap), "--walk-batch", "64", "--no-records",
           "--state", state, "--log", log, "--progress", "120"]
    t0 = time.monotonic()
    with open(log + ".out", "w") as fh:
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise RuntimeError("our scanner did not finish within %d s" % timeout)
    wall = time.monotonic() - t0
    if rc != 0:
        raise RuntimeError("our scanner failed rc=%d, see %s.out" % (rc, log))
    return wall, gaps_in(log), log


def gaps_in(log):
    out = set()
    if os.path.exists(log):
        with open(log) as fh:
            for line in fh:
                m = re.search(r"GAP .*?gap=(\d+) lower=(\d+)", line)
                if m:
                    out.add((int(m.group(1)), int(m.group(2))))
    return out


def run_sieve(start, length, min_gap, tmp):
    """Sieve engine (full enumeration) pointed at the same gap threshold.

    Its threshold is ceil(merit_min * ln(start)), so the merit that lands exactly
    on min_gap is (min_gap - 0.5)/ln(start) - the half gap absorbs the ceil.
    The wall therefore includes work a threshold search never does (every prime,
    every gap), which is the point of the measurement, not an accident of it.
    """
    merit = (min_gap - 0.5) / math.log(start)
    log = os.path.join(tmp, "sieve_%d_%d.log" % (min_gap, length))
    state = log + ".state"
    for f in (log, log + ".out", state):
        if os.path.exists(f):
            os.remove(f)
    cmd = [BIN, "--engine", "sieve", "--start", str(start), "--length", str(length),
           "--merit-min", "%.6f" % merit, "--no-records",
           "--state", state, "--log", log, "--progress", "120"]
    t0 = time.monotonic()
    with open(log + ".out", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.monotonic() - t0
    if rc != 0:
        raise RuntimeError("the sieve engine failed rc=%d, see %s.out" % (rc, log))
    return wall, gaps_in(log), log


def pgs_settings(pgs_dir, min_gap):
    wl, block = PGS_PROFILE.get(min_gap, PGS_FALLBACK)
    path = os.path.join(pgs_dir, "settings.json")
    text = open(path).read()
    text = re.sub(r'"WORD_LENGTH":\s*-?\d+', '"WORD_LENGTH": %d' % wl, text)
    text = re.sub(r'"BLOCK_SIZE":\s*-?\d+', '"BLOCK_SIZE": %d' % block, text)
    open(path, "w").write(text)
    return wl, block


def pgs_report_path(pgs_dir, start, length, min_gap):
    a, b = e12(start), e12(start + length)
    return os.path.join(pgs_dir, "reports", "GapReport_%de12_%de12_%d.txt" % (a, b, min_gap))


def parse_pgs_report(path):
    """(their report interval in seconds, {(gap, lower prime)}) from a GapReport file."""
    text = open(path).read()
    stamp = re.search(r"Start time: (.*) UTC\nEnd time: (.*) UTC", text)
    if not stamp:
        raise RuntimeError("no Start/End time lines in %s" % path)
    fmt = "%Y-%b-%d %H:%M:%S.%f"
    interval = (dt.datetime.strptime(stamp.group(2), fmt)
                - dt.datetime.strptime(stamp.group(1), fmt)).total_seconds()
    gaps = set()
    for line in text.splitlines():
        m = re.match(r"^(\d+)\s+[\d.]+\s+(\d+)\s*$", line)
        if m:
            gaps.add((int(m.group(1)), int(m.group(2))))
    return interval, gaps


def run_pgs(pgs_dir, start, length, min_gap, timeout=1800):
    """(report interval seconds, process wall, {(gap, lower prime)}, report path)."""
    for d in ("logs", "unknowns", "reports"):
        os.makedirs(os.path.join(pgs_dir, d), exist_ok=True)
    wl, block = pgs_settings(pgs_dir, min_gap)
    open(os.path.join(pgs_dir, "worktodo.txt"), "w").write(
        "%d,%d,%d\n" % (e12(start), e12(start + length), min_gap))
    report = pgs_report_path(pgs_dir, start, length, min_gap)
    if os.path.exists(report):
        os.remove(report)
    t0 = time.monotonic()
    # their wrapper needs a tty (it draws progress), so run it under script(1)
    try:
        rc = subprocess.run(["script", "-qec", "python3 main.py", "/dev/null"],
                            cwd=pgs_dir, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        raise RuntimeError("PGS did not finish within %d s" % timeout)
    wall = time.monotonic() - t0
    if rc != 0 or not os.path.exists(report):
        raise RuntimeError("PGS failed rc=%d, no report at %s" % (rc, report))
    interval, gaps = parse_pgs_report(report)
    return interval, wall, gaps, (wl, block)


def busy():
    out = subprocess.run(["pgrep", "-af", "phase0_scan_gpu|prime_gaps"],
                         capture_output=True, text=True).stdout.splitlines()
    return [l for l in out if "p0_pgs_bench" not in l]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pgs-dir", default="/tmp/pgcuda")
    p.add_argument("--start", default="2e20")
    p.add_argument("--thresholds", default="900,1260")
    p.add_argument("--lengths", default="1e12,1e13")
    p.add_argument("--out", default=os.path.join(REPO, "data", "p0_pgs_bench_results.txt"))
    p.add_argument("--tmp", default="/tmp/p0_pgs_bench")
    p.add_argument("--no-pgs", action="store_true")
    p.add_argument("--no-ours", action="store_true")
    p.add_argument("--no-audit", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--with-sieve", action="store_true",
                   help="also time the sieve engine (full enumeration) at the same "
                        "threshold - a different task, reported separately")
    a = p.parse_args()

    start = secs(a.start)
    ths = [int(x) for x in a.thresholds.split(",")]
    lens = [secs(x) for x in a.lengths.split(",")]
    if sorted(lens) == lens:
        lens = lens[::-1]                      # long runs first: cleanest steady state
    os.makedirs(a.tmp, exist_ok=True)
    running = busy()
    if running:
        print("WARNING: other GPU tools are running, the numbers will be low:")
        for line in running:
            print("   " + line)

    backup = None
    if not a.no_pgs:
        backup = os.path.join(a.tmp, "settings.json.orig")
        shutil.copy(os.path.join(a.pgs_dir, "settings.json"), backup)
        if not a.no_warmup:                    # triggers their one-off nvcc build
            print("warm-up (compile + first run) ...", flush=True)
            run_pgs(a.pgs_dir, start, 10 ** 12, ths[0])

    rows, audit, sieve_rows = [], {}, {}
    lines = []
    try:
        for g in ths:
            for ln in lens:
                rec = {"min_gap": g, "length": ln}
                if not a.no_ours:
                    wall, gaps, log = run_ours(start, ln, g, a.tmp)
                    rec["ours_s"], rec["ours_gaps"] = wall, len(gaps)
                    audit[(g, ln)] = audit.get((g, ln), {})
                    audit[(g, ln)]["ours"] = gaps
                    print("ours  minGap %-6d %-7s %8.2f s  %8.3f s/e12  %s" % (
                        g, "%.0e" % ln, wall, wall / (ln / 1e12), log), flush=True)
                if not a.no_pgs:
                    iv, wall, gaps, (wl, bl) = run_pgs(a.pgs_dir, start, ln, g)
                    rec["pgs_s"], rec["pgs_wall"], rec["pgs_gaps"] = iv, wall, len(gaps)
                    rec["pgs_wl"], rec["pgs_block"] = wl, bl
                    audit[(g, ln)] = audit.get((g, ln), {})
                    audit[(g, ln)]["pgs"] = gaps
                    print("PGS   minGap %-6d %-7s %8.2f s  %8.3f s/e12  (WL=%d BLOCK=%.0fe9)"
                          % (g, "%.0e" % ln, iv, iv / (ln / 1e12), wl, bl / 1e9), flush=True)
                if a.with_sieve:
                    wall, gaps, log = run_sieve(start, ln, g, a.tmp)
                    sieve_rows[(g, ln)] = wall
                    audit[(g, ln)] = audit.get((g, ln), {})
                    audit[(g, ln)]["sieve"] = gaps
                    print("sieve minGap %-6d %-7s %8.2f s  %8.3f s/e12  %s" % (
                        g, "%.0e" % ln, wall, wall / (ln / 1e12), log), flush=True)
                rows.append(rec)
    finally:
        if backup:
            shutil.copy(backup, os.path.join(a.pgs_dir, "settings.json"))

    # steady-state decomposition: wall = N x steady + S, solved from the two lengths
    print("\nSTEADY STATE (from the two lengths, wall = N*steady + S)")
    hdr = ("%-8s %10s %10s %12s %12s %10s" %
           ("minGap", "ours s/e12", "PGS s/e12", "ours ints/s", "PGS ints/s", "winner"))
    print(hdr)
    lines.append(hdr)
    for g in ths:
        o = {r["length"]: r for r in rows if r["min_gap"] == g and "ours_s" in r}
        s = {r["length"]: r for r in rows if r["min_gap"] == g and "pgs_s" in r}
        if len(o) < 2 or len(s) < 2:
            continue
        big, small = max(lens), min(lens)
        k = (big - small) / 10 ** 12
        o_st = (o[big]["ours_s"] - o[small]["ours_s"]) / k
        s_st = (s[big]["pgs_s"] - s[small]["pgs_s"]) / k
        o_r, s_r = 10 ** 12 / o_st, 10 ** 12 / s_st
        win = "ours %.2fx" % (s_r / o_r) if o_r > s_r else "PGS %.2fx" % (o_r / s_r)
        row = ("%-8d %10.2f %10.2f %12.3e %12.3e %10s" % (g, o_st, s_st, o_r, s_r, win))
        print(row)
        lines.append(row)
        print("   startup: ours %.2f s, PGS %.2f s   (%.1f B/s = 1e9 ints/s)"
              % (o[big]["ours_s"] - o_st * big / 10 ** 12,
                 s[big]["pgs_s"] - s_st * big / 10 ** 12, o_r / 1e9))

    sieve_line = None
    for g in ths:
        if not sieve_rows:
            break
        big, small = max(lens), min(lens)
        if (g, big) not in sieve_rows or (g, small) not in sieve_rows:
            continue
        k = (big - small) / 10 ** 12
        st = (sieve_rows[(g, big)] - sieve_rows[(g, small)]) / k
        r = 10 ** 12 / st
        print("   sieve engine (full enumeration, same threshold): %.2f s/e12, "
              "%.3e ints/s, startup %.2f s" % (
                  st, r, sieve_rows[(g, big)] - st * big / 10 ** 12))
        sieve_line = (sieve_line or "") + "sieve minGap %d: %.2f s/e12 (%.3e ints/s) " % (
            g, st, r)
    if sieve_line:
        lines.append(sieve_line.strip())

    if not a.no_audit:
        print("\nCROSS-TOOL AUDIT (equal output sets, not just equal speed)")
        for key in sorted(audit):
            d = audit[key]
            if len(d) < 2:
                continue
            names = sorted(d)
            ref = d[names[0]]
            parts = ["%s %d" % (n, len(d[n])) for n in names]
            same = ["%s/%s %d" % (names[0], n, len(ref & d[n])) for n in names[1:]]
            msg = ("minGap %-6d %.0e: %s | identical as (gap, lower prime): %s"
                   % (key[0], key[1], ", ".join(parts), ", ".join(same)))
            print(msg)
            lines.append(msg)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write("=== p0_pgs_bench %s  start=%d  host=%s\n"
                     % (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        start, os.uname().nodename))
            fh.write("\n".join(lines) + "\n\n")
        print("\nappended to %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
