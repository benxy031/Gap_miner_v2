#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
watch_gap_hunt_records.py — automatic record checker for the GAP_HUNT log.

Follows the GAP_HUNT results file (or any log containing its records),
compares every gap against the best-known-merit table
(data/prime_gap_merits.txt, from primegaps.cloudygo.com) and appends every
NEW RECORD to a separate records file.

Record criterion (same as the miner's record_log.c):
    gap present in the table AND merit > best-known merit for that gap.
A higher merit than the known first occurrence means the gap was found at a
SMALLER size — i.e. a new first occurrence.

Usage:
    scripts/watch_gap_hunt_records.py <log-file> [records-out] [options]

    log-file      GAP_HUNT results file (`<gap> <merit> <startprime>` per
                  line) or the stderr log (`[GAP_HUNT] gap=... merit=...
                  start=... end=...` lines).  Both formats are parsed; the
                  format is auto-detected per line.
    records-out   file to append records to (default: gap_hunt_records_found.txt)

Options:
    --once        process the file once and exit (no following)
    --table PATH  merit table (default: data/prime_gap_merits.txt)

Record line format (one per line):
    gap merit startprime best_known_merit yyyy-mm-ddTHH:MM:SS
"""

import sys
import os
import re
import time
import datetime


def load_table(path):
    """Return {gap: best_known_merit} from the merits table."""
    table = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                gap = int(parts[0])
                merit = float(parts[1])
            except ValueError:
                continue
            table[gap] = merit
    return table


RE_STDERR = re.compile(
    r"\[GAP_HUNT\]\s+gap=(\d+)\s+merit=([0-9.]+)\s+start=(\d+)")
RE_OUT = re.compile(r"^\s*(\d+)\s+([0-9.]+)\s+(\d+)\s*$")


def parse_line(line):
    """Return (gap, merit, start_str) or None."""
    m = RE_STDERR.search(line)
    if m:
        return int(m.group(1)), float(m.group(2)), m.group(3)
    m = RE_OUT.match(line)
    if m:
        return int(m.group(1)), float(m.group(2)), m.group(3)
    return None


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2

    once = False
    table_path = "data/prime_gap_merits.txt"
    out_path = "gap_hunt_records_found.txt"
    log_path = None

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--once":
            once = True
        elif a == "--table" and i + 1 < len(args):
            i += 1
            table_path = args[i]
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        elif log_path is None:
            log_path = a
        elif out_path == "gap_hunt_records_found.txt":
            out_path = a
        else:
            print(f"unknown argument: {a}", file=sys.stderr)
            return 2
        i += 1

    if not log_path:
        print("missing log file argument", file=sys.stderr)
        return 2

    table = load_table(table_path)
    print(f"[watch] table: {len(table)} gaps from {table_path}")
    print(f"[watch] log: {log_path}  records out: {out_path}"
          + ("  (follow mode, Ctrl+C to stop)" if not once else "  (once)"))

    checked = 0
    records = 0
    start_str = ""

    def process_line(line, f_out):
        nonlocal checked, records
        rec = parse_line(line)
        if rec is None:
            return
        gap, merit, start_str = rec
        checked += 1
        if gap not in table:
            return  # unknown gap: not a record (miner convention)
        best = table[gap]
        if merit > best:
            records += 1
            stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            print(f"[RECORD] gap={gap} merit={merit:.6f} "
                  f"best_known={best:.6f} start={start_str[:32]}...")
            with open(f_out, "a") as fo:
                fo.write(f"{gap} {merit:.6f} {start_str} {best:.6f} {stamp}\n")
                fo.flush()
            table[gap] = merit  # don't re-log the same gap

    if once:
        with open(log_path, "r") as f:
            for line in f:
                process_line(line.rstrip("\n"), out_path)
        print(f"[watch] done: {checked} records checked, {records} new records")
        return 0

    # Follow mode (tail -F-like, tolerates truncation/rotation).
    while True:
        try:
            with open(log_path, "r") as f:
                if os.path.getsize(log_path) > 0:
                    f.seek(0, os.SEEK_END)
                while True:
                    line = f.readline()
                    if line:
                        process_line(line.rstrip("\n"), out_path)
                    else:
                        time.sleep(0.2)
        except FileNotFoundError:
            time.sleep(1.0)
        except KeyboardInterrupt:
            break
    print(f"[watch] stopped: {checked} records checked, {records} new records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
