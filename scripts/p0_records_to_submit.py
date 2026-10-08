#!/usr/bin/env python3
"""
p0_records_to_submit.py

Turns phase0 walk gap-logs into a submit-ready batch file, continuously.

The walk engine appends one line per gap >= --gap-min to its --log file and
flushes every line, so a record is visible within seconds:

    <epoch> <lower> <gap> <merit> <upper> verified=<0|1> record=<comment> table=<best>

`record=NEW` means: the gap length exists in the local merits snapshot AND the
reported merit beats it (criterion identical to new_src/record_log.c).  This
tool harvests those lines, keeps the strongest number per gap length (smallest
start prime => highest true merit), recomputes the merit as gap/ln(lower)
(the site's convention, same as scripts/verify_gap_candidate.py) and regenerates

    records_to_submit_phase0.txt        # gap merit prime_start

Every row is independently re-verified (OpenSSL endpoints + strict scan of all
odd intermediates) before it is allowed into the data section, and rows listed
in the sent-file are excluded so a POSTed batch is never re-posted.

The file is a pure function of (logs, sent-file, verify-cache), so it can be
regenerated at any time, a watcher restart is harmless, and losing the state
file costs nothing.

Usage:
  python3 scripts/p0_records_to_submit.py --once             # single pass, exit
  python3 scripts/p0_records_to_submit.py                    # watch, 30 s poll
  python3 scripts/p0_records_to_submit.py --once --dry-run   # print, no write
  python3 scripts/p0_records_to_submit.py --check-live       # flag stale rows
  python3 scripts/p0_records_to_submit.py --mark-sent records_to_submit_phase0.txt

After a successful POST, mark the batch so the pending file no longer offers it:
  python3 scripts/p0_records_to_submit.py --mark-sent records_to_submit_phase0.txt
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOGS = ["data/p0_*.log"]
DEFAULT_MAX_LOG_MB = 64.0
DEFAULT_OUT = "records_to_submit_phase0.txt"
DEFAULT_SENT = "data/p0_records_sent.txt"
DEFAULT_STATE = "data/p0_records_watch.state"
DEFAULT_VCACHE = "data/p0_records_verify.json"
VERIFY_TOOL = "scripts/verify_gap_candidate.py"
MERITS_URL = "https://primegaps.cloudygo.com/merits.txt"
UA = "gapminer_v2-p0-records/1.0"

# <epoch> <lower> <gap> <merit> <upper> verified=<0|1> record=<...> table=<...>
LOG_RE = re.compile(
    r"^(?P<epoch>\d{9,})\s+(?P<lower>\d+)\s+(?P<gap>\d+)\s+(?P<merit>\d+\.\d+)"
    r"\s+(?P<upper>\d+)\s+verified=(?P<verified>[01])\s+record=(?P<record>\w+)"
    r"\s+table=(?P<table>\S+)\s*$"
)

_STOP = False
_WARNED_SKIPS: set[str] = set()


def log(msg: str) -> None:
    print(f"[p0 records] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[p0 records] WARNING: {msg}", file=sys.stderr, flush=True)


def utc_str(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def rel(path: str) -> str:
    rel_path = os.path.relpath(path, REPO)
    return path if rel_path.startswith("..") else rel_path


def slice_tag(path: str) -> str:
    name = os.path.basename(path)
    for suffix in (".log", ".out"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("p0_walk_"):
        name = name[len("p0_walk_"):]
    return re.sub(r"_g\d+$", "", name)


def write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ── log harvesting ──────────────────────────────────────────────────────────

def scan_logs(patterns: list[str], max_mb: float) -> tuple[list[dict], list[str]]:
    paths: list[str] = []
    for pattern in patterns:
        full = resolve(pattern)
        hits = sorted(globmod.glob(full)) if any(c in full for c in "*?[") \
            else ([full] if os.path.exists(full) else [])
        for hit in hits:
            if hit not in paths:
                paths.append(hit)

    kept: list[str] = []
    events: list[dict] = []
    for path in paths:
        try:
            size_mb = os.path.getsize(path) / 1e6
        except OSError as exc:
            warn(f"cannot stat {rel(path)}: {exc}")
            continue
        if max_mb and size_mb > max_mb:
            if path not in _WARNED_SKIPS:
                _WARNED_SKIPS.add(path)
                warn(f"skipping {rel(path)} ({size_mb:.0f} MB > --max-log-mb "
                     f"{max_mb:.0f}); pass an explicit --logs or a higher limit "
                     f"if it holds records")
            continue
        kept.append(path)
        try:
            with open(path, errors="replace") as fh:
                for line in fh:
                    m = LOG_RE.match(line)
                    if not m or m.group("record") != "NEW":
                        continue
                    table = m.group("table")
                    events.append({
                        "log": rel(path),
                        "slice": slice_tag(path),
                        "epoch": int(m.group("epoch")),
                        "gap": int(m.group("gap")),
                        "lower": int(m.group("lower")),
                        "upper": int(m.group("upper")),
                        "engine_merit": float(m.group("merit")),
                        "verified": int(m.group("verified")),
                        "table": float(table) if table[0].isdigit() else None,
                    })
        except OSError as exc:
            warn(f"cannot read {rel(path)}: {exc}")
    return events, kept


def merge(events: list[dict]) -> list[dict]:
    """One row per gap length: the smallest start prime (highest true merit)."""
    rows: dict[int, dict] = {}
    for ev in events:
        row = rows.get(ev["gap"])
        if row is None:
            row = rows[ev["gap"]] = {
                "gap": ev["gap"], "events": 0, "unverified_events": 0,
                "slices": set(), "first_epoch": ev["epoch"],
                "last_epoch": ev["epoch"], "table": None,
                "lower": ev["lower"], "upper": ev["upper"],
                "engine_merit": ev["engine_merit"],
            }
        row["events"] += 1
        row["unverified_events"] += 1 - ev["verified"]
        row["slices"].add(ev["slice"])
        row["first_epoch"] = min(row["first_epoch"], ev["epoch"])
        row["last_epoch"] = max(row["last_epoch"], ev["epoch"])
        if ev["table"] is not None:
            row["table"] = ev["table"] if row["table"] is None \
                else max(row["table"], ev["table"])
        if ev["lower"] < row["lower"]:
            row["lower"] = ev["lower"]
            row["upper"] = ev["upper"]
            row["engine_merit"] = ev["engine_merit"]
    for row in rows.values():
        row["merit"] = row["gap"] / math.log(row["lower"])
        row["merit6"] = round(row["merit"], 6)
        # x-gain: how much smaller the record's start prime is than the one the
        # local table's merit implies (same convention merit = gap/ln x).
        row["x_gain"] = "?"
        if row["table"]:
            try:
                gain = math.exp(row["gap"] * (1.0 / row["table"] - 1.0 / row["merit"]))
                row["x_gain"] = f"{gain:.2f}x"
            except (OverflowError, ZeroDivisionError):
                pass
    return sorted(rows.values(), key=lambda r: -r["merit"])


# ── independent verification ────────────────────────────────────────────────

def load_json(path: str) -> dict:
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_json(path: str, data: dict) -> None:
    try:
        write_atomic(path, json.dumps(data, indent=1, sort_keys=True) + "\n")
    except OSError as exc:
        warn(f"cannot write {rel(path)}: {exc}")


def verify_rows(rows: list[dict], cache_path: str, timeout: float,
                reverify: bool) -> None:
    """Annotate rows with row['verify'] = 'PASS' | 'FAIL' | 'PENDING'."""
    cache = {} if reverify else load_json(cache_path)
    now = time.time()
    dirty = False

    for row in rows:
        key = f"{row['gap']}:{row['lower']}"
        entry = cache.get(key)
        if isinstance(entry, dict):
            if entry.get("ok") is True:
                row["verify"] = "PASS"
                row["verify_note"] = entry.get("note", "")
                continue
            if entry.get("ok") is False:
                row["verify"] = "FAIL"
                row["verify_note"] = entry.get("note", "")
                continue
            if entry.get("retry_at", 0) > now:            # no verdict yet
                row["verify"] = "PENDING"
                row["verify_note"] = entry.get("note", "")
                continue

        started = time.time()
        output, note, verdict = "", "", None
        try:
            proc = subprocess.run(
                [sys.executable, resolve(VERIFY_TOOL), str(row["gap"]),
                 f"{row['merit6']:.6f}", str(row["lower"])],
                cwd=REPO, capture_output=True, text=True, timeout=timeout)
            output = (proc.stdout or "") + (proc.stderr or "")
            if "PASS:" in output:
                verdict, note = True, "PASS"
            elif "FAIL" in output or proc.returncode not in (0, 1):
                verdict, note = False, (f"exit={proc.returncode} "
                                        f"{output.strip().splitlines()[-1][:80]}"
                                        if output.strip() else
                                        f"exit={proc.returncode}")
            else:
                note = "no verdict"
        except subprocess.TimeoutExpired:
            note = f"timeout after {timeout:.0f}s"
        except OSError as exc:
            note = f"cannot run {VERIFY_TOOL}: {exc}"

        elapsed = time.time() - started
        if verdict is None:
            cache[key] = {"ok": None, "note": note, "retry_at": time.time() + 3600}
            row["verify"], row["verify_note"] = "PENDING", note
            warn(f"gap={row['gap']} verification inconclusive ({note}); "
                 f"keeping row out of the batch")
        else:
            cache[key] = {"ok": verdict, "note": note,
                          "when": utc_str(int(time.time())),
                          "seconds": round(elapsed, 1)}
            row["verify"], row["verify_note"] = ("PASS" if verdict else "FAIL"), note
            log(f"verify gap={row['gap']} x={row['lower']} -> "
                f"{'PASS' if verdict else 'FAIL'} ({elapsed:.1f}s)")
            if not verdict:
                warn(f"gap={row['gap']} x={row['lower']} FAILED independent "
                     f"verification ({note}) — excluded from the batch")
        dirty = True

    if dirty:
        save_json(cache_path, cache)


# ── live table (optional) ───────────────────────────────────────────────────

def fetch_live() -> dict[int, tuple[float, str]]:
    req = urllib.request.Request(MERITS_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "replace")
    live: dict[int, tuple[float, str]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                live[int(parts[0])] = (float(parts[1]),
                                       parts[2] if len(parts) > 2 else "?")
            except ValueError:
                pass
    return live


# ── sent-file ───────────────────────────────────────────────────────────────

def split_row(line: str) -> tuple[int, int] | None:
    """(gap, start_prime) from 'gap prime' or 'gap merit prime' (+ inline comment)."""
    parts = line.split("#")[0].split()
    if len(parts) < 2:
        return None
    try:
        return (int(parts[0]), int(parts[2] if len(parts) >= 3 else parts[1]))
    except ValueError:
        return None


def load_sent(path: str) -> set[tuple[int, int]]:
    sent: set[tuple[int, int]] = set()
    if not os.path.exists(path):
        return sent
    with open(path, errors="replace") as fh:
        for line in fh:
            key = split_row(line)
            if key:
                sent.add(key)
    return sent


def mark_sent(batch_path: str, sent_path: str, rows: list[dict]) -> int:
    """Append the batch file's rows to the sent-file; return rows added."""
    keys = set()
    if os.path.exists(batch_path):
        with open(batch_path, errors="replace") as fh:
            for line in fh:
                key = split_row(line)
                if key:
                    keys.add(key)
    if not keys:
        log(f"nothing to mark ({rel(batch_path)} has no data rows)")
        return 0

    existing = load_sent(sent_path)
    added = sorted(k for k in keys if k not in existing)
    if not added:
        log(f"all {len(keys)} row(s) already in {rel(sent_path)}")
        return 0

    known = {r["gap"]: r for r in rows}
    new_file = not existing
    with open(sent_path, "a") as fh:
        if new_file:
            fh.write("# phase0 records already POSTed to primegaps.cloudygo.com\n"
                     "# format: gap start_prime   (generated by "
                     "scripts/p0_records_to_submit.py --mark-sent)\n")
        fh.write(f"# marked sent {utc_str(int(time.time()))} "
                 f"from {rel(batch_path)}\n")
        for gap, lower in added:
            fh.write(f"{gap} {lower}")
            row = known.get(gap)
            if row and row["lower"] == lower:
                fh.write(f"   # merit {row['merit6']:.6f}")
            fh.write("\n")
    log(f"marked {len(added)} row(s) as sent in {rel(sent_path)}")
    return len(added)


# ── rendering ───────────────────────────────────────────────────────────────

def render(rows: list[dict], meta: dict, args) -> str:
    accepted = [r for r in rows if r["state"] == "ok"]
    skipped = [r for r in rows if r["state"] != "ok"]
    by_state = {s: [r for r in skipped if r["state"] == s]
                for s in ("sent", "superseded", "failed", "pending")}

    out: list[str] = []
    add = out.append
    add("# phase0 walk records -> submit batch        (GENERATED — do not edit)")
    add("#")
    add(f"# generator  scripts/p0_records_to_submit.py   ({meta['mode']})")
    add(f"# discoverer {args.discoverer}    (submit separately from Gapcoin batches)")
    add(f"# generated  {utc_str(int(time.time()))} UTC   "
        f"rows={len(accepted)} from {meta['events']} record event(s)")
    add("# format     gap merit prime_start        "
        "merit = gap/ln(prime_start), 6 dp")
    add(f"# sources    {meta['sources']}")
    add("#")
    if accepted:
        add("#   gap    merit      was      x-gain  found (UTC)        slices        ev  verify")
        for r in accepted:
            add("#   %-6d %-9.6f %-8s %-6s  %s  %-12s %-3d %s" % (
                r["gap"], r["merit6"],
                f"{r['table']:.4f}" if r["table"] is not None else "?",
                r["x_gain"], utc_str(r["first_epoch"]),
                ",".join(sorted(r["slices"]))[:12],
                r["events"], r["verify"] + (f" ({r['live_note']})"
                                            if r.get("live_note") else "")))
    add("#")
    if args.check_live:
        add(f"# live table : checked against {MERITS_URL} "
            f"({meta['live']}); beat/gain shown per row")
    if args.verify:
        add("# Every row was re-verified independently (scripts/verify_gap_candidate.py:")
        add("# OpenSSL endpoint checks + strict scan of every odd between the primes).")
    else:
        add("# !! INDEPENDENT VERIFICATION DISABLED (--no-verify): rows below are")
        add("# !! only as trustworthy as the engine's own verified=<0|1> flag.")
    for state, title in (("sent", "already POSTed (excluded)"),
                         ("superseded", "superseded — excluded"),
                         ("failed", "FAILED independent verification (excluded)"),
                         ("pending", "verification inconclusive (excluded)")):
        rows_s = by_state[state]
        if not rows_s:
            continue
        add(f"# {title}: {len(rows_s)}")
        for r in rows_s:
            add(f"#   {r['gap']} {r['merit6']:.6f} {r['lower']}   "
                f"[{r['verify_note'] or r['state']}]")
    add("#")
    add("# Send as ONE POST, separately from the Gapcoin batch, and only while the")
    add("# site's commit limiter is idle — a refused POST shadows those numbers")
    add("# for ~9 h (it then answers \"Already processed\"):")
    add("#   python3 scripts/submit_records.py --in "
        f"{rel(resolve(args.out))} \\")
    add(f"#       --discoverer {args.discoverer} "
        f"--date {datetime.now(timezone.utc).date().isoformat()} \\")
    add("#       --no-split-by-date --batch-size 100 --delay 3")
    add("# After a successful POST:")
    add("#   python3 scripts/p0_records_to_submit.py "
        f"--mark-sent {rel(resolve(args.out))}")
    add("#")
    add("# Format: gap merit prime_start")
    for r in accepted:
        add(f"{r['gap']} {r['merit6']:.6f} {r['lower']}")
    return "\n".join(out) + "\n"


# ── one pass ────────────────────────────────────────────────────────────────

def collect(args, sent: set[tuple[int, int]],
            live_holder: dict) -> tuple[list[dict], dict]:
    events, paths = scan_logs(args.logs, args.max_log_mb)
    rows = merge(events)
    if args.verify:
        verify_rows(rows, resolve(args.verify_cache), args.verify_timeout,
                    args.reverify)

    for row in rows:
        row["live_note"] = ""
        row.setdefault("verify", "PENDING" if args.verify else "SKIPPED")
        if (row["gap"], row["lower"]) in sent:
            row["state"], row["verify_note"] = "sent", "already POSTed"
            continue
        if row["unverified_events"] == row["events"]:
            row["state"] = "pending"
            row["verify_note"] = "no verified=1 event"
            continue
        if args.verify:
            verdict = row.get("verify", "PENDING")
            if verdict == "FAIL":
                row["state"] = "failed"
                continue
            if verdict != "PASS":
                row["state"] = "pending"
                continue
        if row["table"] is not None and row["merit6"] <= row["table"]:
            row["state"], row["verify_note"] = "superseded", "table snapshot"
            continue
        live = live_holder.get("map", {}).get(row["gap"])
        if live is not None:
            live_merit, holder = live
            if row["merit6"] <= live_merit + 5e-7:
                row["state"] = "superseded"
                row["verify_note"] = f"live {live_merit:.4f} ({holder})"
                continue
            row["live_note"] = f"beats {live_merit:.4f} {holder}"
        row["state"] = "ok"

    contributors: dict[str, int] = {}
    for ev in events:
        contributors[ev["log"]] = contributors.get(ev["log"], 0) + 1
    sources = ", ".join(f"{name} ({cnt})" for name, cnt in sorted(contributors.items()))
    meta = {
        "events": len(events),
        "scanned": len(paths),
        "sources": sources or "no record events yet",
        "live": live_holder.get("note", "off"),
        "mode": "watch" if not args.once else "single pass",
    }
    return rows, meta


def pass_once(args, sent: set[tuple[int, int]], live_holder: dict,
              prev_keys: set[str] | None,
              npass: int) -> tuple[set[str], dict, str]:
    rows, meta = collect(args, sent, live_holder)
    text = render(rows, meta, args)
    accepted = [r for r in rows if r["state"] == "ok"]
    keys = {f"{r['gap']}:{r['lower']}" for r in accepted}

    if not args.dry_run:
        try:
            write_atomic(resolve(args.out), text)
        except OSError as exc:
            warn(f"cannot write {rel(resolve(args.out))}: {exc}")
        state = (f"pass={npass} epoch={int(time.time())} "
                 f"time={utc_str(int(time.time()))} rows={len(accepted)} "
                 f"events={meta['events']} logs={meta['scanned']} "
                 f"out={rel(resolve(args.out))}\n")
        try:
            write_atomic(resolve(args.state_file), state)
        except OSError as exc:
            warn(f"cannot write {rel(resolve(args.state_file))}: {exc}")

    if args.dry_run:
        print(text, end="")
        return keys, meta, text

    if prev_keys is None:
        log(f"{len(accepted)} row(s) from {meta['events']} record event(s) "
            f"in {meta['scanned']} log(s) -> {rel(resolve(args.out))}")
    else:
        for key in sorted(keys - prev_keys):
            row = next(r for r in accepted if f"{r['gap']}:{r['lower']}" == key)
            log(f"NEW  gap={row['gap']} merit={row['merit6']:.6f} "
                f"was={row['table']} x={row['lower']} "
                f"slice={','.join(sorted(row['slices']))} events={row['events']}")
        for key in sorted(prev_keys - keys):
            log(f"row left the batch: {key}")
    return keys, meta, text


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage:")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See README_PHASE0.md (section: record harvest watcher).")
    ap.add_argument("--logs", nargs="+", default=DEFAULT_LOGS,
                    help=f"gap-log glob(s), default {DEFAULT_LOGS[0]}")
    ap.add_argument("--max-log-mb", type=float, default=DEFAULT_MAX_LOG_MB,
                    help="skip logs larger than this (0 = no limit), "
                         f"default {DEFAULT_MAX_LOG_MB:.0f}")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"submit batch file, default {DEFAULT_OUT}")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="watch poll interval in seconds, default 30")
    ap.add_argument("--once", action="store_true",
                    help="single pass then exit (cron/systemd friendly)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the batch to stdout instead of writing files")
    ap.add_argument("--discoverer", default="D.Benko",
                    help="name written into the send command, default D.Benko")
    ap.add_argument("--sent-file", default=DEFAULT_SENT,
                    help=f"POSTed rows excluded from the batch, default {DEFAULT_SENT}")
    ap.add_argument("--state-file", default=DEFAULT_STATE,
                    help=f"liveness/state line, default {DEFAULT_STATE}")
    ap.add_argument("--verify-cache", default=DEFAULT_VCACHE,
                    help=f"verification verdict cache, default {DEFAULT_VCACHE}")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True,
                    help="independent verification with verify_gap_candidate.py (default)")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="skip independent verification (rows marked unverified)")
    ap.add_argument("--verify-timeout", type=float, default=300.0,
                    help="seconds per verification run, default 300")
    ap.add_argument("--reverify", action="store_true",
                    help="ignore cached verification verdicts and re-run them")
    ap.add_argument("--check-live", action="store_true",
                    help="fetch merits.txt and flag rows the live table has caught")
    ap.add_argument("--live-interval", type=float, default=1800.0,
                    help="seconds between merits.txt refreshes, default 1800")
    ap.add_argument("--mark-sent", metavar="FILE", default=None,
                    help="append FILE's data rows to the sent-file, then do one pass")
    ap.add_argument("--quiet", action="store_true", help="suppress the heartbeat")
    args = ap.parse_args()

    if args.interval < 5:
        ap.error("--interval must be >= 5 seconds")
    if not args.verify:
        warn("--no-verify: rows are written WITHOUT independent verification")

    def on_signal(signum, _frame):
        global _STOP
        _STOP = True
        log(f"signal {signum}: finishing current pass, then exit")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    live_holder: dict = {"map": {}, "note": "off", "fetched": 0.0}
    npass = 0
    prev_keys: set[str] | None = None

    while True:
        npass += 1
        if args.check_live and (time.time() - live_holder["fetched"] >
                                args.live_interval or not live_holder["map"]):
            try:
                live_holder["map"] = fetch_live()
                live_holder["note"] = (f"{len(live_holder['map'])} lengths, "
                                       f"fetched {utc_str(int(time.time()))}")
            except Exception as exc:                       # network, parse, ...
                live_holder["note"] = f"fetch failed: {exc}"
                if live_holder["map"]:
                    live_holder["note"] += " (using previous copy)"
                    warn(f"merits.txt refresh failed: {exc}")
                else:
                    warn(f"merits.txt unavailable ({exc}) — rows unchecked "
                         f"against the live table")
            live_holder["fetched"] = time.time()

        try:
            if args.mark_sent and npass == 1:
                events, _ = scan_logs(args.logs, args.max_log_mb)
                mark_sent(resolve(args.mark_sent), resolve(args.sent_file),
                          merge(events))
            sent = load_sent(resolve(args.sent_file))
            prev_keys, meta, _ = pass_once(args, sent, live_holder, prev_keys,
                                           npass)
        except Exception as exc:                       # never die mid-campaign
            warn(f"pass {npass} failed: {type(exc).__name__}: {exc}")
            if args.once or args.dry_run:
                return 1

        if args.once or args.dry_run or _STOP:
            break
        if not args.quiet and npass % max(1, int(600 / args.interval)) == 0:
            sent_now = len(load_sent(resolve(args.sent_file)))
            log(f"heartbeat pass={npass} rows={len(prev_keys or ())} "
                f"sent={sent_now} live={'on' if args.check_live else 'off'}")
        for _ in range(int(args.interval)):
            if _STOP:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
