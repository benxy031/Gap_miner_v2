#!/usr/bin/env python3
# Copyright (C) 2026  cpugapminer contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
submit_records.py — Submit prime gap records to primegaps.cloudygo.com.

Two input formats are auto-detected:

  export  a record file written by scan_blocks_gap2026.py --export-records,
          i.e. "<gap> <merit> <start prime>" rows with optional
          "# height=<n>" comments.  This is the Gapcoin block database's own
          output and it carries RAW DECIMAL primes (a Gapcoin number is
          h<<shift + adder, which has no primorial form).
  log     miner log text with ">>> GAP FOUND" blocks, whose NUMBER is normally
          already an expression (a * P# / d - b) — the form the site's form
          placeholder documents.

Why raw decimals are sent for export input: the list itself stores them.
prime-gap-list's check.py parses a plain digit string
(`if start.isdigit(): return int(start)`), and 1188 rows of allgaps.sql are
raw decimals (Gapcoin-sourced, e.g. A.Renyer 2025).  The server's reply is
printed verbatim after every POST, so if a future parser change does reject
them, it shows up as "0 processed"/HTTP 500 instead of failing silently.

Usage examples
--------------
Dry-run an export file (nothing is sent):
    python3 scripts/submit_records.py --in records_to_submit_9.txt \\
        --discoverer Gapcoin --dry-run

Send the three strongest records first and inspect the queue:
    python3 scripts/submit_records.py --in records_to_submit_9.txt \\
        --discoverer Gapcoin --sort merit --limit 3

Send everything (one POST per discovery date, dates from gapchain.sqlite3):
    python3 scripts/submit_records.py --in records_to_submit_9.txt \\
        --discoverer Gapcoin --batch-size 10 --delay 3

Parse a miner log instead (GAP FOUND blocks, expression numbers):
    python3 scripts/submit_records.py --log miner.log \\
        --discoverer S.Troisi --dry-run

Read miner output from stdin:
    ./bin/gap_miner ... 2>&1 | python3 scripts/submit_records.py \\
        --discoverer S.Troisi --date 2024-01-15

A record is skipped when it does not improve the live merit table
(primegaps.cloudygo.com/merits.txt) by more than --min-improvement (the table
is published with 4 decimals), when the same number is already on the site's
queue, or when it is the weaker of two attempts at the same gap length (the
list keeps one record per length).

The server's reply is printed after every POST: it says which gapsize was
queued and by how much the merit improves.  Verified records leave the queue
within seconds and show up on /status under "Recent records checked" and in
the published table.

Requirements: Python 3 standard library only.
"""

import argparse
import datetime
import http.cookiejar
import math
import os
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser


# ── Constants ──────────────────────────────────────────────────────────────
SITE_URL   = "https://primegaps.cloudygo.com/"
MERITS_URL = "https://primegaps.cloudygo.com/merits.txt"
UA         = "cpugapminer-submit/1.0"

# Minimum gap size the site will accept (smaller gaps are fully catalogued)
MIN_GAP_SIZE = 1202

# The site parser (primegap-list check.py) accepts these NUMBER grammars only:
#   a*P#/d±b  P#/d±b  a*P#/(Q#*d)±b  a*P#/(Q*d)±b  a*P#/Q#±b  a^k±b
# A pure decimal integer matches none of them: the server ignores it (single
# line) or crashes with HTTP 500 (multi-line batches).
_NUMBER_FORMS = (
    re.compile(r"^\d+\*\(?\d+#\)?/\d+[+-]\d+$"),
    re.compile(r"^\(?\d+#\)?/\d+[+-]\d+$"),
    re.compile(r"^\d+\*\d+#/\(\d+#\*\d+\)[+-]\d+$"),
    re.compile(r"^\d+\*\d+#/\(\d+\*\d+\)[+-]\d+$"),
    re.compile(r"^\d+\*\d+#/\d+#[+-]\d+$"),
    re.compile(r"^\d+\^\d+[+-]\d+$"),
)


def _is_expression_number(n):
    """True if the prime is already in one of the site's expression forms.

    The site accepts spaced expressions (e.g. `a * 337# / 2310 - b`), so
    whitespace is stripped before matching.
    """
    compact = re.sub(r"\s+", "", n)
    return any(f.fullmatch(compact) for f in _NUMBER_FORMS)


# ── Record-export input (scan_blocks_gap2026.py --export-records) ───────────
#
# The live Gapcoin block database is exported for submission like this:
#
#     # Gapcoin prime gap records — exported by scan_blocks.py
#     # Format: gap merit prime_start
#     # height=2531756
#     16816 31.594797 1408288415797097243068…883
#
# i.e. "<gap> <merit> <start prime>" rows, each optionally preceded by a
# "# height=<n>" comment (used here to recover that block's own date).
#
# The gap-hunt watcher (scripts/watch_gap_hunt_records.py) writes the same
# first three fields but appends metadata, which is parsed too:
#
#     37882 26.803420 6311915248…8139 25.602900 2026-09-23T22:22:43Z
#           claim=FIRST_KNOWN_OCCURRENCE coverage=known_table_b2bebffea359
#
# so those files can be submitted directly — and their ISO timestamp gives a
# better discovery date than a database lookup.
_EXPORT_ROW_RE    = re.compile(r"^(\d+)\s+([\d.]+)\s+(\d{6,})\b(.*)$")
_EXPORT_HEIGHT_RE = re.compile(r"^#\s*height\s*=\s*(\d+)")
_EXPORT_ISO_RE    = re.compile(r"(\d{4}-\d{2}-\d{2})T\d{2}:\d{2}:\d{2}Z")
_EXPORT_HEADER_RE = re.compile(r"^(?:date|snapshot)\s*=")


def looks_like_export(text):
    """True when *text* is an --export-records file rather than a miner log."""
    if ">>> GAP FOUND" in text:
        return False
    return any(_EXPORT_ROW_RE.match(l.strip())
               for l in text.splitlines()[:500])


def parse_export(text):
    """Parse an --export-records file into record dicts.

    Returns [{gap, merit, prime, height, date}] in file order.
    """
    records, height = [], None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _EXPORT_HEIGHT_RE.match(line)
        if m:
            height = int(m.group(1))
            continue
        if line.startswith("#"):
            continue
        m = _EXPORT_ROW_RE.match(line)
        if not m:
            if _EXPORT_HEADER_RE.match(line):
                continue      # watcher snapshot header, not a record
            print(f"  Warning: unparsed line skipped: {line[:70]}",
                  file=sys.stderr)
            continue
        iso = _EXPORT_ISO_RE.search(m.group(4))
        records.append({"gap": int(m.group(1)), "merit": float(m.group(2)),
                        "prime": m.group(3), "height": height,
                        "date": iso.group(1) if iso else None})
    return records


def default_db_path():
    """<repo>/gapchain.sqlite3 (the database sync_gapchain_blocks.py fills)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "gapchain.sqlite3")


def load_dates(db_path, heights):
    """{height: 'YYYY-MM-DD'} from gapchain.sqlite3 ({} if unavailable)."""
    heights = {h for h in heights if h}
    if not heights or not os.path.exists(db_path):
        return {}
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        print(f"  Warning: cannot open {db_path}: {exc}", file=sys.stderr)
        return {}
    out = {}
    try:
        for height in heights:
            row = conn.execute("SELECT date FROM gapchain WHERE height=?",
                               (height,)).fetchone()
            if row and row[0]:
                out[height] = str(row[0])
    except sqlite3.Error as exc:
        print(f"  Warning: cannot read dates from {db_path}: {exc}",
              file=sys.stderr)
        return {}
    finally:
        conn.close()
    return out


def site_number_chunks(site_html, width=40):
    """All `width`-digit chunks of the long digit runs on a site page.

    A queued number is rendered in full (wrapped over several lines), so any
    `width`-digit chunk of our prime appearing in a page means the record is
    already there.  Chunking instead of a plain substring test survives the
    line wrapping.
    """
    chunks = set()
    for run in re.findall(r"\d{%d,}" % width, site_html):
        for i in range(len(run) - width + 1):
            chunks.add(run[i:i + width])
            if len(chunks) > 400000:
                return chunks
    return chunks


def on_site(number, chunks, width=40):
    """True when *number* (or a big chunk of it) is on the fetched page."""
    return any(number[i:i + width] in chunks
               for i in range(0, len(number) - width + 1))


# ── Log parser ──────────────────────────────────────────────────────────────
# Matches an entire >>> GAP FOUND block (multi-line, non-greedy).
# Captures: gap, merit, nShift, nAdd.
# nAdd may be:
#   • a raw decimal with an optional " (0x…)" hex annotation (RPC mode)
#   • the full decimal start prime (CRT/scan mode)
#   • a site expression like "3676117443599 * 337# / 2310 - 7578"
_GAP_BLOCK_RE = re.compile(
    r">>> GAP FOUND\b.*?"
    r"gap\s*=\s*(\d+).*?"
    r"merit\s*=\s*([\d.]+).*?"
    r"nShift\s*=\s*(\d+).*?"
    r"nAdd\s*=\s*(.+?)(?:\s+\(0x[0-9a-fA-F]+\))?\s*(?:\n|$)",
    re.DOTALL,
)

# [verify_pow] hash= line printed by the miner after every RPC gap block.
# The hash is displayed big-endian (bytes reversed from internal LE storage),
# but int(hash_hex, 16) == the GMP integer that was imported LE — they match.
_VERIFY_HASH_RE = re.compile(r"\[verify_pow\]\s+hash=([0-9a-fA-F]{64})")

# In RPC/Gapcoin mode nAdd is at most 8 bytes (2^64-1).
# In CRT/scan mode nAdd IS the full prime (hundreds of digits, >> 2^64).
_RPC_NADD_MAX = (1 << 64)


def parse_log(text):
    """
    Scan *text* for >>> GAP FOUND blocks and return a list of
    (gap:int, merit:float, prime_str:str) tuples, deduplicated.

    Two log formats are handled automatically:

    CRT / scan mode  —  nAdd is the full starting prime (large decimal),
    or already a site expression:
        nAdd    = 35133984279...  (80+ digits)
        nAdd    = 3676117443599 * 337# / 2310 - 7578

    RPC / Gapcoin mining mode  —  nAdd is a small addend; the full prime
    is reconstructed from the [verify_pow] hash= line that follows:
        nAdd    = 1844674415710808815 (0x1999999b8ab1aaef)
        ...later...
        [verify_pow] hash=fbe6a3c073c52fd4...  bits=256 hash_ok=1 is_prime=1
        prime = int(hash_hex, 16) << nShift + nAdd
    """
    results = []
    seen = set()
    for m in _GAP_BLOCK_RE.finditer(text):
        gap    = int(m.group(1))
        merit  = float(m.group(2))
        nshift = int(m.group(3))
        nadd_str = m.group(4).strip()

        if nadd_str.isdigit() and int(nadd_str) < _RPC_NADD_MAX:
            # RPC/Gapcoin mode: small addend; the full prime is reconstructed
            # from the [verify_pow] hash= line that follows this block.
            nadd = int(nadd_str)
            hash_m = _VERIFY_HASH_RE.search(text, m.end())
            if not hash_m:
                print(f"  Warning: RPC gap (gap={gap}) has no [verify_pow] hash= "
                      f"line — cannot reconstruct prime; skipping.",
                      file=sys.stderr)
                continue
            hash_int  = int(hash_m.group(1), 16)
            prime_str = str((hash_int << nshift) + nadd)
        else:
            # CRT/scan mode: full decimal prime, or a site expression form.
            prime_str = nadd_str

        key = (gap, prime_str)
        if key not in seen:
            seen.add(key)
            results.append((gap, merit, prime_str))
    return results


# ── Record comparison ───────────────────────────────────────────────────────

def fetch_merits():
    """
    Download merits.txt from the site.
    Returns dict {gap_size: merit} or {} on error.
    """
    print(f"Fetching current records from {MERITS_URL} …", file=sys.stderr)
    try:
        req = urllib.request.Request(MERITS_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        records = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    records[int(parts[0])] = float(parts[1])
                except ValueError:
                    pass
        print(f"  Loaded {len(records):,} existing records.", file=sys.stderr)
        return records
    except Exception as exc:
        print(f"  Warning: could not fetch merits.txt: {exc}", file=sys.stderr)
        return {}


# ── CSRF extraction ─────────────────────────────────────────────────────────

class _CSRFParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.csrf_token = None

    def handle_starttag(self, tag, attrs):
        if tag == "input":
            d = dict(attrs)
            if d.get("name") == "csrf_token":
                self.csrf_token = d.get("value", "")


def _extract_csrf(html):
    p = _CSRFParser()
    p.feed(html)
    return p.csrf_token


# ── HTTP helpers (session via cookie jar) ───────────────────────────────────

def _opener(jar):
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar)
    )


def _get(jar, url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with _opener(jar).open(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _post(jar, url, fields):
    body = urllib.parse.urlencode(fields).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": UA,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": url,
            "Origin":  url.rstrip("/"),
        },
    )
    with _opener(jar).open(req, timeout=90) as resp:
        return resp.read().decode("utf-8")


# ── Response parser ─────────────────────────────────────────────────────────

def _div_text(html, div_id):
    """Text inside the first <div id="div_id">…</div> (best effort)."""
    m = re.search(r'<div[^>]*id\s*=\s*["\']%s["\'][^>]*>(.*?)</div>' % div_id,
                  html, re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    # Strip real tags only: the server message contains "<294>" (a digit-count
    # annotation), which is not a tag and must survive.
    return re.sub(r"</?[a-zA-Z][^>]*>", "", m.group(1)).strip()


def _reply_summary(html):
    """Read a submission reply: (status message, queued item descriptions).

    The page is server-rendered and has no <script>.  A reply carries

        <section><h3>Status Results</h3>
          <div id="results">Adding 374…939<294> gapsize=21224 to queue,
              would improve merit 27.349 to 31.397</div></section>
        <section><h3>1 Queued</h3>
          <div id="queue"><div>21224, C??, 31.3974, Gapcoin, 2026-09-25,
              294, 3747290199…378789939</div></div></section>

    so the message and the accepted items can both be read back verbatim —
    which is how a rejected format (or an unexpected value) becomes visible.
    """
    message = _div_text(html, "results")

    items = []
    m = re.search(r'id\s*=\s*["\']queue["\'][^>]*>(.*?)</section>',
                  html, re.DOTALL | re.IGNORECASE)
    if m:
        for raw in re.findall(r"<div>(.*?)</div>", m.group(1), re.DOTALL):
            text = " ".join(raw.split())
            parts = [p.strip() for p in text.split(",")]
            if len(parts) >= 7:
                gap, _cert, merit, who, date, digits, number = parts[:7]
                items.append(f"{who} {date}: gap={gap} merit={merit} "
                             f"({digits} digits, {number[:10]}…{number[-10:]})")
            elif text:
                items.append(text)
    return message, items


# ── Submission ──────────────────────────────────────────────────────────────

def submit_batch(batch, discoverer, date_str, dry_run):
    """
    Submit one POST of record dicts.  Returns True on (apparent) success.
    """
    lines = [f"{r['gap']} {r['merit']:.4f} {r['prime']}" for r in batch]
    logdata = "\n".join(lines)

    print(f"\n{'─'*72}")
    print(f"{len(batch)} record(s)  discoverer={discoverer}  date={date_str}")
    for r in batch:
        where = f"height={r['height']} " if r.get("height") else ""
        print(f"  gap={r['gap']:8d}  merit={r['merit']:.4f}  {where}"
              f"start=…{r['prime'][-8:]} ({len(r['prime'])} digits)")

    if dry_run:
        print("  [dry-run] logdata that would be POSTed:")
        for line in lines:
            preview = line if len(line) <= 100 else line[:48] + "…" + line[-30:]
            print("    " + preview)
        print("  [dry-run] skipping the submission.")
        return True

    jar = http.cookiejar.CookieJar()

    # GET page to obtain session cookie + CSRF token
    print("  → Fetching CSRF token …", end=" ", flush=True)
    try:
        html = _get(jar, SITE_URL)
    except Exception as exc:
        print(f"FAILED ({exc})")
        return False
    csrf = _extract_csrf(html)
    if not csrf:
        print("FAILED (no csrf_token found in page)")
        return False
    print(f"OK ({csrf[:8]}…)")

    # POST
    print("  → Submitting …", end=" ", flush=True)
    try:
        resp = _post(jar, SITE_URL, {
            "discoverer": discoverer,
            "date":       date_str,
            "logdata":    logdata,
            "csrf_token": csrf,
            "submit":     "Add",
        })
    except Exception as exc:
        print(f"FAILED ({exc})")
        return False
    print("OK")

    # Print the server's own words: the queue message ("would improve merit
    # 27.349 to 31.397"), the accepted items, or an error — so a rejected
    # format is visible instead of silently swallowed.  Verified records leave
    # the queue within seconds; /status lists them under "Recent records
    # checked" and "New Records".
    message, items = _reply_summary(resp)
    if message:
        print("  Server reply:", message[:300])
    else:
        flat = " ".join(re.sub(r"<[^>]+>", " ", resp).split())
        print("  Server reply (no #results block; first 300 chars):")
        print("    " + flat[:300])
    for item in items:
        print("  Queued:", item)

    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Submit cpugapminer gap records to primegaps.cloudygo.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Requirements:")[0].strip(),
    )
    ap.add_argument("input", nargs="?", default=None,
                    help="Record export file (--export-records) or a miner log; "
                         "default: read from stdin")
    ap.add_argument("--discoverer", required=True,
                    help="Short name (3-8 chars), e.g. S.Troisi")
    ap.add_argument("--date", default=None,
                    help="YYYY-MM-DD discovery date for all records (default: "
                         "each record's own block date from --db, else today)")
    ap.add_argument("--log", default=None,
                    help="Explicit input file (same as the positional FILE)")
    ap.add_argument("--in", dest="input_opt", default=None, metavar="FILE",
                    help="Explicit input file (same as the positional FILE)")
    ap.add_argument("--db", default=default_db_path(),
                    help="gapchain.sqlite3 used to date records by block "
                         "height (default: <repo>/gapchain.sqlite3)")
    ap.add_argument("--sort", choices=("none", "gap", "merit"), default="merit",
                    help="Submission order: merit = strongest first (default), "
                         "gap = shortest first, none = file order")
    ap.add_argument("--limit", type=int, default=0,
                    help="Send at most N records, after --sort (default: all)")
    ap.add_argument("--min-gap", type=int, default=MIN_GAP_SIZE,
                    help=f"Ignore gaps below this size (default: {MIN_GAP_SIZE})")
    ap.add_argument("--min-merit", type=float, default=0.0,
                    help="Ignore gaps with merit below this (default: 0)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and preview records, do NOT submit")
    ap.add_argument("--batch-size", type=int, default=10,
                    help="Records per HTTP POST batch (default: 10)")
    ap.add_argument("--delay", type=float, default=3.0,
                    help="Seconds between batches (default: 3)")
    ap.add_argument("--skip-check", action="store_true",
                    help="Submit all found gaps without comparing to existing records")
    ap.add_argument("--min-improvement", type=float, default=1e-4, metavar="D",
                    help="Only submit when merit exceeds the live table by more "
                         "than D (default: 1e-4 — the live table is published "
                         "with 4 decimals, so a 6-decimal merit that merely "
                         "rounds to the published value is NOT an improvement)")
    ap.add_argument("--no-queue-check", action="store_true",
                    help="Do not ask the site which numbers are already queued")
    ap.add_argument("--no-split-by-date", action="store_true",
                    help="Pack every record into one date group instead of one "
                         "POST per discovery date")
    ap.add_argument("--force", action="store_true",
                    help="Allow raw-decimal primes in miner-log input (export "
                         "files always send raw decimals: a Gapcoin number is "
                         "h<<shift + adder and has no primorial form)")
    args = ap.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────
    if not 3 <= len(args.discoverer) <= 8:
        ap.error(f"--discoverer '{args.discoverer}' must be 3-8 characters")

    date_str = args.date or datetime.date.today().isoformat()
    try:
        datetime.date.fromisoformat(date_str)
    except ValueError:
        ap.error(f"--date '{date_str}': use YYYY-MM-DD format")

    # ── Read input ────────────────────────────────────────────────────────
    path = args.input or args.input_opt or args.log
    if path:
        try:
            with open(path, "r", errors="replace") as fh:
                text = fh.read()
        except OSError as exc:
            ap.error(f"Cannot open input file: {exc}")
    else:
        print("Reading from stdin … (Ctrl+C to stop)", file=sys.stderr)
        text = sys.stdin.read()

    # ── Parse (the input format is auto-detected) ─────────────────────────
    if looks_like_export(text):
        kind = "export"
        found = parse_export(text)
        if not args.date:
            # Watcher rows carry their own ISO timestamp; the rest are dated
            # from gapchain.sqlite3 by block height.
            dates = load_dates(args.db, {r["height"] for r in found
                                         if r["height"] and not r["date"]})
            for r in found:
                if not r["date"]:
                    r["date"] = dates.get(r["height"])
            dated = sorted(r["date"] for r in found if r["date"])
            if dated:
                print(f"  Dated {len(dated)}/{len(found)} record(s) "
                      f"({dated[0]} … {dated[-1]})", file=sys.stderr)
        print(f"  Input: export file "
              f"({len(found)} record row(s)){f' — {path}' if path else ''}",
              file=sys.stderr)
    else:
        kind = "log"
        found = [{"gap": g, "merit": m, "prime": n, "height": None,
                  "date": None} for g, m, n in parse_log(text)]
        print(f"\nFound {len(found)} unique GAP FOUND block(s) in log.",
              file=sys.stderr)

    if not found:
        print("Nothing to submit.")
        return

    # ── Filter by merit / gap size ────────────────────────────────────────
    if args.min_merit > 0:
        before = len(found)
        found = [r for r in found if r["merit"] >= args.min_merit]
        print(f"  Merit ≥{args.min_merit}: kept {len(found)}/{before}",
              file=sys.stderr)

    before = len(found)
    found = [r for r in found if r["gap"] >= args.min_gap]
    if len(found) < before:
        print(f"  Dropped {before - len(found)} gap(s) below minimum size "
              f"{args.min_gap}", file=sys.stderr)

    if not found:
        print("Nothing to submit (all records were filtered out).")
        return

    # ── One record per gap length (the list keeps the best one) ───────────
    best = {}
    for r in found:
        old = best.get(r["gap"])
        if old is None or r["merit"] > old["merit"]:
            best[r["gap"]] = r
    if len(best) < len(found):
        print(f"  Kept the strongest attempt per gap length: "
              f"{len(found)} → {len(best)}", file=sys.stderr)
    found = sorted(best.values(), key=lambda r: (r["gap"], -r["merit"]))

    # ── Raw decimals: normal for export input, refused in miner logs ──────
    raw = [r for r in found if not _is_expression_number(r["prime"])]
    if raw and kind == "log" and not args.force:
        for r in raw:
            print(f"  skip gap={r['gap']}: raw decimal prime "
                  f"({len(r['prime'])} digits) in a miner log", file=sys.stderr)
        print("\nA miner log is expected to carry the searched expression "
              "(a * P# / d ± b).  Re-run with --force to send raw decimals, "
              "or feed an --export-records file: raw decimals are that "
              "format's normal content.", file=sys.stderr)
        found = [r for r in found if _is_expression_number(r["prime"])]
        if not found:
            print("\nNothing to submit (raw-decimal records were dropped).")
            return
    elif raw:
        print(f"  {len(raw)}/{len(found)} record(s) are raw decimal primes — "
              f"sent as-is (a Gapcoin number is h<<shift + adder and has no "
              f"primorial form)", file=sys.stderr)

    # ── Compare against the live record table ─────────────────────────────
    if args.skip_check:
        to_submit = found
        print(f"  --skip-check: submitting all {len(to_submit)} gap(s).",
              file=sys.stderr)
    else:
        records = fetch_merits()
        to_submit, skipped = [], 0
        for r in found:
            existing = records.get(r["gap"], 0.0)
            if r["merit"] > existing + args.min_improvement:
                to_submit.append(r)
            else:
                skipped += 1
                print(f"  skip gap={r['gap']}: merit {r['merit']:.6f} ≤ "
                      f"existing {existing:.4f} + {args.min_improvement:g}",
                      file=sys.stderr)
        if skipped:
            print(f"  Skipped {skipped} non-improvement(s).", file=sys.stderr)

    # ── Anything already on the site (queued, not merged yet) ─────────────
    if to_submit and not args.no_queue_check and not args.dry_run:
        try:
            page = _get(http.cookiejar.CookieJar(), SITE_URL)
            chunks = site_number_chunks(page)
            queued = [r for r in to_submit if on_site(r["prime"], chunks)]
            if queued:
                shown = ", ".join(str(r["gap"]) for r in queued[:8])
                print(f"  Already on the site: {len(queued)} record(s) "
                      f"(gap {shown}{' …' if len(queued) > 8 else ''}) — "
                      f"skipped.", file=sys.stderr)
                stay = set(map(id, queued))
                to_submit = [r for r in to_submit if id(r) not in stay]
        except Exception as exc:
            print(f"  (queue check skipped: {exc})", file=sys.stderr)

    if not to_submit:
        print("\nNo new/improved records to submit. Done.")
        return

    # ── Order + limit ─────────────────────────────────────────────────────
    if args.sort == "merit":
        to_submit.sort(key=lambda r: -r["merit"])
    elif args.sort == "gap":
        to_submit.sort(key=lambda r: (r["gap"], -r["merit"]))
    if args.limit > 0 and len(to_submit) > args.limit:
        print(f"  --limit {args.limit}: sending the first {args.limit} of "
              f"{len(to_submit)} record(s) in --sort {args.sort} order.",
              file=sys.stderr)
        to_submit = to_submit[:args.limit]

    # ── One POST group per discovery date (a POST carries a single date) ──
    if args.date or args.no_split_by_date or not any(r["date"] for r in to_submit):
        groups = [(date_str, to_submit)]
    else:
        by_date = {}
        for r in to_submit:
            by_date.setdefault(r["date"] or date_str, []).append(r)
        groups = sorted(by_date.items())
        print(f"  Splitting into {len(groups)} discovery-date group(s) "
              f"({groups[0][0]} … {groups[-1][0]}): one POST per date keeps "
              f"each record's own date on the site.", file=sys.stderr)

    n_posts = sum(math.ceil(len(rs) / args.batch_size) for _, rs in groups)
    print(f"\n{len(to_submit)} record(s) → {n_posts} POST(s) "
          f"(discoverer={args.discoverer}, batch size {args.batch_size})")
    if args.dry_run:
        print("  [dry-run mode — no HTTP requests will be made]")

    # ── Submit ────────────────────────────────────────────────────────────
    post_no = 0
    failures = 0
    for group_date, group_records in groups:
        for i in range(0, len(group_records), args.batch_size):
            batch = group_records[i:i + args.batch_size]
            post_no += 1
            print(f"\n[POST {post_no}/{n_posts}]")
            if not submit_batch(batch, args.discoverer, group_date, args.dry_run):
                failures += 1
                print("  Batch failed; continuing with the next one …")
            if post_no < n_posts and not args.dry_run:
                print(f"  Waiting {args.delay}s before the next POST …")
                time.sleep(args.delay)

    print("\nAll done."
          + (f"  {failures} POST(s) failed." if failures else ""))


if __name__ == "__main__":
    main()
