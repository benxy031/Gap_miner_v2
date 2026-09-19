#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
records_report.py — read the miner record logs (node AND pool) and show, in
text and in graphs, whether the miner is actually finding anything.

Background: the miner appends one line per BPSW-verified candidate to a record
log, and a second line per candidate once the work source has ruled on it:

  discovery line (found)
    <ts> height=<h> shift=<s> header_nonce=<n> nAdd=<d> start=<p> gap=<g>
         merit=<m> best_known_merit=<b> new_record=<yes|no|unknown>
         claim=<FIRST_KNOWN_OCCURRENCE|none> status=<dry-run|queued|queue-full>
  verdict line (what happened to it)
    <ts> height=<h> shift=<s> header_nonce=<n> nAdd=<d> gap=<g> merit=<m>
         status=<accepted|rejected|unresolved|duplicate|send-failed
                |assemble-failed|stale>

Since 2026-09-19 the pool run writes its own file (`gapminer_pool_records.log`)
so a pool share is never confused with a node-accepted gap; this tool reads both
and keeps them apart.

What it reports (per file, then pooled):

  * how many candidates were FOUND, over how long, and at what rate — a
    Poisson interval, and the "nothing found" upper bound when the count is 0
    (an empty log is a measurement, not a missing measurement);
  * the merit distribution: min / median / max, the best gap, and the window
    size scale L = ln(start);
  * whether the observed merit tail matches the geometric/Poisson expectation
    it must follow (a fitted shifted exponential): a threshold table and a
    Kolmogorov-Smirnov distance.  A tail that is shorter than the fit means
    candidates near the threshold are being MISSED; a longer one means the
    threshold or the reporting is wrong.  This is the "is it finding what it
    should" instrument, not a record hunt;
  * how the work source ruled on each candidate (accepted / rejected /
    unresolved / stale ...), PER HOUR — for a pool this is the plot that shows
    a stale-work outage (a run that stops being accepted keeps finding
    candidates at the same rate, so only the verdict series reveals it);
  * in pool mode, WHICH TEMPLATE each submitted solution belonged to
    (`status=submitted template_prevhash=... template_time=...
    template_merit=...`, added 2026-09-19).  A block is only valid for the
    template it was mined on, so this is the field that tells a block the pool
    accepted but never put on chain apart from one it never submitted —
    `scripts/pool_block_audit.py` checks exactly that against the chain;
  * record proximity against `data/prime_gap_merits.txt` (same tables as
    `gap_hunt_stats.py`): the closest candidates to a first-known occurrence,
    any candidate that BEATS the table, and the exact
    `scripts/verify_gap_candidate.py` command to check one.

Caveats it states out loud instead of hiding:

  * a rate is only comparable between files with the SAME threshold: a pool run
    reports from its share target (merit ~15.8), a node run from the network
    difficulty (merit ~23.7), so the node log legitimately has orders of
    magnitude fewer candidates per hour.  Files whose thresholds differ are
    reported separately and flagged, never pooled silently;
  * `sigma` fitted from fewer than ~30 candidates is noise, and so is any
    threshold whose EXPECTED count is small: the verdict only treats a ratio as
    real when it exceeds both 25% and twice the Poisson error of its count;
  * the record log only contains what the miner chose to report — it is not a
    census of all survivors.

Usage:
    scripts/records_report.py                    # auto: both default logs in CWD
    scripts/records_report.py --log a.log --log b.log
    scripts/records_report.py gapminer_pool_records.log --top 10
    scripts/records_report.py --plot                 # -> records_report.png
    scripts/records_report.py --plot out/records.png --since 2026-09-19T12:00
    scripts/records_report.py --source pool          # only pool logs
    scripts/records_report.py --selftest             # verify the tool itself
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import dataclass, field

# ── record-log line grammar ──────────────────────────────────────────────────

TS_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?P<rest>.*)$")
KV_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<val>\S+)")

# The worker writes exactly one of these when it finds a candidate.
DISCOVERY_STATUSES = ("dry-run", "queued", "queue-full")
# The work source (node or pool) then writes one of these for the same candidate.
VERDICT_STATUSES = ("accepted", "rejected", "unresolved", "duplicate",
                    "send-failed", "assemble-failed", "stale")
# Pool mode additionally records WHICH template a solution was submitted
# against (prevhash/time/nDifficulty).  It is neither a discovery nor a verdict:
# it is the context that makes a missing block explainable, so it is kept on the
# candidate and never counted as an outcome of its own.
SUBMIT_CTX_STATUS = "submitted"

DEFAULT_NODE_LOG = "gapminer_records.log"
DEFAULT_POOL_LOG = "gapminer_pool_records.log"
DEFAULT_TABLE = "data/prime_gap_merits.txt"

Z95 = 1.959963985


@dataclass
class Entry:
    """One parsed record-log line."""
    t: float                 # epoch seconds
    iso: str                 # original timestamp text
    path: str
    source: str              # "node" | "pool" (from the file name)
    status: str
    kv: dict = field(default_factory=dict)

    @property
    def is_discovery(self) -> bool:
        return self.status in DISCOVERY_STATUSES

    @property
    def key(self) -> tuple:
        """Identity of the candidate a line belongs to.

        A candidate is (height, shift, header_nonce, nAdd): the header nonce
        changes when the work is re-seated and nAdd is the window/adder offset,
        so the pair identifies the search position that produced the gap.  The
        discovery line and its verdict line share this key, which is what makes
        one candidate out of two lines.
        """
        kv = self.kv
        return (kv.get("height", "?"), kv.get("shift", "?"),
                kv.get("header_nonce", "?"), kv.get("nAdd", "?"))

    def f(self, name: str, default=None):
        return self.kv.get(name, default)

    def fnum(self, name: str):
        v = self.kv.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except ValueError:
            return None


@dataclass
class Candidate:
    """A unique found candidate, with every status line it produced."""
    t: float
    iso: str
    source: str
    path: str
    shift: int | None
    gap: int | None
    merit: float | None
    start: str | None          # decimal string, may be huge
    header_nonce: str | None
    nadd: str | None
    statuses: list[str] = field(default_factory=list)
    # Pool mode only: the template the solution was submitted against.
    template_prevhash: str | None = None   # display order, compares with getblockhash
    template_time: int | None = None
    template_merit: float | None = None

    @property
    def has_template(self) -> bool:
        return bool(self.template_prevhash)

    @property
    def discovery_status(self) -> str:
        for s in self.statuses:
            if s in DISCOVERY_STATUSES:
                return s
        return self.statuses[0] if self.statuses else "?"

    @property
    def verdict_status(self) -> str | None:
        """The terminal verdict, if the work source gave one.

        `stale`/`assemble-failed`/`duplicate`/`send-failed` are local outcomes,
        not work-source verdicts, but they all mean "no accepted/rejected came
        back", so they are reported side by side and never counted as accepts.
        """
        for s in self.statuses:
            if s in VERDICT_STATUSES:
                return s
        return None

    @property
    def ln_start(self) -> float | None:
        if not self.start:
            return None
        try:
            return math.log(int(self.start))
        except (ValueError, OverflowError):
            return None


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_log_line(line: str):
    m = TS_RE.match(line)
    if not m:
        return None
    iso = m.group("ts")
    kv = {k: v for k, v in KV_RE.findall(m.group("rest"))}
    status = kv.get("status")
    if status is None:
        return None
    try:
        t = _iso_to_epoch(iso)
    except ValueError:
        return None
    return iso, t, status, kv


def _iso_to_epoch(iso: str) -> float:
    # "YYYY-MM-DDTHH:MM:SSZ" — parsed by hand to stay independent of the
    # datetime module's Z handling across Python versions.
    from calendar import timegm
    from datetime import datetime
    dt = datetime(int(iso[0:4]), int(iso[5:7]), int(iso[8:10]),
                  int(iso[11:13]), int(iso[14:16]), int(iso[17:19]))
    return float(timegm(dt.timetuple()))


def guess_source(path: str) -> str:
    """The record log's own name says which work source wrote it."""
    base = os.path.basename(path).lower()
    return "pool" if "pool" in base else "node"


def load_file(path: str, since=None, until=None):
    """Parse one log; returns (entries, bad_lines).

    The miner fflushes every line, but it may still be RUNNING while this tool
    reads the file, so a final line without its newline yet is expected: it is
    counted as `partial`, not as malformed, and never as a candidate.
    """
    entries = []
    bad = 0
    source = guess_source(path)
    try:
        fh = open(path, "r", errors="ignore")
    except OSError as exc:
        print(f"!! cannot read {path}: {exc}", file=sys.stderr)
        return [], 0
    with fh:
        raw = fh.read()
    complete = raw.endswith("\n")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for idx, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = parse_log_line(line)
        if parsed is None:
            # Only the very last line may be a write in flight.
            if idx == len(lines) - 1 and not complete:
                continue
            bad += 1
            continue
        iso, t, status, kv = parsed
        if since is not None and t < since:
            continue
        if until is not None and t > until:
            continue
        entries.append(Entry(t=t, iso=iso, path=path, source=source,
                             status=status, kv=kv))
    return entries, bad


def build_candidates(entries):
    """Collapse discovery + verdict + submit-context lines into one Candidate."""
    by_key: dict[tuple, Candidate] = {}
    repeats = 0

    def note_submit_ctx(cand, e):
        """Attach the pool template fields carried by a `submitted` line."""
        cand.template_prevhash = e.f("template_prevhash", cand.template_prevhash)
        t = e.f("template_time")
        if t is not None:
            cand.template_time = _int(t)
        m = e.fnum("template_merit")
        if m is not None:
            cand.template_merit = m

    for e in entries:
        if e.is_discovery:
            key = e.key
            cand = by_key.get(key)
            if cand is None:
                start = e.f("start")
                cand = Candidate(
                    t=e.t, iso=e.iso, source=e.source, path=e.path,
                    shift=_int(e.f("shift")), gap=_int(e.f("gap")),
                    merit=e.fnum("merit"), start=start,
                    header_nonce=e.f("header_nonce"), nadd=e.f("nAdd"))
                by_key[key] = cand
            else:
                # The same search position reported twice: legitimate only if
                # the work was re-issued (e.g. the pool re-sent a template).
                repeats += 1
            cand.statuses.append(e.status)
        else:
            # Verdict or submit-context line: attach to its candidate; if the
            # discovery line fell outside the --since/--until window or was
            # never written, the line still counts (create the candidate from
            # it).  A submit-context line is NOT a verdict: it is recorded on the
            # candidate and never counted as an outcome.
            key = e.key
            cand = by_key.get(key)
            if cand is None:
                cand = Candidate(
                    t=e.t, iso=e.iso, source=e.source, path=e.path,
                    shift=_int(e.f("shift")), gap=_int(e.f("gap")),
                    merit=e.fnum("merit"), start=None,
                    header_nonce=e.f("header_nonce"), nadd=e.f("nAdd"))
                by_key[key] = cand
            if e.status == SUBMIT_CTX_STATUS:
                note_submit_ctx(cand, e)
            else:
                cand.statuses.append(e.status)
    return sorted(by_key.values(), key=lambda c: c.t), repeats


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── statistics ───────────────────────────────────────────────────────────────

def poisson_rate(n: int, hours: float):
    """(rate, lo, hi) candidates per hour, 95% normal-approximation interval.

    For small n the normal approximation is optimistic, so the caller prints
    the interval only as a scale and falls back to the exact zero-count bound
    when n == 0 (95% upper bound = 3.0 events, the standard Poisson rule).
    """
    if hours <= 0:
        return (float("nan"), float("nan"), float("nan"))
    rate = n / hours
    if n == 0:
        return (0.0, 0.0, 3.0 / hours)
    half = Z95 * math.sqrt(n) / hours
    return (rate, max(0.0, rate - half), rate + half)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def fit_sigma(merits, m0=None):
    """Mean-excess (maximum-likelihood) scale of a shifted exponential.

    sigma_hat = mean(m - m0) over m >= m0.  With m0 = the sample minimum this
    is the standard tail estimate (same convention as `gap_hunt_stats.py`); it
    is biased low because the minimum is itself a sample statistic, so a second
    estimate above the median is reported next to it as a stability check.
    """
    if not merits:
        return None, None
    m0 = min(merits) if m0 is None else m0
    tail = [m for m in merits if m >= m0]
    if len(tail) < 2:
        return None, m0
    return sum(m - m0 for m in tail) / len(tail), m0


def sigma_from_quantiles(merits):
    """Independent scale estimate from two quantiles (no MLE bias)."""
    s = sorted(merits)
    n = len(s)
    if n < 8:
        return None
    q25, q75 = quantile(s, 0.25), quantile(s, 0.75)
    c25 = sum(1 for m in s if m >= q25)
    c75 = sum(1 for m in s if m >= q75)
    if q75 <= q25 or c75 == 0:
        return None
    return (q75 - q25) / math.log(c25 / c75)


def ks_exponential(merits, m0, sigma):
    """Kolmogorov-Smirnov distance of the tail against Exp(m0, sigma).

    Returns (D, n, critical_95).  The critical value 1.36/sqrt(n) is the
    no-estimation-required bound; since sigma IS estimated from the same data
    the true bound is lower, so this test is CONSERVATIVE (it under-reports
    disagreement, never invents it).
    """
    tail = sorted(m for m in merits if m >= m0)
    n = len(tail)
    if n < 8 or not sigma or sigma <= 0:
        return None, n, None
    d = 0.0
    for i, m in enumerate(tail, start=1):
        model = 1.0 - math.exp(-(m - m0) / sigma)
        d = max(d, abs(model - (i - 1) / n), abs(model - i / n))
    return d, n, 1.36 / math.sqrt(n)


def load_table(path):
    table = {}
    try:
        fh = open(path, "r")
    except OSError:
        return None
    with fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                table[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return table


def outcome_label(c: Candidate) -> str:
    """The label a candidate's follow-up deserves.

    A `dry-run` candidate is never submitted, so it will never receive a verdict
    — calling that "no verdict yet" would invent a pending state that does not
    exist.  A `queue-full` candidate was dropped by the submission queue.  Only a
    submitted candidate can legitimately be waiting.
    """
    v = c.verdict_status
    if v:
        return v
    d = c.discovery_status
    if d == "dry-run":
        return "dry-run (not submitted)"
    if d == "queue-full":
        return "dropped (queue full)"
    return "submitted (awaiting verdict)"


def fmt_duration(hours: float) -> str:
    if hours < 1 / 60:
        return f"{hours * 3600:.0f}s"
    if hours < 1:
        return f"{hours * 60:.1f}min"
    if hours < 48:
        return f"{hours:.2f}h"
    return f"{hours / 24:.1f}d"


def split_runs(cands, gap_s: float = 1800.0):
    """Split a log into contiguous runs (a time gap > 30 min = a new session).

    One record log accumulates EVERY run the operator ever did with that file
    name, and each run may sit at a different merit threshold.  A single
    exponential fitted across such a mixture is not the distribution of any one
    configuration, so the runs are reported separately and the mixture is named
    as a possible cause of a misfit rather than left implicit.
    """
    runs = []
    current = []
    for c in cands:
        if current and c.t - current[-1].t > gap_s:
            runs.append(current)
            current = []
        current.append(c)
    if current:
        runs.append(current)
    return runs


# ── text report ──────────────────────────────────────────────────────────────

def report_group(title, cands, table, args, out=sys.stdout, comparable=True):
    """Print the analysis of one set of candidates (one file or a pooled set)."""
    cands = sorted(cands, key=lambda c: c.t)
    n = len(cands)
    print(f"== {title}", file=out)
    if n == 0:
        print("   no candidates in the selected window", file=out)
        print(file=out)
        return
    t0, t1 = cands[0].t, cands[-1].t
    hours = (t1 - t0) / 3600.0
    # A zero-width span would divide by zero; use the file's own span instead
    # of pretending the rate is infinite.
    span_ok = hours > 0
    rate, lo, hi = poisson_rate(n, hours) if span_ok else (float("nan"),) * 3
    merits = sorted(c.merit for c in cands if c.merit is not None)
    best = cands[max(range(n), key=lambda i: (cands[i].merit or 0.0))] \
        if merits else cands[0]

    print(f"   span           {cands[0].iso} .. {cands[-1].iso}  "
          f"({fmt_duration(hours)})", file=out)
    print(f"   candidates     {n}", file=out)
    if span_ok:
        print(f"   rate           {rate:.2f}/h  "
              f"(95% CI {lo:.2f}..{hi:.2f}, normal approximation)", file=out)
    else:
        print("   rate           n/a (all candidates carry the same timestamp)",
              file=out)

    if merits:
        print(f"   merit          min {merits[0]:.4f}  "
              f"median {quantile(merits, 0.5):.4f}  max {merits[-1]:.4f}",
              file=out)
        print(f"   best gap       {best.gap}  merit {best.merit:.4f}  "
              f"at {best.iso}  (shift {best.shift}, "
              f"header_nonce {best.header_nonce})", file=out)
        starts = [c.ln_start for c in cands if c.ln_start is not None]
        if starts:
            L = sum(starts) / len(starts)
            print(f"   window scale   L=ln(start) mean {L:.1f}   "
                  f"mean bits {L / math.log(2):.0f}   (n={len(starts)})",
                  file=out)

    # ── status accounting ────────────────────────────────────────────────────
    disc, verd = {}, {}
    for c in cands:
        disc[c.discovery_status] = disc.get(c.discovery_status, 0) + 1
        label = outcome_label(c)
        verd[label] = verd.get(label, 0) + 1
    print("   outcome        " + "  ".join(
        f"{k}={v}" for k, v in sorted(disc.items())), file=out)
    print("   ruled          " + "  ".join(
        f"{k}={v}" for k, v in sorted(verd.items())), file=out)
    if "rejected" in verd or "unresolved" in verd:
        rej_rate = verd.get("rejected", 0) / n * 100.0
        print(f"   !! {verd.get('rejected', 0)} rejected "
              f"({rej_rate:.1f}%), {verd.get('unresolved', 0)} unresolved",
              file=out)
        print("      A rejection with an unchanged target usually means stale "
              "work: the work source rotated its template and its notification "
              "was not adopted (see docs/POOL_STRATUM.md). Unresolved means the "
              "link dropped with the share in flight and is NOT a rejection.",
              file=out)

    # ── tail shape: is the merit distribution the one it must be? ────────────
    sigma, m0 = fit_sigma(merits)
    sigma_q = sigma_from_quantiles(merits)
    rows = []          # (threshold, observed, expected, ratio|None, sigma_ratio)
    if sigma:
        note = "" if len(merits) >= 30 else "   [n<30: NOISY]"
        print(f"   tail scale     sigma(mean-excess over min)={sigma:.3f}"
              + (f"   sigma(quartiles)={sigma_q:.3f}" if sigma_q else "")
              + note, file=out)
        print(f"   tail table     m0={m0:.4f}  N(m>=m0)={len(merits)}"
              + ("   [m0 is the sample minimum: the log itself is thresholded, "
                 "so this is a reference, not the population]" if m0 else ""),
              file=out)
        print("                    m>=" + " " * 8 + "observed  expected  ratio",
              file=out)
        for k in range(1, 6):
            thr = m0 + k * sigma
            obs = sum(1 for m in merits if m >= thr)
            exp = len(merits) * math.exp(-(thr - m0) / sigma)
            ratio = obs / exp if exp >= 5 else None
            sd = 1.0 / math.sqrt(exp) if exp > 0 else float("inf")
            rows.append((thr, obs, exp, ratio, sd))
        for thr, obs, exp, ratio, sd in rows:
            if ratio is None:
                shown, note = "      -", "   [expected <5: ratio not meaningful]"
            elif abs(ratio - 1.0) <= 2.0 * sd:
                shown = f"{ratio:7.2f}"
                note = f"   [within noise: 2 sigma = {2.0 * sd:.2f}]"
            else:
                shown = f"{ratio:7.2f}"
                note = f"   [beyond noise: 2 sigma = {2.0 * sd:.2f}]"
            print(f"                    {thr:10.4f} {obs:8d} {exp:9.1f} {shown}"
                  f"{note}", file=out)
        # A ratio only means something if it is larger than the Poisson error of
        # the count behind it: at 6 expected events a ratio of 0.70 is pure
        # noise (2 sigma = 0.84), and letting that drive the verdict would
        # manufacture a deviation out of the tail of the tail.
        outliers = [(r, e) for (_, _, e, r, sd) in rows
                    if r is not None and abs(r - 1.0) > max(0.25, 2.0 * sd)]
        informative = [r for (_, _, _, r, _) in rows if r is not None]
        if outliers:
            worst = max(abs(r - 1.0) for r, _ in outliers)
            deep_ratio = outliers[-1][0]
            tail_verdict = ("deviates from a single exponential (largest "
                            f"ratio deviation {worst * 100:.0f}%, beyond the "
                            "Poisson noise of the counts)")
            tail_verdict += (", thinner than the fit at the deep end"
                             if deep_ratio < 1 else
                             ", thicker than the fit at the deep end")
        else:
            tail_verdict = ("consistent with a single exponential (no ratio "
                            "beyond both 25% and 2 sigma)")
        print(f"   tail fit       {tail_verdict}", file=out)
        d, nn, crit = ks_exponential(merits, m0, sigma)
        if d is not None:
            print(f"   goodness       KS D={d:.3f} (n={nn}, conservative 95% "
                  f"bound {crit:.3f})"
                  + (" -> deviation is detectable, but at n this large KS "
                     "detects ANY deviation: read the ratio column for the "
                     "effect size" if d > crit else
                     " -> no deviation detected"), file=out)
        if outliers:
            print("                  Possible causes, in order of how often they "
                  "are the real one: the sample MIXES configurations or "
                  "thresholds (see the runs below), sigma really varies over "
                  "the run, or the reported set is incomplete. It is not by "
                  "itself evidence of missed candidates.", file=out)
        elif not informative:
            print("                  (every threshold is too deep to carry a "
                  "usable count: raise --top or collect more candidates)",
                  file=out)
    else:
        print("   tail scale     too few candidates for a tail fit", file=out)

    # Contiguous runs: one log file normally holds several sessions, and each
    # may sit at a different threshold.  Report them so a mixture is visible.
    runs = split_runs(cands)
    if len(runs) > 1:
        print(f"   runs           {len(runs)} session(s) in this file:", file=out)
        for i, run in enumerate(runs[:8], start=1):
            rm = [c.merit for c in run if c.merit is not None]
            rh = (run[-1].t - run[0].t) / 3600.0
            print(f"                    {i:2d}. n={len(run):6d}  "
                  f"min merit={min(rm):.4f}  "
                  f"span={fmt_duration(rh)}  from {run[0].iso}", file=out)
        if len(runs) > 8:
            print(f"                    ... and {len(runs) - 8} more", file=out)
        mins = [min(c.merit for c in run if c.merit is not None)
                for run in runs if any(c.merit is not None for c in run)]
        if len(mins) > 1:
            lo_r, hi_r = min(mins), max(mins)
            if hi_r > 0 and (hi_r - lo_r) / hi_r > 0.02:
                print(f"                  !! these runs sit at different "
                      f"thresholds (min merit {lo_r:.3f}..{hi_r:.3f}): "
                      f"their rates are NOT comparable, and the single-sigma "
                      f"tail fit above is a mixture, not any one run",
                      file=out)

    # ── template identity (pool mode, 2026-09-19 instrumentation) ───────────
    with_tpl = [c for c in cands if c.has_template]
    if with_tpl:
        prevs = {}
        for c in with_tpl:
            prevs.setdefault(c.template_prevhash, []).append(c)
        print(f"   templates      {len(prevs)} distinct template(s) over "
              f"{len(with_tpl)} submitted candidate(s)", file=out)
        for prev, group in sorted(prevs.items(), key=lambda kv: -len(kv[1]))[:5]:
            tm = [c.template_merit for c in group if c.template_merit]
            tt = [c.template_time for c in group if c.template_time]
            desc = f"prevhash {prev[:16]}.. (parents {len(group)} candidate(s))"
            if tm:
                desc += f" net merit {min(tm):.3f}"
            if tt:
                from datetime import datetime, timezone
                desc += (" time " + datetime.fromtimestamp(min(tt), timezone.utc)
                         .strftime("%H:%M:%SZ"))
            print(f"                    {desc}", file=out)

    # ── record proximity ─────────────────────────────────────────────────────
    if table:
        scored = []
        for c in cands:
            if c.gap is None or c.merit is None or c.gap not in table:
                continue
            scored.append((table[c.gap] - c.merit, c))
        scored.sort(key=lambda x: x[0])
        above = [x for x in scored if x[0] < 0]
        if above:
            print(f"   *** {len(above)} candidate(s) ABOVE the known best for "
                  f"their gap length: verify before claiming anything ***",
                  file=out)
            for delta, c in above[:10]:
                print(f"      gap={c.gap} merit={c.merit:.4f} "
                      f"table={table[c.gap]:.4f} delta={-delta:+.4f} "
                      f"{c.iso}", file=out)
                print(f"        scripts/verify_gap_candidate.py {c.gap} "
                      f"{c.merit:.4f} {c.start if c.start else '<start=missing>'}"
                      f"   # {os.path.basename(c.path)}", file=out)
        near = [x for x in scored if x[0] >= 0][:args.top]
        if near:
            print("   closest to a record (needed merit - found merit):",
                  file=out)
            for delta, c in near:
                print(f"      gap={c.gap:>7} merit={c.merit:.4f} "
                      f"needed={table[c.gap]:.4f} delta={delta:.4f}",
                      file=out)

    # ── the verdict the user actually asked for ──────────────────────────────
    if n == 0:
        pass
    elif not span_ok:
        print("   VERDICT        candidates present (rate not computable)",
              file=out)
    elif not comparable:
        print("   VERDICT        withheld: these files report at different "
              "thresholds, so a pooled rate would be meaningless — read the "
              "per-file sections above", file=out)
    else:
        acc = verd.get("accepted", 0)
        if acc and acc == n:
            print(f"   VERDICT        finding work: {n} candidate(s) in "
                  f"{fmt_duration(hours)}, all accepted", file=out)
        else:
            print(f"   VERDICT        finding work: {rate:.2f} candidates/h "
                  f"({n} in {fmt_duration(hours)})", file=out)
    print(file=out)


def make_plot(all_cands, groups, table, args, outpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("!! matplotlib not available — skipping the plot "
              "(pip install matplotlib)", file=sys.stderr)
        return False

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    ax = axes.ravel()
    colors = {"node": "#1f77b4", "pool": "#d62728", "other": "#2ca02c"}

    # (0) cumulative candidates vs time, with the Poisson band of the mean rate.
    # Normalized to each group's own total: absolute counts hide the shape when
    # one file holds 16k candidates and another 316 (the point of this panel is
    # "is it finding steadily?", which is a shape question).
    for name, (cands, _) in groups.items():
        if not cands:
            continue
        t0 = min(c.t for c in cands)
        xs = [(c.t - t0) / 3600.0 for c in sorted(cands, key=lambda c: c.t)]
        n = len(xs)
        ys = [(i + 1) / n for i in range(n)]
        color = colors.get(guess_source(name), colors["other"])
        ax[0].step(xs, ys, where="post", label=f"{os.path.basename(name)} "
                                               f"(n={n})", color=color)
        span = xs[-1] if xs else 0.0
        if span > 0:
            rate = 1.0 / span
            grid = [span * i / 100.0 for i in range(101)]
            ax[0].plot(grid, [rate * x for x in grid], ls=":", lw=1, color=color)
            ax[0].fill_between(
                grid,
                [max(0.0, rate * x - Z95 * math.sqrt(max(rate * x, 1e-12)) / n)
                 for x in grid],
                [min(1.0, rate * x + Z95 * math.sqrt(max(rate * x, 1e-12)) / n)
                 for x in grid],
                alpha=0.15, color=color)
    ax[0].set_title("cumulative candidates, normalized per file\n"
                    "(straight = steady rate; dotted = mean rate, band = 95% Poisson)")
    ax[0].set_xlabel("hours since that file's first candidate")
    ax[0].set_ylabel("share of that file's candidates")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    # (1) rate per wall-clock hour bin
    for name, (cands, _) in groups.items():
        if not cands:
            continue
        t0 = cands[0].t
        xs = [(c.t - t0) / 3600.0 for c in sorted(cands, key=lambda c: c.t)]
        nbins = max(1, min(48, int(math.ceil(max(xs[-1], 1e-6)) * 2)))
        bins = [0] * nbins
        width = (xs[-1] / nbins) if xs[-1] > 0 else 1.0
        for x in xs:
            idx = min(nbins - 1, int(x / max(width, 1e-9)))
            bins[idx] += 1
        rate = [b / max(width, 1e-9) for b in bins]
        centers = [(i + 0.5) * width for i in range(nbins)]
        ax[1].plot(centers, rate, marker="o", ms=3,
                   label=os.path.basename(name),
                   color=colors.get(guess_source(name), colors["other"]))
    ax[1].set_title("candidates per hour (binned, log scale)")
    ax[1].set_xlabel("hours since first candidate")
    ax[1].set_ylabel("candidates / h (log)")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # (2) merit histogram vs the exponential it must follow
    for name, (cands, _) in groups.items():
        merits = sorted(c.merit for c in cands if c.merit is not None)
        if len(merits) < 3:
            continue
        color = colors.get(guess_source(name), colors["other"])
        ax[2].hist(merits, bins=max(8, min(40, len(merits) // 2)),
                   alpha=0.5, label=f"{os.path.basename(name)} (n={len(merits)})",
                   color=color)
        sigma, m0 = fit_sigma(merits)
        if sigma:
            xs = [m0 + i * (merits[-1] - m0) / 200.0 for i in range(201)]
            width = max(1e-9, (merits[-1] - m0) / max(8, min(40, len(merits) // 2)))
            pdf = [len(merits) * width / sigma * math.exp(-(x - m0) / sigma)
                   for x in xs]
            ax[2].plot(xs, pdf, color=color, ls="--", lw=1.5,
                       label=f"exp fit sigma={sigma:.2f}")
    if args.share_target:
        ax[2].axvline(args.share_target, color="k", ls=":", lw=1)
        ax[2].text(args.share_target, 0, " share target", rotation=90, fontsize=7)
    ax[2].set_yscale("log")
    ax[2].set_title("merit distribution vs the fitted exponential")
    ax[2].set_xlabel("merit")
    ax[2].set_ylabel("candidates (log)")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    # (3) tail survival: observed P(merit >= m) against the fit
    for name, (cands, _) in groups.items():
        merits = sorted((c.merit for c in cands if c.merit is not None),
                        reverse=True)
        if len(merits) < 3:
            continue
        color = colors.get(guess_source(name), colors["other"])
        n = len(merits)
        ys = [(i + 1) / n for i in range(n)]
        ax[3].step(merits, ys, where="post", color=color,
                   label=os.path.basename(name))
        sigma, m0 = fit_sigma(merits)
        if sigma:
            xs = [m0 + i * (merits[0] - m0) / 200.0 for i in range(201)]
            ax[3].plot(xs, [math.exp(-(max(x - m0, 0.0)) / sigma) for x in xs],
                       color=color, ls="--", lw=1.2)
    ax[3].set_yscale("log")
    ax[3].set_title("tail: observed P(merit >= m) vs exponential fit\n"
                    "(a curve falling FASTER than its dash line = missing candidates)")
    ax[3].set_xlabel("merit")
    ax[3].set_ylabel("P(merit >= m)")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3)

    # (4) merit vs time — the "what did we find, when" view
    for name, (cands, _) in groups.items():
        pts = [(c.t, c.merit) for c in cands if c.merit is not None]
        if not pts:
            continue
        t0 = min(t for t, _ in pts)
        ax[4].scatter([(t - t0) / 3600.0 for t, _ in pts],
                      [m for _, m in pts], s=10, alpha=0.6,
                      color=colors.get(guess_source(name), colors["other"]),
                      label=os.path.basename(name))
        if table:
            recs = [(c.t, c.merit) for c in cands
                    if c.merit is not None and c.gap in table
                    and c.merit > table[c.gap]]
            if recs:
                ax[4].scatter([(t - t0) / 3600.0 for t, _ in recs],
                              [m for _, m in recs], s=90, marker="*",
                              edgecolor="k", zorder=5,
                              label="ABOVE known best (verify!)")
    ax[4].set_title("merit of every candidate over time")
    ax[4].set_xlabel("hours since first candidate")
    ax[4].set_ylabel("merit")
    ax[4].legend(fontsize=8)
    ax[4].grid(alpha=0.3)

    # (5) gap vs merit, with the required merit for a record per gap length
    if table:
        gs = sorted(table)
        pts = [(g, table[g]) for g in gs if 1000 <= g <= max(
            [c.gap for c in all_cands if c.gap] or [100000]) * 1.2]
        if pts:
            ax[5].plot([g for g, _ in pts], [m for _, m in pts], color="k",
                       lw=1, alpha=0.5, label="needed for a record (table)")
    for name, (cands, _) in groups.items():
        pts = [(c.gap, c.merit) for c in cands
               if c.gap is not None and c.merit is not None]
        if not pts:
            continue
        ax[5].scatter([g for g, _ in pts], [m for _, m in pts], s=12, alpha=0.6,
                      color=colors.get(guess_source(name), colors["other"]),
                      label=os.path.basename(name))
    ax[5].set_title("gap vs merit (at fixed L, merit = gap/L)\n"
                    "against the merit a record needs at that size")
    ax[5].set_xlabel("gap length")
    ax[5].set_ylabel("merit")
    # The table's required merit reaches far above any real find (and dips as low
    # as 25 for small gaps), which would flatten the candidates into the axis.
    # Zoom to the finds: the envelope stays visible where it matters.
    cand_merits = [c.merit for c in all_cands if c.merit is not None]
    if cand_merits:
        ax[5].set_ylim(min(cand_merits) - 2.0, max(cand_merits) + 5.0)
    ax[5].legend(fontsize=8)
    ax[5].grid(alpha=0.3)

    # (6) outcome MIX per hour — the series that reveals a stale-work outage.
    # Shares, not counts: with files of very different size a stacked count bar
    # chart hides the small file completely, and the question here is "what
    # fraction of what we found was accepted, hour by hour".
    order = ["accepted", "rejected", "unresolved", "duplicate", "send-failed",
             "assemble-failed", "stale", "dry-run (not submitted)",
             "dropped (queue full)", "submitted (awaiting verdict)"]
    for name, (cands, _) in groups.items():
        if not cands:
            continue
        t0 = min(c.t for c in cands)
        span_h = max((max(c.t for c in cands) - t0) / 3600.0, 1e-9)
        nbins = max(1, min(48, int(math.ceil(span_h * 2))))
        width = span_h / nbins
        counts = {k: [0] * nbins for k in order}
        for c in cands:
            idx = min(nbins - 1, int((c.t - t0) / 3600.0 / width))
            v = outcome_label(c)
            if v not in counts:
                counts[v] = [0] * nbins
                order.append(v)
            counts[v][idx] += 1
        totals = [sum(counts[k][i] for k in order) for i in range(nbins)]
        bottoms = [0.0] * nbins
        centers = [(i + 0.5) * width for i in range(nbins)]
        palette = {"accepted": "#2ca02c", "rejected": "#d62728",
                   "unresolved": "#ff7f0e", "stale": "#9467bd",
                   "duplicate": "#8c564b", "send-failed": "#7f7f7f",
                   "assemble-failed": "#bcbd22",
                   "dry-run (not submitted)": "#c7d4e8",
                   "dropped (queue full)": "#e8c7d4",
                   "submitted (awaiting verdict)": "#c7c7c7"}
        for key in order:
            vals = [v / t if t else 0.0 for v, t in zip(counts[key], totals)]
            if not any(vals):
                continue
            ax[6].bar(centers, vals, width=width * 0.9, bottom=bottoms,
                      label=f"{os.path.basename(name)}:{key}",
                      color=palette.get(key, "#333333"), alpha=0.85)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax[6].set_ylim(0.0, 1.0)
    ax[6].set_title("outcome MIX per hour (share of that hour's candidates)\n"
                    "a stale-work outage turns its bars red")
    ax[6].set_xlabel("hours since first candidate")
    ax[6].set_ylabel("share of candidates")
    ax[6].legend(fontsize=7, ncol=2)
    ax[6].grid(alpha=0.3, axis="y")

    # (7) the same numbers as text, so the PNG is self-contained
    ax[7].axis("off")
    lines = ["summary"]
    for name, (cands, _) in groups.items():
        if not cands:
            lines.append(f"{os.path.basename(name)}: no candidates")
            continue
        hours = (cands[-1].t - cands[0].t) / 3600.0
        merits = [c.merit for c in cands if c.merit is not None]
        rate = len(cands) / hours if hours > 0 else float("nan")
        sig, m0 = fit_sigma(sorted(merits)) if merits else (None, None)
        rej = sum(1 for c in cands if c.verdict_status == "rejected")
        lines += [
            f"{os.path.basename(name)}",
            f"  candidates {len(cands)}   span {fmt_duration(hours)}",
            f"  rate       {rate:.2f}/h" if hours > 0 else "  rate       n/a",
            f"  merit      {min(merits):.4f}..{max(merits):.4f}" if merits
            else "  merit      n/a",
            f"  sigma      {sig:.3f}" if sig else "  sigma      n/a",
            f"  rejected   {rej}",
        ]
    ax[7].text(0.0, 0.98, "\n".join(lines), va="top", ha="left",
               family="monospace", fontsize=9)

    fig.suptitle("gapminer record-log report — " +
                 ", ".join(f"{os.path.basename(k)}" for k in groups),
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    # --plot out/x.png must work even when out/ does not exist yet; and a write
    # failure has to be a one-line message, not a traceback after a whole report.
    outdir = os.path.dirname(os.path.abspath(outpath))
    try:
        os.makedirs(outdir, exist_ok=True)
    except OSError as exc:
        print(f"!! cannot create the plot directory {outdir}: {exc}",
              file=sys.stderr)
        return False
    try:
        fig.savefig(outpath, dpi=110)
    except (OSError, ValueError) as exc:
        print(f"!! cannot write the plot to {outpath}: {exc}", file=sys.stderr)
        return False
    print(f"plot written: {outpath}")
    return True


# ── main ─────────────────────────────────────────────────────────────────────

def discover_logs():
    """Default input: the two record logs (and their rotated copies) in CWD."""
    found = []
    for pattern in (DEFAULT_NODE_LOG, DEFAULT_POOL_LOG):
        found += sorted(glob.glob(pattern + "*"))
    out, seen = [], set()
    for f in found:
        if not os.path.isfile(f) or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Read the miner record logs and report/plot what is "
                    "being found (node and pool).")
    p.add_argument("logs", nargs="*", help="record log files "
                                          f"(default: {DEFAULT_NODE_LOG} and "
                                          f"{DEFAULT_POOL_LOG} in CWD)")
    p.add_argument("--log", action="append", default=[],
                   help="add a log file (repeatable)")
    p.add_argument("--source", choices=["all", "node", "pool"], default="all",
                   help="restrict to logs written by one work source")
    p.add_argument("--since", default=None, help="ISO time, e.g. 2026-09-19T12:00")
    p.add_argument("--until", default=None, help="ISO time, upper bound")
    p.add_argument("--table", default=DEFAULT_TABLE,
                   help=f"known best merit per gap length (default: {DEFAULT_TABLE})")
    p.add_argument("--top", type=int, default=5,
                   help="how many near-record candidates to list (default 5)")
    p.add_argument("--share-target", type=float, default=None,
                   help="draw this merit as the pool share target")
    p.add_argument("--span-hours", type=float, default=None,
                   help="observation span of a run that found NOTHING, so an "
                        "empty log becomes a 95%% upper bound instead of a "
                        "blank (also inferred when --since and --until are "
                        "both given)")
    p.add_argument("--plot", nargs="?", const="records_report.png", default=None,
                   metavar="PATH", help="write a multi-panel PNG "
                                        "(default: records_report.png)")
    p.add_argument("--selftest", action="store_true",
                   help="check the parser and the statistics on a synthetic "
                        "log with a known distribution, then exit")
    args = p.parse_args(argv)

    if args.selftest:
        return selftest()

    since = _parse_when(args.since)
    until = _parse_when(args.until)

    paths = list(args.logs) + list(args.log)
    if not paths:
        paths = discover_logs()
    if not paths:
        print(f"no record logs found (looked for {DEFAULT_NODE_LOG}* and "
              f"{DEFAULT_POOL_LOG}*)", file=sys.stderr)
        return 2

    table = load_table(args.table)
    if table is None:
        print(f"!! cannot read {args.table}: record proximity skipped",
              file=sys.stderr)

    groups = {}
    thresholds = []
    for path in paths:
        if args.source != "all" and guess_source(path) != args.source:
            continue
        entries, bad = load_file(path, since=since, until=until)
        cands, repeats = build_candidates(entries)
        groups[path] = (cands, bad)
        if bad:
            print(f"!! {path}: {bad} unparsable line(s) ignored", file=sys.stderr)
        if repeats:
            print(f"!! {path}: {repeats} repeated discovery line(s) for the "
                  f"same (height, shift, nonce, nAdd) — re-issued work?",
                  file=sys.stderr)
        if cands:
            merits = [c.merit for c in cands if c.merit is not None]
            if merits:
                thresholds.append((path, min(merits)))

    if not groups:
        print("no logs matched the selected source/window", file=sys.stderr)
        return 2

    all_cands = [c for cands, _ in groups.values() for c in cands]
    if not all_cands:
        # Nothing found is a MEASUREMENT, not a missing measurement: with a
        # known span it becomes a Poisson upper bound on the rate.
        print(f"no candidates in the selected window ({len(groups)} file(s) "
              f"read)")
        for path, (_, bad) in groups.items():
            print(f"   {path}: 0 candidates"
                  + (f" ({bad} unparsable lines)" if bad else ""))
        span_h = _observation_span(args, since, until)
        if span_h and span_h > 0:
            bound = 3.0 / span_h
            print(f"   observation span {fmt_duration(span_h)} -> 0 candidates "
                  f"gives a 95% Poisson upper bound of {bound:.3f}/h "
                  f"({bound * 24:.1f}/day)")
            print("   i.e. the run cannot be said to find more than that; it is "
                  "NOT evidence that the configuration finds nothing")
        else:
            print("   give --since and --until (or --span-hours) to turn "
                  "'nothing' into a rate upper bound")
        return 0

    for path in paths:
        if path in groups:
            cands, _ = groups[path]
            report_group(os.path.basename(path), cands, table, args)

    # Pooled view is only meaningful at a shared threshold; say so instead of
    # silently averaging incomparable rates.
    comparable = True
    if len(groups) > 1:
        lo = min(t for _, t in thresholds)
        hi = max(t for _, t in thresholds)
        if lo > 0 and (hi - lo) / hi > 0.02:
            comparable = False
            print("WARNING: these files report at different thresholds "
                  "(minimum merit differs "
                  f"{lo:.3f}..{hi:.3f}); candidate RATES are not comparable "
                  "between them — compare them per file, and compare a rate "
                  "only against a run at the same threshold.")
            print()
        report_group(f"POOLED ({len(groups)} files)", all_cands, table, args,
                     comparable=comparable)

    if args.plot:
        if not make_plot(all_cands, groups, table, args, args.plot):
            return 1
    return 0


def selftest() -> int:
    """Deterministic check of the parser, the pairing and the statistics.

    Builds a synthetic log with a KNOWN exponential merit distribution (sigma
    1.2, m0 15.0) and checks that the tool recovers the count, the pairing
    (discovery line + verdict line = ONE candidate), the sigma and the outcome
    mix — plus the empty-log rate bound.  No real logs and no mining needed.
    """
    import random
    import tempfile

    print("selftest: synthetic log with sigma=1.2, m0=15.0, n=400")
    rng = random.Random(1234)
    n = 400
    rng_ok = 0
    tmpdir = tempfile.mkdtemp(prefix="records_report_selftest_")
    # The file name must contain "pool" for the source label to come out right.
    path = os.path.join(tmpdir, "gapminer_pool_records.log")
    with open(path, "w") as fh:
        for i in range(n):
            # Inverse-CDF sample of a shifted exponential.
            m = 15.0 + (-math.log(1.0 - rng.random())) * 1.2
            gap = int(round(m * 531.0))
            ts = f"2026-09-19T10:{i // 60:02d}:{i % 60:02d}Z"
            base = (f"{ts} height=0 shift=509 header_nonce={1000 + i} "
                    f"nAdd={i * 7919}")
            if i % 2 == 0:
                fh.write(f"{base} start={10 ** 200 + i} gap={gap} "
                         f"merit={m:.4f} best_known_merit=unknown "
                         f"new_record=unknown claim=none status=queued\n")
                if i % 10 == 0:
                    rng_ok += 1
                    fh.write(f"{base} gap={gap} merit={m:.4f} status=rejected\n")
                else:
                    fh.write(f"{base} gap={gap} merit={m:.4f} status=accepted\n")
            else:
                fh.write(f"{base} start={10 ** 200 + i} gap={gap} "
                         f"merit={m:.4f} best_known_merit=unknown "
                         f"new_record=unknown claim=none status=dry-run\n")
    entries, bad = load_file(path)
    cands, repeats = build_candidates(entries)
    lines = len(entries)
    half = n // 2
    merits = sorted(c.merit for c in cands if c.merit is not None)
    sigma, m0 = fit_sigma(merits)
    rejected = sum(1 for c in cands if c.verdict_status == "rejected")
    accepted = sum(1 for c in cands if c.verdict_status == "accepted")
    not_submitted = sum(1 for c in cands if outcome_label(c).startswith("dry-run"))
    rate, lo, hi = poisson_rate(n, 10.0 / 60.0)

    checks = [
        ("parsed every line", f"{lines} lines, {bad} bad",
         lines == n + half and bad == 0),
        ("one candidate per found gap", f"{len(cands)} candidates (want {n})",
         len(cands) == n),
        ("no repeated discovery line", f"repeats={repeats}", repeats == 0),
        ("verdict attached to its candidate", f"accepted={accepted} "
         f"rejected={rejected} (want {half - rng_ok}/{rng_ok})",
         accepted == half - rng_ok and rejected == rng_ok),
        ("dry-run is not called 'awaiting verdict'",
         f"{not_submitted} dry-run candidates", not_submitted == n - half),
        ("sigma recovered", f"sigma={sigma:.3f} (want 1.2 +-15%)",
         abs(sigma - 1.2) / 1.2 < 0.15),
        ("tail fit accepts a true exponential",
         f"D={ks_exponential(merits, m0, sigma)[0]:.3f}",
         ks_exponential(merits, m0, sigma)[0] <=
         ks_exponential(merits, m0, sigma)[2]),
        ("empty log gives a rate bound", "3.0/1h = 3.0/h",
         abs(poisson_rate(0, 1.0)[2] - 3.0) < 1e-9 and lo < rate < hi),
    ]
    failed = 0
    for name, detail, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name:42} {detail}")
        failed += 0 if ok else 1
    try:
        os.remove(path)
        os.rmdir(tmpdir)
    except OSError:
        pass
    print(f"selftest: {'PASS' if not failed else f'{failed} FAILED'}")
    return 0 if not failed else 1


def _observation_span(args, since, until):
    """Hours of observation, when the caller can supply them."""
    if args.span_hours:
        return float(args.span_hours)
    if since is not None and until is not None and until > since:
        return (until - since) / 3600.0
    return None


def _parse_when(text):
    """Accept 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM' and 'YYYY-MM-DDTHH:MM:SS[Z]'."""
    if not text:
        return None
    t = text.strip().rstrip("Z")
    fmts = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d")
    for fmt in fmts:
        try:
            from calendar import timegm
            from datetime import datetime
            return float(timegm(datetime.strptime(t, fmt).timetuple()))
        except ValueError:
            continue
    raise SystemExit(f"cannot parse time '{text}' "
                     "(use e.g. 2026-09-19T12:00)")


if __name__ == "__main__":
    sys.exit(main())
