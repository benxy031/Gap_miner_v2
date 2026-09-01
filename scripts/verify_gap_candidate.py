#!/usr/bin/env python3
"""
verify_gap_candidate.py

Strict verifier for prime-gap candidates emitted by gap_range_search.

Supported inputs:
1) A full scanner line, e.g.
   gap=1736 merit=19.2961 merit_mode=prev prev=... next=... tag=m=1064
2) Explicit values via --prev and --next.
3) Submit-style triple: "<gap> <merit> <startprime>".

Checks performed:
- Arithmetic gap: next - prev
- Merit consistency: gap / ln(prev)
- Endpoint probable-prime checks (OpenSSL 'prime -checks N')
- Strict intermediate scan: every odd in (prev, next) must be composite

Exit code:
- 0 on valid consecutive-prime gap candidate
- 1 otherwise
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

LINE_RE = re.compile(
    r"gap=(?P<gap>\d+)\s+"
    r"merit=(?P<merit>[0-9]+(?:\.[0-9]+)?)\s+"
    r"merit_mode=[^\s]+\s+"
    r"prev=(?P<prev>\d+)\s+"
    r"next=(?P<next>\d+)"
)

SUBMIT_RE = re.compile(
    r"^\s*(?P<gap>\d+)\s+"
    r"(?P<merit>[0-9]+(?:\.[0-9]+)?)\s+"
    r"(?P<start>\d+)\s*$"
)


@dataclass
class Candidate:
    prev: int
    nextp: int
    reported_gap: Optional[int] = None
    reported_merit: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict checker for reported prime-gap candidates.")
    p.add_argument("--line", type=str, default="", help="Full scanner line with gap/merit/prev/next, or first token of '<gap> <merit> <startprime>'.")
    p.add_argument("--prev", type=int, help="Start prime (prev).")
    p.add_argument("--next", dest="nextp", type=int, help="End prime (next).")
    p.add_argument("--gap", type=int, default=None, help="Reported gap (optional).")
    p.add_argument("--merit", type=float, default=None, help="Reported merit (optional).")
    p.add_argument("--checks-endpoints", type=int, default=64, help="OpenSSL primality rounds for endpoints.")
    p.add_argument("--checks-intermediate", type=int, default=32, help="OpenSSL primality rounds for intermediate odd values.")
    p.add_argument("--merit-tol", type=float, default=1e-3, help="Allowed absolute tolerance between reported and computed merit.")
    p.add_argument("--timeout", type=int, default=20, help="Timeout (seconds) per primality call.")
    p.add_argument("--show-progress", action="store_true", help="Show intermediate scan progress.")
    p.add_argument("positional", nargs="*", help="Optional submit-style tokens: <gap> <merit> <startprime>.")
    return p.parse_args()


def parse_candidate(ns: argparse.Namespace) -> Candidate:
    # Accept submit-style triple from positionals: <gap> <merit> <startprime>
    if not ns.line and len(ns.positional) == 3:
        triple = " ".join(ns.positional)
        m = SUBMIT_RE.match(triple)
        if not m:
            raise ValueError("Could not parse positional submit triple. Expected: <gap> <merit> <startprime>.")
        gap = int(m.group("gap"))
        start = int(m.group("start"))
        return Candidate(
            prev=start,
            nextp=start + gap,
            reported_gap=gap,
            reported_merit=float(m.group("merit")),
        )

    # Accept user style: --line <gap> <merit> <startprime> (without quotes)
    if ns.line and len(ns.positional) == 2:
        triple = f"{ns.line} {ns.positional[0]} {ns.positional[1]}"
        m = SUBMIT_RE.match(triple)
        if m:
            gap = int(m.group("gap"))
            start = int(m.group("start"))
            return Candidate(
                prev=start,
                nextp=start + gap,
                reported_gap=gap,
                reported_merit=float(m.group("merit")),
            )

    if ns.line:
        s = ns.line.strip()
        m = LINE_RE.search(s)
        if m:
            return Candidate(
                prev=int(m.group("prev")),
                nextp=int(m.group("next")),
                reported_gap=int(m.group("gap")),
                reported_merit=float(m.group("merit")),
            )

        m = SUBMIT_RE.match(s)
        if m:
            gap = int(m.group("gap"))
            start = int(m.group("start"))
            return Candidate(
                prev=start,
                nextp=start + gap,
                reported_gap=gap,
                reported_merit=float(m.group("merit")),
            )

        raise ValueError(
            "Could not parse --line. Expected scanner line with gap/merit/prev/next "
            "or submit triple '<gap> <merit> <startprime>'."
        )

    if ns.prev is None or ns.nextp is None:
        raise ValueError("Provide one of: --line, positional <gap> <merit> <startprime>, or both --prev and --next.")

    return Candidate(
        prev=ns.prev,
        nextp=ns.nextp,
        reported_gap=ns.gap,
        reported_merit=ns.merit,
    )


def openssl_is_probable_prime(n: int, checks: int, timeout_sec: int) -> bool:
    proc = subprocess.run(
        ["openssl", "prime", "-checks", str(checks), str(n)],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenSSL primality check failed for n={} (rc={}): {}{}".format(
                n,
                proc.returncode,
                proc.stdout,
                proc.stderr,
            )
        )
    return "is prime" in proc.stdout.lower()


def main() -> int:
    try:
        ns = parse_args()
        c = parse_candidate(ns)
    except Exception as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    if c.nextp <= c.prev:
        print("FAIL: next must be greater than prev.")
        return 1

    actual_gap = c.nextp - c.prev
    computed_merit = actual_gap / math.log(c.prev)

    print(f"prev={c.prev}")
    print(f"next={c.nextp}")
    print(f"actual_gap={actual_gap}")
    print(f"computed_merit={computed_merit:.12f}")

    if c.reported_gap is not None:
        gap_ok = (actual_gap == c.reported_gap)
        print(f"reported_gap={c.reported_gap} match={int(gap_ok)}")
    else:
        gap_ok = True

    if c.reported_merit is not None:
        merit_delta = abs(computed_merit - c.reported_merit)
        merit_ok = merit_delta <= ns.merit_tol
        print(
            "reported_merit={:.12f} delta={:.12e} tol={:.3e} match={}".format(
                c.reported_merit,
                merit_delta,
                ns.merit_tol,
                int(merit_ok),
            )
        )
    else:
        merit_ok = True

    try:
        prev_prime = openssl_is_probable_prime(c.prev, ns.checks_endpoints, ns.timeout)
        next_prime = openssl_is_probable_prime(c.nextp, ns.checks_endpoints, ns.timeout)
    except Exception as exc:
        print(f"FAIL: endpoint primality check failed: {exc}", file=sys.stderr)
        return 1

    print(f"prev_probable_prime={int(prev_prime)}")
    print(f"next_probable_prime={int(next_prime)}")

    if not prev_prime or not next_prime:
        print("FAIL: at least one endpoint is not probable-prime.")
        return 1

    first_intermediate_prime = None
    tested = 0
    n = c.prev + 2
    if (n & 1) == 0:
        n += 1

    while n < c.nextp:
        tested += 1
        if ns.show_progress and tested % 1000 == 0:
            print(f"...tested_intermediate_odds={tested}")
        try:
            if openssl_is_probable_prime(n, ns.checks_intermediate, ns.timeout):
                first_intermediate_prime = n
                break
        except Exception as exc:
            print(f"FAIL: intermediate primality check failed at n={n}: {exc}", file=sys.stderr)
            return 1
        n += 2

    print(f"odd_intermediates_tested={tested}")

    if first_intermediate_prime is not None:
        print(f"FAIL: intermediate probable-prime found: {first_intermediate_prime}")
        print(f"first_subgap={first_intermediate_prime - c.prev}")
        return 1

    if not gap_ok or not merit_ok:
        print("FAIL: arithmetic mismatch vs reported gap/merit.")
        return 1

    print("PASS: strict consecutive-prime gap candidate is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
