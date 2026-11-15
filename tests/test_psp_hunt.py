#!/usr/bin/env python3
"""Regression tests for the standalone pseudoprime hunter."""

import argparse
import fcntl
import json
import math
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from scripts import psp_hunt as hunt
from scripts import psp_math, psp_probe


BRIAN_SMALL_GAPS = (
    (424592033333333333333, 11896666667, 35689999999, 312),
    (992350730874983898601, 31429264441, 31574099761, 314),
    (473732207945649671653, 21754338913, 21776446981, 386),
    (740970121244702824021, 2273429341, 325926171481, 354),
)


def small_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def brute_pseudoprimes(start, end, floor, semiprime_only=False):
    out = set()
    for n in range(start | 1, end, 2):
        if pow(2, n - 1, n) != 1:
            continue
        factors = small_factors(n)
        if len(factors) < 2 or min(factors) <= floor:
            continue
        if semiprime_only and len(factors) != 2:
            continue
        out.add(n)
    return out


def run_cli(*arguments, timeout=30):
    return subprocess.run([sys.executable, str(SCRIPTS / "psp_hunt.py"), *arguments],
                          cwd=ROOT, text=True, capture_output=True, timeout=timeout)


class PredicateTests(unittest.TestCase):
    def test_exact_int_and_bound(self):
        self.assertEqual(hunt.exact_int("1e21"), 10**21)
        self.assertEqual(hunt.exact_int("7.9e28"), 79 * 10**27)
        self.assertEqual(hunt.exact_int("1.20e1"), 12)
        self.assertEqual(hunt.exact_int("100e-2"), 1)
        for value in ("1e-1", "-1", "nan", "1.5"):
            with self.assertRaises(argparse.ArgumentTypeError):
                hunt.exact_int(value)
        self.assertTrue(psp_math.proven_prime_mr12(2))
        self.assertTrue(psp_math.proven_prime_mr12(37))
        self.assertFalse(psp_math.proven_prime_mr12(341))
        self.assertIsNone(psp_math.mr12_witness(psp_math.MR12_LIMIT))
        with self.assertRaises(ValueError):
            psp_math.proven_prime_mr12(psp_math.MR12_LIMIT)

    def test_order_and_sieve_against_brute_force(self):
        primes = hunt.prime_table(200)
        for p in primes[1:]:
            order = 1
            while pow(2, order, p) != 1:
                order += 1
            self.assertEqual(hunt.order_base2(p, primes), order)
        for start, end, floor in ((3, 50, 20), (10, 104, 7), (341, 400, 30)):
            primes = hunt.prime_table(floor)
            expected = [n for n in range(start | 1, end, 2)
                        if all(n == p or n % p for p in primes)]
            self.assertEqual(list(hunt.survivors(start, end, primes)), expected)

    def test_pairs_and_scan_match_small_brute_force(self):
        end = 50_000
        config = {"mode": "pairs", "start": 3, "end": end, "factor_floor": 2,
                  "p_start": 3, "p_end": math.isqrt(end - 1) + 1}
        hunt.validate_config(config)
        primes = hunt.prime_table(math.isqrt(config["p_end"] - 1))
        cursor = hunt.initial_cursor(config)
        records = []
        complete = False
        while not complete:
            cursor, _, batch, complete = hunt.pairs_batch(config, cursor, 53, primes)
            records.extend(batch)
        self.assertEqual({int(r["n"]) for r in records},
                         brute_pseudoprimes(3, end, 2, semiprime_only=True))
        all_pairs = brute_pseudoprimes(3, end, 2, semiprime_only=True)
        for start, stop, floor, p_start, p_end in (
                (341, 342, 2, 11, 12), (342, 50_000, 2, 11, 12),
                (3, 50_000, 10, 11, 50), (1000, 10_000, 2, 3, 90)):
            config = {"mode": "pairs", "start": start, "end": stop,
                      "factor_floor": floor, "p_start": p_start, "p_end": p_end}
            hunt.validate_config(config)
            cursor = hunt.initial_cursor(config)
            records = []
            complete = False
            while not complete:
                cursor, _, batch, complete = hunt.pairs_batch(
                    config, cursor, 53, hunt.prime_table(math.isqrt(p_end - 1)))
                records.extend(batch)
            expected = {n for n in all_pairs if start <= n < stop
                        and floor < small_factors(n)[0]
                        and p_start <= small_factors(n)[0] < p_end}
            self.assertEqual({int(r["n"]) for r in records}, expected)
        for floor in (2, 3, 7, 20):
            scan_config = {"mode": "scan", "start": 3, "end": end, "factor_floor": floor}
            cursor = hunt.initial_cursor(scan_config)
            records = []
            complete = False
            while not complete:
                cursor, _, batch, complete = hunt.scan_batch(
                    scan_config, cursor, 997, hunt.prime_table(floor))
                records.extend(batch)
            self.assertEqual({int(r["n"]) for r in records},
                             brute_pseudoprimes(3, end, floor))
            if floor == 2:
                self.assertIn(561, {int(r["n"]) for r in records})

    def test_prime_square_and_three_factors(self):
        n = 1093**2
        config = {"mode": "pairs", "start": n, "end": n + 1, "factor_floor": 1000,
                  "p_start": 1093, "p_end": 1094}
        cursor, _, hits, complete = hunt.pairs_batch(
            config, hunt.initial_cursor(config), 100, hunt.prime_table(34))
        self.assertTrue(complete)
        self.assertEqual(cursor["p"], 1094)
        self.assertEqual([int(h["n"]) for h in hits], [n])
        self.assertEqual(hits[0]["factors"], ["1093", "1093"])

        config = {"mode": "chernick", "k_start": 1, "k_end": 2, "factor_floor": 2}
        _, _, hits, complete = hunt.chernick_batch(config, hunt.initial_cursor(config), 10)
        self.assertTrue(complete)
        self.assertEqual(hits[0]["n"], "1729")
        self.assertEqual(hits[0]["factors"], ["7", "13", "19"])
        config = {"mode": "scan", "start": 1729, "end": 1730, "factor_floor": 2}
        _, _, hits, complete = hunt.scan_batch(config, hunt.initial_cursor(config), 10, [2])
        self.assertTrue(complete)
        self.assertEqual(hits[0]["n"], "1729")

    def test_all_brian_reference_cases(self):
        items = [tuple(map(int, line.split()))
                 for line in psp_probe.BRIK_800.strip().splitlines()]
        self.assertEqual(len(items), 61)
        euler = strong = 0
        for n, p, q, gap in items:
            config = {"mode": "pairs", "start": n, "end": n + 1, "factor_floor": 10**7,
                      "p_start": p, "p_end": p + 1}
            primes = hunt.prime_table(math.isqrt(p))
            _, _, hits, complete = hunt.pairs_batch(config, hunt.initial_cursor(config),
                                                     10, primes)
            self.assertTrue(complete)
            self.assertEqual(len(hits), 1, n)
            hit = hits[0]
            self.assertEqual(hit["n"], str(n))
            self.assertEqual(hit["factors"], [str(p), str(q)])
            self.assertEqual(hit["gap"]["size"], gap)
            self.assertEqual(hit["gap"]["status"], "proven_mr12")
            for minimum in (800, 802):
                filtered_config = dict(config, min_gap=minimum)
                _, delta, filtered_hits, filtered_complete = hunt.pairs_batch(
                    filtered_config, hunt.initial_cursor(filtered_config), 10, primes)
                self.assertTrue(filtered_complete)
                self.assertEqual(filtered_hits, hits if gap >= minimum else [])
                self.assertEqual(delta["confirmed"], 1)
                self.assertEqual(delta["gap_filtered"], int(gap < minimum))
            euler += hit["euler_jacobi"]
            strong += hit["strong_base2"]
        self.assertEqual((euler, strong), (37, 26))

    def test_gap_filter_proof_failure_and_short_gap_fast_path(self):
        for n, p, q, gap in BRIAN_SMALL_GAPS:
            self.assertEqual(hunt.confirmed_hit(n, "pairs", 10**7, [p, q])["gap"]["size"], gap)
            config = {"mode": "pairs", "start": n, "end": n + 1, "factor_floor": 10**7,
                      "p_start": p, "p_end": p + 1, "min_gap": 800}
            with patch.object(hunt, "confirmed_hit", side_effect=AssertionError("full record")):
                _, delta, hits, complete = hunt.pairs_batch(
                    config, hunt.initial_cursor(config), 10, hunt.prime_table(math.isqrt(p)))
            self.assertTrue(complete)
            self.assertEqual(hits, [])
            self.assertEqual(delta["confirmed"], 1)
            self.assertEqual(delta["gap_filtered"], 1)
        with patch.object(hunt, "proven_prime_mr12", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "endpoint failed"):
                hunt.gap_below(n, 800)
        with self.assertRaisesRegex(ValueError, "MR12 bound"):
            hunt.gap_below(psp_math.MR12_LIMIT, 800)
        with self.assertRaisesRegex(ValueError, "product range"):
            hunt.validate_config(dict(config, end=psp_math.MR12_LIMIT + 1))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            hunt.validate_config(dict(config, min_gap=-1))

    def test_gap_filter_preserves_all_candidate_work(self):
        config = {"mode": "pairs", "start": 3, "end": 50_000, "factor_floor": 2,
                  "p_start": 3, "p_end": 224}
        primes = hunt.prime_table(math.isqrt(config["p_end"]))

        def collect(settings):
            cursor = hunt.initial_cursor(settings)
            records, totals, complete = [], {}, False
            while not complete:
                cursor, delta, batch, complete = hunt.pairs_batch(settings, cursor, 53, primes)
                records.extend(batch)
                for key, value in delta.items():
                    totals[key] = totals.get(key, 0) + value
            return cursor, records, totals

        cursor, records, totals = collect(config)
        for minimum in (10, 20, 800):
            filtered_cursor, hits, stats = collect(dict(config, min_gap=minimum))
            self.assertEqual(filtered_cursor, cursor)
            self.assertEqual(hits, [r for r in records if r["gap"]["size"] >= minimum])
            for key in ("q_tested", "p_completed", "confirmed", "fermat_passers"):
                self.assertEqual(stats[key], totals[key])
            self.assertEqual(stats["gap_filtered"], len(records) - len(hits))

    def test_above_mr12_bound_is_not_silently_discarded(self):
        n = psp_math.MR12_LIMIT
        config = {"mode": "scan", "start": n, "end": n + 1, "factor_floor": 2}
        _, delta, hits, complete = hunt.scan_batch(config, hunt.initial_cursor(config), 1, [2])
        self.assertTrue(complete)
        self.assertEqual(delta["pending"], 1)
        self.assertEqual(hits[0]["status"], "pending")
        self.assertEqual(hunt.gap_context(n)["status"], "probable_gmp")
        self.assertEqual(hunt.gap_context(341)["lower"], "337")
        self.assertEqual(hunt.gap_context(341)["upper"], "347")

    def test_explicit_work_and_proof_limits(self):
        config = {"mode": "scan", "start": 3, "end": 10, "factor_floor": 25_000_001}
        with self.assertRaisesRegex(ValueError, "prime-table limit"):
            hunt.validate_config(config)
        config = {"mode": "pairs", "start": 3, "end": 4 * psp_math.MR12_LIMIT,
                  "factor_floor": 2, "p_start": 3, "p_end": 10}
        with self.assertRaisesRegex(ValueError, "MR12 bound"):
            hunt.validate_config(config)
        config = {"mode": "chernick", "factor_floor": 2, "k_start": 1,
                  "k_end": psp_math.MR12_LIMIT}
        with self.assertRaisesRegex(ValueError, "MR12 bound"):
            hunt.validate_config(config)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_atomic_checkpoint_and_foreign_database(self):
        config = {"version": hunt.VERSION, "mode": "scan", "factor_floor": 2,
                  "start": 3, "end": 400}
        path = self.root / "atomic.sqlite3"
        with hunt.Store(path, config) as store:
            record = {"n": "341", "status": "confirmed", "gap": {"size": 10}}
            with self.assertRaises(sqlite3.IntegrityError):
                store.checkpoint({"n": 400}, {"confirmed": 2}, [record, record], True)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0], 0)
            self.assertEqual(json.loads(store.db.execute(
                "SELECT cursor FROM progress WHERE id=1").fetchone()[0]), {"n": 3})
        foreign = self.root / "foreign.sqlite3"
        with sqlite3.connect(foreign) as db:
            db.execute("CREATE TABLE user_data (value TEXT)")
            db.execute("INSERT INTO user_data VALUES ('preserve')")
        with self.assertRaisesRegex(ValueError, "not a psp_hunt"):
            with hunt.Store(foreign, config):
                pass
        with sqlite3.connect(foreign) as db:
            self.assertEqual(db.execute("SELECT value FROM user_data").fetchone()[0], "preserve")
        foreign_view = self.root / "foreign-view.sqlite3"
        with sqlite3.connect(foreign_view) as db:
            db.execute("CREATE VIEW user_view AS SELECT 1")
        with self.assertRaisesRegex(ValueError, "not a psp_hunt"):
            with hunt.Store(foreign_view, config):
                pass

    def test_pairs_pause_resume_and_configuration_guard(self):
        resumed = self.root / "resumed.sqlite3"
        single = self.root / "single.sqlite3"
        common = ("pairs", "--p-start", "10000000001", "--p-end", "10000003001")
        first = run_cli(*common, "--db", str(resumed), "--batch-size", "5",
                        "--max-batches", "1")
        self.assertEqual(first.returncode, 4, first.stderr)
        second = run_cli(*common, "--db", str(resumed), "--batch-size", "1000")
        self.assertEqual(second.returncode, 0, second.stderr)
        reference = run_cli(*common, "--db", str(single), "--batch-size", "20000")
        self.assertEqual(reference.returncode, 0, reference.stderr)
        with sqlite3.connect(resumed) as a, sqlite3.connect(single) as b:
            self.assertEqual(a.execute("SELECT n,payload FROM results ORDER BY n").fetchall(),
                             b.execute("SELECT n,payload FROM results ORDER BY n").fetchall())
            self.assertEqual(a.execute("SELECT COUNT(*) FROM results").fetchone()[0], 17)
            self.assertEqual(json.loads(a.execute("SELECT stats FROM progress").fetchone()[0])["q_tested"],
                             json.loads(b.execute("SELECT stats FROM progress").fetchone()[0])["q_tested"])
        conflict = run_cli(*common, "--db", str(resumed), "--end", "3e21")
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("configuration differs", conflict.stderr)
        shown = run_cli("show", "--db", str(resumed), "--min-gap", "100", "--limit", "3")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        rows = [json.loads(line) for line in shown.stdout.splitlines()]
        self.assertLessEqual(len(rows), 3)
        self.assertTrue(all(row["gap"]["size"] >= 100 for row in rows))
        completed_again = run_cli(*common, "--db", str(resumed))
        self.assertEqual(completed_again.returncode, 0, completed_again.stderr)
        self.assertIn("COMPLETE", completed_again.stdout)

    def test_generic_scan_brian_control_with_default_floor(self):
        path = self.root / "brian.sqlite3"
        n = 160696461657079906273
        result = run_cli("scan", "--start", str(n), "--end", str(n + 1), "--db", str(path))
        self.assertEqual(result.returncode, 0, result.stderr)
        shown = run_cli("show", "--db", str(path))
        row = json.loads(shown.stdout)
        self.assertEqual(row["n"], str(n))
        self.assertEqual(row["gap"]["size"], 956)
        self.assertEqual(row["factor_floor"], 10_000_000)

    def test_filtered_pairs_resume_output_and_database_identity(self):
        p = 9710332633
        common = ("pairs", "--start", "1e20", "--end", "1e21",
                  "--p-start", str(p), "--p-end", str(p + 1000))
        all_path = self.root / "all.sqlite3"
        filtered_path = self.root / "filtered.sqlite3"
        result = run_cli(*common, "--db", str(all_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run_cli(*common, "--db", str(filtered_path), "--min-gap", "800",
                         "--batch-size", "31", "--max-batches", "1")
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("retained=", result.stdout)
        result = run_cli(*common, "--db", str(filtered_path), "--min-gap", "800",
                         "--batch-size", "10000")
        self.assertEqual(result.returncode, 0, result.stderr)
        hit_lines = [json.loads(line[4:]) for line in result.stdout.splitlines()
                     if line.startswith("HIT ")]
        self.assertTrue(all(r["gap"]["size"] >= 800 for r in hit_lines))
        with sqlite3.connect(all_path) as a, sqlite3.connect(filtered_path) as b:
            expected = a.execute(
                "SELECT n,payload FROM results WHERE CAST(gap AS INTEGER)>=800 ORDER BY n").fetchall()
            self.assertEqual(b.execute("SELECT n,payload FROM results ORDER BY n").fetchall(),
                             expected)
            self.assertTrue(expected)
            all_stats = json.loads(a.execute("SELECT stats FROM progress").fetchone()[0])
            stats = json.loads(b.execute("SELECT stats FROM progress").fetchone()[0])
            for key in ("q_tested", "p_completed", "confirmed"):
                self.assertEqual(stats[key], all_stats[key])
            self.assertGreater(stats["gap_filtered"], 0)
            self.assertEqual(stats["confirmed"] - stats["gap_filtered"], len(expected))
        for minimum in ("0", "802"):
            result = run_cli(*common, "--db", str(filtered_path), "--min-gap", minimum)
            self.assertEqual(result.returncode, 2)
            self.assertIn("configuration differs", result.stderr)
        result = run_cli(*common, "--db", str(all_path), "--min-gap", "800")
        self.assertEqual(result.returncode, 2)
        result = run_cli(*common, "--db", str(all_path), "--min-gap", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        shown = run_cli("show", "--db", str(filtered_path), "--limit", "0")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("gap_filtered=", shown.stderr)
        self.assertIn("min_gap=800", shown.stderr)

    def test_gap_proof_failure_does_not_commit_progress(self):
        n, p, _, _ = map(int, psp_probe.BRIK_800.strip().splitlines()[0].split())
        config = {"version": hunt.VERSION, "mode": "pairs", "start": n, "end": n + 1,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 1, "min_gap": 800}
        with hunt.Store(self.root / "failed-gap.sqlite3", config) as store:
            with patch.object(hunt, "gap_neighbors", side_effect=RuntimeError("proof failure")):
                with self.assertRaisesRegex(RuntimeError, "proof failure"):
                    hunt.pairs_batch(config, store.cursor, 100, hunt.prime_table(math.isqrt(p)))
            self.assertEqual(store.cursor, hunt.initial_cursor(config))
            self.assertEqual(store.stats["batches"], 0)
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM results").fetchone()[0], 0)

    def test_pending_exit_and_exclusive_lock(self):
        path = self.root / "pending.sqlite3"
        n = str(psp_math.MR12_LIMIT)
        result = run_cli("scan", "--start", n, "--end", str(int(n) + 1),
                         "--factor-floor", "2", "--db", str(path))
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("require classification", result.stderr)
        shown = run_cli("show", "--db", str(path), "--status", "pending")
        self.assertEqual(json.loads(shown.stdout)["n"], n)

        locked = self.root / "locked.sqlite3"
        with open(str(locked) + ".lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run_cli("scan", "--start", "3", "--end", "100", "--db", str(locked))
        self.assertEqual(result.returncode, 2)
        self.assertIn("database lock", result.stderr)
        self.assertFalse(locked.exists())
        target = self.root / "target.sqlite3"
        target.touch()
        alias = self.root / "alias.sqlite3"
        alias.symlink_to(target)
        with open(str(target) + ".lock", "a+", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run_cli("scan", "--start", "3", "--end", "100", "--db", str(alias))
        self.assertEqual(result.returncode, 2)
        self.assertIn("database lock", result.stderr)

    def test_show_numeric_order_and_large_filter(self):
        path = self.root / "show.sqlite3"
        config = {"version": hunt.VERSION, "mode": "scan", "factor_floor": 2,
                  "start": 3, "end": 400}
        records = [
            {"n": "99", "status": "confirmed", "gap": {"size": 9}},
            {"n": "100", "status": "confirmed", "gap": {"size": 100}},
            {"n": "101", "status": "confirmed", "gap": {"size": 100}},
            {"n": "100000", "status": "confirmed", "gap": {"size": 10**25}},
            {"n": "1000000", "status": "pending"},
        ]
        with hunt.Store(path, config) as store:
            store.checkpoint({"n": 400}, {"confirmed": 4, "pending": 1}, records, True)
        result = run_cli("show", "--db", str(path), "--status", "all", "--limit", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([json.loads(line)["n"] for line in result.stdout.splitlines()],
                         ["100000", "101", "100", "99", "1000000"])
        self.assertIn("shown=5 matching=5", result.stderr)
        result = run_cli("show", "--db", str(path), "--min-gap", "1e25", "--limit", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["n"], "100000")
        self.assertIn("shown=1 matching=1", result.stderr)

    def test_sigterm_keeps_checkpoint_resumable(self):
        path = self.root / "signal.sqlite3"
        start = 10**21
        common = ("scan", "--start", str(start), "--end", str(start + 10_000_000),
                  "--factor-floor", "2", "--batch-size", "1000", "--db", str(path))
        process = subprocess.Popen([sys.executable, str(SCRIPTS / "psp_hunt.py"), *common],
                                   cwd=ROOT, text=True, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
        self.addCleanup(lambda: process.poll() is None and process.kill())
        deadline = time.monotonic() + 10
        next_n = None
        while time.monotonic() < deadline and process.poll() is None:
            if path.exists():
                try:
                    with sqlite3.connect(path) as db:
                        row = db.execute("SELECT cursor FROM progress").fetchone()
                    next_n = json.loads(row[0])["n"]
                    if next_n > start:
                        break
                except sqlite3.OperationalError:
                    pass
            time.sleep(0.02)
        self.assertIsNone(process.poll(), "search finished before interruption")
        self.assertIsNotNone(next_n)
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 128 + signal.SIGTERM, stderr)
        with sqlite3.connect(path) as db:
            before = json.loads(db.execute("SELECT cursor FROM progress").fetchone()[0])["n"]
        resumed = run_cli(*common, "--max-batches", "1")
        self.assertEqual(resumed.returncode, 4, resumed.stderr)
        with sqlite3.connect(path) as db:
            after = json.loads(db.execute("SELECT cursor FROM progress").fetchone()[0])["n"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
