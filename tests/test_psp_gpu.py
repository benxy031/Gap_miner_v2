"""GPU PSP gates, anchored to BrianK's published semiprimes."""

import ctypes
import contextlib
import json
import math
import os
import random
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
sys.path.insert(0, str(ROOT / "scripts"))
from scripts import psp_hunt as hunt, psp_gpu as gpu, psp_probe
from test_psp_hunt import BRIAN_SMALL_GAPS, run_cli


BRIAN = [tuple(map(int, line.split()))
         for line in psp_probe.BRIK_800.strip().splitlines()]


class OracleFilter:
    def filter(self, ranges):
        return [job.offset + i for job in ranges for i in range(job.count)
                if pow(2, job.p - 1, job.first + i * job.step) == 1], 0.0


class SchedulingTests(unittest.TestCase):
    def test_single_prime_segment_needs_no_sieve_factors(self):
        config = {"mode": "pairs", "start": 3, "end": 100,
                  "factor_floor": 2, "p_start": 3, "p_end": 4}
        primes = tuple(hunt.prime_table(1))
        cache = gpu.PrimeOrderCache(hunt, primes)
        cursor = hunt.initial_cursor(config)
        expected = hunt.pairs_batch(config, cursor, 100, primes)
        self.assertEqual(gpu.pairs_batch(
            hunt, config, cursor, 100, primes, OracleFilter(), cache), expected)
        self.assertEqual(gpu.pairs_batch(
            hunt, config, cursor, 100, primes, OracleFilter()), expected)

    def test_cached_random_small_slices_match_cpu(self):
        rng = random.Random(20261110)
        for case in range(50):
            end = rng.randint(200, 20000)
            floor = rng.choice((2, 3, 7))
            largest = math.isqrt(end - 1)
            p_start = rng.randint(max(3, floor + 1), largest)
            p_end = rng.randint(p_start + 1, largest + 1)
            config = {"mode": "pairs", "start": rng.randint(3, end - 1), "end": end,
                      "factor_floor": floor, "p_start": p_start, "p_end": p_end,
                      "min_gap": rng.choice((0, 5, 20, 50))}
            primes = tuple(hunt.prime_table(math.isqrt(p_end - 1)))
            cache = gpu.PrimeOrderCache(hunt, primes)
            oracle = OracleFilter()
            cursor = hunt.initial_cursor(config)
            batch_size = rng.randint(1, 32)
            with self.subTest(case=case, config=config, batch_size=batch_size), \
                    patch.object(gpu, "SEGMENT", rng.choice((1, 2, 7, 16))):
                complete = False
                while not complete:
                    actual = gpu.pairs_batch(
                        hunt, config, cursor, batch_size, primes, oracle, cache)
                    self.assertEqual(actual, hunt.pairs_batch(config, cursor, batch_size, primes))
                    cursor, _, _, complete = actual

    def test_cached_segments_match_cpu_across_batches_and_rewinds(self):
        config = {"mode": "pairs", "start": 3, "end": 5000,
                  "factor_floor": 2, "p_start": 3, "p_end": 71}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"] - 1)))
        cache = gpu.PrimeOrderCache(hunt, primes)
        oracle = OracleFilter()
        cursor = hunt.initial_cursor(config)
        records, batches = [], 0
        with patch.object(gpu, "SEGMENT", 8), \
                patch.object(hunt, "prime_orders", wraps=hunt.prime_orders) as sieve:
            first = gpu.pairs_batch(hunt, config, cursor, 31, primes, oracle, cache)
            self.assertEqual(first, hunt.pairs_batch(config, cursor, 31, primes))
            self.assertEqual(sieve.call_count, 1)
            self.assertEqual(gpu.pairs_batch(
                hunt, config, cursor, 31, primes, oracle, cache), first)
            self.assertEqual(sieve.call_count, 1)

            narrow = dict(config, p_end=7)
            self.assertEqual(
                gpu.pairs_batch(hunt, narrow, cursor, 10000, primes, oracle, cache),
                hunt.pairs_batch(narrow, cursor, 10000, primes))

            complete = False
            while not complete:
                expected = hunt.pairs_batch(config, cursor, 31, primes)
                actual = gpu.pairs_batch(hunt, config, cursor, 31, primes, oracle, cache)
                self.assertEqual(actual, expected)
                cursor, _, hits, complete = actual
                records.extend(hits)
                batches += 1
            self.assertTrue(records)
            self.assertLess(sieve.call_count, batches)
            self.assertEqual(gpu.pairs_batch(
                hunt, config, hunt.initial_cursor(config), 31, primes, oracle, cache), first)

        with self.assertRaisesRegex(ValueError, "does not match"):
            gpu.pairs_batch(hunt, config, cursor, 31, tuple(list(primes)), oracle, cache)
        with self.assertRaisesRegex(TypeError, "immutable"):
            gpu.PrimeOrderCache(hunt, list(primes))

    def test_cached_empty_segments_do_not_end_search(self):
        config = {"mode": "pairs", "start": 3, "end": 5000,
                  "factor_floor": 2, "p_start": 23, "p_end": 31}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"] - 1)))
        cache = gpu.PrimeOrderCache(hunt, primes)
        cursor = hunt.initial_cursor(config)
        with patch.object(gpu, "SEGMENT", 2), \
                patch.object(hunt, "prime_orders", wraps=hunt.prime_orders) as sieve:
            complete = False
            while not complete:
                actual = gpu.pairs_batch(hunt, config, cursor, 3, primes, OracleFilter(), cache)
                self.assertEqual(actual, hunt.pairs_batch(config, cursor, 3, primes))
                cursor, _, _, complete = actual
            segments = [(call.args[0], call.args[1]) for call in sieve.call_args_list]
            self.assertIn((25, 27), segments)
            self.assertIn((27, 29), segments)

    def test_cpu_scheduler_parity_on_brian_factor_progression(self):
        _, p, _, _ = BRIAN[0]
        config = {"mode": "pairs", "start": 10**20, "end": 10**21,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 1000}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"])))
        cache = gpu.PrimeOrderCache(hunt, primes)
        oracle = OracleFilter()
        a = b = hunt.initial_cursor(config)
        complete = False
        self.assertEqual(gpu.pairs_batch(hunt, config, b, 31, primes, oracle),
                         hunt.pairs_batch(config, a, 31, primes))
        while not complete:
            expected = hunt.pairs_batch(config, a, 31, primes)
            actual = gpu.pairs_batch(hunt, config, b, 31, primes, oracle, cache)
            self.assertEqual(actual, expected)
            a = b = actual[0]
            complete = actual[3]

    def test_range_limit_and_failed_batch_keeps_database(self):
        _, p, _, _ = BRIAN[0]
        config = {"version": hunt.VERSION, "mode": "pairs", "start": 10**20,
                  "end": 10**21, "factor_floor": 10**7,
                  "p_start": p, "p_end": p + 1}
        bad = dict(config, end=p * gpu.FACTOR_LIMIT + 1)
        with self.assertRaisesRegex(ValueError, "2\\^63"):
            gpu.validate_range(bad, 1000)
        for size in (0, gpu.MAX_BATCH + 1):
            with self.assertRaises(ValueError):
                gpu.validate_range(config, size)
        for device in (-1, 1 << 32):
            with self.assertRaisesRegex(ValueError, "32-bit"):
                gpu.Filter(device=device)
        primes = tuple(hunt.prime_table(math.isqrt(p)))
        cache = gpu.PrimeOrderCache(hunt, primes)
        oracle = OracleFilter()
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "checkpoint.sqlite3"
            with hunt.Store(db, config) as store:
                before = store.cursor.copy()
                before_progress = store.db.execute(
                    "select cursor,stats,complete from progress").fetchone()
                with patch.object(hunt, "prime_orders", wraps=hunt.prime_orders) as sieve:
                    with patch.object(OracleFilter, "filter",
                                      side_effect=RuntimeError("CUDA failure")):
                        with self.assertRaisesRegex(RuntimeError, "CUDA failure"):
                            gpu.pairs_batch(
                                hunt, config, store.cursor, 1000, primes, oracle, cache)
                    self.assertEqual(sieve.call_count, 1)
                    actual = gpu.pairs_batch(
                        hunt, config, store.cursor, 1000, primes, oracle, cache)
                    self.assertEqual(actual, hunt.pairs_batch(config, before, 1000, primes))
                    self.assertEqual(sieve.call_count, 1)
                self.assertEqual(store.cursor, before)
                self.assertEqual(store.db.execute("select count(*) from results").fetchone()[0], 0)
                self.assertEqual(store.db.execute(
                    "select cursor,stats,complete from progress").fetchone(), before_progress)
        with patch.object(hunt, "STOP_SIGNAL", signal.SIGTERM):
            result = gpu.pairs_batch(hunt, config, hunt.initial_cursor(config),
                                     1000, [], OracleFilter())
        self.assertEqual(result[0], hunt.initial_cursor(config))
        self.assertEqual(result[2], [])

    def test_signal_during_sieving_leaves_cursor_retryable(self):
        config = {"mode": "pairs", "start": 3, "end": 5000,
                  "factor_floor": 2, "p_start": 3, "p_end": 71}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"] - 1)))
        cache = gpu.PrimeOrderCache(hunt, primes)
        cursor = hunt.initial_cursor(config)
        original = hunt.prime_orders

        def interrupted(*args):
            entries = original(*args)
            hunt.STOP_SIGNAL = signal.SIGTERM
            return entries

        with patch.object(hunt, "STOP_SIGNAL", 0), \
                patch.object(hunt, "prime_orders", side_effect=interrupted) as sieve:
            stopped = gpu.pairs_batch(hunt, config, cursor, 31, primes, OracleFilter(), cache)
            self.assertEqual(stopped[0], cursor)
            self.assertEqual(stopped[2], [])
            hunt.STOP_SIGNAL = 0
            resumed = gpu.pairs_batch(hunt, config, cursor, 31, primes, OracleFilter(), cache)
            self.assertEqual(resumed, hunt.pairs_batch(config, cursor, 31, primes))
            self.assertEqual(sieve.call_count, 1)

    def test_descriptor_limit_commits_prefix_without_skipping(self):
        _, p, _, _ = BRIAN[0]
        config = {"mode": "pairs", "start": 10**20, "end": 10**21,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 1000}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"])))
        cache = gpu.PrimeOrderCache(hunt, primes)
        oracle = OracleFilter()
        reference = hunt.pairs_batch(config, hunt.initial_cursor(config), 100000, primes)
        cursor, complete, records, q_tested = hunt.initial_cursor(config), False, [], 0
        with patch.object(gpu, "MAX_RANGES", 2):
            while not complete:
                cursor, delta, hits, complete = gpu.pairs_batch(
                    hunt, config, cursor, 100000, primes, oracle, cache)
                records.extend(hits)
                q_tested += delta["q_tested"]
        self.assertEqual(cursor, reference[0])
        self.assertEqual(records, reference[2])
        self.assertEqual(q_tested, reference[1]["q_tested"])


@unittest.skipUnless(os.environ.get("PSP_GPU_TEST") == "1",
                     "GPU gates require PSP_GPU_TEST=1 (make test-psp-gpu)")
class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.filter = gpu.Filter(capacity=1 << 20)

    @classmethod
    def tearDownClass(cls):
        cls.filter.close()

    def test_smallest_factor_with_empty_prime_table(self):
        config = {"mode": "pairs", "start": 3, "end": 100,
                  "factor_floor": 2, "p_start": 3, "p_end": 4}
        primes = tuple(hunt.prime_table(1))
        cache = gpu.PrimeOrderCache(hunt, primes)
        cursor = hunt.initial_cursor(config)
        actual = gpu.pairs_batch(hunt, config, cursor, 100, primes, self.filter, cache)
        self.assertEqual(actual, hunt.pairs_batch(config, cursor, 100, primes))
        self.assertEqual(actual[1]["q_tested"], 16)
        self.assertEqual(self.filter.telemetry["filter_passers"], 1)
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "p3.sqlite3"
            result = run_cli("pairs", "--backend", "cuda", "--start", "3", "--end", "100",
                             "--factor-floor", "2", "--p-start", "3", "--p-end", "4",
                             "--db", str(db))
            self.assertEqual(result.returncode, 0, result.stderr)
            with contextlib.closing(sqlite3.connect(db)) as connection:
                stats, complete = connection.execute(
                    "select stats,complete from progress").fetchone()
                self.assertEqual(json.loads(stats)["q_tested"], 16)
                self.assertEqual(complete, 1)

    def test_all_61_brian_psps_with_factors_and_gaps(self):
        self.assertEqual(len(BRIAN), 61)
        euler = strong = 0
        primes = tuple(hunt.prime_table(math.isqrt(max(x[1] for x in BRIAN))))
        cache = gpu.PrimeOrderCache(hunt, primes)
        for n, p, q, gap in BRIAN:
            config = {"mode": "pairs", "start": n, "end": n + 1,
                      "factor_floor": 10**7, "p_start": p, "p_end": p + 1}
            _, _, records, complete = gpu.pairs_batch(
                hunt, config, hunt.initial_cursor(config), 100, primes, self.filter, cache)
            self.assertTrue(complete)
            self.assertEqual(len(records), 1, n)
            record = records[0]
            self.assertEqual(record["n"], str(n))
            self.assertEqual(record["factors"], [str(p), str(q)])
            self.assertEqual(record["gap"]["size"], gap)
            self.assertEqual(record["gap"]["status"], "proven_mr12")
            for minimum in (800, 802):
                filtered_config = dict(config, min_gap=minimum)
                expected = hunt.pairs_batch(
                    filtered_config, hunt.initial_cursor(filtered_config), 100, primes)
                actual = gpu.pairs_batch(
                    hunt, filtered_config, hunt.initial_cursor(filtered_config),
                    100, primes, self.filter, cache)
                self.assertEqual(actual, expected)
                self.assertEqual(actual[2], records if gap >= minimum else [])
                self.assertEqual(actual[1]["confirmed"], 1)
                self.assertEqual(actual[1]["gap_filtered"], int(gap < minimum))
                self.assertEqual(self.filter.telemetry["gap_filtered"], int(gap < minimum))
            euler += record["euler_jacobi"]
            strong += record["strong_base2"]
        self.assertEqual((euler, strong), (37, 26))

    def test_filtered_brian_slice_max_batch_and_mixed_resume(self):
        _, p, _, _ = min(BRIAN, key=lambda item: item[1])
        config = {"mode": "pairs", "start": 10**20, "end": 10**21,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 1000, "min_gap": 800}
        primes = tuple(hunt.prime_table(math.isqrt(config["p_end"])))
        cache = gpu.PrimeOrderCache(hunt, primes)
        expected = hunt.pairs_batch(config, hunt.initial_cursor(config), gpu.MAX_BATCH, primes)
        with gpu.Filter(capacity=gpu.MAX_BATCH) as large:
            actual = gpu.pairs_batch(
                hunt, config, hunt.initial_cursor(config), gpu.MAX_BATCH, primes, large, cache)
        self.assertEqual(actual, expected)
        self.assertTrue(actual[3])
        self.assertGreater(actual[1]["gap_filtered"], 0)
        self.assertTrue(actual[2])
        common = ("pairs", "--start", "1e20", "--end", "1e21",
                  "--p-start", str(p), "--p-end", str(p + 1000), "--min-gap", "800")
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "mixed.sqlite3"
            result = run_cli(*common, "--db", str(db), "--batch-size", "31",
                             "--max-batches", "1")
            self.assertEqual(result.returncode, 4, result.stderr)
            result = run_cli(*common, "--db", str(db), "--backend", "cuda",
                             "--batch-size", str(gpu.MAX_BATCH))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("gap_filtered", result.stdout)
            with contextlib.closing(sqlite3.connect(db)) as connection:
                stored = [json.loads(row[0]) for row in connection.execute(
                    "select payload from results order by n")]
                stats = json.loads(connection.execute("select stats from progress").fetchone()[0])
                self.assertEqual(stored, sorted(expected[2], key=lambda r: r["n"]))
                for key in ("confirmed", "gap_filtered", "q_tested", "p_completed"):
                    self.assertEqual(stats[key], expected[1][key])
                self.assertEqual(connection.execute("pragma integrity_check").fetchone()[0], "ok")

    def test_brian_short_gaps_filtered_without_full_record(self):
        primes = tuple(hunt.prime_table(
            math.isqrt(max(p for n, p, q, gap in BRIAN_SMALL_GAPS))))
        cache = gpu.PrimeOrderCache(hunt, primes)
        for n, p, q, gap in BRIAN_SMALL_GAPS:
            self.assertEqual(hunt.confirmed_hit(n, "pairs", 10**7, [p, q])["gap"]["size"], gap)
            config = {"mode": "pairs", "start": n, "end": n + 1, "factor_floor": 10**7,
                      "p_start": p, "p_end": p + 1, "min_gap": 800}
            with patch.object(hunt, "confirmed_hit", side_effect=AssertionError("full record")):
                result = gpu.pairs_batch(
                    hunt, config, hunt.initial_cursor(config), 100, primes, self.filter, cache)
            self.assertTrue(result[3])
            self.assertEqual(result[2], [])
            self.assertEqual(result[1]["confirmed"], 1)
            self.assertEqual(result[1]["gap_filtered"], 1)

    def test_brian_neighborhoods_and_signed_64_boundary(self):
        ranges = []
        offset = 0
        for _, p, q, _ in BRIAN:
            ranges.append(gpu.Range(p, q - 128, 2, 129, offset))
            offset += 129
            ranges.append(gpu.Range(p, (1 << 63) - 513, 2, 257, offset))
            offset += 257
        expected, _ = OracleFilter().filter(ranges)
        actual, _ = self.filter.filter(ranges)
        self.assertEqual(actual, expected)
        # Mersenne moduli supply true congruences at large widths, not only rejects.
        ranges = []
        for _, p, _, _ in BRIAN:
            for bits in range(32, 64):
                if (p - 1) % bits == 0:
                    ranges.append(gpu.Range(p, (1 << bits) - 1, 2, 1, len(ranges)))
        self.assertTrue(any(job.first == (1 << 63) - 1 for job in ranges))
        expected, _ = OracleFilter().filter(ranges)
        actual, _ = self.filter.filter(ranges)
        self.assertEqual(expected, list(range(len(ranges))))
        self.assertEqual(actual, expected)

    def test_dense_output_compaction_and_invalid_arguments(self):
        _, p, q, _ = BRIAN[0]
        ranges = [gpu.Range(p, q, 2, 1, i) for i in range(4096)]
        actual, _ = self.filter.filter(ranges)
        self.assertEqual(actual, list(range(4096)))
        for job in (gpu.Range(p, q, 1, 2, 0),
                    gpu.Range(p, 1 << 63, 2, 1, 0),
                    gpu.Range(p, q, 0, 1, 0)):
            with self.assertRaises((ValueError, RuntimeError)):
                self.filter.filter([job])
        with self.assertRaises(ValueError):
            self.filter.filter([gpu.Range(p, q, 2, 1, 1)])
        count, milliseconds = ctypes.c_uint64(), ctypes.c_float()
        rc = self.filter.lib.psp_gpu_filter(
            self.filter.ctx, (gpu.Range * 1)(gpu.Range(p, q, 2, 1, 0)),
            1, self.filter.capacity + 1, self.filter.indices,
            ctypes.byref(count), ctypes.byref(milliseconds))
        self.assertNotEqual(rc, 0)

    def test_brian_progression_cpu_gpu_resume_and_cli_failures(self):
        _, p, _, _ = min(BRIAN, key=lambda item: item[1])
        common = ("pairs", "--start", "1e20", "--end", "1e21",
                  "--p-start", str(p), "--p-end", str(p + 1000))
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "mixed.sqlite3"
            reference = Path(directory) / "cpu.sqlite3"
            result = run_cli(*common, "--db", str(db), "--batch-size", "31",
                             "--max-batches", "1")
            self.assertEqual(result.returncode, 4, result.stderr)
            result = run_cli(*common, "--db", str(db), "--backend", "cuda",
                             "--batch-size", "10000", "--max-batches", "5")
            self.assertIn(result.returncode, (0, 4), result.stderr)
            self.assertGreaterEqual(result.stdout.count("CUDA "), 2)
            result = run_cli(*common, "--db", str(db))
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_cli(*common, "--db", str(reference))
            self.assertEqual(result.returncode, 0, result.stderr)
            with contextlib.closing(sqlite3.connect(db)) as a, \
                    contextlib.closing(sqlite3.connect(reference)) as b:
                self.assertEqual(a.execute("select n,payload from results order by n").fetchall(),
                                 b.execute("select n,payload from results order by n").fetchall())
                for key in ("q_tested", "p_completed", "confirmed"):
                    self.assertEqual(json.loads(a.execute("select stats from progress").fetchone()[0])[key],
                                     json.loads(b.execute("select stats from progress").fetchone()[0])[key])
            for option in (("--gpu-library", str(Path(directory) / "missing.so")),
                           ("--device", "99999")):
                failed = run_cli(*common, "--db", str(Path(directory) / (option[0][2:] + ".sqlite3")),
                                 "--backend", "cuda", *option)
                self.assertEqual(failed.returncode, 2, failed.stderr)
                self.assertNotIn("COMPLETE", failed.stdout)

    def test_brian_factor_squares_keep_full_fermat_guard(self):
        _, p, _, _ = BRIAN[0]
        n = p * p
        config = {"mode": "pairs", "start": n, "end": n + 1,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 1}
        primes = tuple(hunt.prime_table(math.isqrt(p)))
        cache = gpu.PrimeOrderCache(hunt, primes)
        _, _, actual, complete = gpu.pairs_batch(
            hunt, config, hunt.initial_cursor(config), 100, primes, self.filter, cache)
        expected = hunt.pairs_batch(config, hunt.initial_cursor(config), 100,
                                    primes)[2]
        self.assertTrue(complete)
        self.assertEqual(actual, expected)
        self.assertEqual(self.filter.telemetry["filter_passers"], 1)

    def test_sigterm_gpu_checkpoint_resumes_on_cpu(self):
        _, p, _, _ = min(BRIAN, key=lambda item: item[1])
        common = ("pairs", "--start", "1e20", "--end", "1e21",
                  "--p-start", str(p), "--p-end", str(p + 100000), "--min-gap", "800")
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "signal.sqlite3"
            output = Path(directory) / "signal.out"
            with output.open("w") as stream:
                process = subprocess.Popen(
                    [sys.executable, str(ROOT / "scripts/psp_hunt.py"),
                     *common, "--backend", "cuda", "--batch-size", "10000",
                     "--db", str(db)], cwd=ROOT, stdout=stream, stderr=stream)
                try:
                    deadline = time.monotonic() + 10
                    checkpointed = False
                    while time.monotonic() < deadline and process.poll() is None:
                        if db.exists():
                            with contextlib.closing(sqlite3.connect(db)) as connection:
                                table = connection.execute(
                                    "select name from sqlite_master where name='progress'").fetchone()
                                if table:
                                    row = connection.execute("select stats from progress").fetchone()
                                    if row and json.loads(row[0])["batches"] > 0:
                                        checkpointed = True
                                        break
                        time.sleep(0.01)
                    self.assertTrue(checkpointed, output.read_text())
                    process.send_signal(signal.SIGTERM)
                    self.assertEqual(process.wait(timeout=20), 143, output.read_text())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=20)
            resumed = run_cli(*common, "--db", str(db), "--batch-size", "10000",
                              "--max-batches", "1")
            self.assertEqual(resumed.returncode, 4, resumed.stderr)
            with contextlib.closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute("pragma integrity_check").fetchone()[0], "ok")


if __name__ == "__main__":
    unittest.main()
