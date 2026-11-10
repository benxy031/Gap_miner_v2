#!/usr/bin/env python3
"""Resumable base-2 Fermat pseudoprime searches; optional CUDA pairs filter."""

import argparse
import contextlib
import fcntl
import json
import math
import signal
import sqlite3
import sys
from pathlib import Path

try:
    import gmpy2
except ImportError:
    raise SystemExit("psp_hunt needs gmpy2 (pip install gmpy2)")

from psp_math import MR12_LIMIT, euler_base2, mr12_witness
from psp_math import proven_prime_mr12, strong_base2


VERSION = 1
MAX_PRIME_TABLE = 25_000_000
STOP_SIGNAL = 0


def exact_int(text):
    """Parse nonnegative decimal/scientific notation without using floats."""
    import re

    match = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:[eE]([+-]?\d+))?", text)
    if not match:
        raise argparse.ArgumentTypeError("expected an exact nonnegative integer")
    whole, fraction, exponent = match.groups()
    fraction = fraction or ""
    scale = int(exponent or "0") - len(fraction)
    digits = int(whole + fraction)
    if scale >= 0:
        return digits * 10 ** scale
    divisor = 10 ** -scale
    if digits % divisor:
        raise argparse.ArgumentTypeError("value is not an integer")
    return digits // divisor


def positive_int(text):
    value = exact_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def prime_table(limit):
    if limit > MAX_PRIME_TABLE:
        raise ValueError(f"prime table limit {limit} exceeds {MAX_PRIME_TABLE}")
    if limit < 2:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            count = (limit - p * p) // p + 1
            sieve[p * p:limit + 1:p] = b"\x00" * count
    return [p for p in range(2, limit + 1) if sieve[p]]


def order_base2(p, primes):
    """Exact multiplicative order of 2 modulo the proven odd prime p."""
    remainder = p - 1
    factors = []
    for divisor in primes:
        if divisor * divisor > remainder:
            break
        if remainder % divisor == 0:
            factors.append(divisor)
            while remainder % divisor == 0:
                remainder //= divisor
    if remainder > 1:
        factors.append(remainder)
    order = p - 1
    for divisor in factors:
        while order % divisor == 0 and gmpy2.powmod(2, order // divisor, p) == 1:
            order //= divisor
    return order


def prime_orders(lo, hi, primes):
    """(p, order of 2 mod p) for every prime p in [lo,hi), lo >= 3.

    One segmented sieve both proves primality (no divisor <= isqrt(hi-1)) and
    factors every p-1, replacing per-p next_prime and trial division; the
    orders are identical to order_base2."""
    import numpy as np
    lo = max(lo, 3)
    if hi <= lo:
        return []
    need = math.isqrt(hi - 1)
    if need >= 2 and (not primes or
                      (primes[-1] < need and int(gmpy2.next_prime(primes[-1])) <= need)):
        raise ValueError(f"prime table does not reach isqrt({hi - 1})")
    length = hi - lo
    composite = np.zeros(length, dtype=bool)
    sieving = [r for r in primes if r <= need]
    for r in sieving:
        start = max(r * r, -(-lo // r) * r) - lo
        composite[start::r] = True
    prime_idx = np.flatnonzero(~composite)
    if not prime_idx.size:
        return []
    is_prime = ~composite
    rem = np.arange(lo - 1, hi - 1, dtype=np.uint64)
    hit_parts, factor_parts = [], []
    for r in sieving:
        first = (1 - lo) % r
        hits = np.arange(first, length, r)
        hits = hits[is_prime[hits]]
        if not hits.size:
            continue
        hit_parts.append(hits)
        factor_parts.append(np.full(hits.size, r, dtype=np.uint64))
        divisor = np.uint64(r)
        while hits.size:
            rem[hits] //= divisor
            hits = hits[rem[hits] % divisor == 0]
    if hit_parts:
        order_of = np.argsort(np.concatenate(hit_parts), kind="stable")
        hit_sorted = np.concatenate(hit_parts)[order_of]
        factor_sorted = np.concatenate(factor_parts)[order_of].tolist()
        bounds = np.searchsorted(hit_sorted, prime_idx).tolist()
    else:
        factor_sorted = []
        bounds = [0] * prime_idx.size
    bounds.append(len(factor_sorted))
    rest = rem[prime_idx].tolist()
    out = []
    for j, idx in enumerate(prime_idx.tolist()):
        p = lo + idx
        order = p - 1
        factors = factor_sorted[bounds[j]:bounds[j + 1]]
        if rest[j] > 1:
            factors.append(rest[j])
        for divisor in factors:
            while order % divisor == 0 and gmpy2.powmod(2, order // divisor, p) == 1:
                order //= divisor
        out.append((p, order))
    return out


def gap_neighbors(n):
    lower = n - 1
    if lower > 2 and lower % 2 == 0:
        lower -= 1
    while lower > 2 and not gmpy2.is_prime(lower):
        lower -= 2
    upper = int(gmpy2.next_prime(n))
    if not 2 <= lower < n < upper:
        raise RuntimeError(f"invalid surrounding gap for {n}")
    if upper < MR12_LIMIT:
        if not proven_prime_mr12(lower) or not proven_prime_mr12(upper):
            raise RuntimeError(f"GMP endpoint failed deterministic MR12 for {n}")
    return lower, upper


def gap_below(n, minimum):
    lower, upper = gap_neighbors(n)
    if upper >= MR12_LIMIT:
        raise ValueError("gap filtering requires bracketing primes below the MR12 bound")
    # Any two proven bracketing primes bound the containing gap from above.
    return upper - lower < minimum


def gap_context(n):
    lower, upper = gap_neighbors(n)
    status = "probable_gmp"
    if upper < MR12_LIMIT:
        first = lower + 1
        if first % 2 == 0:
            first += 1
        for candidate in range(first, upper, 2):
            if proven_prime_mr12(candidate):
                raise RuntimeError(f"prime {candidate} inside claimed gap for {n}")
        status = "proven_mr12"
    return {
        "lower": str(lower),
        "upper": str(upper),
        "size": upper - lower,
        "left": n - lower,
        "right": upper - n,
        "status": status,
    }


def confirmed_hit(n, source, floor, factors=None, witness=None, family_k=None):
    if n <= 2 or n % 2 == 0 or gmpy2.powmod(2, n - 1, n) != 1:
        raise RuntimeError(f"invalid Fermat pseudoprime candidate {n}")
    record = {
        "n": str(n),
        "status": "confirmed",
        "source": source,
        "fermat_base2": True,
        "euler_jacobi": bool(euler_base2(n)),
        "strong_base2": bool(strong_base2(n)),
        "factor_floor": floor,
        "gap": gap_context(n),
    }
    if factors is not None:
        if math.prod(factors) != n or any(
                f <= floor or f >= MR12_LIMIT or not proven_prime_mr12(f)
                for f in factors):
            raise RuntimeError(f"invalid or unproven factorization for {n}")
        record["factors"] = [str(f) for f in factors]
        record["composite_proof"] = {"kind": "prime_factors_mr12"}
    elif witness is not None:
        if mr12_witness(n) != witness:
            raise RuntimeError(f"invalid compositeness witness for {n}")
        record["composite_proof"] = witness
    else:
        raise RuntimeError(f"missing compositeness evidence for {n}")
    if family_k is not None:
        record["k"] = str(family_k)
    return record


def pair_record(config, p, q):
    n = p * q
    minimum = config.get("min_gap", 0)
    if minimum and gap_below(n, minimum):
        return None
    return confirmed_hit(n, "pairs", config["factor_floor"], [p, q])


def pending_hit(n, floor):
    return {
        "n": str(n),
        "status": "pending",
        "source": "scan",
        "fermat_base2": True,
        "euler_jacobi": bool(euler_base2(n)),
        "strong_base2": bool(strong_base2(n)),
        "factor_floor": floor,
        "reason": "MR12 passed outside its proven range; compositeness unresolved",
    }


def survivors(start, end, primes):
    """Odd values in [start,end) with no prime divisor in primes."""
    first = start | 1
    count = max(0, (end - first + 1) // 2)
    flags = bytearray(b"\x01") * count
    for p in primes:
        if p == 2:
            continue
        offset = (-first) % p
        if offset % 2:
            offset += p
        if first + offset == p:
            offset += 2 * p
        index = offset // 2
        if index < count:
            hits = (count - 1 - index) // p + 1
            flags[index::p] = b"\x00" * hits
    return (first + 2 * i for i, flag in enumerate(flags) if flag)


def initial_cursor(config):
    if config["mode"] == "pairs":
        return {"p": config["p_start"], "q": None}
    if config["mode"] == "chernick":
        return {"k": config["k_start"]}
    return {"n": config["start"]}


def empty_stats():
    return {
        "batches": 0,
        "p_completed": 0,
        "q_tested": 0,
        "k_tested": 0,
        "numbers_scanned": 0,
        "sieve_survivors": 0,
        "fermat_passers": 0,
        "confirmed": 0,
        "pending": 0,
    }


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class Store:
    def __init__(self, path, config):
        self.path = Path(path).resolve()
        self.lock = None
        self.db = None
        self.config = config

    def __enter__(self):
        self.lock = open(str(self.path) + ".lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(str(self.path))
            self.db.execute("PRAGMA synchronous=FULL")
            objects = {
                (row[0], row[1]) for row in self.db.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%'")
            }
            if not objects:
                with self.db:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                    self.db.execute(
                        "CREATE TABLE progress (id INTEGER PRIMARY KEY CHECK(id=1), "
                        "cursor TEXT NOT NULL, stats TEXT NOT NULL, complete INTEGER NOT NULL)")
                    self.db.execute(
                        "CREATE TABLE results (n TEXT PRIMARY KEY, status TEXT NOT NULL, "
                        "gap TEXT, payload TEXT NOT NULL)")
                    self.db.execute("INSERT INTO meta VALUES ('config', ?)", (encode(self.config),))
                    self.db.execute("INSERT INTO progress VALUES (1, ?, ?, 0)",
                                    (encode(initial_cursor(self.config)), encode(empty_stats())))
            elif objects != {("table", "meta"), ("table", "progress"), ("table", "results")}:
                raise ValueError("database is not a psp_hunt database")
            row = self.db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
            if row is None or json.loads(row[0]) != self.config:
                raise ValueError("database search configuration differs; use the original settings or a new file")
            row = self.db.execute(
                "SELECT cursor, stats, complete FROM progress WHERE id=1").fetchone()
            if row is None:
                raise ValueError("database has no checkpoint")
            self.cursor, self.stats, complete = json.loads(row[0]), json.loads(row[1]), row[2]
            self.complete = bool(complete)
            return self
        except (OSError, sqlite3.Error, ValueError):
            if self.db is not None:
                self.db.close()
            self.lock.close()
            raise

    def checkpoint(self, cursor, delta, records, complete):
        stats = dict(self.stats)
        stats["batches"] += 1
        for key, value in delta.items():
            stats[key] = stats.get(key, 0) + value
        with self.db:
            for record in records:
                gap = record.get("gap")
                self.db.execute("INSERT INTO results VALUES (?, ?, ?, ?)",
                                (record["n"], record["status"],
                                 str(gap["size"]) if gap else None, encode(record)))
            self.db.execute(
                "UPDATE progress SET cursor=?, stats=?, complete=? WHERE id=1",
                (encode(cursor), encode(stats), int(complete)))
        self.cursor, self.stats, self.complete = cursor, stats, complete

    def __exit__(self, exc_type, exc_value, traceback):
        if self.db is not None:
            self.db.close()
        if self.lock is not None:
            self.lock.close()


def pair_bounds(config, p, order):
    step = order if order % 2 == 0 else 2 * order
    low = max(p, config["factor_floor"] + 1, (config["start"] + p - 1) // p)
    high = (config["end"] - 1) // p
    first = 1 + ((low - 1 + step - 1) // step) * step
    return first, high, step


def pairs_batch(config, cursor, batch_size, primes):
    p_floor = cursor["p"]
    q_resume = cursor["q"]
    p_end = min(config["p_end"], math.isqrt(config["end"] - 1) + 1)
    work = 0
    records = []
    delta = {"p_completed": 0, "q_tested": 0, "fermat_passers": 0, "confirmed": 0}
    if config.get("min_gap", 0):
        delta["gap_filtered"] = 0
    while p_floor < p_end and work < batch_size and not STOP_SIGNAL:
        if q_resume is None:
            p = int(gmpy2.next_prime(p_floor - 1))
            while p < p_end and not proven_prime_mr12(p):
                p = int(gmpy2.next_prime(p))
            if p >= p_end:
                p_floor = p_end
                break
        else:
            p = p_floor
            if not proven_prime_mr12(p):
                raise ValueError("checkpoint contains a non-prime p")
        order = order_base2(p, primes)
        first, high, step = pair_bounds(config, p, order)
        q = first if q_resume is None else q_resume
        if q_resume is not None and (q < first or q > high or (q - 1) % step):
            raise ValueError("checkpoint contains an invalid q cursor")
        while q <= high and work < batch_size and not STOP_SIGNAL:
            work += 1
            delta["q_tested"] += 1
            if gmpy2.powmod(2, p - 1, q) == 1 and proven_prime_mr12(q):
                n = p * q
                if gmpy2.powmod(2, n - 1, n) == 1:
                    record = pair_record(config, p, q)
                    if record is None:
                        delta["gap_filtered"] += 1
                    else:
                        records.append(record)
                    delta["fermat_passers"] += 1
                    delta["confirmed"] += 1
            q += step
        if q <= high:
            p_floor, q_resume = p, q
        else:
            p_floor, q_resume = p + 1, None
            delta["p_completed"] += 1
            work += 1
    return {"p": p_floor, "q": q_resume}, delta, records, p_floor >= p_end


def chernick_batch(config, cursor, batch_size):
    k = cursor["k"]
    records = []
    delta = {"k_tested": 0, "fermat_passers": 0, "confirmed": 0}
    while k < config["k_end"] and delta["k_tested"] < batch_size and not STOP_SIGNAL:
        factors = [6 * k + 1, 12 * k + 1, 18 * k + 1]
        delta["k_tested"] += 1
        if factors[0] > config["factor_floor"] and all(
                gmpy2.is_prime(f, 1) and proven_prime_mr12(f) for f in factors):
            n = math.prod(factors)
            records.append(confirmed_hit(n, "chernick", config["factor_floor"],
                                         factors, family_k=k))
            delta["fermat_passers"] += 1
            delta["confirmed"] += 1
        k += 1
    return {"k": k}, delta, records, k >= config["k_end"]


def scan_batch(config, cursor, batch_size, primes):
    start = cursor["n"]
    end = min(config["end"], start + batch_size)
    records = []
    delta = {"numbers_scanned": end - start, "sieve_survivors": 0,
             "fermat_passers": 0, "confirmed": 0, "pending": 0}
    for n in survivors(start, end, primes):
        delta["sieve_survivors"] += 1
        if gmpy2.powmod(2, n - 1, n) != 1:
            continue
        delta["fermat_passers"] += 1
        witness = mr12_witness(n)
        if witness is not None:
            records.append(confirmed_hit(n, "scan", config["factor_floor"],
                                         witness=witness))
            delta["confirmed"] += 1
        elif n >= MR12_LIMIT:
            records.append(pending_hit(n, config["factor_floor"]))
            delta["pending"] += 1
    return {"n": end}, delta, records, end >= config["end"]


def validate_config(config):
    floor = config["factor_floor"]
    mode = config["mode"]
    if mode in ("pairs", "scan"):
        if config["start"] < 3 or config["end"] <= config["start"]:
            raise ValueError("require 3 <= start < end")
    if mode == "pairs":
        minimum = config.get("min_gap", 0)
        if minimum < 0:
            raise ValueError("min-gap must be nonnegative")
        if minimum and config["end"] > MR12_LIMIT:
            raise ValueError("gap filtering requires a product range below the MR12 bound")
        p_start, p_end = config["p_start"], config["p_end"]
        if p_start < 3 or p_start <= floor or p_end <= p_start:
            raise ValueError("require 3 <= p-start < p-end and p-start > factor-floor")
        max_p = min(p_end - 1, math.isqrt(config["end"] - 1))
        if max_p >= p_start:
            if max_p >= MR12_LIMIT or (config["end"] - 1) // p_start >= MR12_LIMIT:
                raise ValueError("factor range exceeds the proven MR12 bound; narrow the range")
            if math.isqrt(max_p - 1) > MAX_PRIME_TABLE:
                raise ValueError("p range needs a factor table above 25000000; narrow p-end")
    elif mode == "chernick":
        if config["k_start"] < 1 or config["k_end"] <= config["k_start"]:
            raise ValueError("require 1 <= k-start < k-end")
        if 18 * (config["k_end"] - 1) + 1 >= MR12_LIMIT:
            raise ValueError("family factors exceed the proven MR12 bound")
    elif mode == "scan" and floor > MAX_PRIME_TABLE:
        raise ValueError("scan factor-floor exceeds the 25000000 prime-table limit")


def request_stop(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def run_search(args):
    config = {"version": VERSION, "mode": args.mode, "factor_floor": args.factor_floor}
    if args.mode in ("pairs", "scan"):
        config.update(start=args.start, end=args.end)
    if args.mode == "pairs":
        config.update(p_start=args.p_start, p_end=args.p_end)
        if args.min_gap:
            config["min_gap"] = args.min_gap
    if args.mode == "chernick":
        config.update(k_start=args.k_start, k_end=args.k_end)
    validate_config(config)
    gpu_module = None
    if args.mode == "pairs" and args.backend == "cuda":
        import psp_gpu as gpu_module
        gpu_module.validate_range(config, args.batch_size)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with contextlib.ExitStack() as stack:
        store = stack.enter_context(Store(args.db, config))
        gpu = None
        if gpu_module is not None and not store.complete:
            gpu = stack.enter_context(gpu_module.Filter(
                args.device, args.batch_size, args.gpu_library))
        print(f"mode={args.mode} db={args.db} checkpoint={store.cursor}", flush=True)
        primes = []
        if not store.complete and args.mode == "pairs":
            max_p = min(args.p_end - 1, math.isqrt(args.end - 1))
            if max_p >= args.p_start:
                primes = prime_table(math.isqrt(max_p - 1))
        elif not store.complete and args.mode == "scan":
            primes = prime_table(args.factor_floor)
        order_cache = None
        if gpu is not None:
            primes = tuple(primes)
            order_cache = gpu_module.PrimeOrderCache(sys.modules[__name__], primes)
        batches = 0
        while not store.complete and not STOP_SIGNAL:
            if args.mode == "pairs":
                if gpu is not None:
                    result = gpu_module.pairs_batch(
                        sys.modules[__name__], config, store.cursor,
                        args.batch_size, primes, gpu, order_cache)
                else:
                    result = pairs_batch(config, store.cursor, args.batch_size, primes)
            elif args.mode == "chernick":
                result = chernick_batch(config, store.cursor, args.batch_size)
            else:
                result = scan_batch(config, store.cursor, args.batch_size, primes)
            cursor, delta, records, complete = result
            if STOP_SIGNAL and cursor == store.cursor and not records:
                break
            store.checkpoint(cursor, delta, records, complete)
            if gpu is not None:
                print("CUDA " + encode(gpu.telemetry), flush=True)
            batches += 1
            for record in records:
                print("HIT " + encode(record), flush=True)
            filtered_progress = ""
            if config.get("min_gap", 0):
                filtered = store.stats.get("gap_filtered", 0)
                filtered_progress = (
                    f" retained={store.stats['confirmed'] - filtered} gap_filtered={filtered}")
            print(f"checkpoint={store.cursor} confirmed={store.stats['confirmed']} "
                  f"pending={store.stats['pending']}{filtered_progress}", flush=True)
            if args.max_batches and batches >= args.max_batches:
                break
        if STOP_SIGNAL:
            print(f"stopped by signal {STOP_SIGNAL}; restart with the same settings", file=sys.stderr)
            return 128 + STOP_SIGNAL
        if not store.complete:
            print("INCOMPLETE: resume with the same search settings", file=sys.stderr)
            return 4
        if store.stats["pending"]:
            print(f"INCOMPLETE: {store.stats['pending']} Fermat passers require classification",
                  file=sys.stderr)
            return 4
        print(f"COMPLETE stats={encode(store.stats)}", flush=True)
        return 0


def show_results(args):
    path = Path(args.db).resolve()
    uri = path.as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("BEGIN")
        row = db.execute("SELECT value FROM meta WHERE key='config'").fetchone()
        if row is None or json.loads(row[0]).get("version") != VERSION:
            raise ValueError("database is not a supported psp_hunt database")
        config = json.loads(row[0])
        progress = db.execute("SELECT cursor, stats, complete FROM progress WHERE id=1").fetchone()
        if progress is None:
            raise ValueError("database has no checkpoint")
        clauses, params = [], []
        if args.status != "all":
            clauses.append("status = ?")
            params.append(args.status)
        if args.min_gap:
            minimum = str(args.min_gap)
            clauses.append(
                "(gap IS NOT NULL AND (length(gap) > ? OR "
                "(length(gap) = ? AND gap >= ?)))")
            params.extend((len(minimum), len(minimum), minimum))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        matching = db.execute("SELECT COUNT(*) FROM results" + where, params).fetchone()[0]
        limit = min(args.limit, matching) if args.limit else matching
        selected = db.execute(
            "SELECT payload FROM results" + where +
            " ORDER BY length(gap) DESC, gap DESC, length(n) DESC, n DESC LIMIT ?",
            (*params, limit)).fetchall()
        stats = json.loads(progress[1])
    for (payload,) in selected:
        print(payload)
    filtered_progress = ""
    if config.get("min_gap", 0):
        filtered = stats.get("gap_filtered", 0)
        filtered_progress = (
            f" retained={stats['confirmed'] - filtered} gap_filtered={filtered}"
            f" min_gap={config['min_gap']}")
    print(f"shown={len(selected)} matching={matching} complete={bool(progress[2])} "
          f"confirmed={stats['confirmed']} pending={stats['pending']} "
          f"checkpoint={json.loads(progress[0])}{filtered_progress}", file=sys.stderr)
    return 0


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    def search_options(p, default_batch):
        p.add_argument("--db", required=True, help="SQLite results/checkpoint path")
        p.add_argument("--factor-floor", type=exact_int, default=10_000_000,
                       help="all prime factors must be greater than this value (default 10000000)")
        p.add_argument("--batch-size", type=positive_int, default=default_batch,
                       help=f"checkpoint work size (default {default_batch})")
        p.add_argument("--max-batches", type=exact_int, default=0,
                       help="pause after this many checkpoints; 0 means no extra limit")

    pairs = sub.add_parser("pairs", help="complete semiprime search in a smaller-factor slice")
    search_options(pairs, 10_000)
    pairs.add_argument("--start", type=exact_int, default=10**21, help="inclusive product start")
    pairs.add_argument("--end", type=exact_int, default=2 * 10**21, help="exclusive product end")
    pairs.add_argument("--p-start", type=exact_int, required=True, help="inclusive smaller-factor start")
    pairs.add_argument("--p-end", type=exact_int, required=True, help="exclusive smaller-factor end")
    pairs.add_argument("--min-gap", type=exact_int, default=0,
                       help="retain only gaps >= this size; 0 retains all (default 0)")
    pairs.add_argument("--backend", choices=("cpu", "cuda"), default="cpu",
                       help="pairs filter backend (default cpu)")
    pairs.add_argument("--device", type=exact_int, default=0,
                       help="CUDA device index (default 0)")
    pairs.add_argument("--gpu-library", default=str(
        Path(__file__).resolve().parents[1] / "bin/libpsp_pairs_gpu.so"),
        help="CUDA shared library (default bin/libpsp_pairs_gpu.so in repository)")

    family = sub.add_parser("chernick", help="search the three-factor Chernick family")
    search_options(family, 10_000)
    family.add_argument("--k-start", type=exact_int, default=1_666_667, help="inclusive k start")
    family.add_argument("--k-end", type=exact_int, required=True, help="exclusive k end")

    scan = sub.add_parser("scan", help="complete interval scan; slower but includes all factor shapes")
    search_options(scan, 100_000)
    scan.add_argument("--start", type=exact_int, default=10**21, help="inclusive interval start")
    scan.add_argument("--end", type=exact_int, required=True, help="exclusive interval end")

    show = sub.add_parser("show", help="show saved results as JSON lines")
    show.add_argument("--db", required=True, help="SQLite results path")
    show.add_argument("--min-gap", type=exact_int, default=0, help="minimum shown gap, default 0")
    show.add_argument("--limit", type=exact_int, default=20, help="maximum rows, 0 means all")
    show.add_argument("--status", choices=("confirmed", "pending", "all"), default="confirmed")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        return show_results(args) if args.mode == "show" else run_search(args)
    except BlockingIOError:
        print("psp_hunt: another search holds the database lock", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"psp_hunt: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
