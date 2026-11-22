"""CUDA filters for the standalone pseudoprime hunter."""

import bisect
import ctypes
import math
import time
from pathlib import Path

import gmpy2


FACTOR_LIMIT = 1 << 63
MAX_BATCH = 1 << 24
MAX_RANGES = 1 << 20
SCAN_MIN = 1 << 95
SCAN_LIMIT = (1 << 96) - (1 << 79)
# p-interval per prime_orders sieve segment.
SEGMENT = 1 << 21
DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "bin/libpsp_pairs_gpu.so"


class Range(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64)
                for name in ("p", "first", "step", "count", "offset")]


class PrimeOrderCache:
    """Keep one derived sieve segment, indexed by the checkpoint cursor."""

    def __init__(self, hunt, primes):
        if not isinstance(primes, tuple):
            raise TypeError("prime order cache requires an immutable prime table")
        self.hunt = hunt
        self.primes = primes
        self.low = self.high = self.limit = 0
        self.entries = []

    def get(self, low, limit):
        if self.limit == limit and self.low <= low < self.high:
            return self.entries, bisect.bisect_left(self.entries, (low,)), self.high
        high = min(limit, low + SEGMENT)
        entries = self.hunt.prime_orders(low, high, self.primes)
        self.low, self.high, self.limit, self.entries = low, high, limit, entries
        return entries, 0, high


class Filter:
    def __init__(self, device=0, capacity=1 << 20, library=DEFAULT_LIBRARY):
        if not 1 <= capacity <= MAX_BATCH:
            raise ValueError(f"CUDA batch size must be in [1,{MAX_BATCH}]")
        if not 0 <= device < 1 << 31:
            raise ValueError("CUDA device must fit a nonnegative signed 32-bit index")
        self.capacity = capacity
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        required = ("psp_gpu_error", "psp_gpu_create", "psp_gpu_filter", "psp_gpu_destroy")
        if not all(hasattr(self.lib, symbol) for symbol in required):
            raise ValueError("shared library does not expose the PSP CUDA API")
        self.lib.psp_gpu_error.argtypes = []
        self.lib.psp_gpu_error.restype = ctypes.c_char_p
        self.lib.psp_gpu_create.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self.lib.psp_gpu_create.restype = ctypes.c_void_p
        self.lib.psp_gpu_filter.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(Range), ctypes.c_uint32,
            ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float)]
        self.lib.psp_gpu_filter.restype = ctypes.c_int
        self.lib.psp_gpu_destroy.argtypes = [ctypes.c_void_p]
        self.lib.psp_gpu_destroy.restype = ctypes.c_int
        self.indices = (ctypes.c_uint64 * capacity)()
        self.ctx = self.lib.psp_gpu_create(device, capacity)
        if not self.ctx:
            raise RuntimeError(self.error())
        self.telemetry = {}

    def error(self):
        message = self.lib.psp_gpu_error()
        return message.decode("utf-8", errors="replace") if message else "CUDA filter failed"

    def filter(self, ranges):
        if not self.ctx:
            raise RuntimeError("CUDA filter is closed")
        total = 0
        for job in ranges:
            if job.offset != total or job.count < 1:
                raise ValueError("CUDA progression offsets must be contiguous")
            total += job.count
        if total > self.capacity:
            raise ValueError("CUDA candidates exceed allocated capacity")
        if len(ranges) > MAX_RANGES:
            raise ValueError("CUDA progression count exceeds the per-launch limit")
        if not total:
            return [], 0.0
        jobs = (Range * len(ranges))(*ranges)
        count = ctypes.c_uint64()
        milliseconds = ctypes.c_float()
        rc = self.lib.psp_gpu_filter(
            self.ctx, jobs, len(ranges), total, self.indices,
            ctypes.byref(count), ctypes.byref(milliseconds))
        if rc:
            raise RuntimeError(self.error())
        if count.value > total:
            raise RuntimeError("CUDA returned an invalid hit count")
        indices = sorted(self.indices[:count.value])
        if any(index >= total for index in indices) or len(set(indices)) != len(indices):
            raise RuntimeError("CUDA returned invalid or duplicate hit indices")
        return indices, milliseconds.value

    def scan(self, first, offsets):
        import numpy as np

        if not self.ctx:
            raise RuntimeError("CUDA filter is closed")
        if not SCAN_MIN <= first < SCAN_LIMIT or not first & 1:
            raise ValueError("CUDA scan first candidate must be odd and inside the u96 domain")
        if (not isinstance(offsets, np.ndarray) or offsets.dtype != np.uint32 or
                offsets.ndim != 1 or not offsets.flags.c_contiguous):
            raise ValueError("CUDA scan offsets must be a contiguous uint32 array")
        if len(offsets) > self.capacity:
            raise ValueError("CUDA scan candidate count exceeds capacity")
        if not hasattr(self.lib, "psp_gpu_scan"):
            raise RuntimeError("CUDA library does not expose psp_gpu_scan; rebuild it")
        self.lib.psp_gpu_scan.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float)]
        self.lib.psp_gpu_scan.restype = ctypes.c_int
        count, milliseconds = ctypes.c_uint64(), ctypes.c_float()
        rc = self.lib.psp_gpu_scan(
            self.ctx, first & ((1 << 64) - 1), first >> 64,
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)), len(offsets),
            self.indices, ctypes.byref(count), ctypes.byref(milliseconds))
        if rc:
            raise RuntimeError(self.error())
        if count.value > len(offsets):
            raise RuntimeError("CUDA returned an invalid hit count")
        hits = sorted((value & 0xffffffff, (value >> 32) & 0xff, value >> 40)
                      for value in self.indices[:count.value])
        indices = [index for index, _, _ in hits]
        if (any(index >= len(offsets) or witness > 12 or euler > 1
                for index, witness, euler in hits) or
                len(set(indices)) != len(indices)):
            raise RuntimeError("CUDA returned invalid or duplicate hit indices")
        return hits, milliseconds.value

    def close(self):
        if self.ctx:
            ctx, self.ctx = self.ctx, None
            if self.lib.psp_gpu_destroy(ctx):
                raise RuntimeError(self.error())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def validate_range(config, batch_size):
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"CUDA batch size must be in [1,{MAX_BATCH}]")
    maximum_p = min(config["p_end"] - 1, math.isqrt(config["end"] - 1))
    if maximum_p >= config["p_start"] and (
            maximum_p >= FACTOR_LIMIT or
            (config["end"] - 1) // config["p_start"] >= FACTOR_LIMIT):
        raise ValueError("CUDA requires p and q strictly below 2^63")


def validate_scan_range(config, batch_size):
    if not 1 <= batch_size <= MAX_BATCH:
        raise ValueError(f"CUDA batch size must be in [1,{MAX_BATCH}]")
    if config["start"] < SCAN_MIN or config["end"] > SCAN_LIMIT:
        raise ValueError("CUDA scan requires the interval in [2^95, 2^96 - 2^79)")


def scan_batch(hunt, config, cursor, batch_size, primes, gpu):
    import numpy as np

    started = time.perf_counter()
    start = cursor["n"]
    end = min(config["end"], start + batch_size)
    first, flags = hunt.sieve_flags(start, end, primes)
    offsets = np.flatnonzero(np.frombuffer(flags, dtype=np.uint8)).astype(np.uint32)
    hits, milliseconds = gpu.scan(first, offsets)
    records = []
    delta = {"numbers_scanned": end - start, "sieve_survivors": len(offsets),
             "fermat_passers": len(hits), "confirmed": 0, "pending": 0}
    for index, witness, euler in hits:
        n = first + 2 * int(offsets[index])
        if gmpy2.powmod(2, n - 1, n) != 1:
            raise RuntimeError(f"CUDA scan returned a false Fermat passer: {n}")
        record = (hunt.scan_passer(n, config["factor_floor"]) if witness else
                  hunt.pending_hit_values(n, config["factor_floor"], bool(euler), True))
        if record:
            records.append(record)
            delta[record["status"]] += 1
    gpu.telemetry = {"sieve_survivors": len(offsets), "filter_passers": len(hits),
                     "kernel_ms": milliseconds, "batch_seconds": time.perf_counter() - started}
    return {"n": end}, delta, records, end >= config["end"]


def pairs_batch(hunt, config, cursor, batch_size, primes, gpu, order_cache=None):
    if order_cache is not None and (
            order_cache.hunt is not hunt or order_cache.primes is not primes):
        raise ValueError("prime order cache does not match the search context")
    started = time.perf_counter()
    p_floor, q_resume = cursor["p"], cursor["q"]
    p_end = min(config["p_end"], math.isqrt(config["end"] - 1) + 1)
    work = total = 0
    ranges = []
    delta = {"p_completed": 0, "q_tested": 0, "fermat_passers": 0, "confirmed": 0}
    if config.get("min_gap", 0):
        delta["gap_filtered"] = 0
    segment, seg_index, seg_hi = [], 0, p_floor
    while (p_floor < p_end and work < batch_size and
           len(ranges) < MAX_RANGES and not hunt.STOP_SIGNAL):
        if q_resume is None:
            if seg_index == len(segment):
                if seg_hi >= p_end:
                    p_floor = p_end
                    break
                low = max(seg_hi, p_floor)
                if order_cache is None:
                    seg_hi = min(p_end, low + SEGMENT)
                    segment, seg_index = hunt.prime_orders(low, seg_hi, primes), 0
                else:
                    segment, seg_index, seg_hi = order_cache.get(low, p_end)
                continue
            p, order = segment[seg_index]
            seg_index += 1
        else:
            p = p_floor
            if not hunt.proven_prime_mr12(p):
                raise ValueError("checkpoint contains a non-prime p")
            order = hunt.order_base2(p, primes)
        first, high, step = hunt.pair_bounds(config, p, order)
        q = first if q_resume is None else q_resume
        if q_resume is not None and (q < first or q > high or (q - 1) % step):
            raise ValueError("checkpoint contains an invalid q cursor")
        count = min(max(0, (high - q) // step + 1), batch_size - work)
        if count:
            ranges.append(Range(p, q, step, count, total))
            total += count
            work += count
            q += count * step
        if q <= high:
            p_floor, q_resume = p, q
        else:
            p_floor, q_resume = p + 1, None
            delta["p_completed"] += 1
            work += 1

    indices, milliseconds = gpu.filter(ranges)
    offsets = [job.offset for job in ranges]
    records = []
    for index in indices:
        job = ranges[bisect.bisect_right(offsets, index) - 1]
        p = job.p
        q = job.first + (index - job.offset) * job.step
        if gmpy2.powmod(2, p - 1, q) != 1:
            raise RuntimeError(f"CUDA filter returned a false congruence for p={p}, q={q}")
        # q == p (n = p^2) needs no MR12: p is already proven prime.
        if ((q == p or hunt.proven_prime_mr12(q)) and
                gmpy2.powmod(2, p * q - 1, p * q) == 1):
            record = hunt.pair_record(config, p, q)
            if record is None:
                delta["gap_filtered"] += 1
            else:
                records.append(record)
            delta["fermat_passers"] += 1
            delta["confirmed"] += 1
    delta["q_tested"] = total
    gpu.telemetry = {"q_tested": total, "filter_passers": len(indices),
                     "kernel_ms": milliseconds,
                     "batch_seconds": time.perf_counter() - started}
    if config.get("min_gap", 0):
        gpu.telemetry["gap_filtered"] = delta["gap_filtered"]
    return {"p": p_floor, "q": q_resume}, delta, records, p_floor >= p_end
