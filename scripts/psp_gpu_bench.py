#!/usr/bin/env python3
"""Compare CUDA and GMP progression filters on BrianK-derived candidates."""

import argparse
import json
import math
import statistics
import sys
import time

import gmpy2

import psp_gpu
import psp_hunt as hunt
from psp_probe import BRIK_800


def benchmark(args):
    fixtures = [tuple(map(int, line.split())) for line in BRIK_800.strip().splitlines()]
    selected = [min(fixtures, key=lambda item: item[1]), fixtures[0],
                max(fixtures, key=lambda item: item[1])]
    results = []
    primes = tuple(hunt.prime_table(math.isqrt(max(item[1] for item in selected))))
    with psp_gpu.Filter(args.device, args.count, args.gpu_library) as gpu:
        for n, p, q, gap in selected:
            order = hunt.order_base2(p, primes)
            step = order if order % 2 == 0 else 2 * order
            if q + (args.count - 1) * step >= psp_gpu.FACTOR_LIMIT:
                raise ValueError("benchmark progression exceeds CUDA factor limit")
            jobs = [psp_gpu.Range(p, q, step, args.count, 0)]
            gpu.filter(jobs)
            cpu_times, gpu_times, kernel_times = [], [], []
            for _ in range(args.repeats):
                started = time.perf_counter()
                expected = [i for i in range(args.count)
                            if gmpy2.powmod(2, p - 1, q + i * step) == 1]
                cpu_times.append(time.perf_counter() - started)
                started = time.perf_counter()
                actual, kernel_ms = gpu.filter(jobs)
                gpu_times.append(time.perf_counter() - started)
                kernel_times.append(kernel_ms / 1000)
                if actual != expected or 0 not in actual:
                    raise RuntimeError(f"GPU/GMP mismatch in Brian-derived progression p={p}")
            cpu_seconds = statistics.median(cpu_times)
            gpu_seconds = statistics.median(gpu_times)
            kernel_seconds = statistics.median(kernel_times)
            results.append({
                "brian_n": str(n), "p": p, "first_q": q, "step": step, "known_gap": gap,
                "q_tested": args.count, "filter_passers": len(actual),
                "cpu_seconds": cpu_seconds, "gpu_filter_seconds": gpu_seconds,
                "kernel_seconds": kernel_seconds,
                "cpu_q_per_second": args.count / cpu_seconds,
                "gpu_q_per_second": args.count / gpu_seconds,
                "filter_speedup": cpu_seconds / gpu_seconds})

        n, p, q, gap = min(fixtures, key=lambda item: item[1])
        config = {"mode": "pairs", "start": 10**20, "end": 10**21,
                  "factor_floor": 10**7, "p_start": p, "p_end": p + 10000}
        order_cache = psp_gpu.PrimeOrderCache(hunt, primes)
        timings = []
        for backend in ("cpu", "cuda"):
            cursor, complete, records, q_tested = hunt.initial_cursor(config), False, [], 0
            started = time.perf_counter()
            while not complete:
                if backend == "cpu":
                    cursor, delta, batch, complete = hunt.pairs_batch(
                        config, cursor, args.count, primes)
                else:
                    cursor, delta, batch, complete = psp_gpu.pairs_batch(
                        hunt, config, cursor, args.count, primes, gpu, order_cache)
                records.extend(batch)
                q_tested += delta["q_tested"]
            timings.append((time.perf_counter() - started, records, q_tested))
        if timings[0][1:] != timings[1][1:]:
            raise RuntimeError("CPU/GPU full hunt mismatch")
    return {
        "device": args.device, "repeats": args.repeats,
        "scope": "Brian-derived progressions; filter and bounded full hunt, not whole-range forecast",
        "filter": results,
        "full_hunt": {"start": str(config["start"]), "end": str(config["end"]),
                      "p_start": config["p_start"], "p_end": config["p_end"],
                      "q_tested": timings[0][2], "confirmed": len(timings[0][1]),
                      "cpu_seconds": timings[0][0], "gpu_seconds": timings[1][0],
                      "speedup": timings[0][0] / timings[1][0]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=hunt.positive_int, default=1 << 20,
                        help="candidates per progression/batch (default 1048576)")
    parser.add_argument("--repeats", type=hunt.positive_int, default=3,
                        help="timing repeats (default 3)")
    parser.add_argument("--device", type=hunt.exact_int, default=0)
    parser.add_argument("--gpu-library", default=str(psp_gpu.DEFAULT_LIBRARY))
    args = parser.parse_args()
    try:
        print(json.dumps(benchmark(args), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"psp_gpu_bench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
