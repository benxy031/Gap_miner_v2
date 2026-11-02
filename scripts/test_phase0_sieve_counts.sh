#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

while read -r limit odd above_wheel; do
    for scanner in phase0_scan phase0_scan_gpu phase0_scan_gpu_split; do
        args=(--sieve-limit "$limit" --start 200000000000000000000
              --length 1000 --threads 1 --no-test)
        if [[ "$scanner" != phase0_scan ]]; then
            args+=(--cpu-sieve --no-records)
            expected="[phase0-gpu] sieve primes up to $limit: $odd odd primes ($above_wheel > 13 in buckets after the wheel)"
        else
            expected="[phase0] sieve primes up to $limit: $odd odd primes"
        fi
        if ! output=$("./bin/$scanner" "${args[@]}" 2>&1); then
            printf '%s failed for sieve limit %s:\n%s\n' "$scanner" "$limit" "$output" >&2
            exit 1
        fi
        if [[ "$output" != *"$expected"* ]]; then
            printf '%s at limit %s: expected "%s"\n%s\n' \
                "$scanner" "$limit" "$expected" "$output" >&2
            exit 1
        fi
    done
done <<'CASES'
0 0 0
1 0 0
2 0 0
3 1 0
4 1 0
5 2 0
6 2 0
7 3 0
9 3 0
10 3 0
11 4 0
12 4 0
13 5 0
14 5 0
16 5 0
17 6 1
18 6 1
24 8 3
25 8 3
26 8 3
29 9 4
30 9 4
31 10 5
97 24 19
98 24 19
100 24 19
10000000 664578 664573
10000001 664578 664573
CASES

printf 'Phase-0 sieve counts: 28 limits pass for all three scanners.\n'
