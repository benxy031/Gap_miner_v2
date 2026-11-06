#!/bin/sh
set -eu

bin=$1
mock=$2

run_expected() {
    expected=$1
    shift
    if output=$("$@" 2>&1); then
        actual=0
    else
        actual=$?
    fi
    if [ "$actual" -ne "$expected" ]; then
        printf 'Expected exit %s, got %s: %s\n%s\n' "$expected" "$actual" "$*" "$output" >&2
        exit 1
    fi
}

contains() {
    if ! printf '%s\n' "$output" | grep -F -q -- "$1"; then
        printf 'Missing %s in:\n%s\n' "$1" "$output" >&2
        exit 1
    fi
}

omits() {
    if printf '%s\n' "$output" | grep -F -q -- "$1"; then
        printf 'Unexpected %s in:\n%s\n' "$1" "$output" >&2
        exit 1
    fi
}

run_expected 0 env CUDA_VISIBLE_DEVICES= "$bin" verify --cpu --expr 101 --gap 2
contains 'verdict: VALID'
omits 'CUDA device'

run_expected 2 env CUDA_VISIBLE_DEVICES= "$bin" verify --cpu --expr 101 --gap 6
contains 'PRIME BETWEEN: N + 2 = 103'
run_expected 2 env CUDA_VISIBLE_DEVICES= "$bin" verify --cpu --expr 2 --gap 3
contains 'PRIME BETWEEN: N + 1 = 3'
run_expected 1 env CUDA_VISIBLE_DEVICES= "$bin" verify --cpu --expr 101 --gap 4
contains 'verdict: INVALID (endpoint composite)'
run_expected 3 env CUDA_VISIBLE_DEVICES= "$bin" verify --cpu --expr invalid --gap 2
contains 'verdict: PARSE_FAIL'
run_expected 5 sh -c 'CUDA_VISIBLE_DEVICES= "$1" verify --cpu --expr 101 --gap 2 >/dev/full' sh "$bin"
contains 'failed writing verification output'

run_expected 0 env CUDA_VISIBLE_DEVICES= "$bin" sweep --cpu --with-interval --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (2, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
INSERT INTO gaps VALUES (6, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
INSERT INTO gaps VALUES (4, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
EOF
contains 'selected=3 ok=1 composite_end=1 prime_between=1'
contains 'cpu=3 cpu_skip=0 interval_skipped=0 errors=0'
omits 'CUDA device'

run_expected 4 env CUDA_VISIBLE_DEVICES= "$bin" sweep --cpu --cpu-max-digits 1 --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (2, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
EOF
contains 'ok=0'
contains 'cpu_skip=1'

run_expected 4 env CUDA_VISIBLE_DEVICES= "$bin" sweep --cpu --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (2, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, 'invalid');
EOF
contains 'PARSE_FAIL'
contains 'parse_fail=1'

run_expected 5 env CUDA_VISIBLE_DEVICES= "$bin" sweep --cpu --db /dev/stdin --out /dev/full <<'EOF'
INSERT INTO gaps VALUES (4, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
EOF
contains 'failed writing sweep log'

run_expected 5 env CUDA_VISIBLE_DEVICES= "$bin" verify --expr 101 --gap 2
contains 'verdict: ERROR'
omits 'verdict: VALID'
run_expected 5 env CUDA_VISIBLE_DEVICES= "$bin" sweep --with-interval --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (2, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 3, '101');
EOF
contains 'VERIFY_ERROR'
contains 'ok=0'
contains 'errors=1'
run_expected 5 env CUDA_VISIBLE_DEVICES= "$bin" selftest
contains 'SELFTEST: 1 FAILURES'
omits 'SELFTEST: all passed'
run_expected 5 env CUDA_VISIBLE_DEVICES= "$bin" bench --bits 64 --count 1
contains 'bench failed'

for stage in endpoints sieve interval; do
    run_expected 5 env GV_TEST_FAIL="$stage" "$mock" verify --expr 1009 --gap 4
    contains 'verdict: ERROR'
    omits 'verdict: VALID'
    run_expected 5 env GV_TEST_FAIL="$stage" "$mock" sweep --with-interval --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (4, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 4, '1009');
EOF
    contains 'VERIFY_ERROR'
    contains 'ok=0'
    contains 'errors=1'
done

run_expected 0 env GV_TEST_FAIL=sieve "$mock" verify --cpu --expr 101 --gap 2
contains 'verdict: VALID'
run_expected 4 env GV_TEST_FAIL=large "$mock" verify --expr '2^32768 + 1' --gap 2
contains 'verdict: UNVERIFIED'
omits 'verdict: VALID'
run_expected 4 env GV_TEST_FAIL=large "$mock" sweep --with-interval --db /dev/stdin <<'EOF'
INSERT INTO gaps VALUES (2, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 9865, '2^32768 + 1');
EOF
contains 'INTERVAL_SKIPPED'
contains 'ok=0'
contains 'interval_skipped=1'
run_expected 0 env GV_TEST_FAIL=large "$mock" verify --cpu --expr '2^32768 + 1' --gap 2
contains 'verdict: VALID'

run_expected 5 "$mock" sweep --db /dev/null/not-a-file
run_expected 5 "$mock" sweep --db /dev/stdin --out /dev/null/not-a-file <<'EOF'
EOF

while read -r lower gap; do
    run_expected 0 "$bin" verify --expr "$lower" --gap "$gap" --pmax 150000
    contains 'verdict: VALID'
done <<'EOF'
101109460727019857759 602
858219038909369891653 616
2364158537830198303939 882
3056786368761643367417 714
8324202273673221249967 616
EOF
run_expected 1 "$bin" verify --expr 858219038909369891713 --gap 556 --pmax 150000
contains 'start is COMPOSITE'
run_expected 1 "$bin" verify --expr 3056786368761643367851 --gap 280 --pmax 150000
contains 'start is COMPOSITE'
run_expected 2 "$bin" verify --expr 101109460727019857759 --gap 612 --pmax 150000
contains 'PRIME BETWEEN: N + 602 = 101109460727019858361'
run_expected 2 "$bin" verify --expr 101 --gap 6 --pmax 0
contains 'PRIME BETWEEN: N + 2 = 103'
run_expected 2 "$bin" verify --expr 2 --gap 3
contains 'PRIME BETWEEN: N + 1 = 3'
run_expected 2 "$bin" verify --expr 5 --gap 6
contains 'PRIME BETWEEN: N + 2 = 7'
run_expected 0 "$bin" sweep --with-interval --db /dev/stdin --pmax 150000 <<'EOF'
INSERT INTO gaps VALUES (616, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 21, '858219038909369891653');
INSERT INTO gaps VALUES (612, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 21, '101109460727019857759');
INSERT INTO gaps VALUES (60, 0, 'C', 'F?', 'C?', 'test', 2026, 0, 21, '858219038909369891653');
EOF
contains 'selected=3 ok=1 composite_end=1 prime_between=1'
contains 'interval_skipped=0 errors=0'

printf 'CLI regression tests passed\n'
