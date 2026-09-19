#!/usr/bin/env bash
# HISTORICAL (2026-09-19): MINING_JUMP2_POOL round-admission experiment.
#
# VERDICT: REJECTED, code removed.  The knob no longer exists, so running this
# script now just measures the default repeatedly.  Kept as the record of the
# measurement, because the numbers are the evidence for the CLOSED_FINGERPRINTS
# entry.
#
# Hypothesis was: the chain runs all K flight windows as one wave, so windows
# that finish early leave the late rounds nearly empty (average MR batch 2033
# of a possible 6144 candidates at chunk 12), and since the CGBN kernel is
# latency-bound a smaller batch means fewer tests/s -- so admitting at most N
# windows and topping up as they finish should give fewer rounds AND a fuller
# batch.
#
# Measured at chunk 12, 2 workers, 90 s arms (win/s / rounds per 1000 windows /
# tests per second):
#     pool 128   9216 / 73.31 / 735k
#     pool 256  10734 / 49.89 / 857k
#     pool 384  11547 / 42.94 / 920k
#     none      11751 / 39.10 / 929k     <-- the wave wins, monotonically
# The hypothesis was wrong on BOTH counts: a pool ADDS rounds and LOSES
# tests/s.  Mechanism: the wave completes a flight in max(slices-per-window)
# rounds, a pool in sum(slices)/pool >= max, and what the fuller batch buys
# back is worth less than the extra rounds.  pool=0 must behave exactly like
# the env being unset, so treat size 0 as the control arm.
#
# Usage: scripts/ab_pool.sh [secs]   (only the parity tail is meaningful now)
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
SECS="${1:-90}"
BIN=./bin/gapminer
OUT=/tmp/ab_pool
mkdir -p "$OUT"

printf "%-26s %-9s %-9s %-9s %-8s %s\n" "arm" "win/s" "MR/win" "rounds/1k" "tests/s" "gather%|mr%"
for spec in "0:12" "128:12" "256:12" "384:12" "256:16" "384:8" "448:12"; do
    P="${spec%%:*}"; C="${spec##*:}"
    log="$OUT/p${P}_c${C}.log"
    env FUSED_GPU=1 MINING_JUMP2_CHUNK="$C" MINING_JUMP2_POOL="$P" \
        FUSED_STAGE_TIMING=1 \
        timeout --signal=TERM --kill-after=8 "$SECS" \
        "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
        --threads 2 --enable-gpu-fermat --crt-file "$CRT" >"$log" 2>&1
    w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    m=$(grep -oE "GPU MR tests: [0-9]+ \([0-9.]+/window" "$log" | tail -1 |
        grep -oE "\([0-9.]+" | tr -d '(')
    split=$(grep -oE "Fused stage split: .*" "$log" | tail -1 |
            sed 's/Fused stage split: //')
    rounds=$(printf '%s' "$split" | grep -oE "rounds=[0-9]+" | grep -oE "[0-9]+")
    gm=$(printf '%s' "$split" | grep -oE "gather=[0-9.]+%, mr=[0-9.]+%" |
         sed 's/gather=//;s/%, mr=/|/')
    wins=$(grep -oE "Processed: [0-9]+ windows" "$log" | tail -1 | grep -oE "[0-9]+")
    printf "%-26s %-9s %-9s %-9s %-8s %s\n" "pool=$P chunk=$C" "${w:--}" "${m:--}" \
        "$(python3 -c "
r=$rounds if $rounds else 0
w=$wins if $wins else 0
print(f'{1000*r/w:.2f}' if r and w else '--')" 2>/dev/null || echo '--')" \
        "$(python3 -c "
w=$w if $w else 0
m=$m if $m else 0
print(f'{w*m/1000:.0f}k')" 2>/dev/null || echo '--')" \
        "${gm:--}"
done

echo
echo "=== pool regression + parity check (defaults path, pool off) ==="
env FUSED_GPU=1 MINING_JUMP2_VERIFY=1 timeout --signal=TERM --kill-after=8 240 \
    "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
    --threads 2 --enable-gpu-fermat --merit 16 --crt-file "$CRT" \
    >"$OUT/parity_off.log" 2>&1
grep -oE "bad_windows=[0-9]+ bad_pairs=[0-9]+" "$OUT/parity_off.log" |
    awk -F'[= ]' '{bw+=$2; bp+=$4} END {
        printf "pool=off PARITY flights=%d bad_windows=%d bad_pairs=%d\n", NR, bw+0, bp+0}'
