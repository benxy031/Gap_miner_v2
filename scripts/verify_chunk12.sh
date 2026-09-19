#!/usr/bin/env bash
# Verification of the shipped configuration:
#   1. the binary really reports the new chunk default,
#   2. emitted-set parity vs a FULL SCAN at that default (the mandatory gate
#      after any change to the chain's scheduling or stream usage),
#   3. repeated 90 s A/B of chunk 12 (new default) vs 16 vs 32 at 2 workers, so
#      the claim rests on a mean, not one arm (run-to-run spread is ~4-5%).
#
# Usage: scripts/verify_chunk12.sh [arm_secs] [parity_secs]
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
BIN=./bin/gapminer
SECS="${1:-90}"
PARITY_SECS="${2:-240}"
OUT=/tmp/verify_c12
mkdir -p "$OUT"

echo "=== 1. default reported by the binary (stdbuf: unbuffered) ==="
env FUSED_GPU=1 stdbuf -o0 timeout --signal=TERM 8 \
    "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
    --threads 2 --enable-gpu-fermat --crt-file "$CRT" 2>&1 |
    grep -m1 -oE "MINING_JUMP2 no-test chain: K=[0-9]+ windows, chunk=[0-9]+" ||
    echo "  (no chain banner seen)"

echo
echo "=== 2. parity vs FULL SCAN at the default (merit 16, ${PARITY_SECS} s) ==="
env FUSED_GPU=1 MINING_JUMP2_VERIFY=1 FUSED_STAGE_TIMING=1 \
    timeout --signal=TERM --kill-after=8 "$PARITY_SECS" \
    "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
    --threads 2 --enable-gpu-fermat --merit 16 --crt-file "$CRT" \
    >"$OUT/parity.log" 2>&1
grep -oE "bad_windows=[0-9]+ bad_pairs=[0-9]+" "$OUT/parity.log" |
    awk -F'[= ]' '{bw+=$2; bp+=$4} END {
        printf "PARITY flights=%d bad_windows=%d bad_pairs=%d\n", NR, bw+0, bp+0}'
grep -c "VERIFY: full-scan failed" "$OUT/parity.log" |
    sed 's/^/  verifier internal failures: /'

echo
echo "=== 3. repeats at 2 workers (${SECS} s each) ==="
printf "%-10s %-9s %-9s %s\n" "chunk" "rep" "win/s" "MR/win"
for C in 12 16 32; do
    for R in 1 2; do
        log="$OUT/c${C}_r${R}.log"
        env FUSED_GPU=1 MINING_JUMP2_CHUNK="$C" \
            timeout --signal=TERM --kill-after=8 "$SECS" \
            "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
            --threads 2 --enable-gpu-fermat --crt-file "$CRT" >"$log" 2>&1
        w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 |
            grep -oE "[0-9]+")
        m=$(grep -oE "GPU MR tests: [0-9]+ \([0-9.]+/window" "$log" | tail -1 |
            grep -oE "\([0-9.]+" | tr -d '(')
        printf "%-10s %-9s %-9s %s\n" "$C" "$R" "${w:--}" "${m:--}"
    done
done
