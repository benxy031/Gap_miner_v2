#!/usr/bin/env bash
# Round 2 of the chunk/packing A/B:
#  (a) refine the t2 peak around chunk 12 (10/12/14 + a repeat),
#  (b) decide the DEFAULT question: the optimum chunk depends on how many
#      workers share the GPU, so measure t1 at 12/20/32 as well,
#  (c) t4 at the new peak.
#
# Usage: scripts/ab_chunk_round2.sh [secs]
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
SECS="${1:-90}"
BIN=./bin/gapminer
OUT=/tmp/ab_r2
mkdir -p "$OUT"

printf "%-24s %-9s %-9s %-9s %s\n" "arm" "win/s" "MR/win" "rounds/1k" "gather%|mr%"
for spec in "12:2" "10:2" "14:2" "12:3" "16:2" "12:1" "20:1" "32:1" "12:4"; do
    C="${spec%%:*}"; T="${spec##*:}"
    log="$OUT/c${C}_t${T}.log"
    env FUSED_GPU=1 MINING_JUMP2_CHUNK="$C" FUSED_STAGE_TIMING=1 \
        timeout --signal=TERM --kill-after=8 "$SECS" \
        "$BIN" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
        --threads "$T" --enable-gpu-fermat --crt-file "$CRT" >"$log" 2>&1
    w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    m=$(grep -oE "GPU MR tests: [0-9]+ \([0-9.]+/window" "$log" | tail -1 |
        grep -oE "\([0-9.]+" | tr -d '(')
    split=$(grep -oE "Fused stage split: .*" "$log" | tail -1 |
            sed 's/Fused stage split: //')
    rounds=$(printf '%s' "$split" | grep -oE "rounds=[0-9]+" | grep -oE "[0-9]+")
    gm=$(printf '%s' "$split" | grep -oE "gather=[0-9.]+%, mr=[0-9.]+%" |
         sed 's/gather=//;s/%, mr=/|/')
    wins=$(grep -oE "Processed: [0-9]+ windows" "$log" | tail -1 |
           grep -oE "[0-9]+")
    rpw=$(python3 -c "
r=$rounds if $rounds else 0
w=$wins if $wins else 0
print(f'{1000*r/w:.2f}' if r and w else '--')" 2>/dev/null || echo "--")
    printf "%-24s %-9s %-9s %-9s %s\n" "chunk=$C t=$T" "${w:--}" "${m:--}" \
        "$rpw" "${gm:--}"
done
