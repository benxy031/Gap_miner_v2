#!/usr/bin/env bash
# A/B: pre-change baseline binary vs the rebuilt binary (gather-round fix),
# plus a chunk re-sweep.  Everything at the production geometry
# (shift509_p74_covermax_m38, fused chain, live difficulty).
#
# Usage: scripts/ab_gather_fix.sh [secs]
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
SECS="${1:-90}"
NEW=./bin/gapminer
OLD=/tmp/gapminer_baseline
OUT=/tmp/ab_gather
mkdir -p "$OUT"

[ -x "$OLD" ] || { echo "missing $OLD (baseline binary)"; exit 1; }
[ -x "$NEW" ] || { echo "missing $NEW (build first)"; exit 1; }

printf "%-30s %-9s %-9s %-8s %-11s %s\n" \
    "arm (binary/chunk/threads)" "win/s" "MR/win" "rounds" "rounds/1k win" "gather%/mr%"
printf -- "-----------------------------------------------------------------------------------------\n"

run_arm() { # tag binary chunk threads
    local tag="$1" bin="$2" chunk="$3" threads="$4"
    local log="$OUT/$(printf '%s' "$tag" | tr '/' '_').log"
    env FUSED_GPU=1 MINING_JUMP2_CHUNK="$chunk" FUSED_STAGE_TIMING=1 \
        timeout --signal=TERM --kill-after=8 "$SECS" \
        "$bin" --host 127.0.0.1 --port 31397 --user benxy031 --pass xx \
        --threads "$threads" --enable-gpu-fermat --crt-file "$CRT" \
        >"$log" 2>&1
    local w m split rounds gm rpw
    w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    m=$(grep -oE "GPU MR tests: [0-9]+ \([0-9.]+/window" "$log" | tail -1 |
        grep -oE "\([0-9.]+" | tr -d '(')
    split=$(grep -oE "Fused stage split: .*" "$log" | tail -1 |
            sed 's/Fused stage split: //')
    rounds=$(printf '%s' "$split" | grep -oE "rounds=[0-9]+" | grep -oE "[0-9]+")
    gm=$(printf '%s' "$split" | grep -oE "gather=[0-9.]+%, mr=[0-9.]+%" |
         sed 's/gather=//;s/%, mr=/ | /')
    local wins
    wins=$(grep -oE "Processed: [0-9]+ windows" "$log" | tail -1 |
           grep -oE "[0-9]+")
    rpw=$(python3 -c "
r=$rounds if $rounds > 0 else 0
w=$wins if $wins else 0
print(f'{1000*r/w:.2f}' if r and w else '--')" 2>/dev/null || echo "--")
    printf "%-30s %-9s %-9s %-8s %-11s %s\n" "$tag" "${w:--}" "${m:--}" \
        "${rounds:--}" "$rpw" "${gm:--}"
}

echo "== gather-fix A/B (same chunk on both binaries) =="
run_arm "old/chunk32/t2" "$OLD" 32 2
run_arm "new/chunk32/t2" "$NEW" 32 2
run_arm "old/chunk16/t2" "$OLD" 16 2
run_arm "new/chunk16/t2" "$NEW" 16 2

echo
echo "== chunk re-sweep on the new binary (t2) =="
for C in 8 12 20 24; do run_arm "new/chunk$C/t2" "$NEW" "$C" 2; done

echo
echo "== packing sensitivity on the new binary (chunk 16) =="
run_arm "new/chunk16/t1" "$NEW" 16 1
run_arm "new/chunk16/t4" "$NEW" 16 4
