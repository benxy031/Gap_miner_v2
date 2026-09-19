#!/usr/bin/env bash
# chunk sweep WITH the FUSED_STAGE_TIMING split, so the per-round cost of the
# chain (gather vs MR) and the number of rounds are visible next to win/s.
# Use this to (a) re-tune chunk after the gather-async change and (b) check
# whether more workers per GPU now scale better (the removed
# cudaDeviceSynchronize() in gpu_fermat_gather_run() was a device-wide barrier
# taken once per chain round by every worker).
#
# Usage: scripts/ab_chunk_stage.sh [chunks] [threads]
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
SECS="${SECS:-90}"
OUT=/tmp/ab_stage
mkdir -p "$OUT"

ARMS="${1:-16 24 32}"
THREADS="${2:-2}"

printf "%-26s %-8s %-11s %-9s %-22s %s\n" \
    "arm" "win/s" "MR/win" "rounds" "gather%/mr%" "stage split (mark/extract/collect/chain)"
for T in $THREADS; do
    for C in $ARMS; do
        log="$OUT/c${C}_t${T}.log"
        env FUSED_GPU=1 MINING_JUMP2_CHUNK="$C" FUSED_STAGE_TIMING=1 \
            timeout --signal=TERM --kill-after=8 "$SECS" \
            ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 \
            --pass xx --threads "$T" --enable-gpu-fermat \
            --crt-file "$CRT" >"$log" 2>&1
        w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
        m=$(grep -oE "GPU MR tests: [0-9]+ \([0-9.]+/window" "$log" | tail -1 | grep -oE "\([0-9.]+" | tr -d '(')
        split=$(grep -oE "Fused stage split: .*" "$log" | tail -1 | sed 's/Fused stage split: //')
        rounds=$(printf '%s' "$split" | grep -oE "rounds=[0-9]+" | grep -oE "[0-9]+")
        gm=$(printf '%s' "$split" | grep -oE "gather=[0-9.]+%, mr=[0-9.]+%" | sed 's/gather=//;s/%, mr=/ \/ /')
        printf "%-26s %-8s %-11s %-9s %-22s %s\n" \
            "chunk=$C threads=$T" "${w:--}" "${m:--}" "${rounds:--}" "${gm:--}" \
            "$(printf '%s' "$split" | sed 's/ (rounds=.*//')"
    done
done
