#!/usr/bin/env bash
# chunk sweep with MINING_JUMP2_VERIFY parity gate.
#
# MINING_JUMP2_CHUNK sets how many candidates per window are gathered into one
# chain MR round (batch = MINING_JUMP2_BATCH * chunk).  A smaller chunk makes
# more, smaller MR rounds.  Because the chain decides WHICH candidates get
# tested, a chunk change can in principle alter the emitted merit-candidate
# set - that is a correctness question, not a speed question, and it is
# answered by MINING_JUMP2_VERIFY=1 (full-scan replay compared pair-by-pair
# against the chain output, per window, with --merit low enough that many
# pairs are actually emitted).
#
# Usage: scripts/ab_chunk_parity.sh
set -u
cd "$(dirname "$0")/.." || exit 1

CRT="${CRT:-data/crt/m23/shift509_p74_covermax_m38.txt}"
THREADS="${THREADS:-2}"
OUT=/tmp/ab_chunk
mkdir -p "$OUT"

run_arm() { # name chunk merit seconds verify
    local name="$1" chunk="$2" merit="$3" secs="$4" verify="$5"
    local log="$OUT/$name.log"
    local -a envs=(FUSED_GPU=1 MINING_JUMP2_CHUNK="$chunk")
    [ "$verify" = 1 ] && envs+=(MINING_JUMP2_VERIFY=1)
    local -a mer=(--merit "$merit")
    [ "$merit" = "live" ] && mer=()
    env "${envs[@]}" timeout --signal=TERM --kill-after=8 "$secs" \
        ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 \
        --pass xx --threads "$THREADS" --enable-gpu-fermat \
        --crt-file "$CRT" "${mer[@]}" >"$log" 2>&1
    local w n c bad
    w=$(grep -oE "Processed: [0-9]+ windows" "$log" | tail -1 | grep -oE "[0-9]+")
    n=$(grep -oE "Merit candidates: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    c=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    bad=$(grep -oE "bad_windows=[0-9]+ bad_pairs=[0-9]+" "$log" |
          awk -F'[= ]' '{bw+=$2; bp+=$4} END {print bw+0" "bp+0}')
    local fl
    fl=$(grep -c "MINING_JUMP2 VERIFY" "$log")
    printf "%-16s chunk=%-3s merit=%-4s win/s=%-6s windows=%-9s cand=%-4s rate/100k=%s  VERIFY(flights=%s bad_win/bad_pairs=%s)\n" \
        "$name" "$chunk" "$merit" "${c:--}" "${w:--}" "${n:--}" \
        "$(python3 -c "print(f'{100000*${n:-0}/${w:-1}:.2f}')")" \
        "$fl" "${bad:-none}"
}

echo "=== parity arms (merit 16, 240 s): the emitted set must be identical ==="
run_arm chunk16_verify 16 16 240 1
run_arm chunk32_verify 32 16 240 1
run_arm chunk24_verify 24 16 240 1

echo
echo "=== throughput arms (live merit, 90 s) ==="
run_arm chunk16_live 16 live 90 0
run_arm chunk24_live 24 live 90 0
run_arm chunk32_live 32 live 90 0
