#!/usr/bin/env bash
# scripts/p0_rec_campaign.sh - contiguous world-record campaign driver.
#
# Runs back-to-back walk-engine segments at a fixed start increment.  K=96 by
# default (measured +2.5 % over K=64 at the high-start wall; 6.5 GB of VRAM,
# so it is only safe when nothing else holds the card - the bitmap alloc fails
# LOUDLY, never silently).
#
# Resume rule: the scanner refuses a (start,length,threads) mismatch, so an
# interrupted segment MUST be resumed with the values stored in its state file -
# this driver reads them instead of assuming its own position.  A COMPLETED
# segment deletes the state file (scanner behaviour), which is how the driver
# knows to advance the start.
#
# Env (all optional):
#   START  first segment start          (default 1e32)
#   SEG    positions per segment        (default 8.5e14 ~= 1 h on a 3070)
#   GAPMIN gap threshold to report      (default 2002 = merit 27.17 at 1e32)
#   BATCH  --walk-batch                 (default 96)
#   WP     --walk-primes                (default 150000)
#   STATE / LOG / OUT  file paths
#
# Stop with: pkill -TERM -f p0_rec_campaign   (the scanner checkpoints on SIGTERM
# and the next start resumes from the state file).
set -u
cd "$(dirname "$0")/.." || exit 1

BIN=./bin/phase0_scan_gpu
START=${START:-100000000000000000000000000000000}
SEG=${SEG:-850000000000000}
GAPMIN=${GAPMIN:-2002}
BATCH=${BATCH:-96}
WP=${WP:-150000}
STATE=${STATE:-data/p0_rec_1e32.state}
LOG=${LOG:-data/p0_rec_1e32.log}
OUT=${OUT:-data/p0_rec_1e32.out}
SEGMENTS=${SEGMENTS:-0}     # 0 = run until stopped

stamp() { date -Is; }

# print the value of "key VALUE" from the state file (python ints: > 2^53)
state_get() {
  python3 - "$STATE" "$1" <<'EOF'
import sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) == 2 and p[0] == key:
                print(p[1])
                break
except FileNotFoundError:
    pass
EOF
}

cur="$START"
n=0
echo "[campaign] $(stamp) driver up: START=$cur SEG=$SEG GAPMIN=$GAPMIN BATCH=$BATCH" >>"$OUT"

while :; do
    run_start="$cur"
    run_len="$SEG"
    if [ -f "$STATE" ]; then
        s=$(state_get start); l=$(state_get length)
        if [ -n "${s:-}" ] && [ -n "${l:-}" ]; then
            run_start="$s"; run_len="$l"
            echo "[campaign] $(stamp) resuming interrupted segment start=$run_start len=$run_len" >>"$OUT"
        fi
    fi

    "$BIN" --engine walk --start "$run_start" --length "$run_len" \
        --gap-min "$GAPMIN" --walk-primes "$WP" --walk-batch "$BATCH" \
        --log "$LOG" --state "$STATE" --state-every 60 --progress 300 >>"$OUT" 2>&1
    rc=$?

    if [ "$rc" -ne 0 ]; then
        echo "[campaign] $(stamp) segment start=$run_start exited rc=$rc -- stopping" >>"$OUT"
        exit "$rc"
    fi

    cur=$(python3 -c "print(int('$run_start') + int('$run_len'))")
    n=$((n + 1))
    echo "[campaign] $(stamp) segment #$n complete; next start=$cur" >>"$OUT"
    if [ "$SEGMENTS" -gt 0 ] && [ "$n" -ge "$SEGMENTS" ]; then
        echo "[campaign] $(stamp) SEGMENTS=$SEGMENTS reached -- stopping" >>"$OUT"
        exit 0
    fi
done
