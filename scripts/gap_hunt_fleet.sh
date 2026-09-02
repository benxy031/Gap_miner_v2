#!/usr/bin/env bash
# gap_hunt_fleet.sh — run several GAP_HUNT walkers in parallel (multi-size,
# multi-GPU).  Each walker gets its own state/out/log files and a watcher.
#
# Config: gap_hunt_fleet.conf with one line per walker:
#   <crt-file> <device-id> <min-merit>
#
# Empty lines and #-comments are ignored.  Walkers save state on SIGTERM,
# so stopping the fleet is lossless (each resumes from its state file).
set -u

BIN=${BIN:-./bin/gapminer}
CONF=${CONF:-gap_hunt_fleet.conf}
OUTDIR=${OUTDIR:-data}
WATCHER=${WATCHER:-scripts/watch_gap_hunt_records.py}

if [ ! -f "$CONF" ]; then
    echo "config $CONF not found; example:" >&2
    echo "  data/crt/m23/shift507_p74_lex_m30.txt 0 18" >&2
    echo "  data/crt/m23/shift998_p128_m23.txt    1 18" >&2
    exit 1
fi

PIDS=()
ID=0
while IFS= read -r line; do
    line="${line%%#*}"
    set -- $line
    [ -z "$1" ] && continue
    crt=$1; dev=$2; merit=$3
    ID=$((ID + 1))
    state="$OUTDIR/gap_hunt_state_f${ID}.txt"
    out="$OUTDIR/gap_hunt_records_f${ID}.txt"
    log="$OUTDIR/gap_hunt_log_f${ID}.txt"
    echo "[fleet] #$ID: $crt device=$dev merit=$merit"
    "$BIN" --gap-hunt --crt-file "$crt" --gap-hunt-device "$dev" \
        --gap-hunt-min-merit "$merit" \
        --gap-hunt-state "$state" --gap-hunt-out "$out" \
        >>"$log" 2>&1 &
    PIDS+=($!)
    "$WATCHER" "$out" "$OUTDIR/gap_hunt_records_found_f${ID}.txt" \
        >>"$log" 2>&1 &
    PIDS+=($!)
done < "$CONF"

stop() {
    echo "[fleet] stopping..."
    kill -TERM "${PIDS[@]}" 2>/dev/null
    wait "${PIDS[@]}" 2>/dev/null
    echo "[fleet] all walkers saved state and exited"
    exit 0
}
trap stop INT TERM

echo "[fleet] running ${#PIDS[@]} processes (Ctrl+C to stop)"
while true; do
    sleep 60
done
