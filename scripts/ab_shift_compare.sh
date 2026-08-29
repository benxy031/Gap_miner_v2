#!/usr/bin/env bash
# =============================================================================
# ab_shift_compare.sh — A/B test accepted blocks / qualifying gaps per hour
# between two CRT shifts (default: shift 258 vs shift 998).
#
# For each iteration the two shifts are run for DURATION seconds each
# (order alternates per iteration to cancel difficulty drift).  The final
# ROLLING STATS block of each run is parsed and one row is appended to a
# TSV.  A summary table (accepted/hour, candidates/hour, windows/candidate)
# is printed at the end.
#
# By default runs are DRY-RUN (no real submissions).  Set COINBASE_HEX to
# your payout scriptPubKey hex to enable real --enable-submission runs
# (blocks found during the test are then really submitted).
#
# Usage:
#   ./scripts/ab_shift_compare.sh
#   DURATION=3600 ITERATIONS=2 ./scripts/ab_shift_compare.sh
#   COINBASE_HEX= ./scripts/ab_shift_compare.sh   # force dry-run
# =============================================================================
set -u

# ---- Configuration (env-overridable) -------------------------------------
NODE_HOST=${NODE_HOST:-127.0.0.1}
NODE_PORT=${NODE_PORT:-31397}
RPC_USER=${RPC_USER:-benxy031}
RPC_PASS=${RPC_PASS:-xx}
THREADS=${THREADS:-8}
DURATION=${DURATION:-3600}          # seconds per single-shift run
ITERATIONS=${ITERATIONS:-2}         # number of A/B pairs
CRT_258=${CRT_258:-data/crt/m23/shift258_p43_strong_m40.txt}
CRT_998=${CRT_998:-data/crt/m23/shift998_p128_strong_m40.txt}
MINER_ENV=${MINER_ENV:-"FUSED_GPU=1 GPU_SIEVE=1 HALF_CLASS=1"}
COINBASE_HEX=${COINBASE_HEX:-51202fb6ebdfec454316c2fdea3cfaa022af86586c64fb096b03bb2f094abefb90ca}  # payout script; empty = dry-run
BIN=${BIN:-./bin/gapminer}
OUT=${OUT:-/tmp/ab_shift_results.tsv}
LOGDIR=${LOGDIR:-/tmp/ab_shift_logs}

mkdir -p "$LOGDIR"

if [ "$DURATION" -lt 90 ]; then
    echo "WARN: DURATION=$DURATION s is very short (first stats block at 30 s)." >&2
fi

# ---- Node info --------------------------------------------------------------
node_info() {
    curl -s --max-time 10 --user "$RPC_USER:$RPC_PASS" \
        --data-binary '{"jsonrpc":"1.0","id":"ab","method":"getmininginfo","params":[]}' \
        -H 'content-type: application/json' "http://$NODE_HOST:$NODE_PORT/" \
    | python3 -c 'import json,sys
try:
    r=json.load(sys.stdin)["result"]
    print(r.get("difficulty","?"), r.get("blocks","?"))
except Exception:
    print("? ?")' 2>/dev/null
}

# ---- Parse final stats block of a run log -----------------------------------
parse_log() {  # $1 = log file
    local log=$1
    local thr cand att acc rej sta bpsw passd sub

    thr=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    cand=$(grep -oE "Merit candidates: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    bpsw=$(grep -oE "BPSW attempts: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    passd=$(grep -oE "Passed: [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    att=$(grep -oE "Submit: attempts=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    acc=$(grep -oE "accepted=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    rej=$(grep -oE "rejected=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    sta=$(grep -oE "stale=[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+")
    sub=$(grep -oE "Submitted: [0-9]+ \(dry-run\)" "$log" | tail -1 | grep -oE "[0-9]+")

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${thr:-0}" "${cand:-0}" "${bpsw:-0}" "${passd:-0}" \
        "${att:-0}" "${acc:-0}" "${rej:-0}" "${sta:-0}" "${sub:-0}"
}

# ---- One run -----------------------------------------------------------------
run_shift() {  # $1 = shift label, $2 = crt file, $3 = iteration, $4 = order
    local label=$1 crt=$2 it=$3 order=$4
    local mode="dry-run"
    local submit_args=()
    if [ -n "$COINBASE_HEX" ]; then
        mode="submit"
        submit_args=(--enable-submission --coinbase-script-hex "$COINBASE_HEX")
    fi

    local ts diff_height
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    diff_height=$(node_info)
    local difficulty=${diff_height% *}
    local height=${diff_height#* }

    local log="$LOGDIR/ab_${label}_it${it}_${order}.log"
    echo "[AB] [$ts] run shift=$label iter=$it ($mode, ${DURATION}s, threads=$THREADS) difficulty=$difficulty height=$height" | tee -a "$LOGDIR/ab_run.log"

    # shellcheck disable=SC2086
    env $MINER_ENV timeout --signal=TERM "$DURATION" "$BIN" \
        --host "$NODE_HOST" --port "$NODE_PORT" --user "$RPC_USER" --pass "$RPC_PASS" \
        --threads "$THREADS" --enable-gpu-fermat --crt-file "$crt" \
        "${submit_args[@]}" >"$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ] && [ $rc -ne 124 ] && [ $rc -ne 143 ]; then
        echo "[AB] WARN: run shift=$label iter=$it exited rc=$rc (log: $log)" | tee -a "$LOGDIR/ab_run.log"
    fi

    local vals
    vals=$(parse_log "$log")
    # shellcheck disable=SC2086
    set -- $vals
    local thr=$1 cand=$2 bpsw=$3 passd=$4 att=$5 acc=$6 rej=$7 sta=$8 sub=$9

    if [ ! -f "$OUT" ]; then
        printf 'timestamp\tshift\tduration_s\tdifficulty\theight\tmode\twindows_s\tcandidates\tbpsw_attempts\tbpsw_passed\tsubmit_attempts\taccepted\trejected\tstale\tlog\n' >"$OUT"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ts" "$label" "$DURATION" "$difficulty" "$height" "$mode" \
        "$thr" "$cand" "$bpsw" "$passd" "$att" "$acc" "$rej" "$sta" "$log" >>"$OUT"

    echo "[AB] [$ts] shift=$label iter=$it done: win/s=$thr candidates=$cand accepted=$acc rejected=$rej stale=$sta" | tee -a "$LOGDIR/ab_run.log"
}

# ---- Main loop --------------------------------------------------------------
echo "[AB] A/B shift comparison start: $(date '+%F %T')  duration=${DURATION}s iterations=$ITERATIONS" | tee -a "$LOGDIR/ab_run.log"
echo "[AB] A=$CRT_258  B=$CRT_998  out=$OUT" | tee -a "$LOGDIR/ab_run.log"

for it in $(seq 1 "$ITERATIONS"); do
    if [ $((it % 2)) -eq 1 ]; then
        run_shift 258 "$CRT_258" "$it" a
        run_shift 998 "$CRT_998" "$it" b
    else
        run_shift 998 "$CRT_998" "$it" b
        run_shift 258 "$CRT_258" "$it" a
    fi
done

# ---- Summary -----------------------------------------------------------------
echo
echo "=============================== SUMMARY ==============================="
echo "Results file: $OUT"
awk -F'\t' '
NR == 1 { next }
{
    sh=$2; dur=$3; cand=$8; acc=$12;
    sum_cand[sh]+=cand;
    sum_acc[sh]+=acc;
    sum_hours[sh]+=dur/3600.0;
    n[sh]++;
    low[sh]+= (cand < 5 ? 1 : 0);
    if (cand > maxc[sh]) maxc[sh]=cand;
}
END {
    printf "%-8s %-12s %-14s %-14s %-16s %s\n", "shift", "runs", "accepted/h", "candidates/h", "candidates/run", "max/run";
    for (sh in sum_cand) {
        printf "%-8s %-12d %-14.3f %-14.3f %-16.1f %d\n",
            sh, n[sh],
            (sum_hours[sh]>0 ? sum_acc[sh]/sum_hours[sh] : 0),
            (sum_hours[sh]>0 ? sum_cand[sh]/sum_hours[sh] : 0),
            (n[sh]>0 ? sum_cand[sh]/n[sh] : 0),
            maxc[sh];
    }
}' "$OUT"
echo
for sh in 258 998; do
    lowrun=$(awk -F'\t' -v s="$sh" 'NR>1 && $2==s && $8<5 {n++} END{print n+0}' "$OUT")
    if [ "$lowrun" -gt 0 ]; then
        echo "NOTE: shift $sh has $lowrun run(s) with <5 candidates — Poisson noise is high;"
        echo "      run more/longer intervals before trusting the comparison."
    fi
done
if [ -z "$COINBASE_HEX" ]; then
    echo "NOTE: dry-run mode — 'accepted' is 0. Compare 'candidates/h' instead, or set COINBASE_HEX."
fi
echo "======================================================================"
