#!/usr/bin/env bash
# hunt_gpu_bench.sh - head-to-head of the primorial-hunt class on ONE GPU:
#   arm A: Seth Troisi's prime-gap-gpu  (/tmp/gpugpu/gap_search_primorial)
#   arm B: our Gapcoin miner walker     (bin/gapminer --gap-hunt)
#
# Both arms search the SAME task class - gaps anchored at primorial-like
# structures, reported above a merit threshold - on the same card, for the
# same wall time.  The anchors are not identical (his: p#/d * m divisors,
# ours: CRT-cover survivors of a primorial interval), so the metric is the
# hunt-relevant one: gaps >= merit M found per hour, plus each tool's own
# throughput line.
#
# Usage: scripts/hunt_gpu_bench.sh [SECS_PER_ARM] [MIN_MERIT]
# Needs the GPU to itself; appends to data/hunt_gpu_bench_results.txt.
set -u
cd "$(dirname "$0")/.."

SECS=${1:-720}
MERIT=${2:-10}
SETHP=${SETHP:-/tmp/gpugpu/gap_search_primorial}
MINC_SETH=${MINC_SETH:-20000000}   # 200M: one work unit exceeds the sieve phase
CRT=${CRT:-data/crt/m23/shift450_p67_covermax_m30.txt}
DEV=${DEV:-0}
TAG=$(date -u +%Y%m%dT%H%M%SZ)
OUT=data/hunt_gpu_bench_results.txt

log() { echo "[hgb $TAG] $*"; }

log "arm A (SethTro prime-gap-gpu) ${SECS}s, merit>=${MERIT}, p=491 d=210"
timeout "$SECS" "$SETHP" -p 491 -d 210 --mstart 10000000 --minc "$MINC_SETH" \
    --min-merit "$MERIT" --max-prime 1000 --cpu-fraction 0.016 --cpu-threads 3 \
    > /tmp/hgb_seth.log 2>&1
RC_A=$?
# Seth's tool prints finds as "<gap> <merit> <m> * 491# / 210 - <off>"
GA_A=$(grep -cE "^[0-9]+ [0-9]+\.[0-9]+ " /tmp/hgb_seth.log 2>/dev/null || true)
THR_A=$(grep -oE "[0-9.]+ [KM]?/s" /tmp/hgb_seth.log | tail -2 | tr '\n' ' ')

log "arm B (gapminer --gap-hunt ${CRT}) ${SECS}s, merit>=${MERIT}"
timeout "$SECS" bin/gapminer --gap-hunt --crt-file "$CRT" \
    --gap-hunt-min-merit "$MERIT" --gap-hunt-device "$DEV" \
    --gap-hunt-out /tmp/hgb_ours.out \
    > /tmp/hgb_ours.log 2>&1
RC_B=$?
GA_B=$(sort -u /tmp/hgb_ours.out 2>/dev/null | wc -l)
WINS=$(grep -oE "[0-9]+ win/s" /tmp/hgb_ours.log | tail -1)

{
    echo "# hunt_gpu_bench $TAG  secs_per_arm=$SECS min_merit=$MERIT device=$DEV"
    echo "# arm A: SethTro prime-gap-gpu (build: sm_86 BITS=1024, git $(cd /tmp/gpugpu && git rev-parse --short HEAD))"
    echo "# arm B: bin/gapminer --gap-hunt crt=$CRT (git $(git rev-parse --short HEAD))"
    echo "armA rc=$RC_A gaps_merit_ge=${GA_A} lines=merit  throughput=\"${THR_A}\""
    echo "armB rc=$RC_B gaps_merit_ge=${GA_B} wins=\"${WINS}\""
    echo
} >> "$OUT"
log "written to $OUT"
log "armA rc=$RC_A found=$GA_A throughput=$THR_A"
log "armB rc=$RC_B found=$GA_B wins=$WINS"
exit 0
