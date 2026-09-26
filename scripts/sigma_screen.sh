#!/usr/bin/env bash
# sigma_screen.sh — screen a CRT cover by its MEASURED merit-tail slope.
#
# Why: the cover's residue PATTERN sets the merit exponent sigma, not just its
# survivor count. Measured controlled A/B at shift 509 (same primes, same
# gap_target, same 600 s arm), 2026-09-25:
#     optimized cover : sigma(a=8) = 1.260 +- 0.005  (n=70,723), 0.0678 gaps/window
#     random residues : sigma(a=8) = 0.978 +- 0.006  (n=29,361), 0.0256 gaps/window
# i.e. the natural Cramer value is ~1.0 and optimization lifts it to ~1.26 --
# a difference worth 2.65x at merit 8 and ~e^(0.23*m) at depth (extrapolated).
# Since sigma is measurable to ~0.5 % in ten minutes, candidate covers can be
# RANKED BY SIGMA instead of by survivor count (what lex/covermax optimise).
#
# Usage:
#   scripts/sigma_screen.sh <cover-file> [duration_s] [--device N] [--merit M]
# Example:
#   scripts/sigma_screen.sh data/crt/m23/shift720_p98_lex_m45.txt 600
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

COVER=""
DUR=600
DEV=""
MERIT=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEV="$2"; shift 2 ;;
        --merit)  MERIT="$2"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) COVER="$1"; DUR="${2:-600}"; [[ $# -gt 1 ]] && shift 2 || shift ;;
    esac
done
[[ -n "$COVER" && -f "$COVER" ]] || { echo "usage: $0 <cover-file> [duration_s] [--device N]" >&2; exit 2; }

WORK=$(mktemp -d /tmp/sigma_screen.XXXXXX)
OUT="$WORK/out.txt"
ARGS=( --gap-hunt --crt-file "$COVER" --gap-hunt-min-merit "$MERIT"
       --gap-hunt-state "$WORK/state.txt" --gap-hunt-out "$OUT" )
[[ -n "$DEV" ]] && ARGS+=( --gap-hunt-device "$DEV" )

echo "=== screening $(basename "$COVER") for ${DUR}s (min-merit $MERIT)"
head -6 "$COVER" | tr '\n' ' '; echo

timeout "$DUR" ./bin/gapminer "${ARGS[@]}" > "$WORK/arm.log" 2>&1 || true

echo "--- walk"
grep -E "^\[GAP_HUNT\] (walk|k=)" "$WORK/arm.log" | tail -2 | cut -c1-150
echo "--- merit tail (the screen)"
python3 scripts/cover_dist_compare.py "$OUT" --label "$(basename "$COVER" .txt)" \
        --floor "$MERIT" 2>/dev/null | sed -n '3p;/merit tail by anchor/,+2p;/gap-length distribution/,+2p'
echo "--- raw: $OUT ($(wc -l < "$OUT" 2>/dev/null || echo 0) gaps)"
