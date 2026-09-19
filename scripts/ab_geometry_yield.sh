#!/usr/bin/env bash
# Geometry comparison via the in-miner `Yield` line.
#
# expected blocks/h = win/s x P(merit >= m), and the Yield line reports both
# halves separately: `blocks/h expected` (the product) and `per Mwin` (P x 1e6,
# throughput divided out).  The per-Mwin column is the geometry instrument: it
# makes two covers/shifts comparable at a fixed threshold without waiting hours
# for accepted blocks.
#
# Arms are chosen so that the SHIFT effect can be separated from the generator
# and design-merit effects, at the SAME live threshold (all arms run
# back-to-back against the same node, so difficulty drift is shared):
#   509 covermax_m38  - our production reference
#   509 strong_m38    - same design merit, different generator
#   512 covermax_m30  - covermax at the symmetric shift for our primes
#   512 strong_m30
#   512 lex_m30       - the file the shift512 benchmarks in the README used
#
# NOTE: the p75 shift-512 files are row-starved (ctr_bits=3 -> 7 rows); that
# depresses win/s, not per-Mwin.  Read BOTH columns.
#
# Usage: scripts/ab_geometry_yield.sh [secs]
set -u
cd "$(dirname "$0")/.." || exit 1

SECS="${1:-120}"
OUT=/tmp/ab_geom
mkdir -p "$OUT"

FILES="
data/crt/m23/shift509_p74_covermax_m38.txt
data/crt/m23/shift509_p74_strong_m38.txt
data/crt/m23/shift512_p75_covermax_m30.txt
data/crt/m23/shift512_p75_strong_m30.txt
data/crt/m23/shift512_p75_lex_m30.txt
"

printf "%-44s %-9s %-11s %-10s %-9s %s\n" \
    "cover" "win/s" "per Mwin" "blocks/h" "cand" "merit"
for f in $FILES; do
    [ -f "$f" ] || { printf "%-44s (missing)\n" "$(basename "$f")"; continue; }
    log="$OUT/$(basename "$f" .txt).log"
    env FUSED_GPU=1 timeout --signal=TERM --kill-after=8 "$SECS" \
        ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 \
        --pass xx --threads 2 --enable-gpu-fermat --crt-file "$f" \
        >"$log" 2>&1
    w=$(grep -oE "Throughput: [0-9]+ windows/s" "$log" | tail -1 | grep -oE "[0-9]+")
    y=$(grep -oE "Yield: .*" "$log" | tail -1)
    pm=$(printf '%s' "$y" | grep -oE "[0-9.]+ per Mwin" | grep -oE "[0-9.]+")
    bh=$(printf '%s' "$y" | grep -oE "Yield: [0-9.]+" | grep -oE "[0-9.]+")
    cd_=$(printf '%s' "$y" | grep -oE "[0-9]+ candidates" | grep -oE "[0-9]+")
    mt=$(printf '%s' "$y" | grep -oE "merit>=[0-9.]+" | grep -oE "[0-9.]+")
    printf "%-44s %-9s %-11s %-10s %-9s %s\n" "$(basename "$f" .txt)" \
        "${w:--}" "${pm:--}" "${bh:--}" "${cd_:--}" "${mt:--}"
done
echo
echo "(per Mwin = P(merit >= m) x 1e6 measured; cand = cumulative merit candidates)"
