#!/usr/bin/env bash
# Geometry comparison, two thresholds per cover.
#
# `Yield` gives P(merit >= m) directly (per Mwin = P x 1e6), so two thresholds
# per cover yield the local tail exponent
#       sigma = (m_hi - m_lo) / ln(P_lo / P_hi)
# which is the transferable geometry number, plus win/s measured at a MATCHED
# threshold (a lower threshold makes the chain stop earlier, so win/s is only
# comparable between arms run at the same --merit).
#
# CAVEAT (documented in repo memory): the tail is lighter than exponential at
# depth (measured sigma 1.29 at M0=14 -> 0.858 at M0=19 for one cover), so
# sigma fitted at 10..18 OVERESTIMATES the live rate; use the implied
# blocks/h as a RANKING, never as a level.  The level comes from accepted
# blocks/h at live difficulty.
#
# Usage: scripts/ab_geometry_sigma.sh [secs_per_arm]
set -u
cd "$(dirname "$0")/.." || exit 1

SECS="${1:-60}"
OUT=/tmp/ab_geom
mkdir -p "$OUT"

FILES="
data/crt/m23/shift509_p74_covermax_m38.txt
data/crt/m23/shift509_p74_strong_m38.txt
data/crt/m23/shift512_p75_covermax_m30.txt
data/crt/m23/shift512_p75_strong_m30.txt
data/crt/m23/shift512_p75_lex_m30.txt
"

run() { # file merit logfile
    env FUSED_GPU=1 timeout --signal=TERM --kill-after=8 "$1" >/dev/null 2>&1
}
run_arm() { # file merit secs log
    env FUSED_GPU=1 timeout --signal=TERM --kill-after=8 "$3" \
        ./bin/gapminer --host 127.0.0.1 --port 31397 --user benxy031 \
        --pass xx --threads 2 --enable-gpu-fermat --crt-file "$1" \
        --merit "$2" >"$4" 2>&1
}

printf "%-34s %-8s %-10s %-10s %-9s %-9s %s\n" \
    "cover" "win/s@18" "P10/Mwin" "P18/Mwin" "sigma" "b/h@18" "b/h@live (ranking)"
for f in $FILES; do
    b="$(basename "$f" .txt)"
    [ -f "$f" ] || { printf "%-34s (missing)\n" "$b"; continue; }
    run_arm "$f" 10 "$SECS" "$OUT/${b}_m10.log"
    run_arm "$f" 18 "$SECS" "$OUT/${b}_m18.log"
    p10=$(grep -oE "Yield: .*" "$OUT/${b}_m10.log" | tail -1 |
          grep -oE "[0-9.]+ per Mwin" | grep -oE "[0-9.]+")
    p18=$(grep -oE "Yield: .*" "$OUT/${b}_m18.log" | tail -1 |
          grep -oE "[0-9.]+ per Mwin" | grep -oE "[0-9.]+")
    w18=$(grep -oE "Throughput: [0-9]+ windows/s" "$OUT/${b}_m18.log" | tail -1 |
          grep -oE "[0-9]+")
    mlive=$(grep -oE "Difficulty: [0-9.]+" "$OUT/${b}_m18.log" | tail -1 |
            grep -oE "[0-9.]+")
    python3 - "$b" "$w18" "$p10" "$p18" "$mlive" <<'PY'
import math, sys
b, w, p10, p18, mlive = sys.argv[1:6]
w = float(w or 0); p10 = float(p10 or 0) / 1e6; p18 = float(p18 or 0) / 1e6
ml = float(mlive or 0)
# b/h@18 is MEASURED (win/s x P18 x 3600); the live column extrapolates it with
# the fitted sigma, which is only valid as a RANKING (the tail lightens with m,
# so a sigma fitted at 10..18 overestimates the live rate).
if p10 > 0 and p18 > 0 and p10 > p18:
    sigma = 8.0 / math.log(p10 / p18)
    bh18 = w * p18 * 3600.0
    bhlive = bh18 * math.exp(-(ml - 18.0) / sigma) if ml > 18.0 else 0.0
    print(f"{b:<34} {w:<8.0f} {p10*1e6:<10.3f} {p18*1e6:<10.3f} "
          f"{sigma:<9.3f} {bh18:<9.1f} {bhlive:<9.2f} (m={ml:.2f})")
else:
    print(f"{b:<34} {w:<8.0f} {p10*1e6:<10.3f} {p18*1e6:<10.3f} "
          f"{'n/a':<9} {'n/a':<9} n/a (no m18 samples)")
PY
done
