#!/usr/bin/env bash
# ab_cover_sigma.sh - two-threshold cover yield comparison (sigma) with ABBA
# ordering, for the "which cover should production run?" decision.
#
# Why two thresholds: sigma = 8 / ln(P10/P18) is what turns a measured yield
# into a live-threshold prediction, and the absolute P values alone do NOT rank
# covers (a cover can win at merit 10 and lose at the live threshold - measured
# on the 512 m30/m38 files).  Sigma needs BOTH thresholds, and the deep one
# needs a long arm because its rate is ~20-30 per million windows: run this
# with m18 >= ~420 s per arm so n(18) is ~100+ (1-sigma ~10%).
#
# Usage:
#   scripts/ab_cover_sigma.sh [m18_seconds] [m10_seconds] coverA coverB [...]
# Defaults: 420 s at merit 18, 45 s at merit 10 (2 arms per cover, ABBA order).
# Needs FUSED_GPU path flags; the miner is dry-run (no --enable-submission), so
# nothing is submitted and the record log stays clean (run it from /tmp).
set -uo pipefail

T18=${1:-420}; shift || true
T10=${1:-45};  shift || true
if [ "$#" -lt 2 ]; then
    echo "usage: $0 [m18_seconds] [m10_seconds] coverA coverB [...]" >&2
    exit 2
fi
COVERS=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin/gapminer"
OUT="${AB_COVER_OUT:-/tmp/ab_cover_sigma}"
mkdir -p "$OUT"
THREADS=${AB_COVER_THREADS:-2}

# One arm: frozen threshold, dry-run, report the Yield line (per-Mwin rate + n).
run_arm() {
    local cover="$1" merit="$2" secs="$3" tag="$4"
    local log="$OUT/${tag}.log"
    ( cd /tmp && MINING_JUMP2=1 FUSED_GPU=1 timeout -s INT "$secs" \
        "$BIN" --crt-file "$cover" --threads "$THREADS" \
        --enable-gpu-fermat --merit "$merit" > "$log" 2>&1 )
    local win rate n
    win=$(grep -oE "Throughput: [0-9]+" "$log" | tail -1 | awk '{print $2}')
    rate=$(grep -oE "\| [0-9.]+ per Mwin" "$log" | tail -1 | awk '{print $2}')
    n=$(grep -oE "n=[0-9]+" "$log" | tail -1 | cut -d= -f2)
    printf "%s\t%s\t%s\t%s\t%s\n" "$tag" "$(basename "$cover")" "$merit" "${rate:-0}" "${n:-0}" >> "$OUT/arms.tsv"
    echo "  $tag: merit=$merit P/Mwin=${rate:-?} n=${n:-?} win/s=${win:-?}"
}

: > "$OUT/arms.tsv"
echo "=== cover yield A/B (ABBA, ${T18}s @18 + ${T10}s @10 per cover, $THREADS threads) ==="
# ABBA: forward pass, then reverse pass, so drift lands on both covers.
for pass in 1 2; do
    if [ "$pass" = 1 ]; then order=("${COVERS[@]}"); else
        order=(); for ((i=${#COVERS[@]}-1; i>=0; i--)); do order+=("${COVERS[$i]}"); done
    fi
    for cover in "${order[@]}"; do
        base="$(basename "$cover" .txt)"
        echo "pass$pass $base"
        run_arm "$cover" 18 "$T18" "p${pass}_${base}_m18"
        run_arm "$cover" 10 "$T10" "p${pass}_${base}_m10"
    done
done

echo
echo "=== summary (P per Mwin; sigma = 8/ln(P10/P18); ratio at m18) ==="
awk -F'\t' '
{ cov=$2; m=$3; r=$4; s[cov"|"m]+=r; c[cov"|"m]++
  if (!(cov in seen)) { seen[cov]=1; order[++k]=cov } }
END {
    printf "%-34s %12s %12s %8s\n", "cover", "P10/Mwin", "P18/Mwin", "sigma"
    for (i=1; i<=k; i++) {
        cov=order[i]
        p10=(c[cov"|"10]>0)? s[cov"|"10]/c[cov"|"10] : 0
        p18=(c[cov"|"18]>0)? s[cov"|"18]/c[cov"|"18] : 0
        sg=(p10>0 && p18>0)? 8/log(p10/p18) : 0
        printf "%-34s %12.1f %12.2f %8.3f\n", cov, p10, p18, sg
    }
}' "$OUT/arms.tsv"
echo "raw arms: $OUT/arms.tsv  (per-arm logs in $OUT)"
