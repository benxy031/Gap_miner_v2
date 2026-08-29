#!/usr/bin/env bash
# gen_crt_batch.sh — batch-generate CRT covering files via bin/gen_crt.
#
# Serially generates a run of `shiftXXX_pYYY_mZZZ.txt` files. The shift is read
# from gen_crt's own output (it derives shift = ceil(log2(primorial)) + bits),
# so file names always match the actual content — no duplicated prime math.
#
# Examples:
#   scripts/gen_crt_batch.sh                       # primes 67..130, merit 23, bits 8
#   scripts/gen_crt_batch.sh --only 74             # single file: shift509_p74_m23
#   scripts/gen_crt_batch.sh --merit 23,30,33,35   # 4 merit variants, same prime range
#   scripts/gen_crt_batch.sh --start 67 --end 98 \
#       --merit 23 --bits 8 --out-dir data/crt/m23
#   scripts/gen_crt_batch.sh --only 67 --evolution # "strong" (higher quality) variant
#   scripts/gen_crt_batch.sh --only 128 --merit 40 --run-objective
#                        # longest-covered-run objective; files named
#                        # "*_objective_*" (measured WORSE for mining at
#                        # shift 998: sigma 1.87 -> 1.40; keep for research)
#
# Existing files are skipped unless --force is given (idempotent re-runs).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN=bin/gen_crt
OUT_DIR=data/crt/m23
START=67
END=130
MERITS="23"
BITS=8
STRENGTH=19000
IVS=6000
FIXED=8
RANGE=0
EVOLUTION=1
RUN_OBJECTIVE=0
FORCE=0
BUILD=1
ATTEMPTS=3

usage() {
    cat <<EOF
Usage: $0 [options]

Batch-generate CRT covering files with bin/gen_crt.

Options:
  --start N          First prime count (default $START)
  --end N            Last prime count  (default $END)
  --only N           Generate a single prime count (overrides --start/--end)
  --merit LIST       Comma/space separated merit targets (default "$MERITS")
  --bits B           Extra bits: shift - log2(primorial) (default $BITS)
  --out-dir DIR      Output directory (default $OUT_DIR)
  --strength S       Greedy restarts / quality (default $STRENGTH)
  --evolution        Enable evolutionary refinement -> "*_strong_*" file names
  --run-objective    Maximize longest covered run (passes --ctr-run-objective;
                     file names use "_objective" instead of "_strong";
                     selection keeps the longest-run result instead of the
                     fewest-candidates one)
  --ivs I            Evolution population size (default $IVS)
  --fixed F          Primes frozen during evolution (default $FIXED)
  --range R          Percent deviation from --ctr-primes (default $RANGE)
  --force            Overwrite existing files
  --attempts K       Runs per file; keep the fewest-candidates result (default $ATTEMPTS)
  --no-build         Do not auto-build $BIN if missing
  --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)      START="${2:?--start needs a value}"; shift 2 ;;
        --end)        END="${2:?--end needs a value}"; shift 2 ;;
        --only)       START="${2:?--only needs a value}"; END="$START"; shift 2 ;;
        --merit)      MERITS="${2:?--merit needs a value}"; shift 2 ;;
        --bits)       BITS="${2:?--bits needs a value}"; shift 2 ;;
        --out-dir)    OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
        --strength)   STRENGTH="${2:?--strength needs a value}"; shift 2 ;;
        --ivs)        IVS="${2:?--ivs needs a value}"; shift 2 ;;
        --fixed)      FIXED="${2:?--fixed needs a value}"; shift 2 ;;
        --range)      RANGE="${2:?--range needs a value}"; shift 2 ;;
        --evolution)  EVOLUTION=1; shift ;;
        --run-objective) RUN_OBJECTIVE=1; shift ;;
        --force)      FORCE=1; shift ;;
        --attempts)   ATTEMPTS="${2:?--attempts needs a value}"; shift 2 ;;
        --no-build)   BUILD=0; shift ;;
        --help|-h)    usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ $BUILD -eq 1 && ! -x "$BIN" ]]; then
    echo "[build] $BIN missing, running make..."
    make "$BIN"
fi
[[ -x "$BIN" ]] || { echo "error: $BIN not found and not built" >&2; exit 1; }

mkdir -p "$OUT_DIR"
MERITS="${MERITS//,/ }"

run_one() {
    # $1=n  $2=merit  $3=tmp_path
    # stdout on success: "<shift> <n_primes> <n_candidates> <longest_run>"
    local n=$1 merit=$2 tmp=$3
    local args=( --calc-ctr --ctr-primes "$n" --ctr-merit "$merit" --ctr-bits "$BITS"
                 --ctr-strength "$STRENGTH" --ctr-fixed "$FIXED" --ctr-ivs "$IVS"
                 --ctr-range "$RANGE" --ctr-file "$tmp" )
    [[ $EVOLUTION -eq 1 ]] && args+=( --ctr-evolution )
    [[ $RUN_OBJECTIVE -eq 1 ]] && args+=( --ctr-run-objective )

    local out wrote_line shift_val n_val cand run_val
    if ! out=$(./"$BIN" "${args[@]}" 2>&1); then
        return 1
    fi
    wrote_line=$(printf '%s\n' "$out" | grep '^wrote ' || true)
    shift_val=$(printf '%s\n' "$wrote_line" | sed -n 's/.*shift=\([0-9][0-9]*\).*/\1/p')
    n_val=$(printf '%s\n' "$wrote_line" | sed -n 's/^wrote .* (\([0-9][0-9]*\) primes,.*/\1/p')
    cand=$(grep '^n_candidates ' "$tmp" 2>/dev/null | awk '{print $2}')
    run_val=$(printf '%s\n' "$wrote_line" | sed -n 's/.*longest_run=\([0-9][0-9]*\).*/\1/p')
    if [[ -z "$shift_val" || -z "$n_val" || -z "$cand" ]]; then
        return 1
    fi
    printf '%s %s %s %s\n' "$shift_val" "$n_val" "$cand" "${run_val:-0}"
}

total=0; skipped=0; failed=0
for merit in $MERITS; do
    for (( n = START; n <= END; n++ )); do
        best_cand=9223372036854775807
        best_run=0
        best_tmp=""
        best_shift=""
        best_n=""
        ok_attempts=0

        for (( a = 1; a <= ATTEMPTS; a++ )); do
            tmp="$OUT_DIR/.tmp_crt_$$_${n}_m${merit}_${a}.txt"
            if ! res=$(run_one "$n" "$merit" "$tmp"); then
                rm -f "$tmp"
                continue
            fi
            read -r s nv cv rv <<< "$res"
            ok_attempts=$((ok_attempts+1))
            keep=0
            if [[ $RUN_OBJECTIVE -eq 1 ]]; then
                [[ "$rv" -gt "$best_run" ]] && keep=1
            else
                [[ "$cv" -lt "$best_cand" ]] && keep=1
            fi
            if [[ $keep -eq 1 ]]; then
                [[ -n "$best_tmp" ]] && rm -f "$best_tmp"
                best_cand=$cv; best_run=$rv; best_shift=$s; best_n=$nv; best_tmp=$tmp
            else
                rm -f "$tmp"
            fi
        done

        if [[ $ok_attempts -eq 0 ]]; then
            echo "FAIL: n=$n merit=$merit: all $ATTEMPTS attempts failed" >&2
            failed=$((failed+1))
            continue
        fi

        if [[ $RUN_OBJECTIVE -eq 1 ]]; then
            suffix="_objective"
        elif [[ $EVOLUTION -eq 1 ]]; then
            suffix="_strong"
        else
            suffix=""
        fi
        final="$OUT_DIR/shift${best_shift}_p${best_n}${suffix}_m${merit}.txt"

        if [[ -e "$final" && $FORCE -eq 0 ]]; then
            echo "skip $final (exists)"
            skipped=$((skipped+1))
            rm -f "$best_tmp"
            continue
        fi

        mv "$best_tmp" "$final"
        total=$((total+1))
        if [[ $RUN_OBJECTIVE -eq 1 ]]; then
            echo "  ok  $final  ${best_n} primes, ${best_cand} candidates, longest_run ${best_run} (best of $ok_attempts)"
        else
            echo "  ok  $final  ${best_n} primes, ${best_cand} candidates (best of $ok_attempts)"
        fi
    done
done

echo "-------------------------------------------"
echo "done: generated=$total skipped=$skipped failed=$failed"
[[ $failed -eq 0 ]]
