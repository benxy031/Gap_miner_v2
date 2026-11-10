#!/usr/bin/env bash
# End-to-end width, output parity and checkpoint regression tests.
set -euo pipefail

cd "$(dirname "$0")/.."
bin=${GAPMINER_BIN:-./bin/gapminer}
verify=${TEST_GAP_HUNT_BIN:-./bin/test_gap_hunt}
work=$(mktemp -d "${TMPDIR:-/tmp}/gap-hunt-width.XXXXXXXX")
names=(w320_full w320_jump w320_jump2 w640_full w640_jump w640_jump2
       cross320 cross640 overflow_full overflow_serial overflow_prefetch
       overflow_start missing_final missing_periodic)
cleanup() {
    for name in "${names[@]}"; do
        rm -f -- "$work/$name.log" "$work/$name.gaps" "$work/$name.state" \
            "$work/$name.state.tmp"
    done
    rmdir -- "$work"
}
trap cleanup EXIT

cover=tests/data/shift55_width_cover.txt
gpu_env=(GPU_FERMAT_NARROW_WIDTHS=1 GPU_FERMAT_NO_CGBN=0 GPU_MR_KERNEL=cgbn)
fail() {
    echo "FAIL: $1" >&2
    if [[ -n ${2:-} && -f $2 ]]; then tail -n 12 "$2" >&2; fi
    exit 1
}

run_parity() {
    local width=$1 width_cover=$2 merit=$3 kernel=$4
    local mode jump jump2 prefetch name
    for mode in full jump jump2; do
        case "$mode" in
            full)  jump=0; jump2=0; prefetch=0 ;;
            jump)  jump=1; jump2=0; prefetch=0 ;;
            jump2) jump=0; jump2=1; prefetch=1 ;;
        esac
        name=w${width}_$mode
        env "${gpu_env[@]}" GAP_HUNT_KMAX=1024 GAP_HUNT_BATCH=128 \
            GAP_HUNT_JUMP="$jump" GAP_HUNT_JUMP2="$jump2" \
            GAP_HUNT_PREFETCH="$prefetch" \
            "$bin" --gap-hunt --crt-file "$width_cover" \
            --gap-hunt-min-merit "$merit" --gap-hunt-state "$work/$name.state" \
            --gap-hunt-out "$work/$name.gaps" >"$work/$name.log" 2>&1 ||
            fail "$name hunt failed" "$work/$name.log"
        [[ $(awk '$1 == "k" {print $2}' "$work/$name.state") == 1024 ]] ||
            fail "$name checkpoint is not 1024" "$work/$name.log"
    done
    grep -q "$kernel" "$work/w${width}_jump2.log" ||
        fail "$width-bit CGBN kernel not active; build with WITH_CGBN_FERMAT=1" \
            "$work/w${width}_jump2.log"
    [[ -s $work/w${width}_full.gaps ]] ||
        fail "$width-bit parity test emitted no gaps" "$work/w${width}_full.log"
    for mode in jump jump2; do
        diff -q <(LC_ALL=C sort "$work/w${width}_full.gaps") \
                <(LC_ALL=C sort "$work/w${width}_$mode.gaps") ||
            fail "$width-bit $mode output differs from full scan"
    done
    "$verify" "$work/w${width}_jump2.gaps" bpsw
    echo "PASS: full, jump and jump2 emit the same $width-bit gaps"
}
run_parity 320 "$cover" 7 'CGBN kernel active (AL=5, 320-bit, TPI=2)'
run_parity 640 data/crt/m23/shift258_p43_lex_m30.txt 8 \
    'CGBN kernel active (AL=10, 640-bit, TPI=4)'

anchor320=$(BC_LINE_LENGTH=0 bc <<< 'obase=16; 2^320 - 2^47')
anchor640=$(BC_LINE_LENGTH=0 bc <<< 'obase=16; 2^640 - 2^254')
for width in 320 640; do
    if [[ $width == 320 ]]; then
        anchor=$anchor320
        width_cover=$cover
        expected='AL=6, 384-bit'
    else
        anchor=$anchor640
        width_cover=data/crt/m23/shift258_p43_lex_m30.txt
        expected='AL=12, 768-bit'
    fi
    env "${gpu_env[@]}" GAP_HUNT_KMAX=64 GAP_HUNT_BATCH=8 \
        GAP_HUNT_JUMP=0 GAP_HUNT_JUMP2=1 GAP_HUNT_PREFETCH=0 \
        "$bin" --gap-hunt --crt-file "$width_cover" \
        --gap-hunt-start "$anchor" --gap-hunt-min-merit 7 \
        --gap-hunt-state "$work/cross$width.state" \
        --gap-hunt-out "$work/cross$width.gaps" >"$work/cross$width.log" 2>&1 ||
        fail "$width-bit boundary hunt failed" "$work/cross$width.log"
    grep -q "$expected" "$work/cross$width.log" ||
        fail "$width-bit boundary did not widen" "$work/cross$width.log"
    [[ -s $work/cross$width.gaps ]] ||
        fail "$width-bit boundary test emitted no gaps" "$work/cross$width.log"
    "$verify" "$work/cross$width.gaps" bpsw
done
echo "PASS: walks crossing 320/640 bits widen without truncation"

anchor2048=$(BC_LINE_LENGTH=0 bc <<< 'obase=16; 2^2048 - 10*7420738134810')
for mode in full serial prefetch; do
    case "$mode" in
        full)     jump2=0; prefetch=0; expected=10 ;;
        serial)   jump2=1; prefetch=0; expected=10 ;;
        prefetch) jump2=1; prefetch=1; expected=8 ;;
    esac
    name=overflow_$mode
    if env "${gpu_env[@]}" GAP_HUNT_KMAX=64 GAP_HUNT_BATCH=2 \
        GAP_HUNT_JUMP=0 GAP_HUNT_JUMP2="$jump2" GAP_HUNT_PREFETCH="$prefetch" \
        "$bin" --gap-hunt --crt-file "$cover" \
        --gap-hunt-start "$anchor2048" --gap-hunt-min-merit 15 \
        --gap-hunt-state "$work/$name.state" >"$work/$name.log" 2>&1; then
        fail "$name unexpectedly succeeded" "$work/$name.log"
    else
        rc=$?
        [[ $rc == 1 ]] || fail "$name returned $rc, expected 1" "$work/$name.log"
    fi
    grep -q 'exceeds the 32-limb GPU width (k=10' "$work/$name.log" ||
        fail "$name did not stop at the width limit" "$work/$name.log"
    [[ $(awk '$1 == "k" {print $2}' "$work/$name.state") == "$expected" ]] ||
        fail "$name checkpoint is not the first unprocessed flight" "$work/$name.log"
done
echo "PASS: 2048-bit overflow preserves safe flight watermarks"

anchor_start=$(BC_LINE_LENGTH=0 bc <<< 'obase=16; 2^2048 + 2^64')
if env "${gpu_env[@]}" GAP_HUNT_KMAX=64 GAP_HUNT_BATCH=2 \
    GAP_HUNT_JUMP=0 GAP_HUNT_JUMP2=1 GAP_HUNT_PREFETCH=0 \
    "$bin" --gap-hunt --crt-file "$cover" --gap-hunt-start "$anchor_start" \
    --gap-hunt-min-merit 15 --gap-hunt-state "$work/overflow_start.state" \
    >"$work/overflow_start.log" 2>&1; then
    fail "over-width start unexpectedly succeeded" "$work/overflow_start.log"
else
    rc=$?
    [[ $rc == 1 ]] || fail "over-width start returned $rc" "$work/overflow_start.log"
fi
grep -q 'exceeds the 32-limb GPU width (k=0' "$work/overflow_start.log" ||
    fail "over-width start did not stop at k=0" "$work/overflow_start.log"
[[ $(awk '$1 == "k" {print $2}' "$work/overflow_start.state") == 0 ]] ||
    fail "over-width start checkpoint is not 0" "$work/overflow_start.log"
echo "PASS: an over-width anchor fails before processing the first window"

for mode in final periodic; do
    if [[ $mode == final ]]; then max=16; else max=2048; fi
    name=missing_$mode
    if env "${gpu_env[@]}" GAP_HUNT_KMAX="$max" GAP_HUNT_BATCH=128 \
        GAP_HUNT_JUMP=0 GAP_HUNT_JUMP2=1 GAP_HUNT_PREFETCH=0 \
        "$bin" --gap-hunt --crt-file "$cover" --gap-hunt-min-merit 15 \
        --gap-hunt-state "$work/nonexistent/state" >"$work/$name.log" 2>&1; then
        fail "$name unexpectedly succeeded" "$work/$name.log"
    else
        rc=$?
        [[ $rc == 1 ]] || fail "$name returned $rc, expected 1" "$work/$name.log"
    fi
    grep -q "failed to save $mode state" "$work/$name.log" ||
        fail "$name did not report the failed write" "$work/$name.log"
done
echo "PASS: periodic and final state write failures return an error"
