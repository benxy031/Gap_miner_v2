#!/usr/bin/env bash
# CUDA parity, resume and fail-closed regression tests for the center chain.
set -euo pipefail

cd "$(dirname "$0")/.."
bin=${GAPMINER_BIN:-./bin/gapminer}
verify=${TEST_GAP_HUNT_BIN:-./bin/test_gap_hunt}
work=$(mktemp -d "${TMPDIR:-/tmp}/gap-hunt-center.XXXXXXXX")
cleanup() {
    rm -f -- "$work"/*.log "$work"/*.gaps "$work"/*.state \
        "$work"/*.state.tmp "$work"/*.cover "$work"/*.sorted
    rmdir -- "$work"
}
trap cleanup EXIT

cover320=tests/data/shift55_width_cover.txt
cover640=data/crt/m23/shift258_p43_lex_m30.txt
cover2048=data/crt/m30/shift1784_p207_strong_m30.txt
gpu_env=(GPU_FERMAT_NARROW_WIDTHS=1 GPU_FERMAT_NO_CGBN=0 GPU_MR_KERNEL=cgbn)

fail() {
    echo "FAIL: $1" >&2
    if [[ -n ${2:-} && -f $2 ]]; then tail -n 15 "$2" >&2; fi
    exit 1
}

# Arguments: name cover center jump2 batch chunk wave prefetch pair defer kmax merit.
run_hunt() {
    local name=$1 cover=$2 center=$3 jump2=$4 batch=$5 chunk=$6 wave=$7
    local prefetch=$8 pair=$9
    shift 9
    local defer=$1 kmax=$2 merit=$3
    local center_env=()
    if [[ $center != unset ]]; then center_env=("GAP_HUNT_CENTER=$center"); fi
    env -u GAP_HUNT_CENTER -u GHDBG_DUMP_K "${gpu_env[@]}" "${center_env[@]}" \
        GAP_HUNT_KMAX="$kmax" GAP_HUNT_BATCH="$batch" GAP_HUNT_JUMP=0 \
        GAP_HUNT_JUMP2="$jump2" GAP_HUNT_QUARTER=0 \
        GAP_HUNT_JUMP2_CHUNK="$chunk" GAP_HUNT_JUMP2_WAVE="$wave" \
        GAP_HUNT_PREFETCH="$prefetch" GAP_HUNT_PAIR="$pair" \
        GAP_HUNT_DEFER_COUNT="$defer" GAP_HUNT_RESIDUE_INCR=2 \
        "$bin" --gap-hunt --crt-file "$cover" --gap-hunt-device 0 \
        --gap-hunt-min-merit "$merit" --gap-hunt-state "$work/$name.state" \
        --gap-hunt-out "$work/$name.gaps" >"$work/$name.log" 2>&1 ||
        fail "$name hunt failed" "$work/$name.log"
    [[ $(awk '$1 == "k" {print $2}' "$work/$name.state") == "$kmax" ]] ||
        fail "$name checkpoint is not $kmax" "$work/$name.log"
    local effective_center=$center
    if [[ $center == unset ]]; then effective_center=0; fi
    grep -Eq " batch=$batch .*center=$effective_center " "$work/$name.log" ||
        fail "$name used a different batch or center mode" "$work/$name.log"
}

same_gaps() {
    local expected=$1 actual=$2
    [[ -s $work/$expected.gaps ]] ||
        fail "$expected parity is vacuous" "$work/$expected.log"
    LC_ALL=C sort "$work/$expected.gaps" > "$work/$expected.sorted"
    LC_ALL=C sort "$work/$actual.gaps" > "$work/$actual.sorted"
    diff -q "$work/$expected.sorted" "$work/$actual.sorted" ||
        fail "$actual differs from $expected" "$work/$actual.log"
}

run_hunt full320 "$cover320" 0 0 32 16 8 0 1 0 512 7
run_hunt default320 "$cover320" unset 1 32 16 8 1 1 0 512 7
run_hunt jump320 "$cover320" 0 1 32 16 8 1 1 0 512 7
run_hunt center320_small "$cover320" 1 1 8 8 0 0 0 0 512 7
run_hunt center320_mid "$cover320" 1 1 32 64 4 4 1 0 512 7
run_hunt center320_deferred "$cover320" 1 1 64 128 16 16 1 1 512 7
for name in default320 jump320 center320_small center320_mid center320_deferred; do
    same_gaps full320 "$name"
done
grep -q 'CGBN kernel active (AL=5, 320-bit, TPI=2)' "$work/center320_mid.log" ||
    fail "320-bit CGBN kernel is not active" "$work/center320_mid.log"
grep -q 'deferred survivor counts ON' "$work/center320_deferred.log" ||
    fail "deferred-count test did not enable the requested mode" "$work/center320_deferred.log"
"$verify" "$work/center320_mid.gaps" bpsw
echo "PASS: full, default jump2 and center agree across batches, waves and fill modes"

run_hunt resume320 "$cover320" 1 1 32 16 8 4 1 0 256 7
run_hunt resume320 "$cover320" 1 1 32 16 8 4 1 0 512 7
grep -q ' k0=256 ' "$work/resume320.log" ||
    fail "resume did not start at k=256" "$work/resume320.log"
same_gaps full320 resume320
echo "PASS: resumed center walk has no missing or duplicate records"

run_hunt resume_to_center "$cover320" 0 1 32 16 8 4 1 0 256 7
run_hunt resume_to_center "$cover320" 1 1 32 16 8 4 1 0 512 7
run_hunt resume_to_jump "$cover320" 1 1 32 16 8 4 1 0 256 7
run_hunt resume_to_jump "$cover320" 0 1 32 16 8 4 1 0 512 7
for name in resume_to_center resume_to_jump; do
    grep -q ' k0=256 ' "$work/$name.log" ||
        fail "$name did not resume at k=256" "$work/$name.log"
    same_gaps full320 "$name"
done
echo "PASS: checkpoint resumes preserve parity when switching in either direction"

run_hunt full640 "$cover640" 0 0 32 16 8 0 1 0 512 8
run_hunt jump640 "$cover640" 0 1 32 16 8 4 1 0 512 8
run_hunt center640 "$cover640" 1 1 32 16 8 4 1 0 512 8
same_gaps full640 jump640
same_gaps full640 center640
grep -q 'CGBN kernel active (AL=10, 640-bit, TPI=4)' "$work/center640.log" ||
    fail "640-bit CGBN kernel is not active" "$work/center640.log"
"$verify" "$work/center640.gaps" bpsw
echo "PASS: non-vacuous full, jump2 and center parity at 640 bits"

run_hunt full2048 "$cover2048" 0 0 32 16 8 0 1 0 512 8
run_hunt jump2048 "$cover2048" 0 1 32 16 8 4 1 0 512 8
run_hunt center2048 "$cover2048" 1 1 32 16 8 4 1 0 512 8
same_gaps full2048 jump2048
same_gaps full2048 center2048
grep -q 'CGBN kernel active (AL=32, 2048-bit, TPI=8)' "$work/center2048.log" ||
    fail "2048-bit CGBN kernel is not active" "$work/center2048.log"
"$verify" "$work/center2048.gaps" bpsw
echo "PASS: non-vacuous full, jump2 and center parity at 2048 bits"

for bad in '' 2 01 -1 true; do
    if env GAP_HUNT_CENTER="$bad" "$bin" --gap-hunt --crt-file "$cover320" \
        >"$work/invalid.log" 2>&1; then
        fail "invalid GAP_HUNT_CENTER=$bad was accepted" "$work/invalid.log"
    fi
    grep -q 'GAP_HUNT_CENTER must be 0 or 1' "$work/invalid.log" ||
        fail "invalid center option did not explain the error" "$work/invalid.log"
done
for conflict in GAP_HUNT_JUMP2=0 GAP_HUNT_JUMP2=2 GAP_HUNT_JUMP=1 \
                GAP_HUNT_JUMP=bad GAP_HUNT_QUARTER=1; do
    if env GAP_HUNT_CENTER=1 "$conflict" "$bin" --gap-hunt \
        --crt-file "$cover320" >"$work/conflict.log" 2>&1; then
        fail "$conflict was accepted with center" "$work/conflict.log"
    fi
    grep -q 'GAP_HUNT_CENTER=1 requires' "$work/conflict.log" ||
        fail "$conflict did not explain the incompatibility" "$work/conflict.log"
done
echo "PASS: invalid and incompatible center options fail explicitly"

# The larger window contains more than 128 prime pairs but still fits the
# initial survivor buffer, so this exercises the pair cap rather than the sieve.
sed 's/^gap_target .*/gap_target 20000/' "$cover320" > "$work/capacity.cover"
for prefetch in 0 1; do
    rm -f -- "$work/capacity.state" "$work/capacity.gaps"
    if env "${gpu_env[@]}" GAP_HUNT_CENTER=1 GAP_HUNT_JUMP2=1 GAP_HUNT_JUMP=0 \
        GAP_HUNT_QUARTER=0 GAP_HUNT_BATCH=1 GAP_HUNT_KMAX=2 \
        GAP_HUNT_PREFETCH="$prefetch" "$bin" --gap-hunt \
        --crt-file "$work/capacity.cover" --gap-hunt-min-merit 0 \
        --gap-hunt-state "$work/capacity.state" --gap-hunt-out "$work/capacity.gaps" \
        >"$work/capacity.log" 2>&1; then
        fail "pair-capacity exhaustion did not stop the walk" "$work/capacity.log"
    fi
    grep -q 'center scan failed:.*result=-2 pair_capacity=128' "$work/capacity.log" ||
        fail "pair-capacity error was not reported" "$work/capacity.log"
    [[ $(awk '$1 == "k" {print $2}' "$work/capacity.state") == 0 ]] ||
        fail "capacity checkpoint is not the first unprocessed flight" "$work/capacity.log"
    [[ ! -s $work/capacity.gaps ]] ||
        fail "capacity failure emitted an incomplete flight" "$work/capacity.log"
done
echo "PASS: pair-capacity exhaustion preserves safe checkpoints with and without prefetch"

anchor2048=$(BC_LINE_LENGTH=0 bc <<< 'obase=16; 2^2048 - 10*7420738134810')
for prefetch in 0 1; do
    if [[ $prefetch == 0 ]]; then expected=10; else expected=8; fi
    if env "${gpu_env[@]}" GAP_HUNT_CENTER=1 GAP_HUNT_JUMP=0 GAP_HUNT_JUMP2=1 \
        GAP_HUNT_QUARTER=0 GAP_HUNT_KMAX=64 GAP_HUNT_BATCH=2 \
        GAP_HUNT_PREFETCH="$prefetch" "$bin" --gap-hunt --crt-file "$cover320" \
        --gap-hunt-start "$anchor2048" --gap-hunt-min-merit 15 \
        --gap-hunt-state "$work/overflow.state" >"$work/overflow.log" 2>&1; then
        fail "2048-bit overflow unexpectedly succeeded" "$work/overflow.log"
    fi
    grep -q 'exceeds the 32-limb GPU width (k=10' "$work/overflow.log" ||
        fail "overflow did not stop at the width limit" "$work/overflow.log"
    [[ $(awk '$1 == "k" {print $2}' "$work/overflow.state") == "$expected" ]] ||
        fail "overflow checkpoint is not the safe flight watermark" "$work/overflow.log"
    rm -f -- "$work/overflow.state"
done
echo "PASS: GPU-width failures preserve safe checkpoints with and without prefetch"
