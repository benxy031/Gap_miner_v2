#!/usr/bin/env bash
# build_host.sh - builds bin/gapminer.exe + tests with MinGW-w64 (MSYS2 MINGW64)
# and runs the validation tests.  Called by windows/build_all.bat.
#
# Environment:
#   GPU_NLIMBS  must match gapgpu.dll (default 32, see build_gpu_dll.bat)
#   TEST_GPU    optional: run the GPU tests on this GPU only (PCI bus order,
#               as nvidia-smi), e.g. to keep them off cards that are mining.
set -u
cd "$(dirname "$0")/.."
mkdir -p windows/logs
LOG=windows/logs/build_host.log
TLOG=windows/logs/tests.log
: > "$LOG"
: > "$TLOG"

echo "[$(date '+%F %T')] build_host start (MSYSTEM=${MSYSTEM:-?})" | tee -a "$LOG"
gcc --version | head -1 >> "$LOG"

CUDA_WIN="${CUDA_HOME:-${CUDA_PATH_V12_9:-${CUDA_PATH:-}}}"
if [ -z "$CUDA_WIN" ]; then
    echo "ERROR: CUDA toolkit not found (set CUDA_HOME)" | tee -a "$LOG"
    exit 1
fi
CUDA_INC="$(cygpath -u "$CUDA_WIN")/include"
echo "CUDA_INC=$CUDA_INC" >> "$LOG"

if [ ! -f bin/gapgpu.dll ]; then
    echo "ERROR: bin/gapgpu.dll missing - build_gpu_dll.bat must succeed first" | tee -a "$LOG"
    exit 1
fi

GPU_NLIMBS="${GPU_NLIMBS:-32}"
echo "GPU_NLIMBS=$GPU_NLIMBS" >> "$LOG"
make -f Makefile.win -j4 all CUDA_INC="$CUDA_INC" GPU_NLIMBS="$GPU_NLIMBS" >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] make exit code $rc" | tee -a "$LOG"
[ $rc -ne 0 ] && exit $rc

# Runtime DLLs: gapgpu.dll + cudart64_*.dll live in bin/, MSYS2 DLLs in PATH.
export PATH="$PWD/bin:/mingw64/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${TEST_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$TEST_GPU"
fi

run_test() {
    local t="$1"
    echo "===== $t (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}) =====" >> "$TLOG"
    local start=$(date +%s)
    ./bin/$t.exe >> "$TLOG" 2>&1
    local trc=$?
    local dur=$(( $(date +%s) - start ))
    echo "----- $t exit=$trc (${dur}s)" >> "$TLOG"
    printf '%-24s exit=%-3s %4ss\n' "$t" "$trc" "$dur" | tee -a "$LOG"
    return $trc
}

fail=0
for t in test_primality test_gap_detection test_sieve_core test_crt_submission \
         test_block_submission test_halfclass \
         test_gpu_fermat test_gpu_sieve test_gpu_resolve; do
    run_test "$t" || fail=$((fail + 1))
done

echo "[$(date '+%F %T')] tests done, failures=$fail" | tee -a "$LOG"
exit $fail
