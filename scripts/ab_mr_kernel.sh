#!/usr/bin/env bash
# Fixed-time A/B of the GPU MR backend: GPU_MR_KERNEL=cios (32-bit CIOS,
# 1 thread/candidate, no shuffles) vs the default CGBN TPI=8.
#
# Same binary, same CRT file, same thread count, same wall time; only the MR
# backend changes.  Run from the repo root.
#
# Usage: [SECS=45] [THREADS=8] [CRT=<path>] scripts/ab_mr_kernel.sh
#
# Reading the result: "Throughput: N windows/s" is the end-to-end metric;
# "acc/wall" tells whether the run was GPU-bound (>1) or host-bound (<1) --
# a backend win only shows up in windows/s when the run is GPU-bound.
# MINING_JUMP2_BATCH may be exported beforehand to test the batch-geometry
# hypothesis (the CIOS shape needs a big MR batch to fill the GPU).
set -u

SECS=${SECS:-45}
THREADS=${THREADS:-8}
CRT=${CRT:-data/crt/m23/shift507_p74_lex_m31.txt}
JUMP2=${MINING_JUMP2_BATCH:-2048}

if [ ! -f "$CRT" ]; then
    echo "missing CRT file: $CRT" >&2
    exit 1
fi

arm() {
    local name="$1" kern="$2"
    echo "=== arm $name (GPU_MR_KERNEL=${kern:-<unset>}, JUMP2_BATCH=$JUMP2) ==="
    MINING_JUMP2=1 MINING_JUMP2_BATCH="$JUMP2" FUSED_GPU=1 GPU_SIEVE=1 \
        GPU_MR_KERNEL="$kern" \
        timeout --kill-after=8 "$SECS" ./bin/gapminer --crt-file "$CRT" \
        --threads "$THREADS" --enable-gpu-fermat 2>&1 \
    | grep -E "kernel active|BATCH .* ->|Throughput:|acc/wall|Sieve survivors|andidates tested" \
    | tail -8
}

arm CGBN-default ""
arm CIOS "cios"
