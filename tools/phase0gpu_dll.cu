/*
 * phase0gpu_dll.cu - phase0gpu.dll: the Phase-0 kernels behind a plain-C API.
 *
 * Windows only in practice (nvcc + MSVC, see windows/build_phase0.bat): the
 * MinGW-built host tools (phase0_scan_gpu, mr68_gpu, bench_p0sieve) call the
 * wrappers in phase0gpu_api.h.  The file compiles on Linux too (nvcc), which
 * is how the split is validated: `make bin/phase0_scan_gpu_dlltest` links the
 * host TU (g++ -DPHASE0_KERNEL_DLL) against this object and the outputs are
 * diffed against the single-TU binary.
 *
 * Every wrapper forwards the launch geometry the caller computed and then
 * reports cudaGetLastError() as -1, so the host's existing error paths keep
 * working unchanged.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "phase0gpu_api.h"
#include "mr68_kernel.cuh"
#include "p0_mark.cuh"
#include "p0_walk_kern.cuh"

#if defined(_WIN32)
#define P0GPU_EXPORT extern "C" __declspec(dllexport)
#else
#define P0GPU_EXPORT extern "C"
#endif

static int p0gpu_rc(void) {
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

P0GPU_EXPORT int p0gpu_mr68_packed(void *d_base3, void *d_steps, void *d_res,
                                  unsigned cnt, unsigned grid, unsigned tpb,
                                  void *stream) {
    mr68_kernel_packed<<<grid, tpb, 0, (cudaStream_t)stream>>>(
        (const uint32_t *)d_base3, (const uint32_t *)d_steps,
        (uint8_t *)d_res, cnt);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_p0_wheel(const void *d_wpat2, unsigned wheel_p,
                                unsigned wheel_inv2,
                                unsigned long long wheel_r64p,
                                unsigned long long v0_lo,
                                unsigned long long v0_hi, unsigned nwords,
                                void *d_bm, unsigned grid, unsigned tpb,
                                void *stream) {
    p0_wheel_kernel<<<grid, tpb, 0, (cudaStream_t)stream>>>(
        (const uint64_t *)d_wpat2, wheel_p, wheel_inv2, wheel_r64p, v0_lo,
        v0_hi, nwords, (uint64_t *)d_bm);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_p0_mark(const void *d_items, unsigned nitems,
                               const void *d_primes64, const void *d_r64,
                               const void *d_invp, unsigned long long half,
                               unsigned long long v0_lo,
                               unsigned long long v0_hi, unsigned v0_mod30,
                               void *d_bm, unsigned grid, unsigned tpb,
                               void *stream) {
    p0_mark_kernel<<<grid, tpb, 0, (cudaStream_t)stream>>>(
        (const P0Item *)d_items, nitems, (const uint64_t *)d_primes64,
        (const uint64_t *)d_r64, (const uint64_t *)d_invp, half, v0_lo, v0_hi,
        p0_vis_mask30(v0_mod30), (uint64_t *)d_bm);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_p0_compact(const void *d_bm, unsigned nwords,
                                  unsigned long long last_mask, void *d_offs,
                                  void *d_cnt, unsigned grid, void *stream) {
    p0_compact_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
        (const uint64_t *)d_bm, nwords, last_mask, (uint32_t *)d_offs,
        (uint32_t *)d_cnt);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_mr68_from_offsets(unsigned long long v0_lo,
                                         unsigned long long v0_hi,
                                         const void *d_offs,
                                         const void *d_cnt, void *d_rbm,
                                         unsigned grid, void *stream) {
    mr68_from_offsets<<<grid, 128, 0, (cudaStream_t)stream>>>(
        v0_lo, v0_hi, (const uint32_t *)d_offs, (const uint32_t *)d_cnt,
        (uint64_t *)d_rbm);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_mr68_kernel(void *d_cands, void *d_res, unsigned cnt,
                                   unsigned grid, unsigned tpb, void *stream) {
    mr68_kernel<<<grid, tpb, 0, (cudaStream_t)stream>>>(
        (const uint32_t *)d_cands, (uint8_t *)d_res, cnt);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_class30_sieve(const void *primes, const void *invp,
                                     const void *r64, const void *item_pidx,
                                     const void *item_k0, unsigned nitems,
                                     unsigned long long alo,
                                     unsigned long long ahi,
                                     unsigned long long tile_slots,
                                     unsigned long long ntiles, void *g_bm,
                                     unsigned tile_words, const void *wpidx,
                                     int wheel_on, unsigned grid,
                                     unsigned block, size_t shared_bytes,
                                     void *stream) {
    class30_sieve_kernel<<<grid, block, shared_bytes, (cudaStream_t)stream>>>(
        (const uint64_t *)primes, (const uint64_t *)invp,
        (const uint64_t *)r64, (const uint32_t *)item_pidx,
        (const uint32_t *)item_k0, nitems, alo, ahi, tile_slots, ntiles,
        (uint64_t *)g_bm, tile_words, (const uint32_t *)wpidx, wheel_on);
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_walk(const void *bm, unsigned long long nwords,
                            unsigned long long alo, unsigned long long ahi,
                            const void *wtab, unsigned min_gap, void *out,
                            unsigned cap, void *stats, unsigned grid,
                            unsigned block, void *stream) {
    switch (w_test_mode(ahi)) {
    case 2:
        walk_kernel7<2><<<grid, block, 0, (cudaStream_t)stream>>>(
            (const uint64_t *)bm, nwords, alo, ahi, (const uint32_t *)wtab,
            min_gap, (w_gaprec_t *)out, cap, (unsigned long long *)stats);
        break;
    case 1:
        walk_kernel7<1><<<grid, block, 0, (cudaStream_t)stream>>>(
            (const uint64_t *)bm, nwords, alo, ahi, (const uint32_t *)wtab,
            min_gap, (w_gaprec_t *)out, cap, (unsigned long long *)stats);
        break;
    default:
        walk_kernel7<0><<<grid, block, 0, (cudaStream_t)stream>>>(
            (const uint64_t *)bm, nwords, alo, ahi, (const uint32_t *)wtab,
            min_gap, (w_gaprec_t *)out, cap, (unsigned long long *)stats);
        break;
    }
    return p0gpu_rc();
}

P0GPU_EXPORT int p0gpu_c30_tables(const void *clsidx, const void *perm,
                                  const void *pinv) {
    cudaError_t e = cudaMemcpyToSymbol(c30_clsidx, clsidx, sizeof(c30_clsidx));
    if (e == cudaSuccess)
        e = cudaMemcpyToSymbol(c30_perm, perm, sizeof(c30_perm));
    if (e == cudaSuccess)
        e = cudaMemcpyToSymbol(c30_pinv, pinv, sizeof(c30_pinv));
    return (e == cudaSuccess) ? 0 : -1;
}

P0GPU_EXPORT int p0gpu_set_sieve_shared(size_t bytes) {
    cudaError_t e = cudaFuncSetAttribute(
        class30_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)bytes);
    return (e == cudaSuccess) ? 0 : -1;
}
