/*
 * phase0gpu_api.h - plain-C API of phase0gpu.dll (Windows kernel-DLL split).
 *
 * On Windows, nvcc requires MSVC as host compiler, while the Phase-0 tools
 * need POSIX threads + GMP (easiest from MSYS2/MinGW).  The kernels are
 * therefore compiled by nvcc + MSVC into phase0gpu.dll and the tools are
 * built by MinGW (Makefile.win), calling only this API.  On Linux the same
 * .cu files are compiled as one translation unit and this header is unused.
 *
 * All device pointers are raw void* (the allocator is cudart, which both
 * sides share as one runtime DLL/instance), streams are cudaStream_t cast to
 * void*, and every function returns 0 on success / -1 on a CUDA error.
 * Grid/block/shared geometry is computed by the caller (it owns the policy)
 * and forwarded verbatim, so the DLL never decides launch shapes.
 */
#ifndef PHASE0GPU_API_H
#define PHASE0GPU_API_H

#include <stddef.h>
#include <stdint.h>

/* 32-bit limbs per candidate in mr68_kernel.cuh (mirrors #define NW there;
   host code sizes its candidate buffers with it). */
#define P0GPU_NW 3

#ifdef __cplusplus
extern "C" {
#endif

int p0gpu_mr68_packed(void *d_base3, void *d_steps, void *d_res, unsigned cnt,
                      unsigned grid, unsigned tpb, void *stream);

int p0gpu_p0_wheel(const void *d_wpat2, unsigned wheel_p, unsigned wheel_inv2,
                   unsigned long long wheel_r64p, unsigned long long v0_lo,
                   unsigned long long v0_hi, unsigned nwords, void *d_bm,
                   unsigned grid, unsigned tpb, void *stream);

int p0gpu_p0_mark(const void *d_items, unsigned nitems, const void *d_primes64,
                  const void *d_r64, const void *d_invp,
                  unsigned long long half, unsigned long long v0_lo,
                  unsigned long long v0_hi, unsigned v0_mod30, void *d_bm,
                  unsigned grid, unsigned tpb, void *stream);

int p0gpu_p0_compact(const void *d_bm, unsigned nwords,
                     unsigned long long last_mask, void *d_offs, void *d_cnt,
                     unsigned grid, void *stream);

int p0gpu_mr68_from_offsets(unsigned long long v0_lo, unsigned long long v0_hi,
                            const void *d_offs, const void *d_cnt, void *d_rbm,
                            unsigned grid, void *stream);

int p0gpu_mr68_kernel(void *d_cands, void *d_res, unsigned cnt, unsigned grid,
                      unsigned tpb, void *stream);

int p0gpu_class30_sieve(const void *primes, const void *invp, const void *r64,
                        const void *item_pidx, const void *item_k0,
                        unsigned nitems, unsigned long long alo,
                        unsigned long long ahi, unsigned long long tile_slots,
                        unsigned long long ntiles, void *g_bm,
                        unsigned tile_words, const void *wpidx, int wheel_on,
                        unsigned grid, unsigned block, size_t shared_bytes,
                        void *stream);

int p0gpu_walk(const void *bm, unsigned long long nwords,
               unsigned long long alo, unsigned long long ahi,
               const void *wtab, unsigned min_gap, void *out, unsigned cap,
               void *stats, unsigned grid, unsigned block, void *stream);

/* c30_* device constant tables (30 bytes / 2 x 120 bytes host arrays). */
int p0gpu_c30_tables(const void *clsidx, const void *perm, const void *pinv);

/* cudaFuncSetAttribute(class30_sieve_kernel, MaxDynamicSharedMemorySize) */
int p0gpu_set_sieve_shared(size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* PHASE0GPU_API_H */
