/*
 * p0_types.h - host AND device visible types for the Phase-0 GPU toolkit.
 *
 * Deliberately free of CUDA syntax so a plain C/C++ HOST translation unit can
 * include it: the Linux build keeps everything in one .cu, while the Windows
 * build splits the kernels into phase0gpu.dll (nvcc + MSVC) and the host code
 * into MinGW binaries (Makefile.win), and the host still needs P0Item to build
 * the work-item table and p0_vis_mask30 to feed the marking kernel.
 */
#ifndef P0_TYPES_H
#define P0_TYPES_H

#include <stdint.h>

struct P0Item {
    uint32_t pidx;   /* index into the prime table */
    uint32_t k0;     /* first hit index covered by this item */
};

#define P0_CHUNK 64u

/* One reported gap of the class-30 walk engine (device kernel output and
   host consumer, hence host-safe here).  out[0].gap is the count. */
typedef struct { uint64_t slot; uint32_t gap; uint32_t pad; } w_gaprec_t;

/* host: 30-bit mask of slot residues s%30 whose value v0+2s is NOT divisible
   by 3 or 5 (the slots the {3,5} wheel covers); pass v0 mod 30. */
static inline uint32_t p0_vis_mask30(uint32_t v0_mod30) {
    uint32_t m = 0;
    for (uint32_t t = 0; t < 30u; t++) {
        uint32_t n = (v0_mod30 + 2u * t) % 30u;
        if (n % 3u != 0u && n % 5u != 0u) m |= 1u << t;
    }
    return m;
}

#endif /* P0_TYPES_H */
