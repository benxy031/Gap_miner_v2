/*
 * mr68_kernel.cuh - device code for the dedicated 68-bit-class Miller-Rabin
 * kernel (shared by tools/mr68_gpu.cu bench/validate tool and the live
 * tools/phase0_scan_gpu.cu scanner).
 *
 * Validated by: bin/mr68_gpu --validate N (kernel verdicts vs GMP, 0 mismatch
 * required) and end-to-end by phase0_scan_gpu --check N (full prime-sequence
 * comparison vs a pure GMP walk).
 *
 * Design: one thread per candidate; 3 x 32-bit CIOS Montgomery (96-bit
 * container, valid for any odd n < 2^96); base-2 strong probable prime test.
 * The CIOS core and MR tail are ported from the GMP-validated
 * tools/bench_cios_mr.cu math.
 *
 * Include this header ONLY from .cu translation units.
 */

#ifndef MR68_KERNEL_CUH
#define MR68_KERNEL_CUH

#include <stdint.h>

#define NW 3                    /* 32-bit limbs per candidate (96-bit container) */

/* r = a*b*R^-1 mod n, result < n.  n0inv = -n^-1 mod 2^32.
   Ported from tools/bench_cios_mr.cu (validated there against GMP). */
__device__ __forceinline__ static void mont_mul(uint32_t *r, const uint32_t *a,
                                                const uint32_t *b,
                                                const uint32_t *n, uint32_t n0inv) {
    uint32_t t[NW + 2];
#pragma unroll
    for (int i = 0; i < NW + 2; i++) t[i] = 0U;

#pragma unroll
    for (int i = 0; i < NW; i++) {
        uint64_t c = 0;
        const uint64_t bi = (uint64_t)b[i];
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t p = (uint64_t)a[j] * bi + (uint64_t)t[j] + c;
            t[j] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s0 = (uint64_t)t[NW] + c;
        uint64_t s1 = (uint64_t)t[NW + 1] + (s0 >> 32);
        t[NW] = (uint32_t)s0;
        t[NW + 1] = (uint32_t)s1;

        uint32_t m = t[0] * n0inv;
        c = ((uint64_t)t[0] + (uint64_t)m * (uint64_t)n[0]) >> 32;
#pragma unroll
        for (int j = 1; j < NW; j++) {
            uint64_t p = (uint64_t)m * (uint64_t)n[j] + (uint64_t)t[j] + c;
            t[j - 1] = (uint32_t)p;
            c = p >> 32;
        }
        uint64_t s2 = (uint64_t)t[NW] + c;
        t[NW - 1] = (uint32_t)s2;
        t[NW] = (uint32_t)((uint64_t)t[NW + 1] + (s2 >> 32));
        t[NW + 1] = 0U;
    }

    uint32_t borrow = 0, sub[NW];
#pragma unroll
    for (int j = 0; j < NW; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (t[NW] != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < NW; j++) r[j] = need ? sub[j] : t[j];
}

/* r = 2*a mod n */
__device__ __forceinline__ static void dbl_mod(uint32_t *r, const uint32_t *a,
                                               const uint32_t *n) {
    uint32_t c = 0, t[NW];
#pragma unroll
    for (int j = 0; j < NW; j++) {
        uint64_t s = (uint64_t)a[j] * 2ULL + (uint64_t)c;
        t[j] = (uint32_t)s;
        c = (uint32_t)(s >> 32);
    }
    uint32_t borrow = 0, sub[NW];
#pragma unroll
    for (int j = 0; j < NW; j++) {
        uint64_t d = (uint64_t)t[j] - (uint64_t)n[j] - (uint64_t)borrow;
        sub[j] = (uint32_t)d;
        borrow = (uint32_t)((d >> 32) & 1ULL);
    }
    int need = (c != 0U) | (borrow == 0U);
#pragma unroll
    for (int j = 0; j < NW; j++) r[j] = need ? sub[j] : t[j];
}

/* base-2 strong probable prime: 0 = composite, 1 = probable prime.
   n-1 = 2^s * d.  y = 2^d in Montgomery form; then squarings looking for -1. */
__device__ static int mr_base2(const uint32_t *nin) {
    uint32_t n[NW], one[NW], neg[NW], two[NW], y[NW];
    uint32_t n0inv = 1U;
#pragma unroll
    for (int k = 0; k < 5; k++) n0inv = n0inv * (2U - nin[0] * n0inv);
    n0inv = (uint32_t)(0U - n0inv);
#pragma unroll
    for (int j = 0; j < NW; j++) n[j] = nin[j];

    /* b = bit length of n */
    int b = 0;
#pragma unroll
    for (int j = NW - 1; j >= 0; j--) {
        if (b == 0 && n[j] != 0U) b = j * 32 + (32 - __clz(n[j]));
    }

    /* ONE = 2^96 mod n.  q = floor(2^96/n) comes from a double-precision
       estimate that is within +-1 of the true quotient (2^96/n < 2^29 for
       n > 2^67; double rounding contributes < 2^-23 absolute); the 96-bit
       residual (0 - q*n) mod 2^96 is then corrected with one add or subtract
       of n.  Replaces (96-b) serial doublings (~300 instructions) with a
       division plus ~30 integer ops. */
    {
        double dn = (double)n[2];
        dn = dn * 4294967296.0 + (double)n[1];
        dn = dn * 4294967296.0 + (double)n[0];
        uint64_t q = (uint64_t)(79228162514264337593543950336.0 / dn);
        uint32_t qn[NW];
        uint64_t cy = 0;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t p = (uint64_t)q * (uint64_t)n[j] + cy;
            qn[j] = (uint32_t)p;
            cy = p >> 32;
        }
        uint32_t ovf = (uint32_t)cy;          /* q*n >= 2^96 */
        uint32_t bor = 0;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t d64 = 0ULL - (uint64_t)qn[j] - (uint64_t)bor;
            one[j] = (uint32_t)d64;
            bor = (uint32_t)((d64 >> 32) & 1ULL);
        }
        if (ovf) {
            uint32_t c2 = 0;                  /* overshoot: one += n (mod 2^96) */
#pragma unroll
            for (int j = 0; j < NW; j++) {
                uint64_t s64 = (uint64_t)one[j] + (uint64_t)n[j] + (uint64_t)c2;
                one[j] = (uint32_t)s64;
                c2 = (uint32_t)(s64 >> 32);
            }
        } else {
            uint32_t b2 = 0, sub[NW];         /* residual may be >= n */
#pragma unroll
            for (int j = 0; j < NW; j++) {
                uint64_t d64 = (uint64_t)one[j] - (uint64_t)n[j] - (uint64_t)b2;
                sub[j] = (uint32_t)d64;
                b2 = (uint32_t)((d64 >> 32) & 1ULL);
            }
            if (!b2) {
#pragma unroll
                for (int j = 0; j < NW; j++) one[j] = sub[j];
            }
        }
    }
    {
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t d = (uint64_t)n[j] - (uint64_t)one[j] - (uint64_t)borrow;
            neg[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
    dbl_mod(two, one, n);

    /* d = (n-1) >> s */
    uint32_t nm1[NW];
    {
        uint64_t d = (uint64_t)n[0] - 1ULL;
        nm1[0] = (uint32_t)d;
        uint32_t borrow = (uint32_t)((d >> 32) & 1ULL);
#pragma unroll
        for (int j = 1; j < NW; j++) {
            uint64_t dd = (uint64_t)n[j] - (uint64_t)borrow;
            nm1[j] = (uint32_t)dd;
            borrow = (uint32_t)((dd >> 32) & 1ULL);
        }
    }
    int s = 0;
    {
        int j = 0;
        while (j < NW && nm1[j] == 0U) { s += 32; j++; }
        if (j < NW) {
            uint32_t w = nm1[j];
            while ((w & 1U) == 0U) { w >>= 1; s++; }
        }
    }
    uint32_t dsh[NW];
    {
        int sh = s & 31, wi = s >> 5;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint32_t lo = (j + wi < NW) ? nm1[j + wi] : 0U;
            uint32_t hi = (j + wi + 1 < NW) ? nm1[j + wi + 1] : 0U;
            dsh[j] = (sh == 0) ? lo : (uint32_t)((lo >> sh) | (hi << (32 - sh)));
        }
    }
    int dbits = b - s;
    while (dbits > 0 && ((dsh[(dbits - 1) >> 5] >> ((dbits - 1) & 31)) & 1U) == 0U)
        dbits--;

#pragma unroll 1
    for (int j = 0; j < NW; j++) y[j] = two[j];
    /* exponent bits walk a left-shifting 128-bit register: the MSB sits at
       bit 127, so each iteration is one test + one shift.  A dynamically
       indexed dsh[e>>5] register array would compile to predicated selects
       on every iteration of the Montgomery chain.
       The register is two 64-bit words instead of `unsigned __int128`:
       MSVC (the Windows host compiler nvcc requires) has no 128-bit type,
       and nvcc's host pass parses this file.  The 96-bit dsh value only
       ever needs "shift left" and "test bit 127", so the pair is exact. */
    {
    uint64_t xlo = (uint64_t)dsh[0] | ((uint64_t)dsh[1] << 32);
    uint64_t xhi = (uint64_t)dsh[2];
    /* x <<= (128 - dbits + 1): two shifts, any amount (shl128 helper) */
    int sh = 128 - dbits + 1;
    if (sh >= 128) { xlo = 0; xhi = 0; }
    else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
    else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
    for (int e = dbits - 2; e >= 0; e--) {
        /* in place: mont_mul reads a[]/b[] before writing r[] (r may alias) */
        mont_mul(y, y, y, n, n0inv);
        if ((uint32_t)(xhi >> 63)) dbl_mod(y, y, n);
        xhi = (xhi << 1) | (xlo >> 63);
        xlo <<= 1;
    }
    }
    int is_one = 1, is_neg = 1;
#pragma unroll
    for (int j = 0; j < NW; j++) {
        if (y[j] != one[j]) is_one = 0;
        if (y[j] != neg[j]) is_neg = 0;
    }
    if (is_one || is_neg) return 1;
    for (int r = 0; r < s - 1; r++) {
        mont_mul(y, y, y, n, n0inv);
        int got_neg = 1, got_one = 1;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            if (y[j] != neg[j]) got_neg = 0;
            if (y[j] != one[j]) got_one = 0;
        }
        if (got_neg) return 1;
        if (got_one) return 0;
    }
    return 0;
}

__global__ static void mr68_kernel(const uint32_t * __restrict__ cands,
                                   uint8_t * __restrict__ results, uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t *n = cands + (size_t)i * NW;
    results[i] = (uint8_t)mr_base2(n);
}

/* packed variant: candidates are (base3, u32 step) with n = base + 2*step;
   the value is rebuilt in registers - no intermediate 12 B/candidate buffer
   and no separate pack launch. */
__global__ static void mr68_kernel_packed(const uint32_t * __restrict__ base3,
                                          const uint32_t * __restrict__ steps,
                                          uint8_t * __restrict__ results,
                                          uint32_t count) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t d = (uint64_t)steps[i] << 1;
    uint64_t lo = (uint64_t)base3[0] | ((uint64_t)base3[1] << 32);
    uint64_t lo2 = lo + d;
    uint32_t n[NW];
    n[0] = (uint32_t)lo2;
    n[1] = (uint32_t)(lo2 >> 32);
    n[2] = base3[2] + (uint32_t)(lo2 < lo);
    results[i] = (uint8_t)mr_base2(n);
}

/* compact the survivor set of a masked bitmap word into offs[]: each thread
   emits the zero-bit positions of its word; one shared reservation gives every
   thread a unique local slot, one global atomic per block reserves the block's
   slice, so the offsets are dense but unordered ACROSS blocks (the MR stage is
   per-candidate independent - order does not matter).  cntp accumulates the
   total candidate count of the launch. */
__global__ static void p0_compact_kernel(const uint64_t * __restrict__ bm,
                                         uint64_t nwords, uint64_t last_mask,
                                         uint32_t * __restrict__ offs,
                                         uint32_t * __restrict__ cntp) {
    __shared__ uint32_t sbase;
    if (threadIdx.x == 0) sbase = 0;
    __syncthreads();
    uint64_t wi = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int active = (wi < nwords);
    uint64_t w = 0;
    if (active) {
        w = ~bm[wi];
        if (wi == nwords - 1) w &= last_mask;
    }
    uint32_t n = active ? (uint32_t)__popcll((unsigned long long)w) : 0U;
    uint32_t local = atomicAdd(&sbase, n);
    __syncthreads();
    __shared__ uint32_t gbase;
    if (threadIdx.x == 0) gbase = atomicAdd(cntp, sbase);
    __syncthreads();
    if (!active) return;
    uint32_t pos = gbase + local;
    uint64_t base = wi << 6;
    while (w) {
        int b = __ffsll((long long)w) - 1;
        w &= w - 1;
        offs[pos++] = (uint32_t)(base + (uint64_t)b);
    }
}

/* MR stage fed by the compacted offsets: ONE THREAD PER CANDIDATE (full warp
   efficiency for the long Montgomery chains; the bitmap-decoded word variant
   idled lanes on its variable trip counts).  Candidate = v0 + 2*offs[i];
   probable primes set their bit in the verdict bitmap (the only D2H).  A
   grid-stride loop makes the launch correct for any candidate count. */
__global__ static void mr68_from_offsets(uint64_t v0_lo, uint64_t v0_hi,
                                         const uint32_t * __restrict__ offs,
                                         const uint32_t * __restrict__ cntp,
                                         uint64_t * __restrict__ out) {
    const uint32_t cnt = *cntp;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < cnt; i += stride) {
        uint32_t off = offs[i];
        uint64_t lo2 = v0_lo + ((uint64_t)off << 1);
        uint32_t n[NW];
        n[0] = (uint32_t)lo2;
        n[1] = (uint32_t)(lo2 >> 32);
        n[2] = (uint32_t)v0_hi + (uint32_t)(lo2 < v0_lo);
        if (mr_base2(n))
            atomicOr((unsigned long long *)&out[off >> 6], 1ULL << (off & 63u));
    }
}

#endif /* MR68_KERNEL_CUH */
