/* bench_mr_variant.cu - measure what one phase0 MR test really costs, and the
   cheapest legitimate ways to cut it, on real 96-bit candidates.

   Arms (all base-2 Miller-Rabin, identical strength):
     A) NW=3 (96-bit)  - the current mr68 container
     B) NW=2 (64-bit)  - exact for n < 2^64 (our walk is 96-bit, so this arm is
        only a cost probe, NOT a drop-in)
     C) NW=3 (96-bit)  + an extra small-prime screen inside the kernel
        (reject multiples of p<64 before the Fermat chain)

   Build:  nvcc -O3 -arch=sm_86 tools/bench_mr_variant.cu -o bin/bench_mr_variant
   Run:    bin/bench_mr_variant [N]
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ----------------------- NW-limb CIOS Montgomery ----------------------- */
template <int NW>
__device__ __forceinline__ void mont_mul(uint32_t *r, const uint32_t *a,
                                         const uint32_t *b, const uint32_t *n,
                                         uint32_t n0inv) {
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

template <int NW>
__device__ __forceinline__ void dbl_mod(uint32_t *r, const uint32_t *a,
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

/* ---------------- base-2 MR, parametrized on limb count ---------------- */
template <int NW>
__device__ static int mr_base2_generic(const uint32_t *nin) {
    uint32_t n[NW], one[NW], neg[NW], two[NW], y[NW];
    uint32_t n0inv = 1U;
#pragma unroll
    for (int k = 0; k < 5; k++) n0inv = n0inv * (2U - nin[0] * n0inv);
    n0inv = (uint32_t)(0U - n0inv);
#pragma unroll
    for (int j = 0; j < NW; j++) n[j] = nin[j];

    int b = 0;
#pragma unroll
    for (int j = NW - 1; j >= 0; j--)
        if (b == 0 && n[j] != 0U) b = j * 32 + (32 - __clz(n[j]));

    /* ONE = 2^(32*NW) mod n.  Only the large-n path is needed here
       (we benchmark real ~96-bit candidates). */
    double dn = 0;
#pragma unroll
    for (int j = NW - 1; j >= 0; j--) dn = dn * 4294967296.0 + (double)n[j];
    const double twoPow = (NW == 3) ? 79228162514264337593543950336.0
                                    : 18446744073709551616.0;
    uint64_t q = (uint64_t)(twoPow / dn);
    uint32_t qn[NW];
    uint64_t cy = 0;
#pragma unroll
    for (int j = 0; j < NW; j++) {
        uint64_t p = (uint64_t)q * (uint64_t)n[j] + cy;
        qn[j] = (uint32_t)p;
        cy = p >> 32;
    }
    uint32_t ovf = (uint32_t)cy;
    uint32_t bor = 0;
#pragma unroll
    for (int j = 0; j < NW; j++) {
        uint64_t d64 = 0ULL - (uint64_t)qn[j] - (uint64_t)bor;
        one[j] = (uint32_t)d64;
        bor = (uint32_t)((d64 >> 32) & 1ULL);
    }
    if (ovf) {
        uint32_t c2 = 0;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t s64 = (uint64_t)one[j] + (uint64_t)n[j] + (uint64_t)c2;
            one[j] = (uint32_t)s64;
            c2 = (uint32_t)(s64 >> 32);
        }
    } else {
        uint32_t b2 = 0, sub[NW];
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
    {
        uint32_t borrow = 0;
#pragma unroll
        for (int j = 0; j < NW; j++) {
            uint64_t d = (uint64_t)n[j] - (uint64_t)one[j] - (uint64_t)borrow;
            neg[j] = (uint32_t)d;
            borrow = (uint32_t)((d >> 32) & 1ULL);
        }
    }
    dbl_mod<NW>(two, one, n);

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
        if (j < NW) { uint32_t w = nm1[j]; while ((w & 1U) == 0U) { w >>= 1; s++; } }
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
    {
        uint64_t xlo = (uint64_t)dsh[0] | ((uint64_t)(NW > 1 ? dsh[1] : 0) << 32);
        uint64_t xhi = (NW > 2) ? (uint64_t)dsh[2] : 0;
        int sh = 128 - dbits + 1;
        if (sh >= 128) { xlo = 0; xhi = 0; }
        else if (sh >= 64) { xhi = xlo << (sh - 64); xlo = 0; }
        else if (sh > 0) { xhi = (xhi << sh) | (xlo >> (64 - sh)); xlo <<= sh; }
        for (int e = dbits - 2; e >= 0; e--) {
            mont_mul<NW>(y, y, y, n, n0inv);
            if ((uint32_t)(xhi >> 63)) dbl_mod<NW>(y, y, n);
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
        mont_mul<NW>(y, y, y, n, n0inv);
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

/* ------------------- kernels (one thread per candidate) ------------------- */
__global__ static void k_nw3(const uint32_t *cands, uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = (uint8_t)mr_base2_generic<3>(cands + (size_t)i * 3);
}
__global__ static void k_nw2(const uint32_t *cands, uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = (uint8_t)mr_base2_generic<2>(cands + (size_t)i * 3); /* low 64 bits */
}
/* small-prime screen, then NW=3 */
__global__ static void k_screen3(const uint32_t *cands, uint8_t *res, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t *c = cands + (size_t)i * 3;
    /* Correct for a 96-bit n: compute n mod p over ALL limbs
       (r = (r * 2^32 + limb) mod p), and never reject n == p itself. */
    uint32_t top = c[2];
    static const uint32_t P[] = {3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61};
#pragma unroll
    for (int k = 0; k < 17; k++) {
        uint32_t p = P[k];
        if (top == 0u && c[1] == 0u && c[0] == p) continue;  /* n == p: prime */
        uint32_t r = (uint32_t)(((uint64_t)top << 32 | c[1]) % p);
        r = (uint32_t)((((uint64_t)r << 32) | c[0]) % p);
        if (r == 0u) { res[i] = 0; return; }
    }
    res[i] = (uint8_t)mr_base2_generic<3>(c);
}

/* ----------------------------- host ----------------------------- */
static float bench(void (*k)(const uint32_t *, uint8_t *, uint32_t),
                   const uint32_t *dc, uint8_t *dr, uint32_t n) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<(n + 255) / 256, 256>>>(dc, dr, n);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; r++) k<<<(n + 255) / 256, 256>>>(dc, dr, n);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return ms / 5.0f;
}

int main(int argc, char **argv) {
    uint32_t n = (argc > 1) ? (uint32_t)strtoul(argv[1], 0, 10) : (1u << 22);
    uint32_t *h = (uint32_t *)malloc((size_t)n * 3 * 4);
    srand(12345);
    for (uint32_t i = 0; i < n; i++) {
        /* real ~96-bit odd candidates */
        uint32_t a = 0x40000000u | (uint32_t)rand();      /* bit 62-63 set  */
        uint32_t bb = (uint32_t)rand();
        uint32_t cc = ((uint32_t)rand() & ~1u);            /* will +1 below  */
        /* assemble an odd 96-bit value with top limb in [2^62, 2^63) */
        uint64_t v0 = (uint64_t)cc | 1u;
        uint64_t v1 = (uint64_t)bb;
        uint64_t v2 = (uint64_t)(a & 0xC0000000u);
        h[i * 3 + 0] = (uint32_t)v0;
        h[i * 3 + 1] = (uint32_t)(v1);
        h[i * 3 + 2] = (uint32_t)(v2);
    }
    uint32_t *dc; uint8_t *dr;
    cudaMalloc(&dc, (size_t)n * 3 * 4);
    cudaMalloc(&dr, n);
    cudaMemcpy(dc, h, (size_t)n * 3 * 4, cudaMemcpyHostToDevice);

    float ms3 = bench(k_nw3, dc, dr, n);
    float ms2 = bench(k_nw2, dc, dr, n);
    float mss = bench(k_screen3, dc, dr, n);

    printf("N=%u candidates (~96-bit)\n", n);
    printf("  NW=3 (current)      : %.3f ms  -> %.3e tests/s\n", ms3, n / (ms3 / 1e3));
    printf("  NW=2 (64-bit probe) : %.3f ms  -> %.3e tests/s  (%.2fx of NW3)\n",
           ms2, n / (ms2 / 1e3), ms3 / ms2);
    printf("  NW=3 + p<64 screen  : %.3f ms  -> %.3e tests/s  (%.2fx of NW3)\n",
           mss, n / (mss / 1e3), ms3 / mss);
    return 0;
}
