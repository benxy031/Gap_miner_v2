/*
 * class30.cu - class-30 (mod 30) bitmap sieve prototype + CPU arbiter gate.
 *
 * Motivation: in the odd-slot layout every prime p marks one bit per odd
 * multiple (536M/p marks per 2^30 block); primes 3 and 5 mark at all.  In a
 * class-30 layout only the numbers coprime to 30 (8 per 30) exist as bits, so
 *   3, 5  : mark nothing (structural)
 *   p >= 7: mark density 1/p among the slots (vs 1/p among 2x fewer... see doc)
 * -> total marks 1.18e9 -> ~0.44e9 per 2^30 numbers, and the current
 *    atomicOr-bound sieve cost should drop roughly in proportion.
 *
 * geometry:
 *   CLS[8] = {1,7,11,13,17,19,23,29}, A = largest multiple of 30 <= v0
 *   slot s  <->  value = A + 30*(s>>3) + CLS[s&7]
 *   tile = 2^19 slots (64 KB shared), block = ntiles * tileSlots slots.
 *
 * item = (prime, chunk of 64 marks).  Key identities used by the kernel:
 *   64 marks = 240 in the multiple-index m (8 coprimes per 30) =>
 *     value of chunk k0  = v_start + 240*p*k0
 *     slot  of chunk k0  = pos_start + 64*p*k0      (240*p/30*8 = 64*p)
 *   the 8 slot steps from mark j to j+1 cycle with period 8.
 *
 * usage: class30 START N_NUMBERS P [tile_bits] [grid] [block]
 *   env VERIFY=1 -> CPU arbiter compares every slot bit vs true composite flag.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

typedef unsigned __int128 u128;

#define MAXSH (99 * 1024 / 8)

__device__ __constant__ int8_t c_clsidx[30];
__device__ __constant__ uint32_t c_cls[8];
__device__ __constant__ uint32_t c_perm[30];      /* allowed VALUE classes of multiples */
__device__ __constant__ uint32_t c_pinv[30];      /* inverse of pm30 mod 30 */

/* fastmod64 (same core as the engine's t_fastmod64) */
__device__ __host__ static inline uint64_t t_fastmod64(uint64_t a, uint64_t p, uint64_t invp) {
    uint64_t q = ((u128)a * (u128)invp) >> 64;
    uint64_t r = a - q * p;
    while (r >= p) r -= p;
    return r;
}

/* ---------------- device ---------------- */
__global__ void class30_sieve_kernel(const uint64_t * __restrict__ primes,
                                     const uint64_t * __restrict__ invp,
                                     const uint64_t * __restrict__ r64,
                                     const uint32_t * __restrict__ itemPidx,
                                     const uint32_t * __restrict__ itemK0,
                                     uint32_t nitems,
                                     uint64_t Alo, uint64_t Ahi,
                                     uint64_t tileSlots, uint64_t ntiles,
                                     uint64_t * __restrict__ g_bm,
                                     uint32_t tileWords,
                                     int ab_nomark, int ab_noitem,
                                     const uint32_t * __restrict__ wpidx, int wg) {
    extern __shared__ uint64_t sh[];
    __shared__ uint64_t wpat1[127];       /* group 1: {7,11,13},  period 8008  */
    __shared__ uint64_t wpat2[932];       /* group 2: {17,19,23}, period 59432 */
    __shared__ uint32_t wpj[48];          /* 6 primes x 8 AP base slots */
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        /* phase 1: zero sh + zero wheel patterns + wheel-prime setup (disjoint) */
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0;
        if (wg) {
            for (uint32_t w = threadIdx.x; w < (wg >= 2 ? 1059u : 127u); w += blockDim.x) {
                if (w < 127u) wpat1[w] = 0ull;
                else if (w >= 128u) wpat2[w - 128u] = 0ull;
            }
        }
        /* tile's first slot value */
        uint64_t slotBase = tile * tileSlots;              /* multiple of 8 */
        u128 tv = ((u128)Ahi << 64 | Alo) + (u128)(slotBase >> 3) * 30u + 1u;
        uint64_t t_lo = (uint64_t)tv, t_hi = (uint64_t)(tv >> 64);
        /* wheel-prime setup: first mark slot + the 8 AP base slots */
        if (wg && threadIdx.x < 6u) {
            if (wg == 1 && threadIdx.x >= 3u) { }
            else {
            int pi = (int)wpidx[threadIdx.x];
            uint64_t p = primes[pi], pv = invp[pi];
            uint64_t vh = t_fastmod64(t_hi, p, pv);
            uint64_t vl = t_fastmod64(t_lo, p, pv);
            uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
            uint64_t d = vm ? (p - vm) : 0;
            uint64_t phi1 = 1u + d;
            uint64_t r0 = phi1 % 30u;
            uint64_t pm30 = p % 30u;
            uint32_t mask = c_perm[pm30];
            uint64_t kadj = 0;
            while (!((mask >> ((r0 + kadj * pm30) % 30u)) & 1u)) kadj++;
            uint64_t phi = phi1 + kadj * p;
            uint64_t r = phi % 30u;
            uint64_t pos = (phi / 30u) * 8u + (uint64_t)c_clsidx[r];
            uint32_t step[8];
            uint64_t j = (uint64_t)c_clsidx[(r * c_pinv[pm30]) % 30u];
            uint64_t ph = phi;
            for (int s = 0; s < 8; s++) {
                uint64_t j2 = (j + 1u) & 7u;
                uint64_t dm = (uint64_t)c_cls[j2] - (uint64_t)c_cls[j];
                if (j2 == 0u) dm += 30u;
                uint64_t ph2 = ph + dm * p;
                step[s] = (uint32_t)((ph2 / 30u) * 8u + (uint64_t)c_clsidx[ph2 % 30u]
                                   - ((ph / 30u) * 8u + (uint64_t)c_clsidx[ph % 30u]));
                ph = ph2; j = j2;
            }
            uint32_t cum = 0;
            for (int s = 0; s < 8; s++) { wpj[threadIdx.x * 8 + s] = (uint32_t)pos + cum; cum += step[s]; }
            }
        }
        __syncthreads();
        if (!ab_noitem) {
        /* contiguous per-thread runs over the prime-major item table; the
           per-prime setup (fastmods + bump + the 8 slot steps) is reused for
           every chunk of the same prime (chunks differ only by pos += 64p) */
        uint32_t nthreads = blockDim.x;
        uint32_t per = nitems / nthreads, rem = nitems - per * nthreads;
        uint32_t it0 = threadIdx.x * per + (threadIdx.x < rem ? threadIdx.x : rem);
        uint32_t itn = per + (threadIdx.x < rem ? 1u : 0u);
        int cur_pi = -1;
        uint32_t cur_p = 0;
        uint32_t pos_base = 0;
        uint32_t step[8];
        for (uint32_t it = it0; it < it0 + itn; it++) {
            int pi = (int)itemPidx[it];
            uint32_t k0 = itemK0[it];
            if (pi != cur_pi) {
                cur_pi = pi;
                uint64_t p = primes[pi];
                cur_p = (uint32_t)p;
                uint64_t pv = invp[pi];
                uint64_t vh = t_fastmod64(t_hi, p, pv);
                uint64_t vl = t_fastmod64(t_lo, p, pv);
                uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
                uint64_t d = vm ? (p - vm) : 0;
                uint32_t phi1 = (uint32_t)(1u + d);
                uint32_t r0 = phi1 % 30u;
                uint32_t pm30 = (uint32_t)(p % 30u);
                uint32_t mask = c_perm[pm30];
                uint32_t kadj = 0;
                while (!((mask >> ((r0 + kadj * pm30) % 30u)) & 1u)) kadj++;
                uint32_t phi = phi1 + kadj * cur_p;
                uint32_t r = phi % 30u;
                pos_base = (phi / 30u) * 8u + (uint32_t)c_clsidx[r];
                uint32_t j = (uint32_t)c_clsidx[(r * c_pinv[pm30]) % 30u];
                uint32_t ph = phi;
                for (int st = 0; st < 8; st++) {
                    uint32_t j2 = (j + 1u) & 7u;
                    uint32_t dm = (uint32_t)c_cls[j2] - (uint32_t)c_cls[j];
                    if (j2 == 0u) dm += 30u;
                    uint32_t ph2 = ph + dm * cur_p;
                    step[st] = (ph2 / 30u) * 8u + (uint32_t)c_clsidx[ph2 % 30u]
                             - ((ph / 30u) * 8u + (uint32_t)c_clsidx[ph % 30u]);
                    ph = ph2; j = j2;
                }
            }
            uint32_t p32 = pos_base + 64u * cur_p * k0;
            if (p32 >= (uint32_t)tileSlots) continue;
            if (!ab_nomark) {
                for (int r8 = 0; r8 < 8; r8++) {
#pragma unroll
                    for (int st = 0; st < 8; st++) {
                        if (p32 < (uint32_t)tileSlots)
                            atomicOr(((unsigned int *)sh) + (p32 >> 5), 1u << (p32 & 31u));
                        p32 += step[st];
                    }
                }
            }
        }
        }
        if (wg) {
            /* phase 2 (cont.): wheel lay writes only wpat, disjoint from sh above */
            const uint64_t P1 = 8008ull, A1 = 8128ull;
            const uint64_t P2 = 59432ull, A2 = 59496ull;
            const uint64_t PGS[2] = {P1, P2};
            const uint64_t ASZ[2] = {A1, A2};
            for (int g = 0; g < (wg >= 2 ? 2 : 1); g++) {
                for (int wi = 0; wi < 3; wi++) {
                    uint64_t p = (g == 0) ? ((wi == 0) ? 7u : (wi == 1) ? 11u : 13u)
                                          : ((wi == 0) ? 17u : (wi == 1) ? 19u : 23u);
                    uint64_t stride8 = 8u * p;
                    uint64_t nt = ASZ[g] / stride8 + 2u;
                    for (uint64_t idx = threadIdx.x; idx < 8u * nt; idx += blockDim.x) {
                        uint64_t j = idx & 7u, t = idx >> 3;
                        uint64_t pos = (uint64_t)wpj[(g * 3 + wi) * 8 + (int)j] + t * stride8;
                        if (pos >= ASZ[g]) continue;
                        uint64_t q = pos % PGS[g];
                        unsigned long long *pp = (unsigned long long *)((g == 0) ? wpat1 : wpat2);
                        atomicOr(&pp[q >> 6], 1ull << (q & 63u));
                        if (q < 64u) atomicOr(&pp[(q + PGS[g]) >> 6], 1ull << ((q + (PGS[g] & 63u)) & 63u));
                    }
                }
            }
        }
        __syncthreads();
        /* phase 3: fused wheel-OR + flush (single global write per word) */
        {
            uint64_t *out = g_bm + tile * tileWords;
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) {
                uint64_t val = sh[w];
                if (wg) {
                    uint64_t idx = ((uint64_t)64u * w) % 8008ull;
                    uint32_t k = (uint32_t)(idx >> 6), r2 = (uint32_t)(idx & 63u);
                    val |= (wpat1[k] >> r2) | (r2 ? (wpat1[k + 1] << (64 - r2)) : 0ull);
                    if (wg >= 2) {
                        uint64_t idx2 = ((uint64_t)64u * w) % 59432ull;
                        uint32_t k2 = (uint32_t)(idx2 >> 6), r22 = (uint32_t)(idx2 & 63u);
                        val |= (wpat2[k2] >> r22) | (r22 ? (wpat2[k2 + 1] << (64 - r22)) : 0ull);
                    }
                }
                out[w] = val;
            }
        }
        __syncthreads();
    }
}

/* ---------------- host ---------------- */
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int8_t H_CLSIDX[30];
static uint32_t H_CLS[8] = {1, 7, 11, 13, 17, 19, 23, 29};

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: class30 START N P [tile_bits] [grid] [block]\n"); return 2; }
    u128 start = 0;
    for (const char *s = argv[1]; *s; s++) start = start * 10 + (u128)(*s - '0');
    uint64_t N = strtoull(argv[2], NULL, 10);
    uint32_t P = (uint32_t)strtoul(argv[3], NULL, 10);
    int tile_bits = (argc > 4) ? atoi(argv[4]) : 19;          /* slots per tile */
    int grid = (argc > 5) ? atoi(argv[5]) : 92;
    int block = (argc > 6) ? atoi(argv[6]) : 512;
    uint64_t tileSlots = 1ULL << tile_bits;
    uint32_t tileWords = (uint32_t)(tileSlots >> 6);
    if (tileWords > MAXSH) { fprintf(stderr, "tile too big\n"); return 2; }
    /* A = largest multiple of 30 <= v0 ; v0 = start|1 */
    u128 v0 = start | 1;
    u128 A = (v0 / 30u) * 30u;
    uint64_t slot0 = 0;                                        /* slot of A+1 */
    uint64_t frames = (N / 30u) + 1;
    uint64_t slotsPerBlock = frames * 8u;
    uint64_t ntiles = (slotsPerBlock + tileSlots - 1) / tileSlots;
    for (int i = 0; i < 30; i++) H_CLSIDX[i] = -1;
    for (int i = 0; i < 8; i++) H_CLSIDX[H_CLS[i]] = (int8_t)i;
    cudaMemcpyToSymbol(c_clsidx, H_CLSIDX, sizeof(H_CLSIDX));
    cudaMemcpyToSymbol(c_cls, H_CLS, sizeof(H_CLS));
    {
        uint32_t perm[30] = {0}, pinv[30] = {0};
        for (int pm = 0; pm < 30; pm++) {
            int g = 1; for (int d = 2; d <= pm && d <= 30; d++) if (pm % d == 0 && 30 % d == 0) g = d;
            if (pm % 3 == 0 || pm % 5 == 0 || pm % 2 == 0) continue;   /* not coprime */
            uint32_t m = 0;
            for (int j = 0; j < 8; j++) m |= 1u << ((pm * (int)H_CLS[j]) % 30);
            perm[pm] = m;
            for (int x = 1; x < 30; x++) if ((pm * x) % 30 == 1) pinv[pm] = (uint32_t)x;
            (void)g;
        }
        cudaMemcpyToSymbol(c_perm, perm, sizeof(perm));
        cudaMemcpyToSymbol(c_pinv, pinv, sizeof(pinv));
    }

    /* primes 7..P */
    uint8_t *sv = (uint8_t *)calloc((size_t)P + 1, 1);
    for (uint64_t i = 4; i <= P; i += 2) sv[i] = 1;
    for (uint64_t i = 3; i <= P; i += 2)
        if (!sv[i] && i * i <= (uint64_t)P) for (uint64_t j = i * i; j <= (uint64_t)P; j += i) sv[j] = 1;
    size_t np = 0;
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * (P / 2 + 64));
    for (uint64_t i = 7; i <= (uint64_t)P; i += 2)
        if (!sv[i]) hp[np++] = i;                              /* skip 2,3,5 */
    uint64_t *h_iv = (uint64_t *)malloc(np * 8), *h_r = (uint64_t *)malloc(np * 8);
    for (size_t i = 0; i < np; i++) { h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]); h_iv[i] = ~0ULL / hp[i]; }

    /* items: per prime, chunks of 64 marks covering one tile.
       marks per tile = tileSlots/p  (1/p of the tile's coprime values) */
    uint64_t tileVals = (tileSlots / 8u) * 30u;                /* values per tile */
    int wheel_on = getenv("WHEEL23") ? 2 : (getenv("WHEEL713") ? 1 : 0);
    /* SKIP_PMIN: TIMING PROBE ONLY - skips all primes < PMIN in the item table
       (bitmap becomes wrong; measures the atomicOr budget by prime range). */
    uint32_t pmin = getenv("SKIP_PMIN") ? (uint32_t)strtoul(getenv("SKIP_PMIN"), NULL, 10) : 0u;
    size_t icap = 0;
    for (size_t i = 0; i < np; i++) {
        if (wheel_on && hp[i] <= (wheel_on >= 2 ? 23u : 13u)) continue;
        if (hp[i] < pmin) continue;
        icap += (size_t)((tileSlots / hp[i] + 2 + 63) / 64);
    }
    uint32_t *h_ip = (uint32_t *)malloc(icap * 4), *h_ik = (uint32_t *)malloc(icap * 4);
    size_t nii = 0;
    for (size_t i = 0; i < np; i++) {
        if (wheel_on && hp[i] <= (wheel_on >= 2 ? 23u : 13u)) continue;
        if (hp[i] < pmin) continue;
        uint64_t marks = tileSlots / hp[i] + 2;
        uint32_t chunks = (uint32_t)((marks + 63) / 64);
        for (uint32_t c = 0; c < chunks; c++) { h_ip[nii] = (uint32_t)i; h_ik[nii] = c; nii++; }
    }
    uint32_t h_wpidx[6] = {0, 0, 0, 0, 0, 0};
    const uint64_t WP6[6] = {7, 11, 13, 17, 19, 23};
    for (size_t i = 0; i < np; i++)
        for (int wq = 0; wq < 6; wq++) if (hp[i] == WP6[wq]) h_wpidx[wq] = (uint32_t)i;
    printf("[c30] N=%llu P=%u primes=%zu items=%zu tiles=%llu tileSlots=%llu wheel=%d\n",
           (unsigned long long)N, P, np, nii, (unsigned long long)ntiles,
           (unsigned long long)tileSlots, wheel_on);

    uint64_t *d_p, *d_iv, *d_r, *d_bm = NULL;
    uint32_t *d_ip, *d_ik;
    cudaMalloc(&d_p, np * 8); cudaMalloc(&d_iv, np * 8); cudaMalloc(&d_r, np * 8);
    cudaMalloc(&d_ip, nii * 4); cudaMalloc(&d_ik, nii * 4);
    size_t bmBytes = (size_t)ntiles * tileWords * 8;
    cudaMalloc(&d_bm, bmBytes);
    cudaMemcpy(d_p, hp, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip, h_ip, nii * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ik, h_ik, nii * 4, cudaMemcpyHostToDevice);
    uint32_t *d_wpidx; cudaMalloc(&d_wpidx, 6 * 4);
    cudaMemcpy(d_wpidx, h_wpidx, 6 * 4, cudaMemcpyHostToDevice);

    size_t shbytes = (size_t)tileWords * 8;
    cudaFuncSetAttribute(class30_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
    class30_sieve_kernel<<<grid, block, shbytes>>>(d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
        (uint64_t)A, (uint64_t)(A >> 64), tileSlots, ntiles, d_bm, tileWords, getenv("SIEVE_NOMARK") != NULL, getenv("SIEVE_NOITEM") != NULL, d_wpidx, wheel_on);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[c30] %s\n", cudaGetErrorString(e)); return 1; }
    double t0 = now_s();
    for (int r = 0; r < 3; r++)
        class30_sieve_kernel<<<grid, block, shbytes>>>(d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
            (uint64_t)A, (uint64_t)(A >> 64), tileSlots, ntiles, d_bm, tileWords, getenv("SIEVE_NOMARK") != NULL, getenv("SIEVE_NOITEM") != NULL, d_wpidx, wheel_on);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[c30] %s\n", cudaGetErrorString(e)); return 1; }
    double t = (now_s() - t0) / 3.0;
    printf("[c30] sieve=%.4f s -> %.1f B/s (numbers)\n", t, (double)N / t / 1e9);

    if (getenv("VERIFY")) {
        size_t nslots = (size_t)ntiles * tileSlots;
        uint64_t *hb = (uint64_t *)malloc(bmBytes);
        cudaMemcpy(hb, d_bm, bmBytes, cudaMemcpyDeviceToHost);
        /* CPU arbiter in VALUE space: true composite flag per slot */
        uint8_t *ref = (uint8_t *)calloc(nslots, 1);
        for (size_t i = 0; i < np; i++) {
            uint64_t p = hp[i];
            /* first odd multiple of p >= A+1 */
            u128 first = ((A + 1 + p - 1) / p) * p;
            if ((first & 1) == 0) first += p;
            for (u128 v = first; v < A + 1 + (u128)frames * 30u; v += 2 * (u128)p) {
                u128 off = v - A;
                if (off < 1) continue;
                uint64_t f = (uint64_t)(off / 30u);
                int ci = H_CLSIDX[(int)(off % 30u)];
                if (ci < 0) continue;
                size_t s = (size_t)f * 8u + ci;
                if (s < nslots) ref[s] = 1;
            }
        }
        uint64_t nfalse = 0, nmiss = 0, shown = 0;
        size_t realSlots = (size_t)frames * 8u;      /* marks beyond are tile overshoot */
        for (size_t s = 0; s < realSlots; s++) {
            int marked = (int)((hb[s >> 6] >> (s & 63)) & 1ull);
            if (marked != (int)ref[s]) {
                if (marked) nfalse++; else nmiss++;
                if (shown < 8) {
                    u128 v = A + (u128)(s >> 3) * 30u + H_CLS[s & 7u];
                    char buf[48]; int k = 0; u128 tv2 = v;
                    while (tv2) { buf[k++] = (char)('0' + (int)(tv2 % 10)); tv2 /= 10; }
                    printf("[c30] %s slot=%zu v=", marked ? "FALSE_MARK" : "MISS", s);
                    while (k--) putchar(buf[k]);
                    printf("\n");
                    shown++;
                }
            }
        }
        printf("[c30] verify: false=%llu miss=%llu of %zu slots (real %zu)\n",
               (unsigned long long)nfalse, (unsigned long long)nmiss, realSlots, realSlots);
        free(ref); free(hb);
    }
    return 0;
}
