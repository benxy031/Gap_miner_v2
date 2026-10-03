/* tile_bench.cu - CTA-tile shared-memory sieve bench (the PGS-class sieve
   structure, our own implementation): each CTA owns a tile of the range,
   marks it into SHARED memory (shared atomicOr - no global atomics), then
   flushes the tile to global with plain stores (words are owned by the tile).

   Geometry: 1 bit per odd slot (our census bitmap format, all 30-classes).
   Primes 3..P_med marked with a per-prime strided loop (straggler warps are
   part of this v1 measurement; the production version would chunk).

   Usage: tile_bench START_DEC N_NUMBERS P_MED [tile_bits] [grid] [block]
   prints numbers/s for the sieve. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include "p0_mark.cuh"

typedef unsigned __int128 u128;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

__device__ static inline uint64_t t_fastmod64(uint64_t x, uint64_t p, uint64_t invp) {
    uint64_t q = __umul64hi(x, invp);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}

#define MAXSH 16384   /* u64 words of shared bitmap = 128KB max (opt-in) */

__global__ void tile_sieve_kernel(const uint64_t * __restrict__ primes,
                                  const uint64_t * __restrict__ invp,
                                  const uint64_t * __restrict__ r64,
                                  const uint32_t * __restrict__ itemPidx,
                                  const uint32_t * __restrict__ itemK0,
                                  uint32_t nitems,
                                  uint64_t v0_lo, uint64_t v0_hi,
                                  uint64_t tileSlots, uint64_t ntiles,
                                  uint64_t * __restrict__ g_bm,
                                  uint32_t tileWords,
                                  int ab_nozero, int ab_noflush, int ab_nomark, int ab_noitem,
                                  int wheel_on, uint64_t v0m3, uint64_t v0m5, uint64_t v0m7,
                                  uint64_t v0m11, uint64_t v0m13) {
    extern __shared__ uint64_t sh[];
    __shared__ uint64_t pat[236];
    __shared__ uint64_t rp[5];
    const int nozero = ab_nozero, noflush = ab_noflush, nomark = ab_nomark, noitem = ab_noitem;
    const uint64_t wp[5] = {3u, 5u, 7u, 11u, 13u};
    const uint64_t v0m[5] = {v0m3, v0m5, v0m7, v0m11, v0m13};
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        if (!nozero) { for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0; }
        __syncthreads();
        uint64_t off2 = tile * tileSlots * 2;
        u128 tv = ((u128)v0_hi << 64 | v0_lo) + (u128)off2;
        uint64_t t_lo = (uint64_t)tv; uint64_t t_hi = (uint64_t)(tv >> 64);
        if (!noitem)
        for (uint32_t it = threadIdx.x; it < nitems; it += blockDim.x) {
            int pi = (int)itemPidx[it];
            uint64_t p = primes[pi], pv = invp[pi];
            uint64_t vh = t_fastmod64(t_hi, p, pv);
            uint64_t vl = t_fastmod64(t_lo, p, pv);
            uint64_t vm = t_fastmod64(vh * r64[pi] + vl, p, pv);
            uint64_t d = vm ? (p - vm) : 0;
            uint64_t s = t_fastmod64(d * ((p + 1) >> 1), p, pv)
                       + (uint64_t)itemK0[it] * 64u * p;
            if (!nomark) {
                uint32_t j = 0;
                for (; j < 64u && s < tileSlots; j++, s += p)
                    atomicOr(((unsigned int *)sh) + (uint32_t)(s >> 5), 1u << (s & 31u));
                (void)j;
            }
        }
        __syncthreads();
        if (wheel_on) {
            /* ---- wheel pass: mark odd multiples of {3,5,7,11,13} by OR-ing a
               phase-shifted period-15015 pattern (no atomics for these primes;
               their item entries were removed host-side). */
            if (threadIdx.x == 0) {
                uint64_t g0 = tile * tileSlots;
                for (int i = 0; i < 5; i++) {
                    uint64_t p = wp[i];
                    uint64_t B = (v0m[i] + 2u * (g0 % p)) % p;      /* B = v0+2g0 mod p */
                    uint64_t inv2 = (p + 1u) >> 1;
                    rp[i] = ((p - B) % p) * inv2 % p;
                }
            }
            __syncthreads();
            for (uint32_t w = threadIdx.x; w < 236u; w += blockDim.x) pat[w] = 0ull;
            __syncthreads();
            uint64_t tot = 0; (void)tot;
            for (int i = 0; i < 5; i++) {
                uint64_t p = wp[i], r = rp[i];
                /* fill positions [0, 15079) with the periodic pattern so that any
                   64-bit read window starting below 15015 stays inside the array */
                for (uint64_t k = threadIdx.x; ; k += blockDim.x) {
                    uint64_t j = r + k * p;
                    if (j >= 15079u) break;
                    atomicOr((unsigned long long *)&pat[j >> 6], 1ull << (j & 63));
                }
                __syncthreads();
            }
            __syncthreads();
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) {
                uint64_t idx = ((uint64_t)64u * w) % 15015u;
                uint32_t k = (uint32_t)(idx >> 6), r = (uint32_t)(idx & 63);
                uint64_t val = (pat[k] >> r) | (r ? (pat[k + 1] << (64 - r)) : 0ull);
                sh[w] |= val;
            }
            __syncthreads();
        }
        __syncthreads();
        if (!noflush) {
            uint64_t *out = g_bm + tile * tileWords;
            for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x)
                out[w] = sh[w];
        }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: tile_bench START N P_MED [tile_bits] [grid] [block]\n"); return 2; }
    u128 start = 0;
    for (const char *s = argv[1]; *s; s++) start = start * 10 + (u128)(*s - '0');
    uint64_t N = strtoull(argv[2], NULL, 10);
    uint32_t P = (uint32_t)strtoul(argv[3], NULL, 10);
    int tile_bits = (argc > 4) ? atoi(argv[4]) : 20;      /* numbers per tile = 2^bits */
    int grid = (argc > 5) ? atoi(argv[5]) : 138;
    int block = (argc > 6) ? atoi(argv[6]) : 512;
    uint64_t tileNum = 1ULL << tile_bits;
    uint64_t tileSlots = tileNum >> 1;
    uint32_t tileWords = (uint32_t)((tileSlots + 63) / 64);
    if (N % tileNum) { fprintf(stderr, "N must be a multiple of tile size\n"); return 2; }
    uint64_t ntiles = N / tileNum;
    if (tileWords > MAXSH) { fprintf(stderr, "tile too big for shared\n"); return 2; }

    uint8_t *sv = (uint8_t *)calloc((size_t)P + 1, 1);
    for (uint64_t i = 4; i <= (uint64_t)P; i += 2) sv[i] = 1;   /* evens are composite */
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)P / 2 + 64));
    size_t np = 0;
    for (uint64_t i = 3; i <= (uint64_t)P; i++)
        if (!sv[i]) { hp[np++] = i;
            if (i * i <= (uint64_t)P) for (uint64_t j = i * i; j <= (uint64_t)P; j += i) sv[j] = 1; }
    uint64_t *h_iv = (uint64_t *)malloc(np * sizeof(uint64_t));
    uint64_t *h_r = (uint64_t *)malloc(np * sizeof(uint64_t));
    for (size_t i = 0; i < np; i++) {
        h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]);
        h_iv[i] = ~0ULL / hp[i];
    }
    /* items (prime, 64-hit chunk) for one tile geometry */
    int wheel_on = getenv("WHEEL13") != NULL;
    size_t icap = 0;
    for (size_t i = 0; i < np; i++)
        icap += (size_t)((tileSlots / hp[i] + 64) / 64);
    uint32_t *h_ip = (uint32_t *)malloc(icap * 4);
    uint32_t *h_ik = (uint32_t *)malloc(icap * 4);
    size_t nii = 0;
    for (size_t i = 0; i < np; i++) {
        if (wheel_on && hp[i] <= 13u) continue;      /* handled by the wheel pass */
        uint64_t hits = tileSlots / hp[i] + 1;
        uint32_t chunks = (uint32_t)((hits + 63) / 64);
        for (uint32_t c = 0; c < chunks; c++) { h_ip[nii] = (uint32_t)i; h_ik[nii] = c; nii++; }
    }
    printf("[tile] items/tile=%zu wheel=%d\n", nii, wheel_on);
    uint64_t *d_p, *d_iv, *d_r, *d_bm;
    uint32_t *d_ip, *d_ik;
    cudaMalloc(&d_p, np * 8); cudaMalloc(&d_iv, np * 8); cudaMalloc(&d_r, np * 8);
    cudaMalloc(&d_ip, nii * 4); cudaMalloc(&d_ik, nii * 4);
    cudaMalloc(&d_bm, (N >> 1) / 64 * 8);
    cudaMemcpy(d_p, hp, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip, h_ip, nii * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ik, h_ik, nii * 4, cudaMemcpyHostToDevice);

    u128 v0 = start | 1;
    size_t shbytes = (size_t)tileWords * 8;
    cudaFuncSetAttribute(tile_sieve_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);
    int ab_nozero = getenv("SIEVE_NOZERO") != NULL, ab_noflush = getenv("SIEVE_NOFLUSH") != NULL;
    int ab_nomark = getenv("SIEVE_NOMARK") != NULL, ab_noitem = getenv("SIEVE_NOITEM") != NULL;
    printf("[tile] N=%llu P=%u primes=%zu tile=%llu numbers tileWords=%u grid=%d block=%d shared=%zu B\n",
           (unsigned long long)N, P, np, (unsigned long long)tileNum, tileWords, grid, block, shbytes);
    tile_sieve_kernel<<<grid, block, shbytes>>>(d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
        (uint64_t)v0, (uint64_t)(v0 >> 64), tileSlots, ntiles, d_bm, tileWords,
        ab_nozero, ab_noflush, ab_nomark, ab_noitem,
        wheel_on, (uint64_t)(v0 % 3u), (uint64_t)(v0 % 5u), (uint64_t)(v0 % 7u),
        (uint64_t)(v0 % 11u), (uint64_t)(v0 % 13u));
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[tile] %s\n", cudaGetErrorString(e)); return 1; }
    double t0 = now_s();
    for (int r = 0; r < 3; r++)
        tile_sieve_kernel<<<grid, block, shbytes>>>(d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
            (uint64_t)v0, (uint64_t)(v0 >> 64), tileSlots, ntiles, d_bm, tileWords,
            ab_nozero, ab_noflush, ab_nomark, ab_noitem,
            wheel_on, (uint64_t)(v0 % 3u), (uint64_t)(v0 % 5u), (uint64_t)(v0 % 7u),
            (uint64_t)(v0 % 11u), (uint64_t)(v0 % 13u));
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "[tile] %s\n", cudaGetErrorString(e)); return 1; }
    double t = (now_s() - t0) / 3.0;
    printf("[tile] sieve=%.4f s  -> %.1f B/s  (%.2f ns/number)\n",
           t, (double)N / t / 1e9, t / (double)N * 1e9);

    if (getenv("VERIFY")) {
        uint64_t halves = N >> 1;
        uint64_t words = halves / 64;
        uint64_t *hb = (uint64_t *)malloc(words * 8);
        uint8_t *ref = (uint8_t *)calloc((size_t)halves, 1);
        cudaMemcpy(hb, d_bm, words * 8, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < np; i++) {          /* naive CPU marker */
            uint64_t p = hp[i];
            u128 first = (v0 + p - 1) / p * p;     /* first multiple >= v0 */
            if ((first & 1) == 0) first += p;      /* odd multiple only */
            for (u128 m = first; m < v0 + 2 * (u128)halves; m += 2 * (u128)p)
                ref[(uint64_t)((m - v0) / 2)] = 1;
        }
        uint64_t nfalse = 0, nmiss = 0, shown = 0;
        /* candidate broken primes: test their true first-hit slot */
        uint64_t cand[4] = {59, 61, 67, 71};
        for (int ci = 0; ci < 4; ci++) {
            uint64_t q = cand[ci];
            uint64_t vm = (uint64_t)(v0 % q);
            uint64_t s0t = ((q - vm) % q) * ((q + 1) / 2) % q;
            printf("[verify] true s0(%llu)=%llu\n", (unsigned long long)q, (unsigned long long)s0t);
        }
        for (uint64_t s = 0; s < halves; s++) {
            int marked = (int)((hb[s >> 6] >> (s & 63)) & 1ull);
            if (marked != (int)ref[s]) {
                if (marked) nfalse++; else nmiss++;
                if (shown < 12) {
                    u128 v = v0 + 2 * (u128)s;
                    char buf[48]; int k = 0; u128 t2 = v;
                    while (t2) { buf[k++] = (char)('0' + (int)(t2 % 10)); t2 /= 10; }
                    if (!k) buf[k++] = '0';
                    printf("[verify] %s slot=%llu v=", marked ? "FALSE_MARK" : "MISS",
                           (unsigned long long)s);
                    while (k--) putchar(buf[k]);
                    printf(" divis:");
                    for (size_t q = 0, cnt = 0; q < np && cnt < 2; q++)
                        if ((v / hp[q]) * (u128)hp[q] == v) { printf(" %llu", (unsigned long long)hp[q]); cnt++; }
                    printf(" mod59=%llu mod61=%llu mod67=%llu mod71=%llu",
                           (unsigned long long)(s % 59), (unsigned long long)(s % 61),
                           (unsigned long long)(s % 67), (unsigned long long)(s % 71));
                    printf("\n");
                    shown++;
                }
            }
        }
        printf("[verify] false=%llu miss=%llu of %llu slots\n",
               (unsigned long long)nfalse, (unsigned long long)nmiss,
               (unsigned long long)halves);
    }
    if (getenv("VERIFY2")) {
        /* arbiter: p0_mark_kernel (verified) on the SAME per-tile geometry,
           vis filter disabled (all-ones mask) so both mark the same set */
        size_t pni = 0;
        for (size_t i = 0; i < np; i++) {
            uint64_t hits = tileSlots / hp[i] + 1;
            pni += (size_t)((hits + 63) / 64);
        }
        P0Item *pit = (P0Item *)malloc(pni * sizeof(P0Item));
        size_t kk = 0;
        for (size_t i = 0; i < np; i++) {
            uint64_t hits = tileSlots / hp[i] + 1;
            uint32_t chunks = (uint32_t)((hits + 63) / 64);
            for (uint32_t c = 0; c < chunks; c++) { pit[kk].pidx = (uint32_t)i; pit[kk].k0 = c * 64; kk++; }
        }
        P0Item *d_pit; uint64_t *d_bm2;
        cudaMalloc(&d_pit, kk * sizeof(P0Item));
        cudaMalloc(&d_bm2, (size_t)tileWords * 8 * ntiles);
        cudaMemcpy(d_pit, pit, kk * sizeof(P0Item), cudaMemcpyHostToDevice);
        cudaMemset(d_bm2, 0, (size_t)tileWords * 8 * ntiles);
        for (uint64_t tl = 0; tl < ntiles; tl++) {
            u128 tv = v0 + (u128)(tl * tileSlots * 2);
            p0_mark_kernel<<<(uint32_t)((kk + 255) / 256), 256>>>(d_pit, (uint32_t)kk, d_p, d_r, d_iv,
                tileSlots, (uint64_t)tv, (uint64_t)(tv >> 64), 0x3FFFFFFFu,
                d_bm2 + tl * tileWords);
        }
        cudaError_t e2 = cudaDeviceSynchronize();
        if (e2 != cudaSuccess) { fprintf(stderr, "[v2] %s\n", cudaGetErrorString(e2)); return 1; }
        uint64_t words = (N >> 1) / 64;
        uint64_t *hb1 = (uint64_t *)malloc(words * 8);
        uint64_t *hb2 = (uint64_t *)malloc(words * 8);
        cudaMemcpy(hb1, d_bm, words * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb2, d_bm2, words * 8, cudaMemcpyDeviceToHost);
        { FILE *f = fopen("/tmp/tile_bm.bin", "wb"); fwrite(hb1, 1, words * 8, f); fclose(f);
          printf("[v2] dumped /tmp/tile_bm.bin (%llu bytes)\n", (unsigned long long)(words * 8)); }
        uint64_t donly = 0, ponly = 0, sh2 = 0;
        for (uint64_t s = 0; s < (N >> 1); s++) {
            int b1 = (int)((hb1[s >> 6] >> (s & 63)) & 1ull);
            int b2 = (int)((hb2[s >> 6] >> (s & 63)) & 1ull);
            if (b1 != b2) {
                if (b1) donly++; else ponly++;
                if (sh2 < 10) { printf("[v2] slot=%llu tile=%d p0=%d\n", (unsigned long long)s, b1, b2); sh2++; }
            }
        }
        printf("[v2] tile-only=%llu p0-only=%llu of %llu slots (items kk=%zu)\n",
               (unsigned long long)donly, (unsigned long long)ponly,
               (unsigned long long)(N >> 1), kk);
    }
    return 0;
}
