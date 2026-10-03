/*
 * hc_bench.cu - half-class (8 classes of 30) tile sieve prototype + GATE.
 *
 * The HC bitmap stores 1 bit per value coprime to 30 (classes RES = 1,7,11,13,
 * 17,19,23,29 mod 30) = 8 bits per 30 numbers (vs the full odd bitmap's 1 bit
 * per 2 numbers = 3.75x denser in numbers). Bit index = (v/30 - blk0)*8 + cls.
 *
 * GATE: the kernel's HC bitmap must EXACTLY equal the host conversion of the
 * engine's proven full bitmap (/tmp/eng_bm.bin): for every class-valid value in
 * the range, hc_bit == full_bit[slot]. Any mismatch = fail.
 * Also prints the sieve rate for comparison with the full bitmap's 331 B/s.
 */
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <time.h>

typedef unsigned __int128 u128;

#define P_LIM 100000
#define TILE_BLOCKS (1u << 15)              /* 30-blocks per tile              */
#define TILE_NUMBERS (TILE_BLOCKS * 30u)    /* numbers per tile = 983040       */
#define TILE_WORDS (TILE_BLOCKS / 8u)       /* 64-bit words per tile = 4096    */

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

__constant__ uint8_t  c_RES[8] = {1,7,11,13,17,19,23,29};
__constant__ int8_t   c_CLS[30] = {-1,0,-1,-1,-1,-1,-1,1,-1,-1,-1,2,-1,3,-1,-1,
                                   -1,4,-1,5,-1,-1,-1,6,-1,-1,-1,-1,-1,7};
__constant__ uint32_t c_MASK30 = (1u<<1)|(1u<<7)|(1u<<11)|(1u<<13)|(1u<<17)
                                 |(1u<<19)|(1u<<23)|(1u<<29);

typedef struct { uint32_t pidx; uint32_t k0; } HCItem;

__device__ __forceinline__ uint64_t t_fastmod64(uint64_t x, uint64_t p, uint64_t invp) {
    uint64_t q = __umul64hi(x, invp);
    uint64_t r = x - q * p;
    if (r >= p) r -= p;
    return r;
}

/* q = v/30, rem = v%30 for the 68-bit value (v2,v1,v0), v < 2^68, v2 <= 15.
   v = hi32*2^32 + lo32 with hi32 < 2^36; 2^32 = 30*143165576 + 16. */
__device__ __forceinline__ uint64_t div30_68(uint32_t v2, uint32_t v1, uint32_t v0,
                                             uint32_t *rem) {
    uint64_t hi32 = ((uint64_t)v2 << 32) | v1;      /* < 2^36 */
    uint64_t inner = hi32 * 16u + (uint64_t)v0;     /* < 2^41  */
    uint64_t q = hi32 * 143165576ull + inner / 30u;
    *rem = (uint32_t)(inner % 30u);
    return q;
}

__global__ void tile_sieve_hc(const uint64_t * __restrict__ primes,
                              const uint64_t * __restrict__ invp,
                              const uint64_t * __restrict__ r64,
                              const uint32_t * __restrict__ itemPidx,
                              const uint32_t * __restrict__ itemK0,
                              uint32_t nitems,
                              uint64_t tv_lo, uint64_t tv_hi,   /* tile[0] first value */
                              uint64_t blk0,                     /* bitmap's first 30-block */
                              uint64_t blockStride,              /* 30-blocks per tile */
                              uint64_t ntiles,
                              uint32_t skipBelow,                /* tile-0: skip values < start */
                              uint64_t * __restrict__ g_bm, uint32_t tileWords) {
    extern __shared__ uint64_t sh[];
    for (uint64_t tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x) sh[w] = 0;
        __syncthreads();
        u128 tvu = ((u128)tv_hi << 64 | tv_lo) + ((u128)tile * blockStride * 30u);
        uint32_t tv2 = (uint32_t)(tvu >> 64);
        uint32_t tv1 = (uint32_t)(tvu >> 32);
        uint32_t tv0 = (uint32_t)tvu;
        const uint64_t tblk0 = blk0 + tile * blockStride;   /* tile's first 30-block */
        const uint32_t skip = (tile == 0) ? skipBelow : 0u;
        /* first odd multiple of p at/above tv: value offset rel0 = 2*s0 */
        for (uint32_t it = threadIdx.x; it < nitems; it += blockDim.x) {
            int pi = (int)itemPidx[it];
            uint64_t p = primes[pi], pv = invp[pi];
            /* first ODD multiple of p at/above tv: tv is a multiple of 30
               (even) -> solve from tvo = tv+1 (odd); value = tvo + 2*s0. */
            uint64_t lo64o = ((uint64_t)tv1 << 32) | (uint64_t)(tv0 | 1u);
            uint64_t vm = t_fastmod64((uint64_t)tv2 * r64[pi]
                                      + t_fastmod64(lo64o, p, pv), p, pv);
            uint64_t d = vm ? (p - vm) : 0;
            uint64_t s0 = t_fastmod64(d * ((p + 1) >> 1), p, pv);
            uint64_t rel = 1u + 2u * s0 + (uint64_t)itemK0[it] * 128u * p;
            if (rel >= TILE_NUMBERS) continue;
            /* initial absolute value limbs v = tv + rel */
            u128 av = ((u128)tv2 << 64 | (u128)tv1 << 32 | (u128)tv0) + (u128)rel;
            uint32_t v2 = (uint32_t)(av >> 64);
            uint32_t v1 = (uint32_t)(av >> 32);
            uint32_t v0 = (uint32_t)av;
            uint32_t rem;
            uint64_t blk = div30_68(v2, v1, v0, &rem);
            uint32_t s30 = (uint32_t)((2u * p) % 30u);
            uint32_t stepv = (uint32_t)(2u * p);          /* value step, but p<1e5 -> 2p<2e5 ok */
            uint32_t j = 0;
            for (; j < 64u && rel < TILE_NUMBERS; j++, rel += stepv) {
                if (rel >= skip && ((1u << rem) & c_MASK30) != 0u) {
                    uint64_t bit = (blk - tblk0) * 8u + (uint64_t)c_CLS[rem];
                    atomicOr(((unsigned int *)sh) + (uint32_t)(bit >> 5),
                             1u << (bit & 31u));
                }
                /* advance: v += 2p ; rem += s30 ; blk += (2p + rem_old-rem_new)/30 */
                uint32_t rn = rem + s30;
                if (rn >= 30u) rn -= 30u;
                blk += (uint64_t)(stepv + rem - rn) / 30u;
                rem = rn;
            }
            (void)v0; (void)v1; (void)v2;
        }
        __syncthreads();
        uint64_t *out = g_bm + tile * tileWords;
        for (uint32_t w = threadIdx.x; w < tileWords; w += blockDim.x)
            out[w] = sh[w];
        __syncthreads();
    }
}

int main(void) {
    u128 start = ((u128)10 << 64) | (u128)15532559262904483841ULL;  /* 2e20+1 */
    uint64_t blk0 = (uint64_t)(start / 30u);
    u128 base30 = (u128)blk0 * 30u;
    uint64_t tv_lo = (uint64_t)base30, tv_hi = (uint64_t)(base30 >> 64);
    uint64_t startOff = (uint64_t)(start - base30);   /* start mod 30 = 21 */

    uint64_t ntiles = 1092;                        /* ~2^30 numbers, 1092*983040 */
    uint64_t BN = ntiles * TILE_NUMBERS;
    uint64_t totalBlocks = BN / 30u;               /* kernel covers exactly BN/30 blocks */
    uint64_t hcWords = (totalBlocks * 8u + 63u) / 64u;

    printf("[hc] BN=%llu tiles=%llu words=%llu (%.1f MB)\n",
           (unsigned long long)BN, (unsigned long long)ntiles,
           (unsigned long long)hcWords, hcWords * 8.0 / 1e6);

    /* ---- host: primes 7..P, item table ---- */
    uint8_t *sv = (uint8_t *)calloc(P_LIM + 1, 1);
    for (uint64_t i = 4; i <= P_LIM; i += 2) sv[i] = 1;
    uint64_t *hp = (uint64_t *)malloc(sizeof(uint64_t) * (P_LIM / 2 + 64));
    size_t np = 0;
    for (uint64_t i = 7; i <= P_LIM; i++)     /* p=3,5 mark nothing class-valid */
        if (!sv[i]) { hp[np++] = i;
            if (i * i <= P_LIM) for (uint64_t j = i * i; j <= P_LIM; j += i) sv[j] = 1; }
    uint64_t *h_iv = (uint64_t *)malloc(np * 8), *h_r = (uint64_t *)malloc(np * 8);
    for (size_t i = 0; i < np; i++) {
        h_r[i] = (uint64_t)(((u128)1 << 64) % hp[i]);
        h_iv[i] = ~0ULL / hp[i];
    }
    size_t nii = 0, icap = 0;
    for (size_t i = 0; i < np; i++) icap += (size_t)((TILE_NUMBERS / (2 * hp[i]) + 64) / 64 + 1);
    uint32_t *h_ip = (uint32_t *)malloc(icap * 4), *h_ik = (uint32_t *)malloc(icap * 4);
    for (size_t i = 0; i < np; i++) {
        uint64_t hits = TILE_NUMBERS / (2 * hp[i]) + 1;
        uint32_t chunks = (uint32_t)((hits + 63) / 64);
        for (uint32_t c = 0; c < chunks; c++) { h_ip[nii] = (uint32_t)i; h_ik[nii] = c; nii++; }
    }
    printf("[hc] primes=%zu items=%zu\n", np, nii);

    uint64_t *d_p, *d_iv, *d_r, *d_bm; uint32_t *d_ip, *d_ik;
    cudaMalloc(&d_p, np * 8); cudaMalloc(&d_iv, np * 8); cudaMalloc(&d_r, np * 8);
    cudaMalloc(&d_ip, nii * 4); cudaMalloc(&d_ik, nii * 4);
    cudaMalloc(&d_bm, hcWords * 8);
    cudaMemcpy(d_p, hp, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iv, h_iv, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, h_r, np * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ip, h_ip, nii * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ik, h_ik, nii * 4, cudaMemcpyHostToDevice);
    cudaMemset(d_bm, 0, hcWords * 8);
    size_t shbytes = (size_t)TILE_WORDS * 8;
    cudaFuncSetAttribute(tile_sieve_hc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes);

    int grid = 138, block = 512;
    double t0 = now_s();
    tile_sieve_hc<<<grid, block, shbytes>>>(d_p, d_iv, d_r, d_ip, d_ik, (uint32_t)nii,
        tv_lo, tv_hi, blk0, (uint64_t)TILE_BLOCKS, ntiles, (uint32_t)startOff,
        d_bm, TILE_WORDS);
    cudaError_t e = cudaDeviceSynchronize();
    double t = now_s() - t0;
    printf("[hc] %s sieve=%.4f s -> %.1f B/s\n", cudaGetErrorString(e), t,
           (double)BN / t / 1e9);

    /* ---- GATE: compare with host conversion of the full bitmap ---- */
    FILE *f = fopen("/tmp/eng_bm.bin", "rb");
    if (!f) { printf("[hc] no /tmp/eng_bm.bin - skipping gate\n"); return 0; }
    static uint64_t full[1 << 23];       /* 64M words = 512MB? no: 1<<23 * 8 = 64MB */
    uint64_t fullWords = 8388608;        /* 2^30 slots / 64 */
    if (fread(full, 8, fullWords, f) != fullWords) { printf("[hc] short read\n"); return 1; }
    fclose(f);

    uint64_t *hc = (uint64_t *)malloc(hcWords * 8);
    cudaMemcpy(hc, d_bm, hcWords * 8, cudaMemcpyDeviceToHost);

    /* build the expected HC bitmap from the full one (block/class space, u64) */
    uint64_t *exp = (uint64_t *)calloc(hcWords, 8);
    uint64_t nclass = 0, nmark = 0;
    int8_t res[8] = {1,7,11,13,17,19,23,29};
    for (uint64_t blk = 0; blk < totalBlocks; blk++) {
        for (int ci = 0; ci < 8; ci++) {
            int64_t off = (int64_t)(blk * 30u + (uint64_t)res[ci]) - (int64_t)startOff;
            if (off < 0 || (uint64_t)off >= BN) continue;
            uint64_t slot = (uint64_t)off >> 1;
            uint64_t bit = blk * 8u + (uint64_t)ci;
            uint64_t m = (full[slot >> 6] >> (slot & 63)) & 1u;
            exp[bit >> 6] |= m << (bit & 63);
            nclass++; nmark += m;
        }
    }
    uint64_t bad = 0, hcmarks = 0;
    uint64_t dbit[8]; int nd = 0;
    for (uint64_t w = 0; w < hcWords; w++) {
        uint64_t x = hc[w] ^ exp[w];
        if (x) {
            bad += __builtin_popcountll(x);
            while (x && nd < 8) {
                dbit[nd++] = w * 64 + (uint64_t)__builtin_ctzll(x);
                x &= x - 1;
            }
        }
        hcmarks += __builtin_popcountll(hc[w]);
    }
    for (int d = 0; d < nd; d++) {
        uint64_t bit = dbit[d];
        uint64_t blk = bit >> 3, ci = bit & 7;
        int64_t off = (int64_t)(blk * 30u + (uint64_t)res[ci]) - (int64_t)startOff;
        int fs = -1;
        if (off >= 0 && (uint64_t)off < BN) {
            uint64_t sl = (uint64_t)off >> 1;
            fs = (int)((full[sl >> 6] >> (sl & 63)) & 1u);
        }
        printf("[diff] bit=%llu off=%lld full=%d hc=%d\n",
               (unsigned long long)bit, (long long)off, fs,
               (int)((hc[bit >> 6] >> (bit & 63)) & 1u));
    }
    printf("[hc] GATE: class-values=%llu  full-marks=%llu  hc-marks=%llu  diff-bits=%llu -> %s\n",
           (unsigned long long)nclass, (unsigned long long)nmark,
           (unsigned long long)hcmarks, (unsigned long long)bad,
           bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
