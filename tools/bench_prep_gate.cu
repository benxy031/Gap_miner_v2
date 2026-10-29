/*
 * bench_prep_gate.cu - differential gate for P0_MARK_PREP (class30_prep_kernel
 * + the hoisted branch in class30_sieve_kernel).
 *
 * Runs the PRODUCTION kernel twice on one block with identical inputs:
 *   arm 0: prep = NULL   (inline per-row setup)
 *   arm 1: prep = table  (class30_prep_kernel output)
 * then compares the two bitmaps word by word and reports the first mismatch
 * (tile, word, xor) so a divergence can be localised.
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 tools/bench_prep_gate.cu -o bin/bench_prep_gate
 * Usage: ./bin/bench_prep_gate [P] [tiles] [tileSlots]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "p0_walk_kern.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

int main(int argc, char **argv) {
    uint32_t P = (argc > 1) ? (uint32_t)strtoul(argv[1], NULL, 10) : 150000u;
    uint32_t ntiles = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 10) : 8u;
    uint64_t tileSlots = (argc > 3) ? strtoull(argv[3], NULL, 10) : 753664ull;
    uint64_t grid = (argc > 4) ? strtoull(argv[4], NULL, 10) : 0;   /* CTA count, 0 = ntiles */
    uint64_t lastSlots = (argc > 5) ? strtoull(argv[5], NULL, 10) : tileSlots;
    int wheel_on = (argc > 6) ? atoi(argv[6]) : 0;
    uint32_t tileWords = (uint32_t)(tileSlots >> 6);
    const uint32_t itemChunk = 256u;

    /* primes 7..P (same construction as run_walk_engine) */
    std::vector<char> sv(P + 1, 1);
    sv[0] = sv[1] = 0;
    for (uint32_t i = 2; (uint64_t)i * i <= P; i++) if (sv[i]) for (uint32_t j = i * i; j <= P; j += i) sv[j] = 0;
    std::vector<uint64_t> hp, hiv, hr;
    for (uint32_t i = 7; i <= P; i += 2) if (sv[i]) {
        hp.push_back(i); hiv.push_back(~0ULL / i); hr.push_back((~0ULL) % i + 1);
    }
    size_t np = hp.size();

    const uint32_t CLS8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
    int8_t clsidx[30]; uint32_t perm[30] = {0}, pinv[30] = {0};
    for (int i = 0; i < 30; i++) clsidx[i] = -1;
    for (int i = 0; i < 8; i++) clsidx[CLS8[i]] = (int8_t)i;
    for (int pm = 0; pm < 30; pm++) {
        if (clsidx[pm] < 0) continue;
        for (int j = 0; j < 8; j++) perm[pm] |= 1u << ((pm * CLS8[j]) % 30);
        for (int x = 1; x < 30; x++) if ((pm * x) % 30 == 1) pinv[pm] = (uint32_t)x;
    }

    /* item table: one row per (prime, itemChunk marks), as production builds it */
    std::vector<uint32_t> ip, ik;
    for (size_t i = 0; i < np; i++) {
        if (hp[i] <= 13u) continue;
        uint64_t marks = tileSlots / hp[i] + 2;
        uint32_t chunks = (uint32_t)((marks + itemChunk - 1) / itemChunk);
        for (uint32_t c = 0; c < chunks; c++) { ip.push_back((uint32_t)i); ik.push_back(c); }
    }
    uint32_t nitems = (uint32_t)ip.size();

    uint64_t Alo, Ahi;
    {   /* start = 1e32, 30-aligned like the launch site */
        unsigned __int128 A = (unsigned __int128)10000000000000000ULL * 10000000000000000ULL;
        A = (A / 30u) * 30u;
        Alo = (uint64_t)A; Ahi = (uint64_t)(A >> 64);
    }

    uint64_t *d_p, *d_iv, *d_r, *d_bm0, *d_bm1;
    uint32_t *d_ip, *d_ik, *d_wi, *d_prep;
    CK(cudaMalloc(&d_p, np * 8)); CK(cudaMalloc(&d_iv, np * 8)); CK(cudaMalloc(&d_r, np * 8));
    CK(cudaMalloc(&d_ip, nitems * 4)); CK(cudaMalloc(&d_ik, nitems * 4));
    CK(cudaMalloc(&d_bm0, (size_t)ntiles * tileWords * 8));
    CK(cudaMalloc(&d_bm1, (size_t)ntiles * tileWords * 8));
    CK(cudaMalloc(&d_wi, 3 * 4));
    CK(cudaMalloc(&d_prep, (size_t)ntiles * np * 4));
    CK(cudaMemcpy(d_p, hp.data(), np * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_iv, hiv.data(), np * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_r, hr.data(), np * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ip, ip.data(), nitems * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ik, ik.data(), nitems * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpyToSymbol(c30_clsidx, clsidx, sizeof(clsidx)));
    CK(cudaMemcpyToSymbol(c30_perm, perm, sizeof(perm)));
    CK(cudaMemcpyToSymbol(c30_pinv, pinv, sizeof(pinv)));

    size_t shb = (size_t)tileWords * 8;
    CK(cudaFuncSetAttribute(class30_sieve_kernel<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb));

    if (!grid) grid = ntiles;
    uint32_t h_wi[3] = {0, 1, 2};   /* hp starts at 7, so 7/11/13 are indices 0/1/2 */
    CK(cudaMemcpy(d_wi, h_wi, sizeof(h_wi), cudaMemcpyHostToDevice));
    printf("P=%u np=%zu items=%u tiles=%u grid=%llu tileSlots=%llu last=%llu wheel=%d (%u words)\n",
           P, np, nitems, ntiles, (unsigned long long)grid, (unsigned long long)tileSlots,
           (unsigned long long)lastSlots, wheel_on, tileWords);

    /* arm 0: inline */
    CK(cudaMemset(d_bm0, 0, (size_t)ntiles * tileWords * 8));
    class30_sieve_kernel<0><<<(unsigned)grid, 512, shb>>>(d_p, d_iv, d_r, d_ip, d_ik, nitems,
                                                  Alo, Ahi, tileSlots, ntiles, lastSlots,
                                                  d_bm0, tileWords, d_wi, wheel_on, itemChunk, NULL, 0);
    CK(cudaDeviceSynchronize());
    /* build the table, then arm 1 */
    class30_prep_kernel<<<ntiles, 256>>>(d_p, d_iv, d_r, (uint32_t)np, Alo, Ahi,
                                         tileSlots, ntiles, d_prep);
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(d_bm1, 0, (size_t)ntiles * tileWords * 8));
    class30_sieve_kernel<0><<<(unsigned)grid, 512, shb>>>(d_p, d_iv, d_r, d_ip, d_ik, nitems,
                                                  Alo, Ahi, tileSlots, ntiles, lastSlots,
                                                  d_bm1, tileWords, d_wi, wheel_on, itemChunk,
                                                  d_prep, (uint32_t)np);
    CK(cudaDeviceSynchronize());

    size_t words = (size_t)ntiles * tileWords;
    std::vector<uint64_t> a(words), b(words);
    CK(cudaMemcpy(a.data(), d_bm0, words * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), d_bm1, words * 8, cudaMemcpyDeviceToHost));

    size_t diff = 0, first = (size_t)-1;
    for (size_t i = 0; i < words; i++) if (a[i] != b[i]) { if (first == (size_t)-1) first = i; diff++; }
    printf("words=%zu  differing=%zu\n", words, diff);
    if (diff) {
        size_t tile = first / tileWords, w = first % tileWords;
        printf("first mismatch: tile=%zu word=%zu  inline=%016llx  prep=%016llx  xor=%016llx\n",
               tile, w, (unsigned long long)a[first], (unsigned long long)b[first],
               (unsigned long long)(a[first] ^ b[first]));
        printf("  missing marks (inline has, prep lacks): %d\n", __builtin_popcountll(a[first] & ~b[first]));
        printf("  extra marks   (prep has, inline lacks): %d\n", __builtin_popcountll(b[first] & ~a[first]));
        printf("GATE FAIL\n");
        return 1;
    }
    printf("GATE PASS: prep bitmap is bit-identical to the inline setup\n");
    return 0;
}
