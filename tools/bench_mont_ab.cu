/* tools/bench_mont_ab.cu - row-level A/B of carry styles for the Montgomery inner loop.
 *
 * WHY THIS EXISTS
 *   gECC (arXiv 2501.03245, TACO 2025) claims modular-multiplication efficiency is bound by
 *   the NUMBER of IMAD instructions and reduces it by passing carries through predicate
 *   registers / using IADD3 instead of IMAD. Our bench_issue microbench reproduced the
 *   microarchitectural premise on this RTX 3070 (IMAD plain 152.6 G op/s, IADD3 plain
 *   239.7 G op/s, 1:1 mix 287.7 G = 1.89x of IMAD-only). This tool asks the next question:
 *   CAN ANY HAND-WRITTEN CARRY STYLE BEAT WHAT nvcc ALREADY EMITS for our CIOS code?
 *
 * WHAT IS MEASURED
 *   ONE Montgomery row op, verbatim from the inside of CIOS:  T[0..24] += a[0..23] * bi
 *   with a running carry (bi = one limb of b).  24 limb-MACs + the carry resolution per row.
 *   Three correct implementations of the SAME row:
 *     ARM_C      : C code  p = (u64)a[j]*bi + t[j] + c;             (what tools/bench_cios_mr.cu
 *                  and new_src/gpu/gpu_fermat.cu use - the compiler picks the instruction form)
 *     ARM_WIDE   : explicit mad.wide.u32 with a 64-bit addend      (what ptxas fuses the C code
 *                  into: 1 IMAD.WIDE + 2 IADD3 per limb; hand-PTX version of the same shape)
 *     ARM_SPLIT6 : mul.lo + mul.hi + add.cc + addc(extract) + add.cc + addc
 *                  (2 MUL + 4 IADD3 per limb - the "move the carry work to the INT pipe"
 *                  variant, cheapest provably-correct form with a full 32-bit carry word)
 *
 * FINDING THAT SHAPED THE ARM SET (2026-09-29, verified by this tool's own harness):
 *   the 2-slot "mad.lo.cc + madc.hi" pair CANNOT express a flat-row recurrence - the correct
 *   low word needs t[j] + plo + c added at weight 2^0, while madc adds the carry word at
 *   weight 2^32 (that form is only valid in the shifted-accumulator/sppark SOS layout, where
 *   the carry words ARE the accumulator). A first version of this tool built the pair form
 *   anyway and the host __int128 replica flagged it immediately (that is what verification
 *   is for). So the honest flat-row arm set is the three forms above.
 *
 * MEASURED VERDICT (2026-09-29, RTX 3070 @1725 MHz, 46 SMs, all arms bit-exact vs the
 * host __int128 replica over 40k rows, 1104 blocks x 256 threads x 20000 rows):
 *   C      1210.56 G limb-MAC/s (50.44 G rows/s)  <- nvcc's own mix, FASTEST
 *   WIDE   1118.37 G  (-7.6%; hand mad.wide + explicit u64 addend)
 *   SPLIT6 1110.43 G  (-8.3%; 2 MUL + 4 IADD3 per limb, the max-INT-pipe variant)
 *   SASS per row: C = 25 WIDE + 66 IMAD + 70 IADD3 | WIDE = 25 + 42 + 83 | SPLIT6 = 1 + 106 + 65
 *   => the op runs at ~80% of this SM's issue capacity and NO hand-written carry style
 *      beats plain C.  The INT-pipe headroom measured in bench_issue (1.89x on a pure
 *      1:1 IMAD+IADD3 stream) is NOT reachable from this algorithm shape: moving the
 *      carry work to IADD3 costs more instructions than it buys back.  gECC's own
 *      general-modulus gain was 1.17x over CGBN-1 (A100, 256-bit); our CIOS already
 *      beat CGBN by 1.25x in bulk streams (bench_cios_mr), and the paper's remaining
 *      +39% is SM2-only (q_inv = 1) - dead for our random moduli.  No pipeline A/B
 *      follows: the isolated op did not win.
 *
 * HONESTY RULES
 *   - Verification FIRST and against an INDEPENDENT reference: a host __int128 replica of the
 *     row recurrence, exact same xorshift bi sequence. Any arm with a mismatch is reported
 *     and its timing discarded.
 *   - Same threads/blocks/iters for every arm; limb-MACs = rows x 24 is the reported unit.
 *   - asm is volatile, so arms B/C/D lose some scheduler freedom the C arm has; that is part
 *     of the measured difference, not a bug (noted again in the output).
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -o bin/bench_mont_ab tools/bench_mont_ab.cu
 * Run:   ./bin/bench_mont_ab [blocksPerSM=24] [iters=20000] [threads=256]
 * SASS mix per arm: cuobjdump -sass bin/bench_mont_ab > /tmp/sab.txt
 *   awk '/Function : /{f=$3} /IMAD/{i[f]++} /IADD3/{a[f]++} END{for(k in i) print k, "IMAD="i[k], "IADD3="a[k]}' /tmp/sab.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define NL 24   /* 768-bit = 24 x 32-bit limbs, the width of our production CIOS path */

static void ck(const char *what, cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}

/* ---------------------------------------------------------------- xorshift (host+device identical) */
__host__ __device__ __forceinline__ static uint64_t xs64(uint64_t x) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x;
}

/* ---------------------------------------------------------------- the three row arms */
__device__ __forceinline__ static void row_c(uint32_t *t, const uint32_t *a, uint32_t bi) {
    uint64_t c = 0;
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint64_t p = (uint64_t)a[j] * (uint64_t)bi + (uint64_t)t[j] + c;
        t[j] = (uint32_t)p;
        c = p >> 32;
    }
    t[NL] += (uint32_t)c;
}

__device__ __forceinline__ static void row_wide(uint32_t *t, const uint32_t *a, uint32_t bi) {
    uint64_t c = 0;
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint64_t u = (uint64_t)t[j] + c;   /* 64-bit addend -> IMAD.WIDE carries the sum */
        uint64_t p;
        asm volatile("mad.wide.u32 %0, %1, %2, %3;" : "=l"(p) : "r"(a[j]), "r"(bi), "l"(u));
        t[j] = (uint32_t)p;
        c = p >> 32;
    }
    t[NL] += (uint32_t)c;
}

__device__ __forceinline__ static void row_split6(uint32_t *t, const uint32_t *a, uint32_t bi) {
    uint32_t c = 0;
#pragma unroll
    for (int j = 0; j < NL; j++) {
        uint32_t plo, phi, x, cc1, y, c2;
        asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(plo) : "r"(a[j]), "r"(bi));
        asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(phi) : "r"(a[j]), "r"(bi));
        asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(x) : "r"(t[j]), "r"(plo));   /* x, CC=cc1 */
        asm volatile("addc.u32 %0, %1, %2;" : "=r"(cc1) : "r"(0u), "r"(0u));      /* extract cc1 */
        asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(y) : "r"(x), "r"(c));        /* y = x+c, CC=cc2 */
        asm volatile("addc.u32 %0, %1, %2;" : "=r"(c2) : "r"(phi), "r"(cc1));     /* phi + cc1 + cc2 */
        t[j] = y; c = c2;
    }
    t[NL] += c;
}

/* ---------------------------------------------------------------- kernels */
/* Verify kernel: 1 thread, K iterations, full t[] dumped. */
template <int ARM>
__global__ void k_verify(uint32_t iters, uint64_t seed, const uint32_t *a_in, uint32_t *out)
{
    uint32_t a[NL], t[NL + 1];
    for (int j = 0; j < NL; j++) a[j] = a_in[j];
#pragma unroll
    for (int j = 0; j <= NL; j++) t[j] = 0u;
    uint64_t s = seed;
#pragma unroll 1
    for (uint32_t it = 0; it < iters; it++) {
        s = xs64(s);
        uint32_t bi = (uint32_t)s | 1u;
        if (ARM == 0) row_c(t, a, bi);
        else if (ARM == 1) row_wide(t, a, bi);
        else row_split6(t, a, bi);
    }
    for (int j = 0; j <= NL; j++) out[j] = t[j];
}

/* Perf kernel: many threads x iters rows each; checksum out. */
template <int ARM>
__global__ void k_perf(uint32_t iters, uint64_t seed, const uint32_t *a_in, uint32_t *out)
{
    uint32_t a[NL], t[NL + 1];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < NL; j++) a[j] = a_in[j] + ((tid * 2654435761u) & 0xffffu);
#pragma unroll
    for (int j = 0; j <= NL; j++) t[j] = 0u;
    uint64_t s = seed ^ ((uint64_t)tid * 0x9E3779B97F4A7C15ULL);
#pragma unroll 1
    for (uint32_t it = 0; it < iters; it++) {
        s = xs64(s);
        uint32_t bi = (uint32_t)s | 1u;
        if (ARM == 0) row_c(t, a, bi);
        else if (ARM == 1) row_wide(t, a, bi);
        else row_split6(t, a, bi);
    }
    uint32_t sum = 0;
#pragma unroll
    for (int j = 0; j <= NL; j++) sum += t[j];
    if (sum == 0x12345678u) out[tid] = sum;
}

/* ---------------------------------------------------------------- host reference (__int128) */
static void host_rows(const uint32_t *a, uint32_t iters, uint64_t seed, uint32_t *t_out)
{
    uint32_t t[NL + 1];
    for (int j = 0; j <= NL; j++) t[j] = 0u;
    uint64_t s = seed;
    for (uint32_t it = 0; it < iters; it++) {
        s = xs64(s);
        uint32_t bi = (uint32_t)s | 1u;
        unsigned __int128 c = 0;
        for (int j = 0; j < NL; j++) {
            unsigned __int128 p = (unsigned __int128)a[j] * bi + t[j] + c;
            t[j] = (uint32_t)p;
            c = p >> 32;
        }
        t[NL] += (uint32_t)c;
    }
    for (int j = 0; j <= NL; j++) t_out[j] = t[j];
}

/* ---------------------------------------------------------------- main */
static const char *ARMNAME[3] = { "C     ", "WIDE  ", "SPLIT6" };

int main(int argc, char **argv)
{
    int bpsm = argc > 1 ? atoi(argv[1]) : 24;
    uint32_t iters = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 20000;
    int tpb = argc > 3 ? atoi(argv[3]) : 256;

    cudaDeviceProp prop;
    ck("prop", cudaGetDeviceProperties(&prop, 0));
    int blocks_total = bpsm * prop.multiProcessorCount;

    /* inputs: fixed a[], seed - same for all arms */
    uint32_t a[NL];
    for (int j = 0; j < NL; j++) a[j] = 0x9E3779B9u * (uint32_t)(j + 1) + 0x12345677u;
    a[NL - 1] |= 0x80000000u;   /* top bit set like a real 768-bit operand */
    const uint64_t seed = 0x0123456789ABCDEFULL;

    uint32_t *d_a, *d_out;
    ck("malloc", cudaMalloc((void **)&d_a, sizeof(a)));
    ck("malloc2", cudaMalloc((void **)&d_out, sizeof(uint32_t) * (size_t)blocks_total * tpb));
    ck("memcpy", cudaMemcpy(d_a, a, sizeof(a), cudaMemcpyHostToDevice));

    /* ---- 1) verification ---- */
    uint32_t v_iters = 2u * iters > 100000u ? 100000u : 2u * iters;
    printf("=== verification (host __int128 replica, %u rows, same xorshift) ===\n", v_iters);
    uint32_t *h_ref = (uint32_t *)malloc(sizeof(uint32_t) * (NL + 1));
    uint32_t *h_dev = (uint32_t *)malloc(sizeof(uint32_t) * (NL + 1));
    host_rows(a, v_iters, seed, h_ref);
    int bad_arms = 0;
    for (int arm = 0; arm < 3; arm++) {
        uint32_t *d_v; ck("malloc3", cudaMalloc((void **)&d_v, sizeof(uint32_t) * (NL + 1)));
        if (arm == 0) k_verify<0><<<1, 1>>>(v_iters, seed, d_a, d_v);
        else if (arm == 1) k_verify<1><<<1, 1>>>(v_iters, seed, d_a, d_v);
        else k_verify<2><<<1, 1>>>(v_iters, seed, d_a, d_v);
        ck("verify launch", cudaDeviceSynchronize());
        ck("verify copy", cudaMemcpy(h_dev, d_v, sizeof(uint32_t) * (NL + 1), cudaMemcpyDeviceToHost));
        int mism = 0;
        for (int j = 0; j <= NL; j++) if (h_dev[j] != h_ref[j]) mism++;
        printf("arm %s : %s%s\n", ARMNAME[arm], mism ? "MISMATCH " : "bit-exact",
               mism ? "" : " (25/25 words)");
        if (mism) {
            bad_arms++;
            printf("   ref t[0..3]=%08x %08x %08x %08x  t[24]=%08x\n", h_ref[0], h_ref[1], h_ref[2], h_ref[3], h_ref[24]);
            printf("   dev t[0..3]=%08x %08x %08x %08x  t[24]=%08x\n", h_dev[0], h_dev[1], h_dev[2], h_dev[3], h_dev[24]);
        }
        cudaFree(d_v);
    }
    if (bad_arms) { printf("ABORT: %d arm(s) failed verification - no timing reported.\n", bad_arms); return 2; }

    /* ---- 2) dry run of a K-iteration verify-through-perf consistency check is implicit ---- */

    /* ---- 3) throughput ---- */
    printf("\n=== throughput: %d blocks x %d threads x %u rows each ===\n", blocks_total, tpb, iters);
    double total_rows = (double)blocks_total * tpb * (double)iters;
    double limb_macs = total_rows * (double)NL;   /* 1 limb-MAC = 1 multiply-accumulate of one limb */
    for (int arm = 0; arm < 3; arm++) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        /* warm-up */
        if (arm == 0) k_perf<0><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        else if (arm == 1) k_perf<1><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        else k_perf<2><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        ck("warmup", cudaDeviceSynchronize());
        cudaEventRecord(e0);
        if (arm == 0) k_perf<0><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        else if (arm == 1) k_perf<1><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        else k_perf<2><<<blocks_total, tpb>>>(iters, seed, d_a, d_out);
        cudaEventRecord(e1);
        ck("sync", cudaEventSynchronize(e1));
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        double sec = ms / 1000.0;
        printf("arm %s : %8.2f G limb-MAC/s | %8.2f G rows/s | %6.2f ns/row | implied mul(48 rows)=%6.0f ns\n",
               ARMNAME[arm], limb_macs / sec / 1e9, total_rows / sec / 1e9, sec / total_rows * 1e9,
               sec / total_rows * 48.0 * 1e9);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    printf("\nnote: arms WIDE/SPLIT6 use volatile asm (scheduler constrained by construction);\n"
           "      arm C is plain C - the comparison includes that freedom, by design.\n"
           "      SASS mix per arm: see the cuobjdump awk one-liner in the file header.\n");
    cudaFree(d_a); cudaFree(d_out);
    free(h_ref); free(h_dev);
    return 0;
}
