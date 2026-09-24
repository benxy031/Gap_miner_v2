/*
 * Prototype + harness: symmetric Montgomery squaring for CGBN.
 *
 * MEASURED VERDICT (2026-09-24, RTX 3070, sm_86): this shared-memory
 * product + cgbn_mont_reduce_wide architecture is **6-9x SLOWER** than CGBN's
 * `cgbn_mont_sqr` (= mont_mul(a,a)) at every width (384/512/768/1024/1280/
 * 1536/1792/2048 bits, ratio 0.11-0.17), while doing only ~25% fewer
 * multiplies.  The overhead is structural, not a bug to tune away:
 *   - two layout transposes per squaring (stage a/n, load the product back),
 *   - a serial carry ripple over 2W product words,
 *   - per-output-word inner loops with lane-imbalanced trip counts.
 * CGBN's in-register interleaved CIOS (core_mont_xmad.cu) avoids all three,
 * and its interleaved reduction fold is exactly what prevents the triangular
 * trick from being applied without a full rewrite of the row loop.  Together
 * with the ncu profile (compute 46%, ALU 29%, occupancy 22%, "0.3 waves" -
 * i.e. latency-bound, NOT op-bound) this says the remaining kernel lever is
 * concurrency/occupancy, not fewer multiplies.  Kept as a recorded dead end
 * (do not repeat it) rather than deleted.
 *
 * WHY the idea was tried: CGBN's `mont_sqr` is literally
 * `mont_mul(r, a, a, n, np0)` (cgbn/include/cgbn/impl_cuda.cu:1032) and CGBN
 * contains no squaring specialization (the only "sqr" symbols are sqrt).
 * A squaring needs only the upper triangle of the product (i <= j, doubling
 * the off-diagonal terms), i.e. W(W+1)/2 word-products instead of W^2, and our
 * base-2 Miller-Rabin inner loop is ~100% squarings.
 *
 * HOW: the product is built in *shared memory* (no shuffles, no register
 * pressure -- the failure mode that killed the earlier in-register SOS
 * attempt), then handed to CGBN's own `mont_reduce_wide` so that the
 * Montgomery reduction stays CGBN's verified code:
 *
 *   1. stage a[] and n[] into shared memory in plain global-word order
 *   2. each lane computes the raw 64-bit contributions of a contiguous stripe
 *      of product words, using only i <= j pairs and doubling off-diagonals
 *   3. one serial carry ripple over the 2W product words
 *   4. load the product into a cgbn_wide_t and call cgbn_mont_reduce_wide
 *
 * Layout facts used (from CGBN's core_mul_xmad.cu / core_mont_xmad.cu):
 *   - a cgbn_t<BITS,TPI> holds LIMBS = BITS/(32*TPI) words per lane and word
 *     with global index g lives in lane (g % TPI) at position (g / TPI).
 *   - for every width we instantiate (384/512/768/1024/1280/1536/1792/2048
 *     bits at TPI 4 or 8) BITS is an exact multiple of 32*TPI, so PADDING == 0
 *     and the mapping above is exact (static_assert'ed below).
 *
 * Build:  file is standalone (no project headers) :
 *   nvcc -O3 -arch=sm_86 -std=c++17 -I tools/cgbn/include -o bin/bench_sym_sqr \
 *        tools/bench_sym_sqr.cu
 * Run:    ./bin/bench_sym_sqr [bits] [tpi] [candidates] [squarings]
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include <gmp.h>
#include "cgbn/cgbn.h"

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t err_ = (x);                                              \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err_), __FILE__, __LINE__);      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

/* ---------------------------------------------------------------------------
 * Parameters (mirror gpu_fermat.cu's CgbnFermatParams)
 * ------------------------------------------------------------------------- */
template<uint32_t TPI_VAL>
struct SymParams {
    static const uint32_t TPB           = 128;
    static const uint32_t MAX_ROTATION  = 4;
    static const uint32_t SHM_LIMIT     = 0;
    static const bool     CONSTANT_TIME = false;
    static const uint32_t TPI           = TPI_VAL;
};

/* ---------------------------------------------------------------------------
 * The symmetric squaring helper
 * ------------------------------------------------------------------------- */

template<class env_t, uint32_t TPI>
struct SymSqr {
    static const uint32_t BITS = env_t::BITS;
    static const uint32_t W    = BITS / 32u;          /* product words per side */
    static const uint32_t LIMBS= W / TPI;             /* words per lane        */
    static const uint32_t IPB  = 128u / TPI;          /* instances per block   */

    static_assert(BITS % (32u * TPI) == 0u,
                  "sym_mont_sqr assumes CGBN PADDING == 0 (exact word split)");

    /* Shared layout per instance: [0, 2W) product raw sums (u64), 2W words;
       [2W, 3W) staged a, 32-bit words; [3W, 4W) staged n.  The product sums
       are u64 so the serial ripple is trivial. */
    static const uint32_t U64_PER_INST = 2u * W;
    static const uint32_t U32_PER_INST = 2u * W;

    /* Raw (unrippled) accumulation of one product word.
       For output word o:
         - pairs with i + j == o   contribute their HIGH 32 bits
         - pairs with i + j == o-1 contribute their LOW  32 bits
       counting each unordered pair once and doubling it when i != j. */
    __device__ static __forceinline__
    uint64_t word_contrib(const uint32_t *a, uint32_t o)
    {
        uint64_t raw = 0;

        /* i + j == o  (high halves) */
        {
            uint32_t i_lo = (o >= (W - 1u)) ? (o - (W - 1u)) : 0u;
            uint32_t i_hi = o / 2u;                    /* i <= j = o - i */
            for (uint32_t i = i_lo; i <= i_hi; i++) {
                uint32_t j = o - i;
                if (j >= W) continue;                  /* o can exceed 2W-2 */
                uint64_t prod = (uint64_t)a[i] * (uint64_t)a[j];
                uint32_t lo = (uint32_t)prod;
                uint32_t hi = (uint32_t)(prod >> 32);
                if (i != j) { hi = (hi << 1) | (lo >> 31); }
                raw += hi;
            }
        }
        /* i + j == o - 1  (low halves) */
        if (o > 0u) {
            uint32_t s = o - 1u;
            uint32_t i_lo = (s >= (W - 1u)) ? (s - (W - 1u)) : 0u;
            uint32_t i_hi = s / 2u;
            for (uint32_t i = i_lo; i <= i_hi; i++) {
                uint32_t j = s - i;
                if (j >= W) continue;
                uint64_t prod = (uint64_t)a[i] * (uint64_t)a[j];
                uint32_t lo = (uint32_t)prod;
                if (i != j) lo <<= 1;
                raw += lo;
            }
        }
        return raw;
    }

    /* r = a * a * R^-1 mod n  (fully reduced, same contract as
       cgbn_mont_reduce_wide / cgbn_mont_sqr-with-repair). */
    __device__ static __forceinline__
    void mont_sqr_sym(env_t &env, typename env_t::cgbn_t &r,
                      const typename env_t::cgbn_t &a,
                      const typename env_t::cgbn_t &n,
                      uint32_t np0,
                      uint64_t *sh_raw, uint32_t *sh_a, uint32_t *sh_n)
    {
        const uint32_t lane = threadIdx.x & (TPI - 1u);
        const uint32_t mask = ((1u << TPI) - 1u) << (threadIdx.x & ~(TPI - 1u));

        /* 1) stage a and n by global word index */
        #pragma unroll
        for (uint32_t l = 0; l < LIMBS; l++) {
            sh_a[l * TPI + lane] = a._limbs[l];
            sh_n[l * TPI + lane] = n._limbs[l];
        }
        __syncwarp(mask);

        /* 2) stripe of product words per lane: contiguous, one carry ripple
              at the end (serial over the whole product, see 3) */
        const uint32_t stripe = (2u * W + TPI - 1u) / TPI;
        const uint32_t o_begin = lane * stripe;
        const uint32_t o_end = (o_begin + stripe < 2u * W) ? (o_begin + stripe)
                                                          : (2u * W);
        for (uint32_t o = o_begin; o < o_end; o++) {
            sh_raw[o] = word_contrib(sh_a, o);
        }
        __syncwarp(mask);

        /* 3) serial carry ripple over the 2W product words (lane 0).
              Sequential but tiny compared with the multiply work. */
        if (lane == 0u) {
            uint64_t carry = 0;
            for (uint32_t o = 0; o < 2u * W; o++) {
                uint64_t acc = sh_raw[o] + carry;
                sh_raw[o] = acc & 0xffffffffULL;   /* store the resolved word */
                carry = acc >> 32;
            }
            /* The product of two BITS-bit numbers fits exactly in 2W words, so
               the final carry must be 0; expose a violation instead of writing
               past the instance's shared slot (that was an out-of-bounds write
               in the first revision of this prototype). */
            if (carry != 0)
                sh_raw[0] = 0xdeadbeefULL;         /* flagged below as a failure */
        }
        __syncwarp(mask);

        /* 4) load into CGBN's wide layout and let CGBN reduce */
        typename env_t::cgbn_wide_t wide;
        #pragma unroll
        for (uint32_t l = 0; l < LIMBS; l++) {
            wide._low._limbs[l]  = (uint32_t)sh_raw[l * TPI + lane];
            wide._high._limbs[l] = (uint32_t)sh_raw[W + l * TPI + lane];
        }
        cgbn_mont_reduce_wide(env, r, wide, n, np0);
    }
};

/* ---------------------------------------------------------------------------
 * Kernels: baseline (cgbn_mont_sqr) vs symmetric, K squarings per candidate
 * ------------------------------------------------------------------------- */

template<uint32_t BITS, uint32_t TPI_VAL, bool SYM>
__global__ __launch_bounds__(128)
void sqr_kernel(cgbn_mem_t<BITS> *cands, cgbn_mem_t<BITS> *out,
                uint32_t n_cand, uint32_t rounds)
{
    typedef cgbn_context_t<TPI_VAL, SymParams<TPI_VAL>> ctx_t;
    typedef cgbn_env_t<ctx_t, BITS>                     env_t;
    typedef typename env_t::cgbn_t                      bn_t;
    typedef SymSqr<env_t, TPI_VAL>                      S;

    constexpr uint32_t W = BITS / 32u;
    constexpr uint32_t IPB = 128u / TPI_VAL;

    __shared__ uint64_t sh_raw[IPB * 2u * W];
    __shared__ uint32_t sh_a[IPB * W];
    __shared__ uint32_t sh_n[IPB * W];

    const uint32_t inst = (blockIdx.x * blockDim.x + threadIdx.x) / TPI_VAL;
    const uint32_t slot = inst % IPB;
    if (inst >= n_cand) return;

    ctx_t ctx(cgbn_no_checks);
    env_t env(ctx);
    bn_t  a, N;

    cgbn_load(env, N, cands + inst);
    if ((cgbn_extract_bits_ui32(env, N, 0, 1) & 1u) == 0u) {
        cgbn_set_ui32(env, a, 0);
        cgbn_store(env, out + inst, a);
        return;
    }
    cgbn_set_ui32(env, a, 2);
    uint32_t np0 = cgbn_bn2mont(env, a, a, N);   /* a = Mont(2), np0 */

    if (SYM) {
        for (uint32_t k = 0; k < rounds; k++) {
            S::mont_sqr_sym(env, a, a, N, np0,
                            sh_raw + slot * 2u * W,
                            sh_a + slot * W,
                            sh_n + slot * W);
        }
    } else {
        for (uint32_t k = 0; k < rounds; k++) {
            cgbn_mont_sqr(env, a, a, N, np0);
            /* gpu_fermat.cu repairs CGBN's lazy reduction the same way, so
               both variants must end up fully reduced for a fair compare. */
            if (cgbn_compare(env, a, N) >= 0)
                cgbn_sub(env, a, a, N);
        }
    }
    cgbn_store(env, out + inst, a);
}

/* ---------------------------------------------------------------------------
 * Host driver: build random odd moduli, run both variants, compare, time.
 * ------------------------------------------------------------------------- */

template<uint32_t BITS, uint32_t TPI_VAL>
static int run_case(const char *label, uint32_t n_cand, uint32_t rounds)
{
    typedef cgbn_mem_t<BITS> mem_t;
    const size_t words = (size_t)BITS / 32u;
    const size_t bytes = n_cand * words * sizeof(uint32_t);

    std::vector<uint32_t> host_in(n_cand * words), host_base(n_cand * words),
                          host_sym(n_cand * words);

    /* Deterministic odd candidates with the top bit set (GMP), so both kernels
       see identical input and a mismatch cannot be blamed on the data. */
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 20260924u + BITS);
    mpz_t v;
    mpz_init(v);
    for (uint32_t c = 0; c < n_cand; c++) {
        do {
            mpz_urandomb(v, rs, BITS);
            mpz_setbit(v, BITS - 1);
            mpz_setbit(v, 0);
        } while (mpz_probab_prime_p(v, 1) != 0);   /* composite is fine and faster */
        size_t cnt = 0;
        mpz_export(&host_in[c * words], &cnt, -1, sizeof(uint32_t), 0, 0, v);
    }
    mpz_clear(v);
    gmp_randclear(rs);

    mem_t *d_in = nullptr, *d_base = nullptr, *d_sym = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_base, bytes));
    CUDA_CHECK(cudaMalloc(&d_sym, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, host_in.data(), bytes, cudaMemcpyHostToDevice));

    const uint32_t tpb = 128;
    const uint32_t blocks = (n_cand * TPI_VAL + tpb - 1u) / tpb;

    /* warm-up + correctness run (small) */
    sqr_kernel<BITS, TPI_VAL, false><<<blocks, tpb>>>(d_in, d_base, n_cand, rounds);
    sqr_kernel<BITS, TPI_VAL, true ><<<blocks, tpb>>>(d_in, d_sym, n_cand, rounds);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(host_base.data(), d_base, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_sym.data(), d_sym, bytes, cudaMemcpyDeviceToHost));

    size_t mismatches = 0;
    for (uint32_t c = 0; c < n_cand; c++) {
        if (std::memcmp(&host_base[c * words], &host_sym[c * words],
                        words * sizeof(uint32_t)) != 0) {
            if (mismatches < 2) {
                std::printf("  mismatch at candidate %u\n", c);
            }
            mismatches++;
        }
    }

    /* timing */
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    const int iters = 20;

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++)
        sqr_kernel<BITS, TPI_VAL, false><<<blocks, tpb>>>(d_in, d_base, n_cand, rounds);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_base = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_base, e0, e1));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++)
        sqr_kernel<BITS, TPI_VAL, true ><<<blocks, tpb>>>(d_in, d_sym, n_cand, rounds);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms_sym = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_sym, e0, e1));

    const double cands_per_s_base = (double)n_cand * iters / (ms_base / 1000.0);
    const double cands_per_s_sym  = (double)n_cand * iters / (ms_sym / 1000.0);

    std::printf("%-28s rounds=%-4u cand=%-6u  base %8.0f cand/s (%6.3f ms)  "
                "sym %8.0f cand/s (%6.3f ms)  ratio %.3f  %s\n",
                label, rounds, n_cand, cands_per_s_base, ms_base,
                cands_per_s_sym, ms_sym, cands_per_s_sym / cands_per_s_base,
                mismatches ? "MISMATCH!" : "exact");

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_sym));
    return mismatches ? 1 : 0;
}

int main(int argc, char **argv)
{
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount);

    int failures = 0;
    const uint32_t rounds = (argc > 1) ? (uint32_t)atoi(argv[1]) : 128;
    const uint32_t cand   = (argc > 2) ? (uint32_t)atoi(argv[2]) : 4096;

    /* Every width the miner instantiates (see gpu_fermat.cu's dispatch). */
    failures += run_case< 384, 4>("384-bit  (AL=6,  TPI=4)", cand, rounds);
    failures += run_case< 512, 8>("512-bit  (AL=8,  TPI=8)", cand, rounds);
    failures += run_case< 768, 8>("768-bit  (AL=12, TPI=8)", cand, rounds);
    failures += run_case<1024, 8>("1024-bit (AL=16, TPI=8)", cand, rounds);
    failures += run_case<1280, 8>("1280-bit (AL=20, TPI=8)", cand, rounds);
    failures += run_case<1536, 8>("1536-bit (AL=24, TPI=8)", cand, rounds);
    failures += run_case<1792, 8>("1792-bit (AL=28, TPI=8)", cand, rounds);
    failures += run_case<2048, 8>("2048-bit (AL=32, TPI=8)", cand, rounds);

    if (failures) {
        std::printf("FAILED: %d width(s) mismatch\n", failures);
        return 1;
    }
    std::printf("PASS: symmetric squaring is bit-exact at every width\n");
    return 0;
}
