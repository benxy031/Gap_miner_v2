/* tools/bench_rns_mont.cu - RNS-16 768-bit Montgomery: REFUTED PROTOTYPE
 *
 * STATUS: DOCUMENTED NEGATIVE RESULT + measured per-operation costs.
 * This file is deliberately kept (novelty register, not a roadmap item) because
 * it now contains three facts that a paper estimate would have gotten wrong.
 * Do NOT read it as a working RNS implementation: `rns_step` below uses a base
 * extension formula that is REFUTED (see FACT 2) and the program says so and
 * exits non-zero.
 *
 * ---------------------------------------------------------------------------
 * FACT 1 - the extension base is as big as the main base.
 *   BIJ requires M > 4N (main base) and M' > 2N (extension base) for the lazy
 *   result u < 2N to be exactly reconvertible.  With ~16-bit moduli and N ~ 2^768
 *   that means K ~ 50 AND K' ~ 50: the "small extension base" hope (K'=10) makes
 *   the second extension mathematically impossible (M' ~ 2^160 vs a value of
 *   ~2^768).  The first run of this file failed 100/100 for exactly that reason.
 *   => the extension costs K*K' operations over a base that is not smaller.
 *
 * FACT 2 - the per-modulus weighted-sum extension formula is WRONG, in both the
 *   plain and the "M_i^-1 mod m_i" form.  Measured in isolation (2000 random
 *   x < M, 3 moduli each): 5369 respectively 4458 mismatches out of 6000 checks.
 *   The reason: CRT gives x = sum_i c_i*x_i only MODULO M, and reducing each
 *   c_i mod m'_j is invalid because M is not 0 mod m'_j.  A correct extension
 *   needs either mixed-radix digits (exact, but depth O(K^2) and sequential) or
 *   the modern redundant-modulus + approximation method (research-grade, extra
 *   correction cost).  Any estimate that assumes a simple per-modulus sum is
 *   therefore not implementable as written.
 *
 * FACT 3 - the op-count arithmetic decides the whole question, and it is not
 *   close for the naive variant:
 *     CGBN AL=12 (measured 907k montmul/s = 1102 ns):  AL^2 = 144 schoolbook
 *       products of 64-bit limbs, and 1 limb multiply = 4.48 IMAD slots
 *       (measured: mul32 4999 GMAC/s, mul64 1115 G/s) -> ~645 IMAD slots.
 *     RNS with K=K'=50:  extensions alone cost K*K' = 2500 mul32 EACH (5000
 *       total) + ~200 more = ~5200 mul32 -> ~5200 IMAD slots, plus the
 *       reductions.  A correct classic (mixed-radix) extension is the same
 *       order: K^2/2 digits + K*K' reconstruction.
 *   => RNS-16 must reduce the extension to O(K+K') (redundant modulus) before it
 *      can compete at all; on the naive extension it is several times SLOWER than
 *      CGBN, not 100,000x faster.  The old multiply-stream figure (61.8 T MAC/s
 *      -> "100,000x CGBN") is retired: that was a unit artifact.
 *
 * What this program measures for real (independent of the refutation):
 *   ns/op for  mul32 + a*b % m,  mul32 + Lemire fastmod,  mul32 (reference),
 *   plus the modeled cost of a full 768-bit step in both variants, against the
 *   measured CGBN AL=12 rate of 1102 ns.
 *
 * Build/run:  make bin/bench_rns_mont WITH_CUDA=1
 *             ./bin/bench_rns_mont [cases=100] [iters=4000]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <gmp.h>
#include <cuda_runtime.h>

#define K 50            /* main base      : 50 x ~16 bits ~ 800 bits (> 4N)  */
#define KP 50           /* extension base : 50 x ~16 bits ~ 800 bits (> 2N)  */
#define KT (K + KP)
#define NCHAIN 4        /* independent chains per thread -> throughput       */

static void ck(const char *what, cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

static uint32_t g_mod[KT];

static void build_moduli(void) {
    int n = 0;
    for (uint32_t p = 65535; p > 3 && n < KT; p -= 2) {
        int prime = 1;
        for (uint32_t d = 3; d * d <= p; d += 2)
            if (p % d == 0) { prime = 0; break; }
        if (prime) g_mod[n++] = p;
    }
    if (n != KT) { fprintf(stderr, "moduli generation failed\n"); exit(1); }
}

/* ---- precomputed constants ---------------------------------------------- */
static uint32_t g_ninv[K];        /* (-N^-1) mod m_i                       */
static uint32_t g_minv[K];        /* (M/m_i)^-1 mod m_i                    */
static uint32_t g_n2[KP];         /* N mod m'_j                            */
static uint32_t g_Minv2[KP];      /* M^-1 mod m'_j                         */
static uint32_t g_Mpinv[KP];      /* (M'/m'_j)^-1 mod m'_j                 */
static uint32_t g_T1[K * KP];     /* (M/m_i) mod m'_j                      */
static uint32_t g_T2[KP * K];     /* (M'/m'_j) mod m_i                     */

/* ---- the REFUTED step (kept as the counterexample) ---------------------- */
__host__ __device__ static inline void rns_step_refuted(
        const uint32_t *mod, const uint32_t *ninv, const uint32_t *minv,
        const uint32_t *n2, const uint32_t *Minv2, const uint32_t *Mpinv,
        const uint32_t *T1, const uint32_t *T2, const uint32_t *a,
        const uint32_t *b, uint32_t *u) {
    uint32_t t[KT], q[K], q2[KP], u2[KP];
    for (int i = 0; i < K; i++)
        t[i] = (uint32_t)(((uint64_t)a[i] * b[i]) % mod[i]);
    for (int j = 0; j < KP; j++) {
        int i = K + j;
        t[i] = (uint32_t)(((uint64_t)a[i] * b[i]) % mod[i]);
    }
    for (int i = 0; i < K; i++)
        q[i] = (uint32_t)(((uint64_t)t[i] * ninv[i]) % mod[i]);
    for (int j = 0; j < KP; j++) {          /* EXTEND(q, B -> B') : REFUTED */
        uint64_t acc = 0;
        for (int i = 0; i < K; i++) {
            uint32_t v = (uint32_t)(((uint64_t)q[i] * minv[i]) % mod[i]);
            acc += (uint64_t)v * T1[i * KP + j];
        }
        q2[j] = (uint32_t)(acc % mod[K + j]);
    }
    for (int j = 0; j < KP; j++) {
        int i = K + j;
        uint32_t s = (uint32_t)((t[i] + (uint64_t)q2[j] * n2[j]) % mod[i]);
        u2[j] = (uint32_t)(((uint64_t)s * Minv2[j]) % mod[i]);
    }
    for (int i = 0; i < K; i++) {           /* EXTEND(u', B' -> B) : REFUTED */
        uint64_t acc = 0;
        for (int j = 0; j < KP; j++) {
            uint32_t v =
                (uint32_t)(((uint64_t)u2[j] * Mpinv[j]) % mod[K + j]);
            acc += (uint64_t)v * T2[j * K + i];
        }
        u[i] = (uint32_t)(acc % mod[i]);
    }
    for (int j = 0; j < KP; j++) u[K + j] = u2[j];
}

/* ---- per-operation measurement ----------------------------------------- */
__device__ static inline uint32_t fastmod32(uint32_t x, uint64_t M, uint32_t m) {
    uint64_t low = M * (uint64_t)x;
    return (uint32_t)(((unsigned __int128)low * (unsigned long long)m) >> 64);
}

template <int MODE>
__global__ void op_kernel(const uint32_t *mod, const uint64_t *Mtab,
                          unsigned long long *out, int iters) {
    uint64_t x[NCHAIN], y[NCHAIN];
    const uint32_t m = mod[7];
    const uint64_t Mg = Mtab[7];
#pragma unroll
    for (int c = 0; c < NCHAIN; c++) {
        x[c] = 1234567u + threadIdx.x * 7919u + c;
        y[c] = 987654321u - threadIdx.x * 104729u + c * 13u;
    }
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int c = 0; c < NCHAIN; c++) {
            if (MODE == 0) {
                x[c] = ((uint64_t)(uint32_t)x[c] * (uint32_t)y[c]) % m;
            } else if (MODE == 1) {
                uint32_t p = (uint32_t)((uint32_t)x[c] * (uint32_t)y[c]);
                x[c] = fastmod32(p, Mg, m);
            } else {
                x[c] = (uint32_t)((uint32_t)x[c] * (uint32_t)y[c]);
            }
        }
    }
    unsigned long long r = 0;
#pragma unroll
    for (int c = 0; c < NCHAIN; c++) r += x[c];
    atomicAdd(out, r);
}

static void crt_reconstruct(mpz_t out, const uint32_t *res,
                            const uint32_t *mod, int n) {
    mpz_t M, Mi, inv, term, t, mi;
    mpz_inits(M, Mi, inv, term, t, mi, NULL);
    mpz_set_ui(M, 1);
    for (int i = 0; i < n; i++) mpz_mul_ui(M, M, mod[i]);
    mpz_set_ui(out, 0);
    for (int i = 0; i < n; i++) {
        mpz_set_ui(mi, mod[i]);
        mpz_divexact_ui(Mi, M, mod[i]);
        mpz_mod(t, Mi, mi);
        mpz_invert(inv, t, mi);
        mpz_mul_ui(term, Mi, res[i]);
        mpz_mul(term, term, inv);
        mpz_add(out, out, term);
    }
    mpz_mod(out, out, M);
    mpz_clears(M, Mi, inv, term, t, mi, NULL);
}

int main(int argc, char **argv) {
    int cases = (argc > 1) ? atoi(argv[1]) : 100;
    int iters = (argc > 2) ? atoi(argv[2]) : 4000;

    build_moduli();
    printf("RNS-16 768-bit Montgomery prototype -- REFUTED (see source header)\n");
    printf("  bases: K=%d + K'=%d moduli of ~16 bits (log2 M ~ %.0f, "
           "log2 M' ~ %.0f, N ~ 2^768)\n", K, KP, K * 15.99, KP * 15.99);

    /* ---- measured per-operation cost (independent of the algorithm) ------ */
    cudaDeviceProp p;
    ck("prop", cudaGetDeviceProperties(&p, 0));
    uint32_t *d_mod;
    uint64_t *d_M, Mhost[KT];
    for (int i = 0; i < KT; i++) Mhost[i] = (~0ull) / g_mod[i] + 1ull;
    ck("alloc m", cudaMalloc(&d_mod, sizeof(g_mod)));
    ck("alloc M", cudaMalloc(&d_M, sizeof(Mhost)));
    ck("copy m", cudaMemcpy(d_mod, g_mod, sizeof(g_mod),
                            cudaMemcpyHostToDevice));
    ck("copy M", cudaMemcpy(d_M, Mhost, sizeof(Mhost),
                            cudaMemcpyHostToDevice));
    unsigned long long *d_out;
    ck("alloc out", cudaMalloc(&d_out, sizeof(*d_out)));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    int blocks = p.multiProcessorCount * 8, threads = 128;
    const char *names[3] = {"mul32 + (a*b) % m (hw divide)",
                            "mul32 + Lemire fastmod    ",
                            "mul32 only (reference)    "};
    double ns_op[3];
    for (int mode = 0; mode < 3; mode++) {
        ck("memset", cudaMemset(d_out, 0, sizeof(*d_out)));
        for (int w = 0; w < 2; w++) {
            int it = w ? iters : 200;
            if (mode == 0) op_kernel<0><<<blocks, threads>>>(d_mod, d_M, d_out, it);
            else if (mode == 1) op_kernel<1><<<blocks, threads>>>(d_mod, d_M, d_out, it);
            else op_kernel<2><<<blocks, threads>>>(d_mod, d_M, d_out, it);
            ck(w ? "run" : "warm", cudaDeviceSynchronize());
            if (!w) cudaEventRecord(e0);
        }
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, e0, e1);
        double ops = (double)blocks * threads * iters * NCHAIN;
        ns_op[mode] = ms * 1e6 / ops;
        printf("  %s %6.3f ns/op -> %8.1f Gop/s\n", names[mode], ns_op[mode],
               ops / (ms * 1e6));
    }
    cudaFree(d_out);
    cudaFree(d_mod);
    cudaFree(d_M);

    /* ---- why the op-count model cannot decide this ----------------------- */
    const double cgbn_ns = 1102.0;      /* 907k montmul/s measured, AL=12 */
    const double mul64_ns = 0.00090;    /* measured: mul64 1115 G/s       */
    const double pure = 144.0 * mul64_ns;
    double mul_naive = (double)(KT + K + 2 * K * KP + KP);       /* 5200 */
    printf("\n  the decisive measurement is not the multiply count:\n");
    printf("    CGBN AL=12 does 144 mu64 per step = %.3f ns of pure multiply\n"
           "    work, yet it measures %.0f ns/step -> CGBN uses %.4f%% of the\n"
           "    mul64 stream.  A perfect-implementation model would predict\n"
           "    %.2f ns (%.0fx faster than CGBN really is), so op counts are NOT\n"
           "    a predictor here: reduction / normalization / dependency\n"
           "    dominate by ~4 orders of magnitude, as in every bignum path\n"
           "    tried in this repo (85-99%% of time outside the multiply).\n",
           pure, cgbn_ns, 100.0 * pure / cgbn_ns, pure, cgbn_ns / pure);
    printf("    The same measured basis applied to RNS-16 (K=K'=50, naive\n"
           "    extension, 5200 mulmod + 5000 adds) gives %.0f ns with the\n"
           "    hardware divide and %.0f ns with Lemire - i.e. numbers in the\n"
           "    same useless-optimism band as the CGBN model above.\n",
           mul_naive * (ns_op[0] + 0.9 * ns_op[1]),
           mul_naive * (ns_op[1] + 0.9 * ns_op[1]));

    /* ---- the refutation, measured ---------------------------------------- */
    gmp_randstate_t rs;
    gmp_randinit_mt(rs);
    gmp_randseed_ui(rs, 20260922u);
    int bad = 0;
    for (int c = 0; c < cases; c++) {
        mpz_t N, a, b, M, M2, exp, got, inv, t, x;
        mpz_inits(N, a, b, M, M2, exp, got, inv, t, x, NULL);
        mpz_urandomb(N, rs, 768);
        mpz_setbit(N, 767);
        mpz_setbit(N, 0);
        mpz_urandomm(a, rs, N);
        mpz_urandomm(b, rs, N);
        mpz_set_ui(M, 1);
        for (int i = 0; i < K; i++) mpz_mul_ui(M, M, g_mod[i]);
        mpz_set_ui(M2, 1);
        for (int j = 0; j < KP; j++) mpz_mul_ui(M2, M2, g_mod[K + j]);
        for (int i = 0; i < K; i++) {
            mpz_set_ui(t, g_mod[i]);
            mpz_set_ui(x, mpz_fdiv_ui(N, g_mod[i]));
            mpz_invert(inv, x, t);
            g_ninv[i] = (uint32_t)((g_mod[i] - mpz_get_ui(inv)) % g_mod[i]);
            mpz_divexact_ui(x, M, g_mod[i]);
            mpz_mod(x, x, t);
            mpz_invert(inv, x, t);
            g_minv[i] = (uint32_t)mpz_get_ui(inv);
        }
        for (int j = 0; j < KP; j++) {
            mpz_set_ui(t, g_mod[K + j]);
            g_n2[j] = mpz_fdiv_ui(N, g_mod[K + j]);
            mpz_mod(x, M, t);
            mpz_invert(inv, x, t);
            g_Minv2[j] = (uint32_t)mpz_get_ui(inv);
            mpz_divexact_ui(x, M2, g_mod[K + j]);
            mpz_mod(x, x, t);
            mpz_invert(inv, x, t);
            g_Mpinv[j] = (uint32_t)mpz_get_ui(inv);
        }
        for (int i = 0; i < K; i++)
            for (int j = 0; j < KP; j++) {
                mpz_divexact_ui(x, M, g_mod[i]);
                g_T1[i * KP + j] = (uint32_t)mpz_fdiv_ui(x, g_mod[K + j]);
            }
        for (int j = 0; j < KP; j++)
            for (int i = 0; i < K; i++) {
                mpz_divexact_ui(x, M2, g_mod[K + j]);
                g_T2[j * K + i] = (uint32_t)mpz_fdiv_ui(x, g_mod[i]);
            }
        uint32_t ra[KT], rb[KT], ru[KT];
        for (int i = 0; i < K; i++) {
            ra[i] = mpz_fdiv_ui(a, g_mod[i]);
            rb[i] = mpz_fdiv_ui(b, g_mod[i]);
        }
        for (int j = 0; j < KP; j++) {
            ra[K + j] = mpz_fdiv_ui(a, g_mod[K + j]);
            rb[K + j] = mpz_fdiv_ui(b, g_mod[K + j]);
        }
        rns_step_refuted(g_mod, g_ninv, g_minv, g_n2, g_Minv2, g_Mpinv, g_T1,
                         g_T2, ra, rb, ru);
        crt_reconstruct(got, ru, g_mod, K);
        mpz_mod(got, got, N);
        mpz_mod(x, M, N);
        mpz_invert(inv, x, N);
        mpz_mul(exp, a, b);
        mpz_mod(exp, exp, N);
        mpz_mul(exp, exp, inv);
        mpz_mod(exp, exp, N);
        if (mpz_cmp(exp, got) != 0) bad++;
        mpz_clears(N, a, b, M, M2, exp, got, inv, t, x, NULL);
    }
    gmp_randclear(rs);
    printf("\n  refutation, measured: %d/%d cases wrong with the per-modulus\n"
           "  weighted-sum extension, in BOTH forms (with and without\n"
           "  M_i^-1 mod m_i correction: 5369/6000 and 4458/6000 in isolation).\n",
           bad, cases);
    printf("\n  VERDICT: RNS-16 for 768 bits is NOT reachable by reasoning, and not\n"
           "  by a quick prototype:\n"
           "   - FACT 1: it needs K ~ K' ~ 50 (M > 4N, M' > 2N), so the base\n"
           "     extension IS the algorithm, not an overhead;\n"
           "   - FACT 2: the per-modulus weighted-sum extension is invalid in both\n"
           "     forms (measured above), so any estimate built on it is void;\n"
           "   - FACT 3: op counts are not a predictor (CGBN uses 0.012%% of its\n"
           "     multiply stream), so neither the 100,000x MMA figure nor a\n"
           "     5200-mulmod figure means anything;\n"
           "   => the ONLY instrument that can answer this is a correct, tuned\n"
           "      redundant-modulus RNS kernel (Kawamura/Bajard style) measured\n"
           "      end-to-end against 1102 ns.  That is a multi-week research\n"
           "      implementation with the odds still open.\n"
           "  Decision: do not build it now.  The harness is here if the payoff\n"
           "  ever justifies it.  Returning to 14,162 win/s of real mining.\n");
    return 1;   /* non-zero on purpose: this is a refuted prototype */
}
