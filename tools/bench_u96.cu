/* bench_u96.cu - gate + A/B for OUR wall-band test (tools/p0_test_u96.cuh)
 * against perig (tools/perig.cuh) and GMP.
 *
 * GATES (they must pass before any speed number means anything):
 *   gate  : exact 2^((n-1)/2) mod n against mpz_powm on random odd n in the
 *           whole domain [2^95, 2^96 - 2^79 - 1] plus hand-picked edge cases,
 *           on the GPU and on the host path.
 *   verd  : verdict parity vs perig on consecutive odd candidates (any count,
 *           processed in chunks) at a chosen high word.
 *   sweep : verdict parity over many high words spread across the domain,
 *           including both limits, each with a random low part.
 * Then: ns/test of both arms (time).
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 -Itools \
 *            tools/bench_u96.cu -o bin/bench_u96 -lgmp
 * Run:   bin/bench_u96 gate  [N=200000]
 *        bin/bench_u96 verd  [N=4194304] [Ahi=4282402316] [lowbase]
 *        bin/bench_u96 sweep [chunks=64] [N=1048576]
 *        bin/bench_u96 time  [N=1048576] [iters=20] [Ahi=4282402316]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>
#include <cuda_runtime.h>

#include "p0_test_u96.cuh"   /* ours */
#include "perig.cuh"         /* production reference */

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned char u8;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static const u32 AHI_WALL = 4282402316u;                    /* ~7.9e28 */
static const u32 AHI_MIN = 0x80000000u;                     /* n >= 2^95 */
static const u32 AHI_MAX = 0xffffffffu - 0x8000u;           /* last Ahi with n <= 2^96 - 2^79 - 1 */

__global__ static void k_u96(const u64 *lo, u32 hi, u8 *res, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = p0u96::test_hi(lo[i], hi) ? 1 : 0;
}
__global__ static void k_perig(const u64 *lo, u32 hi, u8 *res, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    res[i] = ciosFermatTest128_hi(lo[i], (u64)hi) ? 1 : 0;
}
__global__ static void k_res(const u32 *nl, u32 *out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 r[3];
    p0u96::test(nl[3 * i], nl[3 * i + 1], nl[3 * i + 2], r);
    out[3 * i] = r[0]; out[3 * i + 1] = r[1]; out[3 * i + 2] = r[2];
}

static u64 rs = 88172645463325252ULL;
static u64 rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static void mpz_from3(mpz_t z, u32 a0, u32 a1, u32 a2) {
    mpz_set_ui(z, a2); mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, a1); mpz_mul_2exp(z, z, 32);
    mpz_add_ui(z, z, a0);
}
static void to3(const mpz_t z, u32 *o) {
    mpz_t t; mpz_init_set(t, z);
    for (int i = 0; i < 3; i++) { o[i] = (u32)mpz_get_ui(t) & 0xffffffffu; mpz_fdiv_q_2exp(t, t, 32); }
    mpz_clear(t);
}
static void ref_resid(u32 n0, u32 n1, u32 n2, u32 *r) {
    mpz_t n, e, b, y;
    mpz_inits(n, e, b, y, NULL);
    mpz_from3(n, n0, n1, n2);
    mpz_sub_ui(e, n, 1); mpz_fdiv_q_2exp(e, e, 1);
    mpz_set_ui(b, 2);
    mpz_powm(y, b, e, n);
    to3(y, r);
    mpz_clears(n, e, b, y, NULL);
}
static bool ref_verdict(const u32 *n, const u32 *ref) {
    bool leg = ((n[0] >> 1) ^ (n[0] >> 2)) & 1;
    mpz_t nz, y; mpz_inits(nz, y, NULL);
    mpz_from3(nz, n[0], n[1], n[2]); mpz_from3(y, ref[0], ref[1], ref[2]);
    bool want;
    if (leg) { mpz_add_ui(y, y, 1); want = mpz_cmp(y, nz) == 0; }
    else want = (mpz_cmp_ui(y, 1) == 0);
    mpz_clears(nz, y, NULL);
    return want;
}
static void rand_n(u32 *n) {
    for (;;) {
        u64 a = rnd(), b = rnd();
        u32 n2 = (u32)(b >> 32) | 0x80000000u;
        if (n2 > AHI_MAX) continue;
        n[0] = (u32)a | 1u; n[1] = (u32)(a >> 32); n[2] = n2;
        return;
    }
}

static int cmd_gate(u32 N) {
    u32 *hn = (u32 *)malloc(12ull * N), *hr = (u32 *)malloc(12ull * N);
    for (u32 i = 0; i < N; i++) rand_n(hn + 3 * i);
    const u32 edge[][3] = {
        {1, 0, 0x80000000u}, {3, 0, 0x80000000u}, {0xffffffffu, 0xffffffffu, AHI_MAX},
        {0xfffffffdu, 0xffffffffu, AHI_MAX}, {1, 0, AHI_MAX}, {0x1234567u, 0, AHI_MAX},
        {0xffffffffu, 0xffffffffu, 0x80000000u}, {0x7fffffffu, 0x80000000u, 0x80000000u},
        {1, 1, 0x80000001u}, {0xffffffffu, 0, AHI_MAX - 1}, {1, 0, 0x8fffffffu},
        {0xffffffffu, 0xffffffffu, 0x8fffffffu}, {1, 0, 0x90000000u}, {5, 5, 0xf0000000u},
    };
    for (u32 i = 0; i < sizeof(edge) / sizeof(edge[0]) && i < N; i++) {
        hn[3 * i] = edge[i][0]; hn[3 * i + 1] = edge[i][1]; hn[3 * i + 2] = edge[i][2];
    }
    u32 *dn, *dr;
    CK(cudaMalloc(&dn, 12ull * N)); CK(cudaMalloc(&dr, 12ull * N));
    CK(cudaMemcpy(dn, hn, 12ull * N, cudaMemcpyHostToDevice));
    k_res<<<(N + 127) / 128, 128>>>(dn, dr, N);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hr, dr, 12ull * N, cudaMemcpyDeviceToHost));
    u32 bad_gpu = 0, bad_host = 0, bad_verdict = 0;
    for (u32 i = 0; i < N; i++) {
        u32 *n = hn + 3 * i, ref[3], hh[3];
        ref_resid(n[0], n[1], n[2], ref);
        if (memcmp(ref, hr + 3 * i, 12)) {
            if (bad_gpu < 5) printf("GPU mismatch n=%08x%08x%08x ref=%08x%08x%08x got=%08x%08x%08x\n",
                n[2], n[1], n[0], ref[2], ref[1], ref[0], hr[3 * i + 2], hr[3 * i + 1], hr[3 * i]);
            bad_gpu++;
        }
        if (i < 200000) {
            bool v = p0u96::test(n[0], n[1], n[2], hh);
            if (memcmp(ref, hh, 12)) bad_host++;
            if (v != ref_verdict(n, ref)) bad_verdict++;
        }
    }
    printf("gate N=%u : gpu residue mismatches=%u  host mismatches=%u  verdict mismatches=%u\n",
           N, bad_gpu, bad_host, bad_verdict);
    return (bad_gpu + bad_host + bad_verdict) != 0;
}

/* verdict parity vs perig on count consecutive odd candidates from base */
static int verd_chunk(u32 N, u32 hi, u64 base, u64 *prime_u, u64 *prime_p, u64 *diff) {
    static u64 *hlo = 0; static u64 *dlo; static u8 *d1, *d2, *h1, *h2; static u32 cap = 0;
    if (cap < N) {
        free(hlo); free(h1); free(h2);
        if (cap) { cudaFree(dlo); cudaFree(d1); cudaFree(d2); }
        hlo = (u64 *)malloc(8ull * N); h1 = (u8 *)malloc(N); h2 = (u8 *)malloc(N);
        CK(cudaMalloc(&dlo, 8ull * N)); CK(cudaMalloc(&d1, N)); CK(cudaMalloc(&d2, N));
        cap = N;
    }
    for (u32 i = 0; i < N; i++) hlo[i] = (base | 1ULL) + 2ULL * i;
    CK(cudaMemcpy(dlo, hlo, 8ull * N, cudaMemcpyHostToDevice));
    k_u96<<<(N + 127) / 128, 128>>>(dlo, hi, d1, N);
    k_perig<<<(N + 127) / 128, 128>>>(dlo, hi, d2, N);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h1, d1, N, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h2, d2, N, cudaMemcpyDeviceToHost));
    for (u32 i = 0; i < N; i++) {
        *diff += h1[i] != h2[i]; *prime_u += h1[i]; *prime_p += h2[i];
        if (h1[i] != h2[i] && *diff <= 5)
            printf("MISMATCH hi=%u lo=%llu u96=%d perig=%d\n", hi, (unsigned long long)hlo[i], h1[i], h2[i]);
    }
    return 0;
}

static int cmd_verd(u64 N, u32 hi, u64 base) {
    u64 pu = 0, pp = 0, diff = 0, done = 0;
    const u32 CH = 1u << 22;
    while (done < N) {
        u32 n = (u32)((N - done) < CH ? (N - done) : CH);
        /* the low word must not wrap: base + 2N stays below 2^64 */
        verd_chunk(n, hi, base + 2ULL * done, &pu, &pp, &diff);
        done += n;
    }
    printf("verd N=%llu hi=%u: primes u96=%llu perig=%llu diff=%llu\n",
           (unsigned long long)N, hi, (unsigned long long)pu, (unsigned long long)pp,
           (unsigned long long)diff);
    return diff != 0;
}

static int cmd_sweep(u32 chunks, u32 N) {
    u64 pu = 0, pp = 0, diff = 0;
    for (u32 c = 0; c < chunks; c++) {
        u32 hi;
        if (c == 0) hi = AHI_MIN;
        else if (c == 1) hi = AHI_MAX;
        else if (c == 2) hi = AHI_MAX - 1;
        else if (c == 3) hi = AHI_WALL;
        else hi = AHI_MIN + (u32)(rnd() % ((u64)AHI_MAX - AHI_MIN + 1));
        u64 base = rnd() & ~0xffffffffffffULL;     /* keep headroom: no low-word wrap */
        base = (base >> 1) | 1ULL;
        verd_chunk(N, hi, base, &pu, &pp, &diff);
    }
    printf("sweep chunks=%u N=%u: primes u96=%llu perig=%llu diff=%llu\n",
           chunks, N, (unsigned long long)pu, (unsigned long long)pp, (unsigned long long)diff);
    return diff != 0;
}

template <class K>
static double bench(K k, const u64 *dlo, u32 hi, u8 *dres, u32 n, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<(n + 127) / 128, 128>>>(dlo, hi, dres, n); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < iters; r++) k<<<(n + 127) / 128, 128>>>(dlo, hi, dres, n);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    return (double)ms * 1e-3 / iters;
}

static int cmd_time(u32 N, int iters, u32 hi) {
    u64 *hlo = (u64 *)malloc(8ull * N);
    const u64 base = 0x1234567800000000ULL | 0x9abcdef1ULL;
    for (u32 i = 0; i < N; i++) hlo[i] = (base | 1ULL) + 2ULL * i;
    u64 *dlo; u8 *dres;
    CK(cudaMalloc(&dlo, 8ull * N)); CK(cudaMalloc(&dres, N));
    CK(cudaMemcpy(dlo, hlo, 8ull * N, cudaMemcpyHostToDevice));
    double bu = 1e9, bp = 1e9;
    for (int r = 0; r < 4; r++) {
        double t = bench(k_u96, dlo, hi, dres, N, iters); if (t < bu) bu = t;
        t = bench(k_perig, dlo, hi, dres, N, iters); if (t < bp) bp = t;
    }
    printf("hi=%u u96  : %.3f ms  %.3f ns/test\n", hi, bu * 1e3, 1e9 * bu / N);
    printf("hi=%u perig: %.3f ms  %.3f ns/test\n", hi, bp * 1e3, 1e9 * bp / N);
    printf("perig/u96 = %.3fx\n", bp / bu);
    return 0;
}

int main(int argc, char **argv) {
    const char *c = argc > 1 ? argv[1] : "time";
    if (!strcmp(c, "gate"))
        return cmd_gate(argc > 2 ? (u32)strtoul(argv[2], 0, 10) : 200000u);
    if (!strcmp(c, "verd")) {
        u64 N = argc > 2 ? strtoull(argv[2], 0, 10) : (1ull << 22);
        u32 hi = argc > 3 ? (u32)strtoul(argv[3], 0, 10) : AHI_WALL;
        u64 base = argc > 4 ? strtoull(argv[4], 0, 10) : (0x1234567800000000ULL | 0x9abcdef1ULL);
        if (hi < AHI_MIN || hi > AHI_MAX) { printf("hi outside the u96 domain\n"); return 2; }
        return cmd_verd(N, hi, base);
    }
    if (!strcmp(c, "sweep"))
        return cmd_sweep(argc > 2 ? (u32)strtoul(argv[2], 0, 10) : 64u,
                         argc > 3 ? (u32)strtoul(argv[3], 0, 10) : (1u << 20));
    u32 N = argc > 2 ? (u32)strtoul(argv[2], 0, 10) : (1u << 20);
    int iters = argc > 3 ? atoi(argv[3]) : 20;
    u32 hi = argc > 4 ? (u32)strtoul(argv[4], 0, 10) : AHI_WALL;
    return cmd_time(N, iters, hi);
}
