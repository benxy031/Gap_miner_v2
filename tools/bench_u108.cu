/* GMP step/residue and perig-verdict gates plus timing for p0_test_u108.cuh.
   Build: make bin/bench_u108 WITH_CUDA=1 */
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <gmp.h>
#include <cuda_runtime.h>

#include "p0_walk_kern.cuh"

typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned char u8;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

static const u64 AHI_MIN = 1ULL << 41;
static const u64 AHI_MAX = (1ULL << 44) - 1;
static const u64 AHI_CAMPAIGN = 5421010862427ULL;
static const u32 MAX_BATCH = 1u << 24;
static const u32 edges[][4] = {
    {1, 0, 0, 0x200}, {3, 0, 0, 0x200},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0x200},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xfff},
    {0xfffffffdu, 0xffffffffu, 0xffffffffu, 0xfff},
    {1, 0, 0, 0xfff}, {1, 0, 0, 0x400}, {1, 0, 0, 0x800},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0x3ff},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0x7ff},
    {0x1234567u, 0x87654321u, 0xffffffffu, 0x56b}
};
static u64 rs = 88172645463325252ULL;
static u64 rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

__global__ static void k_u108(const u64 *lo, u64 hi, u8 *out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = p0u108::test_hi(lo[i], hi);
}

__global__ static void k_perig(const u64 *lo, u64 hi, u8 *out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ciosFermatTest128_hi(lo[i], hi);
}

__global__ static void k_gate(const u32 *ns, u32 *out, u8 *verdict, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) verdict[i] = p0u108::test(ns[4*i], ns[4*i+1], ns[4*i+2], ns[4*i+3], out + 4*i);
}

__global__ static void k_step(const u32 *ns, const u32 *xs, const u8 *vs,
                              u32 *out, u32 n) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u32 *m = ns + 4*i, *x = xs + 4*i;
    const u32 np = p0u108::neginv32(m[0]);
    u32 a = x[0], b = x[1], c = x[2], d = x[3];
    p0u108::step<true>(a, b, c, d, m[0], m[1], m[2], m[3], np, vs[i]);
    out[8*i] = a; out[8*i+1] = b; out[8*i+2] = c; out[8*i+3] = d;
    a = x[0]; b = x[1]; c = x[2]; d = x[3];
    p0u108::step<false>(a, b, c, d, m[0], m[1], m[2], m[3], np, 0);
    out[8*i+4] = a; out[8*i+5] = b; out[8*i+6] = c; out[8*i+7] = d;
}

static void *checked_malloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "host allocation failed\n"); exit(1); }
    return p;
}

static void fill_random_n(u32 *n) {
    u64 a = rnd(), b = rnd();
    n[0] = (u32)a | 1u; n[1] = (u32)(a >> 32);
    n[2] = (u32)b; n[3] = 0x200u + (u32)(rnd() % 0xe00u);
}

static void mpz_to4(const mpz_t z, u32 *out) {
    if (mpz_sgn(z) < 0 || mpz_sizeinbase(z, 2) > 128) {
        fprintf(stderr, "reference value exceeds four limbs\n"); exit(1);
    }
    size_t count = 0;
    memset(out, 0, 4 * sizeof(u32));
    mpz_export(out, &count, -1, sizeof(u32), 0, 0, z);
}

static void ref(const u32 *n, u32 *out, bool *verdict) {
    mpz_t m, e, y, b, t;
    mpz_inits(m, e, y, b, t, NULL);
    mpz_import(m, 4, -1, sizeof(u32), 0, 0, n);
    mpz_sub_ui(e, m, 1); mpz_fdiv_q_2exp(e, e, 1);
    mpz_set_ui(b, 2); mpz_powm(y, b, e, m);
    mpz_to4(y, out);
    if (((n[0] >> 1) ^ (n[0] >> 2)) & 1) {
        mpz_sub_ui(t, m, 1); *verdict = mpz_cmp(y, t) == 0;
    } else {
        *verdict = mpz_cmp_ui(y, 1) == 0;
    }
    mpz_clears(m, e, y, b, t, NULL);
}

static int cmd_gate(u32 N) {
    if (!N) { fprintf(stderr, "N must be positive\n"); return 2; }
    u32 *hn = (u32 *)checked_malloc(16ULL * N);
    u32 *hr = (u32 *)checked_malloc(16ULL * N);
    u8 *hv = (u8 *)checked_malloc(N);
    for (u32 i = 0; i < N; i++) fill_random_n(hn + 4*i);
    for (u32 i = 0; i < sizeof(edges)/sizeof(edges[0]) && i < N; i++)
        memcpy(hn + 4*i, edges[i], sizeof(edges[i]));
    u32 *dn, *dr; u8 *dv;
    CK(cudaMalloc(&dn, 16ULL * N)); CK(cudaMalloc(&dr, 16ULL * N)); CK(cudaMalloc(&dv, N));
    CK(cudaMemcpy(dn, hn, 16ULL * N, cudaMemcpyHostToDevice));
    k_gate<<<(N + 127) / 128, 128>>>(dn, dr, dv, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hr, dr, 16ULL * N, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hv, dv, N, cudaMemcpyDeviceToHost));
    u32 bad_gpu = 0, bad_host = 0, bad_verdict = 0;
    for (u32 i = 0; i < N; i++) {
        u32 expected[4], host[4]; bool want;
        u32 *n = hn + 4*i;
        ref(n, expected, &want);
        bool got_host = p0u108::test(n[0], n[1], n[2], n[3], host);
        if (memcmp(expected, hr + 4*i, 16)) {
            if (bad_gpu < 5)
                printf("GPU mismatch n=%08x%08x%08x%08x ref=%08x%08x%08x%08x got=%08x%08x%08x%08x\n",
                       n[3], n[2], n[1], n[0], expected[3], expected[2], expected[1], expected[0],
                       hr[4*i+3], hr[4*i+2], hr[4*i+1], hr[4*i]);
            bad_gpu++;
        }
        if (memcmp(expected, host, 16)) {
            if (bad_host < 5) printf("host residue mismatch at input %u\n", i);
            bad_host++;
        }
        if ((bool)hv[i] != want || got_host != want) {
            if (bad_verdict < 5) printf("verdict mismatch at input %u\n", i);
            bad_verdict++;
        }
    }
    printf("gate N=%u gpu_residue_diff=%u host_residue_diff=%u verdict_diff=%u\n",
           N, bad_gpu, bad_host, bad_verdict);
    CK(cudaFree(dn)); CK(cudaFree(dr)); CK(cudaFree(dv));
    free(hn); free(hr); free(hv);
    return (bad_gpu + bad_host + bad_verdict) != 0;
}

/* Check both step variants against GMP for random representatives and for
   x=0, n-1, n and n+2^105-1 at every edge modulus and every window value. */
static int cmd_step(u32 N) {
    if (!N) { fprintf(stderr, "N must be positive\n"); return 2; }
    u32 *hn = (u32 *)checked_malloc(16ULL*N), *hx = (u32 *)checked_malloc(16ULL*N);
    u32 *ho = (u32 *)checked_malloc(32ULL*N);
    u8 *hv = (u8 *)checked_malloc(N);
    mpz_t m, x, bound, inv, radix, expect, got, canonical;
    mpz_inits(m, x, bound, inv, radix, expect, got, canonical, NULL);
    mpz_set_ui(radix, 1); mpz_mul_2exp(radix, radix, 128);
    gmp_randstate_t rng; gmp_randinit_mt(rng); gmp_randseed_ui(rng, 20261102);
    for (u32 i = 0; i < N; i++) {
        fill_random_n(hn + 4*i);
        const size_t edge_index = i / 64;
        if (edge_index < sizeof(edges)/sizeof(edges[0]))
            memcpy(hn + 4*i, edges[edge_index], sizeof(edges[0]));
        hv[i] = (u8)(i & 15u);
        mpz_import(m, 4, -1, sizeof(u32), 0, 0, hn + 4*i);
        mpz_set_ui(bound, 1); mpz_mul_2exp(bound, bound, 105); mpz_add(bound, bound, m);
        switch ((i / 16) & 3u) {
        case 0: mpz_set_ui(x, 0); break;
        case 1: mpz_sub_ui(x, m, 1); break;
        case 2: mpz_set(x, m); break;
        default: mpz_sub_ui(x, bound, 1); break;
        }
        if (edge_index >= sizeof(edges)/sizeof(edges[0]) && (i & 3u))
            mpz_urandomm(x, rng, bound);
        mpz_to4(x, hx + 4*i);
    }
    u32 *dn, *dx, *dout; u8 *dv;
    CK(cudaMalloc(&dn, 16ULL*N)); CK(cudaMalloc(&dx, 16ULL*N));
    CK(cudaMalloc(&dout, 32ULL*N)); CK(cudaMalloc(&dv, N));
    CK(cudaMemcpy(dn, hn, 16ULL*N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx, 16ULL*N, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, hv, N, cudaMemcpyHostToDevice));
    k_step<<<(N+127)/128, 128>>>(dn, dx, dv, dout, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ho, dout, 32ULL*N, cudaMemcpyDeviceToHost));
    u32 bad_host = 0, bad_mod = 0, bad_bound = 0;
    for (u32 i = 0; i < N; i++) {
        const u32 *n = hn + 4*i, *in = hx + 4*i;
        u32 host[8] = {in[0], in[1], in[2], in[3], in[0], in[1], in[2], in[3]};
        u32 np = p0u108::neginv32(n[0]);
        p0u108::step<true>(host[0], host[1], host[2], host[3], n[0], n[1], n[2], n[3], np, hv[i]);
        p0u108::step<false>(host[4], host[5], host[6], host[7], n[0], n[1], n[2], n[3], np, 0);
        if (memcmp(host, ho + 8*i, 32)) {
            if (bad_host < 5) printf("host/GPU step mismatch at input %u\n", i);
            bad_host++;
        }
        mpz_import(m, 4, -1, sizeof(u32), 0, 0, n);
        mpz_import(x, 4, -1, sizeof(u32), 0, 0, in);
        if (!mpz_invert(inv, radix, m)) { fprintf(stderr, "non-invertible modulus\n"); exit(1); }
        mpz_mul(expect, x, x); mpz_mul(expect, expect, inv); mpz_mod(expect, expect, m);
        mpz_set_ui(bound, 1); mpz_mul_2exp(bound, bound, 105); mpz_add(bound, bound, m);
        for (int variant = 0; variant < 2; variant++) {
            mpz_import(got, 4, -1, sizeof(u32), 0, 0, ho + 8*i + 4*variant);
            if (mpz_cmp(got, bound) >= 0) {
                if (bad_bound < 5) printf("step bound failure at input %u variant %d\n", i, variant);
                bad_bound++;
            }
            mpz_mod(canonical, got, m);
            if (variant == 0) {
                mpz_mul_2exp(x, expect, hv[i]); mpz_mod(x, x, m);
            } else {
                mpz_set(x, expect);
            }
            if (mpz_cmp(canonical, x)) {
                if (bad_mod < 5) printf("step residue mismatch at input %u variant %d\n", i, variant);
                bad_mod++;
            }
        }
    }
    printf("step N=%u host_gpu_diff=%u residue_diff=%u bound_failures=%u\n",
           N, bad_host, bad_mod, bad_bound);
    CK(cudaFree(dn)); CK(cudaFree(dx)); CK(cudaFree(dout)); CK(cudaFree(dv));
    gmp_randclear(rng); mpz_clears(m, x, bound, inv, radix, expect, got, canonical, NULL);
    free(hn); free(hx); free(ho); free(hv);
    return (bad_host + bad_mod + bad_bound) != 0;
}

static void verd_chunk(u32 N, u64 hi, u64 base, u64 *pu, u64 *pp, u64 *diff) {
    u64 *hlo = (u64 *)checked_malloc(8ULL*N), *dlo;
    u8 *h1 = (u8 *)checked_malloc(N), *h2 = (u8 *)checked_malloc(N), *d1, *d2;
    for (u32 i = 0; i < N; i++) hlo[i] = (base | 1ULL) + 2ULL*i;
    CK(cudaMalloc(&dlo, 8ULL*N)); CK(cudaMalloc(&d1, N)); CK(cudaMalloc(&d2, N));
    CK(cudaMemcpy(dlo, hlo, 8ULL*N, cudaMemcpyHostToDevice));
    k_u108<<<(N+127)/128, 128>>>(dlo, hi, d1, N);
    CK(cudaGetLastError());
    k_perig<<<(N+127)/128, 128>>>(dlo, hi, d2, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h1, d1, N, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2, d2, N, cudaMemcpyDeviceToHost));
    for (u32 i = 0; i < N; i++) {
        *pu += h1[i]; *pp += h2[i];
        if (h1[i] != h2[i]) {
            if (*diff < 5) printf("mismatch hi=%llu lo=%llu u108=%u perig=%u\n",
                                  (unsigned long long)hi, (unsigned long long)hlo[i], h1[i], h2[i]);
            (*diff)++;
        }
    }
    CK(cudaFree(dlo)); CK(cudaFree(d1)); CK(cudaFree(d2));
    free(hlo); free(h1); free(h2);
}

static int cmd_verd(u64 N, u64 hi, u64 base) {
    if (!N || hi < AHI_MIN || hi > AHI_MAX || !(base & 1ULL) ||
        N - 1 > (UINT64_MAX - base) / 2) {
        fprintf(stderr, "invalid count, high word or low-word range (lowbase must be odd)\n");
        return 2;
    }
    u64 pu = 0, pp = 0, diff = 0, done = 0;
    while (done < N) {
        u32 n = (u32)((N - done < (1u << 22)) ? N - done : (1u << 22));
        verd_chunk(n, hi, base + 2*done, &pu, &pp, &diff);
        done += n;
    }
    printf("verd N=%llu hi=%llu primes_u108=%llu primes_perig=%llu diff=%llu\n",
           (unsigned long long)N, (unsigned long long)hi,
           (unsigned long long)pu, (unsigned long long)pp, (unsigned long long)diff);
    return diff != 0;
}

static int cmd_sweep(u32 chunks, u32 N) {
    if (!chunks || !N) { fprintf(stderr, "chunks and N must be positive\n"); return 2; }
    u64 pu=0, pp=0, diff=0;
    for (u32 c=0; c<chunks; c++) {
        u64 hi = c == 0 ? AHI_MIN : c == 1 ? AHI_MAX :
                 c == 2 ? AHI_CAMPAIGN : AHI_MIN + rnd() % (AHI_MAX - AHI_MIN + 1);
        u64 base = rnd() >> 1;
        verd_chunk(N, hi, base, &pu, &pp, &diff);
    }
    printf("sweep chunks=%u N=%u primes_u108=%llu primes_perig=%llu diff=%llu\n",
           chunks, N, (unsigned long long)pu, (unsigned long long)pp, (unsigned long long)diff);
    return diff != 0;
}

template<class K>
static double bench(K k, const u64 *lo, u64 hi, u8 *out, u32 N, int iters) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    k<<<(N+127)/128,128>>>(lo, hi, out, N);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a));
    for (int i=0; i<iters; i++) k<<<(N+127)/128,128>>>(lo, hi, out, N);
    CK(cudaGetLastError()); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms * 1e-3 / iters;
}

static int cmd_time(u32 N, int iters, u64 hi) {
    if (!N || iters <= 0 || hi < AHI_MIN || hi > AHI_MAX) {
        fprintf(stderr, "invalid N, iters or high word\n"); return 2;
    }
    u64 *hlo=(u64 *)checked_malloc(8ULL*N), *dlo;
    u8 *dout;
    for (u32 i=0;i<N;i++) hlo[i]=0x123456789abcdef1ULL+2ULL*i;
    CK(cudaMalloc(&dlo,8ULL*N)); CK(cudaMalloc(&dout,N));
    CK(cudaMemcpy(dlo,hlo,8ULL*N,cudaMemcpyHostToDevice));
    for (int r=0;r<4;r++) {
        double a=bench(k_u108,dlo,hi,dout,N,iters);
        double b=bench(k_perig,dlo,hi,dout,N,iters);
        printf("rep=%d hi=%llu u108=%.4f ns/test perig=%.4f ns/test speedup=%.3fx\n",
               r,(unsigned long long)hi,1e9*a/N,1e9*b/N,b/a);
    }
    CK(cudaFree(dlo)); CK(cudaFree(dout)); free(hlo);
    return 0;
}

static u64 parse_number(const char *s, u64 max, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] < '0' || s[0] > '9' || *end || errno || v > max) {
        fprintf(stderr, "invalid %s: %s\n", name, s); exit(2);
    }
    return (u64)v;
}

static int cmd_dispatch() {
    struct Case { u64 hi; int expected; };
    const Case cases[] = {
        {0, 0}, {63, 0}, {64, 3}, {(1ULL << 11) - 1, 3}, {1ULL << 11, 0},
        {(1ULL << 31) - 1, 0}, {1ULL << 31, 4},
        {(1ULL << 32) - (1ULL << 15) - 1, 4},
        {(1ULL << 32) - (1ULL << 15), 0},
        {(1ULL << 41) - 1, 0}, {1ULL << 41, 5}, {AHI_CAMPAIGN, 5},
        {(1ULL << 44) - 1, 5}, {1ULL << 44, 0},
        {(1ULL << 56) - 1, 0}, {1ULL << 56, 1},
        {(1ULL << 62) - 1, 1}, {1ULL << 62, 2}, {UINT64_MAX, 2}
    };
    unsigned failures = 0;
    for (const Case &c : cases) {
        int actual = w_test_mode(c.hi);
        if (actual != c.expected) {
            fprintf(stderr, "dispatch hi=%llu expected=%d actual=%d\n",
                    (unsigned long long)c.hi, c.expected, actual);
            failures++;
        }
    }
    printf("dispatch cases=%zu failures=%u\n", sizeof(cases) / sizeof(cases[0]), failures);
    return failures != 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_u108 dispatch | gate|step [N] | sweep [chunks] [N] | "
                        "verd [N] [hi] [lowbase] | time [N] [iters] [hi]\n");
        return 2;
    }
    if (!strcmp(argv[1], "dispatch")) return cmd_dispatch();
    if (!strcmp(argv[1], "gate") || !strcmp(argv[1], "step")) {
        u32 n = argc > 2 ? (u32)parse_number(argv[2], MAX_BATCH, "N") : 200000u;
        return !strcmp(argv[1], "gate") ? cmd_gate(n) : cmd_step(n);
    }
    if (!strcmp(argv[1], "sweep"))
        return cmd_sweep(argc > 2 ? (u32)parse_number(argv[2], UINT32_MAX, "chunks") : 64u,
                         argc > 3 ? (u32)parse_number(argv[3], MAX_BATCH, "N") : (1u<<20));
    if (!strcmp(argv[1], "verd"))
        return cmd_verd(argc > 2 ? parse_number(argv[2], UINT64_MAX, "N") : (1ULL<<22),
                        argc > 3 ? parse_number(argv[3], UINT64_MAX, "hi") : AHI_CAMPAIGN,
                        argc > 4 ? parse_number(argv[4], UINT64_MAX, "lowbase") : 0x123456789abcdef1ULL);
    if (!strcmp(argv[1], "time"))
        return cmd_time(argc > 2 ? (u32)parse_number(argv[2], MAX_BATCH, "N") : (1u<<20),
                        argc > 3 ? (int)parse_number(argv[3], INT_MAX, "iters") : 20,
                        argc > 4 ? parse_number(argv[4], UINT64_MAX, "hi") : AHI_CAMPAIGN);
    fprintf(stderr, "unknown command\n"); return 2;
}
