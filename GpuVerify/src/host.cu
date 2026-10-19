#include "host.h"
#include "gv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

/* device entry points from prp_kernel.cu */
extern "C" cudaError_t gv_launch_prp_bucket(int n32, const uint32_t *data,
                                            const uint32_t *meta, uint8_t *pass,
                                            int count, cudaStream_t stream);
/* device entry points from sieve_kernel.cu */
extern "C" void launch_marks_clear(uint8_t *marks, uint64_t half_gap);
extern "C" void launch_sieve(uint8_t *marks, const uint64_t *starts,
                             const uint32_t *primes, uint64_t gap, int nprimes);

static int g_gpu_ok = -1; /* -1 unknown, 0 no, 1 yes */
static cudaStream_t g_stream = 0;

int gv_gpu_init(void)
{
    if (g_gpu_ok == 1)
        return 0;
    if (g_gpu_ok == 0)
        return -1;
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count < 1) {
        g_gpu_ok = 0;
        fprintf(stderr, "gpuverify: no CUDA device found, CPU-only mode\n");
        return -1;
    }
    e = cudaSetDevice(0);
    if (e != cudaSuccess) {
        g_gpu_ok = 0;
        return -1;
    }
    cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking);
    g_gpu_ok = 1;
    return 0;
}

int gv_gpu_available(void)
{
    return gv_gpu_init();
}

/* ---- host-side Montgomery constants for one candidate ----
 * data layout: [n][mont(2)][d][mont(N-1)][mont(1)], each `bucket` words.
 */

typedef struct {
    int L;      /* exact limb count of n */
    int bucket; /* padded limb count, one of 16,32,...,1024 */
    int s;      /* n-1 = d * 2^s, s >= 1 */
    int bl;     /* bit length of d */
    uint32_t np0;
} gv_mont_params;

static void gv_mont_prepare(mpz_t n, int bucket, gv_mont_params *p,
                            uint32_t *data)
{
    mpz_t R, t, m2, mn1, d, n1;
    mpz_init(R);
    mpz_init(t);
    mpz_init(m2);
    mpz_init(mn1);
    mpz_init(d);
    mpz_init(n1);

    int L = (int)((mpz_sizeinbase(n, 2) + 31) / 32);
    p->L = L;
    p->bucket = bucket;

    mpz_sub_ui(n1, n, 1);
    unsigned long s = mpz_scan1(n1, 0);
    if (s < 1)
        s = 1; /* n odd => v2(n-1) >= 1 */
    p->s = (int)s;
    mpz_tdiv_q_2exp(d, n1, s);
    p->bl = (int)mpz_sizeinbase(d, 2);

    /* np0 = -n^{-1} mod 2^32 */
    mpz_set_ui(t, mpz_get_ui(n)); /* low 32 bits of n */
    mpz_t mod;
    mpz_init(mod);
    mpz_setbit(mod, 32);
    if (mpz_invert(t, t, mod) != 0)
        p->np0 = (uint32_t)(0 - mpz_get_ui(t));
    else
        p->np0 = 0;
    mpz_clear(mod);

    /* mont(1) = 2^(32*bucket) mod n (the kernel's Montgomery R uses the
     * padded width, not the exact limb count) */
    mpz_setbit(R, 32 * bucket);
    mpz_mod(R, R, n);
    /* mont(2) = 2R mod n */
    mpz_mul_ui(m2, R, 2);
    mpz_mod(m2, m2, n);
    /* mont(N-1) = R*(n-1) mod n */
    mpz_mul(mn1, n1, R);
    mpz_mod(mn1, mn1, n);

    if (data) {
        size_t W = (size_t)bucket;
        memset(data, 0, 5 * W * sizeof(uint32_t));
        size_t count;
        mpz_export(data, &count, -1, 4, 0, 0, n);
        if (count != (size_t)L)
            gv_die("mont_prepare: unexpected limb count for n");
        mpz_export(data + 1 * W, &count, -1, 4, 0, 0, m2);
        mpz_export(data + 2 * W, &count, -1, 4, 0, 0, d);
        mpz_export(data + 3 * W, &count, -1, 4, 0, 0, mn1);
        mpz_export(data + 4 * W, &count, -1, 4, 0, 0, R);
    }

    mpz_clear(R);
    mpz_clear(t);
    mpz_clear(m2);
    mpz_clear(mn1);
    mpz_clear(d);
    mpz_clear(n1);
}

/*
 * Batch strong PRP base 2. Splits candidates into limb buckets and runs
 * one templated kernel launch per bucket. pass_out[i]: 1 pass / 0 composite
 * (proved) / 2 too large for GPU.
 */
int gv_gpu_prp_batch(mpz_t *nums, int count, uint8_t *pass_out)
{
    if (gv_gpu_init() != 0)
        return -1;

    gv_mont_params *par = (gv_mont_params *)calloc((size_t)count, sizeof(*par));
    uint32_t **data = (uint32_t **)calloc((size_t)count, sizeof(*data));
    uint32_t *meta = (uint32_t *)malloc((size_t)count * 4 * sizeof(uint32_t));
    if (!par || !data || !meta)
        gv_die("out of memory");
    int maxb = 0;

    for (int i = 0; i < count; i++) {
        if (mpz_cmp_ui(nums[i], 3) < 0) {
            pass_out[i] = (mpz_cmp_ui(nums[i], 2) == 0) ? 1 : 0;
            continue;
        }
        if (mpz_even_p(nums[i])) {
            pass_out[i] = 0; /* trivial composite, no GPU needed */
            continue;
        }
        int L = (int)((mpz_sizeinbase(nums[i], 2) + 31) / 32);
        if (L > GV_MAX_LIMBS) {
            pass_out[i] = 2; /* too large for GPU */
            continue;
        }
        int b = 16;
        while (b < L)
            b <<= 1;
        par[i].bucket = b;
        data[i] = (uint32_t *)malloc((size_t)b * 5 * sizeof(uint32_t));
        gv_mont_prepare(nums[i], b, &par[i], data[i]);
        meta[i * 4 + 0] = (uint32_t)par[i].L;
        meta[i * 4 + 1] = (uint32_t)par[i].s;
        meta[i * 4 + 2] = (uint32_t)par[i].bl;
        meta[i * 4 + 3] = par[i].np0;
        if (b > maxb)
            maxb = b;
    }

    for (int bucket = 16; bucket <= maxb; bucket <<= 1) {
        int n_cand = 0;
        for (int i = 0; i < count; i++)
            if (par[i].bucket == bucket)
                n_cand++;
        if (n_cand == 0)
            continue;

        size_t bytes = (size_t)n_cand * (size_t)bucket * 5 * sizeof(uint32_t);
        uint32_t *d_host = (uint32_t *)malloc(bytes);
        uint32_t *m_host = (uint32_t *)malloc((size_t)n_cand * 4 * sizeof(uint32_t));
        uint8_t *p_host = (uint8_t *)malloc((size_t)n_cand);
        uint32_t *d_buf, *m_buf;
        uint8_t *p_buf;
        cudaMalloc(&d_buf, bytes);
        cudaMalloc(&m_buf, (size_t)n_cand * 4 * sizeof(uint32_t));
        cudaMalloc(&p_buf, (size_t)n_cand);

        int k = 0;
        for (int i = 0; i < count; i++) {
            if (par[i].bucket != bucket)
                continue;
            memcpy(d_host + (size_t)k * (size_t)bucket * 5, data[i],
                   (size_t)bucket * 5 * sizeof(uint32_t));
            memcpy(m_host + (size_t)k * 4, meta + (size_t)i * 4,
                   4 * sizeof(uint32_t));
            k++;
        }
        cudaMemcpy(d_buf, d_host, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(m_buf, m_host, (size_t)n_cand * 4 * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);

        cudaError_t e = gv_launch_prp_bucket(bucket, d_buf, m_buf, p_buf,
                                             n_cand, g_stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "gpuverify: PRP launch failed: %s\n",
                    cudaGetErrorString(e));
            return -1;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(p_host, p_buf, (size_t)n_cand, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(g_stream);

        k = 0;
        for (int i = 0; i < count; i++)
            if (par[i].bucket == bucket)
                pass_out[i] = p_host[k++];

        cudaFree(d_buf);
        cudaFree(m_buf);
        cudaFree(p_buf);
        free(d_host);
        free(m_host);
        free(p_host);
    }

    for (int i = 0; i < count; i++)
        free(data[i]);
    free(data);
    free(par);
    free(meta);
    return 0;
}

/* ---- small-prime list ---- */

static uint32_t *g_primes = NULL;
static int g_nprimes = 0;
static uint32_t g_pmax = 0;

int gv_get_primes(uint32_t pmax, const uint32_t **primes_out, int *count_out)
{
    if (g_primes && pmax <= g_pmax) {
        *primes_out = g_primes;
        *count_out = g_nprimes;
        return 0;
    }
    uint32_t size = (pmax + 63) / 64;
    uint64_t *bits = (uint64_t *)calloc(size, sizeof(uint64_t));
    uint32_t cap = pmax > 1000 ? pmax / 10 + 100 : 200;
    uint32_t *pr = (uint32_t *)malloc(cap * sizeof(uint32_t));
    int np = 0;
    pr[np++] = 2;
    for (uint32_t q = 3; q <= pmax; q += 2) {
        if (!(bits[q >> 6] & (1ULL << (q & 63)))) {
            if (np >= (int)cap) {
                cap *= 2;
                pr = (uint32_t *)realloc(pr, cap * sizeof(uint32_t));
            }
            pr[np++] = q;
            if ((uint64_t)q * q <= pmax)
                for (uint64_t m = (uint64_t)q * q; m <= pmax; m += 2 * q)
                    bits[m >> 6] |= 1ULL << (m & 63);
        }
    }
    free(bits);
    free(g_primes);
    g_primes = pr;
    g_nprimes = np;
    g_pmax = pmax;
    *primes_out = g_primes;
    *count_out = np;
    return 0;
}

/* ---- interval sieve ---- */

int gv_gpu_sieve_gap(mpz_t N, uint64_t gap, uint32_t pmax,
                     uint8_t **marks_out, uint64_t *half_gap_out)
{
    if (gv_gpu_init() != 0)
        return -1;

    /* interior candidates: N+k for even k in [2, gap-2] -> gap/2 - 1 */
    if (gap < 4) {
        *marks_out = NULL;
        *half_gap_out = 0;
        return 0;
    }
    uint64_t half = gap / 2 - 1;

    const uint32_t *primes;
    int nprimes;
    gv_get_primes(pmax, &primes, &nprimes);

    /* per-prime first even k with N+k == 0 (mod p), skipping N+k == p */
    uint64_t *starts = (uint64_t *)malloc((size_t)nprimes * sizeof(uint64_t));
    mpz_t tmp;
    mpz_init(tmp);
    for (int i = 0; i < nprimes; i++) {
        uint32_t p = primes[i];
        uint64_t k0;
        if (p == 2) {
            /* N odd -> N+k odd iff k even -> never divisible by 2 */
            k0 = gap; /* disable */
        } else {
            uint32_t r = (uint32_t)mpz_mod_ui(tmp, N, p);
            k0 = (p - r) % p;
            if (k0 & 1)
                k0 += p; /* make k even */
            if (k0 < gap) {
                /* if N+k0 == p, that candidate IS the prime p: skip it */
                mpz_add_ui(tmp, N, k0);
                if (mpz_cmp_ui(tmp, p) == 0)
                    k0 += 2ull * p;
            }
        }
        starts[i] = k0;
    }
    mpz_clear(tmp);

    uint8_t *d_marks;
    uint64_t *d_starts;
    uint32_t *d_primes;
    cudaMalloc(&d_marks, half);
    cudaMalloc(&d_starts, (size_t)nprimes * sizeof(uint64_t));
    cudaMalloc(&d_primes, (size_t)nprimes * sizeof(uint32_t));
    cudaMemcpy(d_starts, starts, (size_t)nprimes * sizeof(uint64_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_primes, primes, (size_t)nprimes * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    launch_marks_clear(d_marks, half);
    launch_sieve(d_marks, d_starts, d_primes, gap, nprimes);
    cudaDeviceSynchronize();

    uint8_t *marks = (uint8_t *)malloc(half);
    cudaMemcpy(marks, d_marks, half, cudaMemcpyDeviceToHost);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "gpuverify: sieve failed: %s\n", cudaGetErrorString(e));
        free(marks);
        return -1;
    }

    cudaFree(d_marks);
    cudaFree(d_starts);
    cudaFree(d_primes);
    free(starts);

    *marks_out = marks;
    *half_gap_out = half;
    return 0;
}

/* ---- CPU BPSW / reference ---- */

int gv_bpsw(mpz_t n)
{
    /* GMP 6.3: probab_prime_p includes the strong base-2 + Lucas (BPSW)
     * tests even at reps == 1. */
    return mpz_probab_prime_p(n, 1) != 0;
}

int gv_ref_strong_prp2(mpz_t n)
{
    if (mpz_cmp_ui(n, 2) == 0)
        return 1;
    if (mpz_cmp_ui(n, 2) < 0 || mpz_even_p(n))
        return 0;
    mpz_t d, y, n1;
    mpz_init(d);
    mpz_init(y);
    mpz_init(n1);
    mpz_sub_ui(n1, n, 1);
    unsigned long s = mpz_scan1(n1, 0);
    mpz_tdiv_q_2exp(d, n1, s);

    mpz_set_ui(y, 2);
    mpz_powm(y, y, d, n);
    int pass = mpz_cmp_ui(y, 1) == 0;
    for (unsigned long i = 0; i < s && !pass; i++) {
        if (mpz_cmp(y, n1) == 0) {
            pass = 1;
            break;
        }
        mpz_mul(y, y, y);
        mpz_mod(y, y, n);
    }
    mpz_clear(d);
    mpz_clear(y);
    mpz_clear(n1);
    return pass;
}
