/*
 * bench_hunt_mark.cu -- dev tool: per-kernel cost split of the --gap-hunt
 * window MARK at production geometry.
 *
 * Why: with the fill pipeline + parallel extract in place (2026-10-05) the
 * hunt's per-window wall at shift258 / min-merit 24 is ~104 us, of which
 * GPU_SIEVE_TIMING attributes ~63 us to the mark.  The mark is a SEQUENCE
 * (H2D base_offsets -> memset -> dense split kernel -> suffix batch kernel)
 * and the aggregate number cannot say which member dominates, so any further
 * change would be guesswork.  This tool drives the PRODUCTION kernels (it
 * includes gpu_sieve.cu, like tools/bench_mark.cu) with the hunt's geometry
 * and times each member separately.
 *
 * It also times the folded PAIR mark introduced for the two-window staging
 * round (gpu_sieve_mark_pair_fold / gpu_sieve_mark_pair_kernel_batch), which
 * is the change whose value this bench has to justify: the pair suffix kernel
 * reads the prime/residue/step tables ONCE for two windows instead of twice.
 *
 * Build:  make bin/bench_hunt_mark WITH_CUDA=1
 * Usage:  ./bin/bench_hunt_mark [--primes N] [--window W] [--iters I]
 *                               [--slots S] [--limbs L]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_sieve.cu"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

static uint64_t parse_u64(const char *s) { return strtoull(s, NULL, 10); }

static std::vector<uint64_t> first_primes(size_t n) {
    std::vector<uint64_t> out;
    out.reserve(n);
    for (uint64_t v = 3; out.size() < n; v += 2) {
        int ok = 1;
        for (uint64_t d = 3; d * d <= v; d += 2) {
            if (v % d == 0) { ok = 0; break; }
        }
        if (ok) out.push_back(v);
        if (v > 100000000ULL) break;   /* guard for tiny --primes typos */
    }
    return out;
}

int main(int argc, char **argv) {
    uint64_t n_primes = 2000000;
    uint64_t W = 20574;
    int iters = 200;
    uint64_t slots = 160;
    int limbs = 32;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--primes") && i + 1 < argc)
            n_primes = parse_u64(argv[++i]);
        else if (!strcmp(argv[i], "--window") && i + 1 < argc)
            W = parse_u64(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
            iters = (int)parse_u64(argv[++i]);
        else if (!strcmp(argv[i], "--slots") && i + 1 < argc)
            slots = parse_u64(argv[++i]);
        else if (!strcmp(argv[i], "--limbs") && i + 1 < argc)
            limbs = (int)parse_u64(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--primes N] [--window W] [--iters I]"
                            " [--slots S] [--limbs L]\n", argv[0]);
            return 2;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaSetDevice(0));
    printf("device: %s (%d SMs), primes=%llu W=%llu slots=%llu limbs=%d\n",
           prop.name, prop.multiProcessorCount,
           (unsigned long long)n_primes, (unsigned long long)W,
           (unsigned long long)slots, limbs);

    std::vector<uint64_t> hprimes = first_primes((size_t)n_primes);
    if (hprimes.size() < n_primes) {
        fprintf(stderr, "prime generation short: %zu < %llu\n", hprimes.size(),
                (unsigned long long)n_primes);
        return 1;
    }
    std::vector<uint64_t> hinv(hprimes.size());
    uint64_t n_small = 0;
    for (size_t i = 0; i < hprimes.size(); i++) {
        hinv[i] = (uint64_t)(~0ULL / hprimes[i]);      /* floor(2^64 / p) */
        if (hprimes[i] <= W) n_small++;
    }
    uint64_t chunks = (W + slots - 1) / slots;
    if (chunks > 1024) chunks = 1024;
    printf("n_small=%llu rest=%llu chunks=%llu (dense items/window=%llu)\n",
           (unsigned long long)n_small,
           (unsigned long long)(hprimes.size() - n_small),
           (unsigned long long)chunks,
           (unsigned long long)(n_small * chunks));

    gpu_sieve_ctx *ctx = gpu_sieve_init(0, (size_t)n_primes, W);
    if (!ctx) { fprintf(stderr, "gpu_sieve_init failed\n"); return 1; }

    /* Anchor the residue cache: one full reduction (mode 0) fills
       d_base_mod_p for the anchor window; afterwards every window folds. */
    uint64_t base_limbs[64];
    memset(base_limbs, 0, sizeof(base_limbs));
    base_limbs[0] = 0x9E3779B97F4A7C15ULL;      /* arbitrary non-zero base */
    base_limbs[1] = 0x1234567ULL;
    if (!gpu_sieve_set_residue_step(ctx, base_limbs, limbs, hprimes.data(),
                                    hinv.data(), hprimes.size())) {
        fprintf(stderr, "set_residue_step failed\n");
        return 1;
    }
    if (!gpu_sieve_mark_from_base(ctx, W, 1, base_limbs, limbs, 0,
                                  hprimes.data(), hinv.data(), hprimes.size(),
                                  NULL, 0)) {
        fprintf(stderr, "anchor mark failed\n");
        return 1;
    }
    /* Prime the bitmaps/count once so the pair path has its scratch. */
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    /* Warm-up + reference: the whole mode-2 mark (folded cache). */
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    const int TPB = 128;
    size_t words = (size_t)((W + 63U) >> 6);
    uint32_t *d_ext_scan = NULL;
    CUDA_CHECK(cudaMalloc(&d_ext_scan, 2U * EXTRACT_MAX_BLOCKS * sizeof(uint32_t)));

    for (int pass = 0; pass < 2; pass++) {
        uint64_t m_total = 0, m_set = 0, m_dense = 0, m_suffix = 0;
        uint64_t m_pair_prep = 0, m_pair_dense = 0, m_pair_suffix = 0;
        uint64_t m_pair_memset = 0;
        for (int it = 0; it < iters; it++) {
            uint64_t m = (uint64_t)it + 1;

            /* (a) the production single-window mark (reference) */
            cudaEventRecord(e0, ctx->stream);
            if (!gpu_sieve_mark_from_base_mw(ctx, W, 1, 0, m, hprimes.data(),
                                             hinv.data(), hprimes.size(),
                                             NULL, 0)) {
                fprintf(stderr, "mark_from_base_mw failed\n");
                return 1;
            }
            cudaEventRecord(e1, ctx->stream);
            CUDA_CHECK(cudaEventSynchronize(e1));
            { float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
              m_total += (uint64_t)(ms * 1000.0f + 0.5f); }

            /* (b) memset alone */
            cudaEventRecord(e0, ctx->stream);
            CUDA_CHECK(cudaMemsetAsync(ctx->d_bitmap[0], 0,
                                       words * sizeof(uint64_t), ctx->stream));
            cudaEventRecord(e1, ctx->stream);
            CUDA_CHECK(cudaEventSynchronize(e1));
            { float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
              m_set += (uint64_t)(ms * 1000.0f + 0.5f); }

            /* (c) dense split kernel alone (p <= W) */
            {
                uint64_t iblocks = (n_small * chunks + TPB - 1) / TPB;
                cudaEventRecord(e0, ctx->stream);
                gpu_sieve_mark_dense_split_kernel<<<(unsigned)iblocks, TPB, 0,
                                                    ctx->stream>>>(
                    ctx->d_bitmap[0], W, 1, ctx->d_primes, ctx->d_base_mod_p,
                    ctx->d_step_mod_p, ctx->d_inv_p, m, n_small,
                    (uint32_t)chunks, slots);
                cudaEventRecord(e1, ctx->stream);
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                m_dense += (uint64_t)(ms * 1000.0f + 0.5f);
            }

            /* (d) suffix kernel alone (p > W) */
            {
                uint64_t rest = hprimes.size() - n_small;
                uint64_t rblocks = (rest + TPB - 1) / TPB;
                cudaEventRecord(e0, ctx->stream);
                gpu_sieve_mark_kernel_batch<<<(unsigned)rblocks, TPB, 0,
                                              ctx->stream>>>(
                    ctx->d_bitmap[0], (uint64_t)words, W, 1, ctx->d_base_offsets,
                    1, ctx->d_primes + n_small, ctx->d_base_mod_p + n_small,
                    ctx->d_step_mod_p + n_small, ctx->d_inv_p + n_small, m,
                    rest);
                cudaEventRecord(e1, ctx->stream);
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                m_suffix += (uint64_t)(ms * 1000.0f + 0.5f);
            }

            /* (e) the folded PAIR mark (two windows, one table read) */
            {
                CUDA_CHECK(cudaMemsetAsync(ctx->d_bitmap[0], 0,
                                           words * sizeof(uint64_t),
                                           ctx->stream));
                CUDA_CHECK(cudaMemsetAsync(ctx->d_bitmap[1], 0,
                                           words * sizeof(uint64_t),
                                           ctx->stream));
                cudaEventRecord(e0, ctx->stream);
                if (!gpu_sieve_mark_pair_fold(ctx, W, 1, W, 1, m,
                                              hprimes.data(), hinv.data(),
                                              hprimes.size(), slots)) {
                    fprintf(stderr, "mark_pair_fold failed\n");
                    return 1;
                }
                cudaEventRecord(e1, ctx->stream);
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                m_pair_dense += (uint64_t)(ms * 1000.0f + 0.5f);
            }
        }
        if (pass == 1) {
            double d = (double)iters;
            printf("\nper-call (mean of %d):\n", iters);
            printf("  production mark (fold, 1 window)  %8.2f us\n", m_total / d);
            printf("    of which memset (1 bitmap)      %8.2f us\n", m_set / d);
            printf("    of which dense split kernel     %8.2f us\n", m_dense / d);
            printf("    of which suffix batch kernel    %8.2f us\n", m_suffix / d);
            printf("    unaccounted (H2D/launches)      %8.2f us\n",
                   (m_total - m_set - m_dense - m_suffix) / d);
            printf("  folded PAIR mark (2 windows+memset)%8.2f us  -> %.2f us/window\n",
                   m_pair_dense / d, m_pair_dense / (2.0 * d));
        }
    }

    /* ── Bit-exactness of the PAIR mark vs two single marks ──────────────
       The pair kernel must produce the SAME two bitmaps as marking the two
       windows separately, for both geometry cases that occur in the walk:
       (a) both windows at the same (size, offset) - the bench's arbitrary
           base, and (b) the REAL parity flip: consecutive walk windows differ
           by the odd CRT step P, so one has an even base (first_odd_offset 1,
           size (interval+1)/2) and the next an odd base (offset 0, size
           interval/2).  A pair that got the second window's geometry wrong
           would still "work" (the extract clips) but would drop composites -
           exactly the silent failure this check rules out. */
    {
        size_t wordsW = (size_t)((W + 63U) >> 6);
        std::vector<uint64_t> a0(wordsW, 0), a1(wordsW, 0);
        std::vector<uint64_t> b0(wordsW, 0), b1(wordsW, 0);
        uint64_t m = 7;

        /* (a) same geometry: two single marks vs one pair mark */
        if (!gpu_sieve_mark_from_base_mw(ctx, W, 1, 0, m, hprimes.data(),
                                         hinv.data(), hprimes.size(), NULL, 0))
            return 1;
        CUDA_CHECK(cudaMemcpy(a0.data(), ctx->d_bitmap[0],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        if (!gpu_sieve_mark_from_base_mw(ctx, W, 1, 1, m + 1, hprimes.data(),
                                         hinv.data(), hprimes.size(), NULL, 0))
            return 1;
        CUDA_CHECK(cudaMemcpy(a1.data(), ctx->d_bitmap[1],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        if (!gpu_sieve_mark_pair_fold(ctx, W, 1, W, 1, m, hprimes.data(),
                                      hinv.data(), hprimes.size(), slots))
            return 1;
        CUDA_CHECK(cudaMemcpy(b0.data(), ctx->d_bitmap[0],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b1.data(), ctx->d_bitmap[1],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        size_t d0 = 0, d1 = 0;
        for (size_t i = 0; i < wordsW; i++) {
            d0 += __builtin_popcountll(a0[i] ^ b0[i]);
            d1 += __builtin_popcountll(a1[i] ^ b1[i]);
        }
        printf("\npair-vs-single (same geometry):  bitmap0 differing bits=%zu,"
               " bitmap1 differing bits=%zu  -> %s\n",
               d0, d1, (d0 == 0 && d1 == 0) ? "BIT-EXACT" : "MISMATCH");

        /* (b) parity flip: even base (offset 1, size W) then odd base
           (offset 0, size W-1), which is the shape of two consecutive walk
           windows. */
        std::fill(b0.begin(), b0.end(), 0);
        std::fill(b1.begin(), b1.end(), 0);
        if (!gpu_sieve_mark_from_base_mw(ctx, W, 1, 0, m, hprimes.data(),
                                         hinv.data(), hprimes.size(), NULL, 0))
            return 1;
        CUDA_CHECK(cudaMemcpy(a0.data(), ctx->d_bitmap[0],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        if (!gpu_sieve_mark_from_base_mw(ctx, W - 1, 0, 1, m + 1,
                                         hprimes.data(), hinv.data(),
                                         hprimes.size(), NULL, 0))
            return 1;
        CUDA_CHECK(cudaMemcpy(a1.data(), ctx->d_bitmap[1],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        if (!gpu_sieve_mark_pair_fold(ctx, W, 1, W - 1, 0, m, hprimes.data(),
                                      hinv.data(), hprimes.size(), slots))
            return 1;
        CUDA_CHECK(cudaMemcpy(b0.data(), ctx->d_bitmap[0],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b1.data(), ctx->d_bitmap[1],
                              wordsW * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost));
        d0 = d1 = 0;
        for (size_t i = 0; i < wordsW; i++) {
            d0 += __builtin_popcountll(a0[i] ^ b0[i]);
            d1 += __builtin_popcountll(a1[i] ^ b1[i]);
        }
        printf("pair-vs-single (parity flip):    bitmap0 differing bits=%zu,"
               " bitmap1 differing bits=%zu  -> %s\n",
               d0, d1, (d0 == 0 && d1 == 0) ? "BIT-EXACT" : "MISMATCH");
    }

    cudaFree(d_ext_scan);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    gpu_sieve_destroy(ctx);
    return 0;
}
