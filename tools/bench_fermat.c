/*
 * bench_fermat.c — GPU Fermat kernel throughput benchmark (dev tool).
 *
 * Build: make WITH_CUDA=1 WITH_CGBN_FERMAT=1 bin/bench_fermat
 * Run:   ./bin/bench_fermat [AL] [batch] [iterations]
 *
 * Generates random odd AL*64-bit candidates, packs them at AL-limb stride,
 * and repeatedly calls gpu_fermat_test_batch to measure candidates/second.
 * Prints per-batch wall time, kernel label, and candidates/s.
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#include "../new_src/gpu/gpu_fermat.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t rng64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int al = (argc > 1) ? atoi(argv[1]) : 12;
    int batch = (argc > 2) ? atoi(argv[2]) : 10000;
    int iters = (argc > 3) ? atoi(argv[3]) : 20;

    if (al < 1 || al > GPU_NLIMBS) {
        fprintf(stderr, "AL out of range [1,%d]\n", GPU_NLIMBS);
        return 2;
    }
    if (batch < 1 || iters < 1) return 2;

    gpu_fermat_ctx *ctx = gpu_fermat_init(0, (size_t)batch);
    if (!ctx) {
        fprintf(stderr, "gpu_fermat_init failed\n");
        return 1;
    }
    gpu_fermat_set_limbs(ctx, al);

    uint64_t *cands = (uint64_t *)malloc((size_t)batch * (size_t)al * sizeof(uint64_t));
    uint8_t *results = (uint8_t *)malloc((size_t)batch);
    if (!cands || !results) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    for (int i = 0; i < batch; i++) {
        uint64_t *c = &cands[(size_t)i * (size_t)al];
        for (int j = 0; j < al; j++)
            c[j] = rng64();
        c[0] |= 1ULL;               /* force odd */
        c[al - 1] |= (1ULL << 62);  /* keep top bits wide (~full AL*64 bits) */
    }

    /* Warm up */
    gpu_fermat_test_batch(ctx, cands, results, (size_t)batch);

    int primes = 0;
    double t0 = now_s();
    for (int k = 0; k < iters; k++) {
        int p = gpu_fermat_test_batch(ctx, cands, results, (size_t)batch);
        if (p < 0) {
            fprintf(stderr, "batch failed\n");
            return 1;
        }
        primes += p;
    }
    double t1 = now_s();

    double total = (double)batch * (double)iters;
    double dt = t1 - t0;
    printf("AL=%d (%d-bit)  kernel=%s  batch=%d  iters=%d  primes=%d\n",
           al, al * 64, gpu_fermat_kernel_label(al), batch, iters, primes);
    printf("  total time   : %.4f s\n", dt);
    printf("  per batch    : %.3f ms\n", dt * 1e3 / iters);
    printf("  throughput   : %.0f candidates/s\n", total / dt);
    printf("  per candidate: %.3f us\n", dt * 1e6 / total);

    free(cands);
    free(results);
    gpu_fermat_destroy(ctx);
    return 0;
}
