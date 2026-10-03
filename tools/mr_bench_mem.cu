/* mr_bench_mem.cu - pure MR throughput WITH a memory touch per iteration,
   to test whether the walk's 47k-thread collapse is an L2-footprint effect.
   usage: mr_bench_mem GRID BLOCK MEMBYTES */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
typedef unsigned __int128 u128;
#include "perig.cuh"

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void mrbench_mem(const uint64_t * __restrict__ mem, uint64_t mask,
                            uint64_t v0lo, unsigned long long * d) {
    uint64_t s = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    unsigned long long hit = 0;
    for (int i = 0; i < 2000; i++, s += stride) {
        uint64_t w = mem[(s >> 3) & mask];          /* one 8B touch per test */
        uint64_t v = v0lo + 2ull * s + (w & 0ull);  /* keep the load live */
        hit += (unsigned long long)ciosFermatTest128(v);
    }
    if (hit) *d += hit;
}

int main(int argc, char **argv) {
    int grid = (argc > 1) ? atoi(argv[1]) : 46 * 8;
    int block = (argc > 2) ? atoi(argv[2]) : 256;
    size_t memb = (argc > 3) ? (size_t)atoll(argv[3]) : (size_t)(4ull << 20);
    uint64_t *dmem; cudaMalloc(&dmem, memb);
    cudaMemset(dmem, 1, memb);
    unsigned long long *d; cudaMalloc(&d, 8); cudaMemset(d, 0, 8);
    uint64_t mask = (memb / 8) - 1;
    uint64_t v0lo = 200000000000000000001ULL;
    mrbench_mem<<<grid, block>>>(dmem, mask, v0lo, d);   /* warmup */
    cudaDeviceSynchronize();
    cudaMemset(d, 0, 8);
    double t0 = now_s();
    mrbench_mem<<<grid, block>>>(dmem, mask, v0lo, d);
    cudaError_t e = cudaDeviceSynchronize();
    double t = now_s() - t0;
    double tests = (double)grid * block * 2000;
    printf("[mrmem] %s grid=%dx%d mem=%.1fMB tests=%.0f time=%.3f s -> %.1f M/s (%.2f ns/test)\n",
           cudaGetErrorString(e), grid, block, (double)memb / 1048576.0, tests, t,
           tests / t / 1e6, t / tests * 1e9);
    return 0;
}
