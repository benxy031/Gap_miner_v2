/* mr_bench.cu - raw mr_base2 throughput ceiling on this box (68-bit numbers). */
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <time.h>
typedef unsigned __int128 u128;
#include "mr68_kernel.cuh"

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec; }

__global__ void mrbench(uint64_t v0lo, uint64_t v0hi, uint64_t startSlot,
                        unsigned long long *hit) {
    unsigned long long local = 0;
    uint64_t s = startSlot + (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (int i = 0; i < 2000; i++, s += stride) {
        uint64_t lo2 = v0lo + (s << 1);
        uint32_t n[3] = { (uint32_t)lo2, (uint32_t)(lo2 >> 32),
                          (uint32_t)v0hi + (uint32_t)(lo2 < v0lo) };
        if (mr_base2(n)) local++;
    }
    atomicAdd(hit, local);
}

int main(int argc, char **argv) {
    uint64_t v0lo = 200000000000000000001ULL, v0hi = 10;
    unsigned long long *d; cudaMalloc(&d, 8); cudaMemset(d, 0, 8);
    int grid = (argc > 1) ? atoi(argv[1]) : 46 * 8, block = (argc > 2) ? atoi(argv[2]) : 256;
    mrbench<<<grid, block>>>(v0lo, v0hi, 0, d);   /* warmup */
    cudaDeviceSynchronize();
    cudaMemset(d, 0, 8);
    double t0 = now_s();
    mrbench<<<grid, block>>>(v0lo, v0hi, 123456789ULL, d);
    cudaError_t e = cudaDeviceSynchronize();
    double t = now_s() - t0;
    unsigned long long hit = 0; cudaMemcpy(&hit, d, 8, cudaMemcpyDeviceToHost);
    double tests = (double)grid * block * 2000;
    printf("[mrbench] %s tests=%.0f  hit=%llu  time=%.3f s -> %.1f M tests/s (%.2f ns/test)\n",
           cudaGetErrorString(e), tests, hit, t, tests / t / 1e6, t / tests * 1e9);
    return 0;
}
