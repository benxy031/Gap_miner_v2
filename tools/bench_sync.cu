/* bench_sync: what does ONE host<->device round trip cost on this box?
 *
 * Motivation (2026-10-25): the gap-hunt fill pays exactly one
 * `cudaMemcpyAsync(count, D2H) + cudaStreamSynchronize` per window, and the
 * fill is ~181 us/window while the kernels it waits for sum to ~48 us
 * (isolated pair-mark 38 us + extract ~10 us).  Everything else that could
 * explain the gap has been falsified in the live pipeline: mark chunk size,
 * prime-table depth 2e5..1e7, extract kernel count (fused vs 3 kernels),
 * split on/off, CPU C-state (3 spinners +3% only), and the pageable count
 * copy (pinning it bought 0).
 *
 * This bench isolates the round trip itself, with the GPU otherwise idle, so
 * the number is a LOWER BOUND on the per-window latency the fill pays.
 *
 * Build: make bin/bench_sync WITH_CUDA=1   (or nvcc -O2 -arch=sm_86)
 * Run:   ./bin/bench_sync [--iters N] [--work KERNELS]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

__global__ static void touch_kernel(unsigned int *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = (unsigned int)(i * 3 + 1);
}

int main(int argc, char **argv) {
    int iters = 1000;
    int work = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work") && i + 1 < argc) work = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--iters N] [--work KERNELS]\n",
                       argv[0]); return 1; }
    }
    int dev = 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 1;
    printf("device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    unsigned int *d = NULL, *h_pinned = NULL, h_page = 0;
    cudaStream_t s;
    cudaStreamCreate(&s);
    if (cudaMalloc(&d, 4096 * sizeof(unsigned int)) != cudaSuccess) return 1;
    if (cudaHostAlloc((void **)&h_pinned, 64 * sizeof(unsigned int),
                      cudaHostAllocDefault) != cudaSuccess) return 1;

    /* Warm up (context, JIT, clocks). */
    for (int i = 0; i < 50; i++) {
        touch_kernel<<<16, 256, 0, s>>>(d, 4096);
        cudaStreamSynchronize(s);
    }

    double t0, t1;

    /* 1. Bare sync round trip: launch one tiny kernel, then sync. */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        touch_kernel<<<16, 256, 0, s>>>(d, 4096);
        cudaStreamSynchronize(s);
    }
    t1 = now_us();
    printf("launch+sync (GPU idle)        : %8.2f us/iter\n",
           (t1 - t0) / iters);

    /* 2. Same, but `work` kernels queued before the sync. */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        for (int k = 0; k < work; k++)
            touch_kernel<<<16, 256, 0, s>>>(d, 4096);
        cudaStreamSynchronize(s);
    }
    t1 = now_us();
    printf("launch x%-3d + sync            : %8.2f us/iter\n", work,
           (t1 - t0) / iters);

    /* 3. Count readback into PINNED host memory + sync. */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        touch_kernel<<<16, 256, 0, s>>>(d, 4096);
        cudaMemcpyAsync(h_pinned, d, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    t1 = now_us();
    printf("4B D2H (pinned) + sync        : %8.2f us/iter\n",
           (t1 - t0) / iters);

    /* 4. Count readback into PAGEABLE host memory + sync (what the fill did
          before 2026-10-25). */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        touch_kernel<<<16, 256, 0, s>>>(d, 4096);
        cudaMemcpyAsync(&h_page, d, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
    }
    t1 = now_us();
    printf("4B D2H (pageable) + sync      : %8.2f us/iter\n",
           (t1 - t0) / iters);

    /* 5. Host wake-up cost with no GPU involved at all. */
    t0 = now_us();
    for (int i = 0; i < iters; i++) {
        if (cudaStreamQuery(s) != cudaSuccess) cudaStreamSynchronize(s);
    }
    t1 = now_us();
    printf("cudaStreamQuery (no work)     : %8.2f us/iter\n",
           (t1 - t0) / iters);

    cudaFreeHost(h_pinned);
    cudaFree(d);
    cudaStreamDestroy(s);
    return 0;
}
