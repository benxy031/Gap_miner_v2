/*
 * bench_mark_ceiling.cu - what is the actual ceiling of the class-30 mark loop
 * on this card, and which component of it binds?
 *
 * WHY: production mark (class30_sieve_kernel, tile = 753664 slots / 92 KB,
 * 512 threads, 1 CTA/SM) runs 1.39 s per 1e12 integers at --walk-primes
 * 150000, i.e. 3.72e11 marks / 1.39 s = 268 G marks/s = 3.2 marks/cycle/SM at
 * 1.8 GHz.  That is far below any plausible LSU bound, so the binding
 * constraint is not obvious.  The depth sweep (P=15000 vs P=150000) showed
 * 1440 vs 1026 G marks/s measured in Sigma(1/p) units, which is a row-setup
 * effect and not a per-mark effect.
 *
 * This probe isolates, one factor at a time, on the SAME shared-memory size,
 * block size and address statistics:
 *   A chain    atomicOr, positions advanced by a per-thread step (a serial
 *              address chain - the production shape's dependency)
 *   B indep    atomicOr, eight independent address chains per thread (removes
 *              the serial dependency; same instruction count and same
 *              address-set statistics)
 *   C plain    `|=` instead of atomicOr on the chain (bounds the atomic
 *              premium; NOT a correct sieve - only a cost measurement)
 *   D nostore  identical loop with the store folded into an accumulator whose
 *              use is data dependent (the loop cannot be optimised away)
 *   E ceiling  atomicOr on a per-lane constant address (no index arithmetic at
 *              all: the raw shared-atomic issue rate)
 *
 * Every arm issues the same number of marks (M per thread) so the rates are
 * directly comparable.  Arms are timed with cudaEvent pairs and the reported
 * rate is marks / (elapsed * SMs * clock) so it can be compared with the
 * per-cycle limits (LSU: 32 shared accesses/cycle/SM on sm_86).
 *
 * Build: nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -O2 tools/bench_mark_ceiling.cu -o bin/bench_mark_ceiling
 * Usage: ./bin/bench_mark_ceiling [threads] [marks_per_thread] [reps]
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)

/* Production geometry: 753664 slots = 11776 words = 92 KB of shared. */
static const uint32_t TILE_SLOTS = 753664u;
static const uint32_t TILE_WORDS = TILE_SLOTS / 64u;

__device__ __forceinline__ uint32_t xorshift(uint32_t &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

/* A: one serial address chain per thread, atomicOr per mark. */
__global__ void k_chain(uint32_t marks, uint32_t seed) {
    extern __shared__ uint64_t sh[];
    for (uint32_t w = threadIdx.x; w < TILE_WORDS; w += blockDim.x) sh[w] = 0ull;
    __syncthreads();
    uint32_t s = seed + threadIdx.x * 2654435761u;
    uint32_t pos = xorshift(s) % TILE_SLOTS;
    uint32_t step = 3u + (xorshift(s) % (TILE_SLOTS / 8u));
    for (uint32_t i = 0; i < marks; i++) {
        pos += step;
        if (pos >= TILE_SLOTS) pos -= TILE_SLOTS;
        atomicOr(((unsigned int *)sh) + (pos >> 5), 1u << (pos & 31u));
    }
    __syncthreads();
    if (threadIdx.x == 0 && sh[0] == 0xdeadbeefull) printf("");
}

/* B: eight independent chains (no serial dependency between consecutive
   atomics), identical mark count and identical address statistics. */
__global__ void k_indep(uint32_t marks, uint32_t seed) {
    extern __shared__ uint64_t sh[];
    for (uint32_t w = threadIdx.x; w < TILE_WORDS; w += blockDim.x) sh[w] = 0ull;
    __syncthreads();
    uint32_t s = seed + threadIdx.x * 2654435761u;
    uint32_t pos[8], step[8];
    for (int j = 0; j < 8; j++) {
        pos[j] = xorshift(s) % TILE_SLOTS;
        step[j] = 3u + (xorshift(s) % (TILE_SLOTS / 8u));
    }
    uint32_t outer = marks >> 3;
    for (uint32_t i = 0; i < outer; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            pos[j] += step[j];
            if (pos[j] >= TILE_SLOTS) pos[j] -= TILE_SLOTS;
            atomicOr(((unsigned int *)sh) + (pos[j] >> 5), 1u << (pos[j] & 31u));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && sh[0] == 0xdeadbeefull) printf("");
}

/* C: plain |= (racy; cost measurement only). */
__global__ void k_plain(uint32_t marks, uint32_t seed) {
    extern __shared__ uint64_t sh[];
    for (uint32_t w = threadIdx.x; w < TILE_WORDS; w += blockDim.x) sh[w] = 0ull;
    __syncthreads();
    uint32_t s = seed + threadIdx.x * 2654435761u;
    uint32_t pos = xorshift(s) % TILE_SLOTS;
    uint32_t step = 3u + (xorshift(s) % (TILE_SLOTS / 8u));
    for (uint32_t i = 0; i < marks; i++) {
        pos += step;
        if (pos >= TILE_SLOTS) pos -= TILE_SLOTS;
        uint32_t *w32 = ((uint32_t *)sh) + (pos >> 5);
        *w32 |= 1u << (pos & 31u);
    }
    __syncthreads();
    if (threadIdx.x == 0 && sh[0] == 0xdeadbeefull) printf("");
}

/* D: no store at all - the loop's arithmetic cost only. */
__global__ void k_nostore(uint32_t marks, uint32_t seed) {
    extern __shared__ uint64_t sh[];
    if (threadIdx.x == 0) sh[0] = 0ull;
    __syncthreads();
    uint32_t s = seed + threadIdx.x * 2654435761u;
    uint32_t pos = xorshift(s) % TILE_SLOTS;
    uint32_t step = 3u + (xorshift(s) % (TILE_SLOTS / 8u));
    uint32_t acc = 0;
    for (uint32_t i = 0; i < marks; i++) {
        pos += step;
        if (pos >= TILE_SLOTS) pos -= TILE_SLOTS;
        acc += pos >> 5;
    }
    __syncthreads();
    if (acc == 0xdeadbeefu) atomicOr(((unsigned int *)sh), 1u);   /* keeps it live */
}

/* E: raw shared-atomic issue rate - per-lane FIXED address, no index math. */
__global__ void k_ceiling(uint32_t marks) {
    extern __shared__ uint64_t sh[];
    for (uint32_t w = threadIdx.x; w < TILE_WORDS; w += blockDim.x) sh[w] = 0ull;
    __syncthreads();
    uint32_t *base = (uint32_t *)sh;
    uint32_t lane = threadIdx.x & 31u;              /* 32 distinct banks */
    for (uint32_t i = 0; i < marks; i++) atomicOr(base + lane, 1u);
    __syncthreads();
    if (threadIdx.x == 0 && base[0] == 0xdeadbeefu) printf("");
}

/* F: atomic where all 32 lanes of a warp hit the SAME address (worst case). */
__global__ void k_conflict(uint32_t marks) {
    extern __shared__ uint64_t sh[];
    for (uint32_t w = threadIdx.x; w < TILE_WORDS; w += blockDim.x) sh[w] = 0ull;
    __syncthreads();
    uint32_t *base = (uint32_t *)sh;
    for (uint32_t i = 0; i < marks; i++) atomicOr(base, 1u);
    __syncthreads();
    if (threadIdx.x == 0 && base[0] == 0xdeadbeefu) printf("");
}

typedef void (*fn_t)(uint32_t, uint32_t);
typedef void (*fn1_t)(uint32_t);

static int G_BLOCKS = 46;

static float time_arm(fn_t f, uint32_t marks, uint32_t seed, uint32_t threads, int reps, size_t shbytes) {
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, 0));
        f<<<G_BLOCKS, threads, shbytes>>>(marks, seed);
        CK(cudaEventRecord(b, 0));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    return best;
}
static float time_arm1(fn1_t f, uint32_t marks, uint32_t threads, int reps, size_t shbytes) {
    float best = 1e30f;
    for (int r = 0; r < reps; r++) {
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a, 0));
        f<<<G_BLOCKS, threads, shbytes>>>(marks);
        CK(cudaEventRecord(b, 0));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    }
    return best;
}

int main(int argc, char **argv) {
    uint32_t threads = (argc > 1) ? (uint32_t)atoi(argv[1]) : 512u;
    uint32_t marks   = (argc > 2) ? (uint32_t)atoi(argv[2]) : 40000u;
    int reps         = (argc > 3) ? atoi(argv[3]) : 5;
    size_t shbytes = (size_t)TILE_WORDS * 8;

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    CK(cudaFuncSetAttribute(k_chain,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));
    CK(cudaFuncSetAttribute(k_indep,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));
    CK(cudaFuncSetAttribute(k_plain,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));
    CK(cudaFuncSetAttribute(k_nostore, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));
    CK(cudaFuncSetAttribute(k_ceiling, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));
    CK(cudaFuncSetAttribute(k_conflict,cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shbytes));

    double sms = p.multiProcessorCount, clk = p.clockRate * 1e3;   /* Hz */
    G_BLOCKS = (int)sms;                    /* one 92 KB CTA per SM */
    double total = (double)marks * threads * (double)G_BLOCKS;
    printf("device: %s  SMs=%d  clock=%.3f GHz  shared/CTA=%zu KB  threads=%u  marks/thread=%u  blocks=%d\n",
           p.name, (int)sms, clk / 1e9, shbytes / 1024, threads, marks, G_BLOCKS);
    printf("%-10s %10s %12s %12s\n", "arm", "ms", "Gmarks/s", "marks/cyc/SM");

    struct { const char *name; fn_t f; } arms[] = {
        {"A chain", k_chain}, {"B indep", k_indep}, {"C plain", k_plain}, {"D nostore", k_nostore},
    };
    for (auto &a : arms) {
        float ms = time_arm(a.f, marks, 12345u, threads, reps, shbytes);
        double g = total / (ms * 1e6);
        printf("%-10s %10.3f %12.1f %12.2f\n", a.name, ms, g, g * 1e9 / (sms * clk));
    }
    {
        float ms = time_arm1(k_ceiling, marks, threads, reps, shbytes);
        double g = total / (ms * 1e6);
        printf("%-10s %10.3f %12.1f %12.2f\n", "E ceiling", ms, g, g * 1e9 / (sms * clk));
    }
    {
        float ms = time_arm1(k_conflict, marks, threads, reps, shbytes);
        double g = total / (ms * 1e6);
        printf("%-10s %10.3f %12.1f %12.2f\n", "F conflict", ms, g, g * 1e9 / (sms * clk));
    }
    printf("\nnote: F is per-LANE marks; all 32 lanes of a warp hit one address,\n"
           "      so its useful rate is the same number but with full serialisation.\n");
    return 0;
}
