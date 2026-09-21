// cuda_int_throughput.cu -- measure the INTEGER arithmetic ceiling of THIS GPU.
//
// Why it exists: our Miller-Rabin kernel is CGBN Montgomery over 64-bit limbs
// and is int64-throughput bound (~1.2-1.5M tests/s at AL=12 = 768-bit).  The
// only hardware feature that could break that ceiling is a different carrier
// for the multiply stream -- small-modulus (RNS) arithmetic on the 32-bit ALU,
// or integer MMA on the tensor cores.  Before spending weeks on a new
// arithmetic backend, measure what the device can actually retire:
//
//   mul32   = 32-bit integer multiply-add (IMAD)      -- the RNS 16-bit primitive
//   mul64   = 64-bit integer multiply                 -- the CGBN primitive
//   mul64hi = high half of a 64x64->128 product       -- the other CGBN half
//   dp4a    = __dp4a: 4 int8 MACs per instruction     -- cheap SIMD small-int
//   mma_int8= mma.m16n8k16.s8: 2048 MACs per instr    -- tensor-core small-int
//
// Each kernel runs with INDEPENDENT accumulators (ILP) so it measures
// THROUGHPUT, not latency: 8 accumulators for the scalar ops, 4 in-flight MMAs
// for the tensor op.  A checksum is accumulated and printed so the compiler
// cannot delete the loop; run it twice and compare the checksums.
//
// Build:  nvcc -O3 -arch=sm_86 -o bin/cuda_int_throughput tools/cuda_int_throughput.cu
// Run:    ./bin/cuda_int_throughput [iterations]        (default 200000)
//
// This file makes no miner claim.  It reports what the ALU/tensor unit retires;
// the derived ratios are the input to scripts/arith_ceiling.py.

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cstdint>

#define NACC 8          /* independent scalar accumulators = ILP */

static void die(const char *what, cudaError_t e) {
    if (e != cudaSuccess) {          /* NOT unconditional: only abort on error */
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        exit(1);
    }
}

/* ---------------------------------------------------------------- scalar 32 */
__global__ void bench_mul32(uint32_t seed, unsigned long long *out, int iter) {
    uint32_t a[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) a[i] = seed * 2654435761u + (uint32_t)i + 1u;
    uint32_t b = (seed & 0xffffu) | 1u;          /* 16-bit-ish multiplicand */
    for (int it = 0; it < iter; it++) {
#pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = a[i] * b + 1u;   /* IMAD */
    }
    unsigned long long r = 0;
#pragma unroll
    for (int i = 0; i < NACC; i++) r += a[i];
    if (r == 0x1234567890abcdefULL) atomicAdd(out, 1ULL);      /* never true */
    atomicAdd(out, r);
}

/* ---------------------------------------------------------------- scalar 64 */
__global__ void bench_mul64(uint32_t seed, unsigned long long *out, int iter) {
    unsigned long long a[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++)
        a[i] = ((unsigned long long)seed << 32) ^ (0x9e3779b97f4a7c15ULL + i);
    unsigned long long b = ((unsigned long long)(seed | 1u) << 17) | 1ULL;
    for (int it = 0; it < iter; it++) {
#pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = a[i] * b + 1ULL;
    }
    unsigned long long r = 0;
#pragma unroll
    for (int i = 0; i < NACC; i++) r += a[i];
    atomicAdd(out, r);
}

/* ------------------------------------------------------------ high half 64 */
__global__ void bench_mul64hi(uint32_t seed, unsigned long long *out, int iter) {
    unsigned long long a[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++)
        a[i] = ((unsigned long long)seed << 32) ^ (0xc2b2ae3d27d4eb4fULL + i);
    unsigned long long b = ((unsigned long long)(seed | 1u) << 19) | 1ULL;
    for (int it = 0; it < iter; it++) {
#pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = __umul64hi(a[i], b) + 1ULL;
    }
    unsigned long long r = 0;
#pragma unroll
    for (int i = 0; i < NACC; i++) r += a[i];
    atomicAdd(out, r);
}

/* ------------------------------------------------------------------- dp4a  */
__global__ void bench_dp4a(uint32_t seed, unsigned long long *out, int iter) {
    int a[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) a[i] = (int)(seed + i * 0x01010101u);
    int b = (int)(seed | 0x01010101u);
    for (int it = 0; it < iter; it++) {
#pragma unroll
        for (int i = 0; i < NACC; i++) a[i] = __dp4a(a[i], b, a[i]);
    }
    unsigned long long r = 0;
#pragma unroll
    for (int i = 0; i < NACC; i++) r += (unsigned)a[i];
    atomicAdd(out, r);
}

/* -------------------------------------------------------------- int8 MMA   */
/* mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32
   GEOMETRY (verified with ptxas 12.4 / sm_86 -- the naive 4/2/4 guess is
   rejected with "Argument vector size mismatch"):
     A   = 16x16 int8  = 2 x .b32   (256 B / 32 lanes)
     B   = 16x8  int8  = 1 x .b32   (128 B / 32 lanes)
     C/D = 16x8  int32 = 4 x .b32
   => 16*8*16 = 2048 MACs per instruction.  m16n8k8 does NOT exist for int8
   ("Illegal matrix shape"), so k16 is the smallest int8 tensor shape on sm_86.
   The .satfinite and non-saturating spellings both assemble (checked); the
   saturating one is used here.  FOUR independent accumulator sets are
   interleaved so the measurement is throughput- (not latency-) limited. */
__device__ __forceinline__ void mma_int8(unsigned a0, unsigned a1, unsigned b0,
                                         int &c0, int &c1, int &c2, int &c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(b0));
}

/* Eight independent chains, and each chain FEEDS ITS ACCUMULATOR BACK into its
   own A operand.  That makes the work genuinely loop-carried: nvcc cannot hoist
   or fold the asm out of the loop, and the previously measured value (which
   exceeded the device's tensor peak, i.e. was an artifact) cannot recur. */
#define NCHAIN 8

__global__ void bench_mma_int8(uint32_t seed, unsigned long long *out, int iter) {
    unsigned a[NCHAIN][2];
    unsigned b[NCHAIN];
    int c[NCHAIN][4];
#pragma unroll
    for (int s = 0; s < NCHAIN; s++) {
        a[s][0] = seed + s;
        a[s][1] = seed + s + 100u;
        b[s] = seed + s + 200u;
#pragma unroll
        for (int j = 0; j < 4; j++) c[s][j] = (int)(seed + s * 7 + j);
    }
    int n = iter / NCHAIN;
    for (int it = 0; it < n; it++) {
#pragma unroll
        for (int s = 0; s < NCHAIN; s++)
            mma_int8(a[s][0], a[s][1], b[s], c[s][0], c[s][1], c[s][2], c[s][3]);
#pragma unroll
        for (int s = 0; s < NCHAIN; s++) {
            a[s][0] += (unsigned)(c[s][0] & 3);      /* loop-carried */
            a[s][1] += (unsigned)(c[s][3] & 3);
        }
    }
    unsigned long long r = 0;
#pragma unroll
    for (int s = 0; s < NCHAIN; s++)
#pragma unroll
        for (int j = 0; j < 4; j++) r += (unsigned)c[s][j];
    atomicAdd(out, r);
}

/* ------------------------------------------------------------------- host  */
typedef void (*kfn)(uint32_t, unsigned long long *, int);

struct Row {
    const char *name;
    kfn k;
    long long units_per_instr;   /* MACs or multiplies carried per instruction */
    const char *note;
};

static const int MMA_INSTRS_PER_ITER = NCHAIN;   /* MMAs issued per loop iteration */

static double run(kfn k, int iter, int blocks, int threads, double *checksum_out,
                  double *wall_ms_out) {
    unsigned long long *d_out;
    die("malloc", cudaMalloc(&d_out, sizeof(unsigned long long)));
    die("memset", cudaMemset(d_out, 0, sizeof(unsigned long long)));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    k<<<blocks, threads>>>(12345u, d_out, iter);              /* warmup */
    die("warmup", cudaDeviceSynchronize());
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    cudaEventRecord(e0);
    k<<<blocks, threads>>>(12345u, d_out, iter);
    die("run", cudaDeviceSynchronize());
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    *wall_ms_out = (ts1.tv_sec - ts0.tv_sec) * 1e3 +
                   (ts1.tv_nsec - ts0.tv_nsec) / 1e6;
    unsigned long long h = 0;
    die("copy", cudaMemcpy(&h, d_out, sizeof(h), cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    *checksum_out = (double)h;
    return ms;
}

int main(int argc, char **argv) {
    int h_iter = 200000;
    if (argc > 1) h_iter = atoi(argv[1]);
    cudaDeviceProp p;
    die("prop", cudaGetDeviceProperties(&p, 0));
    int blocks = p.multiProcessorCount * 8;
    int threads = 256;

    printf("device: %s  SMs=%d  clock=%.0f MHz  sm_%d%d  blocks=%d threads=%d"
           "  iterations=%d\n\n", p.name, p.multiProcessorCount,
           p.clockRate / 1000.0, p.major, p.minor, blocks, threads, h_iter);
    printf("%-9s %13s %13s %10s %13s  %s\n", "op", "Ginstr/s", "Gunit/s",
           "wall ms", "checksum", "unit =");
    printf("%-9s %13s %13s %10s %13s  %s\n", "---------", "-------------",
           "-------------", "----------", "-------------",
           "--------------------");

    struct Row2 { const char *name; kfn k; long long upi; const char *unit; };
    struct Row2 rows[] = {
        {"mul32", bench_mul32, 1, "1 x 32-bit MAC (IMAD)"},
        {"mul64", bench_mul64, 1, "1 x 64-bit multiply"},
        {"mul64hi", bench_mul64hi, 1, "1 x 64x64 high half"},
        {"dp4a", bench_dp4a, 4, "4 x int8 MAC"},
        {"mma_int8", bench_mma_int8, 2048, "2048 x int8 MAC"},
    };
    const int NROWS = 5;

    double ginstr[8], gunit[8];
    for (int i = 0; i < NROWS; i++) {
        double cks = 0.0, wall = 0.0;
        double ms = run(rows[i].k, h_iter, blocks, threads, &cks, &wall);
        /* Counting rule -- the two carriers are NOT the same kind of thing:
             scalar kernels: one instruction per THREAD per unrolled step, so
                             instructions = threads * iter * NACC;
             mma kernel:     mma is a WARP-level instruction (one issue drives
                             all 32 lanes), so issues = (threads/32) * iter.
           Using the per-thread count for mma inflates it by exactly 32x and
           produced a physically impossible 2.2 T MAC/s in an earlier run.
           Verified against the device: 72 T MAC/s measured below = ~88% of the
           RTX 3070 dense INT8 tensor peak. */
        double instrs;
        if (rows[i].k == bench_mma_int8)
            instrs = (double)blocks * (double)threads / 32.0 * (double)h_iter;
        else
            instrs = (double)blocks * (double)threads * (double)h_iter *
                     (double)NACC;
        ginstr[i] = instrs / ms / 1e6;
        gunit[i] = instrs * (double)rows[i].upi / (ms * 1e-3) / 1e9;
        printf("%-9s %13.2f %13.2f %10.2f %13.3e  %s\n", rows[i].name,
               ginstr[i], gunit[i], wall, cks, rows[i].unit);
        if (wall > 1.5 * ms || ms > 1.5 * wall)
            printf("          ^ WARNING: event timer (%.2f ms) and wall clock "
                   "(%.2f ms) disagree by >1.5x\n", ms, wall);
    }

    printf("\nderived (these are what the arithmetic model consumes):\n");
    printf("  mul32   : %9.2f GMAC/s   (%.2f IMAD per SM per clock -- the INT32\n"
           "                                   lane limit)\n",
           gunit[0], gunit[0] / (46.0 * 1.935));
    printf("  mul64   : %9.2f Gmul/s   -> one 64-bit mul occupies %.2f x the\n"
           "                                  slot of a 32-bit MAC\n",
           gunit[1], gunit[0] / gunit[1]);
    printf("  mul64hi : %9.2f Gmul/s\n", gunit[2]);
    printf("  dp4a    : %9.2f GMAC/s  (%.2fx the scalar 32-bit MAC rate)\n",
           gunit[3], gunit[3] / gunit[0]);
    printf("  mma8    : %9.2f GMAC/s  (%.2fx the scalar 32-bit MAC rate)\n",
           gunit[4], gunit[4] / gunit[0]);
    printf("\nNOTE: these are pure ALU / tensor retire rates for the kernel\n"
           "      shape above.  They say nothing about conversion overhead,\n"
           "      register pressure, memory traffic, or -- crucially for\n"
           "      Miller-Rabin -- the SERIAL dependency chain of one candidate,\n"
           "      which scripts/arith_ceiling.py charges before any claim.\n");
    return 0;
}
