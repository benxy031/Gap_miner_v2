/*
 * GpuVerify - batched strong probable-prime test, base 2, on the GPU.
 *
 * One candidate per thread. For candidate N (odd):
 *   write N-1 = d * 2^s with d odd;
 *   compute y = 2^d mod N in Montgomery form (square-and-multiply over the
 *   bits of d, exponent bits read from the modulus itself);
 *   pass iff y == 1, or y becomes N-1 within s-1 squarings.
 * A failing strong base-2 test is a proof of compositeness; a passing one
 * is followed up on the host by GMP's BPSW (strong base-2 + Lucas).
 */
#include "mont.cuh"
#include "gv.h"

#define GV_CAND_ARR 5 /* n, m2(2R mod n), d, mont(N-1), mont(1)=R */

template <int N32>
__global__ void prp_kernel(const uint32_t *__restrict__ data,
                           const uint32_t *__restrict__ meta,
                           uint8_t *__restrict__ pass, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count)
        return;

    const uint32_t *src = data + (size_t)tid * N32 * GV_CAND_ARR;
    uint32_t n[N32], m2[N32], d[N32], mn1[N32], m1[N32];
#pragma unroll 4
    for (int i = 0; i < N32; i++) {
        n[i] = src[i];
        m2[i] = src[N32 + i];
        d[i] = src[2 * N32 + i];
        mn1[i] = src[3 * N32 + i];
        m1[i] = src[4 * N32 + i];
    }
    const uint32_t *mm = meta + (size_t)tid * 4;
    int s = (int)mm[1];
    int bl = (int)mm[2]; /* bit length of d */
    uint32_t np0 = mm[3];

    uint32_t y[N32];
#pragma unroll 4
    for (int i = 0; i < N32; i++)
        y[i] = m2[i];

    /* y = 2^d mod n */
    for (int bit = bl - 2; bit >= 0; bit--) {
        mont_sqr<N32>(y, y, n, np0);
        if ((d[bit >> 5] >> (bit & 31)) & 1)
            mont_mul<N32>(y, y, m2, n, np0);
    }

    bool ok = limb_eq<N32>(y, m1) || limb_eq<N32>(y, mn1);
    for (int i = 1; i < s && !ok; i++) {
        mont_sqr<N32>(y, y, n, np0);
        if (limb_eq<N32>(y, mn1)) {
            ok = true;
            break;
        }
    }
    pass[tid] = ok ? 1 : 0;
}

/* host-side launcher: bucket template instantiation */
template <int N32>
static cudaError_t launch_prp(const uint32_t *data, const uint32_t *meta,
                              uint8_t *pass, int count, cudaStream_t stream)
{
    int threads = GV_BLOCK_THREADS;
    int blocks = (count + threads - 1) / threads;
    prp_kernel<N32><<<blocks, threads, 0, stream>>>(data, meta, pass, count);
    return cudaGetLastError();
}

extern "C" cudaError_t gv_launch_prp_bucket(int n32, const uint32_t *data,
                                            const uint32_t *meta, uint8_t *pass,
                                            int count, cudaStream_t stream)
{
    switch (n32) {
    case 16:
        return launch_prp<16>(data, meta, pass, count, stream);
    case 32:
        return launch_prp<32>(data, meta, pass, count, stream);
    case 64:
        return launch_prp<64>(data, meta, pass, count, stream);
    case 128:
        return launch_prp<128>(data, meta, pass, count, stream);
    case 256:
        return launch_prp<256>(data, meta, pass, count, stream);
    case 512:
        return launch_prp<512>(data, meta, pass, count, stream);
    case 1024:
        return launch_prp<1024>(data, meta, pass, count, stream);
    default:
        return cudaErrorInvalidValue;
    }
}
