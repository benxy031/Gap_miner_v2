/*
 * GpuVerify - GPU interval sieve.
 *
 * N is odd; interior candidates N+k with even k (2, 4, ..., gap-2) are the
 * odd numbers. marks[i] corresponds to k = 2*(i+1); a zero byte means the
 * candidate survived all sieving primes.
 *
 * The host precomputes, for every prime p, the smallest even k >= 0 with
 * N+k == 0 (mod p), adjusted so that the candidate N+k==p itself is never
 * marked (that candidate is the prime p), and passes the array in.
 * The kernel never touches big integers.
 *
 * Concurrent threads write distinct bytes (or the same byte with the same
 * value 1), so plain stores are race-free in effect.
 */
#include <stdint.h>

extern "C" __global__ void sieve_kernel(uint8_t *__restrict__ marks,
                                        const uint64_t *__restrict__ starts,
                                        const uint32_t *__restrict__ primes,
                                        uint64_t gap, int nprimes)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nprimes)
        return;
    uint64_t k = starts[j];
    if (k >= gap)
        return;
    uint64_t stride = 2ull * (uint64_t)primes[j];
    for (; k < gap; k += stride)
        marks[(k >> 1) - 1] = 1;
}

/* reset the marks buffer between runs */
extern "C" __global__ void marks_clear_kernel(uint8_t *marks, uint64_t half_gap)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half_gap)
        return;
    marks[i] = 0;
}

/* host-side launchers */
extern "C" void launch_marks_clear(uint8_t *marks, uint64_t half_gap)
{
    int threads = 256;
    int blocks = (int)((half_gap + threads - 1) / threads);
    marks_clear_kernel<<<blocks, threads>>>(marks, half_gap);
}

extern "C" void launch_sieve(uint8_t *marks, const uint64_t *starts,
                             const uint32_t *primes, uint64_t gap, int nprimes)
{
    int threads = 256;
    int blocks = (nprimes + threads - 1) / threads;
    sieve_kernel<<<blocks, threads>>>(marks, starts, primes, gap, nprimes);
}
