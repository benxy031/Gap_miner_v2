#ifndef GV_HOST_H
#define GV_HOST_H

#include <gmp.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* init GPU (idempotent); returns 0 if a CUDA device is usable, -1 otherwise */
int gv_gpu_init(void);
int gv_gpu_available(void);

/*
 * Batch strong PRP test base 2 for count numbers.
 * pass_out[i]: 1 = passed strong base-2 PRP (needs host BPSW follow-up),
 *              0 = composite (proved by the base-2 witness),
 *              2 = too large for the GPU (caller must use the CPU path).
 * Returns 0 on success, -1 on CUDA error.
 */
int gv_gpu_prp_batch(mpz_t *nums, int count, uint8_t *pass_out);

/*
 * Sieve the interval (N, N+gap): returns a host buffer marks[] of
 * (gap/2 - 1) bytes, marks[i]==0 iff the odd candidate N + 2*(i+1)
 * survived all primes <= pmax. Caller frees with free().
 */
int gv_gpu_sieve_gap(mpz_t N, uint64_t gap, uint32_t pmax,
                     uint8_t **marks_out, uint64_t *half_gap_out);

/* BPSW via GMP (strong base-2 + Lucas); 1 = probable prime, 0 = composite */
int gv_bpsw(mpz_t n);

/* CPU reference implementation of the strong base-2 PRP test (selftests) */
int gv_ref_strong_prp2(mpz_t n);

/* host-side prime list up to pmax (static, regenerated when pmax grows) */
int gv_get_primes(uint32_t pmax, const uint32_t **primes_out, int *count_out);

#ifdef __cplusplus
}
#endif

#endif /* GV_HOST_H */
