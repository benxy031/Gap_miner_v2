/*
 * GpuVerify - standalone GPU-accelerated prime-gap verifier.
 * Independent of the gapminer and phase0 packages: no includes from
 * new_src/, phase0/, or tools/. Depends only on CUDA and GMP.
 */
#ifndef GV_H
#define GV_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Max supported modulus size on GPU: 32768 bits -> 1024 32-bit limbs.
 * Numbers with more digits run through the CPU (GMP BPSW) path. */
#define GV_MAX_LIMBS      1024
#define GV_MAX_BITS       (GV_MAX_LIMBS * 32)
#define GV_MAX_DIGITS_GPU (GV_MAX_BITS * 30103 / 100000 + 1) /* ~9865 */

/* Largest number we ever materialize (~4 GB): 1e9 decimal digits. */
#define GV_MAX_MATERIALIZE_DIGITS 1000000000LL

/* parse_expr return code: value too large to materialize */
#define GV_PARSE_TOO_LARGE -2

#define GV_BLOCK_THREADS 256

/* verdicts */
#define GV_OK             0  /* both endpoints BPSW-prime, interval clean   */
#define GV_COMPOSITE_END  1  /* an endpoint is provably composite           */
#define GV_PRIME_BETWEEN  2  /* a prime lies strictly inside the gap        */
#define GV_PARSE_FAIL     3  /* expression or row could not be parsed       */
#define GV_TOO_LARGE_CPU  4  /* >32768 bits: CPU-only verification          */

#define gv_die(msg) do { fprintf(stderr, "gpuverify: %s\n", (msg)); exit(1); } while (0)

#endif /* GV_H */
