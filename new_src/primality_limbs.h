/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Fixed-limb CPU Montgomery primality tests (ported from cpugapminer's
 * src/primality_utils.h, refactored for gapminer_v2).
 *
 * Fast base-2 Fermat / Euler–Plumb tests for multi-limb integers stored
 * as little-endian 64-bit word arrays, using CIOS Montgomery multiplication
 * with __uint128_t (and ADX/BMI2 inner loops with runtime CPUID detection).
 * Several times faster than GMP mpz_powm for fixed limb counts.
 */

#ifndef PRIMALITY_LIMBS_H
#define PRIMALITY_LIMBS_H

#include <stdint.h>
#include <gmp.h>

/* Maximum limb count supported by fermat_test_cpu_nlimbs().
 * 20 limbs = 1280 bits, covering shifts up to 1024. */
#define PRIMALITY_CPU_MAX_LIMBS 20

#define PRIMALITY_PRECOMP_WIN_ODD_COUNT 8

/* Exact-path precompute state for reusing candidate-specific Montgomery setup
 * across multiple base-2 tests on the same odd candidate.  Caches the
 * Montgomery base/window setup plus the prebuilt exponents for win=4
 * exact-path execution. */
typedef struct {
	int nlimbs;
	int win_bits;
	uint64_t ninv;
	uint64_t n[PRIMALITY_CPU_MAX_LIMBS];
	uint64_t base_m[PRIMALITY_CPU_MAX_LIMBS];
	uint64_t win[PRIMALITY_PRECOMP_WIN_ODD_COUNT][PRIMALITY_CPU_MAX_LIMBS];
	uint64_t nm1[PRIMALITY_CPU_MAX_LIMBS];
	uint64_t fermat_exp[PRIMALITY_CPU_MAX_LIMBS];
	uint64_t euler_exp[PRIMALITY_CPU_MAX_LIMBS];
	int fermat_msb;
	int euler_msb;
} primality_exact_precomp_t;

/* 64-bit helpers (same Montogomery core, single limb). */
int primality_miller_rabin_u64(uint64_t n);
int primality_fast_fermat_u64(uint64_t n);

/* Base-2 Fermat test: 2^(n-1) ≡ 1 (mod n).
 * Returns 1 (probably prime) or 0 (composite). */
int fermat_test_cpu_nlimbs(const uint64_t *n, int nlimbs);
int primality_exact_precomp_init(primality_exact_precomp_t *precomp,
								 const uint64_t *n, int nlimbs);
int fermat_test_cpu_nlimbs_precomp(const primality_exact_precomp_t *precomp);

/* Euler–Plumb criterion: 2^((n-1)/2) ≡ ±1 (mod n).
 * ~50% fewer squarings than the Fermat test; comparison done in Montgomery
 * form (no final de-Montgomery multiply). */
int euler_test_cpu_nlimbs(const uint64_t *n, int nlimbs);
int euler_test_cpu_nlimbs_precomp(const primality_exact_precomp_t *precomp);

/* CPU Montgomery backend feature status.
 * On ADX/BMI2 builds, runtime detection may disable the ADX path and use the
 * portable CIOS fallback when the host CPU lacks the required instructions. */
int primality_cpu_adx_compiled(void);
int primality_cpu_adx_enabled(void);

/* mpz adapters + limb helpers for the worker CPU paths. */
int primality_limbs_effective_nl(const uint64_t *limbs, int max_nl);
int primality_limbs_add_u64(uint64_t *limbs, int max_nl, uint64_t delta);
int primality_limbs_export(const mpz_t n, uint64_t *limbs, int max_nl);
int primality_euler_limbs_mpz(const mpz_t n);
int primality_fermat_limbs_mpz(const mpz_t n);

#endif /* PRIMALITY_LIMBS_H */
