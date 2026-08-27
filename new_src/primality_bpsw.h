/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Primality Testing: Baillie-PSW
 *
 * Combines:
 * 1. Miller-Rabin with base 2 (first strong pseudoprime test)
 * 2. Strong Lucas-Selfridge test
 *
 * No composite number is known to pass both tests. This is a strong
 * probable-prime test, not a proof for arbitrary-size Gapcoin candidates.
 */

#ifndef PRIMALITY_BPSW_H
#define PRIMALITY_BPSW_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

/* Miller-Rabin with base 2 (strong pseudoprime test) */
int miller_rabin_base2(mpz_t n);

/* Strong Lucas-Selfridge test */
int lucas_lehmer_test(mpz_t n);

/* Full BPSW probable-prime test: Miller-Rabin(2) + strong Lucas-Selfridge */
int baillie_psw_test(mpz_t n);

/* Strong final probable-prime check for gap boundaries only */
int baillie_psw_verify_gap_boundaries(mpz_t p1, mpz_t p2);

#endif /* PRIMALITY_BPSW_H */
