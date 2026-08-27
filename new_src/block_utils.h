/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Block Template Utilities
 *
 * Convert Gapcoin bits format to shift/target for mining validation
 * 
 * Usage:
 *   uint32_t bits = parse_bits("15589358449e46");
 *   uint32_t shift = bits_to_shift(bits);
 *   // Koristi shift u merit formuli: gap_len / (ln(p1) + shift * ln(2))
 */

#ifndef BLOCK_UTILS_H
#define BLOCK_UTILS_H

#include <stdint.h>
#include <gmp.h>

/*
 * Parse hex string "15589358449e46" (8 hex chars) to uint32_t
 * Returns: uint32_t where upper byte is exponent, lower 3 bytes are mantissa
 */
uint32_t parse_bits(const char *bits_hex);

/*
 * Decode bits (uint32_t) into exponent and mantissa components
 * bits format (little-endian, 4 bytes):
 *   Byte 0: exponent (e.g., 0x15 = 21)
 *   Bytes 1-3: mantissa (e.g., 0x589358)
 */
void decode_bits(uint32_t bits, uint32_t *out_exponent, uint32_t *out_mantissa);

/*
 * Convert Gapcoin bits difficulty to shift factor (in bits)
 * 
 * Gapcoin uses shift to quantify the nonce space size:
 *   shift = log2(target) ≈ 8*(exponent-3) + log2(mantissa)
 * 
 * Used in merit calculation:
 *   merit = gap_length / (ln(p1) + shift * ln(2))
 */
uint32_t bits_to_shift(uint32_t bits);

/*
 * Convert Gapcoin bits to full target value (GMP mpz_t)
 * 
 * target = mantissa × 2^(8×(exponent-3))
 * 
 * Example:
 *   bits = 0x15589358 (exponent=0x15=21, mantissa=0x589358)
 *   target = 0x589358 × 2^(8×(21-3))
 *          = 0x589358 × 2^144
 */
void bits_to_mpz(uint32_t bits, mpz_t target);

/*
 * Convert bits to difficulty (D = 2^256 / target)
 * Useful for logging/display only
 */
double bits_to_difficulty(uint32_t bits);

#endif /* BLOCK_UTILS_H */
