/* DEACTIVATED: unused module - removed from the build (not in Makefile SOURCES).
 * Kept for reference. To restore: re-add to Makefile SOURCES and delete this block.
 */
/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Block Template Utilities — Implementation
 * 
 * ⚠️  GAPCOIN FORMAT (NOT BITCOIN):
 *   bits = "1556478acda3b7" (7 bytes, 14 hex chars)
 *   Bytes 0-1: shift (16-bit little-endian value)
 *   Bytes 2-6: target/difficulty data
 */

#include "block_utils.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

uint32_t parse_bits(const char *bits_hex) {
    /* 
     * Gapcoin bits format (7 bytes):
     *   Input: "1556478acda3b7"
     *   Output: 0x1556 (shift, as uint32_t)
     */
    if (!bits_hex || strlen(bits_hex) != 14) {
        /* Invalid Gapcoin bits format (should be 14 hex chars) */
        return 0;
    }
    
    uint16_t shift_le = 0;
    sscanf(bits_hex, "%4hx", &shift_le);
    
    /* bits_hex is in little-endian format, convert to big-endian uint32_t */
    return (uint32_t)shift_le;
}

void decode_bits(uint32_t bits, uint32_t *out_exponent, uint32_t *out_mantissa) {
    /*
     * Gapcoin format: bits = shift value (already extracted by parse_bits)
     * For compatibility, we don't use exponent/mantissa breakdown
     * Just return the shift value
     */
    if (!out_exponent || !out_mantissa) {
        return;
    }
    
    *out_exponent = 0;      /* Not used in Gapcoin */
    *out_mantissa = bits;   /* This is already the shift value */
}

uint32_t bits_to_shift(uint32_t bits) {
    /*
     * Gapcoin: bits parameter IS the shift value already!
     * No conversion needed.
     *
     * bits = shift (as extracted by parse_bits from "1556478acda3b7")
     */
    return bits;
}

void bits_to_mpz(uint32_t bits, mpz_t target) {
    /*
     * Gapcoin: Cannot compute exact target from shift alone
     * Shift is log2(target), so target = 2^shift
     *
     * This is approximate; real Gapcoin uses additional encoded data
     * For now, we just set target = 2^shift
     */
    if (bits == 0) {
        mpz_set_ui(target, 0);
        return;
    }
    
    /* target ≈ 2^shift */
    mpz_set_ui(target, 1);
    mpz_mul_2exp(target, target, bits);
}

double bits_to_difficulty(uint32_t bits) {
    /*
     * Difficulty ≈ max_target / target
     * For Gapcoin, this depends on network parameters
     */
    
    if (bits == 0) {
        return 0.0;
    }
    
    /* Approximate: higher shift = lower difficulty */
    return pow(2.0, -bits);
}
