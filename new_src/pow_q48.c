/*
 * Gapcoin Q48 proof-of-work arithmetic -- bit-exact port of the node's
 * PoWUtils (Gapcoin/src/PoWCore/PoWUtils.{h,cpp}).
 *
 * Every numeric detail below is copied from that source on purpose: the
 * point of this module is that pow_q48_accepts() answers exactly the
 * question the node's PoW::valid() answers, so a miner can stop submitting
 * gaps the node will reject with "high-hash".
 *
 * Provenance / verification:
 *   tools/pow_q48_oracle.cpp links the *node's* PoWUtils.cpp and prints its
 *   merit/rand/difficulty/target_size for a list of (start, gap) pairs;
 *   scripts/gen_pow_q48_vectors.sh builds it and writes
 *   tests/data/pow_q48_vectors.txt, which tests/test_pow_q48.c compares
 *   against this implementation.
 */

#include "pow_q48.h"

#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>

/* log2(e) * 2^112 -- the constant PoWUtils::PoWUtils() loads. */
static const char *Q48_LOG2E112_HEX = "171547652b82fe1777d0ffda0d23a";

/* ---------------------------------------------------------------------------
 * Internal helpers
 * ------------------------------------------------------------------------- */

/*
 * PoWUtils::mpz_log2(), ported statement by statement.
 *
 * out = floor(log2(src) * 2^accuracy), built bit by bit by the
 * square-and-shift method.  The `mpz_get_ui(mpz_tmp) < 2` test is the node's
 * own condition and is deliberately NOT replaced: on Windows (LLP64)
 * mpz_get_ui() returns 32 bits while on Linux it returns 64, but the branch
 * only inspects a value the loop has just renormalised down to a single digit,
 * so both platforms take the same path -- and staying identical to the node is
 * the entire point of this module.  (Everything else in this file avoids
 * mpz_get_ui() and unsigned long, because there the truncation WOULD bite:
 * Q48 values exceed 2^32 by construction.)
 *
 * src must be > 0; the caller checks that.
 */
static void q48_mpz_log2(mpz_t out, const mpz_t src, uint32_t accuracy) {
    mpz_t tmp, n;
    uint32_t bits = 0;
    uint32_t shift;

    mpz_init(tmp);
    mpz_init_set(n, src);

    /* integer part of log2 (a bit count, so it always fits an unsigned long) */
    mpz_set_ui(out, (uint64_t)(mpz_sizeinbase(n, 2) - 1));

    shift = accuracy + (uint32_t)mpz_get_ui(out);

    /* add accuracy fractional bits */
    mpz_mul_2exp(out, out, accuracy);
    mpz_mul_2exp(n, n, accuracy);

    for (;;) {
        mpz_div_2exp(tmp, n, shift);

        while (mpz_get_ui(tmp) < 2 && bits <= accuracy) {
            mpz_mul(n, n, n);            /* n <- n^2 */
            mpz_div_2exp(n, n, shift);   /* preserve accuracy */
            mpz_div_2exp(tmp, n, shift);
            bits++;
        }

        if (bits > accuracy) break;

        mpz_set_ui(tmp, 1);
        mpz_mul_2exp(tmp, tmp, accuracy - bits);
        mpz_add(out, out, tmp);          /* log += 2^(accuracy - bits) */

        mpz_div_2exp(n, n, 1);           /* n <- n/2 */
    }

    mpz_clear(tmp);
    mpz_clear(n);
}

/*
 * PoWUtils takes the low 64 bits of an mpz by exporting it little endian and
 * overwriting the first min(len,8) bytes of a pre-initialised uint64.  This
 * helper does exactly that, byte for byte, using the same mpz_fdiv_r_2exp +
 * mpz_export path -- no mpz_get_ui(), because on Windows (LLP64) that returns
 * only 32 bits and would silently truncate every Q48 value (win_compat.h
 * documents the same hazard for the *_ui setters).
 *
 * A zero value (zero export length) keeps the initialiser, which the node
 * relies on for min_gap_distance_merit (initialised to 1, never 0, because it
 * is used as a modulus).
 */
static uint64_t q48_low64(const mpz_t value, uint64_t init) {
    uint8_t buf[8];
    uint8_t tmp[8];
    uint64_t result = 0;
    size_t count = 0;
    mpz_t low;
    int i;

    for (i = 0; i < 8; i++) buf[i] = (uint8_t)(init >> (8 * i));
    if (mpz_sgn(value) <= 0) goto done;

    mpz_init(low);
    mpz_fdiv_r_2exp(low, value, 64);   /* value mod 2^64: export stays <= 8 bytes */
    mpz_export(tmp, &count, -1, 1, 0, 0, low);
    mpz_clear(low);

    for (i = 0; i < (int)count && i < 8; i++) buf[i] = tmp[i];

done:
    for (i = 0; i < 8; i++) result |= ((uint64_t)buf[i]) << (8 * i);
    return result;
}

/* Set an mpz from a uint64 without relying on unsigned long being 64 bit
   (Windows is LLP64: unsigned long is 32 bits there). */
static void q48_set_u64(mpz_t dst, uint64_t value) {
    mpz_import(dst, 1, -1, sizeof(uint64_t), 0, 0, &value);
}

/* Minimal length little endian byte array of a positive mpz.  Caller frees. */
static uint8_t *q48_le_bytes(const mpz_t value, size_t *len_out) {
    size_t len = (mpz_sizeinbase(value, 2) + 7) / 8;
    uint8_t *buf = (uint8_t *)malloc(len ? len : 1);
    size_t written = 0;

    if (!buf) {
        *len_out = 0;
        return NULL;
    }
    if (len > 0) {
        mpz_export(buf, &written, -1, 1, 0, 0, value);
    }
    *len_out = written;
    return buf;
}

/* ---------------------------------------------------------------------------
 * Exact node quantities
 * ------------------------------------------------------------------------- */

uint64_t pow_q48_merit(const mpz_t start, const mpz_t end) {
    mpz_t gap, log2e112, log2_start, quotient;
    uint64_t result = 0;

    if (mpz_sgn(start) <= 0 || mpz_sgn(end) < 0 || mpz_cmp(end, start) < 0) {
        return 0;
    }

    mpz_inits(gap, log2e112, log2_start, quotient, NULL);
    mpz_set_str(log2e112, Q48_LOG2E112_HEX, 16);

    mpz_sub(gap, end, start);
    mpz_mul(gap, gap, log2e112);         /* gap * log2(e) * 2^112 */

    q48_mpz_log2(log2_start, start, 64); /* log2(start) * 2^64 */

    if (mpz_sgn(log2_start) > 0) {
        mpz_fdiv_q(quotient, gap, log2_start);
        result = q48_low64(quotient, 0);
    }

    mpz_clears(gap, log2e112, log2_start, quotient, NULL);
    return result;
}

uint64_t pow_q48_min_gap_merit(const mpz_t start) {
    mpz_t tmp, log2e112, log2_start, quotient;
    uint64_t result = 1;                 /* the node's initialiser */

    if (mpz_sgn(start) <= 0) return 1;

    mpz_inits(tmp, log2e112, log2_start, quotient, NULL);
    mpz_set_str(log2e112, Q48_LOG2E112_HEX, 16);

    mpz_set_ui(tmp, 2);
    mpz_mul(tmp, tmp, log2e112);         /* 2 * log2(e) * 2^112 */

    q48_mpz_log2(log2_start, start, 64);

    if (mpz_sgn(log2_start) > 0) {
        mpz_fdiv_q(quotient, tmp, log2_start);
        result = q48_low64(quotient, 1);
    }

    mpz_clears(tmp, log2e112, log2_start, quotient, NULL);
    return result;
}

uint64_t pow_q48_rand(const mpz_t start, const mpz_t end) {
    uint8_t *bs = NULL, *be = NULL, *joined = NULL;
    size_t ls = 0, le = 0;
    unsigned char first[SHA256_DIGEST_LENGTH], second[SHA256_DIGEST_LENGTH];
    uint64_t words[4], result;
    int i;

    if (!start || !end) return 0;
    if (mpz_sgn(start) <= 0 || mpz_sgn(end) <= 0) return 0;

    bs = q48_le_bytes(start, &ls);
    be = q48_le_bytes(end, &le);
    if (!bs || !be || (ls + le) == 0) {
        free(bs);
        free(be);
        return 0;
    }

    joined = (uint8_t *)malloc(ls + le);
    if (!joined) {
        free(bs);
        free(be);
        return 0;
    }
    memcpy(joined, bs, ls);
    memcpy(joined + ls, be, le);

    /* sha256d(LE(start) || LE(end)) */
    SHA256(joined, ls + le, first);
    SHA256(first, SHA256_DIGEST_LENGTH, second);

    free(bs);
    free(be);
    free(joined);

    /* XOR the four little endian 64 bit words of the hash */
    for (i = 0; i < 4; i++) {
        words[i] = 0;
        for (size_t b = 0; b < 8; b++) {
            words[i] |= ((uint64_t)second[i * 8 + b]) << (8 * b);
        }
    }

    result = words[0] ^ words[1] ^ words[2] ^ words[3];
    return result;
}

uint64_t pow_q48_difficulty(const mpz_t start, const mpz_t end) {
    uint64_t merit, min_gap, rnd;

    if (mpz_sgn(start) <= 0 || mpz_cmp(end, start) < 0) return 0;

    merit = pow_q48_merit(start, end);
    min_gap = pow_q48_min_gap_merit(start);
    if (min_gap == 0) min_gap = 1;       /* modulus safety; the node guarantees >= 1 */

    rnd = pow_q48_rand(start, end);

    /* difficulty = gap_size / log(start) + rand(start, end) % merit_of_distance_to_next_gap */
    return merit + (rnd % min_gap);
}

double pow_q48_readable(uint64_t difficulty) {
    return ((double)difficulty) / ((double)POW_Q48_ONE);
}

double pow_q48_from_double(double merit) {
    if (merit <= 0.0) return 0.0;
    return merit * ((double)POW_Q48_ONE);
}

uint64_t pow_q48_target_size(const mpz_t start, uint64_t target_q48) {
    mpz_t log2e112, target, log2_start, product, quotient;
    uint64_t result = 0;

    if (mpz_sgn(start) <= 0 || target_q48 == 0) return 0;

    mpz_inits(log2e112, target, log2_start, product, quotient, NULL);
    mpz_set_str(log2e112, Q48_LOG2E112_HEX, 16);
    q48_set_u64(target, target_q48);

    q48_mpz_log2(log2_start, start, 64);

    if (mpz_sgn(log2_start) > 0) {
        mpz_mul(product, log2_start, target);
        mpz_fdiv_q(quotient, product, log2e112);
        result = q48_low64(quotient, 0);
    }

    mpz_clears(log2e112, target, log2_start, product, quotient, NULL);
    return result;
}

/* ---------------------------------------------------------------------------
 * Verdicts and the rejection certificate
 * ------------------------------------------------------------------------- */

pow_q48_verdict pow_q48_classify(const mpz_t start, const mpz_t end,
                                 uint64_t target_q48) {
    if (!start || !end || target_q48 == 0) return POW_Q48_UNKNOWN;
    if (mpz_sgn(start) <= 0 || mpz_cmp(end, start) < 0) return POW_Q48_UNKNOWN;

    return (pow_q48_difficulty(start, end) >= target_q48) ? POW_Q48_ACCEPT
                                                         : POW_Q48_REJECT;
}

int pow_q48_accepts(const mpz_t start, const mpz_t end, uint64_t target_q48) {
    return pow_q48_classify(start, end, target_q48) == POW_Q48_ACCEPT;
}

void pow_q48_reject_bound(const mpz_t start, uint64_t target_q48,
                          mpz_t out_bound) {
    mpz_t log2e112, log2_start, needed, product;
    uint64_t min_gap;

    mpz_set_ui(out_bound, 0);            /* fail-open: no certificate */
    if (!start || target_q48 == 0 || mpz_sgn(start) <= 0) return;

    min_gap = pow_q48_min_gap_merit(start);

    mpz_inits(log2e112, log2_start, needed, product, NULL);
    mpz_set_str(log2e112, Q48_LOG2E112_HEX, 16);

    if (target_q48 >= min_gap) {
        /* accept requires merit >= target - (min_gap - 1) */
        q48_set_u64(needed, target_q48 - min_gap + 1);
    } else {
        mpz_set_ui(needed, 0);
    }

    q48_mpz_log2(log2_start, start, 64);

    if (mpz_sgn(log2_start) > 0) {
        mpz_mul(product, log2_start, needed);
        mpz_cdiv_q(out_bound, product, log2e112);  /* ceil */
        if (mpz_cmp_ui(out_bound, 2) < 0) mpz_set_ui(out_bound, 2);
    }

    mpz_clears(log2e112, log2_start, needed, product, NULL);
}

int pow_q48_span_rejected(uint64_t span, const mpz_t bound) {
    mpz_t limit;
    int rejected;

    if (!bound || mpz_sgn(bound) <= 0) return 0;   /* no certificate */
    if (span == UINT64_MAX) return 0;              /* degenerate span */

    /* Compare as mpz, not mpz_cmp_ui(): span is a uint64 and the *_ui family
       takes unsigned long, i.e. 32 bits on Windows (LLP64). */
    mpz_init(limit);
    mpz_set_ui(limit, span);
    rejected = (mpz_cmp(bound, limit) > 0) ? 1 : 0;
    mpz_clear(limit);
    return rejected;
}

int pow_q48_span_may_qualify(const mpz_t start, uint64_t span,
                             uint64_t target_q48) {
    mpz_t bound;
    int may;

    /* Fail-open: without a target (or with a degenerate start) the caller's
       own approximate test has to decide. */
    if (!start || target_q48 == 0 || mpz_sgn(start) <= 0) return 1;
    if (span == UINT64_MAX) return 1;

    mpz_init(bound);
    pow_q48_reject_bound(start, target_q48, bound);
    may = !pow_q48_span_rejected(span, bound);
    mpz_clear(bound);
    return may;
}

uint64_t pow_q48_target_from_header(const uint8_t hdr80[80]) {
    uint64_t value = 0;
    int i;

    if (!hdr80) return 0;
    for (i = 7; i >= 0; i--) {
        value = (value << 8) | (uint64_t)hdr80[POW_Q48_HEADER_OFFSET + i];
    }
    return value;
}
