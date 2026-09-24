/*
 * Gapcoin Q48 proof-of-work arithmetic.
 *
 * This is a bit-exact C99 port of the node's own PoWUtils
 * (Gapcoin/src/PoWCore/PoWUtils.{h,cpp}), so that a miner can decide
 * "would the node accept this gap?" without guessing.
 *
 * The node's consensus rule is (PoW.h):
 *
 *     bool valid() { return difficulty() >= target_difficulty; }
 *
 * with
 *
 *     merit(start,end) = floor( gap * log2(e) * 2^112 / floor(log2(start)*2^64) )
 *                        == floor( gap * 2^48 / ln(start) )   [Q48 fixed point]
 *     rand(start,end)  = XOR of the four LE uint64 words of SHA256d(LE(start)||LE(end))
 *     m2(start)        = floor( 2 * log2(e) * 2^112 / floor(log2(start)*2^64) )
 *                        == Q48 merit of a gap of size 2 (max rand a gap can gain)
 *     difficulty       = merit + (rand mod m2)
 *
 * `target_difficulty` is the template's nDifficulty: the 8 bytes at offset
 * 72 of the 80 byte block header prefix (little endian).  Reading it from
 * the header bytes that are about to be submitted is the strongest possible
 * guarantee that we compare against exactly what the node will compare
 * against -- see pow_q48_target_from_header().
 *
 * Two changes of scale are used throughout:
 *   - "Q48"     : uint64_t, value = merit * 2^48 (the node's wire format)
 *   - "readable": double,    value = Q48 / 2^48 (what logs and --merit use)
 *
 * Verified against the node implementation itself by tests/test_pow_q48.c,
 * which checks committed vectors generated with tools/pow_q48_oracle.cpp
 * (see scripts/gen_pow_q48_vectors.sh for provenance).
 */

#ifndef GAPMINER_POW_Q48_H
#define GAPMINER_POW_Q48_H

#include <stdint.h>
#include <stdio.h>   /* must precede gmp.h: mpz_out_str() takes a FILE* */
#include <gmp.h>

/** 2^48: the node's fixed point scale. */
#define POW_Q48_ONE (((uint64_t)1) << 48)

/** Offset of nDifficulty (8 LE bytes) inside the 80 byte header prefix. */
#define POW_Q48_HEADER_OFFSET 72

/** Q48 difficulty -> double merit, exactly as PoWUtils::get_readable_difficulty. */
double pow_q48_readable(uint64_t difficulty);

/** Q48 difficulty -> double merit, without the Q48 round trip. */
double pow_q48_from_double(double merit);

/**
 * Verdict of an exact Q48 comparison.  Follows the fail-open doctrine: a
 * caller that gets POW_Q48_UNKNOWN must examine the candidate, never assume
 * it is worthless.
 */
typedef enum {
    POW_Q48_REJECT = 0,          /* exact: the node's difficulty() < target */
    POW_Q48_ACCEPT = 1,          /* exact: the node's difficulty() >= target */
    POW_Q48_UNKNOWN = 2          /* no target available -> caller must examine */
} pow_q48_verdict;

/* ---------------------------------------------------------------------------
 * Exact node quantities.  All of these reproduce PoWUtils bit for bit and are
 * endian independent (they touch no machine word order except rand(), which
 * uses the documented little endian layout of the hash words).
 * ------------------------------------------------------------------------- */

/**
 * PoWUtils::merit(): floor(gap * 2^48 / ln(start)) truncated to 64 bits.
 * Returns 0 when start <= 0 (degenerate input; the node would divide by 0).
 */
uint64_t pow_q48_merit(const mpz_t start, const mpz_t end);

/**
 * PoWUtils "min_gap_distance_merit": the Q48 merit of a gap of size 2, i.e.
 * the exclusive upper bound of the rand() refinement.  Never 0 (the node
 * initialises it to 1 and leaves that value for a zero quotient), so it is
 * always safe as a modulus.
 */
uint64_t pow_q48_min_gap_merit(const mpz_t start);

/**
 * PoWUtils::rand(): XOR of the four little endian uint64 words of
 * SHA256d(LE_bytes(start) || LE_bytes(end)) -- minimal length byte arrays,
 * exactly as mpz_export(order=-1, size=1) produces them.
 */
uint64_t pow_q48_rand(const mpz_t start, const mpz_t end);

/** PoWUtils::difficulty(): merit + (rand mod min_gap_merit). */
uint64_t pow_q48_difficulty(const mpz_t start, const mpz_t end);

/**
 * PoWUtils::target_size(): floor(target * ln(start)) as an exact integer
 * (value only; the node's mpz_import() call in that one function reads the
 * uint64 with the wrong endianness on little endian hosts, which this port
 * deliberately does not reproduce -- it is not used by the consensus check
 * `valid()`.  See docs/GAPCOIN_Q48.md).
 */
uint64_t pow_q48_target_size(const mpz_t start, uint64_t target_q48);

/**
 * The node's own verdict for this gap against this target.  POW_Q48_UNKNOWN
 * is returned only when target_q48 == 0 (no target known) or the endpoints
 * are degenerate; both mean "examine", never "reject".
 */
pow_q48_verdict pow_q48_classify(const mpz_t start, const mpz_t end,
                                 uint64_t target_q48);

/** 1 iff the node would accept this gap for this target. */
int pow_q48_accepts(const mpz_t start, const mpz_t end, uint64_t target_q48);

/**
 * Certified interval rejection.
 *
 * For a fixed start, gap sizes below the returned bound are *provably* unable
 * to reach the target, without evaluating rand() at all: an accept needs
 *
 *     difficulty = merit(g) + (rand mod m2) >= target
 *     => merit(g) >= target - m2 + 1                      (since rand mod m2 <= m2-1)
 *     => g * log2(e) * 2^112 >= (target - m2 + 1) * floor(log2(start)*2^64)
 *     => g >= ceil( (target - m2 + 1) * log2_64(start) / (log2(e)*2^112) )
 *
 * so any span strictly below that integer bound cannot be accepted.  The
 * bound is computed per start (per window), never per candidate.
 *
 * `out_bound` receives the bound (mpz).  Fail-open: if the target is 0 or the
 * start is degenerate, out_bound is set to 0, which pow_q48_span_rejected()
 * treats as "no certificate" and the caller must examine the span.
 */
void pow_q48_reject_bound(const mpz_t start, uint64_t target_q48, mpz_t out_bound);

/**
 * 1 iff a span is certified unable to reach the target of `bound` (obtained
 * from pow_q48_reject_bound()).  A bound of 0 disables the certificate.
 */
int pow_q48_span_rejected(uint64_t span, const mpz_t bound);

/**
 * Span qualification for a region that must be *examined* rather than judged
 * by its endpoints (e.g. the covered terminal pair of a HALF_CLASS chain,
 * where hidden interior primes can split the span into smaller gaps).
 *
 * Returns 1 when the span may qualify (the caller must examine it) and 0 only
 * when the span is certified unable to reach the target.  Every gap inside the
 * span is shorter than the span, so the rejection certificate applies to the
 * whole region.  Fail-open: an unknown target or a degenerate start returns 1.
 *
 * Exposed (rather than inlined in the worker) so the decision can be tested
 * against the exact verdict without a live node.
 */
int pow_q48_span_may_qualify(const mpz_t start, uint64_t span,
                             uint64_t target_q48);

/**
 * The template's nDifficulty, read from the 80 byte header prefix that is
 * about to be hashed/submitted.  Returns 0 when hdr80 is NULL.
 */
uint64_t pow_q48_target_from_header(const uint8_t hdr80[80]);

#endif /* GAPMINER_POW_Q48_H */
