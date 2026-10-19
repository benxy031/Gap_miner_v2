#ifndef GV_PARSE_H
#define GV_PARSE_H

#include <stdint.h>

#include <gmp.h>

/*
 * Parse a prime-gap start expression into `out`.
 * Supported forms (whitespace-free, a >= 0):
 *   digits
 *   m*p#/d +- a            e.g. 86968566716*79#/29 - 123
 *   p#/d +- a
 *   m*p#/(d#*m2) +- a
 *   m*p#/(d*m2) +- a
 *   m*p#/d# +- a
 *   m*p# +- a              (no divisor; silently skipped by check.py)
 *   p# +- a
 *   b^p +- a               e.g. 10^19662 - 58727
 * Returns 0 on success, -1 on malformed input, GV_PARSE_TOO_LARGE when
 * the value cannot be materialized in memory (estimated decimal digits
 * beyond GV_MAX_MATERIALIZE_DIGITS).
 */
int gv_parse_expr(const char *s, mpz_t out, int quiet);

/*
 * Extract (gap, digits, start-expr) from one `INSERT INTO gaps VALUES ...`
 * line of allgaps.sql. Quote-aware field splitting, so rows with commas
 * or spaces inside quoted fields (which break check.py's regex) parse.
 * Returns 0 on success (expr returned as malloc'ed string), -1 if the
 * line is not a row or is malformed. `line_no` is used for messages only.
 */
int gv_db_extract_row(const char *line, unsigned long line_no,
                      uint64_t *gap_out, int *digits_out, char **expr_out);

#endif /* GV_PARSE_H */
