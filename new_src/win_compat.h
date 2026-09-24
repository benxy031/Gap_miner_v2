/*
 * win_compat.h - Windows (MinGW-w64, LLP64) compatibility layer.
 *
 * Force-included on Windows builds only (Makefile.win: -include new_src/win_compat.h).
 * Has no effect on Linux: everything is guarded by _WIN32.
 *
 * Why this exists
 * ---------------
 * Windows is LLP64: `unsigned long` is 32 bits, while Linux x86-64 (LP64)
 * makes it 64 bits.  GMP's mpz_*_ui() functions take `unsigned long`, so on
 * Windows any uint64_t value >= 2^32 passed to them is silently truncated
 * (e.g. a non-CRT nAdd at a high shift, or a far adder offset).  A truncated
 * base or offset would make the miner test and submit the WRONG number.
 *
 * The wrappers below keep the exact GMP fast path whenever the value fits in
 * an unsigned long and fall back to a temporary mpz for larger values.  The
 * call sites were changed from `(unsigned long)x` to `(uint64_t)x` so that
 * the full 64-bit value reaches the wrapper (identical on LP64 Linux).
 *
 * It also provides setenv() (absent from the MS C runtime).
 */
#ifndef GAPMINER_WIN_COMPAT_H
#define GAPMINER_WIN_COMPAT_H

#ifdef _WIN32

/* stdio.h must precede gmp.h so that gmp declares its FILE* functions
   (mpz_out_str, mpz_inp_str, ...). */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>

/* ---- GMP 64-bit-safe *_ui wrappers ------------------------------------ */

static inline void gmw_tmp_u64(mpz_ptr t, uint64_t v)
{
    __gmpz_init(t);
    __gmpz_import(t, 1, -1, sizeof(v), 0, 0, &v);
}

static inline void gmw_set_u64(mpz_ptr r, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_set_ui(r, (unsigned long)v);
    } else {
        __gmpz_import(r, 1, -1, sizeof(v), 0, 0, &v);
    }
}

static inline void gmw_init_set_u64(mpz_ptr r, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_init_set_ui(r, (unsigned long)v);
    } else {
        __gmpz_init(r);
        __gmpz_import(r, 1, -1, sizeof(v), 0, 0, &v);
    }
}

static inline void gmw_add_u64(mpz_ptr r, mpz_srcptr a, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_add_ui(r, a, (unsigned long)v);
    } else {
        mpz_t t;
        gmw_tmp_u64(t, v);
        __gmpz_add(r, a, t);
        __gmpz_clear(t);
    }
}

static inline void gmw_sub_u64(mpz_ptr r, mpz_srcptr a, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_sub_ui(r, a, (unsigned long)v);
    } else {
        mpz_t t;
        gmw_tmp_u64(t, v);
        __gmpz_sub(r, a, t);
        __gmpz_clear(t);
    }
}

static inline void gmw_mul_u64(mpz_ptr r, mpz_srcptr a, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_mul_ui(r, a, (unsigned long)v);
    } else {
        mpz_t t;
        gmw_tmp_u64(t, v);
        __gmpz_mul(r, a, t);
        __gmpz_clear(t);
    }
}

static inline void gmw_addmul_u64(mpz_ptr r, mpz_srcptr a, uint64_t v)
{
    if (v <= ULONG_MAX) {
        __gmpz_addmul_ui(r, a, (unsigned long)v);
    } else {
        mpz_t t;
        gmw_tmp_u64(t, v);
        __gmpz_addmul(r, a, t);
        __gmpz_clear(t);
    }
}

#undef mpz_set_ui
#undef mpz_init_set_ui
#undef mpz_add_ui
#undef mpz_sub_ui
#undef mpz_mul_ui
#undef mpz_addmul_ui
#define mpz_set_ui(r, v)         gmw_set_u64((r), (uint64_t)(v))
#define mpz_init_set_ui(r, v)    gmw_init_set_u64((r), (uint64_t)(v))
#define mpz_add_ui(r, a, v)      gmw_add_u64((r), (a), (uint64_t)(v))
#define mpz_sub_ui(r, a, v)      gmw_sub_u64((r), (a), (uint64_t)(v))
#define mpz_mul_ui(r, a, v)      gmw_mul_u64((r), (a), (uint64_t)(v))
#define mpz_addmul_ui(r, a, v)   gmw_addmul_u64((r), (a), (uint64_t)(v))

/* ---- POSIX bits missing from the MS C runtime -------------------------- */

static inline int gmw_setenv(const char *name, const char *value, int overwrite)
{
    if (!overwrite && getenv(name) != NULL)
        return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}
#define setenv(n, v, o) gmw_setenv((n), (v), (o))

/* _putenv_s with an empty value removes the variable from the environment. */
static inline int gmw_unsetenv(const char *name)
{
    return _putenv_s(name, "") == 0 ? 0 : -1;
}
#define unsetenv(n) gmw_unsetenv((n))

#endif /* _WIN32 */
#endif /* GAPMINER_WIN_COMPAT_H */
