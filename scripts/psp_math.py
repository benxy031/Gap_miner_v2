#!/usr/bin/env python3
"""Shared exact predicates for base-2 pseudoprime research."""

import gmpy2


MR12_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
MR12_LIMIT = 318665857834031151167461


def jacobi(a, n):
    return gmpy2.jacobi(a, n)


def euler_base2(n):
    """Euler/Jacobi test for odd n > 1."""
    residue = gmpy2.powmod(2, (n - 1) // 2, n)
    return residue == (1 if jacobi(2, n) == 1 else n - 1)


def strong_base(n, base):
    """Strong Miller-Rabin test, treating a base divisible by n as skipped."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    base %= n
    if base == 0:
        return True
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    residue = gmpy2.powmod(base, d, n)
    if residue in (1, n - 1):
        return True
    for _ in range(s - 1):
        residue = residue * residue % n
        if residue == n - 1:
            return True
    return False


def strong_base2(n):
    return strong_base(n, 2)


def mr12_witness(n):
    """Return a compositeness witness, or None if n passes all 12 bases."""
    if n < 2:
        raise ValueError("MR12 requires n >= 2")
    for base in MR12_BASES:
        if n == base:
            return None
        if n % base == 0:
            return {"kind": "factor", "value": base}
        if not strong_base(n, base):
            return {"kind": "strong_miller_rabin", "base": base}
    return None


def proven_prime_mr12(n):
    """Deterministic primality below the proven, exclusive MR12 bound."""
    if n >= MR12_LIMIT:
        raise ValueError("MR12 is not deterministic at or above its limit")
    return n >= 2 and mr12_witness(n) is None
