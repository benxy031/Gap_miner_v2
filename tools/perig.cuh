/* Perig/ciosFermatTest128 port from prime_gaps.cu (PGS), used under its license. */
#ifndef PERIG_CUH
#define PERIG_CUH

/* Host-compilable: outside nvcc the CUDA qualifiers mean nothing, so the host
   harness tools/perig_range.cpp builds with a plain
     g++ -O2 -I tools tools/perig_range.cpp -lgmp                          */
#if !defined(__CUDACC__)
#  ifndef __device__
#    define __device__
#  endif
#  ifndef __host__
#    define __host__
#  endif
#  ifndef __forceinline__
#    define __forceinline__ inline
#  endif
#endif
#include <stdint.h>
/* 128-bit integer.  Native __int128 everywhere except MSVC (the Windows host
   compiler nvcc requires), which has no 128-bit type at all.  The software
   substitute below implements exactly the operations this file needs
   (construct, +, +=, unary -, *, >>, <<, cast to uint64_t and one modulo) and
   compiles for device too.  Define P0_PERIG_U128_SHIM to force it on
   GCC/nvcc: that is how it is validated (build phase0gpu_dll.cu with the
   shim, link the host with -DPHASE0_KERNEL_DLL and diff outputs against the
   native build - see windows/build_phase0.bat and README_PHASE0.md). */
#if !defined(_MSC_VER) && !defined(P0_PERIG_U128_SHIM)
typedef unsigned __int128 uint128_t;
static inline __host__ __device__ uint128_t perig_u128_mod(uint128_t a, uint128_t b) {
    return b ? (a % b) : a;
}
#else
/* 64 x 64 -> high 64 (host- and device-safe; __umul64hi is device-only). */
static inline __host__ __device__ uint64_t perig_mulhi64(uint64_t a, uint64_t b) {
    uint64_t a0 = a & 0xffffffffULL, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffULL, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}
struct uint128_t {
    uint64_t lo;
    uint64_t hi;
    __host__ __device__ uint128_t() : lo(0), hi(0) {}
    __host__ __device__ uint128_t(uint64_t v) : lo(v), hi(0) {}   /* implicit, like __int128 */
    __host__ __device__ explicit operator uint64_t() const { return lo; }
};
static inline __host__ __device__ uint128_t operator+(uint128_t a, uint128_t b) {
    uint128_t r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo ? 1ULL : 0ULL);
    return r;
}
static inline __host__ __device__ uint128_t &operator+=(uint128_t &a, uint128_t b) {
    a = a + b;
    return a;
}
static inline __host__ __device__ uint128_t operator-(uint128_t a) {   /* unary minus */
    uint128_t r;
    r.lo = ~a.lo + 1ULL;
    r.hi = ~a.hi + (r.lo == 0ULL ? 1ULL : 0ULL);
    return r;
}
static inline __host__ __device__ uint128_t operator*(uint128_t a, uint128_t b) {
    uint128_t r;
    uint64_t p0 = a.lo * b.lo;
    uint64_t p1 = perig_mulhi64(a.lo, b.lo);
    uint64_t p2 = a.hi * b.lo;
    uint64_t p3 = a.lo * b.hi;
    r.lo = p0;
    r.hi = p1 + p2 + p3;   /* valid while a.hi*b.hi == 0 (all call sites) */
    return r;
}
static inline __host__ __device__ uint128_t operator>>(uint128_t a, int sh) {
    uint128_t r;
    if (sh <= 0) return a;
    if (sh >= 128) return r;
    if (sh >= 64) {
        r.lo = a.hi >> (sh - 64);
        r.hi = 0;
    } else {
        r.lo = (a.lo >> sh) | (sh ? (a.hi << (64 - sh)) : 0ULL);
        r.hi = a.hi >> sh;
    }
    return r;
}
static inline __host__ __device__ uint128_t operator<<(uint128_t a, int sh) {
    uint128_t r;
    if (sh <= 0) return a;
    if (sh >= 128) return r;
    if (sh >= 64) {
        r.hi = a.lo << (sh - 64);
        r.lo = 0;
    } else {
        r.hi = (a.hi << sh) | (sh ? (a.lo >> (64 - sh)) : 0ULL);
        r.lo = a.lo << sh;
    }
    return r;
}
/* 128-bit modulo by restoring division (shift/subtract); only ever used for
   the once-per-modulus magic constant, so the 128-step loop is cheap. */
static inline __host__ __device__ uint128_t perig_u128_mod(uint128_t a, uint128_t b) {
    uint128_t r;
    if (b.lo == 0 && b.hi == 0) return a;
    for (int i = 127; i >= 0; i--) {
        uint64_t bit = (i >= 64) ? ((a.hi >> (i - 64)) & 1ULL)
                                 : ((a.lo >> i) & 1ULL);
        r.hi = (r.hi << 1) | (r.lo >> 63);
        r.lo = (r.lo << 1) | bit;
        int ge;
        if (r.hi != b.hi) ge = (r.hi > b.hi);
        else ge = (r.lo >= b.lo);
        if (ge) {
            uint64_t t = r.lo - b.lo;
            uint64_t borrow = (r.lo < b.lo) ? 1ULL : 0ULL;
            r.lo = t;
            r.hi = r.hi - b.hi - borrow;
        }
    }
    return r;
}
#endif
#ifndef PARANOID
#define PARANOID 0
#endif
#ifndef HIGH_64
#define HIGH_64 10
#endif

// (a::b) <<= c
__host__ __device__ static inline __attribute__((always_inline))
void my_shld64(uint64_t * a, uint64_t * b, uint64_t c)
{
#if PARANOID
	assert(c < 64);
#endif
	if (c == 0) {
	} else {
		(*a) = ((*a) << c) | ((*b) >> (64 - c));
		(*b) <<= c;
	}
}

// borrow::diff = a - b - borrow_in
static inline __attribute__((always_inline))
__host__ __device__ uint8_t my_sbb64(uint8_t borrow_in, uint64_t a, uint64_t b, uint64_t * diff)
{
    uint64_t tmp1 = a - borrow_in;
    uint8_t borrow = (tmp1 > a);

    uint64_t tmp2 = tmp1 - b;
    borrow |= (tmp2 > tmp1);
    *diff = tmp2;
    return borrow;
}

// count leading zeroes in binary representation
static inline __attribute__((always_inline))
__host__ __device__ uint64_t my_clz64(uint64_t n)
{
#ifdef __CUDA_ARCH__
    return __clzll(n);
#else
	if (n == 0)
		return 64;
	uint64_t r = 0;
	if ((n & (0xFFFFFFFFull << 32)) == 0)
		r += 32, n <<= 32;
	if ((n & (0xFFFFull << 48)) == 0)
		r += 16, n <<= 16;
	if ((n & (0xFFull << 56)) == 0)
		r += 8, n <<= 8;
	if ((n & (0xFull << 60)) == 0)
		r += 4, n <<= 4;
	if ((n & (0x3ull << 62)) == 0)
		r += 2, n <<= 2;
	if ((n & (0x1ull << 63)) == 0)
		r += 1;
	return r;
#endif
}

static inline __attribute__((always_inline))
__host__ __device__ uint64_t montgomeryInverse64(uint64_t mod_lo)
{
	uint64_t x = (3ull * mod_lo) ^ 2ull;	// 5 bits acurate
	uint64_t t = 1ull - mod_lo * x;
	x *= 1 + t;		// 10 bits accurate
	t *= t;
	x *= 1 + t;		// 20 bits accurate
	t *= t;
	x *= 1 + t;		// 40 bits accurate
	t *= t;
	x *= 1 + t;		// 80 bits accurate , i.e. > 64 bits
	return 0 - x;
}

// subtract the modulus 'mod' multiple times from the input number 'res', if needed
static inline __attribute__((always_inline))
__host__ __device__ void ciosSubtract128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo, uint64_t mod_hi)
{
	uint64_t n_lo, n_hi;
	uint64_t t_lo, t_hi;
	uint8_t b;
	n_lo = *res_lo;
	n_hi = *res_hi;
	// save, subtract the modulus until a borrows occurs
	do {
		t_lo = n_lo;
		t_hi = n_hi;
		b = my_sbb64(0, n_lo, mod_lo, &n_lo);
		b = my_sbb64(b, n_hi, mod_hi, &n_hi);
	}
	while (b == 0);
	// get the saved values
	*res_lo = t_lo;
	*res_hi = t_hi;
}

static inline __attribute__((always_inline))
__host__ __device__ void ciosConstants128(uint64_t mod_lo, uint64_t mod_hi, uint64_t * magic_lo, uint64_t * magic_hi)
{
	// computes 2^128 % mod
	uint128_t m = ((uint128_t) mod_hi << 64) + mod_lo;
	uint128_t t = -m;	// 2^128-m
	t = perig_u128_mod(t, m);	// (2^128-m) % m  (portable: see shim above)
	*magic_lo = (uint64_t) t;
	*magic_hi = (uint64_t) (t >> 64);
#if PARANOID
	assert(*magic_hi <= mod_hi);
#endif
}

static inline __attribute__((always_inline))
__host__ __device__ void ciosModSquare128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo, uint64_t mod_hi, uint64_t mmagic)
{
	uint64_t n_lo = *res_lo, n_hi = *res_hi;
	uint128_t cs, cc;
	uint64_t t0, t1, t2, m;

	cc = (uint128_t) n_lo *n_lo;	// #1
	t0 = (uint64_t) cc;
	cc = cc >> 64;
	cc += (uint128_t) n_lo *(n_hi + n_hi);	// #2
	t1 = (uint64_t) cc;
	cc = cc >> 64;
	t2 = (uint64_t) cc;
#if PARANOID
	assert(cc >> 64 == 0);
#endif

	m = t0 * mmagic;	// #3
	cs = (uint128_t) m *mod_lo;	// #4
	cs += t0;
	cs = cs >> 64;

    cs += (uint128_t) m *mod_hi; // CAN OPTIMIZE
	cs += t1;
    
	t0 = (uint64_t) cs;
	cs = cs >> 64;
	cs += t2;
	t1 = (uint64_t) cs;
	cs = cs >> 64;
	t2 = (uint64_t) cs;
#if PARANOID
	assert(cs >> 64 == 0);
#endif

	cc = (uint128_t) n_hi *n_hi;	// #6
	cc += t1;
	t1 = (uint64_t) cc;
	cc = cc >> 64;
	cc += t2;
	t2 = (uint64_t) cc;
#if 0
	// not necessary with 2-bits guard
	cc = cc >> 64;
	uint64_t t3 = (uint64_t) cc;
	assert(t3 == 0);
#endif
#if PARANOID
	assert(cc >> 64 == 0);
#endif

	m = t0 * mmagic;	// #3
	cs = (uint128_t) m *mod_lo;	// #8
	cs += t0;
	cs = cs >> 64;
    
    cs += (uint128_t) m *mod_hi; // CAN OPTIMIZE
    

	cs += t1;
	t0 = (uint64_t) cs;
	cs = cs >> 64;
	cs += t2;
	t1 = (uint64_t) cs;
#if 0
	// not necessary with 2-bits guard
	cs = cs >> 64;
	cs += t3;
	t2 = (uint64_t) cs;
	assert(t2 == 0);
#endif
#if PARANOID
	assert(cs >> 64 == 0);
#endif

	*res_lo = t0;
	*res_hi = t1;

}

__host__ __device__ inline void ciosModSquare3_128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo,
                                                   uint64_t mod_hi, uint64_t mmagic)
{
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
}

#ifdef HIGH_64
__host__ __device__ bool ciosFermatTest128(uint64_t n_lo) {
#else
__host__ __device__ bool ciosFermatTest128(uint64_t n_lo, uint64_t HIGH_64) {
#endif
#if PARANOID
	assert((n_lo & 1) == 1);
#endif
    //const uint64_t n_hi = <stuff here>;

	uint64_t res_lo, res_hi;
	uint64_t one_lo, one_hi;
	int bit;
	// constant -1/m mod 2^64
	uint64_t mmagic = montgomeryInverse64(n_lo);

	// enter montgomery domain
	// constant 2^128 mod m
	ciosConstants128(n_lo, HIGH_64, &one_lo, &one_hi);
	res_hi = one_hi;
	res_lo = one_lo;

    if (HIGH_64 == 0) {
        bit = 64 - my_clz64(n_lo);
        uint64_t msb_bits = bit < 5 ? bit - 1 : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (n_lo >> bit) & msb_mask);

    } else {
        bit = 64 - my_clz64(HIGH_64);
        uint64_t msb_bits = bit < 4 ? bit : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (HIGH_64 >> bit) & msb_mask);

        while (bit >= 3) {
            bit -= 3;
            // square and reduce
            ciosModSquare3_128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((HIGH_64 >> bit) & 7));
        }

        while (bit) {
            bit -= 1;
            // square and reduce
            ciosModSquare128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((HIGH_64 >> bit) & 1));
        }

        bit = 64;
    }
	//}
	while (bit >= 5) {
		bit -= 3;
		// square and reduce
		ciosModSquare3_128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 7));
	}

	while (bit > 1) {
		bit -= 1;
		// square and reduce
		ciosModSquare128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 1));
	}

	// make sure result is strictly less than the modulus
	ciosSubtract128(&res_lo, &res_hi, n_lo, HIGH_64);

	uint64_t legendre = ((n_lo >> 1) ^ (n_lo >> 2)) & 1;	// shortcut calculation of legendre symbol

	uint64_t m1_lo;
	uint64_t m1_hi;
	uint8_t c;
	c = my_sbb64(0, n_lo, one_lo, &m1_lo);
	my_sbb64(c, HIGH_64, one_hi, &m1_hi);

	return ((res_lo == (legendre ? m1_lo : one_lo)) && (res_hi == (legendre ? m1_hi : one_hi)));
}
/* runtime high-word variant of the same test (the phase0 scanner
   --engine walk path passes the scan range's high word) */
__host__ __device__ bool ciosFermatTest128_hi(uint64_t n_lo, uint64_t n_hi)
{
#if PARANOID
	assert((n_lo & 1) == 1);
#endif
    //const uint64_t n_hi = <stuff here>;

	uint64_t res_lo, res_hi;
	uint64_t one_lo, one_hi;
	int bit;
	// constant -1/m mod 2^64
	uint64_t mmagic = montgomeryInverse64(n_lo);

	// enter montgomery domain
	// constant 2^128 mod m
	ciosConstants128(n_lo, n_hi, &one_lo, &one_hi);
	res_hi = one_hi;
	res_lo = one_lo;

    if (n_hi == 0) {
        bit = 64 - my_clz64(n_lo);
        uint64_t msb_bits = bit < 5 ? bit - 1 : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (n_lo >> bit) & msb_mask);

    } else {
        bit = 64 - my_clz64(n_hi);
        uint64_t msb_bits = bit < 4 ? bit : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (n_hi >> bit) & msb_mask);

        while (bit >= 3) {
            bit -= 3;
            // square and reduce
            ciosModSquare3_128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((n_hi >> bit) & 7));
        }

        while (bit) {
            bit -= 1;
            // square and reduce
            ciosModSquare128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((n_hi >> bit) & 1));
        }

        bit = 64;
    }
	//}
	while (bit >= 5) {
		bit -= 3;
		// square and reduce
		ciosModSquare3_128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 7));
	}

	while (bit > 1) {
		bit -= 1;
		// square and reduce
		ciosModSquare128(&res_lo, &res_hi, n_lo, n_hi, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 1));
	}

	// make sure result is strictly less than the modulus
	ciosSubtract128(&res_lo, &res_hi, n_lo, n_hi);

	uint64_t legendre = ((n_lo >> 1) ^ (n_lo >> 2)) & 1;	// shortcut calculation of legendre symbol

	uint64_t m1_lo;
	uint64_t m1_hi;
	uint8_t c;
	c = my_sbb64(0, n_lo, one_lo, &m1_lo);
	my_sbb64(c, n_hi, one_hi, &m1_hi);

	return ((res_lo == (legendre ? m1_lo : one_lo)) && (res_hi == (legendre ? m1_hi : one_hi)));
}
#endif
