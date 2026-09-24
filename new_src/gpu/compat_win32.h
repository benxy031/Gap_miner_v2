/*
 * compat_win32.h - MSVC host-side shims for the CUDA sources (gapgpu.dll).
 *
 * Included by gpu_fermat.cu / gpu_sieve.cu only when _MSC_VER is defined,
 * i.e. when nvcc uses MSVC (cl.exe) as its host compiler on Windows.
 * Linux and MinGW builds never see this file.
 *
 * Provides the small POSIX / GCC subset the .cu host code relies on:
 *   - pthread mutex + condition variable  -> SRWLOCK + CONDITION_VARIABLE
 *   - clock_gettime(CLOCK_MONOTONIC)       -> QueryPerformanceCounter
 *   - GCC __atomic_* builtins (relaxed)    -> Interlocked* intrinsics
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef GAPMINER_GPU_COMPAT_WIN32_H
#define GAPMINER_GPU_COMPAT_WIN32_H

#ifdef _MSC_VER

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdint.h>
#include <time.h>
#include <type_traits>

/* ---- pthread mutex / condition variable -------------------------------- */

typedef SRWLOCK            pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef void               pthread_mutexattr_t;
typedef void               pthread_condattr_t;

static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a)
{
    (void)a;
    InitializeSRWLock(m);
    return 0;
}
static inline int pthread_mutex_destroy(pthread_mutex_t *m)
{
    (void)m;   /* SRW locks need no cleanup */
    return 0;
}
static inline int pthread_mutex_lock(pthread_mutex_t *m)
{
    AcquireSRWLockExclusive(m);
    return 0;
}
static inline int pthread_mutex_unlock(pthread_mutex_t *m)
{
    ReleaseSRWLockExclusive(m);
    return 0;
}
static inline int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a)
{
    (void)a;
    InitializeConditionVariable(c);
    return 0;
}
static inline int pthread_cond_destroy(pthread_cond_t *c)
{
    (void)c;   /* condition variables need no cleanup */
    return 0;
}
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m)
{
    return SleepConditionVariableSRW(c, m, INFINITE, 0) ? 0 : (int)GetLastError();
}
static inline int pthread_cond_signal(pthread_cond_t *c)
{
    WakeConditionVariable(c);
    return 0;
}
static inline int pthread_cond_broadcast(pthread_cond_t *c)
{
    WakeAllConditionVariable(c);
    return 0;
}

/* ---- clock_gettime(CLOCK_MONOTONIC) ------------------------------------- */

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

static inline int clock_gettime(int clk, struct timespec *ts)
{
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER now;
    (void)clk;
    if (freq.QuadPart == 0 && !QueryPerformanceFrequency(&freq))
        return -1;
    if (!QueryPerformanceCounter(&now))
        return -1;
    ts->tv_sec  = (time_t)(now.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((now.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}

/* ---- GCC __atomic builtins used by the host code (all RELAXED) --------- */

#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#endif

template <typename P>
static inline typename std::remove_cv<P>::type gmw_atomic_load(P *p)
{
    typedef typename std::remove_cv<P>::type T;
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic size");
    if (sizeof(T) == 4)
        return (T)InterlockedCompareExchange((volatile LONG *)p, 0, 0);
    return (T)InterlockedCompareExchange64((volatile LONG64 *)p, 0, 0);
}

template <typename T, typename V>
static inline void gmw_atomic_store(T *p, V v)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic size");
    if (sizeof(T) == 4)
        InterlockedExchange((volatile LONG *)p, (LONG)(T)v);
    else
        InterlockedExchange64((volatile LONG64 *)p, (LONG64)(T)v);
}

template <typename T, typename V>
static inline T gmw_atomic_exchange(T *p, V v)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic size");
    if (sizeof(T) == 4)
        return (T)InterlockedExchange((volatile LONG *)p, (LONG)(T)v);
    return (T)InterlockedExchange64((volatile LONG64 *)p, (LONG64)(T)v);
}

template <typename T, typename V>
static inline T gmw_atomic_fetch_add(T *p, V v)
{
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomic size");
    if (sizeof(T) == 4)
        return (T)InterlockedExchangeAdd((volatile LONG *)p, (LONG)(T)v);
    return (T)InterlockedExchangeAdd64((volatile LONG64 *)p, (LONG64)(T)v);
}

#define __atomic_load_n(p, order)          gmw_atomic_load(p)
#define __atomic_store_n(p, v, order)      gmw_atomic_store((p), (v))
#define __atomic_exchange_n(p, v, order)   gmw_atomic_exchange((p), (v))
#define __atomic_fetch_add(p, v, order)    gmw_atomic_fetch_add((p), (v))

#endif /* _MSC_VER */
#endif /* GAPMINER_GPU_COMPAT_WIN32_H */
