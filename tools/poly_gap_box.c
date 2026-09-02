/* poly_gap_box.c — bounded search for runs of consecutive reducible cubics.
 *
 * Every reducible cubic A*x^3+B*x^2+C*x+D has a reduced rational root p/q
 * with q | A and, when D != 0, p | D.  We invert the equation to recover D.
 * D == 0 is handled explicitly because x is then an automatic factor.
 * Exact __int128 arithmetic; exhaustive within the box.
 *
 * 2026-09-03 external-review corrections: explicit D==0 marking (runs
 * crossing D=0 were previously split), __int128 evaluation, dynamically
 * sized bitmap, allocation/input-bounds checks, clarified n5/n6 labels.
 *
 * Build with GCC/Clang: cc -O2 -o poly_gap_box poly_gap_box.c
 * Run:   ./poly_gap_box [A_MAX] [BC_MAX] [D_MAX]
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef __int128 i128;

static void die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static void scan_abc(long long A, long long B, long long C,
                     long long dmax, unsigned char *mark, size_t mark_size,
                     long long *best_len, long long *best_abc,
                     long long *best_run, long long *n5, long long *n6)
{
    memset(mark, 0, mark_size);

    /* A zero constant always gives the factor x. */
    mark[(size_t)dmax] = 1;

    for (long long Q = 1; Q <= A; ++Q) {
        if (A % Q != 0)
            continue;

        const i128 q = (i128)Q;
        const i128 q2 = q * q;
        const i128 q3 = q2 * q;

        for (long long P = -dmax; P <= dmax; ++P) {
            if (P == 0)
                continue; /* D == 0 was handled above. */

            const i128 p = (i128)P;
            const i128 numerator =
                (i128)A * p * p * p +
                (i128)B * p * p * q +
                (i128)C * p * q2;

            if (numerator % q3 != 0)
                continue;

            const i128 value = -(numerator / q3);
            if (value < -(i128)dmax || value > (i128)dmax)
                continue;

            const long long D = (long long)value;
            mark[(size_t)(D + dmax)] = 1;
        }
    }

    long long current = 0;
    for (long long D = -dmax; D <= dmax; ++D) {
        if (mark[(size_t)(D + dmax)]) {
            ++current;
            if (current == 5)
                ++*n5;
            if (current == 6) {
                ++*n6;
                printf("RUN>=6: P(x)=%lldx^3%+lldx^2%+lldx, D=[%lld..%lld]\n",
                       A, B, C, D - 5, D);
            }
        } else {
            current = 0;
        }

        if (current > *best_len) {
            *best_len = current;
            best_abc[0] = A;
            best_abc[1] = B;
            best_abc[2] = C;
            best_run[0] = D - current + 1;
            best_run[1] = D;
        }
    }
}

int main(int argc, char **argv)
{
    const long long amax = argc > 1 ? atoll(argv[1]) : 12;
    const long long bcmax = argc > 2 ? atoll(argv[2]) : 120;
    const long long dmax = argc > 3 ? atoll(argv[3]) : 120;

    if (amax < 1 || bcmax < 0 || dmax < 0)
        die("Expected A_MAX >= 1 and nonnegative BC_MAX and D_MAX.");
    if ((unsigned long long)dmax > (SIZE_MAX - 1u) / 2u)
        die("D_MAX is too large for this address space.");

    const size_t mark_size = (size_t)dmax * 2u + 1u;
    unsigned char *mark = (unsigned char *)malloc(mark_size);
    if (!mark)
        die("Unable to allocate the D bitmap.");

    long long best_len = 0;
    long long best_abc[3] = {0, 0, 0};
    long long best_run[2] = {0, 0};
    long long n5 = 0;
    long long n6 = 0;

    for (long long A = 1; A <= amax; ++A) {
        for (long long B = -bcmax; B <= bcmax; ++B) {
            for (long long C = -bcmax; C <= bcmax; ++C) {
                scan_abc(A, B, C, dmax, mark, mark_size,
                         &best_len, best_abc, best_run, &n5, &n6);
            }
        }
    }

    printf("box: A<=%lld |B|,|C|<=%lld |D|<=%lld\n",
           amax, bcmax, dmax);
    printf("longest run: %lld P(x)=%lldx^3%+lldx^2%+lldx "
           "D=[%lld..%lld]\n",
           best_len, best_abc[0], best_abc[1], best_abc[2],
           best_run[0], best_run[1]);
    printf("maximal runs reaching >=5: %lld; reaching >=6: %lld\n", n5, n6);

    free(mark);
    return EXIT_SUCCESS;
}
