/* poly_gap_box.c — box search for runs of consecutive reducible cubics.
 *
 * For fixed (A,B,C), every reducible P+D = A x^3 + B x^2 + C x + D has a
 * rational root p/q with q | A and p | D.  Inverting that: D = -P(p/q).
 * We enumerate all candidate roots p/q (|p| <= DMAX, q | A) and mark the
 * D bitmap, then scan for the longest run of consecutive marked D.
 * Exact integer arithmetic; exhaustive within the box.
 *
 * Build: cc -O2 -o poly_gap_box poly_gap_box.c
 * Run:   ./poly_gap_box [A_MAX] [BC_MAX] [D_MAX]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long long gA, gB, gC;

static long long gcdll(long long a, long long b) {
    long long t;
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { t = a % b; a = b; b = t; }
    return a;
}

static long long cube(long long x) { return x * x * x; }

static void scan_abc(long long A, long long B, long long C, long long DMAX,
                     long long *best_len, long long *best_abc, long long *best_run,
                     long long *n5, long long *n6)
{
    static unsigned char mark[2048];
    memset(mark, 0, sizeof(mark));
    long long q[32]; int nq = 0;
    for (long long d = 1; d <= A; d++)
        if (A % d == 0) q[nq++] = d;

    for (int qi = 0; qi < nq; qi++) {
        long long Q = q[qi];
        for (long long p = -DMAX; p <= DMAX; p++) {
            if (p == 0) continue;
            long long num = A * cube(p) + B * p * p * Q + C * p * Q * Q;
            if (num % cube(Q)) continue;
            long long D = -(num / cube(Q));
            if (D < -DMAX || D > DMAX) continue;
            mark[D + DMAX] = 1;
        }
    }

    long long cur = 0, best = 0, bestD = 0;
    for (long long D = -DMAX; D <= DMAX; D++) {
        if (mark[D + DMAX]) {
            if (cur == 0) bestD = D;
            cur++;
            if (cur > best) { best = cur; }
            if (cur == 5) (*n5)++;
            if (cur == 6) {
                (*n6)++;
                printf("RUN6: P(x)=%lldx^3%+lldx^2%+lldx%+lld D in [%lld..]\n",
                       A, B, C, D - 5, D);
            }
        } else {
            cur = 0;
        }
        if (cur > *best_len) {
            *best_len = cur;
            best_abc[0] = A; best_abc[1] = B; best_abc[2] = C;
            best_run[0] = D - cur + 1; best_run[1] = D;
        }
    }
}

int main(int argc, char **argv)
{
    long long A_MAX = argc > 1 ? atoll(argv[1]) : 12;
    long long BC_MAX = argc > 2 ? atoll(argv[2]) : 120;
    long long D_MAX = argc > 3 ? atoll(argv[3]) : 120;

    long long best_len = 0, best_abc[3] = {0,0,0}, best_run[2] = {0,0};
    long long n5 = 0, n6 = 0;

    for (long long A = 1; A <= A_MAX; A++) {
        for (long long B = -BC_MAX; B <= BC_MAX; B++) {
            for (long long C = -BC_MAX; C <= BC_MAX; C++) {
                scan_abc(A, B, C, D_MAX, &best_len, best_abc, best_run,
                         &n5, &n6);
            }
        }
    }

    printf("box: A<=%lld |B|,|C|<=%lld |D|<=%lld\n", A_MAX, BC_MAX, D_MAX);
    printf("longest run: %lld  P(x)=%lldx^3%+lldx^2%+lldx  D in [%lld..%lld]\n",
           best_len, best_abc[0], best_abc[1], best_abc[2],
           best_run[0], best_run[1]);
    printf("5-runs started: %lld  6-runs: %lld\n", n5, n6);
    return 0;
}
