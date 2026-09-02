/* poly_gap_six.c — direct search for SIX consecutive reducible cubics.
 *
 * For each D in [-DMAX,DMAX] and each q-combination (q_j | A), the only
 * possible root numerators of P+j are the DIVISORS of D+j (rational root
 * theorem).  Enumerate p_0,p_1 from the divisor lists of D, D+1, solve
 * B,C exactly (2x2, __int128 arithmetic), then verify j=2..5 by scanning
 * the divisor lists of D+2..D+5.  Exhaustive over |D| <= DMAX with NO
 * bound on B or C.
 *
 * Build: cc -O2 -o poly_gap_six poly_gap_six.c
 * Run:   ./poly_gap_six A_MAX DMAX [q_restrict]
 *        q_restrict: 0 = all q_j | A, 1 = only q_j in {1,2}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef __int128 i128;

static i128 iabs128(i128 x) { return x < 0 ? -x : x; }

static long long gcdll(long long a, long long b) {
    long long t;
    if (a < 0) a = -a;
    if (b < 0) b = -b;
    while (b) { t = a % b; a = b; b = t; }
    return a;
}

/* divisor lists for n in [-DMAX-5, DMAX+5] -> index n + DMAX + 5 */
static int *divcnt;
static int **divlists;

static void build_divtables(long long dmax) {
    long long size = 2 * dmax + 11;
    divcnt = calloc(size, sizeof(int));
    divlists = calloc(size, sizeof(int *));
    for (long long d = 1; d <= dmax + 5; d++) {
        for (long long m = d; m <= dmax + 5; m += d) {
            divcnt[dmax + 5 + m]++;
            divcnt[dmax + 5 - m]++;
        }
    }
    for (long long i = 0; i < size; i++)
        divlists[i] = malloc(sizeof(int) * (divcnt[i] ? divcnt[i] : 1));
    memset(divcnt, 0, size * sizeof(int));
    for (long long d = 1; d <= dmax + 5; d++) {
        for (long long m = d; m <= dmax + 5; m += d) {
            divlists[dmax + 5 + m][divcnt[dmax + 5 + m]++] = (int)d;
            divlists[dmax + 5 - m][divcnt[dmax + 5 - m]++] = (int)-d;
        }
    }
}

static void put128(i128 x) {
    if (x < 0) { putchar('-'); x = -x; }
    char buf[64]; int n = 0;
    do { buf[n++] = (char)('0' + (int)(x % 10)); x /= 10; } while (x);
    while (n) putchar(buf[--n]);
}

int main(int argc, char **argv)
{
    long long A_MAX = argc > 1 ? atoll(argv[1]) : 4;
    long long DMAX = argc > 2 ? atoll(argv[2]) : 500000;
    int q_restrict = argc > 3 ? atoi(argv[3]) : 1;

    build_divtables(DMAX);
    long long nfound = 0;

    for (long long A = 1; A <= A_MAX; A++) {
        long long q[16]; int nq = 0;
        for (long long d = 1; d <= A; d++)
            if (A % d == 0 && (!q_restrict || (d == 1 || d == 2)))
                q[nq++] = d;
        /* q-combination enumeration via mixed-radix over 6 positions */
        unsigned long long combos = 1;
        for (int j = 0; j < 6; j++) combos *= (unsigned long long)nq;
        for (unsigned long long cc = 0; cc < combos; cc++) {
            long long qs[6]; unsigned long long t = cc;
            for (int j = 0; j < 6; j++) { qs[j] = q[t % nq]; t /= nq; }
            for (long long D = -DMAX; D <= DMAX; D++) {
                int *c0 = divlists[DMAX + 5 + D];
                int *c1 = divlists[DMAX + 5 + D + 1];
                for (int i0 = 0; i0 < divcnt[DMAX + 5 + D]; i0++) {
                    i128 p0 = c0[i0], q0 = qs[0];
                    for (int i1 = 0; i1 < divcnt[DMAX + 5 + D + 1]; i1++) {
                        i128 p1 = c1[i1], q1 = qs[1];
                        /* solve B,C:  A p^3 + B p^2 q + C p q^2 + D q^3 = 0 */
                        i128 a0 = p0 * p0 * q0, b0 = p0 * q0 * q0;
                        i128 r0 = -A * p0 * p0 * p0 - D * q0 * q0 * q0;
                        i128 a1 = p1 * p1 * q1, b1 = p1 * q1 * q1;
                        i128 r1 = -A * p1 * p1 * p1 - (D + 1) * q1 * q1 * q1;
                        i128 det = a0 * b1 - a1 * b0;
                        if (det == 0) continue;
                        i128 Bn = r0 * b1 - r1 * b0;
                        i128 Cn = a0 * r1 - a1 * r0;
                        if (Bn % det || Cn % det) continue;
                        i128 B = Bn / det, C = Cn / det;
                        int ok = 1;
                        for (int j = 2; j < 6 && ok; j++) {
                            long long idx = DMAX + 5 + D + j;
                            int *cj = divlists[idx];
                            int nj = divcnt[idx];
                            int hit = 0;
                            for (int k = 0; k < nj; k++) {
                                i128 p = cj[k], qq = qs[j];
                                i128 v = A * p * p * p + B * p * p * qq
                                       + C * p * qq * qq + (D + j) * qq * qq * qq;
                                if (v == 0) { hit = 1; break; }
                            }
                            if (!hit) ok = 0;
                        }
                        if (ok) {
                            printf("RUN6: A=%lld D=%lld B=", A, D);
                            put128(B); printf(" C="); put128(C);
                            printf(" p=(%lld,%lld,..) q=(%lld,%lld,..)\n",
                                   (long long)p0, (long long)p1, (long long)q0,
                                   (long long)q1);
                            nfound++;
                        }
                    }
                }
            }
        }
    }
    printf("six-runs found (A<=%lld, |D|<=%lld, q%s): %lld\n",
           A_MAX, DMAX, q_restrict ? " in {1,2}" : " all", nfound);
    return 0;
}
