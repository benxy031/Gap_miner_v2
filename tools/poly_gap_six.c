/* poly_gap_six.c — direct search for SIX consecutive reducible cubics.
 *
 * For each nonzero constant N=D+j, every reduced rational root p/q has
 * p | N and q | A.  Both signs of p must be enumerated.  If N==0, x=0 is
 * already a root.  Two nonzero shifts are selected as anchors, B and C are
 * solved exactly (2x2, __int128 arithmetic), and all remaining shifts are
 * checked exactly.  Exhaustive over |D| <= DMAX with NO bound on B or C.
 *
 * 2026-09-03 external-review corrections: both divisor signs stored (the
 * original kept only +d for positive and -d for negative constants),
 * zero constants automatically have root 0, anchors are the first two
 * nonzero shifts (was fixed shifts 0,1), reduced p/q only, dynamically
 * sized divisor/denominator tables, output labeled as witnesses.
 *
 * Build with GCC/Clang: cc -O2 -o poly_gap_six poly_gap_six.c
 * Run:   ./poly_gap_six A_MAX DMAX [q_restrict]
 *        q_restrict: 0 = all q_j | A, 1 = only q_j in {1,2}
 */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef __int128 i128;

static int *divcnt;
static long long **divlists;
static long long table_center;
static size_t table_size;

static void die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(EXIT_FAILURE);
}

static long long gcdll(long long a, long long b)
{
    if (a < 0)
        a = -a;
    if (b < 0)
        b = -b;
    while (b != 0) {
        const long long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static size_t index_of(long long n)
{
    return (size_t)(table_center + n);
}

static void build_divtables(long long dmax)
{
    if (dmax < 0 || dmax > (LLONG_MAX - 11) / 2)
        die("Invalid DMAX.");
    if ((unsigned long long)dmax > (SIZE_MAX - 11u) / 2u)
        die("DMAX is too large for this address space.");

    table_center = dmax + 5;
    table_size = (size_t)dmax * 2u + 11u;
    divcnt = (int *)calloc(table_size, sizeof(*divcnt));
    divlists = (long long **)calloc(table_size, sizeof(*divlists));
    if (!divcnt || !divlists)
        die("Unable to allocate divisor-table headers.");

    const long long limit = dmax + 5;
    for (long long d = 1; d <= limit; ++d) {
        for (long long m = d; m <= limit; m += d) {
            divcnt[index_of(m)] += 2;  /* +d and -d both divide +m. */
            divcnt[index_of(-m)] += 2; /* +d and -d both divide -m. */
        }
    }

    for (size_t i = 0; i < table_size; ++i) {
        const size_t count = divcnt[i] ? (size_t)divcnt[i] : 1u;
        divlists[i] = (long long *)malloc(count * sizeof(**divlists));
        if (!divlists[i])
            die("Unable to allocate a divisor list.");
        divcnt[i] = 0;
    }

    for (long long d = 1; d <= limit; ++d) {
        for (long long m = d; m <= limit; m += d) {
            const size_t positive = index_of(m);
            const size_t negative = index_of(-m);
            divlists[positive][divcnt[positive]++] = d;
            divlists[positive][divcnt[positive]++] = -d;
            divlists[negative][divcnt[negative]++] = d;
            divlists[negative][divcnt[negative]++] = -d;
        }
    }
}

static void free_divtables(void)
{
    if (divlists) {
        for (size_t i = 0; i < table_size; ++i)
            free(divlists[i]);
    }
    free(divlists);
    free(divcnt);
}

static void put128(i128 x)
{
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    char buffer[64];
    int length = 0;
    do {
        buffer[length++] = (char)('0' + (int)(x % 10));
        x /= 10;
    } while (x != 0);
    while (length != 0)
        putchar(buffer[--length]);
}

int main(int argc, char **argv)
{
    const long long amax = argc > 1 ? atoll(argv[1]) : 4;
    const long long dmax = argc > 2 ? atoll(argv[2]) : 500000;
    const int q_restrict = argc > 3 ? atoi(argv[3]) : 1;

    if (amax < 1 || dmax < 0)
        die("Expected A_MAX >= 1 and DMAX >= 0.");

    build_divtables(dmax);
    unsigned long long witnesses = 0;

    for (long long A = 1; A <= amax; ++A) {
        size_t nq = 0;
        for (long long d = 1; d <= A; ++d) {
            if (A % d == 0 && (!q_restrict || d == 1 || d == 2))
                ++nq;
        }
        if (nq == 0)
            continue;

        long long *q = (long long *)malloc(nq * sizeof(*q));
        if (!q)
            die("Unable to allocate denominator list.");
        size_t qpos = 0;
        for (long long d = 1; d <= A; ++d) {
            if (A % d == 0 && (!q_restrict || d == 1 || d == 2))
                q[qpos++] = d;
        }

        unsigned long long combinations = 1;
        for (int j = 0; j < 6; ++j) {
            if (nq > ULLONG_MAX / combinations)
                die("Too many denominator combinations.");
            combinations *= (unsigned long long)nq;
        }

        for (unsigned long long code = 0; code < combinations; ++code) {
            long long qs[6];
            unsigned long long value = code;
            for (int j = 0; j < 6; ++j) {
                qs[j] = q[value % nq];
                value /= nq;
            }

            for (long long D = -dmax; D <= dmax; ++D) {
                /* Use the first two nonzero constants as independent anchors. */
                int u = -1;
                int v = -1;
                for (int j = 0; j < 6; ++j) {
                    if (D + j == 0)
                        continue;
                    if (u < 0)
                        u = j;
                    else {
                        v = j;
                        break;
                    }
                }
                if (v < 0)
                    die("Internal error: fewer than two nonzero shifts.");

                const size_t iu = index_of(D + u);
                const size_t iv = index_of(D + v);
                for (int ku = 0; ku < divcnt[iu]; ++ku) {
                    const long long pu64 = divlists[iu][ku];
                    if (gcdll(pu64, qs[u]) != 1)
                        continue;
                    const i128 pu = (i128)pu64;
                    const i128 qu = (i128)qs[u];

                    for (int kv = 0; kv < divcnt[iv]; ++kv) {
                        const long long pv64 = divlists[iv][kv];
                        if (gcdll(pv64, qs[v]) != 1)
                            continue;
                        const i128 pv = (i128)pv64;
                        const i128 qv = (i128)qs[v];

                        const i128 au = pu * pu * qu;
                        const i128 bu = pu * qu * qu;
                        const i128 ru =
                            -(i128)A * pu * pu * pu -
                            (i128)(D + u) * qu * qu * qu;
                        const i128 av = pv * pv * qv;
                        const i128 bv = pv * qv * qv;
                        const i128 rv =
                            -(i128)A * pv * pv * pv -
                            (i128)(D + v) * qv * qv * qv;

                        const i128 determinant = au * bv - av * bu;
                        if (determinant == 0)
                            continue;
                        const i128 Bn = ru * bv - rv * bu;
                        const i128 Cn = au * rv - av * ru;
                        if (Bn % determinant != 0 || Cn % determinant != 0)
                            continue;
                        const i128 B = Bn / determinant;
                        const i128 C = Cn / determinant;

                        int valid = 1;
                        for (int j = 0; j < 6 && valid; ++j) {
                            if (j == u || j == v || D + j == 0)
                                continue;

                            const size_t index = index_of(D + j);
                            int hit = 0;
                            for (int k = 0; k < divcnt[index]; ++k) {
                                const long long p64 = divlists[index][k];
                                if (gcdll(p64, qs[j]) != 1)
                                    continue;
                                const i128 p = (i128)p64;
                                const i128 denominator = (i128)qs[j];
                                const i128 equation =
                                    (i128)A * p * p * p +
                                    B * p * p * denominator +
                                    C * p * denominator * denominator +
                                    (i128)(D + j) * denominator * denominator * denominator;
                                if (equation == 0) {
                                    hit = 1;
                                    break;
                                }
                            }
                            if (!hit)
                                valid = 0;
                        }

                        if (valid) {
                            printf("RUN6 witness: A=%lld D=%lld B=", A, D);
                            put128(B);
                            printf(" C=");
                            put128(C);
                            printf(" anchors=(%d,%d) roots=(%lld/%lld,%lld/%lld)\n",
                                   u, v, pu64, qs[u], pv64, qs[v]);
                            ++witnesses;
                        }
                    }
                }
            }
        }
        free(q);
    }

    printf("six-run witnesses (A<=%lld, |D|<=%lld, q%s): %llu\n",
           amax, dmax, q_restrict ? " in {1,2}" : " all",
           witnesses);
    puts("Note: multiple witnesses may describe the same polynomial.");

    free_divtables();
    return EXIT_SUCCESS;
}
