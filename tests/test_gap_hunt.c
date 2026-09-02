/*
 * Copyright (C) 2026  GapMiner V2 contributors
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * GAP_HUNT record validator.
 *
 * Usage: test_gap_hunt <out-file> [bpsw]
 *
 * For every record `<gap> <merit> <startprime>`:
 *   1. gap == end - start where end = start + gap  (trivial identity, sanity)
 *   2. mpz_nextprime(start) == end                 (exactness: the gap is real)
 *   3. optional `bpsw`: both endpoints pass baillie_psw_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gmp.h>

#include "../new_src/primality_bpsw.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <out-file> [bpsw]\n", argv[0]);
        return 2;
    }
    int want_bpsw = (argc > 2 && strcmp(argv[2], "bpsw") == 0);
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    mpz_t start, end, next;
    mpz_inits(start, end, next, NULL);
    char buf[1024];
    unsigned long long checked = 0, bad = 0;
    int lineno = 0;

    while (fgets(buf, sizeof(buf), f)) {
        lineno++;
        if (buf[0] == '#' || buf[0] == '\n')
            continue;
        unsigned long long gap;
        double merit;
        char gapstr[64], meritstr[64], startstr[512];
        int n = sscanf(buf, "%63s %63s %511s", gapstr, meritstr, startstr);
        if (n != 3) {
            fprintf(stderr, "line %d: malformed (%d fields)\n", lineno, n);
            bad++;
            continue;
        }
        gap = strtoull(gapstr, NULL, 10);
        merit = atof(meritstr);
        (void)merit;
        if (mpz_set_str(start, startstr, 10) != 0) {
            fprintf(stderr, "line %d: bad start decimal\n", lineno);
            bad++;
            continue;
        }

        mpz_add_ui(end, start, (unsigned long)gap);

        /* True merit arithmetic: merit must equal gap / ln(start).
           Overflow-safe ln: mpz_get_d overflows above 2^1024 (shift >=
           ~770), which would zero the expected merit here. */
        {
            signed long int e = 0;
            double m = mpz_get_d_2exp(&e, start); /* start = m * 2^e */
            double ln_start = log(m) + (double)e * 0.69314718055994530942;
            double true_merit = (double)gap / ln_start;
            if (fabs(merit - true_merit) > 0.002) {
                fprintf(stderr,
                        "line %d: FAIL merit %.4f != gap/ln(start) %.6f\n",
                        lineno, merit, true_merit);
                bad++;
                continue;
            }
        }

        /* Exactness: the next prime after start must be exactly start + gap. */
        mpz_set(next, start);
        mpz_nextprime(next, next);
        if (mpz_cmp(next, end) != 0) {
            fprintf(stderr, "line %d: FAIL nextprime(start) != end (gap=%llu)\n",
                    lineno, gap);
            bad++;
            continue;
        }
        if (want_bpsw) {
            if (!baillie_psw_test(start) || !baillie_psw_test(end)) {
                fprintf(stderr, "line %d: FAIL BPSW endpoint\n", lineno);
                bad++;
                continue;
            }
        }
        checked++;
    }
    fclose(f);
    mpz_clears(start, end, next, NULL);

    printf("records=%llu bad=%llu\n", checked, bad);
    return bad ? 1 : 0;
}
