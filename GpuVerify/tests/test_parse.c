/* parser validation against the real allgaps.sql */
#include "../src/gv.h"
#include "../src/parse.h"

#include <gmp.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/tmp/allgaps.sql";
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 1; }

    char *line = NULL;
    size_t cap = 0;
    unsigned long lineno = 0, rows = 0, parse_fail = 0, too_large = 0;
    unsigned long digits_mismatch = 0;
    uint64_t max_gap = 0;
    int max_digits = 0;

    mpz_t n;
    mpz_init(n);

    while (getline(&line, &cap, f) != -1) {
        lineno++;
        uint64_t gap;
        int digits;
        char *expr = NULL;
        if (gv_db_extract_row(line, lineno, &gap, &digits, &expr) != 0)
            continue;
        rows++;
        if (gap > max_gap) max_gap = gap;
        if (digits > max_digits) max_digits = digits;
        int rc = gv_parse_expr(expr, n, 1);
        if (rc == GV_PARSE_TOO_LARGE) {
            too_large++;
        } else if (rc != 0) {
            if (parse_fail < 20)
                printf("PARSE_FAIL gap=%llu digits=%d expr=%s\n",
                       (unsigned long long)gap, digits, expr);
            parse_fail++;
        } else {
            /* exact digit count: mpz_sizeinbase may be 1 too big */
            char *ns = mpz_get_str(NULL, 10, n);
            int nd = (int)strlen(ns);
            free(ns);
            if (nd != digits) {
                if (digits_mismatch < 20)
                    printf("DIGITS gap=%llu stored=%d actual=%d expr=%s\n",
                           (unsigned long long)gap, digits, nd, expr);
                digits_mismatch++;
            }
        }
        free(expr);
    }
    free(line);
    fclose(f);
    mpz_clear(n);

    printf("rows=%lu parse_fail=%lu too_large=%lu digits_mismatch=%lu max_gap=%llu max_digits=%d\n",
           rows, parse_fail, too_large, digits_mismatch,
           (unsigned long long)max_gap, max_digits);

    /* the Payne evidence rows */
    static const char *known[] = {
        "10^19662 - 58727", "10^19666 - 98199",
        "10^16104 - 131519", "10^21530 - 61643",
        "19307#/390637 - 1", /* 206048? keep generic: parse only */
        "86968566716 * 79# - 954846259588805228035541589679",
        "24912408518 * 83# + 14843",
        "374087114868254547#/4367551 - 1",
        "10^22 - 1",
    };
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++) {
        if (gv_parse_expr(known[i], n, 1) == 0) {
            char *ns = mpz_get_str(NULL, 10, n);
            printf("OK   %s -> %zu digits\n", known[i], strlen(ns));
            free(ns);
        } else
            printf("FAIL %s\n", known[i]);
    }
    return (parse_fail + digits_mismatch) ? 1 : 0;
}
