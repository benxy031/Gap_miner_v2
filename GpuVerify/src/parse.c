#include "parse.h"
#include "gv.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int all_digits(const char *s)
{
    if (*s == '\0')
        return 0;
    for (; *s; s++)
        if (!isdigit((unsigned char)*s))
            return 0;
    return 1;
}

/* find the last '+'/'-' at parenthesis depth 0, index > 0 */
static const char *find_tail_sign(const char *s)
{
    const char *best = NULL;
    int depth = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '(')
            depth++;
        else if (*p == ')')
            depth--;
        else if (depth == 0 && p > s && (*p == '+' || *p == '-')) {
            if (*(p - 1) != '/' && *(p - 1) != '*' &&
                *(p - 1) != '^' && *(p - 1) != '(')
                best = p;
        }
    }
    return best;
}

int gv_parse_expr(const char *s, mpz_t out, int quiet)
{
    /* copy and strip whitespace */
    size_t len = strlen(s);
    char *buf = malloc(len + 1);
    if (!buf)
        return -1;
    size_t k = 0;
    for (size_t i = 0; i < len; i++)
        if (!isspace((unsigned char)s[i]))
            buf[k++] = s[i];
    buf[k] = '\0';

    int rc = -1;
    if (all_digits(buf)) {
        if (mpz_set_str(out, buf, 10) == 0)
            rc = 0;
        free(buf);
        if (rc && !quiet)
            fprintf(stderr, "parse: bad digits in %s\n", s);
        return rc;
    }

    /* tail: last top-level +/- */
    const char *tail = find_tail_sign(buf);
    int sign = 1;
    char *a_str = NULL;
    char *body = buf;
    if (tail) {
        if (!all_digits(tail + 1)) {
            if (!quiet)
                fprintf(stderr, "parse: bad offset after '%c' in %s\n", *tail, s);
            free(buf);
            return -1;
        }
        sign = (*tail == '-') ? -1 : 1;
        a_str = (char *)tail + 1;
        *((char *)tail) = '\0';
        body = buf;
    }

    mpz_t t, u, a;
    mpz_init(t);
    mpz_init(u);
    mpz_init(a);
    if (a_str)
        mpz_set_str(a, a_str, 10);

    char *hat = strchr(body, '^');
    if (hat) {
        /* b^p */
        char *p = hat + 1;
        *hat = '\0';
        if (!all_digits(body) || !all_digits(p) || *body == '\0' || *p == '\0') {
            if (!quiet)
                fprintf(stderr, "parse: bad power form in %s\n", s);
            goto out;
        }
        unsigned long b = strtoul(body, NULL, 10);
        unsigned long e = strtoul(p, NULL, 10);
        if (b < 2) {
            if (!quiet)
                fprintf(stderr, "parse: base < 2 in %s\n", s);
            goto out;
        }
        double est = (double)e * log10((double)b);
        if (est > GV_MAX_MATERIALIZE_DIGITS) {
            rc = GV_PARSE_TOO_LARGE;
            goto out;
        }
        mpz_ui_pow_ui(t, b, e);
        if (sign > 0)
            mpz_add(out, t, a);
        else
            mpz_sub(out, t, a);
        rc = 0;
        goto out;
    }

    char *hash = strchr(body, '#');
    if (!hash) {
        if (!quiet)
            fprintf(stderr, "parse: no known form matches %s\n", s);
        goto out;
    }

    /* left of '#': optional "m*" coefficient, then the primorial index
     * digits directly before '#'. */
    {
        char *idx_end = hash;      /* at '#' */
        char *idx_start = idx_end;
        while (idx_start > body && isdigit((unsigned char)idx_start[-1]))
            idx_start--;
        if (idx_start == idx_end) {
            if (!quiet)
                fprintf(stderr, "parse: missing primorial index in %s\n", s);
            goto out;
        }
        unsigned long index = strtoul(idx_start, NULL, 10);
        mpz_t m;
        mpz_init_set_ui(m, 1);
        if (idx_start > body) {
            /* must be exactly "m*" before the index */
            if (idx_start[-1] != '*' || idx_start - body < 2) {
                if (!quiet)
                    fprintf(stderr, "parse: bad coefficient in %s\n", s);
                mpz_clear(m);
                goto out;
            }
            char *c = body;
            idx_start[-1] = '\0';
            if (!all_digits(c)) {
                if (!quiet)
                    fprintf(stderr, "parse: bad coefficient in %s\n", s);
                mpz_clear(m);
                goto out;
            }
            mpz_set_str(m, c, 10);
        }
        double est = 0.4342944819 * (double)index +
                     (double)(mpz_sizeinbase(m, 10)); /* upper bound is fine */
        if (est > GV_MAX_MATERIALIZE_DIGITS) {
            mpz_clear(m);
            rc = GV_PARSE_TOO_LARGE;
            goto out;
        }
        mpz_primorial_ui(t, index);
        mpz_mul(t, t, m);
        mpz_clear(m);
    }

    /* right of '#': the remainder is either the divisor part or the tail
     * was already stripped, so anything left here is a divisor spec. */
    /* NOTE: tail was stripped from `buf`, so `hash + 1` now points into
     * the body; the divisor text starts right after the primorial index
     * digits. Walk to the end of the number after '#'. */
    char *r = hash + 1;
    while (*r && isdigit((unsigned char)*r))
        r++;
    /* r points at '/' + divisor or at end of body */

    if (*r == '\0') {
        /* m*p# +- a (no divisor) */
        if (sign > 0)
            mpz_add(out, t, a);
        else
            mpz_sub(out, t, a);
        rc = 0;
        goto out;
    }
    if (*r != '/') {
        if (!quiet)
            fprintf(stderr, "parse: garbage after primorial in %s\n", s);
        goto out;
    }

    {
        char *d = r + 1; /* divisor spec */
        size_t dl = strlen(d);
        unsigned long d1, d2;
        char *star;

        if (all_digits(d)) {
            /* /d */
            if (!mpz_divisible_ui_p(t, strtoul(d, NULL, 10))) {
                if (!quiet)
                    fprintf(stderr, "parse: %s not divisible by %s\n", body, d);
                goto out;
            }
            mpz_divexact_ui(t, t, strtoul(d, NULL, 10));
        } else if (dl >= 3 && d[0] == '(' && d[dl - 1] == ')') {
            char inner[64];
            if (dl - 2 >= (int)sizeof(inner)) {
                if (!quiet)
                    fprintf(stderr, "parse: divisor too long in %s\n", s);
                goto out;
            }
            memcpy(inner, d + 1, dl - 2);
            inner[dl - 2] = '\0';
            star = strchr(inner, '*');
            if (!star) {
                if (!quiet)
                    fprintf(stderr, "parse: bad parenthesized divisor in %s\n", s);
                goto out;
            }
            *star = '\0';
            char *left = inner, *right = star + 1;
            if (all_digits(left) && all_digits(right)) {
                d1 = strtoul(left, NULL, 10);
                d2 = strtoul(right, NULL, 10);
                /* (d# * m2)  vs  (d * m2): '#' directly after d1 */
                size_t l1 = strlen(left);
                if (l1 > 0 && left[l1 - 1] == '#') {
                    left[l1 - 1] = '\0';
                    if (!mpz_divisible_ui_p(t, d1))
                        goto bad_div;
                    mpz_primorial_ui(u, d1);
                    mpz_mul_ui(u, u, d2);
                    if (!mpz_divisible_p(t, u))
                        goto bad_div;
                    mpz_divexact(t, t, u);
                } else {
                    unsigned long dd = d1 * d2;
                    if (!mpz_divisible_ui_p(t, dd))
                        goto bad_div;
                    mpz_divexact_ui(t, t, dd);
                }
            } else {
                if (!quiet)
                    fprintf(stderr, "parse: bad divisor %s\n", s);
                goto out;
            }
        } else if (dl >= 2 && d[dl - 1] == '#') {
            /* /d# */
            char tmp[32];
            if (dl - 1 >= (int)sizeof(tmp)) {
                if (!quiet)
                    fprintf(stderr, "parse: divisor too long in %s\n", s);
                goto out;
            }
            memcpy(tmp, d, dl - 1);
            tmp[dl - 1] = '\0';
            d1 = strtoul(tmp, NULL, 10);
            mpz_primorial_ui(u, d1);
            if (!mpz_divisible_p(t, u))
                goto bad_div;
            mpz_divexact(t, t, u);
        } else {
            if (!quiet)
                fprintf(stderr, "parse: unrecognized divisor in %s\n", s);
            goto out;
        }

        if (sign > 0)
            mpz_add(out, t, a);
        else
            mpz_sub(out, t, a);
        rc = 0;
        goto out;

    bad_div:
        if (!quiet)
            fprintf(stderr, "parse: primorial not divisible by divisor in %s\n", s);
        goto out;
    }

out:
    mpz_clear(t);
    mpz_clear(u);
    mpz_clear(a);
    free(buf);
    return rc;
}

int gv_db_extract_row(const char *line, unsigned long line_no,
                      uint64_t *gap_out, int *digits_out, char **expr_out)
{
    static const char prefix[] = "INSERT INTO gaps VALUES";
    const char *p = line;
    while (*p == ' ' || *p == '\t')
        p++;
    if (strncmp(p, prefix, strlen(prefix)) != 0)
        return -1;
    p += strlen(prefix);
    while (*p == ' ' || *p == '\t')
        p++;
    if (*p != '(')
        return -1;
    p++;

    /* quote-aware field split */
    char *fields[16];
    int nf = 0;
    fields[nf++] = (char *)p;
    int inq = 0;
    for (; *p; p++) {
        if (*p == '\'')
            inq = !inq;
        else if (*p == ',' && !inq) {
            if (nf >= (int)(sizeof(fields) / sizeof(fields[0])) - 1)
                return -1;
            *((char *)p) = '\0';
            fields[nf++] = (char *)p + 1;
        } else if (*p == ')' && !inq) {
            *((char *)p) = '\0';
            break;
        }
    }

    if (nf < 10)
        return -1;

    /* strip spaces */
    for (int i = 0; i < nf; i++) {
        char *q = fields[i];
        while (*q == ' ' || *q == '\t')
            q++;
        fields[i] = q;
        char *e = q + strlen(q);
        while (e > q && (e[-1] == ' ' || e[-1] == '\t'))
            *--e = '\0';
    }

    /* fields: 0=gap 1=primetype 2='C' 3='F?' 4='C?' 5=who 6=year
     *         7=merit 8=digits 9=start */
    char *endp = NULL;
    unsigned long long g = strtoull(fields[0], &endp, 10);
    if (!endp || *endp != '\0' || g == 0)
        return -1;
    long dg = strtol(fields[8], &endp, 10);
    if (!endp || *endp != '\0' || dg <= 0 || dg > 100000000)
        return -1;

    char *e = fields[9];
    size_t el = strlen(e);
    if (el >= 2 && e[0] == '\'' && e[el - 1] == '\'') {
        e[el - 1] = '\0';
        e++;
        el -= 2;
    }
    if (el == 0)
        return -1;

    char *copy = strdup(e);
    if (!copy)
        return -1;

    (void)line_no;
    *gap_out = (uint64_t)g;
    *digits_out = (int)dg;
    *expr_out = copy;
    return 0;
}
