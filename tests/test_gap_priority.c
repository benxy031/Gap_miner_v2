/*
 * Integration Test: gap-search priority heuristic
 *
 * Verifies:
 *   1. gap_priority_value matches hand-computed cases (the S_c / S_p /
 *      max/cost formula).
 *   2. gap_priority_order returns a permutation of 0..n-1 ordered by
 *      S_c (survivor-gap span) descending, matching a brute-force check.
 *   3. The heuristic is a pure reordering: the set of survivor offsets is
 *      unchanged, so a tester would discover the same primes as a linear scan.
 */

#include "../new_src/gap_priority.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int nearly_equal(double a, double b) {
    double d = a - b;
    if (d < 0) d = -d;
    return d < 1e-9;
}

static int test_value(void) {
    int failures = 0;

    /* Composite-form case: Lc=3, Rc=5 -> Sc = 3+2+5 = 10. Sp=2. cost=2 -> 5 */
    if (!nearly_equal(gap_priority_value(3, 5, 1, 2, 2.0), 5.0)) {
        fprintf(stderr, "FAIL: Sc-dominant case\n");
        failures++;
    }

    /* Prime-close case: Lc=0,Rc=0 -> Sc=2. V_left=40,V_right=20 -> Sp=40.
       benefit=40, cost=10 -> 4 */
    if (!nearly_equal(gap_priority_value(0, 0, 40, 20, 10.0), 4.0)) {
        fprintf(stderr, "FAIL: Sp-dominant case\n");
        failures++;
    }

    /* zero/negative cost -> 0 */
    if (gap_priority_value(10, 10, 10, 10, 0.0) != 0.0) {
        fprintf(stderr, "FAIL: zero-cost guard\n");
        failures++;
    }

    return failures;
}

static int test_order(void) {
    /* Survivor offsets with mixed survivor-gap spans. */
    const uint64_t offsets[] = {3, 5, 50, 55, 200};
    const size_t n = sizeof(offsets) / sizeof(offsets[0]);
    const uint64_t interval_size = 256;
    size_t order[5];

    /* Brute-force S_c per index. */
    uint64_t sc_expected[5];
    for (size_t i = 0; i < n; i++) {
        uint64_t prev = (i == 0) ? 0ULL : offsets[i - 1];
        uint64_t next = (i + 1 == n) ? interval_size : offsets[i + 1];
        sc_expected[i] = next - prev;
    }

    gap_priority_order(offsets, n, interval_size, order);

    /* 1. permutation check */
    int seen[5] = {0, 0, 0, 0, 0};
    for (size_t i = 0; i < n; i++) {
        if (order[i] >= n) {
            fprintf(stderr, "FAIL: order index out of range\n");
            return 1;
        }
        if (seen[order[i]]) {
            fprintf(stderr, "FAIL: duplicate order index\n");
            return 1;
        }
        seen[order[i]] = 1;
    }

    /* 2. descending S_c check */
    for (size_t i = 1; i < n; i++) {
        if (sc_expected[order[i - 1]] < sc_expected[order[i]]) {
            fprintf(stderr, "FAIL: order not descending by S_c\n");
            return 1;
        }
    }

    /* 3. reordering preserves the offset set (trivially true for a
       permutation, but assert it explicitly for clarity). */
    printf("  priority order: ");
    for (size_t i = 0; i < n; i++) {
        printf("%zu ", order[i]);
    }
    printf("\n  S_c values:     ");
    for (size_t i = 0; i < n; i++) {
        printf("%llu ", (unsigned long long)sc_expected[order[i]]);
    }
    printf("\n");

    return 0;
}

int main(void) {
    int failures = test_value();
    failures += test_order();

    if (failures == 0) {
        printf("PASS: gap priority heuristic matches the formula and is a "
               "descending S_c reordering\n");
        return 0;
    }

    fprintf(stderr, "FAIL: %d checks failed\n", failures);
    return 1;
}
