#include "host.h"
#include "gv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int prp_calls;

static int failing(const char *stage)
{
    const char *mode = getenv("GV_TEST_FAIL");
    return mode && strcmp(mode, stage) == 0;
}

int gv_gpu_init(void)
{
    return 0;
}

int gv_gpu_available(void)
{
    return 0;
}

int gv_bpsw(mpz_t n)
{
    if (failing("large") && mpz_sizeinbase(n, 2) > GV_MAX_BITS)
        return 1;
    return mpz_probab_prime_p(n, 1) != 0;
}

int gv_ref_strong_prp2(mpz_t n)
{
    return gv_bpsw(n);
}

int gv_gpu_prp_batch(mpz_t *nums, int count, uint8_t *pass_out)
{
    prp_calls++;
    if ((prp_calls == 1 && failing("endpoints")) ||
        (prp_calls > 1 && failing("interval")))
        return -1;
    for (int i = 0; i < count; i++)
        pass_out[i] = gv_bpsw(nums[i]) ? 1 : 0;
    return 0;
}

int gv_gpu_sieve_gap(mpz_t n, uint64_t gap, uint32_t pmax,
                     uint8_t **marks_out, uint64_t *half_gap_out)
{
    (void)n;
    (void)pmax;
    if (failing("sieve"))
        return -1;
    *half_gap_out = gap < 4 ? 0 : gap / 2 - 1;
    *marks_out = *half_gap_out ? calloc((size_t)*half_gap_out, 1) : NULL;
    if (*half_gap_out && !*marks_out) {
        fprintf(stderr, "mock GPU: out of memory\n");
        return -1;
    }
    return 0;
}
