# High-Merit Weight Examples

Date: 2026-08-05

## Purpose

This report gives concrete starting weight profiles and runnable campaign examples
for high-merit search planning, separated for CRT and non-CRT usage.

Scope note:
- CRT examples are directly usable with the selector workflow.
- non-CRT examples are intended as scoring/prioritization experiments for
  smart-window ordering, not hard acceptance filters.

## Scoring Form

score = log(base_prob)
        + w_hl * log(HL_factor)
        - w_cg * cg_ratio
        - w_cand * log(1 + n_candidates)
        - w_pr * log(1 + n_primes)

Where:
- base_prob is the independent-survivor baseline probability.
- HL_factor is the odd-prime divisor factor for target gap G.
- cg_ratio is G / log(x)^2 (Cramer-Granville caution term).
- n_candidates and n_primes are cost proxies.

## Recommended Start Profiles

### CRT Profiles

1) Balanced CRT (default)
- w_hl = 0.60
- w_cg = 2.40
- w_cand = 0.012
- w_pr = 0.022

2) Aggressive Tail CRT (for 33+ / 35+ push)
- w_hl = 0.80
- w_cg = 2.80
- w_cand = 0.010
- w_pr = 0.018

3) Throughput-Safe CRT (when queue pressure is high)
- w_hl = 0.45
- w_cg = 2.10
- w_cand = 0.016
- w_pr = 0.028

### non-CRT Profiles

1) Balanced non-CRT
- w_hl = 0.30
- w_cg = 1.60
- w_cand = 0.008
- w_pr = 0.000

2) Quality non-CRT
- w_hl = 0.45
- w_cg = 2.00
- w_cand = 0.006
- w_pr = 0.000

3) Safety non-CRT
- w_hl = 0.20
- w_cg = 1.30
- w_cand = 0.012
- w_pr = 0.000

## Concrete CLI Examples

### A) CRT ranking run with Balanced CRT

/usr/bin/python3 scripts/high_merit_crt_selector.py \
  --merit-min 30 --merit-max 36 --merit-step 1 \
  --weight-hl 0.60 --weight-cg 2.40 \
  --weight-candidates 0.012 --weight-primes 0.022 \
  --top 5 --out docs/HIGH_MERIT_CAMPAIGN_BALANCED_2026-08-05.md

### B) CRT ranking run with Aggressive Tail CRT

/usr/bin/python3 scripts/high_merit_crt_selector.py \
  --merit-min 30 --merit-max 36 --merit-step 1 \
  --weight-hl 0.80 --weight-cg 2.80 \
  --weight-candidates 0.010 --weight-primes 0.018 \
  --top 5 --out docs/HIGH_MERIT_CAMPAIGN_AGGR_2026-08-05.md

### C) CRT ranking run with Throughput-Safe CRT

/usr/bin/python3 scripts/high_merit_crt_selector.py \
  --merit-min 30 --merit-max 36 --merit-step 1 \
  --weight-hl 0.45 --weight-cg 2.10 \
  --weight-candidates 0.016 --weight-primes 0.028 \
  --top 5 --out docs/HIGH_MERIT_CAMPAIGN_SAFE_2026-08-05.md

## Example Miner Launch Lanes

Use the winner from each per-merit report row. Typical launch pattern:

bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat \
  --target 32.00 --scan-merit 32.00 --crt-file crt/crt_s1001_m35.txt

bin/gap_miner --shift 512 --threads 8 --cuda 0 --fast-fermat \
  --target 33.00 --scan-merit 33.00 --crt-file crt/crt_s512_m35_p3.txt

## Practical A/B Campaign Template

1) Fix run conditions
- Keep same machine, same thread count, same GPU mode, same timebox per run.
- Use identical target merit band plan.

2) Compare three profiles
- Balanced CRT
- Aggressive Tail CRT
- Throughput-Safe CRT

3) Primary decision metrics
- merit >= 30 hits per hour
- merit >= 33 hits per hour
- best merit per hour
- accepted/tested
- tested/s and pps

4) Keep or reject policy
- Keep profile if tail metrics improve and throughput degradation is acceptable.
- Reject profile if throughput collapses without clear tail gain.

## Adjustment Rules

- If selected files look too optimistic for high shift but underperform in real
  runs, increase w_cg by +0.20.
- If queue latency/pressure rises, increase w_cand by +0.002.
- If tail quality is flat despite good throughput, increase w_hl by +0.10 and
  rerun only fixed-header A/B.

## Guardrails

- Do not use HL or total score as a hard reject for final submission logic.
- Use score only for ranking/prioritization order.
- Keep correctness and acceptance semantics unchanged.
