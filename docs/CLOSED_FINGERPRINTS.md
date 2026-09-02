# Closed Fingerprints — routes that are dead until a causal axis changes

Anti-regression registry for gapminer_v2 (mining + GAP_HUNT).  A **closed
fingerprint** is a route that was attempted, measured or proved, and failed
for an identified reason.  The same route must not be retried until one of
its causal axes changes (`data`, `hardware`, `representation`, `boundary`,
`unit of work`, `bottleneck`, `invariant`, `verifier`, `composition`,
`downstream cost`).  A rename, new language, or toy-size change does not
reopen an entry.

Claim labels used in logs and scripts (adopted from the record-grade witness
taxonomy): `EXACT_GAP`, `FIRST_KNOWN_OCCURRENCE`, `FIRST_OCCURRENCE`,
`MAXIMAL_GAP`, `MERIT_RECORD`, `NETWORK_ACCEPTED`.  Our tools only ever emit
`EXACT_GAP` (validator) and `FIRST_KNOWN_OCCURRENCE` (watcher/record log);
the stronger labels require coverage or network evidence we do not have.

## GAP_HUNT

| Fingerprint | Verdict / first cost | Evidence | Reopen trigger |
|---|---|---|---|
| QUARTER_CLASS two-pass scan for record hunting | EXACT but falsified: visible gaps inherit the σ-tail (mean merit ≈ 8), containment trigger fires ~1.3×/window at m=15; CPU resolution ~40 ms → 46 win/s vs 893 full-class | `GAP_HUNT_QUARTER` env (off), parity test identical, commit `7a0e30e` | a hidden-resolution path ~300× cheaper than mini-sieve 10k + GMP MR |
| Cover re-optimization beyond lex-m30 (74 primes) | lex/run/blocks objectives all converge ~1198 survivors vs production 1141; entropy bound with Σ1/p ≈ 2.04 leaves ~1000+ survivors | gen_crt probes Sep 2 | a new objective that provably lowers survivor count below ~900, or a different prime budget |
| "Gap factory" (2–3 survivors to force merit-31 gaps every window) | impossible: 74 prime moduli cannot cover [1,15867) up to 3 survivors (set-cover entropy); prime-moduli covering systems cannot cover Z (Euclid) | analysis Sep 2 | a cover with Σ 1/p ≫ 2.4 (hundreds of primes) at negligible mark cost |
| Synthetic gap construction (no search) | primality of endpoints cannot be constructed without primality search; AKS is a test, not a birth law | theorem cards G0 (sibling corpus) | an identity that co-defines the relation before the integers exist |
| GPU_SIEVE_PAIR (mining experiment, kept for reference) | −58% win/s: monolithic 2-window kernel starves extract/MR kernels | README GPU table | scheduler evidence the mark kernel no longer serializes |
| Same-family fixed-budget mono-CRT retuning for equal M30 yield | current mono-CRT ~87× below M25 per row; retuning cannot change the family's p0/row | sibling corpus R445 `m30` verdict + our lex probe | a different family/boundary, not new residue choices |

## GapMiner (mining pipeline)

| Fingerprint | Verdict / first cost | Evidence | Reopen trigger |
|---|---|---|---|
| shift998 m40 for live merit | no ≥16-merit gaps on the pipeline; MR cost (bits/64)² dominates | production A/B (May–Jun) | CGBN width for 1280-bit candidates measurably cheaper |
| Blocks/run objectives as σ boost | runs never approach merit scale; σ is conditioning-driven, not run-driven | `gen_crt` -B/-d tools + falsification notes | a score that correlates with exact-gap yield, not run count |
| CGBN sliding-window tuning (WIN_BITS=4, binary sq-and-mul, maxrregcount) | WIN_BITS=3 already optimal at 255 regs; all alternatives spill more | memory: GPU Fermat kernel notes | a different AL/NL geometry or SM arch |
| Producer-consumer overlap at shift=512 | 30% window queue drops (sieve outproduces GPU) | memory: monolithic 4T best | a consumer ≥ 2× faster |
| `GPU_SIEVE_PAIR` (see above) | −58% | README | — |
| Smart-scan tail-skip with per-pair hidden resolution at record thresholds | resolution cost > MR savings (same trigger-rate law as QUARTER_CLASS) | quarter-class parity/benchmark | — |

## House rules

1. Before starting any experiment, check this file; a matching fingerprint must
   name the causal axis that changed, or the experiment is a regression replay.
2. Every new closed entry records: objective, representation, authenticated
   facts, first compensating cost, evidence (commit/file), reopen trigger.
3. Closing an entry is a verdict on the route, not on the person or idea;
   dormant routes stay listed with their trigger.
