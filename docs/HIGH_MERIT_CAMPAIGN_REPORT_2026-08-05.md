# High-Merit CRT Selector

Date: 2026-08-05 (data refreshed 2026-08-06 after adding
crt_s1001_m45.txt, crt_s1001_m45_1.txt, crt_s1001_m42_2.txt,
crt_s1001_m42_3.txt to the portfolio)

## Scope

This report targets high-merit search planning (merit 30-36) for the existing
CRT file portfolio in this repository. It combines:

- Current repository behavior (`tools/eval_crt_merit.py`, `src/crt_runtime_cpu.c`,
	`src/crt_heap.c`, `src/gap_dist.c`).
- Public heuristic models for prime-gap rarity and local gap bias.
- A new ranking tool: `scripts/high_merit_crt_selector.py`.

## Theory Snapshot Used In The Model

The report uses a practical blend of the following heuristics:

- Cramer random-prime baseline: local prime probability near `x` is `1/log(x)`.
- Merit scaling: target gap `G ~= merit * log(x)`.
- Hardy-Littlewood singular-series style correction: even gaps with richer odd
	prime factorization are statistically favored via
	`prod_{p odd | G}(p-1)/(p-2)`.
- Cramer-Granville caution term: large normalized gaps `G/log(x)^2` are
	progressively penalized to avoid over-trusting optimistic random independence.

Practical note: these are ranking heuristics, not proofs. They are useful for
choosing which CRT files to test first, not for absolute probability claims.

Model: log(base_prob) + w_hl*log(HL) - w_cg*(gap/log^2) - cost

Weights: hl=0.650, cg=2.200, candidates=0.010, primes=0.020

where:

- `base_prob`: no-prime-before-G and at-least-one-prime-in-[G,W] under
	independent-survivor approximation.
- `HL`: Hardy-Littlewood multiplicative correction based on odd prime divisors
	of `G`.
- `cost`: logarithmic penalty from survivor pressure and CRT footprint.

## Merit 30.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m45.txt | -2.185339 | 1.446e-01 | 1.000 | 0.03443 | 3595 |
| 2 | crt_s1001_m35.txt | -2.201043 | 1.419e-01 | 1.000 | 0.03443 | 3609 |
| 3 | crt_s1001_m45_1.txt | -2.235999 | 1.374e-01 | 1.000 | 0.03443 | 3552 |
| 4 | crt_s1001_m42_3.txt | -2.245176 | 1.361e-01 | 1.000 | 0.03443 | 3545 |
| 5 | crt_s1001_m42_2.txt | -2.441337 | 1.119e-01 | 1.000 | 0.03443 | 3891 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 30.00 --scan-merit 30.00 --crt-file crt/crt_s1001_m45.txt
```

## Merit 31.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m45.txt | -2.033441 | 1.368e-01 | 1.381 | 0.03558 | 3735 |
| 2 | crt_s1001_m35.txt | -2.044756 | 1.349e-01 | 1.381 | 0.03558 | 3754 |
| 3 | crt_s1001_m45_1.txt | -2.073100 | 1.315e-01 | 1.381 | 0.03558 | 3690 |
| 4 | crt_s1001_m42_3.txt | -2.084104 | 1.299e-01 | 1.381 | 0.03558 | 3697 |
| 5 | crt_s1001_m42_2.txt | -2.297137 | 1.051e-01 | 1.381 | 0.03558 | 4037 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 31.00 --scan-merit 31.00 --crt-file crt/crt_s1001_m45.txt
```

## Merit 32.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m35.txt | -1.855916 | 1.283e-01 | 2.001 | 0.03673 | 3900 |
| 2 | crt_s1001_m45.txt | -1.858394 | 1.284e-01 | 2.001 | 0.03673 | 3880 |
| 3 | crt_s1001_m45_1.txt | -1.889628 | 1.244e-01 | 2.001 | 0.03673 | 3835 |
| 4 | crt_s1001_m42_3.txt | -1.892102 | 1.240e-01 | 2.001 | 0.03673 | 3839 |
| 5 | crt_s1001_m42_2.txt | -2.121852 | 9.866e-02 | 2.001 | 0.03673 | 4192 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 32.00 --scan-merit 32.00 --crt-file crt/crt_s1001_m35.txt
```

## Merit 33.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s512_m35_p3.txt | -2.299562 | 8.585e-02 | 2.001 | 0.06199 | 2971 |
| 2 | crt_s1001_m35.txt | -2.367067 | 1.211e-01 | 1.000 | 0.03788 | 4049 |
| 3 | crt_s1001_m45.txt | -2.368383 | 1.213e-01 | 1.000 | 0.03788 | 4026 |
| 4 | crt_s1001_m42_3.txt | -2.396255 | 1.179e-01 | 1.000 | 0.03788 | 3986 |
| 5 | crt_s1001_m45_1.txt | -2.401567 | 1.174e-01 | 1.000 | 0.03788 | 3979 |

Recommended launch command:

```sh
bin/gap_miner --shift 512 --threads 8 --cuda 0 --fast-fermat --target 33.00 --scan-merit 33.00 --crt-file crt/crt_s512_m35_p3.txt
```

## Merit 34.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m35.txt | -2.183219 | 1.138e-01 | 1.467 | 0.03902 | 4197 |
| 2 | crt_s1001_m45.txt | -2.183787 | 1.140e-01 | 1.467 | 0.03902 | 4166 |
| 3 | crt_s1001_m42_3.txt | -2.203402 | 1.117e-01 | 1.467 | 0.03902 | 4143 |
| 4 | crt_s1001_m45_1.txt | -2.205776 | 1.116e-01 | 1.467 | 0.03902 | 4126 |
| 5 | crt_s1001_m42_2.txt | -2.438189 | 8.843e-02 | 1.467 | 0.03902 | 4487 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 34.00 --scan-merit 34.00 --crt-file crt/crt_s1001_m35.txt
```

## Merit 35.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m35.txt | -2.493901 | 1.072e-01 | 1.000 | 0.04017 | 4351 |
| 2 | crt_s1001_m45.txt | -2.496592 | 1.073e-01 | 1.000 | 0.04017 | 4321 |
| 3 | crt_s1001_m45_1.txt | -2.517792 | 1.050e-01 | 1.000 | 0.04017 | 4269 |
| 4 | crt_s1001_m42_3.txt | -2.518615 | 1.048e-01 | 1.000 | 0.04017 | 4294 |
| 5 | crt_s1001_m42_2.txt | -2.751331 | 8.314e-02 | 1.000 | 0.04017 | 4631 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 35.00 --scan-merit 35.00 --crt-file crt/crt_s1001_m35.txt
```

## Merit 36.00

| rank | file | score | base_prob | hl | cg_ratio | candidates |
|---:|---|---:|---:|---:|---:|---:|
| 1 | crt_s1001_m45.txt | -2.566992 | 1.002e-01 | 1.000 | 0.04132 | 4464 |
| 2 | crt_s1001_m42_3.txt | -2.583752 | 9.847e-02 | 1.000 | 0.04132 | 4438 |
| 3 | crt_s1001_m45_1.txt | -2.585347 | 9.840e-02 | 1.000 | 0.04132 | 4411 |
| 4 | crt_s1001_m42_2.txt | -2.819730 | 7.784e-02 | 1.000 | 0.04132 | 4776 |

Recommended launch command:

```sh
bin/gap_miner --shift 1001 --threads 8 --cuda 0 --fast-fermat --target 36.00 --scan-merit 36.00 --crt-file crt/crt_s1001_m45.txt
```

## Summary

Median winner score: -2.185339; min: -2.566992; max: -1.855916

Winner map:

| merit | winner | score | base_prob |
|---:|---|---:|---:|
| 30.00 | crt_s1001_m45.txt | -2.185339 | 1.446e-01 |
| 31.00 | crt_s1001_m45.txt | -2.033441 | 1.368e-01 |
| 32.00 | crt_s1001_m35.txt | -1.855916 | 1.283e-01 |
| 33.00 | crt_s512_m35_p3.txt | -2.299562 | 8.585e-02 |
| 34.00 | crt_s1001_m35.txt | -2.183219 | 1.138e-01 |
| 35.00 | crt_s1001_m35.txt | -2.493901 | 1.072e-01 |
| 36.00 | crt_s1001_m45.txt | -2.566992 | 1.002e-01 |

## Direct Campaign Recommendations

1. Primary lane for merit 30,31: `crt_s1001_m45.txt` (new best pick after adding
	 the m45/m45_1/m42_2/m42_3 files to the portfolio).
2. Primary lane for merit 32,34,35: `crt_s1001_m35.txt`.
3. Crossover lane for merit 33: `crt_s512_m35_p3.txt`.
4. Merit 36 now has an eligible file: `crt_s1001_m45.txt` (previously no file
	 in the portfolio reached merit 36; the old "Merit 45.00 (Extended Example)"
	 placeholder section is now folded into the real Merit 36.00 table above).

For concrete starting profiles and CLI examples (CRT and non-CRT), see
`docs/HIGH_MERIT_WEIGHT_EXAMPLES_2026-08-05.md`.

## How To Reproduce / Retune

```sh
/usr/bin/python3 scripts/high_merit_crt_selector.py \
	--merit-min 30 --merit-max 36 --merit-step 1 \
	--weight-hl 0.65 --weight-cg 2.20 \
	--weight-candidates 0.01 --weight-primes 0.02 \
	--top 5 --out docs/HIGH_MERIT_CAMPAIGN_REPORT_2026-08-05.md
```

Recommended A/B tuning order for this model:

1. Increase `--weight-cg` if high-shift picks look too optimistic.
2. Increase `--weight-candidates` when CPU/GPU queue pressure dominates wall
	 time.
3. Increase `--weight-hl` only after verifying no regressions in accepted
	 quality (avoid overfitting to divisibility structure alone).

## Sources Surveyed For Theory Context

- Prime gap overview and merit context (Wikipedia):
	`https://en.wikipedia.org/wiki/Prime_gap`
- Cramer model and Granville caveat (Wikipedia):
	`https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_conjecture`
- Prime k-tuple admissibility and HL context (Wikipedia):
	`https://en.wikipedia.org/wiki/Prime_k-tuple`
- Bateman-Horn density constant form (Wikipedia):
	`https://en.wikipedia.org/wiki/Bateman%E2%80%93Horn_conjecture`
- Cramer-Granville statement (MathWorld):
	`https://mathworld.wolfram.com/Cramer-GranvilleConjecture.html`
- Polymath bounded-gap sieve context:
	`https://arxiv.org/abs/1407.4897`
