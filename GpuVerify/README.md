# GpuVerify

Standalone GPU-accelerated verifier for prime gaps. **Fully independent of
the gapminer and phase0 packages**: it has its own build, its own sources,
and depends only on CUDA and GMP — no includes from `new_src/`, `phase0/`,
or `tools/`.

It re-checks gap records of the form stored in `allgaps.sql`
(`10^x ± y`, `m*p#/d ± a`, `p# ± a`, plain digits, ...): both endpoints
must be prime and no prime may lie strictly between them.

## Why it exists

The prime-gap-list DB is only lightly checked on import: its `check.py`
endpoint-tests only rows `< 400 digits` (trial division only above that),
never tests consecutiveness, and its regex cannot even parse 714 rows
(see [primegap-list-project/prime-gap-list#38](https://github.com/primegap-list-project/prime-gap-list/issues/38)).
GpuVerify does the full check, batched on the GPU:

* **endpoints**: strong probable-prime base 2 on the GPU, then GMP's BPSW
  (strong base-2 + Lucas) on the host for every GPU passer;
* **interval**: GPU sieve of the gap interval, then the same PRP pipeline
  for every survivor.

Verdict semantics:

| verdict | meaning |
| --- | --- |
| `VALID` | both endpoints BPSW-prime, no prime between (GPU interval scan) |
| `INVALID (endpoint composite)` | proven composite (base-2 witness or Lucas) |
| `INVALID (prime between)` | a prime strictly inside the gap (witness printed) |

A failing strong base-2 test is a proof of compositeness; a passing one is
confirmed with GMP 6.3's `mpz_probab_prime_p(n, 1)`, which includes the
strong Lucas test (BPSW). Prime verdicts are therefore BPSW probable
primes — the same standard used for the Payne-row audit.

## Build

```
cd GpuVerify
make            # -> bin/gpuverify, RTX 3070 (sm_86); override with ARCH=sm_XX
make test       # selftest (no DB needed)
make test-parser
```

Requirements: CUDA toolkit (nvcc 12.x), GMP dev (`libgmp-dev`), Linux.

## Usage

```sh
# one gap: endpoints + interval scan
./bin/gpuverify verify --expr "10^19662 - 58727" --gap 161450 --pmax 1000000

# a primorial-form row
./bin/gpuverify verify --expr "19307#/390637 - 1" --gap 206048

# batch over the DB (endpoints only; add --with-interval for full check)
./bin/gpuverify sweep --db allgaps.sql --min-digits 400 --max-digits 9864 \
    --out sweep.log

# throughput measurement
./bin/gpuverify bench --bits 4096 --count 5000
```

Flags (see also `--help`):

| flag | default | meaning |
| --- | --- | --- |
| `--expr E` | – | start expression (verify) |
| `--gap G` | – | gap size (verify) |
| `--db FILE` | – | allgaps.sql for sweep |
| `--min-digits` / `--max-digits` | 0 / 0 | row filter |
| `--with-interval` | off | also sieve + PRP the gap interval in sweep |
| `--pmax P` | 1048576 | largest sieving prime for interval scans |
| `--limit K` / `--skip K` | 0 / 0 | process rows [skip, skip+limit) |
| `--cpu` | off | skip the GPU, BPSW-only |
| `--cpu-max-digits D` | 200000 | CPU rows above D digits are skipped |
| `--out LOG` | stdout | sweep failure log |

## Design

```
src/
  parse.c          start-expression parser + quote-aware DB row extractor
                   (parses 100% of allgaps.sql; check.py manages 123,188/123,902)
  mont.cuh         CIOS Montgomery mul/sqr on 32-bit limbs (HAC 14.36)
  prp_kernel.cu    batched strong PRP base 2, one candidate per thread
  sieve_kernel.cu  interval sieve (host precomputes N mod p per prime)
  host.cu          mont constants via GMP, batch dispatch, prime list
  main.c           CLI: verify / sweep / selftest / bench
tests/
  test_mont.cu     GPU-vs-GMP cross-check (7,850 random cases, all sizes)
  test_parse.c     full-DB parse check (123,902/123,902, digits verified)
data/bench_results.txt  measured throughput and sweep projection
```

* Numbers up to 32768 bits (≈9,864 digits) run on the GPU in limb buckets
  (16..1024 limbs, zero-padded); larger numbers fall back to CPU BPSW.
* Montgomery constants (`R`, `mont(2)`, `mont(N-1)`, `mont(1)`, `np0`) are
  derived with GMP per candidate; the kernel reads the exponent bits of
  `d = (N-1)/2^s` straight from the modulus.
* The interval sieve marks even offsets `k` (`N+k` odd) with small primes;
  the host precomputes each prime's first hit so the kernel never sees big
  integers, and skips the degenerate `N+k == p` mark (the candidate would
  be the prime p itself).

## Measured (RTX 3070, idle GPU, v1 kernel)

```
  bits   PRP/s       256: 87074   512: 79588  1024: 17460
                      2048: 1388   4096: 82    8192: 3
                      16384: <0.15 (scaling collapses, see below)
```

Projected full-DB endpoint sweep (94,911 rows, 400..9,864 digits):
**≈ 2.5 h** on an idle GPU. The 7,000–9,864-digit bucket dominates
(135 of 149 min). See `data/bench_results.txt` for the per-bucket table.

## Known limits / next steps

* **v1 kernel collapses ≥ 8,192 bits** (one thread per candidate, local
  memory). The highest-leverage optimization is a TPI-style
  warp-per-candidate CIOS like the miner's `tools/mr128_kernel.cuh`
  (expected 10–100× in the top bucket).
* Rows > 9,864 digits (16,110 of them) are CPU BPSW only; interval scans
  are skipped there (survivor PRPs would take days). `--cpu-max-digits`
  guards the tail (largest row: 385,713 digits).
* The sweep is single-GPU, single-stream; multi-GPU would scale linearly.

## Validation log

* CIOS layer: 7,850 random `a*b*R^-1 mod n` cases vs GMP at 16..1024
  limbs — 0 mismatches.
* PRP kernel vs CPU reference on primes (2, 3, 89, 523, 1e9+7, 2^61-1) and
  composites (2047..341550071728321 strong pseudoprime list) — all correct,
  GPU passers correctly re-classified by BPSW.
* The 4 invalid Payne rows reproduce: 161450, 191356, 263402 →
  endpoint composite; 226058 → prime between (witness `10^16104 + 28021`
  re-verified prime); 206048 primorial form → VALID.
* The `m*p# ± a` rows that `check.py` cannot parse (85 small rows) verify
  VALID, matching the earlier GMP sweep.
