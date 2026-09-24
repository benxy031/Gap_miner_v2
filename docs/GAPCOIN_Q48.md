# Gapcoin Q48 proof-of-work arithmetic (and the certificates built on it)

Everything here is about one question a miner must be able to answer exactly:

> Would the node accept this gap, for this block header?

`new_src/pow_q48.{h,c}` answers it bit-exactly, `tests/test_pow_q48.c` proves it
against the node's own code, and this document records where the values come
from and what was measured about their effect on this miner.

## 1. The node's rule

`Gapcoin/src/PoWCore/PoW.h`:

```cpp
bool valid() { return difficulty() >= target_difficulty; }
```

`PoWUtils.cpp` defines the three pieces, all in **Q48 fixed point** (a `uint64_t`
holding the value times $2^{48}$):

| quantity | definition | note |
|---|---|---|
| `merit(start,end)` | $\left\lfloor \mathrm{gap} \cdot \log_2(e) \cdot 2^{112} / \lfloor \log_2(\mathrm{start}) \cdot 2^{64} \rfloor \right\rfloor \bmod 2^{64}$ | i.e. $\mathrm{gap} \cdot 2^{48}/\ln(\mathrm{start})$ truncated |
| `rand(start,end)` | XOR of the four little-endian `uint64` words of `SHA256d(LE(start) ‖ LE(end))` | minimal-length LE byte arrays |
| `m2(start)` | $\lfloor 2 \cdot \log_2(e) \cdot 2^{112} / \lfloor \log_2(\mathrm{start}) \cdot 2^{64} \rfloor \rfloor \bmod 2^{64}$, never 0 | Q48 merit of a gap of size 2 |
| `difficulty(start,end)` | `merit + (rand mod m2)` | the value the node compares |
| `target_size(start,t)` | $\lfloor t \cdot \lfloor \log_2(\mathrm{start}) \cdot 2^{64} \rfloor / (\log_2(e) \cdot 2^{112}) \rfloor$ | $\approx t \cdot \ln(\mathrm{start})$ |

The constants are `log2(e)·2^112 = 0x171547652b82fe1777d0ffda0d23a` and
`log2(e)·2^64 = 0x171547652b82fe177`, both loaded by `PoWUtils::PoWUtils()`. The
custom `mpz_log2()` (a square-and-shift binary logarithm with 64 fractional bits)
is ported statement by statement, because its truncation is part of the result.

`target_difficulty` is the template's `nDifficulty`, written into the 80-byte
header at **offset 72..79, little endian** — the same bytes `PoWUtils` reads back
when the node validates a block. `pow_q48_target_from_header()` reads them from
the header that is about to be submitted, so the check can never drift from the
template it is supposed to describe.

## 2. Why "merit ≥ difficulty" is not the same test

`rand() mod m2` is an **addition of up to `m2`**, and `m2 = 2/ln(start)` in
readable units (≈0.011 at 256 bits, ≈0.002-0.005 at the widths the miner uses).
So the node accepts gaps whose *merit* is below the target by up to `m2`, and it
rejects gaps whose merit is above the target only if `rand()` came out unlucky at
the very boundary. Node-generated vectors in `tests/data/pow_q48_vectors.txt`
(difficulty 22 target, 512-bit start):

| gap | merit | difficulty | node verdict |
|---|---|---|---|
| 3888 | 21.9425 | 21.9470 | reject |
| 3890 | 21.9981 | **22.0073** | **accept** (merit below target, `rand` lifted it) |
| 7792 | 21.9676 | 21.9701 | reject |
| 7794 | 21.9950 | 21.9989 | reject |

A miner that gates on `merit >= threshold` therefore loses blocks, and — worse —
a miner that takes the threshold from `getmininginfo` (`GetDifficulty(tip)`) is
comparing against the **previous** block's target rather than the one its header
carries; when a template raises the target, every queued gap comes back as
`high-hash`. This is why `main` uses the template's `nDifficulty` as the
threshold and re-checks every candidate against the header bytes it is about to
submit (worker log line `node_diff=…`, counter `below_target=`, record-log
outcome `below-target`).

## 3. Certified interval rejection

For a fixed start, an accept needs `merit(g) + (rand mod m2) ≥ target`, and
`rand mod m2 ≤ m2 - 1`, so

```
merit(g) >= target - m2 + 1
=> g * log2(e) * 2^112 >= (target - m2 + 1) * floor(log2(start)*2^64)
=> g >= ceil( (target - m2 + 1) * floor(log2(start)*2^64) / (log2(e)*2^112) )
```

`pow_q48_reject_bound()` returns that integer. **Every gap strictly shorter than
it is provably below target**, without evaluating `rand()` — useful because
evaluating `rand()` needs `SHA256d` over the endpoints, while the bound is
computed once per window and then compared against integer spans.

Two uses in the miner, both fail-open:

- **Covered-region resolve** (`worker_gpu.c`, terminal pair of the HALF_CLASS
  chain): every sub-gap inside a span is shorter than the span, so a span below
  the bound proves that no hidden interior prime can produce a qualifying gap →
  the host-side interior mini-sieve + BPSW resolve is skipped. Counted as
  `Q48 certificate: N covered regions proven below target`. The decision is
  *looser* than the old `merit >= threshold` test (it also admits the `m2`-wide
  boundary band, which is exactly where `rand` can rescue a gap) and it is
  provably sound: it never skips a span the node would accept.
- **Submit gate**: the exact verdict (ACCEPT / REJECT / UNKNOWN) for the
  candidate, from the same arithmetic the node runs.

`POW_Q48_UNKNOWN` (no target known, degenerate input) always means "examine":
the module never converts missing information into a rejection.

### Measured scope (honest accounting)

- The **threshold source** fix removes wasted `submitblock` calls entirely when a
  template's `nDifficulty` differs from the tip's difficulty (the failure mode
  reported from live mining: `submitblock rejected: high-hash`).
- The **exact gate** additionally rescues the `m2`-wide band below the target
  (2 of 92 node vectors are exactly this case). In block terms that band is a
  fraction ~`m2` of merit wide, i.e. well under 1% of near-threshold finds: a
  correctness gain, **not** a throughput lever.
- The **certificate** does not shorten the common path either: it fires only on
  the terminal covered span, and the honest conclusion from measuring the
  existing code is that our smart-scan tail-skip was already an exact bracket
  argument (a prime inside `[needed_gap, gap_target)` closes every owned gap), so
  there was little work left to certify. Its value is that the skip decision is
  now provably sound instead of double-rounded. The counter exists so the claim
  can be checked at runtime rather than believed.

## 4. Provenance and verification

- `tools/pow_q48_oracle.cpp` links the **node's own** `src/PoWCore/PoWUtils.cpp`
  and prints `merit`, `rand`, `difficulty`, `target_size` and the node's
  `valid()` verdict for a deterministic case list (fixed GMP Mersenne-Twister
  seeds, so regeneration is byte-reproducible).
- `scripts/gen_pow_q48_vectors.sh` builds that oracle against a Gapcoin source
  tree and writes `tests/data/pow_q48_vectors.txt` (header records the source
  revision). The fixture is generated, never hand-edited.
- `tests/test_pow_q48.c` compares our implementation against every vector
  (bit-exact on all five quantities), checks `m2` against its closed form, checks
  the certificate's soundness as a property test (never rejects an acceptable
  span), and checks fail-open behaviour for unknown targets and degenerate input.

```bash
make bin/test_pow_q48 && ./bin/test_pow_q48
GAPCOIN_SRC=/path/to/Gapcoin scripts/gen_pow_q48_vectors.sh   # regenerate
```

Licensing: Gapcoin is GPL-3.0-or-later, the same license as this repository
(`LICENSE.md`), so the port and the oracle are license-compatible.

## 5. Windows / LLP64

Windows is LLP64: `unsigned long` is **32 bits**, while every GMP `mpz_*_ui()`
function takes an `unsigned long`. A Q48 difficulty is ~`24 * 2^48` (~2^52), so
any `unsigned long` round trip of a Q48 value silently truncates it (the same
hazard `new_src/win_compat.h` documents for the setters, which is why that header
already redefines `mpz_set_ui`/`mpz_add_ui`/`mpz_mul_ui`/... to 64-bit-safe
wrappers).

Rules the module follows so that the Windows build cannot silently disagree with
the node:

| Hazard | Handling |
|---|---|
| low 64 bits of an mpz | `mpz_fdiv_r_2exp` + `mpz_export` into a byte buffer, byte for byte like the node's own `mpz_export` path — **no `mpz_get_ui`**, which would return only 32 bits |
| uint64 -> mpz | `mpz_import(rop, 1, -1, sizeof(uint64_t), 0, 0, &value)` |
| compare a span against the certificate bound | `mpz_cmp` against a temporary mpz, not `mpz_cmp_ui` |
| `*_ui` setters | only with `(uint64_t)` casts, so `win_compat.h`'s wrappers see the full value (the repository-wide convention) |

One deliberate exception: `q48_mpz_log2()` keeps the node's own
`mpz_get_ui(mpz_tmp) < 2` test verbatim, because staying identical to the node is
the point of the port. That test only inspects a value the loop has just
renormalised down to a single digit, so 32-bit and 64-bit `mpz_get_ui` take the
same branch; the comment in the source says so.

Verification (and its honest limit): a build of `pow_q48.c` + `test_pow_q48.c`
with `mpz_get_ui` masked to 32 bits reproduces the Linux result byte for byte
(92/92 vectors, identical counters), which is what proves the module is
insensitive to the truncation. No MinGW toolchain was available on the
development host, so **no real Windows build of the miner was run** — the
Windows-side change is the source/test lists in `Makefile.win`
(`pow_q48` in `SOURCES`, `test_pow_q48` in `CPU_TESTS`); `pow_q48` is host-side
only, so `windows/gapgpu.def` and the CUDA DLL are untouched. Run the test from
the repository root (`bin\test_pow_q48.exe`): it reads
`tests/data/pow_q48_vectors.txt` relatively, and tolerates CRLF checkouts.
