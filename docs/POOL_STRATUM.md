# Gapcoin pool mode (legacy stratum)

`gapminer` can mine on a Gapcoin pool instead of a local `gapcoind`:

```bash
bin/gapminer --stratum stratum+tcp://gap.suprnova.cc:2434 \
             --stratum-user <worker> --stratum-auth-file ~/.gapminer-pool \
             --crt-file data/crt/m23/shift509_p74_covermax_m38.txt \
             --threads 4 --enable-gpu-fermat --enable-submission
```

In this mode **no node is contacted at all**: the pool supplies the work, and
solutions are returned as `mining.submit` PoW payloads. There is no coinbase,
no block assembly and no `submitblock` — see "Payload" below for why.

## Which port, which protocol

| port | protocol | usable by us |
|---|---|---|
| 2434 | legacy Gapcoin "getwork over JSON" | **yes** — this is what `--stratum` speaks |
| 2433 | suprnova's "new stratum" | no — a private dialect; their own closed miners only ("old miners can connect on port 2434") |

The protocol is the one the official Gapcoin miners use. It is **not** Bitcoin
stratum v1: there is no `mining.subscribe`, no `mining.authorize`, and no
extranonce1/2.

## Wire format

Newline-delimited JSON-RPC over one TCP connection:

| direction | message |
|---|---|
| miner -> pool | `{"id":N,"method":"mining.request","params":["user","pass"]}` |
| pool -> miner | `{"id":N,"result":{"data":"<160 hex>","difficulty":<ndiff>}}` |
| pool -> miner | `{"id":null,"method":"blockchain.block.new","params":{"data":"<160 hex>","difficulty":<ndiff>}}` (documented name; **suprnova does not use it**) |
| pool -> miner | `{"id":null,"method":"mining.notify","params":{"data":"<160 hex>","difficulty":<ndiff>}}` (**what suprnova actually sends** for new work) |
| pool -> miner | `{"id":null,"method":"mining.set_difficulty","params":[<ndiff>]}` (also `[share,net]` or `{"difficulty":N}`) |
| miner -> pool | `{"id":N,"method":"mining.submit","params":["user","pass","<hex>"]}` |
| pool -> miner | `{"id":N,"result":true\|false}` (optionally with `"error":{"code":..,"message":..}`) |

Every `data` field is the **80-byte header prefix**, not a block:

```
version(4) | prevhash(32) | merkleroot(32) | time(4) | nDifficulty(8, LE at 72..79)
```

The 4-byte `nNonce` is **not** in it: the miner picks it and appends it, so the
hashed header is `SHA256d(hdr80 || nonce)` over **84** bytes, and it must have
its top bit set (Gapcoin requires `mpz_sizeinbase(hash,2) == 256`). That is
exactly the sequence `gapcoin_gbt_work_hash()` already implements, which is why
pool work reuses the node path's search machinery unchanged.

## Two difficulties, and they are different things

| value | meaning | used for |
|---|---|---|
| `difficulty` in the JSON | the pool's **share target** (ndiff; merit = `ndiff / 2^48`) | the miner's merit threshold |
| hdr80[72..79] | the **network** nDifficulty of the block the pool is solving | the "(block)" classification / display |

Verified live (2026-09-19, gap.suprnova.cc): the pool handed out
`share = 15.7726 merit` while the header carried `network = 23.7473 merit`, and
the value decoded from bytes 72..79 matched the chain's live difficulty to 4
decimals — independent confirmation of the header layout.

## How new work arrives, and the trap in it

The pool invalidates work by **pushing** a replacement template. On suprnova the
pushed method is **`mining.notify`**, not the documented
`blockchain.block.new`, and its `params` is the same object
(`{data, difficulty}`). The reference client therefore accepts work from *any*
push carrying such an object, and so does this miner: a push that is not
`mining.set_difficulty`/`mining.target` and whose `params` is a work object (or a
one-element array containing one) is treated as new work, and adopting it logs
`new work: ... (pool rotated its template)`.

This is not a stylistic choice. Filtering on the method name fails **silently**:
the miner keeps hashing an abandoned header, and the pool answers every later
share with `result:false` and **no error message**, forever, while the share
target and the network difficulty in the header stay unchanged. Measured on
2026-09-19 (shift509 CRT, 2 threads, real pool): the first `mining.notify` push
arrived ~2 minutes in, and from that moment every share was rejected — 108
accepted then 202 rejected in a row (a 5-minute `STRATUM_DEBUG=1` capture showed
**6 ignored `mining.notify` pushes** and 212/39 accepted/rejected).

Polling cannot substitute for the push. `mining.request` on this pool keeps
answering with the template that was current when the session started: in that
same capture, every one of the ~30 work requests returned one single `data`
value, while the pushes carried a different, newer one. New TCP sessions start
from the cached template too, which is why a fresh session accepts shares at
first and then decays.

To watch the pushes yourself:

```bash
STRATUM_DEBUG=1 ./bin/gapminer --stratum ... 2>&1 | grep -E 'push method|new work'
```

## Payload: the PoW solution, not a block

```
hdr80(80) + nNonce(4, LE) + nShift(2, LE) + nAdd(LE, >= 1 byte)     > 86 bytes
```

This is the same shape `gapcoind`'s legacy getwork submit expects. It is
**not** a serialized block: the pool holds the template, and its merkle root is
already inside the 80 bytes it handed us, so the miner could not build the block
even if it wanted to. `nShift` (2 bytes) is chosen by the miner — our CRT cover's
shift (509) fits, and the pool validates whatever we send.

## "Do we have to tell it what is a share and what is a block?"

**No — there is no such flag, and both cases use the identical envelope.** The
pool alone classifies the solution by merit:

* merit >= its share target -> a valid **share** (hashrate credit),
* merit >= network difficulty -> additionally a **block** for the pool to submit.

The consequence is on our side, not in the protocol: *we* choose the threshold,
and mining only at network difficulty would leave the pool with almost no share
traffic (see the round accounting on the pool's dashboard). So `--stratum`
defaults to the pool's share target and prints it; `--merit <m>` overrides it
(useful to mine only block-level solutions, or to pin a threshold the pool
cannot move).

## Verification

Two independent checks, both reproducible:

1. **Protocol conformance (no network needed).**
   `bin/test_stratum` drives the client against an in-process mock pool and
   validates every payload byte layout (header, little-endian nonce/shift/nAdd,
   the >86-byte floor, the zero-nAdd case), the work handoff, the
   `mining.set_difficulty` / `mining.notify` / unknown-method work pushes, accept
   and reject verdicts, local duplicate suppression, the verdict callback with
   gap metadata, and reconnect with in-flight share accounting. 53 checks, all
   pass, deterministic.
   `scripts/mock_stratum_pool.py` is the same idea as a standalone server, for
   testing `main`'s pool mode end to end.
2. **Live pool.** 90 s on `gap.suprnova.cc:2434` with a real worker:
   **88 shares queued, 88 accepted, 0 rejected, 0 duplicates, 0 send failures**,
   no pool errors, no unrecognised-method complaints. Rejections would have
   arrived with the pool's own error text.
3. **Live pool, full length, after the push fix.** The same 360 s run that
   produced 108 accepted then 228 rejected before the fix produced
   **308 queued / 308 accepted / 0 rejected / 0 duplicates / 0 unresolved**,
   adopting **40** template rotations (`new work: ... (pool rotated its
   template)`) along the way - i.e. the pool rotates its template roughly every
   9 s and every one of those rotations used to be discarded.

## Credentials

The password is deliberately **not** a command-line option (it would be visible
in `ps` to every user on the box). Order of resolution:

1. `--stratum-auth-file <path>` — line 1 = worker, line 2 = password. The file
   is checked for group/other readability and a warning is printed if it is not
   `chmod 600`.
2. `GAPMINER_STRATUM_USER` / `GAPMINER_STRATUM_PASS` environment variables.
3. No password available -> `x`, which pools accept for wallet/anonymous logins
   (the worker name may also be the wallet address).

## Implementation map

| file | role |
|---|---|
| `new_src/stratum.{h,c}` | protocol client: connect/reconnect with backoff, line framing, work publication, submit queue with local dedup, verdict callback, counters |
| `new_src/main.c` | pool mode: endpoint/auth parsing, work materialisation, share submission, pool stats lines, verdict -> record log |
| `tests/test_stratum.c` | mock-pool conformance suite |
| `scripts/mock_stratum_pool.py` | standalone mock pool for end-to-end runs |

Record log in pool mode: the worker writes `status=queued` when it enqueues a
BPSW-verified gap, and the pool's verdict adds `status=accepted`, `rejected`, or
`unresolved` (the connection dropped with the share in flight — this is *not* a
rejection and is never counted as one). `duplicate`/`send-failed` are written
when the client refuses to queue a share.

**Pool runs log to their own file: `gapminer_pool_records.log`**, and node runs
keep `gapminer_records.log`; `--record-log <path>` overrides both. The split is
not cosmetic: a pool share and a node-accepted gap are different objects — the
verdict comes from the pool, the threshold is the pool's share target rather
than the network difficulty, and `height` is always 0 — and the line format has
no field naming the work source, so appending both into one file would change
what that file means for anyone reading it later. The choice is made once at
startup and printed:

```
[RecordLog] Logging BPSW candidates to gapminer_pool_records.log
```

To see what that file is actually saying — how many candidates per hour, at
which merits, how the pool ruled on each, and how close they came to a known
record — use `scripts/records_report.py` (it reads the pool log and the node log
together, keeps them apart, and can plot an 8-panel PNG):

```bash
scripts/records_report.py --source pool --plot
```

The plot path may point into a directory that does not exist yet — it is created
for you (a failed write is reported as one line, never a traceback after the
report has already been printed).

## Known limits

* Port 2433 ("new stratum") is not implemented: it is suprnova's private
  dialect, and implementing it would mean reverse-engineering their miner.
* The legacy protocol carries **no block height**, so pool-mode record-log
  entries carry `height=0`.
* `--crt-file`/`--shift`, sieve depth and the GPU knobs behave exactly as in
  node mode; the pool's header replaces the GBT template 1:1.
* Pool mode ignores `--coinbase-script-hex`: the pool owns the coinbase, and
  payouts follow the pool account, not a local script.
