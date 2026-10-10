#!/usr/bin/env python3
"""p0_walk_chain.py - run phase0 walk slices back-to-back (start += length).

Slice k:
    bin/phase0_scan_gpu --engine walk --start S --length L --gap-min G
        --walk-batch B --state data/p0_state_<tag>_c<k>.txt
        --state-every 30 --progress 60 --log data/p0_walk_<tag>_c<k>_g<G>.log
    (stdout+stderr appended to data/p0_walk_<tag>_c<k>_g<G>.out)

Multiple GPUs: `--devices 0,1` runs one scan PER GPU, each taking the next
slice from a shared queue (slice files stay `p0_walk_<tag>_c<k>_g<G>.*`; the
ledger records the device).  Slices already on disk are detected at start:
completed ones are skipped, unfinished ones (e.g. after a kill) are requeued
first - so the same command restarts the whole continuum.  A slice that does
not complete drains the chain (no new launches) and exits 1; rerunning the
same command requeues/resumes it.  A lock file prevents a second chain on
the same tag, and starting while `p0_walk_<tag>_c*` scanners are alive is
refused (kill them first - their state files make the restart resumable).

Chain state `data/p0_walk_<tag>_chain.state` (start0/index0, fixed for the
continuum) and a one-line-per-slice ledger in `data/p0_walk_<tag>_chain.log`.

Before every slice it re-derives the shortest length that would still beat
the record table at that start (`g/ln(start) > table(g)`) and warns in BOTH
directions: `--gap-min` below it only adds non-record rows; `--gap-min`
above it MISSES winnable record chances in the skipped band.

Usage (one GPU / two GPUs, detached):
    nohup setsid python3 scripts/p0_walk_chain.py \
        --start 133001070000720000000 --length 1070000000000000 \
        --gap-min 1586 --tag 1p33e20 --index-start 3 \
        > data/p0_walk_1p33e20_chain.out 2>&1 &
    # two GPUs: add  --devices 0,1

Options: --slices N (0 = forever), --walk-batch B (default 64),
--devices D[,D...] (default 0), --bin PATH, --state-dir DIR, --table FILE,
--dry-run (print the first slice commands and exit).
"""
import argparse
import collections
import datetime
import math
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def shortest_winnable(L, table):
    """Shortest even g with g/L > best-known table merit for that g."""
    best = {}
    with open(table) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                g = int(p[0])
                t = float(p[1])
            except ValueError:
                continue
            if g not in best or t > best[g]:
                best[g] = t
    for g in sorted(g for g in best if 1000 <= g <= 9990 and g % 2 == 0):
        if g / L > best[g]:
            return g, best[g]
    return None, None


def forecast_lines(tag, gmin, state_dir, table, r_total, top=3, rate=2.254e11):
    """The p0_record_forecast summary for this chain, or a one-line apology.

    Never raises: the chain must keep scanning even when the forecast cannot be
    computed (missing table, unreadable log, ...).
    """
    try:
        import p0_record_forecast as fc
        an = fc.analyze([os.path.join(state_dir, f"p0_walk_{tag}*_g{gmin}.log")],
                        table_path=table, r_total=r_total)
        return fc.compact(an, top=top, rate=rate)
    except (Exception, SystemExit) as exc:
        # SystemExit included on purpose: a library that calls sys.exit() must
        # never be able to take the scanning chain down with it.
        return [f"[chain] forecast: unavailable ({type(exc).__name__}: {exc})"]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first slice start (decimal)")
    ap.add_argument("--length", required=True, help="slice length (decimal)")
    ap.add_argument("--gap-min", type=int, required=True)
    ap.add_argument("--tag", default="1p33e20")
    ap.add_argument("--index-start", type=int, default=1)
    ap.add_argument("--slices", type=int, default=0, help="0 = run forever")
    ap.add_argument("--walk-batch", type=int, default=64)
    ap.add_argument("--devices", default="0",
                    help="comma-separated CUDA device ids, one scan per device (e.g. 0,1)")
    ap.add_argument("--bin", default=os.path.join(REPO, "bin/phase0_scan_gpu"))
    ap.add_argument("--state-dir", default=os.path.join(REPO, "data"))
    ap.add_argument("--table", default=os.path.join(REPO, "data/prime_gap_merits.txt"))
    ap.add_argument("--no-forecast", action="store_true",
                    help="do not print the p0_record_forecast summary after each "
                         "completed slice")
    ap.add_argument("--forecast-top", type=int, default=3,
                    help="open targets shown in the forecast summary (default 3)")
    ap.add_argument("--rate", type=float, default=2.254e11,
                    help="scan rate in ints/s used for the forecast ETA")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if not os.path.exists(a.bin):
        print(f"[chain] FATAL: scanner not found: {a.bin}", flush=True)
        return 2
    L, G = int(a.length), a.gap_min
    devs = [int(x) for x in a.devices.replace(" ", "").split(",") if x]
    if not devs:
        devs = [0]
    if len(set(devs)) != len(devs):
        print("[chain] WARNING: duplicate --devices id; two scans on one GPU may OOM",
              flush=True)
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        nvid = len([l for l in smi.stdout.splitlines() if l.strip()])
        for d in devs:
            if d < 0 or d >= nvid:
                print(f"[chain] FATAL: device {d} not present "
                      f"(nvidia-smi -L shows {nvid})", flush=True)
                return 2
    except OSError:
        pass

    chain_state = os.path.join(a.state_dir, f"p0_walk_{a.tag}_chain.state")
    chain_ledger = os.path.join(a.state_dir, f"p0_walk_{a.tag}_chain.log")
    lock_path = os.path.join(a.state_dir, f"p0_walk_{a.tag}_chain.lock")

    # start0/index0 are fixed for the whole continuum: prefer the stored ones.
    S0, K0 = int(a.start), a.index_start
    if os.path.exists(chain_state):
        d = dict(l.split() for l in open(chain_state).read().split("\n") if l.strip())
        if "start0" in d and "index0" in d:
            S0, K0 = int(d["start0"]), int(d["index0"])
            print(f"[chain] chain state: start0={S0} index0={K0}", flush=True)

    def paths(k):
        return (os.path.join(a.state_dir, f"p0_state_{a.tag}_c{k}.txt"),
                os.path.join(a.state_dir, f"p0_walk_{a.tag}_c{k}_g{G}.log"),
                os.path.join(a.state_dir, f"p0_walk_{a.tag}_c{k}_g{G}.out"))

    def complete_at(k):
        _, _, out = paths(k)
        if not os.path.exists(out):
            return False
        txt = open(out, errors="replace").read()
        return ("range complete" in txt) and ("verification_failures=0" in txt)

    live = subprocess.run(["pgrep", "-f", f"p0_walk_{a.tag}_c"],
                          capture_output=True, text=True).stdout.split()
    if live and not a.dry_run:
        print(f"[chain] FATAL: scanners for tag={a.tag} already running (pids "
              f"{','.join(live)}); kill them first (state files make the chain "
              f"resumable), or use a different --tag", flush=True)
        return 2

    pat = re.compile(re.escape(f"p0_walk_{a.tag}_c") + r"(\d+)"
                     + re.escape(f"_g{G}.out"))
    idxs = [int(m.group(1)) for fn in os.listdir(a.state_dir)
            for m in [pat.fullmatch(fn)] if m]
    next_assign = max(idxs) + 1 if idxs else K0
    requeue = collections.deque(sorted(k for k in idxs if not complete_at(k)))
    print(f"[chain] devices={devs} next_index=c{next_assign}"
          + (f" requeue={[f'c{k}' for k in requeue]}" if requeue else ""), flush=True)

    def start_of(k):
        return S0 + (k - K0) * L

    def launch(k, dev):
        S = start_of(k)
        wg, wt = shortest_winnable(math.log(S), a.table)
        if wg is not None and wg < G:
            print(f"[chain] WARNING: --gap-min {G} is ABOVE the shortest winnable "
                  f"g={wg} (table {wt:.4f}) at start {S}: winnable record chances "
                  f"in [{wg}, {G}) are NOT reported; lower --gap-min to keep them",
                  flush=True)
        elif wg is not None and wg > G:
            print(f"[chain] WARNING: --gap-min {G} is below the shortest winnable "
                  f"g={wg} (table {wt:.4f}): lengths {G}..{wg - 2} produce "
                  f"non-record rows only", flush=True)
        state, log, out = paths(k)
        cmd = [a.bin, "--engine", "walk", "--start", str(S), "--length", str(L),
               "--gap-min", str(G), "--walk-batch", str(a.walk_batch),
               "--device", str(dev), "--state", state, "--state-every", "30",
               "--progress", "60", "--log", log]
        print(f"[chain] {now()} slice c{k} -> device {dev}: start={S} "
              f"(winnable<=g{wg}) -> {os.path.basename(out)}", flush=True)
        if a.dry_run:
            print("[chain] dry-run:", " ".join(cmd), flush=True)
            return None
        fh = open(out, "a")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        return (p, fh, datetime.datetime.now(), S)

    if a.dry_run:
        for dev, k in zip(devs, list(requeue) + [next_assign + i for i in range(len(devs))]):
            launch(k, dev)
        return 0

    with open(chain_state, "w") as fh:
        fh.write(f"start0 {S0}\nindex0 {K0}\n")
    with open(lock_path, "w") as fh:
        fh.write(f"{os.getpid()}\n")

    running = {}                 # dev -> (k, Popen, fh, t0, S)
    completed = 0
    drain = False
    rc_final = 0
    try:
        while True:
            for dev in devs:
                if dev in running or drain:
                    continue
                if a.slices and completed + len(running) >= a.slices:
                    continue
                if requeue:
                    k = requeue.popleft()
                else:
                    k = next_assign
                    next_assign += 1
                running[dev] = (k,) + launch(k, dev)
            if not running:
                break
            time.sleep(5)
            for dev in list(running):
                k, p, fh, t0, S = running[dev]
                if p.poll() is None:
                    continue
                rc = p.returncode
                fh.close()
                wall = (datetime.datetime.now() - t0).total_seconds()
                ok = complete_at(k)
                _, _, out = paths(k)
                txt = open(out, errors="replace").read()
                gm = re.findall(r"gaps reported=(\d+)", txt)
                gaps = gm[-1] if gm else "?"
                bm = re.findall(r"walk_batch=(\d+)", txt)
                bw = bm[-1] if bm else "?"
                with open(chain_ledger, "a") as lf:
                    lf.write(f"{now()}\tslice c{k}\tdev={dev}\tstart={S}\tlen={L}\t"
                             f"gap_min={G}\trc={rc}\twall={wall:.1f}s\tgaps={gaps}\t"
                             f"walk_batch={bw}\tcomplete={int(ok)}\n")
                print(f"[chain] slice c{k} (dev {dev}) finished rc={rc} "
                      f"wall={wall:.0f}s gaps={gaps} walk_batch={bw} "
                      f"complete={int(ok)}", flush=True)
                if bw != str(a.walk_batch):
                    print(f"[chain] WARNING: banner walk_batch={bw} != requested "
                          f"{a.walk_batch} (old binary clamps K>32 -> 32)", flush=True)
                del running[dev]
                if ok:
                    completed += 1
                    if not a.no_forecast:
                        for line in forecast_lines(
                                a.tag, G, a.state_dir, a.table,
                                (a.slices * L) if a.slices else None,
                                top=a.forecast_top, rate=a.rate):
                            print(line, flush=True)
                else:
                    print(f"[chain] slice c{k} did NOT complete; draining "
                          f"(no new launches). Rerun the same command to requeue "
                          f"and resume it.", flush=True)
                    drain = True
                    rc_final = 1
            if drain and not running:
                break
            if a.slices and completed >= a.slices and not running:
                break
    except KeyboardInterrupt:
        print("[chain] interrupted: terminating running scans (state files make "
              "them resumable)", flush=True)
        for (k, p, fh, t0, S) in running.values():
            p.terminate()
            fh.close()
        rc_final = 130
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass

    print(f"[chain] chain exit: completed={completed} rc={rc_final}", flush=True)
    return rc_final


if __name__ == "__main__":
    sys.exit(main())
