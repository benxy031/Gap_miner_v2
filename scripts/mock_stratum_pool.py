#!/usr/bin/env python3
"""Mock Gapcoin pool speaking the LEGACY stratum protocol (suprnova port 2434).

It exists so `gapminer --stratum ...` can be tested end to end without a real
pool, an account or a password.  The protocol implemented here is the one
gap.suprnova.cc:2434 speaks (getwork over newline-delimited JSON-RPC):

    client -> {"id":N,"method":"mining.request","params":["user","pass"]}
    server -> {"id":N,"result":{"data":"<160 hex>","difficulty":<ndiff>}}
    server -> {"id":null,"method":"blockchain.block.new",
               "params":{"data":"<160 hex>","difficulty":<ndiff>}}
    client -> {"id":N,"method":"mining.submit","params":["user","pass","<hex>"]}
    server -> {"id":N,"result":true|false}

The submitted hex is the PoW SOLUTION, not a block:
`hdr80(80) + nNonce(4,LE) + nShift(2,LE) + nAdd(LE, >=1 byte)` (> 86 bytes).
This script validates every payload against the header it handed out and prints
a summary, which makes it a conformance test for the miner's pool mode.

Usage:
    scripts/mock_stratum_pool.py --port 24340 --share-merit 12
    bin/gapminer --stratum 127.0.0.1:24340 --stratum-user test \
        --crt-file data/crt/m23/shift509_p74_covermax_m38.txt \
        --threads 2 --enable-gpu-fermat --enable-submission
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time

NFIX = 1 << 48
HDR_HEX = 160


def merit_to_ndiff(merit: float) -> int:
    return int(merit * NFIX)


def craft_header(seed: int, net_ndiff: int) -> bytes:
    """80-byte Gapcoin header prefix: version|prevhash|merkle|time|nDifficulty."""
    hdr = bytearray()
    hdr += struct.pack("<I", 1)
    hdr += bytes((seed + i) & 0xFF for i in range(32))
    hdr += bytes((seed * 7 + i * 3) & 0xFF for i in range(32))
    hdr += struct.pack("<I", int(time.time()) & 0xFFFFFFFF)
    hdr += struct.pack("<Q", net_ndiff)          # bytes 72..79, little-endian
    assert len(hdr) == 80
    return bytes(hdr)


class Pool:
    def __init__(self, share_ndiff: int, net_ndiff: int, pushes: bool,
                 push_interval: float, reject_every: int) -> None:
        self.lock = threading.Lock()
        self.share_ndiff = share_ndiff
        self.net_ndiff = net_ndiff
        self.pushes = pushes
        self.push_interval = push_interval
        self.reject_every = reject_every
        self.seed = 1
        self.hdr = craft_header(self.seed, net_ndiff)
        self.submits = 0
        self.accepted = 0
        self.invalid = []
        self.connections = 0

    def new_header(self) -> bytes:
        with self.lock:
            self.seed += 1
            self.hdr = craft_header(self.seed, self.net_ndiff)
            return self.hdr

    def current_header(self) -> bytes:
        with self.lock:
            return self.hdr

    def validate(self, params: list, known: bytes) -> bool:
        if not isinstance(params, list) or len(params) != 3:
            return False
        payload = params[2]
        if not isinstance(payload, str):
            return False
        if len(payload) < 174:                     # 87 bytes
            self.invalid.append(f"too short ({len(payload)} hex chars)")
            return False
        if payload[:HDR_HEX] != known.hex():
            self.invalid.append("header does not match the issued one")
            return False
        rest = payload[HDR_HEX:]
        if len(rest) < 12:                         # nonce(4) + shift(2) + nadd(>=1)
            self.invalid.append("missing nonce/shift/nadd")
            return False
        nonce = struct.unpack("<I", bytes.fromhex(rest[:8]))[0]
        shift = struct.unpack("<H", bytes.fromhex(rest[8:12]))[0]
        if shift == 0 or not (1 <= shift <= 4096):
            self.invalid.append(f"implausible shift {shift}")
            return False
        self.last_nonce, self.last_shift = nonce, shift
        return True

    def handle(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = b""
        last_push = time.time()
        while True:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                chunk = None
            except OSError:
                return

            if chunk:
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.dispatch(conn, line.decode("utf-8", "replace").strip())
            elif chunk == b"":
                return

            now = time.time()
            if self.pushes and now - last_push >= self.push_interval:
                last_push = now
                hdr = self.new_header()
                msg = {"id": None, "method": "blockchain.block.new",
                       "params": {"data": hdr.hex(), "difficulty": self.share_ndiff}}
                conn.sendall((json.dumps(msg) + "\n").encode())
                print(f"[mock] pushed new work seed={self.seed}", flush=True)

    def dispatch(self, conn: socket.socket, line: str) -> None:
        if not line:
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            print(f"[mock] bad JSON from miner: {line[:80]}", flush=True)
            return
        method = req.get("method")
        rid = req.get("id", 0)

        if method == "mining.request":
            known = self.current_header()
            msg = {"id": rid,
                   "result": {"data": known.hex(), "difficulty": self.share_ndiff}}
            conn.sendall((json.dumps(msg) + "\n").encode())
            print(f"[mock] work sent to '{req.get('params', ['?'])[0]}' "
                  f"(share {self.share_ndiff / NFIX:.4f} merit)", flush=True)
        elif method == "mining.submit":
            with self.lock:
                self.submits += 1
                n = self.submits
                ok = self.validate(req.get("params"), self.hdr)
                if ok and self.reject_every and n % self.reject_every == 0:
                    ok = False                      # exercise the reject path
                if ok:
                    self.accepted += 1
                nonce = getattr(self, "last_nonce", 0)
                shift = getattr(self, "last_shift", 0)
                payload = req.get("params", ["", "", ""])[2]
            print(f"[mock] submit #{n}: payload={len(payload) // 2} B "
                  f"nonce=0x{nonce:08x} shift={shift} "
                  f"-> {'ACCEPTED' if ok else 'REJECTED'}", flush=True)
            resp = {"id": rid, "result": bool(ok)}
            if not ok:
                resp["error"] = {"code": 20, "message": "invalid share"}
            conn.sendall((json.dumps(resp) + "\n").encode())
        elif method is not None:
            print(f"[mock] ignoring method {method}", flush=True)

    def serve(self, listen: socket.socket) -> None:
        while True:
            try:
                conn, addr = listen.accept()
            except OSError:
                return
            with self.lock:
                self.connections += 1
                c = self.connections
            print(f"[mock] connection #{c} from {addr[0]}:{addr[1]}", flush=True)
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock Gapcoin legacy stratum pool")
    ap.add_argument("--port", type=int, default=24340)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share-merit", type=float, default=12.0,
                    help="share target handed to the miner (default 12)")
    ap.add_argument("--net-merit", type=float, default=23.8,
                    help="network difficulty written into the header (default 23.8)")
    ap.add_argument("--push-interval", type=float, default=0.0,
                    help="seconds between blockchain.block.new pushes (0 = never)")
    ap.add_argument("--reject-every", type=int, default=0,
                    help="reject every Nth valid share (0 = never)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="exit after N seconds (0 = run until killed)")
    args = ap.parse_args()

    pool = Pool(merit_to_ndiff(args.share_merit), merit_to_ndiff(args.net_merit),
                args.push_interval > 0, args.push_interval, args.reject_every)

    listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen.bind((args.host, args.port))
    listen.listen(4)
    print(f"[mock] listening on {args.host}:{args.port} | share target "
          f"{args.share_merit} merit | network {args.net_merit} merit",
          flush=True)

    if args.seconds > 0:
        threading.Timer(args.seconds, listen.close).start()

    try:
        pool.serve(listen)
    except KeyboardInterrupt:
        pass
    finally:
        listen.close()

    print(f"[mock] SUMMARY: connections={pool.connections} submits={pool.submits} "
          f"accepted={pool.accepted} invalid={len(pool.invalid)}", flush=True)
    for reason in pool.invalid[:5]:
        print(f"[mock]   invalid: {reason}", flush=True)
    return 0 if pool.invalid == [] else 1


if __name__ == "__main__":
    sys.exit(main())
