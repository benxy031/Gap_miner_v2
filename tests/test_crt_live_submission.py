"""Local-only regression tests for experimental CRT submission."""

import hashlib
import json
import os
import queue
import signal
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    import gmpy2
except ImportError:
    gmpy2 = None


ROOT = Path(__file__).resolve().parents[1]
MINER = ROOT / "bin/gapminer"
COVER = ROOT / "data/crt/m23/shift720_p98_lex_m30.txt"
NFIX = 1 << 48


def miner_env(mode, allow=None):
    env = os.environ.copy()
    for key in ("CRT_ROW_SELECT_ALLOW_SUBMIT", "HALF_CLASS", "QUARTER_CLASS",
                "GPU_FERMAT_NO_CGBN", "GAPMINER_STRATUM_USER",
                "GAPMINER_STRATUM_PASS"):
        env.pop(key, None)
    env.update(FUSED_GPU="1", MINING_JUMP2="1", MINING_JUMP2_BATCH="512",
               CRT_ROWS_BATCH="64", CRT_ROW_SELECT=mode)
    if allow is not None:
        env["CRT_ROW_SELECT_ALLOW_SUBMIT"] = allow
    return env


@unittest.skipUnless(MINER.is_file(), "build bin/gapminer first")
class SubmissionPolicyTests(unittest.TestCase):
    def run_config(self, mode, allow, *args):
        result = subprocess.run(
            [str(MINER), "--crt-file", "/nonexistent/gapminer-test-cover",
             "--enable-gpu-fermat", *args],
            cwd=ROOT, env=miner_env(mode, allow), capture_output=True, text=True,
            timeout=10, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        return result.stdout + result.stderr

    def test_live_requires_explicit_opt_in_for_both_modes_and_transports(self):
        for mode in ("observe", "select"):
            for pool in (False, True):
                args = ["--enable-submission"]
                if pool:
                    args += ["--stratum", "127.0.0.1:1", "--stratum-user", "test"]
                with self.subTest(mode=mode, pool=pool):
                    output = self.run_config(mode, None, *args)
                    self.assertIn("require CRT_ROW_SELECT_ALLOW_SUBMIT=1", output)

    def test_opt_in_value_is_checked_and_zero_is_not_opt_in(self):
        output = self.run_config("select", "yes")
        self.assertIn("CRT_ROW_SELECT_ALLOW_SUBMIT must be 0 or 1", output)
        output = self.run_config("select", "0", "--enable-submission")
        self.assertIn("require CRT_ROW_SELECT_ALLOW_SUBMIT=1", output)

    def test_node_live_requires_payout_script(self):
        output = self.run_config("select", "1", "--enable-submission")
        self.assertIn("require --coinbase-script-hex", output)
        output = self.run_config("select", "1", "--enable-submission",
                                 "--coinbase-script-hex", "51")
        self.assertIn("Failed to load CRT covering file", output)

    def test_pool_live_opt_in_and_pool_dry_run_pass_policy_gate(self):
        pool = ["--stratum", "127.0.0.1:1", "--stratum-user", "test"]
        for mode in ("observe", "select"):
            with self.subTest(mode=mode, live=True):
                output = self.run_config(mode, "1", *pool, "--enable-submission")
                self.assertIn("Failed to load CRT covering file", output)
            with self.subTest(mode=mode, live=False):
                output = self.run_config(mode, None, *pool)
                self.assertIn("Failed to load CRT covering file", output)

    def test_default_miner_is_not_subject_to_experimental_opt_in(self):
        output = self.run_config("0", None, "--enable-submission")
        self.assertIn("Failed to load CRT covering file", output)
        self.assertNotIn("CRT_ROW_SELECT_ALLOW_SUBMIT", output)


def sha256d(data):
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def compact_size(data, offset):
    first = data[offset]
    if first < 0xfd:
        return first, offset + 1
    size = {0xfd: 2, 0xfe: 4, 0xff: 8}[first]
    return int.from_bytes(data[offset + 1:offset + 1 + size], "little"), offset + 1 + size


def parse_solution(raw, node):
    nonce = int.from_bytes(raw[80:84], "little")
    shift = int.from_bytes(raw[84:86], "little")
    if node:
        count, pos = compact_size(raw, 86)
        adder = raw[pos:pos + count]
        tx_count, tx_pos = compact_size(raw, pos + count)
        if tx_count != 1 or sha256d(raw[tx_pos:]) != raw[36:68]:
            raise AssertionError("mock block coinbase does not match the merkle root")
    else:
        adder = raw[86:]
    if not adder:
        raise AssertionError("empty nAdd")
    value = int.from_bytes(adder, "little")
    digest = sha256d(raw[:84])
    if not (digest[31] & 0x80) or value >= (1 << shift):
        raise AssertionError("invalid Gapcoin hash or nAdd range")
    start = (int.from_bytes(digest, "little") << shift) + value
    return nonce, shift, adder, start


def records(path):
    if not path.exists():
        return []
    return [
        dict(item.split("=", 1) for item in line.split() if "=" in item)
        for line in path.read_text().splitlines()
    ]


class MockRPCHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        method = request["method"]
        if method == "getmininginfo":
            result = {"blocks": 2500000, "difficulty": 13.0, "networkminingpower": 0}
        elif method == "getblocktemplate":
            result = {
                "previousblockhash": "12" * 32, "height": 2500001,
                "curtime": 1760000000, "difficulty": f"{12 * NFIX:016x}",
                "coinbasevalue": 5000000000, "coinbaseaux": {"flags": ""},
                "transactions": [], "version": 1,
            }
        elif method == "getblockcount":
            result = 2500000
        elif method == "submitblock":
            self.server.payloads.put(bytes.fromhex(request["params"][0]))
            result = None
        else:
            self.server.errors.put(method)
            result = None
        body = json.dumps({"id": request.get("id"), "result": result,
                           "error": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockRPC(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), MockRPCHandler)
        self.payloads = queue.Queue()
        self.errors = queue.Queue()


class MockPoolHandler(socketserver.StreamRequestHandler):
    def handle(self):
        while line := self.rfile.readline():
            request = json.loads(line)
            method = request.get("method")
            if method == "mining.request":
                result = {"data": self.server.header.hex(), "difficulty": 12 * NFIX}
            elif method == "mining.submit":
                self.server.payloads.put(bytes.fromhex(request["params"][2]))
                result = True
            else:
                self.server.errors.put(method)
                continue
            self.wfile.write((json.dumps({"id": request["id"], "result": result}) + "\n").encode())
            self.wfile.flush()


class MockPool(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), MockPoolHandler)
        self.payloads = queue.Queue()
        self.errors = queue.Queue()
        self.header = (
            struct.pack("<I", 1) + bytes(range(32)) + bytes(range(32, 64)) +
            struct.pack("<I", 1760000000) + struct.pack("<Q", 27 * NFIX)
        )


@unittest.skipUnless(os.environ.get("GAPMINER_TEST_CRT_GPU_SUBMIT") == "1",
                     "set GAPMINER_TEST_CRT_GPU_SUBMIT=1 for local CUDA integration")
class LocalGPUIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MINER.is_file(), "build the CUDA/CGBN miner first")
        self.assertTrue(COVER.is_file(), "the shift720 test cover is required")
        self.assertIsNotNone(gmpy2, "gmpy2 is required for independent primality checks")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.record = Path(self.temp.name) / "records.log"
        self.output = Path(self.temp.name) / "miner.log"

    def run_miner(self, server, mode, node, submit=True, merit=None):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        port = server.server_address[1]
        args = [
            str(MINER), "--crt-file", str(COVER), "--threads", "4",
            "--enable-gpu-fermat", "--sieve-primes", "17000000",
            "--record-log", str(self.record),
        ]
        if node:
            args += ["--host", "127.0.0.1", "--port", str(port),
                     "--user", "test", "--pass", "test",
                     "--coinbase-script-hex", "51"]
        else:
            args += ["--stratum", f"127.0.0.1:{port}", "--stratum-user", "test"]
        if submit:
            args.append("--enable-submission")
        if merit is not None:
            args += ["--merit", str(merit)]
        env = miner_env(mode, "1" if submit and mode != "0" else None)
        env["MINING_JUMP2_VERIFY"] = "1"
        env["MJ2_DEBUG"] = "1"
        found = []
        with self.output.open("w") as log:
            proc = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        break
                    if submit:
                        try:
                            found.append(server.payloads.get(timeout=0.25))
                        except queue.Empty:
                            pass
                        if len(found) >= 2:
                            break
                    elif any(r.get("status") == "dry-run" for r in records(self.record)):
                        break
                    else:
                        time.sleep(0.25)
                if submit and len(found) >= 2:
                    until = time.monotonic() + 5
                    while time.monotonic() < until:
                        if any(r.get("status") == "accepted" for r in records(self.record)):
                            break
                        time.sleep(0.1)
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

        output = self.output.read_text()
        self.assertEqual(proc.returncode, 0, output[-10000:])
        self.assertIn("CGBN kernel active", output)
        if mode != "0":
            self.assertIn(f"CRT_ROW_SELECT={mode}: {'LIVE' if submit else 'dry-run'}", output)
        expected_merit = 12.0 if merit is None else merit
        merit_pattern = f"{expected_merit:.4f}".replace(".", r"\.")
        self.assertRegex(output,
                         rf"\[MJ2DBG\] flight K=\d+ total=\d+ merit={merit_pattern}\b")
        if node:
            if merit is None:
                self.assertIn("Active merit threshold 12.0000 from template nDifficulty",
                              output)
            else:
                self.assertIn(f"Active merit threshold: {merit:.2f} (CLI override)",
                              output)
                self.assertNotIn("from template nDifficulty", output)
        self.assertIn("bad_windows=0 bad_pairs=0", output)
        self.assertTrue(server.errors.empty(), list(server.errors.queue))
        if not submit:
            self.assertTrue(any(r.get("status") == "dry-run" for r in records(self.record)),
                            output[-10000:])
            self.assertTrue(server.payloads.empty(), "dry-run sent a pool share")
            return

        self.assertGreaterEqual(len(found), 2, output[-10000:])
        lines = records(self.record)
        queued = {(int(r["header_nonce"]), int(r["nAdd"])): r for r in lines
                  if r.get("status") == "queued"}
        accepted = {(int(r["header_nonce"]), int(r["nAdd"])) for r in lines
                    if r.get("status") == "accepted"}
        verified = 0
        for raw in found:
            nonce, shift, adder, start = parse_solution(raw, node)
            self.assertEqual(shift, 720)
            self.assertGreater(len(adder), 64)
            key = (nonce, int.from_bytes(adder, "little"))
            self.assertIn(key, queued)
            record = queued[key]
            self.assertEqual(start, int(record["start"]))
            self.assertTrue(gmpy2.is_prime(start, 25))
            self.assertEqual(int(gmpy2.next_prime(start)) - start, int(record["gap"]))
            if key in accepted:
                verified += 1
        self.assertGreater(verified, 0, "mock verdict did not retain the full nAdd")

    def test_selected_rows_submit_to_local_node(self):
        self.run_miner(MockRPC(), "select", node=True)

    def test_default_miner_starts_at_local_node_template_target(self):
        self.run_miner(MockRPC(), "0", node=True)

    def test_explicit_node_merit_overrides_template_target(self):
        self.run_miner(MockRPC(), "select", node=True, merit=11.5)

    def test_selected_rows_submit_to_local_pool(self):
        self.run_miner(MockPool(), "select", node=False)

    def test_observed_rows_submit_to_local_pool(self):
        self.run_miner(MockPool(), "observe", node=False)

    def test_pool_dry_run_does_not_submit(self):
        self.run_miner(MockPool(), "select", node=False, submit=False)


if __name__ == "__main__":
    unittest.main()
