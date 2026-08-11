"""Minimal remote perception loop test.

This script starts vision_server.py locally and verifies the first-stage TCP
JSON contract used by RemotePerceptionGateway.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VISION_SERVER = ROOT / "vision_server.py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perception.remote_gateway import RemotePerceptionConfig, RemotePerceptionGateway  # noqa: E402


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int, timeout_sec: float = 5.0) -> None:
    deadline = time.time() + timeout_sec
    last_error: OSError | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"server did not open port {port}: {last_error}")


def _start_server(port: int, extra_args: Iterable[str] = ()) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(VISION_SERVER),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mode",
        "mock",
        "--log-level",
        "WARNING",
        *list(extra_args),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port(port)
    except Exception:
        _stop_server(proc)
        raise
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _raw_request(port: int, request: dict) -> dict:
    with socket.create_connection(("127.0.0.1", port), timeout=1.0) as sock:
        sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
        data = b""
        while b"\n" not in data:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
    if not data:
        raise AssertionError("no JSON response received")
    return json.loads(data.split(b"\n", 1)[0].decode("utf-8"))


def test_normal_connection_and_json() -> None:
    port = _free_port()
    proc = _start_server(port)
    try:
        gw = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=port, timeout_sec=1.0))
        assert gw.is_ready() is True

        raw = _raw_request(port, {"req": "detect_gauges"})
        assert raw["type"] == "gauges"
        assert isinstance(raw["detections"], list)
        assert raw["detections"][0]["zone"] == "A"
        assert raw["detections"][0]["status"] in {"low", "normal", "high"}
        assert "bbox" in raw["detections"][0]

        letters = gw.detect_zone_letters()
        assert len(letters) == 4

        gauges = gw.detect_gauges()
        assert len(gauges) >= 1
        assert gauges[0].zone.value == "A"
        assert gauges[0].status.value in {"low", "normal", "high"}

        readings = []
        while True:
            batch = gw.poll_inspection()
            if not batch:
                break
            readings.extend(batch)
        assert len(readings) >= 1
        assert readings[0].zone.value == "A"
    finally:
        _stop_server(proc)


def test_disconnect_reconnect() -> None:
    port = _free_port()
    proc = _start_server(port, ["--disconnect-after", "1"])
    try:
        gw = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=port, timeout_sec=1.0))
        first = gw.detect_zone_letters()
        assert len(first) == 4

        # The previous server-side close may make this call return cache while
        # the gateway notices the disconnect. A following request must reconnect.
        second = gw.detect_zone_letters()
        assert isinstance(second, list)
        third = gw.detect_zone_letters()
        assert len(third) == 4
    finally:
        _stop_server(proc)


def test_timeout_handling() -> None:
    port = _free_port()
    proc = _start_server(port, ["--response-delay-sec", "1.0"])
    try:
        gw = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=port, timeout_sec=0.2))
        assert gw.detect_zone_letters() == []
        assert gw.detect_gauges() == []
        assert gw.poll_inspection() == []
    finally:
        _stop_server(proc)


def test_empty_results() -> None:
    port = _free_port()
    proc = _start_server(port, ["--empty-results"])
    try:
        gw = RemotePerceptionGateway(RemotePerceptionConfig(host="127.0.0.1", port=port, timeout_sec=1.0))
        assert gw.detect_zone_letters() == []
        assert gw.detect_gauges() == []
        assert gw.poll_inspection() == []
    finally:
        _stop_server(proc)


def main() -> int:
    tests = [
        test_normal_connection_and_json,
        test_disconnect_reconnect,
        test_timeout_handling,
        test_empty_results,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("remote perception client checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
