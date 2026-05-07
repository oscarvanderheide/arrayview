"""Codex-app helper for restarting ArrayView and loading one file.

Usage:
    uv run arrayview-codex path/to/file.nii
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid

from arrayview._launcher import _server_alive, _wait_for_port


def _kill_listeners(port: int) -> None:
    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            check=False,
        )
        pids: list[int] = []
        for line in result.stdout.splitlines():
            if f":{port}" not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.append(int(parts[-1]))
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            )
        return

    result = subprocess.run(
        ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids = [int(p) for p in result.stdout.split() if p.isdigit()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    if not pids:
        return
    deadline = time.time() + 2.0
    while time.time() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            return
        time.sleep(0.05)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _start_loaded_server(filepath: str, port: int, sid: str, name: str) -> subprocess.Popen[bytes]:
    script = (
        "from arrayview._launcher import _serve_daemon; "
        f"_serve_daemon({filepath!r}, {port}, {sid!r}, name={name!r}, persist=True)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not _wait_for_port(port, timeout=15.0, tcp_only=True):
        raise RuntimeError(f"ArrayView server failed to start on port {port}")
    if not _server_alive(port, timeout=2.0):
        raise RuntimeError(f"ArrayView server on port {port} did not become healthy")
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="arrayview-codex",
        description=(
            "Restart ArrayView on localhost, load one file, and print the "
            "Codex app browser URL."
        ),
    )
    parser.add_argument("file", help="Array file to load")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    args = parser.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        return 1

    _kill_listeners(args.port)
    sid = uuid.uuid4().hex
    _start_loaded_server(filepath, args.port, sid, os.path.basename(filepath))
    url = f"http://localhost:{args.port}/?sid={sid}"
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
