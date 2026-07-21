"""Codex-app helper for safely loading one file into ArrayView.

Usage:
    uv run arrayview-codex path/to/file.nii
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid

from arrayview._instance_registry import InstanceRegistry
from arrayview._launch_plan import (
    Invocation,
    LaunchIntent,
    Registration,
    create_launch_context,
)
from arrayview._launcher import (
    _find_server_port,
    _load_session_from_filepath,
    _port_in_use,
    _revalidate_launch_server,
    _server_alive,
    _server_pid,
    _wait_for_port,
)


def _start_loaded_server(
    filepath: str, port: int, sid: str, name: str
) -> subprocess.Popen[bytes]:
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
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(f"ArrayView server failed to start on port {port}")
    if not _server_alive(port, timeout=2.0):
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(f"ArrayView server on port {port} did not become healthy")
    actual_pid = _server_pid(port)
    if actual_pid != proc.pid:
        if proc.poll() is None:
            proc.terminate()
        raise RuntimeError(
            f"Port {port} was claimed by another process while ArrayView was starting"
        )
    return proc


def _open_codex_file(filepath: str, requested_port: int) -> str:
    context = create_launch_context(
        LaunchIntent(
            Invocation.CODEX,
            requested_port,
            requested_window="browser",
            persistent=True,
        )
    )
    if not context.plan.ok:
        raise ValueError(f"Invalid launch request: {context.plan.failure.value}")

    name = os.path.basename(filepath)
    expected_server_id = None
    with InstanceRegistry().startup_lock(timeout=20.0):
        port = context.plan.effective_port
        already_running = context.plan.registration is Registration.HTTP_LOAD
        if already_running:
            port = _revalidate_launch_server(context, port)
            expected_server_id = context.evidence.server.server_instance_id
        else:
            if _port_in_use(port):
                port, appeared_running = _find_server_port(port + 1)
                if appeared_running or _port_in_use(port):
                    raise RuntimeError("Could not find a free port for ArrayView")
            if not 1 <= port <= 65535 or _port_in_use(port):
                raise RuntimeError("Could not find a free port for ArrayView")
            sid = uuid.uuid4().hex
            _start_loaded_server(filepath, port, sid, name)

    if already_running:
        result = _load_session_from_filepath(
            port,
            filepath,
            name,
            expected_server_id=expected_server_id,
            release_on_disconnect=True,
        )
        if "error" in result:
            raise RuntimeError(f"ArrayView server rejected the file: {result['error']}")
        sid = str(result["sid"])

    return f"http://localhost:{port}/?sid={sid}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="arrayview-codex",
        description=(
            "Load one file into a compatible ArrayView server, starting one on "
            "a free port when needed, and print the Codex app browser URL."
        ),
    )
    parser.add_argument("file", help="Array file to load")
    parser.add_argument("--port", type=int, default=8000, help="Preferred port")
    args = parser.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.isfile(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        return 1

    try:
        url = _open_codex_file(filepath, args.port)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
