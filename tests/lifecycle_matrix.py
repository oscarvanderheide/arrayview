#!/usr/bin/env python3
"""ArrayView invocation lifecycle verification matrix.

Run from the repository root:

    uv run python tests/lifecycle_matrix.py

The default run is safe: it does not open native windows, browsers, or VS Code
tabs. Rows marked MANUAL document checks that need an interactive UI session.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    status: str
    evidence: str
    detail: str = ""


@dataclass
class MatrixRow:
    area: str
    invariant: str
    mode: str
    check: Callable[[], CheckResult]


def _run(cmd: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _python() -> str:
    return sys.executable


def _short_output(result: subprocess.CompletedProcess[str], limit: int = 600) -> str:
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    combined = "\n".join(part for part in (out, err) if part)
    if len(combined) <= limit:
        return combined
    return combined[: limit - 3] + "..."


def _pytest_check(test_args: list[str], evidence: str) -> CheckResult:
    result = _run([_python(), "-m", "pytest", *test_args, "-q"], timeout=60.0)
    if result.returncode == 0:
        line = _short_output(result).splitlines()[-1]
        return CheckResult("PASS", evidence, line)
    return CheckResult("FAIL", evidence, _short_output(result))


def check_mex_context() -> CheckResult:
    if shutil.which("mex") is None:
        return CheckResult("SKIP", "mex is not installed")
    result = _run(["mex", "check", "--quiet"], timeout=30.0)
    if result.returncode == 0:
        return CheckResult("PASS", ".mex lifecycle routing", _short_output(result))
    return CheckResult("FAIL", ".mex lifecycle routing", _short_output(result))


def check_lifecycle_contract_tests() -> CheckResult:
    return _pytest_check(["tests/test_lifecycle_contract.py"], "tests/test_lifecycle_contract.py")


def check_invocation_contract_matrix() -> CheckResult:
    return _pytest_check(
        ["tests/test_invocation_contract_matrix.py"],
        "invocation x host x placement x OS planner contracts",
    )


def check_cli_contract_tests() -> CheckResult:
    return _pytest_check(["tests/test_cli.py"], "tests/test_cli.py")


def check_api_lifecycle_helpers() -> CheckResult:
    return _pytest_check(
        [
            "tests/test_api.py::TestOverlayWebSocket::test_shell_close_drops_session",
            "tests/test_api.py::TestCliOpenHelpers",
        ],
        "targeted API lifecycle helpers",
    )


def check_node_extension_helpers() -> CheckResult:
    node = shutil.which("node")
    if node is None:
        return CheckResult("SKIP", "node is not installed")
    result = _run(
        [
            node,
            "--check",
            "vscode-extension/extension.js",
        ],
        timeout=20.0,
    )
    if result.returncode != 0:
        return CheckResult("FAIL", "VS Code extension syntax", _short_output(result))
    scripts = [
        "test_lifecycle_helpers.js",
        "test_tunnel_resolution.js",
        "test_tunnel_desktop_loopback.js",
        "test_tunnel_loopback_promotion.js",
        "test_request_journal.js",
        "test_request_deadline.js",
        "test_panel_replay.js",
        "test_panel_readiness.js",
        "test_integrated_browser_readiness.js",
    ]
    for script in scripts:
        result = _run([node, f"vscode-extension/{script}"], timeout=20.0)
        if result.returncode != 0:
            return CheckResult("FAIL", script, _short_output(result))
    return CheckResult("PASS", "VS Code transaction contracts")


def check_vsix_content() -> CheckResult:
    from arrayview._vscode_extension import _VSCODE_EXT_VERSION

    vsix = ROOT / "src/arrayview/arrayview-opener.vsix"
    if not vsix.is_file():
        return CheckResult("FAIL", "bundled VSIX", "src/arrayview/arrayview-opener.vsix missing")
    try:
        with zipfile.ZipFile(vsix) as zf:
            package = json.loads(zf.read("extension/package.json"))
            names = set(zf.namelist())
            helper = zf.read("extension/lifecycle_helpers.js").decode()
    except Exception as exc:
        return CheckResult("FAIL", "bundled VSIX", str(exc))
    if package.get("version") != _VSCODE_EXT_VERSION:
        return CheckResult(
            "FAIL",
            "bundled VSIX version",
            f"package={package.get('version')} python={_VSCODE_EXT_VERSION}",
        )
    required = {
        "extension/extension.js",
        "extension/lifecycle_helpers.js",
        "extension/package.json",
    }
    missing = sorted(required - names)
    if missing:
        return CheckResult("FAIL", "bundled VSIX content", f"missing {missing}")
    if "shouldRemoveSameTunnelRegistration" not in helper:
        return CheckResult("FAIL", "bundled VSIX helper content", "tunnel cleanup helper missing")
    return CheckResult("PASS", "bundled VSIX version/content", f"version {_VSCODE_EXT_VERSION}")


def check_installed_vscode_extension() -> CheckResult:
    from arrayview._vscode_extension import _VSCODE_EXT_VERSION

    vsix = ROOT / "src/arrayview/arrayview-opener.vsix"
    installed = Path.home() / ".vscode/extensions" / f"arrayview.arrayview-opener-{_VSCODE_EXT_VERSION}"
    if not installed.is_dir():
        return CheckResult("WARN", "installed VS Code extension", f"{installed} not installed")
    helper = installed / "lifecycle_helpers.js"
    if not helper.is_file():
        return CheckResult("FAIL", "installed VS Code extension", "lifecycle_helpers.js missing")
    hash_file = installed / ".vsix_hash"
    if hash_file.is_file():
        installed_hash = hash_file.read_text().strip()
        bundled_hash = hashlib.md5(vsix.read_bytes()).hexdigest()
        if installed_hash != bundled_hash:
            return CheckResult(
                "FAIL",
                "installed VS Code extension hash",
                f"installed={installed_hash} bundled={bundled_hash}",
            )
        return CheckResult("PASS", "installed VS Code extension hash", _VSCODE_EXT_VERSION)
    return CheckResult("WARN", "installed VS Code extension hash", ".vsix_hash missing")


def check_live_transient_daemon_shutdown() -> CheckResult:
    try:
        import httpx
        import numpy as np
        import websockets
    except Exception as exc:
        return CheckResult("SKIP", "live daemon shutdown probe dependencies", str(exc))

    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = int(sock.getsockname()[1])

    with tempfile.TemporaryDirectory() as td:
        array_path = Path(td) / "live_shutdown.npy"
        np.save(array_path, np.zeros((8, 8), dtype=np.float32))
        sid = "lifecycle_matrix_sid"
        code = (
            "from arrayview._launcher import _serve_daemon;"
            f"_serve_daemon({str(array_path)!r}, {port}, {sid!r}, "
            "name='lifecycle-matrix', persist=False)"
        )
        proc = subprocess.Popen(
            [_python(), "-c", code],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    out, err = proc.communicate(timeout=1)
                    return CheckResult(
                        "FAIL",
                        "live transient daemon",
                        f"exited before /ping rc={proc.returncode} stdout={out} stderr={err}",
                    )
                try:
                    response = httpx.get(f"http://localhost:{port}/ping", timeout=0.3)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                time.sleep(0.1)
            else:
                return CheckResult("FAIL", "live transient daemon", "did not answer /ping")

            async def _connect_and_close() -> None:
                async with websockets.connect(f"ws://localhost:{port}/ws/{sid}") as ws:
                    first = await asyncio.wait_for(ws.recv(), timeout=5)
                    if '"type":"metadata"' not in first.replace(" ", ""):
                        raise AssertionError(first[:200])

            asyncio.run(_connect_and_close())
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return CheckResult(
                    "FAIL",
                    "live transient daemon",
                    "still alive after viewer WebSocket closed",
                )
            if proc.returncode != 0:
                out, err = proc.communicate(timeout=1)
                return CheckResult(
                    "FAIL",
                    "live transient daemon",
                    f"nonzero rc={proc.returncode} stdout={out} stderr={err}",
                )
            return CheckResult("PASS", "real subprocess + WebSocket", f"localhost:{port}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)


def check_build() -> CheckResult:
    result = _run(["uv", "build"], timeout=60.0)
    if result.returncode == 0:
        return CheckResult("PASS", "uv build", "sdist and wheel built")
    return CheckResult("FAIL", "uv build", _short_output(result))


def manual_native_window() -> CheckResult:
    return CheckResult(
        "MANUAL",
        "native UI session",
        "Run `uv run arrayview <file> --window native` and close the native window.",
    )


def manual_vscode_tab() -> CheckResult:
    return CheckResult(
        "MANUAL",
        "real VS Code tunnel extension host",
        "Verify originating-window selection with multiple tunnel windows; local-client "
        "reachability of the forwarded URL; forwarding privacy and external URI; "
        "request/window/server ACK correlation; extension-host restart; and tunnel "
        "reconnect without window reload or --kill.",
    )


def manual_jupyter() -> CheckResult:
    return CheckResult(
        "MANUAL",
        "real notebook kernel",
        "Run `view(arr)` in Jupyter; verify iframe loss does not kill kernel-owned backend.",
    )


def _matrix(include_build: bool) -> list[MatrixRow]:
    rows = [
        MatrixRow("Project context", "Lifecycle contract is routed and drift-free", "automated", check_mex_context),
        MatrixRow("All invocation contracts", "Core lifecycle invariants stay pinned", "automated", check_lifecycle_contract_tests),
        MatrixRow("Launch planner matrix", "Invocation, host, placement, and OS rules stay orthogonal", "automated", check_invocation_contract_matrix),
        MatrixRow("CLI", "CLI launch/reuse behavior stays pinned", "automated", check_cli_contract_tests),
        MatrixRow("FastAPI/WebSocket", "Shell close and CLI helper release paths work", "automated", check_api_lifecycle_helpers),
        MatrixRow("Transient daemon", "Local transient server exits after viewer disconnect", "real subprocess", check_live_transient_daemon_shutdown),
        MatrixRow("VS Code extension", "Extension lifecycle helpers and syntax are valid", "node", check_node_extension_helpers),
        MatrixRow("VS Code packaging", "Bundled VSIX version/content matches source", "automated", check_vsix_content),
        MatrixRow("VS Code install", "Installed extension matches bundled VSIX when present", "local state", check_installed_vscode_extension),
        MatrixRow("Native window", "Native pywebview launch/close still works", "manual", manual_native_window),
        MatrixRow("Live VS Code tab", "Actual tab opens in target window and closes cleanly", "manual", manual_vscode_tab),
        MatrixRow("Jupyter", "Notebook iframe lifecycle matches kernel ownership", "manual", manual_jupyter),
    ]
    if include_build:
        rows.append(MatrixRow("Package build", "Wheel/sdist include lifecycle artifacts", "automated", check_build))
    return rows


def _format_table(results: list[tuple[MatrixRow, CheckResult]]) -> str:
    headers = ("Status", "Area", "Mode", "Invariant", "Evidence")
    data = [
        (result.status, row.area, row.mode, row.invariant, result.evidence)
        for row, result in results
    ]
    widths = [
        max(len(headers[i]), *(len(item[i]) for item in data))
        for i in range(len(headers))
    ]
    lines = [
        " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        "-+-".join("-" * widths[i] for i in range(len(headers))),
    ]
    for item in data:
        lines.append(" | ".join(item[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip `uv build` to make the matrix faster.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    args = parser.parse_args()

    results: list[tuple[MatrixRow, CheckResult]] = []
    for row in _matrix(include_build=not args.no_build):
        try:
            result = row.check()
        except Exception as exc:
            result = CheckResult("FAIL", row.mode, f"{type(exc).__name__}: {exc}")
        results.append((row, result))

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "status": result.status,
                        "area": row.area,
                        "mode": row.mode,
                        "invariant": row.invariant,
                        "evidence": result.evidence,
                        "detail": result.detail,
                    }
                    for row, result in results
                ],
                indent=2,
            )
        )
    else:
        print(_format_table(results))
        details = [
            (row, result)
            for row, result in results
            if result.detail and result.status in {"FAIL", "WARN", "SKIP", "MANUAL"}
        ]
        if details:
            print("\nDetails:")
            for row, result in details:
                print(f"- {result.status} {row.area}: {result.detail}")

    failed = [result for _, result in results if result.status == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
