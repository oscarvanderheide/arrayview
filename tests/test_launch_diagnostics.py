import json
import socket
import sys

import arrayview._launcher as launcher


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def test_diagnose_serializes_shared_snapshot_and_plan(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(
        sys,
        "argv",
        ["arrayview", "--diagnose", "--port", str(port), "--window", "browser"],
    )

    launcher.arrayview()

    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["snapshot"]["invocation"] == "cli"
    assert diagnostics["snapshot"]["requested_window"] == "browser"
    assert diagnostics["snapshot"]["server"]["port"] == port
    assert diagnostics["plan"]["invocation"] == "cli"
    assert diagnostics["plan"]["requested_port"] == port
    assert diagnostics["plan"]["display"] == "browser"


def test_diagnose_retains_host_probes(monkeypatch, capsys):
    port = _free_port()
    monkeypatch.setattr(sys, "argv", ["arrayview", "--diagnose", "--port", str(port)])

    launcher.arrayview()

    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["loopback"]["host"] == "localhost"
    assert diagnostics["loopback"]["getaddrinfo"]
    assert diagnostics["loopback"]["bind_probe"]["ok"] is True
    assert "webview" in diagnostics["native_dependencies"]
    assert "ipc_hook_recovered" in diagnostics["vscode"]
