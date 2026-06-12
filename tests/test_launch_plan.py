import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest


@pytest.fixture
def clear_launch_env(monkeypatch):
    keys = [
        "ARRAYVIEW_WINDOW",
        "ARRAYVIEW_WINDOW_ID",
        "DISPLAY",
        "TERM_PROGRAM",
        "VSCODE_AGENT_FOLDER",
        "VSCODE_INJECTION",
        "VSCODE_IPC_HOOK_CLI",
        "WAYLAND_DISPLAY",
        "SSH_CLIENT",
        "SSH_CONNECTION",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/ping":
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(
            {
                "ok": True,
                "service": "arrayview",
                "pid": 4321,
                "hostname": "test-host",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def ping_server():
    server = HTTPServer(("localhost", 0), _PingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_launch_data_enums_match_refactor_contract():
    from arrayview._launch_plan import (
        Display,
        Environment,
        Invocation,
        Registration,
        ServerOwner,
        Transport,
    )

    assert {item.value for item in Invocation} == {
        "cli",
        "python",
        "jupyter",
        "julia",
        "stdio",
        "codex",
    }
    assert {item.value for item in Environment} == {
        "terminal",
        "vscode_local",
        "vscode_remote",
        "ssh",
        "jupyter",
        "julia",
    }
    assert {item.value for item in Transport} == {
        "http",
        "stdio_file",
        "stdio_shm",
        "none",
    }
    assert {item.value for item in ServerOwner} == {
        "existing",
        "spawned_daemon",
        "in_process",
        "persistent",
        "external",
    }
    assert {item.value for item in Display} == {
        "native",
        "browser",
        "vscode",
        "inline",
        "none",
    }
    assert {item.value for item in Registration} == {
        "http_load",
        "daemon_startup",
        "in_process_session",
        "stdio_register",
        "relay",
    }


def test_snapshot_classifies_terminal_and_keeps_selected_env(monkeypatch, clear_launch_env):
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import Environment, Invocation, snapshot_launch_environment

    monkeypatch.setenv("ARRAYVIEW_WINDOW", "browser")
    monkeypatch.setenv("TERM_PROGRAM", "not-vscode")
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: "gtk")

    snapshot = snapshot_launch_environment(_free_port(), Invocation.CLI, "native")

    assert snapshot.invocation is Invocation.CLI
    assert snapshot.environment is Environment.TERMINAL
    assert snapshot.requested_window == "native"
    assert snapshot.env_vars == {
        "ARRAYVIEW_WINDOW": "browser",
        "DISPLAY": ":99",
        "TERM_PROGRAM": "not-vscode",
    }
    assert snapshot.config_default is None
    assert snapshot.native_backend == "gtk"
    assert snapshot.server.port_busy is False
    assert snapshot.server.arrayview_server_alive is False


def test_snapshot_reads_config_window_default(monkeypatch, clear_launch_env, tmp_path):
    import arrayview._config as config_mod
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import Environment, snapshot_launch_environment

    config_file = tmp_path / "config.toml"
    config_file.write_text('[window]\nvscode = "vscode"\ndefault = "browser"\n')
    monkeypatch.setattr(config_mod, "CONFIG_PATH", str(config_file))
    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: None)

    snapshot = snapshot_launch_environment(_free_port(), "cli")

    assert snapshot.environment is Environment.VSCODE_LOCAL
    assert snapshot.config_default == "vscode"


@pytest.mark.parametrize(
    "facts, expected",
    [
        ({"in_jupyter": True}, "jupyter"),
        ({"in_julia": True}, "julia"),
        ({"is_vscode_remote": True, "in_vscode_terminal": True}, "vscode_remote"),
        ({"in_vscode_terminal": True}, "vscode_local"),
        ({"ssh_connection": True}, "ssh"),
    ],
)
def test_classify_environment_priority(facts, expected):
    from arrayview._launch_plan import _classify_environment

    defaults = {
        "in_jupyter": False,
        "in_julia": False,
        "in_vscode_terminal": False,
        "is_vscode_remote": False,
        "ssh_connection": False,
        "ssh_client": False,
    }
    defaults.update(facts)

    assert _classify_environment(**defaults).value == expected


def test_snapshot_reports_arrayview_server_ping(monkeypatch, clear_launch_env, ping_server):
    import arrayview._platform as platform_mod
    from arrayview._launch_plan import snapshot_launch_environment

    monkeypatch.setattr(platform_mod, "_in_jupyter", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(platform_mod, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(platform_mod, "_in_vscode_tunnel", lambda: False)
    monkeypatch.setattr(platform_mod, "_native_window_gui", lambda: None)

    snapshot = snapshot_launch_environment(ping_server, "python")

    assert snapshot.server.port_busy is True
    assert snapshot.server.arrayview_server_alive is True
    assert snapshot.server.server_pid == 4321
    assert snapshot.server.server_hostname == "test-host"


def test_importing_launch_plan_keeps_server_and_numpy_lazy():
    code = (
        "import sys; "
        "import arrayview._launch_plan; "
        "print('arrayview._server' in sys.modules, 'numpy' in sys.modules)"
    )
    result = sys.executable

    import subprocess

    completed = subprocess.run(
        [result, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False False"
