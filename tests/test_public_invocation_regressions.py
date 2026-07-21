from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import urllib.parse

import numpy as np
import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


def _isolate_launch_environment(monkeypatch, tmp_path) -> tuple[object, object]:
    home = tmp_path / "home"
    config = tmp_path / "config"
    runtime = tmp_path / "runtime"
    for directory in (home, config, runtime):
        directory.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(runtime))
    for key in (
        "ARRAYVIEW_WINDOW",
        "ARRAYVIEW_WINDOW_ID",
        "VSCODE_AGENT_FOLDER",
        "SSH_CLIENT",
        "SSH_CONNECTION",
    ):
        monkeypatch.delenv(key, raising=False)

    signal_dir = home / ".arrayview"
    signal_dir.mkdir()
    return signal_dir, runtime


def _write_window_registration(
    signal_dir,
    window_id: str,
    *,
    remote_name: str | None,
) -> None:
    (signal_dir / f"window-{window_id}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hookTag": window_id,
                "remoteName": remote_name,
                "signalQueueVersion": 1,
            }
        )
    )


def test_public_cli_explicit_native_in_local_vscode_never_signals_extension(
    monkeypatch, tmp_path, capsys
):
    """The user's local VS Code environment must not override explicit native."""
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    signal_dir, _runtime = _isolate_launch_environment(monkeypatch, tmp_path)
    ipc_hook = str(tmp_path / "vscode-ipc.sock")
    window_id = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
    _write_window_registration(signal_dir, window_id, remote_name=None)

    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.setenv("VSCODE_INJECTION", "1")
    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", ipc_hook)
    monkeypatch.setattr(launch_plan, "_native_window_gui", lambda: "test-native")
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(
            port,
            True,
            True,
            os.getpid(),
            "test-host",
            "existing-server",
            None,
            ("identity-fenced-load", "identity-fenced-mutations"),
            "1",
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: ("existing-server", None, os.getpid()),
    )

    port = _free_port()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arrayview",
            "--diagnose",
            "--port",
            str(port),
            "--window",
            "native",
        ],
    )

    launcher.arrayview()

    diagnostics = json.loads(capsys.readouterr().out)
    assert diagnostics["snapshot"]["in_vscode_terminal"] is True
    assert diagnostics["snapshot"]["is_vscode_remote"] is False
    assert diagnostics["plan"]["environment"] == "vscode_local"
    assert diagnostics["plan"]["display"] == "native"
    assert "explicit_native" in diagnostics["plan"]["reasons"]

    array_path = tmp_path / "parameter_maps.npy"
    np.save(array_path, np.zeros((8, 8), dtype=np.float32))
    registrations = []
    native_opens = []

    def register(**kwargs):
        registrations.append(kwargs)
        return {
            "sid": "native-sid",
            "overlay_sid": None,
            "compare_sids": [],
            "notify_native_shell": kwargs["use_native_shell"],
            "notified": False,
        }

    monkeypatch.setattr(
        launcher,
        "_register_cli_session_with_existing_server",
        register,
    )
    monkeypatch.setattr(
        launcher,
        "_open_cli_native_shell_after_server",
        lambda **kwargs: native_opens.append(kwargs) or True,
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda *args, **kwargs: pytest.fail(
            "an accepted native launch must not enter the browser/VS Code router"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "arrayview",
            str(array_path),
            "--port",
            str(port),
            "--window",
            "native",
            "--verbose",
        ],
    )

    launcher.arrayview()

    output = capsys.readouterr().out
    assert "environment=vscode_local" in output
    assert "display=native" in output
    assert registrations and registrations[0]["use_native_shell"] is True
    assert len(native_opens) == 1
    assert native_opens[0]["sid"] == "native-sid"
    assert list(signal_dir.glob("open-request*")) == []


def test_public_cli_spawned_native_in_local_vscode_stays_native(
    monkeypatch, tmp_path
):
    """A cold CLI launch keeps explicit native authority through startup."""
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    signal_dir, _runtime = _isolate_launch_environment(monkeypatch, tmp_path)
    ipc_hook = str(tmp_path / "vscode-ipc.sock")
    window_id = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
    _write_window_registration(signal_dir, window_id, remote_name=None)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.setenv("VSCODE_INJECTION", "1")
    monkeypatch.setenv("VSCODE_IPC_HOOK_CLI", ipc_hook)
    monkeypatch.setattr(launch_plan, "_native_window_gui", lambda: "test-native")
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(port, False, False),
    )
    monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)

    backend = type("Backend", (), {"pid": 12345})()
    native_process = object()
    native_opens = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: backend)
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *a, **k: True
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: ("spawned-server", "process-start", 12345),
    )
    monkeypatch.setattr(
        launcher,
        "_open_webview_cli_tracked",
        lambda *args, **kwargs: (
            native_opens.append((args, kwargs)) or True,
            native_process,
        ),
    )
    monkeypatch.setattr(
        launcher, "_activate_early_cli_native_shell", lambda **kwargs: True
    )
    monkeypatch.setattr(
        launcher, "_load_compare_sids", lambda port, files, **kwargs: []
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda *args, **kwargs: pytest.fail(
            "a successful spawned native launch must not signal VS Code"
        ),
    )

    array_path = tmp_path / "parameter_maps.npy"
    np.save(array_path, np.zeros((8, 8), dtype=np.float32))
    monkeypatch.setattr(
        sys,
        "argv",
        ["arrayview", str(array_path), "--window", "native"],
    )

    launcher.arrayview()

    assert len(native_opens) == 1
    assert urllib.parse.urlsplit(native_opens[0][0][0]).path == "/shell"
    assert list(signal_dir.glob("open-request*")) == []


def test_public_remote_jupyter_starts_kernel_owned_in_process_server(
    monkeypatch, tmp_path
):
    """Remote notebook placement changes the opener, not kernel ownership."""
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    signal_dir, _runtime = _isolate_launch_environment(monkeypatch, tmp_path)
    _write_window_registration(signal_dir, "remote-window", remote_name="ssh-remote")
    for key in ("TERM_PROGRAM", "VSCODE_INJECTION", "VSCODE_IPC_HOOK_CLI"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(launcher, "_in_jupyter", lambda: True)
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_jupyter", lambda: True)
    monkeypatch.setattr(launcher._platform_mod, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(launch_plan, "_native_window_gui", lambda: None)
    monkeypatch.setattr(
        launch_plan,
        "_server_snapshot",
        lambda port: launch_plan.ServerSnapshot(port, False, False),
    )
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
    monkeypatch.setattr(launcher, "_server_pid", lambda port: None)
    monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
    monkeypatch.setattr(session_mod, "SERVER_LOOP", None)

    class ReadyEvent:
        def clear(self):
            return None

        def wait(self, timeout=None):
            return True

    thread_calls = []

    class ImmediateThread:
        def __init__(self, target=None, daemon=None, name=None):
            self.target = target
            thread_calls.append({"daemon": daemon, "name": name})

        def start(self):
            return self.target()

    server_calls = []

    async def serve_background(port, **kwargs):
        server_calls.append({"port": port, **kwargs})
        await asyncio.sleep(0)

    opened = []
    configured_ports = []
    monkeypatch.setattr(launcher, "_server_ready_event", ReadyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(launcher, "_serve_background", serve_background)
    monkeypatch.setattr(
        launcher,
        "_configure_vscode_port_preview",
        lambda port, **kwargs: configured_ports.append(port),
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda url, **kwargs: opened.append({"url": url, **kwargs}),
    )
    monkeypatch.setattr(launcher, "_print_viewer_location", lambda *a, **k: None)

    before_sids = set(session_mod.SESSIONS)
    try:
        handle = launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="remote-kernel",
            port=_free_port(),
        )

        context = opened[0]["launch_context"]
        assert isinstance(handle, launcher.ViewHandle)
        assert context.intent.invocation is launch_plan.Invocation.JUPYTER
        assert context.placement is launch_plan.Placement.VSCODE_REMOTE
        assert context.caller_scope is launch_plan.CallerScope.KERNEL
        assert context.plan.server_owner is launch_plan.ServerOwner.IN_PROCESS
        assert context.plan.display is launch_plan.Display.VSCODE
        assert thread_calls == [{"daemon": True, "name": "arrayview-server"}]
        assert len(server_calls) == 1
        assert server_calls[0]["owner_mode"] == "kernel"
        assert server_calls[0]["stop_when_closed"] is False
        assert configured_ports == [handle.port]
        assert opened[0]["force_vscode"] is True
    finally:
        for sid in set(session_mod.SESSIONS) - before_sids:
            session_mod.SESSIONS.pop(sid, None)
