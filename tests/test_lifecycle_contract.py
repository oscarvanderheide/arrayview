from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _install_lifecycle_view_mocks(monkeypatch, launcher, session_mod):
    monkeypatch.setattr(launcher, "_server_pid", lambda port: None)
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
    monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
    monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher, "_can_native_window", lambda: False)
    monkeypatch.setattr("arrayview._config.get_window_default", lambda _env: None)
    monkeypatch.setattr("arrayview._platform.detect_environment", lambda: "terminal")
    monkeypatch.setattr(session_mod, "SERVER_LOOP", None)


def test_plain_python_script_view_keeps_server_alive_until_viewer_closes(monkeypatch):
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_lifecycle_view_mocks(monkeypatch, launcher, session_mod)
    monkeypatch.setattr(launcher, "_is_script_mode", lambda: True)

    thread_calls = []

    class _DummyEvent:
        def clear(self):
            return None

        def wait(self, timeout=None):
            return True

    class _DummyThread:
        def __init__(self, target=None, daemon=None, name=None):
            thread_calls.append({"daemon": daemon, "name": name})
            self.target = target

        def start(self):
            return self.target()

    async def _fake_serve_background(
        port, stop_when_closed=False, owner_mode="in_process"
    ):
        thread_calls.append(
            {
                "port": port,
                "stop_when_closed": stop_when_closed,
                "owner_mode": owner_mode,
            }
        )

    monkeypatch.setattr(launcher, "_server_ready_event", _DummyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", _DummyThread)
    monkeypatch.setattr(launcher, "_serve_background", _fake_serve_background)
    monkeypatch.setattr(launcher, "_open_browser", lambda *args, **kwargs: None)

    handle = launcher.view(
        np.zeros((4, 4), dtype=np.float32),
        name="script-view",
        window=False,
    )

    assert isinstance(handle, launcher.ViewHandle)
    assert thread_calls[0]["daemon"] is False
    assert thread_calls[1] == {
        "port": 8123,
        "stop_when_closed": True,
        "owner_mode": "transient",
    }


def test_jupyter_view_is_kernel_owned_and_does_not_stop_on_iframe_disappearance(monkeypatch):
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_lifecycle_view_mocks(monkeypatch, launcher, session_mod)
    monkeypatch.setattr(launcher, "_in_jupyter", lambda: True)
    monkeypatch.setattr(launcher, "_should_use_jupyter_proxy_inline", lambda: False)

    thread_calls = []

    class _DummyEvent:
        def clear(self):
            return None

        def wait(self, timeout=None):
            return True

    class _DummyThread:
        def __init__(self, target=None, daemon=None, name=None):
            thread_calls.append({"daemon": daemon, "name": name})
            self.target = target

        def start(self):
            return self.target()

    async def _fake_serve_background(
        port, stop_when_closed=False, owner_mode="in_process"
    ):
        thread_calls.append(
            {
                "port": port,
                "stop_when_closed": stop_when_closed,
                "owner_mode": owner_mode,
            }
        )

    monkeypatch.setattr(launcher, "_server_ready_event", _DummyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", _DummyThread)
    monkeypatch.setattr(launcher, "_serve_background", _fake_serve_background)

    result = launcher.view(np.zeros((4, 4), dtype=np.float32), name="jupyter-view", inline=True)

    assert result.__class__.__name__ == "IFrame"
    assert thread_calls[0]["daemon"] is True
    assert thread_calls[1] == {
        "port": 8123,
        "stop_when_closed": False,
        "owner_mode": "kernel",
    }


def test_plain_ssh_browser_guidance_keeps_localhost_forwarding_url(monkeypatch, capsys):
    import arrayview._vscode_browser as browser

    monkeypatch.setattr(browser, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: False)
    monkeypatch.setenv("SSH_CLIENT", "localhost 12345 22")
    monkeypatch.setenv("SSH_CONNECTION", "localhost 12345 22")
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(browser.os, "uname", lambda: type("U", (), {"nodename": "ssh-host"})())

    browser._open_browser("http://localhost:8123/?sid=abc", blocking=True)

    out = capsys.readouterr().out
    assert "http://localhost:8123/" in out
    assert "ssh -L 8123:localhost:8123" in out


def test_plain_ssh_keeps_script_mode_transient(monkeypatch):
    import arrayview._launcher as launcher

    monkeypatch.setattr(launcher, "_in_jupyter", lambda: False)
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setenv("SSH_CLIENT", "localhost 12345 22")
    monkeypatch.setenv("SSH_CONNECTION", "localhost 12345 22")

    assert launcher._is_script_mode() is True


def test_vscode_tunnel_without_window_id_uses_focused_window_fallback(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    monkeypatch.setattr(signal, "_find_current_vscode_window_id", lambda: "100")

    for wid in ("100", "200"):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps({"pid": int(wid), "ppids": [10], "fallbackId": True})
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    request = json.loads(
        (signal_dir / signal._VSCODE_SIGNAL_FILENAME).read_text()
    )
    assert request["broadcast"] is True


def test_vscode_tunnel_exact_window_id_is_not_redirected_to_newer_sibling(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "100")

    (signal_dir / "window-100.json").write_text(
        json.dumps({"pid": 100, "ppids": [10], "fallbackId": True, "ts": 1})
    )
    (signal_dir / "window-200.json").write_text(
        json.dumps({"pid": 200, "ppids": [10], "fallbackId": True, "ts": 2})
    )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    assert (signal_dir / "open-request-pid-100.json").exists()
    assert not (signal_dir / "open-request-pid-200.json").exists()


def test_vscode_local_exact_window_id_is_not_redirected_to_newer_sibling(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "100")

    (signal_dir / "window-100.json").write_text(
        json.dumps({"pid": 100, "ppids": [10], "fallbackId": True, "ts": 1})
    )
    (signal_dir / "window-200.json").write_text(
        json.dumps({"pid": 200, "ppids": [10], "fallbackId": True, "ts": 2})
    )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    assert (signal_dir / "open-request-pid-100.json").exists()
    assert not (signal_dir / "open-request-pid-200.json").exists()


def test_vscode_local_stale_window_id_with_multiple_windows_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "stale")
    monkeypatch.setattr(signal, "_find_current_vscode_window_id", lambda: None)

    for wid in ("100", "200"):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps({"pid": int(wid), "ppids": [10], "fallbackId": True})
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is False
    assert "VS Code window is ambiguous" in capsys.readouterr().out
    assert not list(signal_dir.glob("open-request-*"))


def test_vscode_local_missing_window_match_uses_focused_window_fallback(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    monkeypatch.setattr(signal, "_find_current_vscode_window_id", lambda: None)
    monkeypatch.setattr("arrayview._platform._find_vscode_ipc_hook", lambda: None)
    monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: True)

    for wid in ("100", "200"):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps(
                {
                    "pid": int(wid),
                    "ppids": [10],
                    "fallbackId": True,
                    "remoteName": "tunnel",
                }
            )
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    request = json.loads(
        (signal_dir / signal._VSCODE_SIGNAL_FILENAME).read_text()
    )
    assert request["broadcast"] is True


def test_vscode_local_missing_window_match_with_local_windows_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    monkeypatch.setattr(signal, "_find_current_vscode_window_id", lambda: None)
    monkeypatch.setattr("arrayview._platform._find_vscode_ipc_hook", lambda: None)
    monkeypatch.setattr("arrayview._platform._in_vscode_terminal", lambda: True)

    for wid in ("100", "200"):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps({"pid": int(wid), "ppids": [10], "fallbackId": True})
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is False
    assert "VS Code window is ambiguous" in capsys.readouterr().out
    assert not list(signal_dir.glob("open-request-*"))


def test_tmux_multiple_window_ids_are_ambiguous(monkeypatch):
    import arrayview._vscode_signal as signal

    monkeypatch.delenv("ARRAYVIEW_WINDOW_ID", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "tmux")
    monkeypatch.setattr(signal, "get_ppid", lambda _pid: -1)

    class _Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def _run(cmd, *args, **kwargs):
        if cmd[:3] == ["tmux", "display-message", "-p"]:
            return _Result("$1\n")
        if cmd[:3] == ["tmux", "list-clients", "-t"]:
            return _Result("111\n222\n")
        if cmd[:3] == ["ps", "ewwww", "-p"]:
            pid = cmd[-1]
            wid = "win-a" if pid == "111" else "win-b"
            return _Result(f"COMMAND ARRAYVIEW_WINDOW_ID={wid}\n")
        return _Result("")

    monkeypatch.setattr(signal.subprocess, "run", _run)

    assert signal._find_arrayview_window_id() is None


def test_transient_waiter_notices_quick_viewer_connect_close(monkeypatch):
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    session_mod.VIEWER_SOCKETS = 0
    session_mod.VIEWER_CONNECTIONS_SEEN = 0
    sleeps = []

    def _sleep(_seconds):
        sleeps.append(_seconds)
        session_mod.VIEWER_CONNECTIONS_SEEN += 1

    monkeypatch.setattr(launcher.time, "sleep", _sleep)

    launcher._wait_for_viewer_close(
        grace_seconds=0,
        idle_seconds=0,
        connect_timeout=60,
    )

    assert sleeps == [0.2]


def test_transient_daemon_exits_after_quick_viewer_disconnect(tmp_path):
    import asyncio
    import socket
    import subprocess
    import sys
    import time

    import httpx
    import websockets

    with socket.socket() as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    array_path = tmp_path / "live_shutdown.npy"
    np.save(array_path, np.zeros((8, 8), dtype=np.float32))
    sid = "live_shutdown_sid"
    code = (
        "from arrayview._launcher import _serve_daemon;"
        f"_serve_daemon({str(array_path)!r}, {port}, {sid!r}, "
        "name='live-shutdown', persist=False)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                response = httpx.get(f"http://localhost:{port}/ping", timeout=0.3)
                if response.status_code == 200:
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=1)
                raise AssertionError(
                    f"daemon exited before ping: rc={proc.returncode}\nstdout={out}\nstderr={err}"
                )
            time.sleep(0.1)
        else:
            raise AssertionError("daemon did not answer /ping")

        async def _connect_and_close():
            async with websockets.connect(f"ws://localhost:{port}/ws/{sid}") as ws:
                first = await asyncio.wait_for(ws.recv(), timeout=5)
                assert '"type":"metadata"' in first.replace(" ", "")

        asyncio.run(_connect_and_close())
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                "transient daemon stayed alive after last viewer websocket closed"
            )
        assert proc.returncode == 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)


def test_local_vscode_spawned_daemon_uses_transient_backend(monkeypatch):
    import arrayview._launcher as launcher

    spawned = []

    monkeypatch.setattr(launcher, "_configure_vscode_port_preview", lambda port: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
    monkeypatch.setattr(launcher, "_open_cli_spawned_view", lambda **kwargs: None)
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid"})())

    launcher._handle_cli_spawned_daemon(
        port=8000,
        base_file="/tmp/base.npy",
        name="base.npy",
        compare_files=[],
        overlay_files=[],
        dims_override=None,
        use_native_shell=False,
        watch=False,
        window_mode="vscode",
        floating=False,
        is_remote=False,
        vectorfield=None,
        vfield_components_dim=None,
        rgb=False,
        demo_name=None,
        demo_cleanup=False,
    )

    assert spawned
    assert "persist=False" in spawned[0][0][2]


def test_remote_vscode_spawned_daemon_keeps_backend_persistent(monkeypatch):
    import arrayview._launcher as launcher

    spawned = []

    monkeypatch.setattr(launcher, "_configure_vscode_port_preview", lambda port: None)
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(launcher, "_wait_for_port", lambda *args, **kwargs: True)
    monkeypatch.setattr(launcher, "_load_compare_sids", lambda port, files: [])
    monkeypatch.setattr(launcher, "_open_cli_spawned_view", lambda **kwargs: None)
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid"})())

    launcher._handle_cli_spawned_daemon(
        port=8000,
        base_file="/tmp/base.npy",
        name="base.npy",
        compare_files=[],
        overlay_files=[],
        dims_override=None,
        use_native_shell=False,
        watch=False,
        window_mode="vscode",
        floating=False,
        is_remote=True,
        vectorfield=None,
        vfield_components_dim=None,
        rgb=False,
        demo_name=None,
        demo_cleanup=False,
    )

    assert spawned
    assert "persist=True" in spawned[0][0][2]


def test_viewer_sid_tracking_clears_on_websocket_disconnect(tmp_path):
    import arrayview._session as session_mod
    from arrayview._app import app

    session_mod.VIEWER_SOCKETS = 0
    session_mod.VIEWER_SIDS.clear()
    session_mod.VIEWER_SID_COUNTS.clear()
    session_mod.VIEWER_CONNECTIONS_SEEN = 0

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "viewer_sid.npy", arr)

    with TestClient(app) as client:
        sid = client.post(
            "/load", json={"filepath": str(tmp_path / "viewer_sid.npy")}
        ).json()["sid"]

        with client.websocket_connect(f"/ws/{sid}") as ws:
            assert ws.receive_json()["type"] == "metadata"
            assert session_mod.VIEWER_SOCKETS == 1
            assert sid in session_mod.VIEWER_SIDS
            assert session_mod.VIEWER_SID_COUNTS[sid] == 1
            assert session_mod.VIEWER_CONNECTIONS_SEEN == 1

        assert session_mod.VIEWER_SOCKETS == 0
        assert sid not in session_mod.VIEWER_SIDS
        assert sid not in session_mod.VIEWER_SID_COUNTS


def test_release_route_drops_session_and_is_idempotent(tmp_path):
    from arrayview._app import app

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "release.npy", arr)

    with TestClient(app) as client:
        sid = client.post("/load", json={"filepath": str(tmp_path / "release.npy")}).json()["sid"]
        assert client.get(f"/metadata/{sid}").status_code == 200

        released = client.post(f"/release/{sid}")
        assert released.status_code == 200
        assert released.json() == {"sid": sid, "released": True}
        assert client.get(f"/metadata/{sid}").status_code == 404

        released_again = client.post(f"/release/{sid}")
        assert released_again.status_code == 200
        assert released_again.json() == {"sid": sid, "released": False}


def test_shell_close_uses_release_route_semantics(tmp_path):
    from arrayview._app import app

    arr = np.ones((4, 4), dtype=np.float32)
    np.save(tmp_path / "shell-release.npy", arr)

    with TestClient(app) as client:
        sid = client.post(
            "/load", json={"filepath": str(tmp_path / "shell-release.npy")}
        ).json()["sid"]
        assert client.get(f"/metadata/{sid}").status_code == 200

        with client.websocket_connect("/ws/shell") as ws:
            ws.send_json({"action": "close", "sid": sid})

        assert client.get(f"/metadata/{sid}").status_code == 404


def test_vscode_wrapper_backend_check_uses_extension_host_ping():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "pingUrlFromViewerUrl(url)" in source
    assert "await arrayViewStatusOk(pingUrl)" in source
    assert "isArrayViewStatus(payload, expectedServerId)" in source
    assert "viewerReady = true" in source
    assert "postMessage({ type: 'backend-error', url })" in source
    assert "fetch(pingUrl" not in source


def test_vscode_url_panel_dispose_releases_primary_sid():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "function releaseUrlSession(url)" in source
    assert "collectReleaseSidsFromUrl(url)" in source
    assert "/release/${encodeURIComponent(sid)}" in source
    assert "releaseUrlSession(url)" in source


def test_vscode_lifecycle_helpers_with_node():
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [node, "vscode-extension/test_lifecycle_helpers.js"],
        cwd=repo_root,
        check=True,
    )


def test_bundled_vscode_vsix_matches_release_lifecycle_source():
    import json
    import zipfile

    from arrayview._vscode_extension import _VSCODE_EXT_VERSION

    vsix = Path(__file__).resolve().parents[1] / "src/arrayview/arrayview-opener.vsix"
    with zipfile.ZipFile(vsix) as zf:
        package = json.loads(zf.read("extension/package.json"))
        extension_source = zf.read("extension/extension.js").decode()
        helper_source = zf.read("extension/lifecycle_helpers.js").decode()

    assert package["version"] == _VSCODE_EXT_VERSION
    assert "collectReleaseSidsFromUrl(url)" in extension_source
    assert "compare_sids" in helper_source
    assert "overlay_sid" in helper_source
