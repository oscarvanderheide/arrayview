from pathlib import Path
import os
from threading import Thread as RealThread

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _run_async_in_thread(coro_factory):
    """Run an async test probe outside Playwright's main-thread event loop."""
    errors = []

    def target():
        import asyncio

        try:
            asyncio.run(coro_factory())
        except BaseException as exc:
            errors.append(exc)

    worker = RealThread(target=target)
    worker.start()
    worker.join(timeout=15.0)
    assert not worker.is_alive()
    if errors:
        raise errors[0]


def _install_lifecycle_view_mocks(monkeypatch, launcher, session_mod, *, remote=False):
    import arrayview._launch_plan as launch_plan

    monkeypatch.setattr(launcher, "_server_pid", lambda port: None)
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
    monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
    monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: remote)
    monkeypatch.setattr(launcher, "_in_vscode_terminal", lambda: remote)
    monkeypatch.setattr(launcher, "_in_jupyter", lambda: False)
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(launcher, "_can_native_window", lambda: False)
    monkeypatch.setattr(session_mod, "SERVER_LOOP", None)

    def deterministic_snapshot(port, invocation, requested_window=None):
        inv = launch_plan.Invocation(invocation)
        in_jupyter = launcher._in_jupyter()
        environment = (
            launch_plan.Environment.VSCODE_REMOTE
            if remote
            else
            launch_plan.Environment.JUPYTER
            if in_jupyter
            else launch_plan.Environment.TERMINAL
        )
        return launch_plan.LaunchEnvironmentSnapshot(
            invocation=inv,
            requested_window=requested_window,
            environment=environment,
            platform="test",
            env_vars={},
            config_default=None,
            native_backend=None,
            server=launch_plan.ServerSnapshot(port, False, False),
            in_jupyter=in_jupyter,
            in_julia=False,
            in_vscode_terminal=remote,
            is_vscode_remote=remote,
            in_vscode_tunnel=remote,
            ssh_connection=False,
            ssh_client=False,
            hostname="test-host",
        )

    monkeypatch.setattr(
        launch_plan,
        "snapshot_launch_environment",
        deterministic_snapshot,
    )


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
            worker = RealThread(target=self.target)
            worker.start()
            worker.join(timeout=2.0)
            assert not worker.is_alive()

    async def _fake_serve_background(
        port,
        stop_when_closed=False,
        owner_mode="in_process",
        connect_timeout=20.0,
    ):
        thread_calls.append(
            {
                "port": port,
                "stop_when_closed": stop_when_closed,
                "owner_mode": owner_mode,
                "connect_timeout": connect_timeout,
            }
        )

    monkeypatch.setattr(launcher, "_server_ready_event", _DummyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", _DummyThread)
    monkeypatch.setattr(launcher, "_serve_background", _fake_serve_background)
    monkeypatch.setattr(launcher, "_open_browser", lambda *args, **kwargs: None)

    before_sids = set(session_mod.SESSIONS)
    try:
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
            "connect_timeout": 20.0,
        }
    finally:
        for sid in set(session_mod.SESSIONS) - before_sids:
            session_mod.SESSIONS.pop(sid, None)


@pytest.mark.parametrize("remote", [False, True])
def test_jupyter_view_is_kernel_owned_and_does_not_stop_on_iframe_disappearance(
    monkeypatch, remote
):
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_lifecycle_view_mocks(
        monkeypatch, launcher, session_mod, remote=remote
    )
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
            worker = RealThread(target=self.target)
            worker.start()
            worker.join(timeout=2.0)
            assert not worker.is_alive()

    async def _fake_serve_background(
        port,
        stop_when_closed=False,
        owner_mode="in_process",
        connect_timeout=20.0,
    ):
        thread_calls.append(
            {
                "port": port,
                "stop_when_closed": stop_when_closed,
                "owner_mode": owner_mode,
                "connect_timeout": connect_timeout,
            }
        )

    monkeypatch.setattr(launcher, "_server_ready_event", _DummyEvent())
    monkeypatch.setattr(launcher.threading, "Thread", _DummyThread)
    monkeypatch.setattr(launcher, "_serve_background", _fake_serve_background)
    monkeypatch.setattr(launcher, "_open_browser", lambda *args, **kwargs: None)

    before_sids = set(session_mod.SESSIONS)
    try:
        result = launcher.view(
            np.zeros((4, 4), dtype=np.float32),
            name="jupyter-view",
            inline=True,
        )

        assert result.__class__.__name__ == "IFrame"
        assert thread_calls[0]["daemon"] is True
        assert thread_calls[1] == {
            "port": 8123,
            "stop_when_closed": False,
            "owner_mode": "kernel",
            "connect_timeout": 20.0,
        }
    finally:
        for sid in set(session_mod.SESSIONS) - before_sids:
            session_mod.SESSIONS.pop(sid, None)


def test_plain_ssh_browser_guidance_keeps_localhost_forwarding_url(monkeypatch, capsys):
    import arrayview._vscode_browser as browser

    monkeypatch.setattr(browser, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: False)
    monkeypatch.setenv("SSH_CLIENT", "localhost 12345 22")
    monkeypatch.setenv("SSH_CONNECTION", "localhost 12345 22")
    monkeypatch.setattr(browser.sys, "platform", "linux")
    monkeypatch.setattr(browser.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(
        browser.os,
        "uname",
        lambda: type("U", (), {"nodename": "ssh-host"})(),
        raising=False,
    )

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


def test_vscode_tunnel_without_exact_window_with_multiple_hosts_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    monkeypatch.setattr(signal, "_find_current_vscode_window_id", lambda: "100")

    for wid in ("100", "200"):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps(
                {
                    "pid": int(wid),
                    "ppids": [10],
                    "fallbackId": True,
                    "signalQueueVersion": 1,
                }
            )
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is False
    assert "VS Code remote window is ambiguous" in capsys.readouterr().out
    assert not list(signal_dir.glob("open-request-*"))


def test_vscode_tunnel_recovers_exact_window_from_ipc_hook(monkeypatch, tmp_path):
    import hashlib
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    hook = "/tmp/vscode-ipc-current-window"
    window_id = hashlib.sha256(hook.encode()).hexdigest()[:16]
    monkeypatch.setattr("arrayview._platform._find_vscode_ipc_hook", lambda: hook)
    (signal_dir / f"window-{window_id}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hookTag": window_id,
                "remoteName": "tunnel",
                "signalQueueVersion": 1,
            }
        )
    )
    (signal_dir / "window-sibling.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hookTag": "sibling",
                "remoteName": "tunnel",
                "signalQueueVersion": 1,
            }
        )
    )

    assert signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc", "requestId": "req-1"},
        skip_compat=True,
    )

    target = signal_dir / f"open-request-ipc-{window_id}.request-req-1.json"
    assert target.is_file()
    assert json.loads(target.read_text())["windowId"] == window_id
    assert not list(signal_dir.glob("open-request-v0900*"))


def test_vscode_tunnel_exact_window_id_is_not_redirected_to_newer_sibling(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "100")

    (signal_dir / "window-100.json").write_text(
        json.dumps({"pid": 100, "ppids": [10], "fallbackId": True, "ts": 1, "signalQueueVersion": 1})
    )
    (signal_dir / "window-200.json").write_text(
        json.dumps({"pid": 200, "ppids": [10], "fallbackId": True, "ts": 2, "signalQueueVersion": 1})
    )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    assert list(signal_dir.glob("open-request-pid-100.request-*.json"))
    assert not list(signal_dir.glob("open-request-pid-200.request-*.json"))


def test_vscode_tunnel_open_request_is_remote_only(monkeypatch, tmp_path):
    import arrayview._vscode_signal as signal

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    captured = []
    monkeypatch.setattr(
        signal,
        "_write_vscode_signal",
        lambda payload, delay=0.0: captured.append(payload) or True,
    )

    signal._open_via_signal_file("http://localhost:8000/?sid=abc")

    assert captured[0]["remoteOnly"] is True


def test_vscode_extension_disk_check_is_scoped_to_remote_host(monkeypatch, tmp_path):
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    local_dir = home / ".vscode" / "extensions" / "arrayview.arrayview-opener-1.2.3"
    remote_dir = home / ".vscode-server" / "extensions" / "arrayview.arrayview-opener-1.2.3"
    local_dir.mkdir(parents=True)
    (home / ".vscode-server").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))

    assert extension._extension_on_disk("1.2.3", remote=False) is True
    assert extension._extension_on_disk("1.2.3", remote=True) is False

    remote_dir.mkdir(parents=True)
    assert extension._extension_on_disk("1.2.3", remote=True) is True


@pytest.mark.parametrize("remote", [False, True])
def test_vscode_extension_missing_hash_verifies_content_without_reinstall(
    monkeypatch, tmp_path, remote
):
    import hashlib
    import json
    import zipfile
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    local_base = home / ".vscode" / "extensions"
    remote_base = home / ".vscode-server" / "extensions"
    base = remote_base if remote else local_base
    version = extension._VSCODE_EXT_VERSION
    installed = base / f"arrayview.arrayview-opener-{version}"
    installed.mkdir(parents=True)
    if remote:
        (home / ".vscode-server").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))

    vsix = Path(extension.__file__).with_name("arrayview-opener.vsix")
    with zipfile.ZipFile(vsix) as archive:
        for info in archive.infolist():
            if not info.filename.startswith("extension/") or info.is_dir():
                continue
            relative = Path(info.filename).relative_to("extension")
            target = installed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))

    # VS Code rewrites package.json and injects host-specific metadata.
    package_json = installed / "package.json"
    package = json.loads(package_json.read_text())
    package["__metadata"] = {"targetPlatform": "undefined", "installedTimestamp": 1}
    package_json.write_text(json.dumps(package, indent=2))
    monkeypatch.setattr(extension, "_active_extension_version", lambda: version)
    monkeypatch.setattr(
        extension,
        "_run_extension_installer",
        lambda *args, **kwargs: pytest.fail(
            "verified extension content must not trigger a reinstall"
        ),
    )

    assert not (installed / ".vsix_hash").exists()
    assert extension._ensure_vscode_extension(is_remote=remote) is True
    assert (installed / ".vsix_hash").read_text() == hashlib.md5(
        vsix.read_bytes()
    ).hexdigest()


def test_vscode_extension_missing_hash_rejects_changed_content(monkeypatch, tmp_path):
    import zipfile
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    version = extension._VSCODE_EXT_VERSION
    installed = (
        home
        / ".vscode"
        / "extensions"
        / f"arrayview.arrayview-opener-{version}"
    )
    installed.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    vsix = Path(extension.__file__).with_name("arrayview-opener.vsix")
    with zipfile.ZipFile(vsix) as archive:
        for info in archive.infolist():
            if not info.filename.startswith("extension/") or info.is_dir():
                continue
            relative = Path(info.filename).relative_to("extension")
            target = installed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    (installed / "extension.js").write_text("changed")

    assert extension._extension_on_disk(version, str(vsix), remote=False) is False
    assert not (installed / ".vsix_hash").exists()


def test_vscode_port_settings_preserve_unreadable_jsonc(monkeypatch, tmp_path):
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    settings_path = home / ".vscode-server" / "data" / "Machine" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    original = '{\n  "editor.fontSize": 14,\n}\n'
    settings_path.write_text(original)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_is_vscode_remote", lambda: True)
    extension._VSCODE_CONFIGURED_PORTS.clear()

    assert extension._configure_vscode_port_preview(8123) is True

    assert settings_path.read_text() == original


def test_vscode_port_settings_are_idempotent_and_preserve_extra_keys(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    settings_path = home / ".vscode-server" / "data" / "Machine" / "settings.json"
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "remote.portsAttributes": {
                    "8123": {"label": "old", "requireLocalPort": True}
                }
            }
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_is_vscode_remote", lambda: True)
    extension._VSCODE_CONFIGURED_PORTS.clear()

    assert extension._configure_vscode_port_preview(8123) is True
    first = settings_path.read_text()
    assert extension._configure_vscode_port_preview(8123) is True

    attrs = json.loads(first)["remote.portsAttributes"]["8123"]
    assert attrs["privacy"] == "public"
    assert attrs["requireLocalPort"] is True
    assert settings_path.read_text() == first


def test_vscode_extension_cleanup_never_removes_newer_version(monkeypatch, tmp_path):
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    base = home / ".vscode" / "extensions"
    for version in ("0.14.39", "0.14.40", "0.14.41"):
        (base / f"arrayview.arrayview-opener-{version}").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    extension._remove_old_extension_versions("0.14.40", remote=False)

    assert not (base / "arrayview.arrayview-opener-0.14.39").exists()
    assert (base / "arrayview.arrayview-opener-0.14.40").exists()
    assert (base / "arrayview.arrayview-opener-0.14.41").exists()


def test_vscode_extension_keeps_newer_install_without_downgrading(monkeypatch, tmp_path):
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    newer_version = "99.0.0"
    (home / ".vscode" / "extensions" / f"arrayview.arrayview-opener-{newer_version}").mkdir(
        parents=True
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(extension, "_active_extension_version", lambda: newer_version)
    monkeypatch.setattr(
        extension.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("a newer opener must not be reinstalled"),
    )

    assert extension._ensure_vscode_extension() is True


def test_vscode_extension_requires_reload_for_stale_live_host(monkeypatch):
    import arrayview._vscode_extension as extension

    monkeypatch.setattr(extension, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(extension, "_extension_on_disk", lambda *args, **kwargs: True)
    monkeypatch.setattr(extension, "_active_extension_version", lambda: "0.14.40")

    assert extension._ensure_vscode_extension() is False
    assert extension._VSCODE_EXT_RELOAD_REQUIRED is True


def test_vscode_extension_install_timeout_fails_closed_for_stale_host(monkeypatch):
    import arrayview._vscode_extension as extension

    monkeypatch.setattr(extension, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(extension, "_extension_on_disk", lambda *args, **kwargs: False)
    monkeypatch.setattr(extension, "_newer_extension_on_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_active_extension_version", lambda: "0.14.40")
    monkeypatch.setattr(extension, "_find_code_cli", lambda **kwargs: "code")

    def timed_out(*args, **kwargs):
        raise TimeoutError("installer wedged")

    monkeypatch.setattr(extension, "_run_extension_installer", timed_out)

    assert extension._ensure_vscode_extension() is False
    assert extension._VSCODE_EXT_RELOAD_REQUIRED is True


def test_remote_vscode_launch_automatically_runs_exact_installer(monkeypatch, tmp_path):
    import subprocess
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    remote_base = home / ".vscode-server" / "extensions"
    old_version = "0.14.47"
    wanted = extension._VSCODE_EXT_VERSION
    (remote_base / f"arrayview.arrayview-opener-{old_version}").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_extension_on_disk", lambda *args, **kwargs: False)
    monkeypatch.setattr(extension, "_newer_extension_on_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_active_extension_version", lambda: old_version)
    monkeypatch.setattr(extension, "_find_code_cli", lambda **kwargs: "/exact/code")
    monkeypatch.setattr(extension, "_find_vscode_ipc_hook", lambda: "/exact/ipc.sock")
    calls = []

    def install(command, env):
        calls.append((command, env.get("VSCODE_IPC_HOOK_CLI")))
        target = remote_base / f"arrayview.arrayview-opener-{wanted}"
        target.mkdir(parents=True)
        (target / "package.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    monkeypatch.setattr(extension, "_run_extension_installer", install)
    monkeypatch.setattr(extension, "_patch_vscode_extension_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_wait_for_active_extension_version", lambda *args, **kwargs: True)

    assert extension._ensure_vscode_extension(is_remote=True) is True
    assert calls and calls[0][0][0] == "/exact/code"
    assert calls[0][1] == "/exact/ipc.sock"
    assert extension._VSCODE_EXT_RELOAD_REQUIRED is False


def test_active_extension_version_prefers_live_ipc_over_stale_terminal_env(
    monkeypatch, tmp_path
):
    import hashlib
    import json
    import os
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    ipc = "/run/user/1/current-window.sock"
    current_id = hashlib.sha256(ipc.encode()).hexdigest()[:16]
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ARRAYVIEW_WINDOW_ID", "stale-window")
    monkeypatch.setattr(extension, "_find_vscode_ipc_hook", lambda: ipc)
    (signal_dir / "window-stale-window.json").write_text(
        json.dumps({"pid": os.getpid(), "extensionVersion": "0.14.47"})
    )
    (signal_dir / f"window-{current_id}.json").write_text(
        json.dumps({"pid": os.getpid(), "extensionVersion": "0.14.70"})
    )

    assert extension._active_extension_version() == "0.14.70"


def test_active_extension_version_recovers_revived_terminal_from_exact_server(
    monkeypatch, tmp_path
):
    import json
    import arrayview._platform as platform
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ARRAYVIEW_WINDOW_ID", "stale-window")
    monkeypatch.setattr(extension, "_find_vscode_ipc_hook", lambda: None)
    monkeypatch.setattr(
        platform,
        "_current_vscode_remote_cli",
        lambda: "/servers/tunnel/server/bin/remote-cli/code",
    )
    monkeypatch.setattr(platform, "_process_is_alive", lambda pid: pid in (101, 202))
    monkeypatch.setattr(
        platform.os,
        "readlink",
        lambda path: {
            "/proc/101/exe": "/servers/tunnel/server/node",
            "/proc/202/exe": "/servers/ssh/server/node",
        }[path],
    )
    (signal_dir / "window-tunnel.json").write_text(
        json.dumps(
            {
                "pid": 101,
                "remoteName": "tunnel",
                "extensionVersion": "0.14.70",
            }
        )
    )
    (signal_dir / "window-ssh.json").write_text(
        json.dumps(
            {
                "pid": 202,
                "remoteName": "ssh-remote",
                "extensionVersion": "0.14.47",
            }
        )
    )

    assert extension._active_extension_version() == "0.14.70"


def test_revived_terminal_does_not_guess_between_same_server_windows(
    monkeypatch, tmp_path
):
    import json
    import arrayview._platform as platform

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ARRAYVIEW_WINDOW_ID", "stale-window")
    monkeypatch.setattr(
        platform,
        "_current_vscode_remote_cli",
        lambda: "/servers/tunnel/server/bin/remote-cli/code",
    )
    monkeypatch.setattr(platform, "_process_is_alive", lambda pid: pid in (101, 102))
    monkeypatch.setattr(
        platform.os,
        "readlink",
        lambda path: "/servers/tunnel/server/node",
    )
    for wid, pid in (("tunnel-a", 101), ("tunnel-b", 102)):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps({"pid": pid, "remoteName": "tunnel"})
        )

    assert platform._exact_vscode_window_registration(None) is None


def test_extension_activation_wait_requires_fresh_exact_registration(monkeypatch):
    import arrayview._vscode_extension as extension

    previous = (100, "old-instance", 1)
    monkeypatch.setattr(
        extension,
        "_active_extension_registration",
        lambda: {
            "pid": 100,
            "extensionInstanceId": "old-instance",
            "ts": 1,
            "extensionVersion": "0.14.70",
        },
    )
    assert not extension._wait_for_active_extension_version(
        "0.14.70", timeout=0, previous_marker=previous
    )

    monkeypatch.setattr(
        extension,
        "_active_extension_registration",
        lambda: {
            "pid": 101,
            "extensionInstanceId": "new-instance",
            "ts": 2,
            "extensionVersion": "0.14.70",
        },
    )
    assert extension._wait_for_active_extension_version(
        "0.14.70", timeout=0, previous_marker=previous
    )


def test_remote_vscode_setup_touches_only_active_extension_base(
    monkeypatch, tmp_path
):
    import subprocess
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    local_base = home / ".vscode" / "extensions"
    remote_base = home / ".vscode-server" / "extensions"
    old_version = "0.14.47"
    wanted = extension._VSCODE_EXT_VERSION
    (remote_base / f"arrayview.arrayview-opener-{old_version}").mkdir(parents=True)
    unrelated_local = local_base / f"arrayview.arrayview-opener-{wanted}"
    unrelated_local.mkdir(parents=True)
    (unrelated_local / ".vsix_hash").write_text("leave-this-profile-alone")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_active_extension_version", lambda: old_version)
    monkeypatch.setattr(extension, "_extension_on_disk", lambda *args, **kwargs: False)
    monkeypatch.setattr(extension, "_newer_extension_on_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_find_code_cli", lambda **kwargs: "/exact/code")
    monkeypatch.setattr(extension, "_find_vscode_ipc_hook", lambda: "/exact/ipc.sock")

    def install(command, env):
        target = remote_base / f"arrayview.arrayview-opener-{wanted}"
        target.mkdir(parents=True)
        (target / "package.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    monkeypatch.setattr(extension, "_run_extension_installer", install)
    monkeypatch.setattr(extension, "_patch_vscode_extension_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_wait_for_active_extension_version", lambda *args, **kwargs: True)

    assert extension._ensure_vscode_extension(is_remote=True) is True
    assert extension._VSCODE_EXT_RELOAD_REQUIRED is False
    assert (remote_base / f"arrayview.arrayview-opener-{wanted}" / ".vsix_hash").is_file()
    assert (unrelated_local / ".vsix_hash").read_text() == "leave-this-profile-alone"
    assert (remote_base / f"arrayview.arrayview-opener-{old_version}").is_dir()


def test_remote_vscode_install_requires_exact_activation_before_launch(
    monkeypatch, tmp_path
):
    import subprocess
    import arrayview._vscode_extension as extension

    home = tmp_path / "home"
    remote_base = home / ".vscode-server" / "extensions"
    old_version = "0.14.47"
    wanted = extension._VSCODE_EXT_VERSION
    (remote_base / f"arrayview.arrayview-opener-{old_version}").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(extension, "_active_extension_version", lambda: old_version)
    monkeypatch.setattr(extension, "_extension_on_disk", lambda *args, **kwargs: False)
    monkeypatch.setattr(extension, "_newer_extension_on_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_find_code_cli", lambda **kwargs: "/exact/code")
    monkeypatch.setattr(extension, "_find_vscode_ipc_hook", lambda: "/exact/ipc.sock")

    def install(command, env):
        target = remote_base / f"arrayview.arrayview-opener-{wanted}"
        target.mkdir(parents=True)
        (target / "package.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0, "installed", "")

    monkeypatch.setattr(extension, "_run_extension_installer", install)
    monkeypatch.setattr(extension, "_patch_vscode_extension_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(extension, "_wait_for_active_extension_version", lambda *args, **kwargs: False)

    assert extension._ensure_vscode_extension(is_remote=True) is False
    assert extension._VSCODE_EXT_RELOAD_REQUIRED is True


def test_vscode_local_exact_window_id_is_not_redirected_to_newer_sibling(
    monkeypatch, tmp_path
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "100")

    (signal_dir / "window-100.json").write_text(
        json.dumps({"pid": 100, "ppids": [10], "fallbackId": True, "ts": 1, "signalQueueVersion": 1})
    )
    (signal_dir / "window-200.json").write_text(
        json.dumps({"pid": 200, "ppids": [10], "fallbackId": True, "ts": 2, "signalQueueVersion": 1})
    )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is True
    assert list(signal_dir.glob("open-request-pid-100.request-*.json"))
    assert not list(signal_dir.glob("open-request-pid-200.request-*.json"))


def test_vscode_local_stale_window_id_with_multiple_windows_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
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


def test_vscode_remote_stale_window_id_prefers_live_ipc_registration(
    monkeypatch, tmp_path
):
    import hashlib
    import json
    import os
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    ipc_hook = "/run/user/1/current-tunnel.sock"
    current_wid = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "stale-window")
    monkeypatch.setattr(signal, "_find_vscode_ipc_hook", lambda: ipc_hook)
    monkeypatch.setattr(
        "arrayview._platform._find_vscode_ipc_hook", lambda: ipc_hook
    )

    (signal_dir / f"window-{current_wid}.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hookTag": current_wid,
                "remoteName": "tunnel",
                "fallbackId": False,
                "signalQueueVersion": 1,
            }
        )
    )
    (signal_dir / "window-other.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hookTag": "other",
                "remoteName": "ssh-remote",
                "fallbackId": False,
                "signalQueueVersion": 1,
            }
        )
    )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
        is_remote=True,
    )

    assert opened is True
    assert list(signal_dir.glob(f"open-request-ipc-{current_wid}.request-*.json"))
    assert not list(signal_dir.glob("open-request-v0900.request-*.json"))


def test_vscode_remote_revived_terminal_targets_exact_server_registration(
    monkeypatch, tmp_path
):
    import json
    import os
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "stale-window")
    monkeypatch.setattr(signal, "_find_vscode_ipc_hook", lambda: None)
    monkeypatch.setattr(
        signal,
        "_exact_vscode_window_registration",
        lambda ipc: (
            "tunnel",
            {"pid": os.getpid(), "remoteName": "tunnel"},
        ),
    )
    for wid, remote_name in (("tunnel", "tunnel"), ("ssh", "ssh-remote")):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hookTag": wid,
                    "remoteName": remote_name,
                    "fallbackId": False,
                    "signalQueueVersion": 1,
                }
            )
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
        is_remote=True,
    )

    assert opened is True
    assert list(signal_dir.glob("open-request-ipc-tunnel.request-*.json"))
    assert not list(signal_dir.glob("open-request-ipc-ssh.request-*.json"))
    assert not list(signal_dir.glob("open-request-v0900.request-*.json"))


def test_vscode_remote_stale_window_id_without_exact_match_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import os
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(signal, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "stale-window")
    monkeypatch.setattr(signal, "_find_vscode_ipc_hook", lambda: None)
    monkeypatch.setattr("arrayview._platform._find_vscode_ipc_hook", lambda: None)

    for wid, remote_name in (("tunnel", "tunnel"), ("ssh", "ssh-remote")):
        (signal_dir / f"window-{wid}.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "hookTag": wid,
                    "remoteName": remote_name,
                    "fallbackId": False,
                    "signalQueueVersion": 1,
                }
            )
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
        is_remote=True,
    )

    assert opened is False
    assert "VS Code window is ambiguous" in capsys.readouterr().out
    assert not list(signal_dir.glob("open-request-*"))


def test_vscode_local_missing_window_match_with_remote_hosts_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
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
                        "signalQueueVersion": 1,
                    }
            )
        )

    opened = signal._write_vscode_signal(
        {"url": "http://localhost:8000/?sid=abc"},
        skip_compat=True,
    )

    assert opened is False
    assert "VS Code window is ambiguous" in capsys.readouterr().out
    assert not list(signal_dir.glob("open-request-*"))


def test_vscode_local_missing_window_match_with_local_windows_fails_closed(
    monkeypatch, tmp_path, capsys
):
    import json
    import arrayview._vscode_signal as signal

    home = tmp_path / "home"
    signal_dir = home / ".arrayview"
    signal_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
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


def test_in_process_script_server_stops_immediately_after_display_rollback(
    monkeypatch,
):
    from types import SimpleNamespace

    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    monkeypatch.setattr(session_mod, "SESSIONS", {})
    monkeypatch.setattr(session_mod, "PENDING_SESSIONS", set())
    monkeypatch.setattr(session_mod, "VIEWER_SOCKETS", 0)
    monkeypatch.setattr(session_mod, "VIEWER_CONNECTIONS_SEEN", 0)
    server = SimpleNamespace(should_exit=False)

    _run_async_in_thread(
        lambda: launcher._stop_server_when_viewer_closes(
            server, connect_timeout=60.0
        )
    )

    assert server.should_exit is True


def test_persistent_daemon_bounds_failed_open_and_recoverable_disconnects():
    import arrayview._launcher as launcher
    import arrayview._routes_websocket as websocket_routes

    assert launcher._PERSIST_DAEMON_CONNECT_TIMEOUT_SECONDS == 210.0
    assert launcher._LOCAL_VSCODE_CONNECT_TIMEOUT_SECONDS == 70.0
    assert launcher._PERSIST_DAEMON_IDLE_SECONDS == 1800.0
    assert websocket_routes._RECOVERABLE_DISCONNECT_GRACE_SECONDS == 1800.0


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

        _run_async_in_thread(_connect_and_close)
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
    from types import SimpleNamespace

    spawned = []

    monkeypatch.setattr(
        launcher, "_configure_vscode_port_preview", lambda port, **kwargs: None
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: ("spawned-server", "process-start", 12345),
    )
    monkeypatch.setattr(
        launcher, "_load_compare_sids", lambda port, files, **kwargs: []
    )
    monkeypatch.setattr(launcher, "_open_cli_spawned_view", lambda **kwargs: None)
    monkeypatch.setattr(launcher.uuid, "uuid4", lambda: type("U", (), {"hex": "sid"})())
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )
    launch_context = SimpleNamespace(
        placement=SimpleNamespace(value="vscode_local"),
        plan=SimpleNamespace(display=SimpleNamespace(value="vscode")),
    )

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
        launch_context=launch_context,
    )

    assert spawned
    assert "persist=False" in spawned[0][0][2]
    assert "connect_timeout=70.0" in spawned[0][0][2]


def test_remote_vscode_spawned_daemon_keeps_backend_persistent(monkeypatch):
    import arrayview._launcher as launcher

    spawned = []

    monkeypatch.setattr(
        launcher, "_configure_vscode_port_preview", lambda port, **kwargs: None
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda cmd, *args, **kwargs: spawned.append((cmd, kwargs)) or object(),
    )
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        launcher,
        "_server_runtime_identity",
        lambda port: ("spawned-server", "process-start", 12345),
    )
    monkeypatch.setattr(
        launcher, "_load_compare_sids", lambda port, files, **kwargs: []
    )
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


def test_disconnect_owned_session_releases_related_sessions_after_reconnect_grace(
    tmp_path,
):
    import time

    from arrayview._app import app

    path = tmp_path / "disconnect-release.npy"
    np.save(path, np.ones((4, 4), dtype=np.float32))
    compare_path = tmp_path / "disconnect-release-compare.npy"
    np.save(compare_path, np.ones((4, 4), dtype=np.float32) * 2)
    overlay_path = tmp_path / "disconnect-release-overlay.npy"
    np.save(overlay_path, np.ones((4, 4), dtype=np.float32) * 3)

    with TestClient(app) as client:
        sid = client.post(
            "/load",
            json={"filepath": str(path)},
        ).json()["sid"]
        compare_sid = client.post(
            "/load", json={"filepath": str(compare_path)}
        ).json()["sid"]
        overlay_sid = client.post(
            "/load", json={"filepath": str(overlay_path)}
        ).json()["sid"]
        from arrayview._session import SESSIONS

        server_id = client.get("/ping").json()["instance_id"]
        phase_path = f"/viewer-phase/{sid}/related-release-request"
        prepared = client.post(
            phase_path,
            json={
                "phase": "launch-prepared",
                "server_id": server_id,
                "window_id": "related-release-window",
                "token": "related-release-token",
            },
        )
        assert prepared.status_code == 200
        reported = client.post(
            phase_path,
            json={
                "phase": "script-loaded",
                "server_id": server_id,
                "window_id": "related-release-window",
                "token": "related-release-token",
                "viewer_instance_id": "related-release-viewer",
                "release_on_disconnect": True,
                "sid": sid,
                "compare_sid": compare_sid,
                "compare_sids": compare_sid,
                "overlay_sid": overlay_sid,
            },
        )
        assert reported.status_code == 200
        journal = SESSIONS[sid].viewer_phase_journals[
            "related-release-request"
        ]
        journal["disconnect_release_grace_seconds"] = 0.05
        socket_path = (
            f"/ws/{sid}?launch_request_id=related-release-request"
            "&launch_token=related-release-token"
        )
        with client.websocket_connect(socket_path) as ws:
            assert ws.receive_json()["type"] == "metadata"

        with client.websocket_connect(socket_path) as ws:
            assert ws.receive_json()["type"] == "metadata"
            time.sleep(0.1)
            for owned_sid in (sid, compare_sid, overlay_sid):
                assert client.get(f"/metadata/{owned_sid}").status_code == 200

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if all(
                client.get(f"/metadata/{owned_sid}").status_code == 404
                for owned_sid in (sid, compare_sid, overlay_sid)
            ):
                break
            time.sleep(0.05)
        for owned_sid in (sid, compare_sid, overlay_sid):
            assert client.get(f"/metadata/{owned_sid}").status_code == 404


def test_integrated_launch_cleanup_is_scoped_per_request_token(tmp_path):
    import time

    from arrayview._app import app
    from arrayview._session import SESSIONS

    paths = []
    for index in range(3):
        path = tmp_path / f"related-lease-{index}.npy"
        np.save(path, np.ones((4, 4), dtype=np.float32) * (index + 1))
        paths.append(path)

    with TestClient(app) as client:
        primary_sid, first_related_sid, second_related_sid = [
            client.post("/load", json={"filepath": str(path)}).json()["sid"]
            for path in paths
        ]
        SESSIONS[primary_sid].viewer_leases = 2
        SESSIONS[primary_sid].related_release_sids = [first_related_sid]
        SESSIONS[first_related_sid].viewer_leases = 2
        server_id = client.get("/ping").json()["instance_id"]

        def prepare_launch(request_id, token, related_sid):
            phase_path = f"/viewer-phase/{primary_sid}/{request_id}"
            prepared = client.post(
                phase_path,
                json={
                    "phase": "launch-prepared",
                    "server_id": server_id,
                    "window_id": f"window-{request_id}",
                    "token": token,
                },
            )
            assert prepared.status_code == 200
            reported = client.post(
                phase_path,
                json={
                    "phase": "script-loaded",
                    "server_id": server_id,
                    "window_id": f"window-{request_id}",
                    "token": token,
                    "viewer_instance_id": f"viewer-{request_id}",
                    "release_on_disconnect": True,
                    "sid": primary_sid,
                    "compare_sid": related_sid,
                    "compare_sids": related_sid,
                    "overlay_sid": None,
                },
            )
            assert reported.status_code == 200
            journal = SESSIONS[primary_sid].viewer_phase_journals[request_id]
            journal["disconnect_release_grace_seconds"] = 0.05
            return f"/ws/{primary_sid}?launch_request_id={request_id}&launch_token={token}"

        first_socket = prepare_launch(
            "first-request", "first-token", first_related_sid
        )
        second_socket = prepare_launch(
            "second-request", "second-token", second_related_sid
        )

        with client.websocket_connect(first_socket) as first_ws:
            assert first_ws.receive_json()["type"] == "metadata"
            with client.websocket_connect(second_socket) as second_ws:
                assert second_ws.receive_json()["type"] == "metadata"

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if second_related_sid not in SESSIONS:
                    break
                time.sleep(0.02)
            assert second_related_sid not in SESSIONS
            assert first_related_sid in SESSIONS
            assert SESSIONS[primary_sid].viewer_leases == 1
            assert (
                "second-request"
                not in SESSIONS[primary_sid].viewer_phase_journals
            )

            from starlette.websockets import WebSocketDisconnect

            with pytest.raises(WebSocketDisconnect) as stale_reconnect:
                with client.websocket_connect(second_socket):
                    pass
            assert stale_reconnect.value.code == 1008
            assert SESSIONS[primary_sid].viewer_leases == 1

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if primary_sid not in SESSIONS:
                break
            time.sleep(0.02)
        assert primary_sid not in SESSIONS
        assert SESSIONS[first_related_sid].viewer_leases == 1
        assert client.post(f"/release/{first_related_sid}").json()["released"] is True


def test_reconnect_cancels_and_fences_pending_disconnect_release(tmp_path):
    import time

    from arrayview._app import app
    from arrayview._session import SESSIONS

    path = tmp_path / "disconnect-reconnect.npy"
    np.save(path, np.ones((4, 4), dtype=np.float32))

    with TestClient(app) as client:
        sid = client.post(
            "/load",
            json={
                "filepath": str(path),
                "release_on_disconnect": True,
            },
        ).json()["sid"]
        SESSIONS[sid].disconnect_release_grace_seconds = 0.2

        with client.websocket_connect(f"/ws/{sid}") as ws:
            assert ws.receive_json()["type"] == "metadata"

        with client.websocket_connect(f"/ws/{sid}") as ws:
            assert ws.receive_json()["type"] == "metadata"
            time.sleep(0.3)
            assert client.get(f"/metadata/{sid}").status_code == 200

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if client.get(f"/metadata/{sid}").status_code == 404:
                break
            time.sleep(0.02)
        assert client.get(f"/metadata/{sid}").status_code == 404


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


def test_vscode_open_ack_requires_requested_session_metadata():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "sessionMetadataUrlFromViewerUrl(openUrl)" in source
    assert "sessionMetadataUrlFromViewerUrl(url)" in source
    assert "expired before a panel could be opened" in source
    assert "serverReady && await httpStatus2xx(metadataUrl)" in source
    assert "waitForViewerReady(panel, viewerTimeoutMs)" in source
    assert "message.phase === 'frame-rendered'" in source
    assert "await viewerReady" in source
    assert "Viewer session did not become ready" in source


def test_vscode_tunnel_resolution_reuses_verified_routes_and_retries_fresh():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "const TUNNEL_ROUTE_CACHE_FILE" in source
    assert "cache[`${logWindowId}:${port}`]" in source
    assert "REMOTE: cached route ready" in source
    assert "function _asExternalUriAttempt(baseUri)" in source
    assert "a hung promise cannot poison all" in source
    assert "_externalUriInFlight" not in source
    assert "remote.tunnel.closeInline" not in source
    assert "asExternalUri timeout after 15000ms" not in source
    assert "asExternalUri timeout after 20000ms" not in source


def test_vscode_url_panel_dispose_releases_primary_sid():
    source = (Path(__file__).resolve().parents[1] / "vscode-extension" / "extension.js").read_text()

    assert "function releaseUrlSession(url, backendUrl = null, serverId = null)" in source
    assert "collectReleaseSidsFromUrl(url)" in source
    assert "releaseUrlForSid(url, backendUrl, sid)" in source
    assert "releaseUrlSession(url, backendUrl, serverId)" in source
    assert "releaseUrlSession(openUrl, data.url, data.serverId || null);" in source
    assert "X-ArrayView-Expected-Server-ID" in source
    assert "}, 60000)" not in source


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


@pytest.mark.parametrize(
    "script",
    [
        "test_tunnel_resolution.js",
        "test_tunnel_desktop_loopback.js",
        "test_tunnel_loopback_promotion.js",
    ],
)
def test_vscode_tunnel_resolution_with_node(script):
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [node, f"vscode-extension/{script}"],
        cwd=repo_root,
        check=True,
    )


@pytest.mark.parametrize(
    "script",
    [
        "test_request_journal.js",
        "test_request_deadline.js",
        "test_panel_replay.js",
    ],
)
def test_vscode_transaction_contracts_with_node(script):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [node, f"vscode-extension/{script}"],
        cwd=repo_root,
        check=True,
    )


def test_vscode_extension_defaults_integrated_browser_to_remote_proxy():
    import json
    import zipfile

    repo_root = Path(__file__).resolve().parents[1]
    source_package = json.loads(
        (repo_root / "vscode-extension" / "package.json").read_text()
    )
    with zipfile.ZipFile(repo_root / "src/arrayview/arrayview-opener.vsix") as zf:
        bundled_package = json.loads(zf.read("extension/package.json"))

    for package in (source_package, bundled_package):
        assert package["contributes"]["configurationDefaults"][
            "workbench.browser.enableRemoteProxy"
        ] is True


def test_bundled_vscode_vsix_matches_release_lifecycle_source():
    import json
    import zipfile

    from arrayview._vscode_extension import _VSCODE_EXT_VERSION

    vsix = Path(__file__).resolve().parents[1] / "src/arrayview/arrayview-opener.vsix"
    with zipfile.ZipFile(vsix) as zf:
        package = json.loads(zf.read("extension/package.json"))
        extension_source = zf.read("extension/extension.js").decode()
        helper_source = zf.read("extension/lifecycle_helpers.js").decode()
        manifest_source = zf.read("extension.vsixmanifest").decode()

    assert package["version"] == _VSCODE_EXT_VERSION
    identity = (
        'Identity Language="en-US" Id="arrayview-opener" '
        f'Version="{_VSCODE_EXT_VERSION}"'
    )
    assert identity in manifest_source
    assert "collectReleaseSidsFromUrl(url)" in extension_source
    assert "data.remoteOnly === true && !vscode.env.remoteName" in extension_source
    assert "extensionVersion: version" in extension_source
    assert "releaseUrlSession(url, backendUrl = null, serverId = null)" in extension_source
    assert "releaseUrlSession(url, backendUrl, serverId)" in extension_source
    assert "const lockPath = `${ackPath}.lock`" in extension_source
    assert "_atomicWriteJson(ackPath" in extension_source
    assert (
        "vscode.env.remoteName === 'tunnel' && isLoopbackUrl(externalBase)"
        in extension_source
    )
    assert "const signalHardTimeoutMs = remainingSignalMs === null" in extension_source
    assert "Math.max(1000, remainingSignalMs + 1000)" in extension_source
    assert "const TUNNEL_ROUTE_CACHE_FILE" in extension_source
    assert "function _asExternalUriAttempt(baseUri)" in extension_source
    assert "REMOTE: cached route ready" in extension_source
    assert "compare_sids" in helper_source
    assert "overlay_sid" in helper_source
