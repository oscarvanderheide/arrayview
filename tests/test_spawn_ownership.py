from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_spawn_readiness_requires_the_spawned_child_pid(monkeypatch):
    import arrayview._launcher as launcher

    child = _FakeProcess(123)
    monkeypatch.setattr(launcher, "_server_pid", lambda port: 456)
    monkeypatch.setattr(launcher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(launcher.time, "monotonic", iter([0.0, 0.0, 1.0]).__next__)

    assert launcher._wait_for_spawned_server(child, 8123, timeout=0.5) is False

    monkeypatch.setattr(launcher.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(launcher, "_server_pid", lambda port: 123)
    assert launcher._wait_for_spawned_server(child, 8123, timeout=0.5) is True


def test_cli_spawn_failure_terminates_owned_backend_and_preload(
    monkeypatch, tmp_path
):
    import arrayview._launcher as launcher

    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(launcher.sys, "platform", "darwin")
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
    backend = _FakeProcess(321)
    preload = _FakeProcess(654)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: backend)
    monkeypatch.setattr(
        launcher,
        "_open_webview_cli_tracked",
        lambda *a, **k: (True, preload),
    )
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *a, **k: False
    )
    monkeypatch.setattr(
        launcher,
        "_open_cli_spawned_view",
        lambda **kwargs: pytest.fail("display must not open after backend failure"),
    )

    with pytest.raises(SystemExit):
        launcher._handle_cli_spawned_daemon(
            port=8123,
            base_file="/tmp/base.npy",
            name="base.npy",
            compare_files=[],
            overlay_files=[],
            dims_override=None,
            use_native_shell=True,
            watch=False,
            window_mode="native",
            floating=False,
            is_remote=False,
            vectorfield=None,
            vfield_components_dim=None,
            rgb=False,
            demo_name=None,
            demo_cleanup=False,
        )

    assert backend.terminated == 1
    assert preload.terminated == 1


def test_julia_spawn_failure_terminates_only_its_child(monkeypatch, tmp_path):
    import arrayview._launcher as launcher

    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(tmp_path / "runtime"))
    child = _FakeProcess(777)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *a, **k: False
    )
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )
    context = SimpleNamespace(
        plan=SimpleNamespace(registration=SimpleNamespace(value="daemon_startup")),
        placement=SimpleNamespace(value="local"),
    )

    with pytest.raises(RuntimeError, match="did not claim port"):
        launcher._view_subprocess(
            np.zeros((2, 2), dtype=np.float32),
            "julia-array",
            8123,
            False,
            launch_context=context,
        )

    assert child.terminated == 1


def test_serve_spawn_failure_terminates_only_its_child(monkeypatch, tmp_path):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    monkeypatch.setenv("ARRAYVIEW_RUNTIME_DIR", str(tmp_path / "runtime"))
    context = SimpleNamespace(
        plan=SimpleNamespace(
            ok=True,
            failure=None,
            effective_port=8123,
            registration=launch_plan.Registration.DAEMON_STARTUP,
        ),
        evidence=SimpleNamespace(
            is_vscode_remote=False,
            in_vscode_terminal=False,
        ),
    )
    monkeypatch.setattr(launch_plan, "create_launch_context", lambda intent: context)
    monkeypatch.setattr(launcher, "_port_in_use", lambda port: False)
    monkeypatch.setattr(
        launcher, "_configure_vscode_port_preview", lambda *a, **k: None
    )
    child = _FakeProcess(888)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(
        launcher, "_wait_for_spawned_server", lambda *a, **k: False
    )
    monkeypatch.setattr(launcher.sys, "argv", ["arrayview", "--serve"])

    with pytest.raises(SystemExit):
        launcher.arrayview()

    assert child.terminated == 1
