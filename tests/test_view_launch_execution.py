"""Focused execution tests for the shared launch plan in ``view()``."""

from types import SimpleNamespace

import numpy as np
import pytest


def _snapshot(launch_plan, invocation, *, server_alive=False):
    return launch_plan.LaunchEnvironmentSnapshot(
        invocation=invocation,
        requested_window=None,
        environment=(
            launch_plan.Environment.JULIA
            if invocation is launch_plan.Invocation.JULIA
            else launch_plan.Environment.TERMINAL
        ),
        platform="test",
        env_vars={},
        config_default=None,
        native_backend=None,
        server=launch_plan.ServerSnapshot(
            8123,
            server_alive,
            server_alive,
            4242 if server_alive else None,
            "test-host" if server_alive else None,
            "existing-server" if server_alive else None,
            "process-start" if server_alive else None,
            (
                "identity-fenced-load",
                "identity-fenced-mutations",
            )
            if server_alive
            else (),
            "1" if server_alive else None,
        ),
        in_jupyter=False,
        in_julia=invocation is launch_plan.Invocation.JULIA,
        in_vscode_terminal=False,
        is_vscode_remote=False,
        in_vscode_tunnel=False,
        ssh_connection=False,
        ssh_client=False,
        hostname="test-host",
    )


@pytest.mark.parametrize(
    ("display", "expected_window", "expected_inline"),
    [
        ("native", True, False),
        ("browser", False, False),
        ("vscode", False, False),
        ("inline", False, True),
        ("none", False, False),
    ],
)
def test_view_executes_planned_display_for_julia(
    monkeypatch, display, expected_window, expected_inline
):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    snapshots = []
    intents = []
    julia_calls = []
    snapshot = _snapshot(launch_plan, launch_plan.Invocation.JULIA)

    monkeypatch.setattr(launcher, "_is_julia_env", lambda: True)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(
        launch_plan,
        "snapshot_launch_environment",
        lambda port, invocation, requested_window=None: (
            snapshots.append((port, invocation, requested_window)) or snapshot
        ),
    )

    def fake_plan(intent, facts):
        intents.append((intent, facts))
        return SimpleNamespace(
            ok=True,
            failure=None,
            effective_port=8000,
            display=launch_plan.Display(display),
            registration=launch_plan.Registration.DAEMON_STARTUP,
        )

    monkeypatch.setattr(launch_plan, "plan_launch", fake_plan)
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )
    monkeypatch.setattr(
        launcher,
        "_view_julia",
        lambda *args, **kwargs: julia_calls.append((args, kwargs)) or "julia-result",
    )

    result = launcher.view(np.zeros((2, 2)), window="browser")

    assert result == "julia-result"
    assert snapshots == [(8123, launch_plan.Invocation.JULIA, "browser")]
    assert len(intents) == 1
    assert intents[0][1] is snapshot
    assert intents[0][0].window_explicit is True
    assert intents[0][0].inline_explicit is False
    args, kwargs = julia_calls[0]
    assert args[2] == 8000
    assert kwargs["window"] is expected_window
    assert kwargs["inline"] is expected_inline


def test_view_preserves_explicit_inline_intent(monkeypatch):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    captured = []
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: True)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(
        launch_plan,
        "snapshot_launch_environment",
        lambda *args, **kwargs: _snapshot(
            launch_plan, launch_plan.Invocation.JULIA
        ),
    )

    def fake_plan(intent, facts):
        captured.append(intent)
        return SimpleNamespace(
            ok=True,
            failure=None,
            effective_port=8123,
            display=launch_plan.Display.INLINE,
            registration=launch_plan.Registration.DAEMON_STARTUP,
        )

    monkeypatch.setattr(launch_plan, "plan_launch", fake_plan)
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )
    monkeypatch.setattr(launcher, "_view_julia", lambda *args, **kwargs: None)

    launcher.view(np.zeros((2, 2)), inline=False)

    assert captured[0].window_explicit is False
    assert captured[0].inline_explicit is True
    assert captured[0].inline is False


def _install_python_plan(
    monkeypatch, launcher, launch_plan, display, *, existing
):
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: False)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(
        launch_plan,
        "snapshot_launch_environment",
        lambda *args, **kwargs: _snapshot(
            launch_plan,
            launch_plan.Invocation.PYTHON,
            server_alive=existing,
        ),
    )
    monkeypatch.setattr(
        launch_plan,
        "plan_launch",
        lambda intent, facts: SimpleNamespace(
            ok=True,
            failure=None,
            effective_port=8123,
            display=launch_plan.Display(display),
            registration=(
                launch_plan.Registration.HTTP_LOAD
                if existing
                else launch_plan.Registration.IN_PROCESS_SESSION
            ),
        ),
    )
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )


@pytest.mark.parametrize(
    ("display", "requested_window", "expected_force_vscode", "expected_opens"),
    [
        ("browser", "browser", False, 1),
        ("vscode", "vscode", True, 1),
        ("none", "none", None, 0),
    ],
)
def test_existing_server_executes_python_plan(
    monkeypatch,
    display,
    requested_window,
    expected_force_vscode,
    expected_opens,
):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    _install_python_plan(
        monkeypatch, launcher, launch_plan, display, existing=True
    )
    opened = []
    monkeypatch.setattr(launcher, "_server_alive", lambda port: True)
    monkeypatch.setattr(
        launcher,
        "_load_session_from_filepath",
        lambda *args, **kwargs: {"sid": "planned-sid"},
    )
    monkeypatch.setattr(
        launcher, "_open_browser", lambda url, **kwargs: opened.append(kwargs)
    )

    handle = launcher.view(np.zeros((2, 2)), window=requested_window)

    assert isinstance(handle, launcher.ViewHandle)
    assert len(opened) == expected_opens
    if opened:
        assert opened[0]["force_vscode"] is expected_force_vscode


@pytest.mark.parametrize(
    ("display", "requested_window", "expected_force_vscode", "expected_opens"),
    [
        ("browser", "browser", False, 1),
        ("vscode", "vscode", True, 1),
        ("none", "none", None, 0),
    ],
)
def test_in_process_server_executes_python_plan(
    monkeypatch,
    display,
    requested_window,
    expected_force_vscode,
    expected_opens,
):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher
    import arrayview._session as session_mod

    _install_python_plan(
        monkeypatch, launcher, launch_plan, display, existing=False
    )
    opened = []
    monkeypatch.setattr(launcher, "_server_alive", lambda port: False)
    monkeypatch.setattr(launcher, "_server_pid", lambda port: launcher.os.getpid())
    monkeypatch.setattr(launcher, "_is_vscode_remote", lambda: False)
    monkeypatch.setattr(launcher, "_can_native_window", lambda: False)
    monkeypatch.setattr(
        launcher, "_open_browser", lambda url, **kwargs: opened.append(kwargs)
    )
    monkeypatch.setattr(session_mod, "SERVER_LOOP", None)

    handle = launcher.view(np.zeros((2, 2)), window=requested_window)

    assert isinstance(handle, launcher.ViewHandle)
    assert len(opened) == expected_opens
    if opened:
        assert opened[0]["force_vscode"] is expected_force_vscode


def test_julia_window_false_preserves_inline_compatibility(monkeypatch):
    import arrayview._launch_plan as launch_plan
    import arrayview._launcher as launcher

    captured = []
    monkeypatch.setattr(launcher, "_is_julia_env", lambda: True)
    monkeypatch.setattr(launcher._platform_mod, "_in_matlab", lambda: False)
    monkeypatch.setattr(
        launch_plan,
        "snapshot_launch_environment",
        lambda *args, **kwargs: _snapshot(
            launch_plan, launch_plan.Invocation.JULIA
        ),
    )

    def fake_plan(intent, facts):
        captured.append(intent)
        return SimpleNamespace(
            ok=True,
            failure=None,
            effective_port=8123,
            display=launch_plan.Display.INLINE,
            registration=launch_plan.Registration.DAEMON_STARTUP,
        )

    calls = []
    monkeypatch.setattr(launch_plan, "plan_launch", fake_plan)
    monkeypatch.setattr(
        launcher, "_revalidate_launch_server", lambda context, port: port
    )
    monkeypatch.setattr(
        launcher,
        "_view_julia",
        lambda *args, **kwargs: calls.append(kwargs) or "julia-result",
    )

    assert launcher.view(np.zeros((2, 2)), window=False) == "julia-result"
    assert captured[0].requested_window == "inline"
    assert captured[0].inline is True
    assert calls[0]["inline"] is True
    assert calls[0]["window"] is False
