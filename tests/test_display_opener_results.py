from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arrayview import _vscode_browser as browser
from arrayview._vscode_signal import AckState, SignalRequest


def _context(
    *,
    requested_window: str,
    environment: str,
    invocation: str = "cli",
    native_backend: str | None = None,
    ssh: bool = False,
):
    from arrayview._launch_plan import (
        Environment,
        Invocation,
        LaunchEnvironmentSnapshot,
        LaunchIntent,
        ServerSnapshot,
        create_launch_context,
    )

    env = Environment(environment)
    inv = Invocation(invocation)
    evidence = LaunchEnvironmentSnapshot(
        invocation=inv,
        requested_window=requested_window,
        environment=env,
        platform="linux",
        env_vars={},
        config_default=None,
        native_backend=native_backend,
        server=ServerSnapshot(8123, False, False),
        in_jupyter=env is Environment.JUPYTER,
        in_julia=env is Environment.JULIA,
        in_vscode_terminal=env is Environment.VSCODE_LOCAL,
        is_vscode_remote=env is Environment.VSCODE_REMOTE,
        in_vscode_tunnel=env is Environment.VSCODE_REMOTE and not ssh,
        ssh_connection=ssh,
        ssh_client=ssh,
        hostname="test-host",
    )
    return create_launch_context(
        LaunchIntent(inv, 8123, requested_window=requested_window),
        evidence,
    )


def _local(monkeypatch) -> None:
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(browser, "_is_vscode_remote", lambda: False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)


def test_server_id_probe_retries_transient_failure(monkeypatch):
    attempts = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "service": "arrayview",
                    "instance_id": "server-generation-a",
                }
            ).encode()

    def urlopen(*args, **kwargs):
        attempts.append(None)
        if len(attempts) < 3:
            raise TimeoutError("transient")
        return Response()

    monkeypatch.setattr(browser.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(browser.time, "sleep", lambda delay: None)

    assert (
        browser._server_id_for_url("http://localhost:8123/?sid=abc")
        == "server-generation-a"
    )
    assert len(attempts) == 3


def test_system_browser_reports_opened(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(
        browser.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 0})(),
    )

    result = browser._open_browser("http://localhost:8123/", blocking=True)

    assert result == browser.OpenResult(browser.OpenState.OPENED, "system-browser")
    assert result


def test_system_browser_reports_failed(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(
        browser.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 1})(),
    )

    result = browser._open_browser("http://localhost:8123/", blocking=True)

    assert result.state is browser.OpenState.FAILED
    assert result.mechanism == "system-browser"
    assert not result


def test_vscode_signal_reports_backend_ready(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda **kwargs: True)
    monkeypatch.setattr(
        browser, "_configure_vscode_port_preview", lambda port, **kwargs: None
    )
    request = SignalRequest("request-1", "window-1", "server-1", Path("ack"), True)
    monkeypatch.setattr(
        browser,
        "_open_via_signal_file",
        lambda *args, **kwargs: request,
    )
    monkeypatch.setattr(browser, "_server_id_for_url", lambda url: "server-1")
    monkeypatch.setattr(
        browser,
        "_wait_for_vscode_ack",
        lambda request, timeout: SimpleNamespace(state=AckState.BACKEND_READY),
    )
    result = browser._open_browser("http://localhost:8123/", blocking=True)

    assert result == browser.OpenResult(
        browser.OpenState.READY,
        "vscode-signal",
        "request-1",
    )


def test_native_fallback_bypasses_local_vscode(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    opened = []
    monkeypatch.setattr(
        browser,
        "_open_via_signal_file",
        lambda *args, **kwargs: pytest.fail("native fallback must not open a VS Code tab"),
    )
    monkeypatch.setattr(
        browser.subprocess,
        "run",
        lambda args, **kwargs: opened.append(args)
        or type("Completed", (), {"returncode": 0})(),
    )

    result = browser._open_browser(
        "http://localhost:8123/",
        blocking=True,
        prefer_system_browser=True,
    )

    assert opened == [["open", "http://localhost:8123/"]]
    assert result == browser.OpenResult(browser.OpenState.OPENED, "system-browser")


@pytest.mark.parametrize("requested_window", ["browser", "native"])
def test_planned_system_browser_cannot_be_overridden_by_vscode_detection(
    monkeypatch, requested_window
):
    context = _context(
        requested_window=requested_window,
        environment="vscode_local",
        native_backend=None,
    )
    assert context.plan.display.value == "browser"
    monkeypatch.setattr(
        browser,
        "_in_vscode_terminal",
        lambda: pytest.fail("planned route must not re-detect VS Code"),
    )
    monkeypatch.setattr(
        browser,
        "_is_vscode_remote",
        lambda: pytest.fail("planned route must not re-detect remote placement"),
    )
    monkeypatch.setattr(browser.sys, "platform", "darwin")
    monkeypatch.setattr(
        browser,
        "_open_via_signal_file",
        lambda *args, **kwargs: pytest.fail("planned browser must not signal VS Code"),
    )
    opened = []
    monkeypatch.setattr(
        browser.subprocess,
        "run",
        lambda args, **kwargs: opened.append(args)
        or type("Completed", (), {"returncode": 0})(),
    )

    result = browser._open_browser(
        "http://localhost:8123/",
        blocking=True,
        launch_context=context,
    )

    assert opened == [["open", "http://localhost:8123/"]]
    assert result.mechanism == "system-browser"


def test_planned_remote_vscode_passes_frozen_placement_to_signal(monkeypatch):
    context = _context(
        requested_window="vscode",
        environment="vscode_remote",
    )
    monkeypatch.setattr(
        browser,
        "_in_vscode_terminal",
        lambda: pytest.fail("planned route must not re-detect VS Code"),
    )
    monkeypatch.setattr(
        browser,
        "_is_vscode_remote",
        lambda: pytest.fail("planned route must not re-detect remote placement"),
    )
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda **kwargs: True)
    monkeypatch.setattr(
        browser, "_configure_vscode_port_preview", lambda port, **kwargs: None
    )
    monkeypatch.setattr(browser, "_server_id_for_url", lambda url: "server-1")
    captured = {}

    def _signal(*args, **kwargs):
        captured.update(kwargs)
        return SignalRequest("request-1", "window-1", "server-1", Path("ack"), True)

    monkeypatch.setattr(browser, "_open_via_signal_file", _signal)
    monkeypatch.setattr(
        browser,
        "_wait_for_vscode_ack",
        lambda request, timeout: SimpleNamespace(state=AckState.BACKEND_READY),
    )

    result = browser._open_browser(
        "http://localhost:8123/",
        blocking=True,
        launch_context=context,
    )

    assert captured["is_remote"] is True
    assert result.state is browser.OpenState.READY


def test_planned_plain_ssh_guidance_survives_julia_host_classification(monkeypatch):
    context = _context(
        requested_window="browser",
        environment="julia",
        invocation="julia",
        ssh=True,
    )
    monkeypatch.setattr(browser, "_ssh_message_shown", False)
    monkeypatch.setattr(
        browser,
        "_is_vscode_remote",
        lambda: pytest.fail("planned route must not re-detect remote placement"),
    )

    result = browser._open_browser(
        "http://localhost:8123/",
        blocking=True,
        launch_context=context,
    )

    assert result == browser.OpenResult(browser.OpenState.PRINTED, "ssh-guidance")


def test_python_native_failure_uses_the_planned_browser_fallback(monkeypatch):
    import arrayview._launcher as launcher

    context = _context(
        requested_window="native",
        environment="vscode_local",
        invocation="python",
        native_backend="cocoa",
    )
    assert context.plan.display.value == "native"
    assert context.plan.fallback_display.value == "browser"

    class FailedProcess:
        pid = 123
        returncode = 1
        stderr = SimpleNamespace(read=lambda: b"native failed")

        def poll(self):
            return 1

    opened = []
    monkeypatch.setattr(launcher, "_open_webview", lambda *args, **kwargs: FailedProcess())
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda url, **kwargs: opened.append((url, kwargs)),
    )

    launcher._open_webview_with_fallback(
        "http://localhost:8123/shell",
        800,
        600,
        launch_context=context,
    )

    assert opened == [
        (
            "http://localhost:8123/shell",
            {
                "blocking": True,
                "floating": False,
                "title": None,
                "launch_context": context,
                "use_fallback": True,
            },
        )
    ]


def test_python_native_failure_propagates_failed_browser_fallback(monkeypatch):
    import arrayview._launcher as launcher

    context = _context(
        requested_window="native",
        environment="terminal",
        invocation="python",
        native_backend="cocoa",
    )
    monkeypatch.setattr(
        launcher,
        "_open_webview_cli_tracked",
        lambda *args, **kwargs: (False, None),
    )
    monkeypatch.setattr(
        launcher,
        "_open_browser",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("browser handoff failed")
        ),
    )

    with pytest.raises(RuntimeError, match="browser handoff failed"):
        launcher._open_webview_with_fallback(
            "http://localhost:8123/shell",
            800,
            600,
            launch_context=context,
        )


def test_vscode_signal_reports_correlated_failure(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda **kwargs: True)
    monkeypatch.setattr(
        browser, "_configure_vscode_port_preview", lambda port, **kwargs: None
    )
    request = SignalRequest("request-1", "window-1", None, Path("ack"), True)
    monkeypatch.setattr(browser, "_open_via_signal_file", lambda *args, **kwargs: request)
    monkeypatch.setattr(browser, "_server_id_for_url", lambda url: None)
    monkeypatch.setattr(
        browser,
        "_wait_for_vscode_ack",
        lambda request, timeout: SimpleNamespace(
            state=AckState.FAILED,
            message="wrong window",
        ),
    )

    with pytest.raises(RuntimeError, match="wrong window"):
        browser._open_browser(
            "http://localhost:8123/",
            blocking=True,
            force_vscode=True,
        )


def test_stale_vscode_host_fails_before_writing_open_signal(monkeypatch):
    from arrayview import _vscode_extension as extension_state

    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(browser, "_is_vscode_remote", lambda: True)
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda **kwargs: False)
    monkeypatch.setattr(extension_state, "_VSCODE_EXT_RELOAD_REQUIRED", True)
    monkeypatch.setattr(
        browser,
        "_open_via_signal_file",
        lambda *args, **kwargs: pytest.fail("stale host must not receive an open signal"),
    )

    result = browser._open_browser("http://localhost:8123/?sid=abc", blocking=True)

    assert result.state is browser.OpenState.FAILED
    assert result.mechanism == "vscode-extension"
    assert "reload this VS Code window" in result.detail


def test_plain_ssh_reports_printed_guidance(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setenv("SSH_CONNECTION", "client server")
    monkeypatch.setattr(browser, "_ssh_message_shown", False)

    result = browser._open_browser("http://localhost:8123/", blocking=True)

    assert result == browser.OpenResult(browser.OpenState.PRINTED, "ssh-guidance")


def test_nonblocking_reports_background_acceptance(monkeypatch):
    class ThreadStub:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            pass

    monkeypatch.setattr(browser.threading, "Thread", ThreadStub)

    result = browser._open_browser("http://localhost:8123/")

    assert result == browser.OpenResult(browser.OpenState.ACCEPTED, "background-thread")


def test_vscode_request_deadline_matches_display_owner_lifetime():
    script_context = SimpleNamespace(
        caller_scope=SimpleNamespace(value="script")
    )
    interactive_context = SimpleNamespace(
        caller_scope=SimpleNamespace(value="interactive")
    )

    assert browser._vscode_request_max_age_ms(
        blocking=True, is_remote=False, launch_context=interactive_context
    ) == 14_000
    assert browser._vscode_request_max_age_ms(
        blocking=True, is_remote=True, launch_context=interactive_context
    ) == 190_000
    assert browser._vscode_request_max_age_ms(
        blocking=False, is_remote=False, launch_context=script_context
    ) == 14_000
    assert browser._vscode_request_max_age_ms(
        blocking=False, is_remote=True, launch_context=script_context
    ) == 190_000
    assert (
        browser._vscode_request_max_age_ms(
            blocking=False,
            is_remote=True,
            launch_context=interactive_context,
        )
        is None
    )
