from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from arrayview import _vscode_browser as browser
from arrayview._vscode_signal import AckState, SignalRequest


def _local(monkeypatch) -> None:
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: False)
    monkeypatch.setattr(browser, "_is_vscode_remote", lambda: False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)


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
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda: True)
    monkeypatch.setattr(browser, "_configure_vscode_port_preview", lambda port: None)
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
    monkeypatch.setattr(browser, "_schedule_remote_open_retries", lambda *args, **kwargs: None)

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


def test_vscode_signal_reports_correlated_failure(monkeypatch):
    _local(monkeypatch)
    monkeypatch.setattr(browser, "_in_vscode_terminal", lambda: True)
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda: True)
    monkeypatch.setattr(browser, "_configure_vscode_port_preview", lambda port: None)
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
    monkeypatch.setattr(browser, "_ensure_vscode_extension", lambda: False)
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
