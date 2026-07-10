import json
import os
import time

import arrayview._vscode_signal as signal


def _request(tmp_path, **overrides):
    values = {
        "request_id": "req-1",
        "window_id": "window-1",
        "server_id": "server-1",
        "ack_path": tmp_path / "ack.json",
        "written": True,
    }
    values.update(overrides)
    return signal.SignalRequest(**values)


def test_open_request_includes_versioned_ack_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: "window-1")
    captured = []
    monkeypatch.setattr(
        signal,
        "_write_vscode_signal",
        lambda payload, delay=0.0: captured.append(payload) or True,
    )

    request = signal._open_via_signal_file(
        "http://localhost:8123/?sid=session-1", server_id="server-1"
    )

    assert request.written is True
    assert request.window_id == "window-1"
    assert request.server_id == "server-1"
    assert request.ack_path.name == f"open-ack-v0100-{request.request_id}.json"
    assert captured == [
        {
            "action": "open-preview",
            "url": "http://localhost:8123/?sid=session-1",
            "maxAgeMs": signal._VSCODE_SIGNAL_MAX_AGE_MS,
            "protocolVersion": 1,
            "requestId": request.request_id,
            "ackPath": str(request.ack_path),
            "windowId": "window-1",
            "serverId": "server-1",
        }
    ]


def test_open_request_reports_failed_legacy_write(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(signal, "_find_arrayview_window_id", lambda: None)
    monkeypatch.setattr(signal, "_write_vscode_signal", lambda *args, **kwargs: False)

    request = signal._open_via_signal_file("http://localhost:8123/")

    assert request.written is False
    assert not request


def test_wait_accepts_correlated_ack(tmp_path):
    request = _request(tmp_path)
    request.ack_path.write_text(
        json.dumps(
            {
                "state": "backend_ready",
                "requestId": "req-1",
                "windowId": "window-1",
                "serverId": "server-1",
            }
        )
    )

    result = signal._wait_for_vscode_ack(request, timeout=0)

    assert result.state is signal.AckState.BACKEND_READY
    assert result.request_id == "req-1"
    assert result.window_id == "window-1"
    assert result.server_id == "server-1"


def test_wait_rejects_each_correlation_mismatch(tmp_path):
    request = _request(tmp_path)
    valid = {
        "state": "backend_ready",
        "requestId": "req-1",
        "windowId": "window-1",
        "serverId": "server-1",
    }
    for field in ("requestId", "windowId", "serverId"):
        request.ack_path.write_text(json.dumps({**valid, field: "wrong"}))
        result = signal._wait_for_vscode_ack(request, timeout=0)
        assert result.state is signal.AckState.INVALID
        assert field in result.message


def test_wait_is_bounded_when_ack_never_arrives(tmp_path):
    request = _request(tmp_path)
    started = time.monotonic()

    result = signal._wait_for_vscode_ack(request, timeout=0.02, poll_interval=0.005)

    assert result.state is signal.AckState.TIMEOUT
    assert time.monotonic() - started < 0.2


def test_wait_does_not_treat_progress_as_success(tmp_path):
    request = _request(tmp_path)
    request.ack_path.write_text(
        json.dumps(
            {
                "state": "panel_opened",
                "requestId": "req-1",
                "windowId": "window-1",
                "serverId": "server-1",
            }
        )
    )

    result = signal._wait_for_vscode_ack(
        request,
        timeout=0.02,
        poll_interval=0.005,
    )

    assert result.state is signal.AckState.TIMEOUT
    assert result.message == "last extension state: panel_opened"


def test_wait_rejects_malformed_or_unknown_ack(tmp_path):
    request = _request(tmp_path)
    for content in ("not json", json.dumps({"state": "future-state"})):
        request.ack_path.write_text(content)
        result = signal._wait_for_vscode_ack(request, timeout=0)
        assert result.state is signal.AckState.INVALID


def test_cleanup_removes_only_stale_ack_files(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    signal_dir = tmp_path / ".arrayview"
    signal_dir.mkdir()
    stale = signal_dir / "open-ack-v0100-stale.json"
    fresh = signal_dir / "open-ack-v0100-fresh.json"
    unrelated = signal_dir / "window-1.json"
    for path in (stale, fresh, unrelated):
        path.write_text("{}")
    old = time.time() - 600
    os.utime(stale, (old, old))

    removed = signal._cleanup_stale_vscode_acks(max_age_seconds=300)

    assert removed == 1
    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()
