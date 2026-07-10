import os

from fastapi.testclient import TestClient

import arrayview._server as server
import arrayview._session as session_mod


def test_ping_preserves_compatibility_and_reports_stable_runtime(monkeypatch):
    original = session_mod.SERVER_RUNTIME
    monkeypatch.setattr(session_mod, "VIEWER_SOCKETS", 2)
    monkeypatch.setattr(session_mod, "SHELL_SOCKETS", [object()])
    monkeypatch.setattr(server, "SESSIONS", {"one": object(), "two": object()})
    server.configure_server_runtime(
        instance_id="test-instance",
        process_start="test-process-start",
        owner_mode="kernel",
        started_at=123.5,
        port=8123,
    )
    try:
        with TestClient(server.app) as client:
            first = client.get("/ping").json()
            second = client.get("/ping").json()
    finally:
        monkeypatch.setattr(session_mod, "SERVER_RUNTIME", original)

    assert first == second
    assert first["ok"] is True
    assert first["service"] == "arrayview"
    assert first["pid"] == os.getpid()
    assert first["viewer_sockets"] == 2
    assert first["shell_sockets"] == 1
    assert first["instance_id"] == "test-instance"
    assert first["process_start"] == "test-process-start"
    assert first["owner_mode"] == "kernel"
    assert first["started_at"] == 123.5
    assert first["port"] == 8123
    assert first["protocol_version"] == server.SERVER_PROTOCOL_VERSION
    assert first["package_version"] == server._av_version
    assert first["capabilities"] == list(server.SERVER_CAPABILITIES)
    assert first["active_sessions"] == 2
    assert first["active_viewer_sockets"] == 2
    assert first["active_shell_sockets"] == 1


def test_status_matches_ping():
    with TestClient(server.app) as client:
        assert client.get("/status").json() == client.get("/ping").json()


def test_configure_runtime_retains_identity_when_setting_known_port(monkeypatch):
    original = session_mod.SERVER_RUNTIME
    monkeypatch.setattr(session_mod, "SERVER_RUNTIME", original)

    updated = server.configure_server_runtime(port=9000, owner_mode="transient")

    assert updated.instance_id == original.instance_id
    assert updated.process_start == original.process_start
    assert updated.started_at == original.started_at
    assert updated.port == 9000
    assert updated.owner_mode == "transient"
