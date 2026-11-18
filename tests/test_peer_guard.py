"""Other users on the same machine are refused; the owner is never affected."""

import asyncio
import os
import socket
import sys

import pytest

from arrayview import _peer_guard

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="owner lookup is Linux-only"
)


@linux_only
@pytest.mark.parametrize("family,host", [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_peer_uid_of_own_loopback_connection(family, host):
    try:
        srv = socket.socket(family)
        srv.bind((host, 0))
    except OSError:
        pytest.skip(f"{host} unavailable")
    srv.listen()
    client = socket.create_connection((host, srv.getsockname()[1]))
    conn, addr = srv.accept()
    try:
        assert _peer_guard.peer_uid(addr[:2], conn.getsockname()[:2]) == os.geteuid()
        assert not _peer_guard.is_foreign_peer(addr[:2], conn.getsockname()[:2])
    finally:
        client.close()
        conn.close()
        srv.close()


def _run(scope, monkeypatch, uid):
    monkeypatch.setattr(_peer_guard, "peer_uid", lambda c, s: uid)
    monkeypatch.setattr(_peer_guard.sys, "platform", "linux")
    reached = []
    sent = []

    async def app(scope, receive, send):
        reached.append(scope["path"])

    async def send(message):
        sent.append(message)

    async def receive():
        return {}

    # Own thread: browser tests in the same session leave a loop running here.
    import threading

    worker = threading.Thread(
        target=lambda: asyncio.run(_peer_guard.PeerOwnerGuard(app)(scope, receive, send))
    )
    worker.start()
    worker.join()
    return reached, sent


def _scope(kind="http", path="/load"):
    return {"type": kind, "path": path, "client": ("127.0.0.1", 5555), "server": ("127.0.0.1", 8000)}


def test_other_user_is_refused_over_http(monkeypatch):
    reached, sent = _run(_scope(), monkeypatch, os.geteuid() + 1)
    assert reached == []
    assert sent[0]["status"] == 403


def test_other_user_is_refused_over_websocket(monkeypatch):
    reached, sent = _run(_scope("websocket", "/ws/abc"), monkeypatch, os.geteuid() + 1)
    assert reached == []
    assert sent == [{"type": "websocket.close", "code": 4403}]


def test_ping_stays_open_to_other_users(monkeypatch):
    reached, _ = _run(_scope(path="/ping"), monkeypatch, os.geteuid() + 1)
    assert reached == ["/ping"]


@pytest.mark.parametrize("uid", ["self", 0, None])
def test_owner_root_and_unknown_peers_pass(monkeypatch, uid):
    uid = os.geteuid() if uid == "self" else uid
    reached, _ = _run(_scope(), monkeypatch, uid)
    assert reached == ["/load"]


def test_opt_out_allows_other_users(monkeypatch):
    monkeypatch.setenv("ARRAYVIEW_ALLOW_OTHER_USERS", "1")
    reached, _ = _run(_scope(), monkeypatch, os.geteuid() + 1)
    assert reached == ["/load"]


def test_non_loopback_and_test_clients_are_not_judged():
    assert _peer_guard.peer_uid(("10.0.0.5", 4000), ("10.0.0.1", 8000)) is None
    assert _peer_guard.peer_uid(("testclient", 50000), ("testserver", 80)) is None
