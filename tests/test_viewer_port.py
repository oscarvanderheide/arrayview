"""The private viewer port must not leave connections for VS Code to pool.

VS Code's integrated browser reaches a tunnel viewer through a local proxy that
pools one stream per ``host:port`` and reuses it. A stream this server has
closed for idleness looks alive to that pool, and the next page request written
into it is never delivered: the blank tab. ``Connection: close`` on every
response makes the proxy drop the stream instead of pooling it. These tests
drive the real server over real sockets.
"""

from __future__ import annotations

import json
import socket
import uuid

import httpx
import pytest


def _lease_viewer_port(client: httpx.Client) -> int:
    server_id = client.get("/ping").json()["instance_id"]
    lease = client.post(
        "/cold-start-port",
        json={
            "requestId": uuid.uuid4().hex,
            "ttlMs": 3000,
            "expectedServerId": server_id,
        },
    )
    assert lease.status_code == 200, lease.text
    return int(lease.json()["port"])


def _keep_alive_get(sock: socket.socket, path: str) -> tuple[dict[str, str], bytes]:
    """One HTTP/1.1 request that asks to keep the connection, fully read."""
    sock.sendall(
        f"GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n".encode()
    )
    raw = b""
    while b"\r\n\r\n" not in raw:
        chunk = sock.recv(65536)
        if not chunk:
            break
        raw += chunk
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    assert lines[0].startswith("HTTP/1.1 200"), lines[0]
    headers = {
        name.strip().lower(): value.strip()
        for name, _, value in (line.partition(":") for line in lines[1:])
    }
    length = int(headers["content-length"])
    while len(body) < length:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return headers, body


def test_viewer_port_closes_each_connection_after_its_response(client):
    port = _lease_viewer_port(client)
    with socket.create_connection(("localhost", port), timeout=5) as sock:
        headers, body = _keep_alive_get(sock, "/ping")
        assert json.loads(body)["service"] == "arrayview"
        assert headers.get("connection") == "close"
        # The server hangs up itself: nothing is left open to be reused later.
        sock.settimeout(5)
        assert sock.recv(1) == b""


def test_main_port_still_keeps_connections_alive(server_url):
    """Only viewer ports change; every other client keeps ordinary keep-alive."""
    host, port = server_url.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        first, _ = _keep_alive_get(sock, "/ping")
        assert first.get("connection") != "close"
        second, body = _keep_alive_get(sock, "/ping")
        assert json.loads(body)["service"] == "arrayview"
        assert second.get("connection") != "close"


def test_viewer_port_still_upgrades_websockets(client):
    websockets = pytest.importorskip("websockets.sync.client")
    port = _lease_viewer_port(client)
    with websockets.connect(f"ws://localhost:{port}/ws/shell", open_timeout=5) as ws:
        assert ws.protocol.state.name == "OPEN"
