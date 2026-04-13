"""Unit tests for the pre-server loading page (cold-start UX)."""
import socket
import threading
import time
import urllib.request

import pytest

from arrayview._launcher import (
    _LOADING_HTML,
    _LoadingHandler,
    _run_loading_server,
    _with_loading,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _start_loading_server(timeout: float = 5.0) -> tuple[socket.socket, int]:
    """Bind a loading server on an ephemeral port, return (sock, port)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    port = sock.getsockname()[1]
    threading.Thread(
        target=_run_loading_server, args=(sock,), kwargs={"timeout": timeout}, daemon=True
    ).start()
    return sock, port


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_loading_html_has_required_content():
    """_LOADING_HTML contains the spinner markup, dark background, and JS poller."""
    assert "#0c0c0c" in _LOADING_HTML
    assert "av-spinner" in _LOADING_HTML
    assert "Loading ArrayView" in _LOADING_HTML
    assert "location.search" in _LOADING_HTML  # JS poller reads ?target=
    assert "window.location.replace" in _LOADING_HTML


def test_loading_server_responds_200():
    """_run_loading_server accepts connections and returns 200 with loading HTML."""
    _, port = _start_loading_server()
    time.sleep(0.05)  # give thread time to enter accept()

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as resp:
        assert resp.status == 200
        body = resp.read().decode()

    assert "Loading ArrayView" in body
    assert "av-spinner" in body


def test_loading_server_handles_multiple_requests():
    """Loading server continues serving after the first request."""
    _, port = _start_loading_server()
    time.sleep(0.05)

    for _ in range(3):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3) as resp:
            assert resp.status == 200


def test_with_loading_returns_unchanged_when_no_loading_port():
    """_with_loading is a no-op when _loading_port is None."""
    import arrayview._launcher as L
    original = L._loading_port
    try:
        L._loading_port = None
        url = "http://localhost:7778/?sid=abc123"
        assert L._with_loading(url) == url
    finally:
        L._loading_port = original


def test_with_loading_wraps_url_when_port_set():
    """_with_loading wraps the URL when _loading_port is set."""
    import arrayview._launcher as L
    original = L._loading_port
    try:
        L._loading_port = 9999
        url = "http://localhost:7778/?sid=abc123"
        result = L._with_loading(url)
        assert result.startswith("http://127.0.0.1:9999/")
        assert "target=" in result
        assert "localhost" in result  # target URL is encoded in the loading URL
    finally:
        L._loading_port = original


def test_with_loading_url_encodes_target():
    """The target URL is percent-encoded so it's safe in a query string."""
    import arrayview._launcher as L
    original = L._loading_port
    try:
        L._loading_port = 9999
        url = "http://localhost:7778/?sid=abc&compare=xyz"
        result = L._with_loading(url)
        # & in the target must be encoded
        assert "&compare=xyz" not in result  # raw & would break query string
        assert "%26" in result or "compare" in result  # encoded form present
    finally:
        L._loading_port = original
