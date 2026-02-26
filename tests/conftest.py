"""Shared fixtures for arrayview tests."""
import socket
import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn

from arrayview._app import app


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_url():
    """Start a real uvicorn server once for the whole test session."""
    port = _find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            r = httpx.get(f"{base}/ping", timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Test server did not start in time")

    yield base

    srv.should_exit = True


@pytest.fixture
def client(server_url):
    return httpx.Client(base_url=server_url, timeout=15.0)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def arr_2d() -> np.ndarray:
    """100×80 float32 linear gradient, min=0 max=1."""
    return np.linspace(0.0, 1.0, 100 * 80, dtype=np.float32).reshape(100, 80)


@pytest.fixture
def arr_3d() -> np.ndarray:
    """20×64×64 float32 array."""
    return np.random.default_rng(42).standard_normal((20, 64, 64)).astype(np.float32)


@pytest.fixture
def arr_4d() -> np.ndarray:
    """5×20×32×32 float32 array."""
    return np.random.default_rng(7).standard_normal((5, 20, 32, 32)).astype(np.float32)


def register_array(client: httpx.Client, arr: np.ndarray, tmp_path, name: str) -> str:
    """Save arr to a temp .npy file, POST to /load, return the sid."""
    path = tmp_path / f"{name}.npy"
    np.save(path, arr)
    resp = client.post("/load", json={"filepath": str(path), "name": name})
    resp.raise_for_status()
    body = resp.json()
    assert "sid" in body, f"/load returned error: {body}"
    return body["sid"]


@pytest.fixture
def sid_2d(client, arr_2d, tmp_path):
    return register_array(client, arr_2d, tmp_path, "arr2d")


@pytest.fixture
def sid_3d(client, arr_3d, tmp_path):
    return register_array(client, arr_3d, tmp_path, "arr3d")


@pytest.fixture
def sid_4d(client, arr_4d, tmp_path):
    return register_array(client, arr_4d, tmp_path, "arr4d")


# ---------------------------------------------------------------------------
# Browser helper (pytest-playwright `page` fixture is injected automatically)
# ---------------------------------------------------------------------------

@pytest.fixture
def loaded_viewer(page, server_url):
    """
    Factory: navigate the Playwright page to the viewer for a given sid,
    wait until the canvas is visible (WebSocket render complete), return page.

    Usage:
        def test_foo(loaded_viewer, sid_2d):
            page = loaded_viewer(sid_2d)
            ...
    """
    def _load(sid: str):
        page.goto(f"{server_url}/?sid={sid}")
        # Wait for the loading overlay to be replaced by the canvas
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        # Small grace period for the first WebSocket frame to paint
        page.wait_for_timeout(400)
        return page

    return _load
