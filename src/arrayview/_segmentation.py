"""nnInteractive segmentation client — pure HTTP, no nnInteractive dependency."""

from __future__ import annotations

import gzip
import io
import logging
import shutil
import subprocess
import time

import httpx
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1527
_CONNECT_TIMEOUT = 2.0
_LAUNCH_TIMEOUT = 60.0
_REQUEST_TIMEOUT = 120.0

_base_url: str | None = None
_volume_shape: tuple[int, ...] | None = None
_server_process: subprocess.Popen | None = None


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _url(path: str) -> str:
    assert _base_url is not None, "not connected"
    return f"{_base_url}{path}"


def is_connected() -> bool:
    return _base_url is not None


def try_connect(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if an nnInteractive server is reachable."""
    global _base_url
    url = f"http://{host}:{port}"
    try:
        r = httpx.get(f"{url}/docs", timeout=_CONNECT_TIMEOUT)
        if r.status_code < 500:
            _base_url = url
            return True
    except (httpx.ConnectError, httpx.TimeoutException):
        pass
    return False


def try_launch(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str | None:
    """Try to launch nnInteractive server, return error message or None."""
    global _server_process

    # Try uvx first (no pre-install needed), then direct command
    cmd_name = "nninteractive-slicer-server"
    candidates: list[list[str]] = []
    if shutil.which("uvx"):
        candidates.append(["uvx", cmd_name, "--host", host, "--port", str(port)])
    if shutil.which(cmd_name):
        candidates.append([cmd_name, "--host", host, "--port", str(port)])

    if not candidates:
        return (
            "nnInteractive not found. Install uv (pip install uv) "
            "or nninteractive-slicer-server (pip install nninteractive-slicer-server)"
        )

    for cmd in candidates:
        log.info("launching nnInteractive: %s", " ".join(cmd))
        try:
            _server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            continue

        # Wait for server to come up
        deadline = time.monotonic() + _LAUNCH_TIMEOUT
        while time.monotonic() < deadline:
            if _server_process.poll() is not None:
                break  # process exited — try next candidate
            if try_connect(host, port):
                return None  # success
            time.sleep(0.5)

        # Didn't connect — kill and try next
        _server_process.terminate()
        _server_process = None

    return "nnInteractive server failed to start (is CUDA available?)"


def disconnect() -> None:
    """Disconnect and optionally kill launched server."""
    global _base_url, _volume_shape, _server_process
    _base_url = None
    _volume_shape = None
    if _server_process is not None:
        _server_process.terminate()
        _server_process = None


# ---------------------------------------------------------------------------
# Volume upload
# ---------------------------------------------------------------------------

def upload_volume(data: np.ndarray) -> None:
    """Upload a 3D volume to the nnInteractive server as .npy bytes."""
    global _volume_shape
    if data.ndim != 3:
        raise ValueError(f"nnInteractive requires 3D data, got {data.ndim}D")
    buf = io.BytesIO()
    np.save(buf, data)
    _volume_shape = data.shape
    r = httpx.post(
        _url("/upload_image"),
        files={"file": ("volume.npy", buf.getvalue(), "application/octet-stream")},
        timeout=_REQUEST_TIMEOUT,
    )
    r.raise_for_status()


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

def _decode_mask(content: bytes) -> np.ndarray:
    """Decode gzip-compressed packbits mask from server response."""
    assert _volume_shape is not None
    # Server sends Content-Encoding: gzip but httpx may or may not auto-decompress
    try:
        raw = gzip.decompress(content)
    except gzip.BadGzipFile:
        raw = content
    total = int(np.prod(_volume_shape))
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:total]
    return bits.reshape(_volume_shape).astype(np.uint8)


def add_point(coord_zyx: tuple[int, int, int], positive: bool = True) -> np.ndarray:
    """Send a point interaction, return 3D uint8 mask."""
    r = httpx.post(
        _url("/add_point_interaction"),
        json={"voxel_coord": list(coord_zyx), "positive_click": positive},
        timeout=_REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return _decode_mask(r.content)


def add_bbox(
    corner1_zyx: tuple[int, int, int],
    corner2_zyx: tuple[int, int, int],
    positive: bool = True,
) -> np.ndarray:
    """Send a bounding box interaction, return 3D uint8 mask."""
    r = httpx.post(
        _url("/add_bbox_interaction"),
        json={
            "outer_point_one": list(corner1_zyx),
            "outer_point_two": list(corner2_zyx),
            "positive_click": positive,
        },
        timeout=_REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return _decode_mask(r.content)


def _send_mask_interaction(
    endpoint: str, mask_3d: np.ndarray, positive: bool = True,
) -> np.ndarray:
    """Send a 3D binary mask interaction (scribble or lasso), return result mask."""
    buf = io.BytesIO()
    np.save(buf, mask_3d.astype(np.uint8))
    compressed = gzip.compress(buf.getvalue())
    r = httpx.post(
        _url(endpoint),
        files={"file": ("volume.npy.gz", compressed, "application/octet-stream")},
        data={"positive_click": str(positive).lower()},
        timeout=_REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return _decode_mask(r.content)


def add_scribble(mask_3d: np.ndarray, positive: bool = True) -> np.ndarray:
    """Send a scribble interaction (3D mask with marks on one slice)."""
    return _send_mask_interaction("/add_scribble_interaction", mask_3d, positive)


def add_lasso(mask_3d: np.ndarray, positive: bool = True) -> np.ndarray:
    """Send a lasso interaction (3D mask with filled contour on one slice)."""
    return _send_mask_interaction("/add_lasso_interaction", mask_3d, positive)


def reset_interactions() -> None:
    """Upload a zeroed mask to reset all interactions."""
    assert _volume_shape is not None
    zeros = np.zeros(_volume_shape, dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, zeros)
    compressed = gzip.compress(buf.getvalue())
    r = httpx.post(
        _url("/upload_segment"),
        files={"file": ("volume.npy.gz", compressed, "application/octet-stream")},
        timeout=_REQUEST_TIMEOUT,
    )
    r.raise_for_status()
