"""Regression coverage for 1-D arrays.

Reported 2026-07-27: opening a (11289600,) complex64 file reached
viewer-phase mode-change and then hung until the opener's timeout, with no
error anywhere in the UI. The server was raising
``ValueError: buffer is not large enough`` on every frame.

Cause: a 1-D array yields a 1-D slice, so the colormapped result was
``(N, 4)`` rather than ``(H, W, 4)``. Consumers read ``h, w = rgba.shape[:2]``
and so took the 4 RGBA channels to be the image width, advertising a
4-pixel-wide frame whose buffer could not hold it. Size was never the trigger:
a 10,000-element array failed identically.
"""

import numpy as np
import pytest

from conftest import register_array


@pytest.mark.parametrize(
    "name,shape",
    [
        ("oned_tiny", (16,)),
        ("oned_small", (10_000,)),
        # The reported shape: large enough to take the downscale path where
        # PIL, not numpy, was the component that rejected the buffer.
        ("oned_reported", (11_289_600,)),
    ],
)
def test_one_dimensional_array_renders(
    page, client, server_url, tmp_path, name, shape
):
    arr = np.zeros(shape, dtype=np.complex64)
    sid = register_array(client, arr, tmp_path, name)
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=45_000)


def test_one_dimensional_slice_is_promoted_to_a_row(client, arr_2d, tmp_path):
    """The invariant the render pipeline depends on, checked directly."""
    from arrayview._render import extract_slice
    from arrayview import _session as session_mod

    path = tmp_path / "oned.npy"
    np.save(path, np.arange(64, dtype=np.float32))
    resp = client.post("/load", json={"filepath": str(path), "name": "oned"})
    resp.raise_for_status()
    sid = resp.json()["sid"]

    session = session_mod.SESSIONS[sid]
    sliced = extract_slice(session, 0, 0, [0])
    assert sliced.ndim == 2, "a 1-D slice must be presented as a single row"
    assert sliced.shape[0] == 1
    assert sliced.shape[1] == 64
