"""A clicked file's first image is read while its tab is still opening.

The server knows about the file about half a second before the tab has loaded
the page, connected and asked for a frame. These tests pin that the slice the
viewer opens on is already cached by then, so the frame request reads nothing.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


def _viewer_first_request(meta: dict) -> tuple[int, int, list[int]]:
    """What the viewer asks for first, written out from its own startup rule:
    the metadata's axes, the middle slice on spatial axes, 0 on the rest."""
    shape = meta["shape"]
    dim_x, dim_y = meta["default_dims"]
    cases_from = meta.get("collection_spatial_ndim", len(shape))
    ref = max(shape[dim_x], shape[dim_y])
    indices = [
        size // 2
        if axis < cases_from and (axis in (dim_x, dim_y) or size >= 0.25 * ref)
        else 0
        for axis, size in enumerate(shape)
    ]
    return dim_x, dim_y, indices


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# background=True is an Explorer click; False is a command-line launch that
# found the server already running.
@pytest.mark.parametrize("background", [True, False])
@pytest.mark.parametrize(
    "shape",
    [(24, 32, 20), (24, 32, 20, 6), (5, 40, 48, 44), (30, 36)],
)
def test_load_has_the_first_image_cached_before_the_viewer_asks(
    tmp_path, monkeypatch, shape, background
):
    import arrayview._render as render_mod
    import arrayview._session as session_mod
    from arrayview._app import app

    path = tmp_path / "clicked.npy"
    np.save(path, np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape))

    with TestClient(app) as client:
        sid = client.post(
            "/load",
            json={
                "filepath": str(path),
                "background": background,
                "release_on_disconnect": True,
            },
        ).json()["sid"]
        try:
            assert _wait_for(lambda: sid in session_mod.SESSIONS)
            session = session_mod.SESSIONS[sid]
            meta = client.get(f"/metadata/{sid}").json()
            dim_x, dim_y, indices = _viewer_first_request(meta)
            key = (
                dim_x,
                dim_y,
                tuple(
                    None if axis in (dim_x, dim_y) else index
                    for axis, index in enumerate(indices)
                ),
            )
            # No viewer has connected: only the load itself can have read it.
            assert _wait_for(lambda: key in session.raw_cache), (
                key,
                list(session.raw_cache),
            )

            reads = []
            read_slice = render_mod._read_slice

            def counting_read(session, dim_x, dim_y, idx_list, progress):
                reads.append((dim_x, dim_y, list(idx_list)))
                return read_slice(session, dim_x, dim_y, idx_list, progress)

            monkeypatch.setattr(render_mod, "_read_slice", counting_read)
            with client.websocket_connect(f"/ws/{sid}") as ws:
                assert ws.receive_json()["type"] == "metadata"
                ws.send_json(
                    {
                        "seq": 1,
                        "dim_x": dim_x,
                        "dim_y": dim_y,
                        "dim_z": -1,
                        "indices": indices,
                        "colormap": "gray",
                        "dr": 0,
                        "complex_mode": 0,
                        "log_scale": False,
                        "slice_dim": 0,
                        "direction": 1,
                    }
                )
                assert len(ws.receive_bytes()) > 20
            # Reads of neighbouring slices (lookahead) are fine; the image the
            # viewer asked for must not have been read again.
            assert (dim_x, dim_y, indices) not in reads
        finally:
            session_mod.SESSIONS.pop(sid, None)


def test_startup_view_keeps_a_long_case_axis_on_its_first_case():
    from arrayview._analysis import startup_view
    from arrayview._session import Session

    session = Session(np.zeros((24, 32, 20, 80), dtype=np.float32))
    session.collection_spatial_ndim = 3
    dim_x, dim_y, indices = startup_view(session)
    assert indices[3] == 0
    assert indices[2] == 10
    assert {dim_x, dim_y} <= {0, 1, 2}


def test_startup_view_stands_aside_when_the_launch_picks_its_own_view():
    from arrayview._analysis import startup_view, warm_first_slice
    from arrayview._session import Session

    session = Session(np.zeros((24, 32, 20), dtype=np.float32))
    session.initial_view = {"dims": [0, 2]}
    assert startup_view(session) is None
    assert warm_first_slice(session) is False
    assert not session.raw_cache


def test_a_frame_request_that_joins_the_warm_up_still_sees_its_progress():
    """A slow first read must not go silent because the warm-up started it."""
    from arrayview._analysis import warm_first_slice
    from arrayview._render import extract_slice
    from arrayview._session import Session

    reading = threading.Event()
    release = threading.Event()

    class SlowSource:
        shape = (24, 32, 20)
        dtype = np.dtype(np.float32)
        ndim = 3

        def read_slice(self, slicer, progress):
            reading.set()
            assert release.wait(5.0)
            progress(1, 2)
            return np.zeros((24, 32), dtype=np.float32)

    session = Session(SlowSource())
    session._ram_upgrade_started = True
    assert warm_first_slice(session) is True
    assert reading.wait(5.0)

    # The viewer's frame request arrives mid-read and installs its reporter.
    reported = []
    session.progress_sink = lambda done, total: reported.append((done, total))
    joined = []
    asker = threading.Thread(
        target=lambda: joined.append(extract_slice(session, 0, 1, [12, 16, 10]))
    )
    asker.start()
    release.set()
    asker.join(5.0)

    assert len(joined) == 1 and joined[0].shape == (32, 24)
    assert reported == [(1, 2)]
