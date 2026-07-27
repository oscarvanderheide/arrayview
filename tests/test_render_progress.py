"""Progress must be shown while a frame is still being produced.

Reported case: `arrayview recons/parameter_maps_all.npy`, shape
(224,240,204,6,9) float32 on a network mount, sat on a bare spinner for 31s.
Measured: reading the single displayed 215 KB slice takes ~29s and faults in
2.2 GB, because the wanted values are 4 bytes out of every 216-byte block and
the pages between them amount to the whole file. render_rgba itself is 0.2s.

Two things follow, and both are asserted here. Progress is gated on ELAPSED
TIME, not on how much work there is: a 60-slice mosaic of an in-memory array
finishes in milliseconds while one slice of that memmap takes half a minute,
so work volume predicts nothing. And a single slice has no loop to report
from, so the read is chunked along its outer axis.
"""

import time

import numpy as np
import pytest

from conftest import register_array

_CAPTURE = """
const origWS = window.WebSocket;
window.WebSocket = function (...args) {
    const ws = new origWS(...args);
    ws.addEventListener('message', (event) => {
        if (typeof event.data !== 'string') return;
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'render_progress') {
                window._recordProgress(msg.done, msg.total, msg.label || '');
            }
        } catch (_) {}
    });
    return ws;
};
window.WebSocket.prototype = origWS.prototype;
"""


@pytest.fixture
def ungated(monkeypatch):
    """Remove the time gate so the real pipeline can be observed quickly.

    The gate itself is covered separately below; suppressing it here keeps
    these tests about whether progress actually flows end to end.
    """
    import arrayview._routes_websocket as ws_mod

    original = ws_mod._render_progress_reporter
    monkeypatch.setattr(
        ws_mod,
        "_render_progress_reporter",
        lambda loop, ws, label="Loading", quiet_for_s=0.6, min_interval_s=0.2: (
            original(loop, ws, label, 0.0, 0.0)
        ),
    )


def _watch(page, seen):
    page.expose_function(
        "_recordProgress", lambda d, t, label: seen.append((d, t, label))
    )
    page.add_init_script(_CAPTURE)


def test_single_slice_render_is_wired_for_progress(client, tmp_path, monkeypatch):
    """The reported case: one slice, no mosaic, and so no loop of its own.

    Asserted at the wiring level rather than through the browser because the
    HTTP render route warms the caches first on a small test array, so a
    browser-driven assertion would pass or fail on cache timing rather than on
    whether the websocket render path reports at all.
    """
    import arrayview._render as render_mod

    calls = []
    original = render_mod._extract_with_progress

    def spy(data, slicer, progress, target_updates=32):
        calls.append(progress is not None)
        return original(data, slicer, progress, target_updates)

    monkeypatch.setattr(render_mod, "_extract_with_progress", spy)

    path = tmp_path / "wired.npy"
    np.save(path, np.random.rand(40, 40, 6, 3).astype(np.float32))
    resp = client.post("/load", json={"filepath": str(path), "name": "wired"})
    resp.raise_for_status()
    sid = resp.json()["sid"]

    from arrayview import _session as session_mod

    session = session_mod.SESSIONS[sid]
    render_mod.render_rgba(
        session, 0, 1, (0, 0, 0, 0), "gray", 1.0, 0, False, None, None,
        progress=lambda done, total: None,
    )
    assert calls and calls[-1], (
        "render_rgba must pass its progress callback down to the slice read"
    )


def test_mosaic_build_reports_progress(page, client, server_url, tmp_path, ungated):
    seen = []
    _watch(page, seen)
    sid = register_array(
        client, np.random.rand(16, 16, 60, 3).astype(np.float32), tmp_path, "mosaic_prog"
    )
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=45_000)
    seen.clear()
    page.focus("#keyboard-sink")
    page.keyboard.press("z")
    page.wait_for_timeout(2500)
    assert page.evaluate("() => dim_z") >= 0, "the test must actually enter mosaic mode"

    assert seen, "a mosaic build must report progress"
    assert {t for _, t, _ in seen} == {60}, f"total must be the slice count: {seen}"


def test_fast_render_shows_nothing(page, client, server_url, tmp_path):
    """With the real gate in place, a quick load must not flash a bar."""
    seen = []
    _watch(page, seen)
    sid = register_array(
        client, np.random.rand(32, 32).astype(np.float32), tmp_path, "fast"
    )
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=30_000)
    page.wait_for_timeout(1500)
    assert seen == [], f"a fast render must stay silent, got {seen}"


def test_gate_suppresses_until_the_work_is_slow():
    """The gate in isolation: silent while quick, reporting once slow."""
    import arrayview._routes_websocket as ws_mod

    sent = []

    class _Loop:
        def call_soon_threadsafe(self, fn):
            sent.append(fn)

    report = ws_mod._render_progress_reporter(
        _Loop(), object(), label="Reading", quiet_for_s=0.25, min_interval_s=0.0
    )
    report(1, 10)
    assert sent == [], "nothing may be sent while the work still looks fast"
    time.sleep(0.3)
    report(2, 10)
    assert len(sent) == 1, "once past the quiet period progress must be reported"


def test_chunked_extract_matches_a_plain_read():
    """Chunking must change only the reporting, never the data."""
    from arrayview._render import _extract_with_progress

    data = np.arange(9 * 7 * 5, dtype=np.float32).reshape(9, 7, 5)
    slicer = [slice(None), slice(None), 2]
    calls = []
    chunked = _extract_with_progress(data, slicer, lambda d, t: calls.append((d, t)))
    plain = _extract_with_progress(data, slicer, None)

    np.testing.assert_array_equal(chunked, plain)
    assert calls, "a chunked read must report"
    assert calls[-1] == (9, 9), f"the final update must be complete, got {calls[-1]}"
