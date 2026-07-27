"""The viewer must show real progress while a large mosaic is being built.

A mosaic over an array whose display axes are the outer ones spends all of its
time before the first frame exists: a (224,240,204,6,9) file on a network mount
took 31s to first paint, and for all of it the viewer showed only a spinner —
indistinguishable from a hang, which is exactly how it was reported.

Progress is per slice because that is honest rather than decorative: each slice
faults in roughly the same number of pages, so slice i of n is a true fraction
of the remaining wait.
"""

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
                window._recordProgress(msg.done, msg.total);
            }
        } catch (_) {}
    });
    return ws;
};
window.WebSocket.prototype = origWS.prototype;
"""


def _build_mosaic(page, client, server_url, tmp_path, arr, name, seen):
    """Load the array, then enter mosaic mode — the build under test."""
    page.expose_function("_recordProgress", lambda d, t: seen.append((d, t)))
    page.add_init_script(_CAPTURE)
    sid = register_array(client, arr, tmp_path, name)
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=45_000)
    page.focus("#keyboard-sink")
    page.keyboard.press("z")
    page.wait_for_timeout(3000)
    assert page.evaluate("() => dim_z") >= 0, "the test must actually enter mosaic mode"


def test_progress_is_reported_for_a_large_mosaic(page, client, server_url, tmp_path):
    seen = []
    _build_mosaic(
        page, client, server_url, tmp_path,
        np.random.rand(16, 16, 60, 3).astype(np.float32),
        "mosaic_progress", seen,
    )

    assert seen, "a large mosaic must report progress before its frame arrives"
    for done, total in seen:
        assert total == 60, f"total must be the slice count, got {total}"
        assert 1 <= done <= total, f"progress out of range: {done}/{total}"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), (
        "progress must never go backwards"
    )


def test_small_mosaic_reports_no_progress(page, client, server_url, tmp_path):
    """A fast build must not flash a progress bar on its way past."""
    seen = []
    _build_mosaic(
        page, client, server_url, tmp_path,
        np.random.rand(16, 16, 4, 3).astype(np.float32),
        "small_mosaic", seen,
    )
    assert seen == [], f"a 4-slice mosaic must stay silent, got {seen}"
