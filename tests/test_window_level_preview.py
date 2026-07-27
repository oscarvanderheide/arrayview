"""Window/level drag must be local, and must not lie about the result.

Reported as "doesn't really feel smooth". Measured cause: window/level is a
pure LUT operation, but every update cost a full uncompressed RGBA frame
round-trip — 1024 KB for a 512x512 view, with the server spending 11.6 ms
computing the colormap and 66.3 ms shipping the pixels. Even on loopback the
80 ms trailing throttle capped the image at ~12 Hz; over the user's devtunnel
it was ~3 Hz. The histogram highlight is drawn locally at 60 Hz and sits right
next to the image, so the smooth one made the laggy one obvious.

The fix remaps the already-fetched raw slice against the real colormap LUT in
the browser. The property that makes it safe is the second test: a preview that
drifts from what the server would have produced would be worse than a slow one,
so the drag preview must land on the same pixels the committed frame does. That
is why the server serves the exact 256-entry LUT rather than the 32-stop
gradient the viewer already had — the gradient is an approximation that is fine
for a colorbar and wrong for pixels.
"""

import numpy as np
import pytest

from conftest import register_array

_COUNT_FRAMES = """
window._wsFrames = 0;
const orig = window.WebSocket;
window.WebSocket = function (...args) {
    const ws = new orig(...args);
    ws.addEventListener('message', (e) => {
        if (typeof e.data !== 'string') window._wsFrames++;
    });
    return ws;
};
window.WebSocket.prototype = orig.prototype;
"""

# Sample the canvas sparsely; an exact per-pixel compare would be dominated by
# resampling, and the mean over a stride is enough to catch a wrong LUT.
_CANVAS_MEAN = """
() => {
    const c = document.querySelector('canvas');
    const g = c.getContext('2d');
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let sum = 0, n = 0;
    for (let i = 0; i < d.length; i += 4 * 97) { sum += d[i]; n++; }
    return Math.round(sum / n);
}
"""


@pytest.fixture
def dragged(page, client, server_url, tmp_path):
    """Load an array and perform a real 20-step window/level drag."""
    arr = (np.random.rand(120, 120) * 1000).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "wl")
    page.add_init_script(_COUNT_FRAMES)
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=45_000)
    page.wait_for_timeout(800)

    box = page.locator("#canvas-wrap").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    result = {"before": page.evaluate("() => window._wsFrames"), "pixels": []}

    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(20):
        page.mouse.move(cx + i * 6, cy + i * 3)
        page.wait_for_timeout(25)
        result["pixels"].append(page.evaluate(_CANVAS_MEAN))
    result["during"] = page.evaluate("() => window._wsFrames")
    result["preview_mean"] = page.evaluate(_CANVAS_MEAN)

    page.mouse.up()
    page.wait_for_timeout(1500)
    result["committed_mean"] = page.evaluate(_CANVAS_MEAN)
    result["after"] = page.evaluate("() => window._wsFrames")
    return result


def test_drag_does_not_round_trip_to_the_server(dragged):
    sent = dragged["during"] - dragged["before"]
    assert sent <= 2, f"20 drag moves must not stream frames; sent {sent}"
    assert len(set(dragged["pixels"])) >= 3, (
        f"the image must actually change during the drag: {dragged['pixels']}"
    )


def test_release_commits_an_authoritative_render(dragged):
    assert dragged["after"] > dragged["during"], (
        "releasing must request a real server render"
    )


def test_preview_matches_the_committed_frame(dragged):
    """A preview that drifts from the server would be worse than a slow one."""
    assert abs(dragged["preview_mean"] - dragged["committed_mean"]) <= 2, (
        f"preview {dragged['preview_mean']} must match committed "
        f"{dragged['committed_mean']}"
    )
