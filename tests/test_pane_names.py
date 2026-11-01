"""Corner array names stay clear of what else sits on the panes.

Run with:
    pytest tests/test_pane_names.py
"""

import numpy as np
import pytest

from conftest import register_array

pytestmark = pytest.mark.browser

_OVERLAP = """
(pair) => {
    const [a, b] = pair.map(s => typeof s === 'string' ? document.querySelector(s) : s);
    if (!a || !b) return null;
    const r = a.getBoundingClientRect(), q = b.getBoundingClientRect();
    if (!r.width || !q.width) return null;
    return r.left < q.right && r.right > q.left && r.top < q.bottom && r.bottom > q.top;
}
"""


def _focus(page):
    page.evaluate("() => document.getElementById('viewer')?.focus()")


def _two(client, tmp_path):
    rng = np.random.default_rng(4)
    a, b = (rng.random((8, 48, 48)).astype(np.float32) * (1 + i) for i in range(2))
    return register_array(client, a, tmp_path, "names_base"), register_array(client, b, tmp_path, "names_follow")


def test_zoom_map_and_slice_bar_leave_compare_names_readable(page, server_url, client, tmp_path):
    a, b = _two(client, tmp_path)
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
    page.wait_for_selector("#compare-view-wrap.compare-titles-inset", timeout=15_000)
    left_name = "#compare-left-title"
    left_map = ".compare-primary .cmp-mini-map.visible"

    # Zoomed in: the overview map hangs below the name, not on top of it.
    for _ in range(3):
        _focus(page)
        page.keyboard.press("=")
    page.wait_for_selector(left_map, timeout=5_000)
    page.wait_for_timeout(300)
    assert page.evaluate(_OVERLAP, [left_name, left_map]) is False

    # Immersive: names stay on the images and step out from under the dim bar.
    # Side by side only opens it when height is the limit, hence the short window.
    page.set_viewport_size({"width": 2000, "height": 500})
    page.wait_for_timeout(300)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1500)
    for sel in (left_name, "#compare-right-title"):
        assert page.evaluate("s => getComputedStyle(document.querySelector(s)).opacity", sel) == "1"
        assert page.evaluate(_OVERLAP, [sel, "#info"]) is False
    tops = page.evaluate(
        "() => ['#compare-left-title', '#compare-right-title'].map(s => Math.round(document.querySelector(s).getBoundingClientRect().top))"
    )
    assert tops[0] == tops[1]

    # Back out: names return to the corner.
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => !_fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1500)
    assert page.evaluate("() => getComputedStyle(document.getElementById('compare-left-title')).translate") in ("none", "")
