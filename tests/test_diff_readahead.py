"""Compare centre diff pane reads ahead while scrolling and playing."""

from __future__ import annotations

import re
from urllib.parse import unquote

import numpy as np
import pytest

from conftest import register_array

pytestmark = pytest.mark.browser


def _diff_positions(urls: list[str]) -> list[int]:
    out = []
    for url in urls:
        m = re.search(r"[?&]indices=([^&]+)", url)
        if m:
            out.append(int(unquote(m.group(1)).split(",")[0]))
    return out


def test_diff_pane_requests_next_frames_ahead(page, server_url, client, tmp_path):
    rng = np.random.default_rng(3)
    a = rng.random((30, 48, 48)).astype(np.float32)
    sid_a = register_array(client, a, tmp_path, "ra_a")
    sid_b = register_array(client, a * 0.8, tmp_path, "ra_b")

    page.goto(f"{server_url}/?sid={sid_a}&compare_sid={sid_b}&compare_sids={sid_b}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    page.wait_for_function("() => typeof _setCompareCenterMode === 'function'")
    page.evaluate("_setCompareCenterMode(1)")
    page.wait_for_function("() => !!_diffCanvasKey", timeout=10_000)
    page.mouse.click(10, 10)
    assert page.evaluate("current_slice_dim") == 0

    urls: list[str] = []
    page.on("request", lambda r: urls.append(r.url) if "/diff/" in r.url else None)
    start = page.evaluate("indices[0]")
    page.evaluate("_diffAheadLast = null")
    page.evaluate("indices[0] -= 1; updateView()")
    page.wait_for_timeout(300)
    page.evaluate("indices[0] -= 1; updateView()")
    page.wait_for_function(
        "() => (_diffCanvasKey || '').includes('indices=' + encodeURIComponent(indices.join(',')))",
        timeout=10_000,
    )
    page.wait_for_timeout(300)
    asked = set(_diff_positions(urls))
    # Moving down: the next frames below the current one were asked for
    # before the user got there.
    assert {start - 3, start - 4, start - 5} <= asked, sorted(asked)

    # The frame the user steps to next is drawn from what was read ahead,
    # not from a new request.
    urls.clear()
    page.evaluate("indices[0] -= 1; updateView()")
    page.wait_for_function(
        "() => (_diffCanvasKey || '').includes('indices=' + encodeURIComponent(indices.join(',')))",
        timeout=10_000,
    )
    page.wait_for_timeout(200)
    assert start - 3 not in _diff_positions(urls), urls
