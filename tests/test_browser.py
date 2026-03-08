"""Layer 2: Playwright browser tests.

Run with:
    pytest tests/test_browser.py

First run creates baseline snapshots in tests/snapshots/.
Subsequent runs compare against them (1% pixel-change threshold).
"""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

pytestmark = pytest.mark.browser

SNAPSHOTS = Path(__file__).parent / "snapshots"

# ---------------------------------------------------------------------------
# Canvas inspection helpers (evaluated in-browser via JS)
# ---------------------------------------------------------------------------

_JS_CANVAS_INFO = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c || !c.width || !c.height) return null;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let nonBg = 0;
    // background colour is #111 = rgb(17,17,17)
    for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 20 || d[i+1] > 20 || d[i+2] > 20) nonBg++;
    }
    return {width: c.width, height: c.height, nonBgPixels: nonBg};
}
"""

_JS_CENTER_PIXEL = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""

_JS_COMPARE_LEFT_CENTER_PIXEL = """
() => {
    const c = document.querySelector('canvas#compare-left-canvas');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""

_JS_MV_CANVAS_COUNT = """
() => document.querySelectorAll('.mv-canvas').length
"""


def _focus_kb(page):
    """Focus the hidden keyboard-sink textarea so key events are delivered."""
    page.focus("#keyboard-sink")


def _compare_snapshot(page, name: str, threshold: float = 0.01):
    """
    Screenshot the page and compare against a saved baseline.
    Creates the baseline on first run, then skips with a message.
    Subsequent runs fail if more than `threshold` fraction of pixels differ.
    """
    path = SNAPSHOTS / f"{name}.png"
    raw = page.screenshot()

    if not path.exists():
        path.write_bytes(raw)
        pytest.skip(f"Baseline created: snapshots/{name}.png — re-run to compare")

    baseline = Image.open(path).convert("RGB")
    current = Image.open(io.BytesIO(raw)).convert("RGB")

    if baseline.size != current.size:
        current = current.resize(baseline.size, Image.LANCZOS)

    diff = ImageChops.difference(baseline, current)
    total = baseline.width * baseline.height
    # Count pixels with any channel delta > 10 (ignores tiny rendering differences)
    different = sum(1 for px in np.array(diff).reshape(-1, 3) if px.max() > 10)
    frac = different / total

    assert frac <= threshold, (
        f"Visual regression in '{name}': {frac:.1%} of pixels differ "
        f"(threshold {threshold:.0%}). "
        f"Delete snapshots/{name}.png to accept the new look."
    )


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------

class TestSessionExpired:
    def test_invalid_sid_shows_error_in_overlay(self, page, server_url):
        page.goto(f"{server_url}/?sid=invalidXXX000")
        page.wait_for_timeout(1500)
        assert page.is_visible("#loading-overlay")
        text = page.inner_text("#loading-overlay")
        assert "expired" in text.lower() or "not found" in text.lower()


class TestBasicRender:
    def test_canvas_visible_2d(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        assert page.is_visible("canvas#viewer")

    def test_canvas_has_content_2d(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        info = page.evaluate(_JS_CANVAS_INFO)
        assert info is not None, "Canvas not found or has zero size"
        total = info["width"] * info["height"]
        # At least 30% of pixels should be non-background
        assert info["nonBgPixels"] > total * 0.3, (
            f"Canvas looks blank: only {info['nonBgPixels']}/{total} non-background pixels"
        )

    def test_info_line_shows_dimension_labels(self, loaded_viewer, sid_2d):
        # #info shows [x, y] dimension labels (not raw shape numbers)
        page = loaded_viewer(sid_2d)
        text = page.inner_text("#info")
        assert "x" in text and "y" in text

    def test_colorbar_hidden_by_default_shown_after_b(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        assert not page.is_visible("canvas#colorbar"), "Colorbar should start hidden"
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(300)
        assert page.is_visible("canvas#colorbar"), "Colorbar should appear after pressing b"

    def test_3d_array_renders(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        assert page.is_visible("canvas#viewer")
        info = page.evaluate(_JS_CANVAS_INFO)
        assert info["nonBgPixels"] > 0

    def test_4d_array_renders(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        assert page.is_visible("canvas#viewer")
        info = page.evaluate(_JS_CANVAS_INFO)
        assert info["nonBgPixels"] > 0

    def test_loading_overlay_gone_after_render(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        # #loading-overlay should be hidden once canvas is showing
        assert not page.is_visible("#loading-overlay")


# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------

class TestKeyboard:
    def test_c_changes_colormap(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("c")
        page.wait_for_timeout(800)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Center pixel unchanged after pressing c (colormap cycle)"

    def test_help_overlay_opens_and_closes(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        assert not page.is_visible("#help-overlay.visible")
        _focus_kb(page)
        page.keyboard.press("?")
        page.wait_for_selector("#help-overlay.visible", timeout=2_000)
        assert page.is_visible("#help-box")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        assert not page.is_visible("#help-overlay.visible")

    def test_help_overlay_jk_switches_sections(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("?")
        page.wait_for_selector("#help-overlay.visible", timeout=2_000)
        assert page.inner_text(".help-tab.active").strip().lower() == "navigation"
        page.keyboard.press("j")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "axes & views"
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "display"
        page.keyboard.press("k")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "axes & views"

    def test_v_activates_multiview_on_3d(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        assert page.evaluate(_JS_MV_CANVAS_COUNT) == 3

    def test_v_toggles_back_to_single_view(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.keyboard.press("v")
        page.wait_for_timeout(400)
        # Back to single-view: main canvas visible, multi-view hidden
        assert page.is_visible("canvas#viewer")
        assert not page.is_visible("#multi-view-wrap.active")

    def test_B_toggles_side_by_side_compare(self, loaded_viewer, sid_2d, arr_2d, client, tmp_path):
        path = tmp_path / "arr2d_compare.npy"
        np.save(path, arr_2d * 0.5)
        resp = client.post("/load", json={"filepath": str(path), "name": "arr2d_compare"})
        sid_compare = resp.json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.once("dialog", lambda d: d.accept(sid_compare))
        page.keyboard.press("B")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")
        assert page.is_visible("canvas#compare-cb")
        assert "arr2d_compare" in page.inner_text("#compare-right-title").lower()

        page.keyboard.press("B")
        page.wait_for_timeout(350)
        assert not page.is_visible("#compare-view-wrap.active")

    def test_R_registration_overlay_and_n_cycles_compare_target(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path1 = tmp_path / "arr2d_compare_1.npy"
        path2 = tmp_path / "arr2d_compare_2.npy"
        np.save(path1, arr_2d * 0.35)
        np.save(path2, np.flipud(arr_2d))
        sid1 = client.post("/load", json={"filepath": str(path1), "name": "arr2d_compare_1"}).json()["sid"]
        sid2 = client.post("/load", json={"filepath": str(path2), "name": "arr2d_compare_2"}).json()["sid"]
        assert sid1 != sid2

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.once("dialog", lambda d: d.accept(sid1))
        page.keyboard.press("B")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        assert page.is_visible("canvas#compare-right-canvas")
        assert not page.is_visible("canvas#compare-third-canvas")
        assert page.is_visible("canvas#compare-cb")

        page.keyboard.press("R")
        page.wait_for_timeout(400)
        classes = page.get_attribute("#compare-view-wrap", "class") or ""
        assert "registration-mode" in classes

        before = page.evaluate(_JS_COMPARE_LEFT_CENTER_PIXEL)
        page.keyboard.press("]")
        page.wait_for_timeout(250)
        after = page.evaluate(_JS_COMPARE_LEFT_CENTER_PIXEL)
        assert before != after

        page.keyboard.press("n")
        page.wait_for_timeout(800)
        assert "arr2d_compare_2" in page.inner_text("#compare-right-title").lower()

        page.keyboard.press("R")
        page.wait_for_timeout(400)
        classes = page.get_attribute("#compare-view-wrap", "class") or ""
        assert "registration-mode" not in classes
        assert page.is_visible("canvas#compare-right-canvas")

    def test_B_is_locked_for_multi_array_launch(
        self, page, server_url, sid_2d, arr_2d, client, tmp_path
    ):
        path1 = tmp_path / "arr2d_compare_locked_1.npy"
        path2 = tmp_path / "arr2d_compare_locked_2.npy"
        np.save(path1, arr_2d * 0.5)
        np.save(path2, np.flipud(arr_2d))
        sid1 = client.post("/load", json={"filepath": str(path1), "name": "arr2d_compare_locked_1"}).json()["sid"]
        sid2 = client.post("/load", json={"filepath": str(path2), "name": "arr2d_compare_locked_2"}).json()["sid"]

        page.goto(f"{server_url}/?sid={sid_2d}&compare_sid={sid1}&compare_sids={sid1},{sid2}")
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(500)
        assert page.is_visible("canvas#compare-third-canvas")
        assert page.is_visible("canvas#compare-cb")

        _focus_kb(page)
        page.keyboard.press("B")
        page.wait_for_timeout(250)
        assert page.is_visible("#compare-view-wrap.active")
        status = page.inner_text("#status").lower()
        assert "fixed" in status or "multi-array" in status

    def test_multi_array_launch_supports_six_compare_panes(
        self, page, server_url, sid_2d, arr_2d, client, tmp_path
    ):
        compare_sids = []
        for i in range(5):
            path = tmp_path / f"arr2d_compare_{i}.npy"
            np.save(path, arr_2d * (0.2 + 0.1 * i))
            sid_i = client.post(
                "/load", json={"filepath": str(path), "name": f"arr2d_compare_{i}"}
            ).json()["sid"]
            compare_sids.append(sid_i)

        page.goto(
            f"{server_url}/?sid={sid_2d}&compare_sid={compare_sids[0]}&compare_sids={','.join(compare_sids)}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(700)

        assert page.is_visible("canvas#compare-sixth-canvas")
        active_panes = page.evaluate("() => document.querySelectorAll('#compare-panes .compare-pane.active').length")
        assert active_panes == 6
        grid_cols = page.evaluate(
            "() => getComputedStyle(document.querySelector('#compare-panes')).gridTemplateColumns.split(' ').length"
        )
        assert grid_cols == 3

    def test_d_cycles_dynamic_range_shows_toast(self, loaded_viewer, sid_2d):
        # d cycles dynamic range; result appears in #toast
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        toast = page.inner_text("#toast").strip()
        assert "range" in toast.lower(), f"Expected DR toast, got: '{toast}'"

    def test_i_shows_data_info_overlay(self, loaded_viewer, sid_2d):
        # i fetches /info and shows shape/dtype in #data-info
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(600)
        opacity = page.evaluate(
            "() => parseFloat(getComputedStyle(document.querySelector('#data-info')).opacity)"
        )
        assert opacity > 0.5, "#data-info should be visible after pressing i"
        text = page.inner_text("#data-info")
        assert "100" in text or "80" in text, f"Shape not in data-info: '{text}'"

    def test_e_copies_state_to_clipboard(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_timeout(500)
        expected_px = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("e")
        page.wait_for_timeout(800)
        copied = page.evaluate("() => navigator.clipboard.readText()")
        assert f"sid={sid_2d}" in copied
        assert "state=" in copied
        feedback = (page.inner_text("#status") + " " + page.inner_text("#toast")).strip()
        assert "clipboard" in feedback.lower() or "url" in feedback.lower() or "copied" in feedback.lower(), (
            f"Expected clipboard/status message, got: '{feedback}'"
        )
        page2 = page.context.new_page()
        page2.goto(copied)
        page2.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page2.wait_for_timeout(400)
        assert page2.evaluate(_JS_CENTER_PIXEL) == expected_px
        page2.close()

    def test_reusable_url_restores_multiview_mode(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)

        page.keyboard.press("e")
        page.wait_for_timeout(600)
        copied = page.evaluate("() => navigator.clipboard.readText()")
        assert "state=" in copied

        page.goto(copied)
        page.wait_for_selector("#multi-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_MV_CANVAS_COUNT) == 3

    def test_reusable_url_restores_qmri_mode(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap.active", timeout=5_000)

        page.keyboard.press("e")
        page.wait_for_timeout(700)
        copied = page.evaluate("() => navigator.clipboard.readText()")
        assert "state=" in copied

        page.goto(copied)
        page.wait_for_selector("#qmri-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(500)
        assert page.locator("canvas.qv-canvas").count() == 5

    def test_s_puts_status_message(self, loaded_viewer, sid_2d):
        # s triggers download and sets #status to "Screenshot saved."
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("s")
        page.wait_for_timeout(500)
        status = page.inner_text("#status").strip()
        assert "screenshot" in status.lower() or "saved" in status.lower(), (
            f"Expected screenshot status message, got: '{status}'"
        )


# ---------------------------------------------------------------------------
# Visual regression
# ---------------------------------------------------------------------------

class TestROIDrag:
    def test_canvas_drag_shows_roi_stats(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        assert box is not None
        # Drag from upper-left to lower-right of the canvas
        x0 = box["x"] + box["width"] * 0.1
        y0 = box["y"] + box["height"] * 0.1
        x1 = box["x"] + box["width"] * 0.6
        y1 = box["y"] + box["height"] * 0.6
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()
        page.wait_for_timeout(800)
        status = page.inner_text("#status").strip()
        assert "roi" in status.lower(), f"Expected ROI stats in #status, got: '{status}'"


class TestColorbarWindowLevel:
    def test_colorbar_drag_changes_canvas(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Turn on colorbar
        page.keyboard.press("b")
        page.wait_for_timeout(300)
        before = page.evaluate(_JS_CENTER_PIXEL)
        # Drag the colorbar downward (pan window toward higher values → image darkens)
        cb = page.locator("canvas#colorbar")
        box = cb.bounding_box()
        assert box is not None, "Colorbar not visible"
        mid_x = box["x"] + box["width"] / 2
        mid_y = box["y"] + box["height"] / 2
        page.mouse.move(mid_x, mid_y)
        page.mouse.down()
        page.mouse.move(mid_x, mid_y + 60, steps=10)
        page.mouse.up()
        page.wait_for_timeout(600)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Center pixel unchanged after colorbar drag"


class TestCustomColormap:
    def test_C_key_with_valid_colormap_changes_canvas(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        # Use page.keyboard to trigger C; dialog will appear, fill and confirm
        page.on("dialog", lambda d: d.accept("inferno"))
        page.keyboard.press("C")
        page.wait_for_timeout(1200)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Center pixel unchanged after C + inferno colormap"


class TestSessionStorage:
    def test_colormap_persists_across_reload(self, loaded_viewer, sid_2d, server_url):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Press c twice to cycle away from gray
        page.keyboard.press("c")
        page.wait_for_timeout(600)
        page.keyboard.press("c")
        page.wait_for_timeout(600)
        before = page.evaluate(_JS_CENTER_PIXEL)
        # Reload the same URL (sessionStorage persists within session)
        page.goto(f"{server_url}/?sid={sid_2d}")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page.wait_for_timeout(600)
        after = page.evaluate(_JS_CENTER_PIXEL)
        # After restore the colormap should still be the cycled one → same pixel colour
        assert before == after, "Center pixel changed after reload; colormap not persisted"


class TestVisualRegression:
    """
    On first run: saves screenshots to tests/snapshots/ as baselines.
    On subsequent runs: fails if >1% of pixels differ from baseline.
    To reset a baseline, delete the corresponding .png from tests/snapshots/.
    """

    def test_2d_gradient_gray(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _compare_snapshot(page, "2d_gradient_gray")

    def test_3d_midslice_gray(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _compare_snapshot(page, "3d_midslice_gray")

    def test_2d_gradient_viridis(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Cycle to viridis (gray → lipari → navia → viridis after 3 presses)
        for _ in range(3):
            page.keyboard.press("c")
            page.wait_for_timeout(600)
        _compare_snapshot(page, "2d_gradient_viridis")

    def test_3d_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)
        _compare_snapshot(page, "3d_multiview")
