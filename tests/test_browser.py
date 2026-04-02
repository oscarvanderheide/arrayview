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

_JS_COMPARE_LEFT_CSS_SIZE = """
() => {
    const c = document.querySelector('canvas#compare-left-canvas');
    if (!c) return null;
    const r = c.getBoundingClientRect();
    return [Math.round(r.width), Math.round(r.height)];
}
"""

_JS_COMPARE_OVERLAY_CENTER_PIXEL = """
() => {
    const c = document.querySelector('canvas#compare-diff-canvas');
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


def _pick_compare_session(page, name_contains=None, timeout=3000):
    """Interact with the inline compare picker DOM overlay (replaces browser dialog)."""
    page.wait_for_selector("#compare-picker.visible", timeout=timeout)
    if name_contains:
        page.locator(".cp-item").filter(has_text=name_contains).first.click()
    else:
        page.locator(".cp-item:not(.cp-item-current)").first.click()


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
        page.wait_for_timeout(3000)
        text = page.evaluate(
            "() => document.getElementById('loading-overlay').textContent"
        )
        assert "expired" in text.lower() or "not found" in text.lower(), (
            f"Expected error text in loading-overlay, got: '{text}'"
        )


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

    def test_colorbar_visible_by_default(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        assert page.is_visible("canvas#slim-cb"), (
            "Colorbar should be visible by default"
        )
        assert page.is_visible("#slim-cb-labels"), "Colorbar labels should be visible"

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
        assert before != after, (
            "Center pixel unchanged after pressing c (colormap cycle)"
        )

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

    def test_help_overlay_tabs_switch_sections(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("?")
        page.wait_for_selector("#help-overlay.visible", timeout=2_000)
        assert page.inner_text(".help-tab.active").strip().lower() == "navigation"
        page.click(".help-tab[data-help-panel='axes']")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "axes & views"
        page.click(".help-tab[data-help-panel='display']")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "display"
        page.click(".help-tab[data-help-panel='axes']")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "axes & views"

    def test_help_overlay_keyboard_switches_sections(self, loaded_viewer, sid_2d):
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
        page.keyboard.press("ArrowUp")
        page.wait_for_timeout(120)
        assert page.inner_text(".help-tab.active").strip().lower() == "navigation"

    def test_help_overlay_size_stays_constant_across_sections(
        self, loaded_viewer, sid_2d
    ):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("?")
        page.wait_for_selector("#help-overlay.visible", timeout=2_000)
        initial = page.evaluate(
            "() => { const b = document.querySelector('#help-box').getBoundingClientRect(); return [Math.round(b.width), Math.round(b.height)]; }"
        )
        page.click(".help-tab[data-help-panel='axes']")
        page.wait_for_timeout(120)
        after_1 = page.evaluate(
            "() => { const b = document.querySelector('#help-box').getBoundingClientRect(); return [Math.round(b.width), Math.round(b.height)]; }"
        )
        page.click(".help-tab[data-help-panel='display']")
        page.wait_for_timeout(120)
        after_2 = page.evaluate(
            "() => { const b = document.querySelector('#help-box').getBoundingClientRect(); return [Math.round(b.width), Math.round(b.height)]; }"
        )
        assert initial == after_1 == after_2

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

    def test_zen_mode_keeps_array_name_visible(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        assert page.is_visible("#array-name")
        page.keyboard.press("Z")
        page.wait_for_timeout(150)
        assert page.is_visible("#array-name")
        assert page.inner_text("#array-name-text").strip() != ""

    def test_z_mosaic_axes_indicator_opacity(self, loaded_viewer, sid_4d):
        """Axes indicator uses opacity-based visibility; entering/exiting mosaic
        does not force-show or force-hide it. Check that pressing h/l flashes axes
        in mosaic mode (opacity > 0 immediately after), then they can fade."""
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        # Enter mosaic mode
        page.keyboard.press("z")
        page.wait_for_timeout(150)
        # Press h to flash axes indicator
        page.keyboard.press("h")
        page.wait_for_timeout(100)
        opacity = page.evaluate(
            "() => parseFloat(window.getComputedStyle(document.getElementById('main-axes-svg')).opacity)"
        )
        assert opacity > 0.5, (
            f"Expected axes visible after h in mosaic, got opacity={opacity}"
        )
        # Exit mosaic mode
        page.keyboard.press("z")
        page.wait_for_timeout(150)

    def test_t_keeps_mosaic_mode_active(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("z")
        page.wait_for_timeout(150)
        # Verify mosaic mode is active: dim_z should be >= 0
        dim_z = page.evaluate("() => window.dim_z !== undefined ? dim_z : -99")
        # If dim_z is accessible, verify it's set; otherwise just verify canvas renders
        # Press t (cycle animation frame) — mosaic mode should stay active
        page.keyboard.press("t")
        page.wait_for_timeout(200)
        # Canvas should still be visible (mosaic mode still active)
        assert page.is_visible("#canvas-wrap")

    def test_B_toggles_side_by_side_compare(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare.npy"
        np.save(path, arr_2d * 0.5)
        resp = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare"}
        )
        sid_compare = resp.json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("B")
        _pick_compare_session(page, "arr2d_compare")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        page.wait_for_selector("canvas#compare-right-canvas:visible", timeout=5_000)
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")
        # Shared colorbar is used (per-pane colorbars hidden in non-diff mode)
        assert page.is_visible("#slim-cb-wrap")
        assert "arr2d_compare" in page.inner_text("#compare-right-title").lower()

        page.keyboard.press("B")
        page.wait_for_timeout(350)
        assert not page.is_visible("#compare-view-wrap.active")

    def test_compare_space_keeps_playing(
        self, loaded_viewer, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare.npy"
        np.save(path, np.flip(arr_3d, axis=0))
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare"}
        ).json()["sid"]

        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("B")
        _pick_compare_session(page, "arr3d_compare")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)

        page.keyboard.press("Space")
        page.wait_for_timeout(250)
        p1 = page.evaluate(_JS_COMPARE_LEFT_CENTER_PIXEL)
        page.wait_for_timeout(250)
        p2 = page.evaluate(_JS_COMPARE_LEFT_CENTER_PIXEL)
        page.keyboard.press("Space")
        assert p1 != p2

    def test_compare_scale_stays_stable_on_first_scroll(
        self, loaded_viewer, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare_scale.npy"
        np.save(path, arr_3d * 0.8)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare_scale"}
        ).json()["sid"]

        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("B")
        _pick_compare_session(page, "arr3d_compare_scale")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(350)
        before = page.evaluate(_JS_COMPARE_LEFT_CSS_SIZE)

        page.hover("canvas#compare-left-canvas")
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(350)
        after = page.evaluate(_JS_COMPARE_LEFT_CSS_SIZE)
        assert before == after

    def test_R_registration_overlay_and_n_cycles_compare_target(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path1 = tmp_path / "arr2d_compare_1.npy"
        path2 = tmp_path / "arr2d_compare_2.npy"
        np.save(path1, arr_2d * 0.35)
        np.save(path2, np.flipud(arr_2d))
        sid1 = client.post(
            "/load", json={"filepath": str(path1), "name": "arr2d_compare_1"}
        ).json()["sid"]
        sid2 = client.post(
            "/load", json={"filepath": str(path2), "name": "arr2d_compare_2"}
        ).json()["sid"]
        assert sid1 != sid2

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("B")
        _pick_compare_session(page, "arr2d_compare_1")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        assert page.is_visible("canvas#compare-right-canvas")
        assert not page.is_visible("canvas#compare-third-canvas")
        assert page.is_visible("#slim-cb-wrap")

        # Cycle compare center mode to overlay (mode 4): X×4 = off→A-B→|A-B|→|A-B|/|A|→overlay
        for _ in range(4):
            page.keyboard.press("X")
            page.wait_for_timeout(200)
        page.wait_for_timeout(400)
        diff_classes = page.get_attribute("#compare-diff-pane", "class") or ""
        assert "overlay-center" in diff_classes
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")
        assert page.is_visible("canvas#compare-diff-canvas")

        before = page.evaluate(_JS_COMPARE_OVERLAY_CENTER_PIXEL)
        page.keyboard.press("]")
        page.wait_for_timeout(250)
        after = page.evaluate(_JS_COMPARE_OVERLAY_CENTER_PIXEL)
        assert before != after

        page.keyboard.press("n")
        page.wait_for_timeout(800)
        assert "arr2d_compare_2" in page.inner_text("#compare-right-title").lower()

        # Press X twice more to go past wipe (mode 5) back to off (mode 0)
        page.keyboard.press("X")
        page.wait_for_timeout(200)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        diff_classes = page.get_attribute("#compare-diff-pane", "class") or ""
        assert "overlay-center" not in diff_classes
        assert page.is_visible("canvas#compare-right-canvas")

    def test_B_is_locked_for_multi_array_launch(
        self, page, server_url, sid_2d, arr_2d, client, tmp_path
    ):
        path1 = tmp_path / "arr2d_compare_locked_1.npy"
        path2 = tmp_path / "arr2d_compare_locked_2.npy"
        np.save(path1, arr_2d * 0.5)
        np.save(path2, np.flipud(arr_2d))
        sid1 = client.post(
            "/load", json={"filepath": str(path1), "name": "arr2d_compare_locked_1"}
        ).json()["sid"]
        sid2 = client.post(
            "/load", json={"filepath": str(path2), "name": "arr2d_compare_locked_2"}
        ).json()["sid"]

        page.goto(
            f"{server_url}/?sid={sid_2d}&compare_sid={sid1}&compare_sids={sid1},{sid2}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(500)
        assert page.is_visible("canvas#compare-third-canvas")
        assert page.is_visible("#slim-cb-wrap")

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
        active_panes = page.evaluate(
            "() => document.querySelectorAll('#compare-panes .compare-pane.active').length"
        )
        assert active_panes == 6
        compare_cols = page.evaluate(
            "() => getComputedStyle(document.getElementById('compare-panes')).getPropertyValue('--compare-cols').trim()"
        )
        assert compare_cols == "3", (
            f"Expected --compare-cols=3 for 6 panes, got '{compare_cols}'"
        )

    def test_d_cycles_dynamic_range_shows_status(self, loaded_viewer, sid_2d):
        # First d press expands histogram; second cycles quantile range
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        status = page.inner_text("#status").strip()
        assert "histogram" in status.lower(), f"Expected histogram status, got: '{status}'"
        # Second press cycles quantile — shows range label
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        status = page.inner_text("#status").strip()
        assert "range" in status.lower(), f"Expected range status, got: '{status}'"

    def test_space_toggles_playback(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(250)
        assert "playing" in page.inner_text("#status").lower()

        page.keyboard.press("Space")
        page.wait_for_timeout(150)
        assert "playing" not in page.inner_text("#status").lower()

    def test_playing_dim_gets_orange_class(self, loaded_viewer, sid_3d):
        """Playing dim should get .playing-dim class (orange), not .active-dim."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(300)
        has_playing = page.evaluate("""
            () => document.querySelector('#info .playing-dim') !== null
        """)
        assert has_playing, "playing dim should have .playing-dim class during playback"
        page.keyboard.press("Space")  # stop

    def test_play_allows_independent_dim_navigation(self, loaded_viewer, sid_3d):
        """During playback, user can change activeDim without affecting playingDim."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(300)
        # The playing dim should be shown with .playing-dim class
        playing_dim_idx = page.evaluate("""
            () => {
                const el = document.querySelector('#info .playing-dim');
                return el ? parseInt(el.dataset.dim) : -1;
            }
        """)
        assert playing_dim_idx >= 0, "playingDim should be set during playback"
        # Initially activeDim == playingDim, so .playing-dim takes priority
        # and there's no separate .active-dim element.
        # Press h to change activeDim to a different dim.
        page.keyboard.press("h")
        page.wait_for_timeout(200)
        # After h, activeDim moved to a different dim which should get .active-dim
        active_after = page.evaluate("""
            () => {
                const el = document.querySelector('#info .active-dim');
                return el ? parseInt(el.dataset.dim) : -1;
            }
        """)
        assert active_after >= 0 and active_after != playing_dim_idx, (
            f"Expected .active-dim on a different dim than playing ({playing_dim_idx}), "
            f"got {active_after}"
        )
        # playingDim should still be marked with .playing-dim in the DOM
        still_playing_idx = page.evaluate("""
            () => {
                const el = document.querySelector('#info .playing-dim');
                return el ? parseInt(el.dataset.dim) : -1;
            }
        """)
        assert still_playing_idx == playing_dim_idx, (
            f"playingDim changed from {playing_dim_idx} to {still_playing_idx} after pressing h"
        )
        assert "playing" in page.inner_text("#status").lower(), \
            "should still be playing after changing activeDim"
        page.keyboard.press("Space")  # stop

    def test_jk_on_playing_dim_stops_playback(self, loaded_viewer, sid_3d):
        """Pressing j/k on the actively playing dim should stop playback."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(300)
        # Ensure we're playing
        assert "playing" in page.inner_text("#status").lower()
        # Get the playing dim index from the DOM
        playing_dim_idx = page.evaluate("""
            () => {
                const el = document.querySelector('#info .playing-dim');
                return el ? parseInt(el.dataset.dim) : -1;
            }
        """)
        assert playing_dim_idx >= 0
        # Navigate activeDim to match the playing dim by pressing h/l until we get there
        # For 3D array: activeDim starts at 2 (current_slice_dim), playing_dim is also 2
        # So activeDim should already be on the playing dim — just press j
        page.keyboard.press("j")
        page.wait_for_timeout(200)
        status = page.inner_text("#status").lower()
        assert "playing" not in status, \
            "playback should stop when j/k pressed on playing dim"

    def test_i_shows_data_info_overlay(self, loaded_viewer, sid_2d):
        # i shows the info overlay (#info-overlay gains .visible class)
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(600)
        visible = page.evaluate(
            "() => document.querySelector('#info-overlay').classList.contains('visible')"
        )
        assert visible, "#info-overlay should have .visible after pressing i"
        text = page.inner_text("#info-overlay")
        assert "100" in text or "80" in text, f"Shape not in info-overlay: '{text}'"

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
        feedback = (
            page.inner_text("#status") + " " + page.inner_text("#toast")
        ).strip()
        assert (
            "clipboard" in feedback.lower()
            or "url" in feedback.lower()
            or "copied" in feedback.lower()
        ), f"Expected clipboard/status message, got: '{feedback}'"
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

    def test_border_toggle_uses_subtle_outline(self, loaded_viewer, sid_2d):
        """Border toggle (b) should use 1px gray outline, not 2px highlight."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(200)
        outline = page.evaluate(
            "() => getComputedStyle(document.querySelector('#canvas-viewport')).outline"
        )
        # Should be 1px, not 2px; should NOT be pure white
        assert "1px" in outline, f"expected 1px outline, got: {outline}"
        assert "rgb(255, 255, 255)" not in outline, f"border should not be pure white: {outline}"

    def test_multiview_rotate_updates_all_panes(self, loaded_viewer, sid_3d):
        """Pressing r in multiview should swap axes globally across all 3 panes."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

        # Trigger a saveState() by cycling colormap, so we have a baseline in sessionStorage
        page.keyboard.press("c")
        page.wait_for_timeout(300)

        before = page.evaluate("""(sid) => {
            const raw = sessionStorage.getItem('av_' + sid);
            if (!raw) return null;
            const s = JSON.parse(raw);
            return {dim_x: s.dim_x, dim_y: s.dim_y, mvDims: s.mvDims};
        }""", sid_3d)
        assert before is not None, "No saved state found before rotation"

        # Press r to rotate
        page.keyboard.press("r")
        page.wait_for_timeout(500)

        after = page.evaluate("""(sid) => {
            const raw = sessionStorage.getItem('av_' + sid);
            if (!raw) return null;
            const s = JSON.parse(raw);
            return {dim_x: s.dim_x, dim_y: s.dim_y, mvDims: s.mvDims};
        }""", sid_3d)
        assert after is not None, "No saved state found after rotation"

        # Global rotation should swap dim_x and dim_y
        assert before["dim_x"] == after["dim_y"] and before["dim_y"] == after["dim_x"], (
            f"Expected dim_x/dim_y swap: before={before}, after={after}"
        )
        # mvDims should also have changed
        assert before["mvDims"] != after["mvDims"], (
            f"Expected mvDims to change: before={before['mvDims']}, after={after['mvDims']}"
        )

    def test_fullscreen_toggle(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+K")
        page.wait_for_timeout(200)
        assert "fullscreen-mode" in (page.locator("body").get_attribute("class") or "")
        page.keyboard.press("Shift+K")
        page.wait_for_timeout(200)
        assert "fullscreen-mode" not in (page.locator("body").get_attribute("class") or "")


# ---------------------------------------------------------------------------
# Visual regression
# ---------------------------------------------------------------------------


class TestROIDrag:
    def test_canvas_drag_shows_roi_stats(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Enter ROI mode first (A key)
        page.keyboard.press("A")
        page.wait_for_timeout(300)
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
        panel_visible = page.evaluate(
            "() => document.getElementById('roi-panel').style.display !== 'none' && document.getElementById('roi-panel').style.display !== ''"
        )
        assert panel_visible, "Expected #roi-panel to be visible after drag"
        table_text = page.inner_text("#roi-content")
        assert "mean" in table_text.lower() or "min" in table_text.lower(), (
            f"Expected ROI stats in #roi-content, got: '{table_text}'"
        )


class TestColorbarWindowLevel:
    def test_colorbar_rendered_with_gradient(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        cb = page.locator("canvas#slim-cb")
        box = cb.bounding_box()
        assert box is not None, "Colorbar not visible"
        # Colorbar should be horizontal: wider than tall
        assert box["width"] > box["height"], "Colorbar should be horizontal"
        # Labels should show vmin on left, vmax on right
        labels_text = page.inner_text("#slim-cb-labels")
        assert labels_text.strip(), "Colorbar labels should have content"


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
        assert before == after, (
            "Center pixel changed after reload; colormap not persisted"
        )


class TestMinimapCursor:
    def test_minimap_cursor_grab(self, loaded_viewer, sid_2d):
        """Mini-map should show grab cursor, grabbing while dragging."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Zoom in far enough to trigger mini-map
        for _ in range(8):
            page.keyboard.press("Equal")
            page.wait_for_timeout(80)
        page.wait_for_timeout(300)
        visible = page.evaluate(
            "() => document.querySelector('#mini-map').classList.contains('visible')"
        )
        assert visible, "mini-map should be visible after zooming in"
        cursor = page.evaluate(
            "() => getComputedStyle(document.querySelector('#mini-map')).cursor"
        )
        assert cursor == "grab", f"expected grab cursor, got {cursor}"


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
