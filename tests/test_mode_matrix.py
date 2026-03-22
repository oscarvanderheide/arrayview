"""Layer 3: Comprehensive mode-matrix Playwright tests.

Tests systematic enter/exit of every major viewer mode, state transitions,
zoom regression, and features from items 12-15.

Run with:
    pytest tests/test_mode_matrix.py -v

Requires Playwright:
    uv run playwright install chromium
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

# ---------------------------------------------------------------------------
# JS helpers  — DOM-based checks only (no window.eval for closure vars)
# ---------------------------------------------------------------------------

_JS_COMPARE_LEFT_CSS_SIZE = """
() => {
    const c = document.querySelector('canvas#compare-left-canvas');
    if (!c) return null;
    const r = c.getBoundingClientRect();
    return [Math.round(r.width), Math.round(r.height)];
}
"""

_JS_DIFF_PANE_VISIBLE = """
() => {
    const el = document.getElementById('compare-diff-pane');
    return el && el.style.display !== 'none' && el.offsetParent !== null;
}
"""

_JS_WIPE_PANE_VISIBLE = """
() => {
    const el = document.getElementById('compare-wipe-pane');
    return el && el.style.display !== 'none' && el.offsetParent !== null;
}
"""

_JS_ORIENTATION_VISIBLE = """
() => {
    const el = document.getElementById('mv-orientation');
    return el && el.style.display !== 'none';
}
"""

_JS_CANVAS_HAS_CONTENT = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c || !c.width || !c.height) return false;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let nonBg = 0;
    for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 20 || d[i+1] > 20 || d[i+2] > 20) nonBg++;
    }
    return nonBg > 100;
}
"""

_JS_CMP_INNER_DIMENSIONS = """
() => {
    const inner = document.querySelector('.compare-canvas-inner');
    if (!inner) return null;
    return {
        vpW: inner.dataset.vpW ? parseInt(inner.dataset.vpW) : null,
        vpH: inner.dataset.vpH ? parseInt(inner.dataset.vpH) : null,
        styleW: parseInt(inner.style.width) || null,
        styleH: parseInt(inner.style.height) || null,
    };
}
"""

_JS_HAS_OVERLAY_CENTER_CLASS = """
() => {
    const el = document.getElementById('compare-diff-pane');
    return el ? el.classList.contains('overlay-center') : false;
}
"""

_JS_HAS_WIPE_MODE_CLASS = """
() => {
    const el = document.getElementById('compare-view-wrap');
    return el ? el.classList.contains('wipe-mode') : false;
}
"""

_JS_COMPARE_OVERLAY_CENTER_PIXEL = """
() => {
    const c = document.querySelector('canvas#compare-third-canvas');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""


def _focus_kb(page):
    page.focus("#keyboard-sink")


def _pick_compare_session(page, name_contains=None, timeout=3000):
    page.wait_for_selector("#compare-picker.visible", timeout=timeout)
    if name_contains:
        page.locator(".cp-item").filter(has_text=name_contains).first.click()
    else:
        page.locator(".cp-item:not(.cp-item-current)").first.click()


def _enter_compare(page, compare_name):
    """Enter compare mode by pressing B and picking a session."""
    _focus_kb(page)
    page.keyboard.press("B")
    _pick_compare_session(page, compare_name)
    page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
    page.wait_for_timeout(400)


@pytest.fixture
def sid_compare_3d(client, arr_3d, tmp_path):
    """Register a second 3D array for compare testing."""
    path = tmp_path / "arr3d_compare.npy"
    np.save(path, np.flip(arr_3d, axis=0))
    resp = client.post("/load", json={"filepath": str(path), "name": "arr3d_compare"})
    return resp.json()["sid"]


@pytest.fixture
def sid_compare_2d(client, arr_2d, tmp_path):
    """Register a second 2D array for compare testing."""
    path = tmp_path / "arr2d_cmp.npy"
    np.save(path, arr_2d * 0.5)
    resp = client.post("/load", json={"filepath": str(path), "name": "arr2d_cmp"})
    return resp.json()["sid"]


# ---------------------------------------------------------------------------
# Mode enter/exit: verify each mode can be entered and exited cleanly
# ---------------------------------------------------------------------------

class TestModeEnterExit:
    """Enter and exit each major mode, verify DOM state is clean."""

    def test_multiview_enter_exit(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)

        # Enter multi-view
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        mv_count = page.evaluate("() => document.querySelectorAll('.mv-canvas').length")
        assert mv_count == 3
        assert page.evaluate(_JS_ORIENTATION_VISIBLE)

        # Exit multi-view
        page.keyboard.press("v")
        page.wait_for_timeout(400)
        assert not page.is_visible("#multi-view-wrap.active")
        assert not page.evaluate(_JS_ORIENTATION_VISIBLE)
        assert page.is_visible("canvas#viewer")

    def test_compare_enter_exit(self, loaded_viewer, sid_2d, sid_compare_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        assert page.is_visible("#compare-view-wrap.active")
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")

        # Exit
        _focus_kb(page)
        page.keyboard.press("B")
        page.wait_for_timeout(400)
        assert not page.is_visible("#compare-view-wrap.active")
        assert page.is_visible("canvas#viewer")

    def test_diff_mode_cycle(self, loaded_viewer, sid_2d, sid_compare_2d):
        """Cycle through all compare center modes: off->A-B->|A-B|->|A-B|/|A|->overlay->wipe->off."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)

        # X once: 0->1 (A-B) -- diff pane visible
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_DIFF_PANE_VISIBLE)
        status = page.inner_text("#status").lower()
        assert "diff" in status

        # X twice: 1->2 (|A-B|)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_DIFF_PANE_VISIBLE)

        # X three: 2->3 (|A-B|/|A|)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_DIFF_PANE_VISIBLE)

        # X four: 3->4 (overlay)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_HAS_OVERLAY_CENTER_CLASS)
        assert not page.evaluate(_JS_HAS_WIPE_MODE_CLASS)

        # X five: 4->5 (wipe)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_HAS_WIPE_MODE_CLASS)
        assert not page.evaluate(_JS_HAS_OVERLAY_CENTER_CLASS)

        # X six: 5->0 (off)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert not page.evaluate(_JS_DIFF_PANE_VISIBLE)
        assert not page.evaluate(_JS_HAS_WIPE_MODE_CLASS)
        assert not page.evaluate(_JS_HAS_OVERLAY_CENTER_CLASS)

    def test_roi_mode_enter_exit(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        # Enter ROI mode (A key)
        page.keyboard.press("A")
        page.wait_for_timeout(300)
        status = page.inner_text("#status").lower()
        assert "roi" in status

        # Cycle through shapes: A again -> circle -> freehand -> off
        page.keyboard.press("A")
        page.wait_for_timeout(200)
        status = page.inner_text("#status").lower()
        assert "circle" in status

        page.keyboard.press("A")
        page.wait_for_timeout(200)
        status = page.inner_text("#status").lower()
        assert "freehand" in status

        page.keyboard.press("A")
        page.wait_for_timeout(200)
        status = page.inner_text("#status").lower()
        assert "off" in status


# ---------------------------------------------------------------------------
# State transitions: entering one mode from another
# ---------------------------------------------------------------------------

class TestStateTransitions:
    """Enter one mode, then switch to another, verify no residual state."""

    def test_compare_to_multiview_and_back(self, loaded_viewer, sid_3d, sid_compare_3d):
        page = loaded_viewer(sid_3d)
        _enter_compare(page, "arr3d_compare")

        _focus_kb(page)
        # v in compare -> compare multi-view
        page.keyboard.press("v")
        page.wait_for_timeout(800)
        # Should show compare multiview grid (re-uses qmri-view-wrap)
        cmp_mv_canvases = page.evaluate(
            "() => document.querySelectorAll('#qmri-view-wrap .qv-canvas-wrap').length"
        )
        assert cmp_mv_canvases > 0, "Compare multi-view should show canvases"

        # v again -> back to compare
        page.keyboard.press("v")
        page.wait_for_timeout(500)
        assert page.is_visible("#compare-view-wrap.active")

    def test_diff_exit_cleans_up_panes(self, loaded_viewer, sid_2d, sid_compare_2d):
        """After exiting compare with diff active, no residual diff pane."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)

        # Enter diff mode
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_DIFF_PANE_VISIBLE)

        # Exit compare
        page.keyboard.press("B")
        page.wait_for_timeout(400)
        assert page.is_visible("canvas#viewer")
        assert not page.evaluate(_JS_DIFF_PANE_VISIBLE)


# ---------------------------------------------------------------------------
# Item 14: Multi-array 3D mini-view (orientation widget in compare MV)
# ---------------------------------------------------------------------------

class TestCompareMvOrientation:
    def test_orientation_widget_shown_in_compare_mv(
        self, loaded_viewer, sid_3d, sid_compare_3d
    ):
        page = loaded_viewer(sid_3d)
        _enter_compare(page, "arr3d_compare")

        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_timeout(800)
        assert page.evaluate(_JS_ORIENTATION_VISIBLE), \
            "Orientation widget should be visible in compare multi-view"

    def test_orientation_widget_hidden_after_exit_compare_mv(
        self, loaded_viewer, sid_3d, sid_compare_3d
    ):
        page = loaded_viewer(sid_3d)
        _enter_compare(page, "arr3d_compare")
        _focus_kb(page)

        page.keyboard.press("v")
        page.wait_for_timeout(800)
        assert page.evaluate(_JS_ORIENTATION_VISIBLE)

        page.keyboard.press("v")
        page.wait_for_timeout(500)
        assert not page.evaluate(_JS_ORIENTATION_VISIBLE)


# ---------------------------------------------------------------------------
# Item 12: Compare overlay position -- overlay uses center diff pane
# ---------------------------------------------------------------------------

class TestCompareOverlayPosition:
    def test_overlay_uses_diff_pane_not_tertiary(
        self, loaded_viewer, sid_2d, sid_compare_2d
    ):
        """Overlay mode (mode 4) should use the diff pane with overlay-center class."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)

        # Cycle X to overlay: off->A-B->|A-B|->|A-B|/|A|->overlay (4 presses)
        for _ in range(4):
            page.keyboard.press("X")
            page.wait_for_timeout(250)

        page.wait_for_timeout(200)
        assert page.evaluate(_JS_HAS_OVERLAY_CENTER_CLASS), \
            "Diff pane should have overlay-center class"
        assert page.evaluate(_JS_DIFF_PANE_VISIBLE), \
            "Diff pane should be visible for overlay"


# ---------------------------------------------------------------------------
# Item 15: Compare zoom clipping -- canvases clip instead of rearranging
# ---------------------------------------------------------------------------

class TestCompareZoomClipping:
    def test_compare_zoom_clips_not_rearranges(
        self, loaded_viewer, sid_2d, sid_compare_2d
    ):
        """Zooming in compare mode should clip canvases, not rearrange them."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)
        page.wait_for_timeout(300)

        # Get initial canvas size
        size_before = page.evaluate(_JS_COMPARE_LEFT_CSS_SIZE)
        assert size_before is not None

        # Zoom in via + key
        for _ in range(5):
            page.keyboard.press("+")
            page.wait_for_timeout(200)

        page.wait_for_timeout(300)
        size_after = page.evaluate(_JS_COMPARE_LEFT_CSS_SIZE)
        assert size_after is not None

        # Canvas should have grown (zoom in)
        assert size_after[0] >= size_before[0], "Canvas width should grow or stay same on zoom"

        # Both left and right canvases should still be visible (not rearranged vertically)
        left_visible = page.is_visible("canvas#compare-left-canvas")
        right_visible = page.is_visible("canvas#compare-right-canvas")
        assert left_visible and right_visible

    def test_compare_zoom_sets_viewport_on_inner(
        self, loaded_viewer, sid_2d, sid_compare_2d
    ):
        """After zooming past viewport, .compare-canvas-inner gets explicit viewport dims."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)
        page.wait_for_timeout(300)

        # Zoom in significantly
        for _ in range(15):
            page.keyboard.press("+")
            page.wait_for_timeout(100)

        page.wait_for_timeout(300)
        dims = page.evaluate(_JS_CMP_INNER_DIMENSIONS)
        assert dims is not None
        assert dims["vpW"] is not None, "Viewport width should be set on inner"
        assert dims["vpH"] is not None, "Viewport height should be set on inner"


# ---------------------------------------------------------------------------
# Item 13: Wipe interaction
# ---------------------------------------------------------------------------

class TestWipeInteraction:
    def test_bracket_keys_adjust_wipe(self, loaded_viewer, sid_2d, sid_compare_2d):
        """[ and ] keys should adjust wipe position (status message changes)."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, "arr2d_cmp")
        _focus_kb(page)

        # Cycle X to wipe: off->A-B->|A-B|->|A-B|/|A|->overlay->wipe (5 presses)
        for _ in range(5):
            page.keyboard.press("X")
            page.wait_for_timeout(250)

        page.wait_for_timeout(200)
        assert page.evaluate(_JS_HAS_WIPE_MODE_CLASS)

        # Press ] to increase wipe
        page.keyboard.press("]")
        page.wait_for_timeout(200)
        status_after_right = page.inner_text("#status").lower()
        assert "wipe" in status_after_right

        # Press [ twice to decrease wipe
        page.keyboard.press("[")
        page.keyboard.press("[")
        page.wait_for_timeout(200)
        status_after_left = page.inner_text("#status").lower()
        assert "wipe" in status_after_left
        # The percentage should differ
        assert status_after_left != status_after_right, \
            "Wipe percentage should change with [ and ] keys"


# ---------------------------------------------------------------------------
# Zoom regression: verify no layout jumps
# ---------------------------------------------------------------------------

class TestZoomRegression:
    def test_single_array_zoom_in_out_stable(self, loaded_viewer, sid_2d):
        """Zoom in then back out, canvas size should return to original."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        size_before = page.evaluate("""
            () => {
                const c = document.querySelector('canvas#viewer');
                return c ? [parseInt(c.style.width), parseInt(c.style.height)] : null;
            }
        """)

        # Zoom in 3 times
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.wait_for_timeout(300)

        # Zoom out 3 times
        page.keyboard.press("-")
        page.keyboard.press("-")
        page.keyboard.press("-")
        page.wait_for_timeout(300)

        size_after = page.evaluate("""
            () => {
                const c = document.querySelector('canvas#viewer');
                return c ? [parseInt(c.style.width), parseInt(c.style.height)] : null;
            }
        """)

        # Allow 2px tolerance for rounding
        assert abs(size_before[0] - size_after[0]) <= 2, \
            f"Width should return to original: {size_before[0]} vs {size_after[0]}"
        assert abs(size_before[1] - size_after[1]) <= 2, \
            f"Height should return to original: {size_before[1]} vs {size_after[1]}"

    def test_normal_canvas_visible_after_all_modes(self, loaded_viewer, sid_3d, sid_compare_3d):
        """After entering and exiting several modes, the normal canvas should be visible."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)

        # Multi-view on/off
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.keyboard.press("v")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_CANVAS_HAS_CONTENT), "Canvas should have content after multiview exit"

        # Compare on/off
        page.keyboard.press("B")
        _pick_compare_session(page, "arr3d_compare")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("B")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_CANVAS_HAS_CONTENT), "Canvas should have content after compare exit"

    def test_elements_dont_disappear_on_zoom(self, loaded_viewer, sid_2d):
        """Info bar and colorbar should remain visible when zooming."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        assert page.is_visible("#info")
        assert page.is_visible("#slim-cb-wrap")

        # Zoom in
        for _ in range(5):
            page.keyboard.press("+")
        page.wait_for_timeout(300)

        assert page.is_visible("#info"), "Info bar should stay visible when zoomed"
        assert page.is_visible("#slim-cb-wrap"), "Colorbar should stay visible when zoomed"


# ---------------------------------------------------------------------------
# Item 17: Compact mode
# ---------------------------------------------------------------------------

_JS_HAS_COMPACT_MODE_CLASS = """
() => document.body.classList.contains('compact-mode')
"""

_JS_ARRAY_NAME_HIDDEN = """
() => {
    const el = document.getElementById('array-name');
    if (!el) return true;
    const s = getComputedStyle(el);
    return s.opacity === '0' || s.maxHeight === '0px' || s.display === 'none';
}
"""

_JS_CB_IS_VERTICAL = """
() => {
    const wrap = document.getElementById('slim-cb-wrap');
    return wrap && wrap.classList.contains('compact-vertical');
}
"""

class TestCompactMode:
    def test_K_toggles_compact_mode(self, loaded_viewer, sid_2d):
        """K key should toggle compact mode on and off."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        assert not page.evaluate(_JS_HAS_COMPACT_MODE_CLASS), "Should not start in compact mode"

        page.keyboard.press("K")
        page.wait_for_timeout(400)
        assert page.evaluate(_JS_HAS_COMPACT_MODE_CLASS), "K should activate compact mode"
        assert page.evaluate(_JS_ARRAY_NAME_HIDDEN), "Array name should be hidden in compact mode"

        page.keyboard.press("K")
        page.wait_for_timeout(400)
        assert not page.evaluate(_JS_HAS_COMPACT_MODE_CLASS), "K again should deactivate compact mode"

    def test_compact_mode_vertical_colorbar(self, loaded_viewer, sid_2d):
        """In compact mode, the colorbar should get the compact-vertical class."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        page.keyboard.press("K")
        page.wait_for_timeout(500)
        assert page.evaluate(_JS_CB_IS_VERTICAL), \
            "Colorbar should have compact-vertical class in compact mode"

        page.keyboard.press("K")
        page.wait_for_timeout(500)
        assert not page.evaluate(_JS_CB_IS_VERTICAL), \
            "Colorbar should not have compact-vertical class after exiting compact mode"

    def test_compact_mode_info_still_visible(self, loaded_viewer, sid_2d):
        """In compact mode, the dim bar (#info) should still be visible (just smaller)."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)

        page.keyboard.press("K")
        page.wait_for_timeout(400)
        assert page.is_visible("#info"), "Info bar should remain visible in compact mode"
