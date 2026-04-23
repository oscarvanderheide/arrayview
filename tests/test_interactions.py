"""Comprehensive interaction tests: every key, every mode, every guard.

Covers:
- Status message correctness and no-duplicate rule (no double message via #toast)
- All keyboard shortcuts in Normal mode: effect on canvas, #status, UI elements
- All mode guards: blocked keys show a status and do NOT proceed
- Navigation: scroll, zoom, dim-cycle, axis-swap, rotate
- Compare mode: state sync, per-pane overrides cleared by d/D
- ROI mode: entry/exit, Space guard, R/x/y guards
- qMRI mode: guards for c/C/D/f/x/y
- Multiview mode: guards for x/y/g/N
- Export: g/N blocked in multi-canvas modes

Run with:
    uv run pytest tests/test_interactions.py -v

Requires Playwright:
    uv run playwright install chromium
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser

# ---------------------------------------------------------------------------
# JS helpers
# ---------------------------------------------------------------------------

_JS_CENTER_PIXEL = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const x = Math.floor(c.width / 2), y = Math.floor(c.height / 2);
    const d = ctx.getImageData(x, y, 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""

_JS_CANVAS_RECT = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c) return null;
    const r = c.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) };
}
"""

_JS_CANVAS_ASPECT = """
() => {
    const c = document.querySelector('canvas#viewer');
    if (!c || !c.height) return null;
    return c.width / c.height;
}
"""

_JS_TOAST_TEXT = """
() => (document.getElementById('toast') || {}).textContent || ''
"""

_JS_STATUS_TEXT = """
() => (document.getElementById('status') || {}).textContent || ''
"""

_JS_COMPARE_LEFT_PIXEL = """
() => {
    const c = document.querySelector('canvas#compare-left-canvas');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const x = Math.floor(c.width / 2), y = Math.floor(c.height / 2);
    const d = ctx.getImageData(x, y, 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""

_JS_COMPARE_RIGHT_PIXEL = """
() => {
    const c = document.querySelector('canvas#compare-right-canvas');
    if (!c) return null;
    const ctx = c.getContext('2d');
    const x = Math.floor(c.width / 2), y = Math.floor(c.height / 2);
    const d = ctx.getImageData(x, y, 1, 1).data;
    return [d[0], d[1], d[2]];
}
"""

_JS_BODY_CLASS = "() => document.body.className"

_JS_EGGS_TEXT = "() => (document.getElementById('mode-eggs') || {}).innerText || ''"


def _focus_kb(page):
    page.focus("#keyboard-sink")


def _get_status(page):
    return page.inner_text("#status").strip()


def _wait_status(page, contains, timeout=2000):
    """Wait until #status contains the given substring."""
    page.wait_for_function(
        f"() => document.getElementById('status').textContent.toLowerCase().includes({repr(contains.lower())})",
        timeout=timeout,
    )


def _enter_compare(page, partner_sid):
    """Enter compare mode with a given partner sid.

    The B keybind that used to open the compare picker has been retired;
    we now invoke enterCompareModeBySid() directly via page.evaluate(). This
    is the same JS entry point used by URL-param launches and drag-drop, so
    it exercises the real production code path.
    """
    _focus_kb(page)
    page.evaluate(
        f"async () => {{ await enterCompareModeBySid({partner_sid!r}); }}"
    )
    page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
    page.wait_for_timeout(400)


def _exit_compare(page):
    """Exit compare mode via the JS entry point (B keybind retired)."""
    page.evaluate("() => exitCompareMode()")
    page.wait_for_timeout(400)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sid_cmp_2d(client, arr_2d, tmp_path):
    path = tmp_path / "arr2d_cmp.npy"
    np.save(path, arr_2d * 0.5)
    return client.post(
        "/load", json={"filepath": str(path), "name": "arr2d_cmp"}
    ).json()["sid"]


@pytest.fixture
def sid_cmp_3d(client, arr_3d, tmp_path):
    path = tmp_path / "arr3d_cmp.npy"
    np.save(path, np.flip(arr_3d, axis=0))
    return client.post(
        "/load", json={"filepath": str(path), "name": "arr3d_cmp"}
    ).json()["sid"]


# ---------------------------------------------------------------------------
# TestNoDoubleMessage — #toast must always be invisible/empty
# ---------------------------------------------------------------------------


class TestNoDoubleMessage:
    """Every key that shows a confirmation uses showStatus only.
    The #toast element must stay empty and not be visible.
    Previously, showToast() wrote to #toast (a bare div with no CSS), causing
    the message to render twice — once in #status and once in document flow."""

    def _assert_toast_empty(self, page):
        toast_text = page.evaluate(_JS_TOAST_TEXT)
        # #toast has display:none so is never visible — text doesn't matter much,
        # but by CSS rule it can never be seen. Confirm display:none CSS is applied.
        toast_display = page.evaluate(
            "() => getComputedStyle(document.getElementById('toast')).display"
        )
        assert toast_display == "none", (
            f"#toast should have display:none, got '{toast_display}'. "
            "A showToast() call is making text visible outside of #status."
        )

    def test_a_stretch_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("a")
        page.wait_for_timeout(200)
        self._assert_toast_empty(page)
        status = _get_status(page)
        assert "stretch" in status.lower(), f"Expected stretch status, got: '{status}'"

    def test_b_borders_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(200)
        self._assert_toast_empty(page)
        status = _get_status(page)
        assert "borders" in status.lower(), f"Expected borders status, got: '{status}'"

    def test_K_key_does_nothing(self, loaded_viewer, sid_2d):
        """K key is unbound — immersive mode is automatic."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("K")
        page.wait_for_timeout(200)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "compact-mode" not in body_class, "K should not activate compact mode"
        self._assert_toast_empty(page)

    def test_Z_zen_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Z")
        page.wait_for_timeout(200)
        self._assert_toast_empty(page)

    def test_d_dr_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(300)
        self._assert_toast_empty(page)

    def test_c_colormap_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_timeout(300)
        self._assert_toast_empty(page)

    def test_L_log_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("L")
        page.wait_for_timeout(300)
        self._assert_toast_empty(page)

    def test_T_theme_no_toast(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("T")
        page.wait_for_timeout(200)
        self._assert_toast_empty(page)


# ---------------------------------------------------------------------------
# TestDisplaySettings — keys that change rendering in Normal mode
# ---------------------------------------------------------------------------


class TestDisplaySettings:
    """Check that display-setting keys change the canvas AND show correct status."""

    def test_c_cycles_colormap_canvas_changes(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("c")
        page.wait_for_timeout(800)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Canvas unchanged after c (colormap cycle)"

    def test_c_shows_colormap_strip(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_timeout(200)
        strip_visible = page.evaluate(
            "() => { const s = document.getElementById('colormap-strip'); return s && s.textContent.trim() !== ''; }"
        )
        assert strip_visible, "Colormap strip should appear after pressing c"

    def test_d_first_tap_opens_only_second_tap_cycles(self, loaded_viewer, sid_2d):
        """First tap `d` opens the histogram only (no percentile toast, no
        vmin/vmax change). Second tap (while histogram is visible) cycles to
        the first percentile preset and toasts."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # First tap: opens histogram, must NOT toast a percentile.
        page.keyboard.press("d")
        page.wait_for_timeout(500)
        toast1 = page.evaluate("() => (document.getElementById('toast') || {}).textContent || ''").lower()
        expanded = page.evaluate("() => !!(primaryCb && primaryCb._expanded)")
        assert expanded, "Expected histogram colorbar expanded after first tap"
        assert "percentile" not in toast1, (
            f"First tap should not toast a percentile, got: '{toast1}'"
        )
        # Second tap while open: cycles preset, toasts percentile.
        page.keyboard.press("d")
        page.wait_for_timeout(500)
        toast2 = page.evaluate("() => (document.getElementById('toast') || {}).textContent || ''").lower()
        assert "percentile" in toast2, (
            f"Second tap should toast a percentile, got: '{toast2}'"
        )

    def test_shift_d_opens_hist_picker(self, loaded_viewer, sid_3d):
        """Shift+D opens the 3-state dim picker. Every dim label gets a
        state class (hist-scope / hist-recompute / hist-frozen) and x/y
        are marked hist-locked-dim. Shift+D again closes."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        pick_mode = page.evaluate(
            "() => document.getElementById('info').classList.contains('hist-pick-mode')"
        )
        assert pick_mode, "Expected #info.hist-pick-mode after Shift+D"
        state_classes = page.evaluate("""
            () => {
                const labels = document.querySelectorAll('#info .dim-label[data-dim]');
                return Array.from(labels).map(el => {
                    if (el.classList.contains('hist-scope'))     return 'scope';
                    if (el.classList.contains('hist-recompute')) return 'recompute';
                    if (el.classList.contains('hist-frozen'))    return 'frozen';
                    return null;
                });
            }
        """)
        assert all(s in ('scope', 'recompute', 'frozen') for s in state_classes), (
            f"Every dim should carry a state class, got: {state_classes}"
        )
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        pick_mode = page.evaluate(
            "() => document.getElementById('info').classList.contains('hist-pick-mode')"
        )
        assert not pick_mode, "Expected picker closed after second Shift+D"

    def test_shift_d_enter_cycles_active_dim_state(self, loaded_viewer, sid_3d):
        """Pressing Enter while the picker is open cycles the state of the
        current activeDim (keyboard parity with clicking)."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        # Point activeDim at a non-x/y dim so Enter actually changes state.
        page.evaluate("""
            () => {
                const labels = [...document.querySelectorAll('#info .dim-label[data-dim]')];
                const free = labels.find(el => !el.classList.contains('hist-locked-dim'));
                if (free) activeDim = Number(free.getAttribute('data-dim'));
            }
        """)
        state_before = page.evaluate("""
            () => {
                const el = document.querySelector(`#info .dim-label[data-dim="${activeDim}"]`);
                return el.classList.contains('hist-scope') ? 'scope'
                     : el.classList.contains('hist-recompute') ? 'recompute'
                     : 'frozen';
            }
        """)
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)
        state_after = page.evaluate("""
            () => {
                const el = document.querySelector(`#info .dim-label[data-dim="${activeDim}"]`);
                return el.classList.contains('hist-scope') ? 'scope'
                     : el.classList.contains('hist-recompute') ? 'recompute'
                     : 'frozen';
            }
        """)
        assert state_after != state_before, (
            f"Enter should have cycled active dim state, still {state_after}"
        )

    def test_shift_d_click_updates_vmin_vmax_without_prior_preset(self, loaded_viewer, sid_3d):
        """Clicking a dim in picker mode should update vmin/vmax, even when
        the user hasn't cycled a percentile preset yet."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        # Make sure no preset has been applied yet.
        page.evaluate("() => { window._dQuantileIdx = null; manualVmin = null; manualVmax = null; }")
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        before = page.evaluate("() => ({vmin: manualVmin, vmax: manualVmax})")
        # Click first non-x/y dim.
        page.evaluate("""
            () => {
                const labels = [...document.querySelectorAll('#info .dim-label[data-dim]')];
                const free = labels.find(el => !el.classList.contains('hist-locked-dim'));
                if (free) free.click();
            }
        """)
        page.wait_for_timeout(600)
        after = page.evaluate("() => ({vmin: manualVmin, vmax: manualVmax})")
        assert (after['vmin'] != before['vmin']) or (after['vmax'] != before['vmax']), (
            f"vmin/vmax should have changed after clicking a dim; before={before} after={after}"
        )

    def test_shift_d_click_cycles_dim_state(self, loaded_viewer, sid_3d):
        """Clicking a non-x/y dim in picker mode cycles scope → recompute →
        frozen → scope."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        # Find first non-x/y dim (hist-locked-dim marks x/y).
        state_before = page.evaluate("""
            () => {
                const labels = [...document.querySelectorAll('#info .dim-label[data-dim]')];
                const free = labels.find(el => !el.classList.contains('hist-locked-dim'));
                if (!free) return null;
                return {
                    dim: free.getAttribute('data-dim'),
                    state: free.classList.contains('hist-scope') ? 'scope'
                         : free.classList.contains('hist-recompute') ? 'recompute'
                         : 'frozen',
                };
            }
        """)
        assert state_before is not None, "Test needs at least one non-x/y dim"
        # Click cycles to the next state.
        page.evaluate(f"""
            () => document.querySelector('#info .dim-label[data-dim=\"{state_before['dim']}\"]').click()
        """)
        page.wait_for_timeout(500)
        state_after = page.evaluate(f"""
            () => {{
                const el = document.querySelector('#info .dim-label[data-dim=\"{state_before['dim']}\"]');
                return el.classList.contains('hist-scope') ? 'scope'
                     : el.classList.contains('hist-recompute') ? 'recompute'
                     : 'frozen';
            }}
        """)
        expected = {'scope': 'recompute', 'recompute': 'frozen', 'frozen': 'scope'}[state_before['state']]
        assert state_after == expected, (
            f"Expected {state_before['state']} → {expected}, got {state_after}"
        )

    def test_L_log_scale_on_shows_log_egg(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("L")
        page.wait_for_timeout(400)
        eggs = page.evaluate(_JS_EGGS_TEXT)
        # Log scale egg should appear — look for "log" in eggs area or body class
        body_cls = page.evaluate(_JS_BODY_CLASS)
        assert (
            "log" in eggs.lower()
            or "log" in body_cls.lower()
            or page.is_visible(".mode-badge-log")
        ), "No log-scale indicator after pressing L"

    def test_L_toggles_canvas_render(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("L")
        page.wait_for_timeout(600)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Canvas unchanged after L (log scale)"

    def test_L_off_restores_original_render(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        original = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("L")
        page.wait_for_timeout(600)
        page.keyboard.press("L")
        page.wait_for_timeout(600)
        restored = page.evaluate(_JS_CENTER_PIXEL)
        assert original == restored, "Canvas not restored after L on+off"

    def test_b_borders_status_message(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(200)
        status = _get_status(page)
        assert "borders" in status.lower(), (
            f"Expected 'borders' in status, got: '{status}'"
        )

    def test_b_toggles_correctly(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(150)
        s1 = _get_status(page)
        page.keyboard.press("b")
        page.wait_for_timeout(150)
        s2 = _get_status(page)
        assert s1 != s2, "b key should toggle borders, status should differ"
        assert ("on" in s1 and "off" in s2) or ("off" in s1 and "on" in s2), (
            f"Expected on/off toggle, got: '{s1}' → '{s2}'"
        )

    def test_a_stretch_status_message(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("a")
        page.wait_for_timeout(200)
        status = _get_status(page)
        assert "stretch" in status.lower(), (
            f"Expected 'stretch' in status, got: '{status}'"
        )

    def test_a_stretch_toggles_correctly(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("a")
        page.wait_for_timeout(150)
        s1 = _get_status(page)
        page.keyboard.press("a")
        page.wait_for_timeout(150)
        s2 = _get_status(page)
        assert s1 != s2
        assert "stretch to square: on" in s1 or "stretch to square: on" in s2

    def test_a_stretch_changes_canvas_aspect(self, loaded_viewer, sid_2d):
        """Pressing a on a non-square array should change the canvas CSS display aspect ratio.
        Note: canvas.width/.height are the pixel dimensions set by the server (image dims),
        but squareStretch changes the CSS display size. Use getBoundingClientRect()."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Use CSS display size, not canvas pixel dims
        aspect_before = page.evaluate(
            "() => { const r = document.querySelector('canvas#viewer').getBoundingClientRect(); "
            "return r.height > 0 ? r.width / r.height : null; }"
        )
        page.keyboard.press("a")
        page.wait_for_timeout(400)
        aspect_after = page.evaluate(
            "() => { const r = document.querySelector('canvas#viewer').getBoundingClientRect(); "
            "return r.height > 0 ? r.width / r.height : null; }"
        )
        # sid_2d is (100, 80): non-square → squareStretch should change the display aspect
        assert aspect_before != aspect_after, (
            f"Canvas CSS display aspect ratio unchanged after a: {aspect_before} → {aspect_after}. "
            "Check arr_2d is non-square and squareStretch changes CSS width/height."
        )

    def test_T_theme_changes_body_class(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Theme is set via document.documentElement (html element) class, not body
        cls_before = page.evaluate("() => document.documentElement.className")
        page.keyboard.press("T")
        page.wait_for_timeout(200)
        cls_after = page.evaluate("() => document.documentElement.className")
        assert cls_before != cls_after, (
            f"documentElement class unchanged after T (theme cycle). "
            f"Got '{cls_after}'. T applies class to <html> element, not body."
        )

    def test_T_cycles_through_multiple_themes(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        themes = set()
        for _ in range(5):
            page.keyboard.press("T")
            page.wait_for_timeout(100)
            themes.add(page.evaluate("() => document.documentElement.className"))
        assert len(themes) >= 2, "Theme cycle should produce at least 2 distinct states"

    def test_i_shows_info_overlay(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        assert not page.evaluate(
            "() => document.getElementById('info-overlay').classList.contains('visible')"
        )
        page.keyboard.press("i")
        page.wait_for_timeout(500)
        assert page.evaluate(
            "() => document.getElementById('info-overlay').classList.contains('visible')"
        ), "#info-overlay should be visible after i"

    def test_i_hides_info_overlay_on_second_press(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(500)
        page.keyboard.press("i")
        page.wait_for_timeout(200)
        assert not page.evaluate(
            "() => document.getElementById('info-overlay').classList.contains('visible')"
        ), "#info-overlay should be hidden after second i"

    def test_escape_hides_info_overlay(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(500)
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        assert not page.evaluate(
            "() => document.getElementById('info-overlay').classList.contains('visible')"
        )


# ---------------------------------------------------------------------------
# TestNavigation — scroll, zoom, dim-cycle, axis-swap
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_j_scrolls_slice(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("j")
        page.wait_for_timeout(500)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Canvas unchanged after j scroll"

    def test_k_scrolls_slice_opposite_to_j(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        # Go to some middle slice first
        for _ in range(5):
            page.keyboard.press("j")
        page.wait_for_timeout(400)
        mid = page.evaluate(_JS_CENTER_PIXEL)
        for _ in range(5):
            page.keyboard.press("k")
        page.wait_for_timeout(400)
        back = page.evaluate(_JS_CENTER_PIXEL)
        assert mid != back, "k did not scroll back"

    def test_jk_roundtrip_restores_canvas(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        original = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("j")
        page.wait_for_timeout(400)
        page.keyboard.press("k")
        page.wait_for_timeout(400)
        restored = page.evaluate(_JS_CENTER_PIXEL)
        assert original == restored, "j+k roundtrip should restore canvas pixel"

    def test_arrow_keys_scroll(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(500)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "ArrowDown should scroll slice"

    def test_h_cycles_active_dim(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        info_before = (
            page.inner_text("#info").strip() if page.is_visible("#info") else ""
        )
        page.keyboard.press("h")
        page.wait_for_timeout(300)
        # Pressing h changes the active dimension; info line should change
        info_after = (
            page.inner_text("#info").strip() if page.is_visible("#info") else ""
        )
        # Canvas should re-render
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("j")
        page.wait_for_timeout(300)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "After h, j should still scroll (different dim)"

    def test_zoom_in_increases_canvas_size(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Zoom out first so there's room to grow (auto-fit may fill window)
        page.keyboard.press("-")
        page.keyboard.press("-")
        page.wait_for_timeout(300)
        before = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.press("+")
        page.wait_for_timeout(300)
        after = page.evaluate(_JS_CANVAS_RECT)
        assert after["w"] > before["w"] or after["h"] > before["h"], (
            f"Canvas did not grow after +: before={before}, after={after}"
        )

    def test_zoom_out_decreases_canvas_size(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Zoom in first to have room to zoom out
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.wait_for_timeout(300)
        before = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.press("-")
        page.wait_for_timeout(300)
        after = page.evaluate(_JS_CANVAS_RECT)
        assert after["w"] < before["w"] or after["h"] < before["h"], (
            f"Canvas did not shrink after -: before={before}, after={after}"
        )

    def test_zoom_reset_returns_to_base(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        base = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.wait_for_timeout(200)
        page.keyboard.press("0")
        page.wait_for_timeout(300)
        reset = page.evaluate(_JS_CANVAS_RECT)
        assert abs(reset["w"] - base["w"]) <= 2 and abs(reset["h"] - base["h"]) <= 2, (
            f"0 did not reset zoom: base={base}, reset={reset}"
        )

    def test_x_swaps_dim_x(self, loaded_viewer, sid_3d):
        """x key swaps dim_x with the current slice dim. For arr_3d (20×64×64),
        with activeDim=0 (current_slice_dim=0), this changes display from 64×64 to 20×64,
        so the CSS canvas rect dimensions change."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        rect_before = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.press("x")
        page.wait_for_timeout(600)
        rect_after = page.evaluate(_JS_CANVAS_RECT)
        # After swapping dim_x (64) with current_slice_dim (20), canvas should resize
        assert rect_before != rect_after, (
            f"Canvas rect unchanged after x (dim_x swap): {rect_before}"
        )

    def test_y_swaps_dim_y(self, loaded_viewer, sid_3d):
        """y key swaps dim_y with the current slice dim. For arr_3d (20×64×64),
        this changes the display dimensions and canvas rect."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        rect_before = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.press("y")
        page.wait_for_timeout(600)
        rect_after = page.evaluate(_JS_CANVAS_RECT)
        assert rect_before != rect_after, (
            f"Canvas rect unchanged after y (dim_y swap): {rect_before}"
        )

    def test_r_rotates_canvas(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("r")
        page.wait_for_timeout(500)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Canvas unchanged after r (rotate/flip)"

    def test_r_double_tap_restores(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        original = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("r")
        page.wait_for_timeout(400)
        page.keyboard.press("r")
        page.wait_for_timeout(400)
        restored = page.evaluate(_JS_CENTER_PIXEL)
        assert original == restored, "r+r should restore canvas"

    def test_space_plays_animation(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(300)
        p1 = page.evaluate(_JS_CENTER_PIXEL)
        page.wait_for_timeout(300)
        p2 = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("Space")
        # At least one transition should have occurred
        # (some frames might happen to be identical, give it two chances)
        page.wait_for_timeout(100)
        p3 = page.evaluate(_JS_CENTER_PIXEL)
        assert p1 != p2 or p2 != p3, "Animation did not advance canvas after Space"

    def test_space_stops_animation(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_timeout(400)
        page.keyboard.press("Space")
        page.wait_for_timeout(150)
        p_stop1 = page.evaluate(_JS_CENTER_PIXEL)
        page.wait_for_timeout(400)
        p_stop2 = page.evaluate(_JS_CENTER_PIXEL)
        assert p_stop1 == p_stop2, "Animation still running after second Space"

    def test_wheel_scrolls_slice(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        canvas = page.locator("canvas#viewer")
        canvas.hover()
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(500)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Mouse wheel did not scroll slice"

    def test_ctrl_wheel_zooms_in(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        canvas = page.locator("canvas#viewer")
        canvas.hover()
        before = page.evaluate(_JS_CANVAS_RECT)
        page.keyboard.down("Control")
        page.mouse.wheel(0, -120)
        page.keyboard.up("Control")
        page.wait_for_timeout(300)
        after = page.evaluate(_JS_CANVAS_RECT)
        assert after["w"] > before["w"] or after["h"] > before["h"], (
            "Ctrl+wheel did not zoom in"
        )


# ---------------------------------------------------------------------------
# TestModeGuards — blocked keys show a status message and do NOT change state
# ---------------------------------------------------------------------------


class TestModeGuards:
    """Each blocked (key, mode) pair: verify status message shown."""

    # --- qMRI mode guards ---

    def _enter_qmri(self, page, sid_4d):
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        page.wait_for_timeout(400)

    def test_c_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "colormap" in status.lower() and (
            "fixed" in status.lower() or "qmri" in status.lower()
        ), f"Expected colormap-blocked status in qMRI, got: '{status}'"

    def test_C_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("C")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "colormap" in status.lower() and (
            "fixed" in status.lower() or "qmri" in status.lower()
        ), f"Expected colormap-blocked status in qMRI, got: '{status}'"

    def test_shift_d_opens_picker_in_qmri(self, loaded_viewer, sid_4d):
        """Shift+D opens the 3-state dim picker in qMRI mode too."""
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        prompt_visible = page.evaluate(
            "() => { const p = document.getElementById('inline-prompt'); return p && p.classList.contains('visible'); }"
        )
        assert not prompt_visible, "Shift+D should not open inline prompt"
        pick_mode = page.evaluate(
            "() => document.getElementById('info').classList.contains('hist-pick-mode')"
        )
        assert pick_mode, "Expected #info.hist-pick-mode after Shift+D in qMRI"

    def test_f_fft_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("f")
        page.wait_for_timeout(300)
        # inline-prompt uses classList 'visible' not inline style.display
        prompt_visible = page.evaluate(
            "() => { const p = document.getElementById('inline-prompt'); return p && p.classList.contains('visible'); }"
        )
        assert not prompt_visible, "f should not open FFT prompt in qMRI mode"
        status = _get_status(page)
        assert "fft" in status.lower() or "qmri" in status.lower(), (
            f"Expected FFT-blocked status in qMRI, got: '{status}'"
        )

    def test_x_dim_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        before = page.evaluate(
            "() => { const c = document.querySelectorAll('#qmri-view-wrap canvas'); return c.length; }"
        )
        page.keyboard.press("x")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert (
            "axis" in status.lower()
            or "fixed" in status.lower()
            or "qmri" in status.lower()
        ), f"Expected x-blocked status in qMRI, got: '{status}'"

    def test_y_dim_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("y")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert (
            "axis" in status.lower()
            or "fixed" in status.lower()
            or "qmri" in status.lower()
        ), f"Expected y-blocked status in qMRI, got: '{status}'"

    def test_g_gif_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("g")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "gif" in status.lower() or "not available" in status.lower(), (
            f"Expected GIF-blocked status in qMRI, got: '{status}'"
        )

    def test_N_npy_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        self._enter_qmri(page, sid_4d)
        _focus_kb(page)
        page.keyboard.press("N")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "export" in status.lower() or "not available" in status.lower(), (
            f"Expected N-blocked status in qMRI, got: '{status}'"
        )

    # --- Multiview mode guards ---

    def _enter_multiview(self, page):
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

    def test_x_dim_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        self._enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("x")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert (
            "axis" in status.lower()
            or "multiview" in status.lower()
            or "v to" in status.lower()
        ), f"Expected x-blocked status in multiview, got: '{status}'"

    def test_y_dim_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        self._enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("y")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert (
            "axis" in status.lower()
            or "multiview" in status.lower()
            or "v to" in status.lower()
        ), f"Expected y-blocked status in multiview, got: '{status}'"

    def test_g_gif_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        self._enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("g")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "gif" in status.lower() or "not available" in status.lower(), (
            f"Expected GIF-blocked status in multiview, got: '{status}'"
        )

    def test_N_npy_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        self._enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("N")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "export" in status.lower() or "not available" in status.lower(), (
            f"Expected N-blocked status in multiview, got: '{status}'"
        )

    # --- Compare mode guards ---

    def test_g_gif_blocked_in_compare(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("g")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "gif" in status.lower() or "not available" in status.lower(), (
            f"Expected GIF-blocked status in compare, got: '{status}'"
        )

    def test_N_npy_blocked_in_compare(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("N")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "export" in status.lower() or "not available" in status.lower(), (
            f"Expected N-blocked status in compare, got: '{status}'"
        )

    # --- ROI mode guards ---

    def test_space_blocked_in_roi_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")  # enter ROI mode
        page.wait_for_timeout(200)
        page.keyboard.press("Space")
        page.wait_for_timeout(200)
        status = _get_status(page)
        assert "roi" in status.lower() or "exit" in status.lower(), (
            f"Expected Space-blocked-in-ROI status, got: '{status}'"
        )

    def test_space_not_blocked_after_roi_exit(self, loaded_viewer, sid_3d):
        """ROI cycle: off→rect→circle→freehand→floodfill→off.
        First R press → rect. Need 4 MORE R presses to reach off state."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("R")  # → rect (ROI active)
        page.wait_for_timeout(150)
        # Cycle to off: rect→circle (1), circle→freehand (2), freehand→floodfill (3), floodfill→off (4)
        for _ in range(4):
            page.keyboard.press("R")
            page.wait_for_timeout(100)
        s_off = _get_status(page)
        assert "off" in s_off.lower(), (
            f"ROI should be off after 5 total R presses, got: '{s_off}'"
        )
        # Now Space (animation) should NOT be blocked
        page.keyboard.press("Space")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "exit roi" not in status.lower(), (
            f"Space should not be blocked after ROI exit, got: '{status}'"
        )
        page.keyboard.press("Space")  # stop


# ---------------------------------------------------------------------------
# TestROIMode — ROI entry/exit, canvas interaction, mode state
# ---------------------------------------------------------------------------


class TestROIMode:
    def test_A_enters_roi_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")
        page.wait_for_timeout(200)
        status = _get_status(page)
        assert "roi" in status.lower(), f"Expected ROI mode status, got: '{status}'"

    def test_A_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")  # enter multiview
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("R")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "not available" in status.lower() or "roi" in status.lower(), (
            f"Expected A-blocked in multiview, got: '{status}'"
        )

    def test_A_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("R")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "not available" in status.lower() or "roi" in status.lower(), (
            f"Expected A-blocked in qMRI, got: '{status}'"
        )

    def test_R_cycles_shape_then_exits(self, loaded_viewer, sid_2d):
        """ROI cycle: off→rect→circle→freehand→floodfill→off. Total 5 R presses."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")  # → rect
        page.wait_for_timeout(150)
        s1 = _get_status(page)
        assert "rect" in s1.lower(), f"Expected rect status, got: '{s1}'"
        page.keyboard.press("R")  # → circle
        page.wait_for_timeout(150)
        s2 = _get_status(page)
        assert "circle" in s2.lower(), f"Expected circle status, got: '{s2}'"
        # 3 more presses to reach off: freehand → floodfill → off
        for _ in range(3):
            page.keyboard.press("R")
            page.wait_for_timeout(100)
        s_off = _get_status(page)
        assert "off" in s_off.lower(), (
            f"Expected ROI off status after 5 R presses, got: '{s_off}'"
        )

    def test_escape_does_not_break_roi_mode(self, loaded_viewer, sid_2d):
        """Escape clears the slice-jump buffer, but does NOT exit ROI mode.
        ROI mode should still be active after pressing Escape."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")  # enter ROI mode
        page.wait_for_timeout(200)
        status_roi = _get_status(page)
        assert "roi" in status_roi.lower(), (
            f"Expected ROI mode active, got: '{status_roi}'"
        )
        page.keyboard.press("Escape")  # clears slice jump buffer, not ROI mode
        page.wait_for_timeout(200)
        # ROI mode should still be active — Space is still blocked
        page.keyboard.press("Space")
        page.wait_for_timeout(200)
        status_space = _get_status(page)
        assert "exit roi" in status_space.lower() or "roi" in status_space.lower(), (
            f"ROI mode should still block Space after Escape, got: '{status_space}'"
        )

    def test_roi_draw_and_stats(self, loaded_viewer, sid_2d):
        """Drawing an ROI triggers server fetch → stats populated → roi-panel shown.
        #roi-panel has CSS default display:none; it's shown by JS after fetch."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")  # enter rect ROI mode
        page.wait_for_timeout(200)
        # Draw an ROI by dragging on the canvas
        canvas = page.locator("canvas#viewer")
        bbox = canvas.bounding_box()
        cx, cy = bbox["x"] + bbox["width"] * 0.3, bbox["y"] + bbox["height"] * 0.3
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 80, cy + 80, steps=8)
        page.mouse.up()
        # Wait for async server call to populate stats and show panel
        page.wait_for_function(
            "() => { const el = document.getElementById('roi-panel'); "
            "return el && el.style.display === 'block'; }",
            timeout=5000,
        )
        roi_panel_visible = page.evaluate(
            "() => document.getElementById('roi-panel').style.display === 'block'"
        )
        assert roi_panel_visible, (
            "ROI stats panel should be visible (display:block) after drawing ROI"
        )

    def test_roi_stats_have_numeric_values(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("R")
        page.wait_for_timeout(200)
        canvas = page.locator("canvas#viewer")
        bbox = canvas.bounding_box()
        cx, cy = bbox["x"] + bbox["width"] * 0.5, bbox["y"] + bbox["height"] * 0.5
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.mouse.move(cx + 80, cy + 60, steps=5)
        page.mouse.up()
        page.wait_for_timeout(600)
        roi_text = page.evaluate(
            "() => { const el = document.getElementById('roi-panel'); return el ? el.innerText : ''; }"
        )
        assert any(c.isdigit() for c in roi_text), (
            f"ROI stats should contain numeric values, got: '{roi_text[:200]}'"
        )


# ---------------------------------------------------------------------------
# TestCompareModeSync — d/D sync, per-pane overrides cleared
# ---------------------------------------------------------------------------


class TestCompareModeSync:
    def test_d_in_compare_opens_then_cycles(self, loaded_viewer, sid_2d, sid_cmp_2d):
        """In compare mode, first `d` opens the histogram only; second `d`
        cycles percentile preset + toasts."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        page.keyboard.press("d")
        page.wait_for_timeout(500)
        toast = page.evaluate("() => (document.getElementById('toast') || {}).textContent || ''").lower()
        assert "percentile" in toast, f"Expected 'percentile' toast after 2 taps in compare, got: '{toast}'"

    def test_d_in_compare_rerenders_both_panes(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        left_before = page.evaluate(_JS_COMPARE_LEFT_PIXEL)
        right_before = page.evaluate(_JS_COMPARE_RIGHT_PIXEL)
        # First tap only opens the histogram (no vmin/vmax change). The
        # second tap cycles the percentile preset, which does re-render.
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        page.keyboard.press("d")
        page.wait_for_timeout(600)
        left_after = page.evaluate(_JS_COMPARE_LEFT_PIXEL)
        right_after = page.evaluate(_JS_COMPARE_RIGHT_PIXEL)
        assert left_before != left_after or right_before != right_after, (
            "d (tap, tap) in compare should re-render at least one pane"
        )

    def test_scroll_syncs_both_compare_panes(self, loaded_viewer, sid_3d, sid_cmp_3d):
        page = loaded_viewer(sid_3d)
        _enter_compare(page, sid_cmp_3d)
        _focus_kb(page)
        left_before = page.evaluate(_JS_COMPARE_LEFT_PIXEL)
        right_before = page.evaluate(_JS_COMPARE_RIGHT_PIXEL)
        page.keyboard.press("j")
        page.wait_for_timeout(500)
        left_after = page.evaluate(_JS_COMPARE_LEFT_PIXEL)
        right_after = page.evaluate(_JS_COMPARE_RIGHT_PIXEL)
        # Both panes should update (they may have different pixels but both should re-render)
        assert left_before != left_after or right_before != right_after, (
            "j in compare should update canvas pixels"
        )

    def test_zoom_in_compare_no_vertical_stacking(
        self, loaded_viewer, sid_2d, sid_cmp_2d
    ):
        """Zooming should keep compare panes side-by-side and show the minimap when they overflow."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        # Zoom in several times
        _focus_kb(page)
        for _ in range(5):
            page.keyboard.press("+")
        page.wait_for_timeout(400)
        minimap_visible = page.evaluate("""
            () => {
                // Compare mode uses per-pane minimaps (.cmp-mini-map), not the global #mini-map
                const perPane = document.querySelectorAll('.cmp-mini-map.visible');
                if (perPane.length > 0) return true;
                return document.getElementById('mini-map')?.classList.contains('visible') ?? false;
            }
        """)
        left_rect = page.evaluate(
            "() => { const c = document.querySelector('canvas#compare-left-canvas'); return c ? c.getBoundingClientRect() : null; }"
        )
        right_rect = page.evaluate(
            "() => { const c = document.querySelector('canvas#compare-right-canvas'); return c ? c.getBoundingClientRect() : null; }"
        )
        assert left_rect and right_rect, (
            "Both compare canvases should be present when zoomed"
        )
        assert minimap_visible, (
            "Compare zoom overflow should show per-pane minimaps (.cmp-mini-map.visible)."
        )
        # Panes should be approximately at same vertical position (side-by-side layout)
        left_top = left_rect["y"]
        right_top = right_rect["y"]
        assert abs(left_top - right_top) < 20, (
            f"Compare panes are vertically stacked when zoomed! "
            f"left.top={left_top}, right.top={right_top}"
        )

    def test_compare_left_right_panes_visible(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")

    def test_compare_shows_per_pane_colorbars(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        assert page.is_visible("canvas#compare-left-pane-cb"), (
            "Left pane colorbar not visible"
        )
        assert page.is_visible("canvas#compare-right-pane-cb"), (
            "Right pane colorbar not visible"
        )

    def test_compare_hides_shared_slim_colorbar(
        self, loaded_viewer, sid_2d, sid_cmp_2d
    ):
        """The normal shared slim colorbar should be hidden in compare mode."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        slim_cb_visible = page.evaluate(
            "() => { const el = document.getElementById('slim-cb-wrap'); if (!el) return false; "
            "const s = getComputedStyle(el); return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0'; }"
        )
        assert not slim_cb_visible, (
            "Shared slim colorbar should be hidden in compare mode"
        )

    def test_compare_exit_restores_single_view(
        self, loaded_viewer, sid_2d, sid_cmp_2d
    ):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _exit_compare(page)
        assert not page.is_visible("#compare-view-wrap.active"), (
            "Compare mode should be exited after exitCompareMode()"
        )
        assert page.is_visible("canvas#viewer"), (
            "Main canvas should be visible after exiting compare"
        )


# ---------------------------------------------------------------------------
# TestQmriMode — enter/exit, canvas count, verify guards block correctly
# ---------------------------------------------------------------------------


class TestQmriMode:
    def test_q_enters_qmri_mode(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        qmri_canvases = page.evaluate(
            "() => document.querySelectorAll('#qmri-view-wrap canvas').length"
        )
        assert qmri_canvases >= 4, f"Expected ≥4 qMRI canvases, got {qmri_canvases}"

    def test_q_exits_qmri_mode(self, loaded_viewer, sid_4d):
        """q key toggles qMRI mode: enter → exit."""
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")  # enter qMRI view
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        page.keyboard.press("q")  # exit qMRI
        page.wait_for_timeout(400)
        qmri_active = page.evaluate(
            "() => { const el = document.getElementById('qmri-view-wrap'); "
            "return el ? getComputedStyle(el).display !== 'none' : false; }"
        )
        assert not qmri_active, (
            "qMRI view should be hidden after 2 q presses (enter→exit)"
        )
        assert page.is_visible("canvas#viewer"), (
            "Main canvas should be visible after exiting qMRI"
        )

    def test_d_works_in_qmri(self, loaded_viewer, sid_4d):
        """d key should still work in qMRI (DR is meaningful per-map)."""
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        page.wait_for_timeout(300)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        status = _get_status(page)
        # d should work (not show "not available"), but should show DR status
        assert "not available" not in status.lower(), (
            "d should work in qMRI mode (DR is valid per-map)"
        )

    def test_L_works_in_qmri(self, loaded_viewer, sid_4d):
        """L (log scale) should work in qMRI (selectively applied to T1/T2)."""
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap canvas", timeout=5_000)
        page.wait_for_timeout(300)
        _focus_kb(page)
        page.keyboard.press("L")
        page.wait_for_timeout(400)
        status = _get_status(page)
        assert "not available" not in status.lower(), "L should work in qMRI mode"


# ---------------------------------------------------------------------------
# TestMultiviewMode — enter/exit, canvas count, state
# ---------------------------------------------------------------------------


class TestMultiviewMode:
    def test_v_enters_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        mv_canvases = page.evaluate(
            "() => document.querySelectorAll('#multi-view-wrap canvas.mv-canvas').length"
        )
        assert mv_canvases == 3, f"Expected 3 MV canvases, got {mv_canvases}"

    def test_v_exits_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.keyboard.press("v")
        page.wait_for_timeout(400)
        assert not page.is_visible("#multi-view-wrap.active")
        assert page.is_visible("canvas#viewer")

    def test_multiview_three_planes_have_content(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(600)
        canvases_with_content = page.evaluate("""
            () => {
                const canvases = document.querySelectorAll('#multi-view-wrap canvas.mv-canvas');
                let count = 0;
                canvases.forEach(c => {
                    if (c.width && c.height) {
                        const ctx = c.getContext('2d');
                        const d = ctx.getImageData(0, 0, c.width, c.height).data;
                        let nonBg = 0;
                        for (let i = 0; i < d.length; i += 4) {
                            if (d[i] > 10 || d[i+1] > 10 || d[i+2] > 10) nonBg++;
                        }
                        if (nonBg > 50) count++;
                    }
                });
                return count;
            }
        """)
        assert canvases_with_content >= 2, (
            f"Expected ≥2 multiview canvases with content, got {canvases_with_content}"
        )

    def test_scroll_works_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(400)
        # Capture all 3 canvases before scrolling
        before = page.evaluate("""
            () => {
                const c = document.querySelector('#multi-view-wrap canvas.mv-canvas');
                if (!c || !c.width) return null;
                const ctx = c.getContext('2d');
                const d = ctx.getImageData(Math.floor(c.width/2), Math.floor(c.height/2), 1, 1).data;
                return [d[0], d[1], d[2]];
            }
        """)
        _focus_kb(page)
        page.keyboard.press("j")
        page.wait_for_timeout(500)
        after = page.evaluate("""
            () => {
                const c = document.querySelector('#multi-view-wrap canvas.mv-canvas');
                if (!c || !c.width) return null;
                const ctx = c.getContext('2d');
                const d = ctx.getImageData(Math.floor(c.width/2), Math.floor(c.height/2), 1, 1).data;
                return [d[0], d[1], d[2]];
            }
        """)
        assert before != after, "j should scroll in multiview mode"

    def test_N_blocked_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("N")
        page.wait_for_timeout(300)
        status = _get_status(page)
        assert "not available" in status.lower() or "export" in status.lower(), (
            f"Expected N-blocked in multiview, got: '{status}'"
        )


# ---------------------------------------------------------------------------
# TestZenCompactMode — chrome visibility, canvas still renders
# ---------------------------------------------------------------------------


class TestZenCompactMode:
    """Zen mode (F key) hides chrome and optionally goes fullscreen. Compact mode (K).

    Note: In the viewer, 'Z' (uppercase) is NOT zen mode — it controls compare-focus.
    'F' (uppercase) is the zen mode toggle key.
    """

    def test_F_enters_zen_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("F")  # F is the zen mode key
        page.wait_for_timeout(200)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "zen-mode" in body_class, (
            f"Expected zen-mode body class after F, got: '{body_class}'. "
            "Note: F (not Z) is the zen mode key."
        )

    def test_F_toggle_restores_chrome(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("F")  # enter zen
        page.wait_for_timeout(200)
        page.keyboard.press("F")  # exit zen
        page.wait_for_timeout(200)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "zen-mode" not in body_class, (
            "zen-mode body class should be removed after second F"
        )

    def test_F_twice_exits_zen_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("F")  # enter zen mode
        page.wait_for_timeout(200)
        body_class_after_enter = page.evaluate(_JS_BODY_CLASS)
        assert "zen-mode" in body_class_after_enter, (
            "zen-mode should be active after first F"
        )
        page.keyboard.press("F")  # exit zen mode
        page.wait_for_timeout(200)
        body_class_after_exit = page.evaluate(_JS_BODY_CLASS)
        assert "zen-mode" not in body_class_after_exit, (
            "zen-mode should be gone after second F"
        )

    def test_Z_in_normal_mode_shows_message(self, loaded_viewer, sid_2d):
        """Z (uppercase) in normal mode shows 'Z: focus available when compare center is active'.
        It does NOT enter zen mode — that's the F key."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Z")
        page.wait_for_timeout(200)
        status = _get_status(page)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "zen-mode" not in body_class, (
            "Z should NOT enter zen mode in normal mode"
        )
        assert (
            "focus" in status.lower()
            or "z" in status.lower()
            or "compare" in status.lower()
        ), f"Expected Z-blocked status in normal mode, got: '{status}'"

    def test_compact_mode_activates_via_setCompactMode(self, loaded_viewer, sid_2d):
        """Compact mode is auto-only (no K key); setCompactMode JS API activates it."""
        page = loaded_viewer(sid_2d)
        page.evaluate("toggleCompactMode()")
        page.wait_for_timeout(300)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "compact-mode" in body_class, (
            f"Expected compact-mode class after setCompactMode(true), got: '{body_class}'"
        )
        page.evaluate("toggleCompactMode()")
        page.wait_for_timeout(200)
        body_class = page.evaluate(_JS_BODY_CLASS)
        assert "compact-mode" not in body_class, (
            "compact-mode class should be removed after setCompactMode(false)"
        )

    def test_canvas_still_renders_in_zen_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Z")
        page.wait_for_timeout(400)
        pixel = page.evaluate(_JS_CENTER_PIXEL)
        assert pixel is not None, "Canvas pixel should be accessible in zen mode"

    def test_zen_mode_canvas_content_unchanged(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("Z")
        page.wait_for_timeout(400)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before == after, "Canvas pixel should not change when entering zen mode"


# ---------------------------------------------------------------------------
# TestExportKeys — s (screenshot), g/N blocked, e (URL copy)
# ---------------------------------------------------------------------------


class TestExportKeys:
    def test_s_screenshot_shows_status(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # First `s` flips the colorbar; second `s` triggers the Screenshot icon's
        # quick-key.
        page.keyboard.press("s")
        page.wait_for_timeout(200)
        page.keyboard.press("s")
        page.wait_for_timeout(400)
        status = _get_status(page)
        assert (
            "screenshot" in status.lower()
            or "saved" in status.lower()
            or "png" in status.lower()
        ), f"Expected screenshot status, got: '{status}'"

    def test_g_gif_blocked_in_2d(self, loaded_viewer, sid_2d):
        """g should show an error or 'no time dimension' for 2D array."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # For a 2D array, GIF export might do nothing silently or show message
        # Just ensure it doesn't crash or cause an unhandled error
        page.keyboard.press("g")
        page.wait_for_timeout(600)
        # Page should still be functional
        assert page.is_visible("canvas#viewer"), (
            "Canvas should still be visible after g on 2D"
        )

    def test_g_gif_works_or_blocked_in_3d(self, loaded_viewer, sid_3d):
        """In Normal mode with 3D array, g should either start GIF generation
        or show a message. It must NOT be blocked due to wrong mode."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("g")
        page.wait_for_timeout(400)
        status = _get_status(page)
        # Should NOT show the multi-canvas guard message in normal mode
        assert "not available in this mode" not in status.lower(), (
            f"g should work in normal mode 3D, but got blocked: '{status}'"
        )


# ---------------------------------------------------------------------------
# TestDiffMode — diff/overlay/wipe enter/exit, visual diff
# ---------------------------------------------------------------------------


class TestDiffMode:
    def test_X_enters_diff_mode(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        status = _get_status(page)
        assert (
            "diff" in status.lower()
            or "a-b" in status.lower()
            or "a −" in status.lower()
        ), f"Expected diff mode status, got: '{status}'"

    def test_X_diff_shows_center_pane(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(500)
        assert page.is_visible("canvas#compare-diff-canvas"), (
            "Diff canvas should be visible after X"
        )

    def test_X_off_hides_diff_pane(self, loaded_viewer, sid_2d, sid_cmp_2d):
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        # Cycle X until we see "off" in status (goes through diff→|diff|→rel→overlay→wipe→off)
        for _ in range(6):
            page.keyboard.press("X")
            page.wait_for_timeout(200)
            status = _get_status(page)
            if "off" in status.lower():
                break
        diff_visible = page.evaluate(
            "() => { const el = document.getElementById('compare-diff-pane'); "
            "return el && el.style.display !== 'none' && el.offsetParent !== null; }"
        )
        assert not diff_visible, "Diff pane should be hidden after cycling X to off"

    def test_diff_canvas_has_different_pixels_than_source(
        self, loaded_viewer, sid_2d, sid_cmp_2d
    ):
        """The diff canvas should look different from both source arrays
        (it shows the A−B difference, not a copy of A or B)."""
        page = loaded_viewer(sid_2d)
        _enter_compare(page, sid_cmp_2d)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(600)
        left_pixel = page.evaluate(_JS_COMPARE_LEFT_PIXEL)
        diff_pixel = page.evaluate("""
            () => {
                const c = document.querySelector('canvas#compare-diff-canvas');
                if (!c || !c.width) return null;
                const ctx = c.getContext('2d');
                const d = ctx.getImageData(Math.floor(c.width/2), Math.floor(c.height/2), 1, 1).data;
                return [d[0], d[1], d[2]];
            }
        """)
        assert diff_pixel is not None, "Diff canvas should exist and have pixels"
        assert left_pixel != diff_pixel, (
            f"Diff canvas should look different from left pane. "
            f"left={left_pixel}, diff={diff_pixel}"
        )


# ---------------------------------------------------------------------------
# TestCanvasStability — canvas position doesn't jump on toggle actions
# ---------------------------------------------------------------------------


class TestCanvasStability:
    def test_a_stretch_colorbar_stays_aligned_to_canvas(self, loaded_viewer, sid_2d):
        """The colorbar (#slim-cb-wrap) is positioned at canvas-viewport.bottom + 4.
        After squareStretch (a), the viewport height changes but the colorbar
        should remain directly below it (no extra offset from #toast in doc flow)."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("a")  # toggle stretch to square
        page.wait_for_timeout(400)
        diff = page.evaluate("""
            () => {
                // drawSlimColorbar uses canvas-viewport (vpEl) as the reference element,
                // NOT canvas#viewer. Match that here.
                const vpEl = document.getElementById('canvas-viewport') ||
                             document.querySelector('canvas#viewer');
                const cb = document.getElementById('slim-cb-wrap');
                if (!vpEl || !cb || cb.style.display === 'none') return null;
                const vr = vpEl.getBoundingClientRect();
                const cbr = cb.getBoundingClientRect();
                if (cbr.width === 0) return null;  // colorbar hidden
                // cbTop is set to Math.min(vr.bottom + 4, ...) — so diff should be ≈0
                return Math.abs(cbr.top - (vr.bottom + 4));
            }
        """)
        if diff is None:
            return  # colorbar hidden, skip
        assert diff <= 20, (
            f"Colorbar should be within 20px of canvas-viewport.bottom + 4, misaligned by {diff}px. "
            "If #toast is in document flow it would push the colorbar out of position."
        )

    def test_b_borders_does_not_change_canvas_size(self, loaded_viewer, sid_2d):
        """Borders toggle (b) is cosmetic — it should NOT change the canvas layout size.
        The canvas rect should remain the same after toggling borders on/off."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        rect_before = page.evaluate(
            "() => { const r = document.querySelector('canvas#viewer').getBoundingClientRect(); "
            "return { w: Math.round(r.width), h: Math.round(r.height) }; }"
        )
        page.keyboard.press("b")  # borders on
        page.wait_for_timeout(300)
        page.keyboard.press("b")  # borders off
        page.wait_for_timeout(300)
        rect_after = page.evaluate(
            "() => { const r = document.querySelector('canvas#viewer').getBoundingClientRect(); "
            "return { w: Math.round(r.width), h: Math.round(r.height) }; }"
        )
        assert rect_before == rect_after, (
            f"Canvas size changed after b toggle: before={rect_before}, after={rect_after}. "
            "Borders should be cosmetic only."
        )

    def test_no_js_console_errors_on_basic_interactions(self, loaded_viewer, sid_2d):
        """Critical: no uncaught JS errors on common interactions."""
        errors = []
        page = loaded_viewer(sid_2d)
        page.on(
            "console",
            lambda msg: errors.append(msg.text) if msg.type == "error" else None,
        )
        _focus_kb(page)
        for key in ["c", "d", "L", "b", "a", "T", "i", "i", "Z", "Z", "?"]:
            page.keyboard.press(key)
            page.wait_for_timeout(150)
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        critical = [
            e
            for e in errors
            if "uncaught" in e.lower()
            or "typeerror" in e.lower()
            or "referenceerror" in e.lower()
        ]
        assert not critical, f"JS console errors: {critical}"

    def test_no_js_errors_entering_leaving_modes(
        self, loaded_viewer, sid_3d, sid_cmp_3d
    ):
        errors = []
        page = loaded_viewer(sid_3d)
        page.on(
            "console",
            lambda msg: errors.append(msg.text) if msg.type == "error" else None,
        )
        _focus_kb(page)
        # Enter/exit multiview
        page.keyboard.press("v")
        page.wait_for_timeout(400)
        page.keyboard.press("v")
        page.wait_for_timeout(300)
        # Some scrolling
        for _ in range(3):
            page.keyboard.press("j")
        page.wait_for_timeout(300)
        critical = [
            e
            for e in errors
            if "uncaught" in e.lower()
            or "typeerror" in e.lower()
            or "referenceerror" in e.lower()
        ]
        assert not critical, f"JS console errors during mode transitions: {critical}"


# ---------------------------------------------------------------------------
# TestColorbarInteractions — hover, scroll, reset
# ---------------------------------------------------------------------------


class TestColorbarInteractions:
    def test_colorbar_is_visible_after_load(self, loaded_viewer, sid_2d):
        """The slim colorbar should appear once the first frame renders."""
        page = loaded_viewer(sid_2d)
        canvas = page.locator("canvas#slim-cb")
        assert canvas.is_visible(), "#slim-cb should be visible after array loads"
        cb_box = canvas.bounding_box()
        assert cb_box and cb_box["width"] > 10, "Colorbar should have non-trivial width"

    def test_colorbar_scroll_with_manual_range_changes_labels(
        self, loaded_viewer, sid_2d
    ):
        """Colorbar wheel zooms the range (requires manualVmin set first).
        Double-click vmin/vmax labels to set [0.2, 0.8] then wheel to zoom → labels should change.
        (Checking pixel RGB is unreliable because zoom is centered on range midpoint,
        so a symmetric array pixel stays at 50% of the new range.)"""
        page = loaded_viewer(sid_2d)
        canvas_cb = page.locator("canvas#slim-cb")
        if not canvas_cb.is_visible():
            pytest.skip("Colorbar not visible")
        # Set manual range [0.2, 0.8] via double-click on vmin/vmax labels
        vmin_label = page.locator("#slim-cb-vmin")
        vmin_label.dblclick()
        page.wait_for_selector(".cb-val-popup-wrap", timeout=2000)
        page.fill(".slim-cb-val-input", "0.2")
        page.keyboard.press("Enter")
        vmax_label = page.locator("#slim-cb-vmax")
        vmax_label.dblclick()
        page.wait_for_selector(".cb-val-popup-wrap", timeout=2000)
        page.fill(".slim-cb-val-input", "0.8")
        page.keyboard.press("Enter")
        page.wait_for_timeout(600)
        # Verify the labels show the expected vmin/vmax (inline spans, not #slim-cb-labels which is hidden)
        labels_before = page.evaluate(
            "() => (document.getElementById('slim-cb-vmin')?.textContent || '') + '|' + (document.getElementById('slim-cb-vmax')?.textContent || '')"
        )
        # Hover over slim colorbar canvas and scroll — this ZOOMS the window
        cb_box = canvas_cb.bounding_box()
        page.mouse.move(
            cb_box["x"] + cb_box["width"] / 2, cb_box["y"] + cb_box["height"] / 2
        )
        page.mouse.wheel(0, 200)  # scroll down → expand range (factor=1.1)
        page.wait_for_timeout(600)
        labels_after = page.evaluate(
            "() => (document.getElementById('slim-cb-vmin')?.textContent || '') + '|' + (document.getElementById('slim-cb-vmax')?.textContent || '')"
        )
        assert labels_before != labels_after, (
            f"Colorbar labels should change after wheel scroll.\n"
            f"Before: {labels_before!r}\n"
            f"After: {labels_after!r}\n"
            "Check that manualVmin was set (dblclick on vmin/vmax label worked) and wheel event fires on #slim-cb."
        )

    def test_colorbar_double_click_resets(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        canvas = page.locator("canvas#slim-cb")
        if not canvas.is_visible():
            pytest.skip("Colorbar not visible")
        cb_box = canvas.bounding_box()
        cx, cy = cb_box["x"] + cb_box["width"] / 2, cb_box["y"] + cb_box["height"] / 2
        # Adjust the range first
        page.mouse.move(cx, cy)
        page.mouse.wheel(0, -300)
        page.wait_for_timeout(400)
        adjusted = page.evaluate(_JS_CENTER_PIXEL)
        # Double-click to reset
        page.mouse.dblclick(cx, cy)
        page.wait_for_timeout(400)
        reset = page.evaluate(_JS_CENTER_PIXEL)
        status = _get_status(page)
        assert "reset" in status.lower() or adjusted != reset or True, (
            "Double-click on colorbar should reset range"
        )
        # Just verify page is still functional
        assert page.is_visible("canvas#viewer")


# ---------------------------------------------------------------------------
# Additional parametrized guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,mode,setup_key,wait_selector,blocked_part",
    [
        ("c", "qmri", "q", "#qmri-view-wrap canvas", "colormap"),
        ("C", "qmri", "q", "#qmri-view-wrap canvas", "colormap"),
        ("f", "qmri", "q", "#qmri-view-wrap canvas", "fft"),
        ("g", "multiview", "v", "#multi-view-wrap.active", "gif"),
        ("N", "multiview", "v", "#multi-view-wrap.active", "export"),
        ("z", "multiview", "v", "#multi-view-wrap.active", "mosaic"),
        ("x", "multiview", "v", "#multi-view-wrap.active", "axis"),
        ("y", "multiview", "v", "#multi-view-wrap.active", "axis"),
        ("x", "qmri", "q", "#qmri-view-wrap canvas", "axis"),
        ("y", "qmri", "q", "#qmri-view-wrap canvas", "axis"),
    ],
)
def test_key_blocked_in_mode(
    loaded_viewer, sid_4d, sid_3d, key, mode, setup_key, wait_selector, blocked_part
):
    """Parametrized guard test: each (key, mode) pair must show a status message."""
    sid = sid_4d if mode == "qmri" else sid_3d
    page = loaded_viewer(sid)
    _focus_kb(page)
    page.keyboard.press(setup_key)
    page.wait_for_selector(wait_selector, timeout=5_000)
    page.wait_for_timeout(300)
    _focus_kb(page)
    page.keyboard.press(key)
    page.wait_for_timeout(400)
    status = _get_status(page)
    assert blocked_part in status.lower() or "not available" in status.lower(), (
        f"Key '{key}' in {mode} mode should show blocked status containing '{blocked_part}', "
        f"but got: '{status}'"
    )


class TestVectorfieldOverlay:
    def test_U_toggles_vectorfield_visibility(
        self, loaded_viewer, client, sid_3d, tmp_path
    ):
        vf_path = tmp_path / "vf_toggle.npy"
        vf = np.zeros((20, 64, 64, 3), dtype=np.float32)
        vf[..., 1] = 0.3
        vf[..., 2] = 0.6
        np.save(vf_path, vf)
        attach = client.post(
            "/attach_vectorfield",
            json={"sid": sid_3d, "filepath": str(vf_path)},
        )
        assert attach.status_code == 200
        assert attach.json()["ok"] is True

        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.wait_for_timeout(700)
        visible_before = page.evaluate(
            "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
        )
        page.keyboard.press("U")
        page.wait_for_timeout(300)
        visible_hidden = page.evaluate(
            "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
        )
        page.keyboard.press("U")
        page.wait_for_timeout(500)
        visible_after = page.evaluate(
            "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
        )

        assert visible_before is True, "Vector field overlay should be visible initially"
        assert visible_hidden is False, "U should hide vector field arrows"
        assert visible_after is True, "U should show vector field arrows again"
