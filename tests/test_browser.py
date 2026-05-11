"""Layer 2: Playwright browser tests.

Run with:
    pytest tests/test_browser.py

First run creates baseline snapshots in tests/snapshots/.
Subsequent runs compare against them (1% pixel-change threshold).
"""

import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

pytestmark = pytest.mark.browser

SNAPSHOTS = Path(__file__).parent / "snapshots"
DEBUG_DIR = Path(__file__).resolve().parents[1] / "debug"

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

_JS_COMPARE_LEFT_CANVAS_INFO = """
() => {
    const c = document.querySelector('canvas#compare-left-canvas');
    if (!c || !c.width || !c.height) return null;
    const ctx = c.getContext('2d');
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let nonBg = 0;
    for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 20 || d[i+1] > 20 || d[i+2] > 20) nonBg++;
    }
    return {width: c.width, height: c.height, nonBgPixels: nonBg};
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


def _enter_compare(page, partner_sid, timeout=5000):
    """Enter compare mode with a given partner sid.

    The B keybind that used to open the compare picker has been retired;
    we now invoke enterCompareModeBySid() directly via page.evaluate(). This
    is the same JS entry point used by URL-param launches and drag-drop, so
    it exercises the real production code path.
    """
    page.evaluate(
        f"async () => {{ await enterCompareModeBySid({partner_sid!r}); }}"
    )
    page.wait_for_selector("#compare-view-wrap.active", timeout=timeout)


def _exit_compare(page):
    """Exit compare mode via the JS entry point (B keybind retired)."""
    page.evaluate("() => exitCompareMode()")


def _center_of(locator):
    box = locator.bounding_box()
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


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
        # #loading-overlay should be hidden once startup animation completes
        page.wait_for_selector("#loading-overlay", state="hidden", timeout=5000)

    def test_loading_class_cleared_after_render(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        page.wait_for_function(
            "() => !document.body.classList.contains('av-loading')",
            timeout=5000,
        )

    def test_shell_init_tab_loads_without_preview_overlay(self, page, server_url, sid_2d):
        page.goto(f"{server_url}/shell?init_sid={sid_2d}&init_name=arr2d")
        page.wait_for_selector(".tab-pane.active iframe", timeout=5000)
        page.wait_for_function(
            """() => {
                const iframe = document.querySelector('.tab-pane.active iframe');
                return !!iframe && iframe.contentWindow && iframe.getAttribute('src') && iframe.getAttribute('src').includes(`sid=${window.location.search.match(/init_sid=([^&]+)/)[1]}`);
            }""",
            timeout=15000,
        )
        assert page.locator(".tab-preview").count() == 0

    def test_shell_init_compare_big_left_shows_shared_bar_and_stays_stable(
        self, page, server_url, client
    ):
        base_path = DEBUG_DIR / "parameter_maps.nii"
        compare_path = DEBUG_DIR / "parameter_maps_005.nii"
        sid_base = client.post(
            "/load", json={"filepath": str(base_path), "name": base_path.name}
        ).json()["sid"]
        sid_compare = client.post(
            "/load", json={"filepath": str(compare_path), "name": compare_path.name}
        ).json()["sid"]

        page.goto(
            f"{server_url}/shell?init_sid={sid_base}"
            f"&init_name={base_path.name}"
            f"&init_compare_sid={sid_compare}"
            f"&init_compare_sids={sid_compare}"
        )
        page.wait_for_selector(".tab-pane.active iframe", timeout=5_000)
        page.wait_for_function(
            """() => {
                const iframe = document.querySelector('.tab-pane.active iframe');
                return !!iframe && !!iframe.contentWindow
                    && !!iframe.contentDocument
                    && !!iframe.contentDocument.querySelector('#compare-view-wrap.active');
            }""",
            timeout=15_000,
        )

        iframe = page.locator(".tab-pane.active iframe").element_handle()
        frame = iframe.content_frame()
        assert frame is not None
        frame.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        frame.wait_for_timeout(900)
        frame.focus("#keyboard-sink")
        page.keyboard.press("X")
        frame.wait_for_timeout(450)
        page.keyboard.press("G")
        frame.wait_for_timeout(450)

        def _state():
            return frame.evaluate(
                """() => {
                    const wrap = document.querySelector('#compare-view-wrap');
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    return {
                        wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffClipRect: diffClip?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    };
                }"""
            )

        initial = _state()
        assert initial["wrapBigLeft"], f"shell compare init should enter big-left after X then G, got: {initial}"
        assert initial["sharedCbDisplay"] != "none", f"shell compare init should show the shared source colorbar immediately after X then G, got: {initial}"
        assert initial["sharedCbRect"], f"shell compare init should have a measurable shared source colorbar immediately after X then G, got: {initial}"
        assert initial["diffClipRect"] and initial["sourceTopClipRect"] and initial["sourceBottomClipRect"], f"shell compare init should expose measurable clips after X then G, got: {initial}"
        assert abs(initial["sharedCbRect"]["width"] - initial["sourceTopClipRect"]["width"]) <= 1, f"shell compare init should match the shared source colorbar width to the source pane width, got: {initial}"

        source_box = frame.locator(".compare-primary .compare-canvas-clip").bounding_box()
        assert source_box is not None
        page.mouse.move(
            source_box["x"] + source_box["width"] / 2,
            source_box["y"] + source_box["height"] / 2,
        )
        page.mouse.wheel(0, -120)
        frame.wait_for_timeout(1000)
        after = _state()

        assert after["sharedCbDisplay"] != "none", f"shell compare init should keep the shared source colorbar visible after wheel updates, got: {after}"
        assert after["sharedCbRect"] and after["sourceTopClipRect"], f"shell compare init should keep the shared source colorbar and source pane measurable after wheel updates, got: {after}"
        assert abs(after["sharedCbRect"]["width"] - after["sourceTopClipRect"]["width"]) <= 1, f"shell compare init should keep the shared source colorbar width matched after wheel updates, got: {after}"
        assert after["diffClipRect"] and after["sourceTopClipRect"] and after["sourceBottomClipRect"], f"shell compare init should keep measurable clips after wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["top"] - after["sourceTopClipRect"]["top"]) <= 1, f"shell compare init should keep top alignment after wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["bottom"] - after["sourceBottomClipRect"]["bottom"]) <= 1, f"shell compare init should keep bottom alignment after wheel updates, got: {after}"


# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------


class TestKeyboard:
    def test_c_changes_colormap(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        opened = page.evaluate(_JS_CENTER_PIXEL)
        assert before == opened, (
            "First c press should open the colormap menu without cycling"
        )
        page.keyboard.press("c")
        page.wait_for_timeout(800)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, (
            "Center pixel unchanged after pressing c twice (colormap cycle)"
        )

    def test_c_preview_is_anchored_above_colorbar(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        boxes = page.evaluate(
            """() => {
                const box = (el) => {
                    const r = el.getBoundingClientRect();
                    return {
                        top: r.top, bottom: r.bottom,
                        left: r.left, right: r.right,
                        width: r.width, height: r.height,
                    };
                };
                    return {
                        wrap: box(document.getElementById('slim-cb-wrap')),
                        preview: box(document.getElementById('slim-cb-preview')),
                        bar: box(document.getElementById('slim-cb')),
                        active: box(document.querySelector('#slim-cb-preview .cmh-cell.active')),
                        swatch: box(document.querySelector('#slim-cb-preview .cmh-cell.active canvas')),
                    };
                }"""
        )
        assert boxes["preview"]["top"] >= boxes["wrap"]["top"] + 4
        assert boxes["preview"]["left"] >= boxes["wrap"]["left"] + 8
        assert boxes["preview"]["right"] <= boxes["wrap"]["right"] - 8
        assert boxes["preview"]["bottom"] <= boxes["bar"]["top"] - 2, (
            "Colormap previews should sit above the colorbar row"
        )
        active_center = (boxes["active"]["left"] + boxes["active"]["right"]) / 2
        swatch_center = (boxes["swatch"]["left"] + boxes["swatch"]["right"]) / 2
        assert abs(active_center - swatch_center) <= 2, (
            "Active colormap frame should be centered on its swatch"
        )
        assert boxes["swatch"]["left"] >= boxes["active"]["left"] + 4
        assert boxes["swatch"]["right"] <= boxes["active"]["right"] - 4

    def test_c_preview_navigation_and_hover_hold(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        first = page.evaluate("() => colormap_idx")
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(300)
        right = page.evaluate("() => colormap_idx")
        assert right != first, "ArrowRight should preview the next colormap"
        page.keyboard.press("h")
        page.wait_for_timeout(300)
        left = page.evaluate("() => colormap_idx")
        assert left == first, "h should preview the previous colormap"
        page.hover("#slim-cb-preview .cmh-cell[data-cmh-idx='2']")
        page.wait_for_timeout(3300)
        assert page.is_visible("#slim-cb-preview.fade-in"), (
            "Colormap menu should stay open while the mouse is inside it"
        )

    def test_c_preview_uses_integrated_menu_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)
        page.keyboard.press("c")
        page.wait_for_selector("#mv-cb-wrap .cmap-hold-preview.fade-in", timeout=2_000)
        state = page.evaluate(
            """() => ({
                hostVisible: !!document.querySelector('#mv-cb-wrap .cmap-hold-preview.fade-in'),
                stripVisible: !!document.querySelector('#colormap-strip.visible'),
            })"""
        )
        assert state["hostVisible"], f"multiview should open the integrated colormap menu, got: {state}"
        assert not state["stripVisible"], f"multiview should not fall back to the old strip previewer, got: {state}"

    def test_c_preview_uses_integrated_menu_in_diff_mode(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare_cmap.npy"
        np.save(path, arr_2d * 0.5)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare_cmap"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        page.wait_for_selector("canvas#compare-right-canvas:visible", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(400)
        assert page.is_visible("#compare-diff-canvas"), "diff center canvas should be visible after pressing X"
        page.hover("#compare-diff-canvas")
        page.wait_for_timeout(150)
        page.keyboard.press("c")
        page.wait_for_selector("#compare-diff-pane .cmap-hold-preview.fade-in", timeout=2_000)
        state = page.evaluate(
            """() => ({
                hostVisible: !!document.querySelector('#compare-diff-pane .cmap-hold-preview.fade-in'),
                stripVisible: !!document.querySelector('#colormap-strip.visible'),
            })"""
        )
        assert state["hostVisible"], f"diff mode should open the integrated colormap menu on the diff pane, got: {state}"
        assert not state["stripVisible"], f"diff mode should not fall back to the old strip previewer, got: {state}"

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

    def test_slash_opens_plugin_shelf(
        self, loaded_viewer, sid_3d
    ):
        page = loaded_viewer(sid_3d)
        assert not page.is_visible("#special-modes-shelf.visible")
        _focus_kb(page)
        page.keyboard.press("/")
        page.wait_for_selector("#special-modes-shelf.visible", timeout=2_000)
        # Tile set: qMRI, Segmentation, ROI, Overlay, Vector field, Crop.
        # Assert presence of each by id instead of pinning a count, so adding
        # a future plugin doesn't require updating this test.
        for plugin_id in ("qmri", "segmentation", "roi", "overlay", "vectorfield", "crop"):
            assert page.locator(
                f"#special-modes-grid .smode-tile[data-smode-id='{plugin_id}']"
            ).count() == 1, f"missing tile for plugin '{plugin_id}'"
        # Close with Escape
        page.keyboard.press("Escape")
        page.wait_for_function(
            "() => !document.querySelector('#special-modes-shelf.visible')",
            timeout=2_000,
        )

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

    def test_compare_entry_creates_side_by_side_view(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        """enterCompareModeBySid → exitCompareMode round-trip via JS entry points.

        This test used to press B (now retired); the compare entry point is
        now exercised via the same JS function the unified picker calls.
        """
        path = tmp_path / "arr2d_compare.npy"
        np.save(path, arr_2d * 0.5)
        resp = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare"}
        )
        sid_compare = resp.json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        page.wait_for_selector("canvas#compare-right-canvas:visible", timeout=5_000)
        assert page.is_visible("canvas#compare-left-canvas")
        assert page.is_visible("canvas#compare-right-canvas")
        # Shared colorbar is used (per-pane colorbars hidden in non-diff mode)
        assert page.is_visible("#slim-cb-wrap")
        assert "arr2d_compare" in page.inner_text("#compare-right-title").lower()

        _exit_compare(page)
        page.wait_for_timeout(350)
        assert not page.is_visible("#compare-view-wrap.active")

    def test_invalid_compare_url_falls_back_to_base_view(
        self, page, server_url, sid_2d
    ):
        page.goto(
            f"{server_url}/?sid={sid_2d}"
            "&compare_sid=invalid-compare-sid"
            "&compare_sids=invalid-compare-sid"
        )
        page.wait_for_selector("#loading-overlay", state="hidden", timeout=15_000)
        page.wait_for_selector("canvas#viewer", state="visible", timeout=5_000)
        page.wait_for_function("() => !compareActive", timeout=5_000)
        info = page.evaluate(_JS_CANVAS_INFO)
        assert info is not None, "Canvas not found after compare fallback"
        assert info["nonBgPixels"] > 0, "Canvas stayed blank after compare fallback"

    def test_direct_compare_url_renders_compare_canvas(
        self, page, server_url, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare_direct.npy"
        np.save(path, arr_2d * 0.5)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare_direct"}
        ).json()["sid"]

        page.goto(
            f"{server_url}/?sid={sid_2d}"
            f"&compare_sid={sid_compare}"
            f"&compare_sids={sid_compare}"
        )
        page.wait_for_selector("#loading-overlay", state="hidden", timeout=15_000)
        page.wait_for_selector("canvas#compare-left-canvas:visible", timeout=5_000)
        page.wait_for_selector("canvas#compare-right-canvas:visible", timeout=5_000)
        info = page.evaluate(_JS_COMPARE_LEFT_CANVAS_INFO)
        assert info is not None, "Compare canvas not found after direct compare boot"
        assert info["nonBgPixels"] > 0, "Direct compare boot left compare canvas blank"

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
        _enter_compare(page, sid_compare)

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
        _enter_compare(page, sid_compare)
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
        _enter_compare(page, sid1)
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

    def test_compare_center_title_pill_uses_immersive_glass_style_outside_fullscreen(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare_glass_pill.npy"
        np.save(path, np.fliplr(arr_2d))
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare_glass_pill"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(250)

        state = page.evaluate(
            """() => {
                const btn = document.querySelector('#compare-diff-title .compare-center-mode-btn.active');
                if (!btn) return null;
                const style = getComputedStyle(btn);
                const rect = btn.getBoundingClientRect();
                return {
                    text: (btn.textContent || '').replace(/\\s+/g, ' ').trim(),
                    background: style.backgroundColor,
                    borderColor: style.borderColor,
                    borderRadius: style.borderRadius,
                    backdropFilter: style.backdropFilter,
                    webkitBackdropFilter: style.webkitBackdropFilter,
                    boxShadow: style.boxShadow,
                    minWidth: style.minWidth,
                    minHeight: style.minHeight,
                    fullscreen: document.body.classList.contains('fullscreen-mode'),
                };
            }"""
        )

        assert state, "diff title should render an active compare-center pill after pressing X"
        assert not state["fullscreen"], f"test must stay in non-immersive mode, got: {state}"
        assert "A" in state["text"] and "B" in state["text"], f"compare-center pill should render the compare glyph, got: {state}"
        assert state["background"] != "rgba(0, 0, 0, 0)", f"compare-center pill should use a filled glass background outside fullscreen, got: {state}"
        assert state["borderColor"] != "rgba(0, 0, 0, 0)", f"compare-center pill should keep a visible glass border outside fullscreen, got: {state}"
        assert state["backdropFilter"] != "none" or state["webkitBackdropFilter"] != "none", f"compare-center pill should use the immersive blur treatment outside fullscreen, got: {state}"
        assert state["boxShadow"] != "none", f"compare-center pill should keep the immersive island shadow outside fullscreen, got: {state}"
        assert state["minHeight"] == "26px", f"compare-center pill should stay compact outside fullscreen, got: {state}"
        assert state["minWidth"] == "52px", f"compare-center pill should keep a compact pill width outside fullscreen, got: {state}"

    # test_B_is_locked_for_multi_array_launch removed: the B keybind has
    # been retired entirely, so "B is locked" is meaningless. The
    # multi-array URL launch path is still covered by
    # test_multi_array_launch_supports_six_compare_panes below.

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

    def test_direct_multi_array_launch_big_left_shows_shared_bar_and_stays_stable(
        self, page, server_url, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare_direct_big_left.npy"
        np.save(path, arr_3d * 0.8 + 0.1)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare_direct_big_left"}
        ).json()["sid"]

        page.goto(
            f"{server_url}/?sid={sid_3d}"
            f"&compare_sid={sid_compare}"
            f"&compare_sids={sid_compare}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(700)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(350)
        page.keyboard.press("G")
        page.wait_for_timeout(350)

        def _state():
            return page.evaluate(
                """() => {
                    const wrap = document.querySelector('#compare-view-wrap');
                    const info = document.querySelector('#info');
                    const diffTitle = document.querySelector('#compare-diff-title');
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    return {
                        wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                        infoRect: info?.getBoundingClientRect() || null,
                        diffTitleRect: diffTitle?.getBoundingClientRect() || null,
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffClipRect: diffClip?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    };
                }"""
            )

        initial = _state()
        assert initial["wrapBigLeft"], f"direct multi-array launch should enter big-left layout after X then G, got: {initial}"
        assert initial["sharedCbDisplay"] != "none", f"shared source colorbar should be visible immediately on direct multi-array big-left launch, got: {initial}"
        assert initial["sharedCbRect"], f"shared source colorbar should be measurable immediately on direct multi-array big-left launch, got: {initial}"
        assert initial["diffClipRect"] and initial["sourceTopClipRect"] and initial["sourceBottomClipRect"], f"direct multi-array big-left clips should all be measurable, got: {initial}"
        assert initial["diffClipRect"]["right"] <= initial["sourceTopClipRect"]["left"] + 1, f"left diff clip should not overlap the right source column on direct multi-array launch, got: {initial}"
        assert abs(initial["sharedCbRect"]["width"] - initial["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should match the source clip width on direct multi-array launch, got: {initial}"
        assert initial["diffTitleRect"]["top"] >= initial["diffClipRect"]["top"] - 1, f"big-left center title should sit inside the diff pane instead of above it, got: {initial}"
        assert initial["diffTitleRect"]["bottom"] <= initial["diffClipRect"]["bottom"] + 1, f"big-left center title should remain within the diff pane bounds, got: {initial}"
        assert (
            initial["diffTitleRect"]["bottom"] <= initial["infoRect"]["top"] + 1
            or initial["diffTitleRect"]["top"] >= initial["infoRect"]["bottom"] - 1
            or initial["diffTitleRect"]["right"] <= initial["infoRect"]["left"] + 1
            or initial["diffTitleRect"]["left"] >= initial["infoRect"]["right"] - 1
        ), f"big-left center title should not overlap the dimbar on direct multi-array launch, got: {initial}"

        page.mouse.move(
            initial["sourceTopClipRect"]["left"] + initial["sourceTopClipRect"]["width"] / 2,
            initial["sourceTopClipRect"]["top"] + initial["sourceTopClipRect"]["height"] / 2,
        )
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(1000)
        after = _state()

        assert after["sharedCbDisplay"] != "none", f"shared source colorbar should remain visible after direct multi-array wheel updates, got: {after}"
        assert after["diffClipRect"] and after["sourceTopClipRect"] and after["sourceBottomClipRect"], f"direct multi-array big-left clips should remain measurable after wheel updates, got: {after}"
        assert after["diffClipRect"]["right"] <= after["sourceTopClipRect"]["left"] + 1, f"left diff clip should stay out of the right source column after direct multi-array wheel updates, got: {after}"
        assert abs(after["sharedCbRect"]["width"] - after["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should keep matching the source clip width after direct multi-array wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["top"] - after["sourceTopClipRect"]["top"]) <= 1, f"left clip top should stay aligned with the top-right clip after direct multi-array wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["bottom"] - after["sourceBottomClipRect"]["bottom"]) <= 1, f"left clip bottom should stay aligned with the lower-right clip after direct multi-array wheel updates, got: {after}"

    def test_compare_center_auto_layout_picks_horizontal_on_wide_viewport_and_g_sticks(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare_auto_horizontal.npy"
        np.save(path, np.flipud(arr_2d))
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare_auto_horizontal"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        page.set_viewport_size({"width": 1700, "height": 760})
        page.wait_for_timeout(200)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_function(
            """() => {
                const wrap = document.querySelector('#compare-view-wrap');
                return !!wrap && (
                    wrap.classList.contains('compare-center-layout-horizontal')
                    || wrap.classList.contains('compare-center-layout-big-left')
                    || _compareAutoLayoutMode !== null
                );
            }""",
            timeout=5_000,
        )

        def _state():
            return page.evaluate(
                """() => {
                    const wrap = document.querySelector('#compare-view-wrap');
                    return {
                        wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                        wrapHorizontal: wrap?.classList.contains('compare-center-layout-horizontal') || false,
                        compareLayoutMode,
                        compareAutoLayoutMode: _compareAutoLayoutMode,
                    };
                }"""
            )

        initial = _state()
        assert initial["wrapHorizontal"], f"wide viewport should auto-pick horizontal center layout, got: {initial}"
        assert not initial["wrapBigLeft"], f"wide viewport should not auto-pick big-left center layout, got: {initial}"
        assert initial["compareLayoutMode"] is None, f"auto-picked layout should keep compareLayoutMode unset until manual override, got: {initial}"
        assert initial["compareAutoLayoutMode"] == "horizontal", f"wide viewport should cache a horizontal auto-layout choice, got: {initial}"

        page.set_viewport_size({"width": 980, "height": 1240})
        page.wait_for_timeout(500)
        resized = _state()
        assert resized["wrapHorizontal"], f"auto-picked horizontal layout should stay stable across resize until overridden, got: {resized}"
        assert resized["compareLayoutMode"] is None, f"resize alone should not convert the cached auto-layout into a manual override, got: {resized}"

        page.keyboard.press("G")
        page.wait_for_timeout(350)
        overridden = _state()
        assert overridden["wrapBigLeft"], f"G should manually switch the wide auto-layout to big-left, got: {overridden}"
        assert overridden["compareLayoutMode"] == "big-left", f"G should persist a manual big-left override, got: {overridden}"

        page.set_viewport_size({"width": 1700, "height": 760})
        page.wait_for_timeout(500)
        final = _state()
        assert final["wrapBigLeft"], f"manual G override should remain sticky after resizing back to a wide viewport, got: {final}"
        assert final["compareLayoutMode"] == "big-left", f"manual big-left override should stay persisted after resize, got: {final}"

    def test_compare_center_auto_layout_picks_big_left_on_tall_viewport(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        path = tmp_path / "arr2d_compare_auto_big_left.npy"
        np.save(path, np.fliplr(arr_2d))
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr2d_compare_auto_big_left"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        page.set_viewport_size({"width": 980, "height": 1240})
        page.wait_for_timeout(200)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_function(
            """() => {
                const wrap = document.querySelector('#compare-view-wrap');
                return !!wrap && (
                    wrap.classList.contains('compare-center-layout-horizontal')
                    || wrap.classList.contains('compare-center-layout-big-left')
                    || _compareAutoLayoutMode !== null
                );
            }""",
            timeout=5_000,
        )

        state = page.evaluate(
            """() => {
                const wrap = document.querySelector('#compare-view-wrap');
                return {
                    wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                    wrapHorizontal: wrap?.classList.contains('compare-center-layout-horizontal') || false,
                    compareLayoutMode,
                    compareAutoLayoutMode: _compareAutoLayoutMode,
                };
            }"""
        )

        assert state["wrapBigLeft"], f"tall viewport should auto-pick big-left center layout, got: {state}"
        assert not state["wrapHorizontal"], f"tall viewport should not auto-pick horizontal center layout, got: {state}"
        assert state["compareLayoutMode"] is None, f"auto-picked tall layout should keep compareLayoutMode unset until manual override, got: {state}"
        assert state["compareAutoLayoutMode"] == "big-left", f"tall viewport should cache a big-left auto-layout choice, got: {state}"

    def test_direct_debug_parameter_maps_big_left_shows_shared_bar_and_stays_stable(
        self, page, server_url, client
    ):
        base_path = DEBUG_DIR / "parameter_maps.nii"
        compare_path = DEBUG_DIR / "parameter_maps_005.nii"
        sid_base = client.post(
            "/load", json={"filepath": str(base_path), "name": base_path.name}
        ).json()["sid"]
        sid_compare = client.post(
            "/load", json={"filepath": str(compare_path), "name": compare_path.name}
        ).json()["sid"]

        page.goto(
            f"{server_url}/?sid={sid_base}"
            f"&compare_sid={sid_compare}"
            f"&compare_sids={sid_compare}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(900)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(450)
        page.keyboard.press("G")
        page.wait_for_timeout(450)

        def _state():
            return page.evaluate(
                """() => {
                    const wrap = document.querySelector('#compare-view-wrap');
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    return {
                        wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffClipRect: diffClip?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    };
                }"""
            )

        initial = _state()
        assert initial["wrapBigLeft"], f"debug parameter-maps launch should enter big-left after X then G, got: {initial}"
        assert initial["sharedCbDisplay"] != "none", f"debug parameter-maps launch should show shared source colorbar immediately after X then G, got: {initial}"
        assert initial["sharedCbRect"], f"debug parameter-maps launch should have a measurable shared source colorbar immediately after X then G, got: {initial}"
        assert initial["diffClipRect"] and initial["sourceTopClipRect"] and initial["sourceBottomClipRect"], f"debug parameter-maps launch should expose measurable clips after X then G, got: {initial}"
        assert abs(initial["sharedCbRect"]["width"] - initial["sourceTopClipRect"]["width"]) <= 1, f"debug parameter-maps launch should keep the shared source colorbar width matched to the source pane width, got: {initial}"

        page.mouse.move(
            initial["sourceTopClipRect"]["left"] + initial["sourceTopClipRect"]["width"] / 2,
            initial["sourceTopClipRect"]["top"] + initial["sourceTopClipRect"]["height"] / 2,
        )
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(1000)
        after = _state()

        assert after["sharedCbDisplay"] != "none", f"debug parameter-maps launch should keep the shared source colorbar visible after wheel updates, got: {after}"
        assert after["sharedCbRect"] and after["sourceTopClipRect"], f"debug parameter-maps launch should keep measurable source colorbar and source pane after wheel updates, got: {after}"
        assert abs(after["sharedCbRect"]["width"] - after["sourceTopClipRect"]["width"]) <= 1, f"debug parameter-maps launch should keep the shared source colorbar width matched after wheel updates, got: {after}"
        assert after["diffClipRect"] and after["sourceTopClipRect"] and after["sourceBottomClipRect"], f"debug parameter-maps launch should keep measurable clips after wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["top"] - after["sourceTopClipRect"]["top"]) <= 1, f"debug parameter-maps launch should keep top alignment after wheel updates, got: {after}"
        assert abs(after["diffClipRect"]["bottom"] - after["sourceBottomClipRect"]["bottom"]) <= 1, f"debug parameter-maps launch should keep bottom alignment after wheel updates, got: {after}"

    def test_big_left_shared_source_bar_appears_before_delayed_diff_response(
        self, page, server_url, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare_delayed_diff.npy"
        np.save(path, arr_3d * 0.8 + 0.1)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare_delayed_diff"}
        ).json()["sid"]

        def _slow_diff(route):
            time.sleep(0.9)
            route.continue_()

        page.route("**/diff/**", _slow_diff)
        page.goto(
            f"{server_url}/?sid={sid_3d}"
            f"&compare_sid={sid_compare}"
            f"&compare_sids={sid_compare}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(700)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(200)
        page.keyboard.press("G")
        page.wait_for_timeout(120)

        def _state():
            return page.evaluate(
                """() => {
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    return {
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffClipRect: diffClip?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    };
                }"""
            )

        before_diff = _state()
        assert before_diff["sharedCbDisplay"] != "none", f"shared source colorbar should appear immediately even while diff response is still pending, got: {before_diff}"
        assert before_diff["sharedCbRect"], f"shared source colorbar should already be measurable before the diff response returns, got: {before_diff}"
        assert before_diff["sourceTopClipRect"] and before_diff["sourceBottomClipRect"], f"source clips should already be measurable before the diff response returns, got: {before_diff}"
        assert abs(before_diff["sharedCbRect"]["width"] - before_diff["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should match the source clip width before the diff response returns, got: {before_diff}"
        assert before_diff["sharedCbRect"]["top"] >= before_diff["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should sit inside the gap below the top source pane before the diff response returns, got: {before_diff}"
        assert before_diff["sharedCbRect"]["bottom"] <= before_diff["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should stay inside the gap above the bottom source pane before the diff response returns, got: {before_diff}"

        page.wait_for_timeout(1200)
        after_diff = _state()

        assert after_diff["sharedCbDisplay"] != "none", f"shared source colorbar should remain visible after the delayed diff response completes, got: {after_diff}"
        assert after_diff["sharedCbRect"] and after_diff["sourceTopClipRect"] and after_diff["sourceBottomClipRect"], f"shared source colorbar and source clips should stay measurable after the delayed diff response completes, got: {after_diff}"
        assert abs(after_diff["sharedCbRect"]["width"] - after_diff["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should keep matching the source clip width after the delayed diff response completes, got: {after_diff}"
        assert after_diff["sharedCbRect"]["top"] >= after_diff["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should remain inside the gap below the top source pane after the delayed diff response completes, got: {after_diff}"
        assert after_diff["sharedCbRect"]["bottom"] <= after_diff["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should remain inside the gap above the bottom source pane after the delayed diff response completes, got: {after_diff}"
        assert abs(after_diff["sourceTopClipRect"]["width"] - before_diff["sourceTopClipRect"]["width"]) <= 1, f"top-right source pane width should stay stable while the delayed diff response lands, got before={before_diff} after={after_diff}"
        assert abs(after_diff["sourceTopClipRect"]["height"] - before_diff["sourceTopClipRect"]["height"]) <= 1, f"top-right source pane height should stay stable while the delayed diff response lands, got before={before_diff} after={after_diff}"
        assert abs(after_diff["sourceBottomClipRect"]["width"] - before_diff["sourceBottomClipRect"]["width"]) <= 1, f"bottom-right source pane width should stay stable while the delayed diff response lands, got before={before_diff} after={after_diff}"
        assert abs(after_diff["sourceBottomClipRect"]["height"] - before_diff["sourceBottomClipRect"]["height"]) <= 1, f"bottom-right source pane height should stay stable while the delayed diff response lands, got before={before_diff} after={after_diff}"

    def test_direct_debug_parameter_maps_big_left_settles_without_scroll(
        self, page, server_url, client
    ):
        base_path = DEBUG_DIR / "parameter_maps.nii"
        compare_path = DEBUG_DIR / "parameter_maps_005.nii"
        sid_base = client.post(
            "/load", json={"filepath": str(base_path), "name": base_path.name}
        ).json()["sid"]
        sid_compare = client.post(
            "/load", json={"filepath": str(compare_path), "name": compare_path.name}
        ).json()["sid"]

        page.goto(
            f"{server_url}/?sid={sid_base}"
            f"&compare_sid={sid_compare}"
            f"&compare_sids={sid_compare}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(900)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(220)
        page.keyboard.press("G")
        page.wait_for_timeout(160)

        def _state():
            return page.evaluate(
                """() => {
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffIsland = document.querySelector('#compare-diff-pane-cb')?.closest('.cb-island');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    return {
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffIslandRect: diffIsland?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    };
                }"""
            )

        early = _state()
        assert early["sharedCbDisplay"] != "none", f"shared source colorbar should already be visible shortly after X then G, got: {early}"
        assert early["sharedCbRect"], f"shared source colorbar should already be measurable shortly after X then G, got: {early}"
        assert early["sourceTopClipRect"] and early["sourceBottomClipRect"], f"source clips should already be measurable shortly after X then G, got: {early}"
        assert abs(early["sharedCbRect"]["width"] - early["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should already match the source clip width shortly after X then G, got: {early}"
        assert early["diffIslandRect"], f"diff colorbar island should already be measurable shortly after X then G, got: {early}"
        assert abs(early["sharedCbRect"]["height"] - early["diffIslandRect"]["height"]) <= 1, f"shared source and diff colorbar rows should already match heights shortly after X then G, got: {early}"
        assert early["sharedCbRect"]["top"] >= early["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should already sit below the top source pane shortly after X then G, got: {early}"
        assert early["sharedCbRect"]["bottom"] <= early["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should already sit above the bottom source pane shortly after X then G, got: {early}"

        page.wait_for_timeout(800)
        settled = _state()

        assert settled["sharedCbDisplay"] != "none", f"shared source colorbar should stay visible after the big-left layout settles, got: {settled}"
        assert settled["sharedCbRect"] and settled["sourceTopClipRect"] and settled["sourceBottomClipRect"], f"shared source colorbar and source clips should stay measurable after the big-left layout settles, got: {settled}"
        assert abs(settled["sharedCbRect"]["width"] - settled["sourceTopClipRect"]["width"]) <= 1, f"shared source colorbar should stay width-matched after the big-left layout settles, got: {settled}"
        assert abs(settled["sharedCbRect"]["height"] - settled["diffIslandRect"]["height"]) <= 1, f"shared source and diff colorbar rows should stay height-matched after the big-left layout settles, got: {settled}"
        assert settled["sharedCbRect"]["top"] >= settled["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should remain below the top source pane after the big-left layout settles, got: {settled}"
        assert settled["sharedCbRect"]["bottom"] <= settled["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should remain above the bottom source pane after the big-left layout settles, got: {settled}"
        assert abs(settled["sourceTopClipRect"]["width"] - early["sourceTopClipRect"]["width"]) <= 1, f"top-right source pane width should not drift after X then G without any scroll, got early={early} settled={settled}"
        assert abs(settled["sourceTopClipRect"]["height"] - early["sourceTopClipRect"]["height"]) <= 1, f"top-right source pane height should not drift after X then G without any scroll, got early={early} settled={settled}"
        assert abs(settled["sourceBottomClipRect"]["width"] - early["sourceBottomClipRect"]["width"]) <= 1, f"bottom-right source pane width should not drift after X then G without any scroll, got early={early} settled={settled}"
        assert abs(settled["sourceBottomClipRect"]["height"] - early["sourceBottomClipRect"]["height"]) <= 1, f"bottom-right source pane height should not drift after X then G without any scroll, got early={early} settled={settled}"

    def test_d_first_opens_second_cycles_percentile(self, loaded_viewer, sid_2d):
        # First tap `d` opens the histogram only. Second tap cycles + toasts.
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        page.keyboard.press("d")
        page.wait_for_timeout(500)
        toast = page.evaluate(
            "() => (document.getElementById('toast') || {}).textContent || ''"
        ).lower()
        assert "percentile" in toast, f"Expected 'percentile' toast after 2 taps, got: '{toast}'"

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
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        page.keyboard.press("c")
        page.keyboard.press("Enter")
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

    def test_border_defaults_on_and_uses_subtle_outline(self, loaded_viewer, sid_2d):
        """Borders should be on by default and use a subtle 1px outline."""
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        outline = page.evaluate(
            "() => getComputedStyle(document.querySelector('#canvas-viewport')).outline"
        )
        # Should be 1px, not 2px; should NOT be pure white
        assert "1px" in outline, f"expected 1px outline, got: {outline}"
        assert "rgb(255, 255, 255)" not in outline, f"border should not be pure white: {outline}"

    def test_first_border_toggle_turns_default_border_off(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("b")
        page.wait_for_timeout(200)
        state = page.evaluate(
            """() => ({
                status: document.querySelector('#status')?.innerText?.trim() || '',
                outline: getComputedStyle(document.querySelector('#canvas-viewport')).outline,
            })"""
        )
        assert "border" in state["status"].lower() and "off" in state["status"].lower(), (
            f"expected first press to disable default border, got: {state}"
        )
        assert ("0px" in state["outline"]) or ("none" in state["outline"]), (
            f"expected border outline to disappear after first press, got: {state}"
        )

    def test_multiview_default_border_draws_visible_border(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

        state = page.evaluate(
            """() => ({
                boxShadow: getComputedStyle(document.querySelector('.mv-pane')).boxShadow,
                paneClass: document.querySelector('.mv-pane').classList.contains('canvas-bordered'),
                status: document.querySelector('#status')?.innerText?.trim() || '',
            })"""
        )
        assert state["paneClass"], f"multiview pane should start bordered, got: {state}"
        assert state["boxShadow"] != "none", f"multiview border should be visibly drawn, got: {state}"

    def test_compare_center_border_stays_on_clip_and_panes_align(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        partner_path = tmp_path / "arr2d_border_compare.npy"
        np.save(partner_path, arr_2d * 0.5 + 0.25)
        sid_2d_b = client.post(
            "/load", json={"filepath": str(partner_path), "name": "arr2d_border_compare"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        _enter_compare(page, sid_2d_b)
        page.evaluate(
            """() => {
                compareLayoutMode = 'horizontal';
                _setCompareCenterMode(1);
                compareScaleCanvases();
            }"""
        )
        page.wait_for_timeout(350)

        state = page.evaluate(
            """() => {
                const leftWrap = document.querySelector('.compare-primary .compare-canvas-wrap');
                const leftClip = document.querySelector('.compare-primary .compare-canvas-clip');
                const centerClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                const rightClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                const slimVal = document.querySelector('#slim-cb-vmin');
                const compareVal = document.querySelector('#compare-left-pane-cb-vmin');
                const diffVal = document.querySelector('#compare-diff-pane-cb-vmin');
                const compareIsland = document.querySelector('.compare-primary .compare-pane-cb-island');
                const diffIsland = document.querySelector('#compare-diff-pane .compare-pane-cb-island');
                const leftRect = leftClip?.getBoundingClientRect() || null;
                const centerRect = centerClip?.getBoundingClientRect() || null;
                const rightRect = rightClip?.getBoundingClientRect() || null;
                return {
                    bodyRounded: document.body.classList.contains('rounded-panes'),
                    wrapClass: leftWrap?.classList.contains('canvas-bordered') || false,
                    clipClass: leftClip?.classList.contains('canvas-bordered') || false,
                    wrapShadow: leftWrap ? getComputedStyle(leftWrap).boxShadow : '',
                    clipShadow: leftClip ? getComputedStyle(leftClip).boxShadow : '',
                    clipRadius: leftClip ? getComputedStyle(leftClip).borderRadius : '',
                    slimValFont: slimVal ? getComputedStyle(slimVal).fontSize : '',
                    compareValFont: compareVal ? getComputedStyle(compareVal).fontSize : '',
                    diffValFont: diffVal ? getComputedStyle(diffVal).fontSize : '',
                    slimValFamily: slimVal ? getComputedStyle(slimVal).fontFamily : '',
                    compareValFamily: compareVal ? getComputedStyle(compareVal).fontFamily : '',
                    compareIslandGap: compareIsland ? getComputedStyle(compareIsland).gap : '',
                    compareIslandWidth: compareIsland ? getComputedStyle(compareIsland).width : '',
                    diffIslandWidth: diffIsland ? getComputedStyle(diffIsland).width : '',
                    leftRect,
                    centerRect,
                    rightRect,
                };
            }"""
        )

        assert state["bodyRounded"], f"compare mode should inherit rounded panes by default, got: {state}"
        assert state["wrapClass"], f"compare wrapper should still receive the border state class, got: {state}"
        assert state["clipClass"], f"compare viewport clip should receive the border state class, got: {state}"
        assert state["wrapShadow"] == "none", f"compare wrapper should not draw the border around the colorbar island, got: {state}"
        assert state["clipShadow"] != "none", f"compare viewport clip should draw the visible border, got: {state}"
        assert state["clipRadius"] != "0px", f"rounded compare panes should style the clip box, got: {state}"
        assert state["compareValFont"] == state["slimValFont"], f"compare source-pane labels should match normal colorbar font sizing, got: {state}"
        assert state["diffValFont"] == state["slimValFont"], f"diff-pane labels should match normal colorbar font sizing, got: {state}"
        assert state["compareValFamily"] == state["slimValFamily"], f"compare source-pane labels should match normal colorbar font family, got: {state}"
        assert state["compareIslandGap"] == "8px", f"compare source-pane label spacing should match the normal colorbar gap, got: {state}"
        assert state["diffIslandWidth"] == state["compareIslandWidth"], f"diff-pane colorbar island should match the source-pane width rule, got: {state}"
        assert state["leftRect"] and state["centerRect"] and state["rightRect"], f"compare clips should all exist in X mode, got: {state}"
        assert abs(state["leftRect"]["top"] - state["centerRect"]["top"]) <= 1, f"left and center compare clips should align vertically, got: {state}"
        assert abs(state["rightRect"]["top"] - state["centerRect"]["top"]) <= 1, f"right and center compare clips should align vertically, got: {state}"
        assert abs(state["leftRect"]["height"] - state["centerRect"]["height"]) <= 1, f"left and center compare clips should match height, got: {state}"
        assert abs(state["rightRect"]["height"] - state["centerRect"]["height"]) <= 1, f"right and center compare clips should match height, got: {state}"

    def test_compare_center_big_left_uses_shared_source_bar_and_g_toggles(
        self, loaded_viewer, sid_2d, arr_2d, client, tmp_path
    ):
        partner_path = tmp_path / "arr2d_big_left_compare.npy"
        np.save(partner_path, arr_2d * 0.75 + 0.1)
        sid_2d_b = client.post(
            "/load", json={"filepath": str(partner_path), "name": "arr2d_big_left_compare"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        _enter_compare(page, sid_2d_b)
        page.evaluate(
            """() => {
                compareLayoutMode = 'horizontal';
                _setCompareCenterMode(1);
                compareScaleCanvases();
            }"""
        )
        page.wait_for_timeout(250)
        page.keyboard.press("G")
        page.wait_for_timeout(250)

        state = page.evaluate(
            """() => {
                const wrap = document.querySelector('#compare-view-wrap');
                const diffPane = document.querySelector('#compare-diff-pane');
                const diffCanvas = document.querySelector('#compare-diff-canvas');
                const sourceTop = document.querySelector('.compare-primary');
                const sourceBottom = document.querySelector('.compare-secondary');
                const sharedCb = document.querySelector('#slim-cb-wrap');
                const sourceTopTitle = document.querySelector('.compare-primary .compare-title');
                const sourceTopIsland = document.querySelector('.compare-primary .compare-pane-cb-island');
                    const sourceBottomIsland = document.querySelector('.compare-secondary .compare-pane-cb-island');
                    const sourceTopBadge = document.querySelector('.compare-primary .compare-source-pane-badge');
                    const sourceBottomBadge = document.querySelector('.compare-secondary .compare-source-pane-badge');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    const diffIsland = document.querySelector('#compare-diff-pane .compare-pane-cb-island');
                    const diffRect = diffPane?.getBoundingClientRect() || null;
                const sourceTopRect = sourceTop?.getBoundingClientRect() || null;
                const sourceBottomRect = sourceBottom?.getBoundingClientRect() || null;
                const sharedCbRect = sharedCb?.getBoundingClientRect() || null;
                return {
                    wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                    diffRect,
                    sourceTopRect,
                    sourceBottomRect,
                    sharedCbRect,
                    diffCanvasRect: diffCanvas?.getBoundingClientRect() || null,
                    diffClipRect: diffClip?.getBoundingClientRect() || null,
                    sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                    sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                    diffIslandRect: diffIsland?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        sourceTopIslandDisplay: sourceTopIsland ? getComputedStyle(sourceTopIsland).display : 'missing',
                        sourceBottomIslandDisplay: sourceBottomIsland ? getComputedStyle(sourceBottomIsland).display : 'missing',
                        sourceTopBadgeText: sourceTopBadge?.textContent || '',
                        sourceBottomBadgeText: sourceBottomBadge?.textContent || '',
                        sourceTopBadgeDisplay: sourceTopBadge ? getComputedStyle(sourceTopBadge).display : 'missing',
                        sourceBottomBadgeDisplay: sourceBottomBadge ? getComputedStyle(sourceBottomBadge).display : 'missing',
                        sourceTitleWritingMode: sourceTopTitle ? getComputedStyle(sourceTopTitle).writingMode : '',
                    };
                }"""
            )

        assert state["wrapBigLeft"], f"compare should enter big-left layout after G in 2-array center mode, got: {state}"
        assert state["diffRect"] and state["sourceTopRect"] and state["sourceBottomRect"], f"big-left compare panes should all be measurable, got: {state}"
        assert state["diffRect"]["left"] < state["sourceTopRect"]["left"], f"center pane should move to the left of the stacked source panes, got: {state}"
        assert state["diffRect"]["width"] > state["sourceTopRect"]["width"], f"center pane should be wider than each source pane in big-left mode, got: {state}"
        assert abs(state["sourceTopRect"]["left"] - state["sourceBottomRect"]["left"]) <= 1, f"source panes should stay vertically stacked in one right column, got: {state}"
        assert state["sourceBottomRect"]["top"] > state["sourceTopRect"]["bottom"], f"source panes should stack vertically in big-left mode, got: {state}"
        assert state["sharedCbDisplay"] != "none", f"shared source colorbar should be visible in big-left mode, got: {state}"
        assert state["sharedCbRect"], f"shared source colorbar should have a layout box in big-left mode, got: {state}"
        assert state["sharedCbRect"]["top"] >= state["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should sit below the top source pane inside the stacked gap, got: {state}"
        assert state["sharedCbRect"]["bottom"] <= state["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should sit above the bottom source pane inside the stacked gap, got: {state}"
        assert state["sourceTopIslandDisplay"] == "none", f"top source pane should hide its per-pane colorbar in big-left mode, got: {state}"
        assert state["sourceBottomIslandDisplay"] == "none", f"bottom source pane should hide its per-pane colorbar in big-left mode, got: {state}"
        assert state["sourceTopBadgeText"].strip() == "A", f"top source pane should show an A badge in big-left mode, got: {state}"
        assert state["sourceBottomBadgeText"].strip() == "B", f"bottom source pane should show a B badge in big-left mode, got: {state}"
        assert state["sourceTopBadgeDisplay"] != "none", f"top source pane A badge should be visible in big-left mode, got: {state}"
        assert state["sourceBottomBadgeDisplay"] != "none", f"bottom source pane B badge should be visible in big-left mode, got: {state}"
        assert state["sourceTitleWritingMode"] == "vertical-rl", f"source titles should switch to vertical labels in big-left mode, got: {state}"
        assert state["diffCanvasRect"] and state["diffClipRect"] and state["sourceTopClipRect"] and state["sourceBottomClipRect"], f"big-left layout should expose measurable clip rects, got: {state}"
        assert abs(state["diffCanvasRect"]["width"] - state["diffClipRect"]["width"]) <= 1, f"center clip should shrink-wrap the rendered diff width in big-left mode, got: {state}"
        assert abs(state["diffClipRect"]["top"] - state["sourceTopClipRect"]["top"]) <= 1, f"left clip top should align with top-right clip top, got: {state}"
        assert abs(state["diffClipRect"]["bottom"] - state["sourceBottomClipRect"]["bottom"]) <= 1, f"left clip bottom should align with bottom-right clip bottom, got: {state}"
        assert state["diffIslandRect"], f"diff pane colorbar should remain measurable in big-left mode, got: {state}"

    def test_compare_center_big_left_keeps_shared_bar_and_alignment_after_wheel(
        self, loaded_viewer, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_big_left_compare.npy"
        np.save(path, arr_3d * 0.8 + 0.1)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_big_left_compare"}
        ).json()["sid"]

        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        _focus_kb(page)
        page.keyboard.press("X")
        page.wait_for_timeout(350)
        page.keyboard.press("G")
        page.wait_for_timeout(350)

        def _big_left_state():
            return page.evaluate(
                """() => {
                    const info = document.querySelector('#info');
                    const wrap = document.querySelector('#compare-view-wrap');
                    const panes = document.querySelector('#compare-panes');
                    const diffPane = document.querySelector('#compare-diff-pane');
                    const diffTitle = document.querySelector('#compare-diff-title');
                    const sharedCb = document.querySelector('#slim-cb-wrap');
                    const diffClip = document.querySelector('#compare-diff-pane .compare-canvas-clip');
                    const sourceTopClip = document.querySelector('.compare-primary .compare-canvas-clip');
                    const sourceBottomClip = document.querySelector('.compare-secondary .compare-canvas-clip');
                    const diffIsland = document.querySelector('#compare-diff-pane .compare-pane-cb-island');
                    return {
                        infoRect: info?.getBoundingClientRect() || null,
                        wrapBigLeft: wrap?.classList.contains('compare-center-layout-big-left') || false,
                        panesRect: panes?.getBoundingClientRect() || null,
                        gridCols: panes ? getComputedStyle(panes).gridTemplateColumns : '',
                        leftTrack: panes ? getComputedStyle(panes).getPropertyValue('--compare-big-left-left-w').trim() : '',
                        rightTrack: panes ? getComputedStyle(panes).getPropertyValue('--compare-big-left-right-w').trim() : '',
                        diffPaneDisplay: diffPane ? getComputedStyle(diffPane).display : 'missing',
                        compareCenterMode,
                        compareLayoutMode,
                        diffTitleRect: diffTitle?.getBoundingClientRect() || null,
                        sharedCbRect: sharedCb?.getBoundingClientRect() || null,
                        sharedCbDisplay: sharedCb ? getComputedStyle(sharedCb).display : 'missing',
                        diffClipRect: diffClip?.getBoundingClientRect() || null,
                        sourceTopClipRect: sourceTopClip?.getBoundingClientRect() || null,
                        sourceBottomClipRect: sourceBottomClip?.getBoundingClientRect() || null,
                        diffIslandRect: diffIsland?.getBoundingClientRect() || null,
                    };
                }"""
            )

        initial = _big_left_state()
        assert initial["wrapBigLeft"], f"big-left compare layout should be active immediately after G, got: {initial}"
        assert initial["sharedCbDisplay"] != "none", f"shared source colorbar should be visible immediately on big-left entry, got: {initial}"
        assert initial["sharedCbRect"], f"shared source colorbar should be measurable immediately on big-left entry, got: {initial}"
        assert initial["infoRect"] and initial["diffTitleRect"], f"big-left title and dimbar should both be measurable, got: {initial}"
        assert (
            initial["diffTitleRect"]["bottom"] <= initial["infoRect"]["top"] + 1
            or initial["diffTitleRect"]["top"] >= initial["infoRect"]["bottom"] - 1
            or initial["diffTitleRect"]["right"] <= initial["infoRect"]["left"] + 1
            or initial["diffTitleRect"]["left"] >= initial["infoRect"]["right"] - 1
        ), f"big-left center title should not overlap the dimbar, got: {initial}"
        assert initial["diffClipRect"] and initial["sourceTopClipRect"] and initial["sourceBottomClipRect"], f"big-left clips should all be measurable immediately after G, got: {initial}"
        assert initial["diffClipRect"]["right"] <= initial["sourceTopClipRect"]["left"] + 1, f"left diff clip should not overlap the right source column on big-left entry, got: {initial}"

        page.mouse.move(
            initial["sourceTopClipRect"]["left"] + initial["sourceTopClipRect"]["width"] / 2,
            initial["sourceTopClipRect"]["top"] + initial["sourceTopClipRect"]["height"] / 2,
        )
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(1000)
        after = _big_left_state()

        assert after["wrapBigLeft"], f"big-left compare layout should remain active after wheel-driven slice updates, got: {after}"
        assert after["sharedCbDisplay"] != "none", f"shared source colorbar should remain visible after wheel-driven slice updates, got: {after}"
        assert after["diffClipRect"] and after["sourceTopClipRect"] and after["sourceBottomClipRect"], f"big-left clips should remain measurable after wheel-driven slice updates, got: {after}"
        assert abs(after["diffClipRect"]["top"] - after["sourceTopClipRect"]["top"]) <= 1, f"left clip top should stay aligned with the top-right clip after wheel-driven slice updates, got: {after}"
        assert abs(after["diffClipRect"]["bottom"] - after["sourceBottomClipRect"]["bottom"]) <= 1, f"left clip bottom should stay aligned with the lower-right clip after wheel-driven slice updates, got: {after}"
        assert after["diffIslandRect"] and after["sharedCbRect"], f"big-left colorbar rows should stay measurable after wheel-driven slice updates, got: {after}"
        assert after["sharedCbRect"]["top"] >= after["sourceTopClipRect"]["bottom"] - 1, f"shared source colorbar should stay below the top source pane after wheel-driven slice updates, got: {after}"
        assert after["sharedCbRect"]["bottom"] <= after["sourceBottomClipRect"]["top"] + 1, f"shared source colorbar should stay above the bottom source pane after wheel-driven slice updates, got: {after}"

    def test_multiview_hover_border_and_colorbar_clearance(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)
        page.hover(".mv-canvas")
        page.wait_for_timeout(100)

        state = page.evaluate(
            """() => {
                const row = document.querySelector('#mv-panes');
                const pane = document.querySelector('.mv-pane');
                const cb = document.querySelector('#mv-cb-wrap');
                const cbRect = cb.getBoundingClientRect();
                const frame = getComputedStyle(pane, '::after');
                const rowRect = row.getBoundingClientRect();
                return {
                    active: row.classList.contains('mv-crosshair-active'),
                    frameShadow: frame.boxShadow,
                    colorbarBottomGap: Math.round(window.innerHeight - cbRect.bottom),
                    centerDelta: Math.round(Math.abs((rowRect.left + rowRect.width / 2) - window.innerWidth / 2)),
                };
            }"""
        )
        assert state["active"], f"hovering a multiview canvas should activate pane frame, got: {state}"
        assert "inset" in state["frameShadow"], f"pane frame should render above pane contents, got: {state}"
        assert "1.5px" in state["frameShadow"], f"pane frame should match crosshair thickness, got: {state}"
        assert state["colorbarBottomGap"] >= 36, f"multiview colorbar should clear the viewport bottom, got: {state}"
        assert state["centerDelta"] <= 2, f"multiview pane cluster should be horizontally centered, got: {state}"

    def test_multiview_colorbar_gap_matches_normal_mode_in_big_left(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        page.set_viewport_size({"width": 1700, "height": 1100})
        page.wait_for_timeout(200)
        _focus_kb(page)

        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

        def _mv_gap_state():
            return page.evaluate(
                """() => {
                    const panes = document.querySelector('#mv-panes');
                    const cb = document.querySelector('#mv-cb-wrap');
                    if (!panes || !cb) return null;
                    const panesRect = panes.getBoundingClientRect();
                    const cbRect = cb.getBoundingClientRect();
                    return {
                        gap: Math.round(cbRect.top - panesRect.bottom),
                        wrapGap: Math.round(parseFloat(getComputedStyle(document.getElementById('multi-view-wrap')).gap) || 0),
                        orthoLayoutMode,
                    };
                }"""
            )

        horizontal = _mv_gap_state()
        assert horizontal, "multiview colorbar gap should be measurable in horizontal layout"
        assert abs(horizontal["gap"] - horizontal["wrapGap"]) <= 1, (
            f"v-mode colorbar gap should match the multiview container gap with no extra wrapper offset, got: {horizontal}"
        )

        page.evaluate("""() => { _mvSetOrthoLayoutMode('big-left', { silent: true }); }""")
        page.wait_for_timeout(300)
        big_left = _mv_gap_state()
        assert big_left, "multiview colorbar gap should be measurable in big-left layout"
        assert abs(big_left["gap"] - big_left["wrapGap"]) <= 1, (
            f"big-left multiview colorbar gap should match the multiview container gap with no extra wrapper offset, got: {big_left}"
        )

    def test_multiview_auto_layout_picks_horizontal_on_jupyter_like_viewport_and_manual_override_sticks(
        self, loaded_viewer, sid_3d
    ):
        page = loaded_viewer(sid_3d)
        page.set_viewport_size({"width": 1280, "height": 760})
        page.wait_for_timeout(200)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)

        def _state():
            return page.evaluate(
                """() => {
                    const row = document.querySelector('#mv-panes');
                    const cb = document.querySelector('#mv-cb-wrap');
                    const panes = Array.from(document.querySelectorAll('#mv-panes > .mv-pane')).map(p => {
                        const r = p.getBoundingClientRect();
                        return {
                            left: Math.round(r.left),
                            top: Math.round(r.top),
                            width: Math.round(r.width),
                            height: Math.round(r.height),
                        };
                    });
                    return {
                        orthoLayoutMode,
                        orthoAutoLayoutMode: _orthoAutoLayoutMode,
                        rowPosition: row ? getComputedStyle(row).position : 'missing',
                        rowClass: row?.className || '',
                        cbWidth: cb ? Math.round(cb.getBoundingClientRect().width) : -1,
                        panes,
                    };
                }"""
            )

        initial = _state()
        assert initial["orthoLayoutMode"] is None, f"jupyter-like viewport should remain in auto ortho layout mode until manual override, got: {initial}"
        assert initial["orthoAutoLayoutMode"] == "horizontal", f"jupyter-like viewport should auto-pick horizontal ortho layout, got: {initial}"
        assert initial["rowPosition"] != "relative", f"horizontal ortho auto-layout should keep the legacy row flow, got: {initial}"
        assert len(initial["panes"]) == 3, f"multiview should expose exactly three ortho panes, got: {initial}"
        assert max(abs(initial["panes"][i]["top"] - initial["panes"][0]["top"]) for i in range(1, 3)) <= 2, f"horizontal ortho auto-layout should keep panes on one row, got: {initial}"
        assert max(abs(initial["panes"][i]["width"] - initial["panes"][0]["width"]) for i in range(1, 3)) <= 2, f"horizontal ortho auto-layout should keep pane widths uniform, got: {initial}"

        page.set_viewport_size({"width": 1700, "height": 1100})
        page.wait_for_timeout(500)
        resized = _state()
        assert resized["orthoAutoLayoutMode"] == "horizontal", f"ortho auto-layout choice should stay stable across resize until manually overridden, got: {resized}"
        assert resized["rowPosition"] != "relative", f"resizing alone should not flip ortho layout out of horizontal flow, got: {resized}"

        page.keyboard.press("g")
        page.wait_for_timeout(500)
        overridden = _state()
        assert overridden["orthoLayoutMode"] == "big-left", f"manual g cycling should persist a concrete ortho preset override, got: {overridden}"
        assert overridden["rowPosition"] == "relative", f"big-left ortho override should switch mv panes into preset positioning, got: {overridden}"
        assert "mv-promote-enabled" in overridden["rowClass"], f"big-left ortho override should enable the promotable preset chrome, got: {overridden}"
        assert abs(overridden["cbWidth"] - initial["cbWidth"]) <= 1, f"multiview colorbar width should stay stable when g switches to big-left, got initial={initial}, overridden={overridden}"
        assert overridden["panes"][0]["width"] > overridden["panes"][1]["width"], f"big-left ortho override should make the first pane larger than the stacked panes, got: {overridden}"
        assert overridden["panes"][1]["top"] < overridden["panes"][2]["top"], f"big-left ortho override should stack the secondary panes vertically, got: {overridden}"

        page.keyboard.press("g")
        page.wait_for_timeout(500)
        returned = _state()
        assert returned["orthoLayoutMode"] == "horizontal", f"manual g cycling should now return directly to horizontal without vertical or big-top, got: {returned}"
        assert returned["rowPosition"] != "relative", f"returning to horizontal should restore legacy row flow, got: {returned}"
        assert "mv-promote-enabled" not in returned["rowClass"], f"horizontal ortho layout should clear promotable preset chrome, got: {returned}"
        assert abs(returned["cbWidth"] - initial["cbWidth"]) <= 1, f"multiview colorbar width should stay stable when g returns to horizontal, got initial={initial}, returned={returned}"

        page.set_viewport_size({"width": 1280, "height": 760})
        page.wait_for_timeout(500)
        final = _state()
        assert final["orthoLayoutMode"] == "horizontal", f"manual ortho preset override should stay sticky after resizing back to a smaller viewport, got: {final}"
        assert final["rowPosition"] != "relative", f"manual horizontal ortho override should remain active after resize, got: {final}"

    def test_multiview_auto_layout_picks_big_left_on_large_viewport(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        page.set_viewport_size({"width": 1700, "height": 1100})
        page.wait_for_timeout(200)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)

        state = page.evaluate(
            """() => {
                const row = document.querySelector('#mv-panes');
                const panes = Array.from(document.querySelectorAll('#mv-panes > .mv-pane')).map(p => {
                    const r = p.getBoundingClientRect();
                    return {
                        left: Math.round(r.left),
                        top: Math.round(r.top),
                        width: Math.round(r.width),
                        height: Math.round(r.height),
                    };
                });
                return {
                    orthoLayoutMode,
                    orthoAutoLayoutMode: _orthoAutoLayoutMode,
                    rowPosition: row ? getComputedStyle(row).position : 'missing',
                    rowClass: row?.className || '',
                    panes,
                };
            }"""
        )

        assert state["orthoLayoutMode"] is None, f"large viewport should still be in auto ortho layout mode until manually overridden, got: {state}"
        assert state["orthoAutoLayoutMode"] == "big-left", f"large viewport should auto-pick big-left ortho layout, got: {state}"
        assert state["rowPosition"] == "relative", f"big-left ortho auto-layout should use preset positioning, got: {state}"
        assert "mv-promote-enabled" in state["rowClass"], f"big-left ortho auto-layout should enable the promotable preset chrome, got: {state}"
        assert len(state["panes"]) == 3, f"multiview should expose exactly three ortho panes, got: {state}"
        assert state["panes"][0]["width"] > state["panes"][1]["width"], f"big-left ortho auto-layout should make the first pane larger than the stacked panes, got: {state}"
        assert state["panes"][1]["top"] < state["panes"][2]["top"], f"big-left ortho auto-layout should stack the secondary panes vertically, got: {state}"

    def test_compare_multiview_uses_single_shared_colorbar_and_aligned_columns(
        self, loaded_viewer, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare_mv.npy"
        np.save(path, arr_3d * 0.5)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare_mv"}
        ).json()["sid"]

        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        _enter_compare(page, sid_compare)
        page.keyboard.press("v")
        page.wait_for_selector("#qmri-view-wrap.active .qv-row", timeout=5_000)
        page.wait_for_timeout(800)

        state = page.evaluate(
            """() => {
                const rows = Array.from(document.querySelectorAll('#qmri-view-wrap .qv-row')).map(row =>
                    Array.from(row.querySelectorAll('.qv-canvas-wrap')).map(el => {
                        const r = el.getBoundingClientRect();
                        return {
                            left: Math.round(r.left),
                            width: Math.round(r.width),
                            center: Math.round(r.left + r.width / 2),
                        };
                    })
                );
                const sharedCb = document.querySelector('#mv-cb-wrap');
                const sharedRect = sharedCb ? sharedCb.getBoundingClientRect() : null;
                const paneCbs = Array.from(document.querySelectorAll('#qmri-view-wrap .qv-cb-island'))
                    .filter(el => getComputedStyle(el).display !== 'none');
                return {
                    rows,
                    viewportCenter: Math.round(window.innerWidth / 2),
                    sharedCbVisible: !!sharedCb && getComputedStyle(sharedCb).display !== 'none',
                    sharedCbWidth: sharedRect ? Math.round(sharedRect.width) : 0,
                    visiblePaneCbs: paneCbs.length,
                };
            }"""
        )

        assert len(state["rows"]) == 2, f"expected one compare-mv row per array, got: {state}"
        assert all(len(row) == 3 for row in state["rows"]), f"expected 3 panes per row, got: {state}"
        for col in range(3):
            lefts = [row[col]["left"] for row in state["rows"]]
            widths = [row[col]["width"] for row in state["rows"]]
            assert max(lefts) - min(lefts) <= 2, f"column {col} should align across rows, got: {state}"
            assert max(widths) - min(widths) <= 2, f"column {col} should keep the same pane width across rows, got: {state}"
        middle_centers = [row[1]["center"] for row in state["rows"]]
        assert max(abs(center - state["viewportCenter"]) for center in middle_centers) <= 2, (
            f"middle compare-mv pane should stay centered in the viewport, got: {state}"
        )
        assert state["sharedCbVisible"], f"compare-mv should show the shared multiview colorbar, got: {state}"
        assert state["sharedCbWidth"] > 0, f"shared multiview colorbar should have a real width, got: {state}"
        assert state["visiblePaneCbs"] == 0, f"compare-mv should not show per-pane colorbars, got: {state}"

    def test_multiview_rounded_panes_round_visible_box(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

        state = page.evaluate(
            """() => ({
                status: document.querySelector('#status')?.innerText?.trim() || '',
                bodyRounded: document.body.classList.contains('rounded-panes'),
                radius: getComputedStyle(document.querySelector('.mv-pane')).borderRadius,
                overflow: getComputedStyle(document.querySelector('.mv-pane')).overflow,
            })"""
        )

        assert state["bodyRounded"], f"rounded panes body class should be enabled, got: {state}"
        assert state["radius"] != "0px", f"multiview visible box should get rounded corners, got: {state}"
        assert state["overflow"] == "hidden", f"multiview visible box should clip to rounded corners, got: {state}"

        page.keyboard.press("Shift+B")
        page.wait_for_timeout(200)
        toggled = page.evaluate(
            """() => ({
                status: document.querySelector('#status')?.innerText?.trim() || '',
                bodyRounded: document.body.classList.contains('rounded-panes'),
            })"""
        )
        assert not toggled["bodyRounded"], f"Shift+B should toggle default rounded panes off, got: {toggled}"
        assert "rounded panes" in toggled["status"].lower() and "off" in toggled["status"].lower(), (
            f"expected Shift+B to disable rounded panes, got: {toggled}"
        )

    def test_multiview_empty_square_uses_colormap_min_fill(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)
        page.keyboard.press("c")
        page.wait_for_timeout(250)

        state = page.evaluate(
            """() => {
                const pane = document.querySelector('.mv-pane');
                const bg = getComputedStyle(pane).backgroundColor;
                const canvas = document.querySelector('.mv-canvas');
                const canvasBg = getComputedStyle(canvas).backgroundColor;
                const stops = colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]];
                const expected = stops && stops[0] ? `rgb(${stops[0][0]}, ${stops[0][1]}, ${stops[0][2]})` : null;
                return { bg, canvasBg, expected, colormap: currentColormap() };
            }"""
        )

        assert state["expected"] is not None, f"expected current colormap to expose gradient stops, got: {state}"
        assert state["bg"] == state["expected"], (
            f"multiview square background should use the colormap minimum color, got: {state}"
        )
        assert state["canvasBg"] == state["expected"], (
            f"multiview canvas background should use the colormap minimum color, got: {state}"
        )

    def test_multiview_rotate_updates_all_panes(self, loaded_viewer, sid_3d):
        """Pressing r in multiview should swap axes globally across all 3 panes."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(300)

        # Trigger a saveState() by cycling colormap, so we have a baseline in sessionStorage
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        page.keyboard.press("c")
        page.keyboard.press("Enter")
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
        page.keyboard.press("Shift+F")
        page.wait_for_function(
            "() => document.body.classList.contains('fullscreen-mode')",
            timeout=5000,
        )
        assert "fullscreen-mode" in (page.locator("body").get_attribute("class") or "")
        page.keyboard.press("Shift+F")
        page.wait_for_function(
            "() => !document.body.classList.contains('fullscreen-mode')",
            timeout=5000,
        )
        assert "fullscreen-mode" not in (page.locator("body").get_attribute("class") or "")

    def test_inline_embed_starts_non_immersive_and_top_aligned(self, page, server_url, sid_3d):
        page.goto(f"{server_url}/?sid={sid_3d}&inline=1")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page.wait_for_timeout(400)

        assert "fullscreen-mode" not in (page.locator("body").get_attribute("class") or "")
        assert page.evaluate(
            "() => parseFloat(getComputedStyle(document.getElementById('wrapper')).paddingTop || '0') <= 20"
        )


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
    def test_shift_c_picker_is_centered_and_shortlisted(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+C")
        page.wait_for_selector("#cmap-picker.visible", timeout=2_000)
        picker = page.locator("#cmap-picker-box")
        box = picker.bounding_box()
        viewport = page.viewport_size
        assert box is not None, "Colormap picker modal did not render"
        assert viewport is not None, "Viewport size unavailable"
        modal_cx = box["x"] + box["width"] / 2
        modal_cy = box["y"] + box["height"] / 2
        assert abs(modal_cx - viewport["width"] / 2) < viewport["width"] * 0.12
        assert abs(modal_cy - viewport["height"] / 2) < viewport["height"] * 0.14
        page.wait_for_function(
            "() => document.querySelector('#cmap-picker-summary')?.textContent?.includes('suggested colormaps')"
        )
        visible_count = page.locator("#cmap-picker-list .cmap-p-cell:not(.hidden)").count()
        assert visible_count <= 16, f"Expected a shortlist, saw {visible_count} visible colormaps"
        summary = page.inner_text("#cmap-picker-summary")
        assert "suggested colormaps" in summary.lower()
        page.keyboard.type("inferno", delay=20)
        page.wait_for_timeout(250)
        filtered_box = picker.bounding_box()
        assert filtered_box is not None
        assert abs(filtered_box["y"] - box["y"]) < 2, "Picker box shifted vertically while filtering"

    def test_C_key_with_valid_colormap_changes_canvas(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        before = page.evaluate(_JS_CENTER_PIXEL)
        page.keyboard.press("Shift+C")
        page.wait_for_selector("#cmap-picker.visible", timeout=2_000)
        page.keyboard.type("inferno", delay=30)
        page.wait_for_timeout(400)
        page.keyboard.press("Enter")
        page.wait_for_timeout(1200)
        after = page.evaluate(_JS_CENTER_PIXEL)
        assert before != after, "Center pixel unchanged after Shift+C + inferno colormap"


class TestSessionStorage:
    def test_colormap_persists_across_reload(self, loaded_viewer, sid_2d, server_url):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        # Open the colormap menu, cycle away from gray, then commit.
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2_000)
        page.keyboard.press("c")
        page.keyboard.press("Enter")
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


class TestNormalInspectInteractions:
    def test_ctrl_hover_shows_and_hides_loupe(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)

        page.mouse.move(cx, cy)
        page.wait_for_timeout(120)

        hidden_without_ctrl = page.evaluate(
            "() => getComputedStyle(document.getElementById('main-loupe')).display === 'none'"
        )
        assert hidden_without_ctrl, "loupe should stay hidden during a plain hover"

        page.keyboard.down("Control")
        page.wait_for_timeout(120)

        visible = page.evaluate(
            "() => getComputedStyle(document.getElementById('main-loupe')).display !== 'none'"
        )
        assert visible, "loupe should appear during a Control-hover"

        page.keyboard.up("Control")
        page.wait_for_timeout(220)

        hidden = page.evaluate(
            "() => getComputedStyle(document.getElementById('main-loupe')).display === 'none'"
        )
        assert hidden, "loupe should collapse after Control is released"

    def test_ctrl_hover_shows_loupe_in_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)

        canvas = page.locator(".mv-canvas").first
        cx, cy = _center_of(canvas)

        page.mouse.move(cx, cy)
        page.wait_for_timeout(120)
        page.keyboard.down("Control")
        page.wait_for_timeout(120)

        visible = page.evaluate(
            "() => getComputedStyle(document.getElementById('qmri-loupe')).display !== 'none'"
        )
        assert visible, "loupe should appear during a Control-hover in multiview"

        page.keyboard.up("Control")
        page.wait_for_timeout(220)

        hidden = page.evaluate(
            "() => getComputedStyle(document.getElementById('qmri-loupe')).display === 'none'"
        )
        assert hidden, "multiview loupe should collapse after Control is released"

    def test_ctrl_wheel_still_scrolls_main_view(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)

        before = page.evaluate(
            """() => {
                const sliceDim = current_slice_dim;
                indices[sliceDim] = 0;
                renderInfo();
                updateView();
                return { sliceDim, index: indices[sliceDim], zoom: userZoom };
            }"""
        )

        page.mouse.move(cx, cy)
        page.keyboard.down("Control")
        page.wait_for_timeout(120)

        visible_before = page.evaluate(
            "() => getComputedStyle(document.getElementById('main-loupe')).display !== 'none'"
        )
        loupe_before = page.evaluate(
            "() => document.getElementById('main-loupe-canvas').toDataURL()"
        )
        assert visible_before, "loupe should stay visible while Control is held over the main view"

        page.mouse.wheel(0, -120)
        page.wait_for_timeout(220)

        after = page.evaluate(
            "() => ({ index: indices[current_slice_dim], zoom: userZoom, sliceDim: current_slice_dim })"
        )
        visible_after = page.evaluate(
            "() => getComputedStyle(document.getElementById('main-loupe')).display !== 'none'"
        )
        loupe_after = page.evaluate(
            "() => document.getElementById('main-loupe-canvas').toDataURL()"
        )

        page.keyboard.up("Control")

        assert after["sliceDim"] == before["sliceDim"], (
            f"Control-wheel should keep using the same slice dim, got before={before}, after={after}"
        )
        assert after["index"] == before["index"] + 1, (
            f"Control-wheel should still scroll the main view, got before={before}, after={after}"
        )
        assert after["zoom"] == before["zoom"], (
            f"Control-wheel should not change main-view zoom, got before={before}, after={after}"
        )
        assert visible_after, "loupe should remain visible while scrolling with Control held"
        assert loupe_after != loupe_before, "main-view loupe should redraw after wheel-driven slice changes"

    def test_ctrl_wheel_still_scrolls_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)

        canvas = page.locator(".mv-canvas").first
        cx, cy = _center_of(canvas)
        before = page.evaluate(
            """() => {
                const view = mvViews[0];
                indices[view.sliceDir] = 0;
                renderInfo();
                mvViews.forEach(v => { mvDrawFrame(v); mvRender(v); });
                return { sliceDir: view.sliceDir, indices: [...indices] };
            }"""
        )

        page.mouse.move(cx, cy)
        page.keyboard.down("Control")
        page.wait_for_timeout(120)

        visible_before = page.evaluate(
            "() => getComputedStyle(document.getElementById('qmri-loupe')).display !== 'none'"
        )
        loupe_before = page.evaluate(
            "() => document.querySelector('#qmri-loupe canvas').toDataURL()"
        )
        assert visible_before, "loupe should stay visible while Control is held over a multiview pane"

        page.mouse.wheel(0, -120)
        page.wait_for_timeout(220)

        after = page.evaluate("() => [...indices]")
        visible_after = page.evaluate(
            "() => getComputedStyle(document.getElementById('qmri-loupe')).display !== 'none'"
        )
        loupe_after = page.evaluate(
            "() => document.querySelector('#qmri-loupe canvas').toDataURL()"
        )

        page.keyboard.up("Control")

        assert after[before["sliceDir"]] == before["indices"][before["sliceDir"]] + 1, (
            f"Control-wheel should still scroll multiview slices, got before={before}, after={after}"
        )
        assert visible_after, "multiview loupe should remain visible while scrolling with Control held"
        assert loupe_after != loupe_before, "multiview loupe should redraw after wheel-driven slice changes"

    def test_hover_info_click_pins_multiple_and_compare_does_not(self, loaded_viewer, sid_2d, client, arr_2d, tmp_path):
        partner_path = tmp_path / "arr2d_pin_partner.npy"
        np.save(partner_path, np.flipud(arr_2d))
        partner_sid = client.post(
            "/load", json={"filepath": str(partner_path), "name": "arr2d_pin_partner"}
        ).json()["sid"]

        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(180)

        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)
        page.mouse.move(cx - 20, cy - 12)
        page.wait_for_timeout(180)

        hover_card_visible = page.evaluate(
            "() => !!document.querySelector('#main-pixel-info .pixel-hover-card')"
        )
        hover_coords = page.evaluate(
            "() => (document.querySelector('#main-pixel-info .pixel-hover-coords') || {}).textContent || ''"
        )
        assert hover_card_visible, "hover-info mode should show the pinned-style hover card"
        assert "x=" in hover_coords and "y=" in hover_coords, "hover card should include coordinates"

        page.mouse.click(cx, cy)
        page.mouse.click(cx + 80, cy + 30)
        page.wait_for_timeout(240)

        pin_count = page.evaluate(
            "() => document.querySelectorAll('#main-pixel-pins .main-pixel-pin').length"
        )
        pin_value = page.evaluate(
            "() => (document.querySelector('#main-pixel-pins .main-pixel-pin .pixel-pin-value') || {}).textContent || ''"
        )
        assert pin_count == 2, "clicking twice in hover-info mode should create two pins"
        assert pin_value.strip(), "pinned readout should show a numeric value"

        page.keyboard.press("Escape")
        page.wait_for_timeout(180)
        pin_hidden = page.evaluate(
            "() => document.querySelectorAll('#main-pixel-pins .main-pixel-pin').length === 0"
        )
        assert pin_hidden, "Escape should clear all pinned readouts"

        _enter_compare(page, partner_sid)
        _focus_kb(page)
        page.keyboard.press("i")
        page.wait_for_timeout(180)
        cmp_canvas = page.locator("#compare-left-canvas")
        ccx, ccy = _center_of(cmp_canvas)
        page.mouse.click(ccx, ccy)
        page.wait_for_timeout(220)

        compare_pin_hidden = page.evaluate(
            "() => document.querySelectorAll('#main-pixel-pins .main-pixel-pin').length === 0"
        )
        assert compare_pin_hidden, "pinned readout should stay disabled in compare mode"


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
        # Open then cycle to viridis (gray → lipari → navia → viridis).
        for _ in range(4):
            page.keyboard.press("c")
            page.wait_for_timeout(600)
        page.keyboard.press("Enter")
        _compare_snapshot(page, "2d_gradient_viridis")

    def test_3d_multiview(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)
        _compare_snapshot(page, "3d_multiview")
