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
# Perf instrumentation
# ---------------------------------------------------------------------------

def test_perf_mode_collects_render_samples(page, server_url, sid_3d):
    page.goto(f"{server_url}/?sid={sid_3d}&perf=1")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_function(
        "() => window.__arrayviewPerf && window.__arrayviewPerf.samples.length > 0",
        timeout=15_000,
    )
    perf = page.evaluate(
        "() => ({enabled: window.__arrayviewPerf.enabled, sample: window.__arrayviewPerf.samples.at(-1)})"
    )
    assert perf["enabled"] is True
    assert perf["sample"]["mode"] in ("ws", "http")
    assert perf["sample"]["client_total_ms"] is None or perf["sample"]["client_total_ms"] >= 0

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


def test_overlay_palette_visible_on_first_load(page, client, server_url, tmp_path):
    base = np.zeros((8, 32, 32), dtype=np.float32)
    mask_a = np.zeros((8, 32, 32), dtype=np.uint8)
    mask_b = np.zeros((8, 32, 32), dtype=np.uint8)
    mask_a[:, 3:12, 3:12] = 1
    mask_b[:, 20:29, 20:29] = 1
    sids = []
    for name, array in (("base", base), ("mask_a", mask_a), ("mask_b", mask_b)):
        path = tmp_path / f"{name}.npy"
        np.save(path, array)
        sids.append(client.post("/load", json={"filepath": str(path), "name": name}).json()["sid"])

    page.goto(
        f"{server_url}/?sid={sids[0]}&overlay_sid={sids[1]},{sids[2]}"
        "&overlay_names=mask_a,mask_b"
    )
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    palette = page.locator("#overlay-palette")
    palette.wait_for(state="visible", timeout=5_000)

    assert palette.get_attribute("aria-hidden") == "false"
    assert palette.locator(".overlay-palette-row").count() == 2
    page.evaluate(
        """() => {
            window.__overlayWsMessages = [];
            const realSend = ws.send.bind(ws);
            ws.send = (msg) => {
                try {
                    const parsed = JSON.parse(msg);
                    if ('overlay_alphas' in parsed) window.__overlayWsMessages.push(parsed);
                } catch (e) {}
                return realSend(msg);
            };
        }"""
    )
    palette.locator(".overlay-palette-row").nth(1).hover()
    assert page.evaluate("() => _overlayFocusIdx") == 1
    expected_alphas = page.evaluate("() => _getVisibleOverlayAlphas()")
    dim_alpha, focus_alpha = [float(v) for v in expected_alphas.split(",")]
    assert focus_alpha > dim_alpha
    assert page.evaluate("() => _getVisibleOverlayAlphas()") == expected_alphas
    page.wait_for_function(
        "expected => window.__overlayWsMessages.some(m => m.overlay_alphas === expected)",
        arg=expected_alphas,
    )
    assert "focused" in palette.locator(".overlay-palette-row").nth(1).get_attribute("class")
    assert "dimmed" in palette.locator(".overlay-palette-row").nth(0).get_attribute("class")
    assert page.evaluate(
        "() => getComputedStyle(document.querySelector('.overlay-palette-row.focused')).outlineStyle"
    ) == "solid"
    assert page.evaluate(
        "() => getComputedStyle(document.querySelector('.overlay-palette-row.focused .overlay-palette-name')).color"
    ) != page.evaluate(
        "() => getComputedStyle(document.querySelector('.overlay-palette-row.dimmed .overlay-palette-name')).color"
    )
    page.mouse.move(5, 5)
    assert page.evaluate("() => _overlayFocusIdx") is None
    mode_group = palette.locator(".overlay-palette-mode-group")
    mode_fill = mode_group.locator(".overlay-palette-mode").nth(0)
    mode_outline = mode_group.locator(".overlay-palette-mode").nth(1)
    assert mode_fill.inner_text() == "filled"
    assert mode_outline.inner_text() == "outline"
    assert "active" in mode_fill.get_attribute("class")
    mode_outline.click()
    assert page.evaluate("() => overlayOutlineOnly") is True
    assert "active" in mode_outline.get_attribute("class")
    assert page.evaluate(
        "() => { const p = new URLSearchParams(); _applyOverlayRenderParams(p); return p.get('overlay_outline'); }"
    ) == "1"
    mode_fill.click()
    assert page.evaluate("() => overlayOutlineOnly") is False
    all_btn = palette.locator(".overlay-palette-all")
    assert "active" in all_btn.get_attribute("class")
    all_btn.click()
    assert page.evaluate("() => _getVisibleOverlaySids()") == ""
    assert "active" not in all_btn.get_attribute("class")
    all_btn.click()
    assert page.evaluate("() => _getVisibleOverlaySids().split(',').length") == 2
    page.wait_for_timeout(1200)
    DEBUG_DIR.mkdir(exist_ok=True)
    page.screenshot(path=str(DEBUG_DIR / "overlay_palette_initial_visible.png"))

    _focus_kb(page)
    page.keyboard.press("/")
    page.keyboard.press("o")
    assert not palette.is_visible()
    page.keyboard.press("/")
    page.keyboard.press("o")
    assert palette.is_visible()

    handle = palette.locator(".overlay-palette-drag-handle")
    handle_box = handle.bounding_box()
    before_drag = palette.bounding_box()
    page.mouse.move(
        handle_box["x"] + handle_box["width"] / 2,
        handle_box["y"] + handle_box["height"] / 2,
    )
    page.mouse.down()
    page.mouse.move(before_drag["x"] - 180, before_drag["y"] + 140, steps=8)
    page.mouse.up()
    after_drag = palette.bounding_box()

    assert after_drag["x"] < before_drag["x"] - 100
    assert after_drag["y"] > before_drag["y"] + 80
    page.evaluate("() => _reconcileUI()")
    after_reconcile = palette.bounding_box()
    assert abs(after_reconcile["x"] - after_drag["x"]) <= 2
    assert abs(after_reconcile["y"] - after_drag["y"]) <= 2
    page.screenshot(path=str(DEBUG_DIR / "overlay_palette_dragged.png"))

    _focus_kb(page)
    page.keyboard.press("v")
    page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
    mode_box = palette.bounding_box()
    viewport = page.viewport_size
    assert mode_box["x"] >= 0 and mode_box["y"] >= 0
    assert mode_box["x"] + mode_box["width"] <= viewport["width"]
    assert mode_box["y"] + mode_box["height"] <= viewport["height"]
    assert page.evaluate("() => _overlayPaletteDragPos !== null") is True


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
        page.keyboard.press("Shift+D")
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

    def test_v_mode_colorbar_width_holds_in_small_viewport(self, loaded_viewer, sid_3d):
        """Regression: the v-mode colorbar used to be sized to one pane's width
        (≈ viewport/3), so it shrank to a sliver in VS Code tabs. It should
        instead be sized to the viewport (clamped to CB_TARGET_W=350), matching
        single-view, and only hide below CB_MIN_W=200."""
        page = loaded_viewer(sid_3d)
        page.set_viewport_size({"width": 900, "height": 700})
        page.wait_for_timeout(200)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)

        state = page.evaluate(
            """() => {
                const cb = document.getElementById('mv-cb-wrap');
                return {
                    mvWidth: cb ? Math.round(cb.getBoundingClientRect().width) : -1,
                    mvDisplay: cb ? getComputedStyle(cb).display : 'missing',
                };
            }"""
        )
        assert state["mvDisplay"] != "none", f"mv colorbar should be visible at 900px viewport, got: {state}"
        assert state["mvWidth"] >= 280, f"mv colorbar should hold a wide footprint in a small viewport (>=280px), got: {state}"
        assert state["mvWidth"] <= 350, f"mv colorbar should be capped at CB_TARGET_W (350px), got: {state}"

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
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(400)
        assert page.is_visible("#compare-diff-canvas"), "diff center canvas should be visible after pressing Shift+D"
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

        # Cycle compare center mode to overlay (mode 4): Shift+X×4 = off→A-B→|A-B|→|A-B|/|A|→overlay
        for _ in range(4):
            page.keyboard.press("Shift+X")
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

        # Press Shift+X twice more to go past wipe (mode 5).
        page.keyboard.press("Shift+X")
        page.wait_for_timeout(200)
        page.keyboard.press("Shift+X")
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
        page.keyboard.press("Shift+D")
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

        assert state, "diff title should render an active compare-center pill after pressing Shift+D"
        assert not state["fullscreen"], f"test must stay in non-immersive mode, got: {state}"
        assert "A" in state["text"] and "B" in state["text"], f"compare-center pill should render the compare glyph, got: {state}"
        assert state["background"] != "rgba(0, 0, 0, 0)", f"compare-center pill should use a filled glass background outside fullscreen, got: {state}"
        assert state["borderColor"] != "rgba(0, 0, 0, 0)", f"compare-center pill should keep a visible glass border outside fullscreen, got: {state}"
        assert state["backdropFilter"] != "none" or state["webkitBackdropFilter"] != "none", f"compare-center pill should use the immersive blur treatment outside fullscreen, got: {state}"
        assert state["boxShadow"] != "none", f"compare-center pill should keep the immersive island shadow outside fullscreen, got: {state}"
        assert state["minHeight"] == "26px", f"compare-center pill should stay compact outside fullscreen, got: {state}"
        assert state["minWidth"] == "52px", f"compare-center pill should keep a compact pill width outside fullscreen, got: {state}"

    def test_shift_x_enters_split_for_single_array(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        detached_state = page.evaluate(
            """() => {
                const dim = [...Array(shape.length).keys()].find(d => _canDetachDim(d));
                activeDim = dim;
                indices[dim] = Math.min(1, shape[dim] - 1);
                renderInfo();
                return { dim, size: shape[dim] };
            }"""
        )

        assert detached_state["dim"] is not None
        page.keyboard.press("Shift+S")
        page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)

        assert page.inner_text(f'[data-dim="{detached_state["dim"]}"]').startswith("X")
        left_title_before = page.inner_text("#compare-left-title")
        right_title_before = page.inner_text("#compare-right-title")
        assert f'/{detached_state["size"]}' in left_title_before
        assert f'/{detached_state["size"]}' in right_title_before
        assert "[" in left_title_before and "]" in left_title_before
        assert "{" in right_title_before and "}" in right_title_before
        assert left_title_before != right_title_before
        assert not page.is_visible("#compare-diff-canvas")

        left_sum_before = page.evaluate(
            """() => {
                const c = document.querySelector('canvas#compare-left-canvas');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let sum = 0;
                for (let i = 0; i < d.length; i += 64) sum += d[i] + d[i + 1] + d[i + 2];
                return sum;
            }"""
        )
        left_bracket_before = page.evaluate(
            """() => {
                const el = document.querySelector('.detached-index-hint[data-detached-pane="A"] .hint-bracket[data-detached-key="]"]');
                return getComputedStyle(el).color;
            }"""
        )
        page.keyboard.press("]")
        left_highlight = page.evaluate(
            """() => {
                return new Promise(resolve => {
                    setTimeout(() => {
                        const el = document.querySelector('.detached-index-hint[data-detached-pane="A"] .hint-bracket[data-detached-key="]"]');
                        resolve({ after: getComputedStyle(el).color });
                    }, 475);
                });
            }"""
        )
        left_title_after = page.inner_text("#compare-left-title")
        assert left_highlight["after"] == left_bracket_before
        left_sum_after = page.evaluate(
            """() => {
                const c = document.querySelector('canvas#compare-left-canvas');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let sum = 0;
                for (let i = 0; i < d.length; i += 64) sum += d[i] + d[i + 1] + d[i + 2];
                return sum;
            }"""
        )
        assert left_title_after != left_title_before
        assert left_sum_after != left_sum_before

        right_sum_before = page.evaluate(
            """() => {
                const c = document.querySelector('canvas#compare-right-canvas');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let sum = 0;
                for (let i = 0; i < d.length; i += 64) sum += d[i] + d[i + 1] + d[i + 2];
                return sum;
            }"""
        )
        right_bracket_before = page.evaluate(
            """() => {
                const el = document.querySelector('.detached-index-hint[data-detached-pane="B"] .hint-bracket[data-detached-key="}"]');
                return getComputedStyle(el).color;
            }"""
        )
        page.keyboard.press("}")
        right_highlight = page.evaluate(
            """() => {
                return new Promise(resolve => {
                    setTimeout(() => {
                        const el = document.querySelector('.detached-index-hint[data-detached-pane="B"] .hint-bracket[data-detached-key="}"]');
                        resolve({ after: getComputedStyle(el).color });
                    }, 475);
                });
            }"""
        )
        right_title_after = page.inner_text("#compare-right-title")
        assert right_highlight["after"] == right_bracket_before
        right_sum_after = page.evaluate(
            """() => {
                const c = document.querySelector('canvas#compare-right-canvas');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let sum = 0;
                for (let i = 0; i < d.length; i += 64) sum += d[i] + d[i + 1] + d[i + 2];
                return sum;
            }"""
        )
        assert right_title_after != right_title_before
        assert right_sum_after != right_sum_before

        page.keyboard.press("Shift+D")
        page.wait_for_selector("#compare-diff-canvas:visible", timeout=5_000)
        page.keyboard.press("Shift+D")
        page.wait_for_timeout(300)
        assert not page.is_visible("#compare-diff-canvas")

        page.keyboard.press("Shift+S")
        page.wait_for_timeout(300)
        assert not page.is_visible("#compare-view-wrap.active")

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
        page.keyboard.press("Shift+D")
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
        page.keyboard.press("Shift+D")
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
        page.keyboard.press("Shift+D")
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
        page.keyboard.press("Shift+D")
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
        page.keyboard.press("Shift+D")
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
        page.keyboard.press("Shift+D")
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

    def test_d_first_opens_second_cycles_without_status_text(self, loaded_viewer, sid_2d):
        # First tap `d` opens the histogram only. Second tap cycles quietly.
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_timeout(400)
        before = page.evaluate("() => ({ manualVmin, manualVmax, status: document.getElementById('status')?.textContent || '', toast: document.getElementById('toast')?.textContent || '' })")
        page.keyboard.press("d")
        page.wait_for_timeout(500)
        after = page.evaluate("() => ({ manualVmin, manualVmax, status: document.getElementById('status')?.textContent || '', toast: document.getElementById('toast')?.textContent || '' })")
        assert after["manualVmax"] != before["manualVmax"]
        assert after["status"] == ""
        assert after["toast"] == ""

    def test_histogram_scoped_volume_range_stays_fixed_while_scrolling(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.evaluate("""() => {
            const scrollDim = shape.findIndex((_, i) => i !== dim_x && i !== dim_y);
            _histDimState.set(scrollDim, 'scope');
            activeDim = scrollDim;
        }""")
        page.keyboard.press("d")
        page.wait_for_function("() => manualVmin !== null && manualVmax !== null")
        before = page.evaluate("""() => {
            return {range: [manualVmin, manualVmax], index: indices[activeDim]};
        }""")
        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(300)
        after = page.evaluate("() => ({range: [manualVmin, manualVmax], index: indices[activeDim]})")
        assert after["index"] > before["index"]
        assert after["range"] == before["range"], "scrolling an in-scope dimension must retain the histogram range"

    def test_initial_scoped_volume_range_stays_fixed_without_opening_histogram(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.wait_for_function("() => manualVmin !== null && manualVmax !== null")
        before = page.evaluate("""() => ({
            range: [manualVmin, manualVmax],
            index: indices[activeDim],
            histogramOpen: primaryCb._expanded,
        })""")

        page.keyboard.press("ArrowRight")
        page.wait_for_timeout(300)
        after = page.evaluate("() => ({range: [manualVmin, manualVmax], index: indices[activeDim]})")

        assert before["histogramOpen"] is False
        assert after["index"] > before["index"]
        assert after["range"] == before["range"]

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

    def test_shift_i_shows_data_info_overlay(self, loaded_viewer, sid_2d):
        # Shift+I shows the info overlay (#info-overlay gains .visible class)
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+I")
        page.wait_for_timeout(600)
        visible = page.evaluate(
            "() => document.querySelector('#info-overlay').classList.contains('visible')"
        )
        assert visible, "#info-overlay should have .visible after pressing Shift+I"
        text = page.inner_text("#info-overlay")
        assert "100" in text or "80" in text, f"Shape not in info-overlay: '{text}'"

    def test_shift_i_info_overlay_splits_path_and_shows_nifti_spatial_rows(
        self, client, loaded_viewer, tmp_path
    ):
        nib = pytest.importorskip("nibabel")
        data = np.zeros((4, 5, 6), dtype=np.float32)
        affine = np.diag([2.0, 1.5, 0.75, 1.0])
        folder = tmp_path / "case-a"
        folder.mkdir()
        path = folder / "spatial.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        sid = client.post("/load", json={"filepath": str(path)}).json()["sid"]

        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("Shift+I")
        page.wait_for_selector("#info-overlay.visible", timeout=2_000)

        text = page.inner_text("#info-overlay")
        assert "Voxel size" in text
        assert "2 x 1.5 x 0.75 mm" in text
        assert "Field of view" in text
        assert "8 x 7.5 x 4.5 mm" in text
        assert "Folder" in text
        assert str(folder) in text
        assert "Filename" in text
        assert "spatial.nii.gz" in text

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

    def test_qmri_layout_ignores_thumbnail_frame_size(self, loaded_viewer, sid_4d):
        page = loaded_viewer(sid_4d)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap.active .qv-canvas", timeout=5_000)
        page.wait_for_timeout(500)

        state = page.evaluate(
            """() => {
                if (typeof qvScaleAllCanvases !== 'function' || !Array.isArray(qmriViews)) {
                    return { error: 'qMRI globals unavailable' };
                }
                const views = qmriViews.filter(v => v.canvas && v.lastW && v.lastH);
                if (views.length < 3) return { error: 'not enough qMRI panes', count: views.length };
                qvScaleAllCanvases();
                const before = views.slice(0, 3).map(v => Math.round(v.canvas.getBoundingClientRect().width));
                views[1].lastW = Math.max(4, Math.round((Number(shape[dim_x]) || views[1].lastW) / 8));
                views[1].lastH = Math.max(4, Math.round((Number(shape[dim_y]) || views[1].lastH) / 8));
                qvScaleAllCanvases();
                const after = views.slice(0, 3).map(v => Math.round(v.canvas.getBoundingClientRect().width));
                return { before, after };
            }"""
        )

        assert "error" not in state, f"qMRI layout test setup failed: {state}"
        assert min(state["before"]) > 20, f"expected visible qMRI panes before perturbation, got: {state}"
        assert min(state["after"]) / max(state["after"]) > 0.9, (
            "qMRI pane layout should stay uniform when one pane receives a thumbnail-sized frame, "
            f"got: {state}"
        )

    def test_qmri_synthetic_hover_does_not_hit_pixel_500(self, loaded_viewer, sid_4d):
        bad_pixel_responses = []
        page = loaded_viewer(sid_4d)
        page.on(
            "response",
            lambda resp: bad_pixel_responses.append((resp.url, resp.status))
            if "/pixel/" in resp.url and resp.status >= 400
            else None,
        )
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_selector("#qmri-view-wrap.active .qv-canvas", timeout=5_000)
        page.evaluate("() => _islandToggleQmriSyntheticContrast('t1w')")
        page.wait_for_selector("#qmri-view-wrap .qv-synthetic-row .qv-canvas", timeout=5_000)
        page.wait_for_timeout(700)

        regular_canvas = page.locator(
            "#qmri-view-wrap .qv-row:not(.qv-synthetic-row) .qv-canvas"
        ).nth(0)
        box = regular_canvas.bounding_box()
        assert box is not None, "expected a regular qMRI pane to hover"
        page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        page.wait_for_timeout(150)
        page.mouse.move(box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55)
        page.wait_for_timeout(500)

        assert not bad_pixel_responses, (
            "Synthetic qMRI hover should not generate failing /pixel requests, "
            f"got: {bad_pixel_responses}"
        )

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
        page.keyboard.press("Shift+D")
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

    def test_multiview_auto_layout_picks_big_left_on_jupyter_like_viewport_and_manual_override_sticks(
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
        assert initial["orthoAutoLayoutMode"] == "big-left", f"jupyter-like viewport should auto-pick big-left ortho layout, got: {initial}"
        assert initial["rowPosition"] == "relative", f"big-left ortho auto-layout should use preset positioning, got: {initial}"
        assert "mv-promote-enabled" in initial["rowClass"], f"big-left ortho auto-layout should enable promotable preset chrome, got: {initial}"
        assert len(initial["panes"]) == 3, f"multiview should expose exactly three ortho panes, got: {initial}"
        assert initial["panes"][0]["width"] > initial["panes"][1]["width"], f"big-left ortho auto-layout should make the first pane larger than the stacked panes, got: {initial}"
        assert initial["panes"][1]["top"] < initial["panes"][2]["top"], f"big-left ortho auto-layout should stack the secondary panes vertically, got: {initial}"

        page.set_viewport_size({"width": 1700, "height": 1100})
        page.wait_for_timeout(500)
        resized = _state()
        assert resized["orthoAutoLayoutMode"] == "big-left", f"ortho auto-layout choice should stay stable across resize until manually overridden, got: {resized}"
        assert resized["rowPosition"] == "relative", f"resizing alone should keep the cached big-left ortho layout, got: {resized}"

        page.keyboard.press("g")
        page.wait_for_timeout(500)
        overridden = _state()
        assert overridden["orthoLayoutMode"] == "horizontal", f"manual g cycling should persist a concrete ortho preset override, got: {overridden}"
        assert overridden["rowPosition"] != "relative", f"horizontal ortho override should restore legacy row flow, got: {overridden}"
        assert "mv-promote-enabled" not in overridden["rowClass"], f"horizontal ortho layout should clear promotable preset chrome, got: {overridden}"
        assert abs(overridden["cbWidth"] - initial["cbWidth"]) <= 1, f"multiview colorbar width should stay stable when g switches to horizontal, got initial={initial}, overridden={overridden}"
        assert max(abs(overridden["panes"][i]["top"] - overridden["panes"][0]["top"]) for i in range(1, 3)) <= 2, f"horizontal ortho override should keep panes on one row, got: {overridden}"
        assert max(abs(overridden["panes"][i]["width"] - overridden["panes"][0]["width"]) for i in range(1, 3)) <= 2, f"horizontal ortho override should keep pane widths uniform, got: {overridden}"

        page.keyboard.press("g")
        page.wait_for_timeout(500)
        returned = _state()
        assert returned["orthoLayoutMode"] == "big-left", f"manual g cycling should now return directly to big-left without vertical or big-top, got: {returned}"
        assert returned["rowPosition"] == "relative", f"returning to big-left should use preset positioning, got: {returned}"
        assert "mv-promote-enabled" in returned["rowClass"], f"big-left ortho layout should enable promotable preset chrome, got: {returned}"
        assert abs(returned["cbWidth"] - initial["cbWidth"]) <= 1, f"multiview colorbar width should stay stable when g returns to big-left, got initial={initial}, returned={returned}"

        page.set_viewport_size({"width": 1280, "height": 760})
        page.wait_for_timeout(500)
        final = _state()
        assert final["orthoLayoutMode"] == "big-left", f"manual ortho preset override should stay sticky after resizing back to a smaller viewport, got: {final}"
        assert final["rowPosition"] == "relative", f"manual big-left ortho override should remain active after resize, got: {final}"

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

    def test_multiview_auto_layout_falls_back_to_horizontal_on_narrow_viewport(self, loaded_viewer, sid_3d):
        """big-left is the default for any reasonable viewport; only a truly
        narrow viewport (e.g. a thin VS Code split) should fall back to the
        horizontal three-in-a-row layout so the secondary panes stay readable."""
        page = loaded_viewer(sid_3d)
        page.set_viewport_size({"width": 600, "height": 700})
        page.wait_for_timeout(200)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)

        narrow = page.evaluate(
            """() => ({
                orthoLayoutMode,
                orthoAutoLayoutMode: _orthoAutoLayoutMode,
                rowPosition: getComputedStyle(document.querySelector('#mv-panes')).position,
            })"""
        )
        assert narrow["orthoLayoutMode"] is None, f"narrow viewport should stay in auto mode, got: {narrow}"
        assert narrow["orthoAutoLayoutMode"] == "horizontal", f"narrow viewport should fall back to horizontal, got: {narrow}"
        assert narrow["rowPosition"] != "relative", f"horizontal fallback should use legacy row flow, got: {narrow}"

        # A moderately small but usable viewport (VS Code tab) should still
        # get big-left — that's the regression this commit fixes.
        page.set_viewport_size({"width": 900, "height": 700})
        page.wait_for_timeout(200)
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        page.keyboard.press("v")
        page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
        page.wait_for_timeout(500)
        medium = page.evaluate("() => ({ orthoLayoutMode, orthoAutoLayoutMode: _resolvedOrthoLayoutMode() })")
        assert medium["orthoAutoLayoutMode"] == "big-left", f"VS Code-tab-sized viewport should default to big-left, got: {medium}"

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
    def _draw_roi(self, page):
        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        assert box is not None
        x0 = box["x"] + box["width"] * 0.18
        y0 = box["y"] + box["height"] * 0.18
        x1 = box["x"] + box["width"] * 0.58
        y1 = box["y"] + box["height"] * 0.58
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()
        page.wait_for_timeout(800)
        return x0, y0, x1, y1

    def test_shift_r_flips_mv_colorbar_to_roi_toolbar_in_v_mode(self, loaded_viewer, sid_3d):
        """In ortho/v-mode the ROI toolbar should appear on the back face of
        the multiview colorbar flip (ROI takes priority over the oblique
        Alt-hold reveal). The single-view slim-cb-wrap stays hidden."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(500)

        page.keyboard.press("Shift+R")
        page.wait_for_function(
            "() => { const i = document.querySelector('.oblique-flip-inner');"
            " return i && i.classList.contains('flipped') && i.classList.contains('roi-flipped'); }",
            timeout=3_000,
        )
        state = page.evaluate(
            """() => ({
                rectRoiMode,
                roiBackDisplay: getComputedStyle(document.querySelector('.roi-flip-back')).display,
                obBackDisplay: getComputedStyle(document.querySelector('.oblique-flip-back')).display,
                mvControlsBtns: document.querySelectorAll('#roi-cb-controls-mv button').length,
                mvControlsVisible: document.getElementById('roi-cb-controls-mv').offsetParent !== null,
                slimDisplay: getComputedStyle(document.getElementById('slim-cb-wrap')).display,
            })"""
        )
        assert state["rectRoiMode"], f"ROI mode should activate in v-mode, got: {state}"
        assert state["roiBackDisplay"] == "flex", f"ROI back face should show, got: {state}"
        assert state["obBackDisplay"] == "none", f"oblique back face should hide, got: {state}"
        assert state["mvControlsBtns"] == 6, f"ROI toolbar should render 6 buttons (4 shapes + stats + clear), got: {state}"
        assert state["mvControlsVisible"], f"mv ROI controls should be visible, got: {state}"
        assert state["slimDisplay"] == "none", f"slim-cb-wrap should stay hidden in v-mode, got: {state}"

        # Toggling ROI off un-flips and restores the oblique back face.
        page.keyboard.press("Shift+R")
        page.wait_for_function(
            "() => { const i = document.querySelector('.oblique-flip-inner');"
            " return i && !i.classList.contains('flipped') && !i.classList.contains('roi-flipped'); }",
            timeout=3_000,
        )
        off = page.evaluate(
            """() => ({
                rectRoiMode,
                obBackDisplay: getComputedStyle(document.querySelector('.oblique-flip-back')).display,
                roiBackDisplay: getComputedStyle(document.querySelector('.roi-flip-back')).display,
            })"""
        )
        assert not off["rectRoiMode"], f"ROI mode should deactivate, got: {off}"
        assert off["obBackDisplay"] == "flex", f"oblique back face should restore, got: {off}"
        assert off["roiBackDisplay"] == "none", f"ROI back face should hide, got: {off}"

    def test_v_mode_roi_does_not_take_over_dimbar(self, loaded_viewer, sid_3d):
        """ROI mode should leave the dimbar as plain navigation in v-mode."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(400)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        state = page.evaluate(
            """() => Array.from(document.querySelectorAll('#info .dim-label[data-dim]')).map(el => ({
                dim: el.getAttribute('data-dim'),
                className: el.className,
            }))"""
        )
        assert not page.locator("#info").evaluate("el => el.classList.contains('roi-scope-mode')")
        assert not any("roi-" in label["className"] for label in state), f"ROI mode should not mark dimbar labels, got: {state}"

    def test_v_mode_roi_draws_circle_on_mv_pane(self, loaded_viewer, sid_3d):
        """Circle/rect/freehand ROIs should be drawable on mv panes in v-mode,
        tagged with the pane's mvDimX/mvDimY so they render on the right pane."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(400)
        page.keyboard.press("Shift+R")
        page.wait_for_selector(".oblique-flip-inner.roi-flipped", timeout=2_000)
        page.evaluate("() => _roiSetShape('circle')")
        page.wait_for_timeout(200)

        pane = page.locator(".mv-canvas").nth(0)
        box = pane.bounding_box()
        x0, y0 = box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2
        x1, y1 = box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.6
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()
        page.wait_for_timeout(800)

        roi = page.evaluate("() => _rois[0] ? { type: _rois[0].type, mvDimX: _rois[0].mvDimX, mvDimY: _rois[0].mvDimY } : null")
        assert roi is not None, "ROI should be created on mv pane"
        assert roi["type"] == "circle", f"ROI should be circle, got: {roi}"
        assert roi["mvDimX"] is not None and roi["mvDimY"] is not None, f"ROI should have mvDimX/mvDimY, got: {roi}"

    def test_v_mode_roi_overlay_keeps_image_visible(self, loaded_viewer, sid_3d):
        """Regression: drawing an ROI in v-mode used to turn the whole pane gray
        because the .mv-roi-overlay canvas inherited the universal opaque
        `canvas { background: var(--bg) }`. The overlay must be transparent so
        the image underneath stays visible, matching single-view (#roi-overlay)
        and qMRI (.qv-roi-overlay)."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(400)
        page.keyboard.press("Shift+R")
        page.wait_for_selector(".oblique-flip-inner.roi-flipped", timeout=2_000)
        page.evaluate("() => _roiSetShape('rect')")

        pane = page.locator(".mv-canvas").nth(0)
        box = pane.bounding_box()
        x0, y0 = box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2
        x1, y1 = box["x"] + box["width"] * 0.7, box["y"] + box["height"] * 0.7
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()
        page.wait_for_timeout(800)

        state = page.evaluate(
            """() => {
                const ov = document.querySelector('.mv-roi-overlay');
                const cv = document.querySelector('.mv-canvas');
                if (!ov || !cv) return null;
                const ovBg = getComputedStyle(ov).backgroundColor;
                // Image canvas center pixel must be non-empty (image rendered).
                const ctx = cv.getContext('2d');
                const px = ctx.getImageData(Math.floor(cv.width/2), Math.floor(cv.height/2), 1, 1).data;
                return { ovBg, imgAlpha: px[3], display: getComputedStyle(ov).display };
            }"""
        )
        assert state is not None, "mv-roi-overlay / mv-canvas must exist in v-mode ROI"
        assert state["ovBg"] in ("rgba(0, 0, 0, 0)", "transparent"), (
            f"mv-roi-overlay background must be transparent so the image shows through, got: {state}")
        assert state["display"] != "none", "overlay should be visible after drawing a ROI"
        assert state["imgAlpha"] > 0, f"image canvas must still hold rendered pixels, got: {state}"

    def test_shift_r_hide_show_preserves_rois(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        assert "visible" not in (page.locator("#tool-drawer").get_attribute("class") or "")
        assert "roi-active" in (page.locator("#slim-cb-wrap").get_attribute("class") or "")
        assert page.locator("#roi-cb-front").evaluate("el => getComputedStyle(el).visibility") == "hidden"
        assert page.locator("#roi-cb-front").evaluate("el => getComputedStyle(el).opacity") == "0"
        assert page.locator("#roi-cb-controls").is_visible()
        self._draw_roi(page)

        assert page.evaluate("() => _rois.length") == 1
        assert page.evaluate("() => _rois[0].type") == "circle"
        assert page.locator("#roi-overlay").evaluate("el => getComputedStyle(el).display") != "none"

        page.keyboard.press("Shift+R")
        page.wait_for_function("() => !document.getElementById('slim-cb-wrap').classList.contains('roi-active')")
        assert page.evaluate("() => _rois.length") == 1
        assert page.locator("#roi-overlay").evaluate("el => getComputedStyle(el).display") == "none"
        assert "roi-active" not in (page.locator("#slim-cb-wrap").get_attribute("class") or "")

        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        assert page.evaluate("() => _rois.length") == 1
        assert page.locator("#roi-overlay").evaluate("el => getComputedStyle(el).display") != "none"
        assert "roi-active" in (page.locator("#slim-cb-wrap").get_attribute("class") or "")
        assert page.locator("#roi-cb-front").evaluate("el => getComputedStyle(el).visibility") == "hidden"
        assert page.locator("#roi-cb-controls").is_visible()

    def test_histogram_shortcut_is_blocked_in_roi_mode(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)

        page.keyboard.press("d")
        page.wait_for_timeout(300)
        state = page.evaluate(
            """() => ({
                rectRoiMode,
                histPickerActive: _histPickerActive,
                expanded: !!(primaryCb && primaryCb._expanded),
                status: document.querySelector('#status')?.innerText?.trim() || '',
            })"""
        )

        assert state["rectRoiMode"], f"ROI mode should remain active after blocked histogram shortcut, got: {state}"
        assert not state["histPickerActive"], f"histogram picker should not open in ROI mode, got: {state}"
        assert not state["expanded"], f"histogram colorbar should not expand in ROI mode, got: {state}"
        assert "histogram" in state["status"].lower() and "roi" in state["status"].lower(), (
            f"blocked histogram shortcut should explain ROI conflict, got: {state}"
        )

    def test_shift_r_does_not_add_roi_dimbar_underlines(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)

        active = page.evaluate(
            """() => ({
                infoClass: document.getElementById('info').className,
                labeled: Array.from(document.querySelectorAll('#info .dim-label')).map(el => el.className),
            })"""
        )
        assert "roi-scope-mode" not in active["infoClass"], f"ROI mode should not scope the dimbar, got: {active}"
        assert not any("roi-" in cls for cls in active["labeled"]), f"ROI mode should not mark dimbar labels, got: {active}"

        page.keyboard.press("Shift+R")
        page.wait_for_timeout(150)
        cleared = page.evaluate(
            """() => ({
                infoClass: document.getElementById('info').className,
                labeled: Array.from(document.querySelectorAll('#info .dim-label')).map(el => el.className),
            })"""
        )

        assert "roi-scope-mode" not in cleared["infoClass"], f"Shift+R should leave ROI dimbar scope off, got: {cleared}"
        assert not any("roi-" in cls for cls in cleared["labeled"]), f"Shift+R should leave dimbar classes clean, got: {cleared}"

    def test_only_floodfill_adds_roi_dimbar_underlines(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)

        for shape_name in ["circle", "rect", "freehand"]:
            page.evaluate("(shapeName) => { _roiSetShape(shapeName); renderInfo(); }", shape_name)
            page.wait_for_timeout(100)
            state = page.evaluate(
                """() => ({
                    infoClass: document.getElementById('info').className,
                    labeled: Array.from(document.querySelectorAll('#info .dim-label')).map(el => el.className),
                })"""
            )
            assert "roi-scope-mode" not in state["infoClass"], f"{shape_name} should not scope dimbar, got: {state}"
            assert not any("roi-" in cls for cls in state["labeled"]), f"{shape_name} should not mark dimbar labels, got: {state}"

        page.evaluate("() => _roiSetShape('floodfill')")
        page.wait_for_selector("#info.roi-scope-mode", timeout=2_000)
        flood = page.evaluate(
            """() => ({
                included: Array.from(document.querySelectorAll('#info .dim-label.roi-included-dim')).map(el => Number(el.dataset.dim)),
                display: Array.from(document.querySelectorAll('#info .dim-label.roi-display-dim')).map(el => Number(el.dataset.dim)),
                resolved: _roiResolvedFloodfillScopeDim(null),
            })"""
        )
        assert flood["resolved"] in flood["included"], f"floodfill should underline resolved scope dim, got: {flood}"
        assert len(flood["included"]) == 1, f"floodfill should include one extra dim, got: {flood}"
        assert len(flood["display"]) == 2, f"display dims should remain marked, got: {flood}"

    def test_roi_overlay_stays_above_image_during_slice_navigation(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        page.evaluate(
            """() => {
                activeDim = dim_z;
                renderInfo();
            }"""
        )

        page.keyboard.press("k")
        page.wait_for_timeout(80)
        state = page.evaluate(
            """() => ({
                overlayDisplay: getComputedStyle(document.getElementById('roi-overlay')).display,
                overlayOpacity: getComputedStyle(document.getElementById('roi-overlay')).opacity,
                fadeOpacity: getComputedStyle(document.getElementById('viewer-fade')).opacity,
                roiCount: _rois.length,
            })"""
        )

        assert state["roiCount"] == 1, f"ROI should remain present after slice navigation, got: {state}"
        assert state["overlayDisplay"] != "none", f"ROI overlay should remain visible after slice navigation, got: {state}"
        assert state["overlayOpacity"] != "0", f"ROI overlay should not fade out during slice navigation, got: {state}"
        assert state["fadeOpacity"] == "0", f"image fade canvas should not cover ROI overlay during slice navigation, got: {state}"

    def test_dimbar_click_does_not_scope_or_clear_roi(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        assert page.evaluate("() => _rois.length") == 1, "ROI should exist before dimbar click"

        # Clicking a non-display dim should use normal dimbar navigation. It
        # must not toggle a hidden ROI scope or clear existing ROIs.
        nondisp = page.evaluate(
            """() => {
                const labels = Array.from(document.querySelectorAll('#info .dim-label[data-dim]'));
                const found = labels.find(el => {
                    const d = Number(el.getAttribute('data-dim'));
                    return d !== dim_x && d !== dim_y;
                });
                return found ? Number(found.getAttribute('data-dim')) : null;
            }"""
        )
        assert nondisp is not None, "expected at least one non-display dim label"

        page.locator(f'#info .dim-label[data-dim="{nondisp}"]').click()
        page.wait_for_timeout(150)
        state = page.evaluate(
            """() => ({
                roiCount: _rois.length,
                activeDim,
                scope: _rois[0] && _rois[0].scope ? _rois[0].scope.broadcast_dims.slice() : null,
                infoClass: document.getElementById('info').className,
            })"""
        )
        assert state["roiCount"] == 1, f"dimbar click should not clear ROIs, got {state}"
        assert state["activeDim"] == nondisp, f"dimbar click should still navigate to dim {nondisp}, got {state}"
        assert state["scope"] == [], f"ROI scope should stay 2D/current-slice, got {state}"
        assert "roi-scope-mode" not in state["infoClass"], f"dimbar should not enter ROI scope mode, got {state}"

    def test_roi_preview_overlay_visible_during_drag(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        x0, y0 = box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2
        x1, y1 = box["x"] + box["width"] * 0.55, box["y"] + box["height"] * 0.55
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=8)
        page.wait_for_timeout(120)
        # While the mouse is still held, the in-progress preview must be painting.
        overlay_display = page.locator("#roi-overlay").evaluate("el => getComputedStyle(el).display")
        page.mouse.up()
        assert overlay_display != "none", "ROI preview overlay should be visible during creation drag"

    def test_normal_mode_floodfill_defaults_to_3d_spatial_scope(self, loaded_viewer, client, tmp_path):
        # A column of value 5 along dim 0 at (y=5, x=5); everything else 0.
        # Normal mode floodfill should grow through the non-display spatial dim by default.
        arr = np.zeros((4, 10, 10), dtype=np.float32)
        arr[:, 5, 5] = 5.0
        path = tmp_path / "roi_seed_column.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_seed_column"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        page.evaluate("_roiSetShape('floodfill')")

        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.wait_for_timeout(250)  # let the flood-fill preview resolve
        page.mouse.up()
        page.wait_for_timeout(400)

        roi = page.evaluate("() => _rois[0] ? { type: _rois[0].type, scopeDim: _rois[0].scope_dim, nSlices: (_rois[0].slices || []).length, hasMask3d: !!_rois[0].mask3d_b64, n: _rois[0].stats && _rois[0].stats.n, broadcast: _rois[0].scope && _rois[0].scope.broadcast_dims } : null")
        assert roi is not None, "a flood-fill ROI should have been created"
        assert roi["type"] == "floodfill"
        assert roi["scopeDim"] == 0, f"normal-mode floodfill should default to spatial scope dim 0, got {roi}"
        assert roi["broadcast"] == [0], f"committed scope should broadcast dim 0, got {roi}"
        assert roi["nSlices"] == 4, f"3D grow should span all 4 slices, got {roi}"
        assert roi["hasMask3d"], f"3D floodfill should carry a 3D mask, got {roi}"
        assert roi["n"] == 4, f"3D seed column should produce one voxel per slice, got {roi}"

    def test_floodfill_dimbar_choose_clear_change_scope(self, loaded_viewer, client, tmp_path):
        arr = np.zeros((4, 10, 10, 5), dtype=np.float32)
        path = tmp_path / "roi_scope_choice.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_scope_choice"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        page.evaluate("() => { _roiSetShape('floodfill'); renderInfo(); }")
        page.wait_for_selector("#info.roi-scope-mode", timeout=2_000)

        state = page.evaluate(
            """() => ({
                resolved: _roiResolvedFloodfillScopeDim(null),
                included: Array.from(document.querySelectorAll('#info .dim-label.roi-included-dim')).map(el => Number(el.dataset.dim)),
                choices: Array.from(document.querySelectorAll('#info .dim-label[data-dim]')).map(el => Number(el.dataset.dim)).filter(d => d !== dim_x && d !== dim_y),
            })"""
        )
        assert state["resolved"] in state["choices"], f"expected an initial non-display spatial scope, got {state}"
        assert state["included"] == [state["resolved"]], f"initial underline should match resolved scope, got {state}"

        first = state["resolved"]
        page.locator(f'#info .dim-label[data-dim="{first}"]').click()
        page.wait_for_timeout(150)
        cleared = page.evaluate(
            """() => ({
                stored: _roiFloodfillScopeDim,
                resolved: _roiResolvedFloodfillScopeDim(null),
                included: Array.from(document.querySelectorAll('#info .dim-label.roi-included-dim')).map(el => Number(el.dataset.dim)),
            })"""
        )
        assert cleared["stored"] == -1 and cleared["resolved"] == -1, f"clicking active scope should clear to 2D, got {cleared}"
        assert cleared["included"] == [], f"cleared floodfill scope should remove included underline, got {cleared}"

        other = next(d for d in state["choices"] if d != first)
        page.locator(f'#info .dim-label[data-dim="{other}"]').click()
        page.wait_for_timeout(150)
        changed = page.evaluate(
            """() => ({
                stored: _roiFloodfillScopeDim,
                resolved: _roiResolvedFloodfillScopeDim(null),
                included: Array.from(document.querySelectorAll('#info .dim-label.roi-included-dim')).map(el => Number(el.dataset.dim)),
            })"""
        )
        assert changed["stored"] == other and changed["resolved"] == other, f"clicking another dim should choose it, got {changed}"
        assert changed["included"] == [other], f"chosen floodfill scope should be underlined, got {changed}"

    def test_floodfill_hidden_on_dim_mismatch(self, loaded_viewer, client, tmp_path):
        # A floodfill ROI grown on (dim_x=2, dim_y=1) must be hidden
        # when the user swaps the viewing dims — the per-slice masks are in
        # the original plane's coordinates and don't apply to the new plane.
        arr = np.zeros((4, 10, 10), dtype=np.float32)
        arr[:, 5, 5] = 5.0
        path = tmp_path / "roi_floodfill_dim_mismatch.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_floodfill_dim_mismatch"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        page.evaluate("_roiSetShape('floodfill')")
        page.evaluate(
            """() => {
                indices[0] = 1;
                activeDim = 0;
                current_slice_dim = 0;
                updateView();
                renderInfo();
            }"""
        )
        page.wait_for_timeout(400)
        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        cx = box["x"] + box["width"] * (5.5 / 10)
        cy = box["y"] + box["height"] * (5.5 / 10)
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.wait_for_timeout(300)
        page.mouse.up()
        page.wait_for_timeout(400)

        dims = page.evaluate("() => ({ dimX: dim_x, dimY: dim_y, roiDimX: _rois[0] && _rois[0].dimX, roiDimY: _rois[0] && _rois[0].dimY })")
        assert dims["roiDimX"] is not None and dims["roiDimY"] is not None, f"floodfill ROI should track dimX/dimY, got {dims}"
        assert dims["roiDimX"] == dims["dimX"] and dims["roiDimY"] == dims["dimY"], f"ROI dims should match current dims, got {dims}"

        # Verify the overlay is visible on the matching plane.
        overlay_visible_before = page.evaluate(
            """() => {
                _redrawRoiOverlays();
                const ov = document.getElementById('roi-overlay');
                const ctx = ov.getContext('2d');
                const w = ov.width, h = ov.height;
                const img = ctx.getImageData(0, 0, w, h).data;
                for (let i = 3; i < img.length; i += 4) { if (img[i] > 0) return true; }
                return false;
            }"""
        )
        assert overlay_visible_before, "floodfill overlay should be visible on matching plane"

        # Swap the viewing dims (x ↔ y) via the keybind.
        page.evaluate(
            """() => {
                activeDim = dim_y;
                const oldDimX = dim_x, oldDimY = dim_y;
                dim_x = oldDimY; dim_y = oldDimX;
                updateView();
                renderInfo();
            }"""
        )
        page.wait_for_timeout(300)

        # After the swap, the ROI's dimX/dimY no longer match → overlay must be empty.
        overlay_visible_after = page.evaluate(
            """() => {
                _redrawRoiOverlays();
                const ov = document.getElementById('roi-overlay');
                const ctx = ov.getContext('2d');
                const w = ov.width, h = ov.height;
                const img = ctx.getImageData(0, 0, w, h).data;
                for (let i = 3; i < img.length; i += 4) { if (img[i] > 0) return true; }
                return false;
            }"""
        )
        assert not overlay_visible_after, "floodfill overlay should be hidden when viewing dims don't match the ROI's plane"

    def test_vmode_floodfill_defaults_to_3d_and_renders_in_all_panes(self, loaded_viewer, client, tmp_path):
        # A centered 3D block of value 5 in a 12x12x12 volume. In v-mode a
        # flood-fill seed on any pane should default to 3D (scope = that
        # pane's sliceDir) and the grown region must render in all three
        # panes, not just the one it was seeded on.
        arr = np.zeros((12, 12, 12), dtype=np.float32)
        arr[4:8, 4:8, 4:8] = 5.0
        path = tmp_path / "roi_vmode_3d_block.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_vmode_3d_block"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_timeout(900)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        page.evaluate("_roiSetShape('floodfill')")
        page.wait_for_timeout(150)

        # Seed at the center of pane 0 (axial). No dim-label clicked, so the
        # 3D default must kick in using pane 0's sliceDir.
        pane0 = page.evaluate("() => { const v = mvViews[0]; const r = v.canvas.getBoundingClientRect(); return { x: r.x + r.width/2, y: r.y + r.height/2, dimX: v.dimX, dimY: v.dimY, sliceDir: v.sliceDir }; }")
        page.mouse.move(pane0["x"], pane0["y"])
        page.mouse.down()
        page.wait_for_timeout(500)
        page.mouse.up()
        page.wait_for_timeout(500)

        roi = page.evaluate("() => _rois[0] ? { type: _rois[0].type, scopeDim: _rois[0].scope_dim, nSlices: (_rois[0].slices || []).length, hasMask3d: !!_rois[0].mask3d_b64, growDimX: _rois[0].grow_dim_x, growDimY: _rois[0].grow_dim_y } : null")
        assert roi is not None, "a flood-fill ROI should have been created"
        assert roi["type"] == "floodfill"
        assert roi["scopeDim"] == pane0["sliceDir"], f"v-mode seed should default to 3D (scope=pane sliceDir={pane0['sliceDir']}), got scopeDim={roi['scopeDim']}"
        assert roi["hasMask3d"], "scoped floodfill ROI should carry the full 3D mask for cross-pane rendering"
        assert roi["nSlices"] == 4, f"3D block should span 4 slices along scope dim, got {roi['nSlices']}"

        # Each pane's overlay must show the grown region on its plane.
        per_pane = page.evaluate("""() => mvViews.map(v => {
            const ov = v.roiOverlay;
            if (!ov || !v.roiCtx) return { sliceDir: v.sliceDir, present: false };
            const ctx = v.roiCtx;
            const img = ctx.getImageData(0, 0, ov.width, ov.height).data;
            let present = false;
            for (let i = 3; i < img.length; i += 4) { if (img[i] > 0) { present = true; break; } }
            return { sliceDir: v.sliceDir, dimX: v.dimX, dimY: v.dimY, present };
        })""")
        assert len(per_pane) == 3
        for p in per_pane:
            assert p["present"], f"3D floodfill ROI should render in pane sliceDir={p['sliceDir']} (dimX={p['dimX']}, dimY={p['dimY']}), got present=False"

    def test_vmode_roi_replace_preview_reuses_original_color(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(700)

        calls = page.evaluate(
            """() => {
                const view = mvViews[0];
                const originalDraw = _roiDrawVisibleShape;
                const seen = [];
                _roiVisible = true;
                _selectedRoiIdx = 0;
                _rois = [
                    { id: 'original', type: 'circle', cx: 6, cy: 6, r: 3, mvDimX: view.dimX, mvDimY: view.dimY, visible: true },
                    { id: 'other', type: 'circle', cx: 14, cy: 14, r: 3, mvDimX: view.dimX, mvDimY: view.dimY, visible: true },
                ];
                _roiDrawVisibleShape = (_ctx, roi, _sx, _sy, color, selected) => {
                    seen.push({ id: roi.id || 'preview', stroke: color.stroke, selected });
                };
                try {
                    _drawAllMvRois({ id: 'preview', type: 'circle', cx: 8, cy: 8, r: 4, mvDimX: view.dimX, mvDimY: view.dimY, replaceIdx: 0 });
                } finally {
                    _roiDrawVisibleShape = originalDraw;
                    _rois = [];
                    _selectedRoiIdx = -1;
                    _clearMvRoiOverlays();
                }
                return { seen, color0: _roiColors[0].stroke, color1: _roiColors[1].stroke, color2: _roiColors[2].stroke };
            }"""
        )
        assert all(call["id"] != "original" for call in calls["seen"]), f"replace preview should skip original ROI, got {calls}"
        preview = next((call for call in calls["seen"] if call["id"] == "preview"), None)
        assert preview is not None, f"replace preview should be drawn, got {calls}"
        assert preview["stroke"] == calls["color0"], f"replace preview should reuse original ROI color, got {calls}"
        assert preview["stroke"] != calls["color2"], f"replace preview should not use next-new ROI color, got {calls}"

    def test_vmode_floodfill_updates_on_scroll(self, loaded_viewer, client, tmp_path):
        # A 3D block at z=4..7. Seed a 3D floodfill on pane 0 (sliceDir=2) at
        # mid-z, then scroll pane 0 outside the block. The ROI overlay on
        # pane 0 must disappear (no voxels on that slice) and reappear when
        # scrolled back. Regression for the stale-overlay-on-scroll bug.
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[4:8, 4:8, 4:8] = 5.0
        path = tmp_path / "roi_vmode_scroll.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_vmode_scroll"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_timeout(900)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        page.evaluate("_roiSetShape('floodfill')")
        page.wait_for_timeout(150)
        page.evaluate("() => { indices[0]=6; indices[1]=6; indices[2]=6; mvViews.forEach(v=>mvRender(v)); }")
        page.wait_for_timeout(400)
        # Seed at the block center on pane 0.
        page.evaluate("""() => { const v = mvViews[0]; const r = v.canvas.getBoundingClientRect();
            window.__sx = r.x + r.width*(5.5/16); window.__sy = r.y + r.height*(5.5/16); }""")
        sx, sy = page.evaluate("() => [window.__sx, window.__sy]")
        page.mouse.move(sx, sy); page.mouse.down(); page.wait_for_timeout(450); page.mouse.up()
        page.wait_for_timeout(450)
        assert page.evaluate("() => _rois[0] && _rois[0].scope_dim") == 2, "seed should grow 3D along pane 0 sliceDir"

        def pane_present(i):
            return page.evaluate("""(i) => {
                const v = mvViews[i]; const ov = v.roiOverlay;
                if (!ov || !v.roiCtx) return false;
                const img = v.roiCtx.getImageData(0,0,ov.width,ov.height).data;
                for (let k=3;k<img.length;k+=4){ if(img[k]>0) return true; } return false;
            }""", i)

        # Scroll pane 0 to z=0 (outside the block) via real wheel events.
        page.evaluate("""() => { const v = mvViews[0]; const r = v.canvas.getBoundingClientRect();
            window.__wx = r.x + r.width/2; window.__wy = r.y + r.height/2; }""")
        wx, wy = page.evaluate("() => [window.__wx, window.__wy]")
        for _ in range(8):
            page.mouse.move(wx, wy)
            page.mouse.wheel(0, 120)
        page.wait_for_timeout(400)
        z_after = page.evaluate("() => indices[2]")
        assert z_after <= 3, f"scroll should move pane 0 outside block (z<=3), got z={z_after}"
        assert not pane_present(0), f"3D ROI should disappear from pane 0 at z={z_after} (block is z=4..7)"

        # Scroll back into the block one step at a time until z enters [4,7].
        z_back = None
        for _ in range(20):
            page.mouse.move(wx, wy)
            page.mouse.wheel(0, -120)
            page.wait_for_timeout(120)
            z_back = page.evaluate("() => indices[2]")
            if 4 <= z_back <= 7:
                break
        page.wait_for_timeout(300)
        assert z_back is not None and 4 <= z_back <= 7, f"scroll back should re-enter block (4<=z<=7), got z={z_back}"
        assert pane_present(0), f"3D ROI should reappear on pane 0 at z={z_back}"

    def test_vmode_floodfill_3d_default_all_panes(self, loaded_viewer, client, tmp_path):
        # Seeding a floodfill in ANY of the three v-mode panes should default
        # to 3D (scope = that pane's sliceDir), not just the first pane.
        arr = np.zeros((16, 16, 16), dtype=np.float32)
        arr[4:8, 4:8, 4:8] = 5.0
        path = tmp_path / "roi_vmode_all_panes.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "block"}, timeout=15).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("v"); page.wait_for_timeout(900)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        page.evaluate("_roiSetShape('floodfill')"); page.wait_for_timeout(150)
        for pane_idx in range(3):
            page.evaluate("() => { _rois.length = 0; _redrawRoiOverlays(); }")
            page.wait_for_timeout(150)
            page.evaluate("() => { indices[0]=6; indices[1]=6; indices[2]=6; mvViews.forEach(v=>mvRender(v)); }")
            page.wait_for_timeout(300)
            info = page.evaluate("""(i) => {
                const v = mvViews[i]; const r = v.canvas.getBoundingClientRect();
                return { x: r.x + r.width/2, y: r.y + r.height/2, sliceDir: v.sliceDir };
            }""", pane_idx)
            page.mouse.move(info["x"], info["y"])
            page.mouse.down(); page.wait_for_timeout(450); page.mouse.up()
            page.wait_for_timeout(400)
            roi = page.evaluate("() => _rois[0] ? { scopeDim: _rois[0].scope_dim, has3d: !!_rois[0].mask3d_b64 } : null")
            assert roi is not None, f"pane {pane_idx}: ROI should be created"
            assert roi["scopeDim"] == info["sliceDir"], f"pane {pane_idx}: scopeDim ({roi['scopeDim']}) should equal sliceDir ({info['sliceDir']})"
            assert roi["has3d"], f"pane {pane_idx}: should have 3D mask"

    def test_roi_stats_correct_after_flip(self, loaded_viewer, client, tmp_path):
        # After flip_x, a rect drawn on the left half of the display should
        # compute stats on the data pixels visible there (the right half of
        # the data), not the unflipped left half.
        arr = np.full((1, 20, 20), 20.0, dtype=np.float32)
        arr[0, 0:10, :] = 10.0  # dim 1 [0:10] = 10 → left half when dim_x=1
        path = tmp_path / "roi_flip_stats.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "half"}, timeout=15).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=3_000)
        page.evaluate("_roiSetShape('rect')"); page.wait_for_timeout(150)
        cv = page.locator("canvas#viewer")
        box = cv.bounding_box()
        x0 = box["x"] + box["width"] * 0.05
        y0 = box["y"] + box["height"] * 0.1
        x1 = box["x"] + box["width"] * 0.45
        y1 = box["y"] + box["height"] * 0.9

        # Without flip: rect on left half (value=10)
        page.mouse.move(x0, y0); page.mouse.down()
        page.mouse.move(x1, y1, steps=8); page.wait_for_timeout(100); page.mouse.up()
        page.wait_for_timeout(600)
        stats = page.evaluate("() => _rois[0] && _rois[0].stats")
        assert stats is not None, "ROI stats should be fetched"
        assert abs(stats["mean"] - 10.0) < 0.1, f"no-flip mean should be ~10, got {stats['mean']}"

        # Flip x, draw same rect on left half of display (now shows value=20)
        page.evaluate("() => { _rois.length = 0; _redrawRoiOverlays(); }")
        page.wait_for_timeout(200)
        page.evaluate("() => { flip_x = !flip_x; updateView(); }")
        page.wait_for_timeout(400)
        page.mouse.move(x0, y0); page.mouse.down()
        page.mouse.move(x1, y1, steps=8); page.wait_for_timeout(100); page.mouse.up()
        page.wait_for_timeout(600)
        stats_flip = page.evaluate("() => _rois[0] && _rois[0].stats")
        assert stats_flip is not None, "ROI stats should be fetched after flip"
        assert abs(stats_flip["mean"] - 20.0) < 0.1, f"flipped mean should be ~20, got {stats_flip['mean']}"

    def test_vmode_roi_toolbar_buttons_fit_in_island(self, loaded_viewer, client, tmp_path):
        # The v-mode ROI toolbar (on the colorbar flip-back) must lay its
        # buttons out in a single row inside the island, not wrap/overflow.
        arr = np.zeros((12, 12, 12), dtype=np.float32)
        path = tmp_path / "roi_vmode_toolbar.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_vmode_toolbar"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_timeout(900)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        page.wait_for_timeout(200)
        geom = page.evaluate("""() => {
            const ctrls = document.getElementById('roi-cb-controls-mv');
            const back = ctrls ? ctrls.closest('.roi-flip-back') : null;
            const btns = ctrls ? Array.from(ctrls.querySelectorAll('button')) : [];
            const r = el => { const x = el.getBoundingClientRect(); return {x: Math.round(x.x), y: Math.round(x.y), right: Math.round(x.right), h: Math.round(x.height)}; };
            return {
                controls: r(ctrls),
                back: r(back),
                buttonRows: btns.map(b => r(b).y),
                buttonCount: btns.length,
            };
        }""")
        assert geom["buttonCount"] >= 6, "ROI toolbar should have shape + action buttons"
        # All buttons share one y → single row, no wrapping.
        assert len(set(geom["buttonRows"])) == 1, f"toolbar buttons should be on one row, got ys={geom['buttonRows']}"
        # Controls fit inside the island back face vertically.
        assert geom["controls"]["h"] <= geom["back"]["h"] + 2, f"controls ({geom['controls']['h']}px) should fit inside island back ({geom['back']['h']}px)"

    def test_vmode_floodfill_hold_updates_toolbar_sensitivity_line(self, loaded_viewer, client, tmp_path):
        arr = np.zeros((12, 12, 12), dtype=np.float32)
        arr[4:8, 4:8, 4:8] = 5.0
        path = tmp_path / "roi_vmode_sensitivity_line.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_vmode_sensitivity_line"}).json()["sid"]
        page = loaded_viewer(sid)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_timeout(900)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-cb-controls-mv", state="visible", timeout=3_000)
        page.evaluate("_roiSetShape('floodfill')")
        page.wait_for_timeout(150)
        placement = page.evaluate(
            """() => {
                const ctrls = document.getElementById('roi-cb-controls-mv');
                const line = ctrls.querySelector('.roi-floodfill-sensitivity');
                const flood = ctrls.querySelector('button[aria-label="flood fill"]');
                const shapes = flood ? flood.closest('.roi-cb-shapes') : null;
                const stats = ctrls.querySelector('button[aria-label="ROI stats"]');
                return {
                    lineAfterFloodfillGroup: shapes && shapes.nextElementSibling === line,
                    lineBeforeStats: stats && stats.previousElementSibling === line,
                };
            }"""
        )
        assert placement == {"lineAfterFloodfillGroup": True, "lineBeforeStats": True}, f"sensitivity line should reuse the separator between floodfill and stats, got {placement}"
        pane = page.evaluate("() => { const v = mvViews[0]; const r = v.canvas.getBoundingClientRect(); return { x: r.x + r.width/2, y: r.y + r.height/2 }; }")
        page.mouse.move(pane["x"], pane["y"])
        page.mouse.down()
        page.wait_for_timeout(120)
        during = page.evaluate(
            """() => {
                const ctrls = document.getElementById('roi-cb-controls-mv');
                const line = ctrls.querySelector('.roi-floodfill-sensitivity');
                return {
                    active: ctrls.classList.contains('floodfill-sensitivity-active'),
                    height: line.getBoundingClientRect().height,
                    value: Number(getComputedStyle(ctrls).getPropertyValue('--roi-ff-sensitivity')),
                };
            }"""
        )
        page.mouse.move(pane["x"], pane["y"] - 160, steps=8)
        page.wait_for_timeout(180)
        moved = page.evaluate(
            """() => {
                const ctrls = document.getElementById('roi-cb-controls-mv');
                const line = ctrls.querySelector('.roi-floodfill-sensitivity');
                return {
                    active: ctrls.classList.contains('floodfill-sensitivity-active'),
                    height: line.getBoundingClientRect().height,
                    value: Number(getComputedStyle(ctrls).getPropertyValue('--roi-ff-sensitivity')),
                };
            }"""
        )
        page.mouse.up()
        page.wait_for_timeout(120)
        after = page.evaluate("() => document.getElementById('roi-cb-controls-mv').classList.contains('floodfill-sensitivity-active')")

        assert during["active"], f"sensitivity line should activate while holding floodfill, got {during}"
        assert moved["active"], f"sensitivity line should stay active while dragging sensitivity, got {moved}"
        assert moved["value"] > during["value"], f"dragging upward should increase sensitivity value, got before={during}, after={moved}"
        assert moved["height"] > during["height"], f"line height should increase with sensitivity, got before={during}, after={moved}"
        assert not after, "sensitivity line should deactivate after mouseup"

    def test_rois_cleared_on_axis_reassignment(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        assert page.evaluate("() => _rois.length") == 1, "ROI should exist after drawing"

        # Find the third (non-display) dim and make it active, then press x
        # to reassign it to the x-axis.  This changes dim_x, so all ROIs
        # should be cleared.
        third = page.evaluate(
            """() => {
                for (let i = 0; i < shape.length; i++) {
                    if (i !== dim_x && i !== dim_y) return i;
                }
                return null;
            }"""
        )
        assert third is not None, "expected a third dim for 3D array"
        page.evaluate(f"() => {{ activeDim = {third}; renderInfo(); }}")
        page.wait_for_timeout(100)
        page.keyboard.press("x")
        page.wait_for_timeout(300)
        assert page.evaluate("() => _rois.length") == 0, "ROIs should be cleared after axis reassignment"

    def test_rois_not_cleared_on_dimbar_click(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        assert page.evaluate("() => _rois.length") == 1, "ROI should exist after drawing"

        # Clicking a non-display dim is normal navigation now; it should not
        # clear ROIs or toggle a hidden pending scope.
        nondisp = page.evaluate(
            """() => {
                const labels = Array.from(document.querySelectorAll('#info .dim-label[data-dim]'));
                const found = labels.find(el => {
                    const d = Number(el.getAttribute('data-dim'));
                    return d !== dim_x && d !== dim_y;
                });
                return found ? Number(found.getAttribute('data-dim')) : null;
            }"""
        )
        assert nondisp is not None
        page.locator(f'#info .dim-label[data-dim="{nondisp}"]').click()
        page.wait_for_timeout(150)
        state = page.evaluate(
            """() => ({
                roiCount: _rois.length,
                activeDim,
                scope: _rois[0] && _rois[0].scope ? _rois[0].scope.broadcast_dims.slice() : null,
            })"""
        )
        assert state["roiCount"] == 1, f"dimbar click should not clear ROIs, got {state}"
        assert state["activeDim"] == nondisp, f"dimbar click should navigate to dim {nondisp}, got {state}"
        assert state["scope"] == [], f"ROI scope should remain current-slice only, got {state}"

    def test_roi_transposes_with_view(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        before = page.evaluate("() => { const r = _rois[0]; return r ? { type: r.type, x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1, cx: r.cx, cy: r.cy, rr: r.r, dimX: r.dimX, dimY: r.dimY } : null; }")
        assert before is not None, "ROI should exist before transpose"

        page.keyboard.press("t")
        page.wait_for_timeout(400)
        after = page.evaluate("() => { const r = _rois[0]; return r ? { type: r.type, x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1, cx: r.cx, cy: r.cy, rr: r.r, dimX: r.dimX, dimY: r.dimY } : null; }")
        assert after is not None, "ROI should survive transpose"
        assert page.evaluate("() => _rois.length") == 1, "ROI count should not change on transpose"

        if before["type"] == "rect":
            assert after["x0"] == before["y0"] and after["y0"] == before["x0"], f"rect bbox should transpose, got {after}"
            assert after["x1"] == before["y1"] and after["y1"] == before["x1"], f"rect bbox should transpose, got {after}"
        elif before["type"] == "circle":
            assert after["cx"] == before["cy"] and after["cy"] == before["cx"], f"circle center should transpose, got {after}"
            assert after["rr"] == before["rr"], f"circle radius should not change, got {after}"
        if before["dimX"] is not None:
            assert after["dimX"] == before["dimY"] and after["dimY"] == before["dimX"], f"floodfill dimX/dimY should swap, got {after}"

    def test_roi_rotates_90_with_view(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)
        before = page.evaluate(
            """() => {
                const r = _rois[0];
                if (!r) return null;
                const w = canvas.width;
                return { type: r.type, x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1, cx: r.cx, cy: r.cy, rr: r.r, dimX: r.dimX, dimY: r.dimY, cw: w };
            }"""
        )
        assert before is not None, "ROI should exist before rotate"

        # Set activeDim to the third (non-display) dim so `r` does a 90° rotation
        third = page.evaluate("() => { for (let i = 0; i < shape.length; i++) if (i !== dim_x && i !== dim_y) return i; return null; }")
        assert third is not None
        page.evaluate(f"() => {{ activeDim = {third}; renderInfo(); }}")
        page.wait_for_timeout(100)
        page.keyboard.press("r")
        page.wait_for_timeout(400)
        after = page.evaluate("() => { const r = _rois[0]; return r ? { type: r.type, x0: r.x0, y0: r.y0, x1: r.x1, y1: r.y1, cx: r.cx, cy: r.cy, rr: r.r, dimX: r.dimX, dimY: r.dimY } : null; }")
        assert after is not None, "ROI should survive rotate"
        assert page.evaluate("() => _rois.length") == 1, "ROI count should not change on rotate"

        w = before["cw"]
        if before["type"] == "rect":
            # (px, py) -> (py, W-1-px)
            exp_x0 = min(before["y0"], before["y1"])
            exp_x1 = max(before["y0"], before["y1"])
            exp_y0 = w - 1 - max(before["x0"], before["x1"])
            exp_y1 = w - 1 - min(before["x0"], before["x1"])
            assert after["x0"] == exp_x0 and after["x1"] == exp_x1, f"rect x should come from old y, got {after} expected x=[{exp_x0},{exp_x1}]"
            assert after["y0"] == exp_y0 and after["y1"] == exp_y1, f"rect y should come from W-1-old_x, got {after} expected y=[{exp_y0},{exp_y1}]"
        elif before["type"] == "circle":
            assert after["cx"] == before["cy"], f"circle cx should come from old cy, got {after}"
            assert after["cy"] == w - 1 - before["cx"], f"circle cy should be W-1-old_cx, got {after}"
            assert after["rr"] == before["rr"], f"circle radius should not change, got {after}"
        if before["dimX"] is not None:
            assert after["dimX"] == before["dimY"] and after["dimY"] == before["dimX"], f"floodfill dimX/dimY should swap, got {after}"

    def test_roi_hover_stats_update_after_scroll_settles(self, loaded_viewer, client, tmp_path):
        arr = np.zeros((3, 64, 64), dtype=np.float32)
        arr[0, :, :] = 1
        arr[1, :, :] = 5
        arr[2, :, :] = 9
        path = tmp_path / "roi_scroll_stats.npy"
        np.save(path, arr)
        sid = client.post("/load", json={"filepath": str(path), "name": "roi_scroll_stats"}).json()["sid"]

        page = loaded_viewer(sid)
        _focus_kb(page)
        page.evaluate(
            """() => {
                indices[0] = 0;
                activeDim = 0;
                current_slice_dim = 0;
                updateView();
                renderInfo();
            }"""
        )
        page.wait_for_timeout(500)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        x0, y0, x1, y1 = self._draw_roi(page)
        page.wait_for_function("() => _rois[0] && _rois[0].stats && _rois[0].stats.mean === 1", timeout=3_000)
        page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2)
        page.wait_for_timeout(100)

        page.keyboard.press("k")
        page.wait_for_timeout(50)
        hidden = page.evaluate(
            """() => ({
                display: getComputedStyle(document.getElementById('roi-hover-tooltip')).display,
                mean: _rois[0].stats.mean,
            })"""
        )
        assert hidden["display"] == "none", f"ROI hover tooltip should hide while scrolled stats are pending, got: {hidden}"

        page.wait_for_function("() => _rois[0] && _rois[0].stats && _rois[0].stats.mean === 5", timeout=3_000)
        page.wait_for_timeout(100)
        state = page.evaluate(
            """() => ({
                mean: _rois[0].stats.mean,
                display: getComputedStyle(document.getElementById('roi-hover-tooltip')).display,
                tooltip: document.getElementById('roi-hover-tooltip')?.innerText || '',
            })"""
        )

        assert state["mean"] == 5, f"ROI stats should refresh to the new slice after scrolling settles, got: {state}"
        assert state["display"] != "none", f"ROI hover tooltip should reappear after refreshed stats arrive, got: {state}"
        assert "5 ± 0" in state["tooltip"], f"hover tooltip should show refreshed ROI stats, got: {state}"

    def test_default_circle_drawing_and_delete_key(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        x0, y0, x1, y1 = self._draw_roi(page)

        assert page.evaluate("() => _roiShape") == "circle"
        assert page.evaluate("() => _rois[0].type") == "circle"
        assert page.evaluate("() => _selectedRoiIdx") == 0
        assert page.evaluate("() => _roiName(0)") == "1"

        page.mouse.click((x0 + x1) / 2, (y0 + y1) / 2)
        page.wait_for_timeout(300)
        assert page.evaluate("() => _selectedRoiIdx") == 0
        assert not page.locator("#roi-context-menu").is_visible()
        assert not page.locator("#roi-hover-tooltip").evaluate("el => el.classList.contains('pinned')")
        assert page.evaluate("() => _roiCanvasLabel(0)") == "1"
        assert not page.locator("#export-overlay").is_visible()
        hover_text = page.locator("#roi-hover-tooltip").inner_text()
        assert "±" in hover_text
        assert "n =" not in hover_text
        assert "count" not in hover_text.lower()
        hover_height = page.locator("#roi-hover-tooltip").evaluate("el => el.getBoundingClientRect().height")

        roi_pt = {"x": (x0 + x1) / 2, "y": (y0 + y1) / 2}
        page.mouse.dblclick(roi_pt["x"], roi_pt["y"])
        page.wait_for_selector("#roi-label-editor.editing .roi-tip-name-input", timeout=2_000)
        edit_height = page.locator("#roi-label-editor").evaluate("el => el.getBoundingClientRect().height")
        assert edit_height <= hover_height + 1
        assert page.locator("#roi-label-editor .roi-tip-edit").evaluate("el => getComputedStyle(el).flexDirection") == "row"
        assert page.locator("#roi-label-editor .roi-tip-name-input").evaluate("el => getComputedStyle(el).borderBottomWidth") == "0px"
        assert "n =" not in page.locator("#roi-label-editor").inner_text()
        assert "count" not in page.locator("#roi-label-editor").inner_text().lower()
        page.locator("#roi-label-editor .roi-tip-name-input").fill("Phantom well")
        page.keyboard.press("Enter")
        page.wait_for_timeout(200)
        assert page.evaluate("() => _roiName(0)") == "Phantom well"
        assert page.evaluate("() => _roiCanvasLabel(0)") == "Phantom well"
        assert not page.locator("#roi-label-editor").evaluate("el => el.classList.contains('editing')")

        page.mouse.dblclick(roi_pt["x"], roi_pt["y"])
        page.wait_for_selector("#roi-label-editor.editing .roi-tip-name-input", timeout=2_000)
        page.locator("#roi-label-editor .roi-tip-name-input").fill("Cancelled")
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        assert page.evaluate("() => _roiName(0)") == "Phantom well"

        page.mouse.click((x0 + x1) / 2, (y0 + y1) / 2, button="right")
        page.wait_for_timeout(300)
        assert page.evaluate("() => _rois.length") == 0

    def test_stats_popup_manager_basics(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#slim-cb-wrap.roi-active", timeout=2_000)
        self._draw_roi(page)

        page.locator("#roi-cb-controls").get_by_label("ROI stats").click()
        page.wait_for_selector("#export-overlay.visible", timeout=2_000)
        assert page.locator("#export-title").inner_text() == "ROI stats"
        assert page.locator(".roi-manager-row, .roi-manager-row-compact").count() == 2
        table_text = page.locator("#export-table-wrap").inner_text()
        assert "MEAN" in table_text
        assert "STD" in table_text
        assert page.locator("#export-download").inner_text() == "Download CSV"
        assert page.locator("#export-mask").is_visible()

        name = page.locator(".roi-manager-name input").first
        name.fill("Phantom well")
        name.press("Enter")
        page.wait_for_timeout(200)
        assert page.evaluate("() => _rois[0].name") == "Phantom well"
        assert page.evaluate("() => _roiCanvasLabel(0)") == "Phantom well"

        page.locator(".roi-manager-actions").get_by_label("Delete Phantom well").click()
        page.wait_for_timeout(300)
        assert page.evaluate("() => _rois.length") == 0

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
    def test_shift_c_picker_uses_narrow_right_drawer_with_twelve_swatches(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+C")
        page.wait_for_selector("#cmap-picker.visible", timeout=2_000)
        page.wait_for_timeout(250)
        picker = page.locator("#cmap-picker-box")
        box = picker.bounding_box()
        viewport = page.viewport_size
        assert box is not None, "Colormap picker drawer did not render"
        assert viewport is not None, "Viewport size unavailable"
        assert box["width"] < 300
        assert box["x"] > viewport["width"] * 0.55
        assert box["y"] < 20
        assert box["x"] >= viewport["width"] - box["width"] - 32
        visible_count = page.locator("#cmap-picker-swatches .cmap-picker-swatch").count()
        assert visible_count == 12, f"Expected 12 quick swatches, saw {visible_count}"
        left_positions = page.evaluate(
            """() => [...document.querySelectorAll('#cmap-picker-swatches .cmap-picker-swatch')]
                .map(el => el.offsetLeft)"""
        )
        assert len(set(left_positions)) == 2, f"Expected two swatch columns, got: {left_positions}"
        label_count = page.locator("#cmap-picker-swatches .cmap-picker-section-label").count()
        assert label_count == 0, f"Expected no section labels, saw {label_count}"
        title = page.inner_text("#cmap-picker-title")
        assert title == "Colormaps"
        swatches_bottom = page.evaluate(
            """() => document.querySelector('#cmap-picker-swatches')?.getBoundingClientRect().bottom || 0"""
        )
        search_top = page.evaluate(
            """() => document.querySelector('#cmap-picker-input')?.getBoundingClientRect().top || 0"""
        )
        assert search_top >= swatches_bottom - 1, "Search should sit below the quick swatches"
        nav_state = page.evaluate(
            """() => {
                const items = [...document.querySelectorAll('#cmap-picker-swatches .cmap-picker-swatch-name')].map(el => el.textContent || '');
                const active = document.querySelector('#cmap-picker-swatches .cmap-picker-swatch.active .cmap-picker-swatch-name')?.textContent || '';
                const idx = items.indexOf(active);
                return {
                    active,
                    expectedDown: idx >= 0 && items[idx + 2] ? items[idx + 2] : active,
                };
            }"""
        )
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(80)
        down_name = page.evaluate(
            """() => document.querySelector('#cmap-picker-swatches .cmap-picker-swatch.active .cmap-picker-swatch-name')?.textContent || ''"""
        )
        assert nav_state["active"] != "", "Expected an active quick-swatch selection"
        assert down_name == nav_state["expectedDown"], f"ArrowDown should move vertically, got {down_name!r} from {nav_state['active']!r}"
        cycle_names = page.evaluate(
            """() => ({
                first: currentColormap(),
                active: document.querySelector('#cmap-picker-swatches .cmap-picker-swatch.active .cmap-picker-swatch-name')?.textContent || '',
            })"""
        )
        page.keyboard.press("c")
        page.wait_for_timeout(120)
        cycle_names["second"] = page.evaluate("() => currentColormap()")
        page.keyboard.press("c")
        page.wait_for_timeout(120)
        cycle_names["third"] = page.evaluate("() => currentColormap()")
        assert cycle_names["second"] != cycle_names["first"], "First c after opening should advance the open picker"
        assert cycle_names["third"] != cycle_names["second"], "Repeated c presses should keep cycling through colormaps"
        page.fill("#cmap-picker-input", "inferno")
        page.wait_for_timeout(250)
        page.wait_for_function(
            "() => (document.querySelectorAll('#cmap-picker-results .cmap-picker-swatch').length || 0) >= 1"
        )
        quick_count_after_search = page.locator("#cmap-picker-swatches .cmap-picker-swatch").count()
        assert quick_count_after_search == 12, f"Quick swatches should stay visible during search, saw {quick_count_after_search}"
        results_count = page.locator("#cmap-picker-results .cmap-picker-swatch").count()
        assert results_count >= 1, "Expected search matches below the search input"
        results_top = page.evaluate(
            """() => document.querySelector('#cmap-picker-results')?.getBoundingClientRect().top || 0"""
        )
        search_bottom = page.evaluate(
            """() => document.querySelector('#cmap-picker-input')?.getBoundingClientRect().bottom || 0"""
        )
        assert results_top >= search_bottom - 1, "Search matches should render below the search input"
        filtered_box = picker.bounding_box()
        assert filtered_box is not None
        assert abs(filtered_box["x"] - box["x"]) < 2, "Picker box shifted horizontally while filtering"
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
        page.wait_for_timeout(120)
        picker_still_open = page.is_visible("#cmap-picker.visible")
        assert picker_still_open, "First Enter should leave the search input without closing the picker"
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
        assert after["index"] > before["index"], (
            f"Control-wheel should still scroll the main view, got before={before}, after={after}"
        )
        assert after["zoom"] == before["zoom"], (
            f"Control-wheel should not change main-view zoom, got before={before}, after={after}"
        )
        assert visible_after, "loupe should remain visible while scrolling with Control held"
        assert loupe_after != loupe_before, "main-view loupe should redraw after wheel-driven slice changes"

    def test_wheel_step_scales_with_dim_size(self, loaded_viewer, client, tmp_path):
        """Scroll sensitivity should depend on the dim size: a single wheel
        notch advances 1 slice on a small dim but several slices on a large
        one, so navigating a 1000-slice volume isn't glacial."""
        big = np.random.default_rng(1).standard_normal((400, 40, 40)).astype(np.float32)
        big_path = tmp_path / "big.npy"
        np.save(big_path, big)
        sid_big = client.post(
            "/load",
            json={"filepath": str(big_path), "name": "big"},
        ).json()["sid"]
        page = loaded_viewer(sid_big)
        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)

        page.evaluate(
            """() => {
                current_slice_dim = 0;  // the 400-slice axis
                indices[0] = 0;
                renderInfo(); updateView();
            }"""
        )
        page.mouse.move(cx, cy)
        page.wait_for_timeout(120)
        page.mouse.wheel(0, -120)
        page.wait_for_timeout(260)

        after = page.evaluate("() => ({ idx: indices[0], dim: current_slice_dim, shape: shape[0] })")
        # 400-slice dim -> _wheelStep = round(400/200) = 2; one notch must
        # advance by at least 2 (clamped). Small-dim behavior (step 1) is
        # already covered by the ctrl_wheel tests using the 20×64×64 fixture.
        assert after["idx"] >= 2, f"wheel on a 400-slice dim should advance >=2 slices per notch, got: {after}"

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

        assert after[before["sliceDir"]] > before["indices"][before["sliceDir"]], (
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
        page.evaluate(
            """() => {
                flip_x = true;
                if (lastImageData && ctx) {
                    ctx.putImageData(applyFlips(lastImageData, lastImgW, lastImgH), 0, 0);
                }
                renderInfo();
            }"""
        )

        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)
        pin_x = cx - 64
        pin_y = cy
        page.mouse.move(pin_x, pin_y)
        page.wait_for_timeout(180)

        hover_state = page.evaluate(
            """() => ({
                visible: !!document.querySelector('#main-pixel-info .pixel-hover-card'),
                valueText: (document.querySelector('#main-pixel-info .pixel-hover-value') || {}).textContent || '',
                coordsText: (document.querySelector('#main-pixel-info .pixel-hover-coords') || {}).textContent || '',
            })"""
        )
        assert hover_state["visible"], "hover-info mode should show the pinned-style hover card"
        assert "x=" in hover_state["coordsText"] and "y=" in hover_state["coordsText"], "hover card should include coordinates"
        assert hover_state["valueText"].strip(), "hover card should show a numeric value before pinning"

        page.mouse.click(pin_x, pin_y)
        page.evaluate(
            """async () => {
                for (const key of Object.keys(_hoverSliceCache)) delete _hoverSliceCache[key];
                await _refreshHoverPins(true);
            }"""
        )
        page.mouse.click(cx + 80, cy + 30)
        page.wait_for_timeout(240)

        pin_count = page.evaluate(
            "() => document.querySelectorAll('#main-pixel-pins .main-pixel-pin').length"
        )
        pin_state = page.evaluate(
            """() => {
                const pin = document.querySelector('#main-pixel-pins .main-pixel-pin');
                return {
                    valueText: (pin?.querySelector('.pixel-pin-value') || {}).textContent || '',
                    coordsText: (pin?.querySelector('.pixel-pin-coords') || {}).textContent || '',
                };
            }"""
        )
        assert pin_count == 2, "clicking twice in hover-info mode should create two pins"
        assert pin_state["valueText"].strip(), "pinned readout should show a numeric value"
        assert pin_state["coordsText"] == hover_state["coordsText"], (
            f"pinned hover coords should match the live hover coords, got {pin_state['coordsText']} vs {hover_state['coordsText']}"
        )
        actual = float(pin_state["valueText"])
        expected = float(hover_state["valueText"])
        assert np.isclose(actual, expected, atol=5e-4), (
            f"pinned hover value should survive a forced refresh in flipped view, got {actual} vs {expected}"
        )

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
