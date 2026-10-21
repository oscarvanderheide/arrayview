"""Layer 2: Playwright browser tests.

Run with:
    pytest tests/test_browser.py
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.browser

DEBUG_DIR = Path(__file__).resolve().parents[1] / "debug"

# ---------------------------------------------------------------------------
# Perf instrumentation
# ---------------------------------------------------------------------------


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


_JS_MV_CANVAS_COUNT = """
() => document.querySelectorAll('.mv-canvas').length
"""


def _focus_kb(page):
    """Focus the hidden keyboard-sink textarea so key events are delivered."""
    page.focus("#keyboard-sink")


def _preset_overlay_state(page, pane_selector=None, colorbar_selector=None):
    """Read the visible percentile selector and its placement."""
    return page.evaluate(
        """([paneSelector, colorbarSelector]) => {
            const shown = el => {
                if (!el) return false;
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && Number.parseFloat(style.opacity || '1') > 0.05
                    && rect.width > 0 && rect.height > 0;
            };
            const overlays = [...document.querySelectorAll('.percentile-preset-overlay')];
            const visible = overlays.filter(shown);
            const overlay = visible.at(-1) || null;
            const dots = overlay
                ? [...overlay.querySelectorAll('.percentile-preset-dots .percentile-preset-dot')]
                : [];
            const rect = overlay?.getBoundingClientRect() || null;
            const valueRect = overlay?.querySelector('.percentile-preset-value')
                ?.getBoundingClientRect() || null;
            const dotsRect = overlay?.querySelector('.percentile-preset-dots')
                ?.getBoundingClientRect() || null;
            const paneRect = paneSelector
                ? document.querySelector(paneSelector)?.getBoundingClientRect() || null
                : null;
            const colorbarRect = colorbarSelector
                ? document.querySelector(colorbarSelector)?.getBoundingClientRect() || null
                : null;
            const oldPercentVisible = [...document.querySelectorAll(
                '.dmenu-percent-vmin, .dmenu-percent-vmax'
            )].filter(shown).length;
            const oldLocksVisible = [...document.querySelectorAll(
                '.dmenu-lock-vmin, .dmenu-lock-vmax'
            )].filter(shown).length;
            const eggs = document.getElementById('mode-eggs');
            const eggsStyle = eggs ? getComputedStyle(eggs) : null;
            return {
                visibleCount: visible.length,
                text: overlay?.querySelector('.percentile-preset-value')?.textContent?.trim() || '',
                dotCount: dots.length,
                minDotSize: dots.length ? Math.min(...dots.map(dot => {
                    const style = getComputedStyle(dot);
                    return Math.min(
                        Number.parseFloat(style.width),
                        Number.parseFloat(style.height),
                    );
                })) : 0,
                activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                valueRect,
                dotsRect,
                oldPercentVisible,
                oldLocksVisible,
                rect,
                paneRect,
                colorbarRect,
                withinPane: !!(rect && paneRect
                    && rect.left >= paneRect.left - 1
                    && rect.right <= paneRect.right + 1
                    && rect.top >= paneRect.top - 1
                    && rect.bottom <= paneRect.bottom + 1),
                aboveColorbar: !!(rect && colorbarRect && rect.bottom <= colorbarRect.top + 2),
                eggsVisible: !!(eggs && [...eggs.querySelectorAll('.mode-badge')].some(shown)),
                eggsOpacity: eggsStyle ? Number.parseFloat(eggsStyle.opacity || '1') : 0,
            };
        }""",
        [pane_selector, colorbar_selector],
    )


def _wait_for_preset_overlay(page, visible=True, timeout=3000):
    page.wait_for_function(
        """expected => {
            const shown = el => {
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && Number.parseFloat(style.opacity || '1') > 0.05
                    && rect.width > 0 && rect.height > 0;
            };
            return [...document.querySelectorAll('.percentile-preset-overlay')].some(shown)
                === expected;
        }""",
        arg=visible,
        timeout=timeout,
    )


def _wait_for_preset_change(page, previous_text, timeout=3000):
    page.wait_for_function(
        """before => {
            const shown = el => {
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && Number.parseFloat(style.opacity || '1') > 0.05
                    && rect.width > 0 && rect.height > 0;
            };
            const overlay = [...document.querySelectorAll('.percentile-preset-overlay')]
                .find(shown);
            return !!overlay
                && overlay.querySelector('.percentile-preset-value')?.textContent?.trim() !== before;
        }""",
        arg=previous_text,
        timeout=timeout,
    )


def _wait_for_mode_badge(page, visible=True, timeout=2000):
    page.wait_for_function(
        """expected => {
            const badges = [...document.querySelectorAll('#mode-eggs .mode-badge')];
            const shown = badges.some(el => {
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && Number.parseFloat(style.opacity || '1') > 0.05
                    && rect.width > 0 && rect.height > 0;
            });
            return shown === expected;
        }""",
        arg=visible,
        timeout=timeout,
    )


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
        "() => getComputedStyle(document.querySelector('.overlay-palette-row.dimmed')).opacity"
    ) != "1"
    page.mouse.move(5, 5)
    assert page.evaluate("() => _overlayFocusIdx") is None
    # Each mask row draws a strip of where it sits along the slice axis.
    page.wait_for_function(
        "() => document.querySelector('#overlay-palette .overlay-palette-strip-fill')?.getAttribute('d')"
    )
    # Minimizing shrinks the list to a single button; clicking it restores the list.
    palette.locator(".overlay-palette-minimize").click()
    assert "collapsed" in palette.get_attribute("class")
    assert palette.locator(".overlay-palette-row").count() == 0
    assert palette.bounding_box()["width"] < 40
    palette.locator(".overlay-palette-restore").click()
    assert palette.locator(".overlay-palette-row").count() == 2
    style_btn = palette.locator(".overlay-palette-style")
    style_btn.click()
    assert page.evaluate("() => overlayOutlineOnly") is True
    assert "outline" in palette.get_attribute("class")
    assert page.evaluate(
        "() => { const p = new URLSearchParams(); _applyOverlayRenderParams(p); return p.get('overlay_outline'); }"
    ) == "1"
    palette.locator(".overlay-palette-style").click()
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


def test_o_cycles_views_and_remembers_each_one(page, client, server_url, tmp_path):
    path = tmp_path / "vol.npy"
    np.save(path, np.random.rand(30, 40, 50).astype(np.float32))
    sid = client.post("/load", json={"filepath": str(path), "name": "vol"}).json()["sid"]
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_timeout(400)
    state = "() => [dim_x, dim_y, current_slice_dim, flip_x, flip_y]"

    seen = []
    for _ in range(3):
        page.keyboard.press("o")
        page.wait_for_timeout(200)
        seen.append(page.evaluate(state))
    assert sorted(s[2] for s in seen) == [0, 1, 2], "three different slice directions"

    # A rotation made in one view is still there after a full cycle.
    page.keyboard.press("r")
    page.wait_for_timeout(200)
    rotated = page.evaluate(state)
    for _ in range(3):
        page.keyboard.press("o")
        page.wait_for_timeout(200)
    assert page.evaluate(state) == rotated

    # Naming the view and giving a voxel size draws thick voxels taller.
    page.keyboard.press("Shift+A")
    page.fill("#inline-prompt-input", "axial 1 1 3")
    page.keyboard.press("Enter")
    page.wait_for_timeout(200)
    page.keyboard.press("o")
    page.wait_for_timeout(400)
    assert page.evaluate("() => document.querySelector('#mode-eggs .percentile-preset-value').textContent") == "coronal"
    lit = page.evaluate("() => [...document.querySelectorAll('#mode-eggs .percentile-preset-dot')].map(d => d.classList.contains('active'))")
    assert lit == [False, True, False]
    assert page.locator("#mode-eggs canvas.percentile-preset-cube").count() == 0
    w, h, bw, bh = page.evaluate(
        "() => [parseFloat(canvas.style.width), parseFloat(canvas.style.height), canvas.width, canvas.height]"
    )
    sx, sy = page.evaluate("() => [_voxelSizeOf(dim_x), _voxelSizeOf(dim_y)]")
    assert abs((h / w) - (bh * sy) / (bw * sx)) < 0.02


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


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------


class TestKeyboard:


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


    def test_stack_mosaic_enters_navigates_exits(self, loaded_viewer, client, tmp_path):
        """Shift+; prompts for N, shows N panes side by side along the stack
        dim, j/k steps the window by N when the stack dim is active, and the
        same key exits back to a single view without tearing out the canvas."""
        stack_dir = tmp_path / "stack_mosaic_src"
        stack_dir.mkdir()
        rng = np.random.default_rng(7)
        for i in range(6):
            np.save(stack_dir / f"vol_{i}.npy", (rng.standard_normal((5, 40, 40)) + i).astype(np.float32))
        resp = client.post("/load", json={"filepath": str(stack_dir)})
        sid = resp.json()["sid"]

        page = loaded_viewer(sid)
        _focus_kb(page)
        # Must have a collection dim (the stack)
        assert page.evaluate("() => collectionSpatialNdim >= 0")
        total = page.evaluate("() => shape[collectionSpatialNdim]")
        assert total == 6

        page.keyboard.press("Shift+;")
        page.wait_for_selector("#inline-prompt.visible", timeout=5_000)
        page.fill("#inline-prompt-input", "2")
        page.keyboard.press("Enter")
        page.wait_for_function("() => stackMosaicActive", timeout=5_000)
        page.wait_for_function("() => stackMosaicViews.length === 2", timeout=5_000)
        page.wait_for_function(
            "() => stackMosaicViews.every(v => v.lastW > 0 && v.lastH > 0)",
            timeout=10_000,
        )
        assert page.is_visible("#stack-mosaic-wrap.active")
        # Window shows the first two arrays
        assert page.evaluate("() => stackMosaicViews.map(v => v.colIdx)") == [0, 1]
        # Dimbar shows the window range
        dimbar = page.inner_text("#info")
        assert "1–2" in dimbar, f"expected window range in dimbar, got: {dimbar}"

        # Make the stack dim active, then step forward by N with k.
        page.evaluate("() => { activeDim = collectionSpatialNdim; renderInfo(); }")
        page.keyboard.press("k")
        page.wait_for_function(
            "() => stackMosaicViews.map(v => v.colIdx).join(',') === '2,3'",
            timeout=10_000,
        )
        # Step forward again to the last window (4,5).
        page.keyboard.press("k")
        page.wait_for_function(
            "() => stackMosaicViews.map(v => v.colIdx).join(',') === '4,5'",
            timeout=10_000,
        )
        # At the end, stepping forward again is a no-op.
        page.keyboard.press("k")
        page.wait_for_timeout(300)
        assert page.evaluate("() => stackMosaicStart") == 4

        # Resize the window with [ / ].
        page.keyboard.press("]")
        page.wait_for_function("() => stackMosaicN === 3", timeout=5_000)
        page.keyboard.press("[")
        page.wait_for_function("() => stackMosaicN === 2", timeout=5_000)

        # Exit with the same key: single view returns, canvas stays intact.
        page.keyboard.press("Shift+;")
        page.wait_for_function("() => !stackMosaicActive", timeout=5_000)
        assert page.evaluate("() => !!document.getElementById('canvas-wrap')")
        assert page.is_visible("canvas#viewer")


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


    def test_file_picker_lists_open_arrays_and_arrow_keys_walk_the_list(
        self, page, server_url, sid_2d, sid_3d
    ):
        page.goto(f"{server_url}/?sid={sid_2d}")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page.wait_for_timeout(400)
        page.keyboard.press("Shift+Y")
        page.wait_for_selector("#uni-picker.visible", timeout=5_000)
        page.wait_for_timeout(300)

        assert page.locator("#uni-picker img").count() == 0
        open_rows = page.locator(".cp-tab-mark")
        assert open_rows.count() >= 2
        assert page.locator(".cp-item-current").count() == 1

        visited = []
        for _ in range(3):
            page.keyboard.press("ArrowDown")
            page.wait_for_timeout(100)
            visited.append(page.evaluate(
                "() => [...document.querySelectorAll('#uni-picker-list .cp-item')].indexOf(document.activeElement)"
            ))
        assert visited == [0, 1, 2]

        page.keyboard.press("Escape")
        page.wait_for_selector("#uni-picker", state="hidden", timeout=2_000)

    def test_compare_mosaic_hover_toggle_sends_no_rejected_pixel_request(
        self, page, server_url, sid_3d, arr_3d, client, tmp_path
    ):
        path = tmp_path / "arr3d_compare_mosaic.npy"
        np.save(path, arr_3d * 0.5)
        sid_compare = client.post(
            "/load", json={"filepath": str(path), "name": "arr3d_compare_mosaic"}
        ).json()["sid"]
        rejected = []
        page.on(
            "response",
            lambda r: rejected.append(r.url) if "/pixel/" in r.url and r.status >= 400 else None,
        )
        page.goto(
            f"{server_url}/?sid={sid_3d}&compare_sid={sid_compare}&compare_sids={sid_compare}"
        )
        page.wait_for_selector("canvas#compare-left-canvas:visible", timeout=15_000)
        page.keyboard.press("z")
        page.wait_for_timeout(700)
        for _ in range(2):
            page.keyboard.press("i")
            page.wait_for_timeout(400)
        page.mouse.move(400, 300)
        page.wait_for_timeout(400)
        assert rejected == []

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


    def test_compare_panes_touching_tiles_four_arrays_into_a_block(
        self, page, server_url, sid_2d, arr_2d, client, tmp_path
    ):
        compare_sids = []
        for i in range(3):
            path = tmp_path / f"arr2d_touch_{i}.npy"
            np.save(path, arr_2d * (0.3 + 0.2 * i))
            compare_sids.append(client.post(
                "/load", json={"filepath": str(path), "name": f"arr2d_touch_{i}"}
            ).json()["sid"])

        page.goto(
            f"{server_url}/?sid={sid_2d}&compare_sid={compare_sids[0]}&compare_sids={','.join(compare_sids)}"
        )
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        page.wait_for_timeout(500)
        page.evaluate("() => _applyViewerPreference('compare_gapless', true)")
        page.wait_for_timeout(300)

        rects = page.evaluate(
            """() => [...document.querySelectorAll('#compare-panes .compare-pane.active .compare-canvas-clip')]
                .map(e => { const r = e.getBoundingClientRect();
                            return [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)]; })"""
        )
        assert len(rects) == 4
        (l0, t0, r0, b0), (l1, t1, r1, b1), (l2, t2, r2, b2), _ = rects
        # 2×2 block: side by side and stacked panes share an edge.
        assert t0 == t1 and r0 == l1, f"top row panes do not touch: {rects}"
        assert l2 == l0 and t2 == b0, f"rows do not touch: {rects}"
        # With two rows the array names sit inside the images.
        title_top = page.evaluate(
            "() => document.getElementById('compare-left-title').getBoundingClientRect().top"
        )
        assert t0 <= title_top < b0


    def test_d_opens_without_selector_then_cycles_a_stable_percentile_overlay(
        self, loaded_viewer, sid_2d
    ):
        page = loaded_viewer(sid_2d)
        _focus_kb(page)
        page.keyboard.press("Shift+L")
        _wait_for_mode_badge(page)
        initial_range = page.evaluate("() => [manualVmin, manualVmax]")

        page.keyboard.press("d")
        page.wait_for_timeout(450)
        opened = _preset_overlay_state(page, "#viewer", "#slim-cb-wrap")
        assert opened["visibleCount"] == 0
        assert opened["eggsVisible"], "opening the histogram must preserve the active mode badge"
        assert page.evaluate("() => [manualVmin, manualVmax]") == pytest.approx(initial_range)

        page.keyboard.press("d")
        _wait_for_preset_overlay(page)
        page.wait_for_function(
            """before => Math.abs(manualVmin - before[0]) > 1e-9
                || Math.abs(manualVmax - before[1]) > 1e-9""",
            arg=initial_range,
            timeout=3_000,
        )
        first = {
            **_preset_overlay_state(page, "#viewer", "#slim-cb-wrap"),
            **page.evaluate(
                "() => ({ manualVmin, manualVmax, "
                "status: document.getElementById('status')?.textContent || '', "
                "toast: document.getElementById('toast')?.textContent || '' })"
            ),
        }
        states = [first]
        for _ in range(first["dotCount"] - 1):
            page.keyboard.press("d")
            _wait_for_preset_change(page, states[-1]["text"])
            states.append(_preset_overlay_state(page, "#viewer", "#slim-cb-wrap"))

        assert first["visibleCount"] == 1
        assert first["text"] and len({state["text"] for state in states}) == first["dotCount"]
        assert first["dotCount"] > 1 and first["activeIndex"] >= 0
        assert first["minDotSize"] >= 5
        assert (first["manualVmin"], first["manualVmax"]) != pytest.approx(initial_range)
        assert first["oldPercentVisible"] == 0
        assert first["oldLocksVisible"] == 0
        assert first["withinPane"] and first["aboveColorbar"]
        assert first["status"] == ""
        assert first["toast"] == ""

        for rect_name in ("valueRect", "dotsRect"):
            baseline = first[rect_name]
            assert baseline
            baseline_points = (
                baseline["x"],
                baseline["x"] + baseline["width"] / 2,
                baseline["x"] + baseline["width"],
            )
            for state in states[1:]:
                rect = state[rect_name]
                assert rect
                points = (
                    rect["x"],
                    rect["x"] + rect["width"] / 2,
                    rect["x"] + rect["width"],
                )
                assert max(abs(a - b) for a, b in zip(points, baseline_points)) <= 1, (
                    f"{rect_name} moved while cycling labels: {baseline} then {rect}"
                )


    def test_space_toggles_playback(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("Space")
        page.wait_for_function("() => document.querySelector('#info').classList.contains('playback-hud-visible')")
        initial = page.evaluate("() => ({fps: playFps, size: shape[playingDim]})")
        assert initial["fps"] == max(1, min(120, int(initial["size"] / 4 + 0.5)))
        assert page.locator(".playback-fps").inner_text() == f'{initial["fps"]} fps'
        assert page.locator(".playback-controls").evaluate(
            "el => el.parentElement.classList.contains('playing-dim')"
        )

        page.keyboard.press("Shift+Period")
        assert page.locator('.playback-hint[data-step="up"]').evaluate(
            "el => el.classList.contains('is-pressed')"
        )
        adjusted = page.evaluate("() => playFps")
        assert adjusted > initial["fps"]
        assert page.locator(".playback-fps").inner_text() == f"{adjusted} fps"
        assert page.locator("#info").evaluate(
            "el => !el.style.getPropertyValue('--av-fill-tint')"
        )
        assert not page.inner_text("#toast")

        page.keyboard.press("Space")
        assert not page.locator("#info").evaluate(
            "el => el.classList.contains('playback-hud-visible')"
        )


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


    def test_v_mode_roi_draws_circle_on_mv_pane(self, loaded_viewer, sid_3d):
        """Circle/rect/freehand ROIs should be drawable on mv panes in v-mode,
        tagged with the pane's mvDimX/mvDimY so they render on the right pane."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_selector("#mv-panes", state="visible", timeout=5_000)
        page.wait_for_timeout(400)
        page.keyboard.press("Shift+R")
        page.wait_for_selector("#roi-stats-hud .roi-hud-tool", timeout=2_000)
        page.evaluate("() => _roiSetShape('circle')")
        page.wait_for_timeout(200)

        # The ROI panel sits over the top-left of the first pane when the
        # panes fill the window, so draw clear of it.
        pane = page.locator(".mv-canvas").nth(0)
        box = pane.bounding_box()
        x0, y0 = box["x"] + box["width"] * 0.4, box["y"] + box["height"] * 0.4
        x1, y1 = box["x"] + box["width"] * 0.8, box["y"] + box["height"] * 0.8
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=10)
        page.mouse.up()
        page.wait_for_timeout(800)

        roi = page.evaluate("() => _rois[0] ? { type: _rois[0].type, mvDimX: _rois[0].mvDimX, mvDimY: _rois[0].mvDimY } : null")
        assert roi is not None, "ROI should be created on mv pane"
        assert roi["type"] == "circle", f"ROI should be circle, got: {roi}"
        assert roi["mvDimX"] is not None and roi["mvDimY"] is not None, f"ROI should have mvDimX/mvDimY, got: {roi}"


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
        page.wait_for_selector("#roi-stats-hud .roi-hud-tool", timeout=3_000)
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


class TestNormalInspectInteractions:
    @staticmethod
    def _assert_info_cursor_active(page, selector):
        canvas = page.locator(selector).first
        cx, cy = _center_of(canvas)
        page.mouse.move(cx, cy)
        page.wait_for_function(
            """selector => {
                const canvas = document.querySelector(selector);
                const cursor = document.getElementById('info-hover-cursor');
                return canvas?.classList.contains('info-hover-cursor-source')
                    && cursor?.classList.contains('visible');
            }""",
            arg=selector,
        )
        state = page.evaluate(
            """({ selector, cx, cy }) => {
                const canvas = document.querySelector(selector);
                const cursor = document.getElementById('info-hover-cursor');
                const box = cursor.getBoundingClientRect();
                return {
                    canvasCursor: getComputedStyle(canvas).cursor,
                    cursorVisible: getComputedStyle(cursor).visibility,
                    width: box.width,
                    height: box.height,
                    centerX: box.left + box.width / 2,
                    centerY: box.top + box.height / 2,
                };
            }""",
            {"selector": selector, "cx": cx, "cy": cy},
        )
        assert state["canvasCursor"] == "none"
        assert state["cursorVisible"] == "visible"
        assert state["width"] == 18
        assert state["height"] == 18
        assert abs(state["centerX"] - cx) < 1
        assert abs(state["centerY"] - cy) < 1

    @staticmethod
    def _assert_ctrl_loupe_uses_shared_pill(page, canvas_selector, loupe_selector):
        canvas = page.locator(canvas_selector).first
        box = canvas.bounding_box()
        px = box["x"] + box["width"] / 2
        py = box["y"] + 12
        page.mouse.move(px, py)
        page.keyboard.down("Control")
        page.evaluate(
            """({ selector, x, y }) => document.querySelector(selector).dispatchEvent(
                new MouseEvent('mousemove', { clientX: x, clientY: y, ctrlKey: true, bubbles: true })
            )""",
            {"selector": canvas_selector, "x": px, "y": py},
        )
        page.wait_for_selector("#mode-eggs .eggs-value-pill", timeout=5_000)
        state = page.evaluate(
            """({ canvasSelector, loupeSelector }) => {
                const canvas = document.querySelector(canvasSelector);
                const pane = canvas.getBoundingClientRect();
                const eggs = document.getElementById('mode-eggs');
                const pill = eggs.querySelector('.eggs-value-pill').getBoundingClientRect();
                const loupe = document.querySelector(loupeSelector).getBoundingClientRect();
                return {
                    placement: eggs.dataset.pillPlacement,
                    centered: Math.abs((pill.left + pill.right) / 2 - (pane.left + pane.right) / 2) < 1,
                    overlaps: pill.left < loupe.right && pill.right > loupe.left
                        && pill.top < loupe.bottom && pill.bottom > loupe.top,
                    legacyVisible: Array.from(document.querySelectorAll('.cv-pixel-info'))
                        .some(el => getComputedStyle(el).display !== 'none'),
                };
            }""",
            {"canvasSelector": canvas_selector, "loupeSelector": loupe_selector},
        )
        assert state["placement"] == "bottom"
        assert state["centered"]
        assert not state["overlaps"]
        assert not state["legacyVisible"]
        page.keyboard.up("Control")
        page.wait_for_timeout(220)
        assert not page.locator("#mode-eggs .eggs-value-pill").count()


    def test_window_level_drag_to_apex_hits_full_data_range_and_clamps(self, loaded_viewer, sid_3d):
        """Dragging the dot to the triangle's apex must land exactly on
        vmin=data-min, vmax=data-max — and dragging further past the apex
        must not move the values any further (a hard clamp, not just a slow
        one). See test_window_level_dot_starts_at_current_values for why
        this uses the 3-D fixture rather than the 2-D gradient."""
        page = loaded_viewer(sid_3d)
        if page.evaluate("() => _pixelInfoVisible"):
            page.keyboard.press("i")
        page.evaluate("() => primaryCb.ensureHistogramData()")
        page.wait_for_function("() => _histData !== null", timeout=2_000)
        assert not page.evaluate("() => _dmenuEffectiveVminLocked()"), (
            "fixture must exercise the general (unlocked) triangle drag"
        )
        canvas = page.locator("#viewer")
        cx, cy = _center_of(canvas)
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.wait_for_function("() => !!_bcDrag")
        geom = page.evaluate(
            """() => ({
                dot0y: _bcDrag.dot0.y,
                dataMin: _bcDrag.dataMin,
                dataMax: _bcDrag.dataMax,
                triMarginTop: BC_TRI_MT,
            })"""
        )

        # Move well past the amount needed to reach y=0 (the apex).
        page.mouse.move(cx, cy - geom["dot0y"] - 40, steps=6)
        page.wait_for_timeout(120)
        at_apex = page.evaluate(
            """() => {
                const dot = document.getElementById('bc-hud').querySelector('.bc-hud-dot');
                return { vmin: manualVmin, vmax: manualVmax, dotY: Number(dot.getAttribute('cy')) };
            }"""
        )
        assert at_apex["vmin"] == pytest.approx(geom["dataMin"], abs=1e-4)
        assert at_apex["vmax"] == pytest.approx(geom["dataMax"], abs=1e-4)
        assert at_apex["dotY"] == pytest.approx(geom["triMarginTop"], abs=0.5)

        # Keep dragging further up — values must stay put, clamped at the edge.
        page.mouse.move(cx, cy - geom["dot0y"] - 200, steps=4)
        page.wait_for_timeout(120)
        beyond = page.evaluate("() => ({ vmin: manualVmin, vmax: manualVmax })")
        page.mouse.up()
        assert beyond["vmin"] == pytest.approx(at_apex["vmin"], abs=1e-9)
        assert beyond["vmax"] == pytest.approx(at_apex["vmax"], abs=1e-9)


    def test_info_hover_drag_only_inspects(self, loaded_viewer, sid_2d):
        page = loaded_viewer(sid_2d)
        canvas = page.locator("#viewer")
        box = canvas.bounding_box()
        x0, y0 = box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.25
        x1, y1 = box["x"] + box["width"] * 0.65, box["y"] + box["height"] * 0.65
        if not page.evaluate("() => _pixelInfoVisible"):
            page.keyboard.press("i")
        before = page.evaluate(
            "() => ({ range: [manualVmin ?? currentVmin, manualVmax ?? currentVmax], rois: _rois.length })"
        )

        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move(x1, y1, steps=8)
        during = page.evaluate(
            """() => ({
                windowing: !!_bcDrag,
                hasTemporaryOutline: !!document.getElementById('info-region-outline'),
                roiMode: rectRoiMode,
                rois: _rois.length,
            })"""
        )
        page.mouse.up()
        page.wait_for_timeout(180)
        released = page.evaluate(
            """() => ({
                range: [manualVmin ?? currentVmin, manualVmax ?? currentVmax],
                windowing: !!_bcDrag,
                infoActive: _pixelInfoVisible,
                hasTemporaryOutline: !!document.getElementById('info-region-outline'),
                rois: _rois.length,
            })"""
        )

        assert during == {
            "windowing": False,
            "hasTemporaryOutline": False,
            "roiMode": False,
            "rois": before["rois"],
        }
        assert released == {
            "range": before["range"],
            "windowing": False,
            "infoActive": True,
            "hasTemporaryOutline": False,
            "rois": before["rois"],
        }


# Pixel-diff visual regression was removed: tests/snapshots/ is gitignored, so
# the baseline never survives a fresh clone and every run took the
# "create baseline, then skip" path. It never once compared anything.


class TestModeCombinations:
    """Several arrays combined with ortho / qMRI / mosaic."""

    def _two(self, client, tmp_path, arr, name):
        from conftest import register_array
        return [register_array(client, arr * (1 + i), tmp_path, f"{name}{i}") for i in range(2)]

    def test_requested_ortho_with_two_arrays_keeps_the_comparison(
        self, page, server_url, client, tmp_path, arr_3d
    ):
        from arrayview._session import SESSIONS
        a, b = self._two(client, tmp_path, arr_3d, "cmportho")
        SESSIONS[a].initial_view = {"mode": "ortho"}
        page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
        page.wait_for_function("() => compareActive && compareMvActive", timeout=15_000)
        # Panes are titled with the file names, not session ids.
        page.wait_for_function(
            "() => [...document.querySelectorAll('.cq-row-label')].map(e => e.textContent).join(' ').includes('cmportho0')",
            timeout=5_000,
        )

    def test_compare_qmri_and_compare_3plane_switch_without_blanking(
        self, page, server_url, client, tmp_path
    ):
        arr = np.random.default_rng(3).random((3, 12, 32, 32)).astype(np.float32)
        a, b = self._two(client, tmp_path, arr, "cq")
        page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
        page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
        for key, check in [
            ("q", "compareQmriActive && !compareMvActive"),
            ("v", "compareMvActive && !compareQmriActive"),
            ("q", "compareQmriActive && !compareMvActive"),
        ]:
            _focus_kb(page)
            page.keyboard.press(key)
            page.wait_for_function(f"() => {check}", timeout=5_000)
            page.wait_for_timeout(300)
            assert page.locator("#qmri-view-wrap canvas:visible").count() > 0

    def test_z_tiles_a_plain_volume(self, loaded_viewer, sid_3d):
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("z")
        page.wait_for_function("() => dim_z >= 0", timeout=5_000)
        _focus_kb(page)
        page.keyboard.press("z")
        page.wait_for_function("() => dim_z === -1", timeout=5_000)

    def test_wheel_scrolls_slices_in_qmri(self, page, server_url, client, tmp_path):
        """qMRI used to leave the wheel on the maps dim, so scrolling did nothing."""
        from conftest import register_array
        arr = np.random.default_rng(5).random((3, 12, 32, 32)).astype(np.float32)
        sid = register_array(client, arr, tmp_path, "qwheel")
        page.goto(f"{server_url}/?sid={sid}")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_function("() => qmriActive", timeout=5_000)
        page.wait_for_timeout(500)
        box = page.locator("#qmri-view-wrap canvas.qv-canvas").first.bounding_box()
        before = page.evaluate("() => JSON.stringify(indices)")
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for _ in range(3):
            page.mouse.wheel(0, 120)
            page.wait_for_timeout(100)
        page.wait_for_function("b => JSON.stringify(indices) !== b", arg=before, timeout=3_000)

    def test_same_named_arrays_get_folder_in_pane_titles(self, page, server_url, client, tmp_path, arr_3d):
        sids = []
        for folder in ("pat_a", "pat_b"):
            (tmp_path / folder).mkdir()
            path = tmp_path / folder / "t1.npy"
            np.save(path, arr_3d)
            sids.append(client.post("/load", json={"filepath": str(path), "name": "t1.npy"}).json()["sid"])
        page.goto(f"{server_url}/?sid={sids[0]}&compare_sid={sids[1]}&compare_sids={sids[1]}")
        page.wait_for_function(
            """() => { const t = document.getElementById('compare-left-title')?.textContent || '';
                       return t.includes('pat_a/t1.npy'); }""",
            timeout=15_000,
        )
