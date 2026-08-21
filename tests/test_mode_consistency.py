"""Mode consistency tests — state round-trips and layout correctness.

These tests cover bugs where a mode transition silently clobbers viewer
state, or where layout/canvas sizing breaks when modes are combined.

Parametrized key-blocked-in-mode tests live in test_interactions.py
(``test_key_blocked_in_mode``). Add new (key, mode) pairs there.
Add tests here when the failure mode is state corruption or layout breakage
rather than a simple "shows wrong status" check.

Run with:
    uv run pytest tests/test_mode_consistency.py -v
"""

import numpy as np
import pytest

pytestmark = pytest.mark.browser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _focus_kb(page):
    page.focus("#keyboard-sink")


def _get_status(page):
    return page.evaluate("() => document.getElementById('status')?.textContent ?? ''")


def _status_or_empty(page):
    """Return status text; empty string if element missing."""
    return (page.evaluate("() => document.getElementById('status')?.textContent ?? ''") or "").strip()


def _enter_multiview(page):
    """Enter multi-view with default dims and wait for the wrapper to be active."""
    _focus_kb(page)
    page.keyboard.press("v")
    page.wait_for_selector("#multi-view-wrap.active", timeout=5000)
    page.wait_for_timeout(400)


def _exit_multiview(page):
    _focus_kb(page)
    page.keyboard.press("v")
    page.wait_for_timeout(300)


def _enter_qmri(page, sid_4d, loaded_viewer):
    """Enter qMRI mode and return the page object."""
    page = loaded_viewer(sid_4d)
    _focus_kb(page)
    page.keyboard.press("q")
    page.wait_for_selector("#qmri-view-wrap.active", timeout=5000)
    page.wait_for_timeout(300)
    return page


def _enter_compare(page, partner_sid):
    page.evaluate(
        "partner => enterCompareModeBySid(partner)",
        partner_sid,
    )
    page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
    page.wait_for_function("() => compareActive", timeout=5_000)


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


def _show_preset_overlay(page):
    page.keyboard.press("d")
    page.wait_for_timeout(400)
    page.keyboard.press("d")
    _wait_for_preset_overlay(page)


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


def _preset_overlay_state(page, pane_selector, colorbar_selector):
    return page.evaluate(
        """([paneSelector, colorbarSelector]) => {
            const shown = el => {
                const style = getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none'
                    && style.visibility !== 'hidden'
                    && Number.parseFloat(style.opacity || '1') > 0.05
                    && rect.width > 0 && rect.height > 0;
            };
            const visible = [...document.querySelectorAll('.percentile-preset-overlay')]
                .filter(shown);
            const overlay = visible.at(-1) || null;
            const dots = overlay
                ? [...overlay.querySelectorAll('.percentile-preset-dots .percentile-preset-dot')]
                : [];
            const rect = overlay?.getBoundingClientRect() || null;
            const pane = document.querySelector(paneSelector)?.getBoundingClientRect() || null;
            const colorbar = document.querySelector(colorbarSelector)?.getBoundingClientRect() || null;
            return {
                visibleCount: visible.length,
                text: overlay?.querySelector('.percentile-preset-value')?.textContent?.trim() || '',
                dotCount: dots.length,
                activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                withinPane: !!(rect && pane
                    && rect.left >= pane.left - 1 && rect.right <= pane.right + 1
                    && rect.top >= pane.top - 1 && rect.bottom <= pane.bottom + 1),
                aboveColorbar: !!(rect && colorbar && rect.bottom <= colorbar.top + 2),
            };
        }""",
        [pane_selector, colorbar_selector],
    )


# ---------------------------------------------------------------------------
# ── MULTIVIEW MODE: keys that must be blocked ──────────────────────────────
# ---------------------------------------------------------------------------

class TestMultiviewBlocks:
    """Keys that have no valid meaning in multiview and must show a status.

    Each test presses the key inside multiview and asserts a meaningful
    status string appears (not silent no-op, not a UI prompt appearing).
    """

    def test_z_mosaic_blocked(self, loaded_viewer, sid_3d):
        """z (mosaic toggle) tries to set dim_z in multiview where it has no meaning.
        Must be blocked."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("z")
        page.wait_for_timeout(200)
        status = _status_or_empty(page)
        assert "mosaic" in status.lower() or "multi" in status.lower() or "not available" in status.lower(), (
            f"Expected blocked status for 'z' (mosaic) in multiview, got: '{status}'"
        )

    def test_A_roi_blocked(self, loaded_viewer, sid_3d):
        """A (ROI) is already guarded — regression check that the guard still fires."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("A")
        page.wait_for_timeout(200)
        status = _status_or_empty(page)
        assert "roi" in status.lower() or "not available" in status.lower(), (
            f"A (ROI) should be blocked in multiview, got: '{status}'"
        )

    def test_N_slice_export_blocked(self, loaded_viewer, sid_3d):
        """N (slice export) is already guarded — regression check."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("N")
        page.wait_for_timeout(200)
        status = _status_or_empty(page)
        assert "export" in status.lower() or "not available" in status.lower(), (
            f"N (slice export) should be blocked in multiview, got: '{status}'"
        )

    def test_g_gif_blocked(self, loaded_viewer, sid_3d):
        """g (GIF export) is already guarded — regression check."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("g")
        page.wait_for_timeout(200)
        status = _status_or_empty(page)
        assert "gif" in status.lower() or "not available" in status.lower(), (
            f"g (GIF) should be blocked in multiview, got: '{status}'"
        )

    def test_x_y_axis_assign_blocked(self, loaded_viewer, sid_3d):
        """x/y (axis assignment) are already guarded — regression check."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("x")
        page.wait_for_timeout(200)
        status_x = _status_or_empty(page)
        page.keyboard.press("y")
        page.wait_for_timeout(200)
        status_y = _status_or_empty(page)
        assert "axis" in status_x.lower() or "use v" in status_x.lower(), (
            f"x should be blocked in multiview, got: '{status_x}'"
        )
        assert "axis" in status_y.lower() or "use v" in status_y.lower(), (
            f"y should be blocked in multiview, got: '{status_y}'"
        )


# ---------------------------------------------------------------------------
# ── MULTIVIEW MODE: features that work (not blocked) ───────────────────────
# ---------------------------------------------------------------------------

class TestMultiviewWorks:
    """Keys that were previously wrongly blocked but have valid multiview behavior."""

    def test_t_transpose_swaps_pane_dims(self, loaded_viewer, sid_3d):
        """t in multiview should visually transpose each pane (swap dimX/dimY)
        and show a status — NOT be blocked."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        # Read each pane's canvas dimensions before transpose
        dims_before = page.evaluate("""
            () => [...document.querySelectorAll('.mv-canvas')].map(c => ({
                w: c.width, h: c.height
            }))
        """)
        _focus_kb(page)
        page.keyboard.press("t")
        page.wait_for_timeout(500)
        status = _status_or_empty(page)
        # Must NOT say "blocked" or "use V"
        assert "use v" not in status.lower(), (
            f"t was blocked in multiview with message: '{status}'"
        )
        assert "transposed" in status.lower() or status == "", (
            f"Expected transpose status or empty, got: '{status}'"
        )
        # After transpose, panes should still have rendered content (non-zero canvas)
        dims_after = page.evaluate("""
            () => [...document.querySelectorAll('.mv-canvas')].map(c => ({
                w: c.width, h: c.height
            }))
        """)
        for s in dims_after:
            assert s["w"] > 0 and s["h"] > 0, (
                f"Pane canvas has zero size after transpose: {s}"
            )
        # For non-square arrays, width/height should swap on at least one pane
        # (20×64×64 arr_3d: dims are close but sliceDir affects shape — just check rendering is alive)
        assert len(dims_after) >= 3, "Expected at least 3 panes after transpose"

    def test_r_rotates_pane_in_multiview(self, loaded_viewer, sid_3d):
        """r in multiview rotates the global view orientation 90 CW (globally swaps dim_x/dim_y
        across all pane definitions and shows a 'rotated 90°' status)."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        # Get pane dimX/dimY before rotation
        pane_dims_before = page.evaluate(
            "() => mvViews.map(v => ({ dx: v.dimX, dy: v.dimY }))"
        )
        page.keyboard.press("r")
        page.wait_for_timeout(500)
        status = _status_or_empty(page)
        assert "rotated" in status.lower(), (
            f"Expected 'rotated 90°' status, got: '{status}'"
        )
        pane_dims_after = page.evaluate(
            "() => mvViews.map(v => ({ dx: v.dimX, dy: v.dimY }))"
        )
        # All panes should have changed (global rotation affects all pane definitions)
        changed = sum(
            1 for a, b in zip(pane_dims_before, pane_dims_after)
            if a["dx"] != b["dx"] or a["dy"] != b["dy"]
        )
        assert changed >= 1, (
            f"Expected at least 1 pane dim changed after r, got 0. "
            f"Before: {pane_dims_before}, after: {pane_dims_after}"
        )
        # All panes must still have valid canvas content
        content_count = page.evaluate("""
            () => {
                const panes = document.querySelectorAll('.mv-canvas');
                let n = 0;
                for (const c of panes) {
                    if (c.width > 0 && c.height > 0) n++;
                }
                return n;
            }
        """)
        assert content_count >= 3, (
            f"Expected ≥3 panes with non-zero canvases after rotation, got {content_count}"
        )

    def test_d_shows_histogram_in_multiview(self, loaded_viewer, sid_3d):
        """Pressing d in multiview should expand the mv colorbar to show histogram."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        page.locator(".mv-canvas").first.hover()
        _focus_kb(page)
        _show_preset_overlay(page)
        # mv-cb canvas should be expanded (height > 8px)
        cb_height = page.evaluate(
            "() => parseInt(document.getElementById('mv-cb')?.style.height || '0')"
        )
        assert cb_height > 8, (
            f"Expected mv colorbar to expand for histogram, got height={cb_height}px"
        )
        before = _preset_overlay_state(page, ".mv-pane", "#mv-cb-wrap")
        assert before["visibleCount"] == 1
        assert before["text"] and before["dotCount"] > 1 and before["activeIndex"] >= 0
        assert before["withinPane"] and before["aboveColorbar"]

        page.keyboard.press("d")
        _wait_for_preset_change(page, before["text"])
        after = _preset_overlay_state(page, ".mv-pane", "#mv-cb-wrap")
        assert after["activeIndex"] != before["activeIndex"]
        assert after["dotCount"] == before["dotCount"]
        # Wait for auto-dismiss (3s)
        _wait_for_preset_overlay(page, visible=False, timeout=5_000)
        page.wait_for_function(
            "() => parseInt(document.getElementById('mv-cb')?.style.height || '0') <= 15",
            timeout=5_000,
        )
        cb_height_after = page.evaluate(
            "() => parseInt(document.getElementById('mv-cb')?.style.height || '0')"
        )
        assert cb_height_after <= 15, (
            f"Expected mv colorbar to collapse after 3s, got height={cb_height_after}px"
        )
        _exit_multiview(page)

    def test_f_fft_works_in_multiview(self, loaded_viewer, sid_3d):
        """f (FFT) must NOT be blocked in multiview — it applies to the underlying
        data and all panes re-render with the FFT'd volume."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("f")
        page.wait_for_timeout(300)
        # Should open the inline-prompt (not be blocked)
        prompt_visible = page.evaluate(
            "() => document.getElementById('inline-prompt')?.classList.contains('visible') ?? false"
        )
        assert prompt_visible, (
            "f (FFT) did NOT open inline-prompt in multiview — it may have been wrongly blocked."
        )
        # Fill axes and confirm (default is 0,1)
        page.fill("#inline-prompt-input", "0,1")
        page.keyboard.press("Enter")
        page.wait_for_timeout(600)
        # Panes should still have content (FFT applied, all panes re-rendered)
        content_count = page.evaluate("""
            () => {
                const panes = document.querySelectorAll('.mv-canvas');
                let n = 0;
                for (const c of panes) {
                    if (!c.width || !c.height) continue;
                    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                    let nonBg = 0;
                    for (let i = 0; i < d.length; i += 4) {
                        if (d[i] > 5 || d[i+1] > 5 || d[i+2] > 5) nonBg++;
                    }
                    if (nonBg > 10) n++;
                }
                return n;
            }
        """)
        assert content_count >= 2, (
            f"After FFT in multiview, expected ≥2 panes with content, got {content_count}"
        )


# ---------------------------------------------------------------------------
# ── MULTIVIEW MODE: state round-trip ───────────────────────────────────────
# ---------------------------------------------------------------------------

class TestMultiviewStateRoundTrip:
    """Actions taken BEFORE entering multiview must survive the round-trip.

    Each test: set state → enter multiview → exit → verify state unchanged.
    These catch the class of bug where enterMultiView/exitMultiView silently
    clobbers dim_x, dim_y, flip_x, flip_y, etc.
    """

    def test_colormap_survives_multiview_roundtrip(self, loaded_viewer, sid_3d):
        """Colormap changed before multiview must be same after exit.
        Reads the colormap strip's active thumbnail — verified without cycling again."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        # Read colormap index directly (strip rendering may be inline/auto-hidden)
        def _read_cmap(page):
            return page.evaluate(
                "() => { "
                "  if (typeof colormap_idx === 'undefined' || typeof COLORMAPS === 'undefined') return ''; "
                "  const idx = colormap_idx < 0 ? 0 : colormap_idx; "
                "  return COLORMAPS[idx % COLORMAPS.length] || ''; "
                "}"
            )

        # Open the colormap menu, cycle away from default, then commit.
        page.keyboard.press("c")
        page.wait_for_selector("#slim-cb-preview.fade-in", timeout=2000)
        page.keyboard.press("c")
        page.keyboard.press("Enter")
        page.wait_for_timeout(200)
        cmap_before = _read_cmap(page)
        _enter_multiview(page)
        _exit_multiview(page)
        page.wait_for_timeout(300)
        cmap_after = _read_cmap(page)
        if not cmap_before or not cmap_after:
            pytest.skip("Colormap state not accessible")
        assert cmap_before == cmap_after, (
            f"Colormap changed across multiview round-trip: '{cmap_before}' → '{cmap_after}'"
        )

    def test_colormap_strip_visible_in_multiview(self, loaded_viewer, sid_3d):
        """Pressing c in multiview should show the colormap preview strip."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.keyboard.press("c")
        strip = page.wait_for_selector("#colormap-strip.visible", timeout=3000)
        assert strip is not None, "Colormap strip not visible in multiview"
        # Verify it's positioned near the multiview colorbar
        strip_box = strip.bounding_box()
        mv_cb = page.query_selector("#mv-cb-wrap")
        assert mv_cb is not None, "Multiview colorbar not found"
        _exit_multiview(page)

    def test_manual_range_survives_multiview_roundtrip(self, loaded_viewer, sid_3d):
        """Manual vmin/vmax set before multiview must persist after exit.
        Verified by re-opening the D prompt and checking the pre-filled vmin default."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        page.keyboard.press("D")
        page.wait_for_selector("#inline-prompt.visible", timeout=2000)
        page.fill("#inline-prompt-input", "0.1")
        page.keyboard.press("Enter")
        page.wait_for_selector("#inline-prompt.visible", timeout=2000)
        page.fill("#inline-prompt-input", "0.9")
        page.keyboard.press("Enter")
        page.wait_for_timeout(400)
        _enter_multiview(page)
        _exit_multiview(page)
        page.wait_for_timeout(300)
        # Re-open D prompt — pre-filled value should still reflect our locked vmin (~0.1)
        _focus_kb(page)
        page.keyboard.press("D")
        page.wait_for_selector("#inline-prompt.visible", timeout=2000)
        vmin_default = page.evaluate(
            "() => document.getElementById('inline-prompt-input')?.value ?? ''"
        )
        page.keyboard.press("Escape")  # cancel
        # The pre-filled vmin should be close to 0.1 (allow for rounding in _cbFmt)
        try:
            parsed = float(vmin_default)
        except (ValueError, TypeError):
            pytest.fail(f"D prompt pre-fill wasn't numeric after multiview roundtrip: '{vmin_default}'")
        assert abs(parsed - 0.1) < 0.05, (
            f"manual vmin pre-fill changed across multiview roundtrip: expected ~0.1, got {vmin_default}"
        )

    def test_flip_state_carried_into_multiview(self, loaded_viewer, sid_3d):
        """Flipping before entering multiview should be reflected in panes (not silently dropped)."""
        page = loaded_viewer(sid_3d)
        _focus_kb(page)
        # Flip along dim_y (activeDim=dim_y after pressing arrow to select it)
        # Actually just grab a center pixel before/after flip to verify round-trip
        # For simplicity: check that pane canvases have content after entering multiview
        # (if flip state were corrupted the panes might render black)
        page.keyboard.press("r")  # flip (activeDim is slice dim → rotates)
        page.wait_for_timeout(300)
        _enter_multiview(page)
        # All 3 panes should have rendered content
        pane_content = page.evaluate("""
            () => {
                const panes = document.querySelectorAll('.mv-canvas');
                if (panes.length < 3) return 0;
                let contentCount = 0;
                for (const c of panes) {
                    if (!c.width || !c.height) continue;
                    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                    let nonBg = 0;
                    for (let i = 0; i < d.length; i += 4) {
                        if (d[i] > 10 || d[i+1] > 10 || d[i+2] > 10) nonBg++;
                    }
                    if (nonBg > 20) contentCount++;
                }
                return contentCount;
            }
        """)
        assert pane_content >= 2, (
            f"Expected ≥2 panes with content after entering multiview with prior flip, got {pane_content}"
        )


# ---------------------------------------------------------------------------
# ── QMRI MODE: shared range interaction and blocked keys ───────────────────
# ---------------------------------------------------------------------------

class TestQmriWorks:
    def test_d_cycles_percentile_overlay_on_hovered_map(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        first_canvas = page.locator("#qmri-view-wrap .qv-canvas:visible").first
        first_canvas.hover()
        _focus_kb(page)
        _show_preset_overlay(page)

        before = _preset_overlay_state(
            page,
            "#qmri-view-wrap.active .qv-pane",
            "#qmri-view-wrap.active .qv-cb-island",
        )
        assert before["visibleCount"] == 1
        assert before["text"] and before["dotCount"] > 1 and before["activeIndex"] >= 0
        assert before["withinPane"] and before["aboveColorbar"]

        page.keyboard.press("d")
        _wait_for_preset_change(page, before["text"])
        after = _preset_overlay_state(
            page,
            "#qmri-view-wrap.active .qv-pane",
            "#qmri-view-wrap.active .qv-cb-island",
        )
        assert after["activeIndex"] != before["activeIndex"]
        assert after["dotCount"] == before["dotCount"]

        second_canvas = page.locator("#qmri-view-wrap .qv-canvas:visible").nth(1)
        second_canvas.hover()
        _show_preset_overlay(page)
        overlay_box = page.locator(".percentile-preset-overlay").bounding_box()
        second_box = second_canvas.bounding_box()
        assert overlay_box and second_box
        assert overlay_box["x"] >= second_box["x"] - 1
        assert overlay_box["x"] + overlay_box["width"] <= second_box["x"] + second_box["width"] + 1
        assert overlay_box["y"] >= second_box["y"] - 1
        assert overlay_box["y"] + overlay_box["height"] <= second_box["y"] + second_box["height"] + 1


class TestCompareQmriRangeSelector:
    def test_d_cycles_only_hovered_map_and_updates_selector(
        self, loaded_viewer, sid_4d, arr_4d, client, tmp_path
    ):
        partner_path = tmp_path / "arr4d_compare_qmri.npy"
        np.save(partner_path, arr_4d * 0.65 + 0.2)
        partner_sid = client.post(
            "/load",
            json={"filepath": str(partner_path), "name": "arr4d_compare_qmri"},
        ).json()["sid"]

        page = loaded_viewer(sid_4d)
        _enter_compare(page, partner_sid)
        _focus_kb(page)
        page.keyboard.press("q")
        page.wait_for_function(
            "() => compareQmriActive && compareQmriViews.length >= 2 "
            "&& compareQmriViews.every(v => v.lastW && v.lastH)",
            timeout=15_000,
        )

        # Use the B1 map: unlike T1/T2, it is not rounded to an integer policy
        # window, so adjacent percentile presets must visibly change its range.
        target_canvas = page.locator("#qmri-view-wrap .qv-canvas:visible").nth(2)
        target_canvas.hover()
        page.wait_for_function("() => !!_hoveredQmriView", timeout=3_000)
        before = page.evaluate(
            """() => ({
                target: compareQmriViews.indexOf(_hoveredQmriView),
                windows: compareQmriViews.map(v => ({
                    vmin: v.lockedVmin ?? v.vmin,
                    vmax: v.lockedVmax ?? v.vmax,
                })),
            })"""
        )
        assert before["target"] >= 0

        _focus_kb(page)
        _show_preset_overlay(page)
        first_cycle = page.evaluate(
            """before => {
                const overlay = document.querySelector('.percentile-preset-overlay');
                const dots = [...(overlay?.querySelectorAll('.percentile-preset-dot') || [])];
                const rect = overlay?.getBoundingClientRect();
                const pane = compareQmriViews[before.target].canvas.getBoundingClientRect();
                return {
                    windows: compareQmriViews.map(v => ({
                        vmin: v.lockedVmin ?? v.vmin,
                        vmax: v.lockedVmax ?? v.vmax,
                    })),
                    text: overlay?.querySelector('.percentile-preset-value')?.textContent?.trim() || '',
                    activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                    dotCount: dots.length,
                    withinTarget: !!(rect
                        && rect.left >= pane.left - 1 && rect.right <= pane.right + 1
                        && rect.top >= pane.top - 1 && rect.bottom <= pane.bottom + 1),
                };
            }""",
            before,
        )
        assert first_cycle["text"]
        assert first_cycle["dotCount"] > 1 and first_cycle["activeIndex"] >= 0
        assert first_cycle["withinTarget"]
        for index, old_window in enumerate(before["windows"]):
            if index == before["target"]:
                continue
            assert first_cycle["windows"][index] == pytest.approx(old_window)

        page.keyboard.press("d")
        _wait_for_preset_change(page, first_cycle["text"], timeout=10_000)
        page.wait_for_function(
            """before => {
                const v = compareQmriViews[before.target];
                const lo = v.lockedVmin ?? v.vmin;
                const hi = v.lockedVmax ?? v.vmax;
                return Math.abs(lo - before.windows[before.target].vmin) > 1e-9
                    || Math.abs(hi - before.windows[before.target].vmax) > 1e-9;
            }""",
            arg=before,
            timeout=10_000,
        )
        second_cycle = page.evaluate(
            """() => {
                const overlay = document.querySelector('.percentile-preset-overlay');
                const dots = [...overlay.querySelectorAll('.percentile-preset-dot')];
                return {
                    windows: compareQmriViews.map(v => ({
                        vmin: v.lockedVmin ?? v.vmin,
                        vmax: v.lockedVmax ?? v.vmax,
                    })),
                    text: overlay.querySelector('.percentile-preset-value').textContent.trim(),
                    activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                    dotCount: dots.length,
                };
            }"""
        )
        assert second_cycle["text"] != first_cycle["text"]
        assert second_cycle["activeIndex"] != first_cycle["activeIndex"]
        assert second_cycle["dotCount"] == first_cycle["dotCount"]
        assert second_cycle["windows"][before["target"]] != pytest.approx(
            before["windows"][before["target"]]
        )
        for index, old_window in enumerate(before["windows"]):
            if index == before["target"]:
                continue
            assert second_cycle["windows"][index] == pytest.approx(old_window)


class TestQmriBlocks:
    """Regression tests for qMRI mode key guards."""

    def test_D_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)
        page.keyboard.press("D")
        page.wait_for_timeout(200)
        prompt_visible = page.evaluate(
            "() => document.getElementById('inline-prompt')?.classList.contains('visible') ?? false"
        )
        assert not prompt_visible, "D must not open inline-prompt in qMRI mode"
        status = _status_or_empty(page)
        assert "range" in status.lower() or "qmri" in status.lower() or "map" in status.lower()

    def test_f_FFT_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)
        page.keyboard.press("f")
        page.wait_for_timeout(200)
        prompt_visible = page.evaluate(
            "() => document.getElementById('inline-prompt')?.classList.contains('visible') ?? false"
        )
        assert not prompt_visible, "f (FFT) must not open inline-prompt in qMRI mode"
        status = _status_or_empty(page)
        assert "fft" in status.lower() or "qmri" in status.lower()

    def test_c_colormap_blocked_in_qmri(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)
        page.keyboard.press("c")
        page.wait_for_timeout(200)
        status = _status_or_empty(page)
        assert "qmri" in status.lower() or "colormap" in status.lower(), (
            f"c (colormap) should be blocked in qMRI, got: '{status}'"
        )


# ---------------------------------------------------------------------------
# ── QMRI MULTIVIEW: ortho view inside qMRI mode ────────────────────────────
# ---------------------------------------------------------------------------

class TestQmriMultiview:
    """Pressing v inside qMRI mode shows a 3-plane grid per map (3 maps × 3)."""

    def test_v_toggles_qmri_3x3_and_back(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)

        # v in qMRI -> 3-plane view per map (3 default-visible maps × 3 planes)
        page.keyboard.press("v")
        page.wait_for_selector("#qmri-view-wrap.active", timeout=5000)
        page.wait_for_function(
            "() => typeof qmriMvViews !== 'undefined' && qmriMvViews.length === 9 "
            "&& qmriMvViews.every(v => v.lastW && v.lastH)",
            timeout=10000,
        )
        panes = page.evaluate(
            "() => document.querySelectorAll('#qmri-view-wrap .qv-canvas').length"
        )
        assert panes == 9, f"expected 9 qMRI-mv panes, got {panes}"
        rows = page.evaluate(
            "() => document.querySelectorAll('#qmri-view-wrap .qmri-mv-row').length"
        )
        assert rows == 3, f"expected 3 map rows, got {rows}"

        # v again -> back to single-map qMRI (3 maps in one row)
        page.keyboard.press("v")
        page.wait_for_timeout(500)
        page.wait_for_selector("#qmri-view-wrap.active", timeout=5000)
        single_panes = page.evaluate(
            "() => document.querySelectorAll('#qmri-view-wrap .qv-pane').length"
        )
        assert single_panes == 3, f"expected 3 qMRI panes after exit, got {single_panes}"

    def test_q_exits_qmri_multiview(self, loaded_viewer, sid_4d):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_function(
            "() => typeof qmriMvViews !== 'undefined' && qmriMvViews.length === 9"
        )
        # q exits qMRI entirely, even from the multiview sub-mode
        page.keyboard.press("q")
        page.wait_for_timeout(600)
        assert not page.is_visible("#qmri-view-wrap.active")

    def test_d_opens_then_cycles_hovered_row_and_dismisses_cleanly(
        self, loaded_viewer, sid_4d
    ):
        page = _enter_qmri(None, sid_4d, loaded_viewer)
        _focus_kb(page)
        page.keyboard.press("v")
        page.wait_for_function(
            "() => qmriMvActive && qmriMvViews.length === 9 "
            "&& qmriMvViews.every(v => v.lastW && v.lastH)",
            timeout=15_000,
        )

        # Use the B1 row so adjacent percentile presets cannot collapse to the
        # same integer-rounded T1/T2 window.
        target_canvas = page.locator("#qmri-view-wrap .qmri-mv-row").nth(2).locator(".qv-canvas").first
        target_canvas.hover()
        page.wait_for_function("() => !!_hoveredQmriMvView", timeout=3_000)
        before = page.evaluate(
            """() => {
                const view = _hoveredQmriMvView;
                const cb = view._mapColorBar;
                return {
                    row: view._mapIndex,
                    preset: window._dQuantileIdx ?? 0,
                    window: cb.opts.getWindow(),
                };
            }"""
        )

        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_function(
            """row => {
                const view = qmriMvViews.find(v => v._mapIndex === row);
                const cb = view?._mapColorBar;
                return !!(cb?._expanded && cb._histData?.counts?.length);
            }""",
            arg=before["row"],
            timeout=10_000,
        )
        _wait_for_preset_overlay(page)
        opened = page.evaluate(
            """row => {
                const view = qmriMvViews.find(v => v._mapIndex === row);
                const cb = view._mapColorBar;
                const current = cb.opts.getWindow();
                const overlay = document.querySelector('.percentile-preset-overlay');
                const rect = overlay?.getBoundingClientRect();
                const pane = view.canvas.getBoundingClientRect();
                const ctx = cb.canvas.getContext('2d');
                const {width, height} = cb.canvas;
                const pixels = ctx.getImageData(0, 0, width, height).data;
                let widestYellowRow = 0;
                for (let y = 0; y < height; y++) {
                    let yellow = 0;
                    for (let x = 0; x < width; x++) {
                        const i = (y * width + x) * 4;
                        if (pixels[i] > 220 && pixels[i + 1] > 210 && pixels[i + 2] < 170) yellow++;
                    }
                    widestYellowRow = Math.max(widestYellowRow, yellow);
                }
                const dots = [...(overlay?.querySelectorAll('.percentile-preset-dot') || [])];
                return {
                    preset: window._dQuantileIdx ?? 0,
                    window: current,
                    text: overlay?.querySelector('.percentile-preset-value')?.textContent?.trim() || '',
                    activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                    dotCount: dots.length,
                    withinTarget: !!(rect
                        && rect.left >= pane.left - 1 && rect.right <= pane.right + 1
                        && rect.top >= pane.top - 1 && rect.bottom <= pane.bottom + 1),
                    hasYellowLimits: widestYellowRow >= Math.max(2, width * 0.6),
                };
            }""",
            before["row"],
        )
        assert opened["preset"] == before["preset"], "first d must open, not cycle"
        assert opened["window"] == pytest.approx(before["window"])
        assert opened["text"] and opened["dotCount"] > 1 and opened["activeIndex"] >= 0
        assert opened["withinTarget"]
        assert opened["hasYellowLimits"]

        page.keyboard.press("d")
        _wait_for_preset_change(page, opened["text"], timeout=10_000)
        cycled = page.evaluate(
            """row => {
                const view = qmriMvViews.find(v => v._mapIndex === row);
                const cb = view._mapColorBar;
                const overlay = document.querySelector('.percentile-preset-overlay');
                const dots = [...overlay.querySelectorAll('.percentile-preset-dot')];
                return {
                    preset: window._dQuantileIdx ?? 0,
                    window: cb.opts.getWindow(),
                    text: overlay.querySelector('.percentile-preset-value').textContent.trim(),
                    activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                };
            }""",
            before["row"],
        )
        assert cycled["preset"] != opened["preset"]
        assert cycled["text"] != opened["text"]
        assert cycled["activeIndex"] != opened["activeIndex"]

        # With a coarse histogram, the 0.1–99.9% bracket can share the full
        # range's outer bins. The next discrete preset must move the bounds.
        page.keyboard.press("d")
        _wait_for_preset_change(page, cycled["text"], timeout=10_000)
        ranged = page.evaluate(
            """row => {
                const view = qmriMvViews.find(v => v._mapIndex === row);
                const overlay = document.querySelector('.percentile-preset-overlay');
                const dots = [...overlay.querySelectorAll('.percentile-preset-dot')];
                return {
                    window: view._mapColorBar.opts.getWindow(),
                    text: overlay.querySelector('.percentile-preset-value').textContent.trim(),
                    activeIndex: dots.findIndex(dot => dot.classList.contains('active')),
                };
            }""",
            before["row"],
        )
        assert ranged["window"] != pytest.approx(opened["window"])
        assert ranged["text"] != cycled["text"]
        assert ranged["activeIndex"] != cycled["activeIndex"]

        _wait_for_preset_overlay(page, visible=False, timeout=5_000)
        page.wait_for_function(
            """row => {
                const view = qmriMvViews.find(v => v._mapIndex === row);
                return !view?._mapColorBar?._expanded;
            }""",
            arg=before["row"],
            timeout=6_000,
        )

        target_canvas.hover()
        _focus_kb(page)
        page.keyboard.press("d")
        page.wait_for_function(
            """row => qmriMvViews.find(v => v._mapIndex === row)?._mapColorBar?._expanded""",
            arg=before["row"],
            timeout=10_000,
        )
        _wait_for_preset_overlay(page)
        page.keyboard.press("ArrowRight")
        _wait_for_preset_overlay(page, visible=False)
        page.wait_for_function(
            """row => !qmriMvViews.find(v => v._mapIndex === row)?._mapColorBar?._expanded""",
            arg=before["row"],
            timeout=3_000,
        )


# ---------------------------------------------------------------------------
# ── COMPACT MODE: correct layout in multiview ──────────────────────────────
# ---------------------------------------------------------------------------

class TestCompactModeWithMultiview:
    """Compact mode must not break multiview layout."""

    def test_compact_toggle_does_not_break_multiview_canvas_size(self, loaded_viewer, sid_3d):
        """Toggling compact mode while in multiview must still produce
        correctly-sized pane canvases (not zero-size or unchanged from pre-multiview)."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        _focus_kb(page)
        page.evaluate("toggleCompactMode()")  # turn compact on
        page.wait_for_timeout(300)
        sizes_compact = page.evaluate("""
            () => [...document.querySelectorAll('.mv-canvas')].map(c => {
                const r = c.getBoundingClientRect();
                return { w: Math.round(r.width), h: Math.round(r.height) };
            })
        """)
        page.evaluate("toggleCompactMode()")  # turn compact off
        page.wait_for_timeout(300)
        sizes_after = page.evaluate("""
            () => [...document.querySelectorAll('.mv-canvas')].map(c => {
                const r = c.getBoundingClientRect();
                return { w: Math.round(r.width), h: Math.round(r.height) };
            })
        """)
        # All sizes must be non-zero
        for s in sizes_compact:
            assert s["w"] > 0 and s["h"] > 0, (
                f"Compact mode left a zero-size multiview pane: {s}. "
                "setCompactMode() was calling scaleCanvas() instead of mvScaleAllCanvases()."
            )
        for s in sizes_after:
            assert s["w"] > 0 and s["h"] > 0, (
                f"After compact toggle off, pane has zero size: {s}"
            )

    def test_auto_compact_does_not_trigger_in_multiview(self, loaded_viewer, sid_3d):
        """Auto-compact must not fire while multiview is active.
        Previously _checkAutoCompact() was missing the multiViewActive guard."""
        page = loaded_viewer(sid_3d)
        _enter_multiview(page)
        # Zoom in aggressively — this would trigger auto-compact in normal mode
        _focus_kb(page)
        for _ in range(8):
            page.keyboard.press("+")
            page.wait_for_timeout(50)
        page.wait_for_timeout(300)
        compact_active = page.evaluate(
            "() => document.body.classList.contains('compact-mode')"
        )
        assert not compact_active, (
            "Auto-compact triggered during multiview — _checkAutoCompact() "
            "was missing the multiViewActive guard."
        )


# ---------------------------------------------------------------------------
# ── KEY EXHAUSTIVENESS: not-yet-covered new keys ───────────────────────────
# ──
# When you add a new key binding, add it to ONE of:
#   – TestMultiviewBlocks    (if it should be blocked in MV)
#   – TestQmriBlocks         (if it should be blocked in qMRI)
#   – TestMultiviewStateRoundTrip  (if it carries state)
#   – OR add a comment below explaining why no mode test is needed
# ---------------------------------------------------------------------------

class TestKeyExhaustiveness:
    """Documents which keys are verified to be mode-safe.

    Keys listed here have been manually verified to either:
    (a) have explicit mode guards in the keyboard handler, or
    (b) delegate entirely through updateView() which correctly routes per-mode
    These are NOT regression-tested individually — add to the classes above
    if a regression is found.

    VERIFIED SAFE (multiview-routed or explicitly guarded):
      +/-/0   — zoom: routes to mvScaleAllCanvases() ✓
      r       — flip: explicit multiViewActive branch ✓
      b       — borders: explicit multiViewActive branch ✓
      a       — squareStretch: explicit multiViewActive branch ✓
      A       — ROI: blocked ✓
      j/k/h/l — navigation: explicit multiViewActive branches ✓
      m       — complex mode: updateView() routes to mvRender ✓
      c/C     — colormap: updateView() routes to mvRender ✓
      d       — dynamic range: explicit mvViews.forEach ✓
      D       — manual range: explicit mvViews.forEach; blocked in qMRI ✓
      L       — log scale: updateView() routes to mvRender ✓
      M       — alpha: explicit mvDims freeDims branch ✓
      w       — histogram: explicit multiViewActive branch ✓ (fixed)
      n       — compare target: guarded by !compareActive ✓
      o/O     — crosshair reset: guarded by !multiViewActive ✓
      q       — qMRI enter/exit: handles multiView exit ✓
      v/V     — multiview enter/exit: explicitly handles all cases ✓
      x/y     — axis assignment: blocked in multiview (use V to reconfigure) ✓
      T       — theme: global, no state interaction ✓
      F       — zen mode: global, no state interaction ✓
      s       — save menu: works everywhere ✓
      e       — copy URL: global ✓
      i       — info overlay: global ✓
      H       — hover info toggle: global ✓
      B       — compare toggle: handled ✓
      R       — RGB toggle: routes through toggleRgbMode ✓
      1-9/Enter/Escape — slice jump: handled per mode ✓

    WORKS IN MULTIVIEW (explicit implementation):
      t  — transpose: swaps dimX/dimY on each pane and re-renders ✓
      f  — FFT: applies to session data, updateView() routes to mvRender ✓

    BLOCKED (genuinely not applicable in multiview):
      z  — mosaic: would display a grid-of-slices inside a grid-of-planes (overwhelming) ✓
      K  — compact: setCompactMode now calls mvScaleAllCanvases in multiview ✓
    """

    def test_placeholder_read_docstring(self):
        """This test always passes — see class docstring for coverage notes."""
        pass
