from pathlib import Path

import pytest


VIEWER_HTML = Path(__file__).parent.parent / "src" / "arrayview" / "_viewer.html"


def test_colorbar_hover_bar_bin_mapping_branch_present():
    src = VIEWER_HTML.read_text()
    assert "_hoverBarValueAt(frac)" in src
    assert "_hoverBinIndexForPoint(frac, y, cssH)" in src
    assert "const isBarStrip = y >= Math.max(0, cssH - CB_COLLAPSED_H);" in src


def test_colorbar_hover_can_drive_lebesgue_without_histogram_expand():
    src = VIEWER_HTML.read_text()
    assert "let _colorbarLebesgueHover = false;" in src
    assert "return lebesgueMode || _colorbarLebesgueHover;" in src
    assert "if (this._onBinHover) this.ensureHistogramData();" in src
    assert "_startColorbarLebesgueHover();" in src
    assert "_stopColorbarLebesgueHover();" in src


def test_colorbar_hover_callbacks_cover_multiview_and_qmri_modes():
    src = VIEWER_HTML.read_text()
    assert "_drawLebesgueHighlightForColorbar(primaryCb, binIdx);" in src
    assert "_drawLebesgueHighlightForColorbar(window._mvColorBar, binIdx);" in src
    assert "_drawLebesgueHighlightForColorbar(_qvCb, binIdx);" in src
    assert "_drawLebesgueHighlightForColorbar(_cqCb, binIdx);" in src
    assert "if (cb?._lebesgueView) _drawLebesgueHighlightForQmriPane(cb._lebesgueView, binLow, binHigh);" in src
    assert "compareQmriActive" in src
    assert "compareMvActive" in src
    assert "qmriActive" in src
    assert "multiViewActive" in src


@pytest.mark.browser
def test_collapsed_colorbar_hover_shows_lebesgue_overlay(loaded_viewer, sid_2d):
    page = loaded_viewer(sid_2d)
    assert page.evaluate("() => !!window._primaryCb && !window._primaryCb.expanded")

    colorbar = page.locator("canvas#slim-cb")
    box = colorbar.bounding_box()
    assert box is not None
    x = box["x"] + box["width"] * 0.55
    y = box["y"] + box["height"] * 0.5
    page.mouse.move(x, y)
    page.wait_for_timeout(250)
    page.mouse.move(x + 4, y)
    page.wait_for_function(
        """() => {
            const c = document.getElementById('lebesgue-canvas');
            return !!c && getComputedStyle(c).display !== 'none';
        }""",
        timeout=5000,
    )


@pytest.mark.browser
def test_qmri_colorbar_hover_only_highlights_its_own_pane(loaded_viewer, sid_4d):
    page = loaded_viewer(sid_4d)
    page.evaluate("() => enterQmri()")
    page.wait_for_selector("#qmri-view-wrap.active .qv-canvas", timeout=5000)
    page.wait_for_timeout(500)
    assert page.locator("#qmri-view-wrap.active .qv-canvas").count() > 1

    colorbar = page.locator("#qmri-view-wrap.active .qv-cb-island canvas.cb-canvas").nth(0)
    box = colorbar.bounding_box()
    assert box is not None
    x = box["x"] + box["width"] * 0.55
    y = box["y"] + box["height"] * 0.5
    page.mouse.move(x, y)
    page.wait_for_timeout(250)
    page.mouse.move(x + 4, y)
    page.wait_for_function(
        """() => {
            const visible = [...document.querySelectorAll('.lebesgue-overlay')]
                .filter(el => getComputedStyle(el).display !== 'none');
            return visible.length === 1;
        }""",
        timeout=5000,
    )
