"""ROI HUD interactions against a served viewer and real canvas gestures."""

from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.browser


@pytest.mark.parametrize("mode", ["normal", "multiview", "qmri"])
def test_roi_hud_linked_hover(page, server_url, sid_3d, sid_4d, mode):
    sid = sid_4d if mode == "qmri" else sid_3d
    page.goto(f"http://localhost:{urlsplit(server_url).port}/?sid={sid}")
    page.wait_for_function("() => lastImageData !== null")
    page.focus("#keyboard-sink")
    selector = "canvas#viewer"
    if mode == "multiview":
        page.keyboard.press("v")
        selector = ".mv-canvas"
    elif mode == "qmri":
        page.keyboard.press("q")
        selector = ".qv-canvas"
    page.locator(selector).first.wait_for(state="visible")
    page.wait_for_timeout(500)
    page.keyboard.press("Shift+R")
    page.wait_for_function("() => rectRoiMode")
    page.evaluate("() => _roiSetShape('rect')")
    canvas = page.locator(selector).first
    box = canvas.bounding_box()
    assert box
    x0, y0 = box["x"] + box["width"] * .52, box["y"] + box["height"] * .52
    x1, y1 = box["x"] + box["width"] * .78, box["y"] + box["height"] * .78
    page.mouse.move(x0, y0)
    page.mouse.down()
    page.mouse.move(x1, y1, steps=10)
    page.mouse.up()
    row = page.locator('.roi-hud-row[data-roi-idx="0"]')
    row.wait_for(state="visible")
    page.wait_for_timeout(500)
    assert page.locator(".roi-hud-row").count() == 1
    assert "mean" in page.locator("#roi-stats-hud").inner_text()
    assert "std" in page.locator("#roi-stats-hud").inner_text()
    expected = page.evaluate("""() => {
        const r = _rois[0];
        const s = qmriActive ? r.qmriStats?.find(q => q.qmriIdx === qmriViews[0].qmriIdx)?.stats : r.stats;
        return s ? [_roiStatsFmt(s.mean), _roiStatsFmt(s.std)] : null;
    }""")
    assert expected is not None, "Real ROI measurements should reach the HUD"
    assert row.locator("td").all_text_contents()[1:3] == expected
    selected = page.evaluate("() => _selectedRoiIdx")

    page.mouse.move((x0+x1)/2, (y0+y1)/2)
    page.wait_for_function("() => document.querySelector('.roi-hud-row.focused')?.dataset.roiIdx === '0'")
    assert not page.locator("#roi-hover-tooltip").is_visible()
    assert not page.locator("#roi-stats-tooltip").is_visible()
    page.mouse.move(2, 2)
    row.hover()
    page.wait_for_function("() => _roiHudHoverIdx === 0")
    assert "focused" in row.get_attribute("class")
    assert page.evaluate("() => _selectedRoiIdx") == selected
    output = Path("tests/smoke_output")
    output.mkdir(exist_ok=True)
    page.screenshot(path=str(output / f"roi_hud_{mode}.png"))
    hud_box = page.locator("#roi-stats-hud").bounding_box()
    assert hud_box and hud_box["x"] >= 0 and hud_box["y"] >= 0
    assert hud_box["x"] + hud_box["width"] <= page.viewport_size["width"]
    if mode == "normal":
        assert hud_box["x"] >= box["x"] + box["width"], "HUD should use available space outside the image"
    else:
        pane_boxes = page.locator(".mv-pane" if mode == "multiview" else ".qv-pane").all()
        pane_top = min(p.bounding_box()["y"] for p in pane_boxes if p.is_visible())
        if mode == "qmri":
            assert hud_box["y"] + hud_box["height"] <= pane_top, "HUD should sit above the complete panes, clear of their colorbars"
        else:
            assert hud_box["y"] >= pane_top or hud_box["y"] + hud_box["height"] <= pane_top
    page.mouse.move(2, 2)
    page.wait_for_function("() => _roiHudHoverIdx === -1")
    page.focus("#keyboard-sink")
    page.keyboard.press("Shift+R")
    page.locator("#roi-stats-hud").wait_for(state="hidden")
    page.keyboard.press("Shift+R")
    row.wait_for(state="visible")
    if mode == "normal":
        box = canvas.bounding_box()
        page.mouse.move(box["x"] + box["width"] * .18, box["y"] + box["height"] * .52)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * .35, box["y"] + box["height"] * .78, steps=10)
        page.mouse.up()
        page.wait_for_function("() => document.querySelectorAll('.roi-hud-row').length === 2")
        selected = page.evaluate("() => _selectedRoiIdx")
        row.hover()
        assert page.evaluate("() => _roiHudHoverIdx") == 0
        assert page.evaluate("() => _selectedRoiIdx") == selected == 1
        page.focus("#keyboard-sink")
        for theme in range(4):
            page.keyboard.press("T")
            page.wait_for_timeout(100)
            assert row.is_visible()
            page.screenshot(path=str(output / f"roi_hud_theme_{theme}.png"))
        page.keyboard.press("+")
        page.keyboard.press("+")
        page.wait_for_timeout(300)
        assert row.is_visible()
        page.screenshot(path=str(output / "roi_hud_zoom.png"))
    elif mode == "qmri":
        second = page.locator(selector).nth(1).bounding_box()
        page.mouse.move(second["x"] + second["width"] * .65, second["y"] + second["height"] * .65)
        page.wait_for_function("() => _roiHudMapIdx === qmriViews[1].qmriIdx")
        expected = page.evaluate("""() => {
            const s = _rois[0].qmriStats.find(q => q.qmriIdx === qmriViews[1].qmriIdx).stats;
            return [_roiStatsFmt(s.mean), _roiStatsFmt(s.std)];
        }""")
        assert row.locator("td").all_text_contents()[1:3] == expected

    # The header can relocate the HUD, and resize keeps it in the viewport.
    grip = page.locator(".roi-hud-grip")
    grip_box = grip.bounding_box()
    page.mouse.move(grip_box["x"] + 10, grip_box["y"] + 5)
    page.mouse.down()
    page.mouse.move(10, 10, steps=10)
    page.mouse.up()
    moved = page.locator("#roi-stats-hud").bounding_box()
    assert moved["x"] == 10 and moved["y"] == 10
    page.set_viewport_size({"width": 640, "height": 480})
    moved = page.locator("#roi-stats-hud").bounding_box()
    assert moved["x"] + moved["width"] <= 640
    assert moved["y"] + moved["height"] <= 480
    while page.evaluate("() => _rois.length"):
        count = page.locator(".roi-hud-row").count()
        page.locator(".roi-hud-delete").first.click()
        page.wait_for_function("count => _rois.length === count - 1", arg=count)
    page.locator("#roi-stats-hud").wait_for(state="hidden")
