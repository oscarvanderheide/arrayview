"""Browser coverage for the action-gated interactive tutorial."""

from __future__ import annotations

from urllib.parse import urlencode

import numpy as np
import pytest

from arrayview._tutorial import make_tutorial_arrays


def _register(client, tmp_path, name, data):
    path = tmp_path / f"{name}.npy"
    np.save(path, data)
    response = client.post("/load", json={"filepath": str(path), "name": name})
    response.raise_for_status()
    return response.json()["sid"]


@pytest.fixture
def tutorial_page(page, client, server_url, tmp_path):
    base, compare, overlay = make_tutorial_arrays()
    base_sid = _register(client, tmp_path, "tutorial", base)
    compare_sid = _register(client, tmp_path, "comparison-volume", compare)
    overlay_sid = _register(client, tmp_path, "Regions", overlay)
    query = urlencode(
        {
            "sid": base_sid,
            "compare_sid": compare_sid,
            "compare_sids": compare_sid,
            "overlay_sid": overlay_sid,
            "overlay_names": "Regions",
        }
    )
    page.goto(f"{server_url}/?{query}")
    page.wait_for_function(
        "() => document.body.classList.contains('tutorial-active')",
        timeout=15_000,
    )
    page.wait_for_timeout(500)
    return page


def test_tutorial_waits_for_real_actions_and_delays_compare(tutorial_page):
    page = tutorial_page
    state = page.evaluate(
        """() => ({
            title: document.getElementById('tutorial-title').textContent,
            compareActive,
            overlayVisibility: _overlayVisibility,
        })"""
    )
    assert state == {
        "title": "Start with the array",
        "compareActive": False,
        "overlayVisibility": "none",
    }

    page.get_by_role("button", name="Begin").click()
    page.wait_for_timeout(650)
    assert page.locator("#tutorial-title").inner_text() == "Move through slices"

    # A valid but unrelated viewer command must not earn tutorial progress.
    page.keyboard.press("b")
    page.wait_for_timeout(250)
    assert page.locator("#tutorial-title").inner_text() == "Move through slices"

    before = page.evaluate("indices[activeDim]")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(700)
    assert page.evaluate("indices[activeDim]") != before
    assert page.locator("#tutorial-title").inner_text() == "Choose another dimension"


def test_tutorial_section_controls_and_preferences(tutorial_page):
    page = tutorial_page
    page.get_by_role("button", name="Skip section").click()
    assert page.locator("#tutorial-section").text_content() == "Read the data"

    page.get_by_role("button", name="Back").click()
    assert page.locator("#tutorial-section").text_content() == "Explore"

    page.evaluate("_tutorialGo(20)")
    page.get_by_role("button", name="Open Preferences").click()
    page.wait_for_timeout(250)
    assert "visible" in (page.locator("#help-overlay").get_attribute("class") or "")
    assert page.locator(".help-panel[data-help-tab='preferences']").is_visible()
    assert page.get_by_role("button", name="Done exploring").is_visible()

    page.get_by_role("button", name="Done exploring").click()
    page.wait_for_timeout(650)
    assert page.locator("#tutorial-title").inner_text() == (
        "The complete map is always nearby"
    )


def test_tutorial_panel_uses_laptop_margin_without_covering_data(tutorial_page):
    page = tutorial_page
    page.set_viewport_size({"width": 1024, "height": 640})
    page.wait_for_timeout(250)
    bounds = page.evaluate(
        """() => ({
            panel: document.getElementById('tutorial-panel').getBoundingClientRect().toJSON(),
            canvas: document.getElementById('viewer').getBoundingClientRect().toJSON(),
        })"""
    )
    panel = bounds["panel"]
    canvas = bounds["canvas"]
    assert panel["left"] >= canvas["right"] or panel["right"] <= canvas["left"]
