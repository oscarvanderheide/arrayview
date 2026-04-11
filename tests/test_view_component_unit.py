"""Unit tests for the View Component System (Phase 1+).

Each test calls into the browser via page.evaluate() and asserts the return
value. No DOM assertions in this file — that's in test_view_component_integration.py.
"""
import pytest

pytestmark = pytest.mark.browser


def test_display_state_factory_defaults(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("() => makeDisplayState()")
    assert result["vmin"] is None
    assert result["vmax"] is None
    assert result["quantileIdx"] == -1
    assert result["cmapIdx"] == 0
    assert result["logScale"] is False
    assert result["complexMode"] == 0
    assert result["projectionMode"] == 0
    assert result["renderMode"] == "scalar"
    assert result["alphaThreshold"] == 0
    assert result["overlaySids"] == []
    assert result["overlayAlpha"] == 0.45


def test_display_state_overrides(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    result = page.evaluate("() => makeDisplayState({vmin: 5, logScale: true, cmapIdx: 3})")
    assert result["vmin"] == 5
    assert result["logScale"] is True
    assert result["cmapIdx"] == 3
    # other fields still defaulted
    assert result["vmax"] is None
    assert result["quantileIdx"] == -1
