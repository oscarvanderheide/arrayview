"""View-switch glides: each runs, lands and leaves nothing behind.

Frame-by-frame quality is checked by eye (.mex/patterns/animation-verify.md);
these guard the end states.

Run with:
    uv run pytest tests/test_glides.py -v
"""
from __future__ import annotations

_GHOSTS = (
    "() => [...document.body.children]"
    ".filter(e => e.tagName === 'CANVAS' && e.style.position === 'fixed').length"
)
_SETTLED = (
    "() => typeof _frameDisplaySeq !== 'undefined' && _frameDisplaySeq > 0 && !_orthoGlideBusy"
    " && !document.getElementById('canvas-viewport').getAnimations().length"
)


def _settle(page):
    page.wait_for_function(_SETTLED, timeout=10_000)
    assert page.evaluate(_GHOSTS) == 0


def test_mosaic_glides_in_and_back(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    _settle(page)
    page.evaluate("indices[current_slice_dim] = 6; updateView()")
    page.wait_for_timeout(300)
    slice_dim = page.evaluate("current_slice_dim")

    page.keyboard.press("z")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert page.evaluate("dim_z") == slice_dim

    page.keyboard.press("z")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert page.evaluate("dim_z") == -1
    assert page.evaluate("current_slice_dim") == slice_dim
    assert page.evaluate("indices[current_slice_dim]") == 6


def test_mosaic_during_glide_waits_for_it(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    _settle(page)
    page.keyboard.press("z")
    page.keyboard.press("z")  # waits for the first glide to land, then glides back
    page.wait_for_function("() => dim_z === -1", timeout=10_000)
    _settle(page)
