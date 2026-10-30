"""View-switch glides: each runs, lands and leaves nothing behind.

Frame-by-frame quality is checked by eye (.mex/patterns/animation-verify.md);
these guard the end states.

Run with:
    uv run pytest tests/test_glides.py -v
"""
from __future__ import annotations

import numpy as np

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


def test_complex_mode_and_axis_swap_dissolve(loaded_viewer, client, tmp_path):
    from conftest import register_array

    arr = (np.random.default_rng(1).standard_normal((6, 32, 40)) * np.exp(1j)).astype(np.complex64)
    page = loaded_viewer(register_array(client, arr, tmp_path, "cplx"))
    _settle(page)

    assert page.evaluate("_setComplexMode(1), !!_viewCrossfadeGhost")
    page.wait_for_function("() => !_viewCrossfadeGhost", timeout=5_000)
    assert page.evaluate("complexMode") == 1

    # The slice dim is selected: x swaps it onto the horizontal axis.
    old_x = page.evaluate("dim_x")
    assert page.evaluate("(commands['axis.assignX'].run({}, null), !!_viewCrossfadeGhost)")
    page.wait_for_function("() => !_viewCrossfadeGhost", timeout=5_000)
    assert page.evaluate("dim_x") != old_x
