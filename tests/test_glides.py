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


def test_compare_layout_glides(page, server_url, client, arr_3d, tmp_path):
    from conftest import register_array

    a = register_array(client, arr_3d, tmp_path, "a")
    b = register_array(client, arr_3d * 2, tmp_path, "b")
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}")
    page.wait_for_function("() => compareActive", timeout=10_000)
    _settle(page)
    before = page.evaluate("compareLayoutMode")
    page.keyboard.press("g")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert page.evaluate("compareLayoutMode") != before


def test_ortho_layout_glides(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    _settle(page)
    page.keyboard.press("v")
    page.wait_for_function("() => multiViewActive && !_orthoGlideBusy", timeout=10_000)
    page.wait_for_timeout(300)
    page.evaluate("_initOrthoLayoutPicker()")
    before = page.evaluate("_resolvedOrthoLayoutMode()")
    page.evaluate("_orthoLayoutPicker.cycleNext()")
    assert page.evaluate("_orthoGlideBusy")
    page.wait_for_function("() => !_orthoGlideBusy", timeout=10_000)
    assert page.evaluate(_GHOSTS) == 0
    assert page.evaluate("_resolvedOrthoLayoutMode()") != before


def test_stack_mosaic_glides_in_and_back(loaded_viewer, client, tmp_path):
    folder = tmp_path / "stack"
    folder.mkdir()
    for i in range(4):
        np.save(folder / f"vol_{i}.npy", (np.random.default_rng(i).standard_normal((5, 40, 40)) + i).astype(np.float32))
    sid = client.post("/load", json={"filepath": str(folder)}).json()["sid"]
    page = loaded_viewer(sid)
    _settle(page)

    page.keyboard.press("Shift+;")
    page.wait_for_selector("#inline-prompt.visible", timeout=5_000)
    page.fill("#inline-prompt-input", "2")
    page.keyboard.press("Enter")
    page.wait_for_function("() => stackMosaicActive && stackMosaicViews.length === 2", timeout=5_000)
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)

    page.keyboard.press("Shift+;")
    page.wait_for_function("() => !stackMosaicActive", timeout=5_000)
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
