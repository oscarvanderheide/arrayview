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
    # Both slices drawn first: without them there is nothing to glide and it fades.
    page = _compare_page(page, server_url, client, arr_3d, tmp_path)
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


def test_mosaic_glide_does_not_double_the_colorbar_values(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    _settle(page)
    page.keyboard.press("z")
    copies = page.evaluate(
        "() => [...document.body.children].filter(e => e.classList.contains('slim-cb-val')).length"
    )
    assert copies == 0
    _settle(page)


_UNDERLINE_TRACE = """() => { window.__u = []; const t0 = performance.now(); const f = () => {
    window.__u.push([performance.now() - t0, document.getElementById('dim-active-underline').getBoundingClientRect().left]);
    if (performance.now() - t0 < 900) requestAnimationFrame(f);
}; f(); }"""


def test_mosaic_underline_slides_to_new_active_dim(loaded_viewer, sid_4d):
    page = loaded_viewer(sid_4d)
    _settle(page)
    page.evaluate("_dimbarExtentPinned = false; document.getElementById('info').classList.remove('dimbar-expanded')")
    page.wait_for_timeout(400)

    for _ in range(2):
        page.evaluate(_UNDERLINE_TRACE)
        page.keyboard.press("z")
        page.wait_for_timeout(1000)
        trace = page.evaluate("window.__u")
        start, end = trace[0][1], trace[-1][1]
        assert abs(end - start) > 5
        assert page.evaluate(
            "document.querySelector('#info .active-dim.dim-label').getBoundingClientRect().left"
        ) == end
        moving = [t for (t, x), (_, prev) in zip(trace[1:], trace) if abs(x - prev) > 0.1]
        steps = [abs(x - prev) for (_, x), (_, prev) in zip(trace[1:], trace)]
        print(f"slide took {moving[-1] - moving[0]:.0f} ms, biggest step {max(steps):.1f} of {abs(end - start):.1f} px")
        assert 150 < moving[-1] - moving[0] < 400
        assert max(steps) < abs(end - start) * 0.5
        _settle(page)


def _compare_page(page, server_url, client, arr, tmp_path):
    from conftest import register_array

    a = register_array(client, arr, tmp_path, "a")
    b = register_array(client, arr * 2, tmp_path, "b")
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}")
    page.wait_for_function(
        "() => compareActive && !compareRendering && compareFrames.filter(Boolean).length >= 2", timeout=10_000
    )
    _settle(page)
    return page


_ROW_LAYOUT = (
    "views => views.map(v => { const r = v.canvas.getBoundingClientRect();"
    " return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; })"
)


def _all_drawn(views):
    return f"() => {views}.length && {views}.every(v => v.lastImageData)"


def test_compare_ortho_glides_in_and_back(page, server_url, client, arr_3d, tmp_path):
    page = _compare_page(page, server_url, client, arr_3d, tmp_path)
    page.keyboard.press("v")
    assert page.evaluate("_orthoGlideBusy")
    # Every row has a pane showing the slice that was on screen, ready for it to land in.
    assert page.evaluate("compareMvViews.filter(_isOrthoSlicePlane).length") == 2
    # The panes have their places before any frame is back.
    first = page.evaluate(f"({_ROW_LAYOUT})(compareMvViews)")
    assert all(r[2] > 0 for r in first)
    _settle(page)
    page.wait_for_function(_all_drawn("compareMvViews"))
    assert page.evaluate(f"({_ROW_LAYOUT})(compareMvViews)") == first

    page.keyboard.press("v")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert not page.evaluate("compareMvActive")
    assert page.evaluate("_comparePanesShown()")


def test_compare_qmri_glides_in_and_back(page, server_url, client, tmp_path):
    arr = np.random.default_rng(2).random((4, 8, 32, 40)).astype(np.float32)
    page = _compare_page(page, server_url, client, arr, tmp_path)
    page.keyboard.press("q")
    assert page.evaluate("_orthoGlideBusy")
    first = page.evaluate(f"({_ROW_LAYOUT})(compareQmriViews)")
    assert all(r[2] > 0 for r in first)
    _settle(page)
    page.wait_for_function(_all_drawn("compareQmriViews"))
    assert page.evaluate(f"({_ROW_LAYOUT})(compareQmriViews)") == first
    # All panes are visible and nothing is left holding them back.
    page.wait_for_function(
        "() => compareQmriViews.every(v => getComputedStyle(v.canvas.closest('.qv-pane')).opacity === '1'"
        " && getComputedStyle(v.canvas).opacity === '1')",
        timeout=3_000,
    )

    page.keyboard.press("q")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert not page.evaluate("compareQmriActive")


def test_compare_ortho_and_qmri_together_is_refused(page, server_url, client, tmp_path):
    arr = np.random.default_rng(3).random((4, 8, 32, 40)).astype(np.float32)
    page = _compare_page(page, server_url, client, arr, tmp_path)
    page.keyboard.press("v")
    _settle(page)
    page.keyboard.press("q")
    page.wait_for_timeout(300)
    assert page.evaluate("compareMvActive && !compareQmriActive")
    assert "not available" in page.evaluate("document.getElementById('status').textContent")


def test_split_ortho_glides_in_and_back(loaded_viewer, sid_4d):
    page = loaded_viewer(sid_4d)
    _settle(page)
    page.keyboard.press("Shift+S")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    _settle(page)
    page.keyboard.press("v")
    assert page.evaluate("_orthoGlideBusy")
    _settle(page)
    assert page.evaluate("compareMvActive")
    page.keyboard.press("v")
    _settle(page)
    assert page.evaluate("detachedDimMode && _comparePanesShown()")


def test_split_ortho_rows_keep_their_own_index(loaded_viewer, client, tmp_path):
    from conftest import register_array

    arr = np.random.default_rng(4).random((24, 26, 20, 6)).astype(np.float32)
    page = loaded_viewer(register_array(client, arr, tmp_path, "coils"))
    _settle(page)
    page.evaluate("activeDim = 3; renderInfo()")
    page.keyboard.press("Shift+S")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    _settle(page)
    page.keyboard.press("]")  # the left pane's index moves off the main one
    page.wait_for_timeout(500)
    a, b = page.evaluate("[detachedDimIndexA, detachedDimIndexB]")
    assert a != b
    page.keyboard.press("v")
    _settle(page)
    page.wait_for_function("() => compareMvViews.length && compareMvViews.every(v => v.lastImageData)")
    rows = page.evaluate("compareMvViews.map(v => [v.cmvRow, _cmvIndices(v)[3]])")
    assert {r[1] for r in rows if r[0] == 0} == {a}
    assert {r[1] for r in rows if r[0] == 1} == {b}
    # Two different volumes, so the rows' pictures differ.
    assert page.evaluate(
        "(() => { const n = compareMvViews.length / 2, p = compareMvViews[0].lastImageData.data,"
        " q = compareMvViews[n].lastImageData.data; for (let i = 0; i < p.length; i++) if (p[i] !== q[i]) return true; return false; })()"
    )
    # One array: its rows carry no array name.
    assert page.evaluate("document.querySelectorAll('#qmri-view-wrap .cq-row-name').length") == 0


def _maps_page(loaded_viewer, client, tmp_path):
    from conftest import register_array

    arr = np.random.default_rng(5).random((20, 24, 18, 3)).astype(np.float32)
    page = loaded_viewer(register_array(client, arr, tmp_path, "maps"))
    _settle(page)
    return page


_MODE = (
    "() => compareMvActive ? 'split+ortho' : compareQmriActive ? 'split+qmri' : qmriMvActive ? 'qmri+ortho'"
    " : multiViewActive ? 'ortho' : qmriActive ? 'qmri' : compareActive ? 'split' : 'single'"
)


def _press_all(page, keys):
    for k in keys:
        page.keyboard.press(k)
        page.wait_for_function("() => !_orthoGlideBusy && !_crossfading", timeout=10_000)
        page.wait_for_timeout(400)
    _settle(page)
    return page.evaluate(_MODE)


def test_ortho_and_qmri_commute(loaded_viewer, client, tmp_path):
    page = _maps_page(loaded_viewer, client, tmp_path)
    assert _press_all(page, ["v", "q"]) == "qmri+ortho"
    assert _press_all(page, ["v"]) == "qmri"          # v takes ortho away again
    assert _press_all(page, ["v", "q"]) == "ortho"    # and q takes qMRI away
    assert _press_all(page, ["q"]) == "qmri+ortho"


def test_split_and_ortho_commute(loaded_viewer, client, tmp_path):
    from conftest import register_array

    arr = np.random.default_rng(6).random((20, 24, 18, 5)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "vols")
    results = []
    for keys in (["Shift+S", "v"], ["v", "Shift+S"]):
        page = loaded_viewer(sid)
        _settle(page)
        page.evaluate("activeDim = 3; renderInfo()")
        results.append((_press_all(page, keys), page.evaluate("detachedDim")))
    assert results == [("split+ortho", 3), ("split+ortho", 3)]
    # Shift+S again leaves the split and keeps ortho.
    assert _press_all(page, ["Shift+S"]) == "ortho"


def test_split_along_an_ortho_plane_dim_is_refused(loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    _settle(page)
    page.evaluate("activeDim = current_slice_dim; renderInfo()")
    assert _press_all(page, ["Shift+S", "v"]) == "split"
    assert "ortho shows as a plane" in page.evaluate("document.getElementById('status').textContent")


def test_split_ortho_rows_share_the_split_window(loaded_viewer, client, tmp_path):
    from conftest import register_array

    vols = np.random.default_rng(7).random((20, 24, 18, 3)).astype(np.float32)
    vols[..., 1] = vols[..., 1] * 2 + 1  # second volume brighter: per-pane ranges would hide that
    page = loaded_viewer(register_array(client, vols, tmp_path, "vols"))
    _settle(page)
    page.evaluate("activeDim = 3; renderInfo()")
    _press_all(page, ["Shift+S", "v"])
    page.wait_for_function(_all_drawn("compareMvViews"))
    window = page.evaluate("[manualVmin, manualVmax]")
    assert page.evaluate("compareMvViews.map(v => [v.vmin, v.vmax])") == [window] * 6
