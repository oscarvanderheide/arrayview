"""Corner array names stay clear of what else sits on the panes.

Run with:
    pytest tests/test_pane_names.py
"""

import os

import numpy as np
import pytest

from conftest import register_array

pytestmark = pytest.mark.browser

_OVERLAP = """
(pair) => {
    const [a, b] = pair.map(s => typeof s === 'string' ? document.querySelector(s) : s);
    if (!a || !b) return null;
    const r = a.getBoundingClientRect(), q = b.getBoundingClientRect();
    if (!r.width || !q.width) return null;
    return r.left < q.right && r.right > q.left && r.top < q.bottom && r.bottom > q.top;
}
"""


def _focus(page):
    page.evaluate("() => document.getElementById('viewer')?.focus()")


def _two(client, tmp_path):
    rng = np.random.default_rng(4)
    a, b = (rng.random((8, 48, 48)).astype(np.float32) * (1 + i) for i in range(2))
    return register_array(client, a, tmp_path, "names_base"), register_array(client, b, tmp_path, "names_follow")


def test_zoom_map_and_slice_bar_leave_compare_names_readable(page, server_url, client, tmp_path):
    a, b = _two(client, tmp_path)
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
    page.wait_for_selector("#compare-view-wrap.compare-titles-inset", timeout=15_000)
    left_name = "#compare-left-title"
    left_map = ".compare-primary .cmp-mini-map.visible"

    # Zoomed in: the overview map hangs below the name, not on top of it.
    for _ in range(3):
        _focus(page)
        page.keyboard.press("=")
    page.wait_for_selector(left_map, timeout=5_000)
    page.wait_for_timeout(300)
    assert page.evaluate(_OVERLAP, [left_name, left_map]) is False

    # Immersive: names stay on the images and step out from under the dim bar.
    # Side by side only opens it when height is the limit, hence the short window.
    page.set_viewport_size({"width": 2000, "height": 500})
    page.wait_for_timeout(300)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1500)
    for sel in (left_name, "#compare-right-title"):
        assert page.evaluate("s => getComputedStyle(document.querySelector(s)).opacity", sel) == "1"
        assert page.evaluate(_OVERLAP, [sel, "#info"]) is False
    tops = page.evaluate(
        "() => ['#compare-left-title', '#compare-right-title'].map(s => Math.round(document.querySelector(s).getBoundingClientRect().top))"
    )
    assert tops[0] == tops[1]

    # Back out: names return to the corner.
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => !_fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1500)
    assert page.evaluate("() => getComputedStyle(document.getElementById('compare-left-title')).translate") in ("none", "")


def test_brackets_follow_the_mouse_and_move_only_that_pane(page, server_url, client, tmp_path):
    arr = np.random.default_rng(5).random((48, 48, 120)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "names_split")
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("canvas#viewer", timeout=15_000)
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(500)
    bar_height = page.evaluate("() => document.getElementById('info').offsetHeight")

    _focus(page)
    page.evaluate("() => { activeDim = 2; renderInfo(); }")
    page.keyboard.press("Shift+Semicolon")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    page.wait_for_timeout(800)

    def bracketed():
        return page.evaluate(
            """() => compareTitles.map((t, i) => {
                const b = t.querySelector('.hint-bracket');
                return b && getComputedStyle(b).visibility === 'visible' ? i : -1;
            }).filter(i => i >= 0)"""
        )

    def hover(i):
        box = page.locator(f"#{['compare-left', 'compare-right', 'compare-third'][i]}-canvas").bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(100)

    # Every pane names just its own number (the total, /120, is the dim bar's);
    # only the pane under the mouse wears the brackets, none when off the panes.
    assert all(page.evaluate("() => compareTitles.slice(0, 2).map(e => /^\\[?\\d{3}\\]?$/.test(e.textContent))"))
    page.mouse.move(5, 5)
    page.wait_for_timeout(100)
    assert bracketed() == []
    _focus(page)
    before = page.evaluate("() => [...detachedDimIdx]")
    page.keyboard.press("]")  # no pane under the mouse: nothing moves
    page.wait_for_timeout(150)
    assert page.evaluate("() => [...detachedDimIdx]") == before
    hover(0)
    assert bracketed() == [0]
    hover(1)
    assert bracketed() == [1]
    page.mouse.move(5, 5)
    page.wait_for_timeout(100)
    assert bracketed() == []
    hover(1)

    a, b = page.evaluate("() => [...detachedDimIdx]")
    # Neighbours too read as a set, not a run.
    assert page.evaluate("() => document.querySelector('#info .dim-spread-set').textContent") == f"{{{a + 1},{b + 1}}}"
    _focus(page)
    page.keyboard.press("]")
    page.wait_for_timeout(150)
    assert page.evaluate("() => [...detachedDimIdx]") == [a, b + 1]
    assert page.evaluate("() => !!compareTitles[1].querySelector('.index-hint-glow')")

    # Moved apart: the dim bar still lists them as a tight set.
    setText = lambda: page.evaluate("() => document.querySelector('#info .dim-spread-set').textContent")
    assert setText() == f"{{{a + 1},{b + 2}}}"
    # The underline covers the digits and commas, not the braces.
    page.wait_for_timeout(400)
    ul = page.evaluate("""() => {
        const t = document.querySelector('#info .spread-digits').getBoundingClientRect();
        const m = document.getElementById('dim-active-underline').getBoundingClientRect();
        return [Math.round(t.left), Math.round(t.width), Math.round(m.left), Math.round(m.width)];
    }""")
    assert abs(ul[0] - ul[2]) <= 1 and abs(ul[1] - ul[3]) <= 1, ul
    # Extents on: the slash fills in the active colour like any dim's; digits stay plain.
    page.evaluate("() => { _dimbarExtentPinned = true; _reconcileDimbarExtent(); renderInfo(); }")
    look = page.evaluate("""() => {
        const lab = document.querySelector('#info .active-dim.dim-label');
        const dig = getComputedStyle(lab.querySelector('.spread-digits')).webkitTextFillColor;
        return { slash: lab.querySelector('.dim-slash').style.background, dig, text: getComputedStyle(lab).webkitTextFillColor };
    }""")
    assert "linear-gradient" in look["slash"] and look["dig"] == look["text"]
    page.evaluate("() => { _dimbarExtentPinned = false; _reconcileDimbarExtent(); renderInfo(); }")
    hover(1)
    assert page.evaluate("() => document.getElementById('info').offsetHeight") == bar_height
    shots = os.environ.get("AV_SHOT_DIR", str(tmp_path))
    page.screenshot(path=f"{shots}/spread_pair_moved.png")

    # One more pane keeps the hand-set two and follows the last.
    page.keyboard.press("'")
    page.wait_for_function("() => detachedDimIdx.length === 3", timeout=10_000)
    page.wait_for_timeout(1200)
    assert page.evaluate("() => [...detachedDimIdx]") == [a, b + 1, b + 2]
    hover(2)
    _focus(page)
    page.keyboard.press("[")
    page.wait_for_timeout(150)
    # The pane it lands on is taken, so it skips past it.
    assert page.evaluate("() => [...detachedDimIdx]") == [a, b + 1, b]
    assert bracketed() == [2]
    page.screenshot(path=f"{shots}/spread_three_moved.png")


_RECORD_FADE = """
(ms) => new Promise(done => {
    const out = [], t0 = performance.now();
    const step = () => {
        const bar = getComputedStyle(document.getElementById('info')).opacity;
        const line = getComputedStyle(document.getElementById('dim-active-underline')).opacity;
        out.push([+bar, +line]);
        if (performance.now() - t0 < ms) requestAnimationFrame(step); else done(out);
    };
    requestAnimationFrame(step);
})
"""


def test_compact_underline_fades_with_the_dim_bar(page, server_url, client, tmp_path):
    arr = np.random.default_rng(6).random((256, 256, 20)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "names_fade")
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(1500)
    page.evaluate("() => { activeDim = 2; _dimbarExtentPinned = false; _reconcileDimbarExtent(); renderInfo(); }")
    page.wait_for_timeout(400)

    def gap():
        return page.evaluate(
            """() => document.getElementById('dim-active-underline').getBoundingClientRect().top
                - document.querySelector('#info .active-dim.dim-label').getBoundingClientRect().bottom"""
        )

    for _ in ("enter", "exit", "enter again", "exit again"):
        _focus(page)
        page.evaluate("(ms) => { window.__fade = null; (" + _RECORD_FADE + ")(ms).then(r => window.__fade = r); }", 1300)
        page.keyboard.press("F")
        page.wait_for_function("() => window.__fade", timeout=5_000)
        frames = page.evaluate("() => window.__fade")
        # The bar really fades, rather than popping, and the line stays with it.
        assert any(0.2 < bar < 0.8 for bar, _line in frames)
        assert max(abs(bar - line) for bar, line in frames[2:]) < 0.1
        assert frames[-1] == [1, 1]
        assert gap() == 0
        page.wait_for_timeout(200)


def test_split_key_flash_survives_the_bar_redrawing(page, server_url, client, tmp_path):
    arr = np.random.default_rng(7).random((48, 48, 20)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "names_flash")
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(500)
    page.evaluate("() => { activeDim = 2; renderInfo(); }")
    _focus(page)
    page.keyboard.press("Shift+Semicolon")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    page.wait_for_timeout(800)

    # Lit in the same task as the press, and a redraw (a slice arriving) changes nothing.
    state = page.evaluate(
        """async () => {
            const lit = () => {
                const el = compareTitles[0].querySelector('.hint-bracket[data-detached-key="]"]');
                return el.classList.contains('index-hint-glow') ? getComputedStyle(el).webkitTextFillColor : null;
            };
            const yellow = () => {
                const probe = document.createElement('span');
                probe.style.color = 'var(--active-dim)';
                document.body.append(probe);
                const c = getComputedStyle(probe).color;
                probe.remove();
                return c;
            };
            _spreadTarget = 0;
            _bumpSpreadPane(1, ']');
            const first = lit();
            updateCompareTitles();
            return { first, afterRedraw: lit(), yellow: yellow() };
        }"""
    )
    assert state["first"] == state["yellow"]
    assert state["afterRedraw"] == state["yellow"]

    # Over soon after: a later redraw does not bring it back.
    page.wait_for_timeout(500)
    page.evaluate("() => updateCompareTitles()")
    assert page.evaluate("() => !compareTitles[0].querySelector('.index-hint-glow')")


@pytest.mark.parametrize("shape, viewport", [((48, 48, 20), (2000, 500)), ((48, 96, 20), (1600, 500))])
def test_immersive_panes_join_with_touching_off(page, server_url, client, tmp_path, shape, viewport):
    arr = np.random.default_rng(8).random(shape).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "names_join")
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(500)
    page.evaluate("() => { compareGapless = false; activeDim = 2; renderInfo(); }")
    _focus(page)
    page.keyboard.press("Shift+Semicolon")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    page.wait_for_timeout(800)

    def seam():
        return page.evaluate(
            """() => {
                const [a, b] = [...document.querySelectorAll('.compare-pane.active .compare-canvas-clip')]
                    .slice(0, 2).map(c => c.getBoundingClientRect());
                return b.left - a.right;
            }"""
        )

    assert seam() > 0
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1000)
    assert abs(seam()) < 1
    shots = os.environ.get("AV_SHOT_DIR", str(tmp_path))
    page.screenshot(path=f"{shots}/join_{shape[0]}_{shape[1]}.png")

    # Back out: the preference applies again.
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => !_fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1000)
    assert seam() > 0


_RECORD_RECTS = """
(args) => new Promise(done => {
    const [ms, sels] = args, out = [], t0 = performance.now();
    const step = () => {
        const row = {};
        for (const [name, sel] of Object.entries(sels)) {
            const el = document.querySelector(sel), r = el.getBoundingClientRect();
            row[name] = { top: r.top, left: r.left, w: r.width, o: +getComputedStyle(el).opacity };
        }
        out.push(row);
        if (performance.now() - t0 < ms) requestAnimationFrame(step); else done(out);
    };
    requestAnimationFrame(step);
})
"""


def _record_toggle(page, sels, ms=1300):
    _focus(page)
    page.evaluate(
        "(args) => { window.__rects = null; (" + _RECORD_RECTS + ")(args).then(r => window.__rects = r); }", [ms, sels]
    )
    page.keyboard.press("F")
    page.wait_for_function("() => window.__rects", timeout=6_000)
    return page.evaluate("() => window.__rects")


def _open_split(page, server_url, client, tmp_path, name, viewport=(2000, 700)):
    arr = np.random.default_rng(9).random((128, 128, 20)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, name)
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(800)
    page.evaluate("() => { activeDim = 2; _dimbarExtentPinned = false; _reconcileDimbarExtent(); renderInfo(); }")
    _focus(page)
    page.keyboard.press("Shift+Semicolon")
    page.wait_for_function("() => detachedDimMode && compareActive", timeout=10_000)
    page.wait_for_timeout(900)


_SPLIT_SELS = {
    "clip": ".compare-pane.active .compare-canvas-clip",
    "cb": "#slim-cb-wrap",
    "bar": "#info",
}


def test_immersive_glide_in_split_has_no_jumps(page, server_url, client, tmp_path):
    _open_split(page, server_url, client, tmp_path, "names_glide")
    for _ in range(2):
        enter = _record_toggle(page, _SPLIT_SELS)
        # Starts from where it was, moves on: no shove in the first frames.
        assert abs(enter[3]["clip"]["top"] - enter[0]["clip"]["top"]) <= 2
        assert abs(enter[3]["clip"]["w"] - enter[0]["clip"]["w"]) <= 2
        page.wait_for_timeout(200)

        exit_ = _record_toggle(page, _SPLIT_SELS)
        mid = [f for f in exit_ if 0.2 < f["bar"]["o"] < 0.8]
        assert mid, "the bar should fade back in"
        # The colorbar fades in where it stays, with the bar.
        assert all(f["cb"]["top"] == exit_[-1]["cb"]["top"] for f in mid)
        assert max(abs(f["cb"]["o"] - f["bar"]["o"]) for f in mid) < 0.1
        # Lands exactly where it started, without a late step.
        assert exit_[-1]["clip"]["top"] == enter[0]["clip"]["top"]
        assert exit_[-1]["cb"]["top"] == enter[0]["cb"]["top"]
        tail = exit_[-40:]
        assert max(abs(b["clip"]["top"] - a["clip"]["top"]) for a, b in zip(tail, tail[1:])) <= 2
        page.wait_for_timeout(200)
    assert page.evaluate("() => userZoom === _normalFitZoom")


def test_immersive_exit_animates_when_it_would_not_enter_again(page, server_url, client, tmp_path):
    # Square image, square window: from inside, immersive no longer looks worthwhile.
    arr = np.random.default_rng(10).random((128, 128, 20)).astype(np.float32)
    sid = register_array(client, arr, tmp_path, "names_square")
    page.set_viewport_size({"width": 1000, "height": 1000})
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_function("() => shape.length === 3", timeout=15_000)
    page.wait_for_timeout(800)
    _record_toggle(page, {"bar": "#info"})
    assert page.evaluate("() => _fullscreenActive && immersiveTl.progress() === 1")
    frames = _record_toggle(page, {"bar": "#info", "vp": "#canvas-viewport"})
    assert any(0.2 < f["bar"]["o"] < 0.8 for f in frames)
    steps = [abs(b["vp"]["w"] - a["vp"]["w"]) for a, b in zip(frames, frames[1:])]
    assert max(steps) < 40


def test_immersive_key_turns_round_mid_glide(page, server_url, client, tmp_path):
    _open_split(page, server_url, client, tmp_path, "names_turn")
    before = page.evaluate("() => document.querySelector('.compare-pane.active .compare-canvas-clip').getBoundingClientRect().top")
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_timeout(150)
    page.keyboard.press("F")
    page.wait_for_timeout(1000)
    assert page.evaluate("() => !_fullscreenActive && !immersiveTl")
    assert before == page.evaluate("() => document.querySelector('.compare-pane.active .compare-canvas-clip').getBoundingClientRect().top")


def test_immersive_exit_after_resize_leaves_chrome_in_place(page, server_url, client, tmp_path):
    _open_split(page, server_url, client, tmp_path, "names_resize")
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(800)
    page.set_viewport_size({"width": 1900, "height": 640})
    page.wait_for_timeout(400)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => !_fullscreenActive", timeout=5_000)
    page.wait_for_timeout(600)
    state = page.evaluate(
        """() => {
            const line = document.getElementById('dim-active-underline');
            const label = document.querySelector('#info .active-dim.dim-label');
            const clip = document.querySelector('.compare-pane.active .compare-canvas-clip').getBoundingClientRect();
            const cb = document.getElementById('slim-cb-wrap').getBoundingClientRect();
            return {
                lineOpacity: +getComputedStyle(line).opacity,
                gap: line.getBoundingClientRect().top - label.getBoundingClientRect().bottom,
                belowPanes: cb.top - clip.bottom,
                cbBottom: cb.bottom,
            };
        }"""
    )
    assert state["lineOpacity"] == 1
    assert state["gap"] == 0
    assert 0 <= state["belowPanes"] <= 30
    assert state["cbBottom"] <= 640


def test_immersive_exit_with_centre_pane_keeps_bar_still(page, server_url, client, tmp_path):
    rng = np.random.default_rng(11)
    a, b = (register_array(client, rng.random((128, 128, 20)).astype(np.float32), tmp_path, f"names_centre{i}") for i in range(2))
    page.set_viewport_size({"width": 1800, "height": 1000})
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
    page.wait_for_selector("#compare-view-wrap.compare-titles-inset", timeout=15_000)
    page.wait_for_timeout(500)
    page.evaluate("() => _setCompareCenterMode(1)")
    page.wait_for_timeout(800)
    page.evaluate("() => { compareLayoutMode = 'big-left'; _setCompareCenterMode(1); ModeRegistry.scaleAll(); }")
    page.wait_for_timeout(800)
    assert page.evaluate("() => _isCompareBigLeftLayout() && _shouldEnterImmersive()")
    _record_toggle(page, {"bar": "#info"})
    frames = _record_toggle(page, {"bar": "#info", "cb": "#slim-cb-wrap"})
    # Recording starts just before the key lands: only what follows the bar's fade-out counts.
    frames = frames[next(i for i, f in enumerate(frames) if f["bar"]["o"] <= 0.2):]
    visible = [f for f in frames if f["bar"]["o"] > 0.2]
    bar_mids = [f["bar"]["left"] + f["bar"]["w"] / 2 for f in visible]
    assert max(bar_mids) - min(bar_mids) < 2
    shown = [f for f in frames if f["cb"]["o"] > 0.2]
    cb_mids = [f["cb"]["left"] + f["cb"]["w"] / 2 for f in shown]
    assert max(cb_mids) - min(cb_mids) < 2


def _immersive_volume(page, server_url, client, tmp_path, viewport=(2000, 700)):
    sid = register_array(client, np.random.default_rng(5).random((64, 64, 12)).astype(np.float32), tmp_path, "bars")
    page.set_viewport_size({"width": viewport[0], "height": viewport[1]})
    page.goto(f"{server_url}/?sid={sid}")
    page.wait_for_selector("#viewer", timeout=15_000)
    page.wait_for_function("() => _frameDisplaySeq > 0", timeout=15_000)
    page.wait_for_timeout(500)


def test_immersive_split_bars_appear_after_the_panes_land(page, server_url, client, tmp_path):
    _immersive_volume(page, server_url, client, tmp_path)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1200)
    _focus(page)
    page.keyboard.press("Shift+Semicolon")
    frames = page.evaluate(
        """() => new Promise(res => {
            const out = [], t0 = performance.now();
            const step = () => {
                out.push({
                    left: document.querySelector('body > canvas[aria-hidden]')?.getBoundingClientRect().left ?? null,
                    bar: +getComputedStyle(document.getElementById('info')).opacity,
                    cb: +getComputedStyle(document.getElementById('slim-cb-wrap')).opacity,
                });
                if (performance.now() - t0 < 1100) requestAnimationFrame(step); else res(out);
            };
            step();
        })"""
    )
    lefts = [f["left"] for f in frames if f["left"] is not None]
    assert lefts and lefts[0] != lefts[-1]
    # Hidden for as long as the image copy is on its way; it may linger a moment once landed.
    assert all(f["bar"] == 0 and f["cb"] == 0 for f in frames if f["left"] is not None and f["left"] != lefts[-1])
    assert frames[-1]["bar"] == 1 and frames[-1]["cb"] == 1


def test_immersive_ortho_value_readout_sits_above_the_colorbar(page, server_url, client, tmp_path):
    _immersive_volume(page, server_url, client, tmp_path, viewport=(1500, 900))
    _focus(page)
    page.keyboard.press("v")
    page.wait_for_function("() => multiViewActive", timeout=5_000)
    page.wait_for_timeout(1200)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1200)
    _focus(page)
    page.keyboard.press("i")
    box = page.locator("#mv-cb-wrap").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] - 30
    page.mouse.move(x, y - 5)
    page.mouse.move(x, y)
    page.wait_for_selector("#mode-eggs .eggs-value-pill", timeout=5_000)
    page.wait_for_timeout(200)
    assert page.locator("#mode-eggs .eggs-value-pill").bounding_box()["y"] + 18 <= box["y"]


def test_immersive_colorbar_has_no_box_and_ticks_do_not_move_it(page, server_url, client, tmp_path):
    _immersive_volume(page, server_url, client, tmp_path)
    _focus(page)
    page.keyboard.press("F")
    page.wait_for_function("() => _fullscreenActive", timeout=5_000)
    page.wait_for_timeout(1200)
    background = "() => getComputedStyle(document.getElementById('slim-cb-wrap')).backgroundColor"
    bar_top = "() => document.getElementById('slim-cb').getBoundingClientRect().top"
    assert page.evaluate(background) == "rgba(0, 0, 0, 0)"
    assert "0px 0px 0px 1px" in page.evaluate("() => getComputedStyle(document.getElementById('slim-cb')).boxShadow")
    before = page.evaluate(bar_top)
    _focus(page)
    page.keyboard.press("w")
    page.wait_for_selector("#slim-cb-wrap.cb-has-ticks", timeout=5_000)
    page.wait_for_timeout(400)
    assert page.evaluate(bar_top) == before
    last_tick = "() => Math.max(...[...document.querySelectorAll('#slim-cb-wrap .cb-ticks span')].map(e => e.getBoundingClientRect().bottom))"
    assert page.evaluate(last_tick) <= page.evaluate("() => innerHeight")
    # The histogram stays bare too; a colormap row brings the box back.
    page.evaluate("() => document.getElementById('slim-cb-wrap').classList.add('cb-expanded')")
    page.wait_for_timeout(400)
    assert page.evaluate(background) == "rgba(0, 0, 0, 0)"
    page.evaluate("() => document.getElementById('slim-cb-wrap').classList.add('cb-backed')")
    page.wait_for_timeout(400)
    assert page.evaluate(background) != "rgba(0, 0, 0, 0)"


def test_corner_names_fade_under_the_axis_cross(page, server_url, client, tmp_path):
    """The dimbar / r / t axis cross must not sit on top of a pane's corner name."""
    a, b = _two(client, tmp_path)
    page.goto(f"{server_url}/?sid={a}&compare_sid={b}&compare_sids={b}")
    page.wait_for_selector("#compare-view-wrap.compare-titles-inset", timeout=15_000)
    page.wait_for_timeout(500)
    op = "() => +getComputedStyle(document.getElementById('compare-left-title')).opacity"
    assert page.evaluate(op) == 1
    page.evaluate("() => { _orientHovering = true; _showOrientEdgeHighlight(); }")
    page.wait_for_timeout(300)
    assert page.evaluate("() => document.querySelectorAll('.orient-cross.showing').length") == 2
    assert page.evaluate(op) == 0
    page.evaluate("() => { _orientHovering = false; _clearOrientEdgeHighlight(); }")
    page.wait_for_timeout(300)
    assert page.evaluate(op) == 1
