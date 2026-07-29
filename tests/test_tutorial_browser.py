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


WHISPER = """
    () => {
        const w = document.getElementById('tutorial-whisper');
        return {
            key: document.getElementById('tutorial-whisper-key').textContent,
            text: document.getElementById('tutorial-whisper-text').textContent,
            note: document.getElementById('tutorial-whisper-note').textContent,
            visible: w.classList.contains('is-visible'),
            echo: w.classList.contains('is-echo'),
            section: w.classList.contains('is-section'),
            muted: document.getElementById('tutorial-layer')
                .classList.contains('is-muted'),
            index: _tutorialIndex,
            rail: (document.querySelector('.tutorial-rail-item.is-current') || {})
                .textContent,
        };
    }
"""

ASK = "() => document.getElementById('tutorial-whisper-key').textContent === %r"

_ASKING = """
    () => {
        const w = document.getElementById('tutorial-whisper');
        return w.classList.contains('is-visible') && !w.classList.contains('is-echo')
            && document.getElementById('tutorial-whisper-key').textContent !== '';
    }
"""
_ECHOING = """
    () => document.getElementById('tutorial-whisper').classList.contains('is-echo')
"""


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
    return page


def _wait_for_ask(page, key, timeout=20_000):
    page.wait_for_function(ASK % key, timeout=timeout)


def _section(page, section_id):
    """Resolve a section by id. Never hard-code the index — sections get
    inserted, and an index that silently means a different chapter turns a
    real failure into a confusing one."""
    index = page.evaluate(
        f"() => _TUTORIAL_SECTIONS.findIndex(s => s.id === {section_id!r})"
    )
    assert index >= 0, f"no tutorial section with id {section_id!r}"
    return index


def _go_to_section(page, section_id):
    page.evaluate(f"() => _tutorialGoSection({_section(page, section_id)})")


def test_the_tour_opens_on_a_chapter_not_an_instruction(tutorial_page):
    """It used to drop you straight into 'press K'. A tour that changes the
    ground under you — a second array, then overlays — has to say where it
    is before it asks for anything."""
    page = tutorial_page
    page.wait_for_function(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-section')",
        timeout=15_000,
    )
    state = page.evaluate(WHISPER)

    assert state["text"] == "moving", f"the first section should name itself, got {state}"
    assert state["note"], f"a section should say what it covers, got {state}"
    assert not state["key"], f"a chapter heading asks for nothing, got {state}"
    assert page.evaluate("() => compareActive") is False, (
        "the comparison pair must not open before the tour reaches it"
    )
    assert page.evaluate("() => _overlayVisibility") == "none", (
        "the overlay must stay hidden until the tour reaches it"
    )


def test_the_invitation_explains_nothing(tutorial_page):
    """The whole premise: it asks for a key and does not say what the key
    does. Anything that describes the outcome up front is the old tour."""
    page = tutorial_page
    _wait_for_ask(page, "k")
    state = page.evaluate(WHISPER)

    assert state["visible"] and not state["echo"], f"an invitation is not an echo, got {state}"
    assert len(state["text"]) <= 40, (
        f"the invitation should be a whisper, not a paragraph, got {state['text']!r}"
    )
    assert "slice" not in state["text"].lower(), (
        f"the invitation must not give away what the key does, got {state['text']!r}"
    )


def test_every_step_asks_for_a_key_that_is_actually_bound(tutorial_page):
    """`v` and `V` are different commands. A step whose label does not match
    the keymap strands the reader on a key that does something else."""
    page = tutorial_page
    mismatched = page.evaluate(
        """() => {
            const bound = new Map();
            keybinds.forEach(b => {
                if (!b.key) return;
                const label = (b.shift === true ? 'Shift+' : '') + b.key;
                if (!bound.has(label)) bound.set(label, []);
                bound.get(label).push(b.command);
            });
            return _TUTORIAL_STEPS
                .filter(s => s.key && s.expect)
                .filter(s => {
                    const cmds = bound.get(s.key === 'space' ? ' ' : s.key) || [];
                    return !s.expect.some(e => cmds.includes(e));
                })
                .map(s => ({key: s.key, expect: s.expect}));
        }"""
    )
    assert mismatched == [], f"these steps name a key that does not run them: {mismatched}"


def test_the_action_reveals_what_it_did(tutorial_page):
    page = tutorial_page
    _wait_for_ask(page, "k")
    before = page.evaluate("indices[activeDim]")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(400)

    assert page.evaluate("indices[activeDim]") != before, "the key should still act"
    echoed = page.evaluate(WHISPER)
    assert echoed["echo"], f"landing the action should switch to the echo, got {echoed}"
    assert "slice" in echoed["text"].lower(), (
        f"the echo should name what just happened, got {echoed['text']!r}"
    )
    assert echoed["index"] == 0, "the echo belongs to the step you just did"

    # It dissolves on its own into the next invitation. Nothing to dismiss.
    _wait_for_ask(page, "l")
    nxt = page.evaluate(WHISPER)
    assert nxt["index"] == 1, f"the tour should move on by itself, got {nxt}"
    assert not nxt["echo"], f"the next step is an invitation again, got {nxt}"


def test_the_echo_holds_long_enough_to_read(tutorial_page):
    """The tour used to move on before you had finished the line."""
    page = tutorial_page
    _wait_for_ask(page, "k")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(2400)
    state = page.evaluate(WHISPER)
    assert state["echo"] and state["visible"], (
        f"the echo should still be up 2.4s after the action, got {state}"
    )


def test_it_waits_while_you_are_still_exploring(tutorial_page):
    """Room to explore is the point: a new demand must not land on someone
    who is still playing with what they just found."""
    page = tutorial_page
    _wait_for_ask(page, "k")
    page.keyboard.press("ArrowUp")
    page.wait_for_timeout(300)

    # Keep working well past the point where the next step would be due.
    for _ in range(10):
        page.keyboard.press("ArrowUp")
        page.wait_for_timeout(400)
    busy = page.evaluate(WHISPER)
    assert not (busy["visible"] and not busy["echo"]), (
        f"no new invitation should arrive while keys are still coming, got {busy}"
    )

    _wait_for_ask(page, "l")
    assert page.evaluate(WHISPER)["index"] == 1, "it should resume once you stop"


def test_an_unrelated_command_earns_no_progress(tutorial_page):
    page = tutorial_page
    _wait_for_ask(page, "k")
    page.keyboard.press("b")
    page.wait_for_timeout(300)
    state = page.evaluate(WHISPER)
    assert state["index"] == 0 and not state["echo"], (
        f"only the asked-for action should advance the tour, got {state}"
    )


def test_a_step_that_asks_for_several_presses_waits_for_them(tutorial_page):
    """'press it a few times' resolving on the first press makes the line
    a lie, and skips the behaviour it was pointing at."""
    page = tutorial_page
    # Walk into the split step the way a reader would; its `]` is the only
    # counted step that does not open a panel over the frame. Consecutive
    # steps reuse the same key, so wait on the step index, not the label.
    target = page.evaluate(
        "() => _TUTORIAL_STEPS.findIndex(s => s.expect"
        " && s.expect.includes('detachedDim.adjustIndexA'))"
    )
    _go_to_section(page, "views")
    for key in ("v", "v", "Shift+S"):
        page.wait_for_function(_ASKING, timeout=20_000)
        page.keyboard.press(key)
        # Wait for the press to be taken before looking for the next ask,
        # or the still-showing previous label reads as the next one.
        page.wait_for_function(_ECHOING, timeout=20_000)
    page.wait_for_function(f"() => _tutorialIndex === {target}", timeout=20_000)
    _wait_for_ask(page, "]")

    page.keyboard.press("]")
    page.wait_for_timeout(400)
    state = page.evaluate(WHISPER)
    assert not state["echo"], f"one press should not satisfy a two-press step, got {state}"
    assert page.evaluate("() => _tutorialHits") == 1, "the press should still be counted"

    page.keyboard.press("]")
    page.wait_for_timeout(500)
    assert page.evaluate(WHISPER)["echo"], "the second press should complete the step"


def test_the_only_thing_to_click_is_the_section_rail(tutorial_page):
    """No panel, no counter, no progress bar, no dismiss button. The rail
    is the one deliberate exception."""
    page = tutorial_page
    leftovers = page.evaluate(
        """() => ['tutorial-panel', 'tutorial-title', 'tutorial-copy',
                  'tutorial-count', 'tutorial-progress', 'tutorial-action',
                  'tutorial-back', 'tutorial-skip', 'tutorial-restart',
                  'tutorial-close']
            .filter(id => document.getElementById(id))"""
    )
    assert leftovers == [], f"the tutorial chrome should be gone, found {leftovers}"

    assert page.evaluate(
        "() => document.querySelectorAll('#tutorial-whisper button').length"
    ) == 0, "the whisper itself should offer nothing to click"

    rail = page.evaluate(
        "() => Array.from(document.querySelectorAll('.tutorial-rail-item'))"
        ".map(el => el.textContent)"
    )
    assert len(rail) >= 4 and rail[0] == "moving", f"the rail should list sections, got {rail}"


def test_the_whisper_never_blocks_the_array(tutorial_page):
    page = tutorial_page
    page.set_viewport_size({"width": 1024, "height": 640})
    page.wait_for_timeout(300)
    state = page.evaluate(
        """() => {
            const w = document.getElementById('tutorial-whisper');
            const r = w.getBoundingClientRect();
            return {
                pointerEvents: getComputedStyle(w).pointerEvents,
                hitAtCentre: document.elementFromPoint(
                    r.left + r.width / 2, r.top + r.height / 2)?.id || '',
            };
        }"""
    )
    assert state["pointerEvents"] == "none", (
        f"the whisper must not intercept the pointer, got {state}"
    )
    assert not state["hitAtCentre"].startswith("tutorial-"), (
        f"clicks through the whisper should reach what is behind it, got {state}"
    )


def test_it_goes_quiet_behind_the_panel_it_just_asked_for(tutorial_page):
    """The colormap picker opens centred, right over where the line sits.
    Talking underneath it is how the tour used to lose its own echo."""
    page = tutorial_page
    _go_to_section(page, "looking")
    _wait_for_ask(page, "c")

    page.keyboard.press("c")
    page.wait_for_timeout(700)
    state = page.evaluate(WHISPER)
    assert state["muted"], f"the tutorial should step aside for the picker, got {state}"
    assert page.evaluate(
        "() => getComputedStyle(document.getElementById('tutorial-layer')).opacity"
    ) == "0", "a muted tutorial should be fully out of the way"

    # And Escape belongs to the picker, not to the tour.
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)
    assert page.evaluate(
        "() => document.body.classList.contains('tutorial-active')"
    ), "closing a panel must not also end the tutorial"


def test_sections_can_be_switched(tutorial_page):
    page = tutorial_page
    page.wait_for_timeout(600)

    page.keyboard.press("Tab")
    page.wait_for_timeout(600)
    assert page.evaluate(WHISPER)["rail"] == "looking", "Tab should move a section on"

    page.keyboard.press("Shift+Tab")
    page.wait_for_timeout(600)
    assert page.evaluate(WHISPER)["rail"] == "moving", "Shift+Tab should move back"

    page.click(f".tutorial-rail-item[data-section='{_section(page, 'pair')}']")
    page.wait_for_timeout(600)
    state = page.evaluate(WHISPER)
    assert state["rail"] == "two arrays", f"the rail should be clickable, got {state}"
    assert state["section"] and state["text"] == "two arrays", (
        f"arriving in a section should announce it, got {state}"
    )


def test_jumping_into_a_section_puts_the_viewer_where_it_expects(tutorial_page):
    """Sections are entry points, so each one has to set its own stage —
    otherwise skipping ahead lands you in a mode its first step cannot use."""
    page = tutorial_page
    _go_to_section(page, "pair")
    page.wait_for_function("() => compareActive", timeout=20_000)

    # Going back to a single-array section must undo it again.
    _go_to_section(page, "moving")
    page.wait_for_timeout(1200)
    assert page.evaluate("() => compareActive") is False, (
        "a single-array section should close the comparison behind it"
    )


def test_the_stage_hand_steps_run_themselves(tutorial_page):
    """`auto` steps set something up and move on; nothing is asked of the
    reader, so a stall there would strand the whole tour."""
    page = tutorial_page
    page.evaluate(
        """() => {
            const i = _TUTORIAL_STEPS.findIndex(s => s.auto === 'pair');
            _tutorialGo(i);
        }"""
    )
    page.wait_for_function("() => compareActive", timeout=20_000)
    _wait_for_ask(page, "Shift+X")
    assert page.evaluate(WHISPER)["index"] > page.evaluate(
        "() => _TUTORIAL_STEPS.findIndex(s => s.auto === 'pair')"
    ), "the tour should carry on without input"


def test_escape_wakes_you_up(tutorial_page):
    page = tutorial_page
    _wait_for_ask(page, "k")
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert page.evaluate(
        "() => document.body.classList.contains('tutorial-active')"
    ) is False, "Escape should end the tutorial"
    assert page.evaluate(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-visible')"
    ) is False, "the whisper should go with it"


def test_the_last_step_ends_the_tour(tutorial_page):
    page = tutorial_page
    page.evaluate("() => _tutorialGo(_TUTORIAL_STEPS.length - 1)")
    page.wait_for_function(
        "() => !document.body.classList.contains('tutorial-active')",
        timeout=20_000,
    )
    assert page.evaluate(
        "() => sessionStorage.getItem('arrayview:tutorial:v3')"
    ) is None, "a finished tour should not resume on reload"
