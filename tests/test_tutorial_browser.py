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
    () => ({
        key: document.getElementById('tutorial-whisper-key').textContent,
        text: document.getElementById('tutorial-whisper-text').textContent,
        visible: document.getElementById('tutorial-whisper').classList.contains('is-visible'),
        echo: document.getElementById('tutorial-whisper').classList.contains('is-echo'),
        index: _tutorialIndex,
    })
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
    page.wait_for_timeout(500)
    return page


def test_tutorial_opens_on_an_invitation_that_explains_nothing(tutorial_page):
    """The whole premise: it asks for a key and does not say what the key
    does. Anything that describes the outcome up front is the old tour."""
    page = tutorial_page
    state = page.evaluate(WHISPER)

    assert state["visible"], f"the tutorial should whisper straight away, got {state}"
    assert state["key"] == "K", f"the first step should name one key, got {state}"
    assert not state["echo"], f"an invitation is not an echo, got {state}"
    assert len(state["text"]) <= 40, (
        f"the invitation should be a whisper, not a paragraph, got {state['text']!r}"
    )
    assert "slice" not in state["text"].lower(), (
        f"the invitation must not give away what the key does, got {state['text']!r}"
    )
    assert page.evaluate("() => compareActive") is False, (
        "the comparison pair must not open before the tour reaches it"
    )
    assert page.evaluate("() => _overlayVisibility") == "none", (
        "the overlay must stay hidden until the tour reaches it"
    )


def test_the_action_reveals_what_it_did(tutorial_page):
    page = tutorial_page
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
    page.wait_for_timeout(3200)
    nxt = page.evaluate(WHISPER)
    assert nxt["index"] == 1, f"the tour should move on by itself, got {nxt}"
    assert not nxt["echo"], f"the next step is an invitation again, got {nxt}"
    assert nxt["key"] == "L", f"expected the next key, got {nxt}"


def test_an_unrelated_command_earns_no_progress(tutorial_page):
    page = tutorial_page
    page.keyboard.press("b")
    page.wait_for_timeout(300)
    state = page.evaluate(WHISPER)
    assert state["index"] == 0 and not state["echo"], (
        f"only the asked-for action should advance the tour, got {state}"
    )


def test_there_is_no_menu_to_read_or_close(tutorial_page):
    """No panel, no counter, no progress bar, no buttons."""
    page = tutorial_page
    leftovers = page.evaluate(
        """() => ['tutorial-panel', 'tutorial-title', 'tutorial-copy',
                  'tutorial-count', 'tutorial-progress', 'tutorial-action',
                  'tutorial-back', 'tutorial-skip', 'tutorial-restart',
                  'tutorial-close']
            .filter(id => document.getElementById(id))"""
    )
    assert leftovers == [], f"the tutorial chrome should be gone, found {leftovers}"

    buttons = page.evaluate(
        "() => document.querySelectorAll('#tutorial-whisper button').length"
    )
    assert buttons == 0, "the whisper should offer nothing to click"


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


def test_escape_wakes_you_up(tutorial_page):
    page = tutorial_page
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert page.evaluate(
        "() => document.body.classList.contains('tutorial-active')"
    ) is False, "Escape should end the tutorial"
    assert page.evaluate(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-visible')"
    ) is False, "the whisper should go with it"


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
    page.wait_for_timeout(1200)
    assert page.evaluate("() => compareActive"), (
        "the pair step should open the comparison by itself"
    )
    echoed = page.evaluate(WHISPER)
    assert echoed["echo"], f"it should say what appeared, got {echoed}"

    page.wait_for_timeout(3000)
    after = page.evaluate(WHISPER)
    assert after["index"] > page.evaluate(
        "() => _TUTORIAL_STEPS.findIndex(s => s.auto === 'pair')"
    ), f"the tour should carry on without input, got {after}"


def test_the_last_step_ends_the_tour(tutorial_page):
    page = tutorial_page
    page.evaluate("() => _tutorialGo(_TUTORIAL_STEPS.length - 1)")
    page.wait_for_timeout(3400)
    assert page.evaluate(
        "() => document.body.classList.contains('tutorial-active')"
    ) is False, "the final step should end the tutorial on its own"
    assert page.evaluate(
        "() => sessionStorage.getItem('arrayview:tutorial:v2')"
    ) is None, "a finished tour should not resume on reload"
