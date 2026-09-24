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
            section: w.classList.contains('is-section'),
            muted: document.getElementById('tutorial-layer')
                .classList.contains('is-muted'),
            index: _tutorialIndex,
            rail: (document.querySelector('.tutorial-rail-item.is-current') || {})
                .textContent,
        };
    }
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


_INTRO = "() => document.getElementById('tutorial-whisper').classList.contains('is-intro')"


def _read_the_welcome(page):
    """Press Enter through every welcome line, the way a reader would."""
    for line in range(page.evaluate("() => _TUTORIAL_INTRO.length")):
        page.wait_for_function(
            f"() => {_INTRO[6:]} && _tutorialSkip !== null"
            f" && document.getElementById('tutorial-whisper-text').textContent"
            f" === _TUTORIAL_INTRO[{line}].text",
            timeout=15_000,
        )
        page.keyboard.press("Enter")


@pytest.fixture
def toured_page(tutorial_page):
    _read_the_welcome(tutorial_page)
    return tutorial_page


def test_it_says_what_arrayview_is_before_asking_for_anything(tutorial_page):
    """Starting on "press K" over an unexplained picture felt rushed. The
    tour first says what the tool is, and waits for Enter on every line."""
    page = tutorial_page
    page.wait_for_function(_INTRO, timeout=15_000)
    first = page.evaluate(WHISPER)
    assert "arrayview" in first["text"], first
    assert "Enter" in first["note"], f"the first line should say how to continue, got {first}"
    assert not first["key"], f"the welcome asks for no key, got {first}"

    page.wait_for_timeout(6_000)
    still = page.evaluate(WHISPER)
    assert still["text"] == first["text"], "a welcome line must not move on by itself"

    _read_the_welcome(page)
    page.wait_for_function(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-section')",
        timeout=15_000,
    )
    assert page.evaluate(WHISPER)["text"] == "moving"


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


def test_the_tour_opens_on_a_chapter_not_an_instruction(toured_page):
    """It used to drop you straight into 'press K'. A tour that changes the
    ground under you — a second array, then overlays — has to say where it
    is before it asks for anything."""
    page = toured_page
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


def _wait_for_step(page, index, timeout=20_000):
    """Wait until step `index` is on screen and Enter would move past it."""
    page.wait_for_function(
        f"() => _tutorialIndex === {index} && _tutorialSkip !== null"
        " && document.getElementById('tutorial-whisper-text').textContent"
        f" === _TUTORIAL_STEPS[{index}].text",
        timeout=timeout,
    )


def _step_index(page, predicate):
    return page.evaluate(f"() => _TUTORIAL_STEPS.findIndex({predicate})")


def test_a_step_says_what_its_keys_do(toured_page):
    """A bare "press k" with no explanation rushed people past the point.
    Each step names its keys and says what they do."""
    page = toured_page
    _wait_for_step(page, 0)
    state = page.evaluate(WHISPER)
    assert state["key"].split() == ["j", "k"], state
    assert "slice" in state["text"], f"the step should say what j and k do, got {state}"
    assert "Enter" in state["note"], f"the step should say how to move on, got {state}"


def test_trying_the_keys_does_not_move_on(toured_page):
    """Pressing the key once used to jump straight to the next line. Now
    you can keep going for as long as you like."""
    page = toured_page
    _wait_for_step(page, 0)
    before = page.evaluate("indices[activeDim]")
    for _ in range(5):
        page.keyboard.press("k")
        page.wait_for_timeout(150)
    page.wait_for_timeout(4_000)

    assert page.evaluate("indices[activeDim]") != before, "the key should still act"
    state = page.evaluate(WHISPER)
    assert state["index"] == 0, f"trying the keys must not move the tour on, got {state}"
    assert page.evaluate(
        "() => document.getElementById('tutorial-whisper-key').classList.contains('is-tried')"
    ), "the key should show it was tried"

    page.keyboard.press("Enter")
    _wait_for_step(page, 1)


def test_the_text_does_not_jump(toured_page):
    """Headings, steps with a key, and long lines that wrap must all start
    at the same height, and nothing may appear under a line after it lands."""
    page = toured_page
    top = "() => Math.round(document.getElementById('tutorial-whisper-text').getBoundingClientRect().top)"
    tops = set()
    for index in range(page.evaluate("() => _TUTORIAL_SECTION_START[1]")):
        _wait_for_step(page, index)
        tops.add(page.evaluate(top))
        page.wait_for_timeout(1_500)
        tops.add(page.evaluate(top))
        page.keyboard.press("Enter")
    page.wait_for_function(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-section')"
        " && document.getElementById('tutorial-whisper').classList.contains('is-visible')",
        timeout=20_000,
    )
    tops.add(page.evaluate(top))
    assert len(tops) == 1, f"the text moved between lines: {sorted(tops)}"


def test_every_key_on_a_chip_is_bound(toured_page):
    """A chip that names a key the keymap does not bind strands the reader."""
    page = toured_page
    unbound = page.evaluate(
        """() => {
            const bound = new Set(keybinds.map(b => b.key).filter(Boolean));
            bound.add('?');
            return _TUTORIAL_STEPS
                .flatMap(s => (s.key || '').split(/\\s+/).filter(Boolean))
                .map(k => k === 'space' ? ' ' : k)
                .filter(k => !bound.has(k));
        }"""
    )
    assert unbound == [], f"these chip keys do nothing: {unbound}"


def test_every_expected_command_exists(toured_page):
    page = toured_page
    missing = page.evaluate(
        """() => _TUTORIAL_STEPS
            .flatMap(s => s.expect || [])
            .filter(id => id !== 'pointer.draw' && !commands[id])"""
    )
    assert missing == [], f"these steps listen for commands that do not exist: {missing}"


def test_enter_belongs_to_the_choices_while_they_are_open(toured_page):
    """`p` opens a row of choices that Enter closes. That Enter must not
    also skip the step."""
    page = toured_page
    target = _step_index(page, "s => s.key === 'p'")
    _go_to_section(page, "mosaic")
    _wait_for_step(page, target - 1)
    page.keyboard.press("Enter")
    _wait_for_step(page, target)

    page.keyboard.press("p")
    page.wait_for_function("() => !!_modePicker", timeout=5_000)
    page.keyboard.press("Enter")
    page.wait_for_timeout(1_500)
    assert page.evaluate(WHISPER)["index"] == target, "Enter should only close the choices"


def test_a_new_chapter_starts_from_a_plain_view(toured_page):
    """Steps leave things on for you to play with; the next chapter must
    not inherit them."""
    page = toured_page
    target = _step_index(page, "s => s.key === 'z'")
    _go_to_section(page, "mosaic")
    _wait_for_step(page, target)
    page.keyboard.press("z")
    page.wait_for_function("() => dim_z >= 0", timeout=5_000)

    _go_to_section(page, "spectra")
    page.wait_for_function("() => dim_z < 0", timeout=10_000)


def test_the_only_thing_to_click_is_the_section_rail(toured_page):
    """No panel, no counter, no progress bar, no dismiss button. The rail
    is the one deliberate exception."""
    page = toured_page
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


def test_the_whisper_never_blocks_the_array(toured_page):
    page = toured_page
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


def test_it_goes_quiet_behind_the_panel_it_just_asked_for(toured_page):
    """The colormap picker opens centred, right over where the line sits.
    Talking underneath it would go unread."""
    page = toured_page
    _go_to_section(page, "looking")
    _wait_for_step(page, _step_index(page, "s => s.key === 'c'"))

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


def test_sections_can_be_switched(toured_page):
    page = toured_page
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


def test_jumping_into_a_section_puts_the_viewer_where_it_expects(toured_page):
    """Sections are entry points, so each one has to set its own stage —
    otherwise skipping ahead lands you in a mode its first step cannot use."""
    page = toured_page
    _go_to_section(page, "pair")
    page.wait_for_function("() => compareActive", timeout=20_000)

    # Going back to a single-array section must undo it again.
    _go_to_section(page, "moving")
    page.wait_for_timeout(1200)
    assert page.evaluate("() => compareActive") is False, (
        "a single-array section should close the comparison behind it"
    )


def test_the_stage_hand_steps_run_themselves(toured_page):
    """`auto` steps open the comparison themselves, then wait like any
    other step."""
    page = toured_page
    index = _step_index(page, "s => s.auto === 'pair'")
    page.evaluate(f"() => _tutorialGo({index})")
    page.wait_for_function("() => compareActive", timeout=20_000)
    _wait_for_step(page, index)
    page.keyboard.press("Enter")
    _wait_for_step(page, index + 1)


def test_escape_wakes_you_up(toured_page):
    page = toured_page
    _wait_for_step(page, 0)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    assert page.evaluate(
        "() => document.body.classList.contains('tutorial-active')"
    ) is False, "Escape should end the tutorial"
    assert page.evaluate(
        "() => document.getElementById('tutorial-whisper').classList.contains('is-visible')"
    ) is False, "the whisper should go with it"


def test_the_last_step_ends_the_tour(toured_page):
    page = toured_page
    last = page.evaluate("() => _TUTORIAL_STEPS.length - 1")
    page.evaluate(f"() => _tutorialGo({last})")
    _wait_for_step(page, last)
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => !document.body.classList.contains('tutorial-active')",
        timeout=20_000,
    )
    assert page.evaluate(
        "() => sessionStorage.getItem('arrayview:tutorial:v3')"
    ) is None, "a finished tour should not resume on reload"
