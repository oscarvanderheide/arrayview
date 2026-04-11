"""Round-trip state preservation tests.

Phase 0 of the UI maturity strategy. Each test enters a mode, perturbs
state, exits, and asserts collectStateSnapshot() is unchanged from before
entry. Failures here are the input to Phase 1 (finishing collectStateSnapshot
and ensuring every mode enter/exit is symmetric).

THIS TEST IS EXPECTED TO FAIL on day one. The failures are the diagnostic.
Once Phase 1 lands, this should be ~98% green. See dev/plans/ui-maturity-strategy.md.

Run with:
    uv run pytest tests/test_mode_roundtrip.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import register_array

pytestmark = pytest.mark.browser


# ---------------------------------------------------------------------------
# Mode entry/exit specs
# ---------------------------------------------------------------------------
#
# Each mode has an "enter" and "exit" callable taking (page) and returning
# nothing. "arr" says which fixture array to load. "compare" modes require a
# second array registered at test time. The enter callable is expected to
# leave the page in the mode; exit toggles back to baseline.

def _wait(page, ms=200):
    page.wait_for_timeout(ms)


def _press(key):
    def _f(page):
        page.keyboard.press(key)
    return _f


# --- multiview ---
def _enter_multiview(page):
    page.keyboard.press("v")
    page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
    _wait(page, 250)


def _exit_multiview(page):
    page.keyboard.press("v")
    _wait(page, 300)


# --- compare ---
# The B keybind was retired; compare is now opened either via the unified
# picker (Ctrl/Cmd+O) or programmatically via enterCompareModeBySid. For
# round-trip testing we exercise the JS entry point directly — it's the same
# function URL-param loads and the picker both call, and avoids needing to
# drive a modal picker in headless mode.
# The partner sid is stashed on `window._rtripCompareSid` by the test body.
def _enter_compare(page):
    page.evaluate(
        "async () => { await enterCompareModeBySid(window._rtripCompareSid); }"
    )
    page.wait_for_selector("#compare-view-wrap.active", timeout=5_000)
    _wait(page, 400)


def _exit_compare(page):
    page.evaluate("() => exitCompareMode()")
    _wait(page, 400)


# --- qmri (needs 4D) ---
def _enter_qmri(page):
    page.keyboard.press("q")
    _wait(page, 400)


def _exit_qmri(page):
    # qMRI compact cycle: q again takes full -> compact (if n>3) or exits.
    # For 5-map array, pressing q again toggles compact; pressing a third time exits.
    # To cleanly exit, keep pressing q up to 2 more times until qmriActive is false.
    for _ in range(3):
        active = page.evaluate("() => (typeof qmriActive !== 'undefined') && qmriActive")
        if not active:
            break
        page.keyboard.press("q")
        _wait(page, 350)


# --- qmri_compact: enter qMRI, then press q once more to compact; exit the same way ---
def _enter_qmri_compact(page):
    page.keyboard.press("q")
    _wait(page, 350)
    page.keyboard.press("q")  # full -> compact
    _wait(page, 350)


def _exit_qmri_compact(page):
    for _ in range(3):
        active = page.evaluate("() => (typeof qmriActive !== 'undefined') && qmriActive")
        if not active:
            break
        page.keyboard.press("q")
        _wait(page, 350)


# --- mip (multiview + p) ---
def _enter_mip(page):
    page.keyboard.press("v")
    page.wait_for_selector("#multi-view-wrap.active", timeout=5_000)
    _wait(page, 250)
    page.keyboard.press("p")
    _wait(page, 400)


def _exit_mip(page):
    # p toggles MIP off while in multiview; then v to exit multiview
    page.keyboard.press("p")
    _wait(page, 300)
    page.keyboard.press("v")
    _wait(page, 300)


# --- projection ---
def _enter_projection(page):
    page.keyboard.press("p")
    _wait(page, 250)


def _exit_projection(page):
    # Projection cycles: off -> MAX -> MIN -> MEAN -> STD -> SOS -> SUM -> off
    # We entered at mode=1; read PROJECTION_LABELS.length from the page so we
    # don't drift when modes are added, and press `p` enough times to wrap.
    n_labels = page.evaluate("() => PROJECTION_LABELS.length")
    for _ in range(n_labels):  # 1 + n_labels presses total → wraps to 0
        page.keyboard.press("p")
        _wait(page, 100)
    _wait(page, 200)


# --- mosaic (z, needs 4D) ---
def _enter_mosaic(page):
    page.keyboard.press("z")
    _wait(page, 350)


def _exit_mosaic(page):
    page.keyboard.press("z")
    _wait(page, 350)


# --- zen / fullscreen (F) ---
def _enter_zen(page):
    page.keyboard.press("F")
    _wait(page, 300)


def _exit_zen(page):
    page.keyboard.press("F")
    _wait(page, 300)


MODES = {
    "multiview":    {"enter": _enter_multiview,    "exit": _exit_multiview,    "arr": "3d"},
    "compare":      {"enter": _enter_compare,      "exit": _exit_compare,      "arr": "3d-compare"},
    "qmri":         {"enter": _enter_qmri,         "exit": _exit_qmri,         "arr": "4d"},
    "qmri_compact": {"enter": _enter_qmri_compact, "exit": _exit_qmri_compact, "arr": "4d"},
    "mip":          {"enter": _enter_mip,          "exit": _exit_mip,          "arr": "3d"},
    "projection":   {"enter": _enter_projection,   "exit": _exit_projection,   "arr": "3d"},
    "mosaic":       {"enter": _enter_mosaic,       "exit": _exit_mosaic,       "arr": "4d"},
    "zen":          {"enter": _enter_zen,          "exit": _exit_zen,          "arr": "3d"},
}


PERTURBATIONS = {
    "cycle_colormap":      _press("c"),
    "cycle_dynamic_range": _press("d"),
    "toggle_log":          _press("L"),
    "toggle_pixel_info":   _press("i"),
    "change_slice":        _press("l"),
}


# (mode, perturbation) -> reason. Skips document design intent, never bugs.
INTENTIONALLY_NOOP = {
    ("qmri",         "cycle_colormap"):      "qMRI: colormaps fixed per parameter map",
    ("qmri_compact", "cycle_colormap"):      "qMRI: colormaps fixed per parameter map",
    ("qmri",         "toggle_log"):          "qMRI: log scale not applicable to parameter maps",
    ("qmri_compact", "toggle_log"):          "qMRI: log scale not applicable to parameter maps",
    ("qmri",         "cycle_dynamic_range"): "qMRI: dynamic range fixed per parameter map",
    ("qmri_compact", "cycle_dynamic_range"): "qMRI: dynamic range fixed per parameter map",
}


# (mode, perturbation) -> reason. These hang the page (open-ended awaits or
# server fetches the test harness can't satisfy) and need the production code
# to grow a deterministic completion path before they can be exercised. Marked
# xfail with strict=False so they show up but don't block the diagnostic run.
HANGING_COMBOS = {
}


# Fields allowed to differ in every diff (timing/internal, not user-facing state).
# viewStates/mmModeName are Phase 15 additions that mirror existing fields
# (manualVmin/vmax, logScale, colormap_idx) already tracked per-field above.
IGNORED_FIELDS_GLOBAL: set[str] = {"viewStates", "mmModeName"}

# Fields each perturbation is DELIBERATELY supposed to change. These are the
# user-visible side-effects of the key press itself; preserving them across
# a mode round-trip is the *correct* behavior. Without these entries, the
# round-trip test would falsely flag every successful perturbation as a bug.
#
# Example: pressing `c` cycles the colormap. Entering compare, pressing `c`,
# and exiting compare should leave the new colormap in place. That's not a
# state-corruption bug — it's the whole point.
PERTURBATION_EXPECTED_CHANGES: dict[str, set[str]] = {
    "cycle_colormap":      {"colormap_idx", "customColormap"},
    "cycle_dynamic_range": {"manualVmin", "manualVmax"},
    # toggle_log changes logScale AND recomputes vmin/vmax for the new scale
    "toggle_log":          {"logScale", "manualVmin", "manualVmax"},
    "toggle_pixel_info":   {"_pixelInfoVisible"},
    # `l` is a navigation key: may change indices, activeDim, current_slice_dim
    # depending on which mode it fires in. All three are navigation state.
    "change_slice":        {"indices", "activeDim", "current_slice_dim"},
}

# Fields that are intentionally sticky across mode exit (not bugs — design).
# Example: after exiting compare, compareSid/compareName/compareSidList are
# deliberately kept so the user can quickly re-enter compare with the same
# partner (see enterCompareModeBySid call in _fullRestore flow at ~line 5883
# and the toggleRegistrationMode fallback at ~line 10210). These are
# remembered state, not leaked state.
IGNORED_FIELDS_PER_MODE: dict[str, set[str]] = {
    "compare": {"compareSid", "compareName", "compareSidList"},
}

# Additional per-case ignores for genuine mode-specific quirks (empty for now;
# any entry here MUST be documented with *why* it's an allowed exception).
IGNORED_FIELDS_PER_CASE: dict[tuple[str, str], set[str]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snapshot(page):
    """Round-trip collectStateSnapshot() through JSON to get a plain Python dict."""
    return page.evaluate(
        "() => JSON.parse(JSON.stringify(collectStateSnapshot()))"
    )


def _focus(page):
    page.focus("#keyboard-sink")


def _diff(before, after, ignore):
    diffs = {}
    for k in set(before) | set(after):
        if k in ignore:
            continue
        if before.get(k) != after.get(k):
            diffs[k] = (before.get(k), after.get(k))
    return diffs


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perturb_id", list(PERTURBATIONS.keys()))
@pytest.mark.parametrize("mode_id", list(MODES.keys()))
def test_round_trip(
    mode_id,
    perturb_id,
    loaded_viewer,
    sid_3d,
    sid_4d,
    client,
    tmp_path,
    arr_3d,
):
    if (mode_id, perturb_id) in INTENTIONALLY_NOOP:
        pytest.skip(INTENTIONALLY_NOOP[(mode_id, perturb_id)])
    if (mode_id, perturb_id) in HANGING_COMBOS:
        pytest.xfail(HANGING_COMBOS[(mode_id, perturb_id)])

    spec = MODES[mode_id]
    arr_kind = spec["arr"]

    if arr_kind == "3d":
        sid = sid_3d
    elif arr_kind == "4d":
        sid = sid_4d
    elif arr_kind == "3d-compare":
        # For compare: load sid_3d as base, register a second 3D array as compare target.
        sid = sid_3d
        second = np.flip(
            np.random.default_rng(11).standard_normal((20, 64, 64)).astype(np.float32),
            axis=0,
        )
        compare_partner_sid = register_array(
            client, second, tmp_path, "arr3d_rtrip_compare"
        )
    else:
        raise RuntimeError(f"unknown arr kind {arr_kind!r}")

    page = loaded_viewer(sid)
    _focus(page)
    _wait(page, 350)

    if arr_kind == "3d-compare":
        page.evaluate(
            f"() => {{ window._rtripCompareSid = {compare_partner_sid!r}; }}"
        )

    # Sanity: the snapshot function must exist and work before we enter the mode.
    before = _snapshot(page)
    assert isinstance(before, dict) and before, "collectStateSnapshot() returned empty"

    spec["enter"](page)
    _focus(page)
    _wait(page, 200)

    PERTURBATIONS[perturb_id](page)
    _wait(page, 200)

    spec["exit"](page)
    _focus(page)
    _wait(page, 300)

    after = _snapshot(page)

    ignore = (
        IGNORED_FIELDS_GLOBAL
        | PERTURBATION_EXPECTED_CHANGES.get(perturb_id, set())
        | IGNORED_FIELDS_PER_MODE.get(mode_id, set())
        | IGNORED_FIELDS_PER_CASE.get((mode_id, perturb_id), set())
    )
    diff = _diff(before, after, ignore)
    assert not diff, (
        f"Round-trip {mode_id} x {perturb_id} corrupted state. "
        f"Diff (field: before -> after):\n"
        + "\n".join(f"  {k}: {v[0]!r} -> {v[1]!r}" for k, v in sorted(diff.items()))
    )
