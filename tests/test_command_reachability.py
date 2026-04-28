"""Cross-mode command reachability matrix.

For each (mode, essential_command) pair, enters the mode on a freshly
loaded viewer and asserts that evalWhen(commands[cmd].when, makeContext(state))
returns the expected boolean. Expected defaults to True; known legitimate
disables live in INTENTIONAL_DISABLES with a short comment.

This test is about drift detection, not about fixing existing coverage —
if something here flips, either the command's `when` clause changed or a
mode started leaking/hiding context. Either way it deserves a look.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from conftest import register_array
from test_mode_roundtrip import (
    MODES,
    _focus,
    _wait,
)

pytestmark = pytest.mark.browser


# Subset of modes worth exercising — skip MIP (slow, WebGL) and qmri_compact
# (redundant vs qmri for reachability purposes).
CORE_MODES = [
    "multiview",
    "compare",
    "qmri",
    "projection",
    "mosaic",
    "zen",
]


ESSENTIAL_COMMANDS = [
    "colormap.openOrCycle",
    "colormap.cycleNext",
    "zoom.in",
    "zoom.out",
    "zoom.reset",
    "slice.next",
    "slice.prev",
    "mode.toggleZen",
    "playback.toggle",
    "fft.toggle",
    "logscale.toggle",
    "histogram.openOrCycle",
    "info.toggle",
    "screenshot.save",
    "hoverinfo.toggle",
    "toolmenu.open",
]


# (mode, cmd) -> expected enabled bool. Each entry needs a short comment.
INTENTIONAL_DISABLES: dict[tuple[str, str], bool] = {
    # qMRI fixes colormap/range per parameter map → cycleNext is a no-op.
    # (currently still reported enabled because the command has when: [];
    #  listed here only as a template — leave empty until a real drift shows.)
}


@pytest.fixture
def reach_page(loaded_viewer, sid_3d, sid_4d, client, tmp_path):
    """Factory: returns a callable (mode_id) -> page already in that mode."""
    def _enter(mode_id: str):
        spec = MODES[mode_id]
        arr_kind = spec["arr"]
        if arr_kind == "3d":
            sid = sid_3d
        elif arr_kind == "4d":
            sid = sid_4d
        elif arr_kind == "3d-compare":
            sid = sid_3d
            second = np.flip(
                np.random.default_rng(11)
                .standard_normal((20, 64, 64))
                .astype(np.float32),
                axis=0,
            )
            partner = register_array(client, second, tmp_path, "arr_reach_cmp")
            _p = loaded_viewer(sid)
            _focus(_p)
            _wait(_p, 300)
            _p.evaluate(f"() => {{ window._rtripCompareSid = {partner!r}; }}")
            spec["enter"](_p)
            _focus(_p)
            _wait(_p, 200)
            return _p
        else:
            raise RuntimeError(f"unknown arr kind {arr_kind!r}")
        page = loaded_viewer(sid)
        _focus(page)
        _wait(page, 300)
        spec["enter"](page)
        _focus(page)
        _wait(page, 200)
        return page

    return _enter


@pytest.mark.parametrize(
    "mode_id,cmd",
    list(product(CORE_MODES, ESSENTIAL_COMMANDS)),
)
def test_command_reachable_in_mode(reach_page, mode_id, cmd):
    page = reach_page(mode_id)
    enabled = page.evaluate(
        f"() => evalWhen(commands[{cmd!r}].when, makeContext(null))"
    )
    expected = INTENTIONAL_DISABLES.get((mode_id, cmd), True)
    assert enabled == expected, (
        f"{cmd} expected enabled={expected} in mode {mode_id}, got {enabled}"
    )
