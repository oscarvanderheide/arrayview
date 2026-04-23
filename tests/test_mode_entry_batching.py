"""Guard the postMessage batching invariant at every multi-pane mode entry.

Background: in VS Code tab (postMessage transport), render requests issued
from per-view WebSocket `onopen` callbacks land in separate event-loop ticks.
The server's parallel render pool can only batch requests that arrive in the
same tick, so staggered sends force serial rendering and panes visibly pop in
one-by-one.

The fix (commit 78a064c) is two-part, applied at every multi-pane entry:
  1. Pass `postMessageTransport ? null : () => <renderFn>(...)` as the
     `onopen` so postMessage doesn't render per-view.
  2. Call `_kickInitialRenders(views, renderFn)` after the view loop so all
     N initial renders fire in a single tick.

This has regressed silently once already (qMRI was fixed but ortho and the
compare variants were forgotten). A pytest-level static check is the cheapest
way to keep it from drifting again.
"""
from __future__ import annotations

from pathlib import Path

VIEWER_HTML = Path(__file__).parent.parent / "src" / "arrayview" / "_viewer.html"

# (entry function, render function) for each multi-pane mode.
MODE_ENTRIES = [
    ("enterMultiView",    "mvRender"),
    ("enterQmri",         "qvRender"),
    ("enterCompareMv",    "compareMvRender"),
    ("enterCompareQmri",  "compareQmriRender"),
]


def _function_body(src: str, name: str) -> str:
    """Return the source of a top-level JS function declaration, up to the
    next `function ` at the same indentation. Good enough for a static check.
    """
    needle = f"function {name}("
    start = src.find(needle)
    assert start >= 0, f"{name} not found in _viewer.html"
    # Next function at the same 8-space indentation level marks the end.
    end = src.find("\n        function ", start + 1)
    return src[start : end if end >= 0 else len(src)]


def test_helper_defined_once():
    src = VIEWER_HTML.read_text()
    assert src.count("function _kickInitialRenders(") == 1, \
        "_kickInitialRenders helper should be defined exactly once"


def test_every_mode_entry_batches_in_postmessage():
    src = VIEWER_HTML.read_text()
    for entry, render in MODE_ENTRIES:
        body = _function_body(src, entry)
        assert f"postMessageTransport ? null : () => {render}" in body, (
            f"{entry} must pass `postMessageTransport ? null : () => {render}(...)` "
            f"as onopen — otherwise per-view renders stagger inside VS Code tab."
        )
        assert f"_kickInitialRenders(" in body and f", {render})" in body, (
            f"{entry} must call `_kickInitialRenders(<views>, {render})` after the "
            f"view loop so all panes' initial renders fire in one tick."
        )
