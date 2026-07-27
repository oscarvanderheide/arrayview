"""A render failure must name itself instead of presenting as a hang.

Before this, an exception while producing a frame was printed to the server's
stdout and the socket was closed with nothing sent to the viewer. The panel
kept its loading spinner until the VS Code opener's timeout failed the request
tens of seconds later, so a hard backend error reached the user as an
unexplained hang — and, because the opener serialises on one request at a
time, it stalled every click behind it.

The trigger is deliberately injected rather than fished for with a malformed
array: the contract under test is "any exception during rendering is
reported", not the behaviour of one bad input.
"""

import numpy as np
import pytest

from conftest import register_array


@pytest.fixture
def failing_render(monkeypatch):
    import arrayview._routes_websocket as ws_mod

    def _boom(*args, **kwargs):
        raise ValueError("buffer is not large enough")

    monkeypatch.setattr(ws_mod, "render_rgba", _boom)
    return _boom


def test_render_failure_is_shown_in_the_viewer(
    page, client, server_url, tmp_path, failing_render
):
    sid = register_array(
        client, np.zeros((32, 32), dtype=np.float32), tmp_path, "boom"
    )
    page.goto(f"{server_url}/?sid={sid}")

    overlay = page.locator("#loading-overlay")
    overlay.get_by_text("Could not render this array").wait_for(timeout=20_000)
    assert "buffer is not large enough" in overlay.inner_text()


def test_a_failed_array_does_not_break_the_next_one(
    page, client, server_url, tmp_path, monkeypatch
):
    """The behaviour actually asked for: one bad array must not poison the rest."""
    import arrayview._routes_websocket as ws_mod

    real_render = ws_mod.render_rgba
    calls = {"n": 0}

    def _boom_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("buffer is not large enough")
        return real_render(*args, **kwargs)

    monkeypatch.setattr(ws_mod, "render_rgba", _boom_once)

    bad = register_array(
        client, np.zeros((32, 32), dtype=np.float32), tmp_path, "bad"
    )
    page.goto(f"{server_url}/?sid={bad}")
    page.locator("#loading-overlay").get_by_text(
        "Could not render this array"
    ).wait_for(timeout=20_000)

    good = register_array(
        client, np.arange(64 * 64, dtype=np.float32).reshape(64, 64), tmp_path, "good"
    )
    page.goto(f"{server_url}/?sid={good}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=20_000)
