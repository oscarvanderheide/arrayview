"""Cross-mode parametrized test harness — Phase 14 regression gate.

Verifies that modeManager.modeName and modeManager.currentViews are
correctly populated after entering each mode. These tests form the
regression gate for the View Component System.

Run with:
    uv run pytest tests/test_cross_mode_parametrized.py -v
"""
import pytest

pytestmark = pytest.mark.browser


# ---------------------------------------------------------------------------
# Mode parameters
# Each entry: (mode_label, expected_view_count, expected_modename)
# expected_view_count=None means "just check > 0, don't check exact count"
# ---------------------------------------------------------------------------

MODE_PARAMS = [
    pytest.param("normal",       1,    "normal",       id="normal"),
    pytest.param("multiview",    3,    "multiview",    id="multiview"),
    pytest.param("compare",      2,    "compare",      id="compare"),
    pytest.param("qmri",         None, "qmri",         id="qmri"),
    pytest.param("mip",          1,    "mip",          id="mip"),
    pytest.param("compare-mv",   None, "compare-mv",   id="compare-mv"),
    pytest.param("compare-qmri", None, "compare-qmri", id="compare-qmri"),
]

# JS snippets to enter each mode.  Each snippet runs inside an async IIFE.
# Returns {mode, count} when done.
_MODE_JS = {
    "normal": """async () => {
        // normal mode is the boot default — just return current state
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "multiview": """async () => {
        document.getElementById('keyboard-sink')?.focus();
        const evt = new KeyboardEvent('keydown', {key: 'v', bubbles: true});
        document.getElementById('keyboard-sink').dispatchEvent(evt);
        await new Promise(r => setTimeout(r, 800));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "compare": """async () => {
        if (typeof enterCompareModeByMultipleSids !== 'function') {
            return { error: 'no enterCompareModeByMultipleSids' };
        }
        await enterCompareModeByMultipleSids([window.currentSid, window.currentSid]);
        await new Promise(r => setTimeout(r, 400));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "qmri": """async () => {
        if (typeof enterQmri !== 'function') {
            return { error: 'no enterQmri' };
        }
        enterQmri();
        await new Promise(r => setTimeout(r, 600));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "mip": """async () => {
        // MIP requires entering multiview first
        document.getElementById('keyboard-sink')?.focus();
        const evt = new KeyboardEvent('keydown', {key: 'v', bubbles: true});
        document.getElementById('keyboard-sink').dispatchEvent(evt);
        await new Promise(r => setTimeout(r, 800));
        if (typeof enterMipMode !== 'function') {
            return { error: 'no enterMipMode' };
        }
        await enterMipMode();
        await new Promise(r => setTimeout(r, 400));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "compare-mv": """async () => {
        if (typeof enterCompareModeByMultipleSids !== 'function') {
            return { error: 'no enterCompareModeByMultipleSids' };
        }
        await enterCompareModeByMultipleSids([window.currentSid, window.currentSid]);
        await new Promise(r => setTimeout(r, 400));
        if (typeof enterCompareMv !== 'function') {
            return { error: 'no enterCompareMv' };
        }
        enterCompareMv();
        await new Promise(r => setTimeout(r, 600));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",

    "compare-qmri": """async () => {
        if (typeof enterCompareModeByMultipleSids !== 'function') {
            return { error: 'no enterCompareModeByMultipleSids' };
        }
        await enterCompareModeByMultipleSids([window.currentSid, window.currentSid]);
        await new Promise(r => setTimeout(r, 400));
        if (typeof enterCompareQmri !== 'function') {
            return { error: 'no enterCompareQmri' };
        }
        enterCompareQmri();
        await new Promise(r => setTimeout(r, 600));
        return {
            mode: modeManager.modeName,
            count: modeManager.currentViews.length,
        };
    }""",
}

# Modes that need a 4-D array (sid_4d); all others use sid_3d.
_NEEDS_4D = {"qmri", "compare-qmri"}


# ---------------------------------------------------------------------------
# Parametrized test: modeManager state after mode entry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode_label, expected_count, expected_modename", MODE_PARAMS)
def test_mode_populates_modemanager(
    loaded_viewer,
    sid_3d,
    sid_4d,
    mode_label,
    expected_count,
    expected_modename,
):
    """After entering each mode, modeManager reflects the correct name and view count."""
    sid = sid_4d if mode_label in _NEEDS_4D else sid_3d
    page = loaded_viewer(sid)
    page.wait_for_timeout(400)

    js = _MODE_JS[mode_label]
    result = page.evaluate(js)
    page.wait_for_timeout(300)

    # Must not have returned an error object from a missing function
    assert "error" not in (result or {}), (
        f"Mode '{mode_label}' JS returned error: {result.get('error')}"
    )

    # modeManager.modeName must match
    assert result.get("mode") == expected_modename, (
        f"Expected modeName='{expected_modename}', got '{result.get('mode')}'"
    )

    # currentViews must be non-empty
    count = result.get("count", 0)
    assert count > 0, (
        f"Mode '{mode_label}': modeManager.currentViews is empty (count={count})"
    )

    # If an exact count is expected, enforce it
    if expected_count is not None:
        assert count == expected_count, (
            f"Mode '{mode_label}': expected {expected_count} views, got {count}"
        )


# ---------------------------------------------------------------------------
# Additional test: views have required properties (normal mode only)
# ---------------------------------------------------------------------------

def test_modemanager_views_have_required_properties(loaded_viewer, sid_3d):
    """Each view in modeManager.currentViews (normal mode) has id, role, displayState.

    canvas may be null for non-rendering views, so we only verify its type
    is either a canvas element or null.
    """
    page = loaded_viewer(sid_3d)
    page.wait_for_timeout(500)

    result = page.evaluate("""() => {
        return modeManager.currentViews.map(v => ({
            id:           v.id,
            role:         v.role,
            hasDisplayState: typeof v.displayState === 'object' && v.displayState !== null,
            canvasOk:     v.canvas === null || (v.canvas && v.canvas.tagName === 'CANVAS'),
        }));
    }""")

    assert len(result) > 0, "No views found in normal mode"

    for entry in result:
        assert entry["id"] is not None and entry["id"] != "", (
            f"View has missing id: {entry}"
        )
        assert entry["role"] is not None and entry["role"] != "", (
            f"View has missing role: {entry}"
        )
        assert entry["hasDisplayState"] is True, (
            f"View '{entry['id']}' has no displayState"
        )
        assert entry["canvasOk"] is True, (
            f"View '{entry['id']}' canvas is neither a <canvas> nor null"
        )
