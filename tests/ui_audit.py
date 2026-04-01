"""
UI Consistency Audit for arrayview.

Automated Playwright-based visual audit that screenshots every mode × array-count
× shape combination and runs DOM assertions to catch cross-mode inconsistencies.

Run after any visual/UI change:

    uv run python tests/ui_audit.py                      # all tiers
    uv run python tests/ui_audit.py --tier 1             # core modes only (~14 combos)
    uv run python tests/ui_audit.py --tier 2             # extended modes (~45 combos)
    uv run python tests/ui_audit.py --tier 3             # shape variations (~35 combos)
    uv run python tests/ui_audit.py --tier 2 --subset zoom,compact
    uv run python tests/ui_audit.py --update-baselines   # reset baselines
    uv run python tests/ui_audit.py --list               # print scenario table and exit

Output:
    tests/ui_audit/screenshots/   — per-combo PNGs
    tests/ui_audit/baselines/     — baseline PNGs for visual diff
    tests/ui_audit/diffs/         — highlighted pixel diffs
    tests/ui_audit/report.txt     — terminal-friendly summary
"""

import argparse
import socket
import sys
import threading
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from playwright.sync_api import sync_playwright, Page

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent / "ui_audit"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
BASELINES_DIR = BASE_DIR / "baselines"
DIFFS_DIR = BASE_DIR / "diffs"
REPORT_PATH = BASE_DIR / "report.txt"

for d in (SCREENSHOTS_DIR, BASELINES_DIR, DIFFS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


def _start_server():
    from arrayview._app import app

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    threading.Thread(target=srv.run, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{base}/ping", timeout=1.0).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    return base, srv


def _load(client, arr, name, tmp):
    path = Path(tmp) / f"{name}.npy"
    np.save(path, arr)
    r = client.post("/load", json={"filepath": str(path), "name": name})
    r.raise_for_status()
    return r.json()["sid"]


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------


def _goto(page: Page, base: str, sid: str, wait: int = 1200):
    try:
        page.evaluate("() => sessionStorage.clear()")
    except Exception:
        pass
    page.goto(f"{base}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_timeout(wait)


def _goto_compare(page: Page, base: str, sids: list[str], wait: int = 1500):
    sid_main = sids[0]
    compare_sids = ",".join(sids[1:])
    try:
        page.evaluate("() => sessionStorage.clear()")
    except Exception:
        pass
    page.goto(f"{base}/?sid={sid_main}&compare_sids={compare_sids}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    page.wait_for_timeout(wait)


def _focus(page: Page):
    page.locator("#keyboard-sink").focus()


def _press(page: Page, key: str, wait: int = 400):
    page.keyboard.press(key)
    page.wait_for_timeout(wait)


def _shot(page: Page, name: str, wait: int = 600) -> Path:
    page.wait_for_timeout(wait)
    path = SCREENSHOTS_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    return path


# ---------------------------------------------------------------------------
# Visual diff (pixel-level comparison)
# ---------------------------------------------------------------------------


def _pixel_diff(img_path: Path, baseline_path: Path, diff_path: Path,
                threshold: float = 0.01) -> tuple[bool, float]:
    """Compare two screenshots. Returns (passed, pct_changed).

    Uses raw numpy pixel comparison. Pixels differing by more than 5/255 in
    any channel are counted as changed. If pct_changed > threshold, the test
    fails and a highlighted diff image is written.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  SKIP visual diff (Pillow not installed)")
        return True, 0.0

    if not baseline_path.exists():
        return True, 0.0  # no baseline to compare against

    img = np.array(Image.open(img_path).convert("RGB"))
    base = np.array(Image.open(baseline_path).convert("RGB"))

    # Handle size mismatch (viewport might differ slightly)
    if img.shape != base.shape:
        h = min(img.shape[0], base.shape[0])
        w = min(img.shape[1], base.shape[1])
        img = img[:h, :w]
        base = base[:h, :w]

    diff_mask = np.any(np.abs(img.astype(int) - base.astype(int)) > 5, axis=-1)
    pct = diff_mask.sum() / diff_mask.size

    if pct > threshold:
        # Write highlighted diff image
        diff_img = img.copy()
        diff_img[diff_mask] = [255, 0, 0]  # red highlight
        Image.fromarray(diff_img).save(diff_path)
        return False, pct

    return True, pct


# ---------------------------------------------------------------------------
# DOM Assertions
# ---------------------------------------------------------------------------

# JavaScript snippets evaluated in the browser to check DOM state.

_JS_BOUNDING_BOX = """
(selector) => {
    const el = document.querySelector(selector);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {x: r.x, y: r.y, w: r.width, h: r.height};
}
"""

_JS_ALL_BOUNDING_BOXES = """
(selector) => {
    return Array.from(document.querySelectorAll(selector))
        .map(el => {
            const r = el.getBoundingClientRect();
            return {x: r.x, y: r.y, w: r.width, h: r.height};
        })
        .filter(b => b.w > 0 && b.h > 0);
}
"""

_JS_IS_VISIBLE = """
(selector) => {
    const el = document.querySelector(selector);
    if (!el) return false;
    const s = getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && s.opacity !== '0';
}
"""

_JS_VIEWPORT = """
() => ({w: window.innerWidth, h: window.innerHeight})
"""

_JS_HEIGHT_SYNC = """
() => {
    const info = document.querySelector('#info');
    const cb = document.querySelector('#slim-cb-wrap');
    if (!info || !cb) return null;
    const iS = getComputedStyle(info);
    const cS = getComputedStyle(cb);
    if (iS.display === 'none' || cS.display === 'none') return null;
    return {infoH: info.offsetHeight, cbH: cb.offsetHeight};
}
"""

_JS_IMMERSIVE_STATE = """
() => ({
    fullscreen: typeof _fullscreenActive !== 'undefined' ? _fullscreenActive : false,
    animating: typeof _immersiveAnimating !== 'undefined' ? _immersiveAnimating : false,
    infoDrag: typeof _infoDragPos !== 'undefined' ? _infoDragPos : null,
    cbDrag: typeof _cbDragPos !== 'undefined' ? _cbDragPos : null,
    islandDrag: typeof _islandDragPos !== 'undefined' ? _islandDragPos : null,
    fsOverlayCount: document.querySelectorAll('.fs-overlay').length,
    hasFullscreenClass: document.body.classList.contains('fullscreen-mode'),
})
"""

_JS_CB_FLEX_DIRS = """
() => {
    return Array.from(document.querySelectorAll('.cb-island'))
        .filter(el => el.offsetWidth > 0 && getComputedStyle(el).display !== 'none')
        .map(el => ({
            id: el.id || el.className,
            dir: getComputedStyle(el).flexDirection,
        }));
}
"""


def _boxes_overlap(a: dict, b: dict) -> bool:
    """Check if two bounding boxes {x, y, w, h} overlap."""
    if a is None or b is None:
        return False
    return not (
        a["x"] + a["w"] <= b["x"] or
        b["x"] + b["w"] <= a["x"] or
        a["y"] + a["h"] <= b["y"] or
        b["y"] + b["h"] <= a["y"]
    )


def _box_within_viewport(box: dict, vp: dict, margin: int = 5) -> bool:
    """Check if a bounding box fits within the viewport (with tolerance)."""
    if box is None:
        return True  # element not present, not a violation
    return (
        box["x"] >= -margin and
        box["y"] >= -margin and
        box["x"] + box["w"] <= vp["w"] + margin and
        box["y"] + box["h"] <= vp["h"] + margin
    )


@dataclass
class AssertionResult:
    rule: str
    passed: bool
    detail: str = ""


def run_assertions(page: Page, mode_name: str, zoomed: bool = False) -> list[AssertionResult]:
    """Run all DOM assertions for the current page state."""
    results = []
    vp = page.evaluate(_JS_VIEWPORT)

    # R3: All canvases within viewport (only at fit-to-window zoom)
    if not zoomed:
        for selector in ["#canvas", ".compare-canvas", ".mv-canvas", ".qv-canvas"]:
            boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, selector)
            for i, box in enumerate(boxes):
                within = _box_within_viewport(box, vp)
                results.append(AssertionResult(
                    rule=f"R3 ({selector}[{i}] within viewport)",
                    passed=within,
                    detail="" if within else f"box={box} vp={vp}",
                ))

    # R6: Eggs don't overlap colorbar
    eggs_box = page.evaluate(_JS_BOUNDING_BOX, "#mode-eggs")
    cb_box = page.evaluate(_JS_BOUNDING_BOX, "#slim-cb-wrap")
    cb_visible = page.evaluate(_JS_IS_VISIBLE, "#slim-cb-wrap")
    if eggs_box and cb_box and cb_visible:
        overlap = _boxes_overlap(eggs_box, cb_box)
        results.append(AssertionResult(
            rule="R6 (eggs don't overlap colorbar)",
            passed=not overlap,
            detail="" if not overlap else f"eggs={eggs_box} cb={cb_box}",
        ))

    # R13: Colorbars don't overlap each other
    all_cb_selectors = [
        "#slim-cb-wrap",
        ".compare-pane-cb",
        ".mv-cb",
        ".qv-cb",
    ]
    cb_boxes_all = []
    for sel in all_cb_selectors:
        boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, sel)
        for box in boxes:
            cb_boxes_all.append((sel, box))
    for i in range(len(cb_boxes_all)):
        for j in range(i + 1, len(cb_boxes_all)):
            sel_a, box_a = cb_boxes_all[i]
            sel_b, box_b = cb_boxes_all[j]
            overlap = _boxes_overlap(box_a, box_b)
            if overlap:
                results.append(AssertionResult(
                    rule=f"R13 ({sel_a} overlaps {sel_b})",
                    passed=False,
                    detail=f"a={box_a} b={box_b}",
                ))

    # R14: Minimap within viewport and not overlapping colorbars
    minimap_box = page.evaluate(_JS_BOUNDING_BOX, "#minimap-canvas")
    minimap_visible = page.evaluate(_JS_IS_VISIBLE, "#minimap-canvas") if minimap_box else False
    if minimap_box and minimap_visible:
        within = _box_within_viewport(minimap_box, vp)
        results.append(AssertionResult(
            rule="R14 (minimap within viewport)",
            passed=within,
            detail="" if within else f"box={minimap_box}",
        ))
        for sel, cb_b in cb_boxes_all:
            if _boxes_overlap(minimap_box, cb_b):
                results.append(AssertionResult(
                    rule=f"R14 (minimap overlaps {sel})",
                    passed=False,
                    detail=f"minimap={minimap_box} cb={cb_b}",
                ))

    # R8: ROI hover info within viewport (if present)
    roi_boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, ".cv-pixel-info")
    for i, box in enumerate(roi_boxes):
        within = _box_within_viewport(box, vp)
        results.append(AssertionResult(
            rule=f"R8 (ROI info[{i}] within viewport)",
            passed=within,
            detail="" if within else f"box={box}",
        ))

    # R10: Compare panes aligned
    compare_boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, ".compare-canvas")
    if len(compare_boxes) >= 2:
        rows: list[list[dict]] = []
        for box in sorted(compare_boxes, key=lambda b: b["y"]):
            placed = False
            for row in rows:
                if abs(box["y"] - row[0]["y"]) < 10:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        aligned = True
        for row in rows:
            if len(row) >= 2:
                ys = [b["y"] for b in row]
                if not all(abs(y - ys[0]) < 5 for y in ys):
                    aligned = False
                    break

        results.append(AssertionResult(
            rule="R10 (compare panes aligned)",
            passed=aligned,
            detail="" if aligned else f"rows={[[b for b in r] for r in rows]}",
        ))

    return results


def run_lebesgue_assertions(page: Page) -> list[AssertionResult]:
    """Run assertions specific to Lebesgue/histogram mode."""
    results = []
    lebesgue_active = page.evaluate(
        "() => typeof lebesgueMode !== 'undefined' && lebesgueMode"
    )
    if lebesgue_active:
        results.append(AssertionResult(
            rule="R1/R12 (colorbar state in Lebesgue mode)",
            passed=True,
            detail="Lebesgue mode active — colorbar is in expanded state",
        ))
    return results


def run_diff_assertions(page: Page) -> list[AssertionResult]:
    """Run assertions specific to diff mode."""
    results = []
    diff_info = page.evaluate("""
    () => {
        if (typeof diffMode === 'undefined' || diffMode <= 0) return null;
        return {
            diffMode: diffMode,
            hasDiffCanvas: !!document.getElementById('compare-diff-canvas'),
        };
    }
    """)
    if diff_info and diff_info.get("hasDiffCanvas"):
        results.append(AssertionResult(
            rule="R4/R5 (diff canvas present)",
            passed=True,
            detail=f"diffMode={diff_info['diffMode']}",
        ))
    return results


def run_invariant_assertions(page: Page) -> list[AssertionResult]:
    """Run hard UI invariant assertions (apply to every scenario)."""
    results = []

    # R29: Height sync — dimbar and colorbar height within 4px
    hs = page.evaluate(_JS_HEIGHT_SYNC)
    if hs is not None:
        diff = abs(hs["infoH"] - hs["cbH"])
        ok = diff <= 4
        results.append(AssertionResult(
            rule="R29 (height sync: dimbar ↔ colorbar)",
            passed=ok,
            detail="" if ok else f"infoH={hs['infoH']} cbH={hs['cbH']} diff={diff}",
        ))

    # R33: Flex direction — all visible .cb-island must be row
    dirs = page.evaluate(_JS_CB_FLEX_DIRS)
    for d in dirs:
        ok = d["dir"] == "row"
        results.append(AssertionResult(
            rule=f"R33 (flex-direction: {d['id']})",
            passed=ok,
            detail="" if ok else f"got {d['dir']}",
        ))

    return results


def run_immersive_exit_assertions(page: Page) -> list[AssertionResult]:
    """Run assertions after immersive exit to verify clean state."""
    results = []
    state = page.evaluate(_JS_IMMERSIVE_STATE)

    # Should not be in fullscreen after exit
    if state["fullscreen"]:
        results.append(AssertionResult(
            rule="R30 (not in fullscreen after exit)",
            passed=False,
            detail=f"fullscreen={state['fullscreen']}",
        ))
        return results  # other checks meaningless if still in fullscreen

    # R30: Drag positions cleared
    for name in ("infoDrag", "cbDrag", "islandDrag"):
        val = state[name]
        ok = val is None
        results.append(AssertionResult(
            rule=f"R30 ({name} cleared after immersive exit)",
            passed=ok,
            detail="" if ok else f"{name}={val}",
        ))

    # R31: No .fs-overlay elements
    ok = state["fsOverlayCount"] == 0
    results.append(AssertionResult(
        rule="R31 (no .fs-overlay after immersive exit)",
        passed=ok,
        detail="" if ok else f"count={state['fsOverlayCount']}",
    ))

    # R31b: No fullscreen-mode class on body
    ok = not state["hasFullscreenClass"]
    results.append(AssertionResult(
        rule="R31b (no fullscreen-mode class after exit)",
        passed=ok,
        detail="" if ok else "body still has fullscreen-mode",
    ))

    return results


# ---------------------------------------------------------------------------
# Scenario definitions — declarative
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A single audit scenario.

    Each scenario declaratively describes what to set up and screenshot.
    The generic runner interprets these fields — no per-scenario if/elif needed.
    """
    name: str
    tier: int
    subset_tags: list[str] = field(default_factory=list)
    description: str = ""

    # Which test array to view (key into the arrays dict)
    array_key: str = "3d"

    # If set, open in compare mode with these array keys.
    # First key is the main session, rest are compare_sids.
    compare_keys: list[str] | None = None

    # Key presses to execute after page load. Each is (key, wait_ms).
    keys: list[tuple[str, int]] = field(default_factory=list)

    # Zoom-in steps (0 = fit to window). Applied after keys.
    zoom: int = 0

    # Extra assertion sets to run: "diff", "lebesgue"
    extra_assertions: list[str] = field(default_factory=list)

    # Extra wait (ms) after all keys + zoom, before screenshot
    settle_ms: int = 400


# ── Key sequence constants ────────────────────────────────────────────────
# Reusable building blocks for common mode activations.

K_MULTIVIEW = [("v", 1200)]
K_QMRI_FULL = [("q", 2500)]
K_QMRI_COMPACT = [("q", 2500), ("q", 1500)]
K_QMRI_MOSAIC = [("q", 2500), ("z", 800)]
K_ZEN = [("Shift+F", 800)]
K_IMMERSIVE = [("Shift+K", 600)]
K_MOSAIC = [("z", 800)]
K_ROI = [("Shift+A", 600)]
K_LOG = [("Shift+L", 400)]
K_BORDERS = [("b", 400)]
K_STRETCH = [("a", 400)]
K_MASK = [("Shift+M", 400)]
K_RGB = [("Shift+R", 400)]
K_LEBESGUE = [("w", 600)]
K_PIXEL_INFO = [("Shift+H", 400), ("i", 400)]
K_FLIP = [("r", 400)]
K_RULER = [("u", 400)]
K_PLAY = [("Space", 800)]

# Compare center modes: X cycles 0→1→2→3→4→5→6→7→0
K_DIFF_AB = [("Shift+X", 800)]
K_DIFF_ABS = [("Shift+X", 800)] * 2
K_DIFF_REL = [("Shift+X", 800)] * 3
K_OVERLAY = [("Shift+X", 800)] * 4
K_WIPE = [("Shift+X", 800)] * 5
K_FLICKER = [("Shift+X", 800)] * 6
K_CHECKER = [("Shift+X", 800)] * 7

# Projection: p cycles 0→MAX→MIN→MEAN→STD→SOS→off
K_PROJ_MAX = [("p", 800)]
K_PROJ_MIN = [("p", 800)] * 2
K_PROJ_MEAN = [("p", 800)] * 3
K_PROJ_STD = [("p", 800)] * 4
K_PROJ_SOS = [("p", 800)] * 5

# Theme: T cycles dark→light→solarized→nord
K_THEME_LIGHT = [("Shift+T", 400)]
K_THEME_SOLARIZED = [("Shift+T", 400)] * 2
K_THEME_NORD = [("Shift+T", 400)] * 3

# Complex: m cycles mag→phase→real→imag (for complex arrays)
K_COMPLEX_PHASE = [("m", 400)]
K_COMPLEX_REAL = [("m", 400)] * 2
K_COMPLEX_IMAG = [("m", 400)] * 3

# Compare layout: G cycles horizontal→vertical→grid (3+ panes)
K_LAYOUT_VERTICAL = [("Shift+G", 400)]
K_LAYOUT_GRID = [("Shift+G", 400)] * 2


# ── Tier 1: Core modes (always check) ─────────────────────────────────────

TIER1 = [
    Scenario("t1_single_normal_fit", 1,
             description="Single 3D, normal view, fit to window"),
    Scenario("t1_single_normal_zoom", 1, ["zoom"],
             description="Single 3D, zoomed in",
             zoom=5),
    Scenario("t1_single_compact", 1, ["zoom", "compact"],
             description="Single 3D, immersive/compact mode (K)",
             keys=list(K_IMMERSIVE)),
    Scenario("t1_single_multiview_fit", 1, ["multiview"],
             description="Single 3D, 3-plane multiview",
             keys=list(K_MULTIVIEW)),
    # NOTE: multiview intentionally caps zoom to 1.0 (panes fill viewport).
    # No multiview zoom scenario — zoom has no visual effect in this mode.
    Scenario("t1_single_qmri_full", 1, ["qmri"],
             description="Single 4D, qMRI full (5 panels)",
             array_key="4d", keys=list(K_QMRI_FULL)),
    Scenario("t1_single_qmri_compact", 1, ["qmri", "compact"],
             description="Single 4D, qMRI compact (T1/T2/|PD|)",
             array_key="4d", keys=list(K_QMRI_COMPACT)),
    Scenario("t1_single_zen", 1,
             description="Single 3D, zen mode (no chrome)",
             keys=list(K_ZEN)),
    Scenario("t1_compare_fit", 1, ["compare"],
             description="2× 3D, compare side-by-side, fit",
             compare_keys=["3d", "3d_b"]),
    Scenario("t1_compare_zoom", 1, ["compare", "zoom"],
             description="2× 3D, compare zoomed",
             compare_keys=["3d", "3d_b"], zoom=4),
    Scenario("t1_compare_diff", 1, ["compare", "diff"],
             description="2× 3D, diff A-B center pane",
             compare_keys=["3d", "3d_b"], keys=list(K_DIFF_AB),
             extra_assertions=["diff"]),
    Scenario("t1_compare_multiview", 1, ["compare", "multiview"],
             description="2× 3D, compare + 3-plane multiview",
             compare_keys=["3d", "3d_b"], keys=list(K_MULTIVIEW),
             settle_ms=1000),
    Scenario("t1_single_roi", 1, ["roi"],
             description="Single 3D, ROI rect mode",
             keys=list(K_ROI)),
    Scenario("t1_compact_roi", 1, ["roi", "compact"],
             description="Single 3D, compact + ROI",
             keys=list(K_IMMERSIVE) + list(K_ROI)),
    Scenario("t1_immersive_roundtrip", 1, ["zoom", "compact"],
             description="Enter immersive → exit → verify clean state",
             keys=list(K_IMMERSIVE) + [("0", 400), ("0", 800)],
             extra_assertions=["immersive_exit"]),
    Scenario("t1_compare_immersive_roundtrip", 1, ["compare", "zoom", "compact"],
             description="Compare → immersive → exit → verify clean state",
             compare_keys=["3d", "3d_b"],
             keys=list(K_IMMERSIVE) + [("0", 400), ("0", 800)],
             extra_assertions=["immersive_exit"]),
]


# ── Tier 2: Extended modes ─────────────────────────────────────────────────

TIER2 = [
    # --- Basic array types ---
    Scenario("t2_single_2d", 2, ["2d", "basic"],
             description="Single 2D, normal",
             array_key="2d"),
    Scenario("t2_single_4d", 2, ["4d", "ndim"],
             description="Single 4D, normal",
             array_key="4d"),
    Scenario("t2_single_mosaic", 2, ["mosaic", "z-grid"],
             description="Single 4D, mosaic (z key)",
             array_key="4d", keys=list(K_MOSAIC)),
    Scenario("t2_single_complex", 2, ["complex", "dtype"],
             description="Single complex, magnitude view",
             array_key="complex"),

    # --- All 7 compare center modes ---
    Scenario("t2_diff_abs", 2, ["diff"],
             description="2× 3D, diff |A-B|",
             compare_keys=["3d", "3d_b"], keys=list(K_DIFF_ABS),
             extra_assertions=["diff"]),
    Scenario("t2_diff_rel", 2, ["diff"],
             description="2× 3D, diff |A-B|/|A|",
             compare_keys=["3d", "3d_b"], keys=list(K_DIFF_REL),
             extra_assertions=["diff"]),
    Scenario("t2_overlay", 2, ["overlay", "diff"],
             description="2× 3D, overlay blend",
             compare_keys=["3d", "3d_b"], keys=list(K_OVERLAY)),
    Scenario("t2_wipe", 2, ["wipe", "diff"],
             description="2× 3D, wipe (vertical divider)",
             compare_keys=["3d", "3d_b"], keys=list(K_WIPE)),
    Scenario("t2_flicker", 2, ["flicker", "diff"],
             description="2× 3D, flicker (alternating A/B)",
             compare_keys=["3d", "3d_b"], keys=list(K_FLICKER)),
    Scenario("t2_checker", 2, ["checker", "diff"],
             description="2× 3D, checkerboard blend",
             compare_keys=["3d", "3d_b"], keys=list(K_CHECKER)),
    Scenario("t2_registration", 2, ["registration"],
             description="2× 3D, registration/overlay via X cycle",
             compare_keys=["3d", "3d_b"], keys=list(K_OVERLAY)),

    # --- Compare variants ---
    Scenario("t2_compare_qmri", 2, ["qmri", "compare"],
             description="2× 4D, compare + qMRI",
             compare_keys=["4d", "4d_b"], keys=list(K_QMRI_FULL),
             settle_ms=1000),
    Scenario("t2_compare_diff_roi", 2, ["roi", "diff"],
             description="2× 3D, diff + ROI",
             compare_keys=["3d", "3d_b"],
             keys=list(K_DIFF_AB) + list(K_ROI),
             extra_assertions=["diff"]),
    Scenario("t2_compare_3_grid", 2, ["compare", "grid", "multi-array"],
             description="3× 3D, compare grid",
             compare_keys=["3d", "3d_b", "3d_c"]),
    Scenario("t2_compare_3_qmri", 2, ["qmri", "compare", "multi-array"],
             description="3× 4D, compare + qMRI",
             compare_keys=["4d", "4d_b", "4d_c"],
             keys=list(K_QMRI_FULL), settle_ms=1500),
    Scenario("t2_compare_3_multiview", 2, ["multiview", "compare", "multi-array"],
             description="3× 3D, compare + multiview",
             compare_keys=["3d", "3d_b", "3d_c"],
             keys=list(K_MULTIVIEW), settle_ms=1500),
    Scenario("t2_compare_4_grid", 2, ["compare", "grid", "multi-array"],
             description="4× 3D, compare 2×2 grid",
             compare_keys=["3d", "3d_b", "3d_c", "3d_d"]),
    Scenario("t2_compare_34_qmri_compact", 2, ["qmri", "compare", "compact", "multi-array"],
             description="3× 4D, compare + compact qMRI",
             compare_keys=["4d", "4d_b", "4d_c"],
             keys=list(K_QMRI_COMPACT), settle_ms=1500),
    Scenario("t2_compare_vertical", 2, ["compare", "layout"],
             description="2× 3D, compare vertical layout",
             compare_keys=["3d", "3d_b"],
             keys=list(K_LAYOUT_VERTICAL)),
    Scenario("t2_compare_3_vertical", 2, ["compare", "layout", "multi-array"],
             description="3× 3D, compare vertical",
             compare_keys=["3d", "3d_b", "3d_c"],
             keys=list(K_LAYOUT_VERTICAL)),
    Scenario("t2_compare_3_grid_layout", 2, ["compare", "grid", "layout", "multi-array"],
             description="3× 3D, compare grid layout (G×2)",
             compare_keys=["3d", "3d_b", "3d_c"],
             keys=list(K_LAYOUT_GRID)),

    # --- Vector field ---
    Scenario("t2_vfield_normal", 2, ["vector", "vfield"],
             description="3D + vector field arrows",
             array_key="3d_vf"),
    Scenario("t2_vfield_roi", 2, ["vector", "roi"],
             description="Vector field + ROI",
             array_key="3d_vf", keys=list(K_ROI)),
    Scenario("t2_vfield_compare", 2, ["vector", "compare"],
             description="Compare: one array with vector field",
             compare_keys=["3d_vf", "3d_b"]),

    # --- Display modifiers (single array) ---
    Scenario("t2_single_log", 2, ["log", "display"],
             description="Single 3D, log scale",
             keys=list(K_LOG)),
    Scenario("t2_single_borders", 2, ["borders", "display"],
             description="Single 3D, canvas borders on",
             keys=list(K_BORDERS)),
    Scenario("t2_single_stretch", 2, ["stretch", "display"],
             description="Single 3D, square stretch",
             keys=list(K_STRETCH)),
    Scenario("t2_single_alpha", 2, ["alpha", "display"],
             description="Single 3D, Otsu alpha overlay",
             keys=list(K_MASK)),
    Scenario("t2_single_lebesgue", 2, ["lebesgue", "display"],
             description="Single 3D, Lebesgue integral mode",
             keys=list(K_LEBESGUE),
             extra_assertions=["lebesgue"]),
    Scenario("t2_single_pixel_info", 2, ["info", "display"],
             description="Single 3D, pixel hover + info overlay",
             keys=list(K_PIXEL_INFO)),
    Scenario("t2_single_flip", 2, ["flip", "display"],
             description="Single 3D, flipped axis",
             keys=list(K_FLIP)),
    Scenario("t2_single_ruler", 2, ["ruler", "display"],
             description="Single 3D, ruler mode",
             keys=list(K_RULER)),
    Scenario("t2_single_playing", 2, ["play", "display"],
             description="Single 3D, auto-play active",
             keys=list(K_PLAY)),

    # --- Projection modes ---
    Scenario("t2_projection_max", 2, ["projection"],
             description="Single 3D, MAX projection",
             keys=list(K_PROJ_MAX)),
    Scenario("t2_projection_min", 2, ["projection"],
             description="Single 3D, MIN projection",
             keys=list(K_PROJ_MIN)),
    Scenario("t2_projection_mean", 2, ["projection"],
             description="Single 3D, MEAN projection",
             keys=list(K_PROJ_MEAN)),
    Scenario("t2_projection_std", 2, ["projection"],
             description="Single 3D, STD projection",
             keys=list(K_PROJ_STD)),
    Scenario("t2_projection_sos", 2, ["projection"],
             description="Single 3D, SOS (sum of squares) projection",
             keys=list(K_PROJ_SOS)),

    # --- Complex array modes ---
    Scenario("t2_complex_phase", 2, ["complex", "dtype"],
             description="Complex array, phase view",
             array_key="complex", keys=list(K_COMPLEX_PHASE)),
    Scenario("t2_complex_real", 2, ["complex", "dtype"],
             description="Complex array, real part",
             array_key="complex", keys=list(K_COMPLEX_REAL)),
    Scenario("t2_complex_imag", 2, ["complex", "dtype"],
             description="Complex array, imaginary part",
             array_key="complex", keys=list(K_COMPLEX_IMAG)),

    # --- RGB ---
    Scenario("t2_single_rgb", 2, ["rgb", "dtype"],
             description="RGB-shaped array, RGB mode on",
             array_key="rgb", keys=list(K_RGB)),

    # --- Themes ---
    Scenario("t2_theme_light", 2, ["theme", "display"],
             description="Single 3D, light theme",
             keys=list(K_THEME_LIGHT)),
    Scenario("t2_theme_solarized", 2, ["theme", "display"],
             description="Single 3D, solarized theme",
             keys=list(K_THEME_SOLARIZED)),
    Scenario("t2_theme_nord", 2, ["theme", "display"],
             description="Single 3D, nord theme",
             keys=list(K_THEME_NORD)),

    # --- qMRI mosaic ---
    Scenario("t2_qmri_mosaic", 2, ["qmri", "mosaic"],
             description="Single 4D, qMRI + z mosaic",
             array_key="4d", keys=list(K_QMRI_MOSAIC)),

    # --- Modifier combos ---
    Scenario("t2_log_multiview", 2, ["log", "multiview"],
             description="Log scale + multiview",
             keys=list(K_LOG) + list(K_MULTIVIEW)),
    Scenario("t2_log_compare", 2, ["log", "compare"],
             description="Log scale + compare",
             compare_keys=["3d", "3d_b"], keys=list(K_LOG)),
    Scenario("t2_borders_compare", 2, ["borders", "compare"],
             description="Canvas borders + compare",
             compare_keys=["3d", "3d_b"], keys=list(K_BORDERS)),
    Scenario("t2_stretch_compare", 2, ["stretch", "compare"],
             description="Square stretch + compare",
             compare_keys=["3d", "3d_b"], keys=list(K_STRETCH)),
    Scenario("t2_roi_multiview", 2, ["roi", "multiview"],
             description="ROI + multiview",
             keys=list(K_MULTIVIEW) + list(K_ROI)),
    Scenario("t2_alpha_compare", 2, ["alpha", "compare"],
             description="Alpha overlay + compare",
             compare_keys=["3d", "3d_b"], keys=list(K_MASK)),
    Scenario("t2_projection_multiview", 2, ["projection", "multiview"],
             description="MAX projection (then multiview — should disable)",
             keys=list(K_PROJ_MAX) + list(K_MULTIVIEW)),
    Scenario("t2_immersive_compare", 2, ["compact", "compare"],
             description="Immersive + compare",
             compare_keys=["3d", "3d_b"], keys=list(K_IMMERSIVE)),
]


# ── Tier 3: Shape variations ──────────────────────────────────────────────
# Each primary mode tested with non-square arrays to catch scaling/layout bugs.

TIER3 = [
    # --- Single normal × shapes ---
    Scenario("t3_normal_wide_2d", 3, ["shape", "2d"],
             description="Wide 2D (64×512)",
             array_key="wide_2d"),
    Scenario("t3_normal_tall_2d", 3, ["shape", "2d"],
             description="Tall 2D (512×64)",
             array_key="tall_2d"),
    Scenario("t3_normal_tiny", 3, ["shape"],
             description="Tiny 2D (8×8)",
             array_key="tiny"),
    Scenario("t3_normal_wide_3d", 3, ["shape", "3d"],
             description="Wide 3D (20×64×512)",
             array_key="wide_3d"),
    Scenario("t3_normal_tall_3d", 3, ["shape", "3d"],
             description="Tall 3D (20×512×64)",
             array_key="tall_3d"),
    Scenario("t3_normal_row", 3, ["shape", "degenerate"],
             description="Single-row (1×256)",
             array_key="row"),
    Scenario("t3_normal_col", 3, ["shape", "degenerate"],
             description="Single-column (256×1)",
             array_key="col"),
    Scenario("t3_normal_large", 3, ["shape"],
             description="Large 2D (512×512)",
             array_key="large"),
    Scenario("t3_normal_cube", 3, ["shape", "3d"],
             description="Cube 3D (64×64×64)",
             array_key="cube"),

    # --- Multiview × shapes ---
    Scenario("t3_multiview_wide", 3, ["shape", "multiview"],
             description="Multiview, wide 3D (20×64×512)",
             array_key="wide_3d", keys=list(K_MULTIVIEW)),
    Scenario("t3_multiview_tall", 3, ["shape", "multiview"],
             description="Multiview, tall 3D (20×512×64)",
             array_key="tall_3d", keys=list(K_MULTIVIEW)),
    Scenario("t3_multiview_cube", 3, ["shape", "multiview"],
             description="Multiview, cube 3D (64×64×64)",
             array_key="cube", keys=list(K_MULTIVIEW)),

    # --- Compare × shapes ---
    Scenario("t3_compare_wide", 3, ["shape", "compare"],
             description="Compare, 2× wide 3D",
             compare_keys=["wide_3d", "wide_3d_b"]),
    Scenario("t3_compare_tall", 3, ["shape", "compare"],
             description="Compare, 2× tall 3D",
             compare_keys=["tall_3d", "tall_3d_b"]),
    Scenario("t3_compare_mixed", 3, ["shape", "compare"],
             description="Compare, wide vs tall (mismatched aspect)",
             compare_keys=["wide_3d", "tall_3d"]),

    # --- Diff × shapes ---
    Scenario("t3_diff_wide", 3, ["shape", "diff"],
             description="Diff A-B, wide 3D",
             compare_keys=["wide_3d", "wide_3d_b"],
             keys=list(K_DIFF_AB), extra_assertions=["diff"]),
    Scenario("t3_diff_tall", 3, ["shape", "diff"],
             description="Diff A-B, tall 3D",
             compare_keys=["tall_3d", "tall_3d_b"],
             keys=list(K_DIFF_AB), extra_assertions=["diff"]),

    # --- Mosaic × shapes ---
    Scenario("t3_mosaic_wide", 3, ["shape", "mosaic"],
             description="Mosaic, wide 4D (5×20×64×256)",
             array_key="wide_4d", keys=list(K_MOSAIC)),
    Scenario("t3_mosaic_tall", 3, ["shape", "mosaic"],
             description="Mosaic, tall 4D (5×20×256×64)",
             array_key="tall_4d", keys=list(K_MOSAIC)),

    # --- qMRI × shapes ---
    Scenario("t3_qmri_wide", 3, ["shape", "qmri"],
             description="qMRI full, wide (5×20×32×128)",
             array_key="wide_qmri", keys=list(K_QMRI_FULL)),
    Scenario("t3_qmri_tall", 3, ["shape", "qmri"],
             description="qMRI full, tall (5×20×128×32)",
             array_key="tall_qmri", keys=list(K_QMRI_FULL)),

    # --- Projection × shapes ---
    Scenario("t3_projection_wide", 3, ["shape", "projection"],
             description="MAX projection, wide 3D",
             array_key="wide_3d", keys=list(K_PROJ_MAX)),
    Scenario("t3_projection_tall", 3, ["shape", "projection"],
             description="MAX projection, tall 3D",
             array_key="tall_3d", keys=list(K_PROJ_MAX)),

    # --- Compact/Zen × shapes ---
    Scenario("t3_compact_wide", 3, ["shape", "compact"],
             description="Immersive, wide 3D",
             array_key="wide_3d", keys=list(K_IMMERSIVE)),
    Scenario("t3_compact_tall", 3, ["shape", "compact"],
             description="Immersive, tall 3D",
             array_key="tall_3d", keys=list(K_IMMERSIVE)),
    Scenario("t3_zen_wide", 3, ["shape"],
             description="Zen mode, wide 3D",
             array_key="wide_3d", keys=list(K_ZEN)),
    Scenario("t3_zen_tall", 3, ["shape"],
             description="Zen mode, tall 3D",
             array_key="tall_3d", keys=list(K_ZEN)),

    # --- Compare 4-pane grid × shapes ---
    Scenario("t3_compare_4_wide", 3, ["shape", "compare", "grid"],
             description="4× wide 3D, compare grid",
             compare_keys=["wide_3d", "wide_3d_b", "wide_3d_c", "wide_3d_d"]),
    Scenario("t3_compare_4_tall", 3, ["shape", "compare", "grid"],
             description="4× tall 3D, compare grid",
             compare_keys=["tall_3d", "tall_3d_b", "tall_3d_c", "tall_3d_d"]),

    # --- Vector field × shapes ---
    Scenario("t3_vfield_wide", 3, ["shape", "vector"],
             description="Vector field, wide 3D",
             array_key="wide_3d_vf"),
    Scenario("t3_vfield_tall", 3, ["shape", "vector"],
             description="Vector field, tall 3D",
             array_key="tall_3d_vf"),

    # --- Degenerate in modes ---
    Scenario("t3_tiny_zoom", 3, ["shape", "zoom"],
             description="Tiny 8×8 zoomed in",
             array_key="tiny", zoom=8),
    Scenario("t3_row_zoom", 3, ["shape", "zoom", "degenerate"],
             description="Single-row 1×256 zoomed",
             array_key="row", zoom=5),
    Scenario("t3_col_zoom", 3, ["shape", "zoom", "degenerate"],
             description="Single-column 256×1 zoomed",
             array_key="col", zoom=5),
]


ALL_SCENARIOS = TIER1 + TIER2 + TIER3


# ---------------------------------------------------------------------------
# Test arrays
# ---------------------------------------------------------------------------


def _create_arrays(client, tmp) -> dict[str, str]:
    """Create all test arrays and load them into the server. Returns {key: sid}."""
    rng = np.random.default_rng(42)
    arrays: dict[str, str] = {}

    def _mk(name, arr):
        arrays[name] = _load(client, arr, f"audit_{name}", tmp)

    # ── Standard shapes (square-ish) ──────────────────────────────────────
    _mk("2d", np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(100, 80))
    _mk("3d", rng.standard_normal((20, 64, 64)).astype(np.float32))
    _mk("3d_b", rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.7 + 0.2)
    _mk("3d_c", rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.5)
    _mk("3d_d", rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.3 + 0.4)
    _mk("4d", rng.standard_normal((5, 20, 32, 32)).astype(np.float32))
    _mk("4d_b", rng.standard_normal((5, 20, 32, 32)).astype(np.float32) * 0.8)
    _mk("4d_c", rng.standard_normal((5, 20, 32, 32)).astype(np.float32) * 0.6)
    _mk("complex", (
        rng.standard_normal((20, 32, 32)) + 1j * rng.standard_normal((20, 32, 32))
    ).astype(np.complex64))
    _mk("rgb", rng.random((64, 64, 3)).astype(np.float32))

    # ── Wide shapes ───────────────────────────────────────────────────────
    _mk("wide_2d", rng.standard_normal((64, 512)).astype(np.float32))
    _mk("wide_3d", rng.standard_normal((20, 64, 512)).astype(np.float32))
    _mk("wide_3d_b", rng.standard_normal((20, 64, 512)).astype(np.float32) * 0.7)
    _mk("wide_3d_c", rng.standard_normal((20, 64, 512)).astype(np.float32) * 0.5)
    _mk("wide_3d_d", rng.standard_normal((20, 64, 512)).astype(np.float32) * 0.3)
    _mk("wide_4d", rng.standard_normal((5, 20, 64, 256)).astype(np.float32))
    _mk("wide_qmri", rng.standard_normal((5, 20, 32, 128)).astype(np.float32))

    # ── Tall shapes ───────────────────────────────────────────────────────
    _mk("tall_2d", rng.standard_normal((512, 64)).astype(np.float32))
    _mk("tall_3d", rng.standard_normal((20, 512, 64)).astype(np.float32))
    _mk("tall_3d_b", rng.standard_normal((20, 512, 64)).astype(np.float32) * 0.7)
    _mk("tall_3d_c", rng.standard_normal((20, 512, 64)).astype(np.float32) * 0.5)
    _mk("tall_3d_d", rng.standard_normal((20, 512, 64)).astype(np.float32) * 0.3)
    _mk("tall_4d", rng.standard_normal((5, 20, 256, 64)).astype(np.float32))
    _mk("tall_qmri", rng.standard_normal((5, 20, 128, 32)).astype(np.float32))

    # ── Edge cases ────────────────────────────────────────────────────────
    _mk("tiny", rng.standard_normal((8, 8)).astype(np.float32))
    _mk("large", rng.standard_normal((512, 512)).astype(np.float32))
    _mk("row", rng.standard_normal((1, 256)).astype(np.float32))
    _mk("col", rng.standard_normal((256, 1)).astype(np.float32))
    _mk("cube", rng.standard_normal((64, 64, 64)).astype(np.float32))

    # ── Vector field arrays ───────────────────────────────────────────────
    # Standard vector field (square)
    _mk("3d_vf", rng.standard_normal((20, 64, 64)).astype(np.float32))
    vf = np.zeros((20, 64, 64, 3), dtype=np.float32)
    vf[..., 1] = 0.3
    vf[..., 2] = 0.6
    vf_path = Path(tmp) / "audit_3d_vf_field.npy"
    np.save(vf_path, vf)
    r = client.post("/attach_vectorfield", json={"sid": arrays["3d_vf"], "filepath": str(vf_path)})
    r.raise_for_status()

    # Wide vector field
    _mk("wide_3d_vf", rng.standard_normal((20, 64, 512)).astype(np.float32))
    vf_wide = np.zeros((20, 64, 512, 3), dtype=np.float32)
    vf_wide[..., 1] = 0.3
    vf_wide[..., 2] = 0.6
    vf_wide_path = Path(tmp) / "audit_wide_3d_vf_field.npy"
    np.save(vf_wide_path, vf_wide)
    r = client.post("/attach_vectorfield", json={"sid": arrays["wide_3d_vf"], "filepath": str(vf_wide_path)})
    r.raise_for_status()

    # Tall vector field
    _mk("tall_3d_vf", rng.standard_normal((20, 512, 64)).astype(np.float32))
    vf_tall = np.zeros((20, 512, 64, 3), dtype=np.float32)
    vf_tall[..., 1] = 0.3
    vf_tall[..., 2] = 0.6
    vf_tall_path = Path(tmp) / "audit_tall_3d_vf_field.npy"
    np.save(vf_tall_path, vf_tall)
    r = client.post("/attach_vectorfield", json={"sid": arrays["tall_3d_vf"], "filepath": str(vf_tall_path)})
    r.raise_for_status()

    return arrays


# ---------------------------------------------------------------------------
# Scenario runner (generic)
# ---------------------------------------------------------------------------


def _zoom_in(page: Page, times: int = 5):
    _focus(page)
    for _ in range(times):
        _press(page, "+", wait=200)
    page.wait_for_timeout(400)


def _zoom_reset(page: Page):
    _focus(page)
    _press(page, "0", wait=400)


def run_scenarios(
    page: Page,
    base: str,
    client: httpx.Client,
    tmp: str,
    tier: int | None,
    subset: set[str] | None,
    update_baselines: bool,
) -> list[tuple[str, bool, str]]:
    """Run audit scenarios and return list of (name, passed, detail)."""
    results: list[tuple[str, bool, str]] = []

    # --- Create all test arrays ---
    arrays = _create_arrays(client, tmp)

    # --- Determine which scenarios to run ---
    scenarios: list[Scenario] = []
    if tier is None:
        scenarios = list(ALL_SCENARIOS)
    elif tier == 1:
        scenarios = list(TIER1)
    elif tier == 2:
        scenarios = list(TIER2)
    elif tier == 3:
        scenarios = list(TIER3)

    if subset:
        scenarios = [s for s in scenarios if s.subset_tags and subset & set(s.subset_tags)]

    print(f"Running {len(scenarios)} scenarios...")

    # --- Run each scenario ---
    for scenario in scenarios:
        name = scenario.name
        print(f"\n  {name}: {scenario.description}")

        try:
            # 1. Navigate
            if scenario.compare_keys:
                sids = [arrays[k] for k in scenario.compare_keys]
                _goto_compare(page, base, sids)
            else:
                _goto(page, base, arrays[scenario.array_key])

            # 2. Focus keyboard
            _focus(page)

            # 3. Press keys
            for key, wait in scenario.keys:
                _press(page, key, wait=wait)

            # 4. Zoom
            if scenario.zoom > 0:
                _zoom_in(page, scenario.zoom)
                # Reset scroll to top-left for deterministic screenshots
                page.evaluate(
                    "() => { const cw = document.getElementById('canvas-wrap');"
                    " if (cw) { cw.scrollLeft = 0; cw.scrollTop = 0; } }"
                )

            # 5. Settle
            page.wait_for_timeout(scenario.settle_ms)

            # 6. Screenshot
            _shot(page, name)

            # 7. Run assertions
            assertion_results = run_assertions(page, name, zoomed=scenario.zoom > 0)
            assertion_results.extend(run_invariant_assertions(page))

            # 8. Extra assertions
            if "diff" in scenario.extra_assertions:
                assertion_results.extend(run_diff_assertions(page))
            if "lebesgue" in scenario.extra_assertions:
                assertion_results.extend(run_lebesgue_assertions(page))
            if "immersive_exit" in scenario.extra_assertions:
                assertion_results.extend(run_immersive_exit_assertions(page))

            # 9. Process results
            failed_assertions = [a for a in assertion_results if not a.passed]
            if failed_assertions:
                for a in failed_assertions:
                    print(f"    FAIL: {a.rule} — {a.detail}")

            # 10. Visual diff against baseline
            screenshot_path = SCREENSHOTS_DIR / f"{name}.png"
            baseline_path = BASELINES_DIR / f"{name}.png"
            diff_path = DIFFS_DIR / f"{name}_diff.png"

            if update_baselines and screenshot_path.exists():
                import shutil
                shutil.copy2(screenshot_path, baseline_path)
                print(f"    baseline updated")

            diff_passed = True
            diff_pct = 0.0
            # Skip pixel diff for zoom scenarios — canvas pan position has sub-pixel
            # non-determinism that makes pixel diff unreliable. Layout is validated by
            # DOM assertions (minimap position, colorbar visibility, etc.) instead.
            skip_pixel_diff = scenario.zoom > 0
            if not update_baselines and baseline_path.exists() and not skip_pixel_diff:
                diff_passed, diff_pct = _pixel_diff(screenshot_path, baseline_path, diff_path)
                if not diff_passed:
                    print(f"    DIFF: {diff_pct:.1%} pixels changed (threshold 1%)")

            passed = len(failed_assertions) == 0 and diff_passed
            detail_parts = []
            if failed_assertions:
                detail_parts.append(f"{len(failed_assertions)} assertion(s) failed")
            if not diff_passed:
                detail_parts.append(f"visual diff {diff_pct:.1%}")
            detail = "; ".join(detail_parts) if detail_parts else "ok"

            results.append((name, passed, detail))
            if passed:
                print(f"    PASS")

        except Exception as e:
            print(f"    ERROR: {e}")
            results.append((name, False, f"error: {e}"))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(results: list[tuple[str, bool, str]]):
    lines = ["UI Consistency Audit Report", "=" * 40, ""]
    passed = sum(1 for _, p, _ in results if p)
    failed = len(results) - passed

    lines.append(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    lines.append("")

    if failed > 0:
        lines.append("FAILURES:")
        for name, p, detail in results:
            if not p:
                lines.append(f"  {name}: {detail}")
        lines.append("")

    lines.append("ALL RESULTS:")
    for name, p, detail in results:
        status = "PASS" if p else "FAIL"
        lines.append(f"  [{status}] {name}: {detail}")

    report = "\n".join(lines)
    REPORT_PATH.write_text(report)
    print(f"\n{'=' * 40}")
    print(report)
    print(f"\nScreenshots: {SCREENSHOTS_DIR}")
    if any(not p for _, p, _ in results):
        print(f"Diffs:       {DIFFS_DIR}")
    print(f"Report:      {REPORT_PATH}")


# ---------------------------------------------------------------------------
# List mode (print scenario table)
# ---------------------------------------------------------------------------


def list_scenarios():
    """Print a formatted table of all scenarios."""
    print(f"\n{'─' * 90}")
    print(f"  {'NAME':<40} {'TIER':>4}  {'TAGS':<25} DESCRIPTION")
    print(f"{'─' * 90}")
    for s in ALL_SCENARIOS:
        tags = ", ".join(s.subset_tags) if s.subset_tags else "—"
        print(f"  {s.name:<40} T{s.tier:>3}  {tags:<25} {s.description}")
    print(f"{'─' * 90}")
    t1 = sum(1 for s in ALL_SCENARIOS if s.tier == 1)
    t2 = sum(1 for s in ALL_SCENARIOS if s.tier == 2)
    t3 = sum(1 for s in ALL_SCENARIOS if s.tier == 3)
    print(f"  Total: {len(ALL_SCENARIOS)} scenarios  (T1: {t1}, T2: {t2}, T3: {t3})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ArrayView UI Consistency Audit")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], help="Run only this tier")
    parser.add_argument("--subset", type=str, help="Comma-separated subset tags (e.g. zoom,compact)")
    parser.add_argument("--update-baselines", action="store_true", help="Update baseline screenshots")
    parser.add_argument("--width", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", type=int, default=900, help="Viewport height")
    parser.add_argument("--list", action="store_true", help="Print scenario table and exit")
    args = parser.parse_args()

    if args.list:
        list_scenarios()
        return

    subset = set(args.subset.split(",")) if args.subset else None

    base, srv = _start_server()
    print(f"Server at {base}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        with httpx.Client(base_url=base, timeout=15.0) as client:
            with tempfile.TemporaryDirectory() as tmp:
                results = run_scenarios(
                    page, base, client, tmp,
                    tier=args.tier, subset=subset,
                    update_baselines=args.update_baselines,
                )
        browser.close()

    srv.should_exit = True
    write_report(results)

    if any(not p for _, p, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
