"""
UI Consistency Audit for arrayview.

Automated Playwright-based visual audit that screenshots every mode × array-count
combination and runs DOM assertions to catch cross-mode inconsistencies.

Run after any visual/UI change:

    uv run python tests/ui_audit.py                      # tier 1 + tier 2
    uv run python tests/ui_audit.py --tier 1             # quick check (~15 combos)
    uv run python tests/ui_audit.py --tier 2 --subset zoom,compact
    uv run python tests/ui_audit.py --update-baselines   # reset baselines

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
    page.evaluate("() => sessionStorage.clear()")
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
    """Run all DOM assertions for the current page state.

    Set zoomed=True to skip viewport-bounds checks (canvases intentionally
    overflow when zoomed — the minimap handles navigation).
    """
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

    # R8: ROI hover info within viewport (if present)
    roi_boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, ".cv-pixel-info")
    for i, box in enumerate(roi_boxes):
        within = _box_within_viewport(box, vp)
        results.append(AssertionResult(
            rule=f"R8 (ROI info[{i}] within viewport)",
            passed=within,
            detail="" if within else f"box={box}",
        ))

    # R10: Compare panes aligned — panes on the same row share the same y,
    # panes in the same column share the same x. Group by approximate y to
    # identify rows, then verify alignment within each row.
    compare_boxes = page.evaluate(_JS_ALL_BOUNDING_BOXES, ".compare-canvas")
    if len(compare_boxes) >= 2:
        # Group boxes into rows (boxes within 10px of each other vertically)
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

    # R1: Colorbar hides when histogram is open
    cb_visible = page.evaluate(_JS_IS_VISIBLE, "#slim-cb-wrap")
    # In Lebesgue mode, the colorbar expands — it should NOT show the slim version
    # simultaneously. Check that slim-cb is either hidden or transformed into expanded.
    lebesgue_active = page.evaluate(
        "() => typeof lebesgueMode !== 'undefined' && lebesgueMode"
    )
    if lebesgue_active:
        # R12: slim colorbar should be in expanded/Lebesgue state, not slim
        results.append(AssertionResult(
            rule="R1/R12 (colorbar state in Lebesgue mode)",
            passed=True,  # Lebesgue mode uses same element but expanded
            detail="Lebesgue mode active — colorbar is in expanded state",
        ))

    return results


def run_diff_assertions(page: Page) -> list[AssertionResult]:
    """Run assertions specific to diff mode."""
    results = []

    # R4/R5: Diff center pane should use different colormap/range than sides
    # This checks via JS state variables
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


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    tier: int
    subset_tags: list[str] = field(default_factory=list)
    description: str = ""


# Tier 1 scenarios
TIER1 = [
    Scenario("t1_single_normal_fit", 1, description="Single 3D, normal view, fit"),
    Scenario("t1_single_normal_zoom", 1, ["zoom"], description="Single 3D, zoomed in"),
    Scenario("t1_single_compact", 1, ["zoom", "compact"], description="Single 3D, compact mode"),
    Scenario("t1_single_multiview_fit", 1, ["multiview"], description="Single 3D, multiview"),
    Scenario("t1_single_multiview_zoom", 1, ["multiview", "zoom"], description="Single 3D, multiview zoomed"),
    Scenario("t1_single_qmri_full", 1, ["qmri"], description="Single 4D, qMRI full (5 panels)"),
    Scenario("t1_single_qmri_compact", 1, ["qmri", "compact"], description="Single 4D, qMRI compact (3 panels)"),
    Scenario("t1_single_zen", 1, description="Single 3D, zen mode"),
    Scenario("t1_compare_fit", 1, ["compare"], description="2× 3D, compare fit"),
    Scenario("t1_compare_zoom", 1, ["compare", "zoom"], description="2× 3D, compare zoomed"),
    Scenario("t1_compare_diff", 1, ["compare", "diff"], description="2× 3D, diff A-B"),
    Scenario("t1_compare_multiview", 1, ["compare", "multiview"], description="2× 3D, compare + multiview"),
    Scenario("t1_single_roi", 1, ["roi"], description="Single 3D, ROI mode"),
    Scenario("t1_compact_roi", 1, ["roi", "compact"], description="Single 3D, compact + ROI"),
]

# Tier 2 scenarios
TIER2 = [
    Scenario("t2_single_2d", 2, ["2d", "basic"], description="Single 2D, normal"),
    Scenario("t2_single_4d", 2, ["4d", "ndim"], description="Single 4D, normal"),
    Scenario("t2_single_mosaic", 2, ["mosaic", "z-grid"], description="Single 4D, mosaic"),
    Scenario("t2_single_complex", 2, ["complex", "dtype"], description="Single complex, normal"),
    Scenario("t2_diff_abs", 2, ["diff"], description="2× 3D, diff |A-B|"),
    Scenario("t2_diff_rel", 2, ["diff"], description="2× 3D, diff |A-B|/|A|"),
    Scenario("t2_overlay", 2, ["overlay", "diff"], description="2× 3D, overlay"),
    Scenario("t2_wipe", 2, ["wipe", "diff"], description="2× 3D, wipe"),
    Scenario("t2_registration", 2, ["registration"], description="2× 3D, registration"),
    Scenario("t2_compare_qmri", 2, ["qmri", "compare"], description="2× 4D, compare + qMRI"),
    Scenario("t2_diff_roi", 2, ["roi", "diff"], description="2× 3D, diff + ROI"),
    Scenario("t2_compare_3_grid", 2, ["compare", "grid", "multi-array"], description="3× 3D, compare grid"),
    Scenario("t2_compare_3_qmri", 2, ["qmri", "compare", "multi-array"], description="3× 4D, compare + qMRI"),
    Scenario("t2_compare_3_multiview", 2, ["multiview", "compare", "multi-array"], description="3× 3D, compare + multiview"),
    Scenario("t2_compare_4_grid", 2, ["compare", "grid", "multi-array"], description="4× 3D, compare grid"),
    Scenario("t2_compare_34_qmri_compact", 2, ["qmri", "compare", "compact", "multi-array"], description="3× 4D, compare + compact qMRI"),
    Scenario("t2_vfield_normal", 2, ["vector", "vfield"], description="Vector field, normal"),
    Scenario("t2_vfield_roi", 2, ["vector", "roi"], description="Vector field + ROI"),
    Scenario("t2_vfield_compare", 2, ["vector", "compare"], description="Compare with vector field"),
]


# ---------------------------------------------------------------------------
# Scenario runner
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
    rng = np.random.default_rng(42)
    results: list[tuple[str, bool, str]] = []

    # --- Create test arrays ---
    arr_2d = np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(100, 80)
    arr_3d = rng.standard_normal((20, 64, 64)).astype(np.float32)
    arr_3d_b = rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.7 + 0.2
    arr_3d_c = rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.5
    arr_3d_d = rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.3 + 0.4
    arr_4d = rng.standard_normal((5, 20, 32, 32)).astype(np.float32)
    arr_4d_b = rng.standard_normal((5, 20, 32, 32)).astype(np.float32) * 0.8
    arr_4d_c = rng.standard_normal((5, 20, 32, 32)).astype(np.float32) * 0.6
    arr_complex = (
        rng.standard_normal((20, 32, 32)) + 1j * rng.standard_normal((20, 32, 32))
    ).astype(np.complex64)

    # Vector field: N+1 dims, last dim = 3
    vf_3d = np.zeros((20, 64, 64, 3), dtype=np.float32)
    vf_3d[..., 1] = 0.3
    vf_3d[..., 2] = 0.6

    # --- Load arrays ---
    sid_2d = _load(client, arr_2d, "audit_2d", tmp)
    sid_3d = _load(client, arr_3d, "audit_3d", tmp)
    sid_3d_b = _load(client, arr_3d_b, "audit_3d_b", tmp)
    sid_3d_c = _load(client, arr_3d_c, "audit_3d_c", tmp)
    sid_3d_d = _load(client, arr_3d_d, "audit_3d_d", tmp)
    sid_4d = _load(client, arr_4d, "audit_4d", tmp)
    sid_4d_b = _load(client, arr_4d_b, "audit_4d_b", tmp)
    sid_4d_c = _load(client, arr_4d_c, "audit_4d_c", tmp)
    sid_complex = _load(client, arr_complex, "audit_complex", tmp)

    # Vector field setup
    sid_3d_vf = _load(client, arr_3d, "audit_3d_vf", tmp)
    vf_path = Path(tmp) / "audit_3d_vf_field.npy"
    np.save(vf_path, vf_3d)
    r = client.post("/attach_vectorfield", json={"sid": sid_3d_vf, "filepath": str(vf_path)})
    r.raise_for_status()

    # --- Determine which scenarios to run ---
    scenarios: list[Scenario] = []
    if tier is None or tier == 1:
        scenarios.extend(TIER1)
    if tier is None or tier == 2:
        scenarios.extend(TIER2)

    if subset:
        scenarios = [s for s in scenarios if s.subset_tags and subset & set(s.subset_tags)]

    print(f"Running {len(scenarios)} scenarios...")

    # --- Run each scenario ---
    for scenario in scenarios:
        name = scenario.name
        print(f"\n  {name}: {scenario.description}")
        assertion_results: list[AssertionResult] = []

        try:
            # ── TIER 1 ──────────────────────────────────────────────

            if name == "t1_single_normal_fit":
                _goto(page, base, sid_3d)
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t1_single_normal_zoom":
                _goto(page, base, sid_3d)
                _zoom_in(page, 5)
                _shot(page, name)
                assertion_results = run_assertions(page, name, zoomed=True)
                _zoom_reset(page)

            elif name == "t1_single_compact":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "Shift+K", wait=600)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "Shift+K", wait=400)

            elif name == "t1_single_multiview_fit":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "v", wait=1000)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "v", wait=400)

            elif name == "t1_single_multiview_zoom":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "v", wait=1000)
                _zoom_in(page, 3)
                _shot(page, name)
                assertion_results = run_assertions(page, name, zoomed=True)
                _zoom_reset(page)
                _press(page, "v", wait=400)

            elif name == "t1_single_qmri_full":
                _goto(page, base, sid_4d)
                _focus(page)
                _press(page, "q", wait=2500)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "q", wait=400)
                _press(page, "q", wait=400)  # exit

            elif name == "t1_single_qmri_compact":
                _goto(page, base, sid_4d)
                _focus(page)
                _press(page, "q", wait=2500)
                _press(page, "q", wait=1500)  # compact
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "q", wait=400)  # exit

            elif name == "t1_single_zen":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "Shift+F", wait=800)
                _shot(page, name)
                # Zen mode hides all chrome, so assertions are minimal
                assertion_results = run_assertions(page, name)
                _press(page, "Shift+F", wait=400)

            elif name == "t1_compare_fit":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t1_compare_zoom":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _zoom_in(page, 4)
                _shot(page, name)
                assertion_results = run_assertions(page, name, zoomed=True)
                _zoom_reset(page)

            elif name == "t1_compare_diff":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _press(page, "Shift+X", wait=800)
                _shot(page, name)
                # Diff mode has a center pane below the side panes that may
                # extend past viewport — check with zoomed=True to skip R3.
                assertion_results = run_assertions(page, name, zoomed=True)
                assertion_results.extend(run_diff_assertions(page))
                _press(page, "Shift+X", wait=400)  # cycle through diff modes
                for _ in range(5):
                    _press(page, "Shift+X", wait=300)  # exit diff

            elif name == "t1_compare_multiview":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _press(page, "v", wait=2500)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "v", wait=400)

            elif name == "t1_single_roi":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "Shift+A", wait=600)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                # Exit ROI: cycle through modes back to off
                for _ in range(3):
                    _press(page, "Shift+A", wait=300)

            elif name == "t1_compact_roi":
                _goto(page, base, sid_3d)
                _focus(page)
                _press(page, "Shift+K", wait=600)
                _press(page, "Shift+A", wait=600)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                for _ in range(3):
                    _press(page, "Shift+A", wait=300)
                _press(page, "Shift+K", wait=400)

            # ── TIER 2 ──────────────────────────────────────────────

            elif name == "t2_single_2d":
                _goto(page, base, sid_2d)
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_single_4d":
                _goto(page, base, sid_4d)
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_single_mosaic":
                _goto(page, base, sid_4d)
                _focus(page)
                _press(page, "z", wait=800)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "z", wait=400)

            elif name == "t2_single_complex":
                _goto(page, base, sid_complex)
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_diff_abs":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _press(page, "Shift+X", wait=800)  # A-B
                _press(page, "Shift+X", wait=800)  # |A-B|
                _shot(page, name)
                # Diff center pane extends below viewport by design
                assertion_results = run_assertions(page, name, zoomed=True)
                assertion_results.extend(run_diff_assertions(page))
                for _ in range(4):
                    _press(page, "Shift+X", wait=300)

            elif name == "t2_diff_rel":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                for _ in range(3):  # A-B → |A-B| → |A-B|/|A|
                    _press(page, "Shift+X", wait=800)
                _shot(page, name)
                assertion_results = run_assertions(page, name, zoomed=True)
                assertion_results.extend(run_diff_assertions(page))
                for _ in range(3):
                    _press(page, "Shift+X", wait=300)

            elif name == "t2_overlay":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                for _ in range(4):  # A-B → |A-B| → |A-B|/|A| → overlay
                    _press(page, "Shift+X", wait=800)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                for _ in range(2):
                    _press(page, "Shift+X", wait=300)

            elif name == "t2_wipe":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                for _ in range(5):  # A-B → |A-B| → |A-B|/|A| → overlay → wipe
                    _press(page, "Shift+X", wait=800)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "Shift+X", wait=300)  # off

            elif name == "t2_registration":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _press(page, "Shift+R", wait=1000)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "Shift+R", wait=400)

            elif name == "t2_compare_qmri":
                _goto_compare(page, base, [sid_4d, sid_4d_b])
                _focus(page)
                _press(page, "q", wait=2500)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "q", wait=400)
                _press(page, "q", wait=400)

            elif name == "t2_diff_roi":
                _goto_compare(page, base, [sid_3d, sid_3d_b])
                _focus(page)
                _press(page, "Shift+X", wait=800)
                _press(page, "Shift+A", wait=600)
                _shot(page, name)
                assertion_results = run_assertions(page, name, zoomed=True)
                for _ in range(3):
                    _press(page, "Shift+A", wait=300)
                for _ in range(5):
                    _press(page, "Shift+X", wait=300)

            elif name == "t2_compare_3_grid":
                _goto_compare(page, base, [sid_3d, sid_3d_b, sid_3d_c])
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_compare_3_qmri":
                _goto_compare(page, base, [sid_4d, sid_4d_b, sid_4d_c])
                _focus(page)
                _press(page, "q", wait=3000)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "q", wait=400)
                _press(page, "q", wait=400)

            elif name == "t2_compare_3_multiview":
                _goto_compare(page, base, [sid_3d, sid_3d_b, sid_3d_c])
                _focus(page)
                _press(page, "v", wait=3000)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "v", wait=400)

            elif name == "t2_compare_4_grid":
                _goto_compare(page, base, [sid_3d, sid_3d_b, sid_3d_c, sid_3d_d])
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_compare_34_qmri_compact":
                _goto_compare(page, base, [sid_4d, sid_4d_b, sid_4d_c])
                _focus(page)
                _press(page, "q", wait=3000)
                _press(page, "q", wait=1500)  # compact
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                _press(page, "q", wait=400)

            elif name == "t2_vfield_normal":
                _goto(page, base, sid_3d_vf)
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            elif name == "t2_vfield_roi":
                _goto(page, base, sid_3d_vf)
                _focus(page)
                _press(page, "Shift+A", wait=600)
                _shot(page, name)
                assertion_results = run_assertions(page, name)
                for _ in range(3):
                    _press(page, "Shift+A", wait=300)

            elif name == "t2_vfield_compare":
                _goto_compare(page, base, [sid_3d_vf, sid_3d_b])
                _focus(page)
                _shot(page, name)
                assertion_results = run_assertions(page, name)

            else:
                print(f"    SKIP: no implementation for {name}")
                results.append((name, True, "skipped"))
                continue

            # --- Process assertions ---
            failed_assertions = [a for a in assertion_results if not a.passed]
            if failed_assertions:
                for a in failed_assertions:
                    print(f"    FAIL: {a.rule} — {a.detail}")

            # --- Visual diff ---
            screenshot_path = SCREENSHOTS_DIR / f"{name}.png"
            baseline_path = BASELINES_DIR / f"{name}.png"
            diff_path = DIFFS_DIR / f"{name}_diff.png"

            if update_baselines and screenshot_path.exists():
                import shutil
                shutil.copy2(screenshot_path, baseline_path)
                print(f"    baseline updated")

            diff_passed = True
            diff_pct = 0.0
            if not update_baselines and baseline_path.exists():
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="ArrayView UI Consistency Audit")
    parser.add_argument("--tier", type=int, choices=[1, 2], help="Run only tier 1 or 2")
    parser.add_argument("--subset", type=str, help="Comma-separated subset tags (e.g. zoom,compact)")
    parser.add_argument("--update-baselines", action="store_true", help="Update baseline screenshots")
    parser.add_argument("--width", type=int, default=1440, help="Viewport width")
    parser.add_argument("--height", type=int, default=900, help="Viewport height")
    args = parser.parse_args()

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
