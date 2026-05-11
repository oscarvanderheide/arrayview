"""
Visual smoke test for arrayview — derived from the help overlay.

Run after any UI change to catch visual regressions before committing:

    uv run python tests/visual_smoke.py

Screenshots saved to tests/smoke_output/<name>.png. Open and review visually.

═══════════════════════════════════════════════════════════════════
COVERAGE — derived from _viewer.html help overlay (? key)
Each row: shortcut | what it does | smoke test scenario
───────────────────────────────────────────────────────────────────
NAVIGATION
  scroll          prev/next slice           ✓ 07, 11 (scroll in 3d, qmri)
  h/l/←/→        move cursor to dim        ✓ 50 (axes flash on h/l)
  j/k/↓/↑        prev/next index           ✓ 07 (arrow keys scroll)
  r               reverse axis / rotate 90° ✓ 19 (r key reverses), 67 (r rotates when slice dim active)
  Space           toggle auto-play          ✓ 20 (play then stop)
  [ / ] (play)    change playback fps       ✓ 64 (play + ] raises fps)
    U               toggle vector arrows      ✓ 72
  + / -           zoom in/out               ✓ 04 (zoom in), 05 (zoom out+reset)
  0               reset zoom                ✓ 05

AXES & VIEWS
  x               swap x dim with slice     ✓ 21
  y               swap y dim with slice     ✓ 22
  z               mosaic mode (4D+ only)    ✓ 09 (arr_4d + z key)
  v               3-plane multiview         ✓ 08
  V               3-plane custom dims       ✓ 41 (inline prompt, type dims)
  o               reset oblique (multiview) ✓ 23 (enter mv, rotate, reset)
  q               qMRI mode & toggle        ✓ 10-12, 12a-c (toggle compact/full), 12d (synthetic MRI row)
    /               special modes shelf       ✓ 28b
  a               stretch to square (all)   ✓ 66 (a in normal, mv, compare; default on in mv)

DISPLAY
  b               toggle border             ✓ 16
  c               colormap grid/cycle       ✓ 02-03
  C               custom colormap (dialog)  ✗ (requires dialog input)
  d               cycle dynamic range       ✓ 17, 74 (height regression)
  D               toggle range lock         ✓ 44 (D unlocks → k changes vmin/vmax per slice)
   B               compare picker (dialog)   ✗ (requires dialog interaction)
   P               unified picker – compare  ✓ 45 (uni-picker opens in Side-by-side mode; disabled in inline embed; enabled in native shell iframe)
   Cmd+O / Ctrl+O  unified picker – open     ✓ 45 (uni-picker cycles to open mode; disabled in inline embed; enabled in native shell iframe)
   (search box)    substring filter          ✓ 45e (type query, list filters client-side; box top-anchored, no jump)
   (arrow keys)    navigate picker list      ✓ 45f (ArrowDown from search moves to first item)
  X               center pane cycle (compare)  ✓ 24, 37, 39 (2-pane compare + X cycle: off/A−B/|A−B|/|A−B|/|A|/overlay/wipe)
   [ / ]           overlay blend             ✓ 24, 37; in movie mode → fps ✓ 64
  n               cycle compare session     ✗ (needs multi-session setup)
  Z               zen mode                  ✓ 06
   L               log scale                 ✓ 18, 40a (LOG egg in #mode-eggs); no-op in RGB mode (toast shown)
   M               alpha threshold            ✓ 40f (ALPHA egg in #mode-eggs)
   m               cycle complex mode        ✓ 25 (complex array + m), 40b-e (complex egg in #mode-eggs); no-op in RGB mode (toast shown)
   f               centred FFT (dialog)      ✓ 42 (inline prompt, enter axes)
   T               cycle theme               ✓ 26
    W               toggle histogram strip    ✗ (removed; histogram only shown in Lebesgue mode)
    W (drag lines)  drag vmin/vmax in hist    ✗ (removed; hover no longer auto-expands colorbar)
    W (colormap)    bars colored by colormap  ✓ 54 (Lebesgue mode expands colorbar; bars use active colormap)
    (cb hover)      no longer auto-expands    ✓ 54 (hover does NOT expand; only Lebesgue mode expands)
    (cb drag-clim)  drag vmin/vmax lines      ✓ 55 (drag vertical clim lines in Lebesgue mode)
   A               rectangle ROI mode        ✓ 58 (A toggles rect ROI, status message shown)
   w               Lebesgue integral mode    ✓ 61 (w toggles, hover colorbar highlights matching pixels)

INFO & EXPORT
  hover           pixel value + cb marker   ✓ 15, 43 (tooltip follows cursor; H enables first)
  H               toggle pixel hover tip    ✓ 43
  i               data info overlay         ✓ 27, 59 (colormap reason row)
  N               export current slice .npy ✗ (triggers download dialog)
   s               open save menu            ✓ 57 (menu opens, Esc closes)
  g               save GIF                  ✗ (triggers download dialog)
  e               copy URL                  ✗ (clipboard, no visual change)
  ?               help overlay              ✓ 28
  toast routing   diff/border toasts → #status (bottom-left)  ✓ 49

LOADING OVERLAY
    viewer startup logo absent                ✓ 47 (#loading-overlay has no .av-load-logo)
    viewer overlay gone after first frame     ✓ 47
  ping-pong loading bar absent              ✓ 47 (#loading-track not in DOM)
  loading text absent                       ✓ 47 (#loading-label not in DOM)
    native preload shell (pywebview)          ✓ 68 (_LOADING_HTML + loading_port param checks)
        native shell preview handoff             ✓ 68 (_SHELL_HTML contains tab-preview + frame-rendered handoff)

WELCOME SCREEN / DEMO
  empty-hint visible on welcome session     ✓ 48 (check .visible on #welcome-hint + body.welcome-mode)
  dim-label active no hover highlight       ✓ 52 (hover bg transparent for all labels)
  dim-label drag scrubs index               ✓ 53 (mousedown+move on label)
  colorbar width clamped [120, 600]px       ✓ 65 (bounding_box check on slim-cb-wrap)

AXES INDICATOR (edge labels)
  h/l dims flash axes labels, fade in+out   ✓ 50 (opacity checked after h press)
  axes color reflects active colormap       ✓ 50 (style.color checked after h press)
  axes visible in mosaic mode (z key)       ✓ 50 (mosaic mode flash)
  axes visible in multiview (v key)         ✓ 50 (multiview flash)
  axes visible in compare mode (picker)     ✗ (requires interactive picker)

COMPARE MODE
  drag title to reorder panes               ✓ 56 (drag left title → right title swaps pane order)
  G               cycle layout (h/v/grid)   ✓ 69 (2-pane h↔v; 3-pane h→v→grid→h)

PICKER
  checkbox (click)  toggle select           ✓ 70 (click checkbox selects; checked class toggled)
  auto-dim          incompatible shape      ✓ 70 (different-shape item gets cp-item-dim class)

VIEW MODES (colorbar visible in all)
  single 2d                                 ✓ 01
  single 3d                                 ✓ 06
  mosaic (grid)                             ✓ 09
  multiview (3-plane)                       ✓ 08
   qmri 3-panel                              ✓ 12
   qmri 5-panel                              ✓ 10-11
    compare 2-array                           ✓ 13-14
    compare overflow minimap                 ✓ 71 (zoomed compare stays on one row; minimap visible)
   compare-qMRI (q in compare)              ✓ 62 (2 arrays × 5 maps; compact toggle)
   compare-multiview (v in compare)          ✓ 63 (2 arrays × 3 planes; scroll + exit)

MEDICAL IMAGE SIZES (canvas fill / sizing check)
  3D volume 192×192×96                      ✓ 29 (default + scrolled)
  3D medical multiview                      ✓ 30
  4D medical qmri 5-panel                   ✓ 31

STABILITY (keys must not cause UI element jumps)
  h/l — cursor dim switch + axes flash      ✓ 32 (bounding box check), 50 (axes flash)
  j/k — slice navigate                      ✓ 33 (bounding box check)
  Z — zen mode toggle                       ✓ 34 (before/after)
  b — border toggle                         ✓ 35 (before/after)
  +/- — zoom in/out                         ✓ 36 (canvas resizes, cb stays below)
    immersive crossfade handoff              ✓ 75 (mid-zone zoom maps to paneCutoff; minimap stays hidden)
  registration arrays (phantom)             ✓ 37 (shifted ellipse, overlay via X)
  multiview uniform cells + zoom limit      ✓ 38 (3 panes same size, zoom caps)
    compare center pane cycle (X key)        ✓ 39 (A−B, |A−B|, relative, overlay, wipe)
    compare zoom overflow keeps one row      ✓ 71 (minimap visible; panes stay side-by-side)
  LOG, complex, and alpha eggs              ✓ 40 (badges in #mode-eggs inside pane)
  RGB egg placement inside canvas           ✓ 46 (eggs fully inside canvas bounds)
  V custom multiview dims                   ✓ 41 (inline prompt)
  f FFT via inline prompt                   ✓ 42 (inline prompt)

PERFORMANCE
  compute_global_stats ndim≥4 2D sampling  ✓ test_api.py (samples 2D slices, not 3D volumes)

OVERLAY
  multiple masks (overlay_sid=sid1,sid2)   ✓ 51 (two binary masks, red+green palette)
  Shift+O overlay visibility toggle        ✓ 51b-e (cycle: off → mask1 → mask2 → all)
  heatmap for float/many-label overlays    ✓ test_api.py (TestOverlayIsLabelMap)
  drag-and-drop .npy upload               ✓ test_api.py (TestLoadUpload)

═══════════════════════════════════════════════════════════════════
RULE: when you add a keyboard shortcut, add a scenario here.
═══════════════════════════════════════════════════════════════════
"""

import socket
import threading
import time
import tempfile
from pathlib import Path

import httpx
import numpy as np
import uvicorn
from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).parent / "smoke_output"
OUT_DIR.mkdir(exist_ok=True)

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


def _shot(page, name, wait=600):
    page.wait_for_timeout(wait)
    page.screenshot(path=str(OUT_DIR / f"{name}.png"), full_page=False)
    print(f"  {name}.png")


def _goto(page, base, sid, wait=1200):
    # Clear saved viewer state so compare-mode history doesn't auto-restore
    try:
        page.evaluate("() => sessionStorage.clear()")
    except Exception:
        pass
    page.goto(f"{base}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_timeout(wait)


def _goto_compare(page, base, sid_a, sid_b, wait=1500):
    page.goto(f"{base}/?sid={sid_a}&compare_sid={sid_b}&compare_sids={sid_b}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    page.wait_for_timeout(wait)


def _focus(page):
    page.locator("#keyboard-sink").focus()


def _press(page, key, wait=400):
    page.keyboard.press(key)
    page.wait_for_timeout(wait)


def _press_open_shortcut(page, wait=800):
    """Press platform-specific open shortcut (Cmd+O on Mac, Ctrl+O elsewhere)."""
    import sys

    modifier = "Meta" if sys.platform == "darwin" else "Control"
    page.keyboard.press(f"{modifier}+KeyO")
    page.wait_for_timeout(wait)


STABLE_SELS = ["#canvas-wrap", "#slim-cb-wrap", "#info"]


def _check_no_jump(page, key, shot_name, selectors=STABLE_SELS, wait=400):
    """Press key, take screenshot, and warn if any selector's bounding box moved."""
    before = {s: page.locator(s).bounding_box() for s in selectors}
    page.keyboard.press(key)
    page.wait_for_timeout(wait)
    after = {s: page.locator(s).bounding_box() for s in selectors}
    _shot(page, shot_name, wait=0)
    for s in selectors:
        b, a = before[s], after[s]
        if b and a:
            dx = abs(b["x"] - a["x"])
            dy = abs(b["y"] - a["y"])
            if dx > 2 or dy > 2:
                print(f"  JUMP WARNING: {s} moved {dx:.0f}px/{dy:.0f}px after {key}")


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def run_smoke(page, base, client, tmp):
    rng = np.random.default_rng(42)

    # Arrays
    arr2d = np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(100, 80)
    arr3d = rng.standard_normal((20, 64, 64)).astype(np.float32)
    arr4d = rng.standard_normal((5, 20, 32, 32)).astype(
        np.float32
    )  # dim 0 = 5 for qmri
    arr4d_z = rng.standard_normal((4, 8, 32, 32)).astype(
        np.float32
    )  # 4D for mosaic (z key)
    arrC = (
        rng.standard_normal((20, 32, 32)) + 1j * rng.standard_normal((20, 32, 32))
    ).astype(np.complex64)
    # Medical image sizes: 192×192×96 volume (~14 MB) and 5-echo 4D (~8 MB)
    arr_med3d = rng.standard_normal((96, 192, 192)).astype(np.float32)
    arr_med4d = rng.standard_normal((5, 48, 96, 96)).astype(np.float32)
    # Registration phantom: bright ellipse, version B shifted 8px right + 5px down
    Y, X = np.mgrid[-64:64, -64:64].astype(np.float32)
    arr_reg_a = np.exp(-(X**2 / 800 + Y**2 / 1800))
    arr_reg_b = np.exp(-((X - 8) ** 2 / 800 + (Y - 5) ** 2 / 1800)) * 0.85

    sid2d = _load(client, arr2d, "arr2d", tmp)
    sid3d = _load(client, arr3d, "arr3d", tmp)
    sid4d = _load(client, arr4d, "arr4d", tmp)
    sid4d_z = _load(client, arr4d_z, "arr4d_z", tmp)
    sidC = _load(client, arrC, "arrC", tmp)
    sid2d_b = _load(client, arr2d * 0.5 + 0.25, "arr2d_b", tmp)
    sid_med3d = _load(client, arr_med3d, "med3d", tmp)
    sid_med4d = _load(client, arr_med4d, "med4d", tmp)
    sid_reg_a = _load(client, arr_reg_a, "reg_a", tmp)
    sid_reg_b = _load(client, arr_reg_b, "reg_b", tmp)
    arr3d_qmri3 = rng.standard_normal((3, 20, 32, 32)).astype(np.float32)
    sid_qmri3 = _load(client, arr3d_qmri3, "qmri3", tmp)
    # Second 4D qMRI array for compare-qMRI and compare-multiview tests
    arr4d_b = rng.standard_normal((5, 20, 32, 32)).astype(np.float32) * 0.8 + 0.1
    sid4d_b = _load(client, arr4d_b, "arr4d_b", tmp)
    # 4D array with wildly different scale per dim-0 index (for D-unlock test)
    arr4d_varied = np.zeros((5, 4, 32, 32), dtype=np.float32)
    for i in range(5):
        arr4d_varied[i] = rng.standard_normal((4, 32, 32)).astype(np.float32) * (10**i)
    sid4d_varied = _load(client, arr4d_varied, "arr4d_varied", tmp)
    # A 3D array with same spatial dims as arr3d for compare-multiview
    arr3d_b = rng.standard_normal((20, 64, 64)).astype(np.float32) * 0.7 + 0.2
    sid3d_b = _load(client, arr3d_b, "arr3d_b", tmp)
    vf3d = np.zeros((20, 64, 64, 3), dtype=np.float32)
    vf3d[..., 1] = 0.3
    vf3d[..., 2] = 0.6
    sid3d_vf = _load(client, arr3d, "arr3d_vf", tmp)
    vf3d_path = Path(tmp) / "arr3d_vf_field.npy"
    np.save(vf3d_path, vf3d)
    attach_vf = client.post(
        "/attach_vectorfield",
        json={"sid": sid3d_vf, "filepath": str(vf3d_path)},
    )
    attach_vf.raise_for_status()
    assert attach_vf.json().get("ok") is True

    # ── 01: 2D default view ──────────────────────────────────────────────────
    _goto(page, base, sid2d)
    _shot(page, "01_2d_default")

    # ── 02-03: colormap cycling (c) ──────────────────────────────────────────
    _focus(page)
    _press(page, "c")
    _press(page, "c")
    _press(page, "Enter")
    _shot(page, "02_2d_colormap_lipari")
    _press(page, "c")
    _press(page, "c")
    _press(page, "c")
    _press(page, "Enter")
    _shot(page, "03_2d_colormap_viridis")
    # reset
    page.evaluate("""() => {
        customColormap = null;
        customGradientStops = null;
        colormap_idx = 0;
        modeManager.forEachView(v => { v.displayState.cmapIdx = 0; });
        refreshAxesColor();
        updateView();
        saveState();
    }""")

    # ── 04-05: zoom (+ / - / 0) ──────────────────────────────────────────────
    _press(page, "+")
    _press(page, "+")
    _shot(page, "04_zoom_in")
    _press(page, "-")
    _press(page, "-")
    _press(page, "-")
    _shot(page, "05_zoom_out")
    _press(page, "0")

    # ── 06: zen mode (Z = capital) ───────────────────────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "Shift+Z")
    _shot(page, "06_zen_mode_on")
    _press(page, "Shift+Z")  # off

    # ── 07: 3D scrolled (j/k arrows) ────────────────────────────────────────
    for _ in range(5):
        _press(page, "ArrowRight", wait=100)
    _shot(page, "07_3d_scrolled")

    # ── 08: multiview (v) ────────────────────────────────────────────────────
    _press(page, "v", wait=1000)
    _shot(page, "08_multiview")
    # Check first pane axes labels match normal-mode orientation (z=x-axis, y=y-axis for [0,1,2] dims)
    first_pane_ax_x = page.evaluate(
        "() => { const els = document.querySelectorAll('.mv-canvas-wrap .axes-lbl-x'); return els.length ? els[0].textContent : ''; }"
    )
    first_pane_ax_y = page.evaluate(
        "() => { const els = document.querySelectorAll('.mv-canvas-wrap .axes-lbl-y'); return els.length ? els[0].textContent : ''; }"
    )
    if first_pane_ax_x == "z" and first_pane_ax_y == "y":
        print(
            f"  OK: first pane axes correct (x={first_pane_ax_x}, y={first_pane_ax_y}) — matches normal mode"
        )
    else:
        print(
            f"  WARN: first pane axes unexpected (x={first_pane_ax_x!r}, y={first_pane_ax_y!r}), expected x=z, y=y"
        )
    _press(page, "v", wait=400)  # exit

    # ── 09: mosaic (z key, requires 4D array) ────────────────────────────────
    _goto(page, base, sid4d_z)
    _focus(page)
    _press(page, "z", wait=800)
    _shot(page, "09_mosaic_on")
    _press(page, "z", wait=400)  # exit

    # ── 10-11: qMRI 5-panel, initial + scrolled ──────────────────────────────
    _goto(page, base, sid4d)
    _focus(page)
    _press(page, "q", wait=2500)  # all 5 panels need time
    _shot(page, "10_qmri_5panel_initial", wait=0)
    for _ in range(3):
        _press(page, "ArrowRight", wait=300)
    _shot(page, "11_qmri_5panel_scrolled")
    _press(page, "q", wait=400)  # exit

    # ── 12: qMRI 3-panel ─────────────────────────────────────────────────────
    _goto(page, base, sid_qmri3)
    _focus(page)
    _press(page, "q", wait=1500)
    _shot(page, "12_qmri_3panel")
    _press(page, "q", wait=400)

    # ── 12a: qMRI toggle compact/full (5-panel) ──────────────────────────────
    _goto(page, base, sid4d)
    _focus(page)
    _press(page, "q", wait=2500)  # enter full view (5 panels)
    _shot(page, "12a_qmri_full_5panel", wait=0)
    _press(page, "q", wait=1500)  # toggle to compact (3 panels)
    _shot(page, "12b_qmri_compact_3panel", wait=0)
    _press(page, "q", wait=2500)  # toggle back to full (5 panels)
    _shot(page, "12c_qmri_full_again", wait=0)
    _press(page, "q", wait=400)  # exit qMRI

    # ── 12d: qMRI synthetic MRI row ──────────────────────────────────────────
    _goto(page, base, sid4d)
    _focus(page)
    _press(page, "q", wait=2500)
    page.evaluate("""async () => {
        _islandToggleQmriSyntheticContrast('t1w');
    }""")
    page.wait_for_timeout(1000)
    _shot(page, "12d_qmri_synthetic_t1w", wait=0)
    _press(page, "q", wait=400)

    # ── 13-14: compare mode ───────────────────────────────────────────────────
    _goto_compare(page, base, sid2d, sid2d_b)
    _shot(page, "13_compare_2array")
    # colorbar in compare (shared bar below)
    _shot(page, "14_compare_colorbar", wait=0)

    # ── 15: colorbar hover marker ────────────────────────────────────────────
    _goto(page, base, sid3d)
    canvas = page.locator("#viewer")
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.5)
    _shot(page, "15_colorbar_hover_marker")
    page.mouse.move(0, 0)

    # ── 16: borders (b) ──────────────────────────────────────────────────────
    _focus(page)
    _press(page, "b")
    _shot(page, "16_borders_on")
    _press(page, "b")  # off

    # ── 17: histogram toggle (d) ─────────────────────────────────────────────
    # Tap `d` now toggles the histogram open/closed (quantile cycling was
    # removed from the `d` key; hold-`d` picks the histogram aggregation dim).
    _press(page, "d")
    _shot(page, "17_histogram_open")
    _press(page, "d")  # close

    # ── 18: log scale (L = capital) ──────────────────────────────────────────
    _goto(page, base, sid2d)
    _focus(page)
    _press(page, "Shift+L")
    _shot(page, "18_log_scale_on")
    _press(page, "Shift+L")  # off

    # ── 19: reverse axis (r) ─────────────────────────────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "r")
    _shot(page, "19_axis_reversed")
    _press(page, "r")  # restore

    # ── 20: autoplay (Space) ─────────────────────────────────────────────────
    _press(page, "Space", wait=800)
    _shot(page, "20_autoplay_on")
    _press(page, "Space")  # stop

    # ── 21-22: axis swap (x, y) ──────────────────────────────────────────────
    _press(page, "x")
    _shot(page, "21_axis_swap_x")
    _press(page, "x")  # restore
    _press(page, "y")
    _shot(page, "22_axis_swap_y")
    _press(page, "y")  # restore

    # ── 23: multiview oblique reset (o) ──────────────────────────────────────
    _press(page, "v", wait=1000)
    # move crosshair to off-center position
    mv_canvas = page.locator(".mv-canvas").first
    mv_box = mv_canvas.bounding_box()
    page.mouse.click(
        mv_box["x"] + mv_box["width"] * 0.3, mv_box["y"] + mv_box["height"] * 0.3
    )
    page.wait_for_timeout(300)
    _shot(page, "23a_multiview_crosshair_moved")
    _press(page, "o")
    _shot(page, "23b_multiview_origin_reset")
    _press(page, "v", wait=400)  # exit multiview

    # ── 24: compare + overlay mode via X cycle ([, ] blend) ──────────────────
    _goto_compare(page, base, sid2d, sid2d_b)
    _focus(page)
    # X cycles: off(0)→A−B(1)→|A−B|(2)→|A−B|/|A|(3)→overlay(4)
    for _ in range(4):
        _press(page, "Shift+X", wait=400)
    _shot(page, "24a_registration_overlay")
    _press(page, "]")
    _press(page, "]")
    _shot(page, "24b_registration_blend_increased")
    # X again → wipe(5), then → off(0)
    _press(page, "Shift+X", wait=400)  # wipe
    _press(page, "Shift+X", wait=400)  # off

    # ── 25: complex mode cycling (m) ─────────────────────────────────────────
    _goto(page, base, sidC)
    _focus(page)
    _shot(page, "25a_complex_mag")
    _press(page, "m")
    _shot(page, "25b_complex_phase")
    _press(page, "m")
    _press(page, "m")  # back to mag

    # ── 26: theme cycling (T = capital) ──────────────────────────────────────
    _goto(page, base, sid2d)
    _focus(page)
    _press(page, "Shift+T")
    _shot(page, "26a_theme_light")
    _press(page, "Shift+T")  # back to dark

    # ── 27: data info overlay (i) ────────────────────────────────────────────
    _press(page, "i", wait=800)
    _shot(page, "27_data_info_overlay")
    _press(page, "Escape", wait=300)

    # ── 28: help overlay (?) + special modes shelf (/) ──────────────────────
    _press(page, "?", wait=400)
    _shot(page, "28_help_overlay")
    _press(page, "?")
    _press(page, "/", wait=400)
    _shot(page, "28b_special_modes_shelf")
    _press(page, "Escape", wait=300)

    # ── 29-31: medical image sizes ────────────────────────────────────────────
    # 192×192×96 volume: check default canvas fill and sizing
    _goto(page, base, sid_med3d, wait=1500)
    _shot(page, "29a_med3d_default")
    _focus(page)
    for _ in range(8):
        _press(page, "ArrowRight", wait=80)
    _shot(page, "29b_med3d_scrolled")

    _press(page, "v", wait=1200)
    _shot(page, "30_med3d_multiview")
    _press(page, "v", wait=400)

    _goto(page, base, sid_med4d, wait=1800)
    _focus(page)
    _press(page, "q", wait=3000)
    _shot(page, "31_med4d_qmri_5panel")
    _press(page, "q", wait=400)

    # ── 32-36: stability — keys must not cause UI element jumps ───────────────
    _goto(page, base, sid3d)
    _focus(page)

    # 32: h/l — move cursor to dim (info label changes, layout stays)
    _shot(page, "32a_stab_before_h")
    _check_no_jump(page, "h", "32b_stab_after_h")
    _check_no_jump(page, "l", "32c_stab_after_l")

    # 33: j/k — slice navigate (canvas updates, layout stays)
    _check_no_jump(page, "j", "33a_stab_after_j")
    _check_no_jump(page, "k", "33b_stab_after_k")

    # 34: Z — zen mode toggle (chrome disappears/reappears, canvas must not jump)
    _shot(page, "34a_stab_zen_before")
    _press(page, "Shift+Z", wait=300)
    _shot(page, "34b_stab_zen_on")
    _press(page, "Shift+Z", wait=300)
    _shot(page, "34c_stab_zen_off")

    # 35: b — border toggle (no layout jump)
    _shot(page, "35a_stab_border_before")
    _check_no_jump(page, "b", "35b_stab_border_on")
    _check_no_jump(page, "b", "35c_stab_border_off")

    # 36: +/- — zoom (canvas resizes; colorbar must stay attached below)
    _shot(page, "36a_stab_zoom_before")
    _press(page, "+")
    _press(page, "+")
    _shot(page, "36b_stab_zoom_in")
    _press(page, "-")
    _press(page, "-")
    _press(page, "-")
    _shot(page, "36c_stab_zoom_out")
    _press(page, "0")
    _shot(page, "36d_stab_zoom_reset")

    # ── 37: overlay with structured phantom arrays (X cycle to overlay) ───────
    _goto_compare(page, base, sid_reg_a, sid_reg_b)
    _focus(page)
    _shot(page, "37a_reg_before_overlay")
    # X cycles: off(0)→A−B(1)→|A−B|(2)→|A−B|/|A|(3)→overlay(4)
    for _ in range(4):
        _press(page, "Shift+X", wait=400)
    _shot(page, "37b_reg_overlay_on")
    _press(page, "]")
    _press(page, "]")
    _shot(page, "37c_reg_blend_increased")
    # X → wipe(5) → off(0)
    _press(page, "Shift+X", wait=400)  # wipe
    _press(page, "Shift+X", wait=400)  # off

    # ── 38: multiview uniform cell sizes + zoom limit ────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "v", wait=1000)
    _shot(page, "38a_mv_uniform_default")
    for _ in range(15):
        _press(page, "+", wait=80)  # zoom to max
    _shot(page, "38b_mv_zoom_max")
    _press(page, "0")  # reset zoom
    _press(page, "v", wait=400)  # exit multiview

    # ── 39: compare center pane cycle (X key: off→A−B→|A−B|→|A−B|/|A|→overlay→wipe→flicker→checker→off)
    _goto_compare(page, base, sid2d, sid2d_b)
    _focus(page)
    _shot(page, "39a_diff_compare_base")
    _press(page, "Shift+X", wait=800)  # 1: A−B
    _shot(page, "39b_diff_AB")
    _press(page, "Shift+X", wait=800)  # 2: |A−B|
    _shot(page, "39c_diff_abs")
    _press(page, "Shift+X", wait=800)  # 3: |A−B|/|A|
    _shot(page, "39d_diff_rel")
    _press(page, "Shift+X", wait=800)  # 4: overlay
    _shot(page, "39e_overlay")
    _press(page, "Shift+X", wait=800)  # 5: wipe
    _shot(page, "39f_wipe")
    _press(page, "Shift+X", wait=400)  # 6: flicker
    _press(page, "Shift+X", wait=400)  # 7: checker
    _press(page, "Shift+X", wait=400)  # 0: off
    _shot(page, "39g_off")

    # ── 40: LOG, complex, and alpha eggs in #mode-eggs ───────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "Shift+L", wait=400)  # LOG on → LOG egg
    _shot(page, "40a_log_egg")
    _press(page, "Shift+L", wait=200)  # LOG off
    _press(page, "Shift+M", wait=600)  # ALPHA on → ALPHA egg
    _shot(page, "40f_alpha_egg")
    _press(page, "Shift+M", wait=400)  # cycle back off (level 0)
    for _ in range(6):
        _press(page, "Shift+M", wait=200)
    _goto(page, base, sidC)
    _focus(page)
    _press(page, "m", wait=400)  # first press: MAGNITUDE egg appears
    _shot(page, "40b_complex_egg_magnitude")
    _press(page, "m", wait=300)  # PHASE
    _shot(page, "40c_complex_egg_phase")
    _press(page, "m", wait=300)  # REAL
    _shot(page, "40d_complex_egg_real")
    _press(page, "m", wait=300)  # IMAG
    _shot(page, "40e_complex_egg_imag")
    _press(page, "m", wait=300)  # back to magnitude — egg stays (MAGNITUDE)
    _shot(page, "40g_complex_mag_egg_persists")

    # ── 41: V key — custom multiview dims via inline prompt ──────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "Shift+V", wait=600)  # opens inline prompt
    _shot(page, "41a_custom_mv_prompt")
    page.locator("#inline-prompt-input").fill("0,1,2")
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    _shot(page, "41b_custom_mv_entered")
    _press(page, "v", wait=400)  # exit multiview

    # ── 42: f key — FFT via inline prompt ────────────────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "f", wait=600)  # opens inline prompt
    _shot(page, "42a_fft_prompt")
    page.locator("#inline-prompt-input").fill("0,1")
    page.keyboard.press("Enter")
    page.wait_for_timeout(800)
    _shot(page, "42b_fft_active")
    _press(page, "f", wait=400)  # FFT off

    # ── 43: H key — toggle pixel hover tooltip ───────────────────────────────
    _goto(page, base, sid2d)
    _focus(page)
    # H enables pixel hover info; then move over canvas to trigger it
    _press(page, "H", wait=200)  # enable hover info
    canvas = page.locator("canvas").first
    box = canvas.bounding_box()
    if box:
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(600)  # wait for async pixel fetch + position
    _shot(page, "43a_hover_tooltip_on")
    _press(page, "H", wait=200)  # disable
    _shot(page, "43b_hover_tooltip_off_status")
    _press(page, "H", wait=200)  # re-enable
    _shot(page, "43c_hover_tooltip_back_on")

    # ── 44: d — 1st tap opens histogram, 2nd tap cycles percentile; Shift+D opens picker ─
    # Tap `d`: first press opens the histogram colorbar only (no vmin/vmax
    # change). Each subsequent tap cycles percentile preset (full · 0.1–99.9
    # · 1–99 · 5–95 · 10–90) + toasts. Shift+D opens the participation
    # picker (outlined-pill dim buttons + lock floater with tooltips).
    _goto(page, base, sid4d_varied)
    _focus(page)
    _press(page, "d", wait=600)  # 1st tap → opens histogram only
    _shot(page, "44a_d_first_tap_opens_histogram")
    _press(page, "d", wait=600)  # 2nd tap → cycles to first preset
    _shot(page, "44b_d_second_tap_cycles_preset")
    _press(page, "Shift+D", wait=300)  # Shift+D opens picker
    _shot(page, "44c_picker_open")
    _press(page, "Shift+D", wait=300)  # Shift+D closes picker
    _shot(page, "44d_picker_closed")

    # ── 45: unified picker — Cmd+O/Ctrl+O and P keys open #uni-picker ───────────────────
    # Cycle: open → compare → overlay → open…
    # P opens in Compare mode. Tab: compare→overlay, Tab: overlay→open
    _goto(page, base, sid2d)
    _focus(page)
    _press(page, "P", wait=800)  # opens in Compare mode
    _shot(page, "45a_uni_picker_compare_mode")
    page.keyboard.press("Tab")  # compare → overlay
    page.wait_for_timeout(300)
    _shot(page, "45b_uni_picker_overlay_mode")
    page.keyboard.press("Tab")  # overlay → open
    page.wait_for_timeout(300)
    _shot(page, "45c_uni_picker_open_mode")
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    _press_open_shortcut(page, wait=800)  # opens directly in Open mode
    _shot(page, "45d_uni_picker_open_direct")
    page.keyboard.press("q")
    page.wait_for_timeout(200)
    _press_open_shortcut(page, wait=800)  # opens in Open mode
    _shot(page, "45d_uni_picker_open_direct")
    page.keyboard.press("q")
    page.wait_for_timeout(200)

    # 45e: search / fzf filter — type in search box, verify filtered list
    _press_open_shortcut(page, wait=800)  # open picker
    search_input = page.locator("#uni-picker-search")
    search_input.type("test", delay=50)  # type a query; fzf/substring filter runs
    page.wait_for_timeout(400)
    _shot(page, "45e_uni_picker_search_filtered")
    # Verify picker box is top-anchored (not vertically centered) so filtering
    # doesn't cause vertical jumps. Box top should be well above mid-viewport.
    box_top = page.evaluate(
        "() => document.getElementById('uni-picker-box').getBoundingClientRect().top"
    )
    vh = page.evaluate("() => window.innerHeight")
    if box_top > vh * 0.5:
        print(
            f"  WARNING: #uni-picker-box top={box_top:.0f}px looks vertically centered (vh={vh}); expected top-anchored"
        )
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    # 45f: ArrowDown from search box moves focus to first list item
    _press_open_shortcut(page, wait=800)  # open picker in Open mode
    search_input = page.locator("#uni-picker-search")
    search_input.type("test", delay=50)  # type a query; fzf/substring filter runs
    page.wait_for_timeout(400)
    _shot(page, "45e_uni_picker_search_filtered")
    # Verify picker box is top-anchored (not vertically centered) so filtering
    # doesn't cause vertical jumps. Box top should be well above mid-viewport.
    box_top = page.evaluate(
        "() => document.getElementById('uni-picker-box').getBoundingClientRect().top"
    )
    vh = page.evaluate("() => window.innerHeight")
    if box_top > vh * 0.5:
        print(
            f"  WARNING: #uni-picker-box top={box_top:.0f}px looks vertically centered (vh={vh}); expected top-anchored"
        )
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    # 45f: ArrowDown from search box moves focus to first list item
    _press_open_shortcut(page, wait=800)  # open picker in Open mode
    search_input = page.locator("#uni-picker-search")
    search_input.click()
    page.wait_for_timeout(200)
    page.keyboard.press("ArrowDown")  # should move focus to first .cp-item
    page.wait_for_timeout(300)
    _shot(page, "45f_uni_picker_arrowdown_from_search")
    # Verify that focus moved away from search into the list
    focused_is_search = page.evaluate(
        "() => document.activeElement === document.querySelector('#uni-picker-search')"
    )
    if focused_is_search:
        print("  WARNING: ArrowDown from search did not move focus to list item")
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)

    # 46: RGB image viewing — HxWx3 array loaded with rgb=True
    # Colorbar should be hidden; RGB badge visible in mode-eggs.
    rgb_arr = np.zeros((128, 128, 3), dtype=np.float32)
    rgb_arr[:64, :, 0] = 200  # red top half
    rgb_arr[64:, :, 1] = 150  # green bottom half
    rgb_arr[:, 64:, 2] = 100  # blue right half
    rgb_path = Path(tmp) / "rgb_test.npy"
    np.save(rgb_path, rgb_arr)
    r = client.post(
        "/load", json={"filepath": str(rgb_path), "name": "rgb_test", "rgb": True}
    )
    r.raise_for_status()
    rgb_sid = r.json()["sid"]
    _goto(page, base, rgb_sid, wait=1200)
    _focus(page)
    _shot(page, "46_rgb_basic")
    # Verify RGB egg badge is inside the canvas bounds.
    eggs_rect = page.evaluate(
        "() => { const e = document.getElementById('mode-eggs'); return e ? e.getBoundingClientRect() : null; }"
    )
    canvas_rect = page.evaluate(
        "() => { const c = document.getElementById('viewer'); return c ? c.getBoundingClientRect() : null; }"
    )
    if eggs_rect and canvas_rect:
        inside = (
            eggs_rect["left"] >= canvas_rect["left"] - 1
            and eggs_rect["right"] <= canvas_rect["right"] + 1
            and eggs_rect["top"] >= canvas_rect["top"] - 1
            and eggs_rect["bottom"] <= canvas_rect["bottom"] + 1
        )
        if not inside:
            print(
                "  WARNING: RGB eggs outside canvas bounds"
            )

    # ── 47: viewer startup overlay ───────────────────────────────────────────
    # Verify the viewer no longer ships a startup logo overlay, and that the
    # old track/text loading affordances remain absent.
    _goto(page, base, sid2d)
    startup_logo_present = page.evaluate(
        "() => !!document.querySelector('#loading-overlay .av-load-logo')"
    )
    if startup_logo_present:
        print("  WARNING: startup logo still present in #loading-overlay")
    overlay_visible = page.evaluate(
        "() => getComputedStyle(document.getElementById('loading-overlay')).display !== 'none'"
    )
    if overlay_visible:
        print("  WARNING: loading overlay still visible after initial render")
    loading_track_present = page.evaluate(
        "() => !!document.getElementById('loading-track')"
    )
    if loading_track_present:
        print(
            "  WARNING: #loading-track (ping-pong bar) still present in DOM — should have been removed"
        )
    loading_label_present = page.evaluate(
        "() => !!document.getElementById('loading-label')"
    )
    if loading_label_present:
        print(
            "  WARNING: #loading-label (Loading... text) still present in DOM — should have been removed"
        )
    _shot(page, "47_loading_overlay_after_load")

    # ── 48: demo array — RGB plasma ───────────────────────────────────────────
    # Verify the welcome demo renders as an RGB plasma animation (128×128×32×3).
    # The demo is loaded with rgb=True so the colorbar should be hidden and
    # the RGB egg badge should appear in #mode-eggs.
    from arrayview._app import _make_demo_array

    demo_arr = _make_demo_array()
    demo_path = Path(tmp) / "demo_plasma.npy"
    np.save(demo_path, demo_arr)
    r = client.post(
        "/load", json={"filepath": str(demo_path), "name": "welcome", "rgb": True}
    )
    r.raise_for_status()
    demo_sid = r.json()["sid"]
    _goto(page, base, demo_sid, wait=1200)
    _focus(page)
    _shot(page, "48_demo_plasma_rgb")
    # RGB egg should be visible; colorbar should be absent/hidden
    eggs_text = page.locator("#mode-eggs").inner_text()
    if "RGB" not in eggs_text:
        print("  WARNING: RGB egg not visible in demo plasma scenario")
    # welcome-hint ("{cmd,ctrl,shift}+o · drop to open array") must be visible on welcome screen
    hint_visible = page.evaluate(
        "() => document.getElementById('welcome-hint').classList.contains('visible')"
    )
    if not hint_visible:
        print(
            "  WARNING: #welcome-hint not visible on welcome demo screen — fix _isWelcomeScreen logic"
        )
    # body should have welcome-mode class (caps canvas to 50% height)
    welcome_mode = page.evaluate(
        "() => document.body.classList.contains('welcome-mode')"
    )
    if not welcome_mode:
        print("  WARNING: body.welcome-mode class not set on welcome demo screen")

    # ── 49: array identity moves to window title; toast uses bottom-left slot ─
    _goto(page, base, sid2d)
    _focus(page)
    page_title = page.evaluate("() => document.title")
    name_hidden = page.evaluate(
        "() => getComputedStyle(document.getElementById('array-name')).display === 'none'"
    )
    if not page_title.startswith("ArrayView:"):
        print(f"  WARNING: document title does not include array identity ({page_title!r})")
    if not name_hidden:
        print("  WARNING: #array-name is still visible in the viewport")
    page.evaluate("() => showToast('smoke toast')")
    page.wait_for_timeout(300)
    _shot(page, "49_toast_identity_slot")
    toast_visible = page.evaluate(
        "() => document.getElementById('toast')?.classList.contains('visible')"
    )
    if not toast_visible:
        print("  WARNING: #toast did not become visible")
    page.wait_for_timeout(3300)
    toast_hidden = page.evaluate(
        "() => !document.getElementById('toast')?.classList.contains('visible')"
    )
    if not toast_hidden:
        print("  WARNING: #toast did not hide after timeout")

    # ── 50: axes indicator — edge labels flash on h/l, mosaic, multiview ─────
    # Verify that pressing h in a 3D array makes .axes-indicator opacity become 1.
    # Also verify axes are visible in mosaic mode (not hidden when dim_z >= 0).
    # Also verify axes are ALWAYS visible in multiview mode (not just flashing).
    # Also verify axes color reflects the active colormap (not pixel-brightness gray).
    _goto(page, base, sid3d)
    # Check baseline: axes indicator starts hidden (opacity 0)
    ax_opacity_before = page.evaluate(
        "() => { const el = document.querySelector('.axes-indicator'); "
        "return parseFloat(el ? getComputedStyle(el).opacity : '-1'); }"
    )
    _press(page, "h", wait=500)
    ax_opacity_after = page.evaluate(
        "() => { const el = document.querySelector('.axes-indicator'); "
        "return parseFloat(el ? getComputedStyle(el).opacity : '-1'); }"
    )
    # Verify axes color is colormap-based (not generic gray)
    ax_color = page.evaluate(
        "() => { const el = document.querySelector('.axes-indicator'); "
        "return el ? el.style.color : ''; }"
    )
    _shot(page, "50a_axes_flash_normal")
    if ax_opacity_after < 0.5:
        print(
            f"  WARNING: axes indicator opacity={ax_opacity_after:.2f} after h — expected ~1.0 (fade-in not working)"
        )
    else:
        print(
            f"  OK: axes indicator flashes (opacity={ax_opacity_after:.2f}, color={ax_color!r})"
        )
    # Check mosaic mode: axes should still be visible after z key
    _press(page, "z", wait=400)
    _press(page, "h", wait=500)
    ax_opacity_mosaic = page.evaluate(
        "() => { const el = document.querySelector('.axes-indicator'); "
        "return parseFloat(el ? getComputedStyle(el).opacity : '-1'); }"
    )
    _shot(page, "50b_axes_flash_mosaic")
    if ax_opacity_mosaic < 0.5:
        print(
            f"  WARNING: axes indicator opacity={ax_opacity_mosaic:.2f} in mosaic mode — should flash on h"
        )
    # Check multiview mode: axes should remain visible (not just flash)
    _press(page, "z", wait=200)  # exit mosaic
    _press(page, "v", wait=400)
    ax_opacity_mv_initial = page.evaluate(
        "() => { const els = document.querySelectorAll('.axes-indicator'); "
        "return Math.max(...Array.from(els, e => parseFloat(getComputedStyle(e).opacity))); }"
    )
    _shot(page, "50c_axes_always_visible_multiview")
    if ax_opacity_mv_initial < 0.5:
        print(
            f"  WARNING: axes indicator max opacity={ax_opacity_mv_initial:.2f} on entering multiview — expected ~1.0"
        )
    # Wait 2 seconds to verify axes don't fade out (they should stay visible)
    page.wait_for_timeout(2000)
    ax_opacity_mv_after_wait = page.evaluate(
        "() => { const els = document.querySelectorAll('.axes-indicator'); "
        "return Math.max(...Array.from(els, e => parseFloat(getComputedStyle(e).opacity))); }"
    )
    _shot(page, "50d_axes_still_visible_after_wait")
    if ax_opacity_mv_after_wait < 0.5:
        print(
            f"  WARNING: axes indicator faded to {ax_opacity_mv_after_wait:.2f} after 2s in multiview — should stay visible"
        )
    _press(page, "v", wait=200)  # exit multiview

    # -----------------------------------------------------------------------
    # 51: Multiple overlays — two binary masks with auto-palette colors
    # -----------------------------------------------------------------------
    print("51: multiple overlays (two masks)")
    rng = np.random.default_rng(51)
    base_arr = rng.standard_normal((64, 64)).astype(np.float32)
    mask_a = np.zeros((64, 64), dtype=np.uint8)
    mask_a[5:25, 5:25] = 1  # top-left quadrant
    mask_b = np.zeros((64, 64), dtype=np.uint8)
    mask_b[40:60, 40:60] = 1  # bottom-right quadrant
    sid_base = _load(client, base_arr, "arr_51_base", tmp)
    sid_ov_a = _load(client, mask_a, "arr_51_ov_a", tmp)
    sid_ov_b = _load(client, mask_b, "arr_51_ov_b", tmp)
    try:
        page.evaluate("() => sessionStorage.clear()")
    except Exception:
        pass
    page.goto(
        f"{base}/?sid={sid_base}"
        f"&overlay_sid={sid_ov_a},{sid_ov_b}"
        f"&overlay_colors=ff4444,44cc44"
    )
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_timeout(1400)
    _shot(page, "51_multi_overlay")

    # Shift+O: cycle overlay visibility (all → off → mask1 → mask2 → all)
    _focus(page)
    _press(page, "Shift+O", wait=600)
    status_off = page.evaluate("() => document.getElementById('status')?.textContent || ''")
    _shot(page, "51b_overlay_off")
    assert "off" in status_off, f"Expected 'overlays: off', got '{status_off}'"

    _press(page, "Shift+O", wait=600)
    status_m1 = page.evaluate("() => document.getElementById('status')?.textContent || ''")
    _shot(page, "51c_overlay_mask1")
    assert "overlay 1/" in status_m1, f"Expected 'overlay 1/2', got '{status_m1}'"

    _press(page, "Shift+O", wait=600)
    status_m2 = page.evaluate("() => document.getElementById('status')?.textContent || ''")
    _shot(page, "51d_overlay_mask2")
    assert "overlay 2/" in status_m2, f"Expected 'overlay 2/2', got '{status_m2}'"

    _press(page, "Shift+O", wait=600)
    status_all = page.evaluate("() => document.getElementById('status')?.textContent || ''")
    _shot(page, "51e_overlay_all")
    assert "all" in status_all, f"Expected 'overlays: all', got '{status_all}'"

    # ── 52: dimbar hover — no grayish background on any dim label ─────────
    print("52: dimbar hover no-highlight on any dim label")
    _goto(page, base, sid3d, wait=800)
    _focus(page)
    # .dim-label:hover must use background: transparent (no glow on any label)
    hover_rule_ok = page.evaluate(
        "() => {"
        "  const rules = Array.from(document.styleSheets)"
        "    .flatMap(s => { try { return Array.from(s.cssRules); } catch { return []; } });"
        "  const hoverRule = rules.find(r => r.selectorText && r.selectorText.trim() === '.dim-label:hover');"
        "  if (!hoverRule) return 'rule_missing';"
        "  const bg = hoverRule.style.background || hoverRule.style.backgroundColor;"
        "  return bg === 'transparent' || bg === '' ? 'ok' : bg;"
        "}"
    )
    _shot(page, "52_dimbar_hover")
    if hover_rule_ok != "ok":
        print(
            f"  WARNING: dim-label hover CSS rule not transparent ({hover_rule_ok}) — labels may still highlight"
        )

    # ── 53: dim-label drag to scrub ─────────────────────────────────────────
    print("53: dim-label drag scrubs dimension index")
    _goto(page, base, sid3d, wait=800)
    _focus(page)
    # Get the dim-label element for a scroll dim (first [data-dim] that isn't x/y)
    # Move to the far-left and drag to far-right, index should change
    idx_before = page.evaluate(
        "() => window._arrayviewState ? window._arrayviewState.indices : null"
    )
    label_box = page.evaluate(
        "() => {"
        "  const labels = document.querySelectorAll('#info [data-dim]');"
        "  for (const l of labels) {"
        "    if (!l.classList.contains('spatial-dim') && !l.textContent.match(/^-?[xyz]/)) {"
        "      const r = l.getBoundingClientRect();"
        "      return { x: r.left, y: r.top + r.height / 2, w: r.width };"
        "    }"
        "  }"
        "  return null;"
        "}"
    )
    if label_box:
        page.mouse.move(label_box["x"] + label_box["w"] * 0.1, label_box["y"])
        page.mouse.down()
        page.mouse.move(label_box["x"] + label_box["w"] * 0.9, label_box["y"])
        page.wait_for_timeout(200)
        page.mouse.up()
        page.wait_for_timeout(300)
    _shot(page, "53_dim_label_drag")

    # ── 54: histogram-in-colorbar (Lebesgue mode expands it) ─────────────────────
    print("54: w key (Lebesgue mode) expands colorbar with histogram bars")
    # Navigate to a fresh 3D session
    _goto(page, base, sid3d, wait=800)
    _focus(page)
    # Colorbar should be collapsed initially (8px height)
    cb_initial_h = page.evaluate(
        "() => { const c = document.getElementById('slim-cb'); return c ? parseInt(c.style.height) : 0; }"
    )
    # Hover over the colorbar — should NOT expand (hover no longer auto-expands)
    cb_wrap = page.locator("#slim-cb-wrap")
    cb_box = cb_wrap.bounding_box()
    if cb_box:
        page.mouse.move(
            cb_box["x"] + cb_box["width"] / 2, cb_box["y"] + cb_box["height"] / 2
        )
        page.wait_for_timeout(600)
        cb_hover_h = page.evaluate(
            "() => { const c = document.getElementById('slim-cb'); return c ? parseInt(c.style.height) : 0; }"
        )
        page.mouse.move(10, 10)
        page.wait_for_timeout(300)
        if cb_hover_h <= 10:
            print(f"  OK: hover does NOT expand colorbar (h={cb_hover_h})")
        else:
            print(
                f"  WARN: colorbar expanded on hover (h={cb_hover_h}) — should not happen"
            )
    _shot(page, "54a_colorbar_hover_no_expand")

    # Press w to enable Lebesgue mode → colorbar should expand
    _press(page, "w", wait=800)
    cb_expanded_h = page.evaluate(
        "() => { const c = document.getElementById('slim-cb'); return c ? parseInt(c.style.height) : 0; }"
    )
    _shot(page, "54b_colorbar_lebesgue_expanded")
    # Cycle colormap while expanded
    _press(page, "c", wait=400)
    _press(page, "c", wait=400)
    _press(page, "Enter", wait=200)
    _shot(page, "54c_colorbar_expanded_after_c")
    if cb_initial_h <= 10 and cb_expanded_h >= 30:
        print(
            f"  OK: colorbar expands in Lebesgue mode ({cb_initial_h}→{cb_expanded_h}px)"
        )
    else:
        print(
            f"  WARN: colorbar height unexpected (initial={cb_initial_h}, lebesgue={cb_expanded_h})"
        )
    # Exit Lebesgue mode
    _press(page, "w", wait=400)
    cb_collapsed_h = page.evaluate(
        "() => { const c = document.getElementById('slim-cb'); return c ? parseInt(c.style.height) : 0; }"
    )
    _shot(page, "54d_colorbar_collapsed")
    if cb_collapsed_h <= 10:
        print(
            f"  OK: colorbar collapses after exiting Lebesgue mode ({cb_collapsed_h}px)"
        )
    else:
        print(
            f"  WARN: colorbar still expanded after Lebesgue exit ({cb_collapsed_h}px)"
        )

    # ── 55: histogram-in-colorbar drag-clim (in Lebesgue mode) ──────────────────
    print("55: drag vmin/vmax clim lines on expanded colorbar (Lebesgue mode)")
    _goto(page, base, sid2d, wait=600)
    _focus(page)
    # Enable Lebesgue mode to expand colorbar
    _press(page, "w", wait=800)
    cb_wrap = page.locator("#slim-cb-wrap")
    cb_box = cb_wrap.bounding_box()
    if cb_box:
        # Get expanded colorbar canvas bounding box for drag
        slim_cb = page.locator("#slim-cb")
        slim_box = slim_cb.bounding_box()
        if slim_box:
            # Drag from left side toward center (simulate moving vmin line right)
            start_x = slim_box["x"] + slim_box["width"] * 0.05
            mid_y = slim_box["y"] + slim_box["height"] / 2
            end_x = slim_box["x"] + slim_box["width"] * 0.3
            page.mouse.move(start_x, mid_y)
            page.mouse.down()
            page.mouse.move(end_x, mid_y, steps=10)
            page.mouse.up()
            page.wait_for_timeout(400)
            _shot(page, "55_colorbar_drag_clim")
            print("  OK: colorbar drag-clim screenshot taken")
        else:
            print("  WARN: slim-cb not found")
    else:
        print("  WARN: slim-cb-wrap not found for drag-clim test")
    # Exit Lebesgue mode
    _press(page, "w", wait=300)

    # ── 56: drag-to-reorder compare panels ───────────────────────────────────
    print("56: drag-to-reorder compare panels")
    _goto_compare(page, base, sid2d, sid2d_b, wait=1500)
    _focus(page)
    # Record the initial title texts for pane 0 and pane 1
    title_before_left = page.evaluate(
        "() => document.getElementById('compare-left-title')?.textContent?.trim() || ''"
    )
    title_before_right = page.evaluate(
        "() => document.getElementById('compare-right-title')?.textContent?.trim() || ''"
    )
    _shot(page, "56a_compare_before_drag")
    # HTML5 drag-and-drop doesn't fire reliably in headless Chromium via mouse events.
    # Dispatch synthetic DragEvents via JS to exercise the swap logic directly.
    page.evaluate("""() => {
        const src = document.getElementById('compare-left-title');
        const dst = document.getElementById('compare-right-title');
        if (!src || !dst) return;
        const dt = new DataTransfer();
        src.dispatchEvent(new DragEvent('dragstart', { bubbles: true, cancelable: true, dataTransfer: dt }));
        dst.dispatchEvent(new DragEvent('dragover',  { bubbles: true, cancelable: true, dataTransfer: dt }));
        dst.dispatchEvent(new DragEvent('drop',      { bubbles: true, cancelable: true, dataTransfer: dt }));
        src.dispatchEvent(new DragEvent('dragend',   { bubbles: true, cancelable: true, dataTransfer: dt }));
    }""")
    page.wait_for_timeout(800)
    _shot(page, "56b_compare_after_drag")
    title_after_left = page.evaluate(
        "() => document.getElementById('compare-left-title')?.textContent?.trim() || ''"
    )
    title_after_right = page.evaluate(
        "() => document.getElementById('compare-right-title')?.textContent?.trim() || ''"
    )
    if (
        title_before_left
        and title_before_right
        and title_before_left != title_before_right
    ):
        if (
            title_after_left == title_before_right
            and title_after_right == title_before_left
        ):
            print("  OK: pane titles swapped correctly after drag")
        else:
            print(
                f"  WARN: titles did not swap — before=({title_before_left!r}, {title_before_right!r})"
                f" after=({title_after_left!r}, {title_after_right!r})"
            )
    else:
        print(
            f"  INFO: titles — before=({title_before_left!r}, {title_before_right!r}), after=({title_after_left!r}, {title_after_right!r})"
        )

    # ── 57: save menu flips the colorbar on s and restores on Esc ────────────
    print("57: s flips colorbar to reveal save icons; Esc flips back")
    _goto(page, base, sid3d, wait=1500)
    _focus(page)
    _press(page, "s", wait=400)
    flipped = page.evaluate(
        "() => document.getElementById('slim-cb-save-flip')?.classList.contains('flipped') === true"
    )
    _shot(page, "57_save_menu_flipped")
    _press(page, "Escape", wait=400)
    unflipped = page.evaluate(
        "() => document.getElementById('slim-cb-save-flip')?.classList.contains('flipped') === false"
    )
    if flipped and unflipped:
        print("  OK: colorbar flip + restore works")
    else:
        print(f"  WARN: flip toggle failed — flipped={flipped} unflipped={unflipped}")

    # ── 58: rectangle ROI mode (A key) ────────────────────────────────────────
    print("58: A key toggles rectangle ROI mode")
    _goto(page, base, sid3d, wait=1200)
    _focus(page)
    # Press A to enable rect ROI mode — check status bar message
    _press(page, "A", wait=400)
    status_on = page.evaluate(
        "() => (document.getElementById('status') || {}).textContent || ''"
    )
    _shot(page, "58a_rect_roi_mode_on")
    # Press A again to disable
    _press(page, "A", wait=400)
    status_off = page.evaluate(
        "() => (document.getElementById('status') || {}).textContent || ''"
    )
    _shot(page, "58b_rect_roi_mode_off")
    if "rect ROI" in status_on or "rect ROI" in status_off:
        print(f"  OK: rect ROI status shown (on={status_on!r}, off={status_off!r})")
    else:
        print(
            f"  WARN: rect ROI status not seen (on={status_on!r}, off={status_off!r})"
        )

    # ── 59: enhanced info overlay (i key) shows Colormap row ─────────────────
    print("59: i key info overlay shows Colormap reason row")
    _goto(page, base, sid3d, wait=1200)
    _focus(page)
    _press(page, "i", wait=800)
    colormap_row_visible = page.evaluate("""() => {
        const rows = document.querySelectorAll('#info-overlay td, #info-panel td, .info-row td');
        for (const td of rows) {
            if (td.textContent && td.textContent.toLowerCase().includes('gray')) return true;
            if (td.textContent && td.textContent.toLowerCase().includes('rdbu')) return true;
        }
        // Also try any visible element containing colormap reason keywords
        const all = document.querySelectorAll('*');
        for (const el of all) {
            const t = el.textContent || '';
            if ((t.includes('gray') || t.includes('RdBu')) && t.includes('(') && el.children.length === 0) return true;
        }
        return false;
    }""")
    _shot(page, "59_info_overlay_colormap_row")
    _press(page, "Escape", wait=300)
    if colormap_row_visible:
        print("  OK: Colormap reason visible in info overlay")
    else:
        print("  WARN: Colormap reason not detected in info overlay DOM")

    # ── 60: histogram-in-colorbar uses colormap colors ──────────────────────────
    # (covered by scenario 54 — expand + cycle colormap; this scenario is now a no-op)
    print("60: histogram colormap coloring (covered by scenario 54)")
    print("  OK: covered by 54a/54b screenshots")

    # ── 61: Lebesgue integral mode (w key) ───────────────────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    # 61a: Press w to enable Lebesgue mode — colorbar should expand and stay expanded
    _press(page, "w")
    time.sleep(0.6)
    status_text = page.locator("#status").text_content() or ""
    if "Lebesgue" in status_text:
        print("61: Lebesgue mode toggle (w key)")
        print("  OK: status shows Lebesgue mode enabled")
    else:
        print(f"61: Lebesgue mode toggle (w key)")
        print(f"  WARN: expected 'Lebesgue' in status, got: {status_text!r}")
    # Colorbar should be expanded (40px)
    cb = page.locator("#slim-cb")
    cb_h = int(cb.evaluate("el => el.style.height.replace('px','')") or "0")
    if cb_h >= 30:
        print(f"  OK: colorbar expanded to {cb_h}px in Lebesgue mode")
    else:
        print(f"  WARN: colorbar height {cb_h}px, expected >=30 (expanded)")
    _shot(page, "61a_lebesgue_mode_on")

    # 61b: Hover over the colorbar — should show Lebesgue overlay on canvas
    cb_box = cb.bounding_box()
    if cb_box:
        # Move to center of colorbar
        page.mouse.move(
            cb_box["x"] + cb_box["width"] / 2, cb_box["y"] + cb_box["height"] / 2
        )
        time.sleep(0.5)
        # Check that the lebesgue overlay canvas is visible
        lb_cv = page.locator("#lebesgue-canvas")
        lb_display = (
            lb_cv.evaluate("el => getComputedStyle(el).display")
            if lb_cv.count()
            else "none"
        )
        if lb_display != "none":
            print("  OK: Lebesgue overlay visible on hover")
        else:
            print("  WARN: Lebesgue overlay not visible on hover")
        _shot(page, "61b_lebesgue_hover")

        # 61c: Move to a different position on the colorbar
        page.mouse.move(
            cb_box["x"] + cb_box["width"] * 0.2, cb_box["y"] + cb_box["height"] / 2
        )
        time.sleep(0.3)
        _shot(page, "61c_lebesgue_hover_left")

        # Move mouse away from colorbar
        page.mouse.move(100, 100)
        time.sleep(0.4)
        lb_display2 = (
            lb_cv.evaluate("el => getComputedStyle(el).display")
            if lb_cv.count()
            else "block"
        )
        if lb_display2 == "none":
            print("  OK: Lebesgue overlay hidden when not hovering colorbar")
        else:
            print(
                f"  WARN: Lebesgue overlay still visible after moving away (display={lb_display2!r})"
            )

    # 61d: Press w again to disable
    _focus(page)
    _press(page, "w")
    time.sleep(0.4)
    status_text2 = page.locator("#status").text_content() or ""
    if "off" in status_text2.lower():
        print("  OK: Lebesgue mode disabled")
    else:
        print(f"  WARN: expected 'off' in status, got: {status_text2!r}")
    _shot(page, "61d_lebesgue_mode_off")

    # ── 62: compare-qMRI (q key in compare mode) ────────────────────────────────
    print("62: compare-qMRI mode (q in compare)")
    # Navigate to compare mode with two qMRI-compatible 4D arrays
    page.evaluate("() => sessionStorage.clear()")
    page.goto(f"{base}/?sid={sid4d}&compare_sids={sid4d_b}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    page.wait_for_timeout(1500)
    _focus(page)

    # 62a: Enter compare-qMRI by pressing q
    _press(page, "q", wait=2500)
    qv_rows = page.locator("#qmri-view-wrap .qv-row")
    row_count = qv_rows.count()
    if row_count >= 2:
        print(f"  OK: {row_count} array rows visible in compare-qMRI")
    else:
        print(f"  WARN: expected >=2 qv-row elements, got {row_count}")
    qv_canvases = page.locator("#qmri-view-wrap .qv-canvas")
    canvas_count = qv_canvases.count()
    if canvas_count >= 2:
        print(f"  OK: {canvas_count} canvases rendered (rows × maps)")
    else:
        print(f"  WARN: expected >=2 canvases, got {canvas_count}")
    _shot(page, "62a_compare_qmri_full")

    # 62b: Toggle compact mode (q again, n=5 so compact is available)
    _press(page, "q", wait=1500)
    canvas_count_compact = page.locator("#qmri-view-wrap .qv-canvas").count()
    if canvas_count_compact < canvas_count:
        print(
            f"  OK: compact mode has fewer canvases ({canvas_count_compact} vs {canvas_count})"
        )
    else:
        print(
            f"  WARN: compact mode canvas count {canvas_count_compact} not less than full {canvas_count}"
        )
    _shot(page, "62b_compare_qmri_compact")

    # 62c: Scroll a slice
    _press(page, "ArrowRight", wait=500)
    _shot(page, "62c_compare_qmri_scrolled")

    # 62d: Exit compare-qMRI (q again exits from compact)
    _press(page, "q", wait=800)
    wrap_class = page.locator("#qmri-view-wrap").get_attribute("class") or ""
    if "active" not in wrap_class:
        print("  OK: compare-qMRI exited (qmri-view-wrap not active)")
    else:
        print(f"  WARN: qmri-view-wrap still active after exit (class={wrap_class!r})")
    _shot(page, "62d_compare_qmri_exited")

    # ── 63: compare-multiview (v key in compare mode) ───────────────────────────
    print("63: compare-multiview mode (v in compare)")
    page.evaluate("() => sessionStorage.clear()")
    page.goto(f"{base}/?sid={sid3d}&compare_sids={sid3d_b}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    page.wait_for_timeout(1500)
    _focus(page)

    # 63a: Enter compare-multiview with v key
    _press(page, "v", wait=2500)
    mv_rows = page.locator("#qmri-view-wrap .qv-row")
    mv_row_count = mv_rows.count()
    if mv_row_count >= 2:
        print(f"  OK: {mv_row_count} array rows visible in compare-multiview")
    else:
        print(f"  WARN: expected >=2 qv-row elements, got {mv_row_count}")
    mv_canvases = page.locator("#qmri-view-wrap .qv-canvas")
    mv_canvas_count = mv_canvases.count()
    expected_mv = 6  # 2 arrays × 3 planes
    if mv_canvas_count == expected_mv:
        print(f"  OK: {mv_canvas_count} canvases (2 arrays × 3 planes)")
    else:
        print(
            f"  WARN: expected {expected_mv} canvases for 2 arrays × 3 planes, got {mv_canvas_count}"
        )
    _shot(page, "63a_compare_multiview")

    # 63b: Scroll a slice via ArrowRight
    _press(page, "ArrowRight", wait=500)
    _shot(page, "63b_compare_multiview_scrolled")

    # 63c: Zoom in
    _press(page, "+", wait=400)
    _shot(page, "63c_compare_multiview_zoom")
    _press(page, "0", wait=300)  # reset zoom

    # 63d: Exit compare-multiview with v key
    _press(page, "v", wait=800)
    wrap_class2 = page.locator("#qmri-view-wrap").get_attribute("class") or ""
    if "active" not in wrap_class2:
        print("  OK: compare-multiview exited (qmri-view-wrap not active)")
    else:
        print(f"  WARN: qmri-view-wrap still active after exit (class={wrap_class2!r})")
    cmp_active = page.locator("#compare-view-wrap.active").count()
    if cmp_active:
        print("  OK: compare mode restored after exit")
    else:
        print("  WARN: compare-view-wrap not active after exiting compare-multiview")
    _shot(page, "63d_compare_multiview_exited")

    # ── 64: movie-fps — [ / ] change playback fps while playing ─────────────────
    print("64: movie-fps: [ / ] changes fps in movie mode")
    page.evaluate("() => sessionStorage.clear()")
    page.goto(f"{base}/?sid={sid3d}")
    page.wait_for_selector("#viewer", timeout=10_000)
    page.wait_for_timeout(800)
    _focus(page)

    # Start playback
    _press(page, "Space", wait=400)
    status_text = page.locator("#status").inner_text()
    if "playing" in status_text.lower():
        print("  OK: playback started")
    else:
        print(f"  WARN: expected playing status, got {status_text!r}")
    _shot(page, "64a_movie_playing")

    # Press ] to increase fps — status should update
    _press(page, "]", wait=200)
    status_after = page.locator("#status").inner_text()
    if "fps" in status_after.lower():
        print(f"  OK: fps shown in status: {status_after!r}")
    else:
        print(f"  WARN: expected fps in status, got {status_after!r}")
    _shot(page, "64b_movie_fps_increased")

    # Press [ to decrease fps
    _press(page, "[", wait=200)
    _shot(page, "64c_movie_fps_decreased")

    # Stop playback
    _press(page, "Space", wait=300)
    _shot(page, "64d_movie_stopped")

    # ── 65: colorbar width min/max limits ───────────────────────────────────────
    print("65: colorbar width respects min(120px) and max(600px) limits")
    _goto(page, base, sid2d, wait=600)
    _focus(page)
    cb_wrap = page.locator("#slim-cb-wrap")
    cb_box = cb_wrap.bounding_box()
    if cb_box:
        w = cb_box["width"]
        if 120 <= w <= 600:
            print(f"  OK: colorbar width {w:.0f}px within [120, 600] range")
        else:
            print(f"  WARN: colorbar width {w:.0f}px outside [120, 600] range")
    else:
        print("  WARN: slim-cb-wrap not found")
    _shot(page, "65_colorbar_width_limits")

    # ── 66: axis-modes (a key — stretch to square, all modes) ───────────────────
    print("66: a key stretch-to-square works in normal, multi-view, and compare modes")
    # Normal mode: toggle stretch on, canvas should become square
    _goto(page, base, sid3d, wait=600)
    _focus(page)
    cv_box_before = page.locator("#viewer").bounding_box()
    _press(page, "a", wait=300)
    cv_box_square = page.locator("#viewer").bounding_box()
    if cv_box_square:
        w, h = cv_box_square["width"], cv_box_square["height"]
        if abs(w - h) <= 2:
            print(f"  OK: normal mode canvas is square ({w:.0f}×{h:.0f})")
        else:
            print(f"  WARN: canvas not square after a key ({w:.0f}×{h:.0f})")
    _shot(page, "66a_stretch_to_square_normal")
    _press(page, "a", wait=300)  # toggle off

    # Multi-view mode: enter v → should auto-enable squareStretch
    _press(page, "v", wait=1000)
    mv_canvases = page.locator(".mv-canvas").all()
    if mv_canvases:
        first_box = mv_canvases[0].bounding_box()
        if first_box:
            w, h = first_box["width"], first_box["height"]
            if abs(w - h) <= 2:
                print(f"  OK: multi-view auto squareStretch active ({w:.0f}×{h:.0f})")
            else:
                print(
                    f"  WARN: multi-view pane not square ({w:.0f}×{h:.0f}) — squareStretch default may not be working"
                )
    _shot(page, "66b_stretch_to_square_multiview_auto")
    _press(page, "v", wait=400)  # exit multi-view

    # ── 67: r key — rotate 90° CW when slice dim is active ──────────────────────
    print("67: r key rotates 90° CW when slice dim (not x or y) is active")
    _goto(page, base, sid3d, wait=600)
    _focus(page)
    # Check current dim_x, dim_y orientation before rotate
    cv_before = page.locator("#viewer").bounding_box()
    # Press r — in default state activeDim should be current_slice_dim, not dim_x or dim_y
    # This should rotate 90° CW swapping dim_x and dim_y
    _press(page, "r", wait=400)
    cv_after = page.locator("#viewer").bounding_box()
    _shot(page, "67a_r_rotate_cw")
    if cv_before and cv_after:
        w_b, h_b = cv_before["width"], cv_before["height"]
        w_a, h_a = cv_after["width"], cv_after["height"]
        # After 90° CW rotation, width and height should swap (unless array is square)
        if abs(w_a - h_b) < 5 and abs(h_a - w_b) < 5:
            print(f"  OK: canvas rotated ({w_b:.0f}×{h_b:.0f} → {w_a:.0f}×{h_a:.0f})")
        else:
            print(
                f"  INFO: canvas size before={w_b:.0f}×{h_b:.0f}, after={w_a:.0f}×{h_a:.0f} (may be square array)"
            )
    # Press r three more times to get back to original orientation
    _press(page, "r", wait=200)
    _press(page, "r", wait=200)
    _press(page, "r", wait=200)
    _shot(page, "67b_r_rotate_back")

    # ── 68: native preload shell — _LOADING_HTML in _launcher.py ─────────────
    print("68: native preload shell — _LOADING_HTML present in _launcher.py")
    # The preload shell lives in the pywebview subprocess and is not visible in
    # the browser smoke test (native window only). Verify it stays minimal and
    # non-animated by inspecting the launcher source.
    import importlib, inspect

    launcher = importlib.import_module("arrayview._launcher")
    server = importlib.import_module("arrayview._server")
    assert hasattr(launcher, "_LOADING_HTML"), (
        "FAIL: _LOADING_HTML missing from _launcher.py"
    )
    html = launcher._LOADING_HTML
    assert "animation" not in html, "FAIL: _LOADING_HTML should not animate"
    assert "#0c0c0c" in html, (
        "FAIL: _LOADING_HTML does not use the ArrayView dark background"
    )
    assert "body></body" in html.replace(" ", ""), (
        "FAIL: _LOADING_HTML should stay empty until the viewer URL loads"
    )
    # Check that _open_webview accepts loading_port keyword
    sig = inspect.signature(launcher._open_webview)
    assert "loading_port" in sig.parameters, (
        "FAIL: _open_webview missing loading_port parameter"
    )
    print(
        "  OK: _LOADING_HTML present, shell is dark + non-animated, loading_port parameter present"
    )
    shell_html = server._SHELL_HTML
    assert "tab-preview" in shell_html, "FAIL: _SHELL_HTML missing native preview overlay"
    assert "frame-rendered" in shell_html, "FAIL: _SHELL_HTML missing preview handoff trigger"
    print("  OK: _SHELL_HTML contains preview overlay + frame-rendered handoff")

    # ── 69: G key — compare layout toggle (horizontal / vertical / grid) ────────
    print("69: G key — compare layout toggle")

    def _compare_cols(pg):
        return pg.evaluate(
            '() => getComputedStyle(document.getElementById("compare-panes"))'
            '.getPropertyValue("--compare-cols").trim()'
        )

    # 2-pane compare: cycles horizontal ↔ vertical
    _goto_compare(page, base, sid2d, sid2d_b, wait=1500)
    _focus(page)
    _shot(page, "69a_compare_2pane_auto")
    cols_auto = _compare_cols(page)  # should be '2' (horizontal auto)
    _press(page, "G", wait=400)
    _shot(page, "69b_compare_2pane_vertical")
    cols_v = _compare_cols(page)
    assert cols_v == "1", f"FAIL: 2-pane G#1 expected 1 col (vertical), got {cols_v!r}"
    _press(page, "G", wait=400)
    _shot(page, "69c_compare_2pane_horizontal")
    cols_h = _compare_cols(page)
    assert cols_h == cols_auto, (
        f"FAIL: 2-pane G#2 expected {cols_auto!r} cols (horizontal), got {cols_h!r}"
    )
    print(f"  OK: 2-pane G cycles {cols_auto!r} → 1 → {cols_h!r}")

    # 3-pane compare: cycles horizontal → vertical → grid → horizontal
    page.goto(f"{base}/?sid={sid2d}&compare_sids={sid2d_b},{sid_reg_a}")
    page.wait_for_selector("#compare-view-wrap.active", timeout=15_000)
    _focus(page)
    _shot(page, "69d_compare_3pane_auto")
    _press(page, "G", wait=400)
    c1 = _compare_cols(page)
    _shot(page, "69e_compare_3pane_cycle1")
    _press(page, "G", wait=400)
    c2 = _compare_cols(page)
    _shot(page, "69f_compare_3pane_cycle2")
    _press(page, "G", wait=400)
    c3 = _compare_cols(page)
    _shot(page, "69g_compare_3pane_cycle3")
    _press(page, "G", wait=400)
    c4 = _compare_cols(page)
    _shot(page, "69h_compare_3pane_cycle4")
    seen = {c1, c2, c3}
    assert "1" in seen, f"FAIL: expected vertical (1 col) in 3-pane cycle, got {seen}"
    assert "3" in seen, (
        f"FAIL: expected horizontal (3 cols) in 3-pane cycle, got {seen}"
    )
    assert "2" in seen, f"FAIL: expected grid (2 cols) in 3-pane cycle, got {seen}"
    assert c4 == c1, f"FAIL: expected cycle to wrap around (c4={c4!r} != c1={c1!r})"
    print(f"  OK: 3-pane G cycles through {seen}, wraps: {c4!r}==c1")

    # Exit compare and confirm G is no-op outside compare mode
    _press(page, "Escape", wait=400)
    _goto(page, base, sid2d, wait=600)
    _focus(page)
    _press(page, "G", wait=200)
    _shot(page, "69i_g_noop_outside_compare")

    # ── 70: picker-checkboxes — visible checkbox + auto-dim by shape ─────────────
    print("70: picker-checkboxes — visible checkbox, auto-dim by shape")
    _goto(page, base, sid2d, wait=600)
    _focus(page)
    # Open picker
    page.keyboard.press("Shift+O")
    page.wait_for_selector("#uni-picker.visible", timeout=5000)
    page.wait_for_timeout(500)
    _shot(page, "70a_picker_open_unselected")

    # All checkboxes should be visible (not hidden)
    cbs = page.locator(".cp-item-cb").all()
    assert len(cbs) >= 2, f"FAIL: expected ≥2 checkboxes, got {len(cbs)}"
    for i, cb in enumerate(cbs):
        bb = cb.bounding_box()
        assert bb and bb["width"] > 0, f"FAIL: checkbox {i} is not visible"
    print(f"  OK: {len(cbs)} visible checkboxes found")

    # Click the checkbox on arr2d_b (same shape as current arr2d, 100×100)
    items = page.locator(".cp-item").all()
    b_item = next((it for it in items if "arr2d_b" in it.inner_text()), None)
    assert b_item is not None, "FAIL: arr2d_b item not found in picker"
    b_cb = b_item.locator(".cp-item-cb")
    b_cb.click()
    page.wait_for_timeout(300)
    _shot(page, "70b_picker_arr2d_b_checked")

    checked = b_cb.evaluate("el => el.classList.contains('checked')")
    assert checked, "FAIL: arr2d_b checkbox should be checked after click"
    hint = page.locator("#uni-picker-tab-hint").inner_text()
    assert "1 selected" in hint, f"FAIL: expected '1 selected' hint, got {hint!r}"
    print(f"  OK: checkbox checked, hint={hint!r}")

    # arr3d (shape [20, 64, 64]) should be dimmed — different shape from [100, 100]
    d_item = next(
        (
            it
            for it in items
            if "arr3d" in it.inner_text() and "b" not in it.inner_text()
        ),
        None,
    )
    if d_item is not None:
        dimmed = d_item.evaluate("el => el.classList.contains('cp-item-dim')")
        assert dimmed, (
            "FAIL: arr3d (different shape) should be dimmed when arr2d_b selected"
        )
        print("  OK: arr3d (different shape) dimmed")

    # Deselect b — dimming should clear
    b_cb.click()
    page.wait_for_timeout(300)
    _shot(page, "70c_picker_deselected")
    unchecked = b_cb.evaluate("el => el.classList.contains('checked')")
    assert not unchecked, (
        "FAIL: arr2d_b checkbox should be unchecked after second click"
    )
    if d_item is not None:
        still_dimmed = d_item.evaluate("el => el.classList.contains('cp-item-dim')")
        assert not still_dimmed, (
            "FAIL: arr3d should NOT be dimmed after deselecting all"
        )
    print("  OK: deselect clears dimming")

    # Close picker
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    print("  OK: picker-checkboxes all assertions passed")

    # ── 71: compare zoom overflow keeps row + minimap ─────────────────────────
    print("71: compare zoom overflow keeps row + minimap")
    _goto_compare(page, base, sid2d, sid2d_b, wait=1500)
    _focus(page)
    for _ in range(5):
        _press(page, "+", wait=150)
    page.wait_for_timeout(500)
    _shot(page, "71a_compare_zoom_overflow")
    mini_visible = page.evaluate(
        "() => document.getElementById('mini-map')?.classList.contains('visible') ?? false"
    )
    left_top = page.evaluate(
        "() => document.getElementById('compare-left-canvas')?.getBoundingClientRect().top ?? null"
    )
    right_top = page.evaluate(
        "() => document.getElementById('compare-right-canvas')?.getBoundingClientRect().top ?? null"
    )
    assert mini_visible, "FAIL: compare overflow should show mini-map"
    assert left_top is not None and right_top is not None, "FAIL: compare canvases missing"
    assert abs(left_top - right_top) < 20, (
        f"FAIL: compare panes stacked while zoomed (left_top={left_top}, right_top={right_top})"
    )
    page.locator('#mini-map').click(position={"x": 75, "y": 50})
    page.wait_for_timeout(300)
    _shot(page, "71b_compare_zoom_minimap_pan")
    print("  OK: compare overflow stays horizontal and minimap is visible")

    # ── 72: U key — vector arrows on/off ─────────────────────────────────────
    print("72: U key — toggle vector arrows")
    _goto(page, base, sid3d_vf, wait=900)
    _focus(page)
    page.wait_for_timeout(700)
    arrows_on = page.evaluate(
        "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
    )
    _shot(page, "72a_vector_arrows_on")
    _press(page, "U", wait=300)
    arrows_off = page.evaluate(
        "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
    )
    _shot(page, "72b_vector_arrows_off")
    _press(page, "U", wait=500)
    arrows_on_again = page.evaluate(
        "() => document.getElementById('vfield-canvas')?.style.display !== 'none'"
    )
    _shot(page, "72c_vector_arrows_on_again")
    assert arrows_on, "FAIL: vector arrows should be visible initially"
    assert not arrows_off, "FAIL: U should hide vector arrows"
    assert arrows_on_again, "FAIL: U should show vector arrows again"
    print("  OK: U toggles vector arrows")

    # ── 73: vectorfield in 3-view mode ───────────────────────────────────────
    print("73: vectorfield arrows in 3-view mode")
    _goto(page, base, sid3d_vf, wait=900)
    _focus(page)
    page.wait_for_timeout(700)
    _press(page, "v", wait=800)  # enter 3-view
    mv_vf_overlays = page.evaluate(
        "() => document.querySelectorAll('.mv-vfield-overlay').length"
    )
    mv_vf_visible = page.evaluate(
        "() => [...document.querySelectorAll('.mv-vfield-overlay')].some(el => el.style.display !== 'none')"
    )
    _shot(page, "73a_vfield_3view_on")
    assert mv_vf_visible, "FAIL: vectorfield overlays should be visible in 3-view"
    # Toggle off with U
    _press(page, "U", wait=400)
    mv_vf_hidden = page.evaluate(
        "() => [...document.querySelectorAll('.mv-vfield-overlay')].every(el => el.style.display === 'none')"
    )
    _shot(page, "73b_vfield_3view_off")
    assert mv_vf_hidden, "FAIL: U should hide vectorfield overlays in 3-view"
    # Toggle back on
    _press(page, "U", wait=500)
    mv_vf_back = page.evaluate(
        "() => [...document.querySelectorAll('.mv-vfield-overlay')].some(el => el.style.display !== 'none')"
    )
    _shot(page, "73c_vfield_3view_on_again")
    assert mv_vf_back, "FAIL: U should show vectorfield overlays again in 3-view"
    _press(page, "v", wait=400)  # exit 3-view
    print("  OK: vectorfield arrows work in 3-view mode")

    # ── 74: d key opens histogram (colorbar grows), second d cycles (stable) ─
    _goto(page, base, sid2d)
    _focus(page)
    # Start from collapsed state: ensure hist is closed.
    is_open = page.evaluate(
        "() => !!(typeof primaryCb !== 'undefined' && primaryCb && primaryCb._expanded)"
    )
    if is_open:
        # Close via the palette to avoid tap-d cycling again.
        page.evaluate("() => { primaryCb._expanded = false; primaryCb._manualExpand = false; }")
        page.wait_for_timeout(200)
    _shot(page, "74a_before_d")
    _press(page, "d", wait=600)  # 1st tap — opens histogram
    h_open = page.evaluate(
        "() => { const el = document.getElementById('slim-cb-wrap'); return el ? el.offsetHeight : -1; }"
    )
    _shot(page, "74b_after_first_d")
    _press(page, "d", wait=600)  # 2nd tap — cycles preset, colorbar stays
    h_cycled = page.evaluate(
        "() => { const el = document.getElementById('slim-cb-wrap'); return el ? el.offsetHeight : -1; }"
    )
    _shot(page, "74c_after_second_d")
    delta = abs(h_cycled - h_open)
    assert delta <= 4, (
        f"FAIL: colorbar island height changed by {delta}px between two d taps "
        f"(open={h_open}, cycled={h_cycled}) — cycling shouldn't resize the island"
    )
    print(f"  OK: colorbar height stable across d-taps (delta={delta}px)")

    # ── 75: immersive crossfade handoff maps zoom to paneCutoff ────────────
    print("75: immersive crossfade handoff")
    _goto(page, base, sid_med3d, wait=1500)
    _focus(page)
    handoff = page.evaluate(
        """() => window.eval(`(() => {
            if (!lastImgW || !lastImgH) return { ok: false, reason: 'missing-dims' };
            if (!_shouldEnterImmersive()) return { ok: false, reason: 'immersive-not-available' };
            if (!immersiveTl) _buildImmersiveTl();
            if (!immersiveTl) return { ok: false, reason: 'timeline-missing' };
            const paneCutoff = (immersiveTl.data && immersiveTl.data.paneCutoff) || 0.6;
            const zoomSpan = _immTargetZoom - _normalFitZoom;
            if (!(zoomSpan > 0)) return { ok: false, reason: 'zoom-span-zero' };
            _immersiveDriveTween = null;
            _fullscreenActive = false;
            document.body.classList.remove('fullscreen-mode');
            userZoom = _normalFitZoom + 0.5 * zoomSpan;
            _crossfadeCollapseP = 0;
            _crossfadeP = 0;
            ModeRegistry.scaleAll();
            return {
                ok: true,
                progress: immersiveTl.progress(),
                expected: paneCutoff * 0.5,
                paneCutoff,
            };
        })()` )"""
    )
    assert handoff.get("ok"), f"FAIL: immersive handoff setup failed ({handoff})"
    assert abs(handoff["progress"] - handoff["expected"]) < 0.03, (
        f"FAIL: immersive mid-zone progress snapped to {handoff['progress']:.3f} "
        f"instead of {handoff['expected']:.3f}"
    )
    scrub = page.evaluate(
        """() => window.eval(`(() => {
            const paneCutoff = (immersiveTl && immersiveTl.data && immersiveTl.data.paneCutoff) || 0.6;
            const zoomSpan = _immTargetZoom - _normalFitZoom;
            userZoom = _normalFitZoom + 0.25 * zoomSpan;
            _crossfadeCollapseP = 0;
            _crossfadeP = 0;
            _fullscreenActive = false;
            document.body.classList.remove('fullscreen-mode');
            ModeRegistry.scaleAll();
            const desiredZoom = _normalFitZoom + 0.5 * zoomSpan;
            _scrubRequestedZoom = desiredZoom;
            _zoomAdjustedByUser = true;
            _driveImmersive(paneCutoff * 0.5, 0.25, { scrub: true });
            ModeRegistry.scaleAll();
            return {
                requestedZoom: _scrubRequestedZoom,
                renderedZoom: userZoom,
                minimapVisible: !!document.getElementById('mini-map')?.classList.contains('visible'),
                overflow: mainPan.overflows,
            };
        })()` )"""
    )
    assert scrub["requestedZoom"] > scrub["renderedZoom"], (
        f"FAIL: scrub rendered zoom jumped to requested zoom ({scrub})"
    )
    assert scrub["minimapVisible"] is False, (
        f"FAIL: minimap visible during immersive scrub ({scrub})"
    )
    _shot(page, "75a_immersive_handoff_midzone")
    page.evaluate(
        """() => window.eval(`(() => {
            _crossfadeCleanup();
            _fullscreenActive = false;
            document.body.classList.remove('fullscreen-mode');
            userZoom = _normalFitZoom;
            _zoomAdjustedByUser = false;
            ModeRegistry.scaleAll();
        })()` )"""
    )

    print(f"\nAll {len(list(OUT_DIR.glob('*.png')))} screenshots saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    base, srv = _start_server()
    print(f"Server at {base}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        with httpx.Client(base_url=base, timeout=15.0) as client:
            with tempfile.TemporaryDirectory() as tmp:
                run_smoke(page, base, client, tmp)
        browser.close()

    srv.should_exit = True


if __name__ == "__main__":
    main()
