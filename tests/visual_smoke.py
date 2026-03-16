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
  r               reverse axis              ✓ 19 (r key reverses)
  Space           toggle auto-play          ✓ 20 (play then stop)
  + / -           zoom in/out               ✓ 04 (zoom in), 05 (zoom out+reset)
  0               reset zoom                ✓ 05

AXES & VIEWS
  x               swap x dim with slice     ✓ 21
  y               swap y dim with slice     ✓ 22
  z               mosaic mode (4D+ only)    ✓ 09 (arr_4d + z key)
  v               3-plane multiview         ✓ 08
  V               3-plane custom dims       ✓ 41 (inline prompt, type dims)
  o               reset oblique (multiview) ✓ 23 (enter mv, rotate, reset)
  q               qMRI mode & toggle        ✓ 10-12, 12a-c (toggle compact/full)

DISPLAY
  b               toggle border             ✓ 16
  c               cycle colormap            ✓ 02-03
  C               custom colormap (dialog)  ✗ (requires dialog input)
  d               cycle dynamic range       ✓ 17
  D               manual vmin/vmax (dialog) ✓ 44 (inline prompt)
   B               compare picker (dialog)   ✗ (requires dialog interaction)
   P               unified picker – compare  ✓ 45 (uni-picker opens in Side-by-side mode; disabled in inline embed; enabled in native shell iframe)
   Cmd+O / Ctrl+O  unified picker – open     ✓ 45 (uni-picker cycles to open mode; disabled in inline embed; enabled in native shell iframe)
   (search box)    substring filter          ✓ 45e (type query, list filters client-side; box top-anchored, no jump)
   (arrow keys)    navigate picker list      ✓ 45f (ArrowDown from search moves to first item)
  X               diff view (compare mode)  ✓ 39 (2-pane compare + X cycle)
  R               registration overlay      ✓ 24, 37 (compare mode + R)
  [ / ]           registration blend        ✓ 24, 37
  n               cycle compare session     ✗ (needs multi-session setup)
  Z               zen mode                  ✓ 06
   L               log scale                 ✓ 18, 40a (LOG egg in #mode-eggs); no-op in RGB mode (toast shown)
   M               mask threshold            ✓ 40f (MASK egg in #mode-eggs)
   m               cycle complex mode        ✓ 25 (complex array + m), 40b-e (complex egg in #mode-eggs); no-op in RGB mode (toast shown)
  f               centred FFT (dialog)      ✓ 42 (inline prompt, enter axes)
  T               cycle theme               ✓ 26

INFO & EXPORT
  hover           pixel value + cb marker   ✓ 15, 43 (tooltip follows cursor; H enables first)
  H               toggle pixel hover tip    ✓ 43
  i               data info overlay         ✓ 27
  s               save screenshot           ✗ (triggers download dialog)
  g               save GIF                  ✗ (triggers download dialog)
  e               copy URL                  ✗ (clipboard, no visual change)
  ?               help overlay              ✓ 28
  toast routing   diff/border toasts → #status (bottom-left)  ✓ 49

LOADING ANIMATION
  logo pulse-anim while loading-overlay vis ✓ 47 (js eval checks .av-logo-loading)
  logo-b0..b8 IDs present for pulse anim   ✓ 47 (js eval checks rect IDs)
  logo stops after canvas visible           ✓ 47
  ping-pong loading bar absent              ✓ 47 (#loading-track not in DOM)
  loading text absent                       ✓ 47 (#loading-label not in DOM)

WELCOME SCREEN / DEMO
  empty-hint visible on welcome session     ✓ 48 (check .visible on #empty-hint)
  dim-label active no hover highlight       ✓ 52 (hover bg transparent for all labels)
  dim-track drag scrubs index               ✓ 53 (mousedown+move on track)

AXES INDICATOR (edge labels)
  h/l dims flash axes labels, fade in+out   ✓ 50 (opacity checked after h press)
  axes visible in mosaic mode (z key)       ✓ 50 (mosaic mode flash)
  axes visible in multiview (v key)         ✓ 50 (multiview flash)
  axes visible in compare mode (B key)      ✗ (requires interactive picker)

VIEW MODES (colorbar visible in all)
  single 2d                                 ✓ 01
  single 3d                                 ✓ 06
  mosaic (grid)                             ✓ 09
  multiview (3-plane)                       ✓ 08
  qmri 3-panel                              ✓ 12
  qmri 5-panel                              ✓ 10-11
  compare 2-array                           ✓ 13-14

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
  registration arrays (phantom)             ✓ 37 (shifted ellipse, reg overlay)
  multiview uniform cells + zoom limit      ✓ 38 (3 panes same size, zoom caps)
  compare diff view (X key)                ✓ 39 (A−B, |A−B|, relative)
  LOG, complex, and mask eggs               ✓ 40 (badges in #mode-eggs below canvas)
  RGB egg spacing below canvas              ✓ 46 (eggs top > canvas bottom + 30px)
  V custom multiview dims                   ✓ 41 (inline prompt)
  f FFT via inline prompt                   ✓ 42 (inline prompt)

PERFORMANCE
  compute_global_stats ndim≥4 2D sampling  ✓ test_api.py (samples 2D slices, not 3D volumes)

OVERLAY
  multiple masks (overlay_sid=sid1,sid2)   ✓ 51 (two binary masks, red+green palette)
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

    # ── 01: 2D default view ──────────────────────────────────────────────────
    _goto(page, base, sid2d)
    _shot(page, "01_2d_default")

    # ── 02-03: colormap cycling (c) ──────────────────────────────────────────
    _focus(page)
    _press(page, "c")
    _shot(page, "02_2d_colormap_lipari")
    _press(page, "c")
    _press(page, "c")
    _shot(page, "03_2d_colormap_viridis")
    # reset
    for _ in range(4):
        _press(page, "c", wait=100)

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

    # ── 17: dynamic range (d) ────────────────────────────────────────────────
    _press(page, "d")
    _shot(page, "17_dynamic_range_1pct")
    _press(page, "d")
    _press(page, "d")
    _press(page, "d")  # back to 0-100

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

    # ── 24: compare + registration overlay (R, [, ]) ─────────────────────────
    _goto_compare(page, base, sid2d, sid2d_b)
    _focus(page)
    _press(page, "Shift+R")
    _shot(page, "24a_registration_overlay")
    _press(page, "]")
    _press(page, "]")
    _shot(page, "24b_registration_blend_increased")
    _press(page, "Shift+R")  # exit reg mode

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
    _press(page, "Shift+T")
    _shot(page, "26b_theme_solarized")
    _press(page, "Shift+T")
    _shot(page, "26c_theme_nord")
    _press(page, "Shift+T")  # back to dark

    # ── 27: data info overlay (i) ────────────────────────────────────────────
    _press(page, "i", wait=800)
    _shot(page, "27_data_info_overlay")
    _press(page, "Escape", wait=300)

    # ── 28: help overlay (?) ─────────────────────────────────────────────────
    _press(page, "?", wait=400)
    _shot(page, "28_help_overlay")
    _press(page, "?")

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

    # ── 37: registration overlay with structured phantom arrays ───────────────
    _goto_compare(page, base, sid_reg_a, sid_reg_b)
    _focus(page)
    _shot(page, "37a_reg_before_overlay")
    _press(page, "Shift+R")
    _shot(page, "37b_reg_overlay_on")
    _press(page, "]")
    _press(page, "]")
    _shot(page, "37c_reg_blend_increased")
    _press(page, "Shift+R")  # exit reg mode

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

    # ── 39: compare diff view (X key cycles diff modes) ──────────────────────
    _goto_compare(page, base, sid2d, sid2d_b)
    _focus(page)
    _shot(page, "39a_diff_compare_base")
    _press(page, "Shift+X", wait=800)  # diff: A−B
    _shot(page, "39b_diff_AB")
    _press(page, "Shift+X", wait=800)  # diff: |A−B|
    _shot(page, "39c_diff_abs")
    _press(page, "Shift+X", wait=800)  # diff: |A−B|/|A|
    _shot(page, "39d_diff_rel")
    _press(page, "Shift+X", wait=400)  # diff: off
    _shot(page, "39e_diff_off")

    # ── 40: LOG, complex, and mask eggs in #mode-eggs ────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "Shift+L", wait=400)  # LOG on → LOG egg
    _shot(page, "40a_log_egg")
    _press(page, "Shift+L", wait=200)  # LOG off
    _press(page, "Shift+M", wait=600)  # MASK on → MASK egg
    _shot(page, "40f_mask_egg")
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

    # ── 44: D key — manual vmin/vmax via inline prompt ───────────────────────
    _goto(page, base, sid2d)
    _focus(page)
    _press(page, "D", wait=400)  # opens inline prompt for vmin
    _shot(page, "44a_D_vmin_prompt")
    page.locator("#inline-prompt-input").fill("0.2")
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)  # opens inline prompt for vmax
    _shot(page, "44b_D_vmax_prompt")
    page.locator("#inline-prompt-input").fill("0.8")
    page.keyboard.press("Enter")
    page.wait_for_timeout(400)
    _shot(page, "44c_D_range_locked")

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
    _press(page, "O", wait=800)  # open picker in Open mode
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
    # Verify RGB egg badge is at least 30px below the canvas bottom (spacing fix)
    eggs_rect = page.evaluate(
        "() => { const e = document.getElementById('mode-eggs'); return e ? e.getBoundingClientRect() : null; }"
    )
    canvas_rect = page.evaluate(
        "() => { const c = document.getElementById('viewer'); return c ? c.getBoundingClientRect() : null; }"
    )
    if eggs_rect and canvas_rect:
        gap = eggs_rect["top"] - canvas_rect["bottom"]
        if gap < 30:
            print(
                f"  WARNING: RGB eggs gap too small ({gap:.0f}px < 30px) — eggs may overlap canvas"
            )

    # ── 47: logo animation ────────────────────────────────────────────────────
    # Verify the logo animates (opacity pulse) while loading and stops after canvas visible.
    # Also verify #loading-track (ping-pong bar) and #loading-label (text) are gone.
    # Verify SVG rect IDs (logo-b0..b8) required for pulse animation.
    _goto(page, base, sid2d)
    logo_has_class = page.evaluate(
        "() => document.getElementById('av-logo-svg').classList.contains('av-logo-loading')"
    )
    if logo_has_class:
        print("  WARNING: av-logo-loading class still present after canvas loaded")
    missing_rects = page.evaluate(
        "() => Array.from({length:9},(_,i)=>`logo-b${i}`).filter(id=>!document.getElementById(id))"
    )
    if missing_rects:
        print(
            f"  WARNING: SVG logo rect IDs missing (pulse animation broken): {missing_rects}"
        )
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
    _shot(page, "47_logo_after_load")

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
    # empty-hint ("O open · drop file to load") must be visible on welcome screen
    hint_visible = page.evaluate(
        "() => document.getElementById('empty-hint').classList.contains('visible')"
    )
    if not hint_visible:
        print(
            "  WARNING: #empty-hint not visible on welcome demo screen — fix _isWelcomeScreen logic"
        )

    # ── 49: toast routing — showToast() messages appear in #status (bottom-left) ──
    # Press X in normal mode (no compare) — triggers "diff view: only in 2-pane
    # compare mode", previously shown in the top-center #toast div.
    # After the fix, the message must appear in #status (bottom-left fading toast)
    # and the #toast element must not exist in the DOM.
    _goto(page, base, sid2d)
    _focus(page)
    toast_present = page.evaluate("() => !!document.getElementById('toast')")
    if toast_present:
        print(
            "  WARNING: #toast element still present in DOM — should have been removed"
        )
    _press(page, "X", wait=400)
    status_text = page.locator("#status").inner_text()
    _shot(page, "49_toast_in_status")
    if not status_text.strip():
        print(
            "  WARNING: #status is empty after X in normal mode — toast may not route correctly"
        )

    # ── 50: axes indicator — edge labels flash on h/l, mosaic, multiview ─────
    # Verify that pressing h in a 3D array makes .axes-indicator opacity become 1.
    # Also verify axes are visible in mosaic mode (not hidden when dim_z >= 0).
    # Also verify axes are ALWAYS visible in multiview mode (not just flashing).
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
    _shot(page, "50a_axes_flash_normal")
    if ax_opacity_after < 0.5:
        print(
            f"  WARNING: axes indicator opacity={ax_opacity_after:.2f} after h — expected ~1.0 (fade-in not working)"
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

    # ── 53: dim-track drag to scrub ─────────────────────────────────────────
    print("53: dim-track drag scrubs dimension index")
    _goto(page, base, sid3d, wait=800)
    _focus(page)
    # Get the track element for a scroll dim (not x/y)
    # Move to the far-left of the track and drag to far-right, index should change
    idx_before = page.evaluate(
        "() => window._arrayviewState ? window._arrayviewState.indices : null"
    )
    track_box = page.evaluate(
        "() => {"
        "  const track = document.querySelector('.dim-label.dim-chip .dim-track');"
        "  if (!track) return null;"
        "  const r = track.getBoundingClientRect();"
        "  return { x: r.left, y: r.top + r.height / 2, w: r.width };"
        "}"
    )
    if track_box:
        page.mouse.move(track_box["x"] + track_box["w"] * 0.1, track_box["y"])
        page.mouse.down()
        page.mouse.move(track_box["x"] + track_box["w"] * 0.9, track_box["y"])
        page.wait_for_timeout(200)
        page.mouse.up()
        page.wait_for_timeout(300)
    _shot(page, "53_dim_track_drag")

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
