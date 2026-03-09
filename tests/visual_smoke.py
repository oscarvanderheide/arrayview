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
  h/l/←/→        move cursor to dim        ✗ (moves info label only)
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
  V               3-plane custom dims       ✗ (needs dim picker interaction)
  o               reset oblique (multiview) ✓ 23 (enter mv, rotate, reset)
  q               qMRI mode                 ✓ 10-12

DISPLAY
  b               toggle border             ✓ 16
  c               cycle colormap            ✓ 02-03
  C               custom colormap (dialog)  ✗ (requires dialog input)
  d               cycle dynamic range       ✓ 17
  D               manual vmin/vmax (dialog) ✗ (requires dialog input)
  B               compare picker (dialog)   ✗ (requires dialog input)
  R               registration overlay      ✓ 24, 37 (compare mode + R)
  [ / ]           registration blend        ✓ 24, 37
  n               cycle compare session     ✗ (needs multi-session setup)
  Z               zen mode                  ✓ 06
  L               log scale                 ✓ 18
  M               mask threshold            ✗ (visual effect subtle)
  m               cycle complex mode        ✓ 25 (complex array + m)
  f               centred FFT (dialog)      ✗ (requires dialog input)
  T               cycle theme               ✓ 26

INFO & EXPORT
  hover           pixel value + cb marker   ✓ 15
  i               data info overlay         ✓ 27
  s               save screenshot           ✗ (triggers download dialog)
  g               save GIF                  ✗ (triggers download dialog)
  e               copy URL                  ✗ (clipboard, no visual change)
  ?               help overlay              ✓ 28

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
  h/l — cursor dim switch                   ✓ 32 (bounding box check)
  j/k — slice navigate                      ✓ 33 (bounding box check)
  Z — zen mode toggle                       ✓ 34 (before/after)
  b — border toggle                         ✓ 35 (before/after)
  +/- — zoom in/out                         ✓ 36 (canvas resizes, cb stays below)
  registration arrays (phantom)             ✓ 37 (shifted ellipse, reg overlay)

═══════════════════════════════════════════════════════════════════
RULE: when you add a keyboard shortcut, add a scenario here.
═══════════════════════════════════════════════════════════════════
"""

import os
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
    arr2d   = np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(100, 80)
    arr3d   = rng.standard_normal((20, 64, 64)).astype(np.float32)
    arr4d   = rng.standard_normal((5, 20, 32, 32)).astype(np.float32)   # dim 0 = 5 for qmri
    arr4d_z = rng.standard_normal((4, 8, 32, 32)).astype(np.float32)    # 4D for mosaic (z key)
    arrC    = (rng.standard_normal((20, 32, 32)) + 1j * rng.standard_normal((20, 32, 32))).astype(np.complex64)
    # Medical image sizes: 192×192×96 volume (~14 MB) and 5-echo 4D (~8 MB)
    arr_med3d = rng.standard_normal((96, 192, 192)).astype(np.float32)
    arr_med4d = rng.standard_normal((5, 48, 96, 96)).astype(np.float32)
    # Registration phantom: bright ellipse, version B shifted 8px right + 5px down
    Y, X = np.mgrid[-64:64, -64:64].astype(np.float32)
    arr_reg_a = np.exp(-(X**2 / 800 + Y**2 / 1800))
    arr_reg_b = np.exp(-((X - 8)**2 / 800 + (Y - 5)**2 / 1800)) * 0.85

    sid2d      = _load(client, arr2d,   "arr2d",   tmp)
    sid3d      = _load(client, arr3d,   "arr3d",   tmp)
    sid4d      = _load(client, arr4d,   "arr4d",   tmp)
    sid4d_z    = _load(client, arr4d_z, "arr4d_z", tmp)
    sidC       = _load(client, arrC,    "arrC",     tmp)
    sid2d_b    = _load(client, arr2d * 0.5 + 0.25, "arr2d_b", tmp)
    sid_med3d  = _load(client, arr_med3d, "med3d", tmp)
    sid_med4d  = _load(client, arr_med4d, "med4d", tmp)
    sid_reg_a  = _load(client, arr_reg_a, "reg_a", tmp)
    sid_reg_b  = _load(client, arr_reg_b, "reg_b", tmp)
    arr3d_qmri3 = rng.standard_normal((3, 20, 32, 32)).astype(np.float32)
    sid_qmri3 = _load(client, arr3d_qmri3, "qmri3", tmp)

    # ── 01: 2D default view ──────────────────────────────────────────────────
    _goto(page, base, sid2d)
    _shot(page, "01_2d_default")

    # ── 02-03: colormap cycling (c) ──────────────────────────────────────────
    _focus(page)
    _press(page, "c")
    _shot(page, "02_2d_colormap_lipari")
    _press(page, "c"); _press(page, "c")
    _shot(page, "03_2d_colormap_viridis")
    # reset
    for _ in range(4): _press(page, "c", wait=100)

    # ── 04-05: zoom (+ / - / 0) ──────────────────────────────────────────────
    _press(page, "+"); _press(page, "+")
    _shot(page, "04_zoom_in")
    _press(page, "-"); _press(page, "-"); _press(page, "-")
    _shot(page, "05_zoom_out")
    _press(page, "0")

    # ── 06: zen mode (Z = capital) ───────────────────────────────────────────
    _goto(page, base, sid3d)
    _focus(page)
    _press(page, "Shift+Z")
    _shot(page, "06_zen_mode_on")
    _press(page, "Shift+Z")  # off

    # ── 07: 3D scrolled (j/k arrows) ────────────────────────────────────────
    for _ in range(5): _press(page, "ArrowRight", wait=100)
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
    for _ in range(3): _press(page, "ArrowRight", wait=300)
    _shot(page, "11_qmri_5panel_scrolled")
    _press(page, "q", wait=400)  # exit

    # ── 12: qMRI 3-panel ─────────────────────────────────────────────────────
    _goto(page, base, sid_qmri3)
    _focus(page)
    _press(page, "q", wait=1500)
    _shot(page, "12_qmri_3panel")
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

    # ── 17: dynamic range (d) ────────────────────────────────────────────────
    _press(page, "d")
    _shot(page, "17_dynamic_range_1pct")
    _press(page, "d"); _press(page, "d"); _press(page, "d")  # back to 0-100

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
    page.mouse.click(mv_box["x"] + mv_box["width"] * 0.3, mv_box["y"] + mv_box["height"] * 0.3)
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
    _press(page, "]"); _press(page, "]")
    _shot(page, "24b_registration_blend_increased")
    _press(page, "Shift+R")  # exit reg mode

    # ── 25: complex mode cycling (m) ─────────────────────────────────────────
    _goto(page, base, sidC)
    _focus(page)
    _shot(page, "25a_complex_mag")
    _press(page, "m")
    _shot(page, "25b_complex_phase")
    _press(page, "m"); _press(page, "m")  # back to mag

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
    for _ in range(8): _press(page, "ArrowRight", wait=80)
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
    _press(page, "+"); _press(page, "+")
    _shot(page, "36b_stab_zoom_in")
    _press(page, "-"); _press(page, "-"); _press(page, "-")
    _shot(page, "36c_stab_zoom_out")
    _press(page, "0")
    _shot(page, "36d_stab_zoom_reset")

    # ── 37: registration overlay with structured phantom arrays ───────────────
    _goto_compare(page, base, sid_reg_a, sid_reg_b)
    _focus(page)
    _shot(page, "37a_reg_before_overlay")
    _press(page, "Shift+R")
    _shot(page, "37b_reg_overlay_on")
    _press(page, "]"); _press(page, "]")
    _shot(page, "37c_reg_blend_increased")
    _press(page, "Shift+R")  # exit reg mode

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
