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
  R               registration overlay      ✓ 24 (compare mode + R)
  [ / ]           registration blend        ✓ 24
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
    page.goto(f"{base}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
    page.wait_for_timeout(wait)


def _goto_compare(page, base, sid_a, sid_b, wait=2000):
    page.goto(f"{base}/?sid_a={sid_a}&sid_b={sid_b}")
    page.wait_for_timeout(wait)


def _focus(page):
    page.locator("#keyboard-sink").focus()


def _press(page, key, wait=400):
    page.keyboard.press(key)
    page.wait_for_timeout(wait)


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

    sid2d   = _load(client, arr2d,   "arr2d",   tmp)
    sid3d   = _load(client, arr3d,   "arr3d",   tmp)
    sid4d   = _load(client, arr4d,   "arr4d",   tmp)
    sid4d_z = _load(client, arr4d_z, "arr4d_z", tmp)
    sidC    = _load(client, arrC,    "arrC",     tmp)
    sid2d_b = _load(client, arr2d * 0.5 + 0.25, "arr2d_b", tmp)
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
