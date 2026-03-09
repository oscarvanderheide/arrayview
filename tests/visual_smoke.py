"""
Visual smoke test for arrayview.

Loads arrays in every major view mode and saves screenshots to tests/smoke_output/.
Run after any UI change to catch visual regressions before committing.

Usage:
    uv run python tests/visual_smoke.py

Screenshots are written to tests/smoke_output/<name>.png.
Open them with your image viewer to inspect.
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
# Server setup
# ---------------------------------------------------------------------------

def _start_server():
    from arrayview._app import app
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    srv = uvicorn.Server(config)
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
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
    print(f"  saved {name}.png")


def _goto(page, base, sid, wait=1200):
    page.goto(f"{base}/?sid={sid}")
    page.wait_for_selector("#canvas-wrap,#qmri-view-wrap.active,#multi-view-wrap.active", timeout=15_000)
    page.wait_for_timeout(wait)


def _focus(page):
    page.locator("#keyboard-sink").focus()


# ---------------------------------------------------------------------------
# Smoke scenarios
# ---------------------------------------------------------------------------

def run_smoke(page, base, client, tmp):
    rng = np.random.default_rng(42)

    # --- 2D float ---
    arr2d = np.linspace(0, 1, 100 * 80, dtype=np.float32).reshape(100, 80)
    sid2d = _load(client, arr2d, "arr2d", tmp)
    _goto(page, base, sid2d)
    _shot(page, "01_2d_gray")

    _focus(page)
    page.keyboard.press("c")
    _shot(page, "02_2d_lipari")

    page.keyboard.press("c")
    page.keyboard.press("c")
    _shot(page, "03_2d_viridis")

    # Cycle back to gray
    for _ in range(4):
        page.keyboard.press("c")

    # Zoom in/out
    page.keyboard.press("+")
    page.keyboard.press("+")
    _shot(page, "04_2d_zoomed_in")
    page.keyboard.press("-")
    page.keyboard.press("-")

    # Zen mode
    page.keyboard.press("z")
    _shot(page, "05_2d_zen")
    page.keyboard.press("z")

    # --- 3D float ---
    arr3d = rng.standard_normal((20, 64, 64)).astype(np.float32)
    sid3d = _load(client, arr3d, "arr3d", tmp)
    _goto(page, base, sid3d)
    _shot(page, "06_3d_normal")

    # Scroll a few slices
    _focus(page)
    for _ in range(5):
        page.keyboard.press("ArrowRight")
    _shot(page, "07_3d_scrolled")

    # Multiview
    page.keyboard.press("v")
    page.wait_for_timeout(1000)
    _shot(page, "08_3d_multiview")
    page.keyboard.press("v")
    page.wait_for_timeout(400)

    # Mosaic
    page.keyboard.press("z")  # just to make sure zen is off
    page.keyboard.press("z")
    _focus(page)
    page.keyboard.press("m")
    page.wait_for_timeout(800)
    _shot(page, "09_3d_mosaic")
    page.keyboard.press("m")
    page.wait_for_timeout(400)

    # --- QMRI (5-panel, uses twilight_shifted) ---
    arr_qmri = rng.standard_normal((5, 20, 32, 32)).astype(np.float32)
    sid_qmri = _load(client, arr_qmri, "qmri5", tmp)
    _goto(page, base, sid_qmri)
    _focus(page)
    page.keyboard.press("q")
    page.wait_for_timeout(2000)  # all 5 panels need time
    _shot(page, "10_qmri_5panel")

    # Scroll in qmri
    for _ in range(3):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(800)
    _shot(page, "11_qmri_scrolled")

    page.keyboard.press("q")
    page.wait_for_timeout(400)

    # QMRI with 3 panels
    arr_qmri3 = rng.standard_normal((3, 20, 32, 32)).astype(np.float32)
    sid_qmri3 = _load(client, arr_qmri3, "qmri3", tmp)
    _goto(page, base, sid_qmri3)
    _focus(page)
    page.keyboard.press("q")
    page.wait_for_timeout(1500)
    _shot(page, "12_qmri_3panel")
    page.keyboard.press("q")

    # --- Compare mode ---
    sid2d_b = _load(client, arr2d * 0.5 + 0.25, "arr2d_b", tmp)
    page.goto(f"{base}/?sid_a={sid2d}&sid_b={sid2d_b}")
    page.wait_for_timeout(2000)
    _shot(page, "13_compare_2d")

    # --- Colorbar in each mode ---
    # Single view colorbar
    _goto(page, base, sid3d)
    _shot(page, "14_colorbar_single")

    # Hover over canvas to check marker
    canvas = page.locator("#viewer")
    box = canvas.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.5)
    page.wait_for_timeout(400)
    _shot(page, "15_colorbar_marker")

    # Borders on/off
    _focus(page)
    page.keyboard.press("b")
    _shot(page, "16_borders_on")
    page.keyboard.press("b")

    print(f"\nAll screenshots saved to {OUT_DIR}/")
    print("Open them to check for visual issues.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    base, srv = _start_server()
    print(f"Server running at {base}")

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
