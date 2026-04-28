"""Capture multiview entry/exit frames for manual animation review.

Run from the repo root:
    uv run python tests/capture_v_animation.py

This helper launches a temporary local server, opens a synthetic 3-D array in the
browser, toggles multiview on/off with `v`, and writes a frame sequence to
`tests/v_anim_frames/` for visual inspection.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import numpy as np
import uvicorn
from playwright.sync_api import sync_playwright


OUT_DIR = Path(__file__).parent / "v_anim_frames"


def _start_server() -> tuple[str, uvicorn.Server]:
    from arrayview._app import app

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    cfg = uvicorn.Config(app, host="localhost", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()

    base = f"http://localhost:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{base}/ping", timeout=1.0).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    return base, server


def _save_frames(page, prefix: str, count: int, delay_ms: int) -> None:
    for idx in range(count):
        page.screenshot(path=str(OUT_DIR / f"{prefix}_{idx:02d}.png"), full_page=False)
        page.wait_for_timeout(delay_ms)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for png in OUT_DIR.glob("*.png"):
        png.unlink()

    base, _server = _start_server()

    arr = np.random.default_rng(0).normal(size=(40, 96, 96)).astype("float32")
    arr[:, 24:72, 32:64] += 2.5

    with NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, arr)
        tmp_path = tmp.name

    resp = httpx.post(base + "/load", json={"filepath": tmp_path, "name": "v_anim"})
    resp.raise_for_status()
    sid = resp.json()["sid"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page = context.new_page()
        page.goto(f"{base}/?sid={sid}")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page.wait_for_timeout(1200)
        page.locator("#keyboard-sink").focus()

        page.screenshot(path=str(OUT_DIR / "before.png"), full_page=False)
        page.keyboard.press("v")
        _save_frames(page, "enter", count=15, delay_ms=50)
        page.wait_for_timeout(250)
        page.screenshot(path=str(OUT_DIR / "entered.png"), full_page=False)

        page.keyboard.press("v")
        _save_frames(page, "exit", count=15, delay_ms=50)
        page.wait_for_timeout(250)
        page.screenshot(path=str(OUT_DIR / "after.png"), full_page=False)

        browser.close()

    print(f"saved frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
