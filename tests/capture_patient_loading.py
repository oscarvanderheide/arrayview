"""Capture the patient-loading spinner for manual animation review.

Run from the repo root:
    uv run python tests/capture_patient_loading.py

Frames are written to ``tests/patient_loading_frames/``. The canvas bounds are
recorded in ``bounds.txt`` so the sequence also proves the overlay does not move
or resize the viewer.
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


OUT_DIR = Path(__file__).parent / "patient_loading_frames"


def _start_server() -> tuple[str, uvicorn.Server]:
    from arrayview._app import app

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        port = sock.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(app, host="localhost", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    base = f"http://localhost:{port}"
    for _ in range(50):
        try:
            if httpx.get(f"{base}/ping", timeout=1.0).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    return base, server


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for png in OUT_DIR.glob("*.png"):
        png.unlink()

    base, server = _start_server()
    array = np.random.default_rng(0).normal(size=(24, 96, 96)).astype("float32")
    with NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp.name, array)
        path = tmp.name
    sid = httpx.post(base + "/load", json={"filepath": path, "name": "patient spinner"}).json()["sid"]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(f"{base}/?sid={sid}")
        page.wait_for_selector("#canvas-wrap", state="visible", timeout=15_000)
        page.wait_for_selector("#loading-overlay", state="hidden", timeout=15_000)
        page.wait_for_function("() => lastImageData !== null && lastImgW > 0 && lastImgH > 0", timeout=15_000)
        page.wait_for_function(
            """() => {
                const canvas = document.getElementById('viewer');
                const rect = canvas.getBoundingClientRect();
                const current = [rect.x, rect.y, rect.width, rect.height].join(',');
                if (window.__patientCaptureBounds === current) return true;
                window.__patientCaptureBounds = current;
                return false;
            }""",
            polling=200,
            timeout=5_000,
        )
        page.wait_for_timeout(250)
        before = page.locator("#viewer").bounding_box()
        page.evaluate("() => { _displayedPatientIndex = 0; _patientLoadingIndex = 1; _patientLoadingVisible = true; _reconcileUI(); }")
        page.wait_for_selector("#patient-loading-overlay.visible")
        for idx in range(12):
            page.screenshot(path=str(OUT_DIR / f"spinner_{idx:02d}.png"), full_page=False)
            page.wait_for_timeout(75)
        after = page.locator("#viewer").bounding_box()
        (OUT_DIR / "bounds.txt").write_text(f"before={before}\nafter={after}\n", encoding="utf-8")
        page.evaluate("() => _patientFrameDisplayed(1)")
        page.wait_for_selector("#patient-loading-overlay", state="hidden")
        page.screenshot(path=str(OUT_DIR / "complete.png"), full_page=False)
        browser.close()

    server.should_exit = True
    print(f"saved frames to {OUT_DIR}")


if __name__ == "__main__":
    main()
