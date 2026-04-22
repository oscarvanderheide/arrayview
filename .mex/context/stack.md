---
name: stack
description: Technology stack, library choices, and the reasoning behind them. Load when working with specific technologies or making decisions about libraries and tools.
triggers:
  - "library"
  - "package"
  - "dependency"
  - "which tool"
  - "technology"
edges:
  - target: context/decisions.md
    condition: when the reasoning behind a tech choice is needed
  - target: context/conventions.md
    condition: when understanding how to use a technology in this codebase
  - target: context/architecture.md
    condition: when understanding how a library fits into the overall system
last_updated: 2026-04-22
---

# Stack

## Core Technologies

- **Python 3.12+** — minimum required version (`requires-python = ">=3.12"`)
- **FastAPI** — async HTTP + WebSocket server for the network mode (`_server.py`)
- **uvicorn** — ASGI server running FastAPI; lazy-imported via `_uvicorn()` in `_launcher.py`
- **numpy 2.4+** — core array type; the only eager import in the render path
- **uv** — package manager and task runner (`uv run pytest`, `uv build`, `uvx arrayview`)

## Key Libraries

- **matplotlib 3.9+** — colormap LUT generation only; lazy-initialized once by `_init_luts()` in `_render.py`. Never imported at module level.
- **nibabel 5.3+** — NIfTI file loading (`.nii`, `.nii.gz`); lazy-imported via `_nib()` in `_io.py`.
- **zarr 2.17+** — lazy chunk access for `.zarr` / `.zarr.zip` files; chunk preset utility in `_session.py`.
- **pillow 12+** — PNG encoding for slice frames sent over WebSocket.
- **pywebview 6.1+** — native OS window for CLI / script invocations; lazy, only started when `_can_native_window()` is true.
- **qmricolors** — registers `lipari` and `navia` colormaps into matplotlib; Git dependency (`https://github.com/oscarvanderheide/qmricolors.git`). Imported inside `_init_luts()`.
- **scipy** — `.mat` file loading via `scipy.io.loadmat`; lazy in `_io.py`.
- **h5py** — `.h5` / `.hdf5` file loading; lazy in `_io.py`.
- **tifffile** — `.tif` / `.tiff` file loading; lazy in `_io.py`.
- **websockets 14+** — WebSocket transport (used by uvicorn); not imported directly in application code.
- **pytest** (not unittest) — all tests; browser tests require `pytest-playwright` and are marked `@pytest.mark.browser`.
- **httpx** — async HTTP client used in integration tests to call the FastAPI app.
- **psutil** — optional; used by `_total_ram_bytes()` in `_session.py` to adapt cache budgets to available RAM. Falls back to 8 GB if unavailable.
- **PyQt5 / PyQtWebEngine / qtpy** — Linux-only; provides the Qt backend for pywebview on Linux.

## What We Deliberately Do NOT Use

- **No React / Vue / Svelte / bundler** — the entire frontend is a single self-contained `_viewer.html` with inline CSS and JS. No npm, no build step.
- **No concurrent.futures for the render thread** — uses `threading.Thread` + `SimpleQueue` directly to avoid CPython's `_global_shutdown` executor cleanup racing with daemon threads during interpreter exit.
- **No ORM / database** — sessions are in-memory Python dicts; no SQLAlchemy, no SQLite.
- **No Redux or client-side state management library** — viewer state is plain JS variables in `_viewer.html`.
- **No CSS framework** — all styles are custom properties in `:root`; dark theme only (`#0c0c0c` background).

## Version Constraints

- Python 3.12+ required — the codebase uses `X | Y` union syntax in type hints and `match` statements.
- zarr 2.17+ — v3 API not yet adopted; chunk preset utility is tuned for v2 behavior.
- uvicorn 0.41 triggers a `DeprecationWarning` from websockets 14+ (legacy API); suppressed in `pyproject.toml` `filterwarnings`.
