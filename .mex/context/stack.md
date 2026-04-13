---
name: stack
description: Technology stack, library choices, and the reasoning behind them. Load when working with specific technologies or making decisions about libraries and tools.
triggers:
  - "library"
  - "package"
  - "dependency"
  - "which tool"
  - "technology"
  - "format"
  - "colormap"
edges:
  - target: context/decisions.md
    condition: when the reasoning behind a tech choice is needed
  - target: context/conventions.md
    condition: when understanding how to use a technology in this codebase
  - target: context/architecture.md
    condition: when understanding how libraries fit into the overall system
last_updated: 2026-04-13
---

# Stack

## Core Technologies

- **Python 3.12+** — minimum version enforced in `pyproject.toml`; uses modern union types (`int | None`)
- **FastAPI** — REST + WebSocket server; all routes in `_server.py`
- **uvicorn** — ASGI server for FastAPI; lazy-imported in `_launcher.py`
- **numpy 2.4+** — primary array type; all rendering operates on `np.float32` / `np.uint8`
- **hatchling** — build backend; packages `src/arrayview/` including `_viewer.html` and bundled VSIX

## Key Libraries

- **FastAPI** (not Flask, not Django) — async route handlers; WebSocket support built-in
- **nibabel** — NIfTI loading only; lazy import (`_io._nib()`); not used for any other format
- **zarr 2.x** — lazy array access for `.zarr`/`.zarr.zip`; opened with `mode="r"` always
- **matplotlib** — colormap LUTs only; lazy import via `_init_luts()`; never used for rendering figures
- **qmricolors** — registers `lipari` and `navia` colormaps into matplotlib; git dependency (not PyPI)
- **pywebview** — native window; always launched in a fresh subprocess via `_open_webview()`, not in-process
- **h5py** — `.h5`/`.hdf5` and MATLAB v7.3 (`.mat`) fallback; always reads whole dataset with `[()]`
- **tifffile** — TIFF loading; no alternatives considered
- **scipy** — `.mat` v5 loading via `scipy.io.loadmat`; also used for complex structured dtype fix
- **psutil** (optional) — RAM detection for adaptive cache budgets in `_session.py`; falls back to 8 GB if absent
- **pywebview** — Linux requires `PyQt5` + `PyQtWebEngine` + `qtpy` (declared in pyproject as conditional deps)
- **pytest + playwright** — browser tests in `tests/`; marked with `@pytest.mark.browser`

## What We Deliberately Do NOT Use

- **No frontend build tooling** — `_viewer.html` is a single self-contained file; no webpack, vite, or npm
- **No Redux / state management library** — all viewer state is mutable JS variables in `_viewer.html`
- **No SQLAlchemy / databases** — state is in-memory only; `SESSIONS` dict is the sole store
- **No `concurrent.futures` for the render thread** — uses a raw `threading.Thread` + `SimpleQueue` to avoid `_global_shutdown` executor cleanup during interpreter exit
- **No multiprocessing for webview** — pywebview is always a `subprocess.Popen` to avoid bootstrap errors in Jupyter

## Version Constraints

- Python ≥ 3.12 (union type syntax `X | Y` used throughout)
- zarr 2.x API (not zarr 3.x — `zarr.open(mode="r")` syntax)
- VS Code extension version tracked in `_vscode._VSCODE_EXT_VERSION = "0.14.0"`; signal filename defined by `_vscode._VSCODE_SIGNAL_FILENAME` (currently `"open-request-v0900.json"`)
- numpy 2.x (some dtype handling relies on numpy 2 behavior)
