---
name: architecture
description: How the major pieces of arrayview connect and flow. Load when working on system design, integrations, or understanding how components interact.
triggers:
  - "architecture"
  - "system design"
  - "how does X connect to Y"
  - "integration"
  - "flow"
  - "data flow"
  - "display routing"
edges:
  - target: context/stack.md
    condition: when specific technology details are needed
  - target: context/decisions.md
    condition: when understanding why the architecture is structured this way
  - target: context/frontend.md
    condition: when the task involves _viewer.html, modes, reconcilers, or the View Component System
  - target: context/render-pipeline.md
    condition: when the task involves slice extraction, colormaps, caching, or the render thread
last_updated: 2026-04-24
---

# Architecture

## System Overview

```
CLI / Python API (view() or uvx arrayview <file>)
   └─ _launcher.py  →  FastAPI server (_server.py, uvicorn)   [network mode]
                    →  _stdio_server.py                        [VS Code direct webview]

Server (either mode)
   ├─ _session.py   Session objects, global state, render thread, caches
   ├─ _render.py    extract_slice → apply_complex_mode → render_rgba → PNG pipeline
   ├─ _analysis.py / _diff.py / _overlays.py / _vectorfield.py
   │                Shared helpers used by FastAPI and stdio transports
   └─ _io.py        load_data() — npy/npz/nii/zarr/h5/mat/tif/pt routing

Browser (_viewer.html — single self-contained HTML+JS+CSS file)
   ├─ WebSocket /ws/{sid}   binary PNG slices from server
   ├─ GET /meta/{sid}       session metadata on first load
   └─ Canvas + UI           colorbars, eggs, dynamic islands, mode transitions
```

A `view()` call creates a `Session`, starts the FastAPI server if not running,
registers the session via HTTP POST `/load`, then opens a display for the detected
environment (Jupyter inline, VS Code webview panel or direct webview, native
pywebview, or system browser).

## Key Components

- **`_launcher.py`** — CLI parser, `view()` API, `ViewHandle`, server lifecycle, reverse-tunnel relay (`--relay`), file watching. Heavy imports (`_session`, `_render`, `_io`, uvicorn) are all lazy to keep the CLI fast path near-zero cost.
- **`_server.py`** — FastAPI app with all REST and WebSocket routes (`/meta/{sid}`, `/load`, `/slice`, `/ws/{sid}`, `/seg/*`, `/reload`, etc.). Dispatches render work to the render thread via `_render()` from `_session.py`.
- **`_session.py`** — Single source of global mutable state: `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS`, `VIEWER_SIDS`, `SHELL_SOCKETS`. Owns the render thread (`_RENDER_QUEUE`, `_RENDER_THREAD`), prefetch pool, and the `Session` class with its three LRU caches.
- **`_render.py`** — Stateless rendering functions: `extract_slice()`, `apply_complex_mode()`, `render_rgba()`, `render_rgb_rgba()`, `render_mosaic()`, `extract_projection()`. Owns colormap LUTs (`LUTS` dict, lazy-initialized by `_init_luts()`).
- **`_analysis.py`, `_diff.py`, `_overlays.py`, `_vectorfield.py`** — Shared backend helpers for metadata, analysis endpoints, compare/diff rendering, overlay compositing, vector field validation, and arrow sampling. Imported by both `_server.py` and `_stdio_server.py`.
- **`_io.py`** — All file-format loading behind `load_data(filepath)`. Lazy nibabel import for NIfTI. Handles `.npy`, `.npz`, `.nii` and `.nii.gz`, `.zarr`, `.zarr.zip`, `.pt` and `.pth`, `.h5` and `.hdf5`, `.tif` and `.tiff`, `.mat`. Extensions registered in `_SUPPORTED_EXTS`.
- **`_platform.py`** — Environment detection: checks jupyter → vscode → julia → ssh → terminal in priority order. Results cached. Never short-circuit this order.
- **`_vscode.py`** — VS Code extension install/management, signal-file IPC, shared-memory IPC, webview panel and direct webview opening.
- **`_stdio_server.py`** — Alternative to FastAPI for VS Code tunnel (direct webview): JSON on stdin, length-prefixed binary on stdout.
- **`_viewer.html`** — The entire frontend (~24 100 lines). CSS + JS in one file, no build step. Canvas-based rendering, WebSocket binary protocol, all viewing modes, reconcilers, command registry. See `context/frontend.md`.

## Display Routing

| Environment | Default display | Server mode |
|---|---|---|
| Jupyter | Inline iframe | network |
| VS Code local | Webview panel | network |
| VS Code tunnel | Direct webview (stdio) | stdio |
| Julia | System browser | network |
| CLI / Python script | Native pywebview | network |
| SSH terminal | Prints URL — user forwards port with `ssh -L` | network |

Detection logic: `_platform.py`. Display opening: `_launcher.py` + `_vscode.py`.

## External Dependencies

- **FastAPI + uvicorn** — async HTTP/WebSocket server, lazy-imported in `_launcher.py`. Never call uvicorn directly — use the `_uvicorn()` accessor.
- **nibabel** — NIfTI file loading. Lazy-imported in `_io.py` via `_nib()`. Only loaded for `.nii` / `.nii.gz`.
- **numpy** — Core array type throughout. The only non-lazy import in the render path.
- **matplotlib** — Colormap LUT generation only. Lazy, initialized once by `_init_luts()` in `_render.py`.
- **qmricolors** — Registers the `lipari` and `navia` colormaps. Git dependency (`https://github.com/oscarvanderheide/qmricolors.git`).
- **zarr** — Lazy chunk access for `.zarr` / `.zarr.zip`. Chunk presets via `zarr_chunk_preset()` in `_session.py`.
- **pywebview** — Native OS window. Lazy, only started when `_can_native_window()` is true.

## What Does NOT Exist Here

- No persistent storage — sessions are in-memory only; nothing is written to disk by the server.
- No authentication or multi-user access control — server binds to localhost.
- No build step for the frontend — `_viewer.html` is a single static file served from package resources.
- No background job queue — heavy ops run in the render thread or prefetch pool, both owned by `_session.py`.
- No nnInteractive server — `_segmentation.py` is a pure HTTP client to a separately running nnInteractive process.
