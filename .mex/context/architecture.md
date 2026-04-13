---
name: architecture
description: How the major pieces of arrayview connect and flow. Load when working on system design, integrations, or understanding how components interact.
triggers:
  - "architecture"
  - "system design"
  - "how does X connect to Y"
  - "integration"
  - "flow"
  - "display routing"
  - "server mode"
edges:
  - target: context/stack.md
    condition: when specific technology details are needed
  - target: context/decisions.md
    condition: when understanding why the architecture is structured this way
  - target: patterns/vscode-display.md
    condition: when working on VS Code display routing or IPC
  - target: patterns/frontend-change.md
    condition: when working on the frontend viewer
last_updated: 2026-04-13
---

# Architecture

## System Overview

```
CLI / Python API (arrayview file.npy  OR  view(arr))
   ↓
_launcher.py — detects environment, starts server, opens display
   ├─ _platform.py    environment detection (Jupyter/VSCode/SSH/Julia/native)
   └─ _vscode.py      VS Code extension install, signal-file IPC, browser open

Server (one of two transport modes):
   ├─ Network mode: FastAPI (_server.py) + uvicorn → HTTP/WebSocket on localhost:PORT
   └─ Stdio mode:  _stdio_server.py → JSON+binary over stdin/stdout (VS Code tunnel)

Server internals (both modes):
   ├─ _session.py   Session objects, SESSIONS dict, render thread, prefetch, cache budgets
   ├─ _render.py    extract_slice → apply_complex_mode → apply_colormap_rgba → PNG
   └─ _io.py        load_data() for .npy/.npz/.nii/.zarr/.h5/.mat/.tiff/.pt

Frontend:
   └─ _viewer.html  Single ~15k-line self-contained HTML/CSS/JS file
                    ← binary RGBA frames over WebSocket / postMessage
                    → slice index, colormap, mode commands back to server
```

## Key Components

- **`_launcher.py`** (2817 lines) — Main entry point. CLI parser, `view()` API, `ViewHandle`, server lifecycle (port detection, background thread start), SSH relay, demo arrays, file watching. Imports `_server` and `uvicorn` lazily to keep CLI fast path cheap.
- **`_server.py`** (3258 lines) — FastAPI app. All REST endpoints (`/load`, `/reload`, `/seg/*`, `/resample_ras`, …) and WebSocket handler (`/ws/{sid}`). Reads from `Session` objects in `_session.SESSIONS`.
- **`_viewer.html`** (~15,600 lines) — The entire frontend. CSS dark theme + JS viewer state machine in a single file. No build step. Organized by `/* ── Section ── */` comment separators. Receives binary RGBA frames, renders to Canvas.
- **`_session.py`** — `Session` class (LRU raw/rgba/mosaic caches, preload state, vfield, spatial_meta), global state (SESSIONS dict, SERVER_LOOP, VIEWER_SOCKETS), dedicated render thread, prefetch pool.
- **`_render.py`** — `extract_slice` → `apply_complex_mode` → `apply_colormap_rgba` → `render_rgba`. Also handles `render_mosaic`, `render_rgb_rgba`, overlay compositing. Colormap LUTs built lazily via `_init_luts()`.
- **`_platform.py`** — Environment detection: `_in_jupyter()`, `_in_vscode_terminal()`, `_is_vscode_remote()`, `_in_vscode_tunnel()`, `_can_native_window()`, `_is_julia_env()`. Includes ancestor-process walk to recover env vars stripped by `uv run`.
- **`_vscode.py`** — VS Code extension auto-install (VSIX bundled in package), signal-file IPC (filename defined by `_VSCODE_SIGNAL_FILENAME`), shared-memory IPC, `EnvironmentVariableCollection` window ID (stable across uv run env stripping).
- **`_stdio_server.py`** (791 lines) — Alternative transport for VS Code direct webview. Reads JSON requests from stdin, writes binary RGBA frames to stdout. Spawned as subprocess by the VS Code extension.
- **`_io.py`** — `load_data(filepath)` dispatcher. Handles lazy vs eager loading per format. `.nii.gz` is always materialized (gzip not seekable). `.npy` uses `mmap_mode="r"`.

## Display Routing

| Environment | Display method | Server mode |
|---|---|---|
| Jupyter | Inline iframe | network |
| VS Code local terminal | Simple Browser (signal-file IPC) | network |
| VS Code tunnel (remote) | Direct webview via stdio | stdio |
| Julia | System browser | network |
| CLI / Python script (macOS/Win) | Native pywebview window | network |
| SSH (no VS Code) | Prints URL; VS Code ext tries TCP relay | network |

Detection lives in `_platform.py`. Opening logic in `_launcher.py` and `_vscode.py`.

## External Dependencies

- **FastAPI + uvicorn** — HTTP/WebSocket server. Imported lazily from `_server_mod()` in `_launcher.py` to save ~175 ms on CLI fast path (server already running).
- **nibabel** — NIfTI loading. Lazy import inside `_io._nib()`. `.nii.gz` volumes are materialized up front; `.nii` uses mmap.
- **matplotlib + qmricolors** — Colormap LUTs. Both deferred until first render via `_init_luts()`. `qmricolors` registers `lipari` and `navia` colormaps.
- **pywebview** — Native window rendering on macOS/Windows/Linux. Launched in a fresh subprocess (via `_open_webview`) to avoid multiprocessing bootstrap issues from Jupyter.
- **zarr** — Lazy array access for `.zarr` files. Used directly; no abstraction layer.

## What Does NOT Exist Here

- No database — all state is in-memory (`SESSIONS` dict) and dies with the process.
- No authentication — server binds to localhost only; no access control.
- No async rendering — render thread is a dedicated `threading.Thread` pulling from a `SimpleQueue`; deliberately not `concurrent.futures` (immune to interpreter-shutdown executor cleanup).
- No build step for the frontend — `_viewer.html` is a single file edited directly.
- No separate admin / management interface — `_config.py` reads `~/.arrayview/config.toml` directly.
