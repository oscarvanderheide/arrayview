# ArrayView — Repository Instructions

## Overview

**arrayview** is a fast, interactive viewer for multi-dimensional NumPy arrays (and compatible formats). It runs as a local web server (FastAPI + uvicorn) with a JavaScript frontend, displayed either in a native pywebview window or in a browser tab. In VS Code terminals, it opens in VS Code's Simple Browser panel.

## Architecture

```
CLI / Python API
    │
    ├── view()          ← Python entry point (inline IFrame in Jupyter, native window or browser otherwise)
    └── arrayview()     ← CLI entry point (`uvx arrayview file.npy`)
        │
        ├── FastAPI server (uvicorn, in-process thread or subprocess)
        │     ├── /           ← viewer HTML page
        │     ├── /shell      ← pywebview shell (tab management)
        │     ├── /ws/{sid}   ← WebSocket for real-time rendering
        │     ├── /load       ← POST to register new arrays
        │     └── /ping       ← health check
        │
        ├── pywebview subprocess (native window, default)
        └── browser fallback (--browser flag, or when pywebview unavailable)
```

### Key files

| File | Purpose |
|------|---------|
| `src/arrayview/_app.py` | Everything: server, API, rendering, CLI, platform detection |
| `src/arrayview/_viewer.html` | Frontend viewer (HTML/JS/CSS, single file) |
| `src/arrayview/_shell.html` | Shell page for pywebview (tab management) |
| `src/arrayview/__init__.py` | Package init, exports `view` and `arrayview` |
| `vscode-extension/` | VS Code extension source for Simple Browser integration |
| `src/arrayview/arrayview-opener.vsix` | Bundled extension (auto-installed on first use) |

### Server lifecycle

- **Python API (`view()`)**: Server runs in a daemon thread. In interactive sessions (REPL/Jupyter), it stays alive across multiple `view()` calls. In script mode, a non-daemon thread auto-stops after the viewer closes.
- **CLI (`arrayview`)**: Server runs in a detached subprocess (`_serve_daemon`). Exits automatically after the viewer disconnects (with an 8-second grace period for reconnects).
- **Julia (`_view_julia`)**: Always uses a subprocess server, because Julia's GIL prevents in-process threading.

## Cross-Platform Guidelines

arrayview must work across macOS, Linux, and Windows. Keep these rules in mind:

### Display / Window

| Platform | Native window | Browser fallback |
|----------|--------------|-----------------|
| macOS | pywebview (Cocoa/WebKit) — works out of the box | `webbrowser.open()` or VS Code Simple Browser |
| Linux | pywebview with Qt5 (`PyQt5` + `PyQtWebEngine`) — needs `DISPLAY` or `WAYLAND_DISPLAY` | Same |
| Windows | pywebview (EdgeChromium) — works out of the box | Same |
| VS Code remote/tunnel | No native window possible | VS Code Simple Browser via `arrayview-opener` extension |

The `_can_native_window()` function determines availability. When native windows aren't possible, the code falls back to the browser path automatically.

### VS Code Integration

- **Detection**: `_in_vscode_terminal()` checks `TERM_PROGRAM=vscode`, `VSCODE_IPC_HOOK_CLI`, and walks ancestor processes (because `uv run` strips env vars).
- **Extension auto-install**: `_ensure_vscode_extension()` uses the `code` CLI to install the bundled `.vsix`. The version is tracked by `_VSCODE_EXT_VERSION` (must match `vscode-extension/package.json`).
- **Opening URLs**: When in a VS Code terminal, `_open_browser()` routes through the extension via `code --open-url vscode://arrayview.arrayview-opener/...`. When in a regular terminal, it uses `webbrowser.open()`.
- The extension's IPC hook is essential for routing to the correct VS Code instance. `_find_vscode_ipc_hook()` recovers it from ancestor processes when env vars are stripped.

### URL format

Always use `localhost` (not `127.0.0.1`) in user-facing URLs. VS Code's tunnel port-forwarding only auto-forwards `localhost:PORT`.

## Supported Environments

| Environment | How it works |
|-------------|-------------|
| Python REPL / script | `view(array)` opens native window (or browser with `window=False`) |
| Jupyter notebook | Inline IFrame by default; `view(x, window=True)` for native window |
| VS Code integrated terminal | Native window by default; `--browser` opens in Simple Browser panel |
| VS Code tunnel / SSH remote | No native window; auto-opens in Simple Browser via extension |
| Julia (PythonCall) | Subprocess server; detects IJulia for inline IFrames |
| Regular terminal (Ghostty, Terminal.app, etc.) | Native window by default; `--browser` opens in system browser |

## Supported File Formats

`.npy`, `.npz`, `.nii`/`.nii.gz` (NIfTI), `.zarr`, `.pt`/`.pth` (PyTorch), `.h5`/`.hdf5`, `.tif`/`.tiff`, `.mat` (MATLAB)

## Development

```bash
git clone https://github.com/oscarvanderheide/arrayview
cd arrayview
uv sync --group test
uv run playwright install chromium
```

### Tests

```bash
uv run pytest tests/              # all tests (~100s)
uv run pytest tests/test_api.py   # HTTP/API layer only (~40s)
uv run pytest tests/test_browser.py  # Playwright/Chromium visual tests (~60s)
```

Visual regression baselines live in `tests/snapshots/`. Delete a snapshot file to regenerate its baseline.

### Rebuilding the VS Code extension

After modifying `vscode-extension/extension.js` or `vscode-extension/package.json`:

```bash
cd vscode-extension
# Needs `vsce` (npm install -g @vscode/vsce)
vsce package -o ../src/arrayview/arrayview-opener.vsix
```

Then bump `_VSCODE_EXT_VERSION` in `_app.py` to match the new version in `package.json`.

## Code Conventions

- Single file for core logic (`_app.py`) — keeps deployment simple (one `.py` + two `.html` templates + one `.vsix`).
- Platform detection functions (`_is_vscode_remote`, `_in_vscode_terminal`, `_can_native_window`, etc.) are centralized and used consistently.
- All subprocess spawns use explicit `stdout`/`stderr` redirection (suppress noise, keep user-facing messages).
- The render pipeline runs on a dedicated thread (`_render_worker`) to avoid blocking the asyncio event loop.
- Grace periods after viewer disconnection prevent premature server shutdown on page refresh.

## Future Work (not active)

- VS Code tunnel support is functional but can be fragile — see `TUNNEL_PLAN.md` for diagnostics.
- SSH remote support follows the same code paths as tunnel support.
