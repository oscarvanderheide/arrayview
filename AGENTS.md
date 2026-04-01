# ArrayView Agent Guide

## Mission

Interactive viewer for multi-dimensional arrays and medical/scientific volumes. Runs a FastAPI server + HTML/JS frontend displayed in a native window, browser, VS Code Simple Browser, or Jupyter inline iframe.

**Display routing defaults (sane defaults per environment):**

| Where you are | Default display |
|---|---|
| Jupyter kernel | inline iframe |
| VS Code terminal (local) | VS Code Simple Browser tab (network mode) |
| VS Code terminal (tunnel) | VS Code direct webview tab (stdio mode) |
| Julia | system browser |
| otherwise (CLI, Python script) | native pywebview window |

VS Code Simple Browser requires the bundled `arrayview-opener.vsix` extension. VS Code
provides no public API for auto-opening Simple Browser tabs — the extension uses signal-file
IPC. This is fragile; load the `vscode-simplebrowser` skill before touching it.

When debugging CSS/layout issues, identify the root cause before applying fixes. Common pitfalls: canvas buffer vs CSS resolution mismatches, CSS selectors targeting wrong wrapper elements, av-loading class interactions, global declaration ordering.

Key modes: normal, immersive, compare, diff, multiview (3-pane oblique). Key UI elements: dynamic islands (dimbar, colorbar, ROI, segment), colorbars, histograms, minimap. Interaction modes: ROI, SEGMENT. Always check all code paths including tunnel/remote mode and direct webview mode.

For visual/animation features, propose 2-3 options with brief descriptions BEFORE implementing. User has strong aesthetic preferences and has reverted entire sessions of work that didn't match their taste.

## Architecture

```
CLI / Python API
   |
   +- view()          Python entry  (_launcher.py)
   +- arrayview()     CLI entry     (_launcher.py)
   +- python -m arrayview           (__main__.py)
      |
      +- FastAPI server  (_server.py)        ← network mode (default)
      |  +- /           viewer HTML
      |  +- /shell      pywebview shell HTML
      |  +- /ws/{sid}   WebSocket render updates
      |  +- /load       register arrays
      |  +- /seg/*      segmentation endpoints
      |  +- /ping       health check
      |
      +- Stdio server   (_stdio_server.py)   ← direct webview mode (--mode stdio)
         +- stdin/stdout JSON+binary protocol
         +- No network, no WebSocket — VS Code extension spawns subprocess
         +- PythonBridge in extension.js bridges postMessage ↔ stdin/stdout
```

## Core Files

### Backend

| File | Responsibility |
|------|---------------|
| `_launcher.py` | Entry points, process management, window/browser opening |
| `_server.py` | FastAPI app, REST routes, WebSocket handlers, HTML templates |
| `_session.py` | Sessions, global state, caches, render thread, constants |
| `_render.py` | Rendering: colormaps, LUTs, slice extraction, RGBA/mosaic/RGB |
| `_stdio_server.py` | Stdio server for direct webview mode — mirrors `_server.py` render logic over stdin/stdout |
| `_vscode.py` | Extension management, signal-file IPC, direct webview IPC, browser opening |
| `_segmentation.py` | nnInteractive segmentation client — HTTP client, auto-launch, label management |
| `_config.py` | Persistent user preferences via `~/.arrayview/config.toml` |
| `_platform.py` | Platform/environment detection |
| `_io.py` | Array I/O, format detection |
| `_torch.py` | PyTorch DL integration: `view_batch()`, `TrainingMonitor` |
| `__main__.py` | `python -m arrayview` entry point |
| `_app.py` | **Compat shim only** — re-exports; add no logic here |

### Frontend

| File | Responsibility |
|------|---------------|
| `_viewer.html` | Viewer UI (single file, all JS/CSS embedded) |
| `_shell.html` | Shell page for native tab/window management |

### VS Code Extension

| File | Responsibility |
|------|---------------|
| `vscode-extension/extension.js` | Polls signal file, opens `createWebviewPanel`, PythonBridge for direct webview mode |
| `vscode-extension/package.json` | Extension metadata and version |
| `arrayview-opener.vsix` | Packaged extension (auto-installed by Python) |

## Skills

Load the relevant skill before touching the corresponding area.

| Skill | Load when... |
|-------|-------------|
| `vscode-simplebrowser` | Touching extension install, signal-file IPC, `_ensure_vscode_extension()`, `_VSCODE_EXT_VERSION`, or `vscode-extension/` |
| `invocation-consistency` | Any server startup, display-opening, or environment-detection change |
| `ui-consistency-audit` | Any visual feature: zoom, colorbars, canvas events, rendering modes. Or UI changes: keyboard shortcuts, layout, new panels |
| `frontend-designer` | Any styling/layout change to `_viewer.html` |
| `docs-style` | Updating README, in-app help overlay, or docstrings |
| `task-workflow` | Feature or fix tasks |

## Execution Preferences

When implementing plans, always use **subagent-driven development** (`superpowers:subagent-driven-development`) — not inline execution.

## Eggs (Mode Badges)

Eggs are small pill-shaped badges below the canvas showing active visualization state. They are **composable transforms** — each one modifies how the underlying data is displayed, and they stack naturally:

- `FFT` → show frequency domain
- `LOG` → apply log scale
- `MAGNITUDE` / `PHASE` / `REAL` / `IMAG` → complex component
- `RGB` → interpret channels as color
- `ALPHA` → overlay blending level
- `PROJECTION` → reduce along an axis (MAX/MIN/MEAN/STD/SOS)

Stacking makes sense: "LOG of MAGNITUDE of FFT of data" is a meaningful composition.

**ROI and SEGMENT do not belong in this model.** They are **interaction modes** — they take over canvas input (clicks, drags) rather than transforming displayed data. They are mutually incompatible (both consume the overlay canvas and pointer events). Each gets its own dynamic island for controls. Treat them as a separate UI concept from eggs.

- `R` key → ROI mode (shape tools: rectangle, circle, freehand, polygon, floodfill)
- `S` key → SEGMENT mode (nnInteractive: click, bbox, scribble, lasso, freehand; label management)

## Non-Negotiables

- Always use `localhost` (not `127.0.0.1`) — required for VS Code port forwarding
- Never `--force` reinstall the extension if the correct version is already on disk
- Do not add logic to `_app.py` — compat shim only
- Avoid orphan processes; shutdown must be automatic
- Do not regress working display paths when fixing another

## Testing

```bash
uv sync --group test && uv run playwright install chromium

uv run pytest tests/test_api.py -v       # HTTP API (~2s)
uv run pytest tests/test_cli.py -v       # CLI entry points
uv run pytest tests/test_browser.py -v   # Playwright (~100s)
uv run pytest tests/                     # all

uv run python tests/visual_smoke.py      # screenshots → tests/smoke_output/
```

| What changed | Run |
|---|---|
| Server / API | `test_api.py` |
| CLI | `test_cli.py` |
| Viewer UI | `test_browser.py` + `visual_smoke.py` |
| VS Code / platform | manual: VS Code local terminal, VS Code tunnel |

**Visual consistency:** After any UI change, use the `/ui-consistency-audit` skill to verify the feature works consistently across all viewing modes (normal, multi-view, compare, immersive, inline). This replaces reference screenshot comparisons — the UI changes frequently, so consistency across modes matters more than pixel-matching baselines.

## VS Code Integration

All key functions live in `_vscode.py`:
- `_ensure_vscode_extension()` — installs VSIX only if version not already on disk
- `_open_via_signal_file()` — writes signal; extension polls and opens panel (network mode)
- `_open_direct_via_signal_file()` — writes signal for direct webview mode (stdio, no network)
- `_open_direct_via_shm()` — shared memory IPC for zero-copy numpy arrays
- `_schedule_remote_open_retries()` — retry writes for tunnel/first-install latency
- `_configure_vscode_port_preview()` — port forwarding settings

**Two transport modes:**
- **Network mode** (default): FastAPI server + WebSocket, extension opens Simple Browser tab
- **Direct webview mode** (`--mode stdio`): extension spawns subprocess, PythonBridge bridges postMessage ↔ stdin/stdout — no network, used automatically in tunnel sessions

`_VSCODE_EXT_VERSION` in `_vscode.py` must match `vscode-extension/package.json`.
Rebuild: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`

## High-Risk Areas

- Extension install: skip `--force` if correct version on disk; clean stale dirs first
- IPC hook recovery when env vars are stripped by `uv run` or subprocess wrappers
- tmux: `VSCODE_IPC_HOOK_CLI` not inherited through tmux-server; must walk client PIDs
- Signal routing: local → per-window targeted file; remote/tunnel → shared fallback
- Direct webview IPC: stdin/stdout binary protocol, subprocess lifecycle, PythonBridge error handling
- Segmentation server lifecycle: auto-launch vs external server, cleanup on disconnect
- Shared memory IPC: POSIX SHM with resource tracker cleanup (`_open_direct_via_shm`)
- Dynamic islands in all modes: must verify appearance consistency across normal/immersive/multiview
- Julia: always subprocess (GIL); never run server in-process
- Port forwarding in tunnel environments; shutdown lifecycle and orphan process prevention

## Documentation

Docs live in `docs/` (MkDocs Material site) and `README.md` (minimal quickstart).

**Style:** quiet confidence — short sentences, lead with code, imply depth rather than explain everything. No verbose prose, no "Note:", no emojis. Load the `docs-style` skill before editing.

**Keep docs in sync:** when a commit adds, changes, or removes user-facing behavior (new shortcut, changed flag, removed feature), update the relevant docs page and README before committing. Skip this for purely internal changes.

## Source of Truth

End-user usage: `README.md` and `docs/`
