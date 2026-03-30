# ArrayView Agent Guide

## Mission

Interactive viewer for multi-dimensional arrays and medical/scientific volumes. Runs a FastAPI server + HTML/JS frontend displayed in a native window, browser, VS Code Simple Browser, or Jupyter inline iframe.

**Display routing defaults (sane defaults per environment):**

| Where you are | Default display |
|---|---|
| Jupyter kernel | inline iframe |
| VS Code terminal (local or tunnel) | VS Code Simple Browser tab |
| Julia | system browser |
| otherwise (CLI, Python script) | native pywebview window |

VS Code Simple Browser requires the bundled `arrayview-opener.vsix` extension. VS Code provides no public API for auto-opening Simple Browser tabs — the extension uses signal-file IPC. This is fragile; load the `vscode-simplebrowser` skill before touching it.

## Architecture

```
CLI / Python API
   |
   +- view()        Python entry  (_launcher.py)
   +- arrayview()   CLI entry     (_launcher.py)
      |
      +- FastAPI server  (_server.py)
         +- /           viewer HTML
         +- /shell      pywebview shell HTML
         +- /ws/{sid}   WebSocket render updates
         +- /load       register arrays
         +- /ping       health check
```

## Core Files

### Backend

| File | Responsibility |
|------|---------------|
| `_launcher.py` | Entry points, process management, window/browser opening |
| `_server.py` | FastAPI app, REST routes, WebSocket handlers, HTML templates |
| `_session.py` | Sessions, global state, caches, render thread, constants |
| `_render.py` | Rendering: colormaps, LUTs, slice extraction, RGBA/mosaic/RGB |
| `_vscode.py` | Extension management, signal-file IPC, browser opening |
| `_platform.py` | Platform/environment detection |
| `_io.py` | Array I/O, format detection |
| `_torch.py` | PyTorch DL integration: `view_batch()`, `TrainingMonitor` |
| `_app.py` | **Compat shim only** — re-exports; add no logic here |

### Frontend

| File | Responsibility |
|------|---------------|
| `_viewer.html` | Viewer UI (single file, all JS/CSS embedded) |
| `_shell.html` | Shell page for native tab/window management |

### VS Code Extension

| File | Responsibility |
|------|---------------|
| `vscode-extension/extension.js` | Polls signal file, opens `createWebviewPanel` |
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
- `_open_via_signal_file()` — writes signal; extension polls and opens panel
- `_schedule_remote_open_retries()` — retry writes for tunnel/first-install latency
- `_configure_vscode_port_preview()` — port forwarding settings

`_VSCODE_EXT_VERSION` in `_vscode.py` must match `vscode-extension/package.json`.
Rebuild: `cd vscode-extension && vsce package -o ../src/arrayview/arrayview-opener.vsix`

## High-Risk Areas

- Extension install: skip `--force` if correct version on disk; clean stale dirs first
- IPC hook recovery when env vars are stripped by `uv run` or subprocess wrappers
- tmux: `VSCODE_IPC_HOOK_CLI` not inherited through tmux-server; must walk client PIDs
- Signal routing: local → per-window targeted file; remote/tunnel → shared fallback
- Julia: always subprocess (GIL); never run server in-process
- Port forwarding in tunnel environments; shutdown lifecycle and orphan process prevention

## Documentation

Docs live in `docs/` (MkDocs Material site) and `README.md` (minimal quickstart).

**Style:** quiet confidence — short sentences, lead with code, imply depth rather than explain everything. No verbose prose, no "Note:", no emojis. Load the `docs-style` skill before editing.

**Keep docs in sync:** when a commit adds, changes, or removes user-facing behavior (new shortcut, changed flag, removed feature), update the relevant docs page and README before committing. Skip this for purely internal changes.

## Source of Truth

End-user usage: `README.md` and `docs/`
