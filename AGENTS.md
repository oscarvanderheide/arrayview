# ArrayView Agent Guide

This file defines how coding agents should work in this repository.

## Mission

Build and maintain a smooth `arrayview` experience across:

- Local: CLI, Python scripts, Jupyter, Julia via PythonCall
- Display modes: native `pywebview` and browser
- VS Code terminals: browser opens should prefer VS Code Simple Browser
- VS Code remote/tunnel sessions: viewer should open in the developer's VS Code client, not on the remote host browser

## Product Overview

`arrayview` is an interactive viewer for multi-dimensional arrays and medical/scientific volumes.
It runs a local FastAPI server with an HTML/JS frontend, then displays it either:

- in a native `pywebview` window,
- in a browser (including VS Code Simple Browser in VS Code terminals), or
- inline in Jupyter notebooks.

## Architecture

```
CLI / Python API
   |
   +- view()          Python entry point  (_launcher.py)
   +- arrayview()     CLI entry point (`uvx arrayview file.npy`)  (_launcher.py)
      |
      +- FastAPI server  (_server.py)
         +- /           viewer HTML
         +- /shell      pywebview shell HTML
         +- /ws/{sid}   WebSocket for render updates
         +- /load       register arrays
         +- /ping       health check
```

## Core Files

### Backend (server-side)

| File | Responsibility |
|------|---------------|
| `src/arrayview/_launcher.py` | Entry points (`view()`, `arrayview()` CLI), process management, window opening |
| `src/arrayview/_server.py` | FastAPI app, all REST routes, WebSocket handlers, HTML templates |
| `src/arrayview/_session.py` | Sessions, global state, caches, render thread, constants |
| `src/arrayview/_render.py` | Rendering pipeline: colormaps, LUTs, slice extraction, RGBA/mosaic/RGB |
| `src/arrayview/_vscode.py` | VS Code extension management, signal-file IPC, browser opening |
| `src/arrayview/_platform.py` | Platform/environment detection |
| `src/arrayview/_io.py` | Array I/O (load from file, format detection) |
| `src/arrayview/_app.py` | **Compat shim only** — re-exports from the modules above; do not add logic here |

### Frontend

| File | Responsibility |
|------|---------------|
| `src/arrayview/_viewer.html` | Viewer UI (single-file, all JS/CSS embedded) |
| `src/arrayview/_shell.html` | Shell page for native tab/window management |

### VS Code Extension

| File | Responsibility |
|------|---------------|
| `vscode-extension/extension.js` | VS Code opener behavior |
| `vscode-extension/package.json` | Extension metadata and version |
| `src/arrayview/arrayview-opener.vsix` | Packaged extension installed by Python code |

## Skills — When to Use

**Always invoke the relevant skill before touching the corresponding area.**

| Skill | Trigger |
|-------|---------|
| `viewer-ui-checklist` | ANY UI change: keyboard shortcuts, layout, new panels, canvas behavior. Keeps `visual_smoke.py` in sync. |
| `modes-consistency` | ANY visual feature: zoom, eggs, colorbars, canvas events, new rendering modes. Ensures the feature works across all six viewing modes (normal, multi-view, compare, diff, registration, qMRI). |
| `invocation-consistency` | ANY server, startup, or display-opening change. Ensures the feature works across all six invocation paths: CLI, Python script, Jupyter, Julia, VS Code tunnel, plain SSH. |
| `task-workflow` | Feature or fix tasks — enforces one-commit-per-TODO-item workflow and required collateral updates (README/help/tests/CHANGELOG). |

Skill files live in `.claude/skills/` and are symlinked from `~/.claude/skills/`.

## Non-Negotiables

- Run commands directly when possible; avoid asking the user to run routine commands.
- Before trying a new fix, check whether it was already attempted in prior logs/notes.
- If re-trying a previously failed approach, explicitly note why it may work now.
- Avoid manual cleanup requirements for users. Viewer shutdown should be automatic and reliable.
- Do not regress existing working paths while fixing tunnel/remote behavior.
- Do not add logic to `_app.py` — it is a compat shim only. Add new logic to the appropriate module.

## Testing

```bash
uv sync --group test
uv run playwright install chromium

# Fast: HTTP API only (~2s)
uv run pytest tests/test_api.py -v

# CLI entry-point tests
uv run pytest tests/test_cli.py -v

# Browser/Playwright tests (~100s)
uv run pytest tests/test_browser.py -v

# All tests
uv run pytest tests/

# Visual smoke test — run after any UI change, review screenshots
uv run python tests/visual_smoke.py
# Screenshots saved to tests/smoke_output/
```

Visual regression baselines are in `tests/snapshots/`. Delete a snapshot file to reset its baseline.

## Validation Matrix

After any change, verify the affected paths:

| What changed | Minimum checks |
|---|---|
| Server / API | `pytest tests/test_api.py` |
| CLI / entry points | `pytest tests/test_cli.py` |
| Viewer UI | `pytest tests/test_browser.py` + `python tests/visual_smoke.py` |
| VS Code / platform | Manual: VS Code local terminal, VS Code remote/tunnel |
| Large array handling | `pytest tests/test_large_arrays.py` |

Manual smoke paths (for platform/display changes):
- Local CLI: `arrayview file.npy`
- Python script: `view(arr)`
- Jupyter inline (default)
- VS Code terminal browser path
- VS Code remote/tunnel path

## Platform Behavior (Must Preserve)

- Local Python/CLI: native window by default, browser fallback available
- Jupyter: inline iframe by default; explicit window mode still supported
- VS Code terminal: browser opens should target VS Code Simple Browser
- VS Code tunnel/SSH remote: should open in VS Code Simple Browser, never open UI on remote host browser by mistake
- Use `localhost` in URLs (not `127.0.0.1`) for reliable VS Code port forwarding

## VS Code Integration

Key functions (all in `_vscode.py`):

- `_ensure_vscode_extension()` — installs/updates the VSIX; must handle stale versions robustly
- `_configure_vscode_port_preview()` — sets up port forwarding for the viewer URL
- `_open_via_signal_file()` — IPC mechanism to open URLs in the VS Code client
- `_schedule_remote_open_retries()` — retries for tunnel environments where IPC may not be immediately available

`_VSCODE_EXT_VERSION` is defined in `_vscode.py` and must match `vscode-extension/package.json`.
If extension behavior changes, rebuild the VSIX and keep versioning in sync.

## Rebuild VS Code Extension

```bash
cd vscode-extension
vsce package -o ../src/arrayview/arrayview-opener.vsix
```

Then update `_VSCODE_EXT_VERSION` in `src/arrayview/_vscode.py`.

## High-Risk Areas

- VS Code extension install/update detection and stale extension versions
- Recovering VS Code IPC hook when env vars are stripped (`uv run` and subprocesses)
- Deciding when to use native window vs browser vs VS Code Simple Browser
- Port-forward/autoforward behavior in tunnel environments
- Shutdown lifecycle and orphan process prevention

## Workflow For Complex Debugging

1. Start a logfile `LOG_<FEATURE>.md`.
2. Record each significant attempt: hypothesis → change made → result → decision (keep/revert/follow-up).
3. Prefer incremental, testable changes.
4. Verify behavior in the most failure-prone environments: VS Code local terminal, VS Code remote/tunnel.
5. If behavior is not as expected, re-read the logfile before a new attempt.

## Source Of Truth

- End-user usage and setup: `README.md`
