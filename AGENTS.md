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
   +- view()          Python entry point
   +- arrayview()     CLI entry point (`uvx arrayview file.npy`)
      |
      +- FastAPI server (thread or subprocess)
         +- /           viewer HTML
         +- /shell      pywebview shell HTML
         +- /ws/{sid}   websocket for render updates
         +- /load       register arrays
         +- /ping       health check
```

## Core Files

- `src/arrayview/_app.py`: server lifecycle, platform detection, CLI/API wiring
- `src/arrayview/_viewer.html`: viewer UI
- `src/arrayview/_shell.html`: shell page for native tab/window management
- `vscode-extension/extension.js`: VS Code opener behavior
- `vscode-extension/package.json`: extension metadata and version
- `src/arrayview/arrayview-opener.vsix`: packaged extension installed by Python code

## Non-Negotiables

- Run commands directly when possible; avoid asking the user to run routine commands.
- Before trying a new fix, check whether it was already attempted in prior logs/notes.
- If re-trying a previously failed approach, explicitly note why it may work now.
- Avoid manual cleanup requirements for users. Viewer shutdown should be automatic and reliable.
- Do not regress existing working paths while fixing tunnel/remote behavior.

## Workflow For Complex Debugging


1. Start a logfile `LOG_<FEATURE>.md`.
2. Record each significant attempt:
   - hypothesis
   - change made
   - result
   - decision (keep/revert/follow-up)
3. Prefer incremental, testable changes.
4. Verify behavior in the most failure-prone environments:
   - VS Code local terminal
   - VS Code remote/tunnel
5. If behavior is not as expected, check logfile before a new attempt

## High-Risk Areas

- VS Code extension install/update detection and stale extension versions
- Recovering VS Code IPC hook when env vars are stripped (`uv run` and subprocesses)
- Deciding when to use native window vs browser vs VS Code Simple Browser
- Port-forward/autoforward behavior in tunnel environments
- Shutdown lifecycle and orphan process prevention

## Quick Validation Matrix

- `uv run arrayview data.py` (whether it runs at all)
- `uv run pytest tests/test_api.py`
- `uv run pytest tests/test_browser.py` (when UI behavior changed)
- Manual smoke checks:
  - Local CLI (`arrayview file.npy`)
  - Python `view(arr)` in script
  - Jupyter inline default
  - VS Code terminal browser path
  - Remote/tunnel path

## Platform Behavior (Must Preserve)

- Local Python/CLI: native window by default, browser fallback available
- Jupyter: inline iframe by default; explicit window mode still supported
- VS Code terminal: browser opens should target VS Code Simple Browser
- VS Code tunnel/SSH remote: should open in VS Code Simple Browser, never open UI on remote host browser by mistake
- Use `localhost` in URLs (not `127.0.0.1`) for reliable VS Code port forwarding

## VS Code Integration Notes

- `_in_vscode_terminal()` determines whether VS Code routing should be used
- `_find_vscode_ipc_hook()` may need to recover hooks from parent processes
- `_ensure_vscode_extension()` must handle install/update robustness
- `_VSCODE_EXT_VERSION` in `_app.py` must match `vscode-extension/package.json`

If extension behavior changes, rebuild the VSIX and keep versioning in sync.

## Development Commands

```bash
uv sync --group test
uv run playwright install chromium
uv run pytest tests/
uv run pytest tests/test_api.py
uv run pytest tests/test_browser.py
```

Visual snapshots are in `tests/snapshots/`.

## Rebuild VS Code Extension

```bash
cd vscode-extension
vsce package -o ../src/arrayview/arrayview-opener.vsix
```

Then update `_VSCODE_EXT_VERSION` in `src/arrayview/_app.py`.

## Source Of Truth

- End-user usage and setup: `README.md`
