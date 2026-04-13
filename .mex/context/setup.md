---
name: setup
description: Dev environment setup and commands for arrayview. Load when setting up the project or when environment issues arise.
triggers:
  - "setup"
  - "install"
  - "environment"
  - "getting started"
  - "how do I run"
  - "local development"
  - "test"
  - "build"
edges:
  - target: context/stack.md
    condition: when specific technology versions or library details are needed
  - target: context/architecture.md
    condition: when understanding how components connect during setup
  - target: patterns/debug-render.md
    condition: when visual tests fail during setup verification
last_updated: 2026-04-13
---

# Setup

## Prerequisites

- **Python 3.12+** — enforced in `pyproject.toml`
- **uv** — package manager used throughout (all commands below use `uv`)
- **Node.js** — only needed if modifying the VS Code extension (`vscode-extension/`)
- **Playwright browsers** — required for browser tests: `uv run playwright install chromium`

## First-time Setup

1. `uv sync --all-groups` — installs all deps including dev/test/docs groups
2. `uv run pytest tests/test_view_component_unit.py` — verify unit tests pass
3. `uvx arrayview` — verify CLI works (shows usage / opens demo)
4. `uv run python -c "from arrayview import view; import numpy as np; view(np.random.rand(64,64))"` — verify `view()` API

## Environment Variables

- `ARRAYVIEW_HEAVY_OP_LIMIT_MB` (optional) — override heavy-op guard threshold (default: 5000 MB)
- `ARRAYVIEW_RAW_CACHE_MB` (optional) — override raw slice cache budget (default: 5% RAM)
- `ARRAYVIEW_RGBA_CACHE_MB` (optional) — override RGBA cache budget (default: 10% RAM)
- `ARRAYVIEW_MOSAIC_CACHE_MB` (optional) — override mosaic cache budget (default: 2.5% RAM)
- `ARRAYVIEW_WINDOW_ID` — set by VS Code extension via `EnvironmentVariableCollection`; do not set manually

## Common Commands

- `uv run pytest tests/` — run full test suite (excludes browser tests by default)
- `uv run pytest tests/ -m browser` — run browser/playwright tests only
- `uv run pytest tests/visual_smoke.py` — run visual smoke tests
- `uv run pytest tests/test_mode_consistency.py` — run cross-mode consistency tests
- `uvx arrayview <file>` — run CLI on a file
- `uv run python -m arrayview <file>` — alternate CLI invocation
- `uv build` — build wheel (output to `dist/`)
- `uv run mkdocs serve` — preview docs locally

## Common Issues

**VS Code extension not installing:** Check `_vscode._VSCODE_EXT_VERSION` matches the VSIX bundled at `src/arrayview/arrayview-opener.vsix`. The extension auto-installs on first `view()` call in a VS Code terminal. Force reinstall by deleting `~/.vscode/extensions/arrayview.*`.

**`uv run` strips env vars:** `VSCODE_IPC_HOOK_CLI` is stripped by uv. `_platform._find_vscode_ipc_hook()` walks the ancestor process tree to recover it. If detection fails, set `TERM_PROGRAM=vscode` manually to force VS Code display routing.

**pywebview window does not open:** On Linux, requires `DISPLAY` or `WAYLAND_DISPLAY` + PyQt5 + PyQtWebEngine. Check `_can_native_window()` returns True. On macOS/Windows this should always work.

**Viewer shows but is blank:** Server may not have started yet. Check port binding by looking for "Uvicorn running on" in process output. The webview polls the port for up to 30 seconds.

**Test failures in mode_consistency:** Run `uv run pytest tests/test_mode_consistency.py -v` and check which mode fails. Usually a canvas rendering difference — see `patterns/debug-render.md`.
