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
edges:
  - target: context/stack.md
    condition: when specific technology versions or library details are needed
  - target: context/architecture.md
    condition: when understanding how components connect during setup
last_updated: 2026-04-15
---

# Setup

## Prerequisites

- **Python 3.12+** — the minimum supported version; union type hints and match statements are used throughout
- **uv** — package manager and task runner; install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js / npm** — NOT required; there is no frontend build step

## First-time Setup

1. Clone the repo
2. `uv sync --all-groups` — installs all dependencies including dev and test groups
3. `uv run arrayview dev/sample.npy` — smoke test: should open the viewer with a sample array
4. `uv run pytest tests/test_mode_consistency.py` — verify core render consistency passes

For browser-based tests (playwright):
- `uv run playwright install chromium` — one-time browser install

## Environment Variables

- `ARRAYVIEW_RAW_CACHE_MB` (optional) — override raw slice cache budget in MB (default: 5% RAM)
- `ARRAYVIEW_RGBA_CACHE_MB` (optional) — override RGBA tile cache budget in MB (default: 10% RAM)
- `ARRAYVIEW_MOSAIC_CACHE_MB` (optional) — override mosaic cache budget in MB (default: 2.5% RAM)
- `ARRAYVIEW_HEAVY_OP_LIMIT_MB` (optional) — max array size in MB for heavy ops like FFT (default: 5000)

No `.env` file is needed. All env vars are optional overrides; the server runs with sensible adaptive defaults computed from available RAM via `psutil`.

## Common Commands

- `uv run arrayview <file>` — launch viewer on a file (CLI entry point)
- `uvx arrayview <file>` — launch from anywhere without activating the venv
- `uv run pytest tests/<target>` — run a specific test file
- `uv run pytest tests/test_mode_consistency.py` — mode consistency suite (run after render changes)
- `uv run pytest tests/visual_smoke.py` — browser smoke tests (requires playwright)
- `uv run pytest -m "not browser"` — all non-browser tests
- `uv build` — build wheel + sdist in `dist/`

## Common Issues

**Viewer opens then immediately closes:** The pywebview process exits when the Python process exits. If calling `view()` from a script (not the REPL), add `input()` or a loop at the end to keep the process alive, or use `view(arr, block=True)`.

**Port already in use:** The server binds to a random free port by default. If you see a port conflict, check for orphan `uvicorn` processes: `lsof -i :<port>`. Kill with `kill -9 <PID>`. The server is supposed to shut down automatically when all viewer windows close.

**VS Code extension not loading:** Signal-file IPC can fail if the extension version is mismatched. Check `_VSCODE_EXT_VERSION` in `_vscode.py` matches the installed extension. See `.claude/skills/vscode-simplebrowser/SKILL.md` for the full diagnosis flow.

**Playwright tests fail on CI:** Ensure `uv run playwright install chromium` has been run. Browser tests require a display; on headless servers they need `DISPLAY=:0` or Xvfb.

**zarr import errors:** If using zarr v3, note the project is pinned to zarr 2.x API. Do not upgrade zarr past the constraint without auditing `_io.py`.
