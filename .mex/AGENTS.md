---
name: agents
description: Always-loaded project anchor. Read this first. Contains project identity, non-negotiables, commands, and pointer to ROUTER.md for full context.
last_updated: 2026-04-13
---

# arrayview

## What This Is
A Python package for interactively viewing multi-dimensional arrays (numpy, NIfTI, zarr, DICOM, etc.) with a FastAPI backend, a single-file HTML/JS frontend, and multi-environment display routing (Jupyter, VS Code, SSH, native window).

## Non-Negotiables

- Never split `_viewer.html` into separate files — the entire frontend is one self-contained file, no build step
- All heavy imports (numpy, matplotlib, nibabel, FastAPI, uvicorn) must be lazy — CLI fast path must stay near-zero cost
- New rendering features must be consistent across all six invocation environments (CLI, Jupyter, Julia, VS Code local, VS Code tunnel, SSH)
- Global state lives in `_session.py` only — `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS` are never redefined elsewhere
- Render thread must remain a raw `threading.Thread` + `SimpleQueue`, not `concurrent.futures`

## Commands

- Test: `uv run pytest tests/`
- Visual smoke: `uv run pytest tests/visual_smoke.py`
- CLI: `uvx arrayview path/to/file.npy`
- Build: `uv build`
- Docs: `uv run mkdocs serve`

## Scaffold Growth
After every task: if no pattern exists for the task type you just completed, create one. If a pattern or context file is now out of date, update it. The scaffold grows from real work, not just setup. See the GROW step in `ROUTER.md` for details.

## Navigation
At the start of every session, read `ROUTER.md` before doing anything else.
For full project context, patterns, and task guidance — everything is there.
