---
name: router
description: Entry point for task routing, project non-negotiables, and behavioral guardrails. Start here, then load only the minimum extra context a task needs.
last_updated: 2026-07-22
---

# arrayview — Router

Python package for interactively viewing multi-dimensional arrays (numpy, NIfTI, zarr, …) with a FastAPI backend, single-file HTML/JS frontend, and multi-environment display routing (Jupyter, VS Code, SSH, native window).

## Non-Negotiables

- Never split `_viewer.html` — entire frontend is one self-contained file, no build step.
- All heavy imports (numpy, matplotlib, nibabel, FastAPI, uvicorn) must be lazy — CLI fast path stays near-zero cost.
- New rendering features must be consistent across every affected invocation
  row; local VS Code, Remote SSH, and VS Code tunnel are separate rows.
- Global state lives in `_session.py` only — `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS` never redefined elsewhere.
- Render thread must remain a raw `threading.Thread` + `SimpleQueue`, not `concurrent.futures`.

## Context Budget

- Start with this file plus **one context file and/or one pattern file**, loaded only when the task actually needs them.
- Do **not** reopen this router or already-loaded docs on routine follow-ups in the same thread. Reuse what's loaded unless blocked.
- When blocked, load **one more** file — do not recursively fan out through second-hop references.
- For `_viewer.html` follow-ups with a known symbol: go straight to an exact `rg` search and a narrow code read. Do **not** auto-load `context/frontend.md`.
- For UI behavior, animation, or UX-expectation tasks: ask a plain-English clarification question **before** reading git history, large diffs, or broad source sweeps.
- Normal work never loads `.mex/SETUP.md`, `.mex/SYNC.md`, or `patterns/README.md` — those are scaffold-maintenance files.

## Where to Look

Load only the file(s) matching the task. One is usually enough.

| Task | File |
|------|------|
| Current shipped / in-progress status | `context/project-state.md` |
| System architecture, component connections | `context/architecture.md` |
| Startup, shutdown, display ownership, orphans, VS Code tabs, session release | `context/lifecycle.md` |
| Proving a startup/display fix in real use | `patterns/validate-launch-path.md` |
| Diagnosing VS Code local/remote/tunnel delivery | `patterns/debug-vscode-extension-python.md` |
| Specific technology / library choices | `context/stack.md` |
| Code patterns when writing or reviewing | `context/conventions.md` |
| Why something is built the way it is / a new architectural choice | `context/decisions.md` |
| First-time dev environment setup | `context/setup.md` |
| `_viewer.html` — modes, reconcilers, command registry, View Component System | `context/frontend.md` (+ `patterns/frontend-change.md` if changing code) |
| Rendering, colormaps, LUTs, caching, render thread | `context/render-pipeline.md` |
| Any task — scan for a matching pattern first | `patterns/INDEX.md` |
| Adding a file format | `patterns/add-file-format.md` |
| Adding a server route / WebSocket endpoint | `patterns/add-server-endpoint.md` |
| Visual bugs / render artifacts | `patterns/debug-render.md` |
| Verifying animation changes (frame captures) | `patterns/animation-verify.md` |
| Design philosophy, new features, mode additions, interaction model | `../DESIGN.md` |

## Commands

- Test: run targeted `uv run pytest tests/<target>` checks
- Visual smoke: run `uv run python` on `tests/visual_smoke.py`
- Working-tree CLI: `uv run arrayview <file>`
- Installed-package CLI: `uvx arrayview <file>`
- Build: `uv build`
- Drift check: `mex check --quiet`

## After Completing a Task

- [ ] For startup/display work, attach the affected real public launch evidence;
      a green test suite with `MANUAL` host rows still open is not completion.
- [ ] Update `context/project-state.md` if shipped or in-progress status changed.
- [ ] Update any stale `.mex/context/` or `.mex/patterns/` files touched by the task.
- [ ] If this revealed a repeatable workflow, add or update a pattern in `.mex/patterns/`.
