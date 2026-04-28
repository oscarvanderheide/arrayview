---
name: agents
description: Always-loaded project anchor. Read this first. Contains project identity, non-negotiables, commands, and pointer to ROUTER.md for full context.
last_updated: 2026-04-29
---

# ArrayView

## What This Is
Interactive viewer for multi-dimensional arrays with a FastAPI backend, a single-file HTML frontend, and environment-aware display routing across Jupyter, VS Code, SSH, and native windows.

## Non-Negotiables
- Never split `src/arrayview/_viewer.html`; the frontend stays in one file with no build step.
- Keep heavy imports lazy, especially in `_launcher.py`.
- Do not add new logic to `src/arrayview/_app.py`; it is a compat shim only.
- UI visibility changes must flow through the reconcilers, not ad hoc `style.display` or `classList` toggles.
- Keybind changes must update both the command registry and `GUIDE_TABS`.
- Use `localhost`, not `127.0.0.1`, in project changes and test helpers.

## Commands
- Dev: `uv run arrayview <file>`
- Test: `uv run pytest tests/<target>`
- Visual smoke: `uv run python tests/visual_smoke.py`
- Build: `uv build`
- Drift check: `mex check --quiet`
- Resync: `.mex/sync.sh`

## After Every Task
After completing any task: update `.mex/ROUTER.md` project state and any `.mex/` files that are now out of date. If no pattern existed for the task you just completed, create one in `.mex/patterns/`.

## Navigation
At the start of every session, read `.mex/ROUTER.md` before doing anything else.
For full project context, patterns, and task guidance — everything is there.
