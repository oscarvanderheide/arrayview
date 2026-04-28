---
name: extract-server-route-module
description: Moving an existing route cluster out of `_server.py` into a dedicated `_routes_*.py` module while preserving compat surfaces and focused validation.
triggers:
  - "extract route"
  - "route refactor"
  - "split _server.py"
  - "move routes"
edges:
  - target: context/architecture.md
    condition: when deciding whether `_server.py` should keep or delegate a route cluster
  - target: context/project-state.md
    condition: when the extraction changes the active architecture workstream
last_updated: 2026-04-29
---

# Extract Server Route Module

## Context

`_server.py` is now the assembly layer for the FastAPI app. Extract clusters that own meaningful feature behavior. Leave tiny infrastructure routes and shared dependency helpers inline unless there is real coupling pressure.

## Steps

1. Start from one concrete inline cluster in `_server.py` and add direct API coverage for its current contract before moving code.
2. Create a focused `_routes_*.py` module with `register_*_routes(app, ...)` and import dependencies directly from their source modules.
3. Preserve any compat symbols still expected at the `_server.py` surface by importing them back into `_server.py` instead of forcing wide call-site churn.
4. Register the new module from `_server.py` and remove only the extracted route block.
5. Trim `_server.py` imports after the move so the file reads like app assembly, not a half-migrated monolith.

## Gotchas

- `_app.py` is a backward-compat shim. If it re-exports a helper that moved, either update `_app.py` to import from the real module or re-export the symbol from `_server.py` intentionally.
- `register_loading_routes()` still depends on `_notify_shells`; extracting WebSocket code without preserving that callable at the server surface breaks launcher paths.
- Do not over-extract the final infrastructure surface. `/`, `/ping`, `/shell`, `/colormap/{name}`, and the GSAP asset route that serves `src/arrayview/gsap.min.js` are the natural inline end state.

## Verify

- [ ] Direct tests for the extracted cluster exist before the move
- [ ] The same focused tests pass after extraction
- [ ] `uv run python -m py_compile` passes for the touched modules
- [ ] `_server.py` still exposes any compat symbols relied on by `_app.py` or `_launcher.py`
- [ ] `.mex/context/project-state.md` and any stale architecture notes are updated if the extraction changes the intended server shape

## Update Scaffold
- [ ] Update `.mex/ROUTER.md` "Current Project State" if what's working/not built has changed
- [ ] Update any `.mex/context/` files that are now out of date
- [ ] If this is a new task type without a pattern, create one in `.mex/patterns/` and add to `INDEX.md`
