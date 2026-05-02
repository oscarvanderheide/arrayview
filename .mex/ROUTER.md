---
name: router
description: Navigation hub for task routing, project state, and behavioral guidance. Start here, then load the minimum extra context needed for the task.
edges:
  - target: context/project-state.md
    condition: when the task depends on what is currently shipped, in progress, or recently changed
  - target: context/architecture.md
    condition: when working on system design, integrations, or understanding how components connect
  - target: context/stack.md
    condition: when working with specific technologies, libraries, or making tech decisions
  - target: context/conventions.md
    condition: when writing new code, reviewing code, or unsure about project patterns
  - target: context/decisions.md
    condition: when making architectural choices or understanding why something is built a certain way
  - target: context/setup.md
    condition: when setting up the dev environment or running the project for the first time
  - target: context/frontend.md
    condition: when working on _viewer.html — modes, reconcilers, command registry, View Component System
  - target: context/render-pipeline.md
    condition: when working on rendering, colormaps, LUTs, caching, or the render thread
  - target: patterns/INDEX.md
    condition: when starting a task — check the pattern index for a matching pattern file
  - target: ../../DESIGN.md
    condition: when the task touches design philosophy — new features, UI changes, mode additions, or interaction model decisions
last_updated: 2026-05-02
---

# arrayview — Router

## Fast Path

- New task or changed task family: read this file, then load at most one pattern and one context file.
- Follow-up in the same thread and same subsystem: do **not** reopen this router. Reuse what is already loaded unless blocked.
- Small localized `_viewer.html` fix with a known symbol: go straight to exact code search. Do **not** auto-load `context/frontend.md`.
- Normal work should **not** load `.mex/SETUP.md`, `.mex/SYNC.md`, or `patterns/README.md`. Those are scaffold-maintenance files only.

## What This Is
Python package for interactively viewing multi-dimensional arrays (numpy, NIfTI, zarr, etc.) with a FastAPI backend, single-file HTML/JS frontend, and multi-environment display routing (Jupyter, VS Code, SSH, native window).

## Non-Negotiables
- Never split `_viewer.html` — entire frontend is one self-contained file, no build step
- All heavy imports (numpy, matplotlib, nibabel, FastAPI, uvicorn) must be lazy — CLI fast path stays near-zero cost
- New rendering features must be consistent across all six invocation environments
- Global state lives in `_session.py` only — `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS` never redefined elsewhere
- Render thread must remain a raw `threading.Thread` + `SimpleQueue`, not `concurrent.futures`

## Commands
- Test: run `uv run pytest` against `tests/`
- Visual smoke: run `uv run python` on `tests/visual_smoke.py`
- CLI: `uvx arrayview <file>`
- Build: `uv build`
- Drift check: `mex check --quiet`
- Auto drift hook: `mex watch`

## Context Budget

1. Start with this file plus **at most one pattern and one context file**.
2. Treat frontmatter `edges` as **optional suggestions**, not a preload list.
3. Follow **one extra edge only when blocked**. Do not recursively fan out through second-hop edges.
4. Load `context/project-state.md` only when the task depends on current shipped or in-progress work.
5. For UI, animation, or behavior bugs: ask plain-English clarification questions **before** reading git history, large diffs, or broad source sweeps.
6. For follow-up work in the same thread: prefer reusing already-loaded context over reopening router/context/pattern files.

## Common Cases

| Situation | Load |
|-----------|------|
| Follow-up in same area, no task-family change | No new `.mex` files by default |
| Small `_viewer.html` tweak, exact function/id already known | Exact `rg` + narrow code read only |
| `_viewer.html` change touching modes, reconcilers, keybind registry, or unfamiliar sections | `patterns/frontend-change.md`, then `context/frontend.md` only if still needed |
| Backend/server change in a known file family | Matching pattern first, then one context file only if blocked |
| Release validation or broad cross-mode audit | Matching pattern + relevant context file(s) deliberately |

## Current Project State

Load `context/project-state.md` only when you need active-workstream or recent-shipping detail.

## Routing Table

| Task type | Load |
|-----------|------|
| Checking current shipped / in-progress status | `context/project-state.md` |
| Understanding system architecture | `context/architecture.md` |
| Working with a specific technology | `context/stack.md` |
| Writing or reviewing code | `context/conventions.md` |
| Making a design decision | `context/decisions.md` |
| Setting up or running the project | `context/setup.md` |
| Editing `_viewer.html` (frontend) | `context/frontend.md` + `patterns/frontend-change.md` |
| Render pipeline, colormaps, caching | `context/render-pipeline.md` |
| Adding a new file format | `patterns/add-file-format.md` |
| Adding a server route / WebSocket endpoint | `patterns/add-server-endpoint.md` |
| Visual bugs / render artifacts | `patterns/debug-render.md` |
| Verifying animation changes | `patterns/animation-verify.md` |
| Design philosophy, new features, UI changes, mode additions, interaction model | `DESIGN.md` |
| Any specific task | Check `patterns/INDEX.md` for a matching pattern |

## Behavioural Contract

1. **CONTEXT** — Load from the routing table. Start small: this file + one pattern + one context file is the default cap. Do not preload unrelated docs, and do not reopen docs on routine follow-ups.
2. **CLARIFY** — If the task is about UI behavior, animation, or UX expectations, ask plain-English questions before deep investigation.
3. **BUILD** — Do the work. If deviating from an established pattern, say so before writing code.
4. **VERIFY** — If a pattern file was loaded, use its Verify section. Otherwise use the shared Verify Checklist in `context/conventions.md`.
5. **DEBUG** — If verification fails, check `patterns/INDEX.md` for one debug pattern, fix, then re-run VERIFY.
6. **GROW** — After completing the task: create/update patterns, update stale context files, and update `context/project-state.md` if significant.

## After Completing a Task

- [ ] Update `context/project-state.md` if shipped or in-progress status changed
- [ ] Update any stale `.mex/context/` or `.mex/patterns/` files touched by the task
- [ ] If this revealed a repeatable workflow, add or update a pattern in `.mex/patterns/`
