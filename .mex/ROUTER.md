---
name: router
description: Navigation hub for task routing, project state, and behavioural guidance. Consult it when planning work or loading task-specific context.
edges:
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
  - target: patterns/INDEX.md
    condition: when starting a task — check the pattern index for a matching pattern file
last_updated: 2026-04-14
---

# arrayview — Router

## What This Is
Python package for interactively viewing multi-dimensional arrays (numpy, NIfTI, zarr, etc.) with a FastAPI backend, single-file HTML/JS frontend, and multi-environment display routing (Jupyter, VS Code, SSH, native window).

## Non-Negotiables
- Never split `_viewer.html` — entire frontend is one self-contained file, no build step
- All heavy imports (numpy, matplotlib, nibabel, FastAPI, uvicorn) must be lazy — CLI fast path stays near-zero cost
- New rendering features must be consistent across all six invocation environments
- Global state lives in `_session.py` only — `SESSIONS`, `SERVER_LOOP`, `VIEWER_SOCKETS` never redefined elsewhere
- Render thread must remain a raw `threading.Thread` + `SimpleQueue`, not `concurrent.futures`

## Commands
- Test: `uv run pytest tests/`
- Visual smoke: `uv run pytest tests/visual_smoke.py`
- CLI: `uvx arrayview <file>`
- Build: `uv build`

## Current Project State

**Working:**
- CLI (`uvx arrayview file.npy`) and Python API (`view(arr)`) — both stable
- All six display environments: Jupyter inline, VS Code local, VS Code tunnel (stdio), Julia, native pywebview, SSH URL print
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- Rendering pipeline: colormaps, complex modes, mosaic, RGB/RGBA, projections, overlays
- NIfTI spatial metadata, RAS resampling
- VS Code extension v0.14.3 — stable window ID via `EnvironmentVariableCollection`; `arrayview.openInFloatingWindow` setting moves new tabs to a floating window; `view(arr, floating=True)` and `arrayview file.npy --floating` open in a floating window per-call regardless of global setting; `!vscode.env.remoteName` guard removed (remote VS Code supports floating windows)
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)
- Colorbar island flip: `c` and `d` keys trigger 3D `rotateX` card flip (front=colorbar, back=cmap thumbnails/histogram)
- Cold-start loading spinner in VS Code and native shell

**In progress:**
- Smooth immersive transition (`feat/immersive-animation-redesign`) — single-view pinch scrub now detaches title/dimbar/shared colorbar at scrub start, preserves frame-1 viewport position, restores scrub-start positions on reverse, and no longer supports immersive island dragging; cross-mode parity still pending

**Not yet built:**
- Independent split view for mismatched-shape arrays (designed, shelved)
- Admin/config UI (file-based `~/.arrayview/config.toml` only)

## Routing Table

| Task type | Load |
|-----------|------|
| Understanding system architecture | `context/architecture.md` |
| Working with a specific technology | `context/stack.md` |
| Writing or reviewing code | `context/conventions.md` |
| Making a design decision | `context/decisions.md` |
| Setting up or running the project | `context/setup.md` |
| Editing `_viewer.html` (frontend) | `patterns/frontend-change.md` |
| Adding a new file format | `patterns/add-file-format.md` |
| VS Code display / IPC issues | `patterns/vscode-display.md` |
| Visual bugs / render artifacts | `patterns/debug-render.md` |
| Any specific task | Check `patterns/INDEX.md` for a matching pattern |

## Behavioural Contract

1. **CONTEXT** — Load relevant context file(s) from the routing table. Check `patterns/INDEX.md` for a matching pattern.
  Do not preload unrelated context files.
2. **BUILD** — Do the work. If deviating from an established pattern, say so before writing code.
3. **VERIFY** — If a pattern file was loaded, use its own Verify section. Otherwise run this checklist:
   - [ ] New heavy imports are lazy (`_mod = None` / accessor function pattern)
   - [ ] New `Session` field initialized in `__init__` and cleared in `reset_caches()` if cache-related
   - [ ] New file format goes through `_io.load_data()`, extension added to `_SUPPORTED_EXTS`
   - [ ] Frontend changes in `_viewer.html` only — no new JS/CSS files
   - [ ] New rendering functions follow `extract_slice → apply_complex_mode → apply_colormap_rgba` order
   - [ ] Environment detection changes go in `_platform.py` only
   - [ ] Cross-mode consistency verified across all six invocation environments
4. **DEBUG** — If verification fails, check `patterns/INDEX.md` for a debug pattern, fix, re-run VERIFY.
5. **GROW** — After completing the task: create/update patterns, update stale context files, update "Current Project State" above if significant.
