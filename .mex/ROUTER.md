---
name: router
description: Session bootstrap and navigation hub. Read at the start of every session before any task. Contains project state, routing table, and behavioural contract.
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
last_updated: 2026-04-13
---

# Session Bootstrap

If you haven't already read `AGENTS.md`, read it now — it contains the project identity, non-negotiables, and commands.

Then read this file fully before doing anything else in this session.

## Current Project State

**Working:**
- CLI (`uvx arrayview file.npy`) and Python API (`view(arr)`) — both stable
- All six display environments: Jupyter inline, VS Code local (Simple Browser), VS Code tunnel (stdio), Julia, native pywebview, SSH URL print
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- Rendering pipeline: colormaps, complex modes (mag/phase/real/imag), mosaic, RGB/RGBA, projections (max/min/mean/std/sos/sum)
- Overlays: binary mask, multi-label segmentation, float heatmap with nnInteractive integration
- NIfTI spatial metadata display, RAS resampling
- VS Code extension v0.14.0 — stable window ID via `EnvironmentVariableCollection`
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)

**Not yet built:**
- Independent split view for mismatched-shape arrays (designed, shelved — see `project_independent_split_view.md` memory)
- Smooth immersive transition during continuous trackpad zoom (deferred — see memory)
- Admin/config UI — config is file-based (`~/.arrayview/config.toml`) only

**Known issues:**
- White flicker on macOS native window before loading animation appears (minor, tracked in `dev/TODO.md`)
- ColorBar class refactor is partially complete — some colorbars still use legacy inline code

## Routing Table

Load the relevant file based on the current task. Always load `context/architecture.md` first if not already in context this session.

| Task type | Load |
|-----------|------|
| Understanding how the system works | `context/architecture.md` |
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

For every task, follow this loop:

1. **CONTEXT** — Load the relevant context file(s) from the routing table above. Check `patterns/INDEX.md` for a matching pattern. If one exists, follow it. Narrate what you load: "Loading architecture context..."
2. **BUILD** — Do the work. If a pattern exists, follow its Steps. If you are about to deviate from an established pattern, say so before writing any code — state the deviation and why.
3. **VERIFY** — Load `context/conventions.md` and run the Verify Checklist item by item. State each item and whether the output passes. Do not summarise — enumerate explicitly.
4. **DEBUG** — If verification fails or something breaks, check `patterns/INDEX.md` for a debug pattern. Follow it. Fix the issue and re-run VERIFY.
5. **GROW** — After completing the task:
   - If no pattern exists for this task type, create one in `patterns/` using the format in `patterns/README.md`. Add it to `patterns/INDEX.md`. Flag it: "Created `patterns/<name>.md` from this session."
   - If a pattern exists but you deviated from it or discovered a new gotcha, update it with what you learned.
   - If any `context/` file is now out of date because of this work, update it surgically — do not rewrite entire files.
   - Update the "Current Project State" section above if the work was significant.
