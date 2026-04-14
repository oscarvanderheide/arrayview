---
name: project-state
description: Current working, in-progress, and shelved status. Load only when the task depends on active workstreams or recent shipped behavior.
triggers:
  - "current state"
  - "what's in progress"
  - "recent work"
  - "active feature"
  - "shipped recently"
last_updated: 2026-04-15
---

# Project State

## Working

- CLI (`uvx arrayview file.npy`) and Python API (`view(arr)`) — both stable
- All six display environments: Jupyter inline, VS Code local, VS Code tunnel (stdio), Julia, native pywebview, SSH URL print
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- Rendering pipeline: colormaps, complex modes, mosaic, RGB/RGBA, projections, overlays
- NIfTI spatial metadata, RAS resampling
- VS Code extension v0.14.5 — stable window ID via `EnvironmentVariableCollection`; `arrayview.openInFloatingWindow` setting moves new tabs to a floating window; `view(arr, floating=True)` and `arrayview file.npy --floating` open in a floating window per-call regardless of global setting; `!vscode.env.remoteName` guard removed (remote VS Code supports floating windows); floating mode now uses a single persistent shell hub panel (`_shell.html`) so all arrays share one floating window as tabs instead of opening separate windows; fixed: second CLI call now injects tab via `new_tab` postMessage relay (extension -> hub wrapper -> shell iframe) instead of relying on WebSocket notify which wasn't sent in VS Code mode
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)
- Colorbar island flip: `c` and `d` keys trigger 3D `rotateX` card flip (front=colorbar, back=cmap thumbnails/histogram)
- Cold-start loading spinner in VS Code and native shell

## In Progress

- Smooth immersive transition — dimbar scrub geometry is now preserved explicitly and cleared before immersive overlay positioning so settle/exit no longer reuse stale pixel `left/top` values; immersive overlay positioning now waits one animation frame after the class switch before fading back in; shared slim colorbar returns to its normal slot through `drawSlimColorbar()` on reverse instead of being pixel-pinned during scrub. Trackpad/pinch feel and cross-mode parity still need manual validation.

## Not Yet Built

- Independent split view for mismatched-shape arrays (designed, shelved)
- Admin/config UI (file-based `~/.arrayview/config.toml` only)
