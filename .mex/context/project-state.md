---
name: project-state
description: Current working, in-progress, and shelved status. Load only when the task depends on active workstreams or recent shipped behavior.
triggers:
  - "current state"
  - "what's in progress"
  - "recent work"
  - "active feature"
  - "shipped recently"
last_updated: 2026-04-16
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
- Plugin shelf (`/` menu) supports multi-select: spacebar toggles plugins, Enter applies selection. Mutual exclusion enforced (ROI ↔ Segmentation). Cursor indicator shows focused tile via yellow background + left accent bar
- Dynamic island renders sections for all active plugins simultaneously (qMRI pills + ROI shapes/stats separated by divider), replacing the old single-plugin priority chain
- ROI mode works alongside qMRI: drawing on any pane mirrors the ROI to all panes in real-time via per-pane overlay canvases; stats fetched per parameter map and shown as sub-rows in the island
- qMRI map toggle (`_islandToggleQmriMap`) fade animation now covers dimbar and array-name in addition to panes and colorbars

## In Progress

- Smooth immersive transition — stale scrub geometry handoff is fixed, immersive overlay fade-in is held until after the class switch, shared slim colorbar returns through `drawSlimColorbar()` on reverse, and active scrub suppresses minimap/overflow/drag side effects. Single-pane scrub now targets the actual centered immersive viewport rect instead of a hardcoded corner box, the dimbar stays above the pane during scrub, the shared colorbar sits behind the growing pane, and the phantom extra `av-view-wrap` footprint in normal mode was removed by rebinding `NormalLayout` to the real `#viewer` canvas. Cross-mode parity and deeper reverse-pinch validation still need manual verification.
- ROI + qMRI integration refinements: floodfill not yet supported on qMRI panes; ROI hover tooltip not yet wired for qMRI canvases; per-pane stats are re-fetched on each ROI draw but not updated on slice scroll

## Not Yet Built

- Independent split view for mismatched-shape arrays (designed, shelved)
- Admin/config UI (file-based `~/.arrayview/config.toml` only)
