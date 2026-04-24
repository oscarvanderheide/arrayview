---
name: project-state
description: Current working, in-progress, and shelved status. Load only when the task depends on active workstreams or recent shipped behavior.
triggers:
  - "current state"
  - "what's in progress"
  - "recent work"
  - "active feature"
  - "shipped recently"
last_updated: 2026-04-24
---

# Project State

## Working

- CLI (`uvx arrayview file.npy`) and Python API (`view(arr)`) — both stable
- Display environments: Jupyter inline, VS Code local, VS Code tunnel (stdio), Julia, native pywebview, SSH URL print (user forwards port with `ssh -L`)
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- Rendering pipeline: colormaps, complex modes, mosaic, RGB/RGBA, projections, overlays
- Backend transport parity: FastAPI and stdio now share metadata/analysis helpers, compare/diff helpers, overlay compositing, and vector field layout/arrow sampling via `_analysis.py`, `_diff.py`, `_overlays.py`, and `_vectorfield.py`
- NIfTI spatial metadata, RAS resampling
- VS Code extension v0.14.5 — stable window ID via `EnvironmentVariableCollection`; `arrayview.openInFloatingWindow` setting moves new tabs to a floating window; `view(arr, floating=True)` and `arrayview file.npy --floating` open in a floating window per-call regardless of global setting; `!vscode.env.remoteName` guard removed (remote VS Code supports floating windows); floating mode now uses a single persistent shell hub panel (`_shell.html`) so all arrays share one floating window as tabs instead of opening separate windows; fixed: second CLI call now injects tab via `new_tab` postMessage relay (extension -> hub wrapper -> shell iframe) instead of relying on WebSocket notify which wasn't sent in VS Code mode
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)
- Colormap picker: `c` opens an expanded colorbar-island grid without changing the colormap; subsequent `c` taps cycle, hover/hjkl/arrows live-preview, Enter/click commits, Esc cancels, and auto-dismiss pauses while hovered
- Cold-start loading spinner in VS Code and native shell
- Plugin shelf (`/` menu) supports multi-select: spacebar toggles plugins, Enter applies selection. Mutual exclusion enforced (ROI ↔ Segmentation, and overlay/vectorfield ↔ everything else). Cursor indicator shows focused tile via yellow background + left accent bar
- Dynamic island renders sections for all active plugins simultaneously (qMRI pills + ROI shapes/stats separated by divider), replacing the old single-plugin priority chain
- ROI mode works alongside qMRI: drawing on any pane mirrors the ROI to all panes in real-time via per-pane overlay canvases; stats fetched per parameter map and shown as sub-rows in the island
- qMRI map toggle (`_islandToggleQmriMap`) fade animation now covers dimbar and array-name in addition to panes and colorbars
- Segmentation menu shares the ROI layout (yellow accent, magnifier action icon, common `#export-overlay` modal). Pre-activation shows a pulsing "nnInteractive · connecting" row that morphs into the normal shape toolbar once `/seg/activate` resolves
- Overlays are a plugin tile (`OV`): per-overlay row with colour swatch, editable label, eye toggle, × delete; shared opacity slider; `+ add overlay` opens a filesystem picker rooted at the launched file's directory
- Vector field is a plugin tile (`VF`): row with visibility toggle + density/length sliders bi-directionally wired to the `[ ] { }` keyboard commands
- CLI `--overlay FILE` can be repeated to load multiple overlays at launch
- Filesystem picker endpoint (`GET /fs/list`) clamped to `$HOME`. Accepts `base_sid` + `mode` (`overlay` | `vectorfield`) to filter entries by a cheap header-shape peek (`.npy`, `.nii`/`.nii.gz`, `.h5`, `.zarr`); overlay mode requires identical shape, vectorfield mode requires base shape plus one axis of size 3
- Island collapse affordance: inline `~` at the island's top-right animates the panel into the bottom-left `~` hint circle; external `~` hint only visible while the island is actually collapsed. New `/` hint circle at bottom-right opens the plugin shelf

## In Progress

- Smooth immersive transition — stale scrub geometry handoff is fixed, immersive overlay fade-in is held until after the class switch, shared slim colorbar returns through `drawSlimColorbar()` on reverse, and active scrub suppresses minimap/overflow/drag side effects. Single-pane scrub now targets the actual centered immersive viewport rect instead of a hardcoded corner box, the dimbar stays above the pane during scrub, the shared colorbar sits behind the growing pane, and the phantom extra `av-view-wrap` footprint in normal mode was removed by rebinding `NormalLayout` to the real `#viewer` canvas. Cross-mode parity and deeper reverse-pinch validation still need manual verification.
- ROI + qMRI integration refinements: floodfill not yet supported on qMRI panes; ROI hover tooltip not yet wired for qMRI canvases; per-pane stats are re-fetched on each ROI draw but not updated on slice scroll

## Not Yet Built

- Independent split view for mismatched-shape arrays (designed, shelved)
- Admin/config UI (design intent: file-based user config only, no in-app admin panel)
