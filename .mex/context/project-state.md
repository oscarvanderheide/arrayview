---
name: project-state
description: Current working, in-progress, and shelved status. Load only when the task depends on active workstreams or recent shipped behavior.
triggers:
  - "current state"
  - "what's in progress"
  - "recent work"
  - "active feature"
  - "shipped recently"
last_updated: 2026-05-11
---

# Project State

## Working

- CLI (`uvx arrayview file.npy`) and Python API (`view(arr)`) — both stable
- Display environments: Jupyter inline, VS Code local, VS Code tunnel (stdio), Julia, native pywebview, SSH URL print (user forwards port with `ssh -L`)
- File formats: `.npy`, `.npz`, `.nii`/`.nii.gz`, `.zarr`, `.h5`/`.hdf5`, `.mat`, `.tif`/`.tiff`, `.pt`/`.pth`
- Rendering pipeline: colormaps, complex modes, mosaic, RGB/RGBA, projections, overlays
- Backend transport parity: FastAPI and stdio now share metadata/analysis helpers, compare/diff helpers, overlay compositing, and vector field layout/arrow sampling via `_analysis.py`, `_diff.py`, `_overlays.py`, and `_vectorfield.py`
- NIfTI spatial metadata, RAS resampling
- VS Code extension v0.14.4 — stable window ID via `EnvironmentVariableCollection`; `arrayview.openInFloatingWindow` setting moves new tabs to a floating window; `view(arr, floating=True)` and `arrayview file.npy --floating` open in a floating window per-call regardless of global setting; `!vscode.env.remoteName` guard removed (remote VS Code supports floating windows); floating mode now uses a single persistent shell hub panel (`_shell.html`) so all arrays share one floating window as tabs instead of opening separate windows; fixed: second CLI call now injects tab via `new_tab` postMessage relay (extension -> hub wrapper -> shell iframe) instead of relying on WebSocket notify which wasn't sent in VS Code mode; custom-editor fallback now pins `uv` to Python 3.12 so tunnel workspaces with older project interpreters still open arrays
- Colorbar refactor: `ColorBar` JS class partially migrated (in progress)
- Colormap picker: `c` opens an expanded colorbar-island grid without changing the colormap; subsequent `c` taps cycle, hover/hjkl/arrows live-preview, Enter/click commits, Esc cancels, and auto-dismiss pauses while hovered
- Cold-start loading spinner in VS Code and native shell
- Tool menu (`/` menu) supports multi-select where allowed: spacebar toggles tools, Enter applies selection. Mutual exclusion enforced (ROI ↔ Segmentation, and overlay/vectorfield ↔ everything else). Cursor indicator shows focused tile via yellow background + left accent bar
- Dynamic island renders sections for all active plugins simultaneously (qMRI pills + ROI shapes/stats separated by divider), replacing the old single-plugin priority chain
- ROI mode works alongside qMRI: drawing on any pane mirrors the ROI to all panes in real-time via per-pane overlay canvases; stats fetched per parameter map and shown as sub-rows in the island
- qMRI map toggle (`_islandToggleQmriMap`) fade animation now covers dimbar and array-name in addition to panes and colorbars
- Segmentation menu shares the ROI layout (yellow accent, magnifier action icon, common `#export-overlay` modal). Pre-activation shows a pulsing "nnInteractive · connecting" row that morphs into the normal shape toolbar once `/seg/activate` resolves
- Overlays are a plugin tile (`OV`): per-overlay row with colour swatch, editable label, eye toggle, × delete; shared opacity slider; `+ add overlay` opens a filesystem picker rooted at the launched file's directory
- Vector field is a plugin tile (`VF`): row with visibility toggle + density/length sliders bi-directionally wired to the `[ ] { }` keyboard commands
- CLI `--overlay FILE` can be repeated to load multiple overlays at launch
- Filesystem picker endpoint (`GET /fs/list`) clamped to `$HOME`. Accepts `base_sid` + `mode` (`overlay` | `vectorfield`) to filter entries by a cheap header-shape peek (`.npy`, `.nii`/`.nii.gz`, `.h5`, `.zarr`); overlay mode requires identical shape, vectorfield mode requires base shape plus one axis of size 3
- Island collapse affordance: inline `~` at the island's top-right animates the panel into the bottom-left `~` hint circle; external `~` hint only visible while the island is actually collapsed. New `/` hint circle at bottom-right opens the tool menu
- Compare center tool menu: `/` now re-opens the last-used compare center mode from the tool menu, while compare pane header buttons select diff / overlay / wipe directly. Eligible two-array compare layouts can switch into a big-left arrangement with a wider center pane, and the diff colorbar now matches that center-pane width.
- Compare center + ortho auto-layout: layout choice is now auto-picked once on entry from a shared viewport-profile helper, stays stable across resize, and becomes sticky only after manual `G` / `g` override. Compare big-left also now supports in-pane A/B source badges and a shared source colorbar parked in the gap between the stacked source panes.

## Recently Completed

- Normal-mode dimbar readability: inactive non-spatial dims no longer get a blanket reduced parent opacity, so the current index reads bright while `/total` stays subdued via the existing child dim-size styling.
- Multiview colorbar spacing now matches normal mode: entering ortho with `v` and switching to orthogonal `big-left` with `g` no longer increases the pane-to-colorbar gap. Focused browser coverage now compares the normal-view gap against both multiview layouts directly.
- V-mode ortho layout cycling is now a two-state toggle: `g` only switches between `horizontal` and `big-left`. The old `vertical` and `big-top` multiview presets have been removed from the shared ortho preset table, the hold-`g` picker/help text, and the remaining big-pane promotion branches, and the shared multiview colorbar width now stays fixed when `g` toggles between the two surviving layouts.
- The old snapshot gallery sidecar has been removed from the viewer. Saving a screenshot still downloads the PNG, but `Shift+G` no longer toggles any screenshot/gallery UI outside compare mode; `G` is now compare-layout cycling only.
- Loupe activation no longer arms on a plain left-button hold. Normal view and multiview panes now use `Ctrl` hover with no mouse button, so plain drag stays available for pane navigation and other mode-specific gestures. While `Ctrl` is held, wheel input keeps its normal slice-scroll behavior instead of being swallowed or repurposed for zoom, and the loupe now redraws when those wheel-driven slice changes land. Focused browser coverage now checks both the normal-view and multiview `Ctrl` hover paths plus `Ctrl`-wheel scrolling and loupe-refresh behavior. qMRI loupe wiring was updated to match, but direct qMRI browser validation is currently blocked by an existing `q`-mode entry failure in `tests/test_browser.py::TestKeyboard::test_reusable_url_restores_qmri_mode`.
- Jupyter inline notebooks now default to `height=600`, start in normal mode instead of auto-immersive, and top-align the viewer cluster instead of vertically centering it inside the output cell. This removes the wasted band above the dimbar / below the colorbar while preserving manual `Shift+F` immersive entry. Browser coverage now checks the non-immersive inline start state and low wrapper top padding.
- Architecture followthrough: `_server.py` route extraction is complete. Feature domains now live in `_routes_analysis.py`, `_routes_loading.py`, `_routes_persistence.py`, `_routes_segmentation.py`, `_routes_state.py`, `_routes_query.py`, `_routes_export.py`, `_routes_preload.py`, `_routes_vectorfield.py`, `_routes_rendering.py`, and `_routes_websocket.py`. `_server.py` is now the intended assembly surface: FastAPI app setup, shared dependency injection, HTML/template helpers, and the small infrastructure routes for health/UI/assets/colormap metadata.
- Focused API coverage now directly guards segmentation activate/scribble/click-accept/export paths, export/preload/vectorfield routes, slice/projection/diff/grid/gif rendering, large-array grid/gif guardrails, and websocket metadata plus shell-close cleanup.

## In Progress

- Smooth immersive transition — stale scrub geometry handoff is fixed, immersive overlay fade-in is held until after the class switch, shared slim colorbar returns through `drawSlimColorbar()` on reverse, and active scrub suppresses minimap/overflow/drag side effects. Single-pane scrub now targets the actual centered immersive viewport rect instead of a hardcoded corner box, the dimbar stays above the pane during scrub, the shared colorbar sits behind the growing pane, and the phantom extra `av-view-wrap` footprint in normal mode was removed by rebinding `NormalLayout` to the real `#viewer` canvas. Cross-mode parity and deeper reverse-pinch validation still need manual verification.
- ROI + qMRI integration refinements: floodfill not yet supported on qMRI panes; ROI hover tooltip not yet wired for qMRI canvases; per-pane stats are re-fetched on each ROI draw but not updated on slice scroll

## Not Yet Built

- Independent split view for mismatched-shape arrays (designed, shelved)
- Admin/config UI (design intent: file-based user config only, no in-app admin panel)
