---
name: frontend
description: _viewer.html internals — mode matrix, reconcilers, command registry, View Component System, CSS architecture. Load for any task touching _viewer.html.
triggers:
  - "_viewer.html"
  - "frontend"
  - "mode"
  - "reconciler"
  - "keybind"
  - "colorbar"
  - "canvas"
  - "dynamic island"
  - "egg"
  - "command registry"
  - "View Component"
  - "modeManager"
edges:
  - target: context/architecture.md
    condition: when understanding how the frontend connects to the server
  - target: context/decisions.md
    condition: when understanding why the frontend is a single file or why reconcilers exist
  - target: context/conventions.md
    condition: when writing new frontend code and need section separator conventions
  - target: patterns/frontend-change.md
    condition: when making a concrete change to _viewer.html
last_updated: 2026-04-29
---

# Frontend (_viewer.html)

~24 700 lines. Single file, no build step. All CSS and JS inline.

## Section Organization

Section separators: `/* ── Section Name ── */` in CSS, `// ── Section Name ──` in JS.

**CSS (lines ~7–2059)**
| Section | What it covers |
|---------|----------------|
| Theme Variables and Base Layout | CSS custom properties, dark theme palette, root layout grid |
| ColorBar and Dynamic Islands | Colorbar positioning, egg badges, info bar, dimension sliders |
| Generic ColorBar class styles | Shared `.av-colorbar` styles used by the `ColorBar` JS class |
| Compare, Overlay, and Prompt Styles | Side-by-side panes, overlay blend, prompt dialogs |
| Help and Info Overlays | Help shortcut overlay, array-info panel |
| Immersive, Fullscreen, and Compact Mode | Zen mode hide rules, fullscreen layout, compact overrides |

**JavaScript (lines ~2439–24103)**
| Section | What it covers |
|---------|----------------|
| Constants and Transport Setup | WS URL construction, stdio/postMessage transport abstraction |
| Viewer State Variables | All mutable state: slice indices, zoom, mode flags |
| Mode Registry | Mode name → enter/exit function mapping |
| PanManager | Canvas panning state machine |
| Canvas Scaling and Layout | `scaleCanvas()`, `mvScaleAllCanvases()`, `compareScaleCanvases()`, `qvScaleAllCanvases()` |
| Compare Mode | Multi-pane compare infrastructure, drag-to-reorder |
| Colorbar Rendering and Histogram | Colorbar draw routines, histogram morph |
| ColorBar class | Reusable `ColorBar` class (~900 lines) |
| WebSocket and Data Transport | Binary slice receive, request queueing, reconnect |
| Initialization and Metadata Fetch | `/meta` fetch, loading screen |
| Info Bar and Pixel Display | Bottom info bar, hover pixel readout |
| State Persistence and Restore | URL hash state, `sessionStorage` save/restore |
| Rendering Pipeline | `updateView()`, play/animate, screenshot |
| ROI and Selection Modes | Rectangle/circle/freehand ROI, statistics computation, qMRI cross-pane mirroring |
| nnInteractive Segmentation | Click-to-segment UI, mask overlay, undo stack |
| Keyboard Shortcuts | Command registry + command palette |
| Mode Transitions | Compare/multiview/qMRI enter/exit, crosshair animation |
| Scroll, Zoom, and Pan | Mouse wheel, pinch zoom |
| Immersive Mode, Cross-Fade, and Visual Effects | Zen mode, fullscreen, animated transitions |
| UI Validation and Reconciliation | Reconcilers (~line 13666) |
| Colormap Strip and Wipe/Flicker Compare Tools | A/B wipe, flicker, checkerboard |
| Ruler, Line Profile, and Mini-Map | Distance measurement, intensity profile |
| Compact Mode and Touch Input | Touch gesture handling |
| File Picker and Session Management | File browser, drag-and-drop |

## Mode Matrix

| Mode | Scale function | `modeManager.modeName` |
|------|---------------|------------------------|
| Normal | `scaleCanvas()` | `'normal'` |
| Immersive (Zen) | `scaleCanvas()` | `'normal'` |
| Compact | `scaleCanvas()` | `'normal'` |
| Multiview (3-pane oblique) | `mvScaleAllCanvases()` | `'multiview'` |
| Compare | `compareScaleCanvases()` | `'compare'` |
| Diff / Registration / Wipe/Flicker/Checker | `compareScaleCanvases()` | `'compare'` |
| Compare + MV | `compareMvScaleAllCanvases()` | `'compare-mv'` |
| Compare + qMRI | `compareQmriScaleAllCanvases()` | `'compare-qmri'` |
| qMRI | `qvScaleAllCanvases()` | `'qmri'` |
| qMRI Mosaic | `qvScaleAllCanvases()` | `'qmri-mosaic'` |
| MIP (3-D volume) | N/A (WebGL) | `'mip'` |

## Key Concepts

### Reconcilers (~line 22075)
Four functions enforcing consistent UI state across mode changes:
1. **Unified UI reconciler** — master state enforcer
2. **Layout container visibility reconciler** — show/hide mode-specific containers
3. **Compare sub-mode state reconciler** — diff/overlay/wipe/flicker/checkerboard
4. **CB / island visibility reconciler** — colorbar and dynamic island show/hide

**Rule:** All visibility changes go through reconcilers. Never set `style.display` or toggle classes in mode-entry/exit functions — that belongs in the reconcilers.

### Command Registry
Three tables: `commands` (id → `{title, when, run}`), `keybinds` (key+modifiers → command id), `makeContext(state)` (mode/state flag bag). `dispatchCommand(e)` is the keydown prefix handler.

**Rule:** When adding or changing a keybind, update both `commands`/`keybinds` tables AND `GUIDE_TABS` (the static data structure that renders the help overlay). Do not edit overlay HTML directly.

### View Component System (Phases 1–17 mostly complete)
Introduces `View`, `Slicer`, `Layer`, `LayoutStrategy`, `modeManager` alongside the legacy render pipeline.

| Primitive | What it owns |
|-----------|-------------|
| `makeDisplayState(overrides)` | Plain-object factory: `{vmin, vmax, cmapIdx, logScale, complexMode, renderMode, projectionMode, …}` |
| `class View` | Canvas + ColorBar + DisplayState + Slicer + Layers[]. Has `init()`, `render()`, `requestRender()` |
| `class Slicer` | `FreeSliceSlicer`, `OrthogonalSlicer(axis)` |
| `class Layer` | `CrosshairLayer`, `VectorFieldLayer`, `OverlayLayer` — duck-typed, composable |
| `class LayoutStrategy` | `NormalLayout`, `MultiViewLayout`, `CompareLayout`, `QmriLayout`, etc. |
| `const modeManager` | Singleton: `{currentViews[], currentLayout, modeName, enterMode(), …}` |

**Sync-block pattern:** Every `enterXxx()` function ends with a block that creates thin `View` wrappers over existing legacy canvases/colorbars and populates `modeManager.currentViews` + `modeName`. The legacy render pipeline is unchanged.

**Dual-write pattern:** Where commands update legacy globals (`logScale`, `colormap_idx`, etc.), they also write to `view.displayState.xField` so both systems stay in sync.

**Manual range state:** `manualVmin` / `manualVmax` are regular state variables. Write sites must explicitly dual-write to `displayState.vmin` / `displayState.vmax`; the old `Object.defineProperty(window, …)` shim was removed.

### Tool Menu (`/` menu)
`SPECIAL_MODE_TILES` array defines the tool menu contents: qMRI, Segmentation, ROI, Overlay, Vector field, plus the compare-center entry when compare mode is eligible. The menu supports multi-select where allowed (spacebar toggles, Enter applies). Mutual exclusion is enforced via `tile.excludes` arrays — ROI ↔ Segmentation, and Overlay/Vector field are reciprocally exclusive with all other plugins. `_applyShelfSelection()` diffs current state against selection, exits removed tools, then enters new ones wrapped in `crossfade()`. Overlay and Vector field auto-seed the menu at init when arrays are present (`_overlaySids.length > 0`, `hasVectorfield`); their per-tile state lives in `_shelfSelection.has(id)` since there is no separate mode flag.

### Compare Center Tool
Compare mode can expose a unified center pane that cycles between diff, overlay, and wipe. The `/` tool menu re-opens the last-used center mode, while the compare pane header buttons select a specific center mode directly. Eligible layouts can switch to `compare-center-layout-big-left`, which widens the center pane and stacks the source panes on the right. The diff center pane owns its own colorbar state via `_diffCenterColormap` / `_diffCenterColormapStops`.

### Eggs
Pill badges below the canvas showing active transforms: `FFT` `LOG` `MAGNITUDE` `PHASE` `REAL` `IMAG` `RGB` `ALPHA` `PROJECTION`. **ROI** and **SEGMENT** are NOT eggs — they are interaction modes with their own dynamic island UI.

### Dynamic Islands
Floating UI panels. `renderIsland()` collects all active plugins (`qmriActive`, `rectRoiMode`, `_segMode`/`_segActivating`, `_shelfSelection.has('overlay')`, `_shelfSelection.has('vectorfield')`) and renders sections for each with dividers between them. ROI/Segmentation share the same shape-toolbar + rows + magnifier-action layout; segmentation also renders a pulsing "connecting" loading row during `_segActivating`. Overlay section renders per-overlay rows (swatch + editable label + eye + `×`) with a shared opacity slider and a `+ add overlay` button that opens the filesystem picker. Vector field section renders a visibility row plus density and length sliders wired to `vfDensityLevel` / `vfLengthLevel` (both directions — keyboard `[ ] { }` also re-renders the island). An inline `~` collapse button at the island's top-right triggers `_collapseIslandToHint()`. `renderIsland()` early-returns while `_islandSliderDragging` is true so a mid-drag DOM rebuild doesn't kill the grab. Use absolute/fixed positioning — must be tested across normal, immersive, multiview, and compare modes.

### ROI in qMRI Mode
Each qMRI pane has a `.qv-roi-overlay` canvas (transparent, `position: absolute`, `pointer-events: none`). Drawing a ROI on any pane mirrors it to all panes in real-time via `_drawAllQvRois()`. On finalize, `_fetchQvRoiStats()` fetches per-parameter-map statistics in parallel. Results stored as `roi.qmriStats[]` and displayed as sub-rows in the island. Key functions: `_drawAllQvRois`, `_clearQvRoiOverlays`, `_redrawRoiOverlays`, `_finalizeQvRoi`, `_fetchQvRoiStats`. **Important:** the global `canvas { background: var(--bg) }` rule makes all canvas elements opaque — overlay canvases must set `background: transparent` to avoid covering the underlying content.

## CSS Architecture

- **Dark theme only.** `#0c0c0c` background, `#d8d8d8` text, `#f5c842` yellow accents.
- **No CSS framework.** All custom properties in `:root`.
- **Mode-specific overrides** use `.immersive-mode`, `.compact-mode`, `.multiview-mode` classes on `body`.
- **`av-loading` class** on `body` during initial load — hides UI until data arrives.
- **Canvas size management:** `scaleCanvas()` (and variants) are the only legitimate way to set canvas dimensions. Never set canvas `width`/`height` directly.

## WebSocket Binary Protocol
Slice data arrives as raw RGBA bytes. Header format and byte offsets are tightly coupled between `_server.py` (Python) and the WS handler in `_viewer.html` (JS). Changes to one must match the other exactly.
