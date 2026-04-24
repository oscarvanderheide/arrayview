# Architecture

## System Overview

```
CLI / Python API
   ├─ view() / _launcher.py → FastAPI server (_server.py)       [network mode]
   │     /ws/{sid} WebSocket, /load register arrays, /seg/* segmentation
   └─ Stdio server (_stdio_server.py)                           [direct webview mode]
         stdin/stdout JSON+binary, no network — VS Code extension spawns subprocess

Server (either mode)
   ├─ _session.py    Session objects, caches, render thread
   ├─ _render.py     Slice extraction → RGBA → PNG pipeline
   ├─ _analysis.py / _diff.py / _overlays.py / _vectorfield.py
   │                 Shared backend helpers used by FastAPI and stdio
   └─ _io.py         File loading (numpy, nifti, zarr, DICOM, …)

Frontend (_viewer.html — single self-contained HTML file)
   ├─ CSS            Dark theme, mode-specific layouts
   ├─ Canvas         2D rendering via ImageBitmap / putImageData
   ├─ WebSocket      Binary slice transport, JSON metadata
   └─ UI             Colorbars, eggs, dynamic islands, overlays
```

## Display Routing

| Environment        | Default display                    | Server mode |
|--------------------|------------------------------------|-------------|
| Jupyter            | Inline iframe                      | network     |
| VS Code local      | Webview panel (network)            | network     |
| VS Code tunnel     | Direct webview (stdio)             | stdio       |
| Julia              | System browser                     | network     |
| CLI / Python script | Native pywebview                   | network     |
| SSH terminal       | Prints URL — user forwards port with `ssh -L` | network |

Detection logic lives in `_platform.py`. Display opening logic lives in `_launcher.py` (section: ViewHandle and view() API) and `_vscode.py`.

## File Map

| File | Lines | Owns |
|------|------:|------|
| `__init__.py` | 5 | Public API re-exports: `view`, `arrayview`, `ViewHandle`, `TrainingMonitor`, `view_batch`, `zarr_chunk_preset` |
| `__main__.py` | 4 | `python -m arrayview` entry point |
| `_app.py` | 179 | Backward-compat shim — re-exports everything from the split modules |
| `_analysis.py` | 260 | Shared metadata, histogram, Lebesgue, and pixel helpers |
| `_config.py` | 121 | `~/.arrayview/config.toml` read/write, valid window modes/env keys |
| `_io.py` | 253 | Data loading: numpy, NIfTI (lazy nibabel), zarr, DICOM, raw files |
| `_diff.py` | 218 | Shared compare/diff normalization, colorization, and histograms |
| `_launcher.py` | 3052 | **Main entry.** CLI parser, `view()` API, `ViewHandle`, server lifecycle, reverse-tunnel relay, demo arrays, file watching |
| `_overlays.py` | 61 | Shared segmentation/label overlay compositing |
| `_platform.py` | 396 | Environment detection: Jupyter, VS Code, SSH, tunnel, Julia, native-window capability |
| `_render.py` | 834 | Rendering pipeline: colormap LUTs, slice extraction, RGBA/RGB/mosaic rendering, overlay compositing, preload |
| `_segmentation.py` | 227 | nnInteractive segmentation client (pure HTTP, no nnInteractive dependency) |
| `_server.py` | 3704 | FastAPI app, all REST + WebSocket routes, HTML template serving |
| `_session.py` | 344 | `Session` class, global state (sockets, loops), render thread, prefetch, cache budgets, constants |
| `_stdio_server.py` | 767 | Stdio transport for VS Code direct webview — JSON stdin, binary stdout |
| `_torch.py` | 217 | PyTorch integration: `view_batch()`, `TrainingMonitor` (lazy torch import) |
| `_vectorfield.py` | 231 | Shared vector field layout validation and arrow sampling |
| `_vscode.py` | 1014 | VS Code extension install/management, signal-file IPC, shared-memory IPC, browser opening |
| `_viewer.html` | 24103 | **The entire frontend** — CSS + JS in a single file, all viewing modes |
| `_shell.html` | 174 | Tab-bar shell for native pywebview — wraps viewer iframes, manages multi-tab sessions |

## Frontend (_viewer.html)

The frontend is a single self-contained HTML file (~24k lines). No build step, no external dependencies. Organized by section separators (`/* ── Section Name ── */` for CSS, `// ── Section Name ──` for JS).

### Major Sections

**CSS (lines ~7–2059)**
| Section | What it covers |
|---------|----------------|
| Theme Variables and Base Layout | CSS custom properties, dark theme palette, root layout grid |
| ColorBar and Dynamic Islands | Colorbar positioning, egg badges, info bar, dimension sliders |
| Generic ColorBar class styles | Shared `.av-colorbar` styles used by the ColorBar JS class |
| Compare, Overlay, and Prompt Styles | Side-by-side panes, overlay blend, prompt dialogs |
| Help and Info Overlays | Help shortcut overlay, array-info panel |
| Immersive, Fullscreen, and Compact Mode | Zen mode hide rules, fullscreen layout, compact overrides |

**JavaScript (lines ~2439–24103)**
| Section | What it covers |
|---------|----------------|
| Constants and Transport Setup | WS URL construction, stdio/postMessage transport abstraction |
| Viewer State Variables | All mutable state: current slice indices, zoom, mode flags |
| Mode Registry | Mode name → enter/exit function mapping |
| PanManager | Canvas panning state machine (normal + compare modes) |
| Canvas Scaling and Layout | `scaleCanvas()`, `mvScaleAllCanvases()`, `compareScaleCanvases()`, `qvScaleAllCanvases()` |
| Compare Mode | Multi-pane compare infrastructure, drag-to-reorder, sub-modes |
| Colorbar Rendering and Histogram | Colorbar draw routines, histogram morph, fullscreen overlay colorbar |
| ColorBar class | Reusable `ColorBar` class (~900 lines) — draw, histogram, window/level, hover |
| WebSocket and Data Transport | Binary slice receive, request queueing, reconnect |
| Initialization and Metadata Fetch | `/meta` fetch, loading screen, initial render |
| Info Bar and Pixel Display | Bottom info bar, hover pixel readout, coordinate display |
| State Persistence and Restore | URL hash state, `sessionStorage` save/restore |
| Rendering Pipeline | `updateView()`, play/animate, screenshot capture |
| ROI and Selection Modes | Rectangle/ellipse ROI drawing, statistics computation |
| nnInteractive Segmentation | Click-to-segment UI, mask overlay, undo stack |
| Keyboard Shortcuts | Command registry (`commands` / `keybinds` / `makeContext` / `evalWhen` / `dispatchCommand`) + `/`-triggered command palette. Keydown handler is a thin dispatcher prefix |
| Mode Transitions | Compare/multiview/qMRI enter/exit, crosshair animation |
| Scroll, Zoom, and Pan | Mouse wheel slice scroll, pinch zoom, scroll-to-zoom |
| Immersive Mode, Cross-Fade, and Visual Effects | Zen mode, fullscreen (K), animated transitions |
| UI Validation and Reconciliation | Reconcilers that enforce consistent UI state across modes |
| Colormap Strip and Wipe/Flicker Compare Tools | Colormap preview strip, A/B wipe, flicker, checkerboard |
| Ruler, Line Profile, and Mini-Map | Distance measurement, 1D intensity profile, overview mini-map |
| Compact Mode and Touch Input | Compact layout (K), touch gesture handling |
| File Picker and Session Management | File browser, drag-and-drop, session switching |

### Mode Matrix

Each viewing mode uses a specific scale function and reconciler set:

| Mode | Scale function | `modeManager.modeName` | Notes |
|------|---------------|------------------------|-------|
| Normal | `scaleCanvas()` | `'normal'` | Single canvas, colorbar below |
| Immersive (Zen) | `scaleCanvas()` | `'normal'` | Canvas fills viewport, UI hidden |
| Compact | `scaleCanvas()` | `'normal'` | K-key toggle, reduced chrome |
| Multiview (3-pane oblique) | `mvScaleAllCanvases()` | `'multiview'` | 3 Views: axial/coronal/sagittal |
| Compare | `compareScaleCanvases()` | `'compare'` | N Views, one per SID |
| Diff | `compareScaleCanvases()` | `'compare'` | Compare + center diff View |
| Registration | `compareScaleCanvases()` | `'compare'` | Compare + center blend View |
| Wipe/Flicker/Checker | `compareScaleCanvases()` | `'compare'` | Compare + composite center View |
| Compare + MV | `compareMvScaleAllCanvases()` | `'compare-mv'` | N×3 pane grid |
| Compare + qMRI | `compareQmriScaleAllCanvases()` | `'compare-qmri'` | N×K pane grid |
| qMRI | `qvScaleAllCanvases()` | `'qmri'` | K Views, one per parameter map |
| qMRI Mosaic | `qvScaleAllCanvases()` | `'qmri-mosaic'` | Mosaic variant of qMRI |
| MIP (3-D volume) | N/A (WebGL) | `'mip'` | Single View wrapping WebGL canvas |

### CSS Architecture

- **Dark theme only.** Background `#0c0c0c`, text `#d8d8d8`, accents in yellow (`#f5c842`).
- **Monospace UI.** All text uses the system monospace stack.
- **No CSS framework.** All custom properties defined in `:root`.
- **Mode-specific overrides** use `.immersive-mode`, `.compact-mode`, `.multiview-mode` classes on `body`.
- **`av-loading` class** on `body` during initial load — hides UI until data arrives. Removal triggers layout.
- **Hard UI invariants** block at line ~157 — structural rules that must never be overridden.

## Key Concepts

### Sessions
A `Session` object (`_session.py`) holds one array's data, metadata, and three LRU caches (raw slices, RGBA tiles, mosaics). Sessions are stored in the global `SESSIONS` dict keyed by `sid` (hex UUID). Multiple sessions enable compare mode and the file picker.

### Eggs (Mode Badges)
Pill-shaped badges below the canvas showing active visualization transforms. **Composable** — they stack: `FFT` `LOG` `MAGNITUDE` `PHASE` `REAL` `IMAG` `RGB` `ALPHA` `PROJECTION`. Each egg toggles a transform in the rendering pipeline.

**ROI and SEGMENT are NOT eggs.** They are interaction modes that take over canvas input, are mutually incompatible, and each has its own dynamic island UI.

### Dynamic Islands
Floating UI panels that appear/disappear based on context: ROI statistics, segmentation controls, colorbar hover, dimension sliders. Must be tested across all viewing modes (normal, immersive, multiview, compare).

### View Component System (Phases 1–15)

A refactoring in progress that introduces a View/Layer/Slicer/LayoutStrategy/ModeManager model alongside the legacy code. Every mode now registers its views with `modeManager` without replacing the legacy render pipeline.

**Key primitives (all in `_viewer.html`):**

| Primitive | What it owns |
|-----------|-------------|
| `makeDisplayState(overrides)` | Plain-object factory: `{vmin, vmax, cmapIdx, logScale, complexMode, renderMode, projectionMode, …}` |
| `class View` | Canvas + ColorBar + DisplayState + Slicer + Layers[]. Has `init()`, `render()`, `requestRender()`, capability checks |
| `class Slicer` | `FreeSliceSlicer`, `OrthogonalSlicer(axis)` — encapsulates how to fetch slice data |
| `class Layer` | `CrosshairLayer`, `VectorFieldLayer`, `OverlayLayer` — duck-typed, composable render features |
| `class LayoutStrategy` | `NormalLayout`, `MultiViewLayout`, `CompareLayout`, `QmriLayout`, `MipLayout`, etc. |
| `const modeManager` | Singleton: `{currentViews[], currentLayout, modeName, enterMode(), getFocusedView(), getViewUnderMouse(), …}` |
| `const LAYOUT_REGISTRY` | Maps mode name → `() => new XxxLayout()` factory |

**Sync-block pattern:** Every `enterXxx()` function ends with a block that creates thin `View` wrappers over existing legacy canvases/colorbars and populates `modeManager.currentViews` + `modeName`. The legacy render pipeline is unchanged.

**Dual-write pattern:** Where commands update legacy globals (`logScale`, `colormap_idx`, `complexMode`, `projectionMode`), they also write to `view.displayState.xField` so both systems stay in sync.

**Manual range state:** `manualVmin` / `manualVmax` are regular state variables. Write sites must explicitly dual-write to `displayState.vmin` / `displayState.vmax`; the old `Object.defineProperty(window, …)` shim was removed.

**Status:** Phases 1–15 complete. Phase 12 (keybind collapse) and Phase 17 (final cleanup) pending completion of render-pipeline migration from legacy globals to `displayState`.

### Command Registry
All keybinds flow through a VS Code-style command registry in `_viewer.html`. Three tables: `commands` (id → `{title, when, run}`), `keybinds` (key+modifiers → command id), and `makeContext(state)` (mode/state flag bag). `dispatchCommand(e)` is wired as a prefix to the keydown handler; on a match it evaluates `when` against the context and runs the command, otherwise falls through. The help overlay is rendered at runtime from the `GUIDE_TABS` static data structure in `_viewer.html` — when adding or changing a keybind, update `GUIDE_TABS` manually. A `/`-triggered command palette fuzzy-searches all commands. Cross-mode enablement is guarded by `tests/test_command_reachability.py`.

### Reconcilers
Functions in the "UI Validation and Reconciliation" section (~line 13666) that enforce consistent UI state. When mode changes happen, reconcilers update visibility of containers, colorbars, dynamic islands, and compare sub-mode UI. There are four:
1. **Unified UI reconciler** — master state enforcer
2. **Layout container visibility reconciler** — show/hide mode-specific containers
3. **Compare sub-mode state reconciler** — diff/overlay/wipe/flicker/checkerboard state
4. **CB / island visibility reconciler** — colorbar and dynamic island show/hide

### Render Thread
A dedicated daemon thread (`_session.py`) runs all CPU-heavy rendering off the async event loop. `_render()` posts work to `_RENDER_QUEUE` and returns an awaitable `Future`. The prefetch pool (separate 1-thread executor) warms caches for neighboring slices.

## High-Risk Areas

### CSS Pitfalls
- **Canvas buffer vs CSS resolution mismatches.** The canvas element size (CSS pixels) and its buffer size (device pixels) must stay in sync. `scaleCanvas()` handles this — never set canvas dimensions outside a scale function.
- **Selector targeting wrong wrappers.** Normal mode has one `#canvas-wrapper`; compare mode has per-pane wrappers. CSS rules must scope correctly.
- **`av-loading` class interactions.** Many elements are hidden while `av-loading` is on `body`. Removing it too early causes layout flash; too late causes blank screen.

### Dynamic Islands
Must verify island positioning and visibility across normal, immersive, multiview, and compare modes. Islands use absolute/fixed positioning that breaks if parent containers change.

### Keybind Changes
Keybinds live in the `commands` + `keybinds` tables, not in the keydown handler. Adding or changing a keybind means editing those tables and (if needed) extending `makeContext` / `evalWhen`. The help overlay renders from `GUIDE_TABS` — update that data structure when adding or changing keybinds, do not edit overlay HTML directly.

### Layout Debugging
When debugging layout issues: identify the root cause (which scale function, which reconciler, which CSS rule) before applying fixes. Symptoms in one mode often originate from shared code affecting all modes.

### WebSocket Binary Protocol
Slice data arrives as raw binary (RGBA bytes). The header format and byte offsets are tightly coupled between `_server.py` (Python) and the WebSocket handler in `_viewer.html` (JS). Changes to one must match the other.

### Lazy Imports
`_launcher.py` deliberately avoids importing numpy, _session, _render, and _io at module level. This saves ~300-350ms on CLI fast paths (when the server is already running). Breaking this lazy-loading pattern degrades startup time for every invocation.

## Data Flow

### Request Lifecycle: `view()` to Pixels

```
1. view(array)                          # _launcher.py
   ├─ Create Session(data)              # _session.py — assigns sid, inits caches
   ├─ Start server if not running       # _launcher.py → _server.py (uvicorn)
   ├─ Register session via /load        # HTTP POST to FastAPI
   └─ Open display                      # _launcher.py → _vscode.py / pywebview / browser

2. Browser loads _viewer.html           # _server.py serves from package resources
   ├─ Establish WebSocket /ws/{sid}     # or stdio transport for VS Code direct
   └─ Fetch /meta/{sid}                 # Session metadata: shape, dtype, colormaps

3. User scrolls / interacts
   ├─ JS sends slice request            # WebSocket binary or JSON
   ├─ Server dispatches to render thread
   │   ├─ extract_slice()               # _render.py — pull 2D slice from ND array
   │   ├─ apply_complex_mode()          # _render.py — FFT, magnitude, phase, etc.
   │   ├─ render_rgba() / render_rgb()  # _render.py — apply colormap LUT → RGBA
   │   └─ PNG encode                    # sent as binary WebSocket frame
   └─ JS receives binary
       ├─ Decode PNG → ImageBitmap
       ├─ drawImage() to canvas
       └─ Update info bar, colorbar, eggs
```

### Stdio Transport Variant (VS Code Direct Webview)
Same pipeline, but `_stdio_server.py` replaces the FastAPI+WebSocket layer. Messages are JSON on stdin, binary responses are length-prefixed on stdout. The VS Code extension bridges between the webview's `postMessage` and the subprocess stdio.
