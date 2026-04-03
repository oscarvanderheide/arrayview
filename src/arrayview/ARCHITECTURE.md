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
| VS Code local      | Simple Browser (network)           | network     |
| VS Code tunnel     | Direct webview (stdio)             | stdio       |
| Julia              | System browser                     | network     |
| CLI / Python script | Native pywebview                   | network     |
| SSH terminal       | Print URL (user opens browser)     | network     |

Detection logic lives in `_platform.py`. Display opening logic lives in `_launcher.py` (section: ViewHandle and view() API) and `_vscode.py`.

## File Map

| File | Lines | Owns |
|------|------:|------|
| `__init__.py` | 5 | Public API re-exports: `view`, `arrayview`, `ViewHandle`, `TrainingMonitor`, `view_batch` |
| `__main__.py` | 4 | `python -m arrayview` entry point |
| `_app.py` | 179 | Backward-compat shim — re-exports everything from the split modules |
| `_config.py` | 121 | `~/.arrayview/config.toml` read/write, valid window modes/env keys |
| `_io.py` | 253 | Data loading: numpy, NIfTI (lazy nibabel), zarr, DICOM, raw files |
| `_launcher.py` | 2817 | **Main entry.** CLI parser, `view()` API, `ViewHandle`, server lifecycle, SSH relay, demo arrays, file watching |
| `_platform.py` | 396 | Environment detection: Jupyter, VS Code, SSH, tunnel, Julia, native-window capability |
| `_render.py` | 834 | Rendering pipeline: colormap LUTs, slice extraction, RGBA/RGB/mosaic rendering, overlay compositing, preload |
| `_segmentation.py` | 227 | nnInteractive segmentation client (pure HTTP, no nnInteractive dependency) |
| `_server.py` | 3184 | FastAPI app, all REST + WebSocket routes, HTML template serving |
| `_session.py` | 344 | `Session` class, global state (sockets, loops), render thread, prefetch, cache budgets, constants |
| `_stdio_server.py` | 791 | Stdio transport for VS Code direct webview — JSON stdin, binary stdout |
| `_torch.py` | 217 | PyTorch integration: `view_batch()`, `TrainingMonitor` (lazy torch import) |
| `_vscode.py` | 1014 | VS Code extension install/management, signal-file IPC, shared-memory IPC, browser opening |
| `_viewer.html` | 14750 | **The entire frontend** — CSS + JS in a single file, all viewing modes |
| `_shell.html` | 174 | Tab-bar shell for native pywebview — wraps viewer iframes, manages multi-tab sessions |

## Frontend (_viewer.html)

The frontend is a single self-contained HTML file (~15k lines). No build step, no external dependencies. Organized by section separators (`/* ── Section Name ── */` for CSS, `// ── Section Name ──` for JS).

### Major Sections

**CSS (lines ~7–1500)**
| Section | What it covers |
|---------|----------------|
| Theme Variables and Base Layout | CSS custom properties, dark theme palette, root layout grid |
| ColorBar and Dynamic Islands | Colorbar positioning, egg badges, info bar, dimension sliders |
| Generic ColorBar class styles | Shared `.av-colorbar` styles used by the ColorBar JS class |
| Compare, Overlay, and Prompt Styles | Side-by-side panes, overlay blend, prompt dialogs |
| Help and Info Overlays | Help shortcut overlay, array-info panel |
| Immersive, Fullscreen, and Compact Mode | Zen mode hide rules, fullscreen layout, compact overrides |

**JavaScript (lines ~1500–14750)**
| Section | What it covers |
|---------|----------------|
| Constants and Transport Setup | WS URL construction, stdio/postMessage transport abstraction |
| Viewer State Variables | All mutable state: current slice indices, zoom, mode flags |
| Mode Registry | Mode name → enter/exit function mapping |
| PanManager | Canvas panning state machine (normal + compare modes) |
| Canvas Scaling and Layout | `scaleCanvas()`, `mvScaleAllCanvases()`, `compareScaleCanvases()`, `qvScaleAllCanvases()` |
| Compare Mode | Multi-pane compare infrastructure, drag-to-reorder, sub-modes |
| Colorbar Rendering and Histogram | Colorbar draw routines, histogram morph, fullscreen overlay colorbar |
| ColorBar class | Reusable `ColorBar` class (~1100 lines) — draw, histogram, window/level, hover |
| WebSocket and Data Transport | Binary slice receive, request queueing, reconnect |
| Initialization and Metadata Fetch | `/meta` fetch, loading screen, initial render |
| Info Bar and Pixel Display | Bottom info bar, hover pixel readout, coordinate display |
| State Persistence and Restore | URL hash state, `sessionStorage` save/restore |
| Rendering Pipeline | `updateView()`, play/animate, screenshot capture |
| ROI and Selection Modes | Rectangle/ellipse ROI drawing, statistics computation |
| nnInteractive Segmentation | Click-to-segment UI, mask overlay, undo stack |
| Keyboard Shortcuts | All hotkey bindings — single master switch/case block |
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

| Mode | Scale function | Layout | Reconciler notes |
|------|---------------|--------|------------------|
| Normal | `scaleCanvas()` | Single canvas, colorbar below | Base reconciler |
| Immersive (Zen) | `scaleCanvas()` | Canvas fills viewport, UI hidden | Animated enter/exit, dimbar+colorbar drag |
| Compact | `scaleCanvas()` | Reduced chrome | K-key toggle |
| Multiview (3-pane oblique) | `mvScaleAllCanvases()` | 3 canvases (axial/coronal/sagittal) | Layout container visibility reconciler |
| Compare | `compareScaleCanvases()` | 2+ side-by-side panes | Compare sub-mode reconciler (diff/overlay/wipe/flicker/checkerboard) |
| Diff | `compareScaleCanvases()` | Compare variant — center pane shows A-B | Diff colorbar instances |
| Registration | `compareScaleCanvases()` | Compare variant — overlay blend | Cross-fade overlay |
| qMRI | `qvScaleAllCanvases()` | Multi-map quantitative display | Separate scale pipeline |

### CSS Architecture

- **Dark theme only.** Background `#0c0c0c`, text `#d8d8d8`, accents in yellow (`#f5c842`).
- **Monospace UI.** Font: system monospace for data, system sans-serif for labels.
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
