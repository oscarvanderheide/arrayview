# ArrayView Mode & Feature Matrix

Complete reference of all viewing modes, features, overlays, and UI states.
Use this as the source of truth for regression testing and UI audits.

---

## 1. Primary Viewing Modes (mutually exclusive)

| Mode            | State flag(s)         | Key | Notes                                  |
|-----------------|-----------------------|-----|----------------------------------------|
| Normal (single) | (default)             | --  | Default 2D slice view                  |
| Multi-view      | `multiViewActive`     | v   | 3 orthogonal planes (axial/cor/sag)    |
| Multi-view custom | `multiViewActive`   | V   | User picks 3 arbitrary dims            |
| Mosaic / Grid   | (mosaic layout)       | z   | Tile slices along one dim              |
| Projection      | `projectionMode > 0`  | p   | MAX / MIN / MEAN / STD / SOS           |
| qMRI            | `qmriActive`          | q   | Multi-parameter map display (3-6 maps) |
| qMRI compact    | qMRI + compact toggle | q   | Compact 3-panel qMRI                   |
| Compare         | `compareActive`       | picker | 2-6 arrays side-by-side (via unified picker, Ctrl/Cmd+O) |
| Compare + MV    | `compareMvActive`     | v in compare | Rows x 3-planes grid          |
| Compare + qMRI  | `compareQmriActive`   | q in compare | Rows x parameter maps grid    |

## 2. Compare Center-Pane Sub-Modes

When in compare mode, the center pane cycles through these with `X`:

| Sub-mode              | `diffMode` | State flags                   |
|-----------------------|------------|-------------------------------|
| Off                   | 0          | (none)                        |
| A-B (signed diff)     | 1          | `diffMode=1`                  |
| |A-B| (absolute diff) | 2          | `diffMode=2`                  |
| |A-B|/|A| (normalised)| 3          | `diffMode=3`                  |
| Overlay (blend)       | --         | `registrationMode=true`       |
| Wipe                  | --         | `_wipeActive=true`            |
| Flicker               | --         | `_flickerActive=true`         |
| Checkerboard          | --         | `_checkerActive=true`         |

## 3. Compare Layout Variants (cycle with `G`)

| Layout     | Notes                       |
|------------|-----------------------------|
| Horizontal | Side-by-side (default)      |
| Vertical   | Stacked top-to-bottom       |
| Grid       | N-column grid for 3+ arrays |

## 4. Orthogonal Overlays / Features (can combine with any primary mode)

| Feature            | State flag(s)          | Key              | Notes                              |
|--------------------|------------------------|------------------|------------------------------------|
| ROI - Rectangle    | `rectRoiMode`, `_roiShape='rect'`     | A (cycle) | Draw region, see stats    |
| ROI - Circle       | `rectRoiMode`, `_roiShape='circle'`   | A         |                           |
| ROI - Freehand     | `rectRoiMode`, `_roiShape='freehand'` | A         |                           |
| ROI - Flood fill   | `rectRoiMode`, `_roiShape='floodfill'`| A         | Tolerance via `[ / ]`     |
| Histogram/Lebesgue | `lebesgueMode`         | dblclick colorbar | Expanded colorbar w/ KDE          |
| Pixel hover info   | `_pixelInfoVisible`    | H                | Tooltip follows cursor             |
| Vector field       | `vfieldVisible`        | U                | Displacement arrows overlay        |
| Overlay mask       | `overlay_sid`          | (auto)           | Binary mask composited server-side |
| Ruler / measure    | `_rulerMode`           | u                | Click 2 points, see distance       |

## 5. Display Modifiers (toggles, can combine with modes above)

| Modifier        | State flag        | Key | Notes                                 |
|-----------------|-------------------|-----|---------------------------------------|
| Zen / Fullscreen| `_fullscreenActive`| F  | Hide chrome, glassmorphic overlays    |
| Immersive       | zen + fit zoom    | =/+ | Zen mode + zoom to fill              |
| RGB mode        | `rgbMode`         | R   | Render dim of size 3/4 as RGB(A)     |
| Complex mode    | `complexMode`     | m   | Cycle: mag / phase / real / imag     |
| FFT (centered)  | `_fftActive`      | f   | 2D FFT display                       |
| Log scale       | `logScale`        | L   | Logarithmic display                  |
| Mask threshold  | (Otsu levels)     | M   | 8-step threshold                     |
| Square aspect   | (toggle)          | a   | Stretch panes to square              |
| Canvas border   | (toggle)          | b   | Show/hide border around canvas       |

## 6. Colormap & Dynamic Range

| Action             | Key | Notes                                          |
|--------------------|-----|-------------------------------------------------|
| Cycle colormap     | c   | gray > lipari > navia > viridis > plasma > ...  |
| Enter colormap name| C   | Dialog for any matplotlib colormap              |
| Cycle dynamic range| d   | 0-100% > 1-99% > 5-95% > 10-90% > auto         |
| Set vmin/vmax      | D   | Manual entry, locks until next `d`              |
| Theme cycle        | T   | dark > light > solarized > nord                 |

## 7. Colorbar Interactions

| Interaction     | Action                                |
|-----------------|---------------------------------------|
| Drag            | Shift window level (pan vmin/vmax)    |
| Scroll          | Zoom window (narrow/widen range)      |
| Double-click    | Toggle histogram/Lebesgue mode        |

## 8. Data Type Display States

| Data type           | Display handling                   | Auto-detected? |
|---------------------|------------------------------------|----------------|
| Float32 / Float64   | Linear or log colormap             | Yes            |
| Int (8/16/32/64)    | Colormap with quantisation         | Yes            |
| Complex64/128       | complexMode cycle (mag/phase/real/imag) | Yes       |
| Bool                | Binary [0,1] colormap              | Yes            |
| RGB (dim size 3)    | Direct colour or colormap          | Toggle with R  |
| RGBA (dim size 4)   | Direct colour with alpha           | Toggle with R  |

## 9. Array Dimension States

| Dimensions | Navigation              | Display                    |
|------------|-------------------------|----------------------------|
| 2D         | h/l/j/k scroll          | Single canvas              |
| 3D         | + slice scrolling       | Slice index display        |
| 4D+        | Mosaic (z) or qMRI (q)  | Grid or parameter maps     |
| Time dim   | Space (auto-play)       | Animated slicing           |

## 10. Zoom / Pan States

| State            | Entry           | Notes                        |
|------------------|-----------------|------------------------------|
| Fit to window    | 0               | `userZoom = _fitZoom`        |
| Zoomed in        | =/+/Ctrl+scroll | Shows minimap top-right      |
| Panned           | left-drag       | Only when zoomed past fit    |
| Zoom to region   | Shift+drag      | Rectangle selection zoom     |

## 11. UI Chrome Elements

### Always visible
- Canvas viewport (mode-specific element: `#canvas`, `.mv-canvas`, `.compare-canvas`, `.qv-canvas`)
- Colorbar (`#slim-cb-wrap`, per-pane in compare/diff/qMRI)
- Array name (`#array-name`)
- Status bar (`#status`)
- Mode badges / eggs (`#mode-eggs`)

### Conditional
- Histogram/Lebesgue panel (when `lebesgueMode`)
- Diff pane (when `diffMode > 0`)
- 3-plane orientation indicator (when multiview)
- Minimap (when `userZoom > _fitZoom`)
- ROI stats panel (when ROI active)
- Pixel info tooltip (when `_pixelInfoVisible`)
- Help overlay (`?`)
- Data info overlay (`I`)
- Vector arrows overlay (when `vfieldVisible`)
- Ruler lines (when `_rulerMode`)

### Zen mode overrides
- Array name: hidden
- Colorbar: glassmorphic floating overlay
- Mode badges: glassmorphic floating overlay
- Controls: reveal on mouse move

## 12. Mode Badges (Eggs)

| Badge   | Trigger              | Colour     |
|---------|----------------------|------------|
| LOG     | `logScale=true`      | Orange     |
| COMPLEX | complex mode visible | Cyan       |
| MASK    | mask threshold active | Red        |
| RGB     | `rgbMode=true`       | Purple     |
| DIFF    | `diffMode > 0`       | Orange-red |
| PROJ    | `projectionMode > 0` | Purple-blue|
| ROI     | `rectRoiMode=true`   | Green      |
| FFT     | `_fftActive=true`    | Green      |

## 13. Mouse Interactions

| Interaction          | Action                         |
|----------------------|--------------------------------|
| Left-drag (zoomed)   | Pan image                     |
| Right-drag (fit)     | Scrub through slices          |
| Shift+drag           | Zoom to region                |
| Ctrl+scroll          | Zoom in/out                   |
| Hover                | Show pixel value on colorbar  |
| Click (pixel)        | Copy value to clipboard       |
| Drag (colorbar)      | Shift window level            |
| Scroll (colorbar)    | Zoom window range             |
| Dblclick (colorbar)  | Toggle histogram              |
| Drag (compare title) | Reorder panes                 |

## 14. Per-Mode Scale & Colorbar Functions

| Mode        | Scale function             | Colorbar function           |
|-------------|----------------------------|-----------------------------|
| Normal      | `scaleCanvas()`            | `drawSlimColorbar()`        |
| Multi-view  | `mvScaleAllCanvases()`     | `drawMvCbs()`               |
| Compare     | `compareScaleCanvases()`   | `drawAllComparePaneCbs()`   |
| Diff        | (compare scale)            | `drawDiffPaneCb()`          |
| Registration| (compare scale)            | `drawRegBlendCb()`          |
| qMRI        | `qvScaleAllCanvases()`     | inline in `qvRender()`      |

## 15. Invocation Methods (all must work identically)

| Method         | Example                                        |
|----------------|------------------------------------------------|
| CLI            | `arrayview path/to/file.npy`                   |
| Python script  | `import arrayview; arrayview.view(arr)`         |
| Jupyter        | `arrayview.view(arr)` in notebook cell          |
| Julia          | `PythonCall` wrapper                            |
| VS Code tunnel | Remote SSH / tunnel with SimpleBrowser          |
| VS Code ext    | Extension sidebar integration                   |

---

## Regression Testing Priorities

**Every primary mode must support:** zoom, pan, keyboard nav, colorbar interaction, correct colorbar rendering, mode badges, screenshot export.

**Key cross-mode combinations to verify:**
- Normal + each display modifier (RGB, complex, FFT, log, mask)
- Compare + each center-pane sub-mode
- Compare + multi-view
- Multi-view + zoom/pan
- qMRI + compact toggle
- Zen mode + each primary mode
- ROI + each primary mode
- Histogram/Lebesgue + each primary mode
- Each data type x each primary mode
