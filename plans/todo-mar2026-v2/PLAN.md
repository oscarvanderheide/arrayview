# TODO Plan — March 2026 (v2)

Items from `TODO.md`. Each item = one commit. Ordered by complexity (small → large), related items grouped.

---

## 1. Dim bar padding — fixed-width index numbers

**Problem:** When scrolling through a dim of size 256, the total dim bar width shifts at 9→10 and 99→100 because the index text width changes.  
**Fix:** Pad the index number with leading spaces (or use `ch`-based `min-width`) so `[  1/256]` has the same width as `[256/256]`. Use `Math.ceil(Math.log10(shape[i]+1))` digits for padding.  
**Files:** `_viewer.html` (`renderDimensionBar`, `.dim-text` CSS)  
**Test:** `tests/visual_smoke.py` — scroll scenario, verify no layout jump

---

## 2. Multi-view (x,y) label on wrong canvas

**Problem:** In multi-view mode, the `(x,y)` position display appears on the right canvas instead of the left canvas.  
**Fix:** Inspect `showPixelInfo()` and the multi-view mouse handlers — ensure the per-pane pixel info element is attached to the correct pane (the one actually under the cursor).  
**Files:** `_viewer.html` (multi-view pixel info positioning)  
**Test:** `tests/visual_smoke.py` — multi-view screenshot

---

## 3. Diff mode egg (indicator dot)

**Problem:** When a specific diff mode is active (A-B, |A-B|, overlay, wipe), there's no egg badge indicating which mode. Magnitude/phase have eggs, diff should too.  
**Fix:** Add a new `.mode-badge-diff` egg with a unique color not used by existing eggs (RGB=cyan, FFT=magenta, LOG=orange, complex=cyan, mask=green). Use e.g. a coral/salmon color. Show the short mode name text inside the badge. Update `positionEggs()` for all mode branches.  
**Files:** `_viewer.html` (CSS, egg HTML, `positionEggs`, diff mode enter/exit)  
**Test:** `tests/visual_smoke.py`

---

## 4. Compare mode spurious long colorbar

**Problem:** In compare mode there's a weird, long colorbar that shouldn't be there (see `compare_weird_colorbar.png`).  
**Fix:** Investigate `#slim-cb-wrap` visibility when `compareActive` — likely not being hidden on compare enter, or being re-shown by `drawSlimColorbar` during a render cycle. Ensure `#slim-cb-wrap` is hidden when compare mode is active (compare has its own per-pane colorbars).  
**Files:** `_viewer.html` (`enterCompareModeBySid`, `drawSlimColorbar`, compare teardown)  
**Test:** `tests/visual_smoke.py` — compare mode screenshot

---

## 5. Pinch zoom speed — still too fast

**Problem:** Pinch-to-zoom is too sensitive despite `^0.6` dampening.  
**Fix:** Increase dampening exponent from `0.6` to ~`0.35–0.4`, or add a multiplier cap per-frame (e.g., `Math.max(0.95, Math.min(1.05, ratio))` per touch event). Test on trackpad.  
**Files:** `_viewer.html` (pinch handler, `~line 7670`)  
**Test:** Manual trackpad test

---

## 6. Zoom: don't widen canvas past vertical limit

**Problem:** When the max canvas size in vertical direction has been reached, pressing `+` still makes the canvas wider. It should stop growing in both dimensions once either dimension hits the limit.  
**Fix:** In `scaleCanvas()`, after computing CSS size, check if either dimension exceeds viewport. If height is capped but width isn't, cap width proportionally (preserve aspect ratio). The canvas should grow uniformly and clip via the viewport.  
**Files:** `_viewer.html` (`scaleCanvas`)  
**Test:** `tests/visual_smoke.py` — zoom scenario

---

## 7. Click-drag pan when zoomed in

**Problem:** User wants to click-drag (left button) to pan the visible part of the canvas when zoomed past the viewport.  
**Fix:** When `_isCanvasOverflowing()` is true, left-click-drag on the canvas should pan (update `_panX`, `_panY`, call `_clampPan()` and `_applyCanvasPan()`). Currently only right-click-drag pans. Add left-drag pan alongside existing right-drag, but only when overflowing (to not conflict with ROI drawing — guard: not in ROI mode).  
**Files:** `_viewer.html` (canvas mousedown/mousemove/mouseup handlers)  
**Test:** `tests/visual_smoke.py`

---

## 8. Colormap switch: fade colorbar/histogram, shorter previewer fade

**Problem:** When pressing `c` to change colormap, the colorbar/histogram stays static. User wants: (a) fade out the colorbar while the colormap strip previewer is shown, (b) make the previewer fade 1s shorter, (c) fade the colorbar back in when done.  
**Fix:**  
- On `c` keypress: add class `.cb-faded` to `#slim-cb-wrap` (opacity→0.15, transition 0.3s)  
- Strip auto-fade timer: reduce from 2.5s to 1.5s  
- On strip fade: remove `.cb-faded` class (opacity→1, transition 0.5s)  
- Same logic for compare per-pane colorbars when applicable  
**Files:** `_viewer.html` (CSS transition, c key handler, `_cmodStripTimer` fade callback)  
**Test:** `tests/visual_smoke.py`

---

## 9. Text styling: smaller unit-mode font + black-bordered text for H and axis labels

**Problem:**  
(a) Unit mode (U) pixel value text with black border is nice but font is ~15% too large.  
(b) Hover mode (H) pixel info text should also have the black-bordered style.  
(c) Axis direction arrow labels (x,y,z indicators) should also have black-bordered text.  

**Fix:**  
(a) Reduce the canvas `font` size for unit mode text by ~15% (e.g., from 12px to 10px, or whatever the current size × 0.85).  
(b) Apply `strokeText` + `fillText` pattern (black stroke, yellow fill) to hover tooltip text.  
(c) Apply same stroke+fill pattern to axis arrow labels drawn on canvas.  
**Files:** `_viewer.html` (unit mode rendering, hover rendering, axis label rendering)  
**Test:** `tests/visual_smoke.py`

---

## 10. ROI improvements: sharp borders, circular/freehand, more colors, table layout

**Problem:**  
(a) ROI rectangle borders and index numbers are blurry (anti-aliasing on subpixel coords).  
(b) Only rectangular ROIs — user wants circular and freehand.  
(c) Only 5 colors — need more.  
(d) Statistics are vertically arranged — should be a table with each ROI as a row.

**Fix:**  
(a) Snap ROI coordinates to integer pixels: `Math.round(x) + 0.5` for strokes, integer for fills. Use `ctx.imageSmoothingEnabled = false`.  
(b) Add ROI shape modes: press `S` (or a mode cycling key) in ROI mode to cycle rectangle → ellipse → freehand. Ellipse: drag defines bounding box. Freehand: collect points, close path on mouseup.  
(c) Expand `_roiColors` to 10+ colors (add orange, teal, red, lime, violet, etc.).  
(d) Rewrite `#roi-content` as an HTML `<table>` with columns: `#`, Color, Mean, Std, Min, Max, Count. Each ROI = one row. Style per frontend-designer skill.  
**Files:** `_viewer.html` (ROI drawing, ROI panel, CSS), `_server.py` (if ROI stats endpoint needs ellipse/freehand support)  
**Test:** `tests/visual_smoke.py`, `tests/test_api.py`

---

## 11. Diff mode colormap preview fix + revert + transparency + limited colormaps

**Problem:**  
(a) In diff mode, pressing `c` while hovering the center canvas doesn't update the colormap preview strip — it still shows the array colormaps.  
(b) User doesn't like the black-center RdBu. Revert to standard `RdBu_r`. Instead, make values at exactly 0 transparent.  
(c) The center canvas should only offer colormaps that make sense for difference maps (diverging colormaps), not all colormaps.

**Fix:**  
(a) Track which canvas is hovered (`_hoveredPane`). When `c` is pressed in diff mode and hover is over center pane, cycle `_diffColormapIdx` and show the diff colormap strip instead.  
(b) Revert A-B colormap to RdBu_r. Add alpha=0 for the exact zero bin (or a small ε around 0) in the diff rendering.  
(c) Define a `_diffColormaps` subset (diverging: RdBu_r, coolwarm, PiYG, BrBG, PuOr, seismic, etc.) and cycle only within that subset for the center pane.  
**Files:** `_viewer.html` (diff colormap logic, c key handler), `_render.py` (zero-transparency in diff render)  
**Test:** `tests/visual_smoke.py`, `tests/test_api.py`

---

## 12. Compare overlay mode: canvas position + size consistency

**Problem:** In compare mode when selecting overlay, the overlay appears as the 3rd canvas (not the middle one), and canvas sizes change compared to other diff modes.  
**Fix:** Ensure overlay canvas is positioned as the center pane (CSS order 1, between the two compared arrays). Use the same sizing logic as other center modes so canvas dimensions don't jump when switching between A-B / overlay / wipe.  
**Files:** `_viewer.html` (`compareScaleCanvases`, overlay pane ordering, CSS)  
**Test:** `tests/visual_smoke.py`

---

## 13. Wipe interaction: click-drag line, [ ] keys, overlay drag indicator

**Problem:**  
(a) The wipe vertical divider line is blurry.  
(b) Can't click-drag the wipe line directly on the canvas.  
(c) No `[` / `]` keyboard shortcuts to move wipe position.  
(d) In overlay mode, the blend slider below needs a visual indicator that it can be dragged.

**Fix:**  
(a) Snap wipe line to integer pixel (same +0.5 trick). Use lineWidth=2 for visibility.  
(b) Add mousedown/mousemove on the wipe canvas: if click is near the divider (±10px), start dragging `_wipeFrac`.  
(c) Add `[` / `]` keys in compare mode: decrease/increase `_wipeFrac` by 0.05, clamp to [0,1], redraw.  
(d) Style the overlay range input with a visible thumb (yellow, `cursor: grab`), add a subtle label "drag to blend".  
**Files:** `_viewer.html` (wipe drawing, key handler, overlay slider CSS)  
**Test:** `tests/visual_smoke.py`, update help overlay

---

## 14. Multi-array 3D mini-view (v key)

**Problem:** In single-array mode pressing `v` shows a nice 3D orientation widget with three slice planes. In multi-array (compare) mode, this widget doesn't appear.  
**Fix:** When multi-view is entered in compare mode (if applicable) or when viewing 3D+ arrays in compare, show the orientation widget. The widget needs to reference the active/hovered pane's dimensions and slice indices.  
**Files:** `_viewer.html` (`drawMvOrientation`, compare mode multi-view integration)  
**Test:** `tests/visual_smoke.py`

---

## 15. Compare mode zoom: clip canvases instead of vertically rearranging

**Problem:** When scrolling (zoom) in compare mode, at some point the canvases become vertically aligned instead of zooming into sub-regions like single-array mode does.  
**Fix:** Apply the same viewport-clipping architecture from single-array mode to compare mode. Each compare pane should have its own overflow-hidden viewport wrapper. When `userZoom` exceeds the available layout space, panes stay in their grid/row positions and clip internally. Pan within each pane (or synced pan across all panes).  
**Files:** `_viewer.html` (`compareScaleCanvases`, compare pane HTML/CSS, pan logic)  
**Test:** `tests/visual_smoke.py`

---

## 16. Comprehensive automated testing strategy

**Problem:** User spends too much time reporting small bugs, inconsistencies, and weird UI behavior. Playwright tests exist but miss many issues.  
**Fix:** Implement a systematic Playwright test suite that:  
(a) **Mode matrix testing**: For each mode (normal, multi-view, compare, diff, ROI, qMRI), verify: canvas renders, colorbars visible/correct, eggs positioned correctly, keyboard shortcuts work, no elements overflow.  
(b) **State transition testing**: Enter and exit every mode combination, verify no residual DOM state (hidden elements still visible, wrong classes, stale text).  
(c) **Zoom/scroll regression**: Zoom in/out, verify layout stability (no element jumps >2px).  
(d) **Screenshot diffing**: Use Playwright's `toHaveScreenshot` for pixel-level regression detection with threshold.  
(e) **Accessibility checks**: All interactive elements reachable by keyboard, visible focus rings.  
**Files:** `tests/test_browser.py` (extend), new `tests/test_mode_matrix.py`, `tests/conftest.py` (shared fixtures)  
**Test:** The tests themselves

---

## 17. Movable UI elements (colorbar on side, auto-collapse)

**Problem:** Dim bar, logo, name above canvas + colorbar/eggs below canvas limit vertical zoom space on small screens. User wants these elements to be repositionable (e.g., vertical colorbar on the side).  
**Fix:**  
- Add an auto-collapse mode: when canvas height hits 80%+ of window height, auto-hide top chrome (dim bar collapses to a thin strip, logo/name hide). Show on hover at top edge.  
- Support vertical colorbar mode: draw colorbar as a vertical strip on the left or right side of the canvas. Toggle with a keybind (e.g., `Shift+C`).  
- Store preference in `localStorage` for persistence.  
**Files:** `_viewer.html` (CSS, layout logic, colorbar drawing, key handler)  
**Test:** `tests/visual_smoke.py` — auto-collapse scenario

---

## Order of implementation

| # | Item | Complexity |
|---|------|-----------|
| 1 | Dim bar padding | S |
| 2 | Multi-view (x,y) label | S |
| 3 | Diff mode egg | S |
| 4 | Compare spurious colorbar | S |
| 5 | Pinch zoom speed | S |
| 6 | Zoom: don't widen past V-limit | S |
| 7 | Click-drag pan when zoomed | S |
| 8 | Colormap fade animation | M |
| 9 | Text styling (font, borders) | M |
| 10 | ROI improvements | L |
| 11 | Diff colormap preview + fixes | M |
| 12 | Compare overlay position | M |
| 13 | Wipe interaction | M |
| 14 | Multi-array 3D mini-view | M |
| 15 | Compare zoom clipping | L |
| 16 | Comprehensive testing | L |
| 17 | Movable UI elements | L |
