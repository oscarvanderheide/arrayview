# TODO Plan — March 2026

Items from `TODO.md`. Each item = one commit. Ordered by complexity (small → large).

---

## 1. `log` label — vertical text in histogram ✓ DONE

**Problem:** The "log" label in the histogram area extends beyond the colorbar width.  
**Fix:** Rotate canvas context 90° and draw the text vertically (bottom to top) so it fits within the bar.  
**Files:** `_viewer.html` (`_drawHistogramBarsOnColorbar`)  
**Test:** `tests/visual_smoke.py` — add scenario, run smoke

---

## 2. Remove fading `showStatus` text on colormap `c` key ✓ DONE

**Problem:** When pressing `c` to switch colormap, both a fading status text AND the colormap strip preview appear. The strip is enough.  
**Fix:** Remove the `showStatus(...)` call in the `c` key handler. Keep `showColormapStrip()`.  
**Files:** `_viewer.html` (keydown `c` branch)  
**Test:** Manual smoke / existing browser test

---

## 3. Array names in compare mode ✓ DONE

**Problem:** In compare mode, the logo area still shows the first array's name. And compare-title elements already show the name above each canvas — but the top bar is redundant.  
**Fix:** When `compareActive`, set `#array-name-text` to "comparing N arrays" (where N = number of panes). Update it on compare entrance, exit, and layout change.  
**Files:** `_viewer.html` (`enterCompareModeBySid`, `enterCompareModeByMultipleSids`, `exitCompareMode`, `applyCompareLayout`)  
**Test:** `tests/visual_smoke.py` + screenshot

---

## 4. Unify compare center-pane modes into a single `X` cycle/picker

**Problem:** R (registration overlay), W (wipe), and X (diff: A-B, |A-B|, |A-B|/|A|) are three separate keybinds that all show a "third canvas in the middle." They should be unified.  
**Design:**  
- `X` in 2-pane compare mode opens a **compare-type strip** (horizontal row of mode thumbs, like the colormap strip) and advances to the next mode.  
- Modes in cycle order: **off → A-B → |A-B| → |A-B|/|A| → overlay → wipe → off**  
- Each thumb shows: a text label and a small colored indicator  
- `W` in compare mode removed (becomes regular Lebesgue or disabled)  
- `R` in compare mode removed; `R` outside compare still toggles RGB  
- `Z` still focuses the center pane in any active compare-type mode  
- Update help overlay, README, smoke test  
**Files:** `_viewer.html` (key handlers, `toggleWipeMode`, `toggleRegistrationMode`, compare state vars), `README.md`  
**Test:** `tests/visual_smoke.py` scenarios, `tests/test_browser.py`

---

## 5. Better diff colormaps: black-center diverging + afmhot

**Problem:** A-B uses RdBu_r (white center, bad on dark theme). |A-B| and |A-B|/|A| use viridis.  
**Fix:**  
- A-B: use a diverging colormap that goes red → black → blue (or similar dark-center diverging). Create a custom gradient or pick `PuOr_r` inverted or define a custom stops array. Specifically want black at 0.  
- |A-B| and |A-B|/|A|: use `afmhot` (black at 0, bright at max).  
- Allow pressing `c` while hovering the diff/overlay center pane to cycle its colormap independently.  
**Files:** `_viewer.html` (`fetchAndDrawDiff`, diff colormap logic, maybe new `_diffColormapIdx`)  
**Test:** smoke screenshot, new scenario

---

## 6. ROI mode redesign — multiple ROIs + CSV export

**Problem:** Current `A` key ROI mode draws a single rectangle but the UX is broken (gives square ROI). User wants:  
- A dedicated ROI drawing mode (press some key to enter/exit)  
- Draw multiple rectangle ROIs by dragging on the canvas  
- Each ROI shows: mean, std, min, max, num_voxels  
- Export all ROI stats to CSV (keybind or button)  
- Remove the old broken drag-to-ROI behavior from click-drag

**Design:**  
- Reuse `A` key to toggle multi-ROI mode  
- In ROI mode: left-drag draws a new rectangle ROI  
- Each ROI rendered as a yellow rectangle overlay with stats tooltip  
- A stat panel or tooltip shows live stats (fetch from server `/roi_stats` endpoint)  
- Press `Escape` to cancel in-progress ROI, `Delete`/`Backspace` to remove last ROI  
- `E` or `Ctrl+E` to export all ROI stats as CSV (browser download)  
**Files:** `_viewer.html`, `_server.py` (ROI stats endpoint), `_render.py` (stats helper)  
**Test:** `tests/test_api.py` (ROI stats endpoint), `tests/visual_smoke.py`

---

## 7. Minimap + true pan-zoom (zoom past canvas limits)

**Problem:** Currently zoom is capped when canvas would push other UI elements out. User wants:  
- Continue zooming beyond the canvas size cap — instead of enlarging canvas, zoom shows a sub-region  
- Minimap shows a yellow rectangle indicating the currently visible sub-region  
- Pan within the zoomed view is unchanged (right-drag already pans when zoomed)

**Design:**  
- Track `imageZoom` (the actual pixel zoom level) separately from `canvasZoom` (CSS display zoom bounded by layout)  
- When `imageZoom` exceeds the cap, crop the rendered region: compute pixel offset within the array slice and send as `?zoom=<scale>&pan_x=<px>&pan_y=<py>` to the render endpoint (or crop client-side from cached image data)  
- Minimap: show only when `imageZoom > threshold` (e.g. > 1.1× cap). Position at bottom-right of canvas. Yellow rectangle = visible region.  
**Files:** `_viewer.html` (`scaleCanvas`, zoom key handlers, render fetch), `_server.py` (support crop params), `_render.py` (crop/downsample server-side)  
**Test:** `tests/visual_smoke.py`, `tests/test_api.py`

---

## Order of implementation

1. ✅ Plan (this file)
2. `log` vertical
3. Remove `showStatus` on `c`
4. Array names in compare mode
5. Unify compare modes (X picker strip)
6. Diff colormaps (black + afmhot)
7. ROI redesign
8. Minimap zoom

Items 2–4 are in a session.  Items 5–8 require larger effort and may span sessions.
