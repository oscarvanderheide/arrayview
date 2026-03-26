# Plan: Scrollable/Draggable qMRI Mosaic Mode

## Problem

In qMRI mosaic mode (`q` then `z`), all z-slices are rendered as horizontal strips per parameter map. With many z-slices (e.g. 20), each tile becomes tiny and illegible because `qvScaleAllCanvases()` caps the scale to fit everything in the viewport.

## Design

### Minimum tile size
- `MIN_MOSAIC_TILE_PX = 40` — if computed per-tile width falls below this, switch to overflow/pan mode.
- Per-tile width: `(v.lastW * cappedScale) / mosaic_cols` where `mosaic_cols = shape[_qmriMosaicZDim]`.

### Scrolling mechanism
- Use an inner wrapper div (`#qmri-mosaic-inner`) inside `#qmri-view-wrap` with `overflow: hidden` on the outer div.
- `transform: translate(_mosaicPanX, _mosaicPanY)` on the inner div for panning.
- Matches existing `canvas-viewport` / `canvas-inner` pattern in the codebase.

### Input mapping

| Input | Behavior |
|-------|----------|
| Scroll wheel | Changes slice index (unchanged) |
| Left-click drag | Pan the mosaic view |
| `+` / `-` keys | Zoom (can exceed viewport, min tile enforced) |
| `0` key | Reset zoom + reset pan to center |
| Minimap click | Jump to position |

## Implementation Steps

### Step 1: State variables (~line 1436)
```javascript
let _mosaicPanX = 0, _mosaicPanY = 0;
let _mosaicPanDrag = null; // {x, y, panX0, panY0}
let _mosaicOverflowing = false;
```

### Step 2: Modify `qvScaleAllCanvases()` mosaic branch (~line 8572)
- After computing `absMaxScale`, calculate per-tile width.
- If `tileW < MIN_MOSAIC_TILE_PX`, compute `minTileScale = MIN_MOSAIC_TILE_PX * shape[_qmriMosaicZDim] / ready[0].lastW`.
- Use `Math.max(absMaxScale, minTileScale)` as effective scale.
- Apply `userZoom` on top of minimum scale.
- Remove the `userZoom = 1.0` reset that prevents overflow.
- Set `_mosaicOverflowing = true` when CSS width exceeds `maxCanvasW`.

### Step 3: CSS for scroll container (~line 540)
- `#qmri-view-wrap` gets `overflow: hidden` when mosaic overflows.
- Inner wrapper `#qmri-mosaic-inner` receives `transform: translate(...)`.

### Step 4: Pan/clamp/apply functions
- `_mosaicClampPan()` — clamp pan so viewport stays within content bounds.
- `_mosaicApplyPan()` — set transform on inner div.
- Pattern matches `_clampPan()` (line 4940) and `_applyCanvasPan()` (line 4955).

### Step 5: Drag events on qMRI canvases (~line 7475)
- **mousedown**: If overflowing, start drag, record start position.
- **mousemove** (document-level): Update pan, clamp, apply.
- **mouseup**: Clear drag state.
- Cursor: `grab` when hovering overflowing mosaic, `grabbing` during drag.

### Step 6: Reset pan on mode transitions
- When toggling mosaic off (line 5797-5806), reset `_mosaicPanX = _mosaicPanY = 0`.
- When `qvScaleAllCanvases()` determines content fits again, reset pan.

### Step 7: Minimap for overflow
- Small fixed-position div in bottom-right corner.
- Shows viewport rectangle indicating visible portion.
- Click/drag on minimap jumps pan position.
- Alternative: simple horizontal scroll indicator bar instead of full minimap.

### Step 8: Vertical overflow
- Same mechanism for many parameter rows (6+).
- `_mosaicPanY` controls vertical scroll, clamped by `_mosaicClampPan()`.

### Step 9: Keyboard zoom
- `+` increases `userZoom`, tiles get larger, may increase overflow.
- `-` decreases `userZoom`, tiles never below `MIN_MOSAIC_TILE_PX`.
- `0` resets to fit-or-minimum state, resets pan to center.

## Edge Cases
- Single z-slice: no overflow, behaves as before.
- Window resize: `qvScaleAllCanvases()` rechecks overflow, resets pan if content fits.
- Switching between mosaic and normal qMRI: reset all mosaic pan state.

## Files to Modify
Only `src/arrayview/_viewer.html`:
1. CSS section (~line 540-564)
2. State variables (~line 1436)
3. `qvScaleAllCanvases()` mosaic branch (~line 8572-8601)
4. `_rebuildQmriMosaicLayout()` (~line 8406-8475)
5. qMRI canvas event handlers (~line 7475)
6. Mosaic toggle handler (~line 5795-5822)
7. New functions: `_mosaicClampPan()`, `_mosaicApplyPan()`, minimap
