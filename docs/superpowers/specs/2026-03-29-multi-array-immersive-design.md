# Multi-Array Immersive View

## Context

Immersive mode currently works only for single-array viewing. When multiple arrays are loaded (compare mode), immersive is blocked in the keyboard handler (lines 7463-7465 of `_viewer.html`). The goal is to extend immersive mode to multi-array compare mode, producing a result almost identical to mosaic-in-immersive: panes touching with a gray border, array names hidden, colorbar and dim bar as floating islands.

## Behavior

### Entry / Exit
- Same auto-trigger logic as single-array: enters immersive when height is the binding constraint
- `=` to enter, `-` to exit manually
- Works for all compare sub-modes: normal compare, diff (wipe/checkerboard/flicker/overlay)

### Layout
- Panes fill viewport, touching with gray border (same style as mosaic immersive — `--canvas-border` color, ~1-2px)
- Gap reduced from 18px to 0, border between panes instead
- `#wrapper` padding reset to 0

### Hidden Elements
- Array name titles (`.compare-title`) — hidden (opacity 0, not just overlaid pills)
- Per-pane inline colorbars (`.compare-pane-cb-island`) — hidden (replaced by island)

### Islands (Floating Chrome)
- **Dim bar** (`#info`): top-center of viewport, glassmorphic, same as single-array immersive
- **Colorbar** (`#slim-cb-wrap`): bottom-center of viewport, glassmorphic, same as single-array immersive
- Normal multi-array: one shared colorbar island
- Diff mode: existing per-side colorbars become islands (left/right positioned)
- Diff pane stays visible as a regular pane in the grid; its colorbar (when applicable) also as island

### Minimaps
- Per-pane minimaps (`.cmp-mini-map`), positioned top-right of each pane (already exist at `top:6px; right:6px`)
- Show only when zoomed (same `comparePan.overflows` condition)
- No change to minimap rendering logic — just ensure they display correctly in immersive layout

### Animation
- Same 3-phase animation as single-array immersive (~460ms total):
  1. **Fade-out** (150ms): fade out titles, inline colorbars, dim bar, colorbar
  2. **Grow** (160ms): switch to immersive layout, animate canvas zoom from normal fit → immersive fit
  3. **Fade-in** (150ms): fade in island chrome (dim bar, colorbar)
- Exit animation mirrors entry (fade-out islands → shrink → fade-in normal chrome)

### Zoom Calculation
- `_calcAutoFitZoom()` needs no changes — it calculates based on canvas-viewport dimensions, and the compare layout system handles per-pane sizing
- The compare scaler (`ModeRegistry.scaleAll()` for compare mode) already computes per-pane dimensions from available space

## Implementation

### 1. Lift compare-mode block (keyboard handler ~line 7463)
Remove `compareActive` (or rather, the absence of it — currently the auto-trigger condition checks `!multiViewActive && !qmriActive && !compareMvActive && !compareQmriActive` but plain `compareActive` is implicitly allowed). The actual block may be elsewhere — verify the exact condition that prevents immersive in compare mode.

### 2. CSS: `body.fullscreen-mode` compare rules (~line 745)
- Set `#compare-panes` gap to 0
- Add border between panes: `.compare-pane.active + .compare-pane.active { border-left: 1-2px solid var(--canvas-border); }`
- For grid layout (wrapped rows): add top border on wrapped panes
- Hide `.compare-title` (opacity 0, max-height 0)
- Hide `.compare-pane-cb-island` (opacity 0)
- Remove `.compare-canvas-area` padding

### 3. `_enterImmersive()` / `_exitImmersive()` (~lines 12009, 12056)
- Add compare title elements and per-pane colorbars to the fade-out/fade-in element lists
- On exit: reset any inline styles added to compare elements
- No other changes needed — these functions are already mode-agnostic via `ModeRegistry.scaleAll()`

### 4. `_positionFullscreenChrome()` (~line 12148)
- When `compareActive`: position dim bar and colorbar relative to the `#compare-view-wrap` bounding rect instead of `#canvas-viewport`
- Minimap positioning is handled per-pane (already absolute within each pane), so no changes needed there

### 5. `_shouldEnterImmersive()` (~line 11965)
- Remove the canvas-viewport size check or adapt it for compare mode (use `#compare-view-wrap` dimensions)
- Ensure auto-trigger logic works with compare layout dimensions

## Key Files
- `src/arrayview/_viewer.html` — all changes are frontend-only

## Verification
1. Load 2 arrays: `av.view(arr1, arr2)` — verify immersive auto-triggers when height-bound
2. Load 3-4 arrays — verify grid layout in immersive, panes touching with gray border
3. Verify `=`/`-` manually toggles immersive
4. Verify dim bar island at top-center, colorbar island at bottom-center
5. Zoom in — verify per-pane minimaps appear at top-right of each pane
6. Test diff mode — verify diff pane visible, per-side colorbars as islands
7. Test animation smoothness (fade-out → grow → fade-in)
8. Resize window — verify auto-exit when height no longer binding
9. Test with mosaic+compare combinations to ensure no conflicts
