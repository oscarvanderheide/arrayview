# TODO Batch — 2026-03-24

Four independent UI improvements to the arrayview viewer.

## 1. Miniviewer hand cursor

**Problem:** The miniviewer (top-right mini-map when zoomed in) uses `cursor: crosshair`, which doesn't signal that the yellow viewport rectangle is draggable.

**Change:**
- CSS: `#mini-map { cursor: grab; }`
- JS: On `mousedown`, set `cursor: grabbing` on `#mini-map`. On `mouseup`, revert to `grab`.
- No functional change — click/drag already works.

**Files:** `_viewer.html` (CSS line ~619, JS lines ~8500-8512)

## 2. Subtler canvas border

**Problem:** Border toggle (`b`) uses `2px solid var(--highlight)` — too thick, too bright (white in dark theme).

**Change:**
- Update the existing `--canvas-border` variable in all 4 theme blocks to `rgba(180,180,180,0.35)` (same value for all themes). This replaces the current per-theme semi-transparent values.
- `.canvas-bordered canvas, .canvas-bordered { outline: 1px solid var(--canvas-border); }` (was `2px solid var(--highlight)`)
- `.mv-pane.canvas-bordered { outline: 1px solid var(--canvas-border); }` (was `1px solid var(--border)`)
- Check for any other uses of `--canvas-border` and update if needed.

**Files:** `_viewer.html` (CSS lines ~9-42 for themes, line ~69 for normal border, line ~404 for multiview border)

## 3. Multiview rotation as global axis swap

**Problem:** Pressing `r` in multiview only rotates the single pane whose `sliceDir === activeDim`. The other two panes that share those axes don't update, creating an inconsistent view.

**Change:**
- `r` in multiview swaps the global `dim_x`/`dim_y` variables AND updates all `mvViews` pane definitions in-place (no DOM teardown/rebuild, no WebSocket reconnection).
- After swapping globals, recompute each `mvViews[i].dimX` and `mvViews[i].dimY` from the new `mvDims` ordering: `dims = [dim_y, dim_x, third]` where third is the dim that is neither dim_x nor dim_y.
- The 3 defs pattern stays the same: `[{dimX:dims[1], dimY:dims[0]}, {dimX:dims[2], dimY:dims[0]}, {dimX:dims[2], dimY:dims[1]}]`.
- Re-render all 3 panes via `mvRender(v)` for each view. Update axis labels.
- Transfer flip state correctly during the swap.

**Behavior:** If dims are `[y=0, x=1, z=2]` and you press `r`, globals swap so dim_x becomes 0 and dim_y becomes 1. `dims` becomes `[1, 0, 2]`. All 3 panes get updated dimX/dimY and re-render.

**Files:** `_viewer.html` (JS lines ~4749-4762 for `r` handler, lines ~5956-6129 for `enterMultiView`)

## 4. Space play: orange playing dim + independent navigation

**Problem:** During playback, the playing dim is highlighted in yellow (same as active dim), and you can't navigate other dims while playing.

### CSS changes

New variable per theme (insert near existing `--active-dim`):
- Dark: `--playing-dim: #d08770;`
- Light: `--playing-dim: #c05020;`
- Solarized: `--playing-dim: #cb4b16;`
- Nord: `--playing-dim: #d08770;`

New class (insert near `.active-dim` around line ~85):
```css
.playing-dim { color: var(--playing-dim); font-weight: bold; }
.playing-dim .dim-track-fill { background: var(--playing-dim); }
```

### JS changes

**New state:** `let playingDim = -1;` (-1 means not playing)

**`togglePlay()` (~line 4010):** Sets `playingDim = current_slice_dim` before starting animation.

**`stopPlay()` (~line 3976):** Sets `playingDim = -1`.

**`playNext()` (~line 3989):** Change `indices[current_slice_dim]` to `indices[playingDim]` and `shape[current_slice_dim]` to `shape[playingDim]`. This decouples the animation from `current_slice_dim` so that h/l navigation (which updates `current_slice_dim`) doesn't redirect the animation.

**`renderInfo()` (~line 3376):** Priority for dim label classes:
1. `i === playingDim` → `.playing-dim` (orange takes precedence, even when playingDim === activeDim)
2. `i === activeDim` → `.active-dim` (yellow)
3. Otherwise → default styling

**Navigation during playback:** Arrow key handlers (j/k or up/down) operate on `activeDim` as usual. The user can click dim labels or use h/l to change `activeDim` while `playingDim` continues animating independently. Pressing space stops playback.

**Edge case — j/k on the playing dim:** If the user navigates `activeDim` to the same dim that is playing and presses j/k, stop playback (the manual navigation overrides the animation).

**Compare mode:** Playing-dim styling applies in compare mode too — `renderInfo()` is a single function that handles all modes, so the `.playing-dim` class works everywhere.

**Files:** `_viewer.html` (CSS themes ~9-42, new class near ~85, JS `togglePlay` ~4010, `stopPlay` ~3976, `playNext` ~3989, `renderInfo` ~3376, key handlers)

## Testing

- **Miniviewer cursor:** Browser test — verify cursor changes on hover/drag over mini-map.
- **Border:** Browser test — verify outline is thinner and gray.
- **Multiview rotate:** Browser test — verify all 3 panes update after `r`.
- **Play color:** Browser test — verify playing dim gets orange class, active dim stays yellow, both can coexist. Verify j/k on playing dim stops playback.

## Out of scope

- Changing play FPS UI
- Changing miniviewer position or size
- Any other keyboard shortcut changes
