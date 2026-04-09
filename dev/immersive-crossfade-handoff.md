# Immersive Crossfade — Handoff Notes

## What We're Building

When the user zooms in with trackpad (cmd+scroll or pinch), the transition from normal view to immersive view should feel seamless. Currently there's a discrete 3-phase animation (fade → grow → fade, ~460ms) that works for a single keypress (`=`) but is wrong for continuous trackpad zoom.

**The desired behavior (confirmed by user):**
1. Start at default zoom: slice fits pane, pane is maximum size with title/dimbar/colorbar visible
2. As user zooms in: **the pane grows continuously** — title shrinks from above, dimbar/colorbar shrink from below
3. The slice always fills the growing pane — **no overflow, no minimap** during the transition
4. First half: chrome fades out while collapsing. At midpoint (all invisible), layout switches to immersive.
5. Second half: dimbar/colorbar fade in at their immersive floating positions
6. Only AFTER reaching full immersive and zooming further: overflow + minimap appear
7. Reverse direction (zoom out) is symmetric

## Current State of the Code

The v2 implementation is in place and partially working. Commits on `main` after `717955d`.

### What Works
- `_crossfadeCollapseP` variable correctly drives all 6 scale functions to interpolate bottom reserves
- `_applyImmersiveCrossfade()` runs BEFORE scalers in `ModeRegistry.scaleAll()`
- Chrome elements (title, dimbar, colorbar) collapse their heights via `maxHeight` and fade opacity
- Layout class toggle at p=0.5 when chrome is invisible
- Second half fade-in at immersive positions
- `=`/`-`/`0`/scroll handlers all wired to use crossfade
- Old `_enterImmersive`/`_exitImmersive`/`_immersiveAnimating`/`_animOpacity` removed
- All 67 visual smoke tests pass

### What's Still Wrong
The user reports the transition is "close but not quite there." They want to describe exact symptoms in the next session. Known issues likely involve:
- Possible timing/coordination between the chrome collapse and scale function recalculation
- The `uiReserveV()` function measures actual `offsetHeight` of title + dimbar — as we collapse them via `maxHeight`, these measurements should decrease, but there may be margin/padding not being collapsed that keeps the measured height larger than expected
- The colorbar's space is handled differently than title/dimbar (via `_crossfadeCollapseP` in scale functions rather than `uiReserveV()`), which might cause misalignment

## Architecture

### Key Function: `_applyImmersiveCrossfade()`
- Located near line ~14500 in `_viewer.html`
- Called at the TOP of `ModeRegistry.scaleAll()`, before any scaler runs
- Computes `p = clamp((userZoom - normalFit) / (immersiveFit - normalFit), 0, 1)`
- `collapseP = min(1, p * 2)` — 0→1 during first half
- `fadeIn = max(0, (p - 0.5) * 2)` — 0→1 during second half
- Sets `_crossfadeCollapseP = collapseP` for scale functions
- Caches natural heights (`_crossfadeTitleH`, `_crossfadeInfoH`, `_crossfadeCbH`) when crossfade starts
- Collapses chrome via `el.style.maxHeight = (naturalH * (1 - collapseP)) + 'px'`
- Toggles `fullscreen-mode` class at p=0.5
- Fades in immersive chrome during second half

### How Pane Size is Computed
Each scale function computes available height roughly as:
```
maxH = window.innerHeight - uiReserveV() - bottomReserve
```
- `uiReserveV()` (line ~2368): measures actual `offsetHeight` of `#array-name` and `#info`, plus 30px padding. Returns `max(130, measured)`. **Note the min of 130** — this might prevent the reserve from actually decreasing during collapse!
- `bottomReserve`: each scale function has its own (80px for single, 40px for compare, etc.). All now multiplied by `(1 - _crossfadeCollapseP)`.

### Scale Functions Updated
1. `scaleCanvas` — single pane
2. `compareScaleCanvases` — compare mode (2+ arrays side by side)
3. `compareQmriScaleAllCanvases` — compare + qMRI grid
4. `compareMvScaleAllCanvases` — compare + multiview grid
5. `qvScaleAllCanvases` — qMRI 3/5-panel
6. `mvScaleAllCanvases` — multiview 3-plane

### Zoom Sensitivity (separate, completed feature)
- Scroll zoom uses clamped exponential: `Math.pow(2, -clampedDelta * _zoomSensitivity)`
- `_zoomSensitivity = 0.02` (tunable, persisted in state)
- Safari `GestureEvent` handlers added for pinch-to-zoom
- Touch pinch uses direct ratio (no dampening)
- All working correctly — don't change these

## Failed Approaches (v1)

### Attempt 1: Opacity-only crossfade, toggle at p=0.5
- Just controlled chrome opacity, toggled `fullscreen-mode` at p=0.5
- **Failed:** The class toggle changes ~10 layout properties at once (title collapse, padding, dimbar/colorbar repositioning). Visible canvas jump no matter where we toggled.

### Attempt 2: Toggle at p=1 instead of p=0.5
- Moved the toggle to the boundary when all chrome is at opacity 0
- **Failed:** At p=1, pane instantly jumps to immersive size. Chrome appears instantly. No smooth transition.

### Attempt 3: Pre-collapse title before toggle
- Tried to collapse title with inline styles before adding fullscreen-mode
- **Failed:** Title collapsed too early (at any p > 0), causing instant disappearance. Other layout properties (padding, positioning) still caused jumps at the toggle.

## Key Lessons

1. **The pane must grow — not just chrome fading.** The user explicitly rejected opacity-only approaches. The pane needs to physically expand as chrome gives up space.

2. **`fullscreen-mode` CSS class is too coarse.** It changes too many things at once for a smooth transition. The crossfade progressively handles each change separately.

3. **`uiReserveV()` has a `max(130, h)` floor.** This might prevent the reserve from decreasing during chrome collapse. Investigate whether this is clamping the available height.

4. **Scale functions run AFTER the crossfade.** The crossfade sets `_crossfadeCollapseP` and collapses elements, then scalers measure the DOM and compute sizes. This order is critical — v1 had it backwards.

5. **The math works out linearly.** Since p is linear in userZoom, and fit-zoom is linear in available height (when height binds), collapsing chrome proportional to p makes the slice exactly fill the pane at every intermediate point — in theory.

## Files

- **Spec:** `docs/superpowers/specs/2026-04-09-smooth-immersive-crossfade-v2-design.md`
- **Plan:** `docs/superpowers/plans/2026-04-09-smooth-immersive-crossfade-v2.md`
- **Implementation:** all in `src/arrayview/_viewer.html`
- **Tests:** `tests/visual_smoke.py` (all pass, but tests use keyboard zoom, not trackpad — manual testing required)

## What to Do Next

1. Have the user describe exactly what they see at each stage of zooming
2. Check if `uiReserveV()`'s `max(130, h)` floor is preventing the pane from growing during early crossfade
3. Debug by logging `p`, `collapseP`, `uiReserveV()`, `maxH`, and actual element heights during a zoom gesture
4. Consider whether the colorbar also needs its `maxHeight` collapsed (currently only title + dimbar are collapsed; colorbar space is handled via `_crossfadeCollapseP` in scale functions, but the colorbar element itself may still take space in the DOM flow)
