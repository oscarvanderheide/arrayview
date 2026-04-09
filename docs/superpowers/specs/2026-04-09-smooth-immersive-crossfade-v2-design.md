# Smooth Immersive Crossfade v2

**Date:** 2026-04-09
**Status:** Approved
**Supersedes:** `2026-04-09-smooth-immersive-crossfade-design.md` (v1 had layout jump issues)

## Problem

Entering/exiting immersive mode during continuous trackpad zoom needs to feel seamless. The v1 approach tried to control only chrome opacity during the crossfade and toggle the `fullscreen-mode` CSS class at a single point. This caused visible layout jumps because the class toggle changes ~10 layout properties at once (title collapses, padding changes, dimbar/colorbar repositioning). No matter where we toggled (p=0.5 or p=1), the layout shift was visible.

## Key Insight

The pane must **grow continuously** during the crossfade. The slice must always fill the pane — no overflow, no empty space, no minimap until fully immersive and zooming further. This means chrome elements must **physically collapse** (not just fade) to give the pane more room.

## How It Works

### Progress value

```
immFit = _calcAutoFitZoom(w, h, true)   // immersive fit zoom
normFit = _normalFitZoom                 // normal fit zoom  
p = clamp((userZoom - normFit) / (immFit - normFit), 0, 1)
```

### First half (p: 0 → 0.5) — collapse + fade out

Chrome elements physically shrink AND fade out. The pane grows to fill freed space. The slice fills the growing pane (no overflow).

- **Title (`#array-name`):** height collapses from natural to 0. Opacity fades 1→0. Margin/padding collapse proportionally. Uses inline styles (`maxHeight`, `overflow: hidden`).
- **Dimbar (`#info`):** height collapses from natural to 0. Opacity fades 1→0. Same technique.
- **Colorbar area:** The 80px `bottomReserve` in `scaleCanvas` accounts for colorbar + eggs + gaps. This needs to interpolate from 80→0 during the first half.

**Why the slice fills the pane:** `uiReserveV()` already measures actual `offsetHeight` of title and dimbar. As we collapse them, it returns less, `scaleCanvas` computes more `maxH`, and the canvas fills the larger space. The `bottomReserve` interpolation handles the colorbar space. Since p is linear in userZoom, and the fit-zoom calculation is linear in available height, the two match — the slice exactly fills the pane at every intermediate point.

### Midpoint (p = 0.5) — invisible layout switch

All chrome at opacity 0 and collapsed. Toggle `fullscreen-mode` class. The class sets the same collapsed state + floating positioning for dimbar/colorbar. Since chrome is invisible and already collapsed, the switch has zero visual impact.

### Second half (p: 0.5 → 1.0) — fade in at immersive positions

- **Dimbar:** fades in 0→1 at floating position (top of pane, blur backdrop)
- **Colorbar:** fades in 0→1 at floating position (bottom of pane, blur backdrop)  
- **Title:** stays hidden (opacity 0, hidden by `fullscreen-mode` CSS)
- **Pane continues growing** as userZoom increases toward immersive fit

### Fully immersive (p ≥ 1)

Chrome fully visible at immersive positions. Further zooming causes slice to overflow pane → minimap appears. This is the existing immersive zoom behavior.

### Reverse (zooming out)

Symmetric. As p decreases:
- p: 1 → 0.5: dimbar/colorbar fade out at immersive positions
- p = 0.5: remove `fullscreen-mode` class (chrome invisible, no visual impact). Restore inline collapse styles so elements stay collapsed.
- p: 0.5 → 0: chrome expands back into their normal positions AND fades in. Pane shrinks.
- p = 0: fully normal, inline styles cleared.

## Implementation Details

### Chrome collapse technique

For each chrome element (title, dimbar), progressively set:
```javascript
const collapseP = Math.min(1, p * 2);  // 0→1 during first half
el.style.maxHeight = (naturalH * (1 - collapseP)) + 'px';
el.style.overflow = 'hidden';
el.style.opacity = 1 - collapseP;
// Also collapse margin/padding proportionally
```

Cache `naturalH` (element's natural height) when the crossfade starts (p transitions from 0 to >0) so it doesn't change during the crossfade.

### Bottom reserve interpolation

`scaleCanvas` uses `bottomReserve = _fullscreenActive ? 0 : 80`. During the crossfade, this needs to interpolate. Introduce `_crossfadeBottomReserve`:

```javascript
// In scaleCanvas:
const bottomReserve = _fullscreenActive ? 0 : _crossfadeBottomReserve ?? 80;
```

The crossfade function sets `_crossfadeBottomReserve`:
```javascript
const collapseP = Math.min(1, p * 2);
_crossfadeBottomReserve = 80 * (1 - collapseP);  // 80→0 during first half
```

### Colorbar collapse

The colorbar (`#slim-cb-wrap`) is below the pane. It needs to physically collapse during the first half, same as title/dimbar. However, its space is accounted for by `bottomReserve`, not `uiReserveV()`. The `_crossfadeBottomReserve` interpolation handles the space calculation. The colorbar's visual collapse (height + opacity) matches.

### Call site

`_applyImmersiveCrossfade()` runs BEFORE the scalers in `ModeRegistry.scaleAll()`. This ensures:
1. Chrome elements are collapsed to the right height
2. `_crossfadeBottomReserve` is set
3. THEN scale functions measure layout and compute canvas size

### `=` key

Sets `userZoom = immersiveFitZoom` and calls `_scaleAllWithAnim()`. Each animation frame calls `ModeRegistry.scaleAll()` which runs the crossfade. Chrome collapses and fades as the animated zoom sweeps through the zone.

### `_shouldEnterImmersive()` guard

The crossfade only activates when `_shouldEnterImmersive()` returns true (height is the binding constraint). If width binds, zooming past normalFit just causes overflow as usual — no immersive transition.

### Dimbar width

The dimbar keeps its normal width in immersive mode (CSS `padding` override removed from `fullscreen-mode #info`).

## What Gets Removed (same as v1)

- `_enterImmersive()` — replaced by crossfade
- `_exitImmersive()` — replaced by crossfade reversing
- `_immersiveAnimating` flag
- `_animOpacity()` — no callers remain

## Edge Cases

- **Window resize:** `immFit` recalculates, p adjusts, chrome repositions.
- **Rapid direction change:** Stateless — p from current zoom, no animation state.
- **Compare/multiview/qMRI:** Same collapse + fade pattern for mode-specific chrome.
- **`0` key reset:** Animates zoom to normalFit. Crossfade reverses naturally.
