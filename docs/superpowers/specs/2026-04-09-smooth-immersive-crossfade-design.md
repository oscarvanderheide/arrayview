# Smooth Immersive Crossfade

**Date:** 2026-04-09
**Status:** Approved

## Problem

The immersive mode transition uses a 3-phase animation (fade chrome out → grow canvas → fade chrome in, ~460ms) triggered as a discrete event. This works well for a single keypress (`=`) but is completely wrong for continuous trackpad zoom — the animation blocks input, creates a jarring mode switch, and breaks the flow of the gesture.

## Solution

Replace the discrete animation with a **zoom-driven crossfade**: a progress value `p` is computed from the current zoom level, and `p` continuously controls the opacity and layout of all chrome elements. The transition is stateless — no animation timers, no multi-phase state machine.

### Crossfade Progress

```
immFit = _calcAutoFitZoom(w, h, true)   // immersive fit zoom
normFit = _normalFitZoom                 // normal fit zoom
p = clamp((userZoom - normFit) / (immFit - normFit), 0, 1)
```

The zone spans from `_normalFitZoom` to `_immersiveFitZoom` (mode A: crossfade before threshold).

### Two-Half Fade with Invisible Layout Switch

The binary `fullscreen-mode` CSS class cannot be partially applied. The design hides the switch at the midpoint when all chrome is at opacity 0:

| p range | Layout | Chrome behavior |
|---------|--------|----------------|
| 0 | Normal | Full chrome, no transition |
| 0 → 0.5 | Normal | Title, dimbar, colorbar fade out (opacity 1→0) |
| 0.5 | **Switch** | Add/remove `fullscreen-mode` class. All chrome at opacity 0 — switch is invisible |
| 0.5 → 1.0 | Immersive | Dimbar, colorbar fade in at floating island positions (opacity 0→1). Title stays hidden. |
| 1 | Immersive | Fully settled immersive layout |

Opacity formulas:
```
fadeOut = max(0, 1 - p * 2)       // 1→0 over first half
fadeIn  = max(0, (p - 0.5) * 2)   // 0→1 over second half
```

Reverse direction (zoom out) is symmetric — same zone, same formulas, p just decreases.

## Chrome Element Behavior

### Title (`#array-name`)
- Opacity = `fadeOut` (fades during first half)
- After layout switch (p >= 0.5): hidden by `fullscreen-mode` CSS
- On zoom out below p=0.5: reappears as fadeOut increases

### Dimbar (`#info`)
- **Width stays constant** — no resize between normal and immersive
- p < 0.5 (normal layout): opacity = `fadeOut`, bottom-docked position
- p >= 0.5 (immersive layout): opacity = `fadeIn`, floating `position: fixed; top: 8px; left: 50%` with blur backdrop
- Blur backdrop, border, border-radius applied at layout switch when opacity is 0

### Colorbar (`#slim-cb-wrap`)
- Same fade pattern as dimbar
- p < 0.5: opacity = `fadeOut`, inline docked position
- p >= 0.5: opacity = `fadeIn`, floating island position with blur backdrop

### Compare/multiview/qMRI chrome
- Compare titles (`.compare-title`): same fadeOut/fadeIn pattern
- Per-pane colorbars (`.compare-pane-cb-island`): same pattern, `_reconcileCbVisibility()` called at layout switch
- MV colorbar (`#mv-cb-wrap`), qMRI colorbars (`.qv-cb-island`): same pattern

### Mode eggs (`.mode-badge`)
- No special crossfade — inherit backdrop from layout switch

## Crossfade Controller

A single function computes and applies crossfade state. Called on every zoom update.

```js
function _applyImmersiveCrossfade() {
    if (!_shouldEnterImmersive()) {
        // Width-binding: no immersive possible. Ensure normal layout.
        if (_fullscreenActive) { /* snap to normal */ }
        return;
    }

    const immFit = _calcAutoFitZoom(w, h, true);
    const normFit = _normalFitZoom;
    const p = Math.max(0, Math.min(1, (userZoom - normFit) / (immFit - normFit)));

    const fadeOut = Math.max(0, 1 - p * 2);
    const fadeIn  = Math.max(0, (p - 0.5) * 2);

    // Layout switch at midpoint
    const shouldBeFullscreen = p >= 0.5;
    if (shouldBeFullscreen !== _fullscreenActive) {
        _fullscreenActive = shouldBeFullscreen;
        document.body.classList.toggle('fullscreen-mode', shouldBeFullscreen);
        if (shouldBeFullscreen) {
            _reconcileCbVisibility({ animPhase: 'enter-2' });
        } else {
            // Reset island drag positions, wrapper padding, etc.
            _infoDragPos = null; _cbDragPos = null; _islandDragPos = null;
            // Reset inline positioning styles
            _reconcileCbVisibility({ animPhase: 'exit-3' });
        }
    }

    // Apply opacities
    titleEl.style.opacity = fadeOut;
    dimbar.style.opacity = shouldBeFullscreen ? fadeIn : fadeOut;
    colorbar.style.opacity = shouldBeFullscreen ? fadeIn : fadeOut;
    // ... same for compare/mv/qmri chrome
}
```

### Call Site

Called immediately after every `ModeRegistry.scaleAll()` invocation that changes zoom:
- Every wheel event (ctrl+scroll)
- Every pinch/gesture event
- Every `_scaleAllWithAnim` frame (for `=`/`-`/`0` key animations)

### `=` Key Path (Unified)

Replace `_enterImmersive()` with:
```js
userZoom = immersiveFitZoom;
_scaleAllWithAnim();
// Each animation frame calls ModeRegistry.scaleAll() → _applyImmersiveCrossfade()
// Chrome naturally fades/morphs as zoom sweeps through the zone
```

### `-` Key and Scroll Zoom Exit

No special exit logic needed. As `userZoom` decreases, `p` decreases, the crossfade reverses automatically. When `userZoom <= _normalFitZoom`, `p = 0`, fully normal layout.

## Dimbar Width Consistency

In current immersive mode, the dimbar becomes a compact floating island that changes width. This is unnecessary and makes the crossfade harder.

**Change:** The dimbar keeps its normal width in immersive mode. Only its position (bottom-docked → floating top), background (solid → translucent blur), and border-radius change. This simplifies the crossfade (fewer properties to transition) and provides visual continuity.

CSS change: remove any width constraints from `body.fullscreen-mode #info` that differ from normal mode.

## What Gets Removed

- `_enterImmersive()` — replaced by crossfade controller + animated zoom for `=` key
- `_exitImmersive()` — replaced by crossfade reversing as zoom decreases
- `_immersiveAnimating` flag — no longer needed. The crossfade is stateless (computed from zoom level). Keyboard input during transition just works because it modifies `userZoom` which feeds into `p`.
- `_animOpacity()` calls for immersive transitions — opacity is now a direct function of `p`
- The 3-phase fade→grow→fade timing logic

`_scaleAllWithAnim()` itself is kept — still used by `=`/`-`/`0` keys for smooth animated zoom.

## Edge Cases

- **Window resize during crossfade:** `immFit` recalculates, `p` adjusts. Chrome repositions instantly. No special handling.
- **`_shouldEnterImmersive()` returns false:** Crossfade skipped entirely. Zoom just zooms. If currently immersive, snap to normal.
- **Rapid direction changes:** Stateless — p is computed from current zoom, no animation state to cancel.
- **`0` key (reset zoom):** Animates `userZoom` to `_normalFitZoom`. Crossfade reverses naturally through the zone.
- **Compare/multiview/qMRI modes:** Same p-driven fade for mode-specific chrome. `_reconcileCbVisibility()` handles colorbar visibility at layout switch.
- **`_shouldEnterImmersive()` changes during zoom** (e.g., window resize makes width binding): If immersive is no longer appropriate, snap to normal on next `_applyImmersiveCrossfade()` call.
