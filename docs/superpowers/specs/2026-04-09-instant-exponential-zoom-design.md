# Instant Exponential Zoom

**Date:** 2026-04-09
**Status:** Approved

## Problem

Trackpad zoom (ctrl/cmd+scroll and pinch) feels sluggish, laggy, and uneven. Three root causes:

1. **Too small per-event zoom** — `1 + absDelta * 0.004` yields ~1.008x–1.02x for trackpad deltaY values (1–5). Barely perceptible per frame.
2. **150ms animation on every wheel event** — trackpads fire at 60Hz+. Each event cancels the previous animation mid-flight and restarts, so zoom perpetually chases a target it never reaches.
3. **Jagged partial ease-out curves** — the cancel-restart pattern produces uneven zoom speed. Sometimes events cluster (stuck), sometimes they space out (jump).

Additionally, **Safari pinch-to-zoom is broken** — Safari uses proprietary `GestureEvent` instead of synthesizing `ctrl+wheel`. No handler exists.

## Solution

Replace the linear sensitivity formula + animation with a **clamped exponential formula applied instantly**.

### Formula

```js
const clamped = Math.max(-10, Math.min(10, e.deltaY));
const factor = Math.pow(2, -clamped * 0.1);
userZoom = Math.max(lowerBound, Math.min(10.0, userZoom * factor));
```

**Why clamped exponential:**
- Trackpad events (deltaY ~2) pass through unclamped → 7% zoom per event at 60Hz = smooth ramp
- Mouse wheel clicks (deltaY ~100) get clamped to 10 → 50% per click = one perceptible step
- Exponential is symmetric: zoom in then out by the same delta returns to original scale
- Industry standard: used by Figma, pixi-viewport, Fabric.js

**Why no animation:**
- Figma, Excalidraw, tldraw all apply zoom instantly for continuous input
- High-frequency trackpad events already provide perceptual smoothness
- Animation reserved for discrete jumps (keyboard +/-, fit-to-screen, immersive enter/exit)

### Tuning

Single constant: `0.1` (the exponent multiplier). Effects:
- `0.1` → trackpad 7%/event, wheel 50%/click
- `0.07` → trackpad 5%/event, wheel 33%/click

Tune after testing with both input devices.

## Changes

### 1. Ctrl/Cmd + scroll handler (~line 13969)

Replace:
```js
const absDelta = Math.min(Math.abs(e.deltaY), 50);
const zoomFactor = 1 + absDelta * 0.004;
userZoom = e.deltaY > 0
    ? Math.max(userZoom / zoomFactor, _normalFitZoom)
    : Math.min(userZoom * zoomFactor, 10.0);
// ...
_scaleAllWithAnim();
```

With:
```js
const clamped = Math.max(-10, Math.min(10, e.deltaY));
const factor = Math.pow(2, -clamped * 0.1);
userZoom = Math.max(_normalFitZoom, Math.min(10.0, userZoom * factor));
// ...
ModeRegistry.scaleAll();
```

Immersive auto-enter/exit and `_snapZoomToFit()` logic unchanged.

### 2. Safari GestureEvent handlers (new)

```js
let _gestureStartZoom = null;

function _onGestureStart(e) {
    e.preventDefault();
    _gestureStartZoom = userZoom;
}

function _onGestureChange(e) {
    e.preventDefault();
    userZoom = Math.max(_normalFitZoom, Math.min(10.0, _gestureStartZoom * e.scale));
    _zoomAdjustedByUser = true;
    ModeRegistry.scaleAll();
}

function _onGestureEnd(e) {
    e.preventDefault();
    saveState();
}
```

Register on `canvas` and all `compareCanvases`. Safari's `e.scale` is cumulative from `gesturestart` — multiply against cached value, never compound.

### 3. Touch pinch handler (~line 15610)

Replace `Math.pow(d / _pinchDist0, 0.35)` dampening with direct ratio:

```js
userZoom = Math.max(0.1, Math.min(10, _pinchZoom0 * (d / _pinchDist0)));
```

Raw ratio is appropriate since touch events are already smoothly sampled. If too sensitive after testing, add a mild exponent (0.7–0.8).

### 4. MIP mode wheel handler (~line 11712)

Apply same clamped exponential for consistency:

```js
const clamped = Math.max(-10, Math.min(10, e.deltaY));
_mipZoom = Math.max(0.5, Math.min(10.0, _mipZoom * Math.pow(2, -clamped * 0.1)));
```

### 5. Unchanged

- `_scaleAllWithAnim()` — kept for keyboard zoom, immersive transitions, fullscreen
- Colorbar wheel zoom (fixed 1.1x) — different purpose
- Zoom bounds (`_normalFitZoom` to `10.0`)
- Pan, auto-fit, `_clampUniformZoom`, `_scalePane`
- `_zoomRendered` tracking — still used by `_scaleAllWithAnim` for its callers

## Future work

Smooth normal→immersive transition during continuous trackpad zoom (separate design).
