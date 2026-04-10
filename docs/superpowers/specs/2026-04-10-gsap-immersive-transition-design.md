# GSAP Immersive Transition

**Date:** 2026-04-10
**Status:** Approved
**Supersedes:** `2026-04-09-smooth-immersive-crossfade-v2-design.md` (behavior unchanged; replaces manual interpolation engine with GSAP)

## Problem

The v2 crossfade implementation manually interpolates ~5 CSS properties per element across two progress phases. Every missed property (margin, padding, overflow) shows up as a visible jump. The `=` key wires into the zoom-driven animation by sweeping `userZoom` through the crossfade zone, which works but is fragile — any stutter in the zoom sweep causes a stutter in the chrome animation.

## Solution

Replace the manual interpolation in `_applyImmersiveCrossfade()` with a **single paused GSAP timeline** that is scrubbed by zoom progress during scroll/pinch, and played with a fixed duration on `=` keypress. GSAP drives every CSS property in the same rAF tick, eliminating missed-property jumps.

The behavioral model (two-phase fade + collapse, invisible class switch at p=0.5, `fullscreen-mode` CSS driving layout) is **unchanged from v2**. Only the interpolation engine changes.

## Delivery

`gsap.min.js` (GSAP 3.12.x core, ~30KB) is vendored into `src/arrayview/` alongside `_viewer.html`. No CDN, no FLIP plugin, works offline.

```
src/arrayview/
  _viewer.html
  gsap.min.js       ← new vendored file
```

Referenced in `_viewer.html` `<head>` before the main script block:

```html
<script src="gsap.min.js"></script>
```

Verify that `pyproject.toml` / `MANIFEST.in` includes `*.js` files from the package directory so `gsap.min.js` ships with the package.

## Timeline Construction

Built once on viewer init, stored as `immersiveTl`. The timeline is paused — GSAP never auto-plays it.

```javascript
let immersiveTl = null;

function _buildImmersiveTl() {
  const collapseP = { v: 0 };  // proxy for _crossfadeCollapseP

  immersiveTl = gsap.timeline({
    paused: true,
    onUpdate: () => {
      _crossfadeCollapseP = collapseP.v;
      ModeRegistry.scaleAll();
    }
  });

  // Phase 1 (0 → 0.5): collapse and fade out
  immersiveTl
    .to('#array-name', { opacity: 0, maxHeight: 0, marginTop: 0, marginBottom: 0,
                         overflow: 'hidden', duration: 0.5, ease: 'none' }, 0)
    .to('#info',       { opacity: 0, maxHeight: 0, overflow: 'hidden',
                         duration: 0.5, ease: 'none' }, 0)
    .to('#slim-cb-wrap', { opacity: 0, maxHeight: 0, overflow: 'hidden',
                           duration: 0.5, ease: 'none' }, 0)
    .to(collapseP,     { v: 1, duration: 0.5, ease: 'none' }, 0)  // drives _crossfadeCollapseP

    // Midpoint: invisible class switch
    .call(() => {
      document.body.classList.add('fullscreen-mode');
      // restore inline styles that fullscreen-mode CSS now controls
      ['#array-name', '#info', '#slim-cb-wrap'].forEach(sel => {
        const el = document.querySelector(sel);
        el.style.maxHeight = '';
        el.style.overflow = '';
      });
    }, [], 0.5)

    // Phase 2 (0.5 → 1.0): fade in at immersive positions
    .to('#info',         { opacity: 1, duration: 0.5, ease: 'none' }, 0.5)
    .to('#slim-cb-wrap', { opacity: 1, duration: 0.5, ease: 'none' }, 0.5);

  // Mode-specific colorbars (opacity only, both phases)
  _addModeColorbarTweens(immersiveTl);
}
```

The `ease: 'none'` on all tweens keeps the per-property easing linear — the timeline progress value is the easing surface, so easing is applied once (by `tweenTo`) rather than per-property.

### Reverse midpoint

The class switch at p=0.5 must also be handled on the reverse pass. Add a second `.call()` at progress 0.5 that fires when the timeline plays backward:

```javascript
.call(() => {
  if (immersiveTl.reversed()) {
    document.body.classList.remove('fullscreen-mode');
    // Re-apply collapse styles so elements stay visually hidden while GSAP
    // animates them back open during phase 1 reverse (p: 0.5 → 0)
    ['#array-name', '#info', '#slim-cb-wrap'].forEach(sel => {
      const el = document.querySelector(sel);
      el.style.maxHeight = '0px';
      el.style.overflow = 'hidden';
      el.style.opacity = '0';
    });
  }
}, [], 0.5)
```

GSAP fires `.call()` in both directions, so the class toggle and style cleanup happen correctly on exit too.

## Driving the Timeline

### Scroll / pinch (zoom-driven)

Replace the body of `_applyImmersiveCrossfade()` with:

```javascript
function _applyImmersiveCrossfade(userZoom, normFit, immFit) {
  if (!immersiveTl) return;
  const p = Math.max(0, Math.min(1, (userZoom - normFit) / (immFit - normFit)));
  immersiveTl.progress(p);
  // onUpdate fires synchronously, updating _crossfadeCollapseP and calling scaleAll
}
```

`immersiveTl.progress(p)` is synchronous — GSAP immediately writes all tweened values to the DOM and fires `onUpdate`. The scroll handler no longer needs to call `scaleAll` separately after this; `onUpdate` handles it.

### `=` key (timed animation)

```javascript
const goingImmersive = !document.body.classList.contains('fullscreen-mode');
immersiveTl.tweenTo(goingImmersive ? 1 : 0, {
  duration: 0.4,
  ease: 'power2.inOut',
});
```

GSAP drives its own rAF loop for the 400ms duration, calling `onUpdate` each frame (which calls `ModeRegistry.scaleAll()`). No separate zoom sweep needed.

## Mode-Specific Colorbars

`_addModeColorbarTweens()` adds opacity tweens for colorbars that appear in specific modes. All fade out during phase 1 and stay hidden (they don't fade back in — immersive overlays replace them):

- `.compare-pane-cb-island`: opacity 1→0 at position 0, duration 0.5
- `#mv-cb-wrap`: opacity 1→0 at position 0, duration 0.5
- `.qv-cb-island`: opacity 1→0 at position 0, duration 0.5

These selectors may match zero elements in the current mode — GSAP silently skips missing targets.

## `_crossfadeCollapseP` and Bottom Reserves

The proxy object tween drives `_crossfadeCollapseP` exactly as before. All scale functions that multiply their bottom reserve by `(1 - _crossfadeCollapseP)` are **unchanged** — only the variable is now updated by GSAP's `onUpdate` rather than by manual interpolation.

## Natural Height Caching

The v2 design caches each element's `naturalH` when p first moves above 0 so the start values don't shift during the animation. With GSAP, the start state is captured when `_buildImmersiveTl()` is called (at init). If elements haven't been laid out yet at init time, defer `_buildImmersiveTl()` to first use (call it lazily the first time `_applyImmersiveCrossfade()` is invoked).

## What Gets Removed

- Manual interpolation loop inside `_applyImmersiveCrossfade()` (all the `collapseP * naturalH` arithmetic)
- `_crossfadeCleanup()` CSS reset calls (GSAP's `onComplete` / the clean terminal states at p=0 and p=1 handle cleanup)
- `style.transition = 'none'` overrides (GSAP controls transitions directly)

What stays:
- `_crossfadeCollapseP` variable and all its consumers in scale functions
- `fullscreen-mode` CSS class and all associated CSS
- `_positionFullscreenChrome()` (called after scaleAll, positions overlays — unchanged)
- `_shouldEnterImmersive()` guard

## Edge Cases

- **Window resize while animating:** `immFit` recalculates, `_applyImmersiveCrossfade` sets progress to new `p`. If `tweenTo` is in progress, calling `progress()` directly interrupts it — acceptable, resize is an exceptional event.
- **Rapid `=` presses:** `tweenTo` from current position each time; GSAP picks up from wherever the timeline is.
- **Mode without immersive support:** `_shouldEnterImmersive()` returns false, `_applyImmersiveCrossfade` is not called, timeline stays at progress 0.
- **GSAP unavailable (load failure):** Wrap `_buildImmersiveTl()` in a `typeof gsap !== 'undefined'` guard. If GSAP is missing, `immersiveTl` stays null and `_applyImmersiveCrossfade` early-returns — the transition simply won't animate (elements snap to their end state). This is acceptable; a missing vendored file is a deployment error, not a runtime edge case.
