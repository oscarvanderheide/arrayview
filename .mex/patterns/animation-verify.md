---
name: animation-verify
description: Use before AND after any change to GSAP, rAF loops, or CSS transitions in _viewer.html. Design checks up front, frame capture verification after.
triggers:
  - "animation"
  - "transition"
  - "GSAP"
  - "requestAnimationFrame"
  - "crossfade"
  - "tween"
  - "fade"
  - "stutter"
  - "jank"
  - "smooth"
  - "frame"
edges:
  - target: patterns/frontend-change.md
    condition: always — animation changes are always frontend changes
last_updated: 2026-04-26
---

# Animation Verify

**Why this exists:** The test suite (`test_browser.py`, `visual_smoke.py`, `ui_audit.py`)
captures STEADY-STATE screenshots ONLY. "All tests pass" is meaningless for
animation quality.

## Before You Code

Read this BEFORE touching any animation code. Answer these design questions before writing.

### GSAP timelines

- **CSS transition conflict:** Before adding a GSAP tween on any property, check if that
  element has a CSS `transition` (search `_viewer.html` for `<style>` blocks — ~50
  declarations exist). If yes, set `el.style.transition = 'none'` before the tween starts.
  The tween will race with the CSS transition and cause stutter or wrong final values.
- **Cache origins before transitions:** If you need `getBoundingClientRect()` for an
  animation start/end position (multiview entry, compare FLIP), snapshot the rect BEFORE
  mutating layout. Write rect → mutate → tween from stored rect, never the reverse.
- **Build entry AND exit together:** Every tween that animates something ON must also
  animate it OFF. If you add a forward tween, also add the reverse. Both paths go in the
  same timeline (not separate `gsap.to()` calls with different target values).
- **Cleanup inline styles:** After a tween completes (or is killed), call
  `el.style.removeProperty('...')` for every property the tween set. Leftover inline
  `style.opacity = '0'` will persist and break future renders.
- **No hardcoded timing in ms where a design variable exists.** Use the existing
  constants (search for `duration`, `stagger`, `ease` in the file) instead of creating
  new magic numbers.

### requestAnimationFrame loops

- **16 ms budget:** The rAF callback body must complete in under 16 ms on a target machine.
  If you add canvas draws, DOM reads, or style mutations inside the loop, you risk frame
  drops. Move heavy work outside the rAF (pre-compute, cache).
- **Always stop condition:** Every rAF loop must have a stop path. Check `_stop` flags,
  `isPlaying`, or timeline completion before calling `requestAnimationFrame` again.
  Never create a loop that depends on page unload to stop.
- **Don't stack loops:** If the user can restart the animation (Space, flicker, crosshair
  fade), check if `_playRafId` / `_mvCrosshairAnim` / etc. is already non-null before
  starting a new one. Cancel the old one with `cancelAnimationFrame()` first.

### CSS transition changes

- **Check GSAP overlaps:** Before adding a `transition: opacity 200ms` on any element,
  search the JS for GSAP `gsap.to()` / `gsap.from()` targeting the same property on the
  same selector. A CSS transition and a GSAP tween on `opacity` will fight — pick one
  path and use it consistently.
- **Use `transition: none` during layout recalc:** If `fixWrapperAlignment()` or a
  reconciler mutates dimensions/positions that are transitioned, briefly set
  `transition: none` before the mutation and restore after. Otherwise the browser
  animates the layout change, producing a visual "drift".

### How will you verify this?

Before writing the first line, answer: **"What specific frame capture or progress-scrub
test will prove this animation works?"** If the answer is "visual_smoke.py" or
"pytest", the answer is wrong — those tools don't test animation. See Steps below for
the correct procedure.

## Steps

### 1. Run frame-capture

```bash
uv run python tests/capture_v_animation.py
```

Frames saved to `tests/v_anim_frames/`. This captures 30 frames at 50 ms intervals
during the multiview v-key entry/exit GSAP animation, plus steady-state before/after.

### 2. Inspect frames visually

Open the frame directory and scan the sequence:

- [ ] Consecutive frames during the animation differ (movement is happening — NOT the same frame repeated)
- [ ] No element teleports between frames (Δposition > 5 px in a single frame step is a jump)
- [ ] Opacity changes are monotonic during fades (no flickering in/out)
- [ ] Fixed-position elements stay locked when frozen
- [ ] Entry and exit animations look symmetric (not one smooth and one jerky)

### 3. Scrub at key progress points

For pinch immersive crossfade: exercise `rawP` at 0 → 0.25 → 0.5 → 0.75 → 1.0 via
`window.__pinchDebug = true` and the Playwright console:

- [ ] Canvas size progresses monotonically (grows without shrinking mid-transition)
- [ ] Chrome elements stay at freeze position until their fade-out begins
- [ ] No layout jump at the rawP=0 or rawP=1 boundary
- [ ] After `_settleImmersive()` fires: chrome visible at overlay positions, `_scrubDetached` is false

### 4. Verify reverse path

- [ ] Exit animation (reverse scrub) shows the same freeze behavior as entry
- [ ] Chrome elements rejoin document flow cleanly at rawP=0 (no leftover fixed positioning)
- [ ] `_resetImmersiveTransforms()` clears all detach inline styles

## Regression Table

| Symptom | Likely cause | What to check |
|---------|-------------|---------------|
| Element jumps between frames | `getBoundingClientRect()` measured at wrong time | Timing of rect snapshot vs style mutations |
| Animation freezes (same frame 3+ times) | CSS transition racing GSAP on same property | Set `transition: none` before GSAP takeover |
| Chrome drifts during crossfade | `uiReserveV()` or `fixWrapperAlignment()` re-measuring detached elements | `_scrubFrozenReserve` / `_scrubFrozenPt` |
| Exit looks different from entry | Asymmetric tween config or missing re-freeze on reverse | `_scrubDetached` re-enabled in `_exitImmersive()` |
| Startup animation skips | Timeline race with first render frame | `_startupAnimTl` timing vs load event |
| Flickering during auto-play (Space) | rAF callback doing more work than 16 ms budget | `_rafTick` body complexity |

## Gotchas

- **`visual_smoke.py` does NOT test animation** — it takes one screenshot per scenario after transitions settle. An animation can look terrible and still produce the correct final screenshot.
- **`test_browser.py` snapshot tests compare steady-state only** — a 1% pixel threshold can pass even if the animation path was full of jumps.
- **`ui_audit.py` takes static screenshots** — no temporal validation whatsoever.
- **`capture_v_animation.py` is manual-review only** — it has no automated assertions. YOU must inspect the frames yourself and describe what you saw. Do not claim "tests pass" — that phrase has no meaning for animation.

## Verify

- [ ] `capture_v_animation.py` frames inspected — no jumps, monotonic fades, symmetric entry/exit
- [ ] Key progress points scrubbed — canvas monotonic, chrome frozen when expected
- [ ] Reverse path verified — chrome rejoins flow cleanly
- [ ] Frame inspection results described to the user (not "tests pass")
