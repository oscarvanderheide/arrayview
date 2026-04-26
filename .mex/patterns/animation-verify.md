---
name: animation-verify
description: Verifying animation quality — frame-by-frame capture, jank detection, position stability, transition smoothness. Use after any change to GSAP, rAF loops, or CSS transitions in _viewer.html.
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
captures STEADY-STATE screenshots ONLY. A single screenshot after the transition settles cannot
catch jumps, stutter, drift, or temporal breakage. "All tests pass" is meaningless for
animation quality.

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
