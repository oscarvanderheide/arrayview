# Immersive Animation

Spec and failure log for the pinch-to-immersive enter/exit transition.

---

## What it should look like

### The moment you start pinching

The dimbar, colorbar, and title get "detached" from the layout instantly. They are frozen at their exact current on-screen pixel positions — they look identical to before, just hovering above the layout rather than part of it. Because they are no longer in the normal document flow, the canvas has the space to grow — but we don't let it grow yet. We inject just enough extra padding to keep the canvas exactly where it was, so nothing moves on frame 1.

### As you pinch in (progress 0 → 1)

The canvas grows smoothly, tied 1-to-1 with how much you've pinched. No spring, no lag. As the canvas grows, the frozen dimbar and colorbar simultaneously fade from visible to invisible. They never move — they just get dimmer. The growing canvas slides under them without them influencing its position at all. The extra padding injected at scrub start is reduced in sync with pinch progress, which is what drives the growth.

### When the canvas reaches its final immersive size (progress = 1)

The immersive layout class is applied. A0t this point the dimbar and colorbar are completely invisible, so they are silently repositioned to their overlay locations — dimbar near the top of the canvas, colorbar near the bottom. They teleport while invisible, not animate. Then they fade from invisible to visible at those positions. They do not move during this fade-in.

### Pinching back out (progress 1 → 0)

The immersive class is removed immediately. The dimbar and colorbar silently snap back to their scrub-start positions (still invisible, so no visible snap). As the canvas shrinks, the dimbar/colorbar fade back in at their scrub-start positions. When progress reaches 0, the fixed positioning is removed and the chrome elements rejoin the normal document flow.

### The core rule

Chrome elements never move while their opacity is changing. The transition between "normal flow position" and "immersive overlay position" always happens while they are invisible.

---

## Things tried that didn't work

### paneCutoff two-phase timeline
The GSAP timeline was split into two phases: a "pane growth" phase (progress 0 → 0.6) and a "chrome settle" phase (0.6 → 1.0). `paneCutoff = 0.6` was stored on `tl.data`. The scrub mapped `desiredZoom` into the first phase only, then auto-advanced to 1.0 when the zoom hit `immLand`. Problems: the two-phase split made the feel non-linear (the pane would grow at one rate then the chrome would do something different), and reasoning about the mapping was complex.

### Chasing tween / _scrubRequestedZoom lag
Instead of writing `immersiveTl.progress()` directly, a GSAP tween chased the target progress with a 0.25s duration. The idea was to smooth out sudden zoom jumps. In practice it just introduced lag — the timeline progress chased the fingers rather than tracking them, so the pane always felt behind. Also caused `_scrubRequestedZoom` to diverge from `userZoom`, leading to temporary layout overflow and minimap flicker.

### chromeProxy / _scrubChromeReserve phantom tween
Tried to drive the chrome space reservation using an animated proxy object (a GSAP tween on a `{v: 0}` proxy that fed into padding/reserve calculations). The phantom tween fought with `fixWrapperAlignment` recalculations happening every scaleAll frame, causing the pane to jump when the tween value and the layout's own calculation disagreed.

### _chromeFrozen flag + fixWrapperAlignment skip + _preScrubPaddingTop
At the first pinch frame: snapshotted dimbar/colorbar positions, set them to `position: fixed`, bumped `#wrapper` paddingTop by their combined height so the canvas didn't move, set `_chromeFrozen = true`. While `_chromeFrozen`, `fixWrapperAlignment` returned early. On reset/settle, restored `_preScrubPaddingTop`. Also added `z-index: 10` to the fixed chrome so it sat above the canvas. Problems: the paddingTop compensation was not stable across zoom — as `ModeRegistry.scaleAll()` ran each frame and the canvas grew, `childrenH` changed, and since `fixWrapperAlignment` was skipped, the canvas growth and the wrapper's centering fell out of sync. The pane still jumped, just differently.
