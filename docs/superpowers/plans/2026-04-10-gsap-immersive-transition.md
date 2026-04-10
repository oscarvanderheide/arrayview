# GSAP Immersive Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the manual rAF zoom animation with GSAP for snappy, properly-eased immersive transitions on `=`/`-` key presses, and use a GSAP proxy timeline to drive the crossfade CSS interpolation.

**Architecture:** Vendor `gsap.min.js` and serve it via a dedicated FastAPI endpoint. Replace `_scaleAllWithAnim()` with a GSAP tween (400ms, power2.inOut). Refactor `_applyImmersiveCrossfade()` to scrub a paused GSAP proxy timeline instead of computing `collapseP`/`fadeIn` with manual arithmetic. The existing class toggle logic and CSS property application stay intact.

**Tech Stack:** GSAP 3.12.x (core only, no plugins), Python/FastAPI, vanilla JS

**Spec:** `docs/superpowers/specs/2026-04-10-gsap-immersive-transition-design.md`

---

### Task 1: Vendor gsap.min.js

**Files:**
- Create: `src/arrayview/gsap.min.js`

- [ ] **Step 1: Download GSAP core**

```bash
curl -L -o src/arrayview/gsap.min.js "https://cdn.jsdelivr.net/npm/gsap@3.12.7/dist/gsap.min.js"
```

Verify the file exists and is ~72KB:
```bash
ls -la src/arrayview/gsap.min.js
```

- [ ] **Step 2: Verify the file contains GSAP**

```bash
head -c 200 src/arrayview/gsap.min.js
```

Expected: starts with a copyright comment and minified JS containing `gsap`.

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/gsap.min.js
git commit -m "vendor: add gsap 3.12.7 core for immersive transition"
```

---

### Task 2: Serve GSAP from the server and load it in the viewer

**Files:**
- Modify: `src/arrayview/_server.py:28-30` (import), `src/arrayview/_server.py:265-272` (template loading), add endpoint
- Modify: `src/arrayview/_viewer.html:1535` (add script tag before main script)

- [ ] **Step 1: Add GSAP endpoint to _server.py**

Add `Response` to the existing import on line 28:

```python
from fastapi.responses import HTMLResponse, JSONResponse, Response
```

After line 272 (after the `_VIEWER_HTML_TEMPLATE` block), add:

```python
_GSAP_JS: str = (
    _pkg_files("arrayview").joinpath("gsap.min.js").read_text(encoding="utf-8")
)


@app.get("/gsap.min.js")
def serve_gsap():
    """Serve vendored GSAP library (browser caches via ETag)."""
    return Response(content=_GSAP_JS, media_type="application/javascript")
```

- [ ] **Step 2: Add script tag to _viewer.html**

Before line 1535 (`<script>`), add:

```html
    <script src="/gsap.min.js"></script>
```

- [ ] **Step 3: Verify GSAP loads in the browser**

Start the viewer, open browser devtools console, type `gsap.version`. Should return `"3.12.7"`.

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_server.py src/arrayview/_viewer.html
git commit -m "feat: serve vendored GSAP and load in viewer"
```

---

### Task 3: Replace _scaleAllWithAnim with GSAP tween

**Files:**
- Modify: `src/arrayview/_viewer.html:2576-2612` (`_scaleAllWithAnim` function)

This is the core change. The existing function uses a manual rAF loop with 2000ms duration and cubic easing. Replace it with a GSAP tween: 400ms, power2.inOut.

- [ ] **Step 1: Replace _scaleAllWithAnim**

Replace the entire function (lines 2576-2612) with:

```javascript
        let _zoomAnimId = null;   // kept for compat — now unused
        let _zoomRendered = undefined; // zoom level currently shown on screen
        let _zoomTween = null;    // active GSAP tween for zoom animation
        function _scaleAllWithAnim(onDone, easeIn) {
            // Animate zoom from current rendered value to userZoom target.
            // GSAP drives the rAF loop; each frame calls ModeRegistry.scaleAll()
            // which runs _applyImmersiveCrossfade + the mode scaler.
            const targetZoom = userZoom;
            if (_zoomTween) { _zoomTween.kill(); _zoomTween = null; }
            if (_zoomAnimId) { cancelAnimationFrame(_zoomAnimId); _zoomAnimId = null; }
            const fromZoom = (_zoomRendered !== undefined) ? _zoomRendered : targetZoom;
            if (Math.abs(fromZoom - targetZoom) < 0.001) {
                userZoom = targetZoom; _zoomRendered = targetZoom;
                _cbExpandedHLocked = false;
                ModeRegistry.scaleAll();
                if (onDone) onDone();
                return;
            }
            const proxy = { z: fromZoom };
            _zoomTween = gsap.to(proxy, {
                z: targetZoom,
                duration: 0.4,
                ease: easeIn ? 'power3.in' : 'power2.inOut',
                onUpdate: () => {
                    userZoom = proxy.z;
                    _zoomRendered = userZoom;
                    ModeRegistry.scaleAll();
                },
                onComplete: () => {
                    userZoom = targetZoom;
                    _zoomRendered = targetZoom;
                    _cbExpandedHLocked = false;
                    ModeRegistry.scaleAll();
                    _zoomTween = null;
                    if (onDone) onDone();
                },
            });
        }
```

Key changes from the old function:
- `_zoomTween` replaces `_zoomAnimId` as the active animation handle
- `gsap.to(proxy, ...)` replaces `requestAnimationFrame(step)`
- 400ms replaces 2000ms
- `power2.inOut` replaces manual cubic easing (`1 - Math.pow(1-p, 3)`)
- `power3.in` replaces manual ease-in cubic (`Math.pow(p, 3)`) for the `easeIn` path
- All callers (`=`, `-`, `0` key handlers) continue to work unchanged — the function signature is identical

- [ ] **Step 2: Verify = key animation in browser**

1. Open arrayview with a 2D array
2. Press `=` — should smoothly enter immersive in ~400ms with power2.inOut feel
3. Press `-` — should smoothly exit immersive in ~400ms
4. Press `0` — should snap back to fit
5. Scroll/pinch zoom should still work as before (scroll doesn't use `_scaleAllWithAnim`)

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: replace manual zoom animation with GSAP tween (400ms, power2.inOut)"
```

---

### Task 4: Refactor _applyImmersiveCrossfade to use GSAP proxy timeline

**Files:**
- Modify: `src/arrayview/_viewer.html:15061-15245` (`_applyImmersiveCrossfade` and surrounding code)

Replace the manual `collapseP`/`fadeIn` arithmetic with a paused GSAP timeline that drives proxy values. The CSS property application logic stays the same — only the progress computation changes.

- [ ] **Step 1: Add the proxy timeline builder**

Replace the crossfade state variables (lines 15064-15069) and add the timeline builder. Insert right after the `// ── Zoom-driven immersive crossfade` comment (line 15061):

```javascript
        // ── Zoom-driven immersive crossfade ────────────────────────────
        // GSAP proxy timeline: scrubbed by zoom progress during scroll/pinch.
        // Drives collapseP (0→1 in first half). fadeIn is computed manually
        // The actual CSS property application remains manual so we can handle
        // the fullscreen-mode class toggle between the two phases.
        let _crossfadePrev = -1;
        let _crossfadeTitleH = 0;
        let _crossfadeInfoH = 0;
        let _crossfadeCbH = 0;
        let _crossfadeTitleMT = 0, _crossfadeTitleMB = 0;
        let _crossfadeInfoMT = 0, _crossfadeInfoMB = 0;
        const _cfProxy = { collapseP: 0 };
        let _cfTimeline = null;

        function _buildCrossfadeTimeline() {
            if (_cfTimeline) _cfTimeline.kill();
            _cfProxy.collapseP = 0;
            // Timeline maps progress 0→0.5 to collapseP 0→1 (first half of crossfade).
            // fadeIn is NOT driven by the timeline — it depends on absolute zoom values
            // (userZoom relative to immFit), not the normalized progress p.
            _cfTimeline = gsap.timeline({ paused: true })
                .to(_cfProxy, { collapseP: 1, duration: 0.5, ease: 'none' }, 0);
        }
```

- [ ] **Step 2: Replace manual collapseP/fadeIn computation in _applyImmersiveCrossfade**

In the function body (starting at line 15070), replace the `collapseP` and `fadeIn` manual computation (lines 15091-15096) with the timeline scrub:

Replace:
```javascript
            // collapseP: 0→1 during first half (chrome collapses + fades out)
            const collapseP = Math.min(1, p * 2);
            // fadeIn: 0→1 only AFTER pane reaches final size (userZoom past immFit).
            // This prevents chrome from appearing while the pane is still growing,
            // which would cause annoying positional drift of the floating elements.
            // 0.5% zoom range past immFit — matches the 1.005 margin in zoom.in
            const fadeIn = Math.min(1, Math.max(0, (userZoom - immFit) / (immFit * 0.005)));
```

With:
```javascript
            // Scrub the proxy timeline to get collapseP and fadeIn values
            if (!_cfTimeline) _buildCrossfadeTimeline();
            _cfTimeline.progress(p);
            const collapseP = _cfProxy.collapseP;
            // fadeIn: delayed until pane reaches final size (0.5% zoom past immFit)
            // This prevents floating chrome from drifting while the pane is still growing.
            const fadeIn = Math.min(1, Math.max(0, (userZoom - immFit) / (immFit * 0.005)));
```

Note: `fadeIn` stays manual because it depends on `userZoom` relative to `immFit` (a physical zoom threshold), not the normalized progress `p`. The proxy timeline drives `collapseP` only.

- [ ] **Step 3: Reset timeline in cleanup paths**

In `_crossfadeCleanup()` (line 15247), add timeline reset:

After `_crossfadeP = 0;` (line 15249), add:
```javascript
            if (_cfTimeline) _cfTimeline.progress(0);
            _cfProxy.collapseP = 0;
```

In the early-return path of `_applyImmersiveCrossfade` (line 15074-15079), where `_shouldEnterImmersive()` returns false, add after `_crossfadeCollapseP = 0;`:
```javascript
                if (_cfTimeline) _cfTimeline.progress(0);
                _cfProxy.collapseP = 0;
```

- [ ] **Step 4: Verify scroll/pinch crossfade still works**

1. Open arrayview with a 2D array
2. Slowly scroll/pinch to zoom in past normal fit
3. Chrome should smoothly collapse and fade out
4. Continue zooming — at p=0.5, layout switches to immersive
5. Chrome fades in at floating positions
6. Zoom back out — reverse should be symmetric
7. No visible jumps or stutter at any point

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: use GSAP proxy timeline for crossfade collapseP"
```

---

### Task 5: Run existing tests and verify

**Files:** No changes — verification only.

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/oscar/Projects/packages/python/arrayview
uv run pytest tests/ -x -v 2>&1 | tail -40
```

All tests should pass. The GSAP changes are timing-only — no behavioral changes to the viewer.

- [ ] **Step 2: Run interaction tests specifically**

```bash
uv run pytest tests/test_interactions.py -x -v 2>&1 | tail -40
```

Focus on zoom-related tests — they exercise the = key and zoom animations.

- [ ] **Step 3: Manual visual verification checklist**

Test each scenario in the browser:

| Scenario | Expected |
|---|---|
| `=` key from normal fit | Smooth 400ms entry to immersive, power2.inOut feel |
| `-` key from immersive | Smooth 400ms exit to normal, symmetric |
| `=` then `-` rapidly | Picks up from current position, no jump |
| Scroll/pinch into immersive | Same as before (zoom-driven, no timing change) |
| Scroll/pinch out of immersive | Same as before |
| `0` key from immersive | Snaps back to normal fit |
| Window resize during immersive | Layout adjusts, no crash |
| Compare mode `=` key | Smooth immersive entry/exit |
| qMRI mode `=` key | Smooth immersive entry/exit |

- [ ] **Step 4: Screenshot before/after for visual regression**

Take screenshots of immersive state in at least single-view and compare-view modes. Compare with existing baselines in `tests/ui_audit/baselines/`.

---

### Task 6: Clean up and final commit

**Files:**
- Modify: `src/arrayview/_viewer.html` (remove dead code if any)

- [ ] **Step 1: Remove _zoomAnimId if no longer referenced**

Search for `_zoomAnimId` in the file. If the only references are in the new `_scaleAllWithAnim` (the compat fallback), it can stay. If there are external references (e.g., in scroll handlers that cancel the animation), keep it.

```bash
grep -n '_zoomAnimId' src/arrayview/_viewer.html
```

- [ ] **Step 2: Verify GSAP fallback guard**

Check that the code handles the case where GSAP fails to load. In the `_scaleAllWithAnim` function, `gsap.to()` would throw if GSAP isn't loaded. Add a guard at the top of the function:

```javascript
            if (typeof gsap === 'undefined') {
                // Fallback: snap instantly if GSAP unavailable
                _zoomRendered = targetZoom;
                _cbExpandedHLocked = false;
                ModeRegistry.scaleAll();
                if (onDone) onDone();
                return;
            }
```

Similarly in `_buildCrossfadeTimeline`:
```javascript
        function _buildCrossfadeTimeline() {
            if (typeof gsap === 'undefined') return;
            // ... rest of function
        }
```

- [ ] **Step 3: Final commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "chore: add GSAP fallback guards, clean up dead zoom animation code"
```
