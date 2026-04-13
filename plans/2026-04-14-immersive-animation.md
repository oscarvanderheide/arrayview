# Immersive Animation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Commits:** Do NOT commit anything until the user confirms the result is good.

**Goal:** Make the pinch-to-immersive transition smooth — chrome elements freeze at their current screen positions at scrub start and fade out without moving, while the canvas grows cleanly; on entry completion, chrome teleports to overlay positions and fades in.

**Architecture:** Three coordinated changes — (1) detach `#info` from document flow at scrub start and freeze `#slim-cb-wrap` position, (2) bypass `fixWrapperAlignment` during the scrub and drive `paddingTop` from frozen value toward 0, (3) use a frozen reserve value in `uiReserveV()` to prevent mid-scrub DOM re-measurement drift.

**File:** `src/arrayview/_viewer.html` (single file — all tasks modify this file)

---

### Task 1: Add scrub-state variables

**File:** `src/arrayview/_viewer.html:1992`

- [ ] **Add three new variables near the other crossfade state vars (around line 1992)**

```javascript
let _crossfadeCollapseP = 0; // 0→1 during first half of immersive crossfade; scale functions use this to interpolate bottom reserves
let _crossfadeP = 0; // raw crossfade progress 0→1 (normFit→immFit)
let _scrubDetached = false;   // true while pinch scrub has chrome detached (frozen fixed positions)
let _scrubFrozenPt = 0;       // wrapper paddingTop captured at scrub start
let _scrubFrozenReserve = 0;  // uiReserveV() value captured at scrub start (before any crossfadeP changes)
```

---

### Task 2: Detach chrome in `_buildImmersiveTl()`

**File:** `src/arrayview/_viewer.html:17115`

Immediately after `_resetImmersiveTransforms()` and before caching image dims, snapshot `#info`'s screen rect, fix-position it there, and record frozen values.

- [ ] **Insert detach block after `_resetImmersiveTransforms()` at line 17115**

Find this exact block (lines 17115–17124):
```javascript
            _resetImmersiveTransforms();

            const { title: titleEl, info: infoEl, cb: cbWrap } = _immersiveTargets();

            // Cache immersive zoom target once per build.
            const _w = lastImgW || canvas.width;
            const _h = lastImgH || canvas.height;
            if (_w && _h) {
                _immTargetZoom = _calcAutoFitZoom(_w, _h, true) * 1.005;
            }
```

Replace with:
```javascript
            _resetImmersiveTransforms();

            // ── Detach chrome from flow so it can't influence layout during scrub ──
            // Snapshot positions AFTER reset so we measure natural CSS dimensions.
            const wrapper = document.getElementById('wrapper');
            _scrubFrozenPt = parseFloat(wrapper.style.paddingTop) || 14;
            _scrubFrozenReserve = uiReserveV(); // capture before _crossfadeP changes
            const infoDetach = document.getElementById('info');
            if (infoDetach) {
                const r = infoDetach.getBoundingClientRect();
                infoDetach.style.position = 'fixed';
                infoDetach.style.top = r.top + 'px';
                infoDetach.style.left = r.left + 'px';
                infoDetach.style.width = r.width + 'px';
                infoDetach.style.margin = '0';
            }
            _scrubDetached = true;

            const { title: titleEl, info: infoEl, cb: cbWrap } = _immersiveTargets();

            // Cache immersive zoom target once per build.
            const _w = lastImgW || canvas.width;
            const _h = lastImgH || canvas.height;
            if (_w && _h) {
                _immTargetZoom = _calcAutoFitZoom(_w, _h, true) * 1.005;
            }
```

---

### Task 3: Stabilize `uiReserveV()` during scrub

**File:** `src/arrayview/_viewer.html:2663`

Use the frozen reserve value during the scrub so DOM re-measurement of the now-fixed `#info` can't drift.

- [ ] **Modify the `_crossfadeP > 0` branch in `uiReserveV()`**

Find (lines 2663–2664):
```javascript
            if (_crossfadeP > 0) {
                return Math.max(130, h) * (1 - _crossfadeP);
            }
```

Replace with:
```javascript
            if (_crossfadeP > 0) {
                // Use the value captured at scrub start so DOM re-measurement
                // of the fixed-positioned #info can't drift during the scrub.
                const base = (_scrubDetached && _scrubFrozenReserve > 0)
                    ? _scrubFrozenReserve
                    : Math.max(130, h);
                return base * (1 - _crossfadeP);
            }
```

---

### Task 4: Freeze and smoothly drive `paddingTop` in `fixWrapperAlignment()`

**File:** `src/arrayview/_viewer.html:2669`

During the scrub, bypass the normal childrenH recalculation and instead drive `paddingTop` from `_scrubFrozenPt` toward 0 in sync with `_crossfadeCollapseP`.

- [ ] **Step 1: Add scrub bypass at the top of `fixWrapperAlignment()`**

Find the function opening (lines 2669–2702):
```javascript
        function fixWrapperAlignment() {
            // Center content in the viewport-minus-bottom-reserve zone so the
            // fixed-position colorbar + eggs always have room below the canvas.
            const wrapper = document.getElementById('wrapper');
            let childrenH = 0;
            for (const child of wrapper.children) {
```

Replace just the opening block (insert the bypass after `const wrapper = ...`):
```javascript
        function fixWrapperAlignment() {
            // Center content in the viewport-minus-bottom-reserve zone so the
            // fixed-position colorbar + eggs always have room below the canvas.
            const wrapper = document.getElementById('wrapper');

            // During pinch scrub, chrome is detached from flow. Drive paddingTop
            // from the frozen pre-scrub value toward 0 in sync with the canvas
            // growth, then bail — no childrenH recalculation while scrubbing.
            if (_scrubDetached) {
                const pt = Math.max(0, Math.round(_scrubFrozenPt * (1 - _crossfadeCollapseP)));
                wrapper.style.paddingTop = pt + 'px';
                wrapper.style.justifyContent = 'flex-start';
                return;
            }

            let childrenH = 0;
            for (const child of wrapper.children) {
```

- [ ] **Step 2: Remove the existing `_immersiveDriveTween` freeze** (it guarded key-press only; `_scrubDetached` now covers both paths)

Find:
```javascript
            // Freeze wrapper padding only while the normal-mode chrome is
            // still visible. Once the fade-out is complete (_crossfadeP > 0.5
            // corresponds to timeline progress > 0.3), let padding reflow so
            // the hidden title/info footprint can keep lifting the pane before
            // the midpoint class switch.
            if (_immersiveDriveTween && !_fullscreenActive && _crossfadeP <= 0.5) {
                wrapper.style.justifyContent = 'flex-start';
                return;
            }
```

Replace with:
```javascript
            // (freeze moved to _scrubDetached guard above — covers both pinch and key-press)
```

---

### Task 5: Freeze colorbar position during scrub

**File:** `src/arrayview/_viewer.html:3816`

`drawSlimColorbar()` repositions `#slim-cb-wrap` every frame as the canvas grows. During the scrub we want it frozen.

- [ ] **Add guard at top of `drawSlimColorbar()`**

Find (line 3816):
```javascript
        function drawSlimColorbar(markerFrac) {
```

Replace with:
```javascript
        function drawSlimColorbar(markerFrac) {
            // During scrub, colorbar is fading out in place — don't reposition it.
            if (_scrubDetached) return;
```

---

### Task 6: Restore `#info` in `_resetImmersiveTransforms()`

**File:** `src/arrayview/_viewer.html:17081`

`_resetImmersiveTransforms()` is the canonical cleanup for the scrub. Add removal of the detach inline styles and clear `_scrubDetached`.

- [ ] **Add `position`, `top`, `left`, `width`, `margin` to the list of removed inline styles, and clear `_scrubDetached`**

Find the forEach block (lines 17091–17101):
```javascript
            [t.title, t.info, t.cb].forEach(el => {
                if (!el) return;
                el.style.removeProperty('--fly-y');
                el.style.removeProperty('opacity');
                el.style.removeProperty('height');
                el.style.removeProperty('min-height');
                el.style.removeProperty('margin-top');
                el.style.removeProperty('margin-bottom');
                el.style.removeProperty('padding-top');
                el.style.removeProperty('padding-bottom');
                el.style.removeProperty('overflow');
            });
```

Replace with:
```javascript
            [t.title, t.info, t.cb].forEach(el => {
                if (!el) return;
                el.style.removeProperty('--fly-y');
                el.style.removeProperty('opacity');
                el.style.removeProperty('position');
                el.style.removeProperty('top');
                el.style.removeProperty('left');
                el.style.removeProperty('width');
                el.style.removeProperty('margin');
                el.style.removeProperty('height');
                el.style.removeProperty('min-height');
                el.style.removeProperty('margin-top');
                el.style.removeProperty('margin-bottom');
                el.style.removeProperty('padding-top');
                el.style.removeProperty('padding-bottom');
                el.style.removeProperty('overflow');
            });
            _scrubDetached = false;
```

---

### Task 7: Reset `_scrubDetached` in `_settleImmersive()`

**File:** `src/arrayview/_viewer.html:17173`

When the transition commits to immersive, chrome moves to its overlay positions — it's no longer detached in the scrub sense. Clear the flag so subsequent layout passes run normally.

- [ ] **Add `_scrubDetached = false` at the start of `_settleImmersive()`**

Find (lines 17173–17178):
```javascript
        function _settleImmersive() {
            // Called when transition commits to fully immersive (progress = 1.0).
            // Handles class switch + 150ms overlay fade-in.
            if (_fullscreenActive) return; // already settled
            if (_startupAnimTl) { _startupAnimTl.kill(); _startupAnimTl = null; }
            _fullscreenActive = true;
```

Replace with:
```javascript
        function _settleImmersive() {
            // Called when transition commits to fully immersive (progress = 1.0).
            // Handles class switch + 150ms overlay fade-in.
            if (_fullscreenActive) return; // already settled
            _scrubDetached = false; // chrome is now officially at overlay positions
            if (_startupAnimTl) { _startupAnimTl.kill(); _startupAnimTl = null; }
            _fullscreenActive = true;
```

---

### Task 8: Re-freeze on reverse scrub in `_exitImmersive()`

**File:** `src/arrayview/_viewer.html:17203`

When the user pinches back out from settled immersive, `_exitImmersive()` removes the fullscreen class. Re-enable `_scrubDetached` so the reverse scrub gets the same frozen-chrome behavior as the forward scrub.

- [ ] **Clear the detach inline styles and re-enable `_scrubDetached`**

Find the `#info` cleanup in `_exitImmersive()` (lines 17227–17228):
```javascript
            const infoEl2 = document.getElementById('info');
            if (infoEl2) { infoEl2.style.top = ''; infoEl2.style.left = ''; infoEl2.style.transform = ''; }
```

Replace with:
```javascript
            const infoEl2 = document.getElementById('info');
            if (infoEl2) {
                // Clear overlay JS-set styles AND the detach styles set at scrub start.
                infoEl2.style.top = '';
                infoEl2.style.left = '';
                infoEl2.style.transform = '';
                infoEl2.style.position = '';
                infoEl2.style.width = '';
                infoEl2.style.margin = '';
            }
            // Re-freeze the scrub so the reverse path has the same locked-chrome
            // behavior. _scrubFrozenPt and _scrubFrozenReserve still hold the
            // pre-scrub values from _buildImmersiveTl() — valid for reverse.
            _scrubDetached = true;
```

---

### Task 9: Clean up at rawP=0 in the pinch path

**File:** `src/arrayview/_viewer.html:16691`

When the user pinches back to zero, kill the timeline and reset transforms so `#info` re-joins the flow cleanly. Currently there is no rawP=0 cleanup in the pinch path.

- [ ] **Add rawP=0 cleanup branch inside the pinch crossfade block**

Find (lines 16688–16697):
```javascript
                        userZoom = desiredZoom;
                        _zoomAdjustedByUser = true;
                        _scrubRequestedZoom = null;
                        immersiveTl.progress(rawP);
                        if (rawP >= 1) {
                            _settleImmersive();
                        }
                        ModeRegistry.scaleAll();
                        saveState();
                        return;
```

Replace with:
```javascript
                        userZoom = desiredZoom;
                        _zoomAdjustedByUser = true;
                        _scrubRequestedZoom = null;
                        immersiveTl.progress(rawP);
                        if (rawP >= 1) {
                            _settleImmersive();
                        } else if (rawP <= 0) {
                            // Fully pinched back out — kill timeline and restore chrome.
                            immersiveTl.kill();
                            immersiveTl = null;
                            _resetImmersiveTransforms(); // clears _scrubDetached, restores #info
                            _crossfadeCollapseP = 0;
                            _crossfadeP = 0;
                        }
                        ModeRegistry.scaleAll();
                        saveState();
                        return;
```

---

### Task 10: Screenshot verification

- [ ] **Step 1: Enable pinch debug in devtools console**

```javascript
window.__pinchDebug = true;
```

- [ ] **Step 2: Screenshot at rest (rawP=0)**

Confirm: `#info` computed style shows `position: relative`, `wrapper.style.paddingTop` is a normal centering value.

- [ ] **Step 3: Advance one pinch frame, screenshot**

Confirm: `#info` computed style shows `position: fixed`, canvas is same size as step 2 (no jump).

- [ ] **Step 4: Advance to rawP≈0.5, screenshot**

Confirm: canvas has grown, `#info` is at the same pixel position as step 2, opacity ≈ 0.5. Colorbar at pre-scrub position, opacity ≈ 0.5.

- [ ] **Step 5: Advance to rawP=1, screenshot**

Confirm: canvas fills screen, `#info` and colorbar are invisible.

- [ ] **Step 6: After `_settleImmersive()` fires (150ms later), screenshot**

Confirm: `#info` and colorbar visible at overlay positions. `_scrubDetached` is false.

- [ ] **Step 7: Pinch back to rawP≈0.5, screenshot**

Confirm: canvas has shrunk, `#info` at pre-scrub pixel position, opacity ≈ 0.5, no jump.

- [ ] **Step 8: Pinch back to rawP=0, screenshot**

Confirm: `#info` back in normal flow, canvas at original size, no jump.
