# Multi-Array Immersive View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend immersive mode to work when multiple arrays are loaded in compare mode — panes touching with gray border, array names hidden, colorbar and dim bar as floating islands, per-pane minimaps.

**Architecture:** Lift the compare-mode blocking condition in the keyboard handler, add CSS for immersive compare layout (zero gap, gray borders between panes, hidden titles/inline colorbars), update `_shouldEnterImmersive()` to work with compare viewport dimensions, update `compareScaleCanvases()` to call `_positionFullscreenChrome()`, and adapt `_enterImmersive()`/`_exitImmersive()` to fade compare-specific chrome elements.

**Tech Stack:** HTML/CSS/JavaScript (single file: `_viewer.html`)

---

### Task 1: CSS — Immersive compare layout

**Files:**
- Modify: `src/arrayview/_viewer.html:745-763` (fullscreen-mode CSS section)

- [ ] **Step 1: Replace existing fullscreen compare CSS with immersive layout rules**

Find the existing block at ~line 745:

```css
/* Compare mode: panes fill viewport with minimal gap */
body.fullscreen-mode #compare-view-wrap.active {
    gap: 2px;
}
body.fullscreen-mode #compare-panes {
    gap: 2px;
}
body.fullscreen-mode .compare-pane { position: relative; }
/* Per-pane name pills overlaid at top of each pane */
body.fullscreen-mode .compare-title {
    position: absolute; top: 6px; left: 50%; transform: translateX(-50%);
    z-index: 3;
    background: rgba(30, 30, 30, 0.8);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 3px 12px;
    pointer-events: none;
}
```

Replace with:

```css
/* Compare mode immersive: panes touch, gray borders, no titles */
body.fullscreen-mode #compare-view-wrap.active {
    gap: 0;
}
body.fullscreen-mode #compare-panes {
    gap: 0;
}
body.fullscreen-mode .compare-pane {
    position: relative;
    border: 1px solid var(--canvas-border);
}
/* Collapse title completely */
body.fullscreen-mode .compare-title {
    opacity: 0; max-height: 0; overflow: hidden; margin: 0; padding: 0;
    pointer-events: none;
}
/* Hide per-pane inline colorbars (replaced by shared island) */
body.fullscreen-mode .compare-pane-cb-island {
    opacity: 0; max-height: 0; overflow: hidden; margin: 0; padding: 0;
    pointer-events: none;
}
/* Remove canvas-area padding so panes fill completely */
body.fullscreen-mode .compare-canvas-area {
    padding: 0;
}
```

- [ ] **Step 2: Verify visually**

Load 2 arrays, press `=` (after later tasks enable it). Panes should touch with gray border, titles hidden, inline colorbars hidden.

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "style: add immersive CSS for compare mode — zero-gap, gray borders, hidden titles"
```

---

### Task 2: Lift compare-mode block in keyboard handler

**Files:**
- Modify: `src/arrayview/_viewer.html:7463-7465` (keyboard handler)

- [ ] **Step 1: Remove compare-mode blocking from `=` key auto-immersive condition**

Find at ~line 7463:

```javascript
if (!_fullscreenActive && !_zoomAdjustedByUser &&
    !multiViewActive && !qmriActive && !compareMvActive && !compareQmriActive &&
    _shouldEnterImmersive()) {
```

Replace with:

```javascript
if (!_fullscreenActive && !_zoomAdjustedByUser &&
    !multiViewActive && !qmriActive &&
    _shouldEnterImmersive()) {
```

This allows immersive entry in plain `compareActive` mode. `multiViewActive` and `qmriActive` remain blocked (those are mosaic-of-compare and qMRI modes which have their own layouts).

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: allow immersive mode entry in compare (multi-array) mode"
```

---

### Task 3: Update `_shouldEnterImmersive()` for compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html:11965-11980` (`_shouldEnterImmersive()`)

- [ ] **Step 1: Adapt viewport size check for compare mode**

The current function checks `#canvas-viewport` dimensions, which doesn't exist in compare mode. When `compareActive`, use `#compare-view-wrap` instead.

Find at ~line 11965:

```javascript
function _shouldEnterImmersive() {
    // Only enter immersive when height is the binding constraint —
    // i.e. the vertical space limits fit-zoom more than horizontal.
    // Immersive reclaims the 80px bottom chrome reserve; if width was
    // already limiting, entering immersive gains nothing.
    const w = lastImgW || canvas.width;
    const h = lastImgH || canvas.height;
    if (!w || !h) return false;
    const maxW = Math.max(100, window.innerWidth - 80);
    const maxH = Math.max(50, window.innerHeight - uiReserveV() - 80);
    if ((maxH / h) >= (maxW / w)) return false; // width binding: skip
    // Canvas must be wide/tall enough to fit overlaid chrome
    const vpEl = document.getElementById('canvas-viewport');
    if (!vpEl) return false;
    return vpEl.offsetWidth >= 200 && vpEl.offsetHeight >= 150;
}
```

Replace with:

```javascript
function _shouldEnterImmersive() {
    // Only enter immersive when height is the binding constraint —
    // i.e. the vertical space limits fit-zoom more than horizontal.
    // Immersive reclaims the 80px bottom chrome reserve; if width was
    // already limiting, entering immersive gains nothing.
    const w = lastImgW || canvas.width;
    const h = lastImgH || canvas.height;
    if (!w || !h) return false;
    const maxW = Math.max(100, window.innerWidth - 80);
    const maxH = Math.max(50, window.innerHeight - uiReserveV() - 80);
    if ((maxH / h) >= (maxW / w)) return false; // width binding: skip
    // Container must be wide/tall enough to fit overlaid chrome
    const vpEl = compareActive
        ? document.getElementById('compare-view-wrap')
        : document.getElementById('canvas-viewport');
    if (!vpEl) return false;
    return vpEl.offsetWidth >= 200 && vpEl.offsetHeight >= 150;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: _shouldEnterImmersive checks compare-view-wrap in compare mode"
```

---

### Task 4: Update `compareScaleCanvases()` for immersive layout

**Files:**
- Modify: `src/arrayview/_viewer.html:2216-2445` (`compareScaleCanvases()`)

- [ ] **Step 1: Adjust layout reserves for immersive mode**

In `compareScaleCanvases()`, the layout reserves (padding, chrome space) need to change when in immersive mode. Find at ~line 2219:

```javascript
const titleReserve = 28;
const panePadX = 16; // .compare-canvas-area horizontal padding (8 + 8)
```

Replace with:

```javascript
const titleReserve = _fullscreenActive ? 0 : 28;
const panePadX = _fullscreenActive ? 0 : 16;
```

Find at ~line 2244:

```javascript
const uiReserveH = uiReserveV() + 40; // status/info + compare colorbar
const totalW = Math.max(100, window.innerWidth - 80);
const totalH = Math.max(100, window.innerHeight - uiReserveH);
```

Replace with:

```javascript
const uiReserveH = _fullscreenActive ? 0 : (uiReserveV() + 40);
const totalW = Math.max(100, window.innerWidth - (_fullscreenActive ? 0 : 80));
const totalH = Math.max(100, window.innerHeight - uiReserveH);
```

- [ ] **Step 2: Add `_positionFullscreenChrome()` call at end of function**

Find at ~line 2438 (near end of `compareScaleCanvases`):

```javascript
drawAllComparePaneCbs();
if (registrationMode) drawRegBlendCb();
drawSlimColorbar(cbMarkerFrac);
fixWrapperAlignment();
positionEggs();
updateMiniMap();
```

Replace with:

```javascript
drawAllComparePaneCbs();
if (registrationMode) drawRegBlendCb();
drawSlimColorbar(cbMarkerFrac);
fixWrapperAlignment();
positionEggs();
updateMiniMap();
if (_fullscreenActive) _positionFullscreenChrome();
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: compareScaleCanvases adjusts reserves for immersive, calls _positionFullscreenChrome"
```

---

### Task 5: Update `_positionFullscreenChrome()` for compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html:12148-12196` (`_positionFullscreenChrome()`)

- [ ] **Step 1: Use compare-view-wrap as positioning anchor when in compare mode**

When `compareActive`, the dim bar and colorbar should be positioned relative to the `#compare-view-wrap` bounding rect instead of `#canvas-viewport`. The minimap positioning is per-pane (handled by CSS `position:absolute` inside each pane), so skip the global minimap positioning in compare mode.

Find at ~line 12148:

```javascript
function _positionFullscreenChrome() {
    if (!_fullscreenActive) return;
    const vp = document.getElementById('canvas-viewport');
    if (!vp) return;
    const vpRect = vp.getBoundingClientRect();
    // Position dim bar inside canvas, 8px from canvas top
    const infoEl = document.getElementById('info');
    if (infoEl) {
        infoEl.style.top = (vpRect.top + 8) + 'px';
    }
    // Position colorbar overlaid near canvas bottom (10px margin from canvas bottom edge)
    const cbWrap = document.getElementById('slim-cb-wrap');
    if (cbWrap && cbWrap.style.display !== 'none') {
        cbWrap.style.top = '';
        // bottom = distance from viewport bottom to canvas bottom + 10px margin
        const cbBottom = Math.max(10, window.innerHeight - vpRect.bottom + 10);
        cbWrap.style.bottom = cbBottom + 'px';
    }
    // Position minimap: prefer northeast (inside canvas), fallback to east (outside) if dimbar overlaps
    if (_miniMap) {
        const mmW = _miniMap.offsetWidth || 100;
        const mmH = _miniMap.offsetHeight || 100;
        const mmMargin = 8;
        // Northeast candidate: top-right inside canvas
        const neTop = vpRect.top + mmMargin;
        const neRight = vpRect.right - mmW - mmMargin;
        // Check overlap with dimbar island (#info)
        const infoEl = document.getElementById('info');
        let overlaps = false;
        if (infoEl && infoEl.offsetWidth) {
            const infoRect = infoEl.getBoundingClientRect();
            // Overlap if minimap rect intersects info rect
            overlaps = !(neRight + mmW < infoRect.left ||
                         neRight > infoRect.right ||
                         neTop + mmH < infoRect.top ||
                         neTop > infoRect.bottom);
        }
        if (overlaps) {
            // East fallback: right of canvas, vertically centered
            _miniMap.style.top = (vpRect.top + (vpRect.height - mmH) / 2) + 'px';
            _miniMap.style.left = (vpRect.right + 10) + 'px';
        } else {
            // Northeast: inside canvas, top-right corner
            _miniMap.style.top = neTop + 'px';
            _miniMap.style.left = neRight + 'px';
        }
        _miniMap.style.right = 'auto';
    }
}
```

Replace with:

```javascript
function _positionFullscreenChrome() {
    if (!_fullscreenActive) return;
    // Use compare-view-wrap as anchor in compare mode, canvas-viewport otherwise
    const vp = compareActive
        ? document.getElementById('compare-view-wrap')
        : document.getElementById('canvas-viewport');
    if (!vp) return;
    const vpRect = vp.getBoundingClientRect();
    // Position dim bar inside canvas, 8px from canvas top
    const infoEl = document.getElementById('info');
    if (infoEl) {
        infoEl.style.top = (vpRect.top + 8) + 'px';
    }
    // Position colorbar overlaid near canvas bottom (10px margin from canvas bottom edge)
    const cbWrap = document.getElementById('slim-cb-wrap');
    if (cbWrap && cbWrap.style.display !== 'none') {
        cbWrap.style.top = '';
        // bottom = distance from viewport bottom to canvas bottom + 10px margin
        const cbBottom = Math.max(10, window.innerHeight - vpRect.bottom + 10);
        cbWrap.style.bottom = cbBottom + 'px';
    }
    // Per-pane minimaps in compare mode are positioned via CSS (absolute within pane)
    // Only position global minimap for single-view mode
    if (!compareActive && _miniMap) {
        const mmW = _miniMap.offsetWidth || 100;
        const mmH = _miniMap.offsetHeight || 100;
        const mmMargin = 8;
        // Northeast candidate: top-right inside canvas
        const neTop = vpRect.top + mmMargin;
        const neRight = vpRect.right - mmW - mmMargin;
        // Check overlap with dimbar island (#info)
        const infoEl2 = document.getElementById('info');
        let overlaps = false;
        if (infoEl2 && infoEl2.offsetWidth) {
            const infoRect = infoEl2.getBoundingClientRect();
            overlaps = !(neRight + mmW < infoRect.left ||
                         neRight > infoRect.right ||
                         neTop + mmH < infoRect.top ||
                         neTop > infoRect.bottom);
        }
        if (overlaps) {
            _miniMap.style.top = (vpRect.top + (vpRect.height - mmH) / 2) + 'px';
            _miniMap.style.left = (vpRect.right + 10) + 'px';
        } else {
            _miniMap.style.top = neTop + 'px';
            _miniMap.style.left = neRight + 'px';
        }
        _miniMap.style.right = 'auto';
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: _positionFullscreenChrome uses compare-view-wrap anchor in compare mode"
```

---

### Task 6: Update `_enterImmersive()` and `_exitImmersive()` for compare chrome

**Files:**
- Modify: `src/arrayview/_viewer.html:12009-12146` (`_enterImmersive`, `_exitImmersive`)

- [ ] **Step 1: Update `_enterImmersive()` to fade compare-specific elements**

The compare titles and per-pane colorbars need to be faded out during Phase 1 and excluded from Phase 3 fade-in (they stay hidden in immersive). Find at ~line 12009:

```javascript
function _enterImmersive() {
    if (_fullscreenActive || _immersiveAnimating) return;
    _immersiveAnimating = true;

    const infoEl = document.getElementById('info');
    const cbWrap = document.getElementById('slim-cb-wrap');
    const nameEl = document.getElementById('array-name');
    const FADE_OUT = 150, GROW = 160, FADE_IN = 150;

    // Phase 1: fade out non-immersive chrome
    _animOpacity([infoEl, cbWrap, nameEl], 1, 0, FADE_OUT, () => {
        if (!_immersiveAnimating) return;

        // Phase 2: switch to immersive layout, animate canvas grow
        const w = lastImgW || canvas.width;
        const h = lastImgH || canvas.height;
        const fromZoom   = _fitZoom;
        const newFitZoom = _calcAutoFitZoom(w, h, true);

        _fullscreenActive = true;
        document.body.classList.add('fullscreen-mode');

        // Render canvas at fromZoom in immersive layout (no auto-fit flash)
        _autoFitPending = false;
        userZoom = fromZoom;
        ModeRegistry.scaleAll();

        // Animate fromZoom → newFitZoom. _positionFullscreenChrome runs
        // on every frame so islands are at correct positions by phase 3.
        userZoom      = newFitZoom;
        _fitZoom      = newFitZoom;
        _zoomRendered = fromZoom;
        _scaleAllWithAnim();

        // Phase 3: fade in islands (already positioned correctly)
        setTimeout(() => {
            if (!_immersiveAnimating) return;
            _animOpacity([infoEl, cbWrap], 0, 1, FADE_IN, () => {
                _immersiveAnimating = false;
                for (const el of [infoEl, cbWrap, nameEl]) {
                    if (el) { el.style.transition = ''; el.style.opacity = ''; }
                }
            });
        }, GROW);
    });
}
```

Replace with:

```javascript
function _enterImmersive() {
    if (_fullscreenActive || _immersiveAnimating) return;
    _immersiveAnimating = true;

    const infoEl = document.getElementById('info');
    const cbWrap = document.getElementById('slim-cb-wrap');
    const nameEl = document.getElementById('array-name');
    // In compare mode, also fade out titles and per-pane colorbars
    const cmpTitles = compareActive ? Array.from(document.querySelectorAll('.compare-title')) : [];
    const cmpCbIslands = compareActive ? Array.from(document.querySelectorAll('.compare-pane-cb-island')) : [];
    const fadeOutEls = [infoEl, cbWrap, nameEl, ...cmpTitles, ...cmpCbIslands];
    const FADE_OUT = 150, GROW = 160, FADE_IN = 150;

    // Phase 1: fade out non-immersive chrome
    _animOpacity(fadeOutEls, 1, 0, FADE_OUT, () => {
        if (!_immersiveAnimating) return;

        // Phase 2: switch to immersive layout, animate canvas grow
        const w = lastImgW || canvas.width;
        const h = lastImgH || canvas.height;
        const fromZoom   = _fitZoom;
        const newFitZoom = _calcAutoFitZoom(w, h, true);

        _fullscreenActive = true;
        document.body.classList.add('fullscreen-mode');

        // Render canvas at fromZoom in immersive layout (no auto-fit flash)
        _autoFitPending = false;
        userZoom = fromZoom;
        ModeRegistry.scaleAll();

        // Animate fromZoom → newFitZoom. _positionFullscreenChrome runs
        // on every frame so islands are at correct positions by phase 3.
        userZoom      = newFitZoom;
        _fitZoom      = newFitZoom;
        _zoomRendered = fromZoom;
        _scaleAllWithAnim();

        // Phase 3: fade in islands (already positioned correctly)
        setTimeout(() => {
            if (!_immersiveAnimating) return;
            _animOpacity([infoEl, cbWrap], 0, 1, FADE_IN, () => {
                _immersiveAnimating = false;
                for (const el of [infoEl, cbWrap, nameEl]) {
                    if (el) { el.style.transition = ''; el.style.opacity = ''; }
                }
                // Clear inline opacity on compare elements (CSS handles hiding)
                for (const el of [...cmpTitles, ...cmpCbIslands]) {
                    if (el) { el.style.transition = ''; el.style.opacity = ''; }
                }
            });
        }, GROW);
    });
}
```

- [ ] **Step 2: Update `_exitImmersive()` to restore compare chrome**

In the exit animation, compare titles and per-pane colorbars need to be faded back in during Phase 3. Also, the snap-cancel path needs to clear their inline styles.

Find the snap-cancel block at ~line 12060:

```javascript
if (_immersiveAnimating) {
    _immersiveAnimating = false;
    if (_zoomAnimId) { cancelAnimationFrame(_zoomAnimId); _zoomAnimId = null; }
    const infoEl = document.getElementById('info');
    const cbWrap = document.getElementById('slim-cb-wrap');
    const nameEl = document.getElementById('array-name');
    for (const el of [infoEl, cbWrap, nameEl]) {
        if (el) { el.style.transition = ''; el.style.opacity = ''; }
    }
```

Replace with:

```javascript
if (_immersiveAnimating) {
    _immersiveAnimating = false;
    if (_zoomAnimId) { cancelAnimationFrame(_zoomAnimId); _zoomAnimId = null; }
    const infoEl = document.getElementById('info');
    const cbWrap = document.getElementById('slim-cb-wrap');
    const nameEl = document.getElementById('array-name');
    for (const el of [infoEl, cbWrap, nameEl]) {
        if (el) { el.style.transition = ''; el.style.opacity = ''; }
    }
    // Clear compare element inline styles
    document.querySelectorAll('.compare-title, .compare-pane-cb-island').forEach(el => {
        el.style.transition = ''; el.style.opacity = '';
    });
```

Find the Phase 3 fade-in at ~line 12138:

```javascript
_animOpacity([infoEl, cbWrap, nameEl], 0, 1, FADE_IN, () => {
    _immersiveAnimating = false;
    for (const el of [infoEl, cbWrap, nameEl]) {
        if (el) { el.style.transition = ''; el.style.opacity = ''; }
    }
});
```

Replace with:

```javascript
const cmpEls = Array.from(document.querySelectorAll('.compare-title, .compare-pane-cb-island'));
_animOpacity([infoEl, cbWrap, nameEl, ...cmpEls], 0, 1, FADE_IN, () => {
    _immersiveAnimating = false;
    for (const el of [infoEl, cbWrap, nameEl, ...cmpEls]) {
        if (el) { el.style.transition = ''; el.style.opacity = ''; }
    }
});
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: _enterImmersive/_exitImmersive fade compare titles and per-pane colorbars"
```

---

### Task 7: Update `positionEggs()` for immersive compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html:5756` (`positionEggs()`)

- [ ] **Step 1: Allow fullscreen egg positioning in compare mode**

The eggs positioning currently skips fullscreen-specific positioning when `compareActive`. It should use the same centered positioning but anchor to `#compare-view-wrap`.

Find at ~line 5756:

```javascript
if (_fullscreenActive && !multiViewActive && !compareActive && !qmriActive && !compareQmriActive && !compareMvActive) {
```

Replace with:

```javascript
if (_fullscreenActive && !multiViewActive && !qmriActive && !compareQmriActive && !compareMvActive) {
```

Then find the `vpEl` reference inside that block at ~line 5757:

```javascript
const vpEl = document.getElementById('canvas-viewport');
```

Replace with:

```javascript
const vpEl = compareActive
    ? document.getElementById('compare-view-wrap')
    : document.getElementById('canvas-viewport');
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: positionEggs supports immersive compare mode"
```

---

### Task 8: Update `_calcAutoFitZoom()` for compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html:11934-11944` (`_calcAutoFitZoom()`)

- [ ] **Step 1: Adjust reserves for compare mode in immersive**

In compare mode, the zoom calculation is handled by `compareScaleCanvases()` which has its own reserve logic (updated in Task 4). The `_calcAutoFitZoom()` function is only used by `_enterImmersive`/`_exitImmersive` to compute target zoom values. For compare mode, it should use the same reserves as `compareScaleCanvases()`.

Find at ~line 11934:

```javascript
function _calcAutoFitZoom(w, h, forImmersive) {
    if (!w || !h) return 1;
    const immersivePad = 12;
    const bottomReserve = forImmersive ? 0 : 80;
    const maxW = Math.max(100, window.innerWidth  - (forImmersive ? immersivePad * 2 : 80));
    const maxH = Math.max(50,  window.innerHeight - uiReserveV() - bottomReserve
                                                  - (forImmersive ? immersivePad * 2 : 0));
    const baseScale = getBaseScale(w, h);
    const capScale  = Math.min(maxW / w, maxH / h);
    return Math.max(1.0, capScale / baseScale);
}
```

Replace with:

```javascript
function _calcAutoFitZoom(w, h, forImmersive) {
    if (!w || !h) return 1;
    if (compareActive) {
        // In compare mode, let compareScaleCanvases handle the actual sizing.
        // Return a reasonable fit zoom; the compare scaler will compute exact per-pane sizing.
        const uiH = forImmersive ? 0 : (uiReserveV() + 40);
        const maxW = Math.max(100, window.innerWidth - (forImmersive ? 0 : 80));
        const maxH = Math.max(100, window.innerHeight - uiH);
        const baseScale = getBaseScale(w, h);
        const capScale = Math.min(maxW / w, maxH / h);
        return Math.max(1.0, capScale / baseScale);
    }
    const immersivePad = 12;
    const bottomReserve = forImmersive ? 0 : 80;
    const maxW = Math.max(100, window.innerWidth  - (forImmersive ? immersivePad * 2 : 80));
    const maxH = Math.max(50,  window.innerHeight - uiReserveV() - bottomReserve
                                                  - (forImmersive ? immersivePad * 2 : 0));
    const baseScale = getBaseScale(w, h);
    const capScale  = Math.min(maxW / w, maxH / h);
    return Math.max(1.0, capScale / baseScale);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: _calcAutoFitZoom handles compare mode reserves"
```

---

### Task 9: Verify and fix edge cases

**Files:**
- Modify: `src/arrayview/_viewer.html` (various locations)

- [ ] **Step 1: Test 2-array immersive**

Run: `python -c "import numpy as np; import arrayview as av; av.view(np.random.rand(256,256), np.random.rand(256,256))"`

Verify:
- Press `=` → immersive mode activates, panes touch with gray border
- Titles hidden, inline colorbars hidden
- Dim bar island at top-center, colorbar island at bottom-center
- Press `-` → exits cleanly back to normal compare layout
- Titles and per-pane colorbars reappear

- [ ] **Step 2: Test 3-4 array immersive**

Run: `python -c "import numpy as np; import arrayview as av; av.view(np.random.rand(256,256), np.random.rand(256,256), np.random.rand(256,256))"`

Verify grid layout in immersive mode, all panes touching with borders.

- [ ] **Step 3: Test diff mode immersive**

Load 2 arrays, activate diff mode (press `d`), then press `=`. Verify diff pane visible as regular pane.

- [ ] **Step 4: Test per-pane minimaps**

Zoom in with `=` past fit. Verify per-pane minimaps appear at top-right of each pane.

- [ ] **Step 5: Test auto-exit on zoom out**

While immersive and zoomed, press `-` until back to fit. Should auto-exit immersive.

- [ ] **Step 6: Test window resize**

While immersive, resize window to make width the binding constraint. Should auto-exit or layout should adapt.

- [ ] **Step 7: Fix any issues found and commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: edge cases in multi-array immersive mode"
```

---

## Verification Checklist

1. Load 2 arrays → `=` enters immersive → panes touch with gray border, titles hidden
2. Dim bar island at top-center, colorbar island at bottom-center
3. `=` again → zooms in, per-pane minimaps appear
4. `-` zooms out → auto-exits immersive when at fit
5. 3-4 arrays in grid → immersive works correctly
6. Diff mode → diff pane visible in immersive, colorbars as islands
7. Animation smooth (fade-out → grow → fade-in, ~460ms)
8. Single-array immersive unchanged (regression check)
