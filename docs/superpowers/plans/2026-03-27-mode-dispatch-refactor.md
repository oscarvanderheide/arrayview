# Mode Dispatch & Shared Abstractions Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated mode-dispatch if-else chains and copy-pasted pan/pixel-hover logic by introducing a mode registry, PanManager class, and PixelHover helper.

**Architecture:** Add a `ModeRegistry` object that tracks the active mode and provides a single `scaleAll()` dispatch. Add a `PanManager` class that encapsulates the identical mousedown/move/up/clamp/apply drag pattern used by 3 separate pan implementations. Add a `fetchPixelInfo()` helper that consolidates the throttle+fetch+format pattern used by 4 pixel hover implementations. Simplify `positionEggs()` by extracting per-mode anchor queries into the registry.

**Tech Stack:** Vanilla JS (inline in `_viewer.html`), no build step, no external deps.

---

## File Structure

All changes are in one file:

- **Modify:** `src/arrayview/_viewer.html`
  - Add `ModeRegistry` object and mode registrations (~80 lines)
  - Add `PanManager` class (~60 lines)
  - Add `fetchPixelInfo()` helper (~40 lines)
  - Replace 6+ identical dispatch chains with single calls
  - Replace 3 pan implementations with PanManager instances
  - Replace 4 pixel hover implementations with fetchPixelInfo calls
  - Simplify positionEggs with registry-provided anchor queries
- **Test:** `tests/test_api.py` — must keep all 93 passing
- **Test:** Manual browser verification after each task

## Important Context

### The 6-way dispatch pattern (appears 6+ times identically)

```javascript
if (compareQmriActive) compareQmriScaleAllCanvases();
else if (compareMvActive) compareMvScaleAllCanvases();
else if (multiViewActive) mvScaleAllCanvases();
else if (compareActive) compareScaleCanvases();
else if (qmriActive) qvScaleAllCanvases();
else scaleCanvas(canvas.width, canvas.height);
```

Locations: lines ~2822 (resize), ~7146 (zoom+), ~7167 (zoom-), ~7189 (reset), ~11128 (wheel), ~12139 (pinch), plus partial variants at ~7465 (square stretch) and ~11400 (fullscreen).

### The 3 identical pan implementations

| Pan | State vars | Clamp fn | Apply fn |
|-----|-----------|----------|----------|
| Normal | `_panDrag`, `_panX`, `_panY` | `_clampPan()` (line 6453) | `_applyCanvasPan()` (line 6468) |
| Compare | `_cmpPanDrag`, `_cmpPanX`, `_cmpPanY` | `_cmpClampPan()` (line 1554) | `_cmpApplyPan()` (line 1567) |
| Mosaic | `_mosaicPanDrag`, `_mosaicPanX`, `_mosaicPanY` | `_mosaicClampPan()` (line 6480) | `_mosaicApplyPan()` (line 6491) |

All three follow identical structure: mousedown captures `{x, y, panX0, panY0}`, mousemove computes delta and updates, mouseup clears state.

### The 4 pixel hover implementations

| Mode | Handler line | Throttle var | Rate | API call |
|------|-------------|-------------|------|----------|
| Normal | ~7021 | `pixelHoverPending` | 50ms | `/pixel/${sid}?...` |
| Multi-view | ~8924 | `mvPixelPending` | 50ms | `/pixel/${sid}?...` |
| qMRI | ~9142 | `qvPixelPending` | 60ms | `/pixel/${sid}?...` |
| Compare | ~11188 | implicit | — | `/pixel/${sid}?...` |

All compute coordinates the same way, call the same API, format values identically, and display via `showPixelInfo()`.

### Scaling functions and their signatures

| Function | Line | Zoom cap | Return |
|----------|------|----------|--------|
| `scaleCanvas(w, h)` | 1914 | None | void |
| `compareScaleCanvases()` | 1965 | None | void |
| `mvScaleAllCanvases()` | 10883 | Hard cap | 'max' or null |
| `qvScaleAllCanvases()` | 10200 | Hard cap | 'max' or null |
| `compareQmriScaleAllCanvases()` | 9491 | Hard cap | 'max' or null |
| `compareMvScaleAllCanvases()` | 9908 | Hard cap | 'max' or null |

### Call order after scaling (all modes follow this)

```
scale() → fixWrapperAlignment() → draw colorbars → positionEggs()
```

---

## Task 1: Add `ModeRegistry` and `scaleAll()` dispatch

Create the mode registry and replace all 6-way dispatch chains with a single call.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Define the ModeRegistry object**

Insert after the existing mode flag declarations (after `let qmriActive = false;` etc., around line 1470). The registry maps mode names to their scale functions. The active mode is determined dynamically from the existing boolean flags.

```javascript
// ── Mode Registry ────────────────────────────────────────────────
const ModeRegistry = {
    /** Return the scale function for the currently active mode. */
    getScaler() {
        if (compareQmriActive) return compareQmriScaleAllCanvases;
        if (compareMvActive)   return compareMvScaleAllCanvases;
        if (multiViewActive)   return mvScaleAllCanvases;
        if (compareActive)     return compareScaleCanvases;
        if (qmriActive)        return qvScaleAllCanvases;
        return null; // normal mode — needs (w, h) args
    },

    /** Scale all canvases for the current mode. For normal mode, uses last known image dimensions. */
    scaleAll() {
        const scaler = this.getScaler();
        if (scaler) return scaler();
        if (lastImgW && lastImgH) return scaleCanvas(lastImgW, lastImgH);
        if (canvas.width && canvas.height) return scaleCanvas(canvas.width, canvas.height);
    },

    /** Scale with auto-fit (for zoom reset). */
    scaleAllAutoFit() {
        const scaler = this.getScaler();
        if (scaler) return scaler();
        _autoFitPending = true;
        if (lastImgW && lastImgH) return scaleCanvas(lastImgW, lastImgH);
        if (canvas.width && canvas.height) return scaleCanvas(canvas.width, canvas.height);
    },
};
```

NOTE: This uses the EXISTING boolean flags (`compareQmriActive`, `compareMvActive`, etc.) and EXISTING scale functions. No mode registration needed — we just centralize the dispatch logic. The precedence order matches the existing 6-way if-else exactly.

- [ ] **Step 2: Replace the window resize dispatch**

Find the `window.addEventListener('resize', ...)` handler (around line 2822). Replace the 6-way if-else with:

```javascript
ModeRegistry.scaleAll();
```

The existing handler has extra logic for compare mode calling `positionEggs()` explicitly. Check whether `compareScaleCanvases()` already calls `positionEggs()` at the end (it does, at line ~2187). If so, the explicit call is redundant and can be removed.

- [ ] **Step 3: Replace zoom+ dispatch ('+' key)**

Find the '+' key handler (around line 7146). Replace the 6-way if-else:

```javascript
// Before: 6 lines of if-else
// After:
ModeRegistry.scaleAll();
```

- [ ] **Step 4: Replace zoom- dispatch ('-' key)**

Find the '-' key handler (around line 7167). Replace the 6-way if-else with `ModeRegistry.scaleAll();`

- [ ] **Step 5: Replace zoom reset dispatch ('0' key)**

Find the '0' key handler (around line 7189). Replace the 6-way if-else with `ModeRegistry.scaleAllAutoFit();`

The `scaleAllAutoFit` method handles the `_autoFitPending` flag for normal mode.

- [ ] **Step 6: Replace wheel zoom dispatch**

Find the Ctrl+wheel handler (around line 11128). Replace the 6-way if-else with `ModeRegistry.scaleAll();`

- [ ] **Step 7: Replace pinch zoom dispatch**

Find the touch pinch handler (around line 12139). Replace the 6-way if-else with `ModeRegistry.scaleAll();`

- [ ] **Step 8: Replace partial dispatch variants**

Find the square-stretch handler ('S' key, around line 7465). Replace its 4-way if-else with `ModeRegistry.scaleAll();`

Find the fullscreen handler ('F' key, around line 11400). Replace its 3-way if-else with `ModeRegistry.scaleAll();`

- [ ] **Step 9: Run tests and verify in browser**

Run: `uv run pytest tests/test_api.py -x -q`
Expected: All 93 pass

Start server and verify no JS errors in browser console:
```bash
uv run python -c "import arrayview, numpy as np; arrayview.view(np.random.rand(100,100), window=False); import time; time.sleep(30)"
```

- [ ] **Step 10: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: add ModeRegistry.scaleAll() to replace 6-way dispatch chains"
```

---

## Task 2: Add `PanManager` class

Create a reusable class that handles the mousedown/move/up/clamp/apply pattern for canvas panning.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Define the PanManager class**

Insert after the ModeRegistry definition (around line 1490).

```javascript
// ── PanManager ───────────────────────────────────────────────────
class PanManager {
    /**
     * @param {Object} opts
     * @param {function} opts.getViewport — returns {w, h} of the clipping viewport
     * @param {function} opts.getContent — returns {w, h} of the scrollable content
     * @param {function} opts.apply — called with (x, y) to set the CSS transform
     * @param {function} [opts.onDragStart] — called when drag begins
     * @param {function} [opts.onDragEnd] — called when drag ends
     */
    constructor(opts) {
        this.opts = opts;
        this.x = 0;
        this.y = 0;
        this._drag = null; // {startX, startY, panX0, panY0}
    }

    /** Reset pan to origin. */
    reset() { this.x = 0; this.y = 0; this.opts.apply(0, 0); }

    /** Clamp pan within bounds (content must stay within viewport). */
    clamp() {
        const vp = this.opts.getViewport();
        const ct = this.opts.getContent();
        this.x = Math.max(-(ct.w - vp.w), Math.min(0, this.x));
        this.y = Math.max(-(ct.h - vp.h), Math.min(0, this.y));
    }

    /** Apply current pan position via the apply callback. */
    apply() { this.opts.apply(this.x, this.y); }

    /** Call from mousedown to start a drag. */
    startDrag(e) {
        this._drag = { startX: e.clientX, startY: e.clientY, panX0: this.x, panY0: this.y };
        if (this.opts.onDragStart) this.opts.onDragStart();
    }

    /** Call from mousemove to update drag. Returns true if dragging. */
    updateDrag(e) {
        if (!this._drag) return false;
        this.x = this._drag.panX0 + (e.clientX - this._drag.startX);
        this.y = this._drag.panY0 + (e.clientY - this._drag.startY);
        this.clamp();
        this.apply();
        return true;
    }

    /** Call from mouseup to end drag. Returns true if was dragging. */
    endDrag() {
        if (!this._drag) return false;
        this._drag = null;
        if (this.opts.onDragEnd) this.opts.onDragEnd();
        return true;
    }

    /** Whether a drag is in progress. */
    get dragging() { return this._drag !== null; }

    /** Whether content overflows viewport (pan would have effect). */
    get overflows() {
        const vp = this.opts.getViewport();
        const ct = this.opts.getContent();
        return ct.w > vp.w + 1 || ct.h > vp.h + 1;
    }
}
```

- [ ] **Step 2: Create PanManager instance for normal mode**

Find the existing `_panX`, `_panY`, `_panDrag` declarations (around line 6451) and `_clampPan()` / `_applyCanvasPan()` functions (lines 6453-6471). Create a PanManager instance nearby:

```javascript
const mainPan = new PanManager({
    getViewport: () => {
        const vp = document.getElementById('canvas-viewport');
        return { w: vp ? vp.clientWidth : window.innerWidth, h: vp ? vp.clientHeight : window.innerHeight };
    },
    getContent: () => {
        const cssW = parseInt(canvas.style.width) || canvas.width;
        const cssH = parseInt(canvas.style.height) || canvas.height;
        return { w: cssW, h: cssH };
    },
    apply: (x, y) => {
        const inner = document.getElementById('canvas-inner');
        if (inner) inner.style.transform = `translate(${x}px,${y}px)`;
    },
});
```

- [ ] **Step 3: Replace normal mode pan code with mainPan**

Replace all references to `_panDrag`, `_panX`, `_panY` with `mainPan._drag`, `mainPan.x`, `mainPan.y`.

In the mousedown handler (line ~6523): replace `_panDrag = {...}` with `mainPan.startDrag(e)`.

In the mousemove handler (line ~1771): replace the `if (_panDrag !== null)` block with `if (mainPan.updateDrag(e)) { /* done */ }`.

In the mouseup handler (line ~1813): replace `if (_panDrag !== null) { _panDrag = null; }` with `mainPan.endDrag()`.

Replace calls to `_clampPan()` with `mainPan.clamp()` and `_applyCanvasPan()` with `mainPan.apply()`.

In `scaleCanvas()` (line ~1953-1954): replace `_clampPan(); _applyCanvasPan();` with `mainPan.clamp(); mainPan.apply();`.

Where `_panX = 0; _panY = 0;` appears (zoom reset, etc.): replace with `mainPan.reset()`.

Delete the old `_panX`, `_panY`, `_panDrag`, `_clampPan()`, `_applyCanvasPan()` declarations.

- [ ] **Step 4: Create PanManager instance for compare mode**

Find `_cmpPanX`, `_cmpPanY`, `_cmpPanDrag` (line ~1545) and `_cmpClampPan()` / `_cmpApplyPan()` (lines 1554-1577). Create:

```javascript
const comparePan = new PanManager({
    getViewport: () => {
        const cv = compareCanvases[0];
        const clip = cv ? cv.closest('.compare-canvas-clip') : null;
        return clip ? { w: clip.clientWidth, h: clip.clientHeight } : { w: 0, h: 0 };
    },
    getContent: () => {
        const cv = compareCanvases[0];
        const inner = cv ? cv.closest('.compare-canvas-inner') : null;
        const cssW = cv ? (parseInt(cv.style.width) || cv.width) : 0;
        const cssH = cv ? (parseInt(cv.style.height) || cv.height) : 0;
        return { w: cssW, h: cssH };
    },
    apply: (x, y) => {
        document.querySelectorAll('.compare-canvas-inner').forEach(inner => {
            inner.style.transform = `translate(${x}px,${y}px)`;
        });
        // Counter-transform axes indicators
        document.querySelectorAll('.compare-axes-indicator').forEach(ai => {
            ai.style.transform = `translate(${-x}px,${-y}px)`;
        });
    },
});
```

Replace all compare pan references similarly. Delete old `_cmpPanX`, `_cmpPanY`, `_cmpPanDrag`, `_cmpClampPan()`, `_cmpApplyPan()`.

- [ ] **Step 5: Create PanManager instance for mosaic mode**

Find `_mosaicPanX`, `_mosaicPanY`, `_mosaicPanDrag` (line ~1494) and `_mosaicClampPan()` / `_mosaicApplyPan()` (lines 6480-6494). Create:

```javascript
const mosaicPan = new PanManager({
    getViewport: () => {
        const wrap = document.getElementById('qmri-view-wrap');
        return wrap ? { w: wrap.clientWidth, h: wrap.clientHeight } : { w: 0, h: 0 };
    },
    getContent: () => {
        const inner = document.getElementById('qmri-mosaic-inner');
        return inner ? { w: inner.scrollWidth, h: inner.scrollHeight } : { w: 0, h: 0 };
    },
    apply: (x, y) => {
        const inner = document.getElementById('qmri-mosaic-inner');
        if (inner) inner.style.transform = `translate(${x}px,${y}px)`;
    },
    onDragStart: () => {
        const inner = document.getElementById('qmri-mosaic-inner');
        if (inner) inner.classList.add('mosaic-dragging');
    },
    onDragEnd: () => {
        const inner = document.getElementById('qmri-mosaic-inner');
        if (inner) inner.classList.remove('mosaic-dragging');
    },
});
```

Replace all mosaic pan references. Delete old `_mosaicPanX`, `_mosaicPanY`, `_mosaicPanDrag`, `_mosaicClampPan()`, `_mosaicApplyPan()`.

- [ ] **Step 6: Run tests and verify in browser**

Run: `uv run pytest tests/test_api.py -x -q`

Manual verification:
- Normal mode: right-click drag to pan when zoomed in
- Compare mode: drag to pan when zoomed
- qMRI mosaic mode: drag to pan when content overflows

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: replace 3 pan implementations with PanManager class"
```

---

## Task 3: Add `fetchPixelInfo()` helper

Extract the shared pixel hover logic into a reusable function.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Define the fetchPixelInfo helper**

Insert near the existing `showPixelInfo()` function (around line 5670).

```javascript
/**
 * Fetch and display pixel info for a canvas hover event.
 * Encapsulates the throttle + coordinate computation + API call + display pattern.
 *
 * @param {Object} opts
 * @param {MouseEvent} opts.event — the mousemove event
 * @param {HTMLCanvasElement} opts.canvas — the canvas being hovered
 * @param {string} opts.sid — session ID
 * @param {number} opts.dimX — x dimension index
 * @param {number} opts.dimY — y dimension index
 * @param {number[]} opts.indices — current slice indices
 * @param {number} opts.imgW — image width in pixels
 * @param {number} opts.imgH — image height in pixels
 * @param {HTMLElement} opts.displayEl — element to show the value in
 * @param {function} [opts.onResult] — called with (data, px, py) on successful fetch
 * @returns {boolean} true if fetch was initiated (not throttled)
 */
let _pixelHoverThrottle = null;
function fetchPixelInfo(opts) {
    if (_pixelHoverThrottle) return false;
    _pixelHoverThrottle = true;
    setTimeout(() => { _pixelHoverThrottle = false; }, 50);

    const rect = opts.canvas.getBoundingClientRect();
    const px = Math.floor((opts.event.clientX - rect.left) * opts.imgW / rect.width);
    const py = Math.floor((opts.event.clientY - rect.top) * opts.imgH / rect.height);
    if (px < 0 || px >= opts.imgW || py < 0 || py >= opts.imgH) return false;

    const params = new URLSearchParams({
        dim_x: String(opts.dimX),
        dim_y: String(opts.dimY),
        indices: opts.indices.join(','),
        px: String(px),
        py: String(py),
        complex_mode: complexMode,
    });
    if (logScale) params.set('log_scale', '1');

    fetch(`/pixel/${opts.sid}?${params}`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
            if (!d) return;
            const fmt = v => {
                const av = Math.abs(v);
                if (av === 0) return '0';
                if (av >= 1e4 || (av < 1e-2 && av > 0)) return v.toExponential(3);
                return parseFloat(v.toPrecision(4)).toString();
            };
            showPixelInfo(opts.displayEl, fmt(d.value), px, py);
            if (opts.onResult) opts.onResult(d, px, py);
        })
        .catch(() => {});

    return true;
}
```

NOTE: Uses a single shared throttle instead of per-mode throttle variables. This is fine because only one mode is active at a time.

- [ ] **Step 2: Replace normal mode pixel hover**

Find the canvas mousemove handler (around line 7021). Replace the inline pixel fetch code with:

```javascript
fetchPixelInfo({
    event: e,
    canvas: canvas,
    sid: sid,
    dimX: dim_x,
    dimY: dim_y,
    indices: indices,
    imgW: canvas.width,
    imgH: canvas.height,
    displayEl: piEl,
    onResult: (d, px, py) => {
        // Update axes labels, crosshair, etc. (keep existing post-fetch logic)
    },
});
```

Keep the existing post-fetch logic (axes label updates, crosshair positioning) in the `onResult` callback.

- [ ] **Step 3: Replace multi-view pixel hover**

Find the per-view mousemove handler (around line 8924). Replace with `fetchPixelInfo(...)` using view-specific parameters.

- [ ] **Step 4: Replace qMRI pixel hover**

Find the qMRI mousemove handler (around line 9142). Replace with `fetchPixelInfo(...)`. Note: qMRI fetches for ALL views at the same pixel — the `onResult` callback should trigger fetches for other views.

- [ ] **Step 5: Replace compare pixel hover**

Find the compare mousemove handler (around line 11188). Replace with `fetchPixelInfo(...)`.

- [ ] **Step 6: Delete old throttle variables**

Remove `pixelHoverPending`, `mvPixelPending`, `qvPixelPending` declarations.

- [ ] **Step 7: Run tests and verify**

Run: `uv run pytest tests/test_api.py -x -q`

Manual: hover over canvas in each mode — pixel value should display.

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: replace 4 pixel hover implementations with fetchPixelInfo helper"
```

---

## Task 4: Add `getEggsAnchor()` to ModeRegistry and simplify positionEggs

Extract the per-mode anchor element query logic from `positionEggs()` into the ModeRegistry.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add `getEggsAnchorBottom()` to ModeRegistry**

Each mode branch in `positionEggs()` queries different elements to find the bottom edge to position eggs below. Extract this:

```javascript
// Add to ModeRegistry:
getEggsAnchorBottom() {
    if (multiViewActive) {
        const mvCbWrap = document.getElementById('mv-cb-wrap');
        if (mvCbWrap) {
            const r = mvCbWrap.getBoundingClientRect();
            if (r.height > 0) return r.bottom + 12;
        }
        const mvCanvases = document.querySelectorAll('.mv-canvas');
        if (mvCanvases.length) {
            let maxB = 0;
            mvCanvases.forEach(c => { const r = c.getBoundingClientRect(); if (r.bottom > maxB) maxB = r.bottom; });
            return maxB + 36;
        }
        return window.innerHeight - 30;
    }
    if (qmriActive || compareQmriActive || compareMvActive) {
        const views = qmriActive ? qmriViews : compareMvActive ? compareMvViews : compareQmriViews;
        if (views && views.length) {
            let maxB = 0;
            views.forEach(v => { if (v.canvas) { const r = v.canvas.getBoundingClientRect(); if (r.bottom > maxB) maxB = r.bottom; } });
            return maxB + 36;
        }
        return window.innerHeight - 30;
    }
    if (compareActive) {
        const slimWrap = document.getElementById('slim-cb-wrap');
        if (slimWrap && slimWrap.offsetParent !== null) {
            return slimWrap.getBoundingClientRect().bottom + 12;
        }
        // Check per-pane colorbars (diff mode)
        const paneCbs = document.querySelectorAll('.compare-pane-cb-island');
        if (paneCbs.length) {
            let maxB = 0;
            paneCbs.forEach(el => { const r = el.getBoundingClientRect(); if (r.bottom > maxB) maxB = r.bottom; });
            if (maxB > 0) return maxB + 8;
        }
        // Fallback to pane canvases
        const panes = document.querySelectorAll('.compare-pane.active');
        if (panes.length) {
            let maxB = 0;
            panes.forEach(p => { const r = p.getBoundingClientRect(); if (r.bottom > maxB) maxB = r.bottom; });
            return maxB + 36;
        }
        return window.innerHeight - 30;
    }
    // Normal mode
    return null; // handled specially (fullscreen vs normal)
},
```

- [ ] **Step 2: Simplify positionEggs()**

Replace the 4 mode branches with:

```javascript
function positionEggs() {
    const eggsEl = document.getElementById('mode-eggs');
    if (!eggsEl) return;
    if (!eggsEl.children.length) return;

    if (_fullscreenActive && !multiViewActive && !compareActive && !qmriActive) {
        // Fullscreen normal mode: special inline positioning (keep existing logic)
        // ... existing fullscreen branch ...
        return;
    }

    // All other modes: position below the anchor element
    const anchorBottom = ModeRegistry.getEggsAnchorBottom();
    if (anchorBottom !== null) {
        eggsEl.style.position = '';
        eggsEl.style.bottom = '';
        eggsEl.style.flexDirection = '';
        eggsEl.style.alignItems = '';
        const eggsTop = Math.min(anchorBottom, window.innerHeight - 26);
        eggsEl.style.top = eggsTop + 'px';
        eggsEl.style.left = '50%';
        eggsEl.style.transform = 'translateX(-50%)';
    } else {
        // Normal non-fullscreen mode (keep existing logic)
        // ... existing normal branch ...
    }
}
```

- [ ] **Step 3: Run tests and verify**

Run: `uv run pytest tests/test_api.py -x -q`

Manual: check eggs positioning in each mode (the small LOG/complex/mask indicator dots below canvases).

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: simplify positionEggs with ModeRegistry.getEggsAnchorBottom()"
```

---

## Task 5: Add keyboard dispatch helpers to ModeRegistry

Add helpers that encapsulate common keyboard handler mode checks, reducing the scattered per-mode guards.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add mode query helpers to ModeRegistry**

```javascript
// Add to ModeRegistry:

/** Whether any multi-pane mode is active (blocks certain single-view shortcuts). */
get isMultiPane() {
    return multiViewActive || compareActive || qmriActive || compareMvActive || compareQmriActive;
},

/** Whether a grid-based mode is active (qMRI, compare-qMRI, compare-mv). */
get isGrid() {
    return qmriActive || compareQmriActive || compareMvActive;
},

/** Get the active mode name as a string. */
get name() {
    if (compareQmriActive) return 'compareQmri';
    if (compareMvActive) return 'compareMv';
    if (multiViewActive) return 'multiView';
    if (compareActive) return 'compare';
    if (qmriActive) return 'qmri';
    return 'normal';
},
```

- [ ] **Step 2: Replace scattered mode guard patterns in keyboard handler**

Search for patterns like:
```javascript
if (!multiViewActive && !qmriActive && !compareMvActive && !compareQmriActive)
```
Replace with `if (!ModeRegistry.isMultiPane)` or similar.

Search for the `compareQmriActive || compareMvActive || multiViewActive || ...` patterns used for zoom and replace them (already done in Task 1 via `ModeRegistry.scaleAll()`).

Be CONSERVATIVE — only replace patterns that map cleanly to the new helpers. Don't force-fit complex conditionals.

- [ ] **Step 3: Run tests and verify**

Run: `uv run pytest tests/test_api.py -x -q`

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: add ModeRegistry helpers for keyboard handler mode checks"
```

---

## Task 6: Final cleanup and verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/test_api.py -x -q`
Expected: All 93 pass

- [ ] **Step 2: Start server and verify in browser**

```bash
uv run python -c "import arrayview, numpy as np; arrayview.view(np.random.rand(100,100), window=False); import time; time.sleep(60)"
```

Open browser, check:
- [ ] No JS console errors
- [ ] Normal view works (zoom +/-, pan, pixel hover, eggs position)
- [ ] Multi-view (v) works (zoom, pixel hover)
- [ ] Compare (b) works (zoom, pan, pixel hover)
- [ ] qMRI (q) works if data available
- [ ] Keyboard shortcuts work in all modes

- [ ] **Step 3: Run visual smoke test**

Run: `uv run python tests/visual_smoke.py`
Review: `smoke_output/` directory

- [ ] **Step 4: Commit any fixes**

```bash
git add src/arrayview/_viewer.html
git commit -m "chore: final cleanup after mode dispatch refactor"
```
