# Remaining TODO Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ROI shape indicator, compact mode dynamic island styling, and qMRI mosaic z-mode.

**Architecture:** All changes in `_viewer.html` (JS + CSS). Feature 3 (qMRI mosaic) also needs backend changes in `_app.py` for per-parameter z-slice rendering. Each feature is independent.

**Tech Stack:** Vanilla JS, Canvas API, CSS, FastAPI backend

---

## File Structure

| File | Changes |
|------|---------|
| `src/arrayview/_viewer.html` | All 3 features: CSS, HTML elements, JS handlers |
| `src/arrayview/_app.py` | Feature 3 only: qMRI mosaic z-slice endpoint |

---

### Task 1: ROI Shape Indicator (filmstrip-style)

**Files:**
- Modify: `src/arrayview/_viewer.html`

**Context:**
- Colormap previewer (`showColormapStrip()`, lines ~8334-8383) is the template — fade out colorbar, show icons centered where colorbar was, auto-dismiss after 1.5s
- `#colormap-strip` CSS (lines ~548-559): fixed position, flex row, opacity transition
- ROI A-key handler (lines ~4956-4974): cycles `_roiShape` through rect→circle→freehand→off
- `rectRoiMode` and `_roiShape` variables (lines ~1098-1101)
- `renderEggs()` (lines ~3328-3352): builds mode badge HTML

- [ ] **Step 1: Add CSS for `#roi-shape-strip`**

Add after `#colormap-strip` CSS (after line ~559):

```css
#roi-shape-strip {
    position:fixed; display:flex; flex-direction:row; align-items:center; gap:14px;
    opacity:0; pointer-events:none; transition:opacity 0.25s ease;
    z-index:8;
}
#roi-shape-strip.visible { opacity:1; }
.roi-shape-icon { display:flex; flex-direction:column; align-items:center; gap:3px; opacity:0.4; transition:opacity 0.15s ease; }
.roi-shape-icon.active { opacity:1; }
.roi-shape-icon svg { width:24px; height:24px; }
.roi-shape-icon.active svg { width:28px; height:28px; stroke: var(--help-key); stroke-width:2.5; }
.roi-shape-icon span { font-size:9px; color:var(--muted); font-family:monospace; }
.roi-shape-icon.active span { font-size:10px; color:var(--fg); font-weight:bold; }
```

- [ ] **Step 2: Add `#roi-shape-strip` HTML element**

Add next to `#colormap-strip` in the HTML body (near line ~870):

```html
<div id="roi-shape-strip"></div>
```

- [ ] **Step 3: Add `showRoiShapeStrip()` function**

Add after `showColormapStrip()` (after line ~8383). Follows the same pattern: fade colorbar, show icons centered on colorbar position, auto-dismiss.

```javascript
let _roiStripTimer = null;
function showRoiShapeStrip() {
    const strip = document.getElementById('roi-shape-strip');
    if (!strip) return;
    strip.innerHTML = '';
    const shapes = [
        { id: 'rect',     label: 'rect', svg: '<rect x="4" y="4" width="16" height="16" rx="1" fill="none"/>' },
        { id: 'circle',   label: 'circle', svg: '<circle cx="12" cy="12" r="9" fill="none"/>' },
        { id: 'freehand', label: 'free', svg: '<path d="M4 16 C6 4, 18 4, 20 16 C18 20, 6 20, 4 16Z" fill="none"/>' },
    ];
    for (const s of shapes) {
        const icon = document.createElement('div');
        icon.className = 'roi-shape-icon' + (s.id === _roiShape ? ' active' : '');
        const strokeColor = s.id === _roiShape ? 'var(--help-key)' : '#888';
        const strokeW = s.id === _roiShape ? '2.5' : '1.5';
        icon.innerHTML = `<svg viewBox="0 0 24 24" stroke="${strokeColor}" stroke-width="${strokeW}">${s.svg}</svg>`
                        + `<span>${s.label}</span>`;
        strip.appendChild(icon);
    }
    // Position centered on colorbar (same as cmap previewer)
    const cbWrap = document.getElementById(multiViewActive ? 'mv-cb-wrap' : 'slim-cb-wrap');
    if (cbWrap) {
        const cbRect = cbWrap.getBoundingClientRect();
        strip.style.top = (cbRect.top + cbRect.height / 2 - 16) + 'px';
        strip.style.left = (cbRect.left + cbRect.width / 2) + 'px';
        strip.style.transform = 'translateX(-50%)';
        cbWrap.style.opacity = '0';
    }
    strip.classList.add('visible');
    clearTimeout(_roiStripTimer);
    _roiStripTimer = setTimeout(() => {
        strip.classList.remove('visible');
        if (cbWrap) cbWrap.style.opacity = '1';
    }, 1500);
}
```

- [ ] **Step 4: Call `showRoiShapeStrip()` from A-key handler**

In the A-key handler (lines ~4956-4974), add `showRoiShapeStrip()` call after each shape transition. Also add `renderEggs()` calls:

```javascript
if (!rectRoiMode) {
    rectRoiMode = true; _roiShape = 'rect';
    renderEggs();
    showRoiShapeStrip();
} else if (_roiShape === 'rect') {
    _roiShape = 'circle';
    showRoiShapeStrip();
} else if (_roiShape === 'circle') {
    _roiShape = 'freehand';
    showRoiShapeStrip();
} else {
    rectRoiMode = false; _roiShape = 'rect';
    // ... existing cleanup ...
    renderEggs();
}
```

- [ ] **Step 5: Add ROI egg badge in `renderEggs()`**

In `renderEggs()` (line ~3328), add ROI badge after the existing badges:

```javascript
if (rectRoiMode)
    html += `<span class="mode-badge mode-badge-roi">ROI</span>`;
```

Add CSS for the badge (near other `.mode-badge-*` styles):

```css
.mode-badge-roi { background: rgba(45, 90, 39, 0.8); color: #7ec87e; }
```

- [ ] **Step 6: Fade out ROI badge after freehand completion**

In the freehand completion code (lines ~4475-4489), after the ROI is saved, turn off ROI mode and fade the egg:

```javascript
// After freehand ROI saved, exit ROI mode after brief delay
setTimeout(() => {
    rectRoiMode = false;
    _roiShape = 'rect';
    renderEggs();
    showStatus('ROI mode: off');
}, 1000);
```

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: ROI shape indicator filmstrip + egg badge"
```

---

### Task 2: Compact Mode Dynamic Island Styling

**Files:**
- Modify: `src/arrayview/_viewer.html`

**Context:**
- Compact mode CSS (lines ~525-545): `body.compact-mode #info` and `#slim-cb-wrap.compact-overlay`
- `_positionCompactInfo()` (lines ~8252-8262): positions `#info` at top of viewport
- `drawSlimColorbar()` compact path (lines ~2200-2242): positions colorbar at bottom of canvas

- [ ] **Step 1: Update compact `#info` CSS for pill shape**

Replace the existing `body.compact-mode #info` CSS (lines ~533-538):

```css
body.compact-mode #info {
    position: fixed; z-index: 6; font-size: 14px; margin: 0; padding: 4px 14px;
    transform: translateX(-50%); align-self: auto;
    background: rgba(30, 30, 30, 0.85); border-radius: 16px;
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: font-size 0.25s ease;
}
```

Key changes from current:
- `font-size: 14px` (was 12px — restore to original)
- `border-radius: 16px` (was 4px — pill shape)
- `background: rgba(30, 30, 30, 0.85)` (was 0.75 — slightly more opaque)
- `backdrop-filter: blur(12px)` (was 4px — stronger blur)
- `border: 1px solid rgba(255, 255, 255, 0.06)` (new — subtle edge)
- `padding: 4px 14px` (was `2px 8px` — more breathing room)

- [ ] **Step 2: Update compact colorbar CSS for pill shape**

Replace `#slim-cb-wrap.compact-overlay` CSS (lines ~539-542):

```css
body.compact-mode #slim-cb-wrap.compact-overlay {
    background: rgba(30, 30, 30, 0.85); border-radius: 12px;
    padding: 5px 8px; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
```

Key changes:
- `border-radius: 12px` (was 3px — pill shape)
- `background` opacity 0.85 (was 0.72)
- `blur(12px)` (was 4px)
- `border` added for subtle edge

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: compact mode dynamic island pill styling"
```

---

### Task 3: qMRI Mosaic Z-Mode

**Files:**
- Modify: `src/arrayview/_viewer.html` (JS layout, new mode)
- Modify: `src/arrayview/_app.py` (backend z-slice rendering per parameter)

**Context:**
- Z-key handler (lines ~4795-4808): toggles `dim_z` for mosaic mode
- qMRI mode entry (`enterQmri()`, lines ~6258+): sets up qmriViews array
- qMRI views have `v.mapIndex` (0-5), `v.colormap`, `v.label`
- Backend `/ws/{sid}` handles rendering; `dim_z` sent as parameter
- Current mosaic sends `dim_z` to backend which renders a grid of all z-slices
- qMRI has separate WebSocket per view (one per parameter map)

This feature is the most complex. The approach:

1. When Z is pressed in qMRI mode, enter a new sub-mode `qmriMosaicActive`
2. Each qMRI parameter (T₁, T₂, etc.) gets its own row of z-slice thumbnails
3. Backend renders the mosaic grid for each parameter independently (existing dim_z mechanism)
4. Client-side layout: rows with labels on left, vertical colorbar on right
5. When w/d pressed, vertical histograms replace the vertical colorbars

- [ ] **Step 1: Add state variables**

```javascript
let qmriMosaicActive = false;
```

- [ ] **Step 2: Modify Z-key handler for qMRI mode**

In the Z-key handler (lines ~4795-4808), add a branch for qMRI mode:

```javascript
} else if (e.key === 'z') {
    if (activeDim === VFIELD_T_DIM) { showStatus('time dim (vector field)'); return; }
    if (multiViewActive) { showStatus('mosaic: not available in multi-view'); return; }
    if (qmriActive) {
        // Toggle qMRI mosaic sub-mode
        qmriMosaicActive = !qmriMosaicActive;
        if (qmriMosaicActive) {
            enterQmriMosaic();
        } else {
            exitQmriMosaic();
        }
        return;
    }
    // ... existing mosaic toggle for normal mode ...
```

- [ ] **Step 3: Add CSS for qMRI mosaic layout**

```css
#qmri-mosaic-wrap { display:none; width:100%; }
#qmri-mosaic-wrap.active { display:flex; flex-direction:column; align-items:center; gap:6px; padding:0 20px; }
.qm-row { display:flex; align-items:center; gap:4px; }
.qm-row-label { font-size:13px; font-family:monospace; font-weight:bold; width:28px; text-align:right; flex-shrink:0; }
.qm-row-slices { display:flex; gap:2px; overflow-x:auto; }
.qm-row-slices canvas { image-rendering:pixelated; border-radius:2px; }
.qm-row-vcb { width:8px; flex-shrink:0; margin-left:6px; border-radius:3px; }
```

- [ ] **Step 4: Add `#qmri-mosaic-wrap` HTML element**

Add near `#qmri-view-wrap` in the HTML body:

```html
<div id="qmri-mosaic-wrap"></div>
```

- [ ] **Step 5: Implement `enterQmriMosaic()`**

This function:
1. Hides the normal qMRI grid (`#qmri-view-wrap`)
2. Shows `#qmri-mosaic-wrap`
3. For each qMRI parameter, creates a row with label, slice canvases, and vertical colorbar
4. Fetches mosaic data from backend for each parameter (reuses the existing `/ws/{sid}` with `dim_z` set)

```javascript
function enterQmriMosaic() {
    const qvWrap = document.getElementById('qmri-view-wrap');
    if (qvWrap) qvWrap.style.display = 'none';
    const wrap = document.getElementById('qmri-mosaic-wrap');
    wrap.innerHTML = '';
    wrap.classList.add('active');

    const nSlices = shape[qmriDim === dim_x ? dim_y : (qmriDim === dim_y ? dim_x : qmriDim)];
    // Actually use the z-dimension: pick a dim that's not dim_x, dim_y, or qmriDim
    const zDim = findMosaicZDim(); // find suitable z-dim

    for (const v of qmriViews) {
        const row = document.createElement('div');
        row.className = 'qm-row';
        // Label
        const label = document.createElement('div');
        label.className = 'qm-row-label';
        label.style.color = v.labelColor || '#ccc';
        label.textContent = v.shortLabel || v.label?.textContent || '';
        row.appendChild(label);
        // Slices container
        const slicesWrap = document.createElement('div');
        slicesWrap.className = 'qm-row-slices';
        row.appendChild(slicesWrap);
        // Vertical colorbar canvas
        const vcb = document.createElement('canvas');
        vcb.className = 'qm-row-vcb';
        row.appendChild(vcb);
        v._mosaicRow = { slicesWrap, vcb, label };
        wrap.appendChild(row);
    }

    // Request mosaic renders for each parameter
    qmriMosaicRenderAll();
    showStatus('qMRI mosaic: all z-slices');
}
```

- [ ] **Step 6: Implement `exitQmriMosaic()`**

```javascript
function exitQmriMosaic() {
    qmriMosaicActive = false;
    const wrap = document.getElementById('qmri-mosaic-wrap');
    if (wrap) { wrap.classList.remove('active'); wrap.innerHTML = ''; }
    const qvWrap = document.getElementById('qmri-view-wrap');
    if (qvWrap) qvWrap.style.display = '';
    // Re-render normal qMRI views
    qmriViews.forEach(v => { v.rendering = false; v.pending = false; });
    qvScaleAllCanvases();
    showStatus('qMRI mosaic: off');
}
```

- [ ] **Step 7: Implement mosaic rendering**

Each qMRI parameter needs to render its z-slices. The backend already supports `dim_z` — when set, it returns a grid of all z-slices as a single image. We reuse this mechanism:

For each qMRI view, send a WebSocket request with `dim_z` set to the z-dimension. The response is a single mosaic image. Client-side, we either:
- Display the mosaic image directly in one canvas per row (simpler, recommended)
- Or split into individual slice canvases (more complex, deferred)

**Recommended approach:** One canvas per row showing the mosaic grid, with the backend handling the tiling. This reuses existing infrastructure.

- [ ] **Step 8: Implement vertical colorbar drawing**

```javascript
function drawQmriMosaicVCb(v) {
    const vcb = v._mosaicRow?.vcb;
    if (!vcb) return;
    const stops = COLORMAP_GRADIENT_STOPS[v.colormap];
    if (!stops || !colorbarVisible()) { vcb.style.display = 'none'; return; }
    const slicesWrap = v._mosaicRow.slicesWrap;
    const h = slicesWrap.offsetHeight || 40;
    const w = 8;
    const dpr = window.devicePixelRatio || 1;
    vcb.width = Math.round(w * dpr);
    vcb.height = Math.round(h * dpr);
    vcb.style.width = w + 'px';
    vcb.style.height = h + 'px';
    vcb.style.display = '';
    const ctx = vcb.getContext('2d');
    ctx.scale(dpr, dpr);
    // Vertical gradient (top=vmax, bottom=vmin)
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    const n = stops.length;
    for (let i = 0; i < n; i++) {
        const t = n <= 1 ? 0 : i / (n - 1);
        const rgb = stops[n - 1 - i]; // reversed for top=max
        grad.addColorStop(t, `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
}
```

- [ ] **Step 9: Vertical histogram on w/d press**

When `_cbExpanded` is true in qMRI mosaic mode, replace the vertical colorbars with vertical histograms. The histogram data comes from `_fetchHistogram()` per parameter. Each vertical bar shows the distribution rotated 90 degrees.

This extends `drawQmriMosaicVCb()` to check `_cbExpanded` and draw histogram bars vertically when expanded (wider canvas, ~30px, with bars drawn bottom-to-top).

- [ ] **Step 10: Wire up w/d key handlers for mosaic mode**

In the w-key handler, add a branch for `qmriMosaicActive`:
```javascript
if (qmriMosaicActive) {
    _cbExpanded = !_cbExpanded;
    qmriViews.forEach(v => drawQmriMosaicVCb(v));
    showStatus(_cbExpanded ? 'histogram: on' : 'histogram: off');
    return;
}
```

- [ ] **Step 11: Handle exit conditions**

- Pressing Z again exits qMRI mosaic
- Pressing Q exits qMRI mode entirely (must also exit mosaic)
- Pressing V/B/etc. shows "not available in mosaic mode"

In `exitQmri()`, add: `if (qmriMosaicActive) exitQmriMosaic();`

- [ ] **Step 12: Commit**

```bash
git add src/arrayview/_viewer.html src/arrayview/_app.py
git commit -m "feat: qMRI mosaic z-mode with vertical colorbars and histograms"
```
