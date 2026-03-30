# Dynamic Island Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ROI and SEGMENT egg badges + separate panels with a unified draggable dynamic island UI — one island at a time, morph transition between them.

**Architecture:** A single `#dynamic-island` DOM element serves both ROI and SEG modes. Its content is rendered by `renderIsland()` based on which mode is active (`rectRoiMode` or `_segMode`). The island is draggable, remembers position, and morphs (crossfade + height animation) when switching between modes. Existing `#roi-panel`, `#seg-panel`, and ROI/SEG egg badges are removed.

**Tech Stack:** HTML/CSS/JS in `_viewer.html` (single-file viewer), Python FastAPI for new multi-dim ROI export endpoint in `_server.py`.

**Spec:** `docs/superpowers/specs/2026-03-30-dynamic-island-design.md`

---

### Task 1: Add dynamic island HTML + CSS

**Files:**
- Modify: `src/arrayview/_viewer.html:1354-1357` (HTML structure)
- Modify: `src/arrayview/_viewer.html:529-570` (CSS for roi-panel, seg-panel)

- [ ] **Step 1: Add island HTML element**

After line 1355 (`<canvas id="mv-orientation"></canvas>`), replace lines 1356-1357 (the `#roi-panel` and `#seg-panel` divs) with the dynamic island:

```html
<div id="dynamic-island"></div>
```

- [ ] **Step 2: Add island CSS**

Replace the `#roi-panel` CSS block (lines 529-550) and `#seg-panel` CSS block (lines 552-570) with the unified island CSS:

```css
#dynamic-island {
    position: fixed; left: 16px; top: 50%; transform: translateY(-50%);
    background: rgba(24,24,24,0.95); border: 1px solid rgba(255,255,255,0.07);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-radius: 14px; padding: 10px 12px;
    font-size: 11px; color: var(--text); font-family: var(--mono);
    display: none; z-index: 5; min-width: 200px;
    transition: height 0.3s ease, opacity 0.3s ease;
    cursor: default; user-select: none;
}
#dynamic-island.visible { display: block; }
#dynamic-island .island-header {
    display: flex; align-items: center; gap: 6px; margin-bottom: 8px; cursor: grab;
}
#dynamic-island .island-header:active { cursor: grabbing; }
#dynamic-island .island-drag-handle {
    width: 14px; display: flex; flex-direction: column; gap: 2px; opacity: 0.2; flex-shrink: 0;
}
#dynamic-island .island-drag-handle span { display: block; height: 1px; background: #fff; }
#dynamic-island .island-title { font-weight: 600; flex-shrink: 0; }
#dynamic-island .island-title-roi { color: #e8e8e8; }
#dynamic-island .island-title-seg { color: #e8e8e8; }
#dynamic-island .island-dot { color: #333; flex-shrink: 0; }
#dynamic-island .island-shapes { display: flex; align-items: center; gap: 5px; flex: 1; }
#dynamic-island .island-shapes svg { cursor: pointer; transition: opacity 0.15s; }
#dynamic-island .island-shapes svg:hover { opacity: 0.8 !important; }
#dynamic-island .island-download {
    flex-shrink: 0; cursor: pointer; opacity: 0.4; transition: opacity 0.15s;
}
#dynamic-island .island-download:hover { opacity: 0.9; }
#dynamic-island .island-rows { display: flex; flex-direction: column; gap: 3px; }
#dynamic-island .island-row {
    display: flex; align-items: center; gap: 5px; padding: 3px 5px; border-radius: 4px;
    background: rgba(255,255,255,0.02);
}
#dynamic-island .island-row.selected { border: 1px solid rgba(240,198,116,0.12); }
#dynamic-island .island-row.selected-roi { background: rgba(240,198,116,0.08); border-color: rgba(240,198,116,0.12); }
#dynamic-island .island-row.selected-seg { background: rgba(232,122,175,0.08); border-color: rgba(232,122,175,0.12); }
#dynamic-island .island-swatch { width: 7px; height: 7px; border-radius: 2px; flex-shrink: 0; }
#dynamic-island .island-name {
    color: #ccc; flex: 1; font-size: 10px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#dynamic-island .island-name.editable { border-bottom: 1px dashed rgba(255,255,255,0.1); padding-bottom: 1px; cursor: text; }
#dynamic-island .island-name.default-name { font-style: italic; color: #999; }
#dynamic-island .island-stats { color: #777; font-size: 9px; white-space: nowrap; }
#dynamic-island .island-delete { color: #444; font-size: 10px; cursor: pointer; flex-shrink: 0; line-height: 1; }
#dynamic-island .island-delete:hover { color: #ff5050; }
/* Light theme */
:root.light #dynamic-island { background: rgba(245,245,245,0.95); border-color: rgba(0,0,0,0.1); }
:root.light #dynamic-island .island-name { color: #333; }
:root.light #dynamic-island .island-name.default-name { color: #888; }
:root.light #dynamic-island .island-stats { color: #666; }
:root.light #dynamic-island .island-delete { color: #aaa; }
:root.light #dynamic-island .island-delete:hover { color: #cc3030; }
:root.light #dynamic-island .island-drag-handle span { background: #333; }
```

- [ ] **Step 3: Remove old panel CSS**

Delete the old `#roi-panel` CSS (lines 529-550), `#seg-panel` CSS (lines 552-570), `#roi-export-btn` CSS (lines 549-550), `.seg-btn` CSS, `.roi-delete` CSS, and `#roi-panel table/th/td` CSS. These are all replaced by the island CSS above.

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): add dynamic island HTML element and CSS"
```

---

### Task 2: Implement `renderIsland()` — ROI content

**Files:**
- Modify: `src/arrayview/_viewer.html:6000-6045` (near `renderEggs()`)

- [ ] **Step 1: Add island icon constants**

Add these constants near the existing state variables (around line 1588, after `_rois`):

```javascript
let _islandDragPos = null;  // {x, y} — remembered drag position, null = default left
let _islandMorphing = false;

const _ROI_SHAPE_ICONS = {
    rect:      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><rect x="3" y="3" width="18" height="18" rx="1" stroke="currentColor" stroke-width="2.5"/></svg>',
    circle:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2.5"/></svg>',
    freehand:  '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M4 18 Q8 6, 12 14 Q16 22, 20 8" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" fill="none"/></svg>',
    floodfill: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M12 3 Q5 12 5 15.5 A7 7 0 0 0 19 15.5 Q19 12 12 3Z" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round" fill="none"/></svg>',
};
const _SEG_METHOD_ICONS = {
    click:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" fill="currentColor"/><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="2.5" fill="none"/></svg>',
    circle:   '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2.5"/></svg>',
    scribble: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M4 18 Q8 6, 12 14 Q16 22, 20 8" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" fill="none"/></svg>',
    lasso:    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none"><path d="M6 16 Q4 8, 12 6 Q20 4, 18 14 Q16 20, 6 16 Z" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" fill="none"/></svg>',
};
const _DOWNLOAD_ICON = '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M7 2v7"/><path d="M4 7l3 3 3-3"/><path d="M2 11h10"/></svg>';
```

- [ ] **Step 2: Add `renderIsland()` function**

Add this function right after `renderEggs()` (after line 6045):

```javascript
function renderIsland() {
    const el = document.getElementById('dynamic-island');
    if (!el) return;

    const active = rectRoiMode ? 'roi' : _segMode ? 'seg' : null;
    if (!active) {
        el.classList.remove('visible');
        return;
    }

    const isRoi = active === 'roi';
    const title = isRoi ? 'ROI' : 'SEG';
    const accentColor = isRoi ? '#f0c674' : '#e87aaf';
    const shapes = isRoi ? _ROI_SHAPE_ICONS : _SEG_METHOD_ICONS;
    const currentShape = isRoi ? _roiShape : _segMethod;
    const shapeKeys = Object.keys(shapes);

    // Header: drag handle + title + shape icons + download
    let html = `<div class="island-header">`;
    html += `<div class="island-drag-handle"><span></span><span></span><span></span></div>`;
    html += `<span class="island-title island-title-${active}">${title}</span>`;
    html += `<span class="island-dot">·</span>`;
    html += `<div class="island-shapes">`;
    for (const key of shapeKeys) {
        const isActive = key === currentShape;
        const color = isActive ? accentColor : '#555';
        const opacity = isActive ? '1' : '0.5';
        html += `<span data-shape="${key}" style="color:${color};opacity:${opacity}" onclick="_islandSetShape('${key}')">${shapes[key]}</span>`;
    }
    html += `</div>`;
    html += `<span class="island-download" onclick="_islandExport()">${_DOWNLOAD_ICON}</span>`;
    html += `</div>`;

    // Rows
    html += `<div class="island-rows">`;
    if (isRoi) {
        html += _renderRoiRows();
    } else {
        html += _renderSegRows();
    }
    html += `</div>`;

    el.innerHTML = html;
    el.classList.add('visible');
    _positionIsland();
}

function _renderRoiRows() {
    const fmt = v => {
        const a = Math.abs(v);
        if (a === 0) return '0';
        if (a >= 1e4 || (a < 1e-2 && a > 0)) return v.toExponential(3);
        return parseFloat(v.toPrecision(4)).toString();
    };
    let html = '';
    for (let i = 0; i < _rois.length; i++) {
        const r = _rois[i];
        const c = _roiColors[i % _roiColors.length];
        const s = r.stats;
        const selected = i === 0 ? ' selected-roi' : '';
        html += `<div class="island-row${selected}">`;
        html += `<div class="island-swatch" style="background:${c.stroke}"></div>`;
        html += `<span class="island-name">ROI ${i + 1}</span>`;
        if (s) {
            html += `<span class="island-stats">${fmt(s.mean)} ± ${fmt(s.std)}</span>`;
        }
        html += `<span class="island-delete" data-roi="${i}" onclick="_islandDeleteRoi(${i})" title="Delete ROI ${i + 1}">×</span>`;
        html += `</div>`;
    }
    return html;
}
```

- [ ] **Step 3: Remove ROI badge from `renderEggs()`**

In `renderEggs()`, delete the ROI badge block (lines 6024-6031):

```javascript
// DELETE this entire block:
if (rectRoiMode) {
    const roiIcons = { ... };
    html += `<span class="mode-badge mode-badge-roi">ROI ${roiIcons[_roiShape] || ''}</span>`;
}
```

- [ ] **Step 4: Add `renderIsland()` call after `renderEggs()` calls**

Add `renderIsland();` after every call to `renderEggs()` in the ROI A-key handler (line 8532) and wherever `renderEggs()` is called after ROI state changes. The simplest approach: add it at the end of `renderEggs()` itself, after `if (html) positionEggs();`:

```javascript
// At end of renderEggs(), after line 6044:
renderIsland();
```

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): render ROI island with shape strip and stats rows"
```

---

### Task 3: Implement `renderIsland()` — SEG content + editable names

**Files:**
- Modify: `src/arrayview/_viewer.html` (add `_renderSegRows()`, update seg handlers)

- [ ] **Step 1: Add `_renderSegRows()` function**

Add right after `_renderRoiRows()`:

```javascript
let _segLabelNames = {};  // {labelNumber: 'user name'} — persisted per session

function _renderSegRows() {
    let html = '';
    if (!_segLabels || !_segLabels.length) return html;
    for (let i = 0; i < _segLabels.length; i++) {
        const l = _segLabels[i];
        const userSet = _segLabelNames[l.label] !== undefined;
        const name = userSet ? _segLabelNames[l.label] : l.name;
        const nameClass = userSet ? 'editable' : 'editable default-name';
        const selected = i === 0 ? ' selected-seg' : '';
        html += `<div class="island-row${selected}">`;
        html += `<div class="island-swatch" style="background:${l.color}"></div>`;
        html += `<span class="island-name ${nameClass}" data-label="${l.label}" onclick="_islandEditSegName(this)" title="Click to rename">${name}</span>`;
        html += `<span class="island-delete" onclick="_islandDeleteSeg(${l.label})" title="Delete segment">×</span>`;
        html += `</div>`;
    }
    return html;
}
```

- [ ] **Step 2: Add `_segLabels` state and update `_segUpdatePanel()` to store labels**

Add a state variable near the other seg variables (around line 1583):

```javascript
let _segLabels = [];  // labels array from server, kept for island rendering
```

Modify `_segUpdatePanel(labels)` (line 7874) to store labels and render the island instead of the old panel:

```javascript
function _segUpdatePanel(labels) {
    _segLabels = labels || [];
    renderIsland();
}
```

- [ ] **Step 3: Add editable name handler**

```javascript
function _islandEditSegName(spanEl) {
    const label = parseInt(spanEl.dataset.label);
    const current = spanEl.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = current;
    input.style.cssText = 'background:transparent;border:none;border-bottom:1px solid rgba(255,255,255,0.3);color:#ccc;font-size:10px;font-family:var(--mono);width:100%;outline:none;padding:0;';
    spanEl.replaceWith(input);
    input.focus();
    input.select();
    const commit = () => {
        const val = input.value.trim();
        if (val) _segLabelNames[label] = val;
        renderIsland();
    };
    input.addEventListener('blur', commit);
    input.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); input.blur(); }
        if (e.key === 'Escape') { input.blur(); }
        e.stopPropagation();  // prevent viewer shortcuts while editing
    });
    input.addEventListener('keypress', e => e.stopPropagation());
    input.addEventListener('keyup', e => e.stopPropagation());
}
window._islandEditSegName = _islandEditSegName;
```

- [ ] **Step 4: Remove SEG badge from `renderEggs()`**

In `renderEggs()`, delete the SEG badge block (lines 6033-6040):

```javascript
// DELETE this entire block:
if (_segMode) {
    const segIcons = { ... };
    html += `<span class="mode-badge mode-badge-seg">SEG ${segIcons[_segMethod] || ''}</span>`;
}
```

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): render SEG island with editable label names"
```

---

### Task 4: Mutual exclusivity + morph transition

**Files:**
- Modify: `src/arrayview/_viewer.html:8509-8549` (A and S key handlers)

- [ ] **Step 1: Add morph transition function**

```javascript
function _islandMorph(fromMode, toMode) {
    const el = document.getElementById('dynamic-island');
    if (!el) return;
    // Crossfade: fade out current content, render new, fade in
    el.style.opacity = '0';
    setTimeout(() => {
        renderIsland();
        el.style.opacity = '1';
    }, 150);
}
```

- [ ] **Step 2: Modify A key handler to deactivate SEG first**

In the A key handler (line 8509), add SEG deactivation when entering ROI mode. After the mode availability check (line 8512), before the `if (!rectRoiMode)` block:

```javascript
} else if (e.key === 'A') {
    if (compareActive || multiViewActive || qmriActive) {
        showStatus('ROI mode: not available in this mode'); return;
    }
    // Deactivate SEG if active — modes are mutually exclusive
    if (_segMode) {
        _segDeactivate();
        _islandMorph('seg', 'roi');
    }
    if (!rectRoiMode) {
        rectRoiMode = true; _roiShape = 'rect';
    } else if (_roiShape === 'rect') {
        // ... rest of cycling unchanged ...
```

- [ ] **Step 3: Modify S key handler to deactivate ROI first**

In the S key handler (line 8535), add ROI deactivation when entering SEG mode. In the `else` branch (line 8547, mode inactive → activate):

```javascript
} else {
    // Deactivate ROI if active — modes are mutually exclusive
    if (rectRoiMode) {
        rectRoiMode = false; _roiShape = 'rect';
        _rois = []; _roiFreehandPts = [];
        _floodFillTolerance = 0.1;
        _clearRoiOverlay();
        _islandMorph('roi', 'seg');
    }
    _segActivate();
}
```

- [ ] **Step 4: Update `_segDeactivate()` to use island**

Modify `_segDeactivate()` (line 7807) — replace the panel hide with island render:

```javascript
function _segDeactivate() {
    _segMode = false;
    _segDrawing = false;
    _segDrawPoints = [];
    _segDrawStart = null;
    canvas.style.cursor = 'crosshair';
    showStatus('segmentation: off (server stays active · S to resume)');
    renderIsland();  // hides island since _segMode is now false
    _segClearDrawOverlay();
    updateView();
}
```

- [ ] **Step 5: Update ROI off-cycling to use island**

In the A key handler, the `else` branch (line 8522, floodfill → off), replace `document.getElementById('roi-panel').style.display = 'none';` with just letting `renderIsland()` handle it (it already runs via `renderEggs()`):

```javascript
} else {
    rectRoiMode = false; _roiShape = 'rect';
    _rois = []; _roiFreehandPts = [];
    _floodFillTolerance = 0.1;
    _clearRoiOverlay();
}
```

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): mutual exclusivity with morph crossfade transition"
```

---

### Task 5: Draggable island + position memory

**Files:**
- Modify: `src/arrayview/_viewer.html` (add drag handlers, position function)

- [ ] **Step 1: Add `_positionIsland()` function**

```javascript
function _positionIsland() {
    const el = document.getElementById('dynamic-island');
    if (!el) return;
    if (_islandDragPos) {
        // Use remembered position
        el.style.left = _islandDragPos.x + 'px';
        el.style.top = _islandDragPos.y + 'px';
        el.style.transform = 'none';
    } else {
        // Default: left of canvas, vertically centered
        el.style.left = '16px';
        el.style.top = '50%';
        el.style.transform = 'translateY(-50%)';
    }
}
```

- [ ] **Step 2: Add drag event handlers**

```javascript
(function() {
    const island = document.getElementById('dynamic-island');
    if (!island) return;
    let dragging = false, dragOffX = 0, dragOffY = 0;

    island.addEventListener('mousedown', e => {
        if (!e.target.closest('.island-header')) return;
        if (e.target.closest('.island-shapes') || e.target.closest('.island-download')) return;
        dragging = true;
        const rect = island.getBoundingClientRect();
        dragOffX = e.clientX - rect.left;
        dragOffY = e.clientY - rect.top;
        island.style.transition = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', e => {
        if (!dragging) return;
        const x = e.clientX - dragOffX;
        const y = e.clientY - dragOffY;
        island.style.left = x + 'px';
        island.style.top = y + 'px';
        island.style.transform = 'none';
    });

    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        island.style.transition = '';
        // Remember position
        const rect = island.getBoundingClientRect();
        _islandDragPos = { x: rect.left, y: rect.top };
    });
})();
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): draggable with position memory"
```

---

### Task 6: Wire up island actions (shape select, delete, export)

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add `_islandSetShape()` for direct shape/method selection**

```javascript
function _islandSetShape(shape) {
    if (rectRoiMode) {
        _roiShape = shape;
        const modeHint = shape === 'floodfill' ? 'click seed pixel · [ ] tolerance' : 'drag to draw';
        showStatus(`ROI mode: ${shape} · A to cycle · ${modeHint}`);
    } else if (_segMode) {
        _segMethod = shape;
        _segShowMethodStatus();
    }
    renderIsland();
}
window._islandSetShape = _islandSetShape;
```

- [ ] **Step 2: Add `_islandDeleteRoi()`**

```javascript
function _islandDeleteRoi(idx) {
    if (idx < 0 || idx >= _rois.length) return;
    _rois.splice(idx, 1);
    _drawAllRois();
    renderIsland();
    showStatus(`ROI ${idx + 1} deleted`);
}
window._islandDeleteRoi = _islandDeleteRoi;
```

- [ ] **Step 3: Add `_islandDeleteSeg()`**

```javascript
async function _islandDeleteSeg(label) {
    try {
        const r = await fetch(`/seg/delete_label/${sid}?label=${label}`, { method: 'POST' });
        const d = await r.json();
        if (d.status === 'error') { showToast(`segmentation: ${d.message}`); return; }
        _segOverlaySid = d.overlay_sid;
        _segUpdatePanel(d.labels);
        _segRefreshOverlay();
    } catch (err) {
        showToast(`segmentation: ${err.message}`);
    }
}
window._islandDeleteSeg = _islandDeleteSeg;
```

- [ ] **Step 4: Add `_islandExport()`**

```javascript
function _islandExport() {
    if (rectRoiMode) {
        _exportRoisCsv();
    } else if (_segMode) {
        const a = document.createElement('a');
        a.href = `/seg/export/${sid}`;
        a.download = '';
        a.click();
    }
}
window._islandExport = _islandExport;
```

- [ ] **Step 5: Remove old panel event listeners**

Remove the old `roi-export-btn` click listener (line 7469):
```javascript
// DELETE: document.getElementById('roi-export-btn').addEventListener('click', _exportRoisCsv);
```

Remove the old `roi-content` click listener for delete (lines 7470-7479) — replaced by `_islandDeleteRoi`.

Remove the old `seg-export-btn` click listener (line 7907-7909).

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): wire up shape selection, delete, and export actions"
```

---

### Task 7: Remove old ROI/SEG badge CSS

**Files:**
- Modify: `src/arrayview/_viewer.html:438-442,920` (badge CSS)

- [ ] **Step 1: Remove old badge CSS classes**

Delete `.mode-badge-roi` CSS (line 438-439, 443 for light theme), `.mode-badge-seg` CSS (lines 440-442), and the SVG styling rule `.mode-badge-roi svg, .mode-badge-seg svg` (line 920).

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "chore(island): remove legacy ROI/SEG badge CSS"
```

---

### Task 8: Multi-dimensional ROI export — server endpoint

**Files:**
- Modify: `src/arrayview/_server.py:1014-1049` (near existing `/roi/{sid}`)

- [ ] **Step 1: Add `/roi_multi/{sid}` endpoint**

Add after the existing `/roi/{sid}` endpoint:

```python
@app.get("/roi_multi/{sid}")
def get_roi_multi(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    x0: int = 0,
    y0: int = 0,
    x1: int = 0,
    y1: int = 0,
    complex_mode: int = 0,
):
    """ROI stats across dimension combinations for multi-dim export.

    Uses bounding box (x0,y0,x1,y1) for rect ROIs. Circle and freehand ROIs
    are converted to bounding boxes client-side before calling this endpoint.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    arr = session.array
    ndim = arr.ndim
    idx_list = [int(v) for v in indices.split(",")]

    # Build base mask dims — which dims are the ROI plane
    roi_dims = {dim_x, dim_y}
    other_dims = [d for d in range(ndim) if d not in roi_dims]

    def extract_roi(data_2d):
        """Extract ROI region from a 2D slice using bounding box."""
        h, w = data_2d.shape
        xa, xb = max(0, min(x0, x1, w - 1)), min(w, max(x0, x1) + 1)
        ya, yb = max(0, min(y0, y1, h - 1)), min(h, max(y0, y1) + 1)
        roi = data_2d[ya:yb, xa:xb]
        if roi.size == 0:
            return np.array([])
        return roi[np.isfinite(roi)]

    rows = []

    # 1. Base slice (just the ROI plane)
    bitmask = ['0'] * ndim
    bitmask[dim_x] = '1'
    bitmask[dim_y] = '1'
    base_slice = extract_slice(session, dim_x, dim_y, idx_list)
    base_data = apply_complex_mode(base_slice, complex_mode)
    finite = extract_roi(base_data)
    if finite.size:
        rows.append({
            "dims": ''.join(bitmask),
            "min": _safe_float(finite.min()),
            "max": _safe_float(finite.max()),
            "mean": _safe_float(finite.mean()),
            "std": _safe_float(finite.std()),
            "n": int(finite.size),
        })

    # 2. Single-dimension extensions
    for ext_dim in other_dims:
        bitmask_ext = list(bitmask)
        bitmask_ext[ext_dim] = '1'
        # Aggregate across ext_dim
        all_finite = []
        for val in range(arr.shape[ext_dim]):
            idx_copy = list(idx_list)
            idx_copy[ext_dim] = val
            sl = extract_slice(session, dim_x, dim_y, idx_copy)
            data = apply_complex_mode(sl, complex_mode)
            finite = extract_roi(data)
            if finite.size:
                all_finite.append(finite)
        if all_finite:
            combined = np.concatenate(all_finite)
            rows.append({
                "dims": ''.join(bitmask_ext),
                "min": _safe_float(combined.min()),
                "max": _safe_float(combined.max()),
                "mean": _safe_float(combined.mean()),
                "std": _safe_float(combined.std()),
                "n": int(combined.size),
            })

    # 3. All dimensions
    if len(other_dims) > 1:
        bitmask_all = ['1'] * ndim
        all_finite = []
        # Iterate all combinations of other dims
        import itertools
        ranges = [range(arr.shape[d]) for d in other_dims]
        for combo in itertools.product(*ranges):
            idx_copy = list(idx_list)
            for d, val in zip(other_dims, combo):
                idx_copy[d] = val
            sl = extract_slice(session, dim_x, dim_y, idx_copy)
            data = apply_complex_mode(sl, complex_mode)
            finite = extract_roi(data)
            if finite.size:
                all_finite.append(finite)
        if all_finite:
            combined = np.concatenate(all_finite)
            rows.append({
                "dims": ''.join(bitmask_all),
                "min": _safe_float(combined.min()),
                "max": _safe_float(combined.max()),
                "mean": _safe_float(combined.mean()),
                "std": _safe_float(combined.std()),
                "n": int(combined.size),
            })

    return {"rows": rows}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_server.py
git commit -m "feat(island): add /roi_multi endpoint for multi-dim ROI stats"
```

---

### Task 9: Multi-dimensional ROI export — frontend CSV download

**Files:**
- Modify: `src/arrayview/_viewer.html` (update `_exportRoisCsv()`)

- [ ] **Step 1: Update `_exportRoisCsv()` to use multi-dim endpoint**

Replace the existing `_exportRoisCsv()` function (line 7457):

```javascript
async function _exportRoisCsv() {
    if (!_rois.length) return;
    showStatus('exporting ROI stats...');
    const rows = [['roi', 'dims', 'mean', 'std', 'min', 'max', 'n_pixels'].join(',')];

    for (let i = 0; i < _rois.length; i++) {
        const r = _rois[i];
        if (!r.stats) continue;

        if (shape.length <= 2) {
            // 2D array — single row per ROI, bitmask is all 1s
            const bitmask = shape.map(() => '1').join('');
            const s = r.stats;
            rows.push([`ROI ${i + 1}`, bitmask, s.mean, s.std, s.min, s.max, s.n].join(','));
        } else {
            // Multi-dim — fetch from server
            // Convert all ROI types to bounding box for multi-dim query
            let bx0, by0, bx1, by1;
            if (r.type === 'circle') {
                bx0 = Math.round(r.cx - r.r); by0 = Math.round(r.cy - r.r);
                bx1 = Math.round(r.cx + r.r); by1 = Math.round(r.cy + r.r);
            } else if (r.type === 'freehand' && r.points) {
                const xs = r.points.map(p => p[0]), ys = r.points.map(p => p[1]);
                bx0 = Math.min(...xs); by0 = Math.min(...ys);
                bx1 = Math.max(...xs); by1 = Math.max(...ys);
            } else {
                bx0 = r.x0 ?? 0; by0 = r.y0 ?? 0;
                bx1 = r.x1 ?? 0; by1 = r.y1 ?? 0;
            }
            try {
                const params = new URLSearchParams({
                    dim_x: String(dim_x), dim_y: String(dim_y),
                    indices: indices.join(','),
                    x0: String(bx0), y0: String(by0),
                    x1: String(bx1), y1: String(by1),
                    complex_mode: String(complexMode),
                });
                const resp = await fetch(`/roi_multi/${sid}?${params}`);
                const data = await resp.json();
                if (data.rows) {
                    for (const row of data.rows) {
                        rows.push([`ROI ${i + 1}`, row.dims, row.mean, row.std, row.min, row.max, row.n].join(','));
                    }
                }
            } catch (err) {
                showToast(`export error: ${err.message}`);
            }
        }
    }

    const csvText = rows.join('\n');
    const dataUrl = 'data:text/csv;base64,' + btoa(csvText);
    _downloadFile(dataUrl, 'roi_stats.csv');
    showStatus('ROI stats exported');
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(island): multi-dimensional ROI CSV export with bitmask dims"
```

---

### Task 10: Clean up old panel references + smoke test

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Search and remove all `roi-panel` references**

Remove any remaining references to `document.getElementById('roi-panel')` that weren't caught in earlier tasks. These include:
- The A-key handler line that hides roi-panel (line 8526) — should already be gone from Task 4
- Any other `roi-panel` references

- [ ] **Step 2: Search and remove all `seg-panel` references**

Remove any remaining references to `document.getElementById('seg-panel')` — the `_segDeactivate()` panel hide line should already be gone from Task 4.

- [ ] **Step 3: Verify no dead references**

Search the file for `roi-panel`, `seg-panel`, `roi-export-btn`, `seg-export-btn`, `roi-content`, `seg-list` — none should remain.

- [ ] **Step 4: Manual smoke test**

Run the viewer and verify:
```bash
uv run arrayview tests/  # or any test array
```

Test checklist:
- Press A → ROI island appears on the left
- Click shape icons → shape changes, active icon highlights yellow
- Draw ROIs → rows appear with mean ± std
- Press × → ROI deleted
- Press A to cycle through shapes → icon strip updates
- Press A past floodfill → island disappears
- Press S → SEG island appears (if nnInteractive available)
- Click label name → editable inline
- Press A while SEG is active → SEG morphs to ROI
- Press S while ROI is active → ROI morphs to SEG
- Drag island → stays at new position
- Mode switch → island stays at dragged position
- Download icon → CSV export for ROI, NPY for SEG
- Transform eggs (FFT, LOG, etc.) still render normally without ROI/SEG badges

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "chore(island): remove legacy panel references, clean up"
```
