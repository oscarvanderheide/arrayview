# TODO Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 bugs/features from dev/TODO.md in one session.

**Architecture:** Each task is independent — touches `_viewer.html` (frontend), `_server.py` (backend), or `_render.py` (rendering). All changes are self-contained.

**Tech Stack:** Vanilla JS/HTML/CSS frontend, Python/FastAPI backend, NumPy rendering.

---

### Task 1: Add SUM projection mode to keybind `p`

**Files:**
- Modify: `src/arrayview/_render.py:148-154` (PROJECTION_OPS dict)
- Modify: `src/arrayview/_viewer.html:1427` (help text)
- Modify: `src/arrayview/_viewer.html:1658` (PROJECTION_LABELS)

- [ ] **Step 1: Add SUM to PROJECTION_OPS in _render.py**

In `_render.py`, add entry `6: ("sum", np.sum)` to `PROJECTION_OPS`:

```python
PROJECTION_OPS = {
    1: ("max", np.max),
    2: ("min", np.min),
    3: ("mean", np.mean),
    4: ("std", np.std),
    5: ("sos", None),  # sum of squares — custom implementation
    6: ("sum", np.sum),
}
```

- [ ] **Step 2: Add SUM to PROJECTION_LABELS in _viewer.html**

Change line 1658:
```js
const PROJECTION_LABELS = ['MAX', 'MIN', 'MEAN', 'STD', 'SOS', 'SUM'];
```

- [ ] **Step 3: Update help text**

Change line 1427:
```html
<div class="help-row"><span class="key">p</span><span class="desc">cycle projection: off → MAX → MIN → MEAN → STD → SOS → SUM</span></div>
```

- [ ] **Step 4: Verify the cycling logic already handles N labels**

The keybind handler at ~line 9297 cycles `projectionMode` from 0 to `PROJECTION_LABELS.length`, so adding to the labels array is sufficient. No code change needed there.

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_render.py src/arrayview/_viewer.html
git commit -m "feat: add SUM projection mode to keybind p"
```

---

### Task 2: Fix minimap position in immersive view (northeast, not east)

**Files:**
- Modify: `src/arrayview/_viewer.html:13252-13254` (minimap overlap fallback)

The current code at line 13252 positions the minimap to the **east** (right side, vertically centered) when it overlaps the dimbar. The user wants it **northeast** (top-right of canvas, but offset to avoid dimbar).

- [ ] **Step 1: Change overlap fallback to northeast position**

Replace the overlap branch (lines 13252-13254):

Old:
```js
if (overlaps) {
    _miniMap.style.top = (vpRect.top + (vpRect.height - mmH) / 2) + 'px';
    _miniMap.style.left = (vpRect.right + 10) + 'px';
}
```

New — position northeast of canvas (top edge, right of canvas):
```js
if (overlaps) {
    _miniMap.style.top = vpRect.top + mmMargin + 'px';
    _miniMap.style.left = (vpRect.right + 10) + 'px';
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: minimap falls back to northeast (not east) in immersive view"
```

---

### Task 3: Fix 3-view crosshair line width and transparency

**Files:**
- Modify: `src/arrayview/_viewer.html:12267` (lineWidth calculation)
- Modify: `src/arrayview/_viewer.html:12208` (MV_PLANE_STROKES alpha)

The crosshair lines in 3-view mode use a width of 1 canvas pixel (which looks thick on small arrays) and 0.75 alpha (looks blurry). Fix: use a fixed CSS-pixel width and reduce transparency.

- [ ] **Step 1: Change lineWidth to fixed 1.5 CSS pixels**

Replace line 12267:
```js
v.ctx.lineWidth = Math.max(0.5, 1 / scale);
```
with:
```js
v.ctx.lineWidth = Math.max(0.5, 1.5 / scale);
```

This ensures the line is always ~1.5 CSS pixels regardless of zoom/array size.

- [ ] **Step 2: Reduce alpha in MV_PLANE_STROKES**

Replace line 12208:
```js
const MV_PLANE_STROKES = ['rgba(255,90,90,0.75)', 'rgba(100,210,130,0.75)', 'rgba(80,170,255,0.75)'];
```
with:
```js
const MV_PLANE_STROKES = ['rgba(255,90,90,0.9)', 'rgba(100,210,130,0.9)', 'rgba(80,170,255,0.9)'];
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: 3-view crosshair lines use fixed width and less transparency"
```

---

### Task 4: Make dimbar and colorbar islands draggable in immersive view

**Files:**
- Modify: `src/arrayview/_viewer.html` — add drag handlers to `#info` and `#slim-cb-wrap` in immersive mode, reset on exit

The `#dynamic-island` already has drag support via `_islandDragPos`. Replicate this pattern for `#info` (dimbar) and `#slim-cb-wrap` (colorbar) when in immersive mode.

- [ ] **Step 1: Add drag state variables**

After `_islandDragPos` declaration (~line 1593), add:
```js
let _infoDragPos = null;   // {x, y} — immersive dimbar drag position
let _cbDragPos = null;     // {x, y} — immersive colorbar drag position
```

- [ ] **Step 2: Add CSS cursor for draggable islands in immersive mode**

Add to the `body.fullscreen-mode #info` CSS block (~line 767):
```css
body.fullscreen-mode #info { cursor: grab; }
body.fullscreen-mode #info:active { cursor: grabbing; }
body.fullscreen-mode #slim-cb-wrap { cursor: grab; }
body.fullscreen-mode #slim-cb-wrap:active { cursor: grabbing; }
```

- [ ] **Step 3: Add drag handler function**

Add a reusable drag setup function near the existing island drag code. The function should:
- On mousedown: record offset from element origin
- On mousemove: update element position via `position: fixed; left/top`
- On mouseup: save position to the state variable
- Only activate when `document.body.classList.contains('fullscreen-mode')`

```js
function _setupImmersiveDrag(el, posRef, setPosRef) {
    let dragging = false, offX = 0, offY = 0;
    el.addEventListener('mousedown', e => {
        if (!document.body.classList.contains('fullscreen-mode')) return;
        if (e.button !== 0) return;
        dragging = true;
        const rect = el.getBoundingClientRect();
        offX = e.clientX - rect.left;
        offY = e.clientY - rect.top;
        el.style.cursor = 'grabbing';
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!dragging) return;
        const x = e.clientX - offX, y = e.clientY - offY;
        el.style.left = x + 'px';
        el.style.top = y + 'px';
        el.style.right = 'auto';
        el.style.bottom = 'auto';
        el.style.transform = 'none';
    });
    window.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        el.style.cursor = '';
        const rect = el.getBoundingClientRect();
        setPosRef({ x: rect.left, y: rect.top });
    });
}
```

- [ ] **Step 4: Wire up drag handlers after DOM ready**

In the initialization section (near where minimap/crosshair handlers are set up), add:
```js
const _infoEl = document.getElementById('info');
const _cbWrapEl = document.getElementById('slim-cb-wrap');
if (_infoEl) _setupImmersiveDrag(_infoEl, () => _infoDragPos, p => { _infoDragPos = p; });
if (_cbWrapEl) _setupImmersiveDrag(_cbWrapEl, () => _cbDragPos, p => { _cbDragPos = p; });
```

- [ ] **Step 5: Apply saved drag positions in immersive layout**

In the immersive positioning code (`_positionImmersiveIslands` or equivalent where `#info` and `#slim-cb-wrap` get positioned), check the drag pos first:
```js
// For #info:
if (_infoDragPos) {
    infoEl.style.left = _infoDragPos.x + 'px';
    infoEl.style.top = _infoDragPos.y + 'px';
    infoEl.style.transform = 'none';
} else {
    // ... existing default positioning
}

// For #slim-cb-wrap:
if (_cbDragPos) {
    cbWrap.style.left = _cbDragPos.x + 'px';
    cbWrap.style.top = _cbDragPos.y + 'px';
    cbWrap.style.right = 'auto';
    cbWrap.style.bottom = 'auto';
    cbWrap.style.transform = 'none';
} else {
    // ... existing default positioning
}
```

- [ ] **Step 6: Reset drag positions on exit immersive**

In `_exitImmersive()`, reset:
```js
_infoDragPos = null;
_cbDragPos = null;
```

- [ ] **Step 7: Add snap-to-default behavior**

When mouseup fires within 30px of the default position, reset the drag pos to null (snaps back):
```js
// In the mouseup handler, after saving position:
// Calculate distance to default pos and snap if close
```

The default positions are computed during immersive layout. Store them and compare on mouseup.

- [ ] **Step 8: Update help overlay**

Add a help row for the new drag behavior:
```html
<div class="help-row"><span class="key">drag</span><span class="desc">move dimbar / colorbar in immersive mode (snaps back on exit)</span></div>
```

- [ ] **Step 9: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: draggable dimbar and colorbar islands in immersive view"
```

---

### Task 5: Fix pane centering in native window mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — CSS for `#wrapper` or `#viewer-row`

The `#wrapper` uses `display: flex; align-items: center; justify-content: center;` which centers content. In inline (Jupyter iframe) mode this works, but in native window (pywebview iframe via `_shell.html`) the pane is not centered. The `#canvas-wrap` has `align-self: flex-start` which may interact differently across contexts.

- [ ] **Step 1: Investigate the root cause**

Check if the issue is that `#viewer-row` or `#wrapper` doesn't fill the available height in native window mode. The shell embeds the viewer in an iframe — check if `html, body` heights are consistent.

- [ ] **Step 2: Fix centering**

The most likely fix is ensuring `#viewer-row` has `flex: 1` so it expands to fill available vertical space, then content centers within it:
```css
#viewer-row {
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    flex: 1;
    width: 100%; padding: 8px; box-sizing: border-box;
}
```

Test in both inline and native modes to verify no regression.

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: center pane in native window mode"
```

---

### Task 6: Fix ROI island disappearing after freehand draw

**Files:**
- Modify: `src/arrayview/_viewer.html:7812-7816` (freehand mouseup handler)

After freehand drawing, the code sets `rectRoiMode = false` which causes `renderIsland()` to hide the island. The ROI should stay visible — only exit the *drawing* sub-mode, not the entire ROI mode.

- [ ] **Step 1: Remove auto-exit after freehand draw**

Replace lines 7812-7816:
```js
// Auto-exit ROI mode after freehand draw
setTimeout(() => {
    rectRoiMode = false; _roiShape = 'rect';
    renderEggs();
}, 1000);
```
with just:
```js
renderEggs();
```

This keeps ROI mode active (island stays visible, user can draw more ROIs or press A to exit).

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: ROI island stays visible after freehand draw"
```

---

### Task 7: Fix ROI CSV export error (session.array → session.data)

**Files:**
- Modify: `src/arrayview/_server.py:1075` (get_roi_multi endpoint)

`session.array` does not exist — Session stores the array as `session.data`.

- [ ] **Step 1: Fix the attribute name**

Change line 1075:
```python
arr = session.array
```
to:
```python
arr = session.data
```

- [ ] **Step 2: Search for other occurrences of session.array**

Grep for `session.array` in `_server.py` to ensure no other instances.

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_server.py
git commit -m "fix: ROI CSV export uses session.data (not session.array)"
```

---

### Task 8: Adjust initial ROI island position

**Files:**
- Modify: `src/arrayview/_viewer.html:6117-6121` (_positionIsland default branch)

Current default: `left: 16px; top: 50%; transform: translateY(-50%)` — vertically centered on left edge.

User wants: top of island slightly below top of canvas, horizontally centered between left of canvas and left of viewport.

- [ ] **Step 1: Change default position**

Replace the default branch in `_positionIsland()`:
```js
} else {
    const cvRect = canvas.getBoundingClientRect();
    // Horizontally: center between left of viewport (0) and left of canvas
    const centerX = cvRect.left / 2;
    el.style.left = Math.max(8, centerX - el.offsetWidth / 2) + 'px';
    // Vertically: a bit below top of canvas
    el.style.top = (cvRect.top + 40) + 'px';
    el.style.transform = 'none';
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: ROI island initial position near top-left of canvas"
```

---

## Execution Order

Tasks 1, 2, 3, 6, 7 are small and independent — can be parallelized.
Task 4 (draggable islands) is the largest.
Tasks 5 and 8 need visual verification.

Recommended parallel batches:
- **Batch 1:** Tasks 1, 2, 3, 6, 7 (quick fixes, parallel subagents)
- **Batch 2:** Task 4 (draggable islands — single focused subagent)
- **Batch 3:** Tasks 5, 8 (layout fixes — may need iteration)
