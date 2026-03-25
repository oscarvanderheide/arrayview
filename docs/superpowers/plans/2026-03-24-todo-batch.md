# TODO Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 7 TODO items: histogram in multiview, colormap preview in multiview, rotate in multiview, vector-field perf, colorbar preview redesign, zoom/layout fixes, and startup speed improvements.

**Architecture:** All UI fixes are in `src/arrayview/_viewer.html` (single-file frontend). Vector-field perf involves both server (`_server.py`) and client. Speed work touches `__init__.py`, `_launcher.py`, `_platform.py`. Each item is independent and gets its own commit.

**Tech Stack:** Vanilla JS, CSS, HTML5 Canvas, FastAPI, NumPy

**Skills:** @viewer-ui-checklist @modes-consistency @frontend-designer @vscode-simplebrowser @invocation-consistency

---

## Task 1: Show histogram on `d` in multiview mode

The `d` key handler already fetches/shows histogram in normal mode but guards with `!compareActive && !rgbMode`. In multiview, the slim colorbar (`#slim-cb-wrap`) is hidden — multiview has its own `#mv-cb-wrap`. Need to expand the multiview colorbar to show histogram there too.

**Files:**
- Modify: `src/arrayview/_viewer.html` (d handler ~line 4981, multiview colorbar drawing ~line 7554)

- [ ] **Step 1: Check if multiview has histogram support in its colorbar**

Read `drawMvColorbar()` (~line 7554) to see if it handles `_cbExpanded` / histogram drawing. If not, we need to add it.

- [ ] **Step 2: Add histogram rendering to multiview colorbar**

In `drawMvColorbar()` (~line 7554), add histogram rendering when `_cbExpanded` is true, similar to `drawSlimColorbar()`. Use `_histData` (shared state). Resize `#mv-cb` height from 8px to 40px when expanded.

- [ ] **Step 3: Update `d` handler to work in multiview**

Remove the `!compareActive` guard for the histogram expansion (keep it for compare mode). In multiview, after updating views, fetch histogram using `mvViews[0]` dims and expand the MV colorbar:

```js
// In d handler, after the fetch/clearcache .then():
if (!compareActive && !rgbMode) {
    _histData = null; _histDataVersion = null;
    _fetchHistogram().then(() => {
        _cbExpanded = true;
        if (multiViewActive) drawMvColorbar();
        else drawSlimColorbar(cbMarkerFrac);
    });
}
```

- [ ] **Step 4: Auto-dismiss timer should also redraw MV colorbar**

Update the auto-dismiss callback (~line 4999) to call `drawMvColorbar()` when in multiview:

```js
if (!lebesgueMode && !_cbDragActive) {
    _cbExpanded = false; _histData = null;
    if (multiViewActive) drawMvColorbar();
    else drawSlimColorbar(cbMarkerFrac);
}
```

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/test_browser.py -v -k multiview`

```bash
git commit -m "feat: show histogram on d key in multiview mode"
```

---

## Task 2: Fix colormap previewer in multiview

`showColormapStrip()` anchors to `#slim-cb-wrap` which is hidden in multiview. Need to anchor to `#mv-cb-wrap` when in multiview.

**Files:**
- Modify: `src/arrayview/_viewer.html` (`showColormapStrip` ~line 8153)

- [ ] **Step 1: Update `showColormapStrip` to use correct anchor**

In `showColormapStrip()` (~line 8190), change the anchor element selection:

```js
const cbWrap = document.getElementById(multiViewActive ? 'mv-cb-wrap' : 'slim-cb-wrap');
```

- [ ] **Step 2: Test in multiview — press `c` and verify preview strip appears**

- [ ] **Step 3: Commit**

```bash
git commit -m "fix: anchor colormap preview strip to multiview colorbar when active"
```

---

## Task 3: Make `r` rotate x/y in multiview mode

Currently `r` in multiview only flips the active axis. User wants `r` to rotate (swap dimX/dimY) for the active pane, similar to how `t` transposes all panes. The existing single-view `r` already rotates 90 CW when activeDim is a slice dim.

**Files:**
- Modify: `src/arrayview/_viewer.html` (r handler ~line 4747)

- [ ] **Step 1: Change multiview `r` behavior**

Replace the current multiview branch of the `r` handler. When `r` is pressed in multiview:
- Find the pane whose `sliceDir === activeDim` (the pane where activeDim slices through)
- Swap that pane's `dimX` and `dimY` (90 rotation)
- Update `flipDims` accordingly for proper orientation
- Redraw that pane

```js
if (multiViewActive) {
    // Find pane that slices along activeDim — rotate that pane's x/y
    const pane = mvViews.find(v => v.sliceDir === activeDim);
    if (!pane) return;
    const [odx, ody] = [pane.dimX, pane.dimY];
    const ofx = flipDims[odx] || false;
    const ofy = flipDims[ody] || false;
    pane.dimX = ody; pane.dimY = odx;
    flipDims[pane.dimX] = ofy; flipDims[pane.dimY] = !ofx;
    pane.seq++;
    mvDrawFrame(pane); mvRender(pane);
    showStatus('rotated 90°');
    renderInfo(); saveState();
}
```

- [ ] **Step 2: Test — load 3D array, enter multiview, press h/l to select different dims, press r**

Verify the corresponding pane rotates 90 each press.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: r rotates x/y in multiview mode per-pane"
```

---

## Task 4: Fix vector-field performance with high density

**Problem:** `drawVectorOverlay()` always fetches from `/vectorfield/{sid}` even when only `vfLengthLevel` (client-side scale) changed. Also, with high density (`]` pressed many times), the server generates and serializes thousands of arrows as JSON, which is slow.

**Files:**
- Modify: `src/arrayview/_viewer.html` (drawVectorOverlay ~line 2657)
- Modify: `src/arrayview/_server.py` (get_vectorfield ~line 774)

- [ ] **Step 1: Cache vector field data client-side**

Add a cache key and cached data variable. Only re-fetch when the cache key changes (indices, dim_x, dim_y, t_index, density_offset changed). When only vfLengthLevel changes, re-render from cached data.

```js
let _vfieldCache = null; // {key, data}

async function drawVectorOverlay() {
    if (!hasVectorfield || !vfieldVisible || compareActive || multiViewActive || qmriActive) {
        vfieldCanvas.style.display = 'none';
        return;
    }
    // ... existing layout code ...

    const cacheKey = `${dim_x},${dim_y},${indices.join(',')},${vfieldT},${vfDensityLevel}`;
    let data;
    if (_vfieldCache && _vfieldCache.key === cacheKey) {
        data = _vfieldCache.data;
    } else {
        if (_vfieldAbort) _vfieldAbort.abort();
        _vfieldAbort = new AbortController();
        const signal = _vfieldAbort.signal;
        const params = new URLSearchParams({ dim_x, dim_y, indices: indices.join(','), t_index: vfieldT, density_offset: vfDensityLevel });
        const res = await fetch(`/vectorfield/${sid}?${params}`, { signal });
        if (!res.ok || signal.aborted) return;
        data = await res.json();
        if (signal.aborted) return;
        _vfieldCache = { key: cacheKey, data };
    }
    // Render from data (same as current code)
    // ...
}
```

- [ ] **Step 2: Invalidate cache on relevant state changes**

Clear `_vfieldCache = null` when:
- `sid` changes (new array loaded)
- Slice indices change (already handled by cache key)
- `/clearcache` is called

- [ ] **Step 3: Optimize server-side JSON serialization**

In `get_vectorfield()` (~line 842), the list comprehension building `arrows` as nested Python lists is slow for thousands of arrows. Use numpy directly:

```python
# Replace the list comprehension with:
coords = np.column_stack([gx, gy, vx_s, vy_s])  # shape (n_arrows, 4)
arrows = coords.tolist()  # faster than per-element list comp
```

- [ ] **Step 4: Add server-side arrow count cap**

After computing `n_arrows`, cap it to prevent extreme cases:

```python
MAX_ARROWS = 4096
n_arrows = min(n_arrows, MAX_ARROWS)
```

- [ ] **Step 5: Test with vector field data, press `]` multiple times, verify smooth scrolling**

- [ ] **Step 6: Commit**

```bash
git commit -m "perf: cache vector field data client-side, cap arrow count"
```

---

## Task 5: Redesign colorbar/histogram preview with fade transitions

**User wants:** Revert to old preview behavior but with:
1. Colorbar/histogram fades out completely when preview appears
2. Preview thumbnails are larger
3. Previews fade away after 1-2 seconds
4. Colorbar/histogram fades back in

**Files:**
- Modify: `src/arrayview/_viewer.html` (showColormapStrip ~line 8153, CSS for `.cmap-thumb`)

- [ ] **Step 1: Add fade-out to colorbar when preview shows**

In `showColormapStrip()`, fade out the colorbar wrapper:

```js
// Fade out colorbar
const cbEl = document.getElementById(multiViewActive ? 'mv-cb-wrap' : 'slim-cb-wrap');
if (cbEl) cbEl.style.opacity = '0';
```

- [ ] **Step 2: Make preview thumbnails larger**

Update the thumbnail canvas size from `36x6` to `56x10`:

```js
cv.width = 56; cv.height = 10;
```

And update CSS for `.cmap-thumb` to accommodate larger size.

- [ ] **Step 3: Add fade-back transition**

In the strip hide timer callback, fade colorbar back in:

```js
_cmapStripTimer = setTimeout(() => {
    _cmapStrip.classList.remove('visible');
    // Fade colorbar back in
    const cbEl = document.getElementById(multiViewActive ? 'mv-cb-wrap' : 'slim-cb-wrap');
    if (cbEl) cbEl.style.opacity = '1';
}, 1500);  // 1.5s display time
```

- [ ] **Step 4: Add CSS transitions for smooth fade**

```css
#slim-cb-wrap, #mv-cb-wrap { transition: opacity 0.3s ease; }
```

- [ ] **Step 5: Test — press `c` multiple times, verify smooth fade out/in of colorbar with preview**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: redesign colormap preview with fade transitions and larger thumbnails"
```

---

## Task 6: Fix zoom / miniviewer / layout

**User says:** Can't get into miniviewer state, don't like compact mode. Want better colorbar/dim bar positioning for more canvas space.

This requires investigation of what's broken with the current zoom → miniviewer flow, then targeted fixes.

**Files:**
- Modify: `src/arrayview/_viewer.html` (zoom handling ~line 4568, compact mode ~line 8064, minimap ~line 8367)

- [ ] **Step 1: Investigate zoom → miniviewer flow**

Read the zoom handler code carefully. The minimap shows when `_isCanvasOverflowing()` returns true. Check if the zoom cap (from commit 654289a "cap zoom at window") prevents the canvas from ever overflowing, which would prevent the minimap from appearing.

- [ ] **Step 2: Fix miniviewer activation**

If the zoom cap is too aggressive, adjust it. The minimap should appear when:
1. User zooms past the point where the array fills the viewport
2. Canvas should be allowed to grow beyond viewport (with scrolling/panning)
3. Minimap shows a thumbnail with viewport indicator

The zoom cap should only prevent the INITIAL zoom from overflowing, but user-initiated zoom (+/= keys, Ctrl+scroll) should be allowed to exceed viewport.

- [ ] **Step 3: Improve colorbar/dim bar positioning**

Design a layout where:
- Colorbar overlays the bottom of the canvas (semi-transparent background) rather than taking separate vertical space
- Dim bar (info line) overlays the top of the canvas
- This maximizes canvas area

```css
#slim-cb-wrap {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    z-index: 2;
}
```

- [ ] **Step 4: Remove or simplify compact mode**

Since the colorbar/dim bar now overlay the canvas, compact mode becomes less necessary. Either simplify it to just hide the array name, or remove it entirely.

- [ ] **Step 5: Test zoom in/out, verify miniviewer appears, verify layout looks clean**

Run: `uv run pytest tests/test_browser.py -v`

- [ ] **Step 6: Commit**

```bash
git commit -m "fix: restore miniviewer on zoom, overlay colorbar/dim bar for more canvas space"
```

---

## Task 7: Startup speed improvements

**Depends on:** Speed investigation agent results. The profiled total was 1.2-3.3s. User says it still feels like "several seconds."

**Files:**
- Modify: `src/arrayview/__init__.py`, `src/arrayview/_launcher.py`, `src/arrayview/_platform.py`

- [ ] **Step 1: Verify lazy FastAPI import is still in effect**

Check if any new import chain re-introduced the eager FastAPI import. The speed agent found `import arrayview` still takes ~346ms — the S1 optimization may be negated.

- [ ] **Step 2: Fix any import regression**

If FastAPI is being eagerly imported again, trace the import chain and fix it.

- [ ] **Step 3: Lazy matplotlib import**

`_init_luts()` imports matplotlib on first render (~145ms). Pre-build and cache the LUT tables so matplotlib is only needed on first run, not every startup.

- [ ] **Step 4: Reduce browser/webview cold start**

The biggest remaining cost is `_open_webview` / `_open_browser` (200-2000ms). Options:
- Start the browser open in parallel with server startup (don't wait for server ready)
- Use a simpler loading page that shows immediately, then connects to WS when server is ready
- Pre-warm webview if in VS Code (SimpleBrowser is faster than system browser)

- [ ] **Step 5: Profile and verify improvement**

```bash
uv run python -c "import time; t=time.time(); import arrayview; print(f'import: {time.time()-t:.3f}s')"
```

- [ ] **Step 6: Commit**

```bash
git commit -m "perf: fix import regression, reduce startup latency"
```

---

## Task 8: Research competitor features (3D Slicer, arrShow)

This is handled by a background research agent. The output will be written to `docs/competitor-features.md`.

- [ ] **Step 1: Review agent output**
- [ ] **Step 2: Curate and edit the feature list**
- [ ] **Step 3: Commit**

```bash
git commit -m "docs: add competitor feature research (3D Slicer, arrShow)"
```

---

## Order

Tasks are independent. Recommended execution order by complexity (simple first):
1. Task 2 (colormap preview anchor fix — 1 line change)
2. Task 3 (r to rotate in multiview — small)
3. Task 1 (histogram in multiview — medium)
4. Task 4 (vector-field perf — medium)
5. Task 5 (colorbar preview redesign — medium)
6. Task 6 (zoom/layout — larger, needs investigation)
7. Task 7 (speed — depends on agent findings)
8. Task 8 (research — waiting on agent)

Tasks 1-5 can be parallelized (no dependencies between them).
