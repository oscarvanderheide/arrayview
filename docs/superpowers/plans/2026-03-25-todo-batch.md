# TODO Batch — Bug Fixes & UI Polish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix screenshot/gallery, CSV export, ROI deletion UX, colorbar alignment in zoom+borders, colorbar in all modes, and projection mode dim-bar + cycle improvements.

**Architecture:** All changes in `src/arrayview/_viewer.html` (single-file viewer with embedded CSS/JS). No backend changes needed. Each task is independent and gets its own commit.

**Tech Stack:** Vanilla JS, Canvas API, CSS

---

## File Structure

| File | Changes |
|------|---------|
| `src/arrayview/_viewer.html` | All tasks: CSS fixes, JS bug fixes, new UI elements |
| `tests/test_browser.py` | New/updated Playwright tests for screenshot, ROI panel, projection |

---

### Task 1: Fix Screenshot (s) and Gallery (G)

**Problem:** Pressing `s` shows a small image in the top-left corner with annotation text overlaid, but nothing downloads and the viewer becomes unresponsive. Gallery (G) is broken because it depends on screenshot data.

**Root cause:** `_annotateCanvas()` (line 4333) returns a `<canvas>` element, but the caller at line 4377 calls `.toDataURL()` on it. The issue is likely that the canvas returned has dimensions matching the *backing store* (high-DPI `canvas.width/height`) but the `drawImage(src, 0, 0)` call draws the source at its natural size. When the main canvas has CSS scaling (zoom), the backing store pixels and display pixels diverge, causing a tiny image in the corner.

**Files:**
- Modify: `src/arrayview/_viewer.html:4315-4388`

- [ ] **Step 1: Investigate the actual bug by launching the app**

Open the viewer in Playwright, load a test array, press `s`, and observe:
- Does the browser trigger a download?
- Does the canvas element get corrupted (visible small image)?
- Check console for errors.

Run: `uv run pytest tests/test_browser.py -k screenshot -v` to see if existing tests catch this.

- [ ] **Step 2: Fix `saveScreenshot()` to handle zoomed/DPI-scaled canvas**

The fix should ensure:
1. The annotated canvas is created at the correct resolution
2. `link.click()` triggers a download (not navigation)
3. The main canvas is NOT corrupted after screenshot
4. Append `link` to `document.body` before `.click()` and remove after (some browsers require this)

In `saveScreenshot()` around line 4380, change:
```javascript
const link = document.createElement('a');
link.download = filename;
link.href = dataUrl;
link.click();
```
to:
```javascript
const link = document.createElement('a');
link.download = filename;
link.href = dataUrl;
document.body.appendChild(link);
link.click();
document.body.removeChild(link);
```

Also check `_annotateCanvas()` — ensure `ctx.drawImage(src, 0, 0, src.width, src.height)` uses explicit dimensions to avoid DPI scaling issues.

- [ ] **Step 3: Verify fix manually and with tests**

Run: `uv run pytest tests/test_browser.py -k screenshot -v`

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: screenshot download and canvas corruption after pressing s"
```

---

### Task 2: Fix CSV Export in ROI Mode

**Problem:** Pressing the "export CSV" button in ROI mode displays CSV text in the window (black text) instead of triggering a download. No file is saved to disk and the viewer becomes unresponsive.

**Root cause:** Same pattern as screenshot — `a.click()` without appending to DOM may cause navigation instead of download in some browsers/webviews. The `URL.revokeObjectURL` is called immediately after click, possibly before the download starts.

**Files:**
- Modify: `src/arrayview/_viewer.html:4843-4857`

- [ ] **Step 1: Fix `_exportRoisCsv()` download trigger**

Change the function at line 4843:
```javascript
function _exportRoisCsv() {
    if (!_rois.length) return;
    const rows = [['roi','n','min','max','mean','std'].join(',')];
    for (let i = 0; i < _rois.length; i++) {
        const s = _rois[i].stats;
        if (!s) continue;
        rows.push([i + 1, s.n, s.min, s.max, s.mean, s.std].join(','));
    }
    const blob = new Blob([rows.join('\n')], { type: 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'roi_stats.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
```

Key changes:
- MIME type `application/octet-stream` forces download instead of display
- Append `<a>` to DOM before clicking
- Delay `revokeObjectURL` to allow download to start

- [ ] **Step 2: Test manually — draw an ROI, click export CSV, verify .csv downloads**

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: CSV export downloads file instead of displaying text in browser"
```

---

### Task 3: ROI Panel Trash Icons for Deletion

**Problem:** Right-click to delete ROIs feels broken (sometimes screen turns blue — likely browser default context menu leaking through). User wants a minimal trash icon per ROI row in the panel.

**Files:**
- Modify: `src/arrayview/_viewer.html:4816-4842` (ROI panel rendering)
- Modify: `src/arrayview/_viewer.html:5047-5062` (contextmenu handler)

- [ ] **Step 1: Add trash icon column to ROI panel table**

In `_updateRoiPanel()` at line 4826, add a delete column:

Change header:
```javascript
let html = '<table><tr><th></th><th>n</th><th>min</th><th>max</th><th>mean</th><th>std</th><th></th></tr>';
```

Add trash button per row after the std cell (line 4837):
```javascript
html += `<td><span class="roi-delete" data-roi="${i}" title="Delete ROI ${i+1}" style="cursor:pointer;opacity:0.4;font-size:13px" onmouseenter="this.style.opacity=1" onmouseleave="this.style.opacity=0.4">&#x1F5D1;</span></td></tr>`;
```

- [ ] **Step 2: Wire up click handler for trash icons**

After `_updateRoiPanel()` function, add a delegated click handler on `#roi-content`:

```javascript
document.getElementById('roi-content').addEventListener('click', e => {
    const del = e.target.closest('.roi-delete');
    if (!del) return;
    const idx = parseInt(del.dataset.roi);
    if (isNaN(idx) || idx < 0 || idx >= _rois.length) return;
    _rois.splice(idx, 1);
    _drawAllRois();
    _updateRoiPanel();
    showStatus(`ROI ${idx + 1} deleted`);
});
```

- [ ] **Step 3: Fix the contextmenu blue-screen issue**

At line 4583, the generic `canvas.addEventListener('contextmenu', e => e.preventDefault())` prevents the default in ALL cases. But at line 5047, another contextmenu listener tries to handle ROI deletion. The issue: both fire, and the first one's `preventDefault()` may not fully suppress native UI in some webviews.

Fix: Remove the generic handler at line 4583 and merge into the ROI-specific one at 5047:
```javascript
canvas.addEventListener('contextmenu', e => {
    e.preventDefault();
    if (!rectRoiMode) return;
    // ... existing ROI deletion logic ...
});
```

Wait — actually line 4583 already prevents default for ALL cases, which is correct. The blue screen may be from the right-click-drag scrub feature (line 4587) interfering. Ensure `e.preventDefault()` + `e.stopPropagation()` in the contextmenu handler.

- [ ] **Step 4: Test — draw ROIs, verify trash icons appear, click to delete**

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: trash icon per ROI row in panel, fix contextmenu blue screen"
```

---

### Task 4: Colorbar Dynamic Island Alignment with Borders + Zoom

**Problem:** When borders are enabled (b key) and the canvas is zoomed, the colorbar dynamic island becomes misaligned with the canvas.

**Root cause:** The colorbar positioning in `drawSlimColorbar()` (line 2427-2445) uses `canvasRect` (getBoundingClientRect) which includes CSS transforms but not the border offset. When `.canvas-bordered` adds a 1px outline, the effective layout shifts but `canvasRect` doesn't account for it.

**Files:**
- Modify: `src/arrayview/_viewer.html:2427-2445`

- [ ] **Step 1: Use the ui-consistency-audit skill to capture the misalignment**

Launch viewer with borders on (b), zoom in, take screenshot. Identify the exact pixel offset.

- [ ] **Step 2: Fix colorbar positioning to account for border offset**

In `drawSlimColorbar()`, when computing `cbTop` and centering, check if `.canvas-bordered` is active and compensate:

```javascript
const bordered = document.getElementById('canvas-viewport')?.classList.contains('canvas-bordered');
const borderOffset = bordered ? 1 : 0;
const cbTop = Math.min(canvasRect.bottom + 6 + borderOffset, window.innerHeight - 38 - (...));
```

The exact fix depends on what the audit reveals — it could be horizontal misalignment too.

- [ ] **Step 3: Re-audit with borders on + zoom to verify alignment**

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: colorbar alignment when borders enabled during zoom"
```

---

### Task 5: Colorbar Dynamic Island in All Modes

**Problem:** The dynamic island colorbar only appears in normal mode. User wants it in multi-view, qMRI, diff, compare — everywhere.

**Current state:**
- Normal mode: `#slim-cb-wrap` (dynamic island with glassmorphism)
- Multi-view: `#mv-cb-wrap` (plain div inside flexbox, no dynamic island styling)
- Compare: per-pane colorbars (plain, no island styling)
- qMRI: vertical colorbars (no island styling) or slim horizontal (no island styling)

**Approach:** Apply the dynamic island styling (rounded corners, glassmorphism blur, semi-transparent bg) to the colorbar containers in ALL modes. Don't change positioning logic — just apply the visual treatment.

**Files:**
- Modify: `src/arrayview/_viewer.html` — CSS for `#mv-cb-wrap`, compare pane cb, qMRI cb

- [ ] **Step 1: Add dynamic island styling to multi-view colorbar**

Update `#mv-cb-wrap` CSS (line ~450):
```css
#mv-cb-wrap {
    text-align: center; flex-shrink: 0; padding: 5px 14px; margin-top: 6px;
    transition: opacity 0.3s ease;
    border-radius: 14px;
    background: rgba(30, 30, 30, 0.8);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
}
```

- [ ] **Step 2: Add dynamic island styling to compare pane colorbars**

Add CSS for `.cm-pane-cb-wrap` (or whatever the compare pane cb container class is):
```css
.cm-pane-cb-wrap {
    border-radius: 14px;
    background: rgba(30, 30, 30, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 5px 14px;
}
```

- [ ] **Step 3: Add dynamic island styling to qMRI colorbars**

Apply same treatment to qMRI horizontal slim cb and vertical per-pane cbs.

- [ ] **Step 4: Ensure theme variants (light, solarized, nord) apply to all mode colorbars**

Check that the theme-specific background colors from the slim-cb-wrap rules also apply to the new island containers.

- [ ] **Step 5: Visual audit across all modes**

Use ui-consistency-audit to verify consistency.

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: dynamic island colorbar styling in all modes (multiview, compare, qmri)"
```

---

### Task 6: ROI Mode (A) Misalignment Audit & Fix

**Problem:** When A is pressed for ROI mode, there's some misalignment visible.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Use ui-consistency-audit skill with borders on to capture the misalignment**

Launch viewer, press A to enter ROI mode, take screenshots. Identify what's misaligned (ROI overlay canvas vs main canvas? ROI panel? Shape strip?).

- [ ] **Step 2: Fix the identified misalignment**

The ROI overlay canvas (`#roi-overlay`) must be pixel-aligned with the main canvas. Check that both use the same `getBoundingClientRect()` base and that CSS transforms are in sync.

- [ ] **Step 3: Re-audit to verify**

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: ROI mode overlay alignment"
```

---

### Task 7: Projection Mode — Dim Bar "p" Label + Remove "off" from Cycle

**Problem:**
1. In projection mode, the dim being projected over should show "p" in the dim bar (like qMRI shows "q")
2. The egg badge indicates which projection type is active
3. The "off" item should be removed from the `p` cycle — pressing `p` past the last projection type just deactivates (no highlight), same as ROI mode (A)

**Files:**
- Modify: `src/arrayview/_viewer.html:3643-3705` (renderInfo)
- Modify: `src/arrayview/_viewer.html:5861-5875` (p key handler)
- Modify: `src/arrayview/_viewer.html:1276-1277` (PROJECTION_LABELS/COLORS)
- Modify: `src/arrayview/_viewer.html:2079-2100` (showProjectionStrip)

- [ ] **Step 1: Add "p" label in dim bar for projection dimension**

In `renderInfo()` (line 3643), add a check before the dim_x/dim_y checks:

```javascript
// After the qmri check (line 3651), before dim_x/dim_y:
if (projectionMode > 0 && i === current_slice_dim) {
    const pColor = PROJECTION_COLORS[projectionMode];
    const cls = (active || playing) ? `${dimCls} dim-label` : 'dim-label';
    return `<span class="${cls}" ${d} style="color:${pColor}">p<span class="dim-size">/${shape[i]}</span></span>`;
}
```

- [ ] **Step 2: Remove "off" from projection cycle**

Change `PROJECTION_LABELS` and `PROJECTION_COLORS` (line 1276):
```javascript
const PROJECTION_LABELS = ['MAX', 'MIN', 'MEAN', 'STD', 'SOS'];
const PROJECTION_COLORS = ['#4cc9f0', '#ff6b6b', '#80ed99', '#c77dff', '#ffa62b'];
```

Update `p` key handler (line 5861) to cycle 1-based with 0 = off (no strip highlight):
```javascript
} else if (e.key === 'p') {
    if (shape.length < 3) { showStatus('projection: need ≥ 3D array'); return; }
    if (hasVectorfield) { showStatus('projection: not available in vector field mode'); return; }
    if (projectionMode === 0) {
        projectionMode = 1;  // first projection type
    } else if (projectionMode < PROJECTION_LABELS.length) {
        projectionMode++;
    } else {
        projectionMode = 0;  // off — no highlight, just deactivate
    }
    _eggsVisible = projectionMode > 0;
    updateView();
    renderEggs();
    showProjectionStrip();
    if (projectionMode === 0) {
        showStatus('projection: off');
    } else {
        showStatus(`projection: ${PROJECTION_LABELS[projectionMode - 1]} along dim ${current_slice_dim}`);
    }
    saveState();
}
```

- [ ] **Step 3: Update `showProjectionStrip()` to use 1-based indexing**

In `showProjectionStrip()` (line 2079), adjust to only show the actual projection types (no "off" thumb), and highlight based on `projectionMode - 1`:

```javascript
function showProjectionStrip() {
    const el = document.getElementById('compare-mode-strip');
    if (!el) return;
    el.innerHTML = '';
    for (let i = 0; i < PROJECTION_LABELS.length; i++) {
        const thumb = document.createElement('div');
        thumb.className = 'cmode-thumb' + ((projectionMode > 0 && i === projectionMode - 1) ? ' active' : '');
        if (projectionMode > 0 && i === projectionMode - 1) {
            thumb.style.borderColor = PROJECTION_COLORS[i];
            thumb.style.background = PROJECTION_COLORS[i] + '1a';
        }
        const lbl = document.createElement('span');
        lbl.textContent = PROJECTION_LABELS[i];
        if (projectionMode > 0 && i === projectionMode - 1) lbl.style.color = PROJECTION_COLORS[i];
        thumb.appendChild(lbl);
        el.appendChild(thumb);
    }
    el.classList.add('visible');
    clearTimeout(_projStripTimer);
    _projStripTimer = setTimeout(() => el.classList.remove('visible'), 2200);
}
```

- [ ] **Step 4: Update `renderEggs()` to use 1-based indexing**

The egg badge check `if (projectionMode > 0)` should now reference `PROJECTION_LABELS[projectionMode - 1]` and `PROJECTION_COLORS[projectionMode - 1]`.

- [ ] **Step 5: Update all other projection mode references**

Search for `PROJECTION_LABELS[projectionMode]` and `PROJECTION_COLORS[projectionMode]` and update to `[projectionMode - 1]` where `projectionMode > 0`.

Also update the WebSocket message and server-side if `projection_mode` value mapping changed (check if server uses 0=off, 1=max, etc. — if so, no change needed since we're keeping the same integer values sent to server).

- [ ] **Step 6: Use ui-consistency-audit to check for overlapping elements in projection mode**

- [ ] **Step 7: Test and commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: projection dim shows 'p' in dim bar, remove 'off' from p cycle"
```

---

## Execution Order

Tasks are independent but recommended order for efficient testing:

1. **Task 1** (Screenshot fix) — critical bug, quick fix
2. **Task 2** (CSV export fix) — same pattern as Task 1
3. **Task 3** (ROI trash icons) — UX improvement
4. **Task 7** (Projection mode) — feature enhancement
5. **Task 4** (Colorbar border alignment) — needs visual audit
6. **Task 5** (Colorbar all modes) — needs visual audit
7. **Task 6** (ROI misalignment) — needs visual audit
