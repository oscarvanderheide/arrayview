# TODO Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix compact mode UI issues, correct complex projection math, and redesign histogram indicators.

**Architecture:** Backend change in `_render.py` for complex-aware projections. Frontend changes in `_viewer.html` for compact mode CSS/JS, histogram drawing, and colorbar layout.

**Tech Stack:** Python/NumPy (backend), vanilla JS/Canvas (frontend)

---

### Task 1: Complex-Aware Projections

**Files:**
- Modify: `src/arrayview/_render.py:148-196`
- Test: `tests/test_api.py` (add complex projection tests)

- [ ] **Step 1: Write failing tests for complex projections**

Add to `tests/test_api.py`:

```python
import numpy as np

@pytest.fixture
def complex_sid(client, server_url):
    """Load a 3D complex array for projection tests."""
    import tempfile, os
    arr = np.array([[[1+2j, 3+4j], [5+6j, 7+8j]],
                    [[2+1j, 4+3j], [6+5j, 8+7j]]], dtype=np.complex64)
    f = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    np.save(f.name, arr)
    f.close()
    resp = client.post(f"{server_url}/load", json={"filepath": f.name, "name": "complex_test"})
    sid = resp.json()["sid"]
    yield sid
    os.unlink(f.name)


def test_complex_projection_max(client, server_url, complex_sid):
    """max projection on complex data should use magnitude, no warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.ComplexWarning)
        resp = client.get(f"{server_url}/slice/{complex_sid}",
                          params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                                  "proj_dim": 2, "proj_mode": 1})
    assert resp.status_code == 200


def test_complex_projection_min(client, server_url, complex_sid):
    """min projection on complex data should use magnitude, no warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.ComplexWarning)
        resp = client.get(f"{server_url}/slice/{complex_sid}",
                          params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                                  "proj_dim": 2, "proj_mode": 2})
    assert resp.status_code == 200


def test_complex_projection_sos(client, server_url, complex_sid):
    """sos projection on complex data should use z*conj(z), no warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.ComplexWarning)
        resp = client.get(f"{server_url}/slice/{complex_sid}",
                          params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                                  "proj_dim": 2, "proj_mode": 5})
    assert resp.status_code == 200


def test_complex_projection_mean(client, server_url, complex_sid):
    """mean projection on complex data should work without warnings."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.ComplexWarning)
        resp = client.get(f"{server_url}/slice/{complex_sid}",
                          params={"indices": "0,0,0", "dim_x": 1, "dim_y": 0,
                                  "proj_dim": 2, "proj_mode": 3})
    assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -k "complex_projection" -v`
Expected: FAIL with ComplexWarning for max, min, sos tests

- [ ] **Step 3: Implement complex-aware projection logic**

In `src/arrayview/_render.py`, replace lines 180-184 in `extract_projection()`:

```python
    if proj_mode == 5:  # SOS — sum of magnitude-squared
        if np.iscomplexobj(vol):
            result = np.sum((vol * np.conj(vol)).real, axis=proj_axis).astype(np.float32)
        else:
            result = np.sum(vol.astype(np.float64) ** 2, axis=proj_axis).astype(np.float32)
    elif proj_mode == 1 and np.iscomplexobj(vol):  # max by magnitude
        mag = np.abs(vol)
        idx = np.expand_dims(np.argmax(mag, axis=proj_axis), axis=proj_axis)
        result = np.take_along_axis(vol, idx, axis=proj_axis).squeeze(axis=proj_axis)
    elif proj_mode == 2 and np.iscomplexobj(vol):  # min by magnitude
        mag = np.abs(vol)
        idx = np.expand_dims(np.argmin(mag, axis=proj_axis), axis=proj_axis)
        result = np.take_along_axis(vol, idx, axis=proj_axis).squeeze(axis=proj_axis)
    else:
        _, op = PROJECTION_OPS[proj_mode]
        result = op(vol, axis=proj_axis)
```

Also update the `.astype(np.float32)` at line 190 to handle complex results:

```python
    if np.iscomplexobj(result):
        result = np.nan_to_num(result).astype(np.complex64)
    else:
        result = np.nan_to_num(result).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -k "complex_projection" -v`
Expected: All 4 PASS

- [ ] **Step 5: Run full API test suite**

Run: `uv run pytest tests/test_api.py -v`
Expected: All tests pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_render.py tests/test_api.py
git commit -m "fix: complex-aware projections (max/min by magnitude, sos via z·conj(z))"
```

---

### Task 2: Compact Mode — Keep Dim Bar Full Size

**Files:**
- Modify: `src/arrayview/_viewer.html:625-629` (CSS)
- Modify: `src/arrayview/_viewer.html:1725-1729` (`uiReserveV()`)

- [ ] **Step 1: Remove font-size reduction from compact mode #info CSS**

At line 625-629, change:
```css
body.compact-mode #info {
    position: fixed; z-index: 6; font-size: 14px; margin: 0; padding: 4px 14px;
    transform: translateX(-50%); align-self: auto;
    transition: font-size 0.25s ease;
}
```
to:
```css
body.compact-mode #info {
    position: fixed; z-index: 6; margin: 0; padding: 4px 14px;
    transform: translateX(-50%); align-self: auto;
}
```

Remove `font-size: 14px` and `transition: font-size 0.25s ease`.

- [ ] **Step 2: Update uiReserveV() compact return value**

At line 1726-1728, the compact mode returns 50px (assumed small info bar). Since the info bar is now full size, increase to account for the unchanged dim bar height. Change:

```javascript
if (_compactActive) {
    // In compact mode: array-name is hidden, info is tiny
    return 50; // wrapper padding + small info bar
}
```
to:
```javascript
if (_compactActive) {
    // In compact mode: array-name is hidden but info bar keeps full size
    let h = 30; // wrapper padding
    const infoEl = document.getElementById('info');
    if (infoEl) {
        const s = getComputedStyle(infoEl);
        h += infoEl.offsetHeight + parseFloat(s.marginTop) + parseFloat(s.marginBottom);
    }
    return Math.max(80, h);
}
```

- [ ] **Step 3: Verify visually — open viewer, toggle compact mode (K), confirm dim bar stays same size**

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: keep dim bar full size in compact mode"
```

---

### Task 3: Compact Mode — Vmin/Vmax Labels Shown

**Files:**
- Modify: `src/arrayview/_viewer.html:2483-2488`

- [ ] **Step 1: Show vmin/vmax spans in compact mode**

At lines 2483-2488, the compact overlay branch hides the labels. Change:
```javascript
slimCbLabels.style.display = 'none';
// Also hide inline vmin/vmax flanking spans so they don't widen the bar
const _cvmin = document.getElementById('slim-cb-vmin');
const _cvmax = document.getElementById('slim-cb-vmax');
if (_cvmin) _cvmin.style.display = 'none';
if (_cvmax) _cvmax.style.display = 'none';
```
to:
```javascript
slimCbLabels.style.display = 'none';
// Show inline vmin/vmax flanking spans same as normal mode
const _cvmin = document.getElementById('slim-cb-vmin');
const _cvmax = document.getElementById('slim-cb-vmax');
if (_cvmin) _cvmin.style.display = '';
if (_cvmax) _cvmax.style.display = '';
```

Also, later in the function (around line 2569-2575), make sure the compact mode branch also populates the label text. Check that `slimVmin.textContent` and `slimVmax.textContent` are set with values (not cleared) when in compact mode. The non-compact code path at lines 2572-2574 already does this, so the fix above should be sufficient since the compact branch returns after positioning, and the label text is set in the shared path.

Verify: the labels show values when in compact mode, matching normal mode appearance.

- [ ] **Step 2: Verify visually — compact mode shows vmin/vmax flanking the colorbar**

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: show vmin/vmax labels in compact mode same as normal"
```

---

### Task 4: Compact Mode — Eggs Inside Canvas

**Files:**
- Modify: `src/arrayview/_viewer.html:3685-3696`

- [ ] **Step 1: Reposition eggs inside canvas, centered above colorbar**

At lines 3685-3696, replace the compact mode branch:
```javascript
if (_compactActive) {
    const vpEl = document.getElementById('canvas-viewport');
    if (!vpEl) return;
    const vpRect = vpEl.getBoundingClientRect();
    el.style.flexDirection = 'column';
    el.style.alignItems    = 'flex-start';
    el.style.top           = vpRect.top + 'px';
    el.style.bottom        = '';
    el.style.right         = Math.round(window.innerWidth - vpRect.left + 4) + 'px';
    el.style.left          = '';
    el.style.transform     = '';
    return;
```
with:
```javascript
if (_compactActive) {
    const vpEl = document.getElementById('canvas-viewport');
    if (!vpEl) return;
    const vpRect = vpEl.getBoundingClientRect();
    const cbIsland = document.getElementById('slim-cb-island');
    const cbRect = cbIsland ? cbIsland.getBoundingClientRect() : null;
    // Position inside canvas, centered, just above colorbar
    el.style.flexDirection = 'row';
    el.style.alignItems    = 'center';
    el.style.left          = vpRect.left + vpRect.width / 2 + 'px';
    el.style.transform     = 'translateX(-50%)';
    el.style.top           = '';
    el.style.right         = '';
    el.style.bottom        = cbRect
        ? (window.innerHeight - cbRect.top + 6) + 'px'
        : (window.innerHeight - vpRect.bottom + 30) + 'px';
    return;
```

- [ ] **Step 2: Verify visually — eggs appear as horizontal row centered above colorbar in compact mode**

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: position eggs inside canvas above colorbar in compact mode"
```

---

### Task 5: Histogram — Shaded Out-of-Range Bars + Bracket Indicators

**Files:**
- Modify: `src/arrayview/_viewer.html:2605-2691`

- [ ] **Step 1: Add shading to out-of-range histogram bars**

In `_drawHistogramBarsOnColorbar()`, modify the bar drawing loop (lines 2643-2658). Replace:
```javascript
for (let i = 0; i < nBins; i++) {
    const barH = Math.max(0.5, barFrac(counts[i]) * (histAreaH - 2));
    const x = i * binW;
    const y = histAreaH - barH; // bars grow upward from the gradient strip
    let fillStyle;
    if (_cmapData) {
        const binCenter = edges ? (edges[i] + edges[i + 1]) / 2 : dmin + (i + 0.5) * (dmax - dmin) / nBins;
        const frac = Math.max(0, Math.min(1, (binCenter - windowVmin) / windowRange));
        const px = Math.min(_cmapSz - 1, Math.round(frac * (_cmapSz - 1))) * 4;
        fillStyle = `rgba(${_cmapData[px]},${_cmapData[px+1]},${_cmapData[px+2]},0.85)`;
    } else {
        fillStyle = isDark ? 'rgba(180,180,200,0.55)' : 'rgba(80,80,100,0.5)';
    }
    sCtx.fillStyle = fillStyle;
    sCtx.fillRect(x, y, binW - 0.5, barH);
}
```
with:
```javascript
for (let i = 0; i < nBins; i++) {
    const barH = Math.max(0.5, barFrac(counts[i]) * (histAreaH - 2));
    const x = i * binW;
    const y = histAreaH - barH;
    const binCenter = edges ? (edges[i] + edges[i + 1]) / 2 : dmin + (i + 0.5) * (dmax - dmin) / nBins;
    const inRange = binCenter >= windowVmin && binCenter <= windowVmax;
    const alpha = inRange ? 0.85 : 0.2;
    let fillStyle;
    if (_cmapData) {
        const frac = Math.max(0, Math.min(1, (binCenter - windowVmin) / windowRange));
        const px = Math.min(_cmapSz - 1, Math.round(frac * (_cmapSz - 1))) * 4;
        fillStyle = `rgba(${_cmapData[px]},${_cmapData[px+1]},${_cmapData[px+2]},${alpha})`;
    } else {
        fillStyle = isDark ? `rgba(180,180,200,${alpha * 0.65})` : `rgba(80,80,100,${alpha * 0.6})`;
    }
    sCtx.fillStyle = fillStyle;
    sCtx.fillRect(x, y, binW - 0.5, barH);
}
```

- [ ] **Step 2: Replace vertical lines with bracket markers in `_drawClimLines()`**

Replace the full `_drawClimLines()` function (lines 2672-2691):
```javascript
function _drawClimLines(sCtx, cssW, stripH) {
    const { vmin: dmin, vmax: dmax } = _histData;
    const range = dmax - dmin || 1;
    const vminF = ((manualVmin ?? currentVmin) - dmin) / range;
    const vmaxF = ((manualVmax ?? currentVmax) - dmin) / range;
    const hlineColor = isDark ? 'rgba(255,255,100,0.8)' : 'rgba(180,100,0,0.8)';
    sCtx.strokeStyle = hlineColor;
    sCtx.lineWidth = 1.5;
    _histVminX = null; _histVmaxX = null;
    if (vminF >= 0 && vminF <= 1) {
        const x = Math.round(vminF * (cssW - 1)) + 0.5;
        _histVminX = x - 0.5;
        sCtx.beginPath(); sCtx.moveTo(x, 0); sCtx.lineTo(x, stripH); sCtx.stroke();
    }
    if (vmaxF >= 0 && vmaxF <= 1 && Math.abs(vmaxF - vminF) > 0.002) {
        const x = Math.round(vmaxF * (cssW - 1)) + 0.5;
        _histVmaxX = x - 0.5;
        sCtx.beginPath(); sCtx.moveTo(x, 0); sCtx.lineTo(x, stripH); sCtx.stroke();
    }
}
```

with bracket-marker version:
```javascript
function _drawClimLines(sCtx, cssW, stripH) {
    const { vmin: dmin, vmax: dmax } = _histData;
    const range = dmax - dmin || 1;
    const vminF = ((manualVmin ?? currentVmin) - dmin) / range;
    const vmaxF = ((manualVmax ?? currentVmax) - dmin) / range;
    const histAreaH = stripH - CB_COLLAPSED_H;
    const bracketColor = isDark ? 'rgba(255,255,100,0.85)' : 'rgba(180,100,0,0.85)';
    const bracketH = Math.min(6, histAreaH * 0.25);
    const bracketW = 5;
    sCtx.strokeStyle = bracketColor;
    sCtx.lineWidth = 1.5;
    _histVminX = null; _histVmaxX = null;
    // Vmin bracket: ⌐ at top-left, ⌊ at bottom-left
    if (vminF >= 0 && vminF <= 1) {
        const x = Math.round(vminF * (cssW - 1)) + 0.5;
        _histVminX = x - 0.5;
        sCtx.beginPath();
        sCtx.moveTo(x, bracketH); sCtx.lineTo(x, 0); sCtx.lineTo(x + bracketW, 0);
        sCtx.stroke();
        sCtx.beginPath();
        sCtx.moveTo(x, histAreaH - bracketH); sCtx.lineTo(x, histAreaH); sCtx.lineTo(x + bracketW, histAreaH);
        sCtx.stroke();
    }
    // Vmax bracket: mirrored
    if (vmaxF >= 0 && vmaxF <= 1 && Math.abs(vmaxF - vminF) > 0.002) {
        const x = Math.round(vmaxF * (cssW - 1)) + 0.5;
        _histVmaxX = x - 0.5;
        sCtx.beginPath();
        sCtx.moveTo(x, bracketH); sCtx.lineTo(x, 0); sCtx.lineTo(x - bracketW, 0);
        sCtx.stroke();
        sCtx.beginPath();
        sCtx.moveTo(x, histAreaH - bracketH); sCtx.lineTo(x, histAreaH); sCtx.lineTo(x - bracketW, histAreaH);
        sCtx.stroke();
    }
}
```

- [ ] **Step 3: Verify visually — expand histogram with `d` key, confirm shaded bars and bracket indicators**

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: histogram shaded out-of-range bars + bracket vmin/vmax indicators"
```

---

### Task 6: Histogram — Full Width When Expanded

**Files:**
- Modify: `src/arrayview/_viewer.html:2569-2575`

- [ ] **Step 1: Hide flanking labels when histogram is expanded**

At lines 2569-2575 in `drawSlimColorbar()`, change:
```javascript
if (_cbExpanded) {
    slimVmin.textContent = '';
    slimVmax.textContent = '';
} else {
    slimVmin.textContent = _cbFmt(manualVmin ?? currentVmin);
    slimVmax.textContent = _cbFmt(manualVmax ?? currentVmax);
}
```
to:
```javascript
if (_cbExpanded) {
    slimVmin.style.display = 'none';
    slimVmax.style.display = 'none';
} else {
    slimVmin.style.display = '';
    slimVmax.style.display = '';
    slimVmin.textContent = _cbFmt(manualVmin ?? currentVmin);
    slimVmax.textContent = _cbFmt(manualVmax ?? currentVmax);
}
```

- [ ] **Step 2: Verify visually — histogram fills full island width, labels reappear on collapse**

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: histogram uses full width when expanded (hide flanking labels)"
```

---

### Task 7: Histogram Background Color Verification

**Files:**
- Verify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Verify dynamic island background shows through histogram**

Open viewer, press `d` to expand histogram. Confirm the frosted glassmorphism background of `.cb-island` is visible behind the histogram bars. The gradient strip is already hidden when expanded (guarded by `if (!lebesgueMode && !_cbExpanded)` at line 2535).

If the canvas background is opaque and hides the island background: set `sCtx.clearRect(0, 0, ...)` before drawing histogram bars so the canvas is transparent and the island background shows through.

- [ ] **Step 2: If change needed, commit. Otherwise mark as verified.**

---

### Task 8: Colorbar Zoom-Out Bug

**Files:**
- Investigate: `src/arrayview/_viewer.html`

- [ ] **Step 1: Invoke @ui-consistency-audit skill to investigate**

Audit the colorbar positioning across zoom levels. Test: zoom in far, then zoom out past 1.0. Check if colorbar width/position breaks.

- [ ] **Step 2: Fix identified issues**

- [ ] **Step 3: Commit fix**

---

### Task 9: Verify Flickering Fix + Cross-Mode Audit

- [ ] **Step 1: Test `=` key in normal mode — confirm no flickering**

Open viewer with a 3D+ array. Press `=` once in normal mode. Check: does compact mode flicker on/off? If no flickering, mark item 1 as fixed.

- [ ] **Step 2: Run @ui-consistency-audit across all modes**

Verify that all changes (dim bar, eggs, vmin/vmax, histogram) work correctly in: normal, compact, compare, multiview, qMRI, and projection modes.

- [ ] **Step 3: Update docs/TODO.md — mark completed items, remove fixed ones**

- [ ] **Step 4: Final commit**

```bash
git add docs/TODO.md src/arrayview/_viewer.html
git commit -m "docs: update TODO — mark completed items from batch"
```
