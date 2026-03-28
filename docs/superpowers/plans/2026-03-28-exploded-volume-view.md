# Exploded Volume View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Press `E` to explode the current volume into a 3D fanned deck of slices that auto-rotates, with `[`/`]` density control and click-to-exit.

**Architecture:** New HTTP endpoint returns batch JPEG thumbnails for sampled slices. Frontend builds a CSS 3D transform scene with DOM elements (not canvas), spring-animated entry/exit, and requestAnimationFrame auto-rotation. Integrates as a new mode flag (`explodedActive`) in the existing mode system.

**Tech Stack:** Python (FastAPI endpoint, PIL, existing render pipeline), vanilla JS (CSS 3D transforms, rAF), existing colormap/dynamic-range logic.

**Spec:** `docs/superpowers/specs/2026-03-28-exploded-volume-view-design.md`

---

### Task 1: Server endpoint for batch slice thumbnails

**Files:**
- Modify: `src/arrayview/_server.py` (add new endpoint near existing `/thumbnail/{sid}` at ~line 2074)

This endpoint renders multiple slices at once for the exploded view. It reuses the existing `render_rgba()` pipeline so all colormaps, dynamic range, complex modes, and log scale are respected.

- [ ] **Step 1: Add the POST /exploded/{sid} endpoint**

Add this endpoint after the existing `/thumbnail/{sid}` endpoint (~line 2107):

```python
from fastapi import Body

@app.post("/exploded/{sid}")
async def get_exploded_slices(
    sid: str,
    dim_x: int = Body(...),
    dim_y: int = Body(...),
    scroll_dim: int = Body(...),
    indices: list[int] = Body(...),
    width: int = Body(256),
    colormap: str = Body("gray"),
    dr: int = Body(1),
    complex_mode: int = Body(0),
    log_scale: bool = Body(False),
    vmin_override: float | None = Body(None),
    vmax_override: float | None = Body(None),
):
    """Return JPEG thumbnails for multiple slices along scroll_dim."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    ndim = len(session.shape)
    if ndim < 3:
        return JSONResponse({"error": "need >= 3D array"}, status_code=400)

    Image = _pil_image()
    results = []

    # Build base index tuple (middle of each dim)
    base_indices = [s // 2 for s in session.shape]

    for slice_idx in indices:
        idx_list = list(base_indices)
        idx_list[scroll_dim] = min(max(0, slice_idx), session.shape[scroll_dim] - 1)

        rgba = await asyncio.to_thread(
            render_rgba, session, dim_x, dim_y, tuple(idx_list),
            colormap, dr, complex_mode, log_scale,
            vmin_override, vmax_override,
        )

        img = Image.fromarray(rgba[:, :, :3])
        # Maintain aspect ratio, fit within width
        aspect = img.height / img.width
        target_h = max(1, int(width * aspect))
        resample = Image.NEAREST if img.width <= width else Image.LANCZOS
        img = img.resize((width, target_h), resample)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        results.append({"index": slice_idx, "image": f"data:image/jpeg;base64,{b64}"})

    return JSONResponse({"slices": results})
```

Make sure `base64` is imported at the top of the file (add `import base64` if missing). Also ensure `from fastapi.responses import JSONResponse` is imported (check existing imports — it likely already is).

The `render_rgba` import should already be present from `_render.py`. Verify with the existing imports near the top of `_server.py`.

- [ ] **Step 2: Run the server and test manually**

```bash
cd /Users/oscar/Projects/packages/python/arrayview
python -c "
import numpy as np, requests, json
# Create test array and save
a = np.random.rand(32, 64, 64).astype(np.float32)
np.save('/tmp/test_exploded.npy', a)
"
```

Start the server in background, then test:
```bash
python -m arrayview /tmp/test_exploded.npy &
sleep 2
# Get the session ID from /sessions
SID=$(curl -s http://localhost:8000/sessions | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['sid'])")
# Test exploded endpoint
curl -s -X POST "http://localhost:8000/exploded/$SID" \
  -H "Content-Type: application/json" \
  -d '{"dim_x":2,"dim_y":1,"scroll_dim":0,"indices":[0,8,16,24,31],"width":128}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"slices\"])} slices, first image length: {len(d[\"slices\"][0][\"image\"])} chars')"
# Expected: "5 slices, first image length: NNNN chars"
kill %1 2>/dev/null
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_server.py
git commit -m "feat(exploded): add POST /exploded/{sid} batch thumbnail endpoint"
```

---

### Task 2: Exploded view HTML/CSS structure

**Files:**
- Modify: `src/arrayview/_viewer.html` (add CSS styles in the `<style>` block, add container div in the `<body>`)

- [ ] **Step 1: Add CSS for the exploded view**

Find the closing `</style>` tag in `_viewer.html` and add before it:

```css
/* ── Exploded Volume View ── */
#exploded-overlay {
    display: none;
    position: fixed;
    inset: 0;
    z-index: 8000;  /* above canvas, below help overlay (9000) */
    background: rgba(0, 0, 0, 0);
    transition: background 0.4s ease;
}
#exploded-overlay.active {
    display: block;
    background: rgba(0, 0, 0, 0.75);
}
#exploded-scene {
    position: absolute;
    inset: 0;
    perspective: 1200px;
    display: flex;
    align-items: center;
    justify-content: center;
}
#exploded-turntable {
    transform-style: preserve-3d;
    position: relative;
    /* Size is set dynamically based on slice dimensions */
}
.exploded-slice {
    position: absolute;
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 3px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4);
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
    /* backface-visibility: hidden; — intentionally NOT set so slices visible from both sides */
}
.exploded-slice:hover {
    border-color: rgba(250, 204, 21, 0.6);
    box-shadow: 0 2px 20px rgba(250, 204, 21, 0.25);
}
.exploded-slice.active {
    border-color: #facc15;
    box-shadow: 0 0 16px rgba(250, 204, 21, 0.4);
}
.exploded-slice-label {
    position: absolute;
    bottom: -18px;
    left: 50%;
    transform: translateX(-50%);
    font: 10px var(--font-mono, monospace);
    color: rgba(255, 255, 255, 0.5);
    white-space: nowrap;
    pointer-events: none;
}
```

- [ ] **Step 2: Add the container div to the body**

Find the `<!-- help overlay -->` comment in the HTML body (this is a high-z-index element). Add the exploded overlay div BEFORE it:

```html
<!-- exploded volume view -->
<div id="exploded-overlay">
    <div id="exploded-scene">
        <div id="exploded-turntable"></div>
    </div>
</div>
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): add CSS and DOM structure for exploded volume view"
```

---

### Task 3: Core exploded view state and slice fetching

**Files:**
- Modify: `src/arrayview/_viewer.html` (add JS near other mode variables and functions)

- [ ] **Step 1: Add state variables**

Find the mode flag variables (near line 1520, where `let multiViewActive = false;` etc. are declared). Add after them:

```javascript
/* ── Exploded volume view state ── */
let explodedActive = false;
let _explodedAnimating = false;
let _explodedSlices = [];          // [{index, image, el}]
let _explodedDensityLevel = 0;     // -5..+5, same as vfDensityLevel
let _explodedCenterIdx = -1;       // which slice is "centered"
let _explodedRotation = 0;         // current turntable angle in degrees
let _explodedRafId = null;         // requestAnimationFrame ID
let _explodedPauseUntil = 0;       // timestamp: pause rotation until this time
const _EXPLODED_ROTATION_SPEED = 15; // degrees per second
const _EXPLODED_MAX_SLICES = 30;
const _EXPLODED_THUMB_W = 256;
```

- [ ] **Step 2: Add the slice sampling function**

Add this function in the JS section, after the state variables (or in a logical grouping near other utility functions):

```javascript
function _explodedSampleIndices(axisLen, densityLevel) {
    /* Return evenly-spaced slice indices along an axis.
       densityLevel: 0 = adaptive default, ±N adjusts count. */
    const baseCount = Math.min(axisLen, _EXPLODED_MAX_SLICES);
    // √2 per level (half-octave), same convention as vector field density
    const count = Math.max(3, Math.min(axisLen,
        Math.round(baseCount * Math.pow(1.4142, densityLevel))));
    if (count >= axisLen) {
        return Array.from({length: axisLen}, (_, i) => i);
    }
    const step = (axisLen - 1) / (count - 1);
    return Array.from({length: count}, (_, i) => Math.round(i * step));
}
```

- [ ] **Step 3: Add the slice fetching function**

```javascript
async function _explodedFetchSlices(indices) {
    /* Fetch batch thumbnails from the server. Returns array of {index, image}. */
    const body = {
        dim_x: dim_order[dim_order.length - 1],
        dim_y: dim_order[dim_order.length - 2],
        scroll_dim: current_slice_dim,
        indices: indices,
        width: _EXPLODED_THUMB_W,
        colormap: currentColormap(),
        dr: dr_idx,
        complex_mode: complexMode,
        log_scale: logScale,
        vmin_override: manualVmin,
        vmax_override: manualVmax,
    };
    const res = await fetch(`/exploded/${sid}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.slices;  // [{index, image}, ...]
}
```

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): add state variables, sampling, and fetch logic"
```

---

### Task 4: Build the 3D scene and entry animation

**Files:**
- Modify: `src/arrayview/_viewer.html` (add scene building and animation functions)

- [ ] **Step 1: Add the scene building function**

This creates the DOM elements for each slice and positions them in 3D space:

```javascript
function _explodedBuildScene(sliceData) {
    /* Build DOM cards inside #exploded-turntable from fetched slice data. */
    const turntable = document.getElementById('exploded-turntable');
    turntable.innerHTML = '';
    _explodedSlices = [];

    const n = sliceData.length;
    if (n === 0) return;

    // Card dimensions: maintain aspect ratio from first image
    // We'll use a fixed width and compute height after first image loads
    const cardW = 180;  // CSS px
    const cardH = Math.round(cardW * (shape[dim_order[dim_order.length - 2]] / shape[dim_order[dim_order.length - 1]]));
    const spacing = Math.max(4, Math.min(12, 300 / n));  // Z spacing in px

    // Current scroll index for highlighting
    const currentScrollIdx = indices[current_slice_dim];

    for (let i = 0; i < n; i++) {
        const s = sliceData[i];
        const el = document.createElement('div');
        el.className = 'exploded-slice';
        if (s.index === currentScrollIdx) el.classList.add('active');
        el.style.width = cardW + 'px';
        el.style.height = cardH + 'px';
        el.style.backgroundImage = `url(${s.image})`;
        el.style.left = (-cardW / 2) + 'px';
        el.style.top = (-cardH / 2) + 'px';

        // Position along Z axis, centered around 0
        const zOffset = (i - (n - 1) / 2) * spacing;
        el.dataset.zOffset = zOffset;
        el.dataset.sliceIndex = s.index;

        // Scale: slightly larger near center
        const distFromCenter = Math.abs(i - (n - 1) / 2) / ((n - 1) / 2 || 1);
        const scl = 1.05 - 0.05 * distFromCenter;

        // Opacity: fade edges
        const opac = 1.0 - 0.3 * distFromCenter;

        el.style.transform = `translateZ(${zOffset}px) scale(${scl})`;
        el.style.opacity = opac;

        // Slice index label
        const label = document.createElement('div');
        label.className = 'exploded-slice-label';
        label.textContent = s.index;
        el.appendChild(label);

        // Click to exit and jump to this slice
        el.addEventListener('click', () => _explodedExitTo(s.index));

        turntable.appendChild(el);
        _explodedSlices.push({index: s.index, image: s.image, el});
    }

    // Set turntable dimensions for proper centering
    turntable.style.width = cardW + 'px';
    turntable.style.height = cardH + 'px';
}
```

- [ ] **Step 2: Add the spring entry animation**

```javascript
async function _explodedEnter() {
    /* Animate into exploded view. */
    if (explodedActive || _explodedAnimating) return;
    if (shape.length < 3) { showStatus('exploded: need ≥ 3D array'); return; }
    // Block if in any multi-pane mode
    if (ModeRegistry.isMultiPane) { showStatus('exploded: exit current mode first'); return; }

    _explodedAnimating = true;
    explodedActive = true;
    _explodedDensityLevel = 0;
    _explodedRotation = 0;

    const overlay = document.getElementById('exploded-overlay');
    const turntable = document.getElementById('exploded-turntable');

    // 1. Fetch slices
    const axisLen = shape[current_slice_dim];
    const sampleIndices = _explodedSampleIndices(axisLen, _explodedDensityLevel);
    showStatus(`exploding ${sampleIndices.length} slices…`);
    const sliceData = await _explodedFetchSlices(sampleIndices);
    if (sliceData.length === 0) {
        explodedActive = false;
        _explodedAnimating = false;
        showStatus('exploded: failed to fetch slices');
        return;
    }

    // 2. Build scene (cards start invisible)
    _explodedBuildScene(sliceData);

    // 3. Show overlay with background fade
    overlay.classList.add('active');

    // 4. Spring-in animation: cards start collapsed at Z=0 and fan out
    const cards = turntable.querySelectorAll('.exploded-slice');
    cards.forEach(el => {
        el.style.transition = 'none';
        const finalZ = parseFloat(el.dataset.zOffset);
        el.style.transform = `translateZ(0px) scale(0.6)`;
        el.style.opacity = '0';
        // Force reflow
        el.offsetHeight;
        // Animate with spring-like cubic-bezier (overshoot)
        el.style.transition = 'transform 0.7s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.5s ease-out';
        const distFromCenter = Math.abs(parseFloat(el.dataset.zOffset)) /
            (Math.max(..._explodedSlices.map(s => Math.abs(parseFloat(s.el.dataset.zOffset)))) || 1);
        const scl = 1.05 - 0.05 * distFromCenter;
        const opac = 1.0 - 0.3 * distFromCenter;
        el.style.transform = `translateZ(${finalZ}px) scale(${scl})`;
        el.style.opacity = opac;
    });

    // 5. Start auto-rotation after animation settles
    setTimeout(() => {
        _explodedAnimating = false;
        _explodedStartRotation();
    }, 750);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): build 3D scene and spring entry animation"
```

---

### Task 5: Auto-rotation

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add rotation loop and pause logic**

```javascript
function _explodedStartRotation() {
    /* Begin continuous turntable rotation via rAF. */
    if (_explodedRafId) return;
    let lastTime = performance.now();
    const turntable = document.getElementById('exploded-turntable');

    function tick(now) {
        if (!explodedActive) { _explodedRafId = null; return; }
        const dt = (now - lastTime) / 1000;  // seconds
        lastTime = now;

        // Pause rotation on hover / interaction
        if (now < _explodedPauseUntil) {
            _explodedRafId = requestAnimationFrame(tick);
            return;
        }

        _explodedRotation = (_explodedRotation + _EXPLODED_ROTATION_SPEED * dt) % 360;
        turntable.style.transform = `rotateY(${_explodedRotation}deg)`;
        _explodedRafId = requestAnimationFrame(tick);
    }
    _explodedRafId = requestAnimationFrame(tick);
}

function _explodedStopRotation() {
    if (_explodedRafId) {
        cancelAnimationFrame(_explodedRafId);
        _explodedRafId = null;
    }
}

function _explodedPauseRotation(durationMs) {
    /* Pause auto-rotation for durationMs, then resume. */
    _explodedPauseUntil = performance.now() + (durationMs || 2000);
}
```

- [ ] **Step 2: Add hover pause on the overlay**

Add this after the rotation functions:

```javascript
(function() {
    const overlay = document.getElementById('exploded-overlay');
    overlay.addEventListener('mouseenter', () => _explodedPauseRotation(99999));
    overlay.addEventListener('mouseleave', () => _explodedPauseRotation(2000));
})();
```

Wait — we only want to pause on slice hover, not the whole overlay. Revise: attach hover listeners to each slice card inside `_explodedBuildScene`, right after the click listener:

```javascript
        el.addEventListener('mouseenter', () => _explodedPauseRotation(99999));
        el.addEventListener('mouseleave', () => _explodedPauseRotation(2000));
```

Do NOT add the IIFE above. Instead, add these two lines inside the `for` loop in `_explodedBuildScene`, right after the `el.addEventListener('click', ...)` line.

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): add auto-rotation with hover pause"
```

---

### Task 6: Exit animation

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add exit function with spring collapse**

```javascript
function _explodedExitTo(targetSliceIndex) {
    /* Collapse exploded view and jump to targetSliceIndex. */
    if (!explodedActive || _explodedAnimating) return;
    _explodedAnimating = true;
    _explodedStopRotation();

    const overlay = document.getElementById('exploded-overlay');
    const turntable = document.getElementById('exploded-turntable');

    // 1. Collapse cards toward center
    const cards = turntable.querySelectorAll('.exploded-slice');
    cards.forEach(el => {
        el.style.transition = 'transform 0.45s cubic-bezier(0.55, 0, 0.68, 0.53), opacity 0.35s ease-in';
        el.style.transform = 'translateZ(0px) scale(0.6)';
        el.style.opacity = '0';
    });

    // 2. Fade out background
    setTimeout(() => {
        overlay.style.transition = 'background 0.3s ease';
        overlay.style.background = 'rgba(0, 0, 0, 0)';
    }, 250);

    // 3. Clean up and jump to slice
    setTimeout(() => {
        overlay.classList.remove('active');
        overlay.style.transition = '';
        overlay.style.background = '';
        turntable.innerHTML = '';
        _explodedSlices = [];
        explodedActive = false;
        _explodedAnimating = false;

        // Jump to the target slice
        if (targetSliceIndex != null) {
            indices[current_slice_dim] = targetSliceIndex;
            updateView();
            showStatus(`slice ${targetSliceIndex}`);
        }
    }, 550);
}

function _explodedExit() {
    /* Exit exploded view, returning to the nearest-to-center slice. */
    // Find the slice closest to the current scroll index
    const currentIdx = indices[current_slice_dim];
    let bestSlice = _explodedSlices.length > 0 ? _explodedSlices[0].index : currentIdx;
    let bestDist = Infinity;
    for (const s of _explodedSlices) {
        const d = Math.abs(s.index - currentIdx);
        if (d < bestDist) { bestDist = d; bestSlice = s.index; }
    }
    _explodedExitTo(bestSlice);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): add spring collapse exit animation"
```

---

### Task 7: Keyboard integration and density controls

**Files:**
- Modify: `src/arrayview/_viewer.html` (keydown handler)

- [ ] **Step 1: Add early return guard for exploded mode**

In the main keydown handler (the `sink.addEventListener('keydown', ...)` block), add a guard clause right after the existing `_immersiveAnimating` guard (around line 7454). This blocks all keys except the ones we handle while exploded:

```javascript
        // ── Exploded view: block all keys except E, Esc, s, [, ], scroll ──
        if (explodedActive || _explodedAnimating) {
            if (e.key === 'e' || e.key === 'E') { _explodedExit(); return; }
            if (e.key === 'Escape') { _explodedExit(); return; }
            if (e.key === 's') { /* fall through to screenshot handler below */ }
            else if (e.key === '[' || e.key === ']') {
                _explodedDensityLevel = Math.max(-5, Math.min(5,
                    _explodedDensityLevel + (e.key === ']' ? 1 : -1)));
                _explodedRefreshDensity();
                return;
            } else {
                return;  // block everything else
            }
        }
```

- [ ] **Step 2: Add the `E` key binding**

In the keydown handler, find a suitable location in the if/else chain (near other mode toggles like `f` for FFT, `F` for fullscreen). Add:

```javascript
        } else if (e.key === 'E' || e.key === 'e') {
            if (explodedActive) { _explodedExit(); }
            else { _explodedEnter(); }
```

Note: make sure this doesn't conflict with the existing `e` key (copy URL). Check what lowercase `e` currently does. Looking at the spec, the existing `e` key copies a reusable URL. We should use uppercase `E` only. Revise:

```javascript
        } else if (e.key === 'E' && !e.shiftKey) {
            if (explodedActive) { _explodedExit(); }
            else { _explodedEnter(); }
```

Check the existing keydown handler to see if `E` (uppercase) is already bound. If it is, find an alternative or share the binding contextually. The spec says `E`, so use uppercase `E`.

- [ ] **Step 3: Add the density refresh function**

```javascript
async function _explodedRefreshDensity() {
    /* Re-fetch slices at new density and rebuild scene (no full re-animation). */
    if (!explodedActive || _explodedAnimating) return;
    _explodedAnimating = true;
    _explodedStopRotation();

    const turntable = document.getElementById('exploded-turntable');
    const savedRotation = _explodedRotation;

    const axisLen = shape[current_slice_dim];
    const sampleIndices = _explodedSampleIndices(axisLen, _explodedDensityLevel);
    showStatus(`density ${_explodedDensityLevel > 0 ? '+' : ''}${_explodedDensityLevel} (${sampleIndices.length} slices)`);

    const sliceData = await _explodedFetchSlices(sampleIndices);
    if (sliceData.length === 0) { _explodedAnimating = false; return; }

    // Quick fade-out existing cards
    const oldCards = turntable.querySelectorAll('.exploded-slice');
    oldCards.forEach(el => {
        el.style.transition = 'opacity 0.2s ease';
        el.style.opacity = '0';
    });

    await new Promise(r => setTimeout(r, 220));

    // Rebuild and fade in
    _explodedBuildScene(sliceData);
    _explodedRotation = savedRotation;
    turntable.style.transform = `rotateY(${_explodedRotation}deg)`;

    const newCards = turntable.querySelectorAll('.exploded-slice');
    newCards.forEach(el => {
        const origOpac = el.style.opacity;
        el.style.opacity = '0';
        el.offsetHeight;
        el.style.transition = 'opacity 0.25s ease';
        el.style.opacity = origOpac;
    });

    setTimeout(() => {
        _explodedAnimating = false;
        _explodedStartRotation();
    }, 280);
}
```

- [ ] **Step 4: Add scroll handling for centering slices**

Add a wheel listener on the overlay. Place this near the other exploded functions:

```javascript
document.getElementById('exploded-overlay').addEventListener('wheel', (e) => {
    if (!explodedActive || _explodedAnimating) return;
    e.preventDefault();
    _explodedPauseRotation(2000);

    // Shift which slice is highlighted
    const dir = e.deltaY > 0 ? 1 : -1;
    const activeIdx = _explodedSlices.findIndex(s => s.el.classList.contains('active'));
    const newIdx = Math.max(0, Math.min(_explodedSlices.length - 1,
        (activeIdx === -1 ? 0 : activeIdx) + dir));

    _explodedSlices.forEach(s => s.el.classList.remove('active'));
    _explodedSlices[newIdx].el.classList.add('active');
    _explodedCenterIdx = _explodedSlices[newIdx].index;
}, {passive: false});
```

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): keyboard integration, density controls, scroll centering"
```

---

### Task 8: Help overlay and status updates

**Files:**
- Modify: `src/arrayview/_viewer.html` (help overlay HTML, keyboard hints)

- [ ] **Step 1: Add exploded view to the help overlay**

Find the help overlay's keyboard shortcut table in the HTML. Look for the section that lists mode keys (near `f` for FFT, `F` for fullscreen/zen). Add a row:

```html
<tr><td><kbd>E</kbd></td><td>exploded volume view (3D card deck)</td></tr>
```

Also add in a "While in exploded view" section or note:

```html
<tr><td><kbd>[</kbd> <kbd>]</kbd></td><td>decrease / increase slice density (exploded)</td></tr>
<tr><td>scroll</td><td>highlight prev / next slice (exploded)</td></tr>
<tr><td>click</td><td>exit and jump to clicked slice (exploded)</td></tr>
```

Find the exact location by searching for existing keyboard shortcut rows like `<kbd>F</kbd>` for zen/fullscreen mode.

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): add help overlay documentation"
```

---

### Task 9: Visual smoke test

**Files:**
- Modify: `tests/visual_smoke.py` (if it exists — add exploded view test case)

- [ ] **Step 1: Check if visual_smoke.py exists and add a test**

Look for `tests/visual_smoke.py` or similar. If it exists, add a test case for exploded view. The test should:
1. Open a 3D array
2. Press `E` to enter exploded view
3. Verify the overlay is visible
4. Press `]` to increase density
5. Press `E` to exit
6. Verify normal view is restored

If no visual smoke test file exists, create a manual test script:

```python
"""Manual smoke test for exploded volume view.

Run:
    python -m arrayview /tmp/test_exploded.npy

Then:
    1. Press E — should see 3D card deck with auto-rotation
    2. Hover a slice — rotation should pause
    3. Press ] twice — more slices should appear
    4. Press [ three times — fewer slices
    5. Scroll wheel — yellow highlight should move between slices
    6. Click any slice — should exit and jump to that slice index
    7. Press E again — re-enter
    8. Press Escape — should exit
    9. Verify normal view works after exit (scroll, colormap, etc.)
"""
```

Save this as `tests/manual_exploded_smoke.py` (or add to existing smoke test infrastructure).

- [ ] **Step 2: Run a quick manual verification**

```bash
python -c "import numpy as np; np.save('/tmp/test_exploded.npy', np.random.rand(64, 128, 128).astype(np.float32))"
python -m arrayview /tmp/test_exploded.npy
```

Open the viewer, press `E`, verify the 3D deck appears and rotates. Test `[`/`]`, scroll, click exit, `Esc` exit.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test(exploded): add smoke test for exploded volume view"
```

---

### Task 10: Edge cases and polish

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Handle 2D arrays gracefully**

Verify that `_explodedEnter()` already returns early for 2D arrays (the `shape.length < 3` check). Test with a 2D array:

```bash
python -c "import numpy as np; np.save('/tmp/test_2d.npy', np.random.rand(128, 128).astype(np.float32))"
python -m arrayview /tmp/test_2d.npy
```

Press `E` — should show status "exploded: need ≥ 3D array" and do nothing.

- [ ] **Step 2: Handle small volumes (< 3 slices)**

If the scroll axis has < 3 slices, exploded view isn't useful. Add a guard in `_explodedEnter()` after the shape check:

```javascript
    if (shape[current_slice_dim] < 3) {
        showStatus('exploded: need ≥ 3 slices along scroll axis');
        _explodedAnimating = false;
        explodedActive = false;
        return;
    }
```

- [ ] **Step 3: Add explodedActive to ModeRegistry**

Update the `ModeRegistry` object to be aware of exploded mode. In the `isMultiPane` getter, exploded view should NOT count as multi-pane (it's an overlay, not a layout change). But the `name` getter should report it:

```javascript
    get name() {
        if (explodedActive) return 'exploded';
        if (compareQmriActive) return 'compareQmri';
        // ... rest unchanged
    },
```

- [ ] **Step 4: Prevent entering other modes while exploded**

In the mode entry functions (`enterMultiView`, `enterCompareMode`, etc.), add early returns if `explodedActive` is true. Or rely on the keydown guard from Task 7 which blocks all keys — verify this is sufficient.

The keydown guard from Task 7 already blocks all keys except `E`, `Esc`, `s`, `[`, `]` while exploded. So mode-entry keys like `v`, `z`, `q` are already blocked. No additional guards needed.

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat(exploded): edge case handling and ModeRegistry integration"
```
