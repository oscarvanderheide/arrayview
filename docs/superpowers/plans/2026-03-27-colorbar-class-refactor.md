# ColorBar Class Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 7+ independent colorbar implementations with a single `ColorBar` class that all modes instantiate, so features like smooth KDE histogram, colormap preview, and drag interaction work everywhere automatically.

**Architecture:** Define a `ColorBar` class inside `_viewer.html`'s inline `<script>` that owns its own canvas, island DOM, state (animation, histogram, KDE, drag), and event handlers. Each mode creates `ColorBar` instances with config (orientation, size, sync group). Global state (`_cbExpanded`, `_histData`, `_kdeSmooth`, `_cbAnimT`, etc.) moves into instance properties. Modes that need synced colorbars (diff left/right) share a `SyncGroup` that mirrors interactions.

**Tech Stack:** Vanilla JS (inline in `_viewer.html`), no build step, no external deps.

---

## File Structure

All changes are in one file:

- **Modify:** `src/arrayview/_viewer.html` — the monolithic viewer
  - Add `ColorBar` class definition (new section, ~400 lines)
  - Modify mode entry/exit functions to create/destroy `ColorBar` instances
  - Remove old drawing functions and global state
  - Update keyboard handler to delegate to active `ColorBar` instance(s)
- **Test:** `tests/test_api.py` — existing API tests (must keep passing)
- **Test:** `tests/visual_smoke.py` — visual regression (run after each task, review `smoke_output/`)

## Important Context

### Current colorbar DOM patterns

**Normal mode** (`#slim-cb-wrap`, static HTML at line 893):
```html
<div id="slim-cb-wrap">
  <span class="slim-cb-val" id="slim-cb-vmin"></span>
  <div id="slim-cb-col">
    <canvas id="slim-cb"></canvas>
    <div id="slim-cb-tri-zone">
      <div class="slim-cb-tri" id="slim-cb-tri-vmin"></div>
      <div class="slim-cb-tri" id="slim-cb-tri-vmax"></div>
    </div>
  </div>
  <span class="slim-cb-val" id="slim-cb-vmax"></span>
  <div id="slim-cb-labels"></div>
  <div id="slim-cb-tooltip"></div>
</div>
```

**Multi-view** (`#mv-cb-wrap`, created dynamically at line 7748):
```html
<div id="mv-cb-wrap">
  <span class="slim-cb-val" id="mv-cb-vmin"></span>
  <canvas id="mv-cb"></canvas>
  <span class="slim-cb-val" id="mv-cb-vmax"></span>
  <div id="mv-cb-labels"></div>
</div>
```

**Compare per-pane** (`.compare-pane-cb-island`, static HTML at line 924):
```html
<div class="cb-island compare-pane-cb-island">
  <span class="compare-pane-cb-val" id="compare-left-pane-cb-vmin"></span>
  <canvas class="compare-pane-cb" id="compare-left-pane-cb"></canvas>
  <span class="compare-pane-cb-val" id="compare-left-pane-cb-vmax"></span>
  <div class="compare-pane-cb-labels" id="compare-left-pane-cb-labels"></div>
</div>
```

### Key global state to migrate into instance properties

```
_histData, _histDataVersion, _histFetching     → this._histData, etc.
_kdeSmooth, _KDE_NPTS                          → this._kdeSmooth, etc.
_cbAnimT, _cbAnimating, _cbAnimTarget          → this._animT, etc.
_cbExpandedH                                   → this._expandedH
_cbExpanded                                    → this._expanded
_cbWinVminF, _cbWinVmaxF                       → this._winVminF, etc.
_cbHoverFrac                                   → this._hoverFrac
_cbDragActive, _dragTarget                     → this._dragActive, etc.
_cbMouseOver, _cbHoverTimer                    → this._mouseOver, etc.
_histAutoDismissTimer, _histAutoDismissPending  → this._autoDismissTimer, etc.
_histVminX, _histVmaxX                         → this._hitVminX, etc.
_cmapInIsland                                  → this._cmapActive
```

### Constants (shared, not per-instance)

```javascript
const CB_COLLAPSED_H = 13;
const CB_LABEL_H = 14;
const CB_DEFAULT_EXPANDED_H = 62;
const CB_KDE_NPTS = 300;
const CB_HIT_RADIUS = 7;
```

### CSS class `.cb-island` (line 128)

Already exists as a shared glassmorphic container style. The `ColorBar` class should use this class on its wrapper element, so all colorbars get the same theme-aware background.

---

## Task 1: Define `ColorBar` class — core rendering

Create the class with constructor, DOM creation, and the morph renderer (adapted from `_cbMorphDraw`).

**Files:**
- Modify: `src/arrayview/_viewer.html` — add class definition after the constants block (around line 2870)

- [ ] **Step 1: Add the `ColorBar` class skeleton and constructor**

Insert after the existing `_KDE_NPTS` constant (line 2868). The constructor creates the island DOM, canvas, labels, triangles, and tooltip. It accepts a config object.

```javascript
class ColorBar {
    /**
     * @param {Object} opts
     * @param {HTMLElement} opts.container — parent element to append the island into
     * @param {'horizontal'|'vertical'} [opts.orientation='horizontal']
     * @param {string} [opts.id] — unique prefix for element IDs (e.g., 'slim', 'mv', 'cmp-left')
     * @param {boolean} [opts.interactive=true] — enable wheel/drag/hover
     * @param {function} [opts.getStops] — returns current colormap gradient stops array
     * @param {function} [opts.getRange] — returns {vmin, vmax} for current data range
     * @param {function} [opts.getWindow] — returns {vmin, vmax} for current display window
     * @param {function} [opts.setWindow] — called with (vmin, vmax) when user drags
     * @param {function} [opts.fetchHistogram] — async, returns {counts, edges, vmin, vmax}
     * @param {function} [opts.onWindowChange] — called after any vmin/vmax change (for rendering)
     * @param {ColorBar} [opts.syncWith] — mirror interactions to this other ColorBar
     */
    constructor(opts) {
        this.opts = opts;
        this.orientation = opts.orientation || 'horizontal';
        this.interactive = opts.interactive !== false;
        this.id = opts.id || ('cb-' + Math.random().toString(36).slice(2, 8));

        // State
        this._expanded = false;
        this._animT = 0;           // 0 = collapsed, 1 = expanded
        this._animTarget = 0;
        this._animating = false;
        this._expandedH = CB_DEFAULT_EXPANDED_H;
        this._winVminF = 0;
        this._winVmaxF = 1;
        this._winVminTarget = 0;
        this._winVmaxTarget = 1;
        this._hoverFrac = -1;
        this._mouseOver = false;
        this._hoverTimer = null;
        this._dragActive = false;
        this._dragTarget = null;    // 'vmin' | 'vmax' | null
        this._dragX = null;         // for collapsed pan drag
        this._hitVminX = null;
        this._hitVmaxX = null;
        this._autoDismissTimer = null;
        this._autoDismissPending = false;
        this._cmapActive = false;

        // Histogram / KDE
        this._histData = null;
        this._histVersion = null;
        this._histFetching = false;
        this._kdeSmooth = null;

        // Sync
        this._syncWith = opts.syncWith || null;
        this._syncing = false;      // prevent infinite sync loops

        // Build DOM
        this._buildDOM(opts.container);
    }
}
```

- [ ] **Step 2: Add `_buildDOM` method**

Creates the island wrapper, canvas, value labels, triangle zone, and tooltip. Uses the existing `.cb-island` CSS class for consistent glassmorphic styling. For vertical orientation, adjusts flex direction.

```javascript
_buildDOM(container) {
    const el = (tag, cls, id) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (id) e.id = id;
        return e;
    };

    // Island wrapper
    this.wrap = el('div', 'cb-island cb-wrap');
    this.wrap.dataset.cbId = this.id;
    if (this.orientation === 'vertical') {
        this.wrap.style.flexDirection = 'column';
    }

    // Value labels
    this.vminLabel = el('span', 'slim-cb-val');
    this.vmaxLabel = el('span', 'slim-cb-val');

    // Canvas column (canvas + triangles)
    this.colWrap = el('div', 'cb-col');
    this.canvas = el('canvas', 'cb-canvas');
    this.triZone = el('div', 'cb-tri-zone');
    this.triVmin = el('div', 'slim-cb-tri cb-tri-vmin');
    this.triVmax = el('div', 'slim-cb-tri cb-tri-vmax');
    this.triZone.appendChild(this.triVmin);
    this.triZone.appendChild(this.triVmax);
    this.colWrap.appendChild(this.canvas);
    this.colWrap.appendChild(this.triZone);

    // Tooltip
    this.tooltip = el('div', 'cb-tooltip');
    this.tooltip.style.display = 'none';

    // Assemble
    if (this.orientation === 'horizontal') {
        this.wrap.appendChild(this.vminLabel);
        this.wrap.appendChild(this.colWrap);
        this.wrap.appendChild(this.vmaxLabel);
    } else {
        // Vertical: canvas on left, labels above/below
        this.wrap.appendChild(this.vmaxLabel); // top = max
        this.wrap.appendChild(this.colWrap);
        this.wrap.appendChild(this.vminLabel); // bottom = min
    }
    this.wrap.appendChild(this.tooltip);

    container.appendChild(this.wrap);
}
```

- [ ] **Step 3: Add `_buildKDE` and `_kdeAt` methods**

Adapted from the existing global `_buildKDE()` (line 2870) and `_kdeAt()` (line 2900). Identical logic, but operates on `this._histData` and writes to `this._kdeSmooth`.

```javascript
_buildKDE() {
    if (!this._histData || !this._histData.counts || !this._histData.counts.length) {
        this._kdeSmooth = null;
        return;
    }
    const { counts } = this._histData;
    const nBins = counts.length;
    const sorted = [...counts].sort((a, b) => a - b);
    const medianCount = sorted[Math.floor(sorted.length / 2)] || 1;
    const maxCount = Math.max(...counts, 1);
    const useLog = maxCount > 20 * medianCount;
    const vals = useLog ? counts.map(c => Math.log1p(c)) : [...counts];
    const vMax = Math.max(...vals, 1);
    const out = new Float64Array(CB_KDE_NPTS);
    const sigma = CB_KDE_NPTS / nBins * 1.2;
    for (let i = 0; i < nBins; i++) {
        const center = (i + 0.5) / nBins * CB_KDE_NPTS;
        const h = vals[i] / vMax;
        const r = Math.ceil(sigma * 3);
        for (let j = Math.max(0, Math.floor(center - r)); j < Math.min(CB_KDE_NPTS, Math.ceil(center + r)); j++) {
            const d = (j - center) / sigma;
            out[j] += h * Math.exp(-0.5 * d * d);
        }
    }
    let oMax = 0;
    for (let j = 0; j < CB_KDE_NPTS; j++) if (out[j] > oMax) oMax = out[j];
    if (oMax > 0) for (let j = 0; j < CB_KDE_NPTS; j++) out[j] /= oMax;
    this._kdeSmooth = out;
}

_kdeAt(t) {
    if (!this._kdeSmooth) return 0;
    const idx = Math.max(0, Math.min(CB_KDE_NPTS - 1, Math.floor(t * CB_KDE_NPTS)));
    const idx2 = Math.min(idx + 1, CB_KDE_NPTS - 1);
    const f = t * CB_KDE_NPTS - idx;
    return this._kdeSmooth[idx] + (this._kdeSmooth[idx2] - this._kdeSmooth[idx]) * f;
}
```

- [ ] **Step 4: Add `draw` method (the morph renderer)**

Adapted from `_cbMorphDraw` (line 2935). Renders the gradient + KDE smooth histogram + window lines + value labels on `this.canvas`. Handles both horizontal and vertical orientation. This is the single shared renderer that replaces all mode-specific drawing.

```javascript
draw() {
    if (this._cmapActive) return;
    const cvs = this.canvas;
    if (!cvs) return;
    const stops = this.opts.getStops();
    if (!stops || !stops.length) return;
    const dpr = window.devicePixelRatio || 1;
    const isVert = this.orientation === 'vertical';
    const cssW = cvs.clientWidth || 200;
    const aT = this._animT;
    const cssH = CB_COLLAPSED_H + (this._expandedH - CB_COLLAPSED_H) * aT;
    cvs.width = Math.round(cssW * dpr);
    cvs.height = Math.round(cssH * dpr);
    cvs.style.height = cssH + 'px';
    const ctx = cvs.getContext('2d');
    ctx.clearRect(0, 0, cvs.width, cvs.height);
    ctx.save();
    ctx.scale(dpr, dpr);

    // Compute target window fractions
    const win = this.opts.getWindow();
    const range = this.opts.getRange();
    let vminFrac = 0, vmaxFrac = 1;
    if (this._histData) {
        const { vmin: dmin, vmax: dmax } = this._histData;
        const r = dmax - dmin || 1;
        vminFrac = (win.vmin - dmin) / r;
        vmaxFrac = (win.vmax - dmin) / r;
    }
    this._winVminTarget = vminFrac;
    this._winVmaxTarget = vmaxFrac;
    if (this._dragActive) {
        this._winVminF = vminFrac;
        this._winVmaxF = vmaxFrac;
    } else {
        this._winVminF += (vminFrac - this._winVminF) * 0.18;
        this._winVmaxF += (vmaxFrac - this._winVmaxF) * 0.18;
    }
    const effVmin = this._winVminF * aT;
    const effVmax = 1 - (1 - this._winVmaxF) * aT;
    const effW = effVmax - effVmin;

    // Build colormap LUT (256 entries)
    const cmSz = 256;
    const cmLut = new Uint8Array(cmSz * 3);
    const ns = stops.length;
    for (let i = 0; i < cmSz; i++) {
        const t = i / (cmSz - 1);
        const fi = t * (ns - 1);
        const lo = Math.floor(fi), hi = Math.min(lo + 1, ns - 1), f = fi - lo;
        cmLut[i * 3]     = Math.round(stops[lo][0] * (1 - f) + stops[hi][0] * f);
        cmLut[i * 3 + 1] = Math.round(stops[lo][1] * (1 - f) + stops[hi][1] * f);
        cmLut[i * 3 + 2] = Math.round(stops[lo][2] * (1 - f) + stops[hi][2] * f);
    }
    const cmAt = (frac) => {
        const idx = Math.max(0, Math.min(cmSz - 1, Math.round(frac * (cmSz - 1)))) * 3;
        return [cmLut[idx], cmLut[idx + 1], cmLut[idx + 2]];
    };

    const hasKDE = this._kdeSmooth && aT > 0;
    const labelH = CB_LABEL_H * aT;
    const histH = cssH - labelH;

    // Draw pixel columns
    for (let px = 0; px < cssW; px++) {
        const t = px / cssW;
        const inRange = t >= effVmin && t <= effVmax;
        const cmapT = effW > 0.001 ? (t - effVmin) / effW : 0.5;
        let r, g, b;
        if (inRange) {
            [r, g, b] = cmAt(cmapT);
        } else {
            const [cr, cg, cb] = cmAt(t);
            r = Math.round(cr + (60 - cr) * aT);
            g = Math.round(cg + (60 - cg) * aT);
            b = Math.round(cb + (60 - cb) * aT);
        }
        let colH;
        if (hasKDE) {
            const kdeH = Math.max(2, this._kdeAt(t) * (this._expandedH - CB_LABEL_H));
            colH = CB_COLLAPSED_H + (kdeH - CB_COLLAPSED_H) * aT;
        } else {
            colH = histH;
        }
        const y = cssH - colH;
        const alpha = inRange ? 1 : (1 - 0.65 * aT);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fillRect(px, y, 1, colH);
    }

    // Dashed guide lines + value labels in the label zone
    if (aT > 0.1) {
        ctx.globalAlpha = aT;
        if (vminFrac > 0.005 || vmaxFrac < 0.995) {
            ctx.strokeStyle = isDark ? 'rgba(255,255,100,0.3)' : 'rgba(180,100,0,0.3)';
            ctx.lineWidth = 0.75;
            ctx.setLineDash([2, 2]);
            [effVmin, effVmax].forEach(f => {
                if (f <= 0.003 || f >= 0.997) return;
                const x = Math.round(f * cssW) + 0.5;
                ctx.beginPath();
                ctx.moveTo(x, labelH);
                ctx.lineTo(x, cssH);
                ctx.stroke();
            });
            ctx.setLineDash([]);
        }
        // Value labels
        if (labelH > 4 && this._histData) {
            const { vmin: dmin, vmax: dmax } = this._histData;
            const dimColor = isDark ? 'rgba(255,255,255,0.35)' : 'rgba(0,0,0,0.3)';
            const brightColor = isDark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.7)';
            const hoverColor = isDark ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.9)';
            ctx.font = '9px monospace';
            const charW = ctx.measureText('0').width;
            const labelY = labelH - 3;
            const labels = [];
            labels.push({ x: 2, text: _cbFmt(dmin), align: 'start', color: dimColor, prio: 0 });
            labels.push({ x: cssW - 2, text: _cbFmt(dmax), align: 'end', color: dimColor, prio: 0 });
            if (effVmin > 0.01 && effVmin < 0.99) {
                labels.push({ x: Math.round(effVmin * cssW), text: _cbFmt(win.vmin), align: 'center', color: brightColor, prio: 1 });
            }
            if (effVmax > 0.01 && effVmax < 0.99) {
                labels.push({ x: Math.round(effVmax * cssW), text: _cbFmt(win.vmax), align: 'center', color: brightColor, prio: 1 });
            }
            if (this._hoverFrac >= 0 && this._hoverFrac <= 1) {
                const hoverVal = dmin + this._hoverFrac * (dmax - dmin);
                labels.push({ x: Math.round(this._hoverFrac * cssW), text: _cbFmt(hoverVal), align: 'center', color: hoverColor, prio: 2 });
            }
            labels.sort((a, b) => b.prio - a.prio);
            const drawn = [];
            for (const lbl of labels) {
                const tw = lbl.text.length * charW;
                let left, right;
                if (lbl.align === 'start') { left = lbl.x; right = lbl.x + tw; }
                else if (lbl.align === 'end') { left = lbl.x - tw; right = lbl.x; }
                else { left = lbl.x - tw / 2; right = lbl.x + tw / 2; }
                if (drawn.some(d => left < d.right + 4 && right > d.left - 4)) continue;
                ctx.fillStyle = lbl.color;
                ctx.textAlign = lbl.align;
                ctx.fillText(lbl.text, lbl.x, labelY);
                drawn.push({ left, right });
            }
            ctx.textAlign = 'start';
        }
        ctx.globalAlpha = 1;
    }

    ctx.restore();

    // Update triangle positions
    this.triVmin.style.left = (effVmin * 100) + '%';
    this.triVmax.style.left = (effVmax * 100) + '%';

    // Update hit-test positions
    this._hitVminX = Math.round(effVmin * cssW);
    this._hitVmaxX = Math.round(effVmax * cssW);
}
```

- [ ] **Step 5: Add `show`, `hide`, `destroy` methods**

```javascript
show() { this.wrap.style.display = ''; }
hide() { this.wrap.style.display = 'none'; }

destroy() {
    if (this._animating) this._animating = false;
    if (this._autoDismissTimer) clearTimeout(this._autoDismissTimer);
    if (this._hoverTimer) clearTimeout(this._hoverTimer);
    this._detachEvents();
    this.wrap.remove();
}
```

- [ ] **Step 6: Add value label update method**

```javascript
updateLabels() {
    if (this._expanded) {
        this.vminLabel.textContent = '';
        this.vmaxLabel.textContent = '';
    } else {
        const range = this.opts.getRange();
        this.vminLabel.textContent = _cbFmt(range.vmin);
        this.vmaxLabel.textContent = _cbFmt(range.vmax);
    }
}
```

- [ ] **Step 7: Run API tests**

Run: `uv run pytest tests/test_api.py -x -q`
Expected: all 93 pass (class is defined but not yet used)

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add ColorBar class — core rendering (morph + KDE)"
```

---

## Task 2: Add animation and histogram expansion

Add the expand/collapse animation loop and histogram fetch/KDE-build triggers.

**Files:**
- Modify: `src/arrayview/_viewer.html` — add methods to `ColorBar` class

- [ ] **Step 1: Add animation methods**

```javascript
startAnim(expand) {
    this._animTarget = expand ? 1 : 0;
    if (!this._animating) {
        this._animating = true;
        this._animStep();
    }
}

_animStep() {
    if (!this._animating) return;
    const diff = this._animTarget - this._animT;
    if (Math.abs(diff) < 0.003) {
        this._animT = this._animTarget;
    } else {
        this._animT += diff * 0.15;
    }
    this.draw();
    const morphDone = Math.abs(this._animTarget - this._animT) < 0.003;
    const winDone = Math.abs(this._winVminF - this._winVminTarget) < 0.002
                 && Math.abs(this._winVmaxF - this._winVmaxTarget) < 0.002;
    if (morphDone && winDone) {
        this._animating = false;
        return;
    }
    requestAnimationFrame(() => this._animStep());
}
```

- [ ] **Step 2: Add `expand` and `collapse` methods**

These are the public API for the 'd' key and auto-dismiss logic.

```javascript
async expand() {
    if (this._expanded) return;
    this._expanded = true;
    this.triZone.classList.add('expanded');
    this.updateLabels();
    // Fetch histogram and build KDE
    if (this.opts.fetchHistogram) {
        this._histFetching = true;
        try {
            this._histData = await this.opts.fetchHistogram();
            this._histVersion = Date.now();
        } catch (_) { this._histData = null; }
        this._histFetching = false;
    }
    this._buildKDE();
    this.startAnim(true);
    // Sync
    if (this._syncWith && !this._syncing) {
        this._syncWith._syncing = true;
        await this._syncWith.expand();
        this._syncWith._syncing = false;
    }
}

collapse() {
    if (!this._expanded || this._dragActive) return;
    this._expanded = false;
    this._hitVminX = null;
    this._hitVmaxX = null;
    this.triZone.classList.remove('expanded');
    this.startAnim(false);
    this.updateLabels();
    // Sync
    if (this._syncWith && !this._syncing) {
        this._syncWith._syncing = true;
        this._syncWith.collapse();
        this._syncWith._syncing = false;
    }
}

get expanded() { return this._expanded; }
```

- [ ] **Step 3: Add `refreshHistogram` for stale data re-fetch**

```javascript
async refreshHistogram() {
    if (!this._expanded || this._histFetching || !this.opts.fetchHistogram) return;
    this._histFetching = true;
    try {
        this._histData = await this.opts.fetchHistogram();
        this._histVersion = Date.now();
    } catch (_) { this._histData = null; }
    this._histFetching = false;
    this._buildKDE();
    this.draw();
}
```

- [ ] **Step 4: Add auto-dismiss support**

```javascript
startAutoDismiss(ms = 3000) {
    if (this._autoDismissTimer) clearTimeout(this._autoDismissTimer);
    this._autoDismissTimer = setTimeout(() => {
        this._autoDismissTimer = null;
        if (this._mouseOver) {
            this._autoDismissPending = true;
            return;
        }
        if (this._expanded && !this._dragActive) this.collapse();
    }, ms);
}

cancelAutoDismiss() {
    if (this._autoDismissTimer) { clearTimeout(this._autoDismissTimer); this._autoDismissTimer = null; }
    this._autoDismissPending = false;
}
```

- [ ] **Step 5: Add quantile preset cycling**

```javascript
static QUANTILE_PRESETS = [
    { label: 'full range', lo: 0, hi: 1 },
    { label: '0.1\u201399.9%', lo: 0.001, hi: 0.999 },
    { label: '1\u201399%', lo: 0.01, hi: 0.99 },
    { label: '5\u201395%', lo: 0.05, hi: 0.95 },
    { label: '10\u201390%', lo: 0.10, hi: 0.90 },
];

cycleQuantile() {
    if (!this._expanded || !this._histData) return null;
    this._quantileIdx = ((this._quantileIdx || 0) + 1) % ColorBar.QUANTILE_PRESETS.length;
    const preset = ColorBar.QUANTILE_PRESETS[this._quantileIdx];
    if (preset.lo === 0 && preset.hi === 1) {
        return { vmin: null, vmax: null, label: preset.label };
    }
    const { counts, edges, vmin: dmin, vmax: dmax } = this._histData;
    const total = counts.reduce((a, b) => a + b, 0);
    let cumLo = 0, cumHi = 0, qLo = dmin, qHi = dmax;
    for (let i = 0; i < counts.length; i++) {
        cumLo += counts[i];
        if (cumLo / total >= preset.lo) { qLo = edges[i]; break; }
    }
    for (let i = counts.length - 1; i >= 0; i--) {
        cumHi += counts[i];
        if (cumHi / total >= (1 - preset.hi)) { qHi = edges[i + 1]; break; }
    }
    // Sync
    if (this._syncWith && !this._syncing) {
        this._syncWith._syncing = true;
        this._syncWith._quantileIdx = this._quantileIdx;
        this._syncWith._syncing = false;
    }
    return { vmin: qLo, vmax: qHi, label: preset.label };
}
```

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/test_api.py -x -q`

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: ColorBar class — animation, histogram expansion, quantile cycling"
```

---

## Task 3: Add interaction handlers

Attach wheel, drag, hover, and triangle drag handlers to the ColorBar's own DOM elements.

**Files:**
- Modify: `src/arrayview/_viewer.html` — add methods to `ColorBar` class

- [ ] **Step 1: Add `_attachEvents` method**

Called at the end of the constructor (after `_buildDOM`). Sets up all interaction handlers on the instance's own elements.

```javascript
_attachEvents() {
    if (!this.interactive) return;

    // Wheel: zoom window symmetrically
    this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const win = this.opts.getWindow();
        if (win.vmin == null) return;
        const center = (win.vmin + win.vmax) / 2;
        const half = (win.vmax - win.vmin) / 2;
        const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
        const newHalf = half * factor;
        const vmin = center - newHalf;
        const vmax = center + newHalf;
        this.opts.setWindow(vmin, vmax);
        this.draw();
        if (this.opts.onWindowChange) this.opts.onWindowChange();
        this._syncInteraction('wheel', { vmin, vmax });
    }, { passive: false });

    // Mousedown on canvas: start pan drag (collapsed) or threshold drag (expanded)
    this.canvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        if (this._expanded && this._histData) {
            // Check if clicking near a threshold line
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const target = this._hitTest(x);
            if (target) {
                this._startClimDrag(target, e);
                return;
            }
        }
        // Collapsed: start pan drag
        if (!this._expanded) {
            this._dragX = e.clientX;
        }
    });

    // Double-click: reset to auto range
    this.canvas.addEventListener('dblclick', () => {
        this.opts.setWindow(null, null);
        this.draw();
        if (this.opts.onWindowChange) this.opts.onWindowChange();
        this._syncInteraction('reset');
    });

    // Triangle mousedown handlers
    this.triVmin.addEventListener('mousedown', (e) => {
        if (this._expanded && this._histData) this._startClimDrag('vmin', e);
    });
    this.triVmax.addEventListener('mousedown', (e) => {
        if (this._expanded && this._histData) this._startClimDrag('vmax', e);
    });

    // Wrap mouseenter/mouseleave for auto-dismiss
    this.wrap.addEventListener('mouseenter', () => { this._mouseOver = true; });
    this.wrap.addEventListener('mouseleave', () => {
        this._mouseOver = false;
        if (this._hoverFrac >= 0) { this._hoverFrac = -1; this.draw(); }
        if (this._autoDismissPending && !this._dragActive) {
            this._autoDismissPending = false;
            this.collapse();
        }
        if (this._expanded && !this._dragActive && !this._autoDismissTimer) {
            this._hoverTimer = setTimeout(() => this.collapse(), 200);
        }
    });

    // Global mousemove and mouseup for drag operations
    this._onMouseMove = (e) => this._handleMouseMove(e);
    this._onMouseUp = (e) => this._handleMouseUp(e);
    window.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mouseup', this._onMouseUp);
}

_detachEvents() {
    if (this._onMouseMove) window.removeEventListener('mousemove', this._onMouseMove);
    if (this._onMouseUp) window.removeEventListener('mouseup', this._onMouseUp);
}
```

- [ ] **Step 2: Add drag helper methods**

```javascript
_hitTest(x) {
    if (this._hitVminX != null && Math.abs(x - this._hitVminX) < CB_HIT_RADIUS) return 'vmin';
    if (this._hitVmaxX != null && Math.abs(x - this._hitVmaxX) < CB_HIT_RADIUS) return 'vmax';
    return null;
}

_startClimDrag(target, e) {
    e.preventDefault();
    e.stopPropagation();
    this._dragTarget = target;
    this._dragActive = true;
    this.canvas.style.cursor = 'ew-resize';
}

_handleMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const isOverCanvas = x >= 0 && x <= rect.width && e.clientY >= rect.top && e.clientY <= rect.bottom;

    if (this._dragTarget) {
        // Dragging threshold line
        const frac = Math.max(0, Math.min(1, x / rect.width));
        const { vmin: dmin, vmax: dmax } = this._histData;
        const val = dmin + frac * (dmax - dmin);
        const win = this.opts.getWindow();
        if (this._dragTarget === 'vmin') {
            this.opts.setWindow(Math.min(val, win.vmax - 1e-10), win.vmax);
        } else {
            this.opts.setWindow(win.vmin, Math.max(val, win.vmin + 1e-10));
        }
        this.draw();
        if (this.opts.onWindowChange) this.opts.onWindowChange();
        this._syncInteraction('drag', this.opts.getWindow());
    } else if (this._dragX != null) {
        // Panning (collapsed mode)
        const dx = e.clientX - this._dragX;
        this._dragX = e.clientX;
        const cbW = rect.width || 200;
        const win = this.opts.getWindow();
        const range = win.vmax - win.vmin;
        const shift = -(dx / cbW) * range;
        this.opts.setWindow(win.vmin + shift, win.vmax + shift);
        this.draw();
        if (this.opts.onWindowChange) this.opts.onWindowChange();
        this._syncInteraction('pan', this.opts.getWindow());
    } else if (isOverCanvas && this._expanded) {
        // Hover
        this._hoverFrac = Math.max(0, Math.min(1, x / rect.width));
        this.canvas.style.cursor = this._hitTest(x) ? 'ew-resize' : 'default';
        this.draw();
    } else if (!isOverCanvas && this._hoverFrac >= 0) {
        this._hoverFrac = -1;
        this.draw();
    }
}

_handleMouseUp(e) {
    if (this._dragTarget) {
        this._dragTarget = null;
        this._dragActive = false;
        this.canvas.style.cursor = 'default';
        if (this.opts.onWindowChange) this.opts.onWindowChange();
    }
    this._dragX = null;
}
```

- [ ] **Step 3: Add sync interaction method**

```javascript
_syncInteraction(type, data) {
    if (!this._syncWith || this._syncing) return;
    this._syncWith._syncing = true;
    switch (type) {
        case 'wheel':
        case 'drag':
        case 'pan':
            this._syncWith.opts.setWindow(data.vmin, data.vmax);
            this._syncWith.draw();
            if (this._syncWith.opts.onWindowChange) this._syncWith.opts.onWindowChange();
            break;
        case 'reset':
            this._syncWith.opts.setWindow(null, null);
            this._syncWith.draw();
            if (this._syncWith.opts.onWindowChange) this._syncWith.opts.onWindowChange();
            break;
    }
    this._syncWith._syncing = false;
}
```

- [ ] **Step 4: Add constructor call to `_attachEvents`**

At the end of the constructor, after `_buildDOM`:

```javascript
// ... end of constructor:
this._buildDOM(opts.container);
if (this.interactive) this._attachEvents();
```

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_api.py -x -q`

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: ColorBar class — interaction handlers (wheel, drag, hover, sync)"
```

---

## Task 4: Add colormap preview support

Add methods to show/hide colormap thumbnails inside the ColorBar island, replacing the canvas content temporarily.

**Files:**
- Modify: `src/arrayview/_viewer.html` — add methods to `ColorBar` class

- [ ] **Step 1: Add `showCmapPreview` and `hideCmapPreview` methods**

```javascript
showCmapPreview(stripElement) {
    // Move the colormap strip element into this island's colWrap
    this._cmapActive = true;
    this._cmapPrevParent = stripElement.parentElement;
    this._cmapPrevNextSibling = stripElement.nextSibling;
    this.colWrap.insertBefore(stripElement, this.canvas);
    this.canvas.style.display = 'none';
    this.triZone.style.display = 'none';
    // Sync
    if (this._syncWith && !this._syncing) {
        this._syncWith._syncing = true;
        this._syncWith.showCmapPreview(stripElement.cloneNode(true));
        this._syncWith._syncing = false;
    }
}

hideCmapPreview() {
    if (!this._cmapActive) return;
    this._cmapActive = false;
    // Restore strip to original position
    const strip = this.colWrap.querySelector('.colormap-strip, .cmap-strip');
    if (strip && this._cmapPrevParent) {
        if (this._cmapPrevNextSibling) {
            this._cmapPrevParent.insertBefore(strip, this._cmapPrevNextSibling);
        } else {
            this._cmapPrevParent.appendChild(strip);
        }
    } else if (strip) {
        strip.remove();
    }
    this._cmapPrevParent = null;
    this._cmapPrevNextSibling = null;
    this.canvas.style.display = '';
    this.triZone.style.display = '';
    this.draw();
    // Sync
    if (this._syncWith && !this._syncing) {
        this._syncWith._syncing = true;
        this._syncWith.hideCmapPreview();
        this._syncWith._syncing = false;
    }
}
```

- [ ] **Step 2: Run tests and commit**

Run: `uv run pytest tests/test_api.py -x -q`

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: ColorBar class — colormap preview support"
```

---

## Task 5: Add CSS for the new class-based elements

Add CSS rules for the new generic class names used by `ColorBar._buildDOM`. These mirror the existing `#slim-cb-wrap` styles but use class selectors so multiple instances work.

**Files:**
- Modify: `src/arrayview/_viewer.html` — add CSS after the existing `.cb-island` block (around line 146)

- [ ] **Step 1: Add CSS for `.cb-wrap` and children**

Insert after the existing `.cb-island` theme rules (around line 146):

```css
/* ── Generic ColorBar class styles ──────────────────────────── */
.cb-wrap {
    text-align: center; padding: 7px 16px; z-index: 2;
    display: flex; align-items: flex-end; gap: 8px; box-sizing: border-box;
    transition: opacity 0.3s ease;
}
.cb-col {
    flex: 1; min-width: 60px; display: flex; flex-direction: column; position: relative;
}
.cb-canvas {
    display: block; width: 100%; border-radius: 3px; border: none; cursor: default;
}
.cb-tri-zone {
    position: relative; height: 0; overflow: visible;
    transition: height 0.3s cubic-bezier(0.4,0,0.2,1);
}
.cb-tri-zone.expanded { height: 10px; }
.cb-tooltip {
    position: absolute; top: 0; left: 0; font-size: 10px;
    color: var(--fg); background: var(--bg);
    padding: 1px 4px; border-radius: 2px;
    pointer-events: none; white-space: nowrap; z-index: 3;
}
```

- [ ] **Step 2: Run tests and commit**

Run: `uv run pytest tests/test_api.py -x -q`

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: CSS for generic ColorBar class elements"
```

---

## Task 6: Migrate normal mode to `ColorBar`

Replace the existing normal-mode colorbar (global state + `drawSlimColorbar` + `_cbMorphDraw`) with a `ColorBar` instance.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Create `ColorBar` instance for normal mode**

At the point where the page initializes (after DOM is ready), create the primary colorbar. Find where `slimCbCanvas` is first referenced (around line 1696) and replace with:

```javascript
const primaryCb = new ColorBar({
    container: document.getElementById('slim-cb-wrap') || document.getElementById('canvas-viewport'),
    id: 'primary',
    orientation: 'horizontal',
    getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
    getRange: () => ({ vmin: currentVmin, vmax: currentVmax }),
    getWindow: () => ({ vmin: manualVmin ?? currentVmin, vmax: manualVmax ?? currentVmax }),
    setWindow: (lo, hi) => { manualVmin = lo; manualVmax = hi; },
    fetchHistogram: () => _fetchHistogram(),
    onWindowChange: () => { updateView(); triggerPreload(); },
});
```

**Important:** For this task, keep the existing `#slim-cb-wrap` static HTML but have the `ColorBar` instance reuse/adopt its elements instead of creating new ones. Add an `adoptDOM` option to the constructor that takes existing elements instead of creating them. This avoids changing the DOM structure for the initial migration — we can clean that up later.

Add to constructor:
```javascript
if (opts.adoptDOM) {
    this.wrap = opts.adoptDOM.wrap;
    this.canvas = opts.adoptDOM.canvas;
    this.vminLabel = opts.adoptDOM.vminLabel;
    this.vmaxLabel = opts.adoptDOM.vmaxLabel;
    this.triZone = opts.adoptDOM.triZone;
    this.triVmin = opts.adoptDOM.triVmin;
    this.triVmax = opts.adoptDOM.triVmax;
    this.colWrap = opts.adoptDOM.colWrap || this.canvas.parentElement;
    this.tooltip = opts.adoptDOM.tooltip;
} else {
    this._buildDOM(opts.container);
}
```

Then the normal mode creation becomes:
```javascript
const primaryCb = new ColorBar({
    id: 'primary',
    adoptDOM: {
        wrap: document.getElementById('slim-cb-wrap'),
        canvas: document.getElementById('slim-cb'),
        vminLabel: document.getElementById('slim-cb-vmin'),
        vmaxLabel: document.getElementById('slim-cb-vmax'),
        triZone: document.getElementById('slim-cb-tri-zone'),
        triVmin: document.getElementById('slim-cb-tri-vmin'),
        triVmax: document.getElementById('slim-cb-tri-vmax'),
        colWrap: document.getElementById('slim-cb-col'),
        tooltip: document.getElementById('slim-cb-tooltip'),
    },
    // ... callbacks as above
});
```

- [ ] **Step 2: Replace `drawSlimColorbar` calls with `primaryCb.draw()`**

Search for all calls to `drawSlimColorbar(cbMarkerFrac)` in the normal mode path and replace with `primaryCb.draw()` + `primaryCb.updateLabels()`. The function `drawSlimColorbar` still exists for now (compare mode uses it) but its normal-mode branch delegates to `primaryCb`.

- [ ] **Step 3: Replace 'd' key handler to use `primaryCb`**

In the keyboard handler, replace the normal-mode histogram expansion logic with:

```javascript
if (!primaryCb.expanded) {
    await primaryCb.expand();
    primaryCb.startAutoDismiss();
    showStatus('histogram');
} else {
    const q = primaryCb.cycleQuantile();
    if (q) {
        manualVmin = q.vmin;
        manualVmax = q.vmax;
        showStatus(q.label);
        updateView();
    }
}
```

- [ ] **Step 4: Remove old global state variables that are now in `primaryCb`**

Remove or mark as deprecated: `_cbAnimT`, `_cbAnimating`, `_cbAnimTarget`, `_cbWinVminF`, `_cbWinVmaxF`, `_cbHoverFrac`, `_kdeSmooth`, `_histAutoDismissTimer`, `_histAutoDismissPending`, `_cbMouseOver`, `_cbHoverTimer`.

Keep `_histData`, `_histDataVersion`, `_histFetching` as globals for now since multiple systems read them (Lebesgue, compare).

- [ ] **Step 5: Test normal mode thoroughly**

Run: `uv run pytest tests/test_api.py -x -q`
Run: `uv run python tests/visual_smoke.py`

Manual testing:
- Open a single array
- Verify collapsed colorbar looks identical
- Press 'd' — verify smooth KDE histogram expands
- Press 'd' again — verify quantile cycling
- Hover over histogram — verify value labels
- Drag triangles — verify vmin/vmax adjustment
- Press 'c' — verify colormap previewer appears in island
- Wheel on colorbar — verify zoom
- Drag colorbar — verify pan
- Double-click — verify reset

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: migrate normal mode colorbar to ColorBar class"
```

---

## Task 7: Migrate multi-view mode to `ColorBar`

Replace `drawMvColorbar` and the dynamically-created `#mv-cb-wrap` DOM with a `ColorBar` instance.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: In `enterMultiView`, create a `ColorBar` instance**

Replace the manual DOM creation of `#mv-cb-wrap` (lines 7748-7765) with:

```javascript
if (!window._mvColorBar) {
    const mvContainer = document.getElementById('multi-view-wrap');
    window._mvColorBar = new ColorBar({
        container: mvContainer,
        id: 'mv',
        getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
        getRange: () => {
            const v0 = mvViews.find(v => v.lastW);
            return v0 ? { vmin: v0.vmin, vmax: v0.vmax } : { vmin: 0, vmax: 1 };
        },
        getWindow: () => ({ vmin: manualVmin ?? currentVmin, vmax: manualVmax ?? currentVmax }),
        setWindow: (lo, hi) => { manualVmin = lo; manualVmax = hi; },
        fetchHistogram: () => _fetchHistogram(),
        onWindowChange: () => { for (const v of mvViews) mvRender(v); },
    });
}
```

- [ ] **Step 2: Replace `drawMvColorbar` with `_mvColorBar.draw()`**

All calls to `drawMvColorbar()` become `window._mvColorBar?.draw(); window._mvColorBar?.updateLabels();`

- [ ] **Step 3: Update `_redrawActiveColorbar`**

```javascript
function _redrawActiveColorbar() {
    if (multiViewActive || compareMvActive) {
        window._mvColorBar?.draw();
        window._mvColorBar?.updateLabels();
    } else {
        primaryCb.draw();
        primaryCb.updateLabels();
    }
}
```

- [ ] **Step 4: In `exitMultiView`, destroy the instance**

```javascript
if (window._mvColorBar) {
    window._mvColorBar.destroy();
    window._mvColorBar = null;
}
```

- [ ] **Step 5: Delete old `drawMvColorbar` function**

Remove the entire function (now ~40 lines).

- [ ] **Step 6: Update 'd' key handler for multiview**

The 'd' key handler should detect the active colorbar:

```javascript
const activeCb = multiViewActive ? window._mvColorBar : primaryCb;
if (!activeCb) return;
if (!activeCb.expanded) {
    await activeCb.expand();
    activeCb.startAutoDismiss();
    showStatus('histogram');
} else {
    const q = activeCb.cycleQuantile();
    // ... apply quantile
}
```

- [ ] **Step 7: Test multiview mode**

Run: `uv run pytest tests/test_api.py -x -q`

Manual testing:
- Press 'v' to enter multiview
- Verify collapsed colorbar appears below the 3 panes
- Press 'd' — smooth KDE histogram should expand (not bars!)
- Press 'd' again — quantile cycling
- Press 'c' — colormap previewer in island
- Wheel/drag/dblclick on colorbar
- Press 'v' to exit — colorbar destroyed cleanly

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: migrate multiview colorbar to ColorBar class"
```

---

## Task 8: Migrate compare mode shared colorbar

Compare mode (B key, without diff) uses a single shared colorbar. It already goes through `drawSlimColorbar` → `_cbMorphDraw`, so this migration is mostly about making it use `primaryCb` directly and ensuring histogram fetch merges data from both arrays.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Update `primaryCb`'s `fetchHistogram` to handle compare mode**

Modify the callback to detect compare mode:

```javascript
fetchHistogram: () => compareActive ? _fetchHistogramCompare() : _fetchHistogram(),
```

- [ ] **Step 2: Verify compare shared colorbar works with `primaryCb`**

The `drawSlimColorbar` function's compare path should already delegate to `primaryCb.draw()` from Task 6. Verify:
- Press 'b', load a second array
- Colorbar visible below panes
- Press 'd' — histogram shows merged data from both arrays
- Quantile cycling works

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: compare shared colorbar uses primaryCb with merged histogram"
```

---

## Task 9: Migrate diff mode to synced `ColorBar` instances

When 'X' is pressed in compare mode, left and right panes should each get their own full-featured `ColorBar` (synced), and the diff center pane gets an independent `ColorBar`.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Create left/right synced `ColorBar` instances on diff entry**

In the diff mode entry code (where `diffMode` is set), create:

```javascript
const leftContainer = document.querySelector('#compare-left-pane .compare-pane-cb-island')
    || document.getElementById('compare-left-pane');
const rightContainer = document.querySelector('#compare-right-pane .compare-pane-cb-island')
    || document.getElementById('compare-right-pane');

window._diffLeftCb = new ColorBar({
    container: leftContainer,
    id: 'diff-left',
    getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
    getRange: () => ({ vmin: currentVmin, vmax: currentVmax }),
    getWindow: () => ({ vmin: manualVmin ?? currentVmin, vmax: manualVmax ?? currentVmax }),
    setWindow: (lo, hi) => { manualVmin = lo; manualVmax = hi; },
    fetchHistogram: () => _fetchHistogram(),
    onWindowChange: () => { compareRender(); },
});

window._diffRightCb = new ColorBar({
    container: rightContainer,
    id: 'diff-right',
    syncWith: window._diffLeftCb,
    getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
    getRange: () => ({ vmin: currentVmin, vmax: currentVmax }),
    getWindow: () => ({ vmin: manualVmin ?? currentVmin, vmax: manualVmax ?? currentVmax }),
    setWindow: (lo, hi) => { manualVmin = lo; manualVmax = hi; },
    fetchHistogram: () => _fetchHistogram(),
    onWindowChange: () => { compareRender(); },
});
// Wire sync both ways
window._diffLeftCb._syncWith = window._diffRightCb;
```

- [ ] **Step 2: Create independent diff center `ColorBar`**

```javascript
const diffContainer = document.querySelector('#compare-diff-pane .compare-pane-cb-island')
    || document.getElementById('compare-diff-pane');

window._diffCenterCb = new ColorBar({
    container: diffContainer,
    id: 'diff-center',
    getStops: () => _diffCenterColormapStops || COLORMAP_GRADIENT_STOPS['RdBu_r'] || [],
    getRange: () => ({ vmin: _lastDiffVmin ?? -1, vmax: _lastDiffVmax ?? 1 }),
    getWindow: () => ({ vmin: _lastDiffVmin ?? -1, vmax: _lastDiffVmax ?? 1 }),
    setWindow: (lo, hi) => { /* diff vmin/vmax are auto-computed, no manual override for now */ },
    fetchHistogram: null, // diff histogram can be added later
    onWindowChange: () => { compareRender(); },
});
```

- [ ] **Step 3: Hide `primaryCb` during diff mode**

```javascript
primaryCb.hide();
```

- [ ] **Step 4: On diff exit, destroy diff colorbars and restore `primaryCb`**

```javascript
if (window._diffLeftCb) { window._diffLeftCb.destroy(); window._diffLeftCb = null; }
if (window._diffRightCb) { window._diffRightCb.destroy(); window._diffRightCb = null; }
if (window._diffCenterCb) { window._diffCenterCb.destroy(); window._diffCenterCb = null; }
primaryCb.show();
```

- [ ] **Step 5: Replace `drawComparePaneCb` and `drawDiffPaneCb` calls**

In diff mode, replace calls to `drawComparePaneCb(idx)` and `drawDiffPaneCb(vmin, vmax)` with:
```javascript
window._diffLeftCb?.draw();
window._diffLeftCb?.updateLabels();
window._diffRightCb?.draw();
window._diffRightCb?.updateLabels();
window._diffCenterCb?.draw();
window._diffCenterCb?.updateLabels();
```

- [ ] **Step 6: Update 'd' key handler for diff mode**

When pressing 'd' in diff mode, determine which colorbar is "active" based on cursor position or a focus heuristic:
```javascript
if (diffMode > 0) {
    // For now, expand all three; left/right are synced so expanding one expands both
    const cbs = [window._diffLeftCb, window._diffCenterCb].filter(Boolean);
    for (const cb of cbs) {
        if (!cb.expanded) { await cb.expand(); cb.startAutoDismiss(); }
        else { const q = cb.cycleQuantile(); /* apply */ }
    }
    showStatus('histogram');
    return;
}
```

- [ ] **Step 7: Test diff mode**

Manual testing:
- Press 'b' to enter compare, load second array
- Press 'x' to enter diff mode
- Verify left and right panes each have a colorbar beneath them
- Verify diff center has its own colorbar
- Press 'd' — all three show smooth histograms
- Drag triangle on left — right mirrors the change
- Wheel on right — left mirrors
- Press 'x' again to exit diff — colorbars destroyed, shared colorbar restored

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: diff mode uses synced ColorBar instances for left/right + independent center"
```

---

## Task 10: Migrate qMRI horizontal colorbars

Each qMRI pane gets its own `ColorBar` instance with independent vmin/vmax.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: In `enterQmri`, create `ColorBar` instances per view**

Replace the manual colorbar DOM creation in the qMRI view setup with:

```javascript
for (const v of qmriViews) {
    v._colorBar = new ColorBar({
        container: v.paneEl,  // the .qv-pane element
        id: 'qv-' + v.qmriIdx,
        getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
        getRange: () => ({ vmin: v.vmin, vmax: v.vmax }),
        getWindow: () => ({ vmin: v.vmin, vmax: v.vmax }),
        setWindow: (lo, hi) => { v.lockedVmin = lo; v.lockedVmax = hi; },
        fetchHistogram: async () => {
            // Fetch histogram for this specific qMRI view
            const qIndices = [...indices];
            qIndices[qmriDim] = v.qmriIdx;
            const params = new URLSearchParams({
                dim_x, dim_y,
                indices: qIndices.join(','),
                complex_mode: complexMode,
                bins: 64,
            });
            const resp = await fetch(`/histogram/${sid}?${params}`);
            return resp.ok ? resp.json() : null;
        },
        onWindowChange: () => { qvRender(v); },
    });
}
```

- [ ] **Step 2: Replace `drawQvSlimCb(v)` calls with `v._colorBar.draw()`**

- [ ] **Step 3: In `exitQmri`, destroy instances**

```javascript
for (const v of qmriViews) {
    if (v._colorBar) { v._colorBar.destroy(); v._colorBar = null; }
}
```

- [ ] **Step 4: Delete `drawQvSlimCb` function**

- [ ] **Step 5: Update 'd' key handler for qMRI**

```javascript
if (qmriActive) {
    for (const v of qmriViews) {
        if (v._colorBar && !v._colorBar.expanded) {
            await v._colorBar.expand();
            v._colorBar.startAutoDismiss();
        } else if (v._colorBar) {
            const q = v._colorBar.cycleQuantile();
            if (q) { v.lockedVmin = q.vmin; v.lockedVmax = q.vmax; qvRender(v); }
        }
    }
    showStatus('histogram');
    return;
}
```

- [ ] **Step 6: Test qMRI mode**

Manual testing:
- Open array with qMRI dimensions, press 'q'
- Verify each parameter pane has its own horizontal colorbar
- Press 'd' — all panes show smooth KDE histograms
- Interactions work per-pane

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: qMRI mode uses ColorBar instances per parameter view"
```

---

## Task 11: Migrate qMRI mosaic vertical colorbars

The mosaic mode has vertical colorbars. Add vertical rendering to the `draw` method, then instantiate `ColorBar` with `orientation: 'vertical'`.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add vertical rendering to `ColorBar.draw()`**

The current `draw` method renders horizontal pixel columns. For vertical, rotate the logic: iterate over rows instead of columns, with the colormap running bottom-to-top.

At the start of `draw()`, after computing all the shared state (effVmin, effVmax, cmLut, etc.), branch:

```javascript
if (this.orientation === 'vertical') {
    this._drawVertical(ctx, cssW, cssH, aT, effVmin, effVmax, effW, cmAt, labelH);
} else {
    this._drawHorizontal(ctx, cssW, cssH, aT, effVmin, effVmax, effW, cmAt, labelH);
}
```

Extract the current pixel-column loop into `_drawHorizontal`. Add `_drawVertical`:

```javascript
_drawVertical(ctx, cssW, cssH, aT, effVmin, effVmax, effW, cmAt, labelH) {
    const hasKDE = this._kdeSmooth && aT > 0;
    for (let py = 0; py < cssH; py++) {
        const t = 1 - py / cssH;  // bottom=0, top=1 (reversed)
        const inRange = t >= effVmin && t <= effVmax;
        const cmapT = effW > 0.001 ? (t - effVmin) / effW : 0.5;
        let r, g, b;
        if (inRange) {
            [r, g, b] = cmAt(cmapT);
        } else {
            const [cr, cg, cb] = cmAt(t);
            r = Math.round(cr + (60 - cr) * aT);
            g = Math.round(cg + (60 - cg) * aT);
            b = Math.round(cb + (60 - cb) * aT);
        }
        let rowW;
        if (hasKDE) {
            const kdeW = Math.max(2, this._kdeAt(t) * (cssW - 2));
            rowW = CB_COLLAPSED_H + (kdeW - CB_COLLAPSED_H) * aT;
        } else {
            rowW = cssW;
        }
        const alpha = inRange ? 1 : (1 - 0.65 * aT);
        ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
        ctx.fillRect(0, py, rowW, 1);
    }
}
```

- [ ] **Step 2: In mosaic layout rebuild, create `ColorBar` instances per view**

Replace `drawQmriMosaicVCb(v)` with:

```javascript
v._mosaicCb = new ColorBar({
    container: v._mosaicVcbIsland,
    id: 'mosaic-' + v.qmriIdx,
    orientation: 'vertical',
    getStops: () => colormap_idx === -1 ? customGradientStops : COLORMAP_GRADIENT_STOPS[COLORMAPS[colormap_idx]],
    getRange: () => ({ vmin: v.vmin, vmax: v.vmax }),
    getWindow: () => ({ vmin: v.vmin, vmax: v.vmax }),
    setWindow: (lo, hi) => { v.lockedVmin = lo; v.lockedVmax = hi; },
    fetchHistogram: () => _fetchQmriMosaicHist(v),
    onWindowChange: () => { qvRender(v); },
});
```

- [ ] **Step 3: Delete `drawQmriMosaicVCb` function**

- [ ] **Step 4: Test mosaic mode**

Manual testing:
- Enter qMRI, press 'z' for mosaic
- Verify vertical colorbars appear next to each parameter row
- Press 'd' — smooth KDE histograms (rotated, growing rightward from the gradient)
- Exit mosaic — colorbars destroyed

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: qMRI mosaic uses vertical ColorBar instances with smooth KDE"
```

---

## Task 12: Clean up dead code

Remove all the old drawing functions and global state that are no longer used.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Delete old functions**

Remove these functions entirely:
- `_cbMorphDraw` (line ~2935, ~150 lines) — replaced by `ColorBar.draw()`
- `_drawHistogramBarsOnColorbar` (line ~3182, ~70 lines) — no longer used
- `_drawClimLines` (line ~3243, ~60 lines) — integrated into `ColorBar.draw()`
- `_highlightHistBin` (line ~3377, ~50 lines) — moved into class or removed
- `_initCbHistInteraction` IIFE (line ~3506, ~180 lines) — replaced by `ColorBar._attachEvents()`
- `drawMvColorbar` (already removed in Task 7)
- `drawComparePaneCb` (line ~2487, ~60 lines) — if all compare panes use ColorBar
- `drawAllComparePaneCbs` (line ~2616, ~10 lines)
- `drawDiffPaneCb` (line ~2573, ~40 lines)
- `drawQvSlimCb` (already removed in Task 10)
- `drawQmriMosaicVCb` (already removed in Task 11)
- `_buildKDE` global function (line ~2870, ~30 lines) — now a class method
- `_kdeAt` global function (line ~2900, ~6 lines)
- `_cbStartAnim` (line ~2923, ~4 lines)
- `_cbAnimStep` (line ~2928, ~20 lines)
- `_fetchQmriMosaicHist` (line ~9139, ~30 lines) — inlined into ColorBar callback

- [ ] **Step 2: Delete old global state variables**

Remove these declarations:
```
_cbAnimT, _cbAnimating, _cbAnimTarget
_cbWinVminF, _cbWinVmaxF, _cbWinVminTarget, _cbWinVmaxTarget
_cbHoverFrac, _cbMouseOver, _cbHoverTimer
_cbDragActive, _cbDragX, _dragTarget
_histAutoDismissTimer, _histAutoDismissPending
_histVminX, _histVmaxX
_kdeSmooth
_cbExpandedH
```

Keep (still used by Lebesgue and other systems):
```
_histData, _histDataVersion, _histFetching
_cbExpanded (redirect to primaryCb.expanded)
lebesgueMode (global toggle, not per-colorbar)
```

- [ ] **Step 3: Simplify `drawSlimColorbar`**

This function was the main router. After migration, it should be minimal — just handle positioning the island and delegating to `primaryCb.draw()`. Eventually it may be removable entirely, but for now keep it as the positioning/layout function.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/test_api.py -x -q`
Run: `uv run python tests/visual_smoke.py`
Review: `smoke_output/` directory for visual regression

- [ ] **Step 5: Manual regression test all modes**

Test each mode per the modes-consistency checklist:
- [ ] Normal view — colorbar, 'd' histogram, 'c' cmap preview, wheel/drag/dblclick
- [ ] Multi-view (v) — shared colorbar, 'd', 'c'
- [ ] Compare (b) — shared colorbar, merged histogram
- [ ] Diff (x in compare) — synced left/right colorbars, independent center
- [ ] Registration (r in compare) — blend bar unchanged (not migrated)
- [ ] qMRI (q) — per-parameter colorbars with histograms
- [ ] qMRI mosaic (z in qMRI) — vertical colorbars with smooth histograms

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: remove old colorbar drawing functions and global state"
```

---

## Task 13: Final polish — Lebesgue mode integration

The Lebesgue mode ('w' key) highlights pixels matching the hovered histogram bin. This needs to work with the `ColorBar` class.

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Add Lebesgue hover callback to `ColorBar`**

Add an optional `onBinHover` callback:
```javascript
// In _handleMouseMove, when hovering over expanded histogram:
if (this._expanded && this._hoverFrac >= 0 && this.opts.onBinHover) {
    const nBins = this._histData?.counts?.length || 64;
    const binIdx = Math.floor(this._hoverFrac * nBins);
    this.opts.onBinHover(binIdx, this._hoverFrac);
}
```

- [ ] **Step 2: Wire up Lebesgue overlay in `primaryCb`**

```javascript
onBinHover: (binIdx, frac) => {
    if (!lebesgueMode) return;
    _drawLebesgueHighlight(binIdx);
    _highlightHistBin(binIdx);
},
```

- [ ] **Step 3: Test Lebesgue mode**

- Press 'w' — histogram expands, hovering bins highlights matching pixels
- Works in normal mode and multiview

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: Lebesgue mode integrated with ColorBar class"
```
