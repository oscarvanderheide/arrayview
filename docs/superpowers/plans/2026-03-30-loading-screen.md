# Loading Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the jarring dimbar-before-image appearance with a centered pulsing logo that fades out as all UI elements fade in together.

**Architecture:** Add a `av-loading` body class that hides wrapper contents via opacity. Populate `#loading-overlay` with a large logo + array name on init. On first frame arrival, cross-fade from loading to ready state. All changes in `_viewer.html` only.

**Tech Stack:** CSS transitions, vanilla JS

---

### Task 1: Add CSS for loading state

**Files:**
- Modify: `src/arrayview/_viewer.html:47-54` (wrapper/info CSS area)
- Modify: `src/arrayview/_viewer.html:661-664` (#loading-overlay CSS)

- [ ] **Step 1: Add body.av-loading CSS rules**

After the existing `#wrapper.mode-fade` rule (line 54), add:

```css
body.av-loading #array-name,
body.av-loading #info,
body.av-loading #canvas-wrap,
body.av-loading #slim-cb-wrap { opacity: 0; }
#array-name, #info, #canvas-wrap, #slim-cb-wrap {
    transition: opacity 0.3s ease;
}
```

- [ ] **Step 2: Add loading overlay content styles**

After the existing `#loading-overlay` block (line 664), add:

```css
#loading-overlay {
    transition: opacity 0.3s ease;
}
#loading-overlay .av-load-logo {
    width: 64px; height: 64px;
}
#loading-overlay .av-load-name {
    margin-top: 14px; font-size: 11px; color: var(--muted);
    letter-spacing: 0.1em; text-transform: uppercase;
}
```

Note: The `#loading-overlay` already has `display:flex; flex-direction:column; align-items:center; justify-content:center; position:fixed; inset:0; z-index:5;` — no need to repeat those.

- [ ] **Step 3: Commit**

```
git add src/arrayview/_viewer.html
git commit -m "style: add CSS for loading screen fade-in transition"
```

---

### Task 2: Populate loading overlay and add body class on init

**Files:**
- Modify: `src/arrayview/_viewer.html:5854-5885` (init function)

- [ ] **Step 1: Add av-loading class and populate overlay at start of init()**

At the very beginning of `init()` (line 5854), before the `if (!sid)` check, add code that:
1. Adds `av-loading` class to body
2. Clones the logo SVG from `#av-logo-svg`, sets its class to `av-load-logo av-logo-loading` (to trigger the pulse animation), and inserts it into `#loading-overlay`

```javascript
async function init() {
    // ── Loading screen: show pulsing logo while fetching data ──
    const _loadingOverlay = document.getElementById('loading-overlay');
    if (sid) {
        document.body.classList.add('av-loading');
        const loadLogo = document.getElementById('av-logo-svg').cloneNode(true);
        loadLogo.removeAttribute('id');
        loadLogo.classList.add('av-load-logo', 'av-logo-loading');
        _loadingOverlay.appendChild(loadLogo);
    }

    if (!sid) { hideLoadingOverlay(); drawPlasmaDemo(); return; }
```

- [ ] **Step 2: Add array name to overlay after metadata arrives**

After line 5880 (`document.getElementById('array-name-text').textContent = data.name;`), add:

```javascript
const loadNameEl = document.createElement('div');
loadNameEl.className = 'av-load-name';
loadNameEl.textContent = data.name;
_loadingOverlay.appendChild(loadNameEl);
```

- [ ] **Step 3: Commit**

```
git add src/arrayview/_viewer.html
git commit -m "feat: populate loading overlay with pulsing logo and array name"
```

---

### Task 3: Replace instant hide with fade transition on first frame

**Files:**
- Modify: `src/arrayview/_viewer.html:5443-5447` (first frame arrival in ws.onmessage)
- Modify: `src/arrayview/_viewer.html:5788-5791` (hideLoadingOverlay function)

- [ ] **Step 1: Update the first-frame-arrival block**

Replace the existing block at lines 5443-5447:

```javascript
if (document.getElementById('loading-overlay').style.display !== 'none') {
    document.getElementById('loading-overlay').style.display = 'none';
    document.getElementById('canvas-wrap').style.display = '';
    if (_isWelcomeScreen) document.getElementById('welcome-hint').classList.add('visible');
}
```

With:

```javascript
if (document.getElementById('loading-overlay').style.display !== 'none') {
    // Show canvas-wrap immediately (still invisible due to av-loading opacity:0)
    document.getElementById('canvas-wrap').style.display = '';
    if (_isWelcomeScreen) document.getElementById('welcome-hint').classList.add('visible');
    // Fade out overlay, fade in wrapper contents
    const ol = document.getElementById('loading-overlay');
    ol.style.opacity = '0';
    document.body.classList.remove('av-loading');
    ol.addEventListener('transitionend', () => { ol.style.display = 'none'; }, { once: true });
}
```

- [ ] **Step 2: Update hideLoadingOverlay for non-animated paths**

The `hideLoadingOverlay()` function (line 5788) is called from welcome/plasma paths and multi-view entry. These don't need the fade — keep them instant but also clear the body class:

```javascript
function hideLoadingOverlay() {
    document.body.classList.remove('av-loading');
    const ol = document.getElementById('loading-overlay');
    ol.style.opacity = '';
    ol.style.display = 'none';
    document.getElementById('empty-hint').classList.remove('visible');
}
```

- [ ] **Step 3: Commit**

```
git add src/arrayview/_viewer.html
git commit -m "feat: fade-in transition from loading screen to viewer"
```

---

### Task 4: Manual smoke test

- [ ] **Step 1: Test native pywebview loading**

Launch arrayview with a reasonably sized array to observe the loading sequence:

```python
import numpy as np
import arrayview
arrayview.view(np.random.rand(10, 512, 512))
```

Verify:
- Centered pulsing logo appears during loading
- Array name appears below logo after metadata loads
- All UI elements (dimbar, canvas, colorbar) fade in together — no jumping
- Logo animation stops when content appears

- [ ] **Step 2: Test welcome screen (no sid)**

Open arrayview without data to verify the welcome/plasma screen still works:

```python
import arrayview
arrayview.view()
```

Verify: plasma demo appears immediately, no loading screen flash.

- [ ] **Step 3: Commit any fixes if needed**
