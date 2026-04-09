# Instant Exponential Zoom — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sluggish animated zoom with instant clamped-exponential zoom, add Safari GestureEvent support.

**Architecture:** Four targeted edits in `_viewer.html` — replace zoom formula + drop animation in wheel handler, add Safari gesture handlers, fix touch pinch dampening, update MIP wheel zoom. No new files.

**Tech Stack:** Vanilla JavaScript, WheelEvent, GestureEvent (Safari), Touch API

---

### Task 1: Replace ctrl/cmd+scroll zoom formula and drop animation

**Files:**
- Modify: `src/arrayview/_viewer.html:14006-14026`

- [ ] **Step 1: Replace zoom formula and remove animation call**

Replace lines 14006–14026:
```javascript
            // Ctrl / Cmd + scroll → zoom
            if ((e.ctrlKey || e.metaKey) && !_immersiveAnimating) {
                // Use deltaY magnitude to scale zoom — small trackpad pinch
                // gestures get small zoom steps, big scroll wheel clicks get bigger ones
                const absDelta = Math.min(Math.abs(e.deltaY), 50);
                const zoomFactor = 1 + absDelta * 0.004; // ~1.02 for small pinch, ~1.2 for big scroll
                userZoom = e.deltaY > 0
                    ? Math.max(userZoom / zoomFactor, _normalFitZoom)
                    : Math.min(userZoom * zoomFactor, 10.0);
                if (!_snapZoomToFit()) {
                    _zoomAdjustedByUser = true;
                    // Auto-exit immersive on scroll zoom out to fit level
                    if (_fullscreenActive && userZoom <= _fitZoom) {
                        _exitImmersive();
                        showStatus('zoom: fit');
                        saveState();
                        return;
                    }
                }
                _scaleAllWithAnim();
                saveState();
                return;
            }
```

With:
```javascript
            // Ctrl / Cmd + scroll → zoom (clamped exponential, applied instantly)
            if ((e.ctrlKey || e.metaKey) && !_immersiveAnimating) {
                const clamped = Math.max(-10, Math.min(10, e.deltaY));
                const factor = Math.pow(2, -clamped * 0.1);
                userZoom = Math.max(_normalFitZoom, Math.min(10.0, userZoom * factor));
                if (!_snapZoomToFit()) {
                    _zoomAdjustedByUser = true;
                    if (_fullscreenActive && userZoom <= _fitZoom) {
                        _exitImmersive();
                        showStatus('zoom: fit');
                        saveState();
                        return;
                    }
                }
                ModeRegistry.scaleAll();
                saveState();
                return;
            }
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: instant exponential zoom for ctrl/cmd+scroll — drop 150ms animation"
```

---

### Task 2: Add Safari GestureEvent handlers for pinch-to-zoom

**Files:**
- Modify: `src/arrayview/_viewer.html` — insert after the touch pinch section (~line 15678)

- [ ] **Step 1: Add gesture event handlers**

Insert after the `[canvas, ...compareCanvases].forEach` touch listener block (after line 15678):

```javascript
        // ── 9b. Safari pinch-to-zoom (GestureEvent) ─────────────────────
        let _gestureStartZoom = null;
        function _onGestureStart(e) {
            e.preventDefault();
            _gestureStartZoom = userZoom;
        }
        function _onGestureChange(e) {
            e.preventDefault();
            // e.scale is cumulative from gesturestart — multiply against cached start
            userZoom = Math.max(_normalFitZoom, Math.min(10.0, _gestureStartZoom * e.scale));
            _zoomAdjustedByUser = true;
            ModeRegistry.scaleAll();
        }
        function _onGestureEnd(e) {
            e.preventDefault();
            _gestureStartZoom = null;
            saveState();
        }
        [canvas, ...compareCanvases].forEach(el => {
            el.addEventListener('gesturestart',  _onGestureStart,  { passive: false });
            el.addEventListener('gesturechange', _onGestureChange, { passive: false });
            el.addEventListener('gestureend',    _onGestureEnd,    { passive: false });
        });
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add Safari GestureEvent handlers for trackpad pinch-to-zoom"
```

---

### Task 3: Fix touch pinch dampening

**Files:**
- Modify: `src/arrayview/_viewer.html:15666`

- [ ] **Step 1: Replace dampened pinch with direct ratio**

Replace line 15666:
```javascript
                userZoom = Math.max(0.1, Math.min(10, _pinchZoom0 * Math.pow(d / _pinchDist0, 0.35)));
```

With:
```javascript
                userZoom = Math.max(0.1, Math.min(10, _pinchZoom0 * (d / _pinchDist0)));
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: remove harsh 0.35 dampening from touch pinch zoom — use direct ratio"
```

---

### Task 4: Update MIP mode wheel zoom to clamped exponential

**Files:**
- Modify: `src/arrayview/_viewer.html:11748-11752`

- [ ] **Step 1: Replace MIP wheel zoom formula**

Replace lines 11748–11752:
```javascript
        function _mipOnWheel(e) {
            e.preventDefault(); e.stopPropagation();
            _mipZoom = Math.max(0.5, Math.min(10.0, _mipZoom + e.deltaY * 0.005));
            _mipRender();
        }
```

With:
```javascript
        function _mipOnWheel(e) {
            e.preventDefault(); e.stopPropagation();
            const clamped = Math.max(-10, Math.min(10, e.deltaY));
            _mipZoom = Math.max(0.5, Math.min(10.0, _mipZoom * Math.pow(2, -clamped * 0.1)));
            _mipRender();
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: MIP mode wheel zoom uses clamped exponential for consistency"
```

---

### Task 5: Run visual smoke tests

**Files:**
- Run: `tests/visual_smoke.py`

- [ ] **Step 1: Run visual smoke tests to verify no regressions**

```bash
python tests/visual_smoke.py
```

Expected: All existing tests pass. The zoom tests (04, 05, 36, 38, 63c, 71) use keyboard +/-/0 which still go through `_scaleAllWithAnim`, so they should be unaffected.

- [ ] **Step 2: Commit any fixes if needed**
