# Smooth Immersive Crossfade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discrete 3-phase immersive enter/exit animation with a zoom-driven crossfade that makes continuous trackpad zoom seamless.

**Architecture:** A single stateless function `_applyImmersiveCrossfade()` computes progress `p` from `userZoom` and controls chrome opacity + layout switch. Called after every `ModeRegistry.scaleAll()`. Replaces `_enterImmersive()` and `_exitImmersive()`. The `=`/`-`/`0` keys now just set target zoom and animate — the crossfade applies automatically each frame.

**Tech Stack:** Vanilla JavaScript, CSS

---

### Task 1: Add `_applyImmersiveCrossfade()` function

**Files:**
- Modify: `src/arrayview/_viewer.html` — insert new function near `_enterImmersive` (~line 14389)

This is the core of the feature. The function is called after every zoom update and drives all chrome transitions based on the current zoom level.

- [ ] **Step 1: Add the crossfade controller function**

Insert before `_enterImmersive` (around line 14389), after `_calcAutoFitZoom`:

```javascript
        // ── Zoom-driven immersive crossfade ────────────────────────────
        // Computes progress p ∈ [0,1] from userZoom and drives chrome
        // opacity + fullscreen-mode layout switch.  Stateless: called on
        // every zoom update, no timers or animation ids.
        let _crossfadePrev = 0; // track previous p to avoid redundant DOM writes
        function _applyImmersiveCrossfade() {
            if (_immersiveAnimating) return; // legacy animations still running
            const w = lastImgW || canvas.width;
            const h = lastImgH || canvas.height;
            if (!w || !h) return;
            if (!_shouldEnterImmersive()) {
                // Width-binding: immersive not possible.  Ensure normal layout.
                if (_fullscreenActive) {
                    _fullscreenActive = false;
                    document.body.classList.remove('fullscreen-mode');
                    _infoDragPos = null; _cbDragPos = null; _islandDragPos = null;
                    const wrap = document.getElementById('slim-cb-wrap');
                    if (wrap) { wrap.classList.remove('compact-overlay'); wrap.style.width = ''; }
                    const infoEl = document.getElementById('info');
                    if (infoEl) { infoEl.style.top = ''; infoEl.style.left = ''; infoEl.style.transform = ''; }
                    const cbWrap = document.getElementById('slim-cb-wrap');
                    if (cbWrap) { cbWrap.style.top = ''; cbWrap.style.bottom = ''; cbWrap.style.right = ''; }
                    if (_miniMap) { _miniMap.style.top = ''; _miniMap.style.left = ''; _miniMap.style.right = ''; }
                    const wrapper = document.getElementById('wrapper');
                    if (wrapper) wrapper.style.paddingTop = '';
                    _reconcileCbVisibility({ animPhase: 'exit-3' });
                }
                // Clear any residual inline opacity
                _clearCrossfadeOpacity();
                _crossfadePrev = 0;
                return;
            }

            const immFit = _calcAutoFitZoom(w, h, true);
            const normFit = _normalFitZoom;
            if (immFit <= normFit) return; // no room to crossfade
            const p = Math.max(0, Math.min(1, (userZoom - normFit) / (immFit - normFit)));

            // Skip DOM writes if p hasn't changed meaningfully
            if (Math.abs(p - _crossfadePrev) < 0.005 && (p > 0 && p < 1)) return;
            _crossfadePrev = p;

            const fadeOut = Math.max(0, 1 - p * 2);       // 1→0 over first half
            const fadeIn  = Math.max(0, (p - 0.5) * 2);   // 0→1 over second half

            // Layout switch at midpoint
            const shouldBeFullscreen = p >= 0.5;
            if (shouldBeFullscreen !== _fullscreenActive) {
                _fullscreenActive = shouldBeFullscreen;
                document.body.classList.toggle('fullscreen-mode', shouldBeFullscreen);
                if (shouldBeFullscreen) {
                    _reconcileCbVisibility({ animPhase: 'enter-2' });
                    // Reconcile compare center island
                    if (compareActive && (diffMode > 0 || registrationMode || _wipeActive || _flickerActive || _checkerActive)) {
                        const centerIsland = document.getElementById('compare-center-island');
                        if (centerIsland) {
                            const mode = diffMode > 0 ? diffMode : (_wipeActive ? 5 : (_flickerActive ? 6 : (_checkerActive ? 7 : 4)));
                            centerIsland.innerHTML = _centerModeIndicatorHTML(mode);
                        }
                    }
                } else {
                    _infoDragPos = null; _cbDragPos = null; _islandDragPos = null;
                    const wrap = document.getElementById('slim-cb-wrap');
                    if (wrap) { wrap.classList.remove('compact-overlay'); wrap.style.width = ''; }
                    const infoEl = document.getElementById('info');
                    if (infoEl) { infoEl.style.top = ''; infoEl.style.left = ''; infoEl.style.transform = ''; }
                    const cbWrap = document.getElementById('slim-cb-wrap');
                    if (cbWrap) { cbWrap.style.top = ''; cbWrap.style.bottom = ''; cbWrap.style.right = ''; }
                    if (_miniMap) { _miniMap.style.top = ''; _miniMap.style.left = ''; _miniMap.style.right = ''; }
                    const wrapper = document.getElementById('wrapper');
                    if (wrapper) wrapper.style.paddingTop = '';
                    _reconcileCbVisibility({ animPhase: 'exit-3' });
                }
            }

            // Apply opacities
            const titleEl = document.getElementById('array-name');
            const infoEl  = document.getElementById('info');
            const cbWrap  = document.getElementById('slim-cb-wrap');

            if (titleEl) titleEl.style.opacity = (p === 0) ? '' : fadeOut;
            if (infoEl)  infoEl.style.opacity  = (p === 0) ? '' : (shouldBeFullscreen ? fadeIn : fadeOut);
            if (cbWrap)  cbWrap.style.opacity   = (p === 0) ? '' : (shouldBeFullscreen ? fadeIn : fadeOut);

            // Compare/multiview/qMRI chrome
            if (compareActive) {
                document.querySelectorAll('.compare-title').forEach(el => {
                    el.style.opacity = (p === 0) ? '' : fadeOut;
                });
                document.querySelectorAll('.compare-pane-cb-island').forEach(el => {
                    el.style.opacity = (p === 0) ? '' : (shouldBeFullscreen ? fadeIn : fadeOut);
                });
            }
            const mvCb = multiViewActive ? document.getElementById('mv-cb-wrap') : null;
            if (mvCb) mvCb.style.opacity = (p === 0) ? '' : (shouldBeFullscreen ? fadeIn : fadeOut);
            const qmriCbs = qmriActive ? document.querySelectorAll('#qmri-view-wrap .qv-cb-island') : [];
            qmriCbs.forEach(el => {
                el.style.opacity = (p === 0) ? '' : (shouldBeFullscreen ? fadeIn : fadeOut);
            });

            // Position floating chrome when in fullscreen
            if (shouldBeFullscreen) _positionFullscreenChrome();

            // Update _fitZoom so status bar shows correct state
            if (p >= 1) {
                _fitZoom = immFit;
                if (!_fullscreenActive) _normalFitZoom = userZoom;
            }
        }

        function _clearCrossfadeOpacity() {
            const titleEl = document.getElementById('array-name');
            const infoEl  = document.getElementById('info');
            const cbWrap  = document.getElementById('slim-cb-wrap');
            for (const el of [titleEl, infoEl, cbWrap]) {
                if (el) { el.style.opacity = ''; el.style.transition = ''; }
            }
            document.querySelectorAll('.compare-title, .compare-pane-cb-island').forEach(el => {
                el.style.opacity = ''; el.style.transition = '';
            });
            const mvCb = document.getElementById('mv-cb-wrap');
            if (mvCb) { mvCb.style.opacity = ''; mvCb.style.transition = ''; }
            document.querySelectorAll('#qmri-view-wrap .qv-cb-island').forEach(el => {
                el.style.opacity = ''; el.style.transition = '';
            });
        }
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add _applyImmersiveCrossfade — zoom-driven immersive transition

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire crossfade into zoom update paths

**Files:**
- Modify: `src/arrayview/_viewer.html` — `ModeRegistry.scaleAll()` (~line 1972), `_scaleAllWithAnim` (~line 2565)

The crossfade must run after every zoom-driven scale update. There are two places to wire it:

- [ ] **Step 1: Call crossfade after `ModeRegistry.scaleAll()`**

Find `ModeRegistry.scaleAll()` at line ~1972:

```javascript
            scaleAll() {
                const scaler = this.getScaler();
                if (scaler) return scaler();
                if (lastImgW && lastImgH) return scaleCanvas(lastImgW, lastImgH);
                if (canvas.width && canvas.height) return scaleCanvas(canvas.width, canvas.height);
                _reconcileCbVisibility();
            },
```

Replace with:

```javascript
            scaleAll() {
                const scaler = this.getScaler();
                if (scaler) scaler();
                else if (lastImgW && lastImgH) scaleCanvas(lastImgW, lastImgH);
                else if (canvas.width && canvas.height) scaleCanvas(canvas.width, canvas.height);
                else _reconcileCbVisibility();
                _applyImmersiveCrossfade();
            },
```

Note: changed `return scaler()` to just `scaler()` so the crossfade always runs after.

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: wire _applyImmersiveCrossfade into ModeRegistry.scaleAll()

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewrite `=` key (zoom.in) to use crossfade

**Files:**
- Modify: `src/arrayview/_viewer.html` — `zoom.in` command (~line 8445)

Replace the `_enterImmersive()` call path with a simple animated zoom to immersive fit level. The crossfade handles the layout transition automatically.

- [ ] **Step 1: Rewrite the zoom.in command**

Find the `zoom.in` command at line ~8445:

```javascript
            'zoom.in': {
                title: 'Zoom in / enter immersive',
                when: [],
                run: (ctx, e) => {
                    if (_immersiveAnimating) return; // ignore zoom keys during immersive transition
                    if (!_fullscreenActive && !_zoomAdjustedByUser &&
                        _shouldEnterImmersive()) {
                        // First =: enter immersive mode via animated transition
                        _enterImmersive();
                        showStatus('immersive');
                        saveState();
                    } else {
                        // Already immersive or can't enter: zoom in
                        userZoom = userZoom * 1.1;
                        if (!_snapZoomToFit()) {
                            _zoomAdjustedByUser = true;
                        }
                        _scaleAllWithAnim();
                        showStatus(_fullscreenActive && userZoom === _fitZoom ? 'immersive' : userZoom === _fitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                        saveState();
                    }
                },
            },
```

Replace with:

```javascript
            'zoom.in': {
                title: 'Zoom in / enter immersive',
                when: [],
                run: (ctx, e) => {
                    if (!_fullscreenActive && !_zoomAdjustedByUser &&
                        _shouldEnterImmersive()) {
                        // Animate zoom to immersive fit — crossfade handles layout
                        const w = lastImgW || canvas.width;
                        const h = lastImgH || canvas.height;
                        userZoom = _calcAutoFitZoom(w, h, true);
                        _zoomAdjustedByUser = true;
                        _scaleAllWithAnim();
                        showStatus('immersive');
                        saveState();
                    } else {
                        userZoom = userZoom * 1.1;
                        if (!_snapZoomToFit()) {
                            _zoomAdjustedByUser = true;
                        }
                        _scaleAllWithAnim();
                        showStatus(_fullscreenActive && userZoom === _fitZoom ? 'immersive' : userZoom === _fitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                        saveState();
                    }
                },
            },
```

Key changes:
- Removed `_immersiveAnimating` gate (no longer exists as a concept)
- Instead of `_enterImmersive()`, sets `userZoom` to immersive fit and calls `_scaleAllWithAnim()`. Each animation frame calls `ModeRegistry.scaleAll()` → `_applyImmersiveCrossfade()`, so chrome fades naturally as zoom sweeps through the zone.

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: zoom.in uses animated zoom + crossfade instead of _enterImmersive

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Rewrite `-` key (zoom.out) to use crossfade

**Files:**
- Modify: `src/arrayview/_viewer.html` — `zoom.out` command (~line 8468)

Remove the `_exitImmersive()` call. As zoom decreases, the crossfade reverses automatically.

- [ ] **Step 1: Rewrite the zoom.out command**

Find the `zoom.out` command at line ~8468:

```javascript
            'zoom.out': {
                title: 'Zoom out / exit immersive',
                when: [],
                run: (ctx, e) => {
                    if (_immersiveAnimating) return; // ignore zoom keys during immersive transition
                    userZoom = Math.max(userZoom / 1.1, _normalFitZoom);
                    if (!_snapZoomToFit()) {
                        _zoomAdjustedByUser = true;
                        // Auto-exit immersive mode when zooming back to fit
                        if (_fullscreenActive && userZoom <= _fitZoom) {
                            _exitImmersive();
                            showStatus('zoom: fit');
                            saveState();
                            return; // _exitImmersive handles layout; skip _scaleAllWithAnim
                        }
                    }
                    _scaleAllWithAnim();
                    showStatus(userZoom === _fitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                    saveState();
                },
            },
```

Replace with:

```javascript
            'zoom.out': {
                title: 'Zoom out / exit immersive',
                when: [],
                run: (ctx, e) => {
                    userZoom = Math.max(userZoom / 1.1, _normalFitZoom);
                    if (!_snapZoomToFit()) {
                        _zoomAdjustedByUser = true;
                    }
                    _scaleAllWithAnim();
                    showStatus(userZoom <= _normalFitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                    saveState();
                },
            },
```

Key changes:
- Removed `_immersiveAnimating` gate
- Removed `_exitImmersive()` call — crossfade handles exit automatically as zoom decreases
- Simplified status: no special immersive exit status since the transition is gradual

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: zoom.out uses crossfade instead of _exitImmersive

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Simplify `0` key (zoom.reset) and scroll zoom handler

**Files:**
- Modify: `src/arrayview/_viewer.html` — `zoom.reset` command (~line 8572), scroll zoom handler (~line 14012)

Both currently have `_immersiveAnimating` cancellation logic and `_exitImmersive()` calls. Simplify them since the crossfade is stateless.

- [ ] **Step 1: Simplify zoom.reset command**

Find the `zoom.reset` command at line ~8572. Replace the entire `run` function body (everything inside `run: (ctx, e) => { ... }`) with:

```javascript
                run: (ctx, e) => {
                    // If there's buffered input, '0' is part of slice jump; otherwise it's zoom reset
                    if (sliceJumpBuffer.length > 0) {
                        sliceJumpBuffer += e.key;
                        if (sliceJumpTimeout) clearTimeout(sliceJumpTimeout);
                        sliceJumpTimeout = setTimeout(() => { sliceJumpBuffer = ''; showStatus(''); }, 2000);
                        showStatus(`jump to slice: ${sliceJumpBuffer}_`);
                    } else {
                        const now = performance.now();
                        const isDoubleTap = _fullscreenActive && (now - _lastZeroPress < 500);
                        _lastZeroPress = now;
                        if (isDoubleTap) {
                            // Double-tap 0: full reset to normal fit
                            userZoom = 1.0;
                            _zoomAdjustedByUser = false;
                            mainPan.reset();
                            mosaicPan.reset();
                            ModeRegistry.scaleAllAutoFit();
                            showStatus('zoom: fit');
                        } else if (_fullscreenActive) {
                            // First tap in immersive: reset to immersive-fit level
                            const w = lastImgW || canvas.width;
                            const h = lastImgH || canvas.height;
                            userZoom = _calcAutoFitZoom(w, h, true);
                            _fitZoom = userZoom;
                            _zoomAdjustedByUser = false;
                            mainPan.reset();
                            ModeRegistry.scaleAll();
                            showStatus('immersive · press 0 again to exit');
                        } else {
                            // Not in immersive: reset to normal fit
                            userZoom = 1.0;
                            _zoomAdjustedByUser = false;
                            mainPan.reset();
                            mosaicPan.reset();
                            ModeRegistry.scaleAllAutoFit();
                            showStatus('zoom: fit');
                        }
                        saveState();
                    }
                },
```

Key change: removed the `_immersiveAnimating` cancellation block (~20 lines of opacity cleanup, `cancelAnimationFrame`, `_reconcileUI`). The crossfade is stateless and doesn't need cancellation.

- [ ] **Step 2: Simplify scroll zoom handler**

Find the scroll zoom handler at line ~14012:

```javascript
            // Ctrl / Cmd + scroll → zoom (clamped exponential, applied instantly)
            if ((e.ctrlKey || e.metaKey) && !_immersiveAnimating) {
                const clamped = Math.max(-10, Math.min(10, e.deltaY));
                const factor = Math.pow(2, -clamped * _zoomSensitivity);
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

Replace with:

```javascript
            // Ctrl / Cmd + scroll → zoom (clamped exponential, applied instantly)
            if (e.ctrlKey || e.metaKey) {
                const clamped = Math.max(-10, Math.min(10, e.deltaY));
                const factor = Math.pow(2, -clamped * _zoomSensitivity);
                userZoom = Math.max(_normalFitZoom, Math.min(10.0, userZoom * factor));
                if (!_snapZoomToFit()) {
                    _zoomAdjustedByUser = true;
                }
                ModeRegistry.scaleAll();
                saveState();
                return;
            }
```

Key changes:
- Removed `!_immersiveAnimating` gate (no longer needed)
- Removed `_exitImmersive()` call — crossfade handles exit automatically
- Removed special immersive exit branch — `ModeRegistry.scaleAll()` now calls crossfade which handles everything

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: simplify zoom.reset and scroll handler — remove _immersiveAnimating/_exitImmersive

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CSS — keep dimbar width constant in immersive mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — CSS section (~line 901)

- [ ] **Step 1: Update fullscreen-mode #info CSS**

Find at line ~901:

```css
        body.fullscreen-mode #info {
            position: fixed; z-index: 3; left: 50%; transform: translateX(-50%);
            margin: 0 !important; padding: 6px 16px;
            background-color: rgba(30, 30, 30, 0.8);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            pointer-events: none;
        }
```

Replace with:

```css
        body.fullscreen-mode #info {
            position: fixed; z-index: 3; left: 50%; transform: translateX(-50%);
            margin: 0 !important;
            background-color: rgba(30, 30, 30, 0.8);
            backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            pointer-events: none;
        }
```

Change: removed `padding: 6px 16px` override so the dimbar keeps its normal padding and width. The normal-mode CSS already sets appropriate padding. The `border-radius: 14px` is already present in both normal (`body:not(.fullscreen-mode) #info` at line 1030) and fullscreen CSS, so the dimbar already has rounded corners in both modes — the visual difference is just position and backdrop.

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: dimbar keeps constant width in immersive mode — remove padding override

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Remove `_enterImmersive`, `_exitImmersive`, and `_immersiveAnimating`

**Files:**
- Modify: `src/arrayview/_viewer.html` — remove functions (~line 14389-14571), remove flag (~line 14387)

Now that nothing calls these functions, remove them. Also remove `_immersiveAnimating` and any remaining references.

- [ ] **Step 1: Search for remaining references**

Before deleting, search for all references to confirm nothing still uses them:

```bash
grep -n '_enterImmersive\|_exitImmersive\|_immersiveAnimating' src/arrayview/_viewer.html
```

The only remaining references should be:
- The function definitions themselves
- The `_immersiveAnimating` declaration
- The `if (_immersiveAnimating) return;` guard inside `_applyImmersiveCrossfade` (added in Task 1 — remove this guard too since there are no legacy animations left)

- [ ] **Step 2: Remove `_immersiveAnimating` declaration**

Find and remove (line ~14387):
```javascript
        let _immersiveAnimating = false;
```

- [ ] **Step 3: Remove `_enterImmersive` function**

Delete the entire `_enterImmersive` function (lines ~14389-14455, starting with `function _enterImmersive()` through its closing `}`).

- [ ] **Step 4: Remove `_exitImmersive` function**

Delete the entire `_exitImmersive` function (lines ~14457-14571, starting with `function _exitImmersive()` through its closing `}`).

- [ ] **Step 5: Remove the guard in `_applyImmersiveCrossfade`**

In the `_applyImmersiveCrossfade` function (added in Task 1), remove the line:
```javascript
            if (_immersiveAnimating) return; // legacy animations still running
```

- [ ] **Step 6: Remove `_animOpacity` if no longer used**

Search for `_animOpacity` — if it has no remaining callers after removing `_enterImmersive`/`_exitImmersive`, delete it too.

```bash
grep -n '_animOpacity' src/arrayview/_viewer.html
```

If the only references are the function definition and calls within `_enterImmersive`/`_exitImmersive` (which are now deleted), remove the function definition.

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: remove _enterImmersive, _exitImmersive, _immersiveAnimating

Replaced by stateless zoom-driven _applyImmersiveCrossfade.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update Safari gesture and touch pinch handlers

**Files:**
- Modify: `src/arrayview/_viewer.html` — gesture handler (~line 15674), existing `_snapZoomToFit` references

The Safari gesture and touch pinch handlers need to work with the crossfade. Currently they call `ModeRegistry.scaleAll()` directly which now includes the crossfade. But we need to ensure `_zoomAdjustedByUser` is set so the crossfade doesn't fight with auto-fit.

- [ ] **Step 1: Verify gesture handler already works**

Read the Safari gesture handler added in earlier task. It already does:
```javascript
        function _onGestureChange(e) {
            e.preventDefault();
            userZoom = Math.max(_normalFitZoom, Math.min(10.0, _gestureStartZoom * e.scale));
            _zoomAdjustedByUser = true;
            ModeRegistry.scaleAll();
        }
```

This already calls `ModeRegistry.scaleAll()` which now includes `_applyImmersiveCrossfade()`. It sets `_zoomAdjustedByUser = true`. No changes needed.

- [ ] **Step 2: Verify touch pinch handler already works**

The touch pinch handler at ~line 15661:
```javascript
        function _onTouchMove(e) {
            if (e.touches.length === 2 && _pinchDist0 !== null) {
                e.preventDefault();
                const d = _pinchDist(e.touches);
                userZoom = Math.max(0.1, Math.min(10, _pinchZoom0 * (d / _pinchDist0)));
                _zoomAdjustedByUser = true;
                ModeRegistry.scaleAll();
            }
        }
```

Also already calls `ModeRegistry.scaleAll()`. No changes needed.

- [ ] **Step 3: Commit (no-op — verification only)**

No code changes needed. This task is verification that existing handlers work with the new crossfade wiring.

---

### Task 9: Run visual smoke tests and manual verification

**Files:**
- Run: `tests/visual_smoke.py`

- [ ] **Step 1: Run visual smoke tests**

```bash
uv run python tests/visual_smoke.py
```

Expected: All tests pass. The zoom tests (04, 05, 36, 38, 63c, 71) use keyboard `+`/`-`/`0` which now go through `_scaleAllWithAnim` → `ModeRegistry.scaleAll()` → `_applyImmersiveCrossfade()`. The crossfade should be a no-op at normal zoom levels (p=0).

- [ ] **Step 2: Manual verification checklist**

Open arrayview with a test image and verify:
1. **Cmd+scroll zoom in**: canvas grows, chrome smoothly crossfades to immersive
2. **Cmd+scroll zoom out**: chrome smoothly crossfades back to normal
3. **`=` key**: animated zoom to immersive with smooth crossfade
4. **`-` key from immersive**: animated zoom out with smooth reverse crossfade
5. **`0` key in immersive**: first tap resets to immersive fit, double-tap exits to normal
6. **Rapid direction change**: zoom in past threshold, immediately zoom out — no glitches
7. **Compare mode**: crossfade works with compare titles and pane colorbars
8. **Window resize during crossfade zone**: chrome repositions correctly

- [ ] **Step 3: Commit any fixes if needed**
