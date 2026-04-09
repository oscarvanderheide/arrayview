# Smooth Immersive Crossfade v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the discrete 3-phase immersive animation with a zoom-driven crossfade where the pane grows continuously, chrome collapses/expands physically, and the slice always fills the pane.

**Architecture:** A global `_crossfadeCollapseP` variable (0→1 during first half of crossfade) tells all 6 scale functions how much to reduce their bottom reserves. A `_applyImmersiveCrossfade()` function runs before every scale, progressively collapsing chrome heights and controlling opacity. Layout class toggle at p=0.5 when chrome is invisible and already collapsed.

**Tech Stack:** Vanilla JavaScript, CSS

**Spec:** `docs/superpowers/specs/2026-04-09-smooth-immersive-crossfade-v2-design.md`

---

### Task 1: Add `_crossfadeCollapseP` and update all 6 scale functions

**Files:**
- Modify: `src/arrayview/_viewer.html` — variable declaration area (~line 1835), `scaleCanvas` (~line 2532), `compareScaleCanvases` (~line 2603), `compareQmriScaleAllCanvases` (~line 12298), `compareMvScaleAllCanvases` (~line 12733), `qvScaleAllCanvases` (~line 13041), `mvScaleAllCanvases` (~line 13733)

This task adds the global variable and modifies each scale function to interpolate its bottom reserve. No visual change yet — the variable stays at 0 until the crossfade function (Task 2) sets it.

- [ ] **Step 1: Add the global variable**

Find `let _zoomSensitivity = 0.02;` (around line 1835). Insert after it:

```javascript
        let _crossfadeCollapseP = 0; // 0→1 during first half of immersive crossfade; scale functions use this to interpolate bottom reserves
```

- [ ] **Step 2: Update `scaleCanvas`**

Find in `scaleCanvas` (~line 2533):
```javascript
            const bottomReserve = _fullscreenActive ? 0 : 80; // fullscreen: overlaid; normal: colorbar + eggs + gaps
            const immersivePad = 12; // breathing room on each side in immersive mode
            const maxW = Math.max(100, window.innerWidth - (_fullscreenActive ? immersivePad * 2 : 80));
            const maxH = Math.max(50, window.innerHeight - uiReserveV() - bottomReserve - (_fullscreenActive ? immersivePad * 2 : 0));
```

Replace with:
```javascript
            const bottomReserve = _fullscreenActive ? 0 : 80 * (1 - _crossfadeCollapseP);
            const immersivePad = 12;
            const sidePad = _fullscreenActive ? immersivePad * 2 : 80 * (1 - _crossfadeCollapseP);
            const maxW = Math.max(100, window.innerWidth - sidePad);
            const maxH = Math.max(50, window.innerHeight - uiReserveV() - bottomReserve - (_fullscreenActive ? immersivePad * 2 : 0));
```

- [ ] **Step 3: Update `compareScaleCanvases`**

Find in `compareScaleCanvases` (~line 2621–2650):
```javascript
            const titleReserve = _fullscreenActive ? 0 : 28;
            const panePadX = _fullscreenActive ? 0 : 16;
```

Replace with:
```javascript
            const titleReserve = _fullscreenActive ? 0 : 28 * (1 - _crossfadeCollapseP);
            const panePadX = _fullscreenActive ? 0 : 16 * (1 - _crossfadeCollapseP);
```

Find (~line 2627–2628):
```javascript
            const cbExtra = (_anyCenter && !_fullscreenActive) ? (10 + CB_COLLAPSED_H + 4) : 0;
            const panePadY = (_fullscreenActive ? 0 : 12) + cbExtra;
```

Replace with:
```javascript
            const cbExtra = (_anyCenter && !_fullscreenActive) ? (10 + CB_COLLAPSED_H + 4) * (1 - _crossfadeCollapseP) : 0;
            const panePadY = (_fullscreenActive ? 0 : 12 * (1 - _crossfadeCollapseP)) + cbExtra;
```

Find (~line 2648–2650):
```javascript
            const uiReserveH = _fullscreenActive ? 0 : (uiReserveV() + 40);
            const totalW = Math.max(100, window.innerWidth - (_fullscreenActive ? immersivePad * 2 : 80));
            const totalH = Math.max(100, window.innerHeight - uiReserveH - (_fullscreenActive ? immersivePad * 2 : 0));
```

Replace with:
```javascript
            const uiReserveH = _fullscreenActive ? 0 : (uiReserveV() + 40 * (1 - _crossfadeCollapseP));
            const totalW = Math.max(100, window.innerWidth - (_fullscreenActive ? immersivePad * 2 : 80 * (1 - _crossfadeCollapseP)));
            const totalH = Math.max(100, window.innerHeight - uiReserveH - (_fullscreenActive ? immersivePad * 2 : 0));
```

- [ ] **Step 4: Update `compareQmriScaleAllCanvases`**

Find (~line 12314–12318):
```javascript
            const totalH = window.innerHeight - uiReserveV()
                         - labelH
                         - cbRows * cbH
                         - (rows - 1) * rowGap
                         - bottomPad;
```

Replace with:
```javascript
            const totalH = window.innerHeight - uiReserveV()
                         - labelH * (1 - _crossfadeCollapseP)
                         - cbRows * cbH * (1 - _crossfadeCollapseP)
                         - (rows - 1) * rowGap
                         - bottomPad * (1 - _crossfadeCollapseP);
```

- [ ] **Step 5: Update `compareMvScaleAllCanvases`**

Find (~line 12747–12751):
```javascript
            const totalH = window.innerHeight - uiReserveV()
                         - labelH
                         - cbRows * cbH
                         - (nRows - 1) * rowGap
                         - bottomPad;
```

Replace with:
```javascript
            const totalH = window.innerHeight - uiReserveV()
                         - labelH * (1 - _crossfadeCollapseP)
                         - cbRows * cbH * (1 - _crossfadeCollapseP)
                         - (nRows - 1) * rowGap
                         - bottomPad * (1 - _crossfadeCollapseP);
```

- [ ] **Step 6: Update `qvScaleAllCanvases`**

Find (~line 13056–13057):
```javascript
                const totalH = window.innerHeight - uiReserveV()
                             - (n - 1) * rowGap - bottomPad;
```

Replace with:
```javascript
                const totalH = window.innerHeight - uiReserveV()
                             - (n - 1) * rowGap - bottomPad * (1 - _crossfadeCollapseP);
```

- [ ] **Step 7: Update `mvScaleAllCanvases`**

Find (~line 13738–13739):
```javascript
            const cbH = cbWrap ? cbWrap.offsetHeight : 30;
            const uiReserveH = uiReserveV() + cbH + labelH;
```

Replace with:
```javascript
            const cbH = cbWrap ? cbWrap.offsetHeight : 30;
            const uiReserveH = uiReserveV() + (cbH + labelH) * (1 - _crossfadeCollapseP);
```

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add _crossfadeCollapseP — all 6 scale functions interpolate bottom reserves

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add `_applyImmersiveCrossfade()` and wire into `ModeRegistry.scaleAll()`

**Files:**
- Modify: `src/arrayview/_viewer.html` — insert function before `_enterImmersive` (~line 14387), update `ModeRegistry.scaleAll()` (~line 1972)

- [ ] **Step 1: Add the crossfade function**

Insert BEFORE `let _immersiveAnimating = false;` (~line 14387):

```javascript
        // ── Zoom-driven immersive crossfade ────────────────────────────
        // Progressively collapses chrome and grows the pane as zoom increases
        // from normalFit to immersiveFit.  The slice always fills the pane.
        let _crossfadePrev = -1;
        let _crossfadeTitleH = 0;  // cached natural heights
        let _crossfadeInfoH = 0;
        let _crossfadeCbH = 0;
        function _applyImmersiveCrossfade() {
            if (_immersiveAnimating) return;
            const w = lastImgW || canvas.width;
            const h = lastImgH || canvas.height;
            if (!w || !h) return;
            if (!_shouldEnterImmersive()) {
                if (_crossfadePrev > 0) _crossfadeCleanup();
                _crossfadeCollapseP = 0;
                _crossfadePrev = -1;
                return;
            }

            const immFit = _calcAutoFitZoom(w, h, true);
            const normFit = _normalFitZoom;
            if (immFit <= normFit) { _crossfadeCollapseP = 0; return; }
            const p = Math.max(0, Math.min(1, (userZoom - normFit) / (immFit - normFit)));

            if (Math.abs(p - _crossfadePrev) < 0.003 && p > 0 && p < 1) return;
            _crossfadePrev = p;

            // collapseP: 0→1 during first half (chrome collapses + fades out)
            const collapseP = Math.min(1, p * 2);
            // fadeIn: 0→1 during second half (chrome fades in at immersive positions)
            const fadeIn = Math.max(0, (p - 0.5) * 2);

            // Set global for scale functions
            _crossfadeCollapseP = collapseP;

            // Cache natural heights when crossfade starts
            if (p > 0 && _crossfadeTitleH === 0) {
                const t = document.getElementById('array-name');
                const i = document.getElementById('info');
                const c = document.getElementById('slim-cb-wrap');
                _crossfadeTitleH = t ? t.offsetHeight : 30;
                _crossfadeInfoH = i ? i.offsetHeight : 30;
                _crossfadeCbH = c ? c.offsetHeight : 30;
            }

            // ── Collapse chrome heights (first half) ──────────────────
            const titleEl = document.getElementById('array-name');
            const infoEl = document.getElementById('info');
            const cbWrap = document.getElementById('slim-cb-wrap');

            if (p === 0) {
                // Fully normal: clear all inline styles
                if (titleEl) { titleEl.style.maxHeight = ''; titleEl.style.overflow = ''; titleEl.style.opacity = ''; titleEl.style.margin = ''; titleEl.style.padding = ''; }
                if (infoEl) { infoEl.style.maxHeight = ''; infoEl.style.overflow = ''; infoEl.style.opacity = ''; }
                if (cbWrap) { cbWrap.style.maxHeight = ''; cbWrap.style.overflow = ''; cbWrap.style.opacity = ''; }
                _crossfadeTitleH = 0; _crossfadeInfoH = 0; _crossfadeCbH = 0;
            } else if (!_fullscreenActive) {
                // Crossfading in normal layout: collapse heights + fade
                const remainH = 1 - collapseP;
                if (titleEl) {
                    titleEl.style.maxHeight = (_crossfadeTitleH * remainH) + 'px';
                    titleEl.style.overflow = 'hidden';
                    titleEl.style.opacity = remainH;
                }
                if (infoEl) {
                    infoEl.style.maxHeight = (_crossfadeInfoH * remainH) + 'px';
                    infoEl.style.overflow = 'hidden';
                    infoEl.style.opacity = remainH;
                }
                if (cbWrap) {
                    cbWrap.style.maxHeight = (_crossfadeCbH * remainH) + 'px';
                    cbWrap.style.overflow = 'hidden';
                    cbWrap.style.opacity = remainH;
                }
            }

            // ── Layout switch at p=0.5 (chrome invisible + collapsed) ─
            const shouldBeFullscreen = p >= 0.5;
            if (shouldBeFullscreen !== _fullscreenActive) {
                if (shouldBeFullscreen) {
                    // Enter: chrome already collapsed via inline styles above
                    _fullscreenActive = true;
                    document.body.classList.add('fullscreen-mode');
                    _reconcileCbVisibility({ animPhase: 'enter-2' });
                    if (compareActive && (diffMode > 0 || registrationMode || _wipeActive || _flickerActive || _checkerActive)) {
                        const centerIsland = document.getElementById('compare-center-island');
                        if (centerIsland) {
                            const mode = diffMode > 0 ? diffMode : (_wipeActive ? 5 : (_flickerActive ? 6 : (_checkerActive ? 7 : 4)));
                            centerIsland.innerHTML = _centerModeIndicatorHTML(mode);
                        }
                    }
                    // Clear inline collapse styles — fullscreen-mode CSS handles it now
                    if (titleEl) { titleEl.style.maxHeight = ''; titleEl.style.overflow = ''; titleEl.style.margin = ''; titleEl.style.padding = ''; }
                    if (infoEl) { infoEl.style.maxHeight = ''; infoEl.style.overflow = ''; }
                    if (cbWrap) { cbWrap.style.maxHeight = ''; cbWrap.style.overflow = ''; }
                } else {
                    // Exit: restore normal layout, re-apply collapse inline styles
                    _fullscreenActive = false;
                    document.body.classList.remove('fullscreen-mode');
                    _infoDragPos = null; _cbDragPos = null; _islandDragPos = null;
                    const wrap = document.getElementById('slim-cb-wrap');
                    if (wrap) { wrap.classList.remove('compact-overlay'); wrap.style.width = ''; }
                    if (infoEl) { infoEl.style.top = ''; infoEl.style.left = ''; infoEl.style.transform = ''; }
                    if (cbWrap) { cbWrap.style.top = ''; cbWrap.style.bottom = ''; cbWrap.style.right = ''; }
                    if (_miniMap) { _miniMap.style.top = ''; _miniMap.style.left = ''; _miniMap.style.right = ''; }
                    const wrapper = document.getElementById('wrapper');
                    if (wrapper) wrapper.style.paddingTop = '';
                    _reconcileCbVisibility({ animPhase: 'exit-3' });
                    // Re-apply collapse for the shrinking second half → first half
                    const remainH = 1 - collapseP;
                    if (titleEl) { titleEl.style.maxHeight = (_crossfadeTitleH * remainH) + 'px'; titleEl.style.overflow = 'hidden'; }
                    if (infoEl) { infoEl.style.maxHeight = (_crossfadeInfoH * remainH) + 'px'; infoEl.style.overflow = 'hidden'; }
                    if (cbWrap) { cbWrap.style.maxHeight = (_crossfadeCbH * remainH) + 'px'; cbWrap.style.overflow = 'hidden'; }
                }
            }

            // ── Opacity for second half (fade in at immersive positions) ─
            if (_fullscreenActive) {
                if (titleEl) titleEl.style.opacity = '0'; // title stays hidden in immersive
                if (infoEl) infoEl.style.opacity = fadeIn || '';
                if (cbWrap) cbWrap.style.opacity = fadeIn || '';
                if (fadeIn > 0) _positionFullscreenChrome();
            }

            // Compare/multiview/qMRI chrome
            if (compareActive) {
                const cmpFade = _fullscreenActive ? fadeIn : (1 - collapseP);
                document.querySelectorAll('.compare-title').forEach(el => {
                    el.style.opacity = (p === 0) ? '' : (_fullscreenActive ? '0' : (1 - collapseP));
                });
                document.querySelectorAll('.compare-pane-cb-island').forEach(el => {
                    el.style.opacity = (p === 0) ? '' : cmpFade || '';
                });
            }
            const mvCb = multiViewActive ? document.getElementById('mv-cb-wrap') : null;
            if (mvCb) {
                if (p === 0) { mvCb.style.opacity = ''; }
                else { mvCb.style.opacity = _fullscreenActive ? (fadeIn || '') : (1 - collapseP); }
            }
            const qmriCbs = qmriActive ? document.querySelectorAll('#qmri-view-wrap .qv-cb-island') : [];
            qmriCbs.forEach(el => {
                if (p === 0) { el.style.opacity = ''; }
                else { el.style.opacity = _fullscreenActive ? (fadeIn || '') : (1 - collapseP); }
            });

            // Update _fitZoom
            if (p >= 1) _fitZoom = immFit;
        }

        function _crossfadeCleanup() {
            _crossfadeCollapseP = 0;
            const titleEl = document.getElementById('array-name');
            const infoEl = document.getElementById('info');
            const cbWrap = document.getElementById('slim-cb-wrap');
            for (const el of [titleEl, infoEl, cbWrap]) {
                if (el) { el.style.opacity = ''; el.style.transition = ''; el.style.maxHeight = ''; el.style.overflow = ''; el.style.margin = ''; el.style.padding = ''; }
            }
            document.querySelectorAll('.compare-title, .compare-pane-cb-island').forEach(el => {
                el.style.opacity = ''; el.style.transition = '';
            });
            const mvCb = document.getElementById('mv-cb-wrap');
            if (mvCb) { mvCb.style.opacity = ''; mvCb.style.transition = ''; }
            document.querySelectorAll('#qmri-view-wrap .qv-cb-island').forEach(el => {
                el.style.opacity = ''; el.style.transition = '';
            });
            const wrapper = document.getElementById('wrapper');
            if (wrapper) { wrapper.style.padding = ''; wrapper.style.margin = ''; }
            _crossfadeTitleH = 0; _crossfadeInfoH = 0; _crossfadeCbH = 0;
        }
```

- [ ] **Step 2: Wire into `ModeRegistry.scaleAll()`**

Find `ModeRegistry.scaleAll()` (~line 1972):
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
                _applyImmersiveCrossfade();
                const scaler = this.getScaler();
                if (scaler) scaler();
                else if (lastImgW && lastImgH) scaleCanvas(lastImgW, lastImgH);
                else if (canvas.width && canvas.height) scaleCanvas(canvas.width, canvas.height);
                else _reconcileCbVisibility();
            },
```

- [ ] **Step 3: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: add _applyImmersiveCrossfade — zoom-driven chrome collapse + pane growth

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewrite `=` key, `-` key, and scroll handler to use crossfade

**Files:**
- Modify: `src/arrayview/_viewer.html` — `zoom.in` (~line 8445), `zoom.out` (~line 8468), scroll handler (~line 14015)

- [ ] **Step 1: Rewrite `zoom.in`**

Find `'zoom.in'` command (~line 8445). Replace the `run` body:

```javascript
                run: (ctx, e) => {
                    if (_immersiveAnimating) return;
                    if (!_fullscreenActive && !_zoomAdjustedByUser &&
                        _shouldEnterImmersive()) {
                        _enterImmersive();
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
```

With:

```javascript
                run: (ctx, e) => {
                    if (!_fullscreenActive && !_zoomAdjustedByUser &&
                        _shouldEnterImmersive()) {
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
```

- [ ] **Step 2: Rewrite `zoom.out`**

Find `'zoom.out'` command (~line 8468). Replace the `run` body:

```javascript
                run: (ctx, e) => {
                    if (_immersiveAnimating) return;
                    userZoom = Math.max(userZoom / 1.1, _normalFitZoom);
                    if (!_snapZoomToFit()) {
                        _zoomAdjustedByUser = true;
                        if (_fullscreenActive && userZoom <= _fitZoom) {
                            _exitImmersive();
                            showStatus('zoom: fit');
                            saveState();
                            return;
                        }
                    }
                    _scaleAllWithAnim();
                    showStatus(userZoom === _fitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                    saveState();
                },
```

With:

```javascript
                run: (ctx, e) => {
                    userZoom = Math.max(userZoom / 1.1, _normalFitZoom);
                    if (!_snapZoomToFit()) {
                        _zoomAdjustedByUser = true;
                    }
                    _scaleAllWithAnim();
                    showStatus(userZoom <= _normalFitZoom ? 'zoom: fit' : `zoom: ${Math.round(userZoom * 100)}%`);
                    saveState();
                },
```

- [ ] **Step 3: Simplify scroll handler**

Find the ctrl+scroll handler (~line 14015):

```javascript
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

- [ ] **Step 4: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: zoom.in/out/scroll use crossfade instead of _enterImmersive/_exitImmersive

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Simplify `zoom.reset` (0 key)

**Files:**
- Modify: `src/arrayview/_viewer.html` — `zoom.reset` command (~line 8489)

- [ ] **Step 1: Replace the `run` body**

Find `'zoom.reset'` command. Replace the entire `run` body with:

```javascript
                run: (ctx, e) => {
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
                            _crossfadeCleanup();
                            userZoom = 1.0;
                            _zoomAdjustedByUser = false;
                            mainPan.reset();
                            mosaicPan.reset();
                            ModeRegistry.scaleAllAutoFit();
                            showStatus('zoom: fit');
                        } else if (_fullscreenActive) {
                            const w = lastImgW || canvas.width;
                            const h = lastImgH || canvas.height;
                            userZoom = _calcAutoFitZoom(w, h, true);
                            _fitZoom = userZoom;
                            _zoomAdjustedByUser = false;
                            mainPan.reset();
                            ModeRegistry.scaleAll();
                            showStatus('immersive · press 0 again to exit');
                        } else {
                            _crossfadeCleanup();
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

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: simplify zoom.reset — use _crossfadeCleanup instead of _immersiveAnimating cleanup

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CSS — dimbar width constant in immersive

**Files:**
- Modify: `src/arrayview/_viewer.html` — CSS (~line 901)

- [ ] **Step 1: Remove padding override**

Find:
```css
        body.fullscreen-mode #info {
            position: fixed; z-index: 3; left: 50%; transform: translateX(-50%);
            margin: 0 !important; padding: 6px 16px;
```

Replace with:
```css
        body.fullscreen-mode #info {
            position: fixed; z-index: 3; left: 50%; transform: translateX(-50%);
            margin: 0 !important;
```

- [ ] **Step 2: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "fix: dimbar keeps constant width in immersive — remove padding override

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Remove `_enterImmersive`, `_exitImmersive`, `_immersiveAnimating`, `_animOpacity`

**Files:**
- Modify: `src/arrayview/_viewer.html`

- [ ] **Step 1: Find all remaining references**

```bash
grep -n '_enterImmersive\|_exitImmersive\|_immersiveAnimating' src/arrayview/_viewer.html
```

For every reference that is a GUARD (`&& !_immersiveAnimating` or `if (!_immersiveAnimating)`): remove the `_immersiveAnimating` part from the condition. If the entire condition was just `if (!_immersiveAnimating)`, remove the if-wrapper but keep the body.

For `if (_immersiveAnimating) return;` in `_applyImmersiveCrossfade`: delete the line.

- [ ] **Step 2: Delete `let _immersiveAnimating = false;`**

- [ ] **Step 3: Delete `function _enterImmersive()` (entire function, ~65 lines)**

- [ ] **Step 4: Delete `function _exitImmersive()` (entire function, ~115 lines)**

- [ ] **Step 5: Delete `function _animOpacity()` if no callers remain**

```bash
grep -n '_animOpacity' src/arrayview/_viewer.html
```

If only the definition remains, delete it.

- [ ] **Step 6: Verify clean**

```bash
grep -n '_enterImmersive\|_exitImmersive\|_immersiveAnimating\|_animOpacity' src/arrayview/_viewer.html
```

Should return NO results (except possibly comments).

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: remove _enterImmersive, _exitImmersive, _immersiveAnimating, _animOpacity

Replaced by stateless zoom-driven _applyImmersiveCrossfade.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Run visual smoke tests

**Files:**
- Run: `tests/visual_smoke.py`

- [ ] **Step 1: Run smoke tests**

```bash
uv run python tests/visual_smoke.py
```

Expected: All tests pass (zoom tests 04, 05, 36, 38, 63c, 71 use keyboard +/-/0 which go through `_scaleAllWithAnim` → `ModeRegistry.scaleAll()` → crossfade).

- [ ] **Step 2: Fix any failures**

- [ ] **Step 3: Commit fixes if needed**
