# UI Reconciler System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize all scattered UI visibility toggles into grouped reconciler functions so UI state is derived from mode flags in one place.

**Architecture:** Three grouped reconcilers (`_reconcileLayout()`, `_reconcileCompareState()`, `_reconcileCbVisibility()`) called through a `_reconcileUI()` wrapper. Each reads current mode flags and sets the correct DOM state. The existing `_reconcileCbVisibility()` (already implemented) is incorporated into the wrapper.

**Tech Stack:** Pure JS, single file `_viewer.html`

---

## Key File

All changes in: `src/arrayview/_viewer.html`

## Important Context

- `_reconcileCbVisibility()` already exists (~line 13548) and handles: `#slim-cb-wrap`, `.compare-pane-cb-island` fs-overlay, `.qv-cb-island` fs-overlay, `#compare-center-island`
- `_validateUIState()` exists (~line 13453) with Rules 4, 10, 12, 13, 16, 17, 23, 24, 25, 26
- `compareWrap` is a cached reference to `document.getElementById('compare-view-wrap')`
- Mode flags: `compareActive`, `multiViewActive`, `qmriActive`, `compareQmriActive`, `compareMvActive`, `_fullscreenActive`
- Compare sub-mode flags: `diffMode` (0-3), `registrationMode`, `_wipeActive`, `_flickerActive`, `_checkerActive`, `compareFocusMode`
- User preference: `canvasBorders`
- Initial `canvas-wrap` state is `display:none` (loading). Do NOT touch the initial load display toggles (lines ~5628, 5772, 6015) — those are loading lifecycle, not mode transitions.

---

### Task 1: Add reconciler functions (shadow mode)

**Files:** Modify `src/arrayview/_viewer.html`

- [ ] **Step 1: Add `_reconcileLayout()` function**

Add right before `_reconcileCbVisibility()` (~line 13548), after `_validateUIState()`:

```javascript
// ── Layout container visibility reconciler ───────────────────
// Derives which top-level containers are visible from mode flags.
function _reconcileLayout() {
    // 1. Canvas wrap — visible only in single-view (no special mode)
    const canvasWrap = document.getElementById('canvas-wrap');
    if (canvasWrap && !document.body.classList.contains('av-loading')) {
        canvasWrap.style.display = (!compareActive && !multiViewActive && !qmriActive && !compareQmriActive && !compareMvActive) ? '' : 'none';
    }

    // 2. Compare wrap active class — shown for base compare (not sub-modes using other containers)
    compareWrap.classList.toggle('active', compareActive && !compareQmriActive && !compareMvActive);

    // 3. Array name — hidden in compare mode
    const nameEl = document.getElementById('array-name');
    if (nameEl) nameEl.style.display = compareActive ? 'none' : '';

    // 4. Multiview orientation indicator
    const mvOrientation = document.getElementById('mv-orientation');
    if (mvOrientation) mvOrientation.style.display = (multiViewActive || compareMvActive) ? 'block' : 'none';

    // 5. Canvas borders — apply to all relevant containers based on user preference
    const vp = document.getElementById('canvas-viewport');
    if (vp) vp.classList.toggle('canvas-bordered', canvasBorders);
    document.querySelectorAll('.qv-canvas-wrap, .compare-canvas-wrap, .mv-view').forEach(el => {
        el.classList.toggle('canvas-bordered', canvasBorders);
    });
}
```

Note: the `av-loading` guard on canvas-wrap prevents the reconciler from showing it during initial load (before first frame arrives). The loading lifecycle handles that transition.

- [ ] **Step 2: Add `_reconcileCompareState()` function**

Add right after `_reconcileLayout()`:

```javascript
// ── Compare sub-mode state reconciler ────────────────────────
// Derives compare sub-mode UI from center mode flags.
// When !compareActive, cleans up all compare-specific classes.
function _reconcileCompareState() {
    const isCenter = diffMode > 0 || registrationMode || _wipeActive || _flickerActive || _checkerActive;
    const showWipePane = _wipeActive || _flickerActive || _checkerActive;

    // 1. Diff pane visibility and overlay class
    const diffPane = document.getElementById('compare-diff-pane');
    if (diffPane) {
        diffPane.style.display = (compareActive && (diffMode > 0 || registrationMode)) ? 'flex' : 'none';
        diffPane.classList.toggle('overlay-center', compareActive && registrationMode);
    }

    // 2. Wipe pane visibility
    const wipePaneEl = document.getElementById('compare-wipe-pane');
    if (wipePaneEl) wipePaneEl.style.display = (compareActive && showWipePane) ? 'flex' : 'none';

    // 3. Wipe mode class
    compareWrap.classList.toggle('wipe-mode', compareActive && showWipePane);

    // 4. Focus classes
    compareWrap.classList.toggle('focus-diff', compareActive && compareFocusMode && diffMode > 0);
    compareWrap.classList.toggle('focus-reg', compareActive && compareFocusMode && registrationMode);

    // 5. Flex-wrap for 3-column layout
    const panesEl = document.getElementById('compare-panes');
    if (panesEl) {
        panesEl.style.flexWrap = (compareActive && isCenter && compareDisplaySids.length === 2) ? 'nowrap' : '';
    }
}
```

- [ ] **Step 3: Add `_reconcileUI()` wrapper**

Add right before `_reconcileLayout()`:

```javascript
// ── Unified UI reconciler ────────────────────────────────────
// Single entry point that reconciles all UI state from current flags.
// Call after any mode state change.
function _reconcileUI() {
    _reconcileLayout();
    _reconcileCompareState();
    _reconcileCbVisibility();
}
```

- [ ] **Step 4: Add shadow validation rules in `_validateUIState()`**

Add after Rule 26, before the `if (w.length)` line:

```javascript
// Rule 27 (shadow) — layout reconciler: canvas-wrap display matches expected
const _vCanvasWrap = document.getElementById('canvas-wrap');
if (_vCanvasWrap && !document.body.classList.contains('av-loading')) {
    const expectedVisible = !compareActive && !multiViewActive && !qmriActive && !compareQmriActive && !compareMvActive;
    const actuallyVisible = _vCanvasWrap.style.display !== 'none';
    if (actuallyVisible !== expectedVisible) {
        w.push(`Rule 27 (shadow): canvas-wrap display=${_vCanvasWrap.style.display} but expected visible=${expectedVisible}`);
    }
}

// Rule 28 (shadow) — layout reconciler: compareWrap.active matches expected
const _vCmpActive = compareWrap.classList.contains('active');
const _vExpectedCmpActive = compareActive && !compareQmriActive && !compareMvActive;
if (_vCmpActive !== _vExpectedCmpActive) {
    w.push(`Rule 28 (shadow): compareWrap.active=${_vCmpActive} but expected=${_vExpectedCmpActive}`);
}

// Rule 29 (shadow) — layout reconciler: array-name display
const _vNameEl = document.getElementById('array-name');
if (_vNameEl) {
    const nameVisible = _vNameEl.style.display !== 'none';
    if (compareActive && nameVisible) {
        w.push('Rule 29 (shadow): array-name visible but compareActive=true');
    }
}

// Rule 30 (shadow) — compare state: wipe-mode class
const _vHasWipeClass = compareWrap.classList.contains('wipe-mode');
const _vExpectedWipe = compareActive && (_wipeActive || _flickerActive || _checkerActive);
if (_vHasWipeClass !== _vExpectedWipe) {
    w.push(`Rule 30 (shadow): compareWrap.wipe-mode=${_vHasWipeClass} but expected=${_vExpectedWipe}`);
}
```

- [ ] **Step 5: Commit**

```
feat: add _reconcileLayout(), _reconcileCompareState(), _reconcileUI() in shadow mode
```

---

### Task 2: Wire `_reconcileUI()` into dispatch points

**Files:** Modify `src/arrayview/_viewer.html`

- [ ] **Step 1: Upgrade existing `_reconcileCbVisibility()` calls to `_reconcileUI()`**

Find all `_reconcileCbVisibility()` calls (NOT those with `animPhase` param) in mode entry/exit functions and replace with `_reconcileUI()`:

Replace in these functions (search for `_reconcileCbVisibility()` without arguments):
- `enterMultiView()` → change to `_reconcileUI()`
- `enterQmri()` → change to `_reconcileUI()`
- `enterCompareQmri()` → change to `_reconcileUI()`
- `enterCompareMv()` → change to `_reconcileUI()`
- `setFullscreenMode(false)` → change to `_reconcileUI()`
- `_exitImmersive()` snap-to-normal path → change to `_reconcileUI()` (note: `ModeRegistry.scaleAll()` in this path already calls `_reconcileCbVisibility()` so this is belt-and-suspenders)
- Key-0 handler animation-cancel path → change to `_reconcileUI()`
- Key-0 handler double-tap path → change to `_reconcileUI()`

**DO NOT change these** (they use animPhase or are in hot paths):
- `_reconcileCbVisibility({ animPhase: 'enter-2' })` in `_enterImmersive()`
- `_reconcileCbVisibility({ animPhase: 'exit-3' })` in `_exitImmersive()` Phase 3
- `_reconcileCbVisibility()` in `ModeRegistry.scaleAll()` — keep as CB-only for performance
- `_reconcileCbVisibility()` in `_setCompareCenterMode()` — handled in Step 2

- [ ] **Step 2: Update `_setCompareCenterMode()` dispatch**

In `_setCompareCenterMode()`, find the existing `_reconcileCbVisibility()` call and replace with:
```javascript
_reconcileCompareState();
_reconcileCbVisibility();
```
(Don't call full `_reconcileUI()` here — layout doesn't change when toggling center modes.)

- [ ] **Step 3: Add `_reconcileUI()` to functions not yet wired**

Add `_reconcileUI()` call near the end of these functions (before `showStatus`/`renderEggs`/`saveState`):
- `exitCompareMode()` — after all flag resets, before `drawSlimColorbar()`
- `exitMultiView()` — after cleanup, before `renderInfo()`
- `exitQmri()` — after cleanup, before `renderInfo()`
- `exitCompareQmri()` — after cleanup, before `renderInfo()`
- `exitCompareMv()` — after cleanup, before `renderInfo()`
- `enterCompareModeBySid()` — after `compareActive = true` and `compareWrap.classList.add('active')` block
- `enterCompareModeByMultipleSids()` — same position
- `toggleWipeMode()` — after `_wipeActive = !_wipeActive`, replace manual toggles

**Keep all old toggles in place** — belt-and-suspenders.

- [ ] **Step 4: Commit**

```
feat: wire _reconcileUI() into all mode entry/exit dispatch points
```

---

### Task 3: Remove old layout toggles

**Files:** Modify `src/arrayview/_viewer.html`

- [ ] **Step 1: Remove `#canvas-wrap` display toggles from mode functions**

Remove these lines (the reconciler now handles canvas-wrap visibility):

In `enterCompareModeBySid()` (~line 9729):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `enterCompareModeByMultipleSids()` (~line 9769):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `exitCompareMode()` (~line 10369-10371):
```javascript
// REMOVE:
if (!multiViewActive && !qmriActive) {
    document.getElementById('canvas-wrap').style.display = '';
}
```

In `enterMultiView()` (~line 10408):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `exitMultiView()` (~line 10630):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = '';
```

In `enterQmri()` (~line 10671):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `exitQmri()` (~line 10840):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = '';
```

In `enterCompareQmri()` (~line 10877):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `exitCompareQmri()` (~line 11041):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = '';
```

In `enterCompareMv()` (~line 11224):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

In `exitCompareMv()` (~line 11420):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = '';
```

In `compareRender()` (~line 5866):
```javascript
// REMOVE:
document.getElementById('canvas-wrap').style.display = 'none';
```

**DO NOT remove** the initial load sites (lines ~5628, 5772, 6015) — those are loading lifecycle.

- [ ] **Step 2: Remove `compareWrap.active` toggles**

In `enterCompareModeBySid()` (~line 9731):
```javascript
// REMOVE:
compareWrap.classList.add('active');
```

In `enterCompareModeByMultipleSids()` (~line 9771):
```javascript
// REMOVE:
compareWrap.classList.add('active');
```

In `exitCompareMode()` (~line 10366):
```javascript
// REMOVE:
compareWrap.classList.remove('active');
```

In `enterCompareQmri()` (~line 10876):
```javascript
// REMOVE:
compareWrap.classList.remove('active');
```

In `exitCompareQmri()` (~line 11037):
```javascript
// REMOVE:
compareWrap.classList.add('active');
```

In `enterCompareMv()` (~line 11223):
```javascript
// REMOVE:
compareWrap.classList.remove('active');
```

In `exitCompareMv()` (~line 11416):
```javascript
// REMOVE:
compareWrap.classList.add('active');
```

- [ ] **Step 3: Remove `#array-name` display toggles**

In `enterCompareModeBySid()` (~line 9730):
```javascript
// REMOVE:
document.getElementById('array-name').style.display = 'none';
```

In `enterCompareModeByMultipleSids()` (~line 9770):
```javascript
// REMOVE:
document.getElementById('array-name').style.display = 'none';
```

In `exitCompareMode()` (~line 10372):
```javascript
// REMOVE:
document.getElementById('array-name').style.display = '';
```

- [ ] **Step 4: Remove `#mv-orientation` display toggles**

In `enterMultiView()` (~line 10598):
```javascript
// REMOVE:
document.getElementById('mv-orientation').style.display = 'block';
```

In `exitMultiView()` (~line 10629):
```javascript
// REMOVE:
document.getElementById('mv-orientation').style.display = 'none';
```

In `enterCompareMv()` (~line 11379):
```javascript
// REMOVE:
document.getElementById('mv-orientation').style.display = 'block';
```

In `exitCompareMv()` (~line 11410):
```javascript
// REMOVE:
document.getElementById('mv-orientation').style.display = 'none';
```

- [ ] **Step 5: Update comments in `enterCompareMv()`**

The comment "Hide compare panes" (~line 11222) should be updated since `compareWrap.classList.remove('active')` is removed. Change to just reference the reconciler:
```javascript
// Layout reconciled by _reconcileUI() — show qmri-view-wrap (shared container)
```

- [ ] **Step 6: Commit**

```
refactor: remove scattered layout toggles — reconciler is now authoritative
```

---

### Task 4: Remove old compare state toggles

**Files:** Modify `src/arrayview/_viewer.html`

- [ ] **Step 1: Clean up `exitCompareMode()`**

Remove these lines from `exitCompareMode()` (reconciler handles all of them):

```javascript
// REMOVE all of these:
document.getElementById('compare-diff-pane').style.display = 'none';
document.getElementById('compare-diff-pane').classList.remove('overlay-center');
const _wp = document.getElementById('compare-wipe-pane');
if (_wp) _wp.style.display = 'none';
compareWrap.classList.remove('wipe-mode');
// (compareWrap.classList.remove('active') already removed in Task 3)
compareFocusMode = false;
compareWrap.classList.remove('focus-diff', 'focus-reg');
```

**KEEP** `compareFocusMode = false;` (the flag reset), just remove the classList line after it. The reconciler reads `compareFocusMode` to derive the classes.

- [ ] **Step 2: Clean up `_setCompareCenterMode()`**

Remove these lines (reconciler handles them via `_reconcileCompareState()` call):

```javascript
// REMOVE: diff pane display (lines ~2997-2999)
if (diffPane) {
    diffPane.style.display = (diffMode || registrationMode) ? 'flex' : 'none';
    diffPane.classList.toggle('overlay-center', registrationMode);
}

// REMOVE: wipe pane display and wipe-mode class (lines ~3003-3006)
const showWipePane = _wipeActive || _flickerActive || _checkerActive;
if (wipePaneEl) wipePaneEl.style.display = showWipePane ? 'flex' : 'none';
compareWrap.classList.toggle('wipe-mode', showWipePane);

// REMOVE: focus class cleanup at newMode === 0 (lines ~2991-2992)
if (newMode === 0 && compareFocusMode) {
    compareFocusMode = false;
    compareWrap.classList.remove('focus-diff', 'focus-reg');
}

// REMOVE: flex-wrap (line ~3013-3014)
const panesEl = document.getElementById('compare-panes');
const _hasCenterNow = (diffMode > 0 || registrationMode || showWipePane);
if (panesEl) panesEl.style.flexWrap = (_hasCenterNow && compareDisplaySids.length === 2) ? 'nowrap' : '';
```

**KEEP** the `compareFocusMode = false` flag reset (reconciler reads it), the diff pane content updates (`diffPane.classList.toggle('overlay-center'...` is now in reconciler), and all imperative work (fetchAndDrawDiff, _drawWipe, _startFlicker, _drawCheckerboard, showStatus, etc.).

**KEEP** the `const diffPane = document.getElementById('compare-diff-pane');` declaration IF it's still used for content setting (e.g., title innerHTML). Remove if now unused.

- [ ] **Step 3: Clean up `toggleWipeMode()`**

Replace the manual toggles with a reconciler call:

```javascript
// BEFORE:
_wipeActive = !_wipeActive;
compareWrap.classList.toggle('wipe-mode', _wipeActive);
const wipePaneEl = document.getElementById('compare-wipe-pane');
if (wipePaneEl) wipePaneEl.style.display = _wipeActive ? 'flex' : 'none';

// AFTER:
_wipeActive = !_wipeActive;
_reconcileCompareState();
_reconcileCbVisibility();
```

- [ ] **Step 4: Clean up `toggleRegistrationMode()`**

Remove the focus class cleanup (reconciler handles it):

```javascript
// REMOVE (lines ~10316-10318):
if (compareFocusMode) {
    compareFocusMode = false;
    compareWrap.classList.remove('focus-diff', 'focus-reg');
}
```

**KEEP** `compareFocusMode = false;` — but move it before `_setCompareCenterMode(0)` so the flag is correct when the reconciler runs.

- [ ] **Step 5: Clean up Z key handler**

Replace the focus class toggles with a reconciler call:

```javascript
// BEFORE (~line 8716-8718):
compareFocusMode = !compareFocusMode;
compareWrap.classList.toggle('focus-diff', compareFocusMode && diffMode > 0);
compareWrap.classList.toggle('focus-reg', compareFocusMode && registrationMode && diffMode === 0);

// AFTER:
compareFocusMode = !compareFocusMode;
_reconcileCompareState();
```

- [ ] **Step 6: Commit**

```
refactor: remove scattered compare state toggles — reconciler is now authoritative
```

---

### Task 5: Simplify canvas-bordered keydown handler

**Files:** Modify `src/arrayview/_viewer.html`

- [ ] **Step 1: Replace B key handler cascade**

Find the B key handler (~line 8847) with its mode-specific if/else cascade and replace with:

```javascript
// BEFORE:
canvasBorders = !canvasBorders;
if (multiViewActive) {
    mvViews.forEach(v => { if (v.pane) v.pane.classList.toggle('canvas-bordered', canvasBorders); });
} else if (compareQmriActive || compareMvActive) {
    document.querySelectorAll('.qv-canvas-wrap').forEach(el => el.classList.toggle('canvas-bordered', canvasBorders));
} else if (compareActive) {
    document.querySelectorAll('.compare-canvas-wrap').forEach(el => el.classList.toggle('canvas-bordered', canvasBorders));
} else if (qmriActive) {
    document.querySelectorAll('.qv-canvas-wrap').forEach(el => el.classList.toggle('canvas-bordered', canvasBorders));
} else {
    document.getElementById('canvas-viewport').classList.toggle('canvas-bordered', canvasBorders);
}

// AFTER:
canvasBorders = !canvasBorders;
_reconcileLayout();
```

- [ ] **Step 2: Remove F key (zen mode) border cleanup**

Find the F key handler (~line 9656-9659) and replace the manual border removal:

```javascript
// BEFORE:
if (!_zenActive && canvasBorders) {
    canvasBorders = false;
    document.getElementById('canvas-viewport').classList.remove('canvas-bordered');
    document.querySelectorAll('.mv-view, .compare-canvas-wrap, .qv-canvas-wrap').forEach(el => el.classList.remove('canvas-bordered'));
}

// AFTER:
if (!_zenActive && canvasBorders) {
    canvasBorders = false;
    _reconcileLayout();
}
```

- [ ] **Step 3: Remove `exitMultiView()` border re-apply**

Find (~line 10643):
```javascript
// REMOVE:
document.getElementById('canvas-viewport').classList.toggle('canvas-bordered', canvasBorders);
```
The reconciler already handles this when called from exitMultiView().

- [ ] **Step 4: Commit**

```
refactor: replace scattered canvas-bordered toggles with _reconcileLayout()
```

---

### Task 6: Replace shadow validation with permanent rules + update docs

**Files:** Modify `src/arrayview/_viewer.html`, `dev/lessons_learned.md`, `AGENTS.md`

- [ ] **Step 1: Replace shadow Rules 27-30 with permanent versions**

Remove "(shadow)" labels and tighten the validation:

```javascript
// Rule 27 — Layout reconciler: canvas-wrap display matches expected
// (same code, remove "(shadow)" from label)

// Rule 28 — Layout reconciler: compareWrap.active matches expected
// (same code, remove "(shadow)" from label)

// Rule 29 — Layout reconciler: array-name display
// (same code, remove "(shadow)" from label)

// Rule 30 — Compare state: wipe-mode class
// (same code, remove "(shadow)" from label)

// Rule 31 — Compare state: diff pane visibility
const _vDiffPane = document.getElementById('compare-diff-pane');
if (_vDiffPane && !_immersiveAnimating) {
    const expectedDiff = compareActive && (diffMode > 0 || registrationMode);
    const actuallyVisible = _vDiffPane.style.display !== 'none';
    if (actuallyVisible !== expectedDiff) {
        w.push(`Rule 31: compare-diff-pane display=${_vDiffPane.style.display} but expected visible=${expectedDiff}`);
    }
}

// Rule 32 — Compare state: focus classes
if (!_immersiveAnimating) {
    const hasFocusDiff = compareWrap.classList.contains('focus-diff');
    const expectedFocusDiff = compareActive && compareFocusMode && diffMode > 0;
    if (hasFocusDiff !== expectedFocusDiff) {
        w.push(`Rule 32: focus-diff=${hasFocusDiff} but expected=${expectedFocusDiff}`);
    }
}
```

- [ ] **Step 2: Update `dev/lessons_learned.md`**

Add a new section:

```markdown
## UI Reconciler Pattern

**Problem:** UI element visibility scattered across 15+ toggle sites per element. Mode combinatorics make it impossible to keep all sites in sync — same class of bug fixed 4+ times for colorbars alone.

**Solution:** Grouped reconciler functions that derive UI state from mode flags:
- `_reconcileLayout()` — container visibility (canvas-wrap, compareWrap, array-name, mv-orientation, canvas-bordered)
- `_reconcileCompareState()` — compare sub-mode UI (diff/wipe panes, wipe-mode/focus classes, flex-wrap)
- `_reconcileCbVisibility()` — colorbar/island visibility
- `_reconcileUI()` — wrapper that calls all three

**Key insight:** Mode entry/exit functions set flags, then call the reconciler. The reconciler reads flags and computes correct DOM state. No function needs to know what other functions do — the reconciler is the single source of truth.

**When adding new modes or UI elements:** Add the visibility rule to the appropriate reconciler. All existing call sites automatically get the update.
```

- [ ] **Step 3: Update `AGENTS.md`**

Add to the "Non-Negotiables" section:

```markdown
- UI visibility changes go through reconcilers (_reconcileUI/_reconcileLayout/_reconcileCompareState/_reconcileCbVisibility), not inline style.display or classList toggles in mode functions
```

- [ ] **Step 4: Commit**

```
feat: finalize UI reconciler validation rules, update docs and AGENTS.md
```
