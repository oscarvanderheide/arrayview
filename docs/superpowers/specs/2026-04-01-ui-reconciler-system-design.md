# UI Reconciler System

## Problem

UI element visibility/state is toggled in 15+ scattered locations per element. When one site is updated but others aren't, bugs appear. The combinatorial explosion of modes (normal, compare, multiview, qMRI, compare-qMRI, compare-mv) × fullscreen × compare sub-modes makes this intractable to maintain by hand.

## Solution

Grouped reconciler functions that derive UI state from mode flags. Each reconciler reads current state and sets the correct DOM — no switches, no mode-specific branches, just formulas.

## Architecture

```
_reconcileUI()                    ← unified entry point
  ├── _reconcileLayout()          ← which containers are visible
  ├── _reconcileCompareState()    ← compare sub-mode derived UI
  └── _reconcileCbVisibility()    ← already implemented
```

### Calling strategy

| Trigger | What to call |
|---------|-------------|
| Mode entry/exit | `_reconcileUI()` |
| `_setCompareCenterMode()` | `_reconcileCompareState()` + `_reconcileCbVisibility()` |
| Animation phases | `_reconcileCbVisibility({ animPhase })` only |
| `ModeRegistry.scaleAll()` | `_reconcileCbVisibility()` only |

## `_reconcileLayout()`

Derives container visibility from mode flags.

| Element | Rule |
|---------|------|
| `#canvas-wrap` display | Visible when not in compare, multiview, qMRI, compareQmri, or compareMv |
| `compareWrap.active` class | When `compareActive && !compareQmriActive && !compareMvActive` |
| `#array-name` display | Hidden when `compareActive` |
| `#mv-orientation` display | Shown when `multiViewActive \|\| compareMvActive` |
| `canvas-bordered` class | Applied to `#canvas-viewport`, `.qv-canvas-wrap`, `.compare-canvas-wrap`, `.mv-view` when `canvasBorders` |

## `_reconcileCompareState()`

Derives compare sub-mode UI from center mode flags. Skipped when `!compareActive`.

| Element | Rule |
|---------|------|
| `#compare-diff-pane` display | `(diffMode > 0 \|\| registrationMode) ? 'flex' : 'none'` |
| `#compare-diff-pane.overlay-center` | When `registrationMode` |
| `#compare-wipe-pane` display | `(_wipeActive \|\| _flickerActive \|\| _checkerActive) ? 'flex' : 'none'` |
| `compareWrap.wipe-mode` class | Same condition as wipe pane |
| `compareWrap.focus-diff` class | `compareFocusMode && diffMode > 0` |
| `compareWrap.focus-reg` class | `compareFocusMode && registrationMode` |
| `#compare-panes` flex-wrap | `nowrap` when center mode active with 2 display SIDs |

## `_reconcileCbVisibility()`

Already implemented. Handles `#slim-cb-wrap`, `.compare-pane-cb-island` fs-overlay, `.qv-cb-island` fs-overlay, `#compare-center-island`.

## `_reconcileUI()` wrapper

```javascript
function _reconcileUI() {
    _reconcileLayout();
    if (compareActive) _reconcileCompareState();
    _reconcileCbVisibility();
}
```

## Validation rules

Add to `_validateUIState()`:

| Rule | Checks |
|------|--------|
| R27 | `#canvas-wrap` display matches expected for current mode |
| R28 | `compareWrap.active` matches `compareActive && !compareQmriActive && !compareMvActive` |
| R29 | `#array-name` display matches `!compareActive` |
| R30 | `canvas-bordered` class present on all selectors iff `canvasBorders` |
| R31 | `#compare-diff-pane` display matches `diffMode > 0 \|\| registrationMode` |
| R32 | `compareWrap.wipe-mode` matches `_wipeActive \|\| _flickerActive \|\| _checkerActive` |
| R33 | `compareWrap.focus-diff`/`focus-reg` match `compareFocusMode` × mode flags |

## Not included

Interaction-driven elements (pixel info tooltips, minimap, info-overlay) — their visibility depends on transient hover/mouse state, not mode flags. The reconciler pattern doesn't fit.

## Implementation approach

Same incremental strategy as the CB visibility reconciler:
1. Add functions alongside existing code (shadow mode)
2. Wire into dispatch points (belt-and-suspenders)
3. Remove old scattered toggles
4. Add permanent validation rules
