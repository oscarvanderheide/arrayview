---
name: modes-consistency
description: Use when implementing any visual feature in arrayview that touches canvas rendering, zoom, eggs, colorbars, keyboard shortcuts, or layout. Ensures the feature is applied consistently across ALL viewing modes, not just the one being worked on.
---

# ArrayView Modes Consistency Checklist

## Rule

Every visual feature in `_viewer.html` MUST be implemented for all applicable modes. Implementing it only in one mode and shipping is a bug, not a partial feature.

## The Six Modes

| Mode | State flag | Scale function | Entry key |
|------|-----------|---------------|-----------|
| **Normal** | (default) | `scaleCanvas()` | — |
| **Multi-view** | `multiViewActive` | `mvScaleAllCanvases()` | V / v |
| **Compare** | `compareActive` | `compareScaleCanvases()` | B / P |
| **Diff** | `diffMode > 0` (inside compare) | `compareScaleCanvases()` | X (in compare) |
| **Registration** | `registrationMode` (inside compare) | `compareScaleCanvases()` | R (in compare) |
| **qMRI** | `qmriActive` | `qvScaleAllCanvases()` | q |

Note: Overlay mode (`overlay_sid` URL param) is composited server-side into the normal frame — check backend rendering too.

## Common Feature Categories & Where to Implement

### Zoom / Canvas Sizing

Every mode has a dedicated scale function. When changing zoom behavior, check **all four**:

- `scaleCanvas(w, h)` [Normal] — applies `baseScale * userZoom`, snaps `userZoom` to cap at top
- `mvScaleAllCanvases()` [Multi-view] — iterates `mvViews`, caps `userZoom` via per-view calculation
- `compareScaleCanvases()` [Compare / Diff / Registration] — grid layout, caps `userZoom` before applying
- `qvScaleAllCanvases()` [qMRI] — grid layout for parameter maps

**Cap pattern** (already applied in compare): compute `capZoom` from max allowable scale across all panes, then `if (userZoom > capZoom) userZoom = capZoom;` before the sizing loop.

### Eggs (Mode Indicator Dots — LOG, complex, mask, overlay badges)

All eggs are positioned by `positionEggs()`. The function branches by mode. When changing egg placement rules, check:

- Normal branch: uses `#slim-cb-wrap` bounding box if visible, else canvas bottom
- Multi-view branch: uses `#mv-cb-wrap` bounding box if present, else estimates 36px below panes
- Compare branch: walks `.compare-canvas-wrap` rects to find the tallest pane bottom
- qMRI branch: uses `.qv-canvas-wrap` rects + 36px estimate

When adding new egg types or changing vertical anchor, update all four branches.

### Colorbar / Window-Level Interaction

Colorbars are drawn by separate per-mode functions — there is no shared abstraction:

| Mode | Colorbar function(s) |
|------|---------------------|
| Normal | `drawSlimColorbar(markerFrac)` |
| Compare | `drawComparePaneCb(idx)` + `drawAllComparePaneCbs()` |
| Diff (in compare) | `drawDiffPaneCb(vmin, vmax)` |
| Registration (in compare) | `drawRegBlendCb()` |
| Multi-view | `drawMvCbs()` (called inside `mvScaleAllCanvases`) |
| qMRI | Drawn inline per view in `qvRender()` |

When adding colorbar interactivity (e.g., drag, scroll), check which colorbars the feature should apply to, and implement it per-element.

### Keyboard Shortcuts

The `keydown` handler on `#keyboard-sink` dispatches by active mode. New shortcuts must:

1. Not conflict with mode-specific keys — check the table in _viewer.html's keyboard section
2. Have an explicit guard when the shortcut only makes sense in specific modes (e.g., `if (!compareActive) return;`)
3. Fall through correctly when multiple modes are active (e.g., `registrationMode` requires `compareActive`)

### Canvas Elements Present Per Mode

| Mode | Canvas elements |
|------|----------------|
| Normal | `#canvas` |
| Multi-view | `.mv-canvas` × 3  (inside `.mv-view`) |
| Compare | `.compare-canvas` × 2–6  (inside `.compare-canvas-inner`) |
| Diff | `#compare-diff-canvas` (shown only when `diffMode > 0` and `compareSids.length === 2`) |
| Registration | 3rd `.compare-canvas` = blended overlay; `compareRegistrationFrame` drives it |
| qMRI | `.qv-canvas` × 3–6 |

Features that attach event listeners to canvas elements (mouse, wheel, etc.) must attach to the correct per-mode set of canvases, not just `document.getElementById('canvas')`.

## Step-by-Step Implementation Checklist

When implementing a new feature, run through this list:

1. **Identify applicable modes**: Does this feature affect canvas sizing? colorbars? eggs? cursor? Yes → all modes.

2. **Normal mode** — implement first, verify it works.

3. **Compare mode** (includes Diff and Registration) — `compareScaleCanvases()` or the relevant compare-specific functions.

4. **Multi-view mode** — `mvScaleAllCanvases()`, or per-view listener attachment if it's an event feature.

5. **qMRI mode** — `qvScaleAllCanvases()`, or inline in `qvRender()` if it's a per-pane colorbar feature.

6. **Overlay mode (backend)** — if the feature affects how frames are composited, check the `/slice`, `/frame`, `/diff`, and `/mosaic` endpoints in `_app.py`. Overlay compositing happens in `_composite_overlay_mask()` before PNG encoding.

7. **Mosaic / z-grid** — when feature involves the diff endpoint (`/diff`), check that `dim_z` is correctly passed and handled server-side (backend `get_diff` must produce a mosaic grid when `dim_z >= 0`).

8. **State snapshot** — if the feature introduces a new state variable, add it to `collectStateSnapshot()` and `applyStateSnapshot()` so it is preserved across page reloads and compare mode transitions.

## Red Flags — STOP

- "I only changed it for normal mode, the others are TODO" → implement all applicable modes now
- "`compareScaleCanvases` and `scaleCanvas` now behave differently for zoom capping" → pick one pattern and apply to both
- "I added a canvas listener to `#canvas` and now it doesn't work in compare" → multi-canvas modes have different DOM structures
- "I only pass `dim_z` to some endpoints but not others" → check every endpoint that renders an image and needs the mosaic path

## Quick Sanity Test

After implementing, run:
```
uv run pytest tests/test_api.py -x
uv run python tests/visual_smoke.py   # then review smoke_output/
```

And manually verify:
- [ ] Feature works in normal view
- [ ] Feature works in compare (press B → pick second array)
- [ ] Feature works in diff (press X while in compare)
- [ ] Feature works in multi-view (press v)
- [ ] Feature works in qMRI if applicable (press q — needs array with 3–6 dim of size 3–6)
- [ ] Eggs remain correctly anchored below canvases in each mode after feature is applied
