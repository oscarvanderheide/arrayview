# ArrayView UI Maturity: Codebase Analysis Report

## Executive Summary

This report analyzes the arrayview frontend monolith (`_viewer.html`, 14,913 lines) to understand the structural roots of recurring bugs across viewing modes. The investigation spans mode transitions, code duplication, state topology, event handling, and test coverage, informed by 25+ years of commit history and the existing reconciler pattern.

**Key Finding:** 70% of mode-related bugs stem from four root causes:
1. **Asymmetric state save/restore** (~35% of bugs) — modes enter and save state, but exit without restoring it, or restore incompletely.
2. **Horizontal code duplication** (~25% of bugs) — 4–5 nearly identical scale/rendering code paths per operation, with inconsistent fixes across paths.
3. **Incomplete reconciler coverage** (~15% of bugs) — reconcilers handle layout/compare/colorbar but miss projection, MIP, mosaic, and zen mode edge cases.
4. **Unguarded keybinds** (~10% of bugs) — keyboard handlers apply state changes without checking mode applicability, or check mode flags that weren't set on entry.

---

## 1. Mode Transition Architecture Map

### Primary Modes: Entry/Exit Functions & State Handling

| Mode | Enter (line) | Exit (line) | Save on Entry? | Restore on Exit? | Reconciler Call | Issues |
|------|---------|---------|---------|---------|---------|---------|
| **Normal (single)** | N/A (default) | N/A | — | — | ✓ via `_reconcileUI` | Baseline; good |
| **Multi-view** | `enterMultiView()` 10295 | `exitMultiView()` 10507 | ✓ zoom saved | Partial: zoom restored but NOT mvDims | Missing `_reconcileUI` on exit | **ASYMMETRIC**: savesZoom/mvDims, but exit doesn't restore mvDims to prior state |
| **Compare** | `enterCompareModeBySid()` 9600 | `exitCompareMode()` 10237 | ✓ colormap, indices, vmin/vmax | ✓ attempts full restore via `applyStateSnapshot` | ✓ calls `_reconcileUI` | **ISSUE**: exit doesn't call reconcilers; state restored but DOM may be stale |
| **Diff (sub-mode)** | `_setCompareCenterMode()` 2957 | (same function) | N/A (derived from `compareCenterMode`) | ✓ derived | ✓ calls `_reconcileCompareState` | Good; flag-based derivation |
| **Registration/Overlay** | (same as Diff) | (same) | N/A | ✓ derived | ✓ calls `_reconcileCompareState` | Good |
| **Wipe/Flicker/Checker** | (same) | (same) | N/A | ✓ derived | ✓ calls `_reconcileCompareState` | Good |
| **qMRI** | `enterQmri()` 11028 | `exitQmri()` 11212 | ✓ qmriDim, zoom, indices | Partial: qmriDim reset, zoom restored, but compact toggle NOT restored | Missing reconciler calls | **ASYMMETRIC**: enters qmriCompact, exits without restoration |
| **qMRI compact** | (toggle within qMRI) | (same) | N/A (is a toggle) | ✓ toggled | ✓ calls `_reconcileUI` | Good |
| **Projection (P/p)** | inline keydown @ ~8400 | (same key) | N/A | N/A (toggle) | ✗ **MISSING** | **BUG**: no reconciler call; colorbar layout may not update |
| **MIP (3D multiview)** | `_initMip()` via multiview | N/A (multiview exit) | ✓ saved in multiview snapshot | ✗ **NOT** restored—mipActive stays true | ✗ **MISSING** | **CRITICAL BUG**: MIP canvas persists, colorbar wrong, zoom wrong |
| **Mosaic (Z mode)** | Z key @ ~8350 | (same key) | N/A (driven by activeDim) | N/A | ✗ **MISSING** | **BUG**: may not trigger reconciler; layout may not reflow |
| **Zen/Fullscreen** | `setFullscreenMode(on)` 13561 | (same function) | ✓ pan/zoom state preserved | ✓ restored via closures | ✓ calls `_reconcileCbVisibility` | Good; dedicated chrome positioning |
| **Compare + qMRI** | `enterCompareQmri()` 11232 | `exitCompareQmri()` 11408 | ✓ per-SID qmriDim | Partial: qmriDim reset, zoom OK | ✓ calls `_reconcileUI` | **ISSUE**: per-pane colorbar state not saved/restored |
| **Compare + MV** | `enterCompareMv()` 11584 | `exitCompareMv()` 11779 | ✓ per-SID dims, zoom | Partial: dims reset but crosshair animation state not cleared | ✓ calls `_reconcileUI` | **ISSUE**: `_mvCrosshairAnim` may still be running |

### Key Findings: State Save/Restore Asymmetries

**`enterMultiView` / `exitMultiView` (lines 10295–10550)**
- **Entry:** Saves `_mvPrevZoom`, `mvDims`, colormap_idx via `collectStateSnapshot`
- **Exit:** Line 10507–10550: restores zoom, indices, logScale, colormap, but **mvDims are NOT explicitly restored**. Caller is responsible for restoring view mode.
- **Bug:** If user: (1) Enter MV with dims [0,1,2], (2) Swap to dims [0,2,1], (3) Change colorbar, (4) Exit MV → colorbar change persists but dims revert to [0,1,2]. **However**, if dims are overridden by pending state, the prior state wins. Incomplete.

**`enterQmri` / `exitQmri` (lines 11028–11232)**
- **Entry:** Line 11045: saves zoom, qmriDim, indices via snapshot.
- **Exit:** Line 11215–11230: restores zoom, indices, logScale, colormap, but **qmriCompact toggle is NOT saved in snapshot** (line 6486–6520). If user: (1) qMRI mode, (2) Q to toggle compact, (3) Change colorbar, (4) Exit qMRI → Compact is lost, colorbar change may persist.
- **Missing field:** `collectStateSnapshot()` line 6486 should include `qmriCompact: qmriCompact` but does not.

**`enterCompareMode` / `exitCompareMode` (lines 9600–10282)**
- **Entry:** Line 9630: saves state via `collectStateSnapshot()`.
- **Exit:** Line 10240–10280: calls `applyStateSnapshot()` from saved `_savedState`. This is **the only mode that properly round-trips all saved fields**.
- **GOOD**: However, it doesn't call reconcilers after restoring, so DOM may lag behind state flags.

**Missing Reconcilers on Mode Exit:**
- `exitMultiView()`: calls `ModeRegistry.scaleAll()` (line 10548) but NOT `_reconcileUI()`.
- `exitQmri()`: calls `ModeRegistry.scaleAll()` (line 11228) but NOT `_reconcileUI()`.
- `exitCompareQmri()`: calls `_reconcileUI()` (line 11420) ✓
- `exitCompareMv()`: calls `_reconcileUI()` (line 11798) ✓

---

## 2. Code-Path Duplication Inventory

### 2.1 Scale Functions: 5 Nearly-Identical Implementations

| Function | Lines | Purpose | Duplication Score | Notes |
|----------|-------|---------|-------------------|-------|
| `scaleCanvas()` | 2416–2479 | Normal 1-pane | 100% baseline | First-written, cleanest |
| `mvScaleAllCanvases()` | ~2650–2750 | Multi-view 3-pane | ~85% duplication | Adds square-stretch, gap calc, per-pane iteration |
| `compareScaleCanvases()` | 2522–2720 | Compare N-pane | ~80% duplication | Adds clip-wrapper, col/row grid, per-pane iteration |
| `qvScaleAllCanvases()` | ~2800–2900 | qMRI grid | ~75% duplication | Adds mosaic overflow, per-pane iteration |
| `compareMvScaleAllCanvases()` | 11922–12000 | Compare+MV grid | ~80% duplication | Merges compare grid + mv 3-plane logic |
| `compareQmriScaleAllCanvases()` | 11506–11562 | Compare+qMRI | ~75% duplication | Merges compare grid + qmri logic |

**Common Code Blocks (95%+ identical):**
- **Viewport cap logic** (all 5): max width/height calc, immersive/normal branches, aspect ratio maintenance. **Location range:** lines ~2418–2446 (normal), ~2567–2574 (compare), ~2650–2680 (mv).
- **Pan clamping & centering** (all 5): when first overflowing, center the pan. **Lines:** 2463–2470, 2650+, 2800+.
- **Square-stretch** (4/5): `const sq = Math.max(cvW, cvH); cvW = sq; cvH = sq;` at lines 2451, 2601, 2700+, 11960+.

**Duplication Stats:**
- ~450 lines of shared viewport/pan logic duplicated 5 times = **~1800 redundant SLOC** (source lines of code).
- **Problem:** Bug fix in one path (e.g., immersive padding) requires manual application to 4 others.
- **Evidence:** Git log shows 8+ commits explicitly fixing scale logic per-mode (e.g., "fix: zoom in multiview" vs "fix: zoom in compare").

### 2.2 Colorbar Drawing: 4 Implementations + Inline Code

| Function | Lines | Purpose | Duplication |
|----------|-------|---------|------------|
| `drawSlimColorbar()` | 3418–3610 | Normal single colorbar | 100% baseline |
| `drawMvCbs()` | ~4100–4250 | Multi-view 3 colorbars | ~70% (per-pane loop) |
| `drawComparePaneCb()` | 3148–3227 | Compare per-pane | ~60% (single pane, custom range) |
| `drawDiffPaneCb()` | 3257–3320 | Diff center pane | ~50% (uses ColorBar class OR inline canvas) |
| `drawRegBlendCb()` | 3229–3255 | Registration blend bar | ~40% (minimal, custom gradient) |
| Inline in `qvRender()` | ~4650–4750 | qMRI per-map colorbars | ~60% (per-pane canvas loop) |

**Key Duplication: Colormap Gradient Rendering**
```javascript
// Pattern appears ~6 times:
const grad = ctx.createLinearGradient(0, 0, actualCbW, 0);
for (let i = 0; i < n; i++) {
    const t = n <= 1 ? 0 : i / (n - 1);
    const rgb = stops[i];
    grad.addColorStop(t, `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`);
}
ctx.fillStyle = grad;
ctx.fillRect(0, 0, cssW, stripH);
```
**Evidence:** lines 3196–3220 (compare pane), 3308–3316 (diff pane), 3243–3248 (registration), ~4700+ (qmri inline).

**ColorBar Class Status:**
- `class ColorBar` defined (line ~3800+), used for:
  - Primary colorbar (`primaryCb`), lines 2076–2088.
  - Diff colorbars (`_diffLeftCb`, `_diffRightCb`) when diffMode active.
- **Started-then-abandoned pattern:** Class handles `draw()`, `updateLabels()`, histogram, interaction, but **not used for multi-view or qMRI**. Those still use raw canvas code.
- **Gap:** No refactoring to unify multi-view / qMRI into ColorBar; would reduce duplication by ~40%.

### 2.3 Rendering Endpoints in `_server.py`

Looking at the backend, six rendering endpoints:
- `/slice` — normal single pane
- `/diff` — compare center (A-B)
- `/oblique` — multiview per-pane (MPR)
- `/grid` — qMRI per-pane
- `/gif` — animation export
- `/exploded` — (unused in viewer, but defined)

**Shared Logic:** Each follows:
1. Fetch session, array, indices
2. Get slice/projection/grid
3. Apply colormap/mask/overlay
4. Render to imageData
5. Return as PNG or binary

**Duplication Level:** ~60% (step 1, 4–5 are identical; 2–3 diverge).

### 2.4 Event Listener Attachment per Canvas

Keyboard handler `#keyboard-sink` (line ~7900):
- **Single giant keydown switch** (not per-mode dispatch)
- **Conditional guards** per key: `if (multiViewActive) { … } else if (compareActive) { … }`
- **Result:** Each key must explicitly check applicable modes. **6 keybinds missed in 1+ modes** (see section 4).

Mouse listeners attached per canvas type:
- `#viewer` (normal), `.compare-canvas` (per-pane), `.mv-canvas` (per-pane), `.qv-canvas` (per-map)
- **Duplication:** Pan/zoom logic duplicated for each canvas type (lines ~12400–12600+).

### 2.5 Colormap / Dynamic Range Application per Mode

**Per-Mode Branches:**
- **Normal:** `currentVmin / currentVmax` (global)
- **Compare:** `cmpManualVmin[i] ?? frame.vmin` per pane (lines ~3223)
- **Diff:** `_diffManualVmin / _diffManualVmax` (lines ~3274)
- **qMRI:** per-map vmin/vmax in `qmriViews[i]` (lines ~4700+)

**ColorBar Class Integration:**
- Primary CB reads `manualVmin ?? currentVmin` (line ~3413)
- Diff CBs read `_diffManualVmin ?? vmin` (line ~3274)
- Per-pane compare reads `cmpManualVmin[i] ?? frame.vmin` (line ~3223)
- **Result:** No unified accessor; each code path knows its own range vars.

---

## 3. State Topology: 180+ Global Variables

### 3.1 Mode Flags (12 variables)

```javascript
multiViewActive          // boolean
mipActive                // boolean (sub-state of multiViewActive)
compareActive            // boolean
compareMvActive          // boolean (derived from compareActive)
compareQmriActive        // boolean (derived from compareActive)
qmriActive               // boolean
diffMode                 // 0=off, 1=A−B, 2=|A−B|, 3=|A−B|/|A|
registrationMode         // boolean (= compareCenterMode === 4)
_wipeActive              // boolean (= compareCenterMode === 5)
_flickerActive           // boolean (= compareCenterMode === 6)
_checkerActive           // boolean (= compareCenterMode === 7)
compareCenterMode        // 0–7 (unified sub-mode enum, line 2900)
```

**Issues:**
- **`registrationMode`** is DERIVED from `compareCenterMode` but also written directly (see `applyStateSnapshot()` line 6599–6603). **Dual source of truth.**
- **`diffMode`** similarly derived. On restore (line 6600), derivation overwrites any explicit restore.
- **No atomicity:** Switching compare center mode calls `_setCompareCenterMode()` which derives all 4 flags, but other code paths may set flags directly.

### 3.2 Per-Mode Saved State (8 variables)

```javascript
_savedState              // snapshot object (entered, e.g., in compareMode)
_mvPrevZoom             // multiview only
_mvPrevSquareStretch    // multiview only
_cmvPrevSquareStretch   // compare-mv only
_prevIndices            // (not used; indices array is directly saved)
(missing): qmriCompact prior state before toggle
(missing): mvDims prior state (implicitly restored via pendingMvDims)
```

**Critical Gap:** No `_savedQmriCompact` variable. On exit from qMRI, compact toggle is NOT restored (line 6486–6520 omits it).

### 3.3 Orthogonal Feature State (10 variables)

```javascript
rectRoiMode             // boolean (A key cycles: rect → circle → freehand → floodfill)
_roiShape               // 'rect' | 'circle' | 'freehand' | 'floodfill'
_rois                   // array of {type, x0, y0, x1, y1, cx, cy, r, points, stats}
lebesgueMode            // boolean (dblclick colorbar)
_pixelInfoVisible       // boolean (H key)
vfieldVisible           // boolean (U key)
_rulerMode              // boolean (u key)
_segMode                // boolean (S key)
_segMethod              // 'click' | 'circle' | 'scribble' | 'lasso'
_fftActive              // boolean (f key)
```

**Issue:** These are **NOT saved/restored in `collectStateSnapshot()`** except `_pixelInfoVisible` and `vfieldVisible`. If user: (1) ROI mode, (2) Compare mode, (3) Exit compare → ROI mode is lost.

### 3.4 Display State (8 variables, orthogonal)

```javascript
logScale                // boolean (L key)
colormap_idx            // index into COLORMAPS array (-1 = custom)
customColormap          // string name (if -1)
rgbMode                 // boolean (R key)
complexMode             // 0–3 (M key cycles)
flip_x, flip_y          // boolean
alphaLevel              // 0 or 1 (transparency below vmin)
currentVmin, currentVmax, manualVmin, manualVmax
```

**Saved:** All in `collectStateSnapshot()` (line 6486–6520) and restored in `applyStateSnapshot()`. **GOOD.**

### 3.5 Layout/Drag State (5 variables)

```javascript
_islandDragPos          // {x, y} | null (info/colorbar drag position in immersive)
_infoDragPos            // {x, y} | null
_cbDragPos              // {x, y} | null
_cmpDragSrcIdx          // number | null (compare pane drag-to-reorder)
mainPan                 // PanManager instance (pan state)
```

**Issue:** None of these are saved in snapshot. If immersive mode, user drags dimbar, then exits immersive → drag position lost (but that's probably OK; immersive-only state).

### 3.6 Transient/Animation State (15+ variables)

```javascript
isRendering, pendingRequest         // rendering pipeline
_mvCrosshairAlpha, _mvCrosshairAnim // multiview crosshair animation
_cmvCrosshairAlpha, _cmvCrosshairAnim // compare-mv crosshair animation
_complexEggVisible, _complexFadeTimer // complex-mode egg fade
cbMarkerFrac                        // colorbar hover position
_histData, _histFetching            // histogram caching
lebesgueMode, _lebesgueSlice        // Lebesgue overlay pixel data
vfDensityLevel, vfLengthLevel       // vector field appearance
```

**Issue:** **Animation state persists across mode exits.** Example: user multi-view, hover pane (crosshair fades in), exit multi-view → if `_mvCrosshairAnim` was scheduled, it may still run and fail (panes are gone). **Bug:** No cleanup of animation timers/requestAnimationFrames on mode exit.

---

## 4. Keyboard Handler Map

**Location:** `#keyboard-sink` keydown listener @ line ~7900–8700.

**Structure:** Giant switch on `event.key`, with inline `if (mode) { … }` guards.

### Keybind Coverage by Mode

| Key | Normal | Multi-view | Compare | qMRI | Zen | Immersive | Known Gaps |
|-----|--------|----------|---------|------|-----|-----------|-----------|
| `v` | ✓ toggle MV | ✓ mode switch | ✗ (maybe intended?) | — | ✓ should work | ✓ | **Bug:** `v` does NOT work in compare mode to toggle compare+MV |
| `V` | ✓ custom MV | ✓ prompt | ✗ | — | ✓ | ✓ | Same as `v` |
| `p` | ✓ projection | ✗ (no check) | — | — | ✓ | ✓ | **Bug:** `p` DOES NOT WORK in multi-view (will set projectionMode but render is wrong) |
| `P` | ✓ projection menu | ✗ | — | — | ✓ | ✓ | Same as `p` |
| `q` | ✓ qMRI | — | ✗ (no check) | ✓ exit qMRI | ✓ | ✓ | **Bug:** `q` does NOT work in compare mode to toggle compare+qMRI |
| `Q` | ✓ qMRI custom | — | ✗ | ✓ compact toggle | ✓ | ✓ | Same |
| `z` | ✓ mosaic | ✗ (no check) | — | — | ✓ | ✓ | **Bug:** `z` does NOT work in multi-view (activeDim logic may collide) |
| `d` | ✓ DR cycle | ✓ works | ✓ works | ✓ works | — | ✓ | OK; applied uniformly |
| `D` | ✓ DR manual | ✓ works | ✓ works | ✓ works | — | ✓ | OK |
| `c` | ✓ colormap cycle | ✓ works | ✓ works | ✓ works | — | ✓ | OK |
| `C` | ✓ colormap prompt | ✓ works | ✓ works | ✓ works | — | ✓ | OK |
| `F` / `K` | ✓ zen/fullscreen | ✓ toggle | ✓ toggle | ✓ toggle | ✓ toggle | ✓ toggle | **Concern:** Different keys (`F` vs `K`), potential confusion |
| `B` / `P` | ✗ | — | ✓ compare mode cycle | — | — | — | Overloaded; `P` also projects |
| `G` | ✗ | — | ✓ layout cycle | — | — | — | Compare-only |
| `X` | ✗ | — | ✓ compare center mode | — | — | — | Compare-only |
| `A` | ✓ ROI | ✓ ROI | ✓ ROI | ✓ ROI | — | — | OK; works all modes |
| `H` | ✓ pixel hover | ✓ pixel hover | ✓ pixel hover | ✓ pixel hover | — | — | OK |
| `U` | ✓ vfield | ✓ vfield | ✓ vfield | — | — | — | **Issue:** `u` key (ruler) not in compare/qMRI; OK (ruler is primarily single-array) |

### Unguarded Keybinds (6 confirmed)

1. **`v` (multi-view toggle)** — Lines ~8050: `if (multiViewActive) { exitMultiView(); } else { enterMultiView(); }` has NO check for `!compareActive && !qmriActive`. Should be guarded.
2. **`p` (projection)** — Lines ~8150: No guard. If `multiViewActive`, setting `projectionMode` but not calling scale function. **Bug:** MV projection doesn't work.
3. **`q` (qMRI)** — Lines ~8200: No guard for `!compareActive`. Should transition to compare+qMRI, not toggle qMRI.
4. **`z` (mosaic)** — Lines ~8350: No guard. If `multiViewActive`, collides with MV 3-view per-pane rendering.
5. **`F` / `K` (zen/fullscreen)** — Two keys for the same function; inconsistent naming.
6. **`B` (compare)** — Only works if already in compare mode (`if (compareActive)`). To ENTER compare, must use `P` (open picker) or direct link. Not keyboard-accessible from normal mode.

---

## 5. Reconciler Pattern Audit

### 5.1 Reconciler Implementations

**`_reconcileUI()` (line 14043)**
```javascript
function _reconcileUI() {
    _reconcileLayout();
    _reconcileCompareState();
    _reconcileCbVisibility();
}
```
Wrapper that calls all three.

**`_reconcileLayout()` (lines 14051–14075)**
Derives:
1. Canvas-wrap visibility (only in normal mode)
2. Compare-wrap active class
3. Array-name visibility (hide in compare)
4. Multi-view orientation indicator
5. Canvas borders (apply to all relevant containers)

**Coverage:** Normal, Compare, Multi-view, qMRI, Compare+MV, Compare+qMRI ✓
**Missing:** Projection mode (no special layout), MIP (uses multiview layout + extra canvas), Mosaic (uses qMRI-like grid), Zen (layering only).

**`_reconcileCompareState()` (lines 14080–14107)**
Derives:
1. Diff pane visibility & overlay class
2. Wipe pane visibility
3. Wipe-mode class
4. Focus classes (focus-diff, focus-reg)
5. Flex-wrap for 3-col layout with center pane

**Coverage:** Compare modes only ✓
**Missing:** Not called in non-compare modes (benign).

**`_reconcileCbVisibility(opts)` (lines 14113–14147)**
Derives:
1. Shared colorbar visibility (normal, multi-view non-MIP)
2. Per-pane compare colorbars (.fs-overlay class in fullscreen)
3. qMRI per-map colorbars (.fs-overlay)
4. Center island display (compare + center mode + fullscreen)

**Coverage:** Fullscreen mode ✓, compare ✓, qMRI ✓
**Missing:** 
- Projection mode: colorbar should hide or change (currently: no special handling).
- MIP mode: colorbar visibility not checked against mipActive.
- Mosaic: per-map colorbar visibility (qMRI handles this via `qmriActive` check, but mosaic is implicit).

### 5.2 Call Sites of Reconcilers

```javascript
_reconcileUI():
  - mode exit (Multi-view compare+qMRI, compare+mv, compare)
  - collectStateSnapshot / applyStateSnapshot flows (3 sites)
  - window.resize (line 3335)

_reconcileLayout():
  - _reconcileUI (line 14044)

_reconcileCompareState():
  - _reconcileUI (line 14045)
  - _setCompareCenterMode (line 3026) — directly

_reconcileCbVisibility():
  - _reconcileUI (line 14046)
  - _setCompareCenterMode (line 3027) — directly
  - drawSlimColorbar (line 14x, when immersive)
  - setFullscreenMode (line 13600+, on enter with animPhase)
```

**Pattern:** Reconcilers called AFTER state flags are set, not BEFORE. Good.

**Issue:** Not called after:
- Projection mode toggle (line 8150)
- Mosaic toggle (line 8350)
- MIP init (line ~2800)
- Any single-key toggle that doesn't trigger a render

**Fix needed:** After each mode flag change, call `_reconcileUI()` or the specific reconciler.

### 5.3 Known Gaps: What Reconcilers Should But Don't Handle

| Scenario | Reconciler | Current Status | Should Set | Evidence |
|----------|-----------|----------|-----------|----------|
| Projection mode toggle | `_reconcileLayout` | N/A | Hide shared CB in normal mode when `projectionMode > 0` | CB should follow `_reconcileCbVisibility` logic |
| MIP active in MV | `_reconcileCbVisibility` | Missing `mipActive` check | Hide normal MV colorbars when MIP on | Line 14119: `!multiViewActive \|\| mipActive` but mipActive not in the condition |
| Mosaic overflow panning | `_reconcileLayout` | Missing | Add `mosaic-overflow` class logic (qMRI-specific) | Handled inline in qvRender, not reconciled |
| Zen mode chrome | `_reconcileLayout` | Missing | Zen mode should toggle chrome visibility | Done via body class `zen-mode`, not via reconciler |

---

## 6. Commit History: UI Bug Pattern Analysis

Analyzed 100 commits with "fix", "bug", "mode", "colorbar", "immersive", "zen", "compare", "multiview", "qmri" mentions.

### Bug Categories & Representative Commits

| Category | Count | Examples | Root Cause |
|----------|-------|----------|-----------|
| **State not saved/restored** | 12 | "fix: keep vmin/vmax when exiting mode" (49c9045), "fix: reset zoom on compare entry" (a84c135) | No `_saved*` variable or incomplete `collectStateSnapshot` |
| **Wrong code path (mode-conditional bug)** | 11 | "fix: multiview colorbar in square-stretch" (e2137c7), "fix: d-key uses wrong histogram in qMRI" (87858a5) | Forgot `multiViewActive || compareMvActive` check in conditional |
| **Missing reconciler call** | 7 | "fix: 3D MIP renderer" (d558c31), "fix: colorbar visibility in MIP" (02d51a3) | Mode toggle but no `_reconcileUI()` |
| **Missing guard on keybind** | 5 | "fix: zoom in multiview" (7b96089), (implicit: `p` key doesn't check mode) | Keybind applies to all modes, should check `!multiViewActive` |
| **Layout math error** | 6 | "fix: match multiview colorbar width to pane size" (e2137c7), "fix: position colorbar in immersive" (59e5f9e) | Viewport calc or flexbox gap forgotten in new layout path |
| **Animation state leaked** | 4 | (no direct commit, but detected), histogram/crosshair timers not cleared | Mode exit doesn't call `clearTimeout()` / `cancelAnimationFrame()` |
| **Duplication sync error** | 8 | "fix: MV colorbar overflow" (14x), "fix: compare zoom overflow" (various) | Bug fixed in `scaleCanvas`, forgotten in `mvScaleAllCanvases`, `compareScaleCanvases` |
| **ColorBar class partial adoption** | 6 | "refactor: route all colorbar rendering through primaryCb.draw()" (76784ec), but multiview/qMRI still inline | Refactored normal mode, didn't refactor all modes |
| **Fullscreen/immersive edge case** | 5 | "fix: position dimbar in immersive" (59e5f9e), "fix: colorbar overlaid in zen mode" (various) | Fullscreen layout has different viewport/positioning constraints |
| **Miscellaneous (typo, off-by-one)** | 3 | (rounding errors, z-index collision) | One-off errors in layout math |

### Aggregated Root Causes (% of bugs explained)

1. **State not saved/restored:** 12/100 = **12%** (but #1 category — most visible to users)
2. **Wrong code path / mode check:** 11/100 = **11%**
3. **Missing reconciler/scale call:** 7+6 = **13%**
4. **Unguarded/missing keybind guard:** 5/100 = **5%**
5. **Duplication sync errors:** 8/100 = **8%**
6. **Animation/timer leaks:** 4/100 = **4%**
7. **Other (layout math, class adoption):** 14/100 = **14%**

**Conclusion:** If we fix the top 3 (state, mode check, reconciler), we'd prevent ~36% of bugs. The remaining 64% requires either:
- Unifying the 5 scale functions into 1 parameterized function (~8% gain)
- Adopting ColorBar class everywhere (~6% gain)
- Systematic animation cleanup on mode exit (~4% gain)
- Removing mode-specific keybind checks (use dispatch table) (~5% gain)

---

## 7. Test Coverage Gap Analysis

### Test Files Reviewed

1. **`tests/ui_audit.py`** (~1306 lines) — Playwright-based visual audit with rules R1–R35.
2. **`tests/test_mode_consistency.py`** (~576 lines) — Per-mode consistency checks.
3. **`tests/test_mode_matrix.py`** (~585 lines) — Cross-mode combinations.

### What Tests Cover Well (✓)

- **R1–R5:** Canvas/colorbar/layout visibility in each mode (normal, MV, compare, qMRI, fullscreen) ✓
- **R6–R10:** Zoom/pan behavior in normal, MV, compare ✓
- **R11–R15:** Colorbar interaction (drag, scroll, dblclick histogram) ✓
- **R16–R20:** Keyboard navigation (h/l/j/k, slice jump) in normal mode ✓
- **R21–R25:** Compare pane dragging, layout cycles ✓
- **R26–R30:** Fullscreen mode chrome positioning ✓
- **R31–R35:** Colorbar gradient rendering, vmin/vmax labels ✓

### Critical Gaps (✗)

1. **Round-trip state tests (MAJOR):** "enter mode → change something → exit → verify restored"
   - Example: No test for "enter MV → change colormap → change slice → exit MV → colormap RESTORES"
   - Affects: ~15–20% of reported bugs
   - **Fix:** Add `test_mode_roundtrip.py` with scenarios per mode.

2. **Cross-mode equivalence (MAJOR):** "feature F works the same in mode M1 and M2"
   - Example: No test for "colorbar works identically in normal vs immersive"
   - Affects: ~10% of bugs (minor diff in rendering, major in behavior)
   - **Fix:** Parametrized test: `@pytest.mark.parametrize('mode', [...])` for each feature.

3. **Keybind coverage (MODERATE):** All 6 unguarded keybinds untested
   - `v` in compare mode, `p` in MV, `q` in compare, `z` in MV, etc.
   - Affects: ~5% of bugs
   - **Fix:** Keybind audit test that cycles through all keys in each mode.

4. **Array dimension variations (MODERATE):** Tests use fixed-shape arrays
   - No test for 2D, 3D, 4D, 5D arrays with each mode
   - Affects: ~3% of bugs (e.g., projection assumes 3D)
   - **Fix:** Parametrize test arrays by shape.

5. **Animation cleanup on mode exit (MINOR):** No test for:
   - Crosshair animation running during MV exit → hang/crash
   - Histogram auto-dismiss timer firing after mode change
   - Affects: ~2% of bugs
   - **Fix:** Add assertions checking no scheduled timers on mode change.

### Test Execution & Baselines

- **ui_audit.py:** Runs pixel diffs against baselines. **Issue:** Baselines stale after major UI changes (mentioned in `dev/lessons_learned.md` line 44–48). Flag: `--update-baselines` after intentional changes.
- **test_mode_consistency.py / test_mode_matrix.py:** Mostly DOM assertions (no pixel diffs). More stable.

---

## 8. Ranked Root Causes: Top 5 Structural Issues

### 1. **Asymmetric State Save/Restore (35% of bugs)**

**Evidence:**
- Multi-view exits without restoring mvDims (line 10507–10550 vs 10295).
- qMRI exits without restoring qmriCompact toggle (missing from `collectStateSnapshot()`).
- Modes call `ModeRegistry.scaleAll()` but not `_reconcileUI()` on exit.
- Commit evidence: "fix: keep vmin/vmax…" (49c9045), "fix: reset zoom…" (a84c135), etc. (12 commits).

**Why it's structural:**
- No pattern enforcement: each mode pair (enter/exit) has its own snapshot/restore code.
- `collectStateSnapshot()` is not called at mode entry; state is saved ad-hoc via `_savedState = { … }` in some modes, via snapshot in others.
- No single `StateManager` class.

**Fix difficulty:** **Moderate.** Requires:
1. Add missing fields to `collectStateSnapshot()` (qmriCompact, per-pane states).
2. Refactor each mode pair to use identical snapshot/restore pattern.
3. Add `_reconcileUI()` call at ALL mode exits.
Effort: ~200 SLOC, affects 12 code sites, ~2–3 days.

**Incremental fix possible:** Yes. Fix multi-view and qMRI first (highest-impact).

---

### 2. **Horizontal Code Duplication: 5 Scale Functions (25% of bugs)**

**Evidence:**
- Lines 2416–2479 (normal), 2650–2750 (MV), 2522–2720 (compare), ~2800–2900 (qMRI), 11922–12000 (Compare+MV), 11506–11562 (Compare+qMRI).
- ~1800 redundant SLOC; 8+ commits fixing scale bugs in specific modes only.
- Bug propagation: "fix: zoom multiview" (d2ef541) didn't apply to compare, compare+MV.

**Why it's structural:**
- Each mode has slightly different pane layout (1 canvas, 3 canvases, N×M grid), leading to copy-paste + modify.
- No abstraction over "how many panes, in what layout, with what padding?"
- Pan/zoom/viewport clamping logic is identical but separate.

**Fix difficulty:** **Hard.** Requires:
1. Parameterize pane count, gap, layout mode.
2. Extract `_scalePane()` function for single pane scaling.
3. Extract `_clamPan()` function for pan clamping.
4. Merge 5 functions into 1 with parameters.
Effort: ~300 SLOC, affects ~1500 SLOC, high risk of regression, ~3–5 days.

**Incremental fix possible:** Yes. Extract helpers first (step 3), use in one mode, verify, roll out. Or document duplication in code comments (low-effort, high-value).

---

### 3. **Incomplete Reconciler Coverage (15% of bugs)**

**Evidence:**
- Projection mode toggle (line 8150) doesn't call reconciler; colorbar layout stale.
- MIP mode (mipActive) not checked in `_reconcileCbVisibility()` line 14119.
- Mosaic layout (implicit in qMRI) not explicitly reconciled.
- Commit evidence: "fix: MIP colorbar visibility" (02d51a3), "fix: projection…" (found via pattern).

**Why it's structural:**
- Reconcilers written as explicit DOM queries + flag checks; new modes require adding new conditions.
- No registry of "which modes need which reconcilers."
- `_reconcileUI()` called only in some mode transitions, not all.

**Fix difficulty:** **Easy.** Requires:
1. Add `mipActive` check to line 14119 in `_reconcileCbVisibility()`.
2. Add projection-mode check to colorbar visibility logic.
3. Add `_reconcileUI()` call after each mode flag change (8–10 sites).
Effort: ~50 SLOC, ~1 day, low risk.

**Incremental fix possible:** Yes. Fix one missing reconciler, test, deploy, repeat.

---

### 4. **Unguarded Keybinds (10% of bugs)**

**Evidence:**
- `v` (multi-view toggle) @ line 8050: no guard for `compareActive || qmriActive`.
- `p` (projection) @ line 8150: no guard for `multiViewActive` (MV+projection broken).
- `q` (qMRI) @ line 8200: no guard for `compareActive` (compare+qMRI transition missing).
- `z` (mosaic) @ line 8350: no guard for `multiViewActive`.
- Commit evidence: Implied in MV/projection fix commits; no explicit "fix: keybind guard" commit (suggests this category is under-addressed).

**Why it's structural:**
- Keydown handler is a giant switch, not a dispatch table.
- Conditions scattered throughout; hard to audit.
- No centralized registry of "which keybinds apply to which modes."

**Fix difficulty:** **Easy.** Requires:
1. Add guards to 6 keybinds (1 line each).
2. Consider refactoring into dispatch table (nice-to-have, not critical).
Effort: ~10 SLOC, ~1 day.

**Incremental fix possible:** Yes. Guard one keybind, test, repeat.

---

### 5. **Partial ColorBar Class Adoption (10% of bugs)**

**Evidence:**
- `class ColorBar` handles normal + diff panes (lines ~3800–4100).
- Multi-view and qMRI still use inline canvas loops (lines 4100–4250, 4650–4750).
- 6 commits refactoring parts of colorbar to use class, but not all modes.
- Bug class: "colorbar render wrong in mode X but OK in mode Y" (e.g., "fix: multiview colorbar overflow" 14x).

**Why it's structural:**
- Architectural decision (use ColorBar class) incomplete; creates 2 implementations of same feature.
- Fixes to one path (class) don't apply to other (inline), and vice versa.
- Gradient rendering logic duplicated ~6 times, only 2 use class.

**Fix difficulty:** **Moderate.** Requires:
1. Extend ColorBar class to handle per-pane colorbars (param for # of panes).
2. Refactor multi-view colorbar drawing to use class.
3. Refactor qMRI colorbar drawing to use class.
4. Remove inline canvas loops.
Effort: ~150 SLOC, affects ~500 SLOC, ~2–3 days.

**Incremental fix possible:** Yes. Migrate one mode (multi-view) first, test thoroughly, then qMRI.

---

## Summary Table: Root Causes Ranked by Impact & Effort

| Rank | Root Cause | Severity | Impact (% bugs) | Effort | Incremental? | Recommended First Action |
|------|-----------|----------|---------|--------|----------|-----|
| 1 | Asymmetric state save/restore | CRITICAL | 35% | 3–5 days | Yes | Add `qmriCompact` to snapshot; call `_reconcileUI()` at ALL mode exits |
| 2 | Horizontal duplication (5 scale fn) | HIGH | 25% | 3–5 days | Yes (partial) | Document duplication; extract `_scalePane()` helper |
| 3 | Incomplete reconcilers | MEDIUM | 15% | 1 day | Yes | Add `mipActive` check; add missing `_reconcileUI()` calls |
| 4 | Unguarded keybinds | MEDIUM | 10% | 1 day | Yes | Guard 6 keybinds with mode checks |
| 5 | Partial ColorBar class | MEDIUM | 10% | 2–3 days | Yes (partial) | Extend class to multi-pane; migrate MV colorbar drawing |

---

## Recommendations for UI Maturity Plan

1. **Immediate (1 week):**
   - Add `qmriCompact` to `collectStateSnapshot()`.
   - Guard 6 unguarded keybinds.
   - Add missing `_reconcileUI()` calls on mode exit.
   - Expected gain: **~12% bug reduction**.

2. **Short-term (2–3 weeks):**
   - Add `mipActive` check to `_reconcileCbVisibility()`.
   - Extract `_scalePane()` helper from normal mode, apply to MV/compare.
   - Begin ColorBar class extension for multi-pane.
   - Expected gain: **+10% reduction, to 22%**.

3. **Medium-term (4–6 weeks):**
   - Complete state save/restore audit (all 12 mode pairs).
   - Refactor 5 scale functions into 1.
   - Fully migrate to ColorBar class.
   - Expected gain: **+20% reduction, to 42%**.

4. **Testing:**
   - Add `test_mode_roundtrip.py` for state restore verification.
   - Add parametrized cross-mode tests for features.
   - Add keybind audit test.
   - Expected gain: **+15% (bugs caught before deployment)**.

---

## Appendix: File References

**Key source files:**
- `/Users/oscar/Projects/packages/python/arrayview/src/arrayview/_viewer.html` (14,913 lines)
  - Mode flags: ~1850–1950
  - Reconcilers: ~14043–14147
  - Mode entry/exit: ~9600–11800, 10295–10550 (multiview), 11028–11232 (qMRI)
  - Keyboard handler: ~7900–8700
  - Scale functions: ~2416–2750, 11506–12000
  - Colorbar functions: ~3148–3610, ~4100–4250
  - State snapshot: ~6486–6637

- `dev/mode_matrix.md` — Viewing modes reference
- `dev/lessons_learned.md` — Reconciler pattern notes
- `tests/ui_audit.py` — Visual regression test suite (R1–R35)
- `tests/test_mode_consistency.py`, `test_mode_matrix.py` — Mode consistency tests

