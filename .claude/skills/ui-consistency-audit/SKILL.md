---
name: ui-consistency-audit
description: Use when the user explicitly requests a full visual audit, when validating UI work for a release, or when diagnosing a cross-mode visual regression in arrayview. Proactively identifies all affected mode combinations, prescribes per-mode behavior, checks for UI clashes, and runs a Playwright-based visual audit after implementation. Absorbs and replaces the modes-consistency skill.
---

# ArrayView UI Consistency Audit

## Rule

This is the **full** visual audit path. Do not invoke it by default for every UI change. During normal feature development, do targeted verification for the specific area touched. Use this skill when the user explicitly asks for a broad visual check, when a regression spans modes/layouts, or when validating UI work for a release. This skill has two phases: a **proactive phase** (before coding) that plans the cross-mode behavior, and a **reactive phase** (after coding) that runs an automated visual audit.

---

## Phase 1: Proactive (Before Coding)

### Step 1 — Classify the Change

What area of the UI is being touched?

| Category | Examples |
|----------|---------|
| **Canvas rendering / sizing / zoom** | scaleCanvas, zoom cap, compact mode, miniview |
| **Chrome element** | colorbar, dimbar, eggs, array name, logo, miniview, colormap previewer |
| **Keyboard shortcut** | new or modified shortcut |
| **Mode entry/exit** | toggling multiview, qMRI, compare, diff, ROI, zen, compact |
| **Overlay feature** | ROI, vector field, histogram/Lebesgue, hover tooltip |
| **Multi-array layout** | compare pane alignment, diff center pane, grid layout |

### Step 2 — Enumerate Affected Modes

Use the code-area-to-mode mapping (Section 7) to list every mode combination that could be affected. Do not skip modes — if in doubt, include them.

### Step 3 — Prescribe Behavior Per Mode

For **each** affected mode, explicitly state:

1. How the feature behaves in this mode
2. Whether it's the same as normal mode or different (and why)
3. Which UI elements it could clash with (use the design rules in Section 5)
4. What happens at fit-to-window zoom AND zoomed-in (compact/miniview) zoom

### Step 4 — Check Against Design Rules

Walk through the principles (Section 5a) and specific rules (Section 5b). Flag any potential violations.

### Step 5 — Get Approval

Present the mode-by-mode behavior plan to the user before implementing.

---

## Phase 2: Reactive (After Coding)

After implementation, run the visual audit:

```bash
# Tier 1 only (quick, ~15 screenshots × 2 zoom levels)
uv run python tests/ui_audit.py --tier 1

# Full audit (tier 1 + tier 2, ~40 screenshots × 2 zoom levels)
uv run python tests/ui_audit.py

# Specific subset (when you know which modes are affected)
uv run python tests/ui_audit.py --tier 2 --subset zoom,compact,compare

# Update baselines after intentional changes
uv run python tests/ui_audit.py --update-baselines
```

**Output:** Terminal summary (pass/fail per combination) + screenshots in `tests/ui_audit/screenshots/`. Visual diffs in `tests/ui_audit/diffs/` when baselines exist.

### Step 3 — Visual Design Inspection (MANDATORY)

DOM assertions catch mechanical violations. They **cannot** catch:
- Bad layout decisions (elements in wrong position, wasted screen space)
- Overlapping elements not covered by a specific rule
- Aesthetically wrong choices (miniview floating in a weird spot, panes misaligned)
- Features that technically work but look broken to a human

After the audit script runs, you MUST:

1. **Read every screenshot** from the affected tier using the Read tool (it renders PNGs visually)
2. **For each screenshot, evaluate as a UI designer:**
   - Are elements logically positioned? Would a user know where to look?
   - Is screen real estate maximized? Are canvases as large as they can be?
   - Do overlapping or adjacent elements clash? (colorbars, minimaps, eggs, labels)
   - Is the layout consistent with the same feature in other modes?
   - Would this look good on a 13" laptop screen?
3. **Report findings** — list every issue found, with the screenshot name and a concrete description
4. **Do NOT suppress assertions to make the audit pass.** If an assertion fails, the UI has a bug. Fix the UI, not the assertion. If the assertion is genuinely wrong (testing the wrong thing), fix the assertion AND explain why.

**Red flags for this step:**
- "The assertion failed so I marked it as zoomed=True" → NO. That's suppressing a real bug.
- "All assertions passed so the UI is fine" → NO. Assertions are a minimum bar, not a complete check.
- "I didn't look at the screenshots because the report said PASS" → NO. Always look.

**This is the verification step.** The `superpowers:verification-before-completion` skill requires running this before claiming work is done.

---

## Section 3: Mode Matrix & Tiers

### Tier 1 — Always Check

| Array config | Mode | Screenshot name |
|---|---|---|
| 1× 3D | normal (fit) | `t1_single_normal_fit` |
| 1× 3D | normal (zoomed) | `t1_single_normal_zoom` |
| 1× 3D | compact | `t1_single_compact` |
| 1× 3D | multiview | `t1_single_multiview_fit` |
| 1× 3D | multiview (zoomed) | `t1_single_multiview_zoom` |
| 1× 4D (dim0=5) | qMRI full | `t1_single_qmri_full` |
| 1× 4D (dim0=5) | qMRI compact | `t1_single_qmri_compact` |
| 1× 3D | zen | `t1_single_zen` |
| 2× 3D | compare | `t1_compare_fit` |
| 2× 3D | compare (zoomed) | `t1_compare_zoom` |
| 2× 3D | diff (A-B) | `t1_compare_diff` |
| 2× 3D | compare + multiview | `t1_compare_multiview` |
| 1× 3D | normal + ROI | `t1_single_roi` |
| 1× 3D | compact + ROI | `t1_compact_roi` |

### Tier 2 — Check When Relevant

| Array config | Mode | Trigger keywords |
|---|---|---|
| 1× 2D | normal | `2d, basic` |
| 1× 4D | normal | `4d, ndim` |
| 1× 4D | mosaic (z) | `mosaic, z-grid` |
| 1× complex | normal | `complex, dtype` |
| 2× 3D | diff |A-B| | `diff` |
| 2× 3D | diff |A-B|/|A| | `diff` |
| 2× 3D | overlay | `overlay, wipe, diff` |
| 2× 3D | wipe | `wipe, diff` |
| 2× 3D | registration | `registration` |
| 2× 4D | compare + qMRI | `qmri, compare` |
| 2× 3D | diff + ROI | `roi, diff` |
| 3× 3D | compare grid | `compare, grid, multi-array` |
| 3× 4D | compare + qMRI | `qmri, compare, multi-array` |
| 3× 3D | compare + multiview | `multiview, compare, multi-array` |
| 4× 3D | compare grid | `compare, grid, multi-array` |
| 3-4× 4D | compare + compact qMRI | `qmri, compare, compact, multi-array` |
| 1× 3D + vfield | vector field normal | `vector, vfield` |
| 1× 3D + vfield | vector field + ROI | `vector, roi` |
| 2× 3D (1+vfield) | compare with vector field | `vector, compare` |

---

## Section 4: The Six+ Modes (Reference)

| Mode | State flag | Scale function | Entry | Canvas elements |
|------|-----------|---------------|-------|-----------------|
| **Normal** | (default) | `scaleCanvas()` | — | `#canvas` |
| **Compact** | compact mode active | `scaleCanvas()` | K / auto on zoom | `#canvas` + miniview |
| **Multi-view** | `multiViewActive` | `mvScaleAllCanvases()` | v | `.mv-canvas` × 3 |
| **Compare** | `compareActive` | `compareScaleCanvases()` | B / P | `.compare-canvas` × 2–6 |
| **Diff** | `diffMode > 0` | `compareScaleCanvases()` | X (in compare) | `#compare-diff-canvas` |
| **Registration** | `registrationMode` | `compareScaleCanvases()` | R (in compare) | 3rd `.compare-canvas` |
| **qMRI** | `qmriActive` | `qvScaleAllCanvases()` | q | `.qv-canvas` × 3–6 |
| **Compact qMRI** | qMRI compact toggle | `qvScaleAllCanvases()` | q (second press) | `.qv-canvas` × 3 (T1,T2,|PD|) |
| **Zen** | zen mode | — | F | fullscreen, no chrome |
| **Vector field** | vectorfield attached | — | U toggle arrows | arrows on canvas |

**Orthogonal overlays** (can be active in any mode):
- **ROI** (A key): rect → circle → freehand → off
- **Histogram/Lebesgue** (w key): expanded colorbar with distribution
- **Hover tooltip** (H key): pixel value follows cursor
- **Overlay mask** (`overlay_sid` URL param): composited server-side

---

## Section 5a: Design Principles

1. **No overlapping chrome** — UI elements must never visually overlap each other or the canvas. If two elements compete for the same space, one must hide (e.g. colorbar hides when histogram is open).

2. **Consistent positioning** — an element appearing in mode A should appear in the same logical position in mode B. The miniview overlay and the mini 3D pane previewer must share a position (top-right).

3. **Zoom never clips** — canvases must stay within viewport bounds at all zoom levels. In multi-canvas modes, canvases grow until constrained, then zoom together with a single shared miniview (top-right, showing one representative slice).

4. **Maximize canvas real estate** — chrome is minimal and collapses when not needed. On a laptop screen, the array data is always the primary focus. No persistent toolbars or buttons.

5. **Mode transitions are reversible** — toggling a mode on then off returns to the exact prior state. No leftover artifacts, no shifted elements, no orphaned DOM.

6. **Orthogonal features compose cleanly** — ROI, vector field, histogram, colorbar, hover tooltip work independently. Enabling one must not break or interfere with another. ROI mode should be possible in any mode combination.

7. **Multi-array consistency** — when multiple arrays are loaded, all canvases zoom/scroll/navigate together. Colormaps on side panes match; diff center pane uses an independent colormap and dynamic range.

8. **Compare panes stay close** — pane slots shrink-wrap their canvas viewport when zoomed out so that panes remain adjacent for easy visual comparison. Never use fixed pane widths that create large gaps between canvases.

---

## Section 5b: Specific Rules (Growing Catalog)

These rules are codified as DOM assertions in `tests/ui_audit.py`. When a new inconsistency is found and fixed, add a rule here AND a corresponding assertion in the audit script.

| # | Rule | DOM assertion |
|---|------|---------------|
| R1 | Colorbar hides when histogram/Lebesgue is open | `#slim-cb-wrap` has `display:none` or `opacity:0` when Lebesgue mode active |
| R2 | Miniview position matches mini 3D pane previewer | Miniview overlay bounding box is anchored top-right, same position as multiview pane previewer |
| R3 | All canvases within viewport | Every `.compare-canvas`, `.mv-canvas`, `.qv-canvas`, `#canvas` bounding box fits within `window.innerWidth × innerHeight` |
| R4 | Diff colormaps: sides same, center different | Colormap name on diff center pane differs from side panes |
| R5 | Diff dynamic range is independent | vmin/vmax on diff center canvas differs from side panes |
| R6 | Eggs don't overlap colorbar | Bounding boxes of `#mode-eggs` children don't intersect `#slim-cb-wrap` |
| R7 | Fullscreen mode overlays chrome on canvas | `#info`, `#slim-cb-wrap` have `position:fixed` and glassmorphic styling when `body.fullscreen-mode` is active |
| R8 | ROI hover info within viewport | `.cv-pixel-info` bounding box is within viewport |
| R9 | No element jump on mode toggle | Pressing a mode toggle key then pressing it again returns bounding boxes to ±2px of original |
| R10 | Compare panes aligned | All `.compare-canvas` elements share the same `y` coordinate (horizontal layout) or same `x` (vertical layout) |
| R11 | Vector field arrows hidden when U toggled off | No arrow SVG/canvas elements visible after U press |
| R12 | Histogram not visible when colorbar is in slim mode | Lebesgue expanded state is mutually exclusive with slim colorbar |
| R13 | Colorbars don't overlap each other | No pair of colorbar elements (slim, compare-pane, mv, qv) have intersecting bounding boxes |
| R14 | Minimap within viewport and not overlapping colorbars | `#minimap-canvas` bounding box is within viewport and doesn't intersect any colorbar |
| R15 | Colorbar labels flank horizontally in all modes | Every `.cb-island`, `.compare-pane-cb-island`, `.qv-cb-island` uses `flex-direction: row` with vmin/vmax spans flanking the gradient canvas — never column layout with labels below |
| R16 | Per-pane colorbar width matches canvas viewport width (diff mode only) | Each `.compare-pane-cb` and `.qv-cb` canvas CSS width equals its data canvas viewport width. Per-pane colorbars only visible in diff mode. |
| R17 | Fullscreen mode eggs are visible over image | `#mode-eggs .mode-badge` elements have sufficient background opacity (>= 0.5) when `body.fullscreen-mode` is active, since they overlay the canvas |
| R18 | Histogram has visible background distinct from page | When `_cbExpanded`, the histogram area has an explicit background fill matching the dynamic island aesthetic, not transparent over the page background |
| R19 | Shared colorbar visible in all non-diff compare modes | `#slim-cb-wrap` is visible when `compareActive && !diffMode`. Per-pane colorbars hidden. |
| R20 | Compare panes shrink-wrap when zoomed out | Pane slot width ≤ viewport canvas width + padding, not fixed to max available. Panes stay close for visual comparison |
| R21 | Per-pane colorbars visible only in diff mode | `.compare-pane-cb-island` elements visible only when `diffMode > 0`. Three colorbars (left, center, right) at same vertical position. |
| R22 | Array names include logo in compare mode | Each `.compare-title` contains SVG logo + name text, styled like `#array-name` |
| R23 | Fullscreen mode overlays are glassmorphic islands | When `body.fullscreen-mode`, `#info` and `#slim-cb-wrap` have glassmorphic styling with `backdrop-filter: blur(12px)` |
| R24 | Diff c/d keys are mouse-aware | When `diffMode > 0`, `c` and `d` keys affect center pane when `_mouseOverCenterPane`, side panes otherwise |
| R25 | No duplicate colorbars in multiview | `#slim-cb-wrap` hidden when `multiViewActive` or `compareMvActive` (multiview has its own per-pane colorbars) |
| R26 | Shared colorbar width capped in compare mode | `#slim-cb-wrap` width ≤ 500px in compare mode |
| R27 | Mode eggs hidden during loading | `#mode-eggs` empty/hidden until at least one frame has been received |
| R28 | Compare+multiview crosshair fade works | Crosshair lines in compare+multiview fade in/out on hover, scoped to the hovered array's panes |
| R29 | Dimbar/colorbar height sync | `#info` and `#slim-cb-wrap` offsetHeight within 4px when both visible. Enforced by `_validateUIState()` (runtime) and `run_invariant_assertions()` (Playwright) |
| R30 | Drag positions cleared on immersive exit | `_infoDragPos`, `_cbDragPos`, `_islandDragPos` all null when `!_fullscreenActive`. Enforced by `_validateUIState()` and `run_immersive_exit_assertions()` |
| R31 | No .fs-overlay outside immersive | Zero `.fs-overlay` elements when `!_fullscreenActive && !_immersiveAnimating`. Enforced by `_validateUIState()` and `run_immersive_exit_assertions()` |
| R32 | Per-pane cbs only in immersive+center | `.compare-pane-cb-island.fs-overlay` count > 0 only when `_fullscreenActive && centerMode`. Enforced by `_validateUIState()` |
| R33 | Colorbar flex-direction always row | All visible `.cb-island` computed `flex-direction === 'row'`. Enforced by CSS (`!important`) and `_validateUIState()` and `run_invariant_assertions()` |
| R34 | Shared colorbar visible in compare non-center | `#slim-cb-wrap` not `display:none` when `compareActive && !centerMode`. Enforced by `_validateUIState()` |
| R35 | Islands fully inside or outside pane | Dynamic islands must be fully inside pane bounds (immersive) or fully outside in normal flow (non-immersive) — never partially overlapping |

---

## Section 5c: Colorbar Layout Invariant

All colorbars across ALL modes follow the same structure: `[vmin span] [gradient canvas] [vmax span]` in a horizontal flex row inside a `.cb-island` container. This applies to:
- Normal mode: `#slim-cb-wrap` with `#slim-cb-vmin`, `#slim-cb`, `#slim-cb-vmax`
- Compare/diff: `.compare-pane-cb-island` with `.compare-pane-cb-val` spans + `.compare-pane-cb` canvas
- qMRI: `.qv-cb-island` with `.qv-cb-val` spans + `.qv-cb` canvas

**Never** use column layout with labels below. **Never** use `flex: 1` on colorbar canvases inside inline-flex islands (causes sizing loops). Always set explicit island width = viewport canvas width, with `box-sizing: border-box` so `.cb-island` padding doesn't inflate the width beyond the canvas. Always test with non-square arrays — square arrays hide width mismatches.

**Colorbar-to-canvas gap**: The gap between canvas and colorbar must be visually consistent across modes. Both single and compare mode use the shared `#slim-cb-wrap` colorbar positioned via `chromeGap` (measured from `#info` bottom to canvas top). Per-pane colorbars (`.compare-pane-cb-island`) are only visible in diff mode.

**Compare mode colorbar**: A single shared `#slim-cb-wrap` replaces per-pane colorbars. All arrays share colormap and dynamic range (union of all vmins/vmaxes). Histogram aggregates bins from all sessions. Per-pane colorbars only appear in diff mode (3 colorbars: left, center, right).

---

## Section 5d: Enforcement Layers

Rules are enforced through multiple layers. When adding new rules, add enforcement at the appropriate layer(s):

| Layer | Mechanism | Where | When it runs |
|-------|-----------|-------|--------------|
| **CSS** | `!important` structural constraints | `_viewer.html` CSS (line ~156) | Always — structurally impossible to violate |
| **Runtime JS** | `_validateUIState()` | `_viewer.html` JS (after `_positionFullscreenChrome`) | After every layout change, gated behind `?debug_ui=1` |
| **Playwright** | `run_invariant_assertions()` + `run_immersive_exit_assertions()` | `tests/ui_audit.py` | During `ui_audit.py` runs (all tiers) |
| **Skill** | This document (Section 5b) | `.claude/skills/ui-consistency-audit/SKILL.md` | When AI agent invokes the skill |

**Cross-reference:**

| Rule | CSS | Runtime JS | Playwright | Notes |
|------|-----|-----------|------------|-------|
| R29 (height sync) | — | ✓ | ✓ | |
| R30 (drag cleanup) | — | ✓ | ✓ | Only after immersive exit |
| R31 (fs-overlay cleanup) | — | ✓ | ✓ | Only after immersive exit |
| R32 (per-pane cb visibility) | — | ✓ | — | |
| R33 (flex-direction row) | ✓ | ✓ | ✓ | CSS makes violation impossible |
| R34 (shared cb in compare) | — | ✓ | — | |
| R35 (island containment) | — | — | — | Design principle, enforced by code review |

---

## Section 6: Common Feature Categories

### Zoom / Canvas Sizing

Four dedicated scale functions. When changing zoom behavior, check **all four**:

- `scaleCanvas(w, h)` — Normal/Compact
- `mvScaleAllCanvases()` — Multi-view
- `compareScaleCanvases()` — Compare / Diff / Registration
- `qvScaleAllCanvases()` — qMRI

### Eggs (Mode Badges)

Positioned by `positionEggs()` which branches by mode:
- Normal: uses `#slim-cb-wrap` bounding box if visible, else canvas bottom
- Multi-view: uses `#mv-cb-wrap` or estimates 36px below panes
- Compare: walks `.compare-canvas-wrap` rects for tallest pane bottom
- qMRI: uses `.qv-canvas-wrap` rects + 36px

### Colorbar

Separate per-mode drawing functions — no shared abstraction:

| Mode | Function |
|------|----------|
| Normal | `drawSlimColorbar(markerFrac)` |
| Compare | `drawComparePaneCb(idx)` + `drawAllComparePaneCbs()` |
| Diff | `drawDiffPaneCb(vmin, vmax)` |
| Registration | `drawRegBlendCb()` |
| Multi-view | `drawMvCbs()` |
| qMRI | Inline per view in `qvRender()` |

### Keyboard Shortcuts

The `keydown` handler on `#keyboard-sink` dispatches by active mode. New shortcuts must:
1. Not conflict with mode-specific keys
2. Have explicit guards when mode-specific (e.g., `if (!compareActive) return;`)
3. Fall through correctly when multiple modes are active

---

## Section 7: Code-Area-to-Mode Mapping

Use this to determine which tier 2 subsets to audit based on what code you're touching.

| Code area | Tier 2 subsets to include |
|-----------|--------------------------|
| `scaleCanvas` / zoom / `userZoom` | `zoom, compact` — all zoom-related modes |
| `compareScaleCanvases` / `.compare-` | `compare, diff, registration, grid, multi-array` |
| `mvScaleAllCanvases` / `.mv-` | `multiview` — single + compare multiview |
| `qvScaleAllCanvases` / `.qv-` / `qmri` | `qmri, compare` — single + compare qMRI, compact qMRI |
| `positionEggs` / `renderEggs` / `#mode-eggs` | all modes (eggs appear everywhere) |
| `drawSlimColorbar` / `#slim-cb` | `zoom, compact, diff` — colorbar visibility rules |
| Lebesgue / histogram / `w` key | `zoom, compact, diff` — mutual exclusivity with colorbar |
| ROI / `A` key / `.roi-` | `roi` — ROI across all modes |
| `diffMode` / `X` key / `#compare-diff` | `diff` — all diff variants |
| Vector field / `U` key / arrows | `vector` — normal, ROI, compare |
| `#minimap` / miniview | `zoom, compact, multiview` — position consistency |
| Overlay / `overlay_sid` | `overlay` — server-side compositing |
| Layout / `G` key / grid | `grid, multi-array` — 3-4 array layouts |

---

## Section 8: Test Arrays

The audit script creates these arrays to exercise different code paths:

| Array | Shape | Purpose |
|-------|-------|---------|
| `arr_3d` | (20, 64, 64) float32 | Primary test array, 3D navigation |
| `arr_3d_b` | (20, 64, 64) float32 | Second 3D for compare |
| `arr_2d` | (100, 80) float32 | 2D baseline |
| `arr_4d` | (5, 20, 32, 32) float32 | qMRI eligible (dim0=5) |
| `arr_4d_b` | (5, 20, 32, 32) float32 | Second 4D for compare qMRI |
| `arr_complex` | (20, 32, 32) complex64 | Complex dtype, triggers mag/phase/real/imag |
| `arr_3d_qmri3` | (3, 20, 32, 32) float32 | 3-panel qMRI (dim0=3) |
| `vf_3d` | (20, 64, 64, 3) float32 | Vector field (N+1 dims, last dim=3) |
| `arr_3d_c` | (20, 64, 64) float32 | Third array for 3-pane compare |
| `arr_3d_d` | (20, 64, 64) float32 | Fourth array for 4-pane grid |

---

## Red Flags — STOP

- "I only implemented it for normal mode" → enumerate all affected modes now
- "The colorbar is visible but the histogram is also showing" → R1 violation
- "It works but I didn't check zoomed-in" → run both zoom levels
- "ROI mode isn't relevant here" → ROI is always relevant, it's orthogonal
- "Compare mode doesn't need this" → if it affects the canvas or chrome, compare needs it
- "I'll update the audit script later" → update it in the same task as the rule
- "The screenshots look fine to me" → also check the DOM assertions passed
- "The assertion failed so I set zoomed=True to skip it" → that's suppressing a real bug, fix the UI
- "All assertions passed so no need to look at screenshots" → assertions are a minimum bar, always visually inspect
- "Vector field is separate" → still check it doesn't clash with the new feature
