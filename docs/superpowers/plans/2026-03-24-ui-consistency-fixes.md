# UI Consistency Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 10 UI layout issues identified by visual audit across compare, diff, multiview, qMRI, and compact modes.

**Architecture:** All fixes are in `src/arrayview/_viewer.html` (single-file viewer with embedded JS/CSS). Changes target layout/scaling functions and CSS. Each task modifies specific functions identified by line number ranges. Tasks are ordered by dependency — diff layout first (since other compare-mode fixes depend on the 3-column layout working), then colorbar/miniview fixes, then grid fixes, then polish.

**Tech Stack:** HTML/CSS/JS (vanilla), Playwright (verification via `tests/ui_audit.py`)

**Single file caveat:** All tasks modify `_viewer.html`. Execute sequentially — do NOT parallelize.

---

### Task 1: Diff mode — horizontal 3-pane layout [A] [A-B] [B]

**Files:**
- Modify: `src/arrayview/_viewer.html` — `compareScaleCanvases()` (lines 1616-1744), diff pane CSS (lines 412-414, 443)

**Problem:** When diffMode > 0 with 2 panes, the layout calculates `cols = 3` correctly (line 1632), but the diff center pane ends up below the viewport at y=871 in a 900px window. The second data pane (array B) is pushed off-screen.

**Fix:** The diff center pane (`#compare-diff-pane`) uses CSS `order: 1` to sit between panes 0 and 2. The issue is that the pane sizing doesn't account for the 3-column layout properly — the slot width calculation uses `baseCols` (2) for sizing but `cols` (3) for the grid. The pane flex basis must be recalculated for 3 columns when the center pane is visible.

- [ ] **Step 1:** Read the current `compareScaleCanvases()` implementation and the diff pane HTML structure to identify the exact sizing bug.
- [ ] **Step 2:** Fix `compareScaleCanvases()` so that when `hasCenterPane` is true and the effective column count is 3, the slot width divides the available space by 3 (not 2). The diff pane should also receive the same slot width as the data panes.
- [ ] **Step 3:** Verify the fix by running `uv run python tests/ui_audit.py --tier 1` and visually inspecting `t1_compare_diff.png` — all 3 panes should be horizontally aligned.
- [ ] **Step 4:** Commit.

---

### Task 2: Compare-zoom — clamp colorbars to pane width

**Files:**
- Modify: `src/arrayview/_viewer.html` — `drawComparePaneCb()` (lines 1974-2023)

**Problem:** At zoom > 100%, each per-pane colorbar's width is set to `parseInt(cv.style.width)` which is the full (zoomed) canvas width. When two canvases overflow the viewport, their colorbars overlap in the center.

**Fix:** Clamp `cssW` to the visible portion of the pane within the viewport. Use the canvas's `.compare-canvas-inner` clip width (stored in `dataset.vpW`) instead of the full canvas width.

- [ ] **Step 1:** In `drawComparePaneCb(idx)`, after computing `cssW`, clamp it: if the pane's `.compare-canvas-inner` has a `vpW` dataset value that's smaller than `cssW`, use that instead. Also set `cbCanvas.style.maxWidth` to the inner viewport width.
- [ ] **Step 2:** Verify by running audit: `t1_compare_zoom.png` should show non-overlapping colorbars.
- [ ] **Step 3:** Commit.

---

### Task 3: Compare-zoom — per-canvas miniview inset

**Files:**
- Modify: `src/arrayview/_viewer.html` — `updateMiniMap()` (lines 8439-8507), minimap CSS (lines 617-625)

**Problem:** In zoomed compare mode, a single miniview floats in the top-right corner of the viewport, outside both canvases. Each canvas should have its own miniview inset in its top-right corner.

**Fix:** When `compareActive && _cmpIsOverflowing()`:
1. Create/reuse one minimap per visible compare pane (up to the number of active panes).
2. Position each minimap absolutely inside its `.compare-canvas-inner` container (top-right corner, inset by ~8px).
3. Each minimap renders the downscaled frame for that specific pane.
4. Hide the global `#mini-map` when per-pane minimaps are active.

- [ ] **Step 1:** Add CSS for `.compare-mini-map` — positioned absolute within `.compare-canvas-inner`, top-right, 80×80px, same styling as `#mini-map`.
- [ ] **Step 2:** Modify `updateMiniMap()`: when `compareShouldShow`, iterate over active compare panes and create/update per-pane minimaps instead of the single global one. Each renders its pane's `compareFrames[i].imageData`.
- [ ] **Step 3:** Add viewport rectangles to each per-pane minimap (same logic as `updateMiniMapViewport()` but per-pane).
- [ ] **Step 4:** Verify: `t1_compare_zoom.png` should show each canvas with its own miniview inset.
- [ ] **Step 5:** Commit.

---

### Task 4: Multi-array grid centering + higher default zoom

**Files:**
- Modify: `src/arrayview/_viewer.html` — `compareScaleCanvases()` (lines 1616-1744), `compareMvScaleAllCanvases()` (lines 6940-6980), `qvScaleAllCanvases()` (lines 7087-7126)

**Problem:** Multi-array grids (3-4 arrays) appear shifted and small. The default zoom factor (0.65) leaves too much dead space when there are many panes.

**Fix:**
1. In `compareScaleCanvases()`, ensure the pane grid is horizontally centered (check `#compare-panes` flex centering).
2. In `compareMvScaleAllCanvases()` and `qvScaleAllCanvases()` (which handle compare-multiview and compare-qMRI), increase the default scale factor from `0.65` to something like `0.85` when there are 3+ arrays, so canvases fill more of the available space.
3. Reduce `sidePad` from 120px/80px to something tighter for multi-array scenarios.

- [ ] **Step 1:** In `compareMvScaleAllCanvases()`, change `defaultScale = absMaxScale * 0.65` to use a higher factor (e.g. `0.85`) when `nRows >= 3`.
- [ ] **Step 2:** In `qvScaleAllCanvases()`, same adjustment when in compare-qMRI mode with multiple rows.
- [ ] **Step 3:** Reduce padding values (`sidePad`, `colGap`, `rowGap`) for multi-array grids.
- [ ] **Step 4:** Verify: `t2_compare_3_multiview.png` and `t2_compare_34_qmri_compact.png` should show larger, better-centered canvases.
- [ ] **Step 5:** Commit.

---

### Task 5: Multi-array qMRI — shared colorbars at bottom per column

**Files:**
- Modify: `src/arrayview/_viewer.html` — qMRI layout builder (in `enterQmri()` / `enterCompareQmri()`), `qvScaleAllCanvases()`, `drawQvSlimCb()`

**Problem:** In compare-qMRI mode, each pane has its own tiny colorbar eating vertical space. With 3 arrays × 5 maps = 15 colorbars, this wastes significant space.

**Fix:**
1. In compare-qMRI mode (multiple rows), hide per-pane colorbars on non-bottom rows.
2. Only show one colorbar per map column at the bottom of the grid (the bottom row's colorbars).
3. Reduce gap/padding between panes.

- [ ] **Step 1:** In the compare-qMRI layout builder, add a CSS class `.qv-row-inner` (non-bottom rows) that hides `.qv-cb` and `.qv-cb-labels` via `display: none`.
- [ ] **Step 2:** Reduce `.qv-row` gap from 24px to 8px and `#qmri-view-wrap` gap from 14px to 6px when in compare-qMRI mode.
- [ ] **Step 3:** Update `qvScaleAllCanvases()` to account for fewer colorbars in height calculation.
- [ ] **Step 4:** Verify: `t2_compare_qmri.png` should show tighter grid with colorbars only at bottom.
- [ ] **Step 5:** Commit.

---

### Task 6: Compare-multiview — add missing shared colorbar

**Files:**
- Modify: `src/arrayview/_viewer.html` — `enterCompareMv()` (lines 6649-6801), `compareMvScaleAllCanvases()` (lines 6940-6980)

**Problem:** Compare-multiview mode (v key in compare) shows no colorbar at all.

**Fix:** After building the compare-multiview grid, show the shared `#slim-cb-wrap` colorbar below the grid (same approach as single-array multiview which uses `#mv-cb-wrap`). Or reuse the existing `drawCmvSlimCb()` pattern and ensure the shared colorbar is visible.

- [ ] **Step 1:** In `enterCompareMv()`, ensure `#slim-cb-wrap` or equivalent is made visible and positioned below the grid.
- [ ] **Step 2:** In `compareMvScaleAllCanvases()`, redraw the shared colorbar after scaling.
- [ ] **Step 3:** Verify: `t1_compare_multiview.png` should show a colorbar below the grid.
- [ ] **Step 4:** Commit.

---

### Task 7: qMRI 5-panel — uniform size, 2×3 grid layout

**Files:**
- Modify: `src/arrayview/_viewer.html` — `qvScaleAllCanvases()` (lines 7087-7126), qMRI layout builder

**Problem:** The 5-panel qMRI has 3 panes on top (larger) and 2 on bottom (smaller, centered). All should be the same size in a 2×3 grid with the bottom-right cell empty.

**Fix:**
1. Change the layout from "centered bottom row" to a strict 2×3 grid where the bottom row is left-aligned.
2. All panes use the same scale factor (they already do via `qvScaleAllCanvases()` — the size difference is from the centering causing different flex behavior).
3. For n=5, add an invisible spacer pane in position 6 to fill the grid.

- [ ] **Step 1:** In the qMRI layout builder, when n=5, add an empty invisible `.qv-pane` spacer as the 6th element to make a full 2×3 grid.
- [ ] **Step 2:** Change `.qv-row` from `justify-content: center` to `justify-content: flex-start` (or keep center but ensure both rows have 3 elements so they align).
- [ ] **Step 3:** Verify: `t1_single_qmri_full.png` should show 5 same-sized panes in a 2×3 grid with bottom-right empty.
- [ ] **Step 4:** Commit.

---

### Task 8: Compact mode colorbar — remove vmin/vmax labels, tighter container

**Files:**
- Modify: `src/arrayview/_viewer.html` — compact colorbar code (lines 2143-2186), compact CSS (lines 534-541)

**Problem:** In compact mode, the vmin/vmax labels make the colorbar container larger and ugly. The user wants a clean slim bar with no labels and a tighter background container.

**Fix:**
1. In the compact-mode branch of `drawSlimColorbar()`, hide `slimCbLabels` (`display: none`).
2. Reduce padding on `.compact-overlay` from `3px 6px` to `2px 4px`.
3. Tighten the wrap height to just the colorbar strip.

- [ ] **Step 1:** In the compact-mode branch (around line 2180), set `slimCbLabels.style.display = 'none'` instead of showing them.
- [ ] **Step 2:** Reduce `.compact-overlay` padding and the wrap width calculation (remove the 12px that accounts for label padding).
- [ ] **Step 3:** Verify: `t1_single_compact.png` should show a clean colorbar with no labels.
- [ ] **Step 4:** Commit.

---

### Verification

After all tasks:

- [ ] Run full audit: `uv run python tests/ui_audit.py`
- [ ] Update baselines: `uv run python tests/ui_audit.py --update-baselines`
- [ ] Run existing tests: `uv run pytest tests/test_api.py -x` and `uv run python tests/visual_smoke.py`
- [ ] Visually inspect all screenshots in `tests/ui_audit/screenshots/`
