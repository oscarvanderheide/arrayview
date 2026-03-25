# Unified Chrome Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the viewer chrome layout across 1–6 arrays so the vertical stack is always: dim bar → name(s) → canvas(es) → shared colorbar.

**Architecture:** Reorder single-mode chrome (swap dim bar and name), replace per-pane compare colorbars with a single shared colorbar, replace compact mode with manual fullscreen (K key), and add diff-mode-specific per-pane colorbars with mouse-aware keyboard controls.

**Tech Stack:** Single-file HTML/JS (`_viewer.html`), CSS, Canvas 2D API

**Spec:** `docs/superpowers/specs/2026-03-25-unified-chrome-layout-design.md`

---

## File Map

All changes are in `src/arrayview/_viewer.html` unless noted.

| Area | Lines (approx) | What changes |
|------|----------------|--------------|
| CSS: `#array-name` | 190-196 | Reorder in flex layout |
| CSS: `.compare-title` | 528-529 | Restyle to match `#array-name` |
| CSS: `.compare-pane-cb-island` | 536 | Remove or repurpose for diff mode only |
| CSS: compact mode rules | 624-640 | Replace with fullscreen mode rules |
| CSS: `#compare-view-wrap` | 511-527 | Adjust for shared colorbar |
| HTML: `#array-name`, `#info` | 801-830 | Swap vertical order |
| HTML: compare pane structure | 850-1042 | Per-pane colorbars hidden by default, shown in diff |
| JS: `uiReserveV()` | 1742-1762 | Update for new chrome order |
| JS: `compareScaleCanvases()` | 1843-1999 | Remove per-pane colorbar space, position shared CB |
| JS: `drawSlimColorbar()` | 2473-2640 | Remove compare-mode early return, handle all modes |
| JS: `drawComparePaneCb()` | 2294-2415 | Restrict to diff mode only |
| JS: `drawAllComparePaneCbs()` | 2417-2425 | Restrict to diff mode only |
| JS: Lebesgue/histogram | 2910, 5651 | Remove `compareActive` guard, aggregate bins |
| JS: compact mode | 9343-9424 | Replace with fullscreen mode |
| JS: keyboard handler | 5328+ | Add K key, modify c/d for diff mouse awareness |
| Tests: `test_browser.py` | various | Update for new layout |

---

### Task 1: Swap dim bar and array name in single mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — HTML structure (~lines 798-835), CSS (~lines 190-196), `uiReserveV()` (~line 1742)

The single-mode vertical order must change from `[name, dimbar, canvas, colorbar]` to `[dimbar, name, canvas, colorbar]`.

- [ ] **Step 1: Read the current HTML structure around lines 798-835**

Understand the exact order of `#array-name`, `#info` (dim bar), `#data-info`, and the canvas viewport.

- [ ] **Step 2: Swap `#info` and `#array-name` in the HTML**

Move the `#info` div (dim bar island) ABOVE `#array-name` in the DOM order. The flex container `#top-bar` (or equivalent parent) uses `flex-direction: column`, so DOM order = visual order.

- [ ] **Step 3: Update `uiReserveV()`**

The function at ~line 1742 calculates top chrome height. It references `#array-name` and `#info` — ensure the height calculation is still correct after the swap. The total reserved height should be the same, just the order changes.

- [ ] **Step 4: Verify `drawSlimColorbar()` positioning**

The colorbar positioning at ~line 2569 uses `canvasRect.top - infoEl.getBoundingClientRect().bottom` for the gap. After the swap, `#info` is above `#array-name`, so the gap calculation between canvas and colorbar should still reference the canvas rect, not the info rect. Check and fix if needed.

- [ ] **Step 5: Test manually — load single array, verify layout**

Run server, load a single array, confirm: dim bar on top, name below, canvas, colorbar.

> **Note:** Visual regression test snapshots (in `tests/snapshots/`) will be stale after this change. Delete them to regenerate baselines once the full redesign is stable (after Task 7). Do not try to fix snapshot tests incrementally.

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: swap dim bar above array name in single mode"
```

---

### Task 2: Shared colorbar in compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — `drawSlimColorbar()` (~line 2473), `drawComparePaneCb()` (~line 2294), `compareScaleCanvases()` (~line 1843), per-pane colorbar CSS (~line 536)

Remove per-pane colorbars. Make `drawSlimColorbar()` work for compare mode with a single shared colorbar.

> **Note:** Diff mode will be visually broken (no per-pane colorbars) until Task 6 re-introduces them for diff only. Do not test diff mode between Tasks 2–5.

- [ ] **Step 1: Remove the early return in `drawSlimColorbar()`**

At ~line 2475, `drawSlimColorbar()` has:
```js
if (compareActive) {
    drawAllComparePaneCbs();
    drawCompareColorbar();
    slimCbCanvas.style.display = 'none';
    if (wrap) { wrap.style.display = 'none'; ... }
    return;
}
```
Remove this block so the shared colorbar draws in compare mode too. The shared colorbar should position below all compare panes.

- [ ] **Step 2: Position shared colorbar below compare panes**

In `drawSlimColorbar()`, when `compareActive`, compute `canvasRect` from the `#compare-panes` element (or the bottommost pane) instead of `#canvas-viewport`.

**Enforce symmetric gap:** Measure the gap between the name row bottom and the canvas top. Use the same gap value between the canvas bottom and the colorbar top. Store this as a variable (e.g. `chromeGap`) and use it for both positions.

- [ ] **Step 3: Compute shared colorbar width**

The colorbar width should match the total pane grid width (per spec). Use the `#compare-panes` bounding rect width. Do NOT apply the single-mode 600px max cap — multi-array grids can be wider.

- [ ] **Step 4: Compute shared dynamic range**

The shared `vmin`/`vmax` should be the union of all loaded arrays' ranges:
```js
const sharedVmin = Math.min(...compareFrames.filter(Boolean).map(f => manualVmin ?? f.vmin));
const sharedVmax = Math.max(...compareFrames.filter(Boolean).map(f => manualVmax ?? f.vmax));
```

- [ ] **Step 5: Hide per-pane colorbars by default**

In `drawComparePaneCb()`, add an early return when not in diff mode:
```js
if (!diffMode) {
    // Hide per-pane colorbar
    if (cbIsland) cbIsland.style.display = 'none';
    return;
}
```

- [ ] **Step 6: Update `compareScaleCanvases()` padding**

Reduce `panePadY` since per-pane colorbars are gone. The bottom padding in `.compare-canvas-area` can shrink (was 32px for colorbar space, now ~8px or so). Update the CSS `padding` and the JS `panePadY` constant.

- [ ] **Step 7: Remove the `#compare-cb-wrap` shared colorbar element**

At ~line 1038, there's an old `#compare-cb-wrap` that's hidden (`display: none`). Remove it entirely — we use `#slim-cb-wrap` for everything now.

- [ ] **Step 8: Update `positionEggs()` for compare mode**

The eggs positioning at ~line 3752 references per-pane canvas-wraps including colorbars. Now it should reference the shared colorbar position instead.

- [ ] **Step 9: Remove dead code — `drawCompareColorbar()`**

This function is no longer called after removing the `compareActive` early return. Delete it entirely.

- [ ] **Step 10: Test manually — load 2 arrays, verify single shared colorbar**

> **Note:** Do not test diff mode (X key) yet — it will be broken until Task 6.

- [ ] **Step 11: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: single shared colorbar in compare mode"
```

---

### Task 3: Per-pane names in compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — `.compare-title` CSS (~line 528), compare pane HTML (~line 850+), `#array-name` behavior in compare mode

Unify the name display: per-pane name with logo, styled like `#array-name`, positioned between dim bar and canvas.

- [ ] **Step 1: Restyle `.compare-title` to match `#array-name`**

Update CSS at ~line 528:
```css
.compare-title {
    font-size: 11px; color: var(--muted); letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    max-width: 95%; cursor: grab; user-select: none;
    display: flex; align-items: center; justify-content: center; gap: 5px;
}
```

- [ ] **Step 2: Add logo SVG to each `.compare-title`**

In the `updateCompareTitles()` function (find it), prepend the logo SVG (same as in `#array-name`) to each title's innerHTML. Or add it in the HTML template and just update the text portion.

- [ ] **Step 3: Hide `#array-name` in compare mode**

When `compareActive` is true, hide `#array-name` (it's replaced by per-pane titles). In `enterCompareMode*()` functions (~lines 6330-6374), add:
```js
document.getElementById('array-name').style.display = 'none';
```
And in `exitCompareMode()`, restore it:
```js
document.getElementById('array-name').style.display = '';
```

**Important:** Verify that `drawSlimColorbar()` (modified in Task 2) does NOT use `#array-name` bounding rect for positioning in compare mode. It should reference `#compare-panes` for canvas rect. If it falls back to `#array-name` for any calculation, use the `#info` (dim bar) element instead, which remains visible.

- [ ] **Step 4: Remove "COMPARING N ARRAYS" text**

Find where the `#array-name-text` is set to "comparing N arrays" in compare mode and remove that logic. The name element is hidden in compare mode now, so this text is unnecessary.

- [ ] **Step 5: Test — load 2 arrays, verify per-pane logos + names**

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "refactor: per-pane name+logo in compare mode, remove COMPARING header"
```

---

### Task 4: Re-enable histogram/Lebesgue in compare mode

**Files:**
- Modify: `src/arrayview/_viewer.html` — histogram guards (~lines 2910, 5651), bin aggregation logic

- [ ] **Step 1: Remove `compareActive` guard on `w` key**

At ~line 5651, the `w` key handler shows a toast and returns if `compareActive`. Remove this guard so histogram mode can be activated in compare mode.

- [ ] **Step 2: Remove `compareActive` guard on Lebesgue hover**

At ~line 2910, Lebesgue mode is disabled when `compareActive`. Remove this guard.

- [ ] **Step 3: Aggregate histogram bins across sessions**

When `compareActive` and the histogram is requested, fetch histogram data for all loaded sessions and sum the bins. The existing `/histogram/{sid}` endpoint returns per-session data. The client-side code should:
```js
// Fetch histograms for all compare sessions, sum bins
const allSids = getCompareRenderSids();
const histPromises = allSids.map(sid => fetch(`/histogram/${sid}?...`).then(r => r.json()));
const allHists = await Promise.all(histPromises);
// Sum bins element-wise
const mergedBins = allHists[0].bins.map((_, i) =>
    allHists.reduce((sum, h) => sum + h.bins[i], 0)
);
```

- [ ] **Step 4: Test — load 2 arrays, press `w`, verify histogram shows with aggregated data**

- [ ] **Step 5: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: re-enable histogram in compare mode with aggregated bins"
```

---

### Task 5: Fullscreen mode (replace compact)

**Files:**
- Modify: `src/arrayview/_viewer.html` — compact mode CSS (~lines 624-640), `setCompactMode()` (~line 9359), `_checkAutoCompact()` (~line 9394), keyboard handler

- [ ] **Step 1: Remove auto-compact logic**

Delete `_checkAutoCompact()` function (~lines 9394-9414) and all calls to it. Remove the auto-compact trigger from `scaleCanvas()` and other scale functions.

- [ ] **Step 2: Rename compact mode to fullscreen mode**

Rename `_compactActive` → `_fullscreenActive`, `setCompactMode()` → `setFullscreenMode()`, `toggleCompactMode()` → `toggleFullscreenMode()`. Update the body class from `compact-mode` to `fullscreen-mode`. Update all references.

- [ ] **Step 3: Update fullscreen CSS**

Replace `body.compact-mode` CSS rules (~lines 624-640) with `body.fullscreen-mode` rules:
- Canvas fills viewport (remove chrome margins)
- Dim bar overlaid top-center: `position: fixed; z-index: 3; top: 8px; left: 50%; transform: translateX(-50%);` with `.cb-island` glassmorphic styling
- Colorbar overlaid bottom-center: same approach, `bottom: 8px`
- Per-pane name pills overlaid at top of each pane: `position: absolute; top: 6px; left: 50%; transform: translateX(-50%);` within each pane (which needs `position: relative`)
- Overlay containers: `pointer-events: none;` — interactive children (dim bar segments, colorbar canvas) get `pointer-events: auto;`
- z-index: overlays at 3 (below help overlay at 50, below tooltips at 5-8)
- Always visible, semi-transparent — no hover-reveal, no auto-hide

- [ ] **Step 4: Add K key handler**

In the keyboard handler (~line 5328+), add:
```js
} else if (e.key === 'k' || e.key === 'K') {
    toggleFullscreenMode();
    showToast(_fullscreenActive ? 'fullscreen: on' : 'fullscreen: off');
```

- [ ] **Step 5: Update fullscreen for compare mode**

When fullscreen is active AND compare mode is active:
- Panes fill viewport with minimal gap (2px)
- Per-pane name pills overlaid on each canvas (glassmorphic)
- Shared dim bar overlaid top-center
- Shared colorbar overlaid bottom-center

- [ ] **Step 6: Test — single array, press K, verify fullscreen layout**

- [ ] **Step 7: Test — 2 arrays, press K, verify fullscreen compare layout**

- [ ] **Step 8: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: fullscreen mode (K key) replaces compact mode"
```

---

### Task 6: Diff mode per-pane colorbars

**Files:**
- Modify: `src/arrayview/_viewer.html` — `drawComparePaneCb()` (~line 2294), keyboard handler for `c`/`d`, diff mode entry/exit

This task re-introduces per-pane colorbars **only** when diff mode is active.

- [ ] **Step 1: Show per-pane colorbars in diff mode**

In `drawComparePaneCb()` (modified in Task 2 to early-return when not in diff mode), implement the diff-mode colorbar logic:
- Left + right panes: draw identical colorbars using the shared colormap and shared vmin/vmax
- Center (diff) pane: draw with the diff colormap and diff vmin/vmax
- All three colorbars at the same vertical position and same physical size

- [ ] **Step 2: Hide shared colorbar in diff mode**

In `drawSlimColorbar()`, when `diffMode > 0`, hide the shared `#slim-cb-wrap` (the per-pane colorbars replace it).

- [ ] **Step 3: Track mouse position over center pane**

Use the existing `_mouseOverCenterPane` variable (~line 1381). Ensure mouseover/mouseout events on the diff canvas set this flag.

- [ ] **Step 4: Make `c` key mouse-aware in diff mode**

In the `c` key handler, when `diffMode > 0`:
- If `_mouseOverCenterPane`: cycle through diff-appropriate colormaps (already defined in `DIFF_COLORMAPS` / `ABS_DIFF_COLORMAPS` at ~line 1379)
- Otherwise: cycle the shared colormap for both side panes. Show colormap previewer under both side panes.

- [ ] **Step 5: Make `d` key mouse-aware in diff mode**

In the `d` key handler, when `diffMode > 0`:
- If `_mouseOverCenterPane`: cycle dynamic range for diff pane only
- Otherwise: cycle dynamic range for both side panes together. Show histogram aggregated from both arrays.

- [ ] **Step 6: Restore shared colorbar on diff mode exit**

When diff mode is exited (X key toggles it off), hide per-pane colorbars and restore the shared `#slim-cb-wrap`.

- [ ] **Step 7: Test — 2 arrays, press X for diff, verify 3 colorbars**

- [ ] **Step 8: Test — hover over diff pane, press c/d, verify independent control**

- [ ] **Step 9: Commit**

```bash
git add src/arrayview/_viewer.html
git commit -m "feat: diff mode per-pane colorbars with mouse-aware c/d keys"
```

---

### Task 7: Update tests

**Files:**
- Modify: `tests/test_browser.py` — update assertions for new layout
- Modify: `tests/ui_audit.py` (if exists) — update audit rules

- [ ] **Step 1: Update browser tests that check colorbar visibility**

Tests that assert per-pane colorbar visibility in compare mode need updating. The shared colorbar should be visible instead.

- [ ] **Step 2: Update browser tests that check chrome positioning**

Tests that assert `#array-name` is above `#info` need reversing.

- [ ] **Step 3: Update or remove compact mode tests**

Replace compact mode test assertions with fullscreen mode equivalents.

- [ ] **Step 4: Add test for K key fullscreen toggle**

```python
def test_fullscreen_toggle(self, loaded_viewer, sid_3d):
    page = loaded_viewer(sid_3d)
    page.keyboard.press("k")
    assert "fullscreen-mode" in page.locator("body").get_attribute("class")
    page.keyboard.press("k")
    assert "fullscreen-mode" not in page.locator("body").get_attribute("class")
```

- [ ] **Step 5: Add test for shared colorbar in compare mode**

- [ ] **Step 6: Run full test suite, fix any remaining failures**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 7: Commit**

```bash
git add tests/
git commit -m "test: update tests for unified chrome layout"
```

---

### Task 8: Update UI audit skill rules

**Files:**
- Modify: `.claude/skills/ui-consistency-audit/SKILL.md`

- [ ] **Step 1: Update R19/R20 rules from earlier in this session**

Revise R19 (colorbar gap) and R20 (pane shrink-wrap) to reflect the new shared colorbar approach.

- [ ] **Step 2: Add new rules**

- R21: Shared colorbar visible in all non-diff compare modes
- R22: Per-pane colorbars visible only in diff mode
- R23: Array names include logo in compare mode
- R24: Fullscreen mode overlays are glassmorphic islands

- [ ] **Step 3: Remove obsolete rules about per-pane colorbars**

Rules R15, R16 (per-pane colorbar width/layout) become diff-mode-only.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ui-consistency-audit/SKILL.md
git commit -m "docs: update ui-consistency-audit skill for unified chrome"
```
