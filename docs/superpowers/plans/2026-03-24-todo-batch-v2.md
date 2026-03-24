# TODO Batch (2026-03-24) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 4 independent UI improvements: miniviewer hand cursor, subtler canvas border, multiview global rotation, and play-dim coloring with independent navigation.

**Architecture:** All changes are in `src/arrayview/_viewer.html` (single-file viewer with embedded CSS/JS). Tasks 1-2 are CSS-only with minor JS. Task 3 rewrites the multiview `r` handler. Task 4 adds new state (`playingDim`) and CSS class (`.playing-dim`).

**Tech Stack:** HTML/CSS/JS (vanilla), Playwright browser tests

**Skills to check:** @viewer-ui-checklist, @modes-consistency, @vscode-simplebrowser, @invocation-consistency, @frontend-designer

---

### Task 1: Miniviewer hand cursor

**Files:**
- Modify: `src/arrayview/_viewer.html:619` (CSS), `src/arrayview/_viewer.html:8500-8512` (JS)
- Test: `tests/test_browser.py`

- [ ] **Step 1: Write failing test**

In `tests/test_browser.py`, add to the `TestViewer` class:

```python
def test_minimap_cursor_grab(self, loaded_viewer, sid_2d):
    """Mini-map should show grab cursor, grabbing while dragging."""
    page = loaded_viewer(sid_2d)
    _focus_kb(page)
    # Zoom in far enough to trigger mini-map
    for _ in range(8):
        page.keyboard.press("Equal")
        page.wait_for_timeout(80)
    page.wait_for_timeout(300)
    visible = page.evaluate(
        "() => document.querySelector('#mini-map').classList.contains('visible')"
    )
    assert visible, "mini-map should be visible after zooming in"
    cursor = page.evaluate(
        "() => getComputedStyle(document.querySelector('#mini-map')).cursor"
    )
    assert cursor == "grab", f"expected grab cursor, got {cursor}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_minimap_cursor_grab -v`
Expected: FAIL — cursor is "crosshair"

- [ ] **Step 3: Implement CSS change**

In `_viewer.html` line 619, change `cursor:crosshair` to `cursor:grab`:

```css
/* line 619 */
cursor:grab;
```

- [ ] **Step 4: Implement JS grabbing cursor on drag**

In `_viewer.html` lines 8500-8512, add cursor changes to the mousedown/mouseup handlers:

```javascript
if (_miniMap) {
    _miniMap.addEventListener('mousedown', (e) => {
        _miniMapDragging = true;
        _miniMap.style.cursor = 'grabbing';
        _miniMapJumpTo(e);
    });
    document.addEventListener('mousemove', (e) => {
        if (!_miniMapDragging) return;
        _miniMapJumpTo(e);
    });
    document.addEventListener('mouseup', () => {
        _miniMapDragging = false;
        _miniMap.style.cursor = 'grab';
    });
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_minimap_cursor_grab -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html tests/test_browser.py
git commit -m "feat: miniviewer shows grab/grabbing cursor on hover and drag"
```

---

### Task 2: Subtler canvas border

**Files:**
- Modify: `src/arrayview/_viewer.html:7-42` (CSS themes), `src/arrayview/_viewer.html:69` (border rule), `src/arrayview/_viewer.html:404` (multiview border)
- Test: `tests/test_browser.py`

- [ ] **Step 1: Write failing test**

```python
def test_border_toggle_uses_subtle_outline(self, loaded_viewer, sid_2d):
    """Border toggle (b) should use 1px gray outline, not 2px highlight."""
    page = loaded_viewer(sid_2d)
    _focus_kb(page)
    page.keyboard.press("b")
    page.wait_for_timeout(200)
    outline = page.evaluate(
        "() => getComputedStyle(document.querySelector('#canvas-viewport')).outline"
    )
    # Should be 1px, not 2px; should NOT be pure white
    assert "1px" in outline, f"expected 1px outline, got: {outline}"
    assert "rgb(255, 255, 255)" not in outline, f"border should not be pure white: {outline}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_border_toggle_uses_subtle_outline -v`
Expected: FAIL — outline is "2px solid rgb(255, 255, 255)"

- [ ] **Step 3: Verify `--canvas-border` usage**

Run: `grep -n 'canvas-border' src/arrayview/_viewer.html`
Confirm it's only used in the 4 theme declarations (lines 11, 20, 29, 38) and the two outline rules (lines 69, 404). No other consumers.

- [ ] **Step 4: Update CSS themes — set `--canvas-border` to uniform gray**

In all 4 theme blocks, change `--canvas-border` value:

```css
/* line 11 (dark) */
--canvas-border: rgba(180,180,180,0.35);

/* line 20 (light) */
--canvas-border: rgba(180,180,180,0.35);

/* line 29 (solarized) */
--canvas-border: rgba(180,180,180,0.35);

/* line 38 (nord) */
--canvas-border: rgba(180,180,180,0.35);
```

- [ ] **Step 5: Update border rules to use `--canvas-border` and 1px**

```css
/* line 69 — was: 2px solid var(--highlight) */
.canvas-bordered canvas, .canvas-bordered { outline: 1px solid var(--canvas-border); }

/* line 404 — was: 1px solid var(--border) */
.mv-pane.canvas-bordered { outline: 1px solid var(--canvas-border); }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_border_toggle_uses_subtle_outline -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/arrayview/_viewer.html tests/test_browser.py
git commit -m "fix: make canvas border subtler — 1px gray for all themes"
```

---

### Task 3: Multiview rotation as global axis swap

**Files:**
- Modify: `src/arrayview/_viewer.html:4749-4762` (r key handler)
- Test: `tests/test_browser.py`

- [ ] **Step 1: Write failing test**

```python
def test_multiview_rotate_updates_all_panes(self, loaded_viewer, sid_3d):
    """Pressing r in multiview should swap axes globally across all 3 panes."""
    page = loaded_viewer(sid_3d)
    _focus_kb(page)
    page.keyboard.press("v")
    page.wait_for_timeout(500)

    # Get initial dimX/dimY for all 3 panes
    before = page.evaluate(
        "() => window.mvViews.map(v => ({dimX: v.dimX, dimY: v.dimY, sliceDir: v.sliceDir}))"
    )

    # Press r (activeDim should be on a non-spatial dim by default in multiview)
    # First switch activeDim to a spatial dim so r works
    page.evaluate("() => { window.activeDim = window.mvDims[0]; }")
    page.keyboard.press("r")
    page.wait_for_timeout(300)

    after = page.evaluate(
        "() => window.mvViews.map(v => ({dimX: v.dimX, dimY: v.dimY, sliceDir: v.sliceDir}))"
    )

    # All panes should have changed, not just one
    changed_count = sum(1 for b, a in zip(before, after) if b != a)
    assert changed_count >= 2, (
        f"Expected at least 2 panes to change after r, but only {changed_count} changed. "
        f"Before: {before}, After: {after}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_multiview_rotate_updates_all_panes -v`
Expected: FAIL — only 1 pane changes

- [ ] **Step 3: Rewrite the multiview branch of the `r` handler**

Replace lines 4750-4762 (the `if (multiViewActive) { ... }` block) with:

```javascript
if (multiViewActive) {
    // Global axis swap: swap dim_x and dim_y, rebuild all pane definitions
    const oldDimX = dim_x, oldDimY = dim_y;
    dim_x = oldDimY; dim_y = oldDimX;
    // Transfer flip state
    const ofx = flipDims[oldDimX] || false, ofy = flipDims[oldDimY] || false;
    flipDims[dim_x] = ofy; flipDims[dim_y] = !ofx;
    // Recompute mvDims and pane definitions in-place
    const third = mvDims.find(d => d !== oldDimX && d !== oldDimY);
    mvDims = [dim_y, dim_x, third];
    const newDefs = [
        { dimX: mvDims[1], dimY: mvDims[0], sliceDir: mvDims[2] },
        { dimX: mvDims[2], dimY: mvDims[0], sliceDir: mvDims[1] },
        { dimX: mvDims[2], dimY: mvDims[1], sliceDir: mvDims[0] },
    ];
    mvViews.forEach((v, i) => {
        v.dimX = newDefs[i].dimX; v.dimY = newDefs[i].dimY;
        v.sliceDir = newDefs[i].sliceDir;
        v.seq++;
    });
    mvViews.forEach(v => { mvDrawFrame(v); mvRender(v); });
    // Update axis labels — axLblX/axLblY are strings on the view object,
    // DOM text lives in v.axesSvg querySelector('.axes-lbl-x'/'.axes-lbl-y')
    const _axLetters = ['x', 'y', 'z'];
    mvViews.forEach((v, i) => {
        const newX = _axLetters[mvDims.indexOf(v.dimX)] || '?';
        const newY = _axLetters[mvDims.indexOf(v.dimY)] || '?';
        v.axLblX = newX; v.axLblY = newY;
        if (v.axesSvg) {
            v.axesSvg.querySelector('.axes-lbl-x').textContent = newX;
            v.axesSvg.querySelector('.axes-lbl-y').textContent = newY;
        }
    });
    showStatus('rotated 90°');
    renderInfo(); saveState();
} // closing brace for if (multiViewActive)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_browser.py::TestViewer::test_multiview_rotate_updates_all_panes -v`
Expected: PASS

- [ ] **Step 5: Manual sanity check**

Run the viewer with a 3D array, enter multiview (v), press r, and verify all 3 panes update visually. Check that crosshairs and axis labels are correct.

- [ ] **Step 6: Commit**

```bash
git add src/arrayview/_viewer.html tests/test_browser.py
git commit -m "fix: multiview r now swaps axes globally across all 3 panes"
```

---

### Task 4: Space play — orange playing dim + independent navigation

**Files:**
- Modify: `src/arrayview/_viewer.html:7-42` (CSS themes), `src/arrayview/_viewer.html:85` (new class), `src/arrayview/_viewer.html:1038` (new state), `src/arrayview/_viewer.html:3376-3436` (renderInfo), `src/arrayview/_viewer.html:3976-3980` (stopPlay), `src/arrayview/_viewer.html:3989` (playNext), `src/arrayview/_viewer.html:4010-4021` (togglePlay), `src/arrayview/_viewer.html:5040-5080` (j/k handlers)
- Test: `tests/test_browser.py`

- [ ] **Step 1: Write failing tests**

```python
def test_playing_dim_gets_orange_class(self, loaded_viewer, sid_3d):
    """Playing dim should get .playing-dim class (orange), not .active-dim."""
    page = loaded_viewer(sid_3d)
    _focus_kb(page)
    page.keyboard.press("Space")
    page.wait_for_timeout(300)
    has_playing = page.evaluate("""
        () => document.querySelector('#info .playing-dim') !== null
    """)
    assert has_playing, "playing dim should have .playing-dim class during playback"
    page.keyboard.press("Space")  # stop

def test_play_allows_independent_dim_navigation(self, loaded_viewer, sid_3d):
    """During playback, user can change activeDim without affecting playingDim."""
    page = loaded_viewer(sid_3d)
    _focus_kb(page)
    page.keyboard.press("Space")
    page.wait_for_timeout(300)
    # Get the playing dim and active dim before
    playing_dim = page.evaluate("() => window.playingDim")
    active_before = page.evaluate("() => window.activeDim")
    assert playing_dim >= 0, "playingDim should be set during playback"
    # Change activeDim with h (move to previous dim)
    page.keyboard.press("h")
    page.wait_for_timeout(200)
    # Verify activeDim actually changed (h wasn't a no-op)
    active_after = page.evaluate("() => window.activeDim")
    assert active_after != active_before, (
        f"activeDim didn't change after h: still {active_after}"
    )
    # playingDim should be unchanged
    still_playing = page.evaluate("() => window.playingDim")
    assert still_playing == playing_dim, (
        f"playingDim changed from {playing_dim} to {still_playing} after pressing h"
    )
    is_playing = page.evaluate("() => window.isPlaying")
    assert is_playing, "should still be playing after changing activeDim"
    page.keyboard.press("Space")  # stop

def test_jk_on_playing_dim_stops_playback(self, loaded_viewer, sid_3d):
    """Pressing j/k on the actively playing dim should stop playback."""
    page = loaded_viewer(sid_3d)
    _focus_kb(page)
    page.keyboard.press("Space")
    page.wait_for_timeout(300)
    # Navigate activeDim to the playing dim
    playing_dim = page.evaluate("() => window.playingDim")
    page.evaluate(f"() => {{ window.activeDim = {playing_dim}; }}")
    # Press j (decrement) — should stop playback
    page.keyboard.press("j")
    page.wait_for_timeout(200)
    is_playing = page.evaluate("() => window.isPlaying")
    assert not is_playing, "playback should stop when j/k pressed on playing dim"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_browser.py -k "playing_dim or play_allows or jk_on_playing" -v`
Expected: FAIL — no `.playing-dim` class, no `playingDim` variable

- [ ] **Step 3: Add CSS — `--playing-dim` variable and `.playing-dim` class**

Add `--playing-dim` to each theme block (after `--active-dim`):

```css
/* dark (line ~12, after --active-dim: #f5c842;) */
--playing-dim: #d08770;

/* light (line ~21, after --active-dim: #c47d00;) */
--playing-dim: #c05020;

/* solarized (line ~30, after --active-dim: #b58900;) */
--playing-dim: #cb4b16;

/* nord (line ~39, after --active-dim: #ebcb8b;) */
--playing-dim: #d08770;
```

Add new class after `.active-dim` (line ~85):

```css
.playing-dim { color: var(--playing-dim); font-weight: bold; }
.playing-dim .dim-track-fill { background: var(--playing-dim); }
```

- [ ] **Step 4: Add `playingDim` state variable**

Near line 1038, add after `isPlaying`:

```javascript
let isPlaying = false, playInterval = null, playingVfieldT = false;
let playingDim = -1;  // -1 = not playing; set to dim index during playback
```

- [ ] **Step 5: Update `togglePlay()` to set `playingDim`**

At line ~4012, after `isPlaying = true;`:

```javascript
function togglePlay() {
    if (isPlaying) { stopPlay(); return; }
    isPlaying = true;
    setStatus(`▶ playing  (Space to stop · [ / ] fps: ${playFps})`);
    if (activeDim === VFIELD_T_DIM) {
        playingVfieldT = true;
        playingDim = -1; // vfield-T playback uses vfieldT, not indices[]
        playLastFrameTime = performance.now();
        playNextVfieldT();
    } else {
        playingDim = current_slice_dim;
        playNext();
    }
}
```

- [ ] **Step 6: Update `stopPlay()` to clear `playingDim`**

At line ~3979, add `playingDim = -1;`:

```javascript
function stopPlay() {
    if (_playRafId) { cancelAnimationFrame(_playRafId); _playRafId = null; }
    if (playInterval) { clearTimeout(playInterval); playInterval = null; }
    isPlaying = false; playingVfieldT = false; playingDim = -1; setStatus('');
}
```

- [ ] **Step 7: Update `playNext()` to use `playingDim` instead of `current_slice_dim`**

At line ~3989, change:

```javascript
// WAS: indices[current_slice_dim] = (indices[current_slice_dim] + 1) % shape[current_slice_dim];
indices[playingDim] = (indices[playingDim] + 1) % shape[playingDim];
```

- [ ] **Step 8: Update `renderInfo()` — add `.playing-dim` priority**

In `renderInfo()` (~line 3378), add `playingDim` awareness. Replace line 3378:

```javascript
// WAS: const active = (i === activeDim);
const active = (i === activeDim);
const playing = (playingDim >= 0 && i === playingDim);
const dimCls = playing ? 'playing-dim' : 'active-dim'; // orange if playing, yellow if active
```

Then replace every use of the literal string `'active-dim'` that depends on the `active` variable with `dimCls`. Specifically, change each `active ? 'active-dim ...'` to `(active || playing) ? dimCls + ' ...'`. Here are all 8 branches with the exact replacements:

**Branch 1 — qmri dim (line ~3381):**
```javascript
// WAS: return active ? `<span class="active-dim q-dim dim-label" ${d}>q</span>`
return (active || playing) ? `<span class="${dimCls} q-dim dim-label" ${d}>q</span>`
```

**Branch 2 — multiview spatial (line ~3392):**
```javascript
// WAS: const cls = active ? 'active-dim dim-label' : 'spatial-dim dim-label';
const cls = (active || playing) ? `${dimCls} dim-label` : 'spatial-dim dim-label';
```

**Branch 3 — multiview non-spatial (line ~3399):**
```javascript
// WAS: if (active) return `<span class="active-dim dim-label dim-chip" ${d}>${inner}</span>`;
if (active || playing) return `<span class="${dimCls} dim-label dim-chip" ${d}>${inner}</span>`;
```

**Branch 4 — dim_x (line ~3404):**
```javascript
// WAS: return active ? `<span class="active-dim dim-label" ${d}>${inner}</span>`
return (active || playing) ? `<span class="${dimCls} dim-label" ${d}>${inner}</span>`
```

**Branch 5 — dim_y (line ~3409):**
```javascript
// WAS: return active ? `<span class="active-dim dim-label" ${d}>${inner}</span>`
return (active || playing) ? `<span class="${dimCls} dim-label" ${d}>${inner}</span>`
```

**Branch 6 — dim_z (line ~3414):**
```javascript
// WAS: return active ? `<span class="active-dim dim-label" ${d}>${inner}</span>`
return (active || playing) ? `<span class="${dimCls} dim-label" ${d}>${inner}</span>`
```

**Branch 7 — default slice dim (line ~3421):**
```javascript
// WAS: return active ? `<span class="active-dim dim-label dim-chip" ${d}>${inner}</span>`
return (active || playing) ? `<span class="${dimCls} dim-label dim-chip" ${d}>${inner}</span>`
```

**Branch 8 — no change needed.** The vfield-T section (line ~3430) uses its own class (`vfield-t-dim`) and is unaffected.

- [ ] **Step 9: Add j/k guard — stop playback when navigating the playing dim**

In the j/k handlers (~lines 5040-5080), add at the top of each handler (after `e.preventDefault()`):

```javascript
// j handler (ArrowDown)
if (isPlaying && activeDim === playingDim) { stopPlay(); }

// k handler (ArrowUp) — same guard
if (isPlaying && activeDim === playingDim) { stopPlay(); }
```

This stops playback and then falls through to the normal navigation logic.

- [ ] **Step 10: Run all play-related tests**

Run: `uv run pytest tests/test_browser.py -k "playing_dim or play_allows or jk_on_playing or space_toggles" -v`
Expected: All PASS

- [ ] **Step 11: Commit**

```bash
git add src/arrayview/_viewer.html tests/test_browser.py
git commit -m "feat: playing dim shows in orange, allow independent dim navigation during playback"
```

---

### Task 5: Final verification and collateral updates

**Files:**
- Modify: `tests/visual_smoke.py` (if it exists, per @viewer-ui-checklist)
- Check: README.md for any relevant updates

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Check visual_smoke.py**

If `tests/visual_smoke.py` exists, check if any of the changed shortcuts (b, r, Space) are covered and update if needed.

- [ ] **Step 3: Check README for keyboard shortcut documentation**

Verify the README keyboard shortcuts table still matches the behavior. No new shortcuts were added, but verify `r` description mentions "global axis swap in multiview" and Space mentions "independent dim navigation".

- [ ] **Step 4: Update help overlay if needed**

Check the help overlay text for `r` and Space in `_viewer.html` (~line 904 for r, search for Space). Update descriptions if they're now inaccurate.

- [ ] **Step 5: Update TODO**

Move the 4 completed items to the DONE section in `docs/todo.md`.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "docs: update help text, README, and TODO for completed items"
```
