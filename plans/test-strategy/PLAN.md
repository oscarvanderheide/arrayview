# Behavioral Specification & Test Strategy

## What This Document Is

A complete definition of what every interaction should do in every mode — not just "does it work?" but "should it be allowed here, and if so, what exactly should happen?" This is both a design specification and a test blueprint.

## Modes

| # | Mode | Purpose | Entry | State flag |
|---|------|---------|-------|------------|
| 1 | **Normal** | Single 2D slice, full features | default | — |
| 2 | **Multiview** | 3 orthogonal planes for 3D spatial understanding | `v` | `multiViewActive` |
| 3 | **Compare** | Side-by-side arrays for visual comparison | `B`/`P` | `compareActive` |
| 4 | **Diff** | Center pane showing computed difference (sub-mode of Compare) | `X` | `diffMode > 0` |
| 5 | **Overlay** | Blended registration view (sub-mode of Compare) | `X` | `registrationMode` |
| 6 | **Wipe** | Split-screen comparison (sub-mode of Compare) | `X` | `_wipeActive` |
| 7 | **qMRI** | Parameter map grid with domain-specific colormaps | `q` | `qmriActive` |
| 8 | **Zen** | Minimal chrome for presentations | `Z`/`F` | `zenMode` |
| 9 | **Compact** | Reduced chrome when height-constrained | `K` | `compactMode` |
| 10 | **ROI** | Drawing regions for statistics | `A` | `rectRoiMode` |

Zen and Compact are "overlay" modes — they modify chrome but don't change the underlying mode. ROI is also an overlay on Normal mode only.

---

## Feature × Mode Behavioral Matrix

Legend:
- ✓ = allowed, works as described
- ✗ = should be **disabled** (show status message or silently ignore)
- ★ = allowed but with **mode-specific behavior** (described in notes)
- 🐛 = **currently broken** or **missing guard** (needs fix)

### Keyboard: Display Settings

| Feature | Normal | Multiview | Compare | Diff/Overlay/Wipe | qMRI | Zen | Compact | ROI |
|---------|--------|-----------|---------|-------------------|------|-----|---------|-----|
| **c** colormap | ✓ | ✓ all planes | ✓ all data panes | ★ center colormap only (restricted pool) | 🐛→✗ | ✓ | ✓ | ✓ |
| **C** custom cmap | ✓ | ✓ all planes | ✓ all data panes | ★ apply to center only | 🐛→✗ | ✓ | ✓ | ✓ |
| **d** DR cycle | ✓ | ✓ all planes | ★ all panes + clear per-pane overrides | ★ data panes re-render; diff recomputed | ✓ all maps | ✓ | ✓ | ✓ |
| **D** manual range | ✓ | ✓ all planes | ★ all data panes + clear per-pane overrides | — not applied to diff | 🐛→✗ | ✓ | ✓ | ✓ |
| **L** log scale | ✓ | ✓ all planes | ✓ all data panes | ★ diff recomputed (not log-scaled itself) | ★ T₁/T₂ only (correct!) | ✓ | ✓ | ✓ |
| **m** complex | ✓ | ✓ all planes | ✓ all data panes | ★ diff recomputed | ✗ (correct!) | ✓ | ✓ | ✓ |
| **M** mask | ✓ | ✗ | ✗ | ✗ | ✗ (correct!) | ✓ | ✓ | ✗ |
| **R** RGB | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| **f** FFT | ✓ | ✓ | ✓ all panes | ★ diff recomputed | 🐛→✗ | ✓ | ✓ | ✗ |
| **T** theme | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **b** borders | ✓ | ✓ all planes | ✓ all panes | ✓ | ✓ all maps | ✓ | ✓ | ✓ |
| **a** stretch | ✓ | ✓ all planes | ✓ all panes | ✓ | ✓ all maps | ✓ | ✓ | ✗ |
| **w** Lebesgue | ✓ | ✓ | ✓ | ✓ | ★ per-map? or global? | ✓ | ✓ | ✗ |

**Notes on "disabled in qMRI":**
- `c`/`C`: Each parameter map has a domain-specific colormap (lipari→T₁, navia→T₂, gray→|PD|, twilight→∠PD). Cycling to a shared colormap would be meaningless. **Currently no guard — bug.**
- `D`: Maps have completely different value ranges (T₁: 0-3000ms, T₂: 0-100ms, PD: 0-1). A single manual vmin/vmax makes no sense. **Currently no guard — bug.**
- `f`: FFT of parameter maps is physically meaningless. **Currently no guard — bug.**

**Notes on "disabled in ROI":**
- `a` stretch: Changes canvas aspect ratio, which invalidates drawn ROI positions.
- `f` FFT: Transforms to frequency domain, completely changing what ROI coordinates mean.
- `M` mask: One analysis overlay at a time.
- `w` Lebesgue: Colorbar mode, not compatible with ROI stats workflow.

### Keyboard: Navigation

| Feature | Normal | Multiview | Compare | Diff | qMRI | Zen | ROI |
|---------|--------|-----------|---------|------|------|-----|-----|
| **j/k** scroll | ✓ | ✓ navigate activeDim, all planes update | ✓ all panes sync | ★ diff recomputed | ✓ all maps | ✓ | ★ ROIs persist, stats update |
| **h/l** dim cycle | ✓ | ✓ cycle through all dims | ✓ all panes | ✓ | ✓ non-qMRI dims | ✓ | ✓ |
| **x** assign dim_x | ✓ | ✗ (use V to reconfigure) | ✓ all panes | ✓ | ✗ | ✓ | ★ clears ROIs |
| **y** assign dim_y | ✓ | ✗ (use V to reconfigure) | ✓ all panes | ✓ | ✗ | ✓ | ★ clears ROIs |
| **r** reverse/rotate | ✓ | ★ flip per-dim (special branch) | ✓ all panes | ✓ | ✓ all maps | ✓ | ★ clears ROIs |
| **0-9+Enter** slice jump | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Space** play/pause | ✓ | ✓ | ✓ all panes sync | ★ diff updates on each frame | ✓ all maps | ✓ | ✗ (exit ROI first) |

**Notes on navigation in ROI mode:**
- Scrolling (j/k) with ROIs drawn: ROIs are spatial regions in (dim_x, dim_y). When scrolling a different dimension (e.g., through slices of a 3D volume), the ROI pixel coordinates remain valid. Stats should recompute for the new slice — this makes ROIs act as a "region tracker" across slices. **Useful behavior!**
- When dim_x or dim_y changes (x/y keys): ROI pixel positions become meaningless. Clear all ROIs.
- When rotating (r key): ROI positions become invalid. Clear all ROIs.
- Space (play/pause): Animating with ROIs is confusing — the stats would update every frame but you can't read them. Disable or auto-exit ROI mode.

### Keyboard: Zoom & Layout

| Feature | Normal | Multiview | Compare | Diff | qMRI | Zen | ROI |
|---------|--------|-----------|---------|------|------|-----|-----|
| **+/-** zoom | ✓ | ✓ all planes equal | ✓ all panes equal, viewport clips | ✓ center matches | ✓ all maps | ✓ | ✓ |
| **0** reset zoom | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **K** compact | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| **Z/F** zen | ✓ | ✓ hides chrome, keeps planes | ✓ hides chrome, keeps panes | ✓ | ✓ | toggle | ✓ |
| **G** layout cycle | ✗ | ✗ | ✓ h→v→grid→h | ✓ | ✗ | ✗ | ✗ |

### Keyboard: Export

| Feature | Normal | Multiview | Compare | Diff | qMRI | Zen | ROI |
|---------|--------|-----------|---------|------|------|-----|-----|
| **s** screenshot | ✓ | ✓ composite | ✓ composite | ✓ includes center | ✓ composite | ✓ | ✓ includes ROIs |
| **g** GIF | ✓ | 🐛→✗ | 🐛→✗ | ✗ | 🐛→✗ | ✓ | ✗ |
| **N** export .npy | ✓ | 🐛→✗ | 🐛→✗ | ✗ | 🐛→✗ | ✓ | ✓ |
| **e** copy URL | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **?** help | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **i** info | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ (no chrome) | ✓ |

**Notes on export guards:**
- `g` (GIF): Server generates GIF for a single session over one dimension. In multiview/compare/qMRI, it's ambiguous which session/view to export. The server endpoint only takes a single sid. Should be disabled in all multi-canvas modes. **Currently no guard — bug.**
- `N` (export .npy): Same issue — exports one 2D slice from one session. Ambiguous in multi-canvas modes. **Currently no guard — bug.**
- `s` (screenshot): Works everywhere because it captures what's on screen. The `saveScreenshot()` function already composes multiview canvases side-by-side. Should also work for compare and qMRI.

### Keyboard: Mode Switching

| Feature | Normal | Multiview | Compare | qMRI |
|---------|--------|-----------|---------|------|
| **v** multiview | enter MV | exit MV | enter compare-MV | exit qMRI, enter MV |
| **V** custom MV dims | prompt → enter MV | prompt → re-enter MV | enter compare-MV | exit qMRI, prompt → enter MV |
| **q** qMRI | enter qMRI | exit MV, enter qMRI | enter compare-qMRI | toggle compact/exit |
| **B** compare | enter compare | exit MV, enter compare? | exit compare | exit compare |
| **P** picker | open picker | open picker? | open picker | open picker? |
| **A** ROI | enter ROI | ✗ | ★ see ROI design below | ✗ |
| **X** center cycle | ✗ | ✗ | cycle (2 pane only) | ✗ |

### Mouse: Canvas Interactions

| Interaction | Normal (fit) | Normal (zoomed) | Multiview | Compare (fit) | Compare (zoomed) | qMRI | ROI |
|-------------|-------------|----------------|-----------|---------------|-----------------|------|-----|
| **Click** | focus | focus | navigate crosshair | focus | focus | focus | start ROI draw |
| **Drag** | — | pan | crosshair nav / oblique rotate | — | pan all panes | — | draw ROI shape |
| **Wheel** | scroll slice | scroll slice | scroll slice (per-plane dim) | scroll slice (all panes) | scroll slice | scroll slice (all maps) | scroll slice (stats update) |
| **Ctrl+Wheel** | zoom | zoom | zoom all | zoom all | zoom all | zoom all | zoom |
| **Hover** | — | — | — | — | — | — | ★ show ROI stats tooltip |

### Mouse: Colorbar Interactions

| Interaction | Normal | Multiview | Compare (per-pane) | Compare (global) | Diff center | qMRI (per-map) |
|-------------|--------|-----------|-------------------|-----------------|-------------|----------------|
| **Hover** | marker line | marker line | marker line | marker line | marker line | marker line |
| **Wheel** | zoom range (narrow/widen) | zoom range | zoom THIS pane's range | zoom all panes' range | zoom diff range | zoom THIS map's range |
| **Drag** | shift range | shift range | shift THIS pane | shift all | shift diff range | shift THIS map |
| **Double-click** | reset to auto | reset | reset THIS pane | reset all | reset diff | reset THIS map |

**Note on Compare colorbar semantics:**
- Per-pane colorbar interaction (wheel/drag on one pane's colorbar) creates a per-pane override → useful for investigating one array specifically
- Global colorbar interaction (the shared colorbar below all panes) affects all panes → maintains fair comparison
- `d` and `D` keys always reset per-pane overrides back to shared range

### Mouse: Special Interactions

| Interaction | When available | Behavior |
|-------------|---------------|----------|
| **Title drag** | Compare mode | Reorder panes (swap) |
| **Blend bar drag** | Overlay/registration mode | Adjust alpha |
| **Wipe line drag** | Wipe mode | Adjust split position |
| **Minimap click** | Zoomed + minimap visible | Jump viewport to position |
| **File drop** | Always | Load file as new session |

---

## Compare Mode Behavioral Contract

### Principle: Fair Comparison

When comparing arrays, visual differences must come from DATA differences, not display setting differences. Therefore:

### Synchronized State (all data panes share these)

| Setting | Why |
|---------|-----|
| Colormap | Same colors → only data differences visible |
| DR level | Same contrast windowing |
| vmin/vmax | Same value-to-color mapping |
| Log scale | Same transform |
| Complex mode | Same representation |
| Zoom | Same spatial scale |
| Pan offset | Same viewport position |
| Scroll indices | Same slice |
| Stretch mode | Same aspect ratio |
| Axis assignment | Same orientation |

### Per-Pane Overrides (exception to sync)

Only colorbar wheel/drag creates per-pane vmin/vmax overrides. This is an intentional "drill-down" — the user explicitly adjusts one pane's range to investigate.

**Critical rule:** `d` and `D` keys MUST reset ALL per-pane overrides (`cmpManualVmin.fill(null); cmpManualVmax.fill(null)`) to restore fair comparison. **Currently broken — `d` and `D` don't clear per-pane overrides.**

### Center/Diff Pane Independence

The diff pane is a DERIVED view. It has:
- Its own colormap (restricted pool: RdBu_r for signed diff, afmhot for |A-B| and |A-B|/|A|)
- Its own vmin/vmax (computed from the difference, not from the data panes)
- `c` on center pane cycles diff-appropriate colormaps only
- `d`/`D`/`L` do NOT directly change the diff display — they change the input arrays, which causes diff recomputation

### Layout Invariant

Panes MUST stay in their grid positions regardless of zoom. `flex-wrap: wrap` → `display: grid` with `grid-template-columns: repeat(var(--compare-cols), 1fr)`. When zoomed, canvas overflows into a clipping container; it NEVER causes reflowing.

---

## ROI Design for Compare Mode

### Current: ROI is blocked in compare/multiview/qMRI

### Proposed: Allow ROI in compare mode

**How it works:**
1. Press `A` in compare mode → enter ROI drawing
2. Draw on ANY data pane → same pixel region is marked on ALL panes
3. Stats are computed per-array for each ROI
4. Stats shown as **hover tooltip** when mousing over a ROI (not a persistent panel — too cluttered)
5. CSV export includes all arrays' stats: columns = `roi, array_name, n, min, max, mean, std`
6. ROI drawing on diff/overlay/wipe pane: **disabled** (these are derived views)

**Why hover instead of panel:**
- With 2-6 arrays × multiple ROIs, a table becomes unreadable
- Hover shows stats for the SPECIFIC pane you're interested in
- CSV export provides the full data when you need it

**Implementation notes:**
- ROI coordinates are in (dim_x, dim_y) pixel space → identical for all panes showing the same spatial slice
- Scrolling through slices: ROIs persist, stats recompute per-slice → "region tracking"
- Each pane's backend session provides stats independently via `/roi/{sid}`

---

## Known Bugs (discovered by analysis)

### Missing Mode Guards

| # | Key | Mode | Current | Correct | Status |
|---|-----|------|---------|---------|--------|
| 1 | `c`/`C` | qMRI | ~~No guard~~ | Blocked — maps have domain-specific colormaps | ✅ Fixed |
| 2 | `D` | qMRI | ~~No guard~~ | Blocked — maps have different units/ranges | ✅ Fixed |
| 3 | `f` | qMRI | ~~No guard~~ | Blocked — physically meaningless | ✅ Fixed |
| 4 | `g` | multi-canvas | ~~No guard~~ | Blocked — ambiguous which session | ✅ Fixed |
| 5 | `N` | multi-canvas | ~~No guard~~ | Blocked — ambiguous which view | ✅ Fixed |
| 6 | `x`/`y` | multiview | ~~No guard~~ | Blocked — use V to reconfigure planes | ✅ Fixed |
| 7 | `x`/`y` | qMRI | ~~No guard~~ | Blocked — spatial dims fixed for parameter maps | ✅ Fixed |
| 8 | `r` | multiview | ★ Already has special branch — flips per-dim. Correct! | No fix needed | ✅ Correct |
| 9 | `Space` | ROI mode | ~~No guard~~ | Blocked — exit ROI first | ✅ Fixed |

### State Synchronization Bugs

| # | Bug | Root Cause | Status |
|---|-----|-----------|--------|
| 10 | `d` in compare doesn't clear per-pane overrides | `d` handler zeros `manualVmin/max` but not `cmpManualVmin/max[]` | ✅ Fixed |
| 11 | `D` in compare doesn't clear per-pane overrides | `D` sets global range but doesn't reset per-pane | ✅ Fixed |

### Layout Bugs

| # | Bug | Root Cause | Status |
|---|-----|-----------|--------|
| 11 | Compare panes wrap vertically when zoomed | `#compare-panes` uses `flex-wrap: wrap`; `--compare-cols` set but never consumed by CSS | ✅ Fixed (78e5b43) |

### Visual/Polish Bugs

These were all fixed in recent commits (roi-improvements, dim-bar-padding, text-styling, zoom-aspect-lock, pinch-zoom-speed, wipe-interaction, compare-colorbar, diff-colormap-preview, multi-view-order, orientation widget):

| # | Bug | Status |
|---|-----|--------|
| 12 | Diff colormap preview broken | ✅ Fixed (fa6b664) |
| 13 | Extra/stale colorbar in compare mode | ✅ Fixed (6abba16) |
| 14 | Multiview (x,y) label on wrong canvas | ✅ Fixed (882478f) |
| 15 | 3D orientation widget missing in multi-array compare | ✅ Fixed (0ad682d) |
| 16 | Dim bar width jumps at digit rollover | ✅ Fixed (cd65f74) |
| 17 | Blurry ROI borders/numbers | ✅ Fixed (8de9223) |
| 18 | Blurry wipe line | ✅ Fixed (66dee07) |
| 19 | Zoom at max vertical expands width | ✅ Fixed (cdadf62) |
| 20 | Pinch zoom too fast | ✅ Fixed (cd65f74) |
| 21 | CSV export doesn't download / can't exit after | ✅ Fixed (ed82b19) |

---

## Feature Ideas (emerged from analysis)

### 1. ROI stats on hover (instead of panel)
When hovering over a drawn ROI, show a tooltip with its stats. In compare mode, show stats for the hovered PANE's array. Cleaner than a persistent panel, scales to many ROIs/arrays.

### 2. ROI as region tracker
When scrolling through slices, ROIs persist (they define a spatial region). Stats update for each new slice. Useful for tracking how a region's values change across slices/timepoints.

### 3. ROI in compare mode
Draw once on any pane → same region on all panes. See stats per-array. CSV export includes all arrays. Detailed design above.

### 4. GIF export for compare/multiview
Currently blocked because the server can only GIF a single session. Could be implemented by having the frontend compose frames (capture each compare layout → assemble into GIF). This is hard to implement well — low priority.

### 5. Per-map colorbar interaction in qMRI
Users should be able to wheel/drag individual map colorbars to adjust per-map vmin/vmax, similar to compare per-pane overrides. May already work — needs verification.

### 6. Screenshot in compare mode
`saveScreenshot()` should composite all compare panes (like it does for multiview). Verify this works correctly, including diff/overlay center pane.

### 7. Export what's visible
Instead of blocking `N`/`g` in multi-canvas modes, could export the "focused" or "hovered" view. But this adds ambiguity. Blocking is cleaner.

### 8. Play + ROI
When pressing Space in ROI mode, could animate while tracking ROI stats over time. Would need a plot/graph of stats vs. slice index. Cool but complex — future feature.

---

## Implementation Plan

### Phase 1: Add Missing Guards (small, high-value fixes)

Each is a one-line `if` guard + status message. One commit per fix or batch related fixes:

1. Guard `c`/`C` in qMRI mode
2. Guard `D` in qMRI mode  
3. Guard `f` in qMRI mode
4. Guard `g` in multiview/compare/qMRI
5. Guard `N` in multiview/compare/qMRI
6. Guard `x`/`y` in multiview and qMRI
7. Guard `Space` in ROI mode (or auto-exit ROI)

### Phase 2: Fix State Synchronization

9. `d` handler: add `cmpManualVmin.fill(null); cmpManualVmax.fill(null);` + `compareRender()`
10. `D` handler: same + add guard to not prompt for diff pane

### Phase 3: Fix Layout

11. `#compare-panes`: `flex-wrap: wrap` → CSS grid

### Phase 4: Fix Visual Bugs

12–21: One commit each (see bug list above)

### Phase 5: Build Test Infrastructure

- `tests/test_feature_modes.py`: Parametrized Playwright tests verifying every cell in the matrix above
- `tests/test_compare_sync.py`: Verify synchronized state, per-pane overrides, diff independence
- `tests/test_transitions.py`: Mode enter/exit clean state verification
- JS console error capture in `conftest.py`
- Download interception for export tests

### Phase 6: ROI improvements

- Allow ROI in compare mode with hover-tooltip stats
- ROI persistence across slice navigation
- CSV export with per-array columns in compare mode

---

## How to Verify

After Phase 1-4 (fixes), run:
```bash
uv run pytest tests/test_api.py tests/test_browser.py tests/test_mode_matrix.py -v
uv run python tests/visual_smoke.py  # review smoke_output/
```

After Phase 5 (tests), the test suite itself becomes the verification:
```bash
uv run pytest tests/ -v
```

For each cell in the Feature × Mode matrix, there should be a test that either:
- Asserts the feature works correctly in that mode, OR
- Asserts the feature is blocked with a status message in that mode
