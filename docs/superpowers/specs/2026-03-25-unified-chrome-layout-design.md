# Unified Chrome Layout

Redesign the viewer chrome (dim bar, array names, colorbars) to use a single consistent layout for 1–6 arrays, replacing the current divergent single-mode and compare-mode chrome.

## Current State

- **Single array**: logo + name at top → dim bar → canvas → shared colorbar (position: fixed)
- **Compare (2+ arrays)**: "COMPARING N ARRAYS" + logo → dim bar → per-pane titles → canvases with per-pane colorbars
- **Compact mode**: auto-enters on zoom, collapses chrome onto canvas
- **Zen mode**: fullscreen via `requestFullscreen()`, hides all chrome

Single and multi-array modes have different vertical stacking, different colorbar strategies, and different chrome behavior, making it hard to maintain consistency.

## Design

### Normal Mode — Unified Vertical Stack

Same structure for all array counts:

1. **Dim bar island** (top center) — shared, always the topmost element
2. **Name island(s)** — plain text (matching current single-mode `#array-name` style: 11px monospace, uppercase, letter-spacing 0.08em, muted color, with logo). One per loaded array.
   - 1 array: single name, centered
   - 2 arrays: two names, each centered above its pane
   - 3 arrays: three names in a row
   - 4 arrays: 2×2 grid — name above each pane in the grid
   - 5-6 arrays: names above each pane in the existing grid layout (3-col grid)
3. **Canvas(es)** — same pane shrink-wrap behavior (panes stay close when zoomed out)
4. **Shared colorbar island** (bottom center) — single colorbar spanning below all panes

The gap between the bottom of the canvases and the top of the colorbar matches the gap between the name row and the top of the canvases (symmetric spacing).

### Colorbar

- **Remove per-pane colorbars** in compare mode entirely
- **Single shared colorbar** in all modes — reuse the existing `#slim-cb-wrap` / `drawSlimColorbar()` mechanism
- All arrays share the same colormap and dynamic range. The shared range is the union: `vmin = min(all arrays' vmin)`, `vmax = max(all arrays' vmax)`
- Colorbar width: matches the total width of the pane grid (for 1 array, matches the canvas viewport width — same as today)
- **Re-enable histogram/Lebesgue mode** for multi-array: aggregate bins from all loaded arrays into a single histogram. Implementation detail — likely client-side merge of per-session histogram data already fetched.

### Fullscreen Mode (replaces compact)

Manual toggle only, triggered by **K** key. No auto-enter on zoom.

This is a CSS layout change (not `requestFullscreen()`), distinct from zen mode (F key) which uses the browser Fullscreen API and hides all chrome.

- Canvas(es) fill the viewport edge-to-edge
- All chrome overlaid as glassmorphic dynamic islands (same `.cb-island` styling: `background: rgba(30,30,30,0.8)`, `backdrop-filter: blur(12px)`, `border-radius: 14px`):
  - **Dim bar**: overlaid top-center
  - **Per-pane name pills**: overlaid at top of each pane, smaller glassmorphic pills
  - **Shared colorbar**: overlaid bottom-center
- Multi-array: panes touch (minimal gap, e.g. 2px), filling the full viewport in the same grid layout as normal mode
- Overlays follow existing z-index conventions: below help overlay and tooltips, above canvases. They do not auto-hide and are pointer-events passthrough except for interactive elements (dim bar clicks, colorbar drag).
- Window resize recalculates layout (same as current behavior — handled by the scale functions on the `resize` event).

### Grid Layouts

| Arrays | Default layout | Future (G key) |
|--------|---------------|-----------------|
| 1      | 1×1           | —               |
| 2      | 1×2           | —               |
| 3      | 1×3           | —               |
| 4      | 2×2           | 1×4             |
| 5      | 2×3           | —               |
| 6      | 2×3           | 3×2             |

G key to cycle grid layouts is future work, not part of this spec.

### Diff Mode (2 arrays only)

In diff mode (X key), the layout switches from one shared colorbar to **per-pane colorbars**:

- **Left + right panes** (the two loaded arrays): each gets its own colorbar, same vmin/vmax, same colormap. These are visually identical — two copies of the shared colorbar, positioned below each pane at the same vertical position and same physical size.
- **Center pane** (diff result): gets its own independent colorbar (or no colorbar / a slider / nothing, depending on the diff variant). Same vertical position and size as the side colorbars.

**Keyboard interaction depends on mouse position:**

| Key | Mouse on side panes | Mouse on center pane |
|-----|--------------------|-----------------------|
| `c` (cycle colormap) | Cycles colormap for both side panes together. Colormap previewer appears under both side panes. | Cycles through a limited subset of colormaps appropriate for the diff variant (e.g. diverging maps for A−B, sequential for \|A−B\|). |
| `d` (cycle dynamic range) | Changes dynamic range for both side panes together. Histogram appears for both, bins aggregated from both arrays. | Changes dynamic range for the diff pane only. |

When diff mode is exited, the layout reverts to a single shared colorbar.

### Removed

- "COMPARING N ARRAYS" header text
- Top-center logo in compare mode (logo moves to per-pane name islands)
- Per-pane colorbars (`.compare-pane-cb-island`, `drawComparePaneCb`, `drawAllComparePaneCbs`)
- Auto-compact on zoom
- Separate compact mode (merged into fullscreen)

### Kept As-Is

- Zen mode (F key) — browser fullscreen via `requestFullscreen()`, hides ALL chrome. Orthogonal to this redesign. Composable: you can be in fullscreen mode (K) and then press F for zen.
- Diff/registration/wipe/overlay center pane — continues to work within the compare pane grid
- ROI, vector field, hover tooltip — orthogonal overlays, unaffected
- Transition behavior between array counts (load/unload) — instant re-layout, same as current

## Scope

This is a large refactor touching:
- `_viewer.html`: chrome layout CSS, `compareScaleCanvases()`, `drawSlimColorbar()`, removal of `drawComparePaneCb()` / `drawAllComparePaneCbs()`, compact mode logic → fullscreen mode, keyboard handler
- Histogram aggregation across sessions (client-side merge of per-session data)
- Test updates for new layout
