# TODO Batch — Design Spec

## Overview

Nine TODO items grouped into four themes: compact mode fixes, colorbar zoom, complex projection math, and histogram UX redesign.

## 1. Compact Mode Flickering (Item 1)

**Status**: Likely already fixed by recent dynamic island positioning commits. Verify during implementation — if confirmed fixed, remove from TODO.

## 2. Dim Bar in Compact Mode (Item 2)

**Problem**: `body.compact-mode #info` reduces font-size to 14px.

**Fix**: Remove the `font-size: 14px` from the compact-mode `#info` CSS rule. Keep the array name hidden (shows on hover). The dim bar stays identical to normal mode — only vertical space savings come from collapsing the array name.

**Files**: `_viewer.html` line ~625-629

## 3. Eggs Inside Canvas in Compact Mode (Item 3)

**Problem**: Eggs stack vertically to the LEFT of the canvas in compact mode.

**Fix**: Position eggs inside the canvas viewport, centered horizontally, just above the compact colorbar overlay. Use `position: absolute` relative to the canvas viewport. Semi-transparent so they don't fully obscure the image.

**Layout**: `[canvas image] → [eggs row, centered] → [colorbar island]` stacked bottom-up inside viewport.

**Files**: `_viewer.html` `positionEggs()` lines ~3685-3696

## 4. Vmin/Vmax Labels in Compact Mode (Item 4)

**Problem**: Compact mode hides vmin/vmax inline spans (`display: none`).

**Fix**: Show the flanking `#slim-cb-vmin` and `#slim-cb-vmax` spans in compact mode, same as normal mode. The compact overlay colorbar becomes: `[vmin] ═══gradient═══ [vmax]`.

Also apply to other modes (qMRI etc) — use ui-consistency-audit to verify.

**Files**: `_viewer.html` `drawSlimColorbar()` lines ~2483-2488

## 5. Colorbar Zoom-Out Bug (Item 5)

**Problem**: Colorbar becomes weird when zooming out.

**Approach**: Use ui-consistency-audit skill to investigate and fix. Likely a sizing/positioning edge case in `drawSlimColorbar()` or `scaleCanvasSpace()` when canvas dimensions shrink below certain thresholds.

**Files**: `_viewer.html` — colorbar positioning logic

## 6. Complex Image Projections (Item 6)

**Problem**: Projections on complex arrays trigger `ComplexWarning` because `np.max/min/mean/std` and squaring are applied directly to complex values.

**Fix in `_render.py` `extract_projection()`**:

- **max**: Find indices of max `|z|` along proj_axis, gather those complex values. Return complex array.
- **min**: Find indices of min `|z|` along proj_axis, gather those complex values. Return complex array.
- **mean**: `np.mean(vol, axis=proj_axis)` — valid on complex, returns complex. No warning.
- **std**: `np.std(vol, axis=proj_axis)` — valid on complex, returns real. No warning.
- **sos**: `np.sum(z * np.conj(z), axis=proj_axis)` — magnitude-squared sum. Returns real float.

After projection, the result passes through `apply_complex_mode()` which handles mag/phase/real/imag display. For max/min/sos the result is already informed by magnitude; for mean/std it preserves complex structure for the display mode to interpret.

Guard: only apply complex-aware logic when `np.iscomplexobj(vol)`. Real arrays use existing code path unchanged.

**Files**: `_render.py` `extract_projection()` lines ~157-196

## 7. Histogram Background Color (Item 7)

**Problem**: Histogram bars render on transparent background, showing the window behind.

**Fix**: The histogram already draws on the canvas inside the `.cb-island` dynamic island container, which has a frosted glassmorphism background. The colorbar gradient strip is already hidden when expanded (`!_cbExpanded` guard on line 2535). No code change needed — the dynamic island background already shows through. Verify visually and ensure this holds across all themes (dark, light, solarized, nord).

**Files**: Verify only — `_viewer.html`

## 8. Histogram Vmin/Vmax Indicators (Item 8)

**Problem**: Two plain yellow vertical lines for vmin/vmax look harsh.

**Fix**: Replace with two combined indicators:

1. **Shaded out-of-range bars**: Bars outside the vmin–vmax window draw at reduced opacity (~0.25 vs ~0.85). Provides instant visual understanding of the clipping region.

2. **Bracket markers**: Small L-shaped bracket marks at vmin/vmax positions (like crop handles). Top-left + bottom-left bracket at vmin, top-right + bottom-right bracket at vmax. These provide precise drag targets for the existing vmin/vmax drag interaction.

Remove the current full-height vertical lines from `_drawClimLines()`.

**Files**: `_viewer.html` `_drawHistogramBarsOnColorbar()` ~2605-2670, `_drawClimLines()` ~2672-2691

## 9. Histogram Full Width (Item 9)

**Problem**: Vmin/vmax label spans flank the histogram, taking horizontal space.

**Fix**: When `_cbExpanded` is true, hide the flanking `#slim-cb-vmin` and `#slim-cb-vmax` spans so the histogram canvas fills the full island width. The vmin/vmax position is already communicated by the shading + bracket indicators on the histogram itself.

This is already partially implemented (lines 2569-2571 clear the text content when expanded). Change to `display: none` instead of just clearing text, so the space is reclaimed.

**Files**: `_viewer.html` `drawSlimColorbar()` ~2566-2576

## Implementation Order

1. Complex projection math (item 6) — backend, independent
2. Dim bar compact mode (item 2) — CSS only
3. Vmin/vmax in compact mode (item 4) — small JS change
4. Eggs positioning in compact mode (item 3) — JS positioning
5. Histogram shading + brackets (item 8) — canvas drawing
6. Histogram full width (item 9) — layout change
7. Histogram background verification (item 7) — verify only
8. Colorbar zoom-out bug (item 5) — investigate with audit
9. Verify flickering fix (item 1) — verify only
10. Cross-mode consistency audit — ui-consistency-audit skill
