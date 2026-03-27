# Dynamic Islands Only in Immersive Mode

**Date:** 2026-03-27

## Problem

The glassmorphic dynamic islands (colorbar, dim bar, histogram) look great in immersive mode
where they overlay the canvas, but are unnecessary visual weight in non-immersive mode where
these elements sit in normal layout flow outside the canvas.

## Design

### Non-immersive mode (default)

**Dim bar (`#info`):** Raw text, no background/border/blur/padding. Just the dimension labels
centered above the canvas.

**Colorbars (all modes):** Horizontal `[vmin] [gradient] [vmax]` layout stays. Remove
glassmorphic background, blur, border-radius, padding from the island wrapper. Add a thin
gray border (same as `b`-key pane border) on the gradient canvas element only.

**Histogram (expanded colorbar):** No island background. The KDE curve gets a thin stroke
outline tracing the filled area's top edge + baseline, giving the distribution shape a
visible boundary against the page background.

### Immersive mode (`body.fullscreen-mode`)

No changes — keep all existing glassmorphic island styling. The islands are needed because
elements overlay the canvas in this mode.

### Affected elements

| Element | Selector(s) | Non-immersive change |
|---------|------------|---------------------|
| Dim bar | `#info` | Strip background, blur, border, border-radius, padding |
| Normal colorbar | `#slim-cb-wrap` | Strip island styling, add canvas border |
| Multiview colorbar | `#mv-cb-wrap` | Strip island styling, add canvas border |
| qMRI colorbars | `.qv-cb-island` | Strip island styling, add canvas border |
| Compare colorbars | `.compare-pane-cb-island` | Strip island styling, add canvas border |
| Generic island | `.cb-island` | Strip island styling when not fullscreen |
| Histogram | KDE draw in `ColorBar.draw()` | Add stroke path, remove background fill |

### CSS strategy

Use `body:not(.fullscreen-mode)` selectors to override island styling. The existing
`body.fullscreen-mode` rules already enforce immersive styling, so no changes needed there.

### KDE stroke implementation

In `ColorBar.draw()`, after filling the KDE area, trace a `ctx.stroke()` path along:
1. The baseline (bottom of histogram area)
2. Up along the KDE curve shape
3. Close path back to baseline start

Use a thin (1-1.5px) stroke in the same gray as the `b`-key pane border.

### Border color reference

The `b`-key pane border uses `rgba(255, 255, 255, 0.15)` in dark theme. Use the same value
for colorbar canvas borders and KDE stroke. Theme-aware variants for light/solarized/nord.
