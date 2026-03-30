# Dynamic Island for ROI & SEGMENT Modes

## Problem

The egg badge system is designed for composable data transforms (FFT, LOG, MAGNITUDE, etc.) that stack naturally. ROI and SEGMENT are interaction modes — they take over canvas input rather than transforming displayed data. They are also mutually incompatible. Representing them as eggs is a category error.

Previous attempts (floating fan-out arc at cursor, inline badge icons) were either too intrusive or too limited — neither provided space for the ROI table, stats, export, or segmentation label management.

## Solution

A draggable **dynamic island** panel that appears when ROI or SEGMENT mode is active. One island at a time — activating one dismisses the other with an Apple Dynamic Island-style morph transition.

## Island Layout

Both ROI and SEGMENT islands share the same visual structure:

```
┌─────────────────────────────────┐
│ ≡  ROI · □ ○ ~ ◆    ↓         │  ← header: drag handle, title, shape icons, download
├─────────────────────────────────┤
│ ■ ROI 1      142.3 ± 28.1   × │  ← row: color swatch, name, stats, delete
│ ■ ROI 2       89.7 ± 15.4   × │
│ ■ ROI 3      201.0 ± 42.8   × │
└─────────────────────────────────┘
```

### Header

- **Drag handle** (three horizontal lines, left edge) — mousedown to drag the island anywhere in the viewport
- **Title** — "ROI" or "SEG", white, bold
- **Dot separator** — subtle `·`
- **Shape/method icon strip** — all available shapes shown as small SVG icons. Active shape highlighted in mode color (yellow for ROI, pink for SEG). Inactive shapes gray at 50% opacity. Click an icon to select it directly. Press A/S to cycle as before.
- **Download icon** (right edge) — triggers export

### ROI Rows

Each drawn ROI gets a row:
- **Color swatch** — 7px rounded square, unique color per ROI
- **Name** — "ROI 1", "ROI 2", etc. (auto-numbered)
- **Stats** — `mean ± std` for the current 2D slice, right-aligned, muted color
- **Delete** — `×` button to remove the ROI

### SEGMENT Rows

Each accepted segmentation label gets a row:
- **Color swatch** — 7px rounded square, matches overlay color
- **Name** — default "Label 1", "Label 2", etc. Click to edit inline. Dashed underline hints editability. Default names shown in italic, user-set names in regular weight.
- **Delete** — `×` button to remove the label

No voxel count shown in the row — keep it minimal.

## ROI Shapes (press A to cycle)

1. **rect** — drag to draw rectangle
2. **circle** — drag to draw ellipse
3. **freehand** — drag to draw freeform path
4. **floodfill** — click seed pixel, `[`/`]` to adjust tolerance

## SEG Methods (press S to cycle)

1. **click** — single click seed point
2. **circle** — drag bounding ellipse
3. **scribble** — freehand lines
4. **lasso** — closed contour polygon

## Mutual Exclusivity

ROI and SEGMENT modes cannot be active simultaneously:
- Pressing **S** while ROI island is visible: ROI island morphs into SEG island
- Pressing **A** while SEG island is visible: SEG island morphs into ROI island
- The morph transition: container stays in place, content crossfades, height animates if row count differs (~300ms)

## Positioning

- **Default position**: left side of the canvas pane
- **Draggable**: user can drag the island anywhere in the viewport via the header drag handle
- **Position memory**: once dragged, the island remembers its position. Both ROI and SEG islands share the same remembered position. Position persists across mode switches within a session.
- **Canvas space**: normally positioned adjacent to the canvas with sufficient horizontal space. If space is tight, the island overlays the canvas edge (glassmorphic background ensures readability)

## Styling

- **Background**: `rgba(24, 24, 24, 0.95)` with `backdrop-filter: blur(16px)`
- **Border**: `1px solid rgba(255, 255, 255, 0.07)`
- **Border radius**: 14px
- **Font**: monospace, 11px base, 10px for row content, 9px for stats
- **ROI accent color**: `#f0c674` (yellow)
- **SEG accent color**: `#e87aaf` (pink)
- **Selected row**: subtle tinted background + border matching mode color
- **Theme-aware**: respects dark/light/solarized/nord themes via CSS variables

## ROI Export Format

CSV download with bitmask dimension encoding.

The bitmask has one digit per array dimension. `1` = dimension included in the statistic, `0` = fixed at current index. The ROI plane dimensions are always `1`; additional `1`s mean the statistic is aggregated across that dimension.

Example: 5D array, ROI drawn on dimensions 1 and 3 (bitmask base: `01010`):

```csv
roi,dims,mean,std,min,max,n_pixels
ROI 1,01010,142.3,28.1,12,255,347
ROI 1,11010,135.1,31.2,3,255,17347
ROI 1,01110,140.2,29.8,5,253,8680
ROI 1,01011,138.9,30.1,8,252,6940
ROI 1,11111,128.4,35.7,0,255,347000
ROI 2,01010,89.7,15.4,22,201,182
ROI 2,11010,91.2,18.3,8,215,9100
...
```

Rows per ROI:
1. **Base slice** — just the 2D ROI plane (the two `1` bits)
2. **Single-dimension extensions** — one row per additional dimension, toggling one extra `1`
3. **All dimensions** — all bits set to `1`

## SEG Export Format

NPY file containing the label mask (existing behavior via `/seg/export/{sid}`).

## What Changes

- Remove ROI and SEG from the egg badge system (`renderEggs()`)
- Remove the existing `#seg-panel` — its functionality moves into the SEG island
- Add new island DOM element, shared between ROI and SEG
- ROI egg badge → island header
- SEG egg badge + seg-panel → island header + label rows
- Existing A/S key handlers gain mutual-exclusivity logic

## What Stays the Same

- All egg badges for composable transforms (FFT, LOG, MAGNITUDE, RGB, ALPHA, PROJECTION) remain unchanged
- ROI drawing mechanics (overlay canvas, shapes, freehand points, floodfill tolerance)
- SEG interaction mechanics (nnInteractive server communication, mask handling)
- A/S key cycling behavior (just adds dismissal of the other mode)
- All other keyboard shortcuts
