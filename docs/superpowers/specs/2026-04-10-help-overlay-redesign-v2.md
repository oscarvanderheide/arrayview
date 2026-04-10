# Help Overlay Redesign v2

## Problem

The current help overlay was built yesterday as a curated Guide with 6 tabs. The structure is good but the content needs significant improvement: descriptions are too terse to help new users understand what a mode is for, some keybinds are wrong or missing, mouse interactions aren't mentioned, and the tab grouping doesn't reflect the actual feature landscape (ROI and Segmentation shouldn't be separate tabs; overlays need their own tab).

## Solution

Revise the help overlay content: 7 tabs with clear what/why/when descriptions, corrected keybinds, mouse interactions, and a new Overlays tab. The visual styling and infrastructure (tabbed layout, lazy build, theme support) are already in place — this is a content-only change to the `GUIDE_TABS` data structure.

## Design

### Tab Structure (7 tabs)

| # | Tab | Purpose |
|---|-----|---------|
| 1 | Basics | Navigation, zoom, playback, info, export — works everywhere |
| 2 | Display | Color, range, colorbar interactions, transforms (FFT, projections, complex) |
| 3 | Ortho View | 3-panel crosshair mode for 3D volumes |
| 4 | Compare | Side-by-side viewing with diff/overlay/wipe/flicker/checkerboard operations |
| 5 | Tools | ROI, Ruler, Segmentation (and future Crop) — draw-on-image interactions |
| 6 | Overlays | Segmentation masks and vector fields loaded via CLI flags |
| 7 | qMRI | Quantitative MRI parameter maps with units and tissue presets |

### Tab Content

Each tab has:
1. **Description** — 2-3 plain-English sentences explaining what/why/when. No jargon.
2. **Keybinds & interactions** — grouped by sub-category where needed. Each entry has:
   - Key/action in blue monospace
   - Action label in white
   - Optional hint in muted gray

**Principles:**
- Generic keybinds (arrows, scroll, c, d, L, f, etc.) only appear in Basics or Display — never repeated in mode-specific tabs
- Mouse interactions are included where they matter (colorbar, crosshair drag, pane reorder)
- Use R/r instead of Shift+R/R (case conveys shift)
- Basics and Display use 2-column layout; all others use single-column
- Tools tab uses sub-sections with mini-descriptions for each tool

### Tab 1: Basics

**Description:** "The essentials for exploring any array. Move through slices, zoom into details, or play through a volume as an animation. These controls work everywhere."

**Layout:** 2 columns

**Left column:**

*Slices & Dimensions:*
- `← → ↑ ↓` — Step through slices
- `x / y` — Change viewing dimension (hint: choose which axes map to horizontal and vertical)
- `z` — Mosaic — tile all slices in a grid
- `1–9 Enter` — Jump to a specific slice number

*Playback:*
- `Space` — Play / pause animation
- `< >` — Slower / faster playback

*Info & Search:*
- `i` — Toggle hover pixel info (hint: shows coordinates and value under cursor)
- `I` — Info panel (hint: array shape, dtype, memory layout)
- `/` — Command palette (hint: search all available commands)

**Right column:**

*Zoom & Pan:*
- `Pinch / ⌘Scroll` — Zoom in / out (hint: Ctrl+Scroll on Windows / Linux)
- `0` — Reset zoom to fit
- `Drag` — Pan the image (when zoomed in)
- `F` — Zen mode — hide UI chrome (hint: strips away panels and controls to focus on the image)

*Export:*
- `s` — Save screenshot
- `g` — Save GIF animation
- `e` — Copy shareable URL

### Tab 2: Display

**Description:** "Control how your data looks. Adjust colors and the display range to bring out features that aren't visible with default settings."

**Layout:** 2 columns

**Left column:**

*Color & Range:*
- `c` — Cycle colormap (hint: grayscale, viridis, turbo, and more)
- `d` — Cycle display range presets (hint: clips outliers — press repeatedly to tighten)
- `L` — Toggle log scale (hint: when values span several orders of magnitude)
- `T` — Cycle theme (hint: dark, light, solarized, nord)
- `b` — Toggle canvas borders

*Colorbar:*
- `Scroll` — Narrow or widen the display range
- `Drag handles` — Adjust min / max bounds
- `Drag bar` — Shift the visible range up or down
- `Dbl-click` — Expand histogram
- `Dbl-click label` — Type an exact min or max value

**Right column:**

*Transforms:*
- `f` — FFT — view frequency domain (hint: press again to return to spatial domain)
- `p` — Cycle projections (hint: MAX → MIN → MEAN → STD → off)
- `m` — Complex mode cycle (hint: magnitude → phase → real → imaginary)

### Tab 3: Ortho View

**Description:** "Three synchronized panels showing your volume from three directions at once. A crosshair links the views — clicking in one panel updates the other two. Great for exploring 3D data like MRI scans, where you need to see axial, sagittal, and coronal slices together."

**Layout:** 1 column

*Controls:*
- `v` — Enter or exit ortho view
- `Click / Drag` — Set crosshair position (hint: click any panel to jump all views to that point, or drag to scrub)
- `Drag lines` — Reposition slice planes (hint: each colored line corresponds to a slice in another panel)
- `p` — Maximum intensity projection / MIP (hint: 3D volume rendering — drag to rotate)
- `o` — Reset to origin

### Tab 4: Compare

**Description:** "View two or more arrays side by side, blend them together, or compute a difference map. Useful for checking registration, comparing model outputs, or inspecting before-and-after changes. Open multiple arrays with the file picker first."

**Layout:** 1 column

*Layout:*
- `G` — Cycle layout (hint: horizontal → vertical → grid)
- `Drag title` — Reorder panes by dragging their title bar

*Operations (requires exactly 2 arrays):*
- `X` — Cycle compare operation (hint: off → A−B → |A−B| → relative → overlay → wipe → flicker → checkerboard)
- `[ ]` — Adjust operation parameter (hint: blend amount, flicker speed, or checkerboard tile size)

### Tab 5: Tools

**Description:** "Draw on the image to measure, segment, or crop. Each tool uses click or drag to define a region, and shows results in real time."

**Layout:** 1 column with sub-sections

**ROI — Region of Interest**
Sub-description: "Draw a region to see live statistics (mean, std, min, max). The histogram updates to show only the selected area."
- `R` — Toggle ROI mode
- `r` — Cycle shape (hint: rectangle → circle → freehand → flood fill)
- `Drag` — Draw ROI region
- `Right-click` — Delete ROI under cursor
- `Esc` — Clear all and exit

**Ruler**
Sub-description: "Click two points to measure pixel distance."
- `u` — Toggle ruler mode

**Segmentation**
Sub-description: "Interactive segmentation with nnInteractive. Click to add or exclude regions from the label."
- `S` — Toggle segmentation mode
- `Click` — Add region (positive prompt)
- `Shift+Click` — Exclude region (negative prompt)
- `Esc` — Exit segmentation

**Crop** (not yet implemented — omit from help until the PR lands; the sub-section structure supports adding it later)

### Tab 6: Overlays

**Description:** "Extra layers displayed on top of your array. Load segmentation masks with `--overlay` or deformation fields with `--vectorfield` when launching arrayview. Press O to toggle visibility for all overlays at once."

**Layout:** 1 column with sub-sections

**Segmentation Masks**
Sub-description: "Binary masks rendered as colored layers on the image. Load one or more with `--overlay mask1.nii,mask2.nii`. Each mask gets its own color."
- `O` — Cycle visibility (hint: all → none → individual masks if multiple)
- `[ ]` — Adjust opacity

**Vector Field**
Sub-description: "Deformation or flow fields rendered as arrows. Load with `--vectorfield field.nii`."
- `O` — Toggle visibility
- `[ ]` — Adjust arrow density
- `{ }` — Adjust arrow length

### Tab 7: qMRI

**Description:** "For quantitative MRI parameter maps — T1, T2, FA, and others. Voxel values are shown with proper units and the display adjusts to tissue-specific intensity ranges. Use this when viewing parameter maps to get meaningful readouts instead of raw numbers."

**Layout:** 1 column

*Controls:*
- `q` — Enter / cycle / exit qMRI mode (hint: steps through available parameter types)
- `z` — Toggle qMRI mosaic (hint: see all parameter maps side by side in a grid)

### UI Chrome Changes

- **Remove topbar** "press ? to close" — users discover this naturally
- **Remove footer left** tab-switch hint (h/l / ←/→) — same reason
- **Keep footer right** "Full reference →" link to docs

### Code Changes Already Committed

- `O` (Shift+O) now handles both mask overlay cycling AND vector field toggle (commit d610509)
- `U` keybinding for vector field removed
- `overlay.cycleVisibility` command `when` condition changed from `['hasOverlay']` to `[]` to handle vfield-only case

## Scope

- Replace `GUIDE_TABS` content in `_viewer.html` (data structure only, no DOM logic changes needed)
- Remove topbar hint and footer-left hint from the HTML template
- Update `buildHelpOverlay()` if needed to support the Tools/Overlays sub-section layout (`.help-subsection-title` + `.help-subsection-desc` elements)
- Verify all keybinds mentioned are correct against actual command registry
- Out of scope: docs reference page, visual_smoke.py updates (separate task)
