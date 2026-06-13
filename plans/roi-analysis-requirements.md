# ROI Analysis Requirements

## Purpose

ROI analysis should be a lightweight MRI/qMRI measurement tool. It should make
quick measurements easy, support repeated phantom-well measurements, and export
the resulting statistics and label masks.

ROI analysis is not a replacement for segmentation. Full white matter / gray
matter segmentation, voxel-label editing, automatic well detection, and mask
roundtripping belong outside ROI v1.

The current uncommitted ROI upgrade work is superseded by this document. Useful
backend or mask-export ideas may be reused, but the UI and interaction model
should restart from these requirements.

## Primary Use Cases

- Draw a quick circular or freehand ROI in in-vivo MRI data to inspect mean,
  standard deviation, minimum, and maximum values.
- Draw multiple true circular ROIs inside cylindrical gel phantoms in transverse
  slices and compare their measurements.
- In qMRI mode, draw one ROI and measure the same region across visible
  parameter maps.
- Export ROI statistics as CSV.
- Export a single integer `.npy` label mask matching the source array shape.

## Interaction Model

- `Shift+R` toggles ROI mode visibility and interaction.
- When ROI mode turns off, ROI shapes, buttons, and hover affordances disappear.
- Turning ROI mode back on restores the session's existing ROI shapes.
- ROI geometry persists only for the current viewer session.
- The default ROI shape is a true circle.
- Additional ROI shapes are rectangle/square, freehand, and flood fill.
- Click an existing ROI to select it.
- `Delete` or `Backspace` deletes the selected ROI.
- No automatic or suggest-only circle detection in ROI v1.

## UI Model

- Everyday ROI controls live on a colorbar flip surface, not in the right tool
  drawer.
- The colorbar flip should expose only the common drawing controls: shape
  picker, stats popup button, and minimal clear/hide actions if needed.
- Statistics open in a popup/modal.
- The stats popup is the ROI manager for v1:
  - one row per ROI by default
  - qMRI rows expand or reveal per-map measurements
  - rename ROI
  - delete ROI
  - export CSV
  - export label mask `.npy`
  - advanced dimensional extent controls
- Do not put rename/delete controls directly on the canvas in v1.
- Keep the canvas visually quiet; ROI outlines and labels are the only persistent
  ROI marks while ROI mode is active.

## Statistics and Dimensional Scope

- Normal mode default: a new ROI applies only to the current displayed slice.
- qMRI default: a new ROI applies to the visible qMRI parameter maps.
- Advanced dimensional extent controls live in the stats popup per ROI.
- Advanced controls must use plain-language labels, not debug-like labels such
  as `d2:135` or `d3:all`.
- Stats should report at least count, mean, standard deviation, minimum, and
  maximum.
- CSV export should include ROI name, shape, dimensional extent, and qMRI map
  labels where applicable.

## Masks and Segmentation Boundary

- ROI mask export writes one `.npy` integer label mask matching the source array
  shape.
- Label `0` is background.
- ROI labels are `1..N` in list order.
- ROI mask loading and roundtrip editing are out of scope for ROI v1.
- Segmentation remains the tool for voxel-label editing, external segmentation
  workflows, brush/lasso editing, and future mask roundtripping.

## Visibility and Color

- ROI outlines must remain visible across colormaps, themes, and local image
  brightness.
- Keep per-ROI identity colors, but do not rely on color alone for visibility.
- Use an adaptive high-contrast halo or dual-outline rendering for ROI borders.
- ROI labels should use contrast-aware text/halo rendering.

## Implementation Direction

- Rebuild ROI around a small session-local model:
  `id`, `name`, `shape`, `geometry`, `scope`, `visible`, `selected`.
- Use true shape masks for stats and export. Do not fall back to bounding boxes
  for circle, freehand, or flood fill.
- Route ROI visibility through existing UI reconcilers and colorbar flip
  patterns.
- Update the command registry and `GUIDE_TABS` when finalizing shortcuts and
  help text.
- Update `docs/measurement.md` only after implementation matches this spec.

## Test Plan

- Spec review first: confirm this document captures the intended ROI direction
  before rebuilding the implementation.
- Backend tests:
  - true circle mask stats
  - freehand and floodfill mask stats
  - qMRI per-map stats
  - `.npy` label mask shape and label values
- Browser tests:
  - `Shift+R` hide/show preserves session ROIs
  - default drawing shape is circle
  - stats popup supports rename, delete, CSV export, and mask export
  - qMRI mode shows per-map stats
  - ROI outlines remain readable across themes/colormaps
- Regression tests:
  - simple ROI drawing does not require the right drawer
  - no circle suggestion UI or API
  - no cryptic dimensional labels

## Assumptions

- ROI and segmentation remain separate tools.
- ROI v1 prioritizes fast measurement and export over full mask management.
- Session-only ROI geometry is acceptable.
- Exported CSV and `.npy` masks are the persistence boundary for ROI v1.
