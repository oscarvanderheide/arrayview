---
name: modes-consistency
description: Use when a visual feature touches canvas rendering, zoom, eggs, colorbars, shortcuts, or layout across modes. Keeps implementation and verification consistent without escalating to a full audit by default.
---

# ArrayView Modes Consistency Checklist

## Fast Path

Use this when a change can affect more than one viewing mode.
Skip it for tiny local text/style tweaks that do not touch canvas behavior, colorbars, shortcuts, layout, or mode transitions.
If the user explicitly asks for a full visual audit or release validation, use `ui-consistency-audit` instead.

## Rule

If a feature applies to more than one mode, implement it for all applicable modes now. Shipping “normal mode only” is a bug.

## Mode Map

| Mode | Main owner |
|------|------------|
| Normal | `scaleCanvas()` |
| Multi-view | `mvScaleAllCanvases()` |
| Compare / Diff / Registration | `compareScaleCanvases()` |
| qMRI | `qvScaleAllCanvases()` |

## Check These Areas

- Zoom or canvas sizing: update every relevant scale function
- Eggs: update `positionEggs()` branches
- Colorbars and range interaction: check each per-mode colorbar path
- Shortcuts: guard by mode and update help/registry consistently
- Canvas listeners: attach to the correct canvas set, not just `#canvas`
- New UI state: persist it through snapshot/restore if needed

## Backend Note

If the feature changes rendered image composition, check the rendering backend in:

- `_routes_rendering.py` for `/slice`, `/diff`, and related routes
- `_render.py` / `_overlays.py` for overlay compositing such as `_composite_overlay_mask()`

Do not treat `_app.py` as the implementation surface; it is a compat shim.

## Minimal Workflow

1. Identify which modes the feature should affect.
2. Implement normal mode first.
3. Implement compare-family behavior.
4. Implement multiview behavior.
5. Implement qMRI behavior if applicable.
6. Verify snapshot/restore if new state was added.

## Red Flags

- Only normal mode was changed
- Similar logic diverges between `scaleCanvas()` and `compareScaleCanvases()`
- A listener was attached only to `#canvas`
- Render-route params changed in one backend path but not the others

## Verification

- [ ] Works in normal view
- [ ] Works in compare or diff if applicable
- [ ] Works in multiview if applicable
- [ ] Works in qMRI if applicable
- [ ] Eggs/colorbars/layout anchors still line up after the change
