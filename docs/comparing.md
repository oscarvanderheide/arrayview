# Comparing

## Opening multiple arrays

```bash
uvx arrayview base.npy moving.npy           # 2 arrays
uvx arrayview base.npy a.npy b.npy c.npy    # up to 6
```

Or use the file picker (`Shift+Y`) or drag-and-drop.

## Comparing arrays from other tabs

Press `/` then `c` (**Compare with…**) to pick an array that is open in
another tab. It opens side by side with the current one. Arrays with a
different shape are listed but greyed out. This works in VS Code tabs, where
`Cmd/Ctrl+O` is taken by VS Code.

## Switching between arrays

`n` cycles the compare target array (when multiple arrays are loaded).

## Center pane

`X` cycles through comparison modes:

- **A−B** — signed difference
- **|A−B|** — absolute difference
- **|A−B|/|A|** — normalized difference
- **Overlay** — blend two arrays
- **Wipe** — cursor-following split
- **Flicker** — rapid A/B toggle
- **Checkerboard** — alternating tiles

In the three difference views the center pane's title shows numbers for
what is on screen: **rmse** (root-mean-square of A−B), **max** (largest
|A−B|) and **rel** (rmse relative to the RMS of A, as a percentage).

`Z` zooms into the center pane.

## Layouts

`g` cycles: horizontal, vertical, grid. With two arrays and a center pane,
it toggles horizontal and big-left. Drag panel titles to reorder panes.

## Overlay

```bash
uvx arrayview volume.nii.gz --overlay mask.nii.gz
```

Binary mask (0/1), same spatial shape. `[`/`]` adjusts blend opacity. Multiple overlays get automatic palette colors. Hover an overlay HUD row to focus that mask while dimming the others. Use the overlay HUD or `/ o` to switch between filled masks and outline-only contours.

For cohort review, use `--match` so image and overlay patterns are paired into
aligned stacks:

```bash
uvx arrayview --match "data/**/*_0000.nii.gz" --overlay "gt=data/**/labels/*.nii.gz"
```

## Vector field

```bash
uvx arrayview volume.nii.gz --vectorfield disp.nii.gz
```

Deformation field with arrow overlay. `U` toggles arrows. `[`/`]` adjusts density. `{`/`}` adjusts arrow length.

## Linking tabs

Press `/` then `l` (**Link tabs**) in two or more viewer tabs. Moving through
slices in one linked tab then moves the others to the same position, for every
dimension the arrays have in common (same number of dimensions, same size).
A link icon in the lower-left corner shows the tab is linked; click it, or
`/ l` again, to unlink.
