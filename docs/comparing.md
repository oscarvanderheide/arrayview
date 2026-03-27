# Comparing

## Opening multiple arrays

```bash
uvx arrayview base.npy moving.npy           # 2 arrays
uvx arrayview base.npy a.npy b.npy c.npy    # up to 6
```

File picker: `Cmd/Ctrl+O`, select multiple files, Enter. Or drag-and-drop a file onto the viewer — compatible shapes go to compare.

## Center pane

`X` cycles through comparison modes:

- **A−B** — signed difference
- **|A−B|** — absolute difference
- **|A−B|/|A|** — normalized difference
- **Overlay** — blend two arrays
- **Wipe** — cursor-following split
- **Flicker** — rapid A/B toggle
- **Checkerboard** — alternating tiles

`Z` zooms into the center pane.

## Layouts

`G` cycles: horizontal, vertical, grid. Drag panel titles to reorder panes.

## Overlay

```bash
uvx arrayview scan.nii.gz --overlay mask.nii.gz
```

Binary mask (0/1), same spatial shape. `[`/`]` adjusts blend opacity. Multiple overlays get automatic palette colors.

## Vector field

```bash
uvx arrayview scan.nii.gz --vectorfield disp.nii.gz
```

Deformation field with arrow overlay. `U` toggles arrows. `[`/`]` adjusts density. `{`/`}` adjusts arrow length.
