# Comparing

## Opening multiple arrays

```bash
uvx arrayview base.npy moving.npy           # 2 arrays, side by side to compare
uvx arrayview base.npy a.npy b.npy c.npy    # up to 4, side by side
uvx arrayview run_*.npy                     # 5 or more: one at a time
uvx arrayview a.npy b.npy --overlay mask.npy   # the mask on every array it fits
uvx arrayview a.npy b.npy --watch              # each pane follows its own file
```

Several files open as one array, with the files along a new last dimension in
the order you typed them (a `|` in the dimension bar marks it). Files may have
different shapes. Two to four open side by side; five or more show one at a
time. A folder or `--match` collection does the same. `--compare FILE` is just
one more file. An `--overlay` mask is drawn on every array of the same shape,
and `--watch` reloads whichever file changed.

From Python, `view(a, b)` does the same. The arrays may differ in size, as long
as they have the same number of dimensions:

```python
ha, hb = arrayview.view(a, b)   # side by side, panes named "a" and "b"
hb.update(b2)                   # replaces only the second pane
arrayview.view(a, b, overlay=mask)
```

Each array still gets its own handle; closing any of them closes the view.
The same call works from Julia and MATLAB (see [MATLAB and Julia](foreign-hosts.md)).

Only colour images (`--rgb` / `rgb=`), a `--vectorfield`, and arrays with
different numbers of dimensions still open in the older compare view.

## Side by side

`s` shows the active dimension in several panes — the files you loaded, or
any other non-spatial dimension such as echoes or time points. Pressing `s`
opens a strip of pane counts (two to six) at the bottom of the picture: press
a digit or click a count, or move with the arrows and press Enter. `s` again
or Esc closes the strip without changing anything.

The dimension bar shows the dimension as a purple `s`; each pane names its own
index in its corner (and, for loaded files, the part of the file name that
differs). In side by side, `s` goes back to one image, and holding `s` brings
the strip back for another number of panes: choose, then release `s`.

- `[`/`]` move just the pane under the mouse along the dimension, so you can
  put echo 1 next to echo 5.
- **Two panes compare.** Scrolling moves both through slices together. `X`
  opens the center pane (below).
- **Three or more browse.** They show neighbouring positions; `j`/`k` on that
  dimension pages to the next few.

`v` (3-plane) and `q` (qMRI) show one row per pane.

Or use the file picker (`Shift+Y`) or drag-and-drop.

## Comparing with the array on screen

Press `/` then `c` (**Compare with…**). The file picker opens with the array
on screen ticked as the first pane. Press `1`–`9` to compare with one of the
open arrays straight away, or tick several open arrays or files (up to six
panes) and press Enter. Arrays of a different shape fade out. This works in
VS Code tabs, where `Cmd/Ctrl+O` is taken by VS Code.

The chosen arrays then reopen together the way several files on the command
line do: one array with them along a new last dimension, shown side by side.
Dropping a file and choosing **Compare side by side** does the same. When
that is not possible (different shapes, RGB, overlays on screen) they open in
the older compare view.

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
