# Measurement

## ROI

`i` (or double-clicking a pane) shows or hides ROI mode. It shows the value under the cursor, and dragging draws a region. Hiding ROI mode keeps the session's ROIs.

Draw on the canvas to measure a region. It starts with your default shape, a circle unless you set `viewer.roi_shape` (Preferences, or `arrayview config set viewer.roi_shape freehand`).

Shapes: circle, rectangle, freehand, flood fill. The button at the bottom right, left of the logo, shows the shape you are drawing; click it to change shape. The menu says how to draw the shape under the pointer, and links to Preferences for the shape `i` starts with. Holding `Alt` (`Option` on Mac) opens the same menu at the pointer: release over a button to pick it, or tap `Alt` to keep the menu open.

For flood fill, `[` / `]` adjusts tolerance. While holding the seed, drag up/down to take in more or less, and right/left for smoother or rougher edges (from off to 5; it starts at 2.5 and remembers your last setting). For a 3D fill, the card beside the cursor says when the full 3D region is still catching up. Smoothing blurs the image lightly before deciding what joins, then rounds the region's edge and fills small holes.

Click an existing ROI to select it. `Delete` / `Backspace` removes it.

The numbers panel appears with the first ROI, at the top right; drag its header to put it somewhere else, and it opens there next time. `×` or `Esc` closes it; it comes back with the next ROI you draw. It lists every ROI with its mean ± standard deviation. Hovering a row, or the ROI itself, highlights that ROI and shows its mean ± std next to it on the image. Click a row for its distribution (a histogram on a value axis shared by all ROIs, with a tick at the mean), pixel count, min and max; in qMRI it lists the numbers for every map. Double-click a name to rename it. `copy` puts the numbers on the clipboard as a table that pastes straight into a spreadsheet; `CSV` downloads the same table. It has one row per ROI with its mean, std, min, max and pixel count; in qMRI each map gets its own mean/std/min/max columns. `mask` downloads a label mask.

### Side by side

In side by side (`s`, or several arrays opened together) a ROI drawn on any
pane appears on every pane, and the panel lists one line of numbers per pane
under each ROI — the same region in recon A and recon B, or in echo 1 and
echo 5. Hovering a ROI shows each pane's own mean ± std next to it. The numbers
follow when you scroll, move a pane with `[`/`]`, or add and remove panes. ROIs
carry over when you go back to one image. `copy` and `CSV` give one row per ROI
and pane. With a difference image in the middle (`Shift+X`: A−B, |A−B| or
relative) the ROI shows there too, with its own line of numbers.

### Mosaic, 3-plane and turning the view

In a mosaic (`z`, `Shift+→`, `Shift+↓`) a ROI drawn on one tile appears on
every tile, and the panel lists one line per tile, each measured on that
tile's slice. Hovering shows the numbers of the tile under the pointer.

In 3-plane view (`v`) a ROI belongs to the pane it was drawn on and is
measured on that pane's plane.

`t`, `r`, `x`/`y` and mirroring turn ROIs with the image, so they keep
measuring the same pixels. `o` switches to another slicing direction: ROIs
drawn in the old one are hidden (their row in the panel is dimmed) and come
back when you return to it.

ROIs are not available in 3-plane or qMRI rows of a side-by-side view, or when
the panes differ in size. Flood fill is not available side by side or in a
mosaic.

`N` exports the active ROI or segmentation mask as `.npy`.

## Ruler

`u` — enter ruler mode. Click two points to measure pixel distance. Press `u` again to exit.

## Pixel info

Hover over the image to see coordinates and value reflected on the colorbar.

Click the colorbar to copy that value to the clipboard.

`i` — keep the pixel value visible in the pane pill. The cursor's center turns yellow while this inspection mode is active. Hold `Ctrl` for the loupe; its value stays at the bottom unless the loupe gets close, then moves to the top.

Inspection mode does not draw regions. Enter ROI mode when you want measurements or region statistics.

While pixel info is on, use middle-drag or `Ctrl`/`Cmd`+`Shift`-drag to adjust the display range explicitly. When zoomed in, plain drag still pans.

`I` — show a data info overlay: shape, dtype, size, file path.

## Export

| Key | Action |
|-----|--------|
| `Shift+S` | Open save options (screenshot PNG, GIF, .npy export) |
| `e` | Copy a reusable URL to clipboard |
| `E` | Copy the current view as a line of code |

Screenshots download as PNG. GIF saves an animation along the current slice dimension. `.npy` export saves the current slice.

`E` copies one line that reopens what you are looking at: the viewing mode (ortho, mosaic, qMRI and their combinations), the displayed axes, the slice position, the colormap, a range you set, log scale, and which part of complex data is shown (magnitude, phase, real, imaginary). Settings still at their defaults are left out. An array passed from Python gives a `view()` call using your variable's name; from Julia, an `arrayview.view(...)` call; an array opened from a file gives an `arrayview` command:

```python
view(vol, dims=(2, 1), index=(8, 32, 24), cmap="gray", vmin=-0.1235, vmax=3.142)
```

```bash
arrayview ~/data/vol.npy --index 3,32,24 --cmap magma --log --vmin=-1.5 --vmax 2
```

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
