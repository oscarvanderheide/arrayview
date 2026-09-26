# Measurement

## ROI

`Shift+R` shows or hides ROI mode. Hiding ROI mode keeps the session's ROIs.

Draw on the canvas to measure a region. The default shape is a circle.

Shapes: circle, rectangle, freehand, flood fill. Switch via the colorbar controls.

For flood fill, `[` / `]` adjusts tolerance.

Click an existing ROI to select it. `Delete` / `Backspace` removes it.

`Stats` opens the ROI manager: rename/delete ROIs, adjust extent, export CSV, or export a label mask.

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
| `s` | Open save options (screenshot PNG, GIF, .npy export) |
| `e` | Copy a reusable URL to clipboard |
| `E` | Copy the current view as a line of code |

Screenshots download as PNG. GIF saves an animation along the current slice dimension. `.npy` export saves the current slice.

`E` copies one line that reopens what you are looking at: the displayed axes, the slice position, the colormap, a range you set, and log scale. Settings still at their defaults are left out. An array passed from Python gives a `view()` call using your variable's name; from Julia, an `arrayview.view(...)` call; an array opened from a file gives an `arrayview` command:

```python
view(vol, dims=(2, 1), index=(8, 32, 24), cmap="gray", vmin=-0.1235, vmax=3.142)
```

```bash
arrayview ~/data/vol.npy --index 3,32,24 --cmap magma --log --vmin=-1.5 --vmax 2
```

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
