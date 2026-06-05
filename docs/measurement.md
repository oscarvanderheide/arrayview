# Measurement

## ROI

`Shift+R` opens ROI options. Draw on the canvas to measure a region — live stats (mean, std, min, max) appear in the floating panel.

Shapes: rectangle, ellipse, freehand, flood fill. Switch via the shape icons in the panel.

For flood fill, `[` / `]` adjusts tolerance.

Click an existing ROI to select it. `Delete` / `Backspace` removes it.

`N` exports the ROI mask as `.npy`.

## Ruler

`u` — enter ruler mode. Click two points to measure pixel distance. Press `u` again to exit.

## Pixel info

Hover over the image to see coordinates and value reflected on the colorbar.

Click the colorbar to copy that value to the clipboard.

`i` — toggle a persistent hover tooltip that follows the cursor.

`I` — show a data info overlay: shape, dtype, size, file path.

## Export

| Key | Action |
|-----|--------|
| `s` | Open save options (screenshot PNG, GIF, .npy export) |
| `e` | Copy a reusable URL to clipboard |

Screenshots download as PNG. GIF saves an animation along the current slice dimension. `.npy` export saves the current slice.

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
