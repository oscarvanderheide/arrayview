# Viewing

Once an array is loaded, these modes change how it's displayed.

## Normal

The default view. Shows a single 2D slice. Scroll to move through slices. `h`/`l` changes the scroll dimension; `j`/`k` steps one slice at a time.

## Multi-view

`v` — splits the canvas into three orthogonal planes (axial, coronal, sagittal). Drag the crosshair to navigate all planes at once. Shift+drag rotates the view obliquely. `V` opens custom dimension selection; `o` resets the orientation.

## Mosaic

`z` — lays out a grid of slices along the next dimension. Scrolling then steps through the dimension after that.

## Projections

`p` — cycles through statistical projections along the scroll axis: MAX, MIN, MEAN, STD, SOS. Useful for quickly surveying the full extent of a volume.

## FFT

`f` — displays a centered FFT of the array. Prompts for which axes to transform.

## RGB

`R` — renders the array as direct color. The first or last dimension must be size 3 (RGB) or 4 (RGBA). Also available as `--rgb` on the CLI or `rgb=True` in Python.

## Immersive

`F` — fullscreen with all chrome hidden. Move the mouse to briefly reveal controls.

## Complex data

`m` — cycles through magnitude, phase, real, and imaginary parts for complex-valued arrays.

## Navigation basics

- Scroll: slices. `+`/`-`: zoom. `0`: fit to window.
- `x`/`y`: swap the display axis with the scroll axis. `t`: transpose.
- `Space`: auto-play through slices.
