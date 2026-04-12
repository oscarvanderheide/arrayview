# Viewing

## Normal

The default view. Shows a single 2D slice. Scroll to move through slices. `h`/`l` changes the scroll dimension; `j`/`k` steps one slice at a time.

## Multi-view

`v` — splits the canvas into three orthogonal planes (axial, coronal, sagittal). Drag the crosshair to navigate. Shift+drag for oblique rotation. `V` opens custom dimension selection; `o` resets the orientation.

## Mosaic

`z` — lays out a grid of slices along the next dimension. Scrolling then steps through the dimension after that.

## Projections

`p` — cycles through statistical projections along the scroll axis: MAX, MIN, MEAN, STD, SOS. 
## FFT

`f` — displays a centered FFT of the array. Prompts for which axes to transform.

## RGB

`R` — renders the array as direct color. The first or last dimension must be size 3 (RGB) or 4 (RGBA). Also available as `--rgb` on the CLI or `rgb=True` in Python.

## Immersive

`F` — fullscreen with all chrome hidden. Move the mouse to briefly reveal controls.

## Complex data

`m` — cycles through magnitude, phase, real, and imaginary parts for complex-valued arrays.

## Navigation

Scroll through slices. `/` opens the special modes shelf for ortho, compare, qMRI, ROI, segmentation, FFT, projections, and overlays. The direct keys still work. `0` resets the zoom. `?` shows the full key guide.
