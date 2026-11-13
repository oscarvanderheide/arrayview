# Viewing

## Normal

The default view. Shows a single 2D slice. Scroll to move through slices. `h`/`l` changes the scroll dimension; `j`/`k` steps one slice at a time.

`r` rotates 90 degrees or flips the axis. `t` transposes x and y.

`o` steps through the three views of a volume (axial → coronal → sagittal), the same planes the multi-view shows. Each view keeps its own `r`/`t` rotation and flip. NIfTI views are named and set upright from the file's stored orientation, whatever order it stores its axes in. For arrays without one (e.g. `.npy`), `Shift+A` lets you name the current view and give the voxel size, e.g. `axial 1 1 3`.

When the voxel size is known (NIfTI, or set with `Shift+A`), images are drawn in their true proportions, so thick slices look tall rather than squashed. `a` (stretch to square) overrides this.

## Multi-view

`v` — splits the canvas into three orthogonal planes (axial, coronal, sagittal). Click or drag in any pane to move the crosshair. Hold Alt and drag a line to tilt that plane (oblique). `V` lets you pick which dimensions the three planes use; `o` moves the crosshair back to the origin.

## Mosaic

`z` — lays out a grid of slices along the next dimension. Scrolling then steps through the dimension after that; a plain 3-D volume shows all its slices at once.

`Shift+→` lays the active dimension out across the screen, `Shift+↓` lays it out down. Use one dimension each way for a grid of both, for example echoes across and time points down. `Shift+←` / `Shift+↑` take the across / down dimension out again, and `z` leaves the mosaic. `r`, `t`, `o` and `w` work on the tiles.

## Projections

`p` — cycles through statistical projections along the scroll axis: MAX, MIN, MEAN, STD, SOS, SUM, then off. The first press shows the options; click one to jump to it.

## FFT

`f` — displays a centered FFT of the array. Prompts for which axes to transform.

## RGB

Rendered as direct color via `--rgb` on the CLI or `rgb=True` in Python. The first or last dimension must be size 3 (RGB) or 4 (RGBA). No direct keybind — set at launch.

## Immersive

`Shift+F` — the image fills the window from top to bottom, with the dimbar and colorbar floating on it. Only opens when height is what limits the image; when width is the limit (side-by-side compare, a wide array in a narrow window) it says so and stays put. In ortho view and compare-centre it needs the big-left layout, and switching away from big left closes it.

## Quick zoom

`w` — in a view with several panes (ortho, qMRI, compare, side by side), shows the pane under the mouse on its own, as large as the free space allows. The other panes blur behind it, and the dimbar and a shared colorbar slide to the window edges to make room. Scrolling, hovering and the other keys keep working on it. Press `w` again or `Esc` to bring the other panes back.

## Fill the window (VS Code)

`Shift+W` — makes the VS Code tab fill the whole VS Code window: the side bars, bottom panel, activity bar, status bar and any split editors are hidden. Press `Shift+W` again to bring back exactly what was open before. The tab does not move or reload.

## Zen mode

`Z` — hides UI chrome (dimbar, info, island) for a clean view. Press `Z` again to restore.

## Complex data

`m` — cycles through magnitude, phase, real, and imaginary parts for complex-valued arrays.

## qMRI

`q` — opens qMRI for a parameter dimension. Hold Alt/Option while qMRI is active to choose quantitative parameters and synthesised contrasts. Synthetic contrasts show TE/TR/TI sliders below each pane when available.

## Navigation

Scroll through slices. `/` opens the tool launcher for ortho, compare, qMRI, ROI, segmentation, FFT, projections, and overlays. The direct keys still work. `0` resets the zoom. `?` shows the full key guide.

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
