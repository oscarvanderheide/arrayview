# Display

## Colormaps

`c` opens the colormap grid. Press `c` again to cycle; `Enter` accepts, `Esc` cancels.
`C` enters any matplotlib colormap by name.

## Dynamic range

`d` opens the histogram and cycles the 0–100%, 0.1–99.9%, 1–99%, 5–95%, and 10–90% range presets. The active range and preset dots appear briefly over the image.
`D` opens a histogram dim picker (3-state toggle for which dimensions participate).

## Window / level

Drag the colorbar to shift the window. Scroll on the colorbar to narrow or widen the range. Double-click to reset.

## Log scale

`L` toggles logarithmic display.

## Themes

`T` cycles: dark (default), light.

## Masking

`M` cycles Otsu threshold levels (8 steps). Useful for isolating structures.

## Layout

`b` toggles a border around the canvas. `B` toggles rounded panes. `a` stretches panes to a square aspect ratio.

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
