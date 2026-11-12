# Display

## Colormaps

`c` opens the colormap list. Type to search any colormap by name, or press `c` again to cycle; `Enter` accepts, `Esc` cancels.

## Dynamic range

The first `d` press opens the histogram without changing or classifying the current range. Further presses cycle the 0–100%, 0.1–99.9%, 1–99%, 5–95%, and 10–90% presets and briefly show the active range and preset dots over the image.
While the histogram is open, the dimension labels are coloured by how each dimension counts toward the range. Click a dimension to change how it counts; hover it to see what each colour means.

## Window / level

With pixel inspection off, hold and drag on the image to adjust window/level. The cursor grows and labels the two drag directions while the adjustment is active.

Drag the colorbar to shift the window. Scroll on the colorbar to narrow or widen the range. Double-click it to open the histogram; click the number at either end to type a value, and the padlock next to it keeps that value fixed.

## Log scale

`L` toggles logarithmic display.

## Themes

`T` cycles: dark (default), light.

## Masking

`M` makes everything below the range minimum transparent. Press `M` again to turn it off.

`Shift+4` crops the display to the nonzero part of the array plus a small margin. It only changes the view, not the data.

## Layout

`b` toggles a border around the canvas. `B` toggles rounded panes. `a` stretches panes to a square aspect ratio. `Shift+D` shows or hides the size of each dimension in the dimension bar.

## Caveat

ArrayView works across six invocation environments (CLI, Python script, Jupyter, Julia, VS Code, SSH). Not every feature has been verified in every mode. If something behaves differently than documented, check the [remote](remote.md) page for environment-specific notes or open an issue.
