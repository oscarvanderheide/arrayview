import matplotlib.pyplot as plt
import numpy as np

__all__ = ["ArrayView"]


image_plot_help_str = r"""
$\bf{Hotkeys:}$
    $\bf{h:}$ show/hide hotkey menu.
    $\bf{x/y/z:}$ set current axis as x/y/z.
    $\bf{t:}$ swap between x and y.
    $\bf{c:}$ cycle colormaps.
    $\bf{left/right:}$ change current axis.
    $\bf{up/down:}$ change slice along current axis.
    $\bf{a:}$ toggle hide all labels, titles and axes.
    $\bf{m/p/r/i/l:}$  magnitude/phase/real/imaginary/log mode.
    $\bf{[/]:}$ change brightness.
    $\bf{\{/\}:}$ change contrast.
    $\bf{n:}$ enter vmin/vmax in separate boxes (tab to switch).
    $\bf{d:}$ cycle dynamic range (5%-95%, 1%-99%, 10%-90%, full).
    $\bf{space:}$ auto-play through current slice dimension.
    $\bf{s:}$ save as png.
    $\bf{g/v:}$ save as gif/video by along current axis.
    $\bf{q:}$ refresh.
    $\bf{0-9:}$ enter slice number.
    $\bf{enter:}$ set current axis as slice number.
"""


class ArrayView(object):
    """Plot array as image.

    Press 'h' for a menu for hotkeys.
    """

    def __init__(
        self,
        im,
        x=-1,
        y=-2,
        z=None,
        c=None,
        hide_axes=False,
        mode=None,
        colormap=None,
        vmin=None,
        vmax=None,
        title="",
        interpolation="nearest",
        save_basename="Figure",
        fps=10,
        initial_slices=None,
    ):
        if im.ndim < 2:
            raise TypeError(f"Image dimension must at least be two, got {im.ndim}")

        self.im = im
        self.shape = self.im.shape
        self.ndim = self.im.ndim
        if initial_slices is not None:
            self.slices = list(initial_slices)
        else:
            self.slices = [s // 2 for s in self.shape]
        self.flips = [1] * self.ndim
        self.x = x % self.ndim
        self.y = y % self.ndim
        self.z = z % self.ndim if z is not None else None
        self.c = c % self.ndim if c is not None else None
        self.d = max(self.ndim - 3, 0)
        self.hide_axes = hide_axes
        self.show_help = False
        self.title = title
        self.interpolation = interpolation
        self.mode = mode
        self.colormaps = ["gray", "viridis", "RdBu_r", "magma"]
        self.colormap_idx = 0
        if colormap is not None:
            try:
                self.colormap_idx = self.colormaps.index(colormap)
            except ValueError:
                pass
        self.colormap = self.colormaps[self.colormap_idx]
        self.entering_slice = False
        self.entering_vminmax = False
        self.vminmax_focus = "vmin"  # "vmin" or "vmax"
        self.entered_vmin = ""
        self.entered_vmax = ""
        self.field_selected = True  # Whether current field is selected for overwrite
        self.quantile_cycle = 0  # 0=5%-95%, 1=1%-99%, 2=10%-90%, 3=full range
        self.vmin = vmin
        self.vmax = vmax
        self.save_basename = save_basename
        self.fps = fps

        # Auto-play state
        self.auto_playing = False
        self.auto_play_timer = None
        self.auto_play_interval = 200  # milliseconds between frames

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.axim = None
        self.help_text = None
        self.colorbar = None
        self.entry_text = None  # Text widget for vmin/vmax entry

        # For blitting
        self.background = None

        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

        self.update_all()
        plt.show(block=True)

    def _smart_round(self, value):
        """Smart formatting with appropriate decimal places, no scientific notation."""
        if value is None:
            return ""
        if value == 0:
            return "0"

        # Determine number of decimal places based on magnitude
        abs_val = abs(value)

        if abs_val >= 100:
            # For large values, no decimal places
            return f"{value:.0f}"
        elif abs_val >= 10:
            # For values 10-100, 1 decimal place
            return f"{value:.1f}"
        elif abs_val >= 1:
            # For values 1-10, 2 decimal places
            return f"{value:.2f}"
        elif abs_val >= 0.01:
            # For values 0.01-1, 3 decimal places
            return f"{value:.3f}"
        else:
            # For very small values, 4 decimal places
            return f"{value:.4f}"

    def _cycle_quantile_range(self):
        """Cycle through different quantile-based dynamic range settings."""
        # Get current image data for quantile calculation
        imv = self._get_current_image_data()

        # Define quantile pairs: (lower_percentile, upper_percentile, description)
        quantile_settings = [
            (5, 95, "5%-95%"),
            (1, 99, "1%-99%"),
            (10, 90, "10%-90%"),
            (0, 100, "full range"),
        ]

        # Cycle to next setting
        self.quantile_cycle = (self.quantile_cycle + 1) % len(quantile_settings)
        lower_pct, upper_pct, desc = quantile_settings[self.quantile_cycle]

        if lower_pct == 0 and upper_pct == 100:
            # Full range
            self.vmin = imv.min()
            self.vmax = imv.max()
        else:
            # Calculate quantiles
            self.vmin = np.percentile(imv, lower_pct)
            self.vmax = np.percentile(imv, upper_pct)

        # Brief visual feedback (could be enhanced with a temporary text display)
        print(
            f"Dynamic range: {desc} quantiles (vmin={self.vmin:.3f}, vmax={self.vmax:.3f})"
        )

    def _get_current_image_data(self):
        """Get the current visible image data for quantile calculations."""
        idx = tuple(
            slice(None, None, self.flips[i])
            if i in [self.x, self.y, self.z, self.c]
            else self.slices[i]
            for i in range(self.ndim)
        )
        imv = self.im[idx]

        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims.insert(0, self.z)
        if self.c is not None:
            imv_dims.append(self.c)
        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
        imv = array_to_image(imv, color=self.c is not None)

        # Apply the same transformations as in update_image
        if self.mode is None:
            mode = "r" if np.isrealobj(imv) else "m"
        else:
            mode = self.mode

        if mode == "m":
            imv = np.abs(imv)
        elif mode == "p":
            imv = np.angle(imv)
        elif mode == "r":
            imv = np.real(imv)
        elif mode == "i":
            imv = np.imag(imv)
        elif mode == "l":
            imv = np.log(
                np.abs(imv),
                out=np.full_like(imv, -31, dtype=float),
                where=np.abs(imv) != 0,
            )

        return imv

    def _toggle_auto_play(self):
        """Toggle auto-play through the current slice dimension."""
        if self.auto_playing:
            # Stop auto-play
            self._stop_auto_play()
            print("Auto-play stopped")
        else:
            # Start auto-play (only if current dimension can be sliced)
            if self.d not in [self.x, self.y, self.z, self.c]:
                self._start_auto_play()
                print(
                    f"Auto-play started on dimension {self.d} (shape: {self.shape[self.d]})"
                )
            else:
                print(f"Cannot auto-play on display axis {self.d}")

    def _start_auto_play(self):
        """Start the auto-play timer."""
        self.auto_playing = True
        self._schedule_next_frame()

    def _stop_auto_play(self):
        """Stop the auto-play timer."""
        self.auto_playing = False
        if self.auto_play_timer is not None:
            self.auto_play_timer.stop()
            self.auto_play_timer = None

    def _schedule_next_frame(self):
        """Schedule the next frame update."""
        if self.auto_playing:
            # Use matplotlib's timer for smooth animation
            self.auto_play_timer = self.fig.canvas.new_timer(
                interval=self.auto_play_interval
            )
            self.auto_play_timer.add_callback(self._auto_play_step)
            self.auto_play_timer.start()

    def _auto_play_step(self):
        """Step to the next slice in auto-play mode."""
        if not self.auto_playing:
            return

        # Advance to next slice (wrap around)
        if self.d not in [self.x, self.y, self.z, self.c]:
            self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[self.d]

            # Use fast update for smooth animation
            self._update_slice_fast(
                1
            )  # direction doesn't matter since we set slice directly

            # Schedule next frame
            self._schedule_next_frame()

    def _on_close(self, event):
        """Clean up resources when figure is closed."""
        self._stop_auto_play()

    def _on_close(self, event):
        """Clean up resources when figure is closed."""
        self._stop_auto_play()

    def _on_resize(self, event):
        """Invalidate background on resize."""
        self.background = None

    def _on_draw(self, event):
        """Cache the background after a full draw."""
        if event.canvas.supports_blit:
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def update_all(self):
        """Update everything and perform a full, slow redraw. This is the 'correct' path."""
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw_idle()

    def _update_slice_fast(self, direction):
        """Update only the image slice and blit for maximum speed. Title will be stale."""
        if self.d not in [self.x, self.y, self.z, self.c]:
            self.slices[self.d] = (self.slices[self.d] + direction) % self.shape[self.d]
        else:
            self.flips[self.d] *= -1  # direction doesn't matter for flip

        # Update image data without triggering a full redraw
        self.update_image()

        # Blit only the image artist
        if self.background is not None and self.axim is not None:
            self.fig.canvas.restore_region(self.background)
            self.ax.draw_artist(self.axim)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()

    def key_press(self, event):
        # Use update_all for up/down to ensure title updates correctly
        if event.key == "up":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] + 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1
            self.update_all()
            return
        if event.key == "down":
            if self.d not in [self.x, self.y, self.z, self.c]:
                self.slices[self.d] = (self.slices[self.d] - 1) % self.shape[self.d]
            else:
                self.flips[self.d] *= -1
            self.update_all()
            return

        # Slow, correct path for all other keys
        needs_redraw = True
        if event.key == "left":
            self.d = (self.d - 1) % self.ndim
        elif event.key == "right":
            self.d = (self.d + 1) % self.ndim
        elif event.key == "c":
            self.colormap_idx = (self.colormap_idx + 1) % len(self.colormaps)
            self.colormap = self.colormaps[self.colormap_idx]
        elif event.key == "x" and self.d not in [self.x, self.z, self.c]:
            self.x, self.y = (self.y, self.x) if self.d == self.y else (self.d, self.y)
        elif event.key == "y" and self.d not in [self.y, self.z, self.c]:
            self.x, self.y = (self.y, self.x) if self.d == self.x else (self.x, self.d)
        elif event.key == "z" and self.d not in [self.x, self.y, self.c]:
            self.z = None if self.d == self.z else self.d
        elif event.key == "t":
            self.x, self.y = self.y, self.x
        elif event.key == "a":
            self.hide_axes = not self.hide_axes
        elif event.key in ["m", "p", "r", "i", "l"]:
            self.vmin, self.vmax, self.mode = None, None, event.key
        elif event.key == "h":
            self.show_help = not self.show_help
        elif event.key == "q":
            self.vmin, self.vmax = None, None
        elif event.key == "]":
            if self.vmin is not None:
                width = self.vmax - self.vmin
                self.vmin, self.vmax = self.vmin - width * 0.1, self.vmax - width * 0.1
        elif event.key == "[":
            if self.vmin is not None:
                width = self.vmax - self.vmin
                self.vmin, self.vmax = self.vmin + width * 0.1, self.vmax + width * 0.1
        elif event.key == "}":
            if self.vmin is not None:
                width = self.vmax - self.vmin
                center = (self.vmax + self.vmin) / 2
                self.vmin, self.vmax = (
                    center - width * 1.1 / 2,
                    center + width * 1.1 / 2,
                )
        elif event.key == "{":
            if self.vmin is not None:
                width = self.vmax - self.vmin
                center = (self.vmax + self.vmin) / 2
                self.vmin, self.vmax = (
                    center - width * 0.9 / 2,
                    center + width * 0.9 / 2,
                )
        elif event.key == "f":
            self.fig.canvas.manager.full_screen_toggle()
            needs_redraw = False
        elif event.key == "s":
            self.save_movie(is_gif=False, is_image=True)
            needs_redraw = False
        elif event.key == "d":
            # Cycle through quantile-based dynamic ranges
            self._cycle_quantile_range()
        elif event.key == " ":
            # Toggle auto-play through current slice dimension
            self._toggle_auto_play()
            needs_redraw = False
        elif event.key in ["g", "v"]:
            self.save_movie(is_gif=(event.key == "g"))
            needs_redraw = False
        elif event.key == "n":
            # Start entering vmin,vmax with separate boxes
            self.entering_vminmax = True
            self.entering_slice = False
            self.vminmax_focus = "vmin"
            # Pre-fill with current values using smart rounding
            self.entered_vmin = self._smart_round(self.vmin)
            self.entered_vmax = self._smart_round(self.vmax)
            self.field_selected = True  # Start with vmin selected
        elif event.key == "tab":
            if self.entering_vminmax:
                # Switch focus between vmin and vmax
                self.vminmax_focus = "vmax" if self.vminmax_focus == "vmin" else "vmin"
                self.field_selected = True  # Auto-select new field
        elif event.key in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
            "-",
            ",",
            "backspace",
        ]:
            if self.entering_vminmax:
                # Handle numeric entry for focused field
                current_field = (
                    self.entered_vmin
                    if self.vminmax_focus == "vmin"
                    else self.entered_vmax
                )

                if event.key == "backspace":
                    if self.field_selected:
                        current_field = ""  # Clear entire field if selected
                        self.field_selected = False
                    else:
                        current_field = current_field[:-1]  # Normal backspace
                elif event.key in [
                    "0",
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    ".",
                    "-",
                ]:
                    # Handle numeric input with auto-selection logic
                    if self.field_selected:
                        # Replace entire field content
                        current_field = event.key
                        self.field_selected = False
                    else:
                        # Append to existing content with validation
                        # Only allow one decimal point and one minus sign at the start
                        if event.key == "." and "." in current_field:
                            pass  # Skip duplicate decimal point
                        elif event.key == "-" and (
                            current_field != "" or "-" in current_field
                        ):
                            pass  # Skip minus sign if not at start or already present
                        else:
                            current_field += event.key

                # Update the appropriate field
                if self.vminmax_focus == "vmin":
                    self.entered_vmin = current_field
                else:
                    self.entered_vmax = current_field
            elif self.d not in [self.x, self.y, self.z, self.c]:
                # Handle slice entry (existing functionality)
                if self.entering_slice:
                    if event.key == "backspace":
                        self.entered_slice //= 10
                    elif event.key in [
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                    ]:
                        self.entered_slice = self.entered_slice * 10 + int(event.key)
                elif event.key in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    self.entering_slice = True
                    self.entered_slice = int(event.key)
        elif event.key == "enter":
            if self.entering_vminmax:
                # Apply vmin,vmax values from separate fields
                try:
                    if self.entered_vmin.strip():
                        self.vmin = float(self.entered_vmin.strip())
                    if self.entered_vmax.strip():
                        self.vmax = float(self.entered_vmax.strip())
                except ValueError:
                    pass  # Invalid number, ignore
                self.entering_vminmax = False
                self.entered_vmin = ""
                self.entered_vmax = ""
            elif self.entering_slice:
                # Apply slice value (existing functionality)
                self.entering_slice = False
                if self.entered_slice < self.shape[self.d]:
                    self.slices[self.d] = self.entered_slice
        else:
            needs_redraw = False

        if needs_redraw:
            self.update_all()

    def update_image(self):
        idx = tuple(
            slice(None, None, self.flips[i])
            if i in [self.x, self.y, self.z, self.c]
            else self.slices[i]
            for i in range(self.ndim)
        )
        imv = self.im[idx]

        imv_dims = [self.y, self.x]
        if self.z is not None:
            imv_dims.insert(0, self.z)
        if self.c is not None:
            imv_dims.append(self.c)
        imv = np.transpose(imv, np.argsort(np.argsort(imv_dims)))
        imv = array_to_image(imv, color=self.c is not None)

        if self.mode is None:
            self.mode = "r" if np.isrealobj(imv) else "m"

        if self.mode == "m":
            imv = np.abs(imv)
        elif self.mode == "p":
            imv = np.angle(imv)
        elif self.mode == "r":
            imv = np.real(imv)
        elif self.mode == "i":
            imv = np.imag(imv)
        elif self.mode == "l":
            imv = np.log(
                np.abs(imv),
                out=np.full_like(imv, -31, dtype=float),
                where=np.abs(imv) != 0,
            )

        if self.vmin is None:
            self.vmin = imv.min()
        if self.vmax is None:
            self.vmax = imv.max()

        if self.axim is None:
            self.axim = self.ax.imshow(
                imv,
                vmin=self.vmin,
                vmax=self.vmax,
                cmap=self.colormap,
                origin="lower",
                interpolation=self.interpolation,
                animated=True,  # Keep animated for blitting
            )
            self.colorbar = self.fig.colorbar(self.axim, ax=self.ax)
        else:
            self.axim.set_data(imv)
            self.axim.set_clim(self.vmin, self.vmax)
            self.axim.set_cmap(self.colormap)
            self.axim.set_extent([0, imv.shape[1], 0, imv.shape[0]])

        if self.help_text is None:
            bbox_props = dict(boxstyle="round", pad=1, fc="white", alpha=0.95, lw=0)
            self.help_text = self.ax.text(
                imv.shape[0] / 2,
                imv.shape[1] / 2,
                image_plot_help_str,
                ha="center",
                va="center",
                linespacing=1.5,
                ma="left",
                size=8,
                bbox=bbox_props,
            )
        self.help_text.set_visible(self.show_help)

        # Add or update entry text for vmin/vmax
        if self.entry_text is None:
            # Position the text below the image
            self.entry_text = self.fig.text(
                0.5,
                0.02,
                "",
                ha="center",
                va="bottom",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
            )

        # Update entry text content for separate vmin/vmax boxes
        entry_msg = ""
        if self.entering_vminmax:
            vmin_display = self.entered_vmin + (
                "_" if self.vminmax_focus == "vmin" else ""
            )
            vmax_display = self.entered_vmax + (
                "_" if self.vminmax_focus == "vmax" else ""
            )

            # Create vertical layout with brackets around focused field
            # Show selection indicator if field is selected
            if self.vminmax_focus == "vmin":
                if self.field_selected:
                    vmin_content = f"«{self.entered_vmin}»"
                else:
                    vmin_content = vmin_display
                entry_msg = (
                    f"[vmin: {vmin_content}]\n vmax: {vmax_display}\n(Tab ↕, Enter ✓)"
                )
            else:
                if self.field_selected:
                    vmax_content = f"«{self.entered_vmax}»"
                else:
                    vmax_content = vmax_display
                entry_msg = (
                    f" vmin: {vmin_display}\n[vmax: {vmax_content}]\n(Tab ↕, Enter ✓)"
                )

        self.entry_text.set_text(entry_msg)
        self.entry_text.set_visible(bool(entry_msg))

    def update_axes(self):
        if not self.hide_axes:
            caption_parts = []
            for i in range(self.ndim):
                part = ""
                if self.flips[i] == -1 and i in [self.x, self.y, self.z, self.c]:
                    part += "-"
                if i == self.x:
                    part += "x"
                elif i == self.y:
                    part += "y"
                elif i == self.z:
                    part += "z"
                elif i == self.c:
                    part += "c"
                elif i == self.d and self.entering_slice:
                    part += str(self.entered_slice) + "_"
                else:
                    part += str(self.slices[i])

                if i == self.d:
                    part = f"[{part}]"
                caption_parts.append(part)

            caption = " ".join(caption_parts)
            self.ax.set_title(caption)
            self.fig.suptitle(self.title)
            self.ax.xaxis.set_visible(True)
            self.ax.yaxis.set_visible(True)
            self.ax.title.set_visible(True)
            if self.colorbar:
                self.colorbar.ax.set_visible(True)
        else:
            self.ax.set_title("")
            self.fig.suptitle("")
            self.ax.xaxis.set_visible(False)
            self.ax.yaxis.set_visible(False)
            self.ax.title.set_visible(False)
            if self.colorbar:
                self.colorbar.ax.set_visible(False)

    def save_movie(self, is_gif=False, is_image=False):
        """Save image, GIF, or video along the current axis."""
        from matplotlib.animation import FuncAnimation, PillowWriter

        if is_image:
            # Save single image
            filename = f"{self.save_basename}.png"
            self.fig.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"Saved image: {filename}")
            return

        # Save animation (GIF or video)
        if self.d in [self.x, self.y, self.z, self.c]:
            print("Cannot create animation along display axis")
            return

        # Get the range of slices for the current dimension
        num_frames = self.shape[self.d]
        original_slice = self.slices[self.d]

        def animate(frame):
            self.slices[self.d] = frame
            self.update_image()
            return [self.axim]

        # Create animation
        anim = FuncAnimation(
            self.fig,
            animate,
            frames=num_frames,
            interval=1000 // self.fps,
            blit=True,
            repeat=True,
        )

        try:
            if is_gif:
                filename = f"{self.save_basename}_axis_{self.d}.gif"
                print(f"Saving GIF: {filename} ...")
                anim.save(filename, writer=PillowWriter(fps=self.fps))
                print(f"Saved GIF: {filename}")
            else:
                # Try to save as MP4
                filename = f"{self.save_basename}_axis_{self.d}.mp4"
                print(f"Saving video: {filename} ...")
                try:
                    anim.save(filename, writer="ffmpeg", fps=self.fps)
                    print(f"Saved video: {filename}")
                except Exception as e:
                    # Fallback to GIF if ffmpeg not available
                    print(f"Video save failed ({e}), falling back to GIF...")
                    filename = f"{self.save_basename}_axis_{self.d}.gif"
                    anim.save(filename, writer=PillowWriter(fps=self.fps))
                    print(f"Saved GIF: {filename}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        finally:
            # Restore original slice position
            self.slices[self.d] = original_slice
            self.update_all()


# Original, correct functions
def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while np.prod(mshape) < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)


def array_to_image(arr, color=False):
    if color and not (arr.max() == 0 and arr.min() == 0):
        arr = arr / np.abs(arr).max()
    if arr.ndim == 2 or (color and arr.ndim == 3):
        return arr
    if color:
        img_shape = arr.shape[-3:]
        batch = np.prod(arr.shape[:-3])
        mshape = mosaic_shape(batch)
    else:
        img_shape = arr.shape[-2:]
        batch = np.prod(arr.shape[:-2])
        mshape = mosaic_shape(batch)
    if np.prod(mshape) == batch:
        img = arr.reshape((batch,) + img_shape)
    else:
        img = np.zeros((np.prod(mshape),) + img_shape, dtype=arr.dtype)
        img[:batch, ...] = arr.reshape((batch,) + img_shape)
    img = img.reshape(mshape + img_shape)
    if color:
        img = np.transpose(img, (0, 2, 1, 3, 4))
        img = img.reshape((img_shape[0] * mshape[0], img_shape[1] * mshape[1], 3))
    else:
        img = np.transpose(img, (0, 2, 1, 3))
        img = img.reshape((img_shape[0] * mshape[0], img_shape[1] * mshape[1]))
    return img
