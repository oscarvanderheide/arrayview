## Setup

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
import matplotlib.animation as animation


## Main class
class ArrayShow:
    """
    The ArrayShow class is designed to visualize 3D or higher-dimensional numpy arrays using matplotlib.

    It has "viewing" dimensions and a "scroll" dimension:
        - Viewing dimensions are the two dimensions that are displayed as a 2D slice.
        - The scroll dimension is the dimension through which the array is scrolled.
        - The remaining dimensions are referred to as "fixed_dims".
    """

    def __init__(self, array, view_dims=[0, 1], scroll_dim=2, cmap="viridis"):
        self.array = array
        self.ndim = array.ndim

        assert (
            self.ndim > 2
        ), "Array must have at least 3 dimensions: just use imshow instead"

        # By default, view the first two dimensions and scroll through the third
        self.view_dims = view_dims
        self.scroll_dim = scroll_dim
        self.fixed_dims = self.calculate_fixed_dims()

        # Extract kk
        self.scroll_index = array.shape[self.scroll_dim] // 2
        self.slice_indices = self.initialize_slice_indices()

        # Show absolute value of complex arrays by default
        self.display_mode = "real" if np.isrealobj(array) else "abs"

        # Initialize the figure and axis
        # Make the figure bigger
        # plt.rcParams["figure.figsize"] = [12, 8]

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        # self.fig, self.ax = plt.subplots(1, 1)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.cmap = cmap
        self.im = self.ax.imshow(
            self.get_current_slice(), cmap=self.cmap, vmin=0, vmax=2
        )

        # self.ani = None
        # Create UI components for user interaction
        self.setup_buttons()

        # Add this line at the end of __init__
        plt.show()

    def calculate_fixed_dims(self):
        """Given the view dimensions and the scroll dimension, calculate the remaining, fixed dimensions"""

        # Turn stuff into sets
        all_dims_set = set(range(self.ndim))
        scroll_dim_set = set([self.scroll_dim])
        view_dims_set = set(self.view_dims)
        # Subtract sets
        fixed_dims = list(all_dims_set - scroll_dim_set - view_dims_set)

        return fixed_dims

    def initialize_slice_indices(self):
        """Calculate the indices for the slice to be displayed. Note that viewing dimensions are set to `:`"""
        # First, create a list of all zeros
        slice_indices = [0] * self.ndim

        # Set the scroll dimension index to the current slice index
        slice_indices[self.scroll_dim] = self.scroll_index

        # Set the viewing dimensions indices to `:`
        slice_indices[self.view_dims[0]] = slice(None)
        slice_indices[self.view_dims[1]] = slice(None)
        return slice_indices

    def get_current_slice(self):
        """Extract and arrange the current slice for display.

        For 3D arrays, slices are arranged in a grid that favors a wider screen.
        """
        current_slice = self.array[tuple(self.slice_indices)]
        if current_slice.ndim == 3:
            current_slice = self._arrange_slices_grid(current_slice)
        elif current_slice.ndim > 3:
            raise NotImplementedError(
                "Display for arrays with more than 3 dimensions is not implemented yet."
            )
        return self._apply_display_mode(current_slice)

    def _arrange_slices_grid(self, slices_3d, screen_aspect=16 / 9):
        """Arrange the slices in a grid with more columns than rows
        based on a given screen aspect ratio.

        Args:
            slices_3d (ndarray): 3D array with shape (height, width, num_slices)
            screen_aspect (float): Desired screen aspect ratio (width/height)

        Returns:
            ndarray: A 2D array with the slices arranged in a grid.
        """
        num_slices = slices_3d.shape[2]
        grid_cols = int(math.ceil(math.sqrt(num_slices * screen_aspect)))
        grid_rows = int(math.ceil(num_slices / grid_cols))
        # Pad the slices if necessary so that we have a full grid.
        padded = np.zeros(
            (slices_3d.shape[0], slices_3d.shape[1], grid_rows * grid_cols),
            dtype=slices_3d.dtype,
        )
        padded[:, :, :num_slices] = slices_3d
        reshaped = padded.reshape(
            slices_3d.shape[0], slices_3d.shape[1], grid_rows, grid_cols
        )
        grid = np.block(
            [[reshaped[:, :, i, j] for j in range(grid_cols)] for i in range(grid_rows)]
        )
        return grid

    def _apply_display_mode(self, img):
        """Apply the current display mode (real, imag, abs, or angle) to the image."""
        if self.display_mode == "real":
            return np.real(img)
        elif self.display_mode == "imag":
            return np.imag(img)
        elif self.display_mode == "abs":
            return np.abs(img)
        elif self.display_mode == "angle":
            return np.angle(img)
        else:
            raise ValueError(f"Invalid display mode: {self.display_mode}")

    def update_view(self):
        """Update the image data and redraw the canvas"""
        try:
            current_slice = self.get_current_slice()

            if current_slice.ndim != 2:
                print(f"Error: Expected 2D slice, got {current_slice.ndim}D slice")
                return

            # Always recreate the image to ensure proper display
            self.ax.clear()
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.im = self.ax.imshow(current_slice, cmap=self.cmap, vmin=0, vmax=2)

            # Update the figure title with the current slice information
            dim_info = []
            for i in range(self.array.ndim):
                if i in self.view_dims:
                    dim_info.append(f"dim{i}=view")
                elif i == self.scroll_dim:
                    dim_info.append(f"dim{i}={self.scroll_index}")
                else:
                    dim_info.append(f"dim{i}={self.slice_indices[i]}")

            self.ax.set_title(", ".join(dim_info))
            self.im.axes.figure.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating view: {e}")

    # def update_text_boxes(self):
    #     for i in range(self.array.ndim):
    #         if i in self.view_dims:
    #             self.text_boxes[i].set_val(":")
    #         else:
    #             self.text_boxes[i].set_val(
    #                 str(self.slice_indices if i == self.scroll_dim else 0)
    #             )

    def update_scroll_index(self, delta=0):
        """Update the scroll index based on mousewheel, keypress or buttonpress"""
        self.scroll_index += delta
        # Make sure slice index is valid
        self.scroll_index = np.clip(
            self.scroll_index, 0, self.array.shape[self.scroll_dim] - 1
        )
        # Update the index in slice_indices
        self.slice_indices[self.scroll_dim] = self.scroll_index
        # Update the text in the TextBox corresponding to the current scroll_dim
        self.text_boxes[self.scroll_dim].set_val(str(self.scroll_index))

        #             self.text_boxes[i].set_val(
        #         ":" if i in self.view_dims else str(self.slice_indices[i])
        #     )

    def onscroll(self, event):
        if event.button == "up":
            self.update_scroll_index(delta=1)
        elif event.button == "down":
            self.update_scroll_index(delta=-1)
        # Redraw the canvas
        self.update_view()

    def onkeypress(self, event):
        if event.key == "j":
            self.update_scroll_index(delta=1)
        elif event.key == "k":
            self.update_scroll_index(delta=-1)
        elif event.key == "l":
            new_scroll_dim = self.calculate_next_scroll_dim("next")
            self.change_scroll_dim(new_scroll_dim)
        elif event.key == "h":
            new_scroll_dim = self.calculate_next_scroll_dim("prev")
            self.change_scroll_dim(new_scroll_dim)
        elif event.key == "d":
            self.debug_slice()
        self.update_view()

    def calculate_next_scroll_dim(self, direction):
        if direction == "next":
            # Filter out the fixed dimensions that are larger than scroll_dim
            larger_dims = [dim for dim in self.fixed_dims if dim > self.scroll_dim]
            if larger_dims:
                # Use the smallest integer larger than scroll_dim
                return min(larger_dims)
            else:
                # Use the smallest integer in fixed_dims
                return min(self.fixed_dims)

        elif direction == "prev":
            # Filter out the fixed dimensions that are smaller than scroll_dim
            smaller_dims = [dim for dim in self.fixed_dims if dim < self.scroll_dim]
            if smaller_dims:
                # Use the largest integer smaller than scroll_dim
                return max(smaller_dims)
            else:
                # Use the largest integer in fixed_dims
                return max(self.fixed_dims)
        else:
            print(f"Invalid direction: {direction}")
        return None

    def change_scroll_dim(self, new_scroll_dim):
        assert (
            new_scroll_dim < self.ndim
        ), "New scroll dimension must be less than the number of dimensions in the array"

        if new_scroll_dim in self.view_dims:
            print(
                f"Error: New scroll dimension {new_scroll_dim} should not be a view dimension"
            )
            return None

        self.text_boxes[self.scroll_dim].text_disp.set_fontweight("regular")
        self.scroll_dim = new_scroll_dim
        # Update the fixed_dims
        self.fixed_dims = self.calculate_fixed_dims()
        # Set scroll index to the current index of the new scroll dimension
        self.scroll_index = self.slice_indices[self.scroll_dim]

        self.text_boxes[self.scroll_dim].text_disp.set_fontweight("bold")

        return None

    def start_auto_scroll(self, event):
        if self.ani is None:
            self.slice_indices = 0
            self.ani = animation.FuncAnimation(
                self.im.figure, self.auto_scroll, interval=1000 / 30, save_count=50
            )

    def auto_scroll(self, frame):
        self.change_slice_index(1)
        if self.slice_indices >= self.array.shape[self.scroll_dim]:
            self.ani.event_source.stop()
            self.ani = None

    # def change_display_mode(self, label):
    #     self.display_mode = label
    #     self.update_view()

    def debug_slice(self):
        """Print debugging information about the current slice"""
        current_slice = self.get_current_slice()
        print(f"Current slice shape: {current_slice.shape}")
        print(f"View dims: {self.view_dims}")
        print(f"Scroll dim: {self.scroll_dim}")
        print(f"Scroll index: {self.scroll_index}")
        print(f"Slice indices: {self.slice_indices}")

    def setup_buttons(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.fig.canvas.mpl_connect("key_press_event", self.onkeypress)

        self.button_down = []
        self.button_up = []
        self.text_boxes = []
        self.min_labels = []  # New list to store min value labels
        self.max_labels = []  # New list to store max value labels

        x_pos = 0.5
        y_pos = 0.9

        height = 0.02
        width = 0.035
        dist_between_buttons = 0.05

        for i in range(self.array.ndim):
            # Max value label (above up button)
            ax_max = plt.axes(
                [
                    x_pos + (i - self.array.ndim * 0.5) * dist_between_buttons,
                    y_pos + 3 * height,
                    width,
                    height / 2,
                ],
                frameon=False,
            )
            ax_max.axis("off")
            max_value = self.array.shape[i] - 1  # Show max index for all dimensions
            max_label = ax_max.text(
                0.5, 0.5, f"{max_value}", ha="center", va="center", fontsize=10
            )
            self.max_labels.append(max_label)

            # Up button
            ax_up = plt.axes(
                [
                    x_pos + (i - self.array.ndim * 0.5) * dist_between_buttons,
                    y_pos + 2 * height,
                    width,
                    height,
                ]
            )
            self.button_up.append(Button(ax_up, "↑"))
            self.button_up[i].on_clicked(
                lambda event, dim=i: (
                    self.change_scroll_dim(dim),
                    self.update_scroll_index(1),
                    self.update_view(),
                )
                if dim not in self.view_dims
                else None
            )

            # Text box
            ax_text = plt.axes(
                [
                    x_pos + (i - self.array.ndim * 0.5) * dist_between_buttons,
                    y_pos + 1 * height,
                    width,
                    height,
                ]
            )
            self.text_boxes.append(TextBox(ax_text, ""))
            self.text_boxes[i].set_val(
                ":" if i in self.view_dims else str(self.slice_indices[i])
            )
            if i == self.scroll_dim:
                self.text_boxes[i].text_disp.set_fontweight("bold")
            self.text_boxes[i].on_submit(
                lambda text, dim=i: (
                    self.change_scroll_dim(dim),
                    self.update_scroll_index(int(text) - self.scroll_index),
                    self.update_view(),
                )
            )

            # Down button
            ax_down = plt.axes(
                [
                    x_pos + (i - self.array.ndim * 0.5) * dist_between_buttons,
                    y_pos,
                    width,
                    height,
                ]
            )
            self.button_down.append(Button(ax_down, "↓"))
            self.button_down[i].on_clicked(
                lambda event, dim=i: (
                    self.change_scroll_dim(dim),
                    self.update_scroll_index(-1),
                    self.update_view(),
                )
                if dim not in self.view_dims
                else None
            )

            # Min value label (below down button)
            ax_min = plt.axes(
                [
                    x_pos + (i - self.array.ndim * 0.5) * dist_between_buttons,
                    y_pos - height / 2,
                    width,
                    height / 2,
                ],
                frameon=False,
            )
            ax_min.axis("off")
            min_label = ax_min.text(
                0.5, 0.5, "0", ha="center", va="center", fontsize=10
            )
            self.min_labels.append(min_label)


# if __name__ == "__main__":
#     array = np.random.rand(224, 224, 64, 2, 3)
#     array = np.load(
#         "/Users/oscar/tmp/npy_export/20250205_174627_sense_test_protoarray_Acq_@@_CSM.npy"
#     )
#     array = np.permute_dims(array, [1, 2, 0])
#     print(array.shape)
#     # array is 3d, i want to make it 4d by repeating the last dimension
#     array = np.repeat(array[..., np.newaxis], 32, axis=-1)
#     for i in range(32):
#         array[:, :, :, i] *= (32 - i) / 32
#
#     print(array.shape)
#     viewer = ArrayShow(array)
#     plt.show()
