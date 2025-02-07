## Setup
import matplotlib

# matplotlib.use("Qt5Agg")  # Or "TkAgg"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import matplotlib.animation as animation

## Main class


class ArrayShow:
    """
    The ArrayShow class is designed to visualize 3D or higher-dimensional numpy arrays using matplotlib.

    It has "viewing" dimensions and a "slice" dimension:
        - Viewing dimensions are the two dimensions that are displayed as a 2D slice.
        - The slice dimension is the dimension through which the array is scrolled.

    Initialization:
        - The class is initialized with a numpy array.
        - It checks that the array has at least 3 dimensions and sets up initial viewing parameters,
        including the dimensions to view and the dimension to scroll through.

    View Setup:
        - Initializes the current view slice and sets the display mode (real, imaginary, absolute, or angle).
        - Creates a matplotlib figure and axis to display the array slice.

    Slice Management:
        - Methods like `initialize_current_view`, `get_current_slice`, and `update_view` manage the current slice of the array being displayed.
        - The slice index can be changed via scrolling or key presses.

    User Interaction:
        - Sets up buttons and text boxes for user interaction.
        - Users can change the slice index, view dimensions, and display mode using these controls.
        - Supports auto-scrolling through slices.

    Event Handling:
        - Connects scroll and key press events to their respective handlers to update the view dynamically.

    Main Execution:
        - When run as a script, it creates an instance of `ArrayShower` with a random array and displays the matplotlib window.

    This class provides an interactive way to explore multi-dimensional arrays visually.
    """

    def __init__(self, array, view_dims=[0,1], scroll_dim=2):
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
        self.fig, self.ax = plt.subplots(1, 1)
        self.cmap = "gray"
        self.im = self.ax.imshow(self.get_current_slice(), cmap=self.cmap)

        # self.ani = None
        # Create UI components for user interaction
        self.setup_buttons()

    def calculate_fixed_dims(self):
        """Given the view dimensions and the scroll dimension, calculate the remaining, fixed dimensions"""

        # Turn stuff into sets
        all_dims_set = set(range(array.ndim))
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
        """Given the slice indices, extract the current slice from the array"""
        current_slice = self.array[tuple(self.slice_indices)]
        if self.display_mode == "real":
            return np.real(current_slice)
        elif self.display_mode == "imag":
            return np.imag(current_slice)
        elif self.display_mode == "abs":
            return np.abs(current_slice)
        elif self.display_mode == "angle":
            return np.angle(current_slice)

    def update_view(self):
        """Update the image data and redraw the canvas"""
        current_slice = self.get_current_slice()
        if current_slice.ndim != 2:
            print(f"Error: Expected 2D slice, got {current_slice.ndim}D slice")
        print(
            f"Shape of current slice: {current_slice.shape}, scroll_dim: {self.scroll_dim}, scroll_index: {self.scroll_index}"
        )
        self.im.set_data(current_slice)
        # self.update_text_boxes()
        self.im.axes.figure.canvas.draw_idle()

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

        self.scroll_dim = new_scroll_dim
        # Update the fixed_dims
        self.fixed_dims = self.calculate_fixed_dims()
        # Set scroll index to the current index of the new scroll dimension
        self.scroll_index = self.slice_indices[self.scroll_dim]

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

    def change_view_dims(self):
        # Create a figure and hide it
        fig, ax = plt.subplots()
        # fig.canvas.set_window_title("Enter View Dimensions")
        fig.patch.set_visible(False)
        ax.axis("off")

        # Prompt user for input
        dims = plt.ginput(n=1, timeout=-1, show_clicks=False)
        plt.close(fig)

        if dims:
            try:
                dims_str = input(
                    "Enter view dimensions as comma separated list (e.g., 800,600): "
                )
                width, height = map(int, dims_str.split(","))
                self.view_dims = (width, height)
                print(f"View dimensions changed to: {self.view_dims}")
            except ValueError:
                print(
                    "Invalid input. Please enter dimensions as comma separated integers."
                )

    def change_display_mode(self, label):
        self.display_mode = label
        self.update_view()

    def setup_buttons(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.fig.canvas.mpl_connect("key_press_event", self.onkeypress)

        self.button_down = []
        self.button_up = []
        self.text_boxes = []

        for i in range(self.array.ndim):
            ax_up = plt.axes([0.1 + i * 0.2, 0.15, 0.03, 0.05])
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

            ax_text = plt.axes([0.1 + i * 0.2, 0.1, 0.03, 0.05])
            self.text_boxes.append(TextBox(ax_text, f"Dim {i}"))
            self.text_boxes[i].set_val(
                ":" if i in self.view_dims else str(self.slice_indices[i])
            )
            self.text_boxes[i].on_submit(
                lambda text, dim=i: (
                    self.change_scroll_dim(dim),
                    self.update_scroll_index(int(text)-self.scroll_index),
                    self.update_view(),
                )
            )

            ax_down = plt.axes([0.1 + i * 0.2, 0.05, 0.03, 0.05])
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

        # scroll_button_ax = plt.axes([0.85, 0.01, 0.1, 0.08])
        # scroll_button = Button(scroll_button_ax, "Auto Scroll")
        # scroll_button.on_clicked(self.start_auto_scroll)


if __name__ == "__main__":
    array = np.random.rand(224, 224, 192, 32)
    viewer = ArrayShow(array)
    plt.show()
