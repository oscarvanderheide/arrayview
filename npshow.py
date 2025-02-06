import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
import matplotlib.animation as animation


class ArrayShower:
    def __init__(self, ax, array):
        """
        Initialize the NDTracker with the given axis and array.

        Parameters:
        ax (matplotlib.axes.Axes): The axis to plot on.
        array (numpy.ndarray): The n-dimensional array to visualize.
        """
        self.ax = ax
        self.array = array
        self.ndim = array.ndim
        self.view_dims = [0, 1]
        self.scroll_dim = 2
        self.slice_index = array.shape[self.scroll_dim] // 2
        self.display_mode = "real" if np.isrealobj(array) else "abs"
        self.im = ax.imshow(self.get_current_slice(), cmap="gray")
        self.update_title()
        self.ani = None

    def get_current_slice(self):
        """
        Get the current slice of the array based on the view and scroll dimensions.

        Returns:
        numpy.ndarray: The current slice of the array.
        """
        slice_indices = [
            self.slice_index if i == self.scroll_dim else 0 for i in range(self.ndim)
        ]
        slice_indices[self.view_dims[0]] = slice(None)
        slice_indices[self.view_dims[1]] = slice(None)
        slice_data = self.array[tuple(slice_indices)]
        if self.display_mode == "real":
            return np.real(slice_data)
        elif self.display_mode == "imag":
            return np.imag(slice_data)
        elif self.display_mode == "abs":
            return np.abs(slice_data)
        elif self.display_mode == "angle":
            return np.angle(slice_data)

    def update_title(self):
        """
        Update the title of the plot with the current slice index.
        """
        self.ax.set_title(f"Slice index: {self.slice_index}")

    def update(self):
        """
        Update the image data and redraw the canvas.
        """
        self.slice_index = np.clip(
            self.slice_index, 0, self.array.shape[self.scroll_dim] - 1
        )
        self.im.set_data(self.get_current_slice())
        self.update_title()
        self.im.axes.figure.canvas.draw()

    def change_dim(self, dim_type, new_dim):
        """
        Change the view or scroll dimension.

        Parameters:
        dim_type (str): The type of dimension to change ('slice_dim1', 'slice_dim2', 'scroll_dim').
        new_dim (int): The new dimension index.
        """
        new_dim = int(new_dim)
        if (
            dim_type == "slice_dim1"
            and new_dim != self.view_dims[1]
            and new_dim != self.scroll_dim
        ):
            self.view_dims[0] = new_dim
        elif (
            dim_type == "slice_dim2"
            and new_dim != self.view_dims[0]
            and new_dim != self.scroll_dim
        ):
            self.view_dims[1] = new_dim
        elif (
            dim_type == "scroll_dim"
            and new_dim != self.view_dims[0]
            and new_dim != self.view_dims[1]
        ):
            self.scroll_dim = new_dim
            self.slice_index = self.array.shape[self.scroll_dim] // 2
        self.update()

    def onscroll(self, event):
        """
        Handle scroll events to update the slice index.

        Parameters:
        event (matplotlib.backend_bases.MouseEvent): The scroll event.
        """
        if event.button == "up":
            self.slice_index += 1
        elif event.button == "down":
            self.slice_index -= 1
        self.update()

    def onkeypress(self, event):
        """
        Handle key press events to update the slice index or change dimensions.

        Parameters:
        event (matplotlib.backend_bases.KeyEvent): The key press event.
        """
        if event.key == "j":
            self.slice_index += 1
        elif event.key == "k":
            self.slice_index -= 1
        elif event.key == "h":
            self.change_dim("scroll_dim", (self.scroll_dim - 1) % self.ndim)
        elif event.key == "l":
            self.change_dim("scroll_dim", (self.scroll_dim + 1) % self.ndim)
        self.update()

    def start_auto_scroll(self, event):
        """
        Start automatic scrolling through the slices.

        Parameters:
        event (matplotlib.backend_bases.Event): The event that triggers auto scroll.
        """
        if self.ani is None:
            self.slice_index = 0
            self.ani = animation.FuncAnimation(
                self.im.figure, self.auto_scroll, interval=1000 / 30, save_count=50
            )

    def auto_scroll(self, frame):
        """
        Automatically scroll through the slices.

        Parameters:
        frame (int): The current frame number.
        """
        self.slice_index += 1
        if self.slice_index >= self.array.shape[self.scroll_dim]:
            self.ani.event_source.stop()
            self.ani = None
            return
        self.update()

    def change_display_mode(self, label):
        """
        Change the display mode of the array (real, imag, abs, angle).

        Parameters:
        label (str): The new display mode.
        """
        self.display_mode = label
        self.update()


def npshow(array):
    fig, ax = plt.subplots(1, 1)
    tracker = ArrayShower(ax, array)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
    fig.canvas.mpl_connect("key_press_event", tracker.onkeypress)

    dim_labels = [str(i) for i in range(array.ndim)]
    slice1_ax = plt.axes([0.1, 0.01, 0.2, 0.08], frameon=False)
    slice1_radio = RadioButtons(slice1_ax, dim_labels)
    slice1_radio.on_clicked(lambda label: tracker.change_dim("slice_dim1", label))

    slice2_ax = plt.axes([0.35, 0.01, 0.2, 0.08], frameon=False)
    slice2_radio = RadioButtons(slice2_ax, dim_labels)
    slice2_radio.on_clicked(lambda label: tracker.change_dim("slice_dim2", label))

    scroll_ax = plt.axes([0.6, 0.01, 0.2, 0.08], frameon=False)
    scroll_radio = RadioButtons(scroll_ax, dim_labels)
    scroll_radio.on_clicked(lambda label: tracker.change_dim("scroll_dim", label))

    scroll_button_ax = plt.axes([0.85, 0.01, 0.1, 0.08])
    scroll_button = Button(scroll_button_ax, "Auto Scroll")
    scroll_button.on_clicked(tracker.start_auto_scroll)

    display_mode_ax = plt.axes([0.85, 0.15, 0.1, 0.08])
    display_mode_radio = RadioButtons(display_mode_ax, ["real", "imag", "abs", "angle"])
    display_mode_radio.on_clicked(tracker.change_display_mode)

    plt.show()


# read in a npy array:
array = np.random.rand(224, 224, 192, 32)
npshow(array)

# xx = np.load("/Users/oscar/tmp/npy_export/20250205_174704_SENSE_2_Acq_@@_CSM.npy")
# x = np.transpose(xx, (1, 2, 0))  # put coils last
# x.shape
#
# fig, ax = plt.subplots(4, 4)
#
# for i in range(x.shape[-1]):
#     # get div and rem mod 4
#     div, rem = divmod(i, 4)
#     print(div, rem)
#     ax[div, rem].imshow(np.abs(x[:, :, i]), cmap="viridis", vmin=0, vmax=1)
#     ax[div, rem].axis("off")
#     # set the color limits to the range of the data
#     ax[div, rem].set_title(f"coil {i}")
#     ax[div, rem].set_aspect("equal")
#
# plt.show()
# plt.clim(vmin=0, vmax=2)  # Adjust these values as needed
# #
