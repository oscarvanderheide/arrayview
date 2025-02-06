import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, TextBox
import matplotlib.animation as animation


class ArrayShower:
    def __init__(self, array):
        self.array = array
        self.ndim = array.ndim
        self.view_dims = [0, 1]
        self.scroll_dim = 2
        self.slice_index = array.shape[self.scroll_dim] // 2
        self.display_mode = "real" if np.isrealobj(array) else "abs"
        self.fig, self.ax = plt.subplots(1, 1)
        self.im = self.ax.imshow(self.get_current_slice(), cmap="gray")
        self.update_title()
        self.ani = None
        self.setup_buttons()
        # plt.show()  # Automatically display the plot

    def get_current_slice(self):
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
        self.ax.set_title(f"Slice index: {self.slice_index}")

    def update(self):
        # Ensure slice_index is within valid range
        self.slice_index = np.clip(
            self.slice_index, 0, self.array.shape[self.scroll_dim] - 1
        )
        # Update the image data
        current_slice = self.get_current_slice()
        if current_slice.ndim != 2:
            print(f"Error: Expected 2D slice, got {current_slice.ndim}D slice")
        self.im.set_data(current_slice)
        self.update_title()
        self.update_text_boxes()
        self.im.axes.figure.canvas.draw_idle()  # Use draw_idle to update the canvas

    def update_text_boxes(self):
        for i in range(self.array.ndim):
            if i in self.view_dims:
                self.text_boxes[i].set_val(":")
            else:
                self.text_boxes[i].set_val(
                    str(self.slice_index if i == self.scroll_dim else 0)
                )

    def change_dim(self, dim_type, new_dim):
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
        if event.button == "up":
            self.slice_index += 1
        elif event.button == "down":
            self.slice_index -= 1
        self.slice_index = np.clip(
            self.slice_index, 0, self.array.shape[self.scroll_dim] - 1
        )
        self.update()

    def onkeypress(self, event):
        if event.key == "j":
            self.slice_index += 1
        elif event.key == "k":
            self.slice_index -= 2
        # elif event.key == "h":
        #     self.change_dim("scroll_dim", (self.scroll_dim - 1) % self.ndim)
        # elif event.key == "l":
        #     self.change_dim("scroll_dim", (self.scroll_dim + 1) % self.ndim)

        # Ensure slice_index is within valid range
        # self.slice_index = np.clip(
        #     self.slice_index, 0, self.array.shape[self.scroll_dim] - 1
        # )
        self.update()

    def start_auto_scroll(self, event):
        if self.ani is None:
            self.slice_index = 0
            self.ani = animation.FuncAnimation(
                self.im.figure, self.auto_scroll, interval=1000 / 30, save_count=50
            )

    def auto_scroll(self, frame):
        self.slice_index += 1
        if self.slice_index >= self.array.shape[self.scroll_dim]:
            self.ani.event_source.stop()
            self.ani = None
            return
        self.update()

    def change_display_mode(self, label):
        self.display_mode = label
        self.update()

    def setup_buttons(self):
        self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
        self.fig.canvas.mpl_connect("key_press_event", self.onkeypress)

        self.button_down = []
        self.button_up = []
        self.text_boxes = []

        dim_labels = [str(i) for i in range(self.array.ndim)]
        for i in range(self.array.ndim):
            ax_up = plt.axes([0.1 + i * 0.2, 0.15, 0.1, 0.05])
            self.button_up.append(Button(ax_up, f"Up {i}"))
            self.button_up[i].on_clicked(
                lambda event, dim=i: self.change_dim_index(dim, 1)
            )

            ax_text = plt.axes([0.1 + i * 0.2, 0.1, 0.1, 0.05])
            self.text_boxes.append(TextBox(ax_text, f"Dim {i}"))
            self.text_boxes[i].set_val(
                ":"
                if i in self.view_dims
                else str(self.slice_index if i == self.scroll_dim else 0)
            )
            self.text_boxes[i].on_submit(
                lambda text, dim=i: self.update_dim_index(dim, text)
            )

            ax_down = plt.axes([0.1 + i * 0.2, 0.05, 0.1, 0.05])
            self.button_down.append(Button(ax_down, f"Down {i}"))
            self.button_down[i].on_clicked(
                lambda event, dim=i: self.change_dim_index(dim, -1)
            )
            # btn_down.on_clicked(lambda event, dim=i: self.change_dim_index(dim, -1))

        scroll_button_ax = plt.axes([0.85, 0.01, 0.1, 0.08])
        scroll_button = Button(scroll_button_ax, "Auto Scroll")
        scroll_button.on_clicked(self.start_auto_scroll)

    def change_dim_index(self, dim, delta):
        print(dim, delta)
        if dim in self.view_dims:
            return
        self.slice_index = np.clip(
            self.slice_index + delta, 0, self.array.shape[dim] - 1
        )
        self.update()

    def update_dim_index(self, dim, text):
        if text == ":":
            if dim not in self.view_dims:
                self.view_dims.append(dim)
                if len(self.view_dims) > 2:
                    self.view_dims.pop(0)
        else:
            try:
                index = int(text)
                if dim == self.scroll_dim:
                    self.slice_index = np.clip(index, 0, self.array.shape[dim] - 1)
                else:
                    self.change_dim("scroll_dim", dim)
                    self.slice_index = np.clip(index, 0, self.array.shape[dim] - 1)
            except ValueError:
                pass
        self.update()

    # def setup_buttons(self):
    #     self.fig.canvas.mpl_connect("scroll_event", self.onscroll)
    #     self.fig.canvas.mpl_connect("key_press_event", self.onkeypress)
    #
    #     dim_labels = [str(i) for i in range(self.array.ndim)]
    #     slice1_ax = plt.axes([0.1, 0.01, 0.2, 0.08], frameon=False)
    #     slice1_radio = RadioButtons(slice1_ax, dim_labels)
    #     slice1_radio.on_clicked(lambda label: self.change_dim("slice_dim1", label))
    #
    #     slice2_ax = plt.axes([0.35, 0.01, 0.2, 0.08], frameon=False)
    #     slice2_radio = RadioButtons(slice2_ax, dim_labels)
    #     slice2_radio.on_clicked(lambda label: self.change_dim("slice_dim2", label))
    #
    #     scroll_ax = plt.axes([0.6, 0.01, 0.2, 0.08], frameon=False)
    #     scroll_radio = RadioButtons(scroll_ax, dim_labels)
    #     scroll_radio.on_clicked(lambda label: self.change_dim("scroll_dim", label))
    #
    #     scroll_button_ax = plt.axes([0.85, 0.01, 0.1, 0.08])
    #     scroll_button = Button(scroll_button_ax, "Auto Scroll")
    #     scroll_button.on_clicked(self.start_auto_scroll)
    #
    #     display_mode_ax = plt.axes([0.85, 0.15, 0.1, 0.08])
    #     display_mode_radio = RadioButtons(
    #         display_mode_ax, ["real", "imag", "abs", "angle"]
    #     )
    #     display_mode_radio.on_clicked(self.change_display_mode)


if __name__ == "__main__":
    array = np.random.rand(224, 224, 192, 32)
    viewer = ArrayShower(array)
    plt.show()
