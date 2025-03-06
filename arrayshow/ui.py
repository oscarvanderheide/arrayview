import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
from typing import Callable, List
import numpy as np

class ArrayShowUI:
    """Handle graphical UI stuff like buttons, popup windows, etc."""
    def __init__(self, fig, ax, ndim: int, array_shape, initial_view_dims, on_scroll: Callable, on_keypress: Callable, on_buttonpress: Callable):
        self.fig = fig
        self.ax = ax
        self.ndim = ndim
        self.array_shape = array_shape
        self.initial_view_dims = initial_view_dims

        # Connect events
        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', on_keypress)
        self.fig.canvas.mpl_connect('button_press_event', on_buttonpress)
        
        # Initialize UI components
        self.button_up: List[Button] = []
        self.button_down: List[Button] = []
        self.text_boxes: List[TextBox] = []
        self.view_dim_buttons: CheckButtons = None
        
        self._setup_dimension_controls()
        self._setup_view_controls()
    
    def _setup_dimension_controls(self):
        """Setup the dimension controls (up/down buttons and text boxes)"""
        # Calculate button sizes and positions
        button_width = 0.015
        button_height = 0.03
        text_width = 0.035
        spacing = 0.01
        
        for i in range(self.ndim):
            # Calculate positions
            base_x = 0.85
            base_y = 0.9 - (i * (button_height + spacing) * 3)
            
            # Create up button
            up_ax = self.fig.add_axes([base_x, base_y, button_width, button_height])
            up_button = Button(up_ax, '↑')
            up_button.dim = i  # Store dimension index
            self.button_up.append(up_button)
            
            # Create down button
            down_ax = self.fig.add_axes([base_x, base_y - button_height - spacing, button_width, button_height])
            down_button = Button(down_ax, '↓')
            down_button.dim = i  # Store dimension index
            self.button_down.append(down_button)
            
            # Create text box
            text_ax = self.fig.add_axes([base_x + 2*button_width + spacing, base_y - button_height/2, text_width, button_height])
            initial_text = ':' if i in self.initial_view_dims else '0'
            text_box = TextBox(text_ax, f'/ {self.array_shape[i]}', initial=initial_text, label_pad=0.1)
            text_box.dim = i  # Store dimension index
            text_box.label.set_position((1.2, 0.5))  # Center the label above
            text_box.label.set_horizontalalignment('left')  # Center align the text
            self.text_boxes.append(text_box)

    def _setup_view_controls(self):
        """Setup the view control buttons (display mode, etc.)"""
        # Add view dimension checkboxes
        check_ax = self.fig.add_axes([0.85, 0.1, 0.1, 0.2])
        labels = [f'Dim {i}' for i in range(self.ndim)]
        initial_states = [False] * self.ndim
        self.view_dim_buttons = CheckButtons(check_ax, labels, initial_states)
        
        # Add display mode buttons
        modes_ax = self.fig.add_axes([0.85, 0.05, 0.1, 0.03])
        self.mode_button = Button(modes_ax, 'Mode')
    
    def update_dimension_text(self, dim: int, value: str):
        """Update the text in a specific dimension's text box"""
        self.text_boxes[dim].set_val(value)
    
    def update_view_checkboxes(self, view_dims: List[int]):
        """Update which dimensions are checked in the view controls"""
        for i in range(self.ndim):
            current_state = self.view_dim_buttons.get_status()[i]
            desired_state = i in view_dims
            if current_state != desired_state:
                self.view_dim_buttons.set_active(i)

    def show_view_dims_popup(self, current_view_dims: List[int], callback: Callable):
        """Create a popup dialog for selecting view dimensions."""
        popup_fig = plt.figure(figsize=(5, 2))
        popup_fig.canvas.manager.set_window_title("Set View Dimensions")
        
        # Make the figure background lighter for better visibility
        popup_fig.patch.set_facecolor('#f0f0f0')
        
        # Add instructions text
        popup_fig.text(0.5, 0.85, 
                    "Enter dimension indices separated by commas\n(e.g. 0,1)", 
                    ha='center', 
                    va='center')
        
        # Add text box for input with clear styling
        ax_textbox = popup_fig.add_axes([0.2, 0.4, 0.6, 0.2])
        current_dims_str = ",".join(str(dim) for dim in current_view_dims)
        textbox = TextBox(ax_textbox, "", initial=current_dims_str)
        
        # Add cancel button with better positioning
        ax_button = popup_fig.add_axes([0.35, 0.1, 0.3, 0.15])
        button = Button(ax_button, 'Cancel')
        button.on_clicked(lambda event: plt.close(popup_fig))
        
        def submit(text):
            try:
                new_view_dims = [int(dim.strip()) for dim in text.split(',')]
                if len(new_view_dims) >= 2:  # Basic validation
                    callback(new_view_dims)
                    plt.close(popup_fig)
                else:
                    popup_fig.text(0.5, 0.25, 
                                "Error: Need at least 2 dimensions", 
                                ha='center', 
                                color='red')
                    popup_fig.canvas.draw_idle()
            except ValueError:
                popup_fig.text(0.5, 0.25, 
                            "Error: Invalid format", 
                            ha='center', 
                            color='red')
                popup_fig.canvas.draw_idle()
        
        textbox.on_submit(submit)
        plt.show()

    def update_dimension_text_style(self, scroll_dim: int):
        """Update text box styles to highlight scroll dimension"""
        for i, text_box in enumerate(self.text_boxes):
            # Make scroll dimension bold, others normal
            text_box.label.set_weight('bold' if i == scroll_dim else 'normal')
            text_box.label.figure.canvas.draw_idle()
            text_box.text_disp.set_weight('bold' if i == scroll_dim else 'normal')
            text_box.text_disp.figure.canvas.draw_idle()