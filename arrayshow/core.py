import numpy as np
from typing import List, Union, Tuple
import matplotlib.pyplot as plt
from .state import ArrayShowState
from .events import ArrayShowEventSystem
from .ui import ArrayShowUI

class ArrayShow:
    def __init__(
        self,
        array: np.ndarray,
        view_dims=[0, 1],
        scroll_dim=2,
        cmap="viridis",
    ):
        # Validate input
        assert array.ndim > 2, "Array must have at least 3 dimensions, otherwise just use plot or imshow"
        
        # The array, and information on which dimensions are scrolling and viewing dimensions are stored in a state.
        self.state = ArrayShowState(array, view_dims, scroll_dim)

        # I don't understand this thing yet
        self.events = ArrayShowEventSystem()
        
        # Initialize the plot figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        # Disable default key bindings
        for key in ['k', 'j']:  # Add any other keys you want to override
            self.fig.canvas.mpl_disconnect(
                self.fig.canvas.manager.key_press_handler_id
            )

        # Initialize the graphical UI
        self.ui = ArrayShowUI(
            self.fig, 
            self.ax, 
            array.ndim,
            array.shape,
            view_dims,
            self.onscroll,
            self.onkeypress,
            self.onbuttonpress
        )
        
        # Add view dimensions change callback
        self.ui.on_view_dims_change = self._handle_view_dims_change
        
        # Initialize the image axis used for displaying slices of the array
        self._initialize_axis(cmap)

        # DUNNO
        self.events.subscribe('state_changed', self.update_view)
        self.events.subscribe('dimension_text_changed', self.ui.update_dimension_text)
        self.events.subscribe('scroll_dim_changed', self.ui.update_dimension_text_style)

        # Update the dimension text for the current scroll dimension (make it bold)
        self.ui.update_dimension_text_style(scroll_dim)
        
        # Auto-show the plot figure
        plt.show()

    def _initialize_axis(self, cmap):
        """Initialize the image axis (no ticks, colormap)"""
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.cmap = cmap
        self.im = self.ax.imshow(
            self.get_current_slice(),
            cmap=self.cmap,
            vmin=0,
            vmax=2
        )

    def get_current_slice(self):
        """Extract the current slice based on state information."""
        current_slice = self.state.array[tuple(self.state.slice_indices)]
        if current_slice.ndim == 3:
            current_slice = self._arrange_slices_grid(current_slice)
        elif current_slice.ndim > 3:
            raise NotImplementedError(
                "Display for arrays with more than 3 dimensions is not implemented yet."
            )
        return self._apply_display_mode(current_slice)

    def _arrange_slices_grid(self, slices_3d, screen_aspect=16/9):
        """Arrange 3D slices in a grid optimized for screen aspect ratio."""
        num_slices = slices_3d.shape[2]
        grid_cols = int(np.ceil(np.sqrt(num_slices * screen_aspect)))
        grid_rows = int(np.ceil(num_slices / grid_cols))
        
        # Pad array to fill grid
        padded = np.zeros(
            (slices_3d.shape[0], slices_3d.shape[1], grid_rows * grid_cols),
            dtype=slices_3d.dtype
        )
        padded[:, :, :num_slices] = slices_3d
        
        # Reshape and arrange in grid
        reshaped = padded.reshape(
            slices_3d.shape[0], slices_3d.shape[1], grid_rows, grid_cols
        )
        return np.block([[reshaped[:, :, i, j] 
                        for j in range(grid_cols)] 
                        for i in range(grid_rows)])

    def _apply_display_mode(self, img):
        """Apply the current display mode to the image."""
        if self.state.display_mode == "real":
            return np.real(img)
        elif self.state.display_mode == "imag":
            return np.imag(img)
        elif self.state.display_mode == "abs":
            return np.abs(img)
        elif self.state.display_mode == "angle":
            return np.angle(img)
        else:
            raise ValueError(f"Invalid display mode: {self.state.display_mode}")

    def update_view(self):
        """Update the display with current state."""
        try:
            current_slice = self.get_current_slice()
            if current_slice.ndim != 2:
                raise ValueError(f"Expected 2D slice, got {current_slice.ndim}D slice")

            self.im.set_data(current_slice)
            self.im.axes.figure.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating view: {e}")

    def onscroll(self, event):
        """Handle scroll events."""
        if event.button == "up":
            delta = 1
        elif event.button == "down":
            delta = -1
        else:
            return
            
        # Update state
        self.state.update_scroll_index(delta)
        # Emit event
        self.events.emit('dimension_text_changed', 
                self.state.scroll_dim, 
                str(self.state.scroll_index))
        self.events.emit('scroll_changed')
        self.events.emit('state_changed')

    def onkeypress(self, event):
        """Handle keyboard events."""
        if event.key in ["j", "k"]:  # Scroll up/down
            delta = 1 if event.key == "j" else -1
            self.state.update_scroll_index(delta)
            self.events.emit('scroll_changed')
            self.events.emit('dimension_text_changed', 
                    self.state.scroll_dim, 
                    str(self.state.scroll_index))
            
        elif event.key in ["h", "l"]:  # Change scroll dimension
            direction = "prev" if event.key == "h" else "next"
            new_dim = self.state.calculate_next_scroll_dim(direction)
            if new_dim is not None:
                self.state.set_scroll_dim(new_dim)
                self.events.emit('scroll_changed')
                self.events.emit('scroll_dim_changed', new_dim)
                
        elif event.key == "d":  # Debug info
            self._debug_state()
            
        elif event.key == ":":  # Show view dimensions popup
            self.ui.show_view_dims_popup(
                self.state.view_dims,
                self._handle_view_dims_change
            )
        
        self.events.emit('state_changed')
    
    def onbuttonpress(self, event):
        """Handle button press events."""
        if event.inaxes is None:
            return
            
        # Find which button was clicked by checking axes
        clicked_button = None
        for button in self.ui.button_up + self.ui.button_down:
            if event.inaxes == button.ax:
                clicked_button = button
                break
        
        if clicked_button is None:
            return
            
        dim = clicked_button.dim  # Get dimension associated with button
        
        # Check if it's an up or down button
        if clicked_button in self.ui.button_up:
            delta = 1
        elif clicked_button in self.ui.button_down:
            delta = -1
        else:
            return
        
        if dim == self.state.scroll_dim:
            # If clicking scroll dimension buttons, change index
            self.state.update_scroll_index(delta)
            self.events.emit('scroll_changed')
            self.events.emit('dimension_text_changed', 
                    self.state.scroll_dim, 
                    str(self.state.scroll_index))
        elif dim in self.state.fixed_dims:
            # If clicking fixed dimension buttons, make it the new scroll dim
            self.state.set_scroll_dim(dim)
            self.events.emit('scroll_changed')
            self.events.emit('scroll_dim_changed', dim)
        
        self.events.emit('state_changed')

    def _handle_view_dims_change(self, new_view_dims: List[int]) -> None:
        """Handle changes to view dimensions from popup."""
        try:
            # Validate input
            if len(new_view_dims) < 2:
                raise ValueError("Must specify at least 2 dimensions")
            
            if len(new_view_dims) != len(set(new_view_dims)):
                raise ValueError("Duplicate dimensions not allowed")
                
            if any(dim >= self.state.ndim for dim in new_view_dims):
                raise ValueError(f"Dimensions must be < {self.state.ndim}")
                
            if any(dim < 0 for dim in new_view_dims):
                raise ValueError("Dimensions must be non-negative")

            # Update state
            if self.state.set_view_dimensions(new_view_dims):
                self.events.emit('view_dims_changed')
                self.events.emit('scroll_dim_changed', self.state.scroll_dim)
                self.events.emit('state_changed')
                
        except ValueError as e:
            print(f"Invalid view dimensions: {e}")

    def _debug_state(self):
        """Print current state information for debugging."""
        print(f"Array shape: {self.state.array.shape}")
        print(f"View dims: {self.state.view_dims}")
        print(f"Scroll dim: {self.state.scroll_dim}")
        print(f"Scroll index: {self.state.scroll_index}")
        print(f"Fixed dims: {self.state.fixed_dims}")
        print(f"Slice indices: {self.state.slice_indices}")
