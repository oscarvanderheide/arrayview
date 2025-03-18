from typing import List, Union, Tuple
import numpy as np

class ArrayShowState:
    def __init__(self, array: np.ndarray, view_dims: List[int] = [0, 1], scroll_dim: int = 2):
        """Holds the array and information on which dimensions to view/scroll"""
        self.array = array
        self.ndim = array.ndim
        self.view_dims = view_dims
        self.scroll_dim = scroll_dim
        self.scroll_index = 0 #array.shape[scroll_dim] // 2
        self.fixed_dims = self._calculate_fixed_dims()
        self.slice_indices = self._initialize_slice_indices()
        self.display_mode = "real" if np.isrealobj(array) else "abs"
    
    def _calculate_fixed_dims(self) -> List[int]:
        """Given the viewing and scrolling dimensions, calculate the fixed dimensions."""
        all_dims_set = set(range(self.ndim))
        scroll_dim_set = {self.scroll_dim}
        view_dims_set = set(self.view_dims)
        return list(all_dims_set - scroll_dim_set - view_dims_set)
    
    def _initialize_slice_indices(self) -> List[Union[int, slice]]:
        """Initialize the slice indices for the array."""
        slice_indices = [0] * self.ndim
        slice_indices[self.scroll_dim] = self.scroll_index
        slice_indices[self.view_dims[0]] = slice(None)
        slice_indices[self.view_dims[1]] = slice(None)
        return slice_indices
    
    def update_scroll_index(self, delta: int) -> None:
        """Update scroll index with bounds checking."""
        self.scroll_index += delta
        # Clip to valid range
        self.scroll_index = np.clip(
            self.scroll_index, 
            0, 
            self.array.shape[self.scroll_dim] - 1
        )
        # Update slice indices
        self.slice_indices[self.scroll_dim] = self.scroll_index

    def calculate_next_scroll_dim(self, direction: str) -> Union[int, None]:
        """Calculate the next scroll dimension in the given direction."""
        if not self.fixed_dims:  # If there are no fixed dimensions
            return None
            
        # Get current index in fixed_dims
        try:
            current_idx = list(self.fixed_dims).index(self.scroll_dim)
        except ValueError:
            # Current scroll dim not in fixed dims, start from beginning
            return min(self.fixed_dims)
            
        if direction == "next":
            if current_idx < len(self.fixed_dims) - 1:
                # Move to next dimension
                return list(self.fixed_dims)[current_idx + 1]
            else:
                # Wrap around to smallest
                return min(self.fixed_dims)
        elif direction == "prev":
            if current_idx > 0:
                # Move to previous dimension
                return list(self.fixed_dims)[current_idx - 1]
            else:
                # Wrap around to largest
                return max(self.fixed_dims)
        
        return None

    def set_scroll_dim(self, new_dim: int) -> None:
        """Set the scroll dimension with validation."""
        if new_dim >= self.ndim:
            raise ValueError("Scroll dimension must be less than array dimensions")
        if new_dim in self.view_dims:
            raise ValueError("Scroll dimension cannot be a view dimension")
            
        self.scroll_dim = new_dim
        self.fixed_dims = self._calculate_fixed_dims()
        # Reset scroll index for new dimension
        self.scroll_index = 0
        self.slice_indices[self.scroll_dim] = self.scroll_index

    def set_view_dimensions(self, new_view_dims: List[int]) -> bool:
        """Set new view dimensions with validation.
        
        Returns:
            bool: True if dimensions were changed, False otherwise
        """
        if set(new_view_dims) == set(self.view_dims):
            return False
            
        # Update scroll dimension if needed
        if self.scroll_dim in new_view_dims:
            # Find first available dimension not in view_dims
            for i in range(self.ndim):
                if i not in new_view_dims:
                    self.scroll_dim = i
                    self.scroll_index = 0
                    break
        
        self.view_dims = new_view_dims
        self.fixed_dims = self._calculate_fixed_dims()
        self._update_slice_indices()
        return True
        
    def _update_slice_indices(self) -> None:
        """Update slice indices after dimension changes."""
        for i in range(self.ndim):
            if i in self.view_dims:
                self.slice_indices[i] = slice(None)
            elif i == self.scroll_dim:
                self.slice_indices[i] = self.scroll_index
            else:  # fixed dimensions
                self.slice_indices[i] = 0