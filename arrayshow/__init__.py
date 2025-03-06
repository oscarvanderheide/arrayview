from .core import ArrayShow
from typing import List

__version__ = "0.1.0"
__all__ = ["arrayshow"]

def arrayshow(
    array, 
    view_dims: List[int] = [0, 1], 
    scroll_dim: int = 2, 
    cmap: str = "viridis"
) -> ArrayShow:
    """Display an interactive view of a multidimensional array.
    
    Args:
        array: numpy array with at least 3 dimensions
        view_dims: dimensions to show in the 2D view (default: [0, 1])
        scroll_dim: dimension to scroll through (default: 2)
        cmap: matplotlib colormap to use (default: "viridis")
    
    Returns:
        ArrayShow: The viewer instance
    """
    return ArrayShow(array, view_dims, scroll_dim, cmap)
