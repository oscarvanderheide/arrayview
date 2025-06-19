import sys

from .viewer import NDArrayViewer

# --- Import PyQt5 directly ---
try:
    from PyQt5 import QtWidgets
except ImportError:
    print(
        "Error: This application requires PyQt5. Please install it (`pip install pyqt5`)."
    )
    sys.exit(1)


__version__ = "0.1.0"
__all__ = ["arrayshow"]


def arrayshow(array) -> NDArrayViewer:
    """Display an interactive view of a multidimensional array.

    Args:
        array: numpy array with at least 2 dimensions

    Returns:
        NDArrayViewer: The viewer instance
    """

    app = QtWidgets.QApplication(sys.argv)
    viewer = NDArrayViewer(array)
    print("\nViewer started.")
    viewer.show()

    sys.exit(app.exec())
    return None
