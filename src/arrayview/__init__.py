import sys

from .arrayview import ArrayView


__version__ = "0.1.0"
__all__ = ["arrayview"]


def arrayview(array) -> None:
    """Display an interactive view of a multidimensional array.

    Args:
        array: numpy array with at least 2 dimensions

    Returns:
        None: No return value
    """
    ArrayView(array)
    return None

    # # Check if QApplication already exists (e.g., in Jupyter or when called from REPL)
    # app = QtWidgets.QApplication.instance()
    # if app is None:
    #     # Create new application if none exists
    #     app = QtWidgets.QApplication(sys.argv)
    #     # Set up proper cleanup when app quits
    #     app.setQuitOnLastWindowClosed(True)
    #     new_app = True
    # else:
    #     # Use existing application (common in interactive environments)
    #     new_app = False

    # viewer = NDArrayViewer(array)
    # print("\nViewer started. Close the window when finished.")
    # viewer.show()

    # if new_app:
    #     # Only start event loop and exit if we created a new application
    #     # This prevents interfering with existing event loops (e.g., Jupyter, REPL)
    #     try:
    #         app.exec_()
    #     except KeyboardInterrupt:
    #         print("\nViewer interrupted by user.")
    #     finally:
    #         # Clean shutdown without sys.exit()
    #         viewer.close()
    # else:
    #     # In interactive environments, just return the viewer
    #     # The existing event loop will handle the window
    #     pass

    # return viewer
