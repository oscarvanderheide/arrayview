import subprocess
import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.io import loadmat

from arrayview import ArrayView


def parse_slice_indices(slice_str, ndim):
    """Parse slice indices from string format.

    Args:
        slice_str: String like "x,y,10,0" or ":,:,10,0" or "x,y,1" (partial)
        ndim: Number of dimensions in the array

    Returns:
        dict: Dictionary with 'slices' list and axis mappings
    """
    if not slice_str:
        return None

    parts = slice_str.split(",")

    # Allow partial specifications - pad with defaults for missing dimensions
    if len(parts) > ndim:
        raise ValueError(f"Slice string has too many parts: {len(parts)} > {ndim}")

    # Pad parts with default values (:) for missing dimensions
    while len(parts) < ndim:
        parts.append(":")

    slices = [None] * ndim
    x_axis = None
    y_axis = None
    z_axis = None

    for i, part in enumerate(parts):
        part = part.strip()
        if part == ":":
            # This dimension will be used as a display axis
            if x_axis is None:
                x_axis = i
            elif y_axis is None:
                y_axis = i
            elif z_axis is None:
                z_axis = i
        elif part == "x":
            x_axis = i
        elif part == "y":
            y_axis = i
        elif part == "z":
            z_axis = i
        else:
            # This should be a slice index
            try:
                slices[i] = int(part)
            except ValueError:
                raise ValueError(f"Invalid slice value: {part}")

    # Set default axes if not specified
    if x_axis is None:
        x_axis = -1  # Last dimension
    if y_axis is None:
        y_axis = -2  # Second to last dimension

    return {"slices": slices, "x": x_axis, "y": y_axis, "z": z_axis}


def spawn_viewer(filepath, slice_str=None):
    """Spawn a new process to display the array viewer."""
    # Check if we should spawn a new process or run directly
    if "--direct" not in sys.argv:
        # Spawn new process with --direct flag to avoid infinite recursion
        cmd = [sys.executable, __file__, str(filepath)]
        if slice_str:
            cmd.append(slice_str)
        cmd.append("--direct")

        try:
            # Use subprocess.Popen to start detached process
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            print(f"ArrayView window spawned for: {filepath}")
            print("You can run the command again to open additional windows.")
            return
        except Exception as e:
            print(f"Failed to spawn process: {e}")
            print("Falling back to direct execution...")

    # Direct execution - this will block but only in the spawned process
    run_viewer_direct(filepath, slice_str)


def run_viewer_direct(filepath, slice_str=None):
    """Run the viewer directly (this will block)."""
    import matplotlib.pyplot as plt

    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)

    # Create an empty viewer window first
    av = ArrayView(im=None, title=f"Loading {filepath.name}...")
    plt.pause(0.01)  # Allow GUI to render

    # Load data based on file extension
    if filepath.suffix == ".gz" or filepath.suffix == ".nii":
        array = nib.load(filepath).get_fdata()
    elif filepath.suffix == ".npy":
        array = np.load(filepath)
    elif filepath.suffix == ".npz":
        # Load .npz file (may contain multiple arrays)
        try:
            npz_data = np.load(filepath)
            array_keys = list(npz_data.keys())
        except Exception as e:
            print(f"Error reading NPZ file: {e}")
            sys.exit(1)

        if len(array_keys) == 0:
            print("Error: No arrays found in NPZ file")
            sys.exit(1)
        elif len(array_keys) == 1:
            array = npz_data[array_keys[0]]
        else:
            print("Multiple arrays found in NPZ file:")
            for i, key in enumerate(array_keys):
                shape = npz_data[key].shape
                dtype = npz_data[key].dtype
                print(f"  {i + 1}. {key} (shape: {shape}, dtype: {dtype})")
            while True:
                try:
                    choice = input(
                        f"Select array to visualize (1-{len(array_keys)}): "
                    ).strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(array_keys):
                        selected_key = array_keys[choice_idx]
                        array = npz_data[selected_key]
                        print(f"Selected: {selected_key}")
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter a number between 1 and {len(array_keys)}"
                        )
                except (ValueError, KeyboardInterrupt):
                    print(
                        f"Invalid input. Please enter a number between 1 and {len(array_keys)}"
                    )
    elif filepath.suffix == ".mat":
        # Try to load with scipy first (for older .mat files)
        try:
            mat_data = loadmat(filepath)
            # Filter out MATLAB metadata keys that start with '__'
            array_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        except NotImplementedError:
            # Fall back to h5py for MATLAB v7.3 files
            try:
                with h5py.File(filepath, "r") as f:
                    array_keys = []
                    mat_data = {}

                    def collect_datasets(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            # Only include datasets with more than 0 dimensions
                            if obj.ndim > 0:
                                array_keys.append(name)
                                mat_data[name] = obj[:]

                    # Visit all items in the file
                    f.visititems(collect_datasets)

            except Exception as e:
                print(f"Error reading MATLAB file: {e}")
                sys.exit(1)

        if len(array_keys) == 0:
            print("Error: No data arrays found in MATLAB file")
            sys.exit(1)
        elif len(array_keys) == 1:
            # Single array - use it directly
            array = mat_data[array_keys[0]]
        else:
            # Multiple arrays - let user choose
            print("Multiple arrays found in MATLAB file:")
            for i, key in enumerate(array_keys):
                shape = mat_data[key].shape
                dtype = mat_data[key].dtype
                print(f"  {i + 1}. {key} (shape: {shape}, dtype: {dtype})")

            while True:
                try:
                    choice = input(
                        f"Select array to visualize (1-{len(array_keys)}): "
                    ).strip()
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(array_keys):
                        selected_key = array_keys[choice_idx]
                        array = mat_data[selected_key]
                        print(f"Selected: {selected_key}")
                        break
                    else:
                        print(
                            f"Invalid choice. Please enter a number between 1 and {len(array_keys)}"
                        )
                except (ValueError, KeyboardInterrupt):
                    print(
                        f"Invalid input. Please enter a number between 1 and {len(array_keys)}"
                    )
    else:
        print(f"Error: Unsupported file format: {filepath.suffix}")
        print("Supported formats: .nii.gz, .nii, .npy, .npz, .mat")
        sys.exit(1)

    # Handle structured complex arrays from MATLAB
    if hasattr(array, "dtype") and array.dtype.names:
        # Check if it's a structured array with real/imag fields
        if "real" in array.dtype.names and "imag" in array.dtype.names:
            # Convert to proper complex array
            array = array["real"] + 1j * array["imag"]
        elif len(array.dtype.names) == 1:
            # Single field structured array - extract the field
            field_name = array.dtype.names[0]
            array = array[field_name]

    array = 1000000 * array

    # Parse slice indices if provided
    slice_config = None
    if slice_str:
        try:
            slice_config = parse_slice_indices(slice_str, array.ndim)
        except ValueError as e:
            print(f"Error parsing slice indices: {e}")
            sys.exit(1)

    # Prepare initial slices and axes
    initial_slices = [s // 2 for s in array.shape]
    x_axis, y_axis, z_axis = -1, -2, None

    if slice_config:
        x_axis = slice_config.get("x", x_axis)
        y_axis = slice_config.get("y", y_axis)
        z_axis = slice_config.get("z", z_axis)
        for i, slice_idx in enumerate(slice_config["slices"]):
            if slice_idx is not None:
                if slice_idx < array.shape[i]:
                    initial_slices[i] = slice_idx
                else:
                    print(
                        f"Warning: slice index {slice_idx} exceeds dimension {i} size {array.shape[i]}"
                    )

    # Update viewer with loaded data
    av.x, av.y, av.z = x_axis, y_axis, z_axis
    av.title = filepath.name
    av.set_data(array, initial_slices=initial_slices)

    # Keep the plot window open and responsive
    plt.show()


def main():
    """Command line interface for arrayview."""
    if len(sys.argv) < 2:
        print("Usage: arrayview <filename> [slice_indices]")
        print("Supported formats: .nii.gz, .nii, .npy, .mat")
        print("Example: arrayview my_array.nii.gz")
        print("Example: arrayview my_array.nii.gz 'x,y,10,0'")
        print("Example: arrayview my_array.nii.gz ':,:,10,0'")
        print("  - Use 'x', 'y', 'z' to specify display axes")
        print("  - Use ':' for automatic axis assignment")
        print("  - Use numbers for slice indices")
        sys.exit(1)

    # Filter out --direct flag for argument parsing
    filtered_args = [arg for arg in sys.argv[1:] if arg != "--direct"]

    if len(filtered_args) < 1:
        print("Error: filename required")
        sys.exit(1)

    filepath = filtered_args[0]
    slice_str = filtered_args[1] if len(filtered_args) > 1 else None

    spawn_viewer(filepath, slice_str)


if __name__ == "__main__":
    main()
