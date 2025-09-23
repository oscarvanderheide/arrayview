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


def main():
    """Command line interface for arrayview."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: arrayview <filename> [slice_indices]")
        print("Supported formats: .nii.gz, .nii, .npy, .mat")
        print("Example: arrayview my_array.nii.gz")
        print("Example: arrayview my_array.nii.gz 'x,y,10,0'")
        print("Example: arrayview my_array.nii.gz ':,:,10,0'")
        print("  - Use 'x', 'y', 'z' to specify display axes")
        print("  - Use ':' for automatic axis assignment")
        print("  - Use numbers for slice indices")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    slice_str = sys.argv[2] if len(sys.argv) == 3 else None

    if not filepath.exists():
        print(f"Error: File {filepath} does not exist")
        sys.exit(1)

    # Load data based on file extension
    if filepath.suffix == ".gz" or filepath.suffix == ".nii":
        array = nib.load(filepath).get_fdata()
    elif filepath.suffix == ".npy":
        array = np.load(filepath)
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
        print("Supported formats: .nii.gz, .nii, .npy, .mat")
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

    # Parse slice indices if provided
    slice_config = None
    if slice_str:
        try:
            slice_config = parse_slice_indices(slice_str, array.ndim)
        except ValueError as e:
            print(f"Error parsing slice indices: {e}")
            sys.exit(1)

    # Create ArrayView with slice configuration
    if slice_config:
        # Prepare initial slices with default middle values
        initial_slices = [s // 2 for s in array.shape]

        # Update with specified slice indices
        for i, slice_idx in enumerate(slice_config["slices"]):
            if slice_idx is not None:
                if slice_idx < array.shape[i]:
                    initial_slices[i] = slice_idx
                else:
                    print(
                        f"Warning: slice index {slice_idx} exceeds dimension {i} size {array.shape[i]}"
                    )

        # Create ArrayView with initial slices
        av = ArrayView(
            array,
            x=slice_config["x"],
            y=slice_config["y"],
            z=slice_config["z"],
            initial_slices=initial_slices,
        )
    else:
        av = ArrayView(array)


if __name__ == "__main__":
    main()
