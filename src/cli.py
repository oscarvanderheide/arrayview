import sys
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from scipy.io import loadmat

from arrayview import ArrayView


def main():
    """Command line interface for arrayview."""
    if len(sys.argv) != 2:
        print("Usage: arrayview <filename>")
        print("Supported formats: .nii.gz, .nii, .npy, .mat")
        print("Example: arrayview my_array.nii.gz")
        sys.exit(1)

    filepath = Path(sys.argv[1])

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
                    array_keys = list(f.keys())
                    # Create a dictionary-like structure for consistency
                    mat_data = {key: f[key][:] for key in array_keys}
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

    ArrayView(array)


if __name__ == "__main__":
    main()
