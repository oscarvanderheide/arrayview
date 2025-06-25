import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from .arrayshow import ImagePlot


def main():
    """Command line interface for arrayshow."""
    if len(sys.argv) != 2:
        print("Usage: arrayshow <filename>")
        print("Supported formats: .nii.gz, .nii, .npy")
        print("Example: arrayshow my_array.nii.gz")
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
    else:
        print(f"Error: Unsupported file format: {filepath.suffix}")
        print("Supported formats: .nii.gz, .nii, .npy")
        sys.exit(1)

    # Show array
    ImagePlot(array)


if __name__ == "__main__":
    main()
