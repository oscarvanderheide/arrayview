"""Data loading, tensor conversion, and file-type utilities."""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Lazy nibabel import  (only needed for .nii / .nii.gz files)
# ---------------------------------------------------------------------------
_nib_mod = None


def _nib():
    """Lazy nibabel import."""
    global _nib_mod
    if _nib_mod is None:
        import nibabel

        _nib_mod = nibabel
    return _nib_mod


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _fix_mat_complex(arr):
    """Convert structured complex dtype from scipy.io.loadmat to native complex64.

    scipy.io.loadmat sometimes returns complex arrays as structured arrays with
    'real' and 'imag' fields instead of numpy's native complex dtype.
    """
    if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
        return (arr["real"] + 1j * arr["imag"]).astype(np.complex64)
    return arr


def load_data(filepath):
    if filepath.endswith(".npy"):
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".npz"):
        npz = np.load(filepath)
        keys = list(npz.keys())
        if len(keys) == 1:
            return npz[keys[0]]
        raise ValueError(
            f".npz contains multiple arrays: {keys}. "
            "Load it manually and pass the array to view()."
        )
    elif filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return _nib().load(filepath).dataobj
    elif filepath.endswith(".zarr") or filepath.endswith(".zarr.zip"):
        import zarr

        return zarr.open(filepath, mode="r")
    elif filepath.endswith(".pt") or filepath.endswith(".pth"):
        try:
            import torch
        except ImportError:
            raise ImportError("Install torch to load .pt/.pth files.")
        obj = torch.load(filepath, map_location="cpu", weights_only=True)
        return _tensor_to_numpy(obj, filepath)
    elif filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        import h5py

        f = h5py.File(filepath, "r")
        keys = list(f.keys())
        if len(keys) == 1:
            return f[keys[0]][()]
        raise ValueError(
            f".h5 file contains multiple datasets: {keys}. "
            "Load it manually and pass the array to view()."
        )
    elif filepath.endswith(".tif") or filepath.endswith(".tiff"):
        import tifffile

        return tifffile.imread(filepath)
    elif filepath.endswith(".mat"):
        try:
            import scipy.io

            mat = scipy.io.loadmat(filepath)
            arrays = {
                k: v
                for k, v in mat.items()
                if not k.startswith("_") and isinstance(v, np.ndarray)
            }
            if len(arrays) == 1:
                return _fix_mat_complex(next(iter(arrays.values())))
            raise ValueError(
                f".mat file contains multiple arrays: {list(arrays.keys())}. "
                "Load it manually and pass the array to view()."
            )
        except NotImplementedError:
            # MATLAB v7.3 files use HDF5 — scipy cannot load them; fall back to h5py.
            import h5py

            f = h5py.File(filepath, "r")
            arrays = {k: f[k] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
            if len(arrays) == 1:
                return np.array(next(iter(arrays.values())))
            raise ValueError(
                f".mat (v7.3) file contains multiple datasets: {list(arrays.keys())}. "
                "Load it manually with h5py and pass the array to view()."
            )
    else:
        raise ValueError(
            "Unsupported format. Supported: .npy, .npz, .nii/.nii.gz, .zarr, "
            ".pt/.pth, .h5/.hdf5, .tif/.tiff, .mat"
        )


def _tensor_to_numpy(obj, source="object"):
    """Convert a tensor-like object (PyTorch, etc.) to a numpy array."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    if hasattr(obj, "__array__"):
        return np.array(
            obj
        )  # copy — ensures external GC (e.g. Julia) can't free the backing memory
    raise ValueError(f"Cannot convert {type(obj)} from {source} to a numpy array.")


# ---------------------------------------------------------------------------
# File-type utilities
# ---------------------------------------------------------------------------

_SUPPORTED_EXTS = frozenset(
    [
        ".npy",
        ".npz",
        ".nii",
        ".nii.gz",
        ".zarr",
        ".zarr.zip",
        ".pt",
        ".pth",
        ".h5",
        ".hdf5",
        ".tif",
        ".tiff",
        ".mat",
    ]
)

# Formats that load the entire array into RAM (no mmap/lazy access).
# RAM guard only applies to these.
FULL_LOAD_EXTS = frozenset([".pt", ".pth", ".tif", ".tiff", ".mat"])


def _peek_file_shape(fpath: str, ext: str):
    """Try to return shape quickly without loading the full array. Returns None on failure."""
    try:
        if ext == ".npy":
            arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
            return list(arr.shape)
        if ext in (".nii", ".nii.gz"):
            return list(_nib().load(fpath).shape)
    except Exception:
        pass
    return None
