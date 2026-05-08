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


def list_npz_keys(filepath):
    """Return [{key, shape, dtype}] for each ndarray in an .npz file.

    Filters out 0-d arrays (scalars) and non-ndarray entries.
    """
    npz = np.load(filepath)
    keys = []
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim >= 1:
            keys.append({"key": k, "shape": list(arr.shape), "dtype": str(arr.dtype)})
    npz.close()
    return keys


def _select_npz_array(npz, filepath):
    """Load the first array from a multi-array .npz file.

    The in-viewer NPZ picker handles array selection — no terminal prompt.
    """
    keys = list(npz.keys())
    return npz[keys[0]]


def _load_nifti_with_meta(filepath):
    """Load a NIfTI file, canonical-reorient, return (array, meta).

    meta is a dict with keys:
      affine            : original 4x4 affine (RAS+ mm)
      affine_canonical  : 4x4 affine after as_closest_canonical
      voxel_sizes       : tuple (sx, sy, sz) in mm, post-reorient
      axis_labels       : tuple of 3 strs from {"R","L","A","P","S","I"}
                          — positive direction of each canonical axis
      is_oblique        : bool — True if rotation part has off-diagonal magnitude > 1e-3
                          after normalizing voxel sizes
    """
    nib = _nib()
    img = nib.load(filepath)
    original_affine = np.asarray(img.affine, dtype=np.float64)
    canon = nib.as_closest_canonical(img)
    affine_canonical = np.asarray(canon.affine, dtype=np.float64)

    # NOTE: reorient requires materializing axis permutes/flips. .nii.gz is
    # already eager (gzip not seekable), so this is free; .nii loses mmap as a
    # necessary cost to apply the reorient.
    arr = np.asarray(canon.dataobj)

    rot = affine_canonical[:3, :3]
    voxel_sizes = tuple(float(np.linalg.norm(rot[:, i])) for i in range(3))

    # Direction of each canonical axis (sign of diagonal after normalizing)
    norm_rot = np.zeros((3, 3))
    for i in range(3):
        if voxel_sizes[i] > 0:
            norm_rot[:, i] = rot[:, i] / voxel_sizes[i]
    pos_labels = ("R", "A", "S")
    neg_labels = ("L", "P", "I")
    axis_labels = tuple(
        pos_labels[i] if norm_rot[i, i] >= 0 else neg_labels[i] for i in range(3)
    )

    # Oblique = off-diagonal of normalized rotation has |val| > tol
    off_diag_max = 0.0
    for i in range(3):
        for j in range(3):
            if i != j:
                off_diag_max = max(off_diag_max, abs(norm_rot[i, j]))
    is_oblique = bool(off_diag_max > 1e-3)

    meta = {
        "affine": original_affine,
        "affine_canonical": affine_canonical,
        "voxel_sizes": voxel_sizes,
        "axis_labels": axis_labels,
        "is_oblique": is_oblique,
    }
    return arr, meta


def load_data_with_meta(filepath, key=None):
    """Like load_data but also returns spatial metadata for NIfTI files.

    Returns (array, meta_or_None). meta is None for non-NIfTI formats.
    """
    if filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return _load_nifti_with_meta(filepath)
    return load_data(filepath, key=key), None


def load_data(filepath, key=None):
    if filepath.endswith(".npy"):
        # Eager-load small-to-medium files into RAM.  mmap_mode="r" on a C-order
        # 4D+ array forces scattered page faults for every orthogonal slice
        # (elements are thousands of bytes apart), making first renders very slow.
        # Sequential read is much faster and the PENDING_SESSIONS background thread
        # hides the upfront cost.  Keep mmap only for files that don't fit in RAM.
        _NPY_EAGER_LIMIT = int(os.environ.get("AV_NPY_EAGER_BYTES", 2 * 1024**3))
        if os.path.getsize(filepath) < _NPY_EAGER_LIMIT:
            return np.load(filepath)
        return np.load(filepath, mmap_mode="r")
    elif filepath.endswith(".npz"):
        npz = np.load(filepath)
        try:
            if key is not None:
                return npz[key]
            keys = list(npz.keys())
            if len(keys) == 1:
                return npz[keys[0]]
            return _select_npz_array(npz, filepath)
        finally:
            npz.close()
    elif filepath.endswith(".nii.gz"):
        # Gzip streams aren't seekable, so nibabel's lazy dataobj decompresses
        # large portions of the file on every arbitrary slice access. Materialize
        # the volume up front so subsequent slicing is cheap.
        return np.asarray(_nib().load(filepath).dataobj)
    elif filepath.endswith(".nii"):
        # Uncompressed NIfTI is memory-mapped via dataobj — slicing is cheap.
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
