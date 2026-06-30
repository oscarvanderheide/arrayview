"""Data loading, tensor conversion, and file-type utilities."""

from __future__ import annotations

import fnmatch
import os
import threading
from collections import OrderedDict

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


def _is_viewable_mat_array(value):
    """True for MATLAB arrays ArrayView can display directly."""
    return (
        isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.dtype.kind in ("b", "i", "u", "f", "c")
    )


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


def list_mat_keys(filepath):
    """Return [{key, shape, dtype}] for each array in a .mat file.

    Handles both scipy-loadable .mat files and v7.3 HDF5-based files.
    Filters out metadata keys (starting with '_') and non-array entries.
    """
    try:
        import scipy.io

        mat = scipy.io.loadmat(filepath)
        keys = []
        for k, v in mat.items():
            if k.startswith("_"):
                continue
            if _is_viewable_mat_array(v):
                keys.append({"key": k, "shape": list(v.shape), "dtype": str(v.dtype)})
        return keys
    except NotImplementedError:
        import h5py

        f = h5py.File(filepath, "r")
        keys = []
        for k in f.keys():
            ds = f[k]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) >= 1 and ds.dtype.kind in ("b", "i", "u", "f", "c"):
                keys.append({"key": k, "shape": list(ds.shape), "dtype": str(ds.dtype)})
        f.close()
        return keys


def list_array_keys(filepath):
    """Return [{key, shape, dtype}] for each array in a multi-array file.

    Dispatches by extension: .npz or .mat.
    """
    if filepath.endswith(".npz"):
        return list_npz_keys(filepath)
    if filepath.endswith(".mat"):
        return list_mat_keys(filepath)
    return []


def default_array_key(filepath):
    """Return the first selectable array key for multi-array formats, if any."""
    keys = list_array_keys(filepath)
    return keys[0]["key"] if keys else None


def _select_npz_array(npz, filepath):
    """Load the first array from a multi-array .npz file.

    The in-viewer array picker handles array selection — no terminal prompt.
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


# ---------------------------------------------------------------------------
# NIfTI series — lazy 4D/5D view over a directory of same-shape volumes
# ---------------------------------------------------------------------------


class _NiftiSeries:
    """Lazy 4D/5D view over a directory of same-shape NIfTI files.

    ``file_matrix`` is a P x M list-of-lists of filepaths (P patients,
    M modalities).  With M == 1 the shape is ``(*vol_shape, P)``; with
    M > 1 it is ``(*vol_shape, P, M)``.  ``__getitem__`` maps the trailing
    stack axis/axes to the source file, opens it (canonical reorient,
    LRU-cached), and forwards the spatial key so only the requested slice
    is materialised.
    """

    _av_lazy = True

    def __init__(self, file_matrix, vol_shape, dtype, spatial_meta=None):
        self._file_matrix = file_matrix
        self._vol_shape = tuple(vol_shape)
        self._dtype = np.dtype(dtype)
        self._spatial_meta = spatial_meta
        n_patients = len(file_matrix)
        n_modalities = len(file_matrix[0]) if file_matrix else 0
        if n_modalities <= 1:
            self.shape = (*self._vol_shape, n_patients)
            self._stack_axes = (len(self.shape) - 1,)
        else:
            self.shape = (*self._vol_shape, n_patients, n_modalities)
            self._stack_axes = (len(self.shape) - 2, len(self.shape) - 1)
        self.ndim = len(self.shape)
        self._vol_cache: OrderedDict = OrderedDict()
        self._vol_cache_cap = int(
            os.environ.get("ARRAYVIEW_NIFTI_SERIES_VOL_CACHE", 3)
        )
        self._cache_lock = threading.Lock()

    @property
    def dtype(self):
        return self._dtype

    def _get_volume(self, p_idx, m_idx):
        cache_key = (p_idx, m_idx)
        with self._cache_lock:
            if cache_key in self._vol_cache:
                self._vol_cache.move_to_end(cache_key)
                return self._vol_cache[cache_key]
        filepath = self._file_matrix[p_idx][m_idx]
        nib = _nib()
        img = nib.load(filepath)
        canon = nib.as_closest_canonical(img)
        vol = np.asarray(canon.dataobj)
        with self._cache_lock:
            self._vol_cache[cache_key] = vol
            self._vol_cache.move_to_end(cache_key)
            while len(self._vol_cache) > self._vol_cache_cap:
                self._vol_cache.popitem(last=False)
        return vol

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        has_special = any(k is None or k is Ellipsis for k in key)
        if not has_special and len(key) <= self.ndim:
            padded = list(key) + [slice(None)] * (self.ndim - len(key))
            stack_vals = [padded[i] for i in self._stack_axes]
            if all(isinstance(s, (int, np.integer)) for s in stack_vals):
                spatial_positions = [
                    i for i in range(self.ndim) if i not in self._stack_axes
                ]
                spatial_key = tuple(padded[i] for i in spatial_positions)
                n_p = len(self._file_matrix)
                n_m = len(self._file_matrix[0]) if self._file_matrix else 0
                p_idx = int(stack_vals[0])
                if p_idx < 0:
                    p_idx += n_p
                if len(self._stack_axes) == 1:
                    m_idx = 0
                else:
                    m_idx = int(stack_vals[1])
                    if m_idx < 0:
                        m_idx += n_m
                vol = self._get_volume(p_idx, m_idx)
                return vol[spatial_key]
        return np.asarray(self)[key]

    def __array__(self, dtype=None):
        result = np.empty(self.shape, dtype=self._dtype)
        for p, row in enumerate(self._file_matrix):
            for m, fpath in enumerate(row):
                vol = self._get_volume(p, m)
                if len(self._stack_axes) == 1:
                    result[..., p] = vol
                else:
                    result[..., p, m] = vol
        if dtype is not None:
            result = result.astype(dtype)
        return result


def _load_nifti_series(path, select=None):
    """Build a lazy ``_NiftiSeries`` from a directory of NIfTI files.

    Walks *path* recursively, groups ``.nii``/``.nii.gz`` by immediate parent
    folder (= patient).  With *select* (a list of fnmatch patterns), picks one
    file per pattern per patient → 5D ``(*vol, P, M)``.  Without *select*,
    requires exactly one NIfTI per patient → 4D ``(*vol, P)``.

    Returns ``(series, spatial_meta)``.
    """
    nib = _nib()

    patients: dict[str, list[str]] = {}
    for root, _dirs, files in os.walk(path):
        nii_files = sorted(
            os.path.join(root, f)
            for f in files
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        if nii_files:
            patients[root] = nii_files

    if not patients:
        raise ValueError(
            f"No NIfTI files (.nii/.nii.gz) found under {path!r}. "
            "If the folder contains DICOM (.dcm) files, convert them to NIfTI "
            "first (e.g. `dcm2niix -o <out> <dicom_dir>`)."
        )

    select_patterns = select or []
    patient_dirs = sorted(patients.keys())
    file_matrix: list[list[str]] = []

    if select_patterns:
        for pdir in patient_dirs:
            selected: list[str] = []
            for pattern in select_patterns:
                matches = [
                    f
                    for f in patients[pdir]
                    if fnmatch.fnmatch(os.path.basename(f), pattern)
                ]
                if not matches:
                    raise ValueError(
                        f"Patient folder {pdir!r}: no file matches --select "
                        f"pattern {pattern!r}. Available: "
                        f"{[os.path.basename(f) for f in patients[pdir]]}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"Patient folder {pdir!r}: multiple files match "
                        f"--select pattern {pattern!r}: "
                        f"{[os.path.basename(f) for f in matches]}. "
                        "Make patterns more specific."
                    )
                selected.append(matches[0])
            file_matrix.append(selected)
    else:
        for pdir in patient_dirs:
            nii_files = patients[pdir]
            if len(nii_files) == 1:
                file_matrix.append([nii_files[0]])
            else:
                raise ValueError(
                    f"Patient folder {pdir!r} contains {len(nii_files)} NIfTI "
                    f"files: {[os.path.basename(f) for f in nii_files]}. "
                    "Use --select PATTERN to pick one (or more) per patient. "
                    "Example: --select '*t1*' --select '*t2*' --select '*flair*'"
                )

    ref_shape = None
    ref_dtype = None
    for row in file_matrix:
        for fpath in row:
            img = nib.load(fpath)
            shape = tuple(int(s) for s in img.shape)
            dtype = img.get_data_dtype()
            if ref_shape is None:
                ref_shape = shape
                ref_dtype = dtype
            elif shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: {os.path.basename(fpath)!r} has shape "
                    f"{shape}, expected {ref_shape}."
                )
            elif dtype != ref_dtype:
                raise ValueError(
                    f"Dtype mismatch: {os.path.basename(fpath)!r} has dtype "
                    f"{dtype}, expected {ref_dtype}."
                )

    _, spatial_meta = _load_nifti_with_meta(file_matrix[0][0])
    series = _NiftiSeries(file_matrix, ref_shape, ref_dtype, spatial_meta=spatial_meta)
    return series, spatial_meta


def load_data_with_meta(filepath, key=None, select=None):
    """Like load_data but also returns spatial metadata for NIfTI files.

    Returns (array, meta_or_None). meta is None for non-NIfTI formats.
    When *filepath* is a directory, loads it as a NIfTI series (see
    ``_load_nifti_series``).
    """
    if os.path.isdir(filepath):
        return _load_nifti_series(filepath, select=select)
    if filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return _load_nifti_with_meta(filepath)
    return load_data(filepath, key=key), None


def load_data(filepath, key=None):
    if os.path.isdir(filepath):
        series, _meta = _load_nifti_series(filepath)
        return series
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
                if not k.startswith("_") and _is_viewable_mat_array(v)
            }
            if key is not None:
                return _fix_mat_complex(arrays[key])
            if len(arrays) == 1:
                return _fix_mat_complex(next(iter(arrays.values())))
            raise ValueError(
                f".mat file contains multiple arrays: {list(arrays.keys())}. "
                "Select one in the viewer or pass a key."
            )
        except NotImplementedError:
            # MATLAB v7.3 files use HDF5 — scipy cannot load them; fall back to h5py.
            import h5py

            f = h5py.File(filepath, "r")
            arrays = {k: f[k] for k in f.keys() if isinstance(f[k], h5py.Dataset)}
            if key is not None:
                return np.array(arrays[key])
            if len(arrays) == 1:
                return np.array(next(iter(arrays.values())))
            raise ValueError(
                f".mat (v7.3) file contains multiple datasets: {list(arrays.keys())}. "
                "Select one in the viewer or pass a key."
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
        if os.path.isdir(fpath):
            series, _ = _load_nifti_series(fpath)
            return list(series.shape)
        if ext == ".npy":
            arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
            return list(arr.shape)
        if ext in (".nii", ".nii.gz"):
            return list(_nib().load(fpath).shape)
    except Exception:
        pass
    return None
