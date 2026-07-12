"""Data loading, tensor conversion, and file-type utilities."""

from __future__ import annotations

import fnmatch
import glob
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
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


def _nifti_header_with_meta(filepath):
    """Return a NIfTI image and canonical spatial metadata without reading voxels."""
    nib = _nib()
    img = nib.load(filepath)
    original_affine = np.asarray(img.affine, dtype=np.float64)
    source_ornt = nib.orientations.io_orientation(original_affine)
    canonical_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source_ornt, canonical_ornt)
    affine_canonical = original_affine @ nib.orientations.inv_ornt_aff(
        transform, img.shape[:3]
    )

    rot = affine_canonical[:3, :3]
    voxel_sizes = tuple(float(np.linalg.norm(rot[:, i])) for i in range(3))
    norm_rot = np.zeros((3, 3))
    for i in range(3):
        if voxel_sizes[i] > 0:
            norm_rot[:, i] = rot[:, i] / voxel_sizes[i]
    pos_labels = ("R", "A", "S")
    neg_labels = ("L", "P", "I")
    axis_labels = tuple(
        pos_labels[i] if norm_rot[i, i] >= 0 else neg_labels[i] for i in range(3)
    )
    off_diag_max = max(
        (abs(norm_rot[i, j]) for i in range(3) for j in range(3) if i != j),
        default=0.0,
    )
    canonical_shape = tuple(int(img.shape[int(axis)]) for axis in transform[:, 0])
    return img, {
        "affine": original_affine,
        "affine_canonical": affine_canonical,
        "voxel_sizes": voxel_sizes,
        "axis_labels": axis_labels,
        "is_oblique": bool(off_diag_max > 1e-3),
        "canonical_shape": canonical_shape,
    }


def _nifti_display_array(proxy):
    """Materialize a NIfTI proxy without carrying float64 into display caches."""
    dtype = np.dtype(proxy.dtype)
    display_dtype = np.float32 if np.issubdtype(dtype, np.floating) and dtype.itemsize > 4 else dtype
    return np.asarray(proxy, dtype=display_dtype)


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
    img, meta = _nifti_header_with_meta(filepath)
    canon = _nib().as_closest_canonical(img)

    # NOTE: reorient requires materializing axis permutes/flips. .nii.gz is
    # already eager (gzip not seekable), so this is free; .nii loses mmap as a
    # necessary cost to apply the reorient.
    arr = np.asarray(canon.dataobj)

    return arr, meta


# ---------------------------------------------------------------------------
# File series — lazy 4D/5D view over a directory of same-shape volumes
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
        self._vol_cache_bytes = 0
        self._vol_cache_max_bytes = int(
            os.environ.get("ARRAYVIEW_NIFTI_CACHE_BYTES", 2 * 1024**3)
        )
        self._cache_lock = threading.Lock()
        self._load_locks = {}

    @property
    def dtype(self):
        return self._dtype

    def _get_volume(self, p_idx, m_idx):
        cache_key = (p_idx, m_idx)
        with self._cache_lock:
            if cache_key in self._vol_cache:
                self._vol_cache.move_to_end(cache_key)
                return self._vol_cache[cache_key]
            load_lock = self._load_locks.setdefault(cache_key, threading.Lock())
        with load_lock:
            with self._cache_lock:
                if cache_key in self._vol_cache:
                    self._vol_cache.move_to_end(cache_key)
                    return self._vol_cache[cache_key]
            filepath = self._file_matrix[p_idx][m_idx]
            if filepath is None:
                return np.zeros(self._vol_shape, dtype=self._dtype)
            nib = _nib()
            img = nib.load(filepath)
            canon = nib.as_closest_canonical(img)
            # Keep uncompressed NIfTI proxy-backed so nibabel/OS paging can read
            # only the requested slice. Gzip files must be decompressed once and
            # are therefore materialized into the byte-bounded cache.
            vol = canon.dataobj if filepath.endswith(".nii") else _nifti_display_array(canon.dataobj)
            with self._cache_lock:
                self._vol_cache[cache_key] = vol
                self._vol_cache.move_to_end(cache_key)
                self._vol_cache_bytes += int(getattr(vol, "nbytes", 0))
                while self._vol_cache_bytes > self._vol_cache_max_bytes and len(self._vol_cache) > 1:
                    _key, evicted = self._vol_cache.popitem(last=False)
                    self._vol_cache_bytes -= int(getattr(evicted, "nbytes", 0))
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


class _FileSeries:
    """Lazy 4D/5D view over a directory of same-shape array files.

    Like ``_NiftiSeries`` but works with any supported file format
    (.npy, .npz, .zarr, .pt/.pth, .h5/.hdf5, .tif/.tiff, .mat).
    Uses ``load_data`` for each volume — no nibabel reorientation.
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
            os.environ.get("ARRAYVIEW_FILE_SERIES_VOL_CACHE", 3)
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
        if filepath is None:
            return np.zeros(self._vol_shape, dtype=self._dtype)
        vol = load_data(filepath)
        with self._cache_lock:
            self._vol_cache[cache_key] = vol
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


class _RaggedFileSeries:
    """Lazy collection view over same-rank, same-dtype files with varying shapes.

    The public ``shape`` is a max-spatial-shape facade so existing viewer
    metadata and indexing code can keep a stable ndim.  Rendering code should
    use ``spatial_shape_for_indices`` / ``clamp_indices`` before slicing.
    """

    _av_lazy = True
    _av_ragged = True

    def __init__(self, file_matrix, spatial_shapes, dtype, spatial_meta=None, all_nifti=False):
        self._file_matrix = file_matrix
        self._spatial_shapes = [
            [tuple(int(s) for s in shape) for shape in row]
            for row in spatial_shapes
        ]
        self._dtype = np.dtype(dtype)
        self._spatial_meta = spatial_meta
        self._all_nifti = bool(all_nifti)
        self._spatial_ndim = len(self._spatial_shapes[0][0])
        self._vol_shape = tuple(
            max(row[m][d] for row in self._spatial_shapes for m in range(len(row)))
            for d in range(self._spatial_ndim)
        )
        n_patients = len(file_matrix)
        n_modalities = len(file_matrix[0]) if file_matrix else 0
        if n_modalities <= 1:
            self.shape = (*self._vol_shape, n_patients)
            self._stack_axes = (len(self.shape) - 1,)
        else:
            self.shape = (*self._vol_shape, n_patients, n_modalities)
            self._stack_axes = (len(self.shape) - 2, len(self.shape) - 1)
        self.ndim = len(self.shape)
        self.nbytes = int(
            sum(np.prod(shape) for row in self._spatial_shapes for shape in row)
            * self._dtype.itemsize
        )
        self._vol_cache: OrderedDict = OrderedDict()
        self._vol_cache_bytes = 0
        self._vol_cache_max_bytes = int(
            os.environ.get("ARRAYVIEW_NIFTI_CACHE_BYTES", 2 * 1024**3)
        )
        self._cache_lock = threading.Lock()
        self._load_locks = {}

    @property
    def dtype(self):
        return self._dtype

    @property
    def ragged_spatial_shapes(self):
        return self._spatial_shapes

    def _normalize_collection_indices(self, idx_list):
        n_p = len(self._file_matrix)
        n_m = len(self._file_matrix[0]) if self._file_matrix else 0
        p_idx = int(idx_list[self._stack_axes[0]]) if len(idx_list) > self._stack_axes[0] else 0
        if p_idx < 0:
            p_idx += n_p
        p_idx = max(0, min(n_p - 1, p_idx))
        if len(self._stack_axes) == 1:
            m_idx = 0
        else:
            m_idx = int(idx_list[self._stack_axes[1]]) if len(idx_list) > self._stack_axes[1] else 0
            if m_idx < 0:
                m_idx += n_m
            m_idx = max(0, min(n_m - 1, m_idx))
        return p_idx, m_idx

    def spatial_shape_for_indices(self, idx_list):
        p_idx, m_idx = self._normalize_collection_indices(idx_list)
        return self._spatial_shapes[p_idx][m_idx]

    def clamp_indices(self, idx_list):
        out = list(idx_list)
        spatial_shape = self.spatial_shape_for_indices(out)
        for dim, size in enumerate(spatial_shape):
            if dim < len(out):
                out[dim] = max(0, min(int(size) - 1, int(out[dim])))
        for axis in self._stack_axes:
            if axis < len(out):
                out[axis] = self._normalize_collection_indices(out)[0 if axis == self._stack_axes[0] else 1]
        return out

    def _get_volume(self, p_idx, m_idx):
        cache_key = (p_idx, m_idx)
        with self._cache_lock:
            if cache_key in self._vol_cache:
                self._vol_cache.move_to_end(cache_key)
                return self._vol_cache[cache_key]
            load_lock = self._load_locks.setdefault(cache_key, threading.Lock())
        with load_lock:
            with self._cache_lock:
                if cache_key in self._vol_cache:
                    self._vol_cache.move_to_end(cache_key)
                    return self._vol_cache[cache_key]
            filepath = self._file_matrix[p_idx][m_idx]
            if filepath is None:
                return np.zeros(self._spatial_shapes[p_idx][m_idx], dtype=self._dtype)
            if self._all_nifti:
                nib = _nib()
                img = nib.load(filepath)
                proxy = nib.as_closest_canonical(img).dataobj
                vol = proxy if filepath.endswith(".nii") else _nifti_display_array(proxy)
            else:
                vol = load_data(filepath)
            with self._cache_lock:
                self._vol_cache[cache_key] = vol
                self._vol_cache.move_to_end(cache_key)
                self._vol_cache_bytes += int(getattr(vol, "nbytes", 0))
                while self._vol_cache_bytes > self._vol_cache_max_bytes and len(self._vol_cache) > 1:
                    _key, evicted = self._vol_cache.popitem(last=False)
                    self._vol_cache_bytes -= int(getattr(evicted, "nbytes", 0))
            return vol

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        has_special = any(k is None or k is Ellipsis for k in key)
        if not has_special and len(key) <= self.ndim:
            padded = list(key) + [slice(None)] * (self.ndim - len(key))
            stack_vals = [padded[i] for i in self._stack_axes]
            if all(isinstance(s, (int, np.integer)) for s in stack_vals):
                p_idx, m_idx = self._normalize_collection_indices(padded)
                spatial_shape = self._spatial_shapes[p_idx][m_idx]
                spatial_positions = [
                    i for i in range(self.ndim) if i not in self._stack_axes
                ]
                spatial_key = []
                for src_i, dim in enumerate(spatial_positions):
                    val = padded[dim]
                    if isinstance(val, (int, np.integer)):
                        idx = int(val)
                        if idx < 0:
                            idx += spatial_shape[src_i]
                        idx = max(0, min(spatial_shape[src_i] - 1, idx))
                        spatial_key.append(idx)
                    else:
                        spatial_key.append(val)
                return self._get_volume(p_idx, m_idx)[tuple(spatial_key)]
        raise TypeError("Ragged collections require concrete collection indices.")

    def __array__(self, dtype=None):
        raise TypeError("Ragged collections cannot be materialized as one ndarray.")


def _is_nifti_path(filepath):
    """True if *filepath* ends with .nii or .nii.gz."""
    return filepath.endswith(".nii") or filepath.endswith(".nii.gz")


def _get_ext(filepath):
    """Return the extension of *filepath*, normalising .nii.gz and .zarr.zip."""
    lower = filepath.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    if lower.endswith(".zarr.zip"):
        return ".zarr.zip"
    return os.path.splitext(lower)[1]


def _is_supported_collection_path(filepath):
    return _get_ext(filepath) in _SUPPORTED_EXTS and os.path.exists(filepath)


def _strip_array_ext(filepath):
    base = os.path.basename(filepath)
    lower = base.lower()
    for suffix in (".nii.gz", ".zarr.zip"):
        if lower.endswith(suffix):
            return base[: -len(suffix)]
    return os.path.splitext(base)[0]


def _default_collection_case_key(filepath):
    stem = _strip_array_ext(filepath)
    return re.sub(r"_[0-9]{4}$", "", stem)


def _collection_case_key(filepath, case_regex=None):
    if not case_regex:
        return _default_collection_case_key(filepath)
    match = re.search(case_regex, filepath)
    if match is None:
        raise ValueError(
            f"{filepath!r} does not match --case-regex {case_regex!r}."
        )
    groups = match.groupdict()
    if "case" not in groups:
        raise ValueError("--case-regex must define a named (?P<case>...) group.")
    return groups["case"]


def _collection_pattern_paths(pattern):
    paths = sorted(
        os.path.abspath(p)
        for p in glob.glob(pattern, recursive=True)
        if _is_supported_collection_path(p)
    )
    if not paths:
        raise ValueError(f"No supported array files match pattern {pattern!r}.")
    return paths


def _collection_pattern_map(pattern, case_regex):
    paths = _collection_pattern_paths(pattern)
    by_case = {}
    for path in paths:
        case = _collection_case_key(path, case_regex=case_regex)
        if case in by_case:
            raise ValueError(
                f"Pattern {pattern!r} matched multiple files for case {case!r}: "
                f"{os.path.basename(by_case[case])!r} and {os.path.basename(path)!r}."
            )
        by_case[case] = path
    return by_case


def _collection_ancestor_case_key(filepath, depth):
    parent = os.path.abspath(filepath)
    for _ in range(depth):
        parent = os.path.dirname(parent)
    return os.path.basename(parent)


def _collection_paths_map(paths, pattern, *, ancestor_depth):
    by_case = {}
    for path in paths:
        case = _collection_ancestor_case_key(path, ancestor_depth)
        if not case or case in by_case:
            return None
        by_case[case] = path
    return by_case


def _infer_collection_layout_maps(path_lists, patterns):
    """Infer a shared per-case directory depth for collection patterns."""
    if not path_lists or len(path_lists[0]) < 2:
        return None, None
    for depth in (1, 2, 3):
        maps = [
            _collection_paths_map(paths, pattern, ancestor_depth=depth)
            for paths, pattern in zip(path_lists, patterns)
        ]
        if any(by_case is None for by_case in maps):
            continue
        case_ids = set(maps[0])
        if case_ids and all(set(by_case) == case_ids for by_case in maps[1:]):
            return maps, depth
    return None, None


def _series_from_file_matrix(
    file_matrix, *, load="lazy", stack="auto", scan_progress=None, scan_label=None
):
    if load not in {"lazy", "eager"}:
        raise ValueError("load must be 'lazy' or 'eager'.")
    if stack not in {"auto", "dense", "ragged"}:
        raise ValueError("stack must be 'auto', 'dense', or 'ragged'.")
    if not file_matrix or not file_matrix[0]:
        raise ValueError("Cannot build an empty file series.")

    all_paths = [path for row in file_matrix for path in row if path is not None]
    if not all_paths:
        raise ValueError("Cannot build a file series without any files.")
    all_nifti = all(_is_nifti_path(path) for path in all_paths)
    ref_shape = None
    ref_dtype = None
    spatial_meta = None

    if all_nifti:
        spatial_shapes = []
        for fpath, img, item_meta in _iter_nifti_headers(
            all_paths, scan_progress=scan_progress, scan_label=scan_label
        ):
            shape = tuple(item_meta["canonical_shape"])
            dtype = img.get_data_dtype()
            spatial_shapes.append(shape)
            if ref_shape is None:
                ref_shape = shape
                ref_dtype = dtype
                spatial_meta = item_meta
            elif len(shape) != len(ref_shape):
                raise ValueError(
                    f"Rank mismatch: {os.path.basename(fpath)!r} has shape "
                    f"{shape}, expected rank {len(ref_shape)}."
                )
            elif dtype != ref_dtype:
                raise ValueError(
                    f"Dtype mismatch: {os.path.basename(fpath)!r} has dtype "
                    f"{dtype}, expected {ref_dtype}."
                )
        shape_matrix = []
        pos = 0
        for row in file_matrix:
            row_shapes = []
            for path in row:
                if path is None:
                    row_shapes.append(ref_shape)
                else:
                    row_shapes.append(spatial_shapes[pos])
                    pos += 1
            shape_matrix.append(row_shapes)
        shapes_differ = any(shape != ref_shape for shape in spatial_shapes)
        if stack == "dense" and shapes_differ:
            raise ValueError("Dense stacking requires every file to have the same shape.")
        if stack == "ragged" or shapes_differ:
            series = _RaggedFileSeries(
                    file_matrix,
                    shape_matrix,
                    ref_dtype,
                    spatial_meta=spatial_meta,
                    all_nifti=True,
            )
        else:
            series = _NiftiSeries(file_matrix, ref_shape, ref_dtype, spatial_meta)
        if load == "eager":
            for p, row in enumerate(file_matrix):
                for m, path in enumerate(row):
                    if path is not None:
                        series._get_volume(p, m)
        return series, spatial_meta

    spatial_shapes = []
    for fpath in all_paths:
        arr = load_data(fpath)
        shape = arr.shape
        dtype = arr.dtype
        spatial_shapes.append(shape)
        if ref_shape is None:
            ref_shape = shape
            ref_dtype = dtype
        elif len(shape) != len(ref_shape):
            raise ValueError(
                f"Rank mismatch: {os.path.basename(fpath)!r} has shape "
                f"{shape}, expected rank {len(ref_shape)}."
            )
        elif dtype != ref_dtype:
            raise ValueError(
                f"Dtype mismatch: {os.path.basename(fpath)!r} has dtype "
                f"{dtype}, expected {ref_dtype}."
            )
        if scan_progress is not None:
            scan_progress(scan_label, fpath)
    shape_matrix = []
    pos = 0
    for row in file_matrix:
        row_shapes = []
        for path in row:
            if path is None:
                row_shapes.append(ref_shape)
            else:
                row_shapes.append(spatial_shapes[pos])
                pos += 1
        shape_matrix.append(row_shapes)
    shapes_differ = any(shape != ref_shape for shape in spatial_shapes)
    if stack == "dense" and shapes_differ:
        raise ValueError("Dense stacking requires every file to have the same shape.")
    if stack == "ragged" or shapes_differ:
        series = _RaggedFileSeries(file_matrix, shape_matrix, ref_dtype, spatial_meta=None)
    else:
        series = _FileSeries(file_matrix, ref_shape, ref_dtype, spatial_meta=None)
    if load == "eager":
        for p, row in enumerate(file_matrix):
            for m, path in enumerate(row):
                if path is not None:
                    series._get_volume(p, m)
    return series, None


def _collection_scan_workers():
    raw = os.environ.get("ARRAYVIEW_COLLECTION_SCAN_WORKERS")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    cpu = os.cpu_count() or 4
    return max(1, min(8, cpu))


def _iter_nifti_headers(paths, *, scan_progress=None, scan_label=None):
    workers = _collection_scan_workers()
    if workers <= 1 or len(paths) < 8:
        for path in paths:
            img, item_meta = _nifti_header_with_meta(path)
            if scan_progress is not None:
                scan_progress(scan_label, path)
            yield path, img, item_meta
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, result in zip(paths, executor.map(_nifti_header_with_meta, paths)):
            img, item_meta = result
            if scan_progress is not None:
                scan_progress(scan_label, path)
            yield path, img, item_meta


def load_dir_collection(
    base_patterns,
    overlays=None,
    case_regex=None,
    *,
    load="lazy",
    stack="auto",
    scan_progress=None,
):
    """Load recursive collection patterns as aligned lazy image/overlay stacks.

    *base_patterns* are image channel/modality patterns.  *overlays* is an
    ordered list of ``(name, pattern)`` pairs. A third true value marks a
    sparse overlay whose missing cases should render as empty masks. By
    default, each pattern's sorted matches are paired by position. With
    *case_regex*, files are paired by the regex's named ``case`` group instead.
    """
    if not base_patterns:
        raise ValueError("--stack requires at least one positional image pattern.")
    overlays = overlays or []

    inferred_case_depth = None
    if case_regex:
        base_maps = [
            _collection_pattern_map(pattern, case_regex=case_regex)
            for pattern in base_patterns
        ]
        case_ids = sorted(base_maps[0].keys())
        missing = []
        for idx, by_case in enumerate(base_maps[1:], start=2):
            for case in case_ids:
                if case not in by_case:
                    missing.append(f"image pattern {idx} missing case {case!r}")
        if missing:
            raise ValueError("; ".join(missing))
        base_matrix = [[by_case[case] for by_case in base_maps] for case in case_ids]
    else:
        base_lists = [_collection_pattern_paths(pattern) for pattern in base_patterns]
        inferred_maps, inferred_case_depth = _infer_collection_layout_maps(
            base_lists, base_patterns
        )
        if inferred_maps is not None:
            case_ids = sorted(inferred_maps[0])
            base_matrix = [
                [by_case[case] for by_case in inferred_maps] for case in case_ids
            ]
        else:
            n_cases = len(base_lists[0])
            for idx, paths in enumerate(base_lists[1:], start=2):
                if len(paths) != n_cases:
                    raise ValueError(
                        f"Image pattern {idx} matched {len(paths)} file(s), "
                        f"expected {n_cases} to match image pattern 1."
                    )
            case_ids = [
                _default_collection_case_key(path)
                for path in base_lists[0]
            ]
            if len(set(case_ids)) != len(case_ids):
                case_ids = [
                    os.path.basename(os.path.dirname(path))
                    or _default_collection_case_key(path)
                    for path in base_lists[0]
                ]
            base_matrix = [list(row) for row in zip(*base_lists)]
    data, spatial_meta = _series_from_file_matrix(
        base_matrix,
        load=load,
        stack=stack,
        scan_progress=scan_progress,
        scan_label="images",
    )
    spatial_shape = data._vol_shape

    overlay_items = []
    overlay_reports = []
    for overlay_spec in overlays:
        name, pattern = overlay_spec[:2]
        allow_missing = len(overlay_spec) > 2 and bool(overlay_spec[2])
        report_pattern = overlay_spec[3] if len(overlay_spec) > 3 else pattern
        if isinstance(pattern, dict):
            by_case = pattern
            missing_cases = [case for case in case_ids if case not in by_case]
            if missing_cases and not allow_missing:
                raise ValueError(
                    f"Overlay {name!r} is missing case(s): "
                    f"{', '.join(repr(c) for c in missing_cases)}."
                )
            matrix = [[by_case.get(case)] for case in case_ids]
            extras = sorted(set(by_case.keys()) - set(case_ids))
        elif case_regex:
            by_case = _collection_pattern_map(pattern, case_regex=case_regex)
            missing_cases = [case for case in case_ids if case not in by_case]
            if missing_cases and not allow_missing:
                raise ValueError(
                    f"Overlay {name!r} is missing case(s): "
                    f"{', '.join(repr(c) for c in missing_cases)}."
                )
            matrix = [[by_case.get(case)] for case in case_ids]
            extras = sorted(set(by_case.keys()) - set(case_ids))
        elif inferred_case_depth is not None:
            paths = _collection_pattern_paths(pattern)
            by_case = _collection_paths_map(
                paths, pattern, ancestor_depth=inferred_case_depth
            )
            if by_case is None:
                raise ValueError(
                    f"Overlay {name!r} has multiple files for an inferred case. "
                    "Use --case-regex to describe this layout explicitly."
                )
            missing_cases = [case for case in case_ids if case not in by_case]
            if missing_cases and not allow_missing:
                raise ValueError(
                    f"Overlay {name!r} is missing case(s): "
                    f"{', '.join(repr(c) for c in missing_cases)}."
                )
            matrix = [[by_case.get(case)] for case in case_ids]
            extras = sorted(set(by_case) - set(case_ids))
        else:
            paths = _collection_pattern_paths(pattern)
            if len(paths) != len(case_ids) and not allow_missing:
                raise ValueError(
                    f"Overlay {name!r} matched {len(paths)} file(s), "
                    f"expected {len(case_ids)} to match image pattern 1."
                )
            if allow_missing:
                raise ValueError(
                    "Could not infer case directories for sparse overlays. "
                    "Use a per-case directory layout or pass --case-regex."
                )
            matrix = [[path] for path in paths]
            extras = []
        ov_data, _ov_meta = _series_from_file_matrix(
            matrix,
            load=load,
            stack=stack,
            scan_progress=scan_progress,
            scan_label=name,
        )
        if not getattr(data, "_av_ragged", False) and ov_data._vol_shape != spatial_shape:
            raise ValueError(
                f"Overlay {name!r} has spatial shape {ov_data._vol_shape}, "
                f"expected {spatial_shape}."
            )
        overlay_items.append({"name": name, "data": ov_data, "pattern": report_pattern})
        overlay_reports.append(
            {
                "name": name,
                "pattern": report_pattern,
                "ignored_cases": extras,
                "missing_cases": (
                    missing_cases
                    if case_regex or inferred_case_depth is not None
                    else []
                ),
            }
        )

    summary = {
        "cases": case_ids,
        "base_patterns": list(base_patterns),
        "overlays": overlay_reports,
        "shape": tuple(int(s) for s in data.shape),
        "spatial_shape": tuple(int(s) for s in spatial_shape),
    }
    return data, spatial_meta, overlay_items, summary


def _load_file_series(path, select=None, *, load="lazy", stack="auto"):
    """Build a lazy series from a directory of supported array files.

    Walks *path* recursively, groups files of any supported format by
    immediate parent folder (= patient).  With *select* (a list of fnmatch
    patterns), picks one file per pattern per patient → 5D ``(*vol, P, M)``.
    Without *select*, requires exactly one file per patient → 4D ``(*vol, P)``.

    If every file is NIfTI (.nii/.nii.gz), delegates to
    ``_load_nifti_series`` for canonical reorientation.  Otherwise builds
    a ``_FileSeries`` using ``load_data`` per volume.

    Returns ``(series, spatial_meta)``.
    """
    # ── collect files by parent directory ──────────────────────────
    patients: dict[str, list[str]] = {}
    all_nifti = True

    for root, dirs, files in os.walk(path):
        supported = sorted(
            os.path.join(root, f)
            for f in files
            if _get_ext(f) in _SUPPORTED_EXTS
        )
        if supported:
            patients[root] = supported
            if all_nifti:
                all_nifti = all(_is_nifti_path(p) for p in supported)
            # A directory with supported files is the series unit.  Do not
            # treat nested sub-folders as additional patients.
            dirs[:] = []

    if not patients:
        raise ValueError(
            f"No supported array files found under {path!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTS))}"
        )

    # NIfTI-only → preserve canonical reorientation + lazy dataobj slicing
    if all_nifti:
        return _load_nifti_series(path, select=select, load=load, stack=stack)

    # ── build file matrix ──────────────────────────────────────────
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
                        f"Folder {pdir!r}: no file matches --select "
                        f"pattern {pattern!r}. Available: "
                        f"{[os.path.basename(f) for f in patients[pdir]]}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"Folder {pdir!r}: multiple files match "
                        f"--select pattern {pattern!r}: "
                        f"{[os.path.basename(f) for f in matches]}. "
                        "Make patterns more specific."
                    )
                selected.append(matches[0])
            file_matrix.append(selected)
    else:
        for pdir in patient_dirs:
            files = patients[pdir]
            if len(files) == 1:
                file_matrix.append([files[0]])
            else:
                raise ValueError(
                    f"Folder {pdir!r} contains {len(files)} files: "
                    f"{[os.path.basename(f) for f in files]}. "
                    "Use --select PATTERN to pick one (or more) per folder. "
                    "Example: --select '*t1*' --select '*t2*' --select '*flair*'"
                )

    return _series_from_file_matrix(file_matrix, load=load, stack=stack)


def _load_nifti_series(path, select=None, *, load="lazy", stack="auto"):
    """Build a lazy ``_NiftiSeries`` from a directory of NIfTI files.

    Walks *path* recursively, groups ``.nii``/``.nii.gz`` by immediate parent
    folder (= patient).  With *select* (a list of fnmatch patterns), picks one
    file per pattern per patient → 5D ``(*vol, P, M)``.  Without *select*,
    requires exactly one NIfTI per patient → 4D ``(*vol, P)``.

    Returns ``(series, spatial_meta)``.
    """
    nib = _nib()

    patients: dict[str, list[str]] = {}
    for root, dirs, files in os.walk(path):
        nii_files = sorted(
            os.path.join(root, f)
            for f in files
            if f.endswith(".nii") or f.endswith(".nii.gz")
        )
        if nii_files:
            patients[root] = nii_files
            # A directory with NIfTI files is the series unit.  Do not treat
            # nested mask/derived-output folders as additional patients.
            dirs[:] = []

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

    return _series_from_file_matrix(file_matrix, load=load, stack=stack)


def load_data_with_meta(filepath, key=None, select=None, *, load="lazy", stack="auto"):
    """Like load_data but also returns spatial metadata for NIfTI files.

    Returns (array, meta_or_None). meta is None for non-NIfTI formats.
    When *filepath* is a directory, loads it as a file series (see
    ``_load_file_series``).
    """
    if os.path.isdir(filepath):
        return _load_file_series(filepath, select=select, load=load, stack=stack)
    if filepath.endswith(".nii") or filepath.endswith(".nii.gz"):
        return _load_nifti_with_meta(filepath)
    return load_data(filepath, key=key), None


def load_data(filepath, key=None):
    if os.path.isdir(filepath):
        series, _meta = _load_file_series(filepath)
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
            series, _ = _load_file_series(fpath)
            return list(series.shape)
        if ext == ".npy":
            arr = np.load(fpath, mmap_mode="r", allow_pickle=False)
            return list(arr.shape)
        if ext in (".nii", ".nii.gz"):
            return list(_nib().load(fpath).shape)
    except Exception:
        pass
    return None
