"""Shared metadata, histogram, and pixel helpers."""

from __future__ import annotations

import itertools

import numpy as np

from arrayview import _session as _session_mod
from arrayview._render import apply_complex_mode, extract_slice
from arrayview._synthetic_mri import qmri_display_slice
from arrayview._vectorfield import _vfield_n_times


def _visible_shape(session) -> list[int]:
    """Return the display shape for normal or RGB sessions."""
    shape = session.spatial_shape if session.rgb_axis is not None else session.shape
    return [int(s) for s in shape]


def _build_metadata(session) -> dict:
    """Build metadata for HTTP and stdio transports."""
    target_shape = session.spatial_shape if session.rgb_axis is not None else session.shape
    meta = {
        "shape": [int(s) for s in target_shape],
        "is_complex": bool(np.iscomplexobj(session.data)),
        "name": session.name,
        "has_vectorfield": session.vfield is not None,
        "vfield_n_times": _vfield_n_times(session),
        "is_rgb": session.rgb_axis is not None,
        "has_source_file": bool(getattr(session, "filepath", None)),
    }
    default_dims = _session_mod._startup_dims_for_data(session.data, target_shape)
    if default_dims is not None:
        meta["default_dims"] = [int(default_dims[0]), int(default_dims[1])]
    if getattr(session, "spatial_meta", None) is not None:
        sm = session.spatial_meta
        meta["spatial_meta"] = {
            "voxel_sizes": list(sm["voxel_sizes"]),
            "axis_labels": list(sm["axis_labels"]),
            "is_oblique": bool(sm["is_oblique"]),
        }
        meta["ras_resample_active"] = bool(
            getattr(session, "ras_resample_active", False)
        )
    if getattr(session, "npz_keys", None):
        meta["npz_keys"] = session.npz_keys
    return meta


def _session_summary(session) -> dict:
    """Build one `/sessions` entry."""
    dtype_str = str(getattr(session.data, "dtype", "unknown"))
    return {
        "sid": session.sid,
        "name": session.name,
        "shape": [int(x) for x in session.shape],
        "filepath": session.filepath,
        "dtype": dtype_str,
        "estimated_mem": session._estimated_mem,
    }


def _safe_float(v) -> float | None:
    """Return a finite JSON-safe float."""
    f = float(v)
    return f if np.isfinite(f) else None


def _histogram_payload(finite: np.ndarray, bins: int) -> dict:
    """Build a histogram response from finite values."""
    if finite.size == 0:
        return {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        return {
            "counts": [int(finite.size)],
            "edges": [vmin, vmax + 1e-9],
            "vmin": vmin,
            "vmax": vmax,
        }
    bins = max(8, min(int(bins), 512))
    counts, edges = np.histogram(finite, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": [float(e) for e in edges],
        "vmin": vmin,
        "vmax": vmax,
    }


def _slice_histogram(
    session,
    dim_x: int,
    dim_y: int,
    indices,
    complex_mode: int = 0,
    bins: int = 128,
    qmri_role: str = "",
) -> dict:
    """Return a histogram for one 2-D slice."""
    idx_tuple = (
        tuple(int(v) for v in indices.split(","))
        if isinstance(indices, str)
        else tuple(indices)
    )
    raw = (
        qmri_display_slice(session, dim_x, dim_y, list(idx_tuple), qmri_role)
        if qmri_role
        else extract_slice(session, dim_x, dim_y, list(idx_tuple))
    )
    data = apply_complex_mode(raw, complex_mode)
    finite = data.ravel()
    finite = finite[np.isfinite(finite)]
    return _histogram_payload(finite, bins)


def _parse_fixed_indices(fixed_indices: str) -> dict[int, int]:
    fixed: dict[int, int] = {}
    if fixed_indices:
        for pair in fixed_indices.split(","):
            if ":" in pair:
                d, v = pair.split(":", 1)
                fixed[int(d)] = int(v)
    return fixed


def _resolve_aggregation_dims(
    session,
    dim_x: int,
    dim_y: int,
    scroll_dim: int = -1,
    scroll_dims: str = "",
) -> list[int]:
    agg_dims: list[int] = []
    if scroll_dims:
        for tok in scroll_dims.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                d = int(tok)
            except ValueError:
                continue
            if (
                0 <= d < len(session.shape)
                and d != dim_x
                and d != dim_y
                and d not in agg_dims
            ):
                agg_dims.append(d)
    if not agg_dims and scroll_dim >= 0 and scroll_dim != dim_x and scroll_dim != dim_y:
        agg_dims = [scroll_dim]
    return agg_dims


def _volume_histogram(
    session,
    dim_x: int,
    dim_y: int,
    scroll_dim: int = -1,
    scroll_dims: str = "",
    fixed_indices: str = "",
    complex_mode: int = 0,
    bins: int = 64,
    qmri_role: str = "",
) -> dict:
    """Return a sampled volume histogram."""
    fixed = _parse_fixed_indices(fixed_indices)
    agg_dims = _resolve_aggregation_dims(
        session, dim_x, dim_y, scroll_dim, scroll_dims
    )

    cache_key = (
        dim_x,
        dim_y,
        tuple(agg_dims),
        tuple(sorted(fixed.items())),
        complex_mode,
        qmri_role,
    )
    if not hasattr(session, "_volume_hist_cache"):
        session._volume_hist_cache = {}
    cached = session._volume_hist_cache.get(cache_key)
    if cached is not None and cached.get("_data_version") == session.data_version:
        return cached["result"]

    max_samples = 16
    if not agg_dims:
        sample_combos = [tuple()]
    else:
        per_dim_counts = [session.shape[d] for d in agg_dims]
        k = len(agg_dims)
        per_dim_target = max(1, int(round(max_samples ** (1.0 / k))))
        sample_per_dim: list[list[int]] = []
        for n in per_dim_counts:
            m = min(n, per_dim_target)
            if n <= m:
                sample_per_dim.append(list(range(n)))
            else:
                step = n / m
                sample_per_dim.append([int(i * step) for i in range(m)])
        sample_combos = list(
            itertools.islice(itertools.product(*sample_per_dim), max_samples)
        )

    pixels = []
    for combo in sample_combos:
        idx_list = [s // 2 for s in session.shape]
        for d, si in zip(agg_dims, combo):
            idx_list[d] = si
        for d, v in fixed.items():
            idx_list[d] = v
        raw = (
            qmri_display_slice(session, dim_x, dim_y, idx_list, qmri_role)
            if qmri_role
            else extract_slice(session, dim_x, dim_y, idx_list)
        )
        data = apply_complex_mode(raw, complex_mode)
        finite = data.ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size > 0:
            pixels.append(finite)

    if pixels:
        result = _histogram_payload(np.concatenate(pixels), bins)
    else:
        result = {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
    session._volume_hist_cache[cache_key] = {
        "_data_version": session.data_version,
        "result": result,
    }
    return result


def _lebesgue_slice(
    session,
    dim_x: int,
    dim_y: int,
    indices,
    complex_mode: int = 0,
    log_scale: bool = False,
) -> np.ndarray:
    """Return float32 slice data for Lebesgue mode."""
    idx_tuple = (
        tuple(int(v) for v in indices.split(","))
        if isinstance(indices, str)
        else tuple(indices)
    )
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode).astype(np.float32)
    if log_scale:
        data = np.log10(np.abs(data) + 1).astype(np.float32)
    return data


def _pixel_value(
    session,
    dim_x: int,
    dim_y: int,
    indices,
    px: int,
    py: int,
    complex_mode: int = 0,
    qmri_role: str = "",
) -> float | None:
    """Return a finite pixel value or None."""
    if session.rgb_axis is not None:
        return None
    idx_tuple = (
        tuple(int(v) for v in indices.split(","))
        if isinstance(indices, str)
        else tuple(indices)
    )
    raw = (
        qmri_display_slice(session, dim_x, dim_y, list(idx_tuple), qmri_role)
        if qmri_role
        else extract_slice(session, dim_x, dim_y, list(idx_tuple))
    )
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    if 0 <= py < h and 0 <= px < w:
        return _safe_float(data[py, px])
    return None
