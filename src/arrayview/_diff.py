"""Shared compare/diff rendering helpers."""

from __future__ import annotations

import numpy as np

from arrayview._render import (
    lut_rgba,
    LUTS,
    _build_mosaic_grid,
    _compute_vmin_vmax,
    _ensure_lut,
    _prepare_display,
    apply_complex_mode,
    extract_slice,
)


from arrayview._imaging import ensure_image as _pil_image


def _render_normalized(session, dim_x, dim_y, idx_tuple, dr, complex_mode, log_scale):
    """Extract and normalize a slice to float32 [0, 1]."""
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    return normalized.astype(np.float32)


def _render_normalized_mosaic(
    session,
    dim_x,
    dim_y,
    dim_z,
    idx_tuple,
    dr,
    complex_mode,
    log_scale,
):
    """Return normalized mosaic grid and separator mask."""
    n = session.shape[dim_z]
    idx_list = list(idx_tuple)
    frames_raw = [
        extract_slice(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_list[j] for j in range(len(session.shape))],
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)
    if log_scale:
        vmin = float(np.percentile(all_data, 1))
        vmax = float(np.percentile(all_data, 99))
    else:
        vmin, vmax = _compute_vmin_vmax(session, all_data, dr, complex_mode)

    grid, _, _ = _build_mosaic_grid(all_data, n)

    nan_mask = np.isnan(grid)
    if vmax > vmin:
        normalized = np.clip(
            np.where(nan_mask, 0.0, (grid - vmin) / (vmax - vmin)),
            0,
            1,
        )
    else:
        normalized = np.zeros_like(grid)
    return normalized.astype(np.float32), nan_mask


def _resize_normalized_like(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize normalized source data to target shape."""
    if source.shape == target.shape:
        return source
    image = _pil_image().fromarray((source * 255).astype(np.uint8), mode="L")
    image = image.resize((target.shape[1], target.shape[0]), _pil_image().BILINEAR)
    return np.array(image, dtype=np.float32) / 255.0


def _render_values(session, dim_x, dim_y, idx_tuple, complex_mode, log_scale):
    """Extract a slice in the array's own units (complex/log applied, no rescale)."""
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    if log_scale:
        data = np.log1p(np.abs(data))
    return np.asarray(data, dtype=np.float32)


def _render_values_mosaic(session, dim_x, dim_y, dim_z, idx_tuple, complex_mode, log_scale):
    """Mosaic of every dim_z slice in the array's own units, plus separator mask."""
    idx_list = list(idx_tuple)
    frames = [
        _render_values(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_list[j] for j in range(len(session.shape))],
            complex_mode,
            log_scale,
        )
        for i in range(session.shape[dim_z])
    ]
    all_data = np.stack(frames)
    grid, _, _ = _build_mosaic_grid(all_data, len(frames))
    return grid.astype(np.float32), np.isnan(grid)


def _resize_values_like(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize float source data to target shape without quantising it."""
    if source.shape == target.shape:
        return source
    image = _pil_image().fromarray(np.nan_to_num(source).astype(np.float32), mode="F")
    image = image.resize((target.shape[1], target.shape[0]), _pil_image().BILINEAR)
    return np.array(image, dtype=np.float32)


def _compute_diff(
    session_a,
    session_b,
    dim_x,
    dim_y,
    indices,
    dim_z,
    dr,
    complex_mode,
    log_scale,
    diff_mode,
    indices_a=None,
    indices_b=None,
    stats_out=None,
):
    """Return raw diff, display range, colormap, and optional separator mask.

    When *stats_out* is a dict it is filled with summary numbers for the
    compared view (see `_diff_summary`).
    """
    def _parse_indices(raw_indices):
        if raw_indices is None:
            return ()
        if isinstance(raw_indices, str):
            return tuple(int(x) for x in raw_indices.split(",") if x != "")
        return tuple(int(x) for x in raw_indices)

    idx_tuple = _parse_indices(indices)
    ndim_a = len(session_a.shape)
    ndim_b = len(session_b.shape)
    idx_a = _parse_indices(indices_a)[:ndim_a] if indices_a is not None else idx_tuple[:ndim_a]
    idx_b = _parse_indices(indices_b)[:ndim_b] if indices_b is not None else idx_tuple[:ndim_b]
    nan_mask = None

    # True A−B in the arrays' own units. Each side used to be rescaled to its
    # own per-slice display range first, which cancelled real differences
    # (B = 0.9·A showed as zero) and made the map flicker while scrolling.
    if dim_z >= 0:
        a, nan_mask_a = _render_values_mosaic(
            session_a, dim_x, dim_y, dim_z, idx_a, complex_mode, log_scale
        )
        b, nan_mask_b = _render_values_mosaic(
            session_b, dim_x, dim_y, dim_z, idx_b, complex_mode, log_scale
        )
        b = _resize_values_like(b, a)
        if nan_mask_b.shape == nan_mask_a.shape:
            nan_mask = nan_mask_a | nan_mask_b
        else:
            nan_mask = nan_mask_a
    else:
        a = _render_values(session_a, dim_x, dim_y, idx_a, complex_mode, log_scale)
        b = _render_values(session_b, dim_x, dim_y, idx_b, complex_mode, log_scale)
        b = _resize_values_like(b, a)
    if stats_out is not None:
        stats_out.update(_diff_summary(a, b, nan_mask))
    if diff_mode == 1:
        raw = a - b
        lo = float(np.nanpercentile(raw, 1))
        hi = float(np.nanpercentile(raw, 99))
        vmax = max(abs(lo), abs(hi)) or 1.0
        vmin = -vmax
        colormap = "icefire"
    elif diff_mode == 2:
        raw = np.abs(a - b)
        vmax = float(np.nanmax(raw)) or 1.0
        vmin = 0.0
        colormap = "afmhot"
    else:
        raw = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        raw = np.clip(raw, 0.0, 2.0).astype(np.float32)
        vmax = float(np.nanmax(raw)) or 1.0
        vmin = 0.0
        colormap = "afmhot"

    return raw, vmin, vmax, colormap, nan_mask


def _diff_summary(a: np.ndarray, b: np.ndarray, nan_mask: np.ndarray | None) -> dict[str, float]:
    """RMS of A−B, largest |A−B|, and RMS of A−B relative to the RMS of A.

    Pixels that are not finite in either array (or are mosaic separators)
    are left out. Empty or all-zero inputs give NaN for what is undefined.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if nan_mask is not None and nan_mask.shape == valid.shape:
        valid &= ~nan_mask
    if not valid.any():
        return {"rmse": float("nan"), "max_abs": float("nan"), "rel": float("nan")}
    d = a[valid] - b[valid]
    rmse = float(np.sqrt(np.mean(d * d)))
    rms_a = float(np.sqrt(np.mean(a[valid] ** 2)))
    return {
        "rmse": rmse,
        "max_abs": float(np.max(np.abs(d))),
        "rel": rmse / rms_a if rms_a > 0 else float("nan"),
    }


def _render_diff_rgba(
    raw: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str,
    nan_mask: np.ndarray | None,
) -> np.ndarray:
    """Map raw diff data to RGBA."""
    if vmax > vmin:
        normalized = np.clip(np.nan_to_num((raw - vmin) / (vmax - vmin)), 0, 1)
    else:
        normalized = np.zeros_like(raw)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut_rgba(lut, (normalized * 255).astype(np.uint8))
    if nan_mask is not None and nan_mask.shape == rgba.shape[:2]:
        rgba[nan_mask] = [22, 22, 22, 255]
    return rgba


def _diff_histogram(raw: np.ndarray, bins: int) -> dict[str, object]:
    """Return histogram payload for raw diff data."""
    raw = raw[np.isfinite(raw)]
    if raw.size == 0:
        raw = np.zeros(1, dtype=np.float32)
    vmin = float(raw.min())
    vmax = float(raw.max())
    counts, edges = np.histogram(raw, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "vmin": vmin,
        "vmax": vmax,
    }
