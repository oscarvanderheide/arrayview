"""Shared compare/diff rendering helpers."""

from __future__ import annotations

import numpy as np

from arrayview._render import (
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
):
    """Return raw diff, display range, colormap, and optional separator mask."""
    idx_tuple = tuple(int(x) for x in indices.split(",")) if isinstance(indices, str) else indices
    ndim_a = len(session_a.shape)
    ndim_b = len(session_b.shape)
    idx_a = idx_tuple[:ndim_a]
    idx_b = idx_tuple[:ndim_b]
    nan_mask = None

    if dim_z >= 0:
        a, nan_mask_a = _render_normalized_mosaic(
            session_a,
            dim_x,
            dim_y,
            dim_z,
            idx_a,
            dr,
            complex_mode,
            log_scale,
        )
        b, nan_mask_b = _render_normalized_mosaic(
            session_b,
            dim_x,
            dim_y,
            dim_z,
            idx_b,
            dr,
            complex_mode,
            log_scale,
        )
        nan_mask = nan_mask_a | nan_mask_b
    else:
        a = _render_normalized(
            session_a,
            dim_x,
            dim_y,
            idx_a,
            dr,
            complex_mode,
            log_scale,
        )
        b = _render_normalized(
            session_b,
            dim_x,
            dim_y,
            idx_b,
            dr,
            complex_mode,
            log_scale,
        )

    b = _resize_normalized_like(b, a)
    if diff_mode == 1:
        raw = a - b
        vmin, vmax = -1.0, 1.0
        colormap = "RdBu_r"
    elif diff_mode == 2:
        raw = np.abs(a - b)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "afmhot"
    else:
        raw = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        raw = np.clip(raw, 0.0, 2.0).astype(np.float32)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "afmhot"

    return raw, vmin, vmax, colormap, nan_mask


def _render_diff_rgba(
    raw: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str,
    nan_mask: np.ndarray | None,
) -> np.ndarray:
    """Map raw diff data to RGBA."""
    if vmax > vmin:
        normalized = np.clip((raw - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(raw)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    if nan_mask is not None and nan_mask.shape == rgba.shape[:2]:
        rgba[nan_mask] = [22, 22, 22, 255]
    return rgba


def _diff_histogram(raw: np.ndarray, bins: int) -> dict[str, object]:
    """Return histogram payload for raw diff data."""
    vmin = float(raw.min())
    vmax = float(raw.max())
    counts, edges = np.histogram(raw.ravel(), bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": edges.tolist(),
        "vmin": vmin,
        "vmax": vmax,
    }
