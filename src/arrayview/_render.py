"""Rendering pipeline: colormaps, LUTs, slice extraction, RGBA, mosaic, RGB, preload."""

import time

import numpy as np

from arrayview._session import (
    COLORMAPS,
    DR_PERCENTILES,
    SESSIONS,
)

# ---------------------------------------------------------------------------
# Colormap LUTs (deferred initialisation)
# ---------------------------------------------------------------------------
LUTS: dict = {}

_mpl_colormaps = None


def _init_luts():
    """Build colourmap LUTs on first use (defers matplotlib + qmricolors import)."""
    global _mpl_colormaps
    if LUTS:
        return  # already initialised
    import qmricolors as _qc  # noqa: F401 — registers lipari, navia
    from matplotlib import colormaps

    _mpl_colormaps = colormaps
    for name in COLORMAPS:
        lut = np.concatenate(
            [
                (colormaps[name](np.arange(256) / 255.0) * 255).astype(np.uint8)[:, :3],
                np.full((256, 1), 255, dtype=np.uint8),
            ],
            axis=1,
        )
        LUTS[name] = lut
    for name in COLORMAPS:
        COLORMAP_GRADIENT_STOPS[name] = _lut_to_gradient_stops(LUTS[name])


def _lut_to_gradient_stops(lut, n=32):
    indices = np.linspace(0, 255, n, dtype=int)
    return [[int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])] for i in indices]


COLORMAP_GRADIENT_STOPS: dict = {}
COMPLEX_MODES = ["mag", "phase", "real", "imag"]
REAL_MODES = ["real", "mag"]
OVERLAY_COLOR = np.array([255, 80, 80], dtype=np.float32)
OVERLAY_ALPHA = np.float32(0.45)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def mosaic_shape(batch):
    mshape = [int(batch**0.5), batch // int(batch**0.5)]
    while mshape[0] * mshape[1] < batch:
        mshape[1] += 1
    if (mshape[0] - 1) * (mshape[1] + 1) == batch:
        mshape[0] -= 1
        mshape[1] += 1
    return tuple(mshape)


# ---------------------------------------------------------------------------
# Slice extraction & display preparation
# ---------------------------------------------------------------------------


def _compute_vmin_vmax(session, data, dr, complex_mode=0):
    if complex_mode == 1 and np.iscomplexobj(session.data):
        return (-float(np.pi), float(np.pi))
    if complex_mode == 0 and len(session.shape) <= 3 and dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
        if vmin != vmax:
            return vmin, vmax
    pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
    return float(np.percentile(data, pct_lo)), float(np.percentile(data, pct_hi))


def extract_slice(session, dim_x, dim_y, idx_list):
    key = (dim_x, dim_y, tuple(idx_list))
    if key in session.raw_cache:
        session.raw_cache.move_to_end(key)
        return session.raw_cache[key]

    slicer = [
        slice(None) if i in (dim_x, dim_y) else idx_list[i]
        for i in range(len(session.shape))
    ]
    extracted = np.array(session.data[tuple(slicer)])
    if dim_x < dim_y:
        extracted = extracted.T
    if np.iscomplexobj(extracted):
        result = np.nan_to_num(extracted).astype(np.complex64)
    else:
        result = np.nan_to_num(extracted).astype(np.float32)

    session.raw_cache[key] = result
    session._raw_bytes += result.nbytes
    while session._raw_bytes > session.RAW_CACHE_BYTES and session.raw_cache:
        _, v = session.raw_cache.popitem(last=False)
        session._raw_bytes -= v.nbytes
    return result


def apply_complex_mode(raw, complex_mode):
    if np.iscomplexobj(raw):
        if complex_mode == 1:
            result = np.angle(raw)
        elif complex_mode == 2:
            result = raw.real.copy()
        elif complex_mode == 3:
            result = raw.imag.copy()
        else:
            result = np.abs(raw)
    else:
        result = np.abs(raw) if complex_mode == 1 else raw
    return np.nan_to_num(result).astype(np.float32)


def _compute_otsu_threshold(data) -> float:
    """Compute Otsu's threshold on the absolute values of non-zero, finite elements."""
    flat = np.abs(np.asarray(data, dtype=np.float64).ravel())
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) < 10:
        return 0.0
    if len(flat) > 1_000_000:
        rng = np.random.default_rng(42)
        flat = rng.choice(flat, 1_000_000, replace=False)
    hist, edges = np.histogram(flat, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    total = float(hist.sum())
    if total == 0:
        return 0.0
    w_b = np.cumsum(hist) / total
    w_f = 1.0 - w_b
    mu_cum = np.cumsum(hist * centers)
    mu_b = np.where(w_b > 0, mu_cum / np.maximum(w_b * total, 1e-10), 0.0)
    mu_f = np.where(
        w_f > 0, (mu_cum[-1] / total - mu_b * w_b) / np.maximum(w_f, 1e-10), 0.0
    )
    sigma_b_sq = w_b * w_f * (mu_b - mu_f) ** 2
    return float(centers[int(np.argmax(sigma_b_sq))])


def _prepare_display(
    session, raw, complex_mode, dr, log_scale, vmin_override=None, vmax_override=None
):
    data = apply_complex_mode(raw, complex_mode)
    if vmin_override is not None and vmax_override is not None:
        if log_scale:
            data = np.log1p(np.abs(data)).astype(np.float32)
        return data, vmin_override, vmax_override
    if log_scale:
        data = np.log1p(np.abs(data)).astype(np.float32)
        if complex_mode == 0 and len(session.shape) <= 3 and dr in session.global_stats:
            raw_vmin, raw_vmax = session.global_stats[dr]
            vmin = float(np.log1p(abs(raw_vmin)))
            vmax = float(np.log1p(abs(raw_vmax)))
            if vmin == vmax:
                pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
                vmin = float(np.percentile(data, pct_lo))
                vmax = float(np.percentile(data, pct_hi))
        else:
            pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
            vmin = float(np.percentile(data, pct_lo))
            vmax = float(np.percentile(data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, data, dr, complex_mode)
    return data, vmin, vmax


def _ensure_lut(name: str) -> bool:
    """Ensure name is in LUTS. Returns True if valid."""
    _init_luts()
    if name in LUTS:
        return True
    try:
        cmap = _mpl_colormaps[name]
    except (KeyError, TypeError):
        return False
    rgba = (cmap(np.arange(256) / 255.0) * 255).astype(np.uint8)
    lut = np.concatenate([rgba[:, :3], np.full((256, 1), 255, dtype=np.uint8)], axis=1)
    LUTS[name] = lut
    COLORMAP_GRADIENT_STOPS[name] = _lut_to_gradient_stops(lut)
    return True


# ---------------------------------------------------------------------------
# RGBA rendering
# ---------------------------------------------------------------------------


def apply_colormap_rgba(
    session,
    raw,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    data, vmin, vmax = _prepare_display(
        session,
        raw,
        complex_mode,
        dr,
        log_scale,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
    )
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    mask_thr = getattr(session, "mask_threshold", 0.0)
    if mask_thr > 0:
        abs_raw = np.abs(raw)
        transparent = abs_raw < mask_thr
    elif vmin > 0 and vmax > vmin:
        transparent = data < vmin
    else:
        transparent = data == np.float32(0)
    if transparent.any():
        rgba = rgba.copy()
        rgba[transparent, 3] = 0
    return rgba


# ---------------------------------------------------------------------------
# RGB mode
# ---------------------------------------------------------------------------


def _detect_rgb_axis(shape: tuple) -> int:
    """Return the axis index for the RGB/RGBA channel dimension."""
    if len(shape) < 2:
        raise ValueError(
            f"Array must have at least 2 dimensions for RGB mode, got shape {shape}"
        )
    last = shape[-1]
    first = shape[0]
    if last in (3, 4):
        return len(shape) - 1
    if first in (3, 4):
        return 0
    raise ValueError(
        f"RGB mode requires the first or last dimension to have size 3 or 4 "
        f"(for RGB or RGBA). Got shape {shape} — first={first}, last={last}."
    )


def _setup_rgb(session) -> None:
    """Detect and store the RGB channel axis on a session."""
    axis = _detect_rgb_axis(session.shape)
    session.rgb_axis = axis
    session.spatial_shape = tuple(s for i, s in enumerate(session.shape) if i != axis)


def render_rgb_rgba(session, dim_x: int, dim_y: int, idx_list: list) -> np.ndarray:
    """Render an RGB/RGBA session slice to a H*W*4 uint8 RGBA array."""
    rgb_axis = session.rgb_axis
    ndim_actual = len(session.shape)

    cache_key = ("rgb", dim_x, dim_y, tuple(idx_list))
    if cache_key in session.rgba_cache:
        session.rgba_cache.move_to_end(cache_key)
        return session.rgba_cache[cache_key]

    if rgb_axis == 0:
        actual_dim_x = dim_x + 1
        actual_dim_y = dim_y + 1
    else:
        actual_dim_x = dim_x
        actual_dim_y = dim_y

    slicer: list = []
    spatial_i = 0
    for actual_i in range(ndim_actual):
        if actual_i == rgb_axis:
            slicer.append(slice(None))
        else:
            if actual_i in (actual_dim_x, actual_dim_y):
                slicer.append(slice(None))
            else:
                slicer.append(int(idx_list[spatial_i]))
            spatial_i += 1

    arr = np.array(session.data[tuple(slicer)])

    free_actual = sorted([actual_dim_x, actual_dim_y, rgb_axis])
    pos_dx = free_actual.index(actual_dim_x)
    pos_dy = free_actual.index(actual_dim_y)
    pos_rgb = free_actual.index(rgb_axis)

    arr = arr.transpose(pos_dy, pos_dx, pos_rgb)  # (H, W, C)
    arr = np.nan_to_num(arr)

    if arr.dtype.kind == "f":
        if arr.max() > 1.5:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w, c = arr.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = arr[:, :, :3]
    rgba[:, :, 3] = arr[:, :, 3] if c == 4 else np.uint8(255)

    session.rgba_cache[cache_key] = rgba
    session._rgba_bytes += rgba.nbytes
    while session._rgba_bytes > session.RGBA_CACHE_BYTES and session.rgba_cache:
        _, v = session.rgba_cache.popitem(last=False)
        session._rgba_bytes -= v.nbytes

    return rgba


def render_rgba(
    session,
    dim_x,
    dim_y,
    idx_tuple,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    has_override = vmin_override is not None and vmax_override is not None
    if not has_override:
        key = (
            dim_x,
            dim_y,
            idx_tuple,
            colormap,
            dr,
            complex_mode,
            log_scale,
            getattr(session, "mask_threshold", 0.0),
        )
        if key in session.rgba_cache:
            session.rgba_cache.move_to_end(key)
            return session.rgba_cache[key]
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    rgba = apply_colormap_rgba(
        session,
        raw,
        colormap,
        dr,
        complex_mode,
        log_scale,
        vmin_override=vmin_override,
        vmax_override=vmax_override,
    )
    if not has_override:
        session.rgba_cache[key] = rgba
        session._rgba_bytes += rgba.nbytes
        while session._rgba_bytes > session.RGBA_CACHE_BYTES and session.rgba_cache:
            _, v = session.rgba_cache.popitem(last=False)
            session._rgba_bytes -= v.nbytes
    return rgba


# ---------------------------------------------------------------------------
# Overlay compositing
# ---------------------------------------------------------------------------


def _extract_overlay_mask(
    overlay_sid: str | None,
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    """Return a boolean mask slice for overlay compositing, or None."""
    if not overlay_sid:
        return None
    ov_session = SESSIONS.get(str(overlay_sid))
    if ov_session is None:
        return None

    ov_ndim = ov_session.data.ndim
    if dim_x >= ov_ndim or dim_y >= ov_ndim:
        return None

    idx_candidates: list[tuple[int, ...]] = [
        tuple(int(idx_tuple[i]) if i < len(idx_tuple) else 0 for i in range(ov_ndim))
    ]
    if len(idx_tuple) >= ov_ndim:
        trailing = tuple(int(v) for v in idx_tuple[-ov_ndim:])
        if trailing != idx_candidates[0]:
            idx_candidates.append(trailing)

    for ov_idx in idx_candidates:
        try:
            ov_raw = extract_slice(ov_session, dim_x, dim_y, list(ov_idx))
        except Exception:
            continue
        if ov_raw.shape != expected_shape:
            continue
        mask = np.isfinite(ov_raw) & (ov_raw > 0.5)
        if mask.any():
            return mask
    return None


def _composite_overlay_mask(
    rgba: np.ndarray, mask: np.ndarray | None, alpha: float = float(OVERLAY_ALPHA)
) -> np.ndarray:
    """Alpha-composite a red segmentation mask on top of an RGBA frame."""
    if mask is None:
        return rgba

    out = rgba.copy()
    base_rgb = out[mask, :3].astype(np.float32) / 255.0
    base_a = out[mask, 3].astype(np.float32) / 255.0

    ov_a = np.float32(alpha)
    out_a = ov_a + base_a * (1.0 - ov_a)
    denom = np.maximum(out_a, 1e-6)

    ov_rgb = OVERLAY_COLOR / 255.0
    out_rgb = (ov_a * ov_rgb + (1.0 - ov_a) * base_a[:, None] * base_rgb) / denom[
        :, None
    ]

    out[mask, :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[mask, 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Mosaic rendering
# ---------------------------------------------------------------------------


def render_mosaic(
    session,
    dim_x,
    dim_y,
    dim_z,
    idx_tuple,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
):
    idx_norm = list(idx_tuple)
    idx_norm[dim_z] = 0
    key = (dim_x, dim_y, dim_z, tuple(idx_norm), colormap, dr, complex_mode, log_scale)
    if key in session.mosaic_cache:
        session.mosaic_cache.move_to_end(key)
        return session.mosaic_cache[key]

    n = session.shape[dim_z]
    frames_raw = [
        extract_slice(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_tuple[j] for j in range(len(session.shape))],
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)

    if log_scale:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, all_data, dr, complex_mode)

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    GAP = 2
    total_h = rows * H + (rows - 1) * GAP
    total_w = cols * W + (cols - 1) * GAP
    grid = np.full((total_h, total_w), np.nan, dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        r0, c0 = r * (H + GAP), c * (W + GAP)
        grid[r0 : r0 + H, c0 : c0 + W] = all_data[k]

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, vmin, grid)
    if vmax > vmin:
        normalized = np.clip((filled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(filled)

    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    rgba[nan_mask] = [22, 22, 22, 255]
    session.mosaic_cache[key] = rgba
    session._mosaic_bytes += rgba.nbytes
    while session._mosaic_bytes > session.MOSAIC_CACHE_BYTES and session.mosaic_cache:
        _, v = session.mosaic_cache.popitem(last=False)
        session._mosaic_bytes -= v.nbytes
    return rgba


# ---------------------------------------------------------------------------
# Preload (background cache warming)
# ---------------------------------------------------------------------------


def _run_preload(
    session,
    gen,
    dim_x,
    dim_y,
    idx_list,
    colormap,
    dr,
    slice_dim,
    dim_z=-1,
    complex_mode=0,
    log_scale=False,
):
    shape = session.spatial_shape if session.rgb_axis is not None else session.shape
    n = shape[slice_dim]
    H = shape[dim_y]
    W = shape[dim_x]
    if dim_z >= 0:
        nz = shape[dim_z]
        mrows, mcols = mosaic_shape(nz)
        size_bytes = n * (mrows * H) * (mcols * W) * 4
    else:
        size_bytes = n * H * W * 4

    with session.preload_lock:
        session.preload_total = n
        session.preload_done = 0
        if size_bytes > 500 * 1024 * 1024:
            session.preload_skipped = True
            return
        session.preload_skipped = False

    for i in range(n):
        if session.preload_gen != gen:
            return
        idx = list(idx_list)
        idx[slice_dim] = i
        if dim_z >= 0:
            render_mosaic(
                session,
                dim_x,
                dim_y,
                dim_z,
                tuple(idx),
                colormap,
                dr,
                complex_mode,
                log_scale,
            )
        else:
            if session.rgb_axis is not None:
                render_rgb_rgba(session, dim_x, dim_y, list(idx))
            else:
                render_rgba(
                    session,
                    dim_x,
                    dim_y,
                    tuple(idx),
                    colormap,
                    dr,
                    complex_mode,
                    log_scale,
                )
        with session.preload_lock:
            session.preload_done = i + 1
        time.sleep(0.005)


__all__ = [
    # LUTs / colormaps
    "LUTS",
    "_mpl_colormaps",
    "_init_luts",
    "_lut_to_gradient_stops",
    "COLORMAP_GRADIENT_STOPS",
    "_ensure_lut",
    "apply_colormap_rgba",
    # Constants
    "COMPLEX_MODES",
    "REAL_MODES",
    "OVERLAY_COLOR",
    "OVERLAY_ALPHA",
    # Utility
    "mosaic_shape",
    # Slice extraction
    "_compute_vmin_vmax",
    "extract_slice",
    "apply_complex_mode",
    "_compute_otsu_threshold",
    "_prepare_display",
    # RGB
    "_detect_rgb_axis",
    "_setup_rgb",
    "render_rgb_rgba",
    # Core render
    "render_rgba",
    # Overlay
    "_extract_overlay_mask",
    "_composite_overlay_mask",
    # Mosaic
    "render_mosaic",
    # Preload
    "_run_preload",
]
