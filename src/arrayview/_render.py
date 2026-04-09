"""Rendering pipeline: colormaps, LUTs, slice extraction, RGBA, mosaic, RGB, preload."""

import time

import numpy as np

from arrayview._session import (
    COLORMAPS,
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
    from matplotlib.colors import LinearSegmentedColormap

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

    # Black-center diverging colormap (blue → black → red) for diff A−B
    _bkdiv = LinearSegmentedColormap.from_list(
        "RdBu_r_black",
        [(0.0, (0.0, 0.3, 1.0)), (0.5, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.2, 0.0))],
    )
    _bkdiv_lut = np.concatenate(
        [
            (_bkdiv(np.arange(256) / 255.0) * 255).astype(np.uint8)[:, :3],
            np.full((256, 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )
    LUTS["RdBu_r_black"] = _bkdiv_lut
    COLORMAP_GRADIENT_STOPS["RdBu_r_black"] = _lut_to_gradient_stops(_bkdiv_lut)


def _lut_to_gradient_stops(lut, n=32):
    indices = np.linspace(0, 255, n, dtype=int)
    return [[int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])] for i in indices]


COLORMAP_GRADIENT_STOPS: dict = {}
COMPLEX_MODES = ["mag", "phase", "real", "imag"]
REAL_MODES = ["real", "mag"]
OVERLAY_COLOR = np.array([255, 80, 80], dtype=np.float32)
OVERLAY_ALPHA = np.float32(0.45)

# Categorical colors for integer segmentation labels (label 1..N → index 0..N-1)
LABEL_COLORS = np.array(
    [
        [255,  80,  80],  # 1  red
        [ 80, 160, 255],  # 2  blue
        [ 80, 210,  80],  # 3  green
        [255, 175,  50],  # 4  orange
        [185,  80, 255],  # 5  purple
        [255, 100, 190],  # 6  pink
        [ 60, 210, 195],  # 7  teal
        [240, 220,  50],  # 8  yellow
        [160, 110,  60],  # 9  brown
        [180, 180, 180],  # 10 gray
    ],
    dtype=np.float32,
)


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


def _compute_vmin_vmax(session, data, dr=0, complex_mode=0):
    if complex_mode == 1 and np.iscomplexobj(session.data):
        return (-float(np.pi), float(np.pi))
    return float(np.percentile(data, 1)), float(np.percentile(data, 99))


def extract_slice(session, dim_x, dim_y, idx_list):
    # Mask out indices along the displayed dims — they're slice(None) in the
    # slicer below, so the extracted data doesn't depend on them.
    key_idx = tuple(None if i in (dim_x, dim_y) else idx_list[i] for i in range(len(idx_list)))
    key = (dim_x, dim_y, key_idx)
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
    # scipy.io.loadmat returns complex arrays as structured dtypes; convert here.
    if extracted.dtype.names and "real" in extracted.dtype.names and "imag" in extracted.dtype.names:
        extracted = (extracted["real"] + 1j * extracted["imag"]).astype(np.complex64)
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


PROJECTION_OPS = {
    1: ("max", np.max),
    2: ("min", np.min),
    3: ("mean", np.mean),
    4: ("std", np.std),
    5: ("sos", None),  # sum of squares — custom implementation
    6: ("sum", np.sum),
}


def extract_projection(session, dim_x, dim_y, idx_list, proj_dim, proj_mode):
    """Extract a 2D projection by collapsing proj_dim with the given operation.

    proj_mode: 1=max, 2=min, 3=mean, 4=std, 5=sos (sum of squares)
    """
    # Mask out indices along the displayed/projected dims — they're slice(None).
    key_idx = tuple(
        None if i in (dim_x, dim_y, proj_dim) else idx_list[i]
        for i in range(len(idx_list))
    )
    key = ("proj", dim_x, dim_y, key_idx, proj_dim, proj_mode)
    if key in session.raw_cache:
        session.raw_cache.move_to_end(key)
        return session.raw_cache[key]

    # Build slicer: display dims + projection dim get slice(None), rest use idx
    slicer = []
    for i in range(len(session.shape)):
        if i in (dim_x, dim_y, proj_dim):
            slicer.append(slice(None))
        else:
            slicer.append(idx_list[i])
    vol = np.array(session.data[tuple(slicer)])

    # Determine the axis of proj_dim in the extracted sub-array
    kept_dims = sorted([dim_x, dim_y, proj_dim])
    proj_axis = kept_dims.index(proj_dim)

    if proj_mode == 5:  # SOS — sum of magnitude-squared
        if np.iscomplexobj(vol):
            result = np.sum((vol * np.conj(vol)).real, axis=proj_axis).astype(np.float32)
        else:
            result = np.sum(vol.astype(np.float64) ** 2, axis=proj_axis).astype(np.float32)
    elif proj_mode == 1 and np.iscomplexobj(vol):  # max by magnitude
        mag = np.abs(vol)
        idx = np.expand_dims(np.argmax(mag, axis=proj_axis), axis=proj_axis)
        result = np.take_along_axis(vol, idx, axis=proj_axis).squeeze(axis=proj_axis)
    elif proj_mode == 2 and np.iscomplexobj(vol):  # min by magnitude
        mag = np.abs(vol)
        idx = np.expand_dims(np.argmin(mag, axis=proj_axis), axis=proj_axis)
        result = np.take_along_axis(vol, idx, axis=proj_axis).squeeze(axis=proj_axis)
    else:
        _, op = PROJECTION_OPS[proj_mode]
        result = op(vol, axis=proj_axis)

    # Transpose to match extract_slice convention (dim_x < dim_y → transpose)
    if dim_x < dim_y:
        result = result.T

    if np.iscomplexobj(result):
        result = np.nan_to_num(result).astype(np.complex64)
    else:
        result = np.nan_to_num(result).astype(np.float32)
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
        vmin = float(np.percentile(data, 1))
        vmax = float(np.percentile(data, 99))
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
    alpha_on = getattr(session, "alpha_level", 0) > 0
    if alpha_on:
        if vmax > vmin:
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
            getattr(session, "alpha_level", 0),
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


def render_projection_rgba(
    session,
    dim_x,
    dim_y,
    idx_tuple,
    proj_dim,
    proj_mode,
    colormap,
    dr,
    complex_mode=0,
    log_scale=False,
    vmin_override=None,
    vmax_override=None,
):
    """Render a statistical projection (max/min/mean/std/sos) to RGBA."""
    raw = extract_projection(
        session, dim_x, dim_y, list(idx_tuple), proj_dim, proj_mode
    )
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
    """Return the raw overlay slice (float32 H×W) for compositing, or None.

    Caller should inspect the overlay session dtype to decide label vs heatmap mode.
    """
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
        if np.any(np.isfinite(ov_raw) & (ov_raw != 0)):
            return ov_raw.astype(np.float32)
    return None


def _overlay_is_label_map(overlay_sid: str | None, ov_raw: np.ndarray | None = None) -> bool:
    """Return True if the overlay should be rendered as discrete integer labels.

    Returns False (heatmap) for float arrays or integer arrays with more than
    16 unique non-zero values in the current slice (continuous segmentation scores).
    """
    if not overlay_sid:
        return False
    ov_session = SESSIONS.get(str(overlay_sid))
    if ov_session is None:
        return False
    if not np.issubdtype(ov_session.data.dtype, np.integer):
        return False  # float → heatmap
    if ov_raw is not None:
        unique_vals = np.unique(ov_raw[np.isfinite(ov_raw)])
        n_nonzero = int(np.sum(unique_vals != 0))
        if n_nonzero > 16:
            return False  # too many labels → treat as heatmap
    return True


def _composite_overlay_mask(
    rgba: np.ndarray,
    ov_raw: np.ndarray | None,
    alpha: float = float(OVERLAY_ALPHA),
    is_label: bool = False,
    override_color: np.ndarray | None = None,
) -> np.ndarray:
    """Alpha-composite an overlay on top of an RGBA frame.

    - is_label=True  → treat ov_raw as integer labels (1..N), each label gets a
                        distinct categorical colour from LABEL_COLORS.  If
                        override_color is supplied AND the mask is binary (exactly
                        one unique non-zero value), that colour is used instead of
                        LABEL_COLORS — for palette-assigned overlay lists.
    - is_label=False → treat ov_raw as a float heatmap in [0,∞); normalise to
                       [0,1] and map through a desaturated jet-like palette.
      In both cases pixels where ov_raw == 0 are transparent (no overlay).
    """
    if ov_raw is None:
        return rgba

    out = rgba.copy()
    ov_a = np.float32(alpha)

    if is_label:
        # Label map: each integer value 1..N gets a distinct colour.
        labels = np.round(ov_raw).astype(np.int32)
        unique_labels = np.unique(labels)
        nonzero_labels = unique_labels[unique_labels > 0]
        # Binary mask with a palette override colour
        use_override = override_color is not None and len(nonzero_labels) == 1
        for lbl in nonzero_labels:
            mask = labels == lbl
            if use_override:
                color = override_color
            else:
                color = LABEL_COLORS[(lbl - 1) % len(LABEL_COLORS)]
            _blend_color(out, mask, color, ov_a)
    else:
        # Float heatmap: normalise to [0,1], apply a desaturated warm-cool LUT.
        valid = np.isfinite(ov_raw) & (ov_raw > 0)
        if not valid.any():
            return rgba
        vmax_ov = float(np.max(ov_raw[valid]))
        if vmax_ov == 0:
            return rgba
        norm = np.clip(ov_raw / vmax_ov, 0.0, 1.0)
        # Desaturated jet: blend blue→cyan→yellow→red with reduced saturation
        # We apply the colormap only where the overlay is non-zero.
        ov_colors = _desaturated_jet(norm[valid])
        mask = valid
        _blend_pixels(out, mask, ov_colors, ov_a)

    return out


def _blend_color(
    out: np.ndarray, mask: np.ndarray, color: np.ndarray, ov_a: float
) -> None:
    """In-place Porter-Duff 'over' blend of a single colour onto out[mask]."""
    base_rgb = out[mask, :3].astype(np.float32) / 255.0
    base_a = out[mask, 3].astype(np.float32) / 255.0
    out_a = ov_a + base_a * (1.0 - ov_a)
    denom = np.maximum(out_a, 1e-6)
    ov_rgb = color / 255.0
    out_rgb = (ov_a * ov_rgb + (1.0 - ov_a) * base_a[:, None] * base_rgb) / denom[:, None]
    out[mask, :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[mask, 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)


def _blend_pixels(
    out: np.ndarray, mask: np.ndarray, colors: np.ndarray, ov_a: float
) -> None:
    """In-place Porter-Duff 'over' blend of per-pixel colours onto out[mask]."""
    base_rgb = out[mask, :3].astype(np.float32) / 255.0
    base_a = out[mask, 3].astype(np.float32) / 255.0
    out_a = ov_a + base_a * (1.0 - ov_a)
    denom = np.maximum(out_a, 1e-6)
    ov_rgb = colors / 255.0
    out_rgb = (ov_a * ov_rgb + (1.0 - ov_a) * base_a[:, None] * base_rgb) / denom[:, None]
    out[mask, :3] = np.clip(out_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out[mask, 3] = np.clip(out_a * 255.0, 0.0, 255.0).astype(np.uint8)


# Pre-built desaturated-jet LUT (256 × 3, float32, values in [0, 255])
def _build_desaturated_jet() -> np.ndarray:
    """Build a 256-entry RGB LUT resembling jet but with ~60% saturation."""
    t = np.linspace(0.0, 1.0, 256)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    # Desaturate: lerp toward luminance (60% colour, 40% gray)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    sat = 0.6
    r = sat * r + (1.0 - sat) * lum
    g = sat * g + (1.0 - sat) * lum
    b = sat * b + (1.0 - sat) * lum
    lut = np.stack([r, g, b], axis=1) * 255.0
    return lut.astype(np.float32)


_DESATURATED_JET_LUT = _build_desaturated_jet()


def _desaturated_jet(t: np.ndarray) -> np.ndarray:
    """Map t ∈ [0,1] array to RGB colours using the desaturated-jet LUT."""
    idx = np.clip((t * 255.0).astype(np.int32), 0, 255)
    return _DESATURATED_JET_LUT[idx]


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
    mosaic_cols=None,
    vmin_override=None,
    vmax_override=None,
):
    idx_norm = list(idx_tuple)
    idx_norm[dim_z] = 0
    key = (dim_x, dim_y, dim_z, tuple(idx_norm), colormap, dr, complex_mode, log_scale, mosaic_cols, vmin_override, vmax_override)
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

    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = float(vmin_override), float(vmax_override)
    elif complex_mode == 1 and np.iscomplexobj(session.data):
        vmin, vmax = -float(np.pi), float(np.pi)
    else:
        vmin = float(np.percentile(all_data, 1))
        vmax = float(np.percentile(all_data, 99))

    if mosaic_cols is not None:
        cols = mosaic_cols
        rows = 1
    else:
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
    "_overlay_is_label_map",
    "LABEL_COLORS",
    # Mosaic
    "render_mosaic",
    # Preload
    "_run_preload",
]
