"""Shared vector field layout and arrow sampling."""

from __future__ import annotations

import numpy as np


_MAX_VFIELD_ARROWS = 65536


def _normalize_axis(axis: int, ndim: int, flag_name: str) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"{flag_name} must be in [-{ndim}, {ndim - 1}], got {axis}."
        )
    return axis


def _resolve_vfield_layout(
    vf_shape: tuple[int, ...],
    image_shape: tuple[int, ...],
    components_dim: int | None = None,
) -> dict[str, object]:
    """Resolve component, time, and spatial axes for a vector field."""
    vf_shape = tuple(int(s) for s in vf_shape)
    image_shape = tuple(int(s) for s in image_shape)
    if len(vf_shape) < len(image_shape) + 1:
        raise ValueError(
            f"vector field shape {vf_shape} is too small for image shape {image_shape}."
        )

    if components_dim is not None:
        comp_dim = _normalize_axis(
            components_dim, len(vf_shape), "--vectorfield-components-dim"
        )
        if vf_shape[comp_dim] != 3:
            raise ValueError(
                f"--vectorfield-components-dim points to axis {comp_dim}, but that axis has size {vf_shape[comp_dim]} instead of 3."
            )
    else:
        candidates = [i for i, s in enumerate(vf_shape) if s == 3]
        if not candidates:
            raise ValueError(
                f"vector field shape {vf_shape} has no axis of size 3 for the xyz displacement components; specify one with --vectorfield-components-dim."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"vector field shape {vf_shape} has multiple axes of size 3 ({candidates}); specify the xyz displacement axis with --vectorfield-components-dim."
            )
        comp_dim = candidates[0]

    remaining_axes = [ax for ax in range(len(vf_shape)) if ax != comp_dim]
    if len(remaining_axes) == len(image_shape):
        time_dim = None
        spatial_axes = tuple(remaining_axes)
    elif len(remaining_axes) == len(image_shape) + 1:
        time_dim = remaining_axes[0]
        spatial_axes = tuple(remaining_axes[1:])
    else:
        raise ValueError(
            f"vector field shape {vf_shape} is incompatible with image shape {image_shape}; expected spatial dims {image_shape} plus one component axis of size 3, with at most one extra leading time axis."
        )

    vf_spatial_shape = tuple(vf_shape[ax] for ax in spatial_axes)
    if vf_spatial_shape != image_shape:
        raise ValueError(
            f"vector field spatial shape {vf_spatial_shape} does not match image shape {image_shape}."
        )

    return {
        "components_dim": comp_dim,
        "time_dim": time_dim,
        "spatial_axes": spatial_axes,
        "n_times": int(vf_shape[time_dim]) if time_dim is not None else 1,
    }


def _configure_vectorfield(
    session,
    vf_data,
    components_dim: int | None = None,
) -> dict[str, object]:
    """Attach vector field data and cache its resolved layout on the session."""
    layout = _resolve_vfield_layout(
        tuple(int(s) for s in np.shape(vf_data)),
        tuple(int(s) for s in session.spatial_shape),
        components_dim,
    )
    session.vfield = vf_data
    session.vfield_component_dim = int(layout["components_dim"])
    session.vfield_time_dim = layout["time_dim"]
    session.vfield_spatial_axes = tuple(int(a) for a in layout["spatial_axes"])
    return layout


def _get_vfield_layout(session) -> dict[str, object] | None:
    """Return a cached vector field layout, resolving it if needed."""
    if session.vfield is None:
        return None
    if (
        session.vfield_component_dim is not None
        and session.vfield_spatial_axes is not None
    ):
        return {
            "components_dim": int(session.vfield_component_dim),
            "time_dim": session.vfield_time_dim,
            "spatial_axes": tuple(int(a) for a in session.vfield_spatial_axes),
            "n_times": int(np.shape(session.vfield)[session.vfield_time_dim])
            if session.vfield_time_dim is not None
            else 1,
        }
    return _configure_vectorfield(session, session.vfield)


def _vfield_n_times(session) -> int:
    """Return vector field frame count for metadata."""
    if session.vfield is None:
        return 0
    return int((_get_vfield_layout(session) or {}).get("n_times", 1))


def _vfield_counts_for_level(
    density_offset: int,
    H: int,
    W: int,
    base_stride: int,
):
    """Return arrow count, effective stride, and grid mode for a density level."""
    level = max(-5, min(5, int(density_offset)))
    total = max(1, H * W)
    if level >= 5:
        return min(total, _MAX_VFIELD_ARROWS), 1.0, True

    sparse_stride = max(1, base_stride * 2)
    n_min = max(1, max(1, H // sparse_stride) * max(1, W // sparse_stride))
    n_max = max(n_min * 2, min(total // 2, _MAX_VFIELD_ARROWS // 2))
    log_min = float(np.log2(n_min))
    log_max = float(np.log2(n_max))
    t = (level + 5) / 9.0
    log_n = log_min + t * (log_max - log_min)
    n_arrows = max(1, min(total, int(round(2.0**log_n))))
    effective_stride = float((total / n_arrows) ** 0.5)
    return n_arrows, effective_stride, False


def _compute_vfield_arrows(
    session,
    dim_x,
    dim_y,
    idx_tuple,
    t_index=0,
    density_offset=0,
):
    """Compute sampled vector field arrows for a 2-D view."""
    if session.vfield is None:
        return None
    layout = _get_vfield_layout(session)
    if layout is None:
        return None

    vf = session.vfield
    slices = [slice(None)] * vf.ndim
    time_dim = layout["time_dim"]
    if time_dim is not None:
        t = max(0, min(int(layout["n_times"]) - 1, t_index))
        slices[int(time_dim)] = t

    spatial_axes = tuple(int(ax) for ax in layout["spatial_axes"])
    comp_dim = int(layout["components_dim"])
    vf_x_axis = spatial_axes[dim_x]
    vf_y_axis = spatial_axes[dim_y]
    for img_dim, vf_axis in enumerate(spatial_axes):
        if img_dim in (dim_x, dim_y):
            continue
        slices[vf_axis] = int(idx_tuple[img_dim])

    vf_slice = np.asarray(vf[tuple(slices)], dtype=np.float32)
    free_axes = [ax for ax, sl in enumerate(slices) if isinstance(sl, slice)]
    axis_pos = {ax: i for i, ax in enumerate(free_axes)}
    vf_slice = vf_slice.transpose(
        axis_pos[vf_y_axis], axis_pos[vf_x_axis], axis_pos[comp_dim]
    )

    H, W = vf_slice.shape[:2]
    n_comp = vf_slice.shape[2]
    n_spatial = len(spatial_axes)
    comp_offset = n_spatial - n_comp
    cy = dim_y - comp_offset
    cx = dim_x - comp_offset
    vy_comp = (
        vf_slice[:, :, cy] if 0 <= cy < n_comp else np.zeros((H, W), dtype=np.float32)
    )
    vx_comp = (
        vf_slice[:, :, cx] if 0 <= cx < n_comp else np.zeros((H, W), dtype=np.float32)
    )

    base_stride = max(1, max(H, W) // 32)
    n_arrows_target, effective_stride, use_grid = _vfield_counts_for_level(
        density_offset,
        H,
        W,
        base_stride,
    )
    if use_grid:
        ys = np.arange(0, H, dtype=int)
        xs = np.arange(0, W, dtype=int)
        gy_grid, gx_grid = np.meshgrid(ys, xs, indexing="ij")
        gy = gy_grid.ravel()
        gx = gx_grid.ravel()
        if gy.size > _MAX_VFIELD_ARROWS:
            keep = np.linspace(0, gy.size - 1, _MAX_VFIELD_ARROWS).astype(int)
            gy = gy[keep]
            gx = gx[keep]
    else:
        n_arrows = min(n_arrows_target, _MAX_VFIELD_ARROWS)
        rng = np.random.default_rng(int(H) * 10007 + int(W))
        gy = rng.integers(0, H, n_arrows).astype(int)
        gx = rng.integers(0, W, n_arrows).astype(int)

    vx_s = vx_comp[gy, gx]
    vy_s = vy_comp[gy, gx]
    mags = np.sqrt(vx_s**2 + vy_s**2)
    nonzero = mags[mags > 0]
    p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
    scale = float(effective_stride * 0.75 / max(p95, 1e-9))

    coords = np.column_stack([gx, gy, vx_s, vy_s]).astype(np.float32)
    return {"arrows": coords, "scale": scale, "stride": int(round(effective_stride))}
