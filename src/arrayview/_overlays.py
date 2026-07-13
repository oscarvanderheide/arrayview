"""Shared overlay compositing helpers."""

from __future__ import annotations

import numpy as np

from arrayview._render import (
    _build_mosaic_grid,
    _composite_overlay_mask,
    _extract_overlay_mask,
    _overlay_is_label_map,
)


def _parse_hex_color(hex_str: str) -> np.ndarray | None:
    """Parse a 6-char hex RGB color."""
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return np.array(
            [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
            dtype=np.uint8,
        )
    except ValueError:
        return None


def _composite_overlays(
    rgba: np.ndarray,
    overlay_sid_str: str | None,
    overlay_colors_str: str | None,
    overlay_alpha: float,
    overlay_alphas_str: str | None,
    overlay_outline: bool,
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    shape_hw: tuple[int, int],
    base_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Composite one or more overlay masks onto RGBA pixels."""
    if not overlay_sid_str:
        return rgba
    sids = [s.strip() for s in overlay_sid_str.split(",") if s.strip()]
    colors_raw = (
        [c.strip() for c in overlay_colors_str.split(",")] if overlay_colors_str else []
    )
    alphas_raw = (
        [a.strip() for a in overlay_alphas_str.split(",")] if overlay_alphas_str else []
    )
    for i, sid in enumerate(sids):
        color = _parse_hex_color(colors_raw[i]) if i < len(colors_raw) else None
        alpha = overlay_alpha
        if i < len(alphas_raw):
            try:
                alpha = float(alphas_raw[i])
            except ValueError:
                alpha = overlay_alpha
        alpha = max(0.0, min(1.0, alpha))
        ov_raw = _extract_overlay_mask(
            sid,
            dim_x,
            dim_y,
            idx_tuple,
            expected_shape=shape_hw,
            base_shape=base_shape,
        )
        rgba = _composite_overlay_mask(
            rgba,
            ov_raw,
            alpha=alpha,
            is_label=_overlay_is_label_map(sid, ov_raw),
            override_color=color,
            outline_only=overlay_outline,
        )
    return rgba


def _composite_mosaic_overlays(
    rgba: np.ndarray,
    overlay_sid_str: str | None,
    overlay_colors_str: str | None,
    overlay_alpha: float,
    overlay_alphas_str: str | None,
    overlay_outline: bool,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    idx_tuple: tuple[int, ...],
    base_shape: tuple[int, ...],
    mosaic_cols: int | None = None,
) -> np.ndarray:
    """Composite each overlay as a mosaic using the base mosaic's frame indices.

    Missing overlay axes are resolved per frame by ``_extract_overlay_mask``.
    That lets a mask omit, for example, an echo axis without materialising a
    repeated full-volume copy.
    """
    if not overlay_sid_str:
        return rgba
    sids = [sid.strip() for sid in overlay_sid_str.split(",") if sid.strip()]
    colors_raw = (
        [color.strip() for color in overlay_colors_str.split(",")]
        if overlay_colors_str
        else []
    )
    alphas_raw = (
        [alpha.strip() for alpha in overlay_alphas_str.split(",")]
        if overlay_alphas_str
        else []
    )
    tile_shape = (int(base_shape[dim_y]), int(base_shape[dim_x]))
    n_frames = int(base_shape[dim_z])

    for i, sid in enumerate(sids):
        color = _parse_hex_color(colors_raw[i]) if i < len(colors_raw) else None
        try:
            alpha = float(alphas_raw[i]) if i < len(alphas_raw) else overlay_alpha
        except ValueError:
            alpha = overlay_alpha
        alpha = max(0.0, min(1.0, alpha))

        frames = []
        for frame in range(n_frames):
            frame_idx = list(idx_tuple)
            frame_idx[dim_z] = frame
            ov_raw = _extract_overlay_mask(
                sid,
                dim_x,
                dim_y,
                tuple(frame_idx),
                expected_shape=tile_shape,
                base_shape=base_shape,
            )
            frames.append(
                ov_raw if ov_raw is not None else np.zeros(tile_shape, dtype=np.float32)
            )
        grid, _, _ = _build_mosaic_grid(frames, n_frames, cols=mosaic_cols)
        # Gaps are transparent, not NaN label values or heatmap samples.
        grid = np.nan_to_num(grid, nan=0.0)
        rgba = _composite_overlay_mask(
            rgba,
            grid,
            alpha=alpha,
            is_label=_overlay_is_label_map(sid, grid),
            override_color=color,
            outline_only=overlay_outline,
        )
    return rgba
