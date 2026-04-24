"""Shared overlay compositing helpers."""

from __future__ import annotations

import numpy as np

from arrayview._render import (
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
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    shape_hw: tuple[int, int],
) -> np.ndarray:
    """Composite one or more overlay masks onto RGBA pixels."""
    if not overlay_sid_str:
        return rgba
    sids = [s.strip() for s in overlay_sid_str.split(",") if s.strip()]
    colors_raw = (
        [c.strip() for c in overlay_colors_str.split(",")] if overlay_colors_str else []
    )
    for i, sid in enumerate(sids):
        color = _parse_hex_color(colors_raw[i]) if i < len(colors_raw) else None
        ov_raw = _extract_overlay_mask(
            sid,
            dim_x,
            dim_y,
            idx_tuple,
            expected_shape=shape_hw,
        )
        rgba = _composite_overlay_mask(
            rgba,
            ov_raw,
            alpha=overlay_alpha,
            is_label=_overlay_is_label_map(sid, ov_raw),
            override_color=color,
        )
    return rgba
