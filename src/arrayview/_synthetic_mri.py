"""qMRI parameter roles, unit normalization, and synthetic MRI signals."""

from __future__ import annotations

import numpy as np

from arrayview._render import LUTS, _build_mosaic_grid, _ensure_lut, apply_complex_mode, extract_slice


QMRI_ROLE_ORDER: dict[int, list[str]] = {
    3: ["t1", "t2", "pd"],
    4: ["t1", "t2", "pd", "phase"],
    5: ["t1", "t2", "b1", "pd", "phase"],
    6: ["t1", "t2", "b1", "db0", "pd", "phase"],
}


SYNTHETIC_PRESETS: dict[str, dict[str, float | str]] = {
    "t1w": {"label": "T1W", "kind": "se", "te": 10.0, "tr": 650.0},
    "t2w": {"label": "T2W", "kind": "se", "te": 100.0, "tr": 4500.0},
    "pdw": {"label": "PDW", "kind": "se", "te": 10.0, "tr": 8000.0},
    "t2flair": {"label": "T2-FLAIR", "kind": "ir", "te": 90.0, "tr": 15000.0, "ti": 3100.0},
    "stir": {"label": "STIR", "kind": "ir", "te": 100.0, "tr": 15000.0, "ti": 300.0},
    "psir": {"label": "PSIR", "kind": "psir", "te": 10.0, "tr": 6000.0, "ti": 500.0},
}


def qmri_roles_for_size(size: int) -> list[str]:
    """Return qMRI semantic roles for a parameter dimension size."""
    return list(QMRI_ROLE_ORDER.get(int(size), []))


def qmri_role_index(param_size: int, role: str) -> int | None:
    """Return the map index for a qMRI semantic role."""
    roles = qmri_roles_for_size(param_size)
    try:
        return roles.index(role)
    except ValueError:
        return None


def normalize_relaxation_ms(data: np.ndarray, role: str | None = None) -> np.ndarray:
    """Return T1/T2 data in ms when maps appear to be stored in seconds."""
    arr = np.asarray(data, dtype=np.float32)
    if role not in {"t1", "t2"}:
        return arr
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return arr
    central = float(np.nanmedian(finite))
    if 0.0 < central <= 5.0:
        return (arr * np.float32(1000.0)).astype(np.float32)
    return arr


def qmri_display_slice(
    session,
    dim_x: int,
    dim_y: int,
    idx_list: list[int],
    role: str | None = None,
) -> np.ndarray:
    """Extract a qMRI slice and normalize T1/T2 units for display if needed."""
    raw = extract_slice(session, dim_x, dim_y, idx_list)
    return normalize_relaxation_ms(raw, role)


def _extract_qmri_role_slice(
    session,
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    qmri_dim: int,
    role: str,
) -> np.ndarray:
    param_size = int(session.shape[qmri_dim])
    role_idx = qmri_role_index(param_size, role)
    if role_idx is None:
        raise ValueError(f"qMRI role {role!r} unavailable for parameter size {param_size}")
    idx_list = list(idx_tuple)
    idx_list[qmri_dim] = role_idx
    raw = extract_slice(session, dim_x, dim_y, idx_list)
    return normalize_relaxation_ms(raw, role)


def synthetic_qmri_slice(
    session,
    dim_x: int,
    dim_y: int,
    idx_tuple: tuple[int, ...],
    qmri_dim: int,
    contrast: str,
    *,
    te: float | None = None,
    tr: float | None = None,
    ti: float | None = None,
) -> np.ndarray:
    """Compute a synthetic MRI contrast from T1/T2/PD qMRI maps."""
    contrast_id = str(contrast).lower()
    preset = SYNTHETIC_PRESETS.get(contrast_id)
    if preset is None:
        raise ValueError(f"unknown synthetic MRI contrast: {contrast}")
    if not (0 <= qmri_dim < len(session.shape)):
        raise ValueError("qmri_dim out of range")
    if int(session.shape[qmri_dim]) not in QMRI_ROLE_ORDER:
        raise ValueError("qMRI parameter dimension size must be 3-6")

    te_ms = float(preset.get("te", 0.0) if te is None else te)
    tr_ms = float(preset.get("tr", 0.0) if tr is None else tr)
    ti_ms = float(preset.get("ti", 0.0) if ti is None else ti)

    t1 = _extract_qmri_role_slice(session, dim_x, dim_y, idx_tuple, qmri_dim, "t1")
    t2 = _extract_qmri_role_slice(session, dim_x, dim_y, idx_tuple, qmri_dim, "t2")
    pd = _extract_qmri_role_slice(session, dim_x, dim_y, idx_tuple, qmri_dim, "pd")

    t1 = np.maximum(t1.astype(np.float32), np.float32(1e-6))
    t2 = np.maximum(t2.astype(np.float32), np.float32(1e-6))
    pd = np.nan_to_num(pd.astype(np.float32))

    e1 = np.exp(np.float32(-tr_ms) / t1)
    e2 = np.exp(np.float32(-te_ms) / t2)
    kind = str(preset.get("kind", "se"))
    if kind == "se":
        signal = pd * (np.float32(1.0) - e1) * e2
    else:
        recovery = np.float32(1.0) - np.float32(2.0) * np.exp(np.float32(-ti_ms) / t1) + e1
        if kind == "ir":
            recovery = np.abs(recovery)
        signal = pd * recovery * e2
    return np.nan_to_num(signal).astype(np.float32)


def synthetic_window(data: np.ndarray, contrast: str) -> tuple[float, float]:
    """Return display window for a synthetic MRI slice."""
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    if str(contrast).lower() == "psir":
        mx = float(np.percentile(np.abs(finite), 99))
        return (-mx, mx) if mx > 0 else (-1.0, 1.0)
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def render_qmri_mosaic_rgba(
    session,
    dim_x: int,
    dim_y: int,
    dim_z: int,
    idx_tuple: tuple[int, ...],
    colormap: str,
    role: str,
    complex_mode: int = 0,
    log_scale: bool = False,
    mosaic_cols: int | None = None,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Render qMRI mosaic frames after T1/T2 unit normalization."""
    frames = []
    for i in range(session.shape[dim_z]):
        idx = [i if j == dim_z else idx_tuple[j] for j in range(len(session.shape))]
        raw = qmri_display_slice(session, dim_x, dim_y, idx, role)
        data = apply_complex_mode(raw, complex_mode)
        if log_scale:
            data = np.log1p(np.abs(data)).astype(np.float32)
        frames.append(data)
    all_data = np.stack(frames)
    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = float(vmin_override), float(vmax_override)
    else:
        vmin = float(np.percentile(all_data, 1))
        vmax = float(np.percentile(all_data, 99))
    grid, _, _ = _build_mosaic_grid(
        all_data, session.shape[dim_z], cols=mosaic_cols
    ) if mosaic_cols is not None else _build_mosaic_grid(all_data, session.shape[dim_z])
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
    return rgba, vmin, vmax
