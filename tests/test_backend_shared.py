"""Shared backend helper tests."""

import numpy as np

from arrayview._analysis import (
    _build_metadata,
    _lebesgue_slice,
    _pixel_value,
    _slice_histogram,
    _volume_histogram,
)
from arrayview._diff import _compute_diff, _diff_histogram, _render_diff_rgba
from arrayview._overlays import _composite_overlays
from arrayview._session import SESSIONS, Session
from arrayview._vectorfield import (
    _compute_vfield_arrows,
    _configure_vectorfield,
    _vfield_counts_for_level,
    _vfield_n_times,
)


def test_vectorfield_density_levels_are_monotonic():
    counts = []
    grid_flags = []
    for level in range(-5, 6):
        n_arrows, effective_stride, use_grid = _vfield_counts_for_level(
            level, 64, 64, 2
        )
        counts.append(n_arrows)
        grid_flags.append(use_grid)
        assert effective_stride > 0

    assert counts == sorted(counts)
    assert grid_flags[-1] is True
    assert all(flag is False for flag in grid_flags[:-1])


def test_configure_vectorfield_non_trailing_component_axis():
    session = Session(np.zeros((6, 8, 10), dtype=np.float32))
    vf = np.zeros((6, 3, 8, 10), dtype=np.float32)
    vf[:, 1, :, :] = 0.25
    vf[:, 2, :, :] = 0.5

    layout = _configure_vectorfield(session, vf, components_dim=1)
    assert layout["components_dim"] == 1
    assert layout["spatial_axes"] == (0, 2, 3)
    assert _vfield_n_times(session) == 1

    result = _compute_vfield_arrows(
        session, dim_x=2, dim_y=1, idx_tuple=(3, 4, 5), density_offset=0
    )
    assert result is not None
    assert result["arrows"].shape[1] == 4
    assert result["scale"] > 0
    assert result["stride"] >= 1


def test_compute_diff_modes_and_histogram():
    a = Session(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    b = Session(np.linspace(1, 0, 64, dtype=np.float32).reshape(8, 8))

    raw, vmin, vmax, colormap, nan_mask = _compute_diff(
        a, b, dim_x=1, dim_y=0, indices="0,0", dim_z=-1,
        dr=1, complex_mode=0, log_scale=False, diff_mode=1,
    )
    assert raw.shape == (8, 8)
    assert (vmin, vmax, colormap, nan_mask) == (-1.0, 1.0, "RdBu_r", None)

    hist = _diff_histogram(raw, bins=16)
    assert len(hist["counts"]) == 16
    assert len(hist["edges"]) == 17

    rgba = _render_diff_rgba(raw, vmin, vmax, colormap, nan_mask)
    assert rgba.shape == (8, 8, 4)

    raw_abs, vmin_abs, vmax_abs, colormap_abs, _ = _compute_diff(
        a, b, dim_x=1, dim_y=0, indices="0,0", dim_z=-1,
        dr=1, complex_mode=0, log_scale=False, diff_mode=2,
    )
    assert np.all(raw_abs >= 0)
    assert vmin_abs == 0.0
    assert vmax_abs > 0
    assert colormap_abs == "afmhot"


def test_composite_overlays_uses_shared_session_lookup():
    base = Session(np.zeros((4, 4), dtype=np.float32))
    overlay = Session(np.zeros((4, 4), dtype=np.uint8))
    overlay.data[1:3, 1:3] = 1
    SESSIONS[overlay.sid] = overlay
    try:
        rgba = np.zeros((4, 4, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        out = _composite_overlays(
            rgba,
            overlay.sid,
            "00ff00",
            0.75,
            dim_x=1,
            dim_y=0,
            idx_tuple=(0, 0),
            shape_hw=(4, 4),
        )
    finally:
        SESSIONS.pop(overlay.sid, None)

    assert out[2, 2, 1] > out[2, 2, 0]
    assert out[2, 2, 1] > out[2, 2, 2]
    assert np.array_equal(out[0, 0], rgba[0, 0])


def test_analysis_helpers_cover_metadata_histograms_and_pixels():
    session = Session(np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6))

    meta = _build_metadata(session)
    assert meta["shape"] == [3, 4, 5, 6]
    assert meta["is_complex"] is False

    hist = _slice_histogram(
        session, dim_x=3, dim_y=2, indices="0,0,0,0", bins=16
    )
    assert len(hist["counts"]) == 16
    assert len(hist["edges"]) == 17

    volume_hist = _volume_histogram(
        session,
        dim_x=3,
        dim_y=2,
        scroll_dims="0,1",
        fixed_indices="",
        bins=8,
    )
    assert len(volume_hist["counts"]) == 8
    assert len(volume_hist["edges"]) == 9

    lebesgue = _lebesgue_slice(
        session, dim_x=3, dim_y=2, indices="0,0,0,0"
    )
    assert lebesgue.shape == (5, 6)
    assert lebesgue.dtype == np.float32

    two_d = Session(np.arange(20, dtype=np.float32).reshape(4, 5))
    assert _pixel_value(two_d, 1, 0, "0,0", px=4, py=3) == 19.0
    assert _pixel_value(two_d, 1, 0, "0,0", px=99, py=99) is None


def test_app_shim_re_exports():
    """_app.py re-exports key symbols from the new submodule layout."""
    import arrayview._app as appmod

    # Core API
    assert hasattr(appmod, "view")
    assert hasattr(appmod, "ViewHandle")
    assert hasattr(appmod, "arrayview")
    assert hasattr(appmod, "Session")
    assert hasattr(appmod, "SESSIONS")
    assert hasattr(appmod, "app")

    # Rendering
    assert hasattr(appmod, "render_rgba")
    assert hasattr(appmod, "render_mosaic")
    assert hasattr(appmod, "LUTS")

    # IO
    assert hasattr(appmod, "load_data")
    assert hasattr(appmod, "_tensor_to_numpy")

    # Platform
    assert hasattr(appmod, "_in_jupyter")
    assert hasattr(appmod, "_in_vscode_terminal")

    # VS Code (now in submodules)
    assert hasattr(appmod, "_ensure_vscode_extension")
    assert hasattr(appmod, "_open_via_signal_file")
    assert hasattr(appmod, "_open_browser")
    assert hasattr(appmod, "_write_vscode_signal")

    # Launcher
    assert hasattr(appmod, "_serve_background")
    assert hasattr(appmod, "_server_alive")
    assert hasattr(appmod, "_open_webview")

    # Server
    assert hasattr(appmod, "_VIEWER_HTML_TEMPLATE")
    assert hasattr(appmod, "_pil_image")

    # Analysis / diff / vectorfield / overlays
    assert hasattr(appmod, "_safe_float")
    assert hasattr(appmod, "_render_normalized")
    assert hasattr(appmod, "_vfield_n_times")
    assert hasattr(appmod, "_notify_shells")
