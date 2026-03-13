"""Backward-compatibility shim.

This module re-exports everything that was originally defined in _app.py,
now split across _session, _render, _io, _platform, _vscode, _server,
and _launcher.  Existing imports like ``from arrayview._app import view``
continue to work unchanged.
"""

# Re-export: stdlib / third-party used by tests that do ``import arrayview._app as appmod``
import urllib  # noqa: F401 — used by test_cli (appmod.urllib)

# ---------------------------------------------------------------------------
# _session.py — Session, global state, caches, constants
# ---------------------------------------------------------------------------
from arrayview._session import (  # noqa: F401
    _verbose,
    _vprint,
    SERVER_LOOP,
    VIEWER_SOCKETS,
    SHELL_SOCKETS,
    _window_process,
    PENDING_SESSIONS,
    _RENDER_QUEUE,
    _RENDER_THREAD,
    _render_worker,
    _ensure_render_thread,
    _render,
    _PREFETCH_POOL,
    _PREFETCH_LOCK,
    _get_prefetch_pool,
    _schedule_prefetch,
    Session,
    SESSIONS,
    COLORMAPS,
    DR_PERCENTILES,
    DR_LABELS,
    ZARR_LARGE_XY_TILE,
    ZARR_T_DEPTH,
    PREFETCH_NEIGHBORS,
    PREFETCH_BUDGET_BYTES,
    HEAVY_OP_LIMIT_BYTES,
    _DEFAULT_HEAVY_OP_MB,
    _estimate_array_bytes,
    _total_ram_bytes,
    _cache_budget,
    _RAW_CACHE_BYTES,
    _RGBA_CACHE_BYTES,
    _MOSAIC_CACHE_BYTES,
    zarr_chunk_preset,
)
import arrayview._session as _session_mod  # noqa: F401

# ---------------------------------------------------------------------------
# _render.py — Rendering pipeline
# ---------------------------------------------------------------------------
from arrayview._render import (  # noqa: F401
    LUTS,
    _mpl_colormaps,
    _init_luts,
    _lut_to_gradient_stops,
    COLORMAP_GRADIENT_STOPS,
    COMPLEX_MODES,
    REAL_MODES,
    OVERLAY_COLOR,
    OVERLAY_ALPHA,
    mosaic_shape,
    _compute_vmin_vmax,
    extract_slice,
    apply_complex_mode,
    _compute_otsu_threshold,
    _prepare_display,
    _ensure_lut,
    apply_colormap_rgba,
    _detect_rgb_axis,
    _setup_rgb,
    render_rgb_rgba,
    render_rgba,
    _extract_overlay_mask,
    _composite_overlay_mask,
    render_mosaic,
    _run_preload,
)

# ---------------------------------------------------------------------------
# _io.py — Data loading, tensor conversion, file utilities
# ---------------------------------------------------------------------------
from arrayview._io import (  # noqa: F401
    load_data,
    _tensor_to_numpy,
    _SUPPORTED_EXTS,
    _peek_file_shape,
)

# ---------------------------------------------------------------------------
# _platform.py — Environment detection
# ---------------------------------------------------------------------------
from arrayview._platform import (  # noqa: F401
    _in_jupyter,
    _in_vscode_terminal,
    _is_vscode_remote,
    _in_vscode_tunnel,
    _can_native_window,
    _find_code_cli,
    _find_vscode_ipc_hook,
    _VSCODE_IPC_HOOK_CACHE,
    _is_julia_env,
    _in_julia_jupyter,
)
import arrayview._platform as _platform_mod  # noqa: F401

# ---------------------------------------------------------------------------
# _vscode.py — VS Code integration
# ---------------------------------------------------------------------------
from arrayview._vscode import (  # noqa: F401
    _vscode_app_bundle,
    _VSCODE_EXT_INSTALLED,
    _VSCODE_EXT_FRESH_INSTALL,
    _VSCODE_EXT_VERSION,
    _VSCODE_SIGNAL_FILENAME,
    _VSCODE_COMPAT_SIGNAL_FILENAMES,
    _VSCODE_PORT_SETTINGS_SETTLE_SECONDS,
    _VSCODE_SIGNAL_MAX_AGE_MS,
    _bundled_vscode_vsix_version,
    _patch_vscode_extension_metadata,
    _ensure_vscode_extension,
    _configure_vscode_port_preview,
    _open_via_signal_file,
    _schedule_remote_open_retries,
    _write_vscode_signal,
    _print_viewer_location,
    _open_browser,
)

# ---------------------------------------------------------------------------
# _server.py — FastAPI app, routes, WebSocket endpoints, HTML templates
# ---------------------------------------------------------------------------
from arrayview._server import (  # noqa: F401
    app,
    _pil_image,
    _SHELL_HTML,
    _VIEWER_HTML_TEMPLATE,
    _notify_shells,
    _safe_float,
    _vfield_n_times,
    _render_normalized,
    _render_normalized_mosaic,
    MASK_MULTIPLIERS,
)

# ---------------------------------------------------------------------------
# _launcher.py — Entry points, process management, view(), arrayview() CLI
# ---------------------------------------------------------------------------
from arrayview._launcher import (  # noqa: F401
    _uvicorn,
    _ICON_PNG_PATH,
    _get_icon_png_path,
    _open_webview,
    _open_webview_with_fallback,
    _open_webview_cli,
    _server_alive,
    _server_pid,
    _port_in_use,
    _wait_for_port,
    _find_server_port,
    _server_ready_event,
    _serve_background,
    view,
    _is_script_mode,
    _stop_server_when_viewer_closes,
    _wait_for_viewer_close,
    _view_julia,
    _view_subprocess,
    _serve_empty,
    _serve_daemon,
    _make_demo_array,
    arrayview,
)
