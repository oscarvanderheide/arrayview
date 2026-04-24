"""Stdio-based server for VS Code webview message passing.

Reads JSON requests from stdin (one per line), renders slices using the
existing pipeline, writes binary responses to stdout with a u32 length prefix.

Protocol
--------
Input : one JSON object per line on stdin
Output: [u32 length][binary payload]

The binary payload has the same format as the WebSocket binary response for
slice requests, and is length-prefixed JSON for metadata/register/sessions.
"""

import concurrent.futures as _futures
import json
import arrayview._session as _session_mod
import os
import select
import struct
import sys
import traceback
import uuid

import numpy as np

from arrayview._diff import _compute_diff, _diff_histogram, _render_diff_rgba
from arrayview._io import load_data_with_meta
from arrayview._overlays import _composite_overlays
from arrayview._render import (
    _ensure_lut,
    _init_luts,
    _prepare_display,
    apply_complex_mode,
    extract_projection,
    extract_slice,
    render_mosaic,
    render_projection_rgba,
    render_rgb_rgba,
    render_rgba,
    _setup_rgb,
)
from arrayview._session import SESSIONS, Session
from arrayview._vectorfield import _compute_vfield_arrows, _vfield_n_times


# ---------------------------------------------------------------------------
# Lazy PIL import
# ---------------------------------------------------------------------------
_pil_image_mod = None


def _pil_image():
    global _pil_image_mod
    if _pil_image_mod is None:
        from PIL import Image

        _pil_image_mod = Image
    return _pil_image_mod


# ---------------------------------------------------------------------------
# Response writing
# ---------------------------------------------------------------------------


def _write_response(data: bytes) -> None:
    """Write a length-prefixed binary response to stdout."""
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _write_json(obj: dict) -> None:
    """Write a JSON response as a length-prefixed UTF-8 payload."""
    _write_response(json.dumps(obj).encode())


def _write_error(msg: str) -> None:
    _write_json({"error": msg})


def _write_skip_response(msg: dict) -> None:
    """Write a minimal 1×1 skip frame for an evicted slice request."""
    seq = int(msg.get("seq", 0))
    header = np.array([seq, 1, 1], dtype=np.uint32).tobytes()
    vminmax = np.array([0.0, 0.0], dtype=np.float32).tobytes()
    pixel = b"\x00\x00\x00\x00"  # transparent RGBA
    _write_response(header + vminmax + pixel)


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


def _handle_register(msg: dict) -> None:
    """Register a new session from a file path."""
    file_path = msg["path"]
    name = msg.get("name") or __import__("os").path.basename(file_path)
    options = msg.get("options", {})

    data, spatial_meta = load_data_with_meta(file_path)
    session = Session(data=data, filepath=file_path, name=name)
    session.spatial_meta = spatial_meta
    if spatial_meta is not None:
        session.original_volume = data

    if options.get("rgb"):
        _setup_rgb(session)

    SESSIONS[session.sid] = session

    shape = [
        int(s)
        for s in (
            session.spatial_shape if session.rgb_axis is not None else session.shape
        )
    ]

    _write_json(
        {
            "sid": session.sid,
            "shape": shape,
            "name": session.name,
            "is_complex": bool(np.iscomplexobj(session.data)),
            "has_vectorfield": session.vfield is not None,
            "is_rgb": session.rgb_axis is not None,
        }
    )


def _handle_metadata(msg: dict) -> None:
    """Return session metadata."""
    sid = msg["sid"]
    session = SESSIONS.get(sid)
    if not session:
        _write_error("session not found")
        return

    shape = [
        int(s)
        for s in (
            session.spatial_shape if session.rgb_axis is not None else session.shape
        )
    ]
    _write_json(
        {
            "shape": shape,
            "is_complex": bool(np.iscomplexobj(session.data)),
            "name": session.name,
            "has_vectorfield": session.vfield is not None,
            "vfield_n_times": _vfield_n_times(session),
            "is_rgb": session.rgb_axis is not None,
            **(
                {
                    "default_dims": [int(d) for d in default_dims]
                }
                if (
                    default_dims := _session_mod._startup_dims_for_data(
                        session.data,
                        session.spatial_shape if session.rgb_axis is not None else session.shape,
                    )
                ) is not None
                else {}
            ),
        }
    )


def _handle_sessions() -> None:
    """Return list of active sessions."""
    result = []
    for s in SESSIONS.values():
        dtype_str = str(getattr(s.data, "dtype", "unknown"))
        result.append(
            {
                "sid": s.sid,
                "name": s.name,
                "shape": [int(x) for x in s.shape],
                "filepath": s.filepath,
                "dtype": dtype_str,
                "estimated_mem": s._estimated_mem,
            }
        )
    _write_json(result)


def _handle_clearcache(msg: dict) -> None:
    """Clear caches for a session."""
    sid = msg["sid"]
    session = SESSIONS.get(sid)
    if session:
        session.reset_caches()
    _write_json({"status": "ok"})


def _parse_query(endpoint: str) -> dict[str, str]:
    """Extract query parameters from an endpoint URL."""
    from urllib.parse import unquote
    if "?" not in endpoint:
        return {}
    qs = endpoint.split("?", 1)[1]
    params: dict[str, str] = {}
    for part in qs.split("&"):
        if "=" in part:
            k, v = part.split("=", 1)
            params[unquote(k)] = unquote(v)
    return params


def _handle_histogram(sid: str, params: dict[str, str]) -> None:
    """Compute histogram of the current slice and write JSON response."""
    session = SESSIONS.get(sid)
    if not session:
        _write_error("session not found")
        return
    dim_x = int(params.get("dim_x", "0"))
    dim_y = int(params.get("dim_y", "1"))
    indices_str = params.get("indices", "")
    idx_tuple = tuple(int(v) for v in indices_str.split(",") if v)
    complex_mode = int(params.get("complex_mode", "0"))
    bins = max(8, min(int(params.get("bins", "128")), 512))

    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    flat = data.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        _write_json({"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0})
        return
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        _write_json({
            "counts": [int(finite.size)],
            "edges": [vmin, vmax + 1e-9],
            "vmin": vmin,
            "vmax": vmax,
        })
        return
    counts, edges = np.histogram(finite, bins=bins)
    _write_json({
        "counts": counts.tolist(),
        "edges": [float(e) for e in edges],
        "vmin": vmin,
        "vmax": vmax,
    })


def _handle_volume_histogram(sid: str, params: dict[str, str]) -> None:
    """Compute histogram sampled across scroll dimension and write JSON."""
    session = SESSIONS.get(sid)
    if not session:
        _write_error("session not found")
        return
    dim_x = int(params.get("dim_x", "0"))
    dim_y = int(params.get("dim_y", "1"))
    scroll_dim = int(params.get("scroll_dim", "0"))
    fixed_indices_str = params.get("fixed_indices", "")
    complex_mode = int(params.get("complex_mode", "0"))
    bins = max(8, min(int(params.get("bins", "64")), 512))

    # Parse fixed indices (dim:idx pairs)
    fixed: dict[int, int] = {}
    if fixed_indices_str:
        for pair in fixed_indices_str.split(","):
            if ":" in pair:
                d, v = pair.split(":", 1)
                fixed[int(d)] = int(v)

    # Check cache
    cache_key = (dim_x, dim_y, scroll_dim, tuple(sorted(fixed.items())), complex_mode)
    if not hasattr(session, "_volume_hist_cache"):
        session._volume_hist_cache = {}
    cached = session._volume_hist_cache.get(cache_key)
    if cached is not None and cached.get("_data_version") == session.data_version:
        _write_json(cached["result"])
        return

    # Subsample up to 16 evenly-spaced slices along scroll_dim
    n = session.shape[scroll_dim]
    max_samples = 16
    if n <= max_samples:
        sample_indices = list(range(n))
    else:
        step = n / max_samples
        sample_indices = [int(i * step) for i in range(max_samples)]

    pixels = []
    for si in sample_indices:
        idx_list = [s // 2 for s in session.shape]
        idx_list[scroll_dim] = si
        for d, v in fixed.items():
            idx_list[d] = v
        raw = extract_slice(session, dim_x, dim_y, idx_list)
        data = apply_complex_mode(raw, complex_mode)
        flat = data.ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size > 0:
            pixels.append(finite)

    if not pixels:
        result = {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
        session._volume_hist_cache[cache_key] = {
            "_data_version": session.data_version, "result": result,
        }
        _write_json(result)
        return

    merged = np.concatenate(pixels)
    vmin = float(merged.min())
    vmax = float(merged.max())
    if vmin == vmax:
        result = {
            "counts": [int(merged.size)],
            "edges": [vmin, vmax + 1e-9],
            "vmin": vmin,
            "vmax": vmax,
        }
    else:
        counts, edges = np.histogram(merged, bins=bins)
        result = {
            "counts": counts.tolist(),
            "edges": [float(e) for e in edges],
            "vmin": vmin,
            "vmax": vmax,
        }

    session._volume_hist_cache[cache_key] = {
        "_data_version": session.data_version, "result": result,
    }
    _write_json(result)


def _handle_lebesgue(sid: str, params: dict[str, str]) -> None:
    """Return raw slice as JSON-wrapped base64 float32 for Lebesgue mode."""
    session = SESSIONS.get(sid)
    if not session:
        _write_error("session not found")
        return
    dim_x = int(params.get("dim_x", "0"))
    dim_y = int(params.get("dim_y", "1"))
    indices_str = params.get("indices", "")
    idx_tuple = tuple(int(v) for v in indices_str.split(",") if v)
    complex_mode = int(params.get("complex_mode", "0"))
    log_scale = params.get("log_scale", "false").lower() == "true"

    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode).astype(np.float32)
    if log_scale:
        data = np.log10(np.abs(data) + 1).astype(np.float32)
    import base64
    _write_json({
        "_binary": True,
        "data": base64.b64encode(data.tobytes()).decode(),
        "headers": {
            "X-ArrayView-Width": str(data.shape[1]),
            "X-ArrayView-Height": str(data.shape[0]),
        },
    })


def _handle_vectorfield_fetch(sid: str, params: dict) -> None:
    """Return downsampled vector field arrows for a 2-D view."""
    session = SESSIONS.get(sid)
    if not session or session.vfield is None:
        _write_json({"error": "no vectorfield"})
        return
    dim_x = int(params.get("dim_x", 0))
    dim_y = int(params.get("dim_y", 0))
    indices = params.get("indices", "")
    idx_tuple = tuple(int(x) for x in indices.split(",")) if indices else ()
    t_index = int(params.get("t_index", 0))
    density_offset = int(params.get("density_offset", 0))
    result = _compute_vfield_arrows(session, dim_x, dim_y, idx_tuple, t_index, density_offset)
    if result is None:
        _write_json({"error": "vectorfield computation failed"})
        return
    _write_json({
        "arrows": result["arrows"].tolist(),
        "scale": result["scale"],
        "stride": result["stride"],
    })


def _handle_pixel(sid: str, params: dict) -> None:
    """Return the raw value at a single pixel."""
    session = SESSIONS.get(sid)
    if not session:
        _write_json({"error": "session not found"})
        return
    if session.rgb_axis is not None:
        _write_json({"value": None})
        return
    dim_x = int(params.get("dim_x", 0))
    dim_y = int(params.get("dim_y", 0))
    indices = params.get("indices", "")
    idx_tuple = tuple(int(x) for x in indices.split(",")) if indices else ()
    px = int(params.get("px", 0))
    py = int(params.get("py", 0))
    complex_mode = int(params.get("complex_mode", 0))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    if 0 <= py < h and 0 <= px < w:
        v = data[py, px]
        val = None if np.isnan(v) or np.isinf(v) else float(v)
    else:
        val = None
    _write_json({"value": val})


def _handle_diff(sid_a: str, sid_b: str, params: dict) -> None:
    """Render a diff image and return as base64 JPEG with metadata headers."""
    import base64
    import io

    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        _write_json({"error": "session not found"})
        return

    dim_x = int(params.get("dim_x", 0))
    dim_y = int(params.get("dim_y", 0))
    indices = params.get("indices", "")
    dim_z = int(params.get("dim_z", -1))
    dr = int(params.get("dr", 1))
    complex_mode = int(params.get("complex_mode", 0))
    log_scale = params.get("log_scale", "false").lower() == "true"
    diff_mode = int(params.get("diff_mode", 1))
    diff_colormap = params.get("diff_colormap", "")
    _vmin_ov = params.get("vmin_override")
    _vmax_ov = params.get("vmax_override")
    vmin_override = float(_vmin_ov) if _vmin_ov is not None else None
    vmax_override = float(_vmax_ov) if _vmax_ov is not None else None

    try:
        raw, vmin, vmax, colormap, nan_mask = _compute_diff(
            session_a, session_b, dim_x, dim_y, indices,
            dim_z, dr, complex_mode, log_scale, diff_mode,
        )
    except Exception as e:
        _write_json({"error": str(e)})
        return

    if vmin_override is not None:
        vmin = vmin_override
    if vmax_override is not None:
        vmax = vmax_override
    if diff_colormap and _ensure_lut(diff_colormap):
        colormap = diff_colormap

    rgba = _render_diff_rgba(raw, vmin, vmax, colormap, nan_mask)
    img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)

    _write_json({
        "_binary": True,
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "headers": {
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
            "X-ArrayView-Colormap": colormap,
        },
    })


def _handle_diff_histogram(sid_a: str, sid_b: str, params: dict) -> None:
    """Compute histogram of the diff between two sessions."""
    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        _write_json({"error": "session not found"})
        return

    dim_x = int(params.get("dim_x", 0))
    dim_y = int(params.get("dim_y", 0))
    indices = params.get("indices", "")
    dim_z = int(params.get("dim_z", -1))
    dr = int(params.get("dr", 1))
    complex_mode = int(params.get("complex_mode", 0))
    log_scale = params.get("log_scale", "false").lower() == "true"
    diff_mode = int(params.get("diff_mode", 1))
    bins = int(params.get("bins", 64))

    try:
        raw, _, _, _, _ = _compute_diff(
            session_a, session_b, dim_x, dim_y, indices,
            dim_z, dr, complex_mode, log_scale, diff_mode,
        )
    except Exception as e:
        _write_json({"error": str(e)})
        return

    _write_json(_diff_histogram(raw, bins))


def _handle_fetch_proxy(msg: dict) -> None:
    """Handle proxied fetch requests from the viewer."""
    endpoint = msg.get("endpoint", "")
    parts = endpoint.split("?")[0].strip("/").split("/")

    if not parts:
        _write_error("missing endpoint")
        return

    route = parts[0]
    sid = parts[1] if len(parts) > 1 else None
    params = _parse_query(endpoint)

    if route == "metadata" and sid:
        _handle_metadata({"sid": sid})
    elif route == "clearcache" and sid:
        _handle_clearcache({"sid": sid})
    elif route == "sessions":
        _handle_sessions()
    elif route == "histogram" and sid:
        _handle_histogram(sid, params)
    elif route == "volume-histogram" and sid:
        _handle_volume_histogram(sid, params)
    elif route == "lebesgue" and sid:
        _handle_lebesgue(sid, params)
    elif route == "preload" and sid:
        # Stub: preloading is a background optimisation not yet supported
        # in stdio mode.  Return success so the viewer doesn't error.
        _write_json({"status": "started"})
    elif route == "preload_status" and sid:
        _write_json({"done": 0, "total": 0, "skipped": True})
    elif route == "vectorfield" and sid:
        _handle_vectorfield_fetch(sid, params)
    elif route == "pixel" and sid:
        _handle_pixel(sid, params)
    elif route == "diff" and len(parts) >= 3:
        _handle_diff(parts[1], parts[2], params)
    elif route == "diff-histogram" and len(parts) >= 3:
        _handle_diff_histogram(parts[1], parts[2], params)
    else:
        _write_error(f"unsupported endpoint: {endpoint}")


# Slice render pool: renders multi-pane modes (qMRI, multiview, compare) in
# parallel instead of serializing on the stdin read loop.  Responses are still
# written to stdout in arrival order by the caller.  Per-cache locks on the
# Session protect raw/rgba/mosaic caches from concurrent access.
_SLICE_POOL: _futures.ThreadPoolExecutor | None = None


def _get_slice_pool() -> _futures.ThreadPoolExecutor:
    global _SLICE_POOL
    if _SLICE_POOL is None:
        from arrayview._session import _RENDER_WORKERS
        _SLICE_POOL = _futures.ThreadPoolExecutor(
            max_workers=_RENDER_WORKERS,
            thread_name_prefix="arrayview-stdio-render",
        )
    return _SLICE_POOL


def _build_slice_payload(msg: dict) -> bytes:
    """Render a slice and return its binary payload.

    Raises on missing session / bad input; does NOT write to stdout so this is
    safe to call from a thread pool.  The caller is responsible for writing
    the returned bytes (or the error) in arrival order.
    """
    sid = msg["sid"]
    session = SESSIONS.get(sid)
    if not session:
        raise RuntimeError("session not found")

    seq = int(msg.get("seq", 0))
    dim_x = int(msg["dim_x"])
    dim_y = int(msg["dim_y"])
    idx_tuple = tuple(int(x) for x in msg["indices"])
    colormap = str(msg.get("colormap", "gray"))
    dr = int(msg.get("dr", 1))
    dim_z = int(msg.get("dim_z", -1))
    complex_mode = int(msg.get("complex_mode", 0))
    log_scale = bool(msg.get("log_scale", False))
    _vmin_ov = msg.get("vmin_override")
    _vmax_ov = msg.get("vmax_override")
    vmin_override = float(_vmin_ov) if _vmin_ov is not None else None
    vmax_override = float(_vmax_ov) if _vmax_ov is not None else None
    canvas_w = int(msg.get("canvas_w", 0))
    canvas_h = int(msg.get("canvas_h", 0))
    _mc = msg.get("mosaic_cols")
    mosaic_cols = int(_mc) if _mc is not None else None
    projection_mode = int(msg.get("projection_mode", 0))
    projection_dim = int(msg.get("projection_dim", -1))

    if session.rgb_axis is not None:
        rgba = render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple))
        h, w = rgba.shape[:2]
        vmin, vmax = 0.0, 255.0
    elif dim_z >= 0:
        rgba = render_mosaic(
            session, dim_x, dim_y, dim_z, idx_tuple, colormap, dr,
            complex_mode, log_scale, mosaic_cols=mosaic_cols,
        )
        h, w = rgba.shape[:2]
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        _, vmin, vmax = _prepare_display(
            session, raw, complex_mode, dr, log_scale,
            vmin_override=vmin_override, vmax_override=vmax_override,
        )
    elif projection_mode > 0 and projection_dim >= 0:
        rgba = render_projection_rgba(
            session, dim_x, dim_y, idx_tuple, projection_dim, projection_mode,
            colormap, dr, complex_mode, log_scale, vmin_override, vmax_override,
        )
        h, w = rgba.shape[:2]
        raw = extract_projection(
            session, dim_x, dim_y, list(idx_tuple), projection_dim, projection_mode,
        )
        _, vmin, vmax = _prepare_display(
            session, raw, complex_mode, dr, log_scale,
            vmin_override=vmin_override, vmax_override=vmax_override,
        )
    else:
        rgba = render_rgba(
            session, dim_x, dim_y, idx_tuple, colormap, dr,
            complex_mode, log_scale, vmin_override, vmax_override,
        )
        h, w = rgba.shape[:2]
        raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
        _, vmin, vmax = _prepare_display(
            session, raw, complex_mode, dr, log_scale,
            vmin_override=vmin_override, vmax_override=vmax_override,
        )

        # Overlay compositing
        overlay_sid = msg.get("overlay_sid")
        overlay_colors = msg.get("overlay_colors")
        overlay_alpha = float(msg.get("overlay_alpha", 0.45))
        rgba = _composite_overlays(
            rgba, overlay_sid, overlay_colors, overlay_alpha,
            dim_x, dim_y, idx_tuple, (h, w),
        )

    # Thumbnail downsample
    if canvas_w and canvas_h and (w > canvas_w or h > canvas_h):
        pil = _pil_image().fromarray(
            rgba.astype(np.uint8) if rgba.dtype != np.uint8 else rgba,
            mode="RGBA",
        )
        pil.thumbnail((canvas_w, canvas_h), _pil_image().LANCZOS)
        rgba = np.array(pil)
        h, w = rgba.shape[:2]

    # Build binary payload (same format as WebSocket)
    header = np.array([seq, w, h], dtype=np.uint32).tobytes()
    vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
    payload = header + vminmax + rgba.tobytes()

    # Append vectorfield binary trailer
    if session.vfield is not None:
        vf_density = int(msg.get("vf_density", 0))
        vf_t = int(msg.get("vf_t", 0))
        vf_result = _compute_vfield_arrows(
            session, dim_x, dim_y, idx_tuple,
            t_index=vf_t, density_offset=vf_density,
        )
        if vf_result is not None:
            arrows = vf_result["arrows"]
            vf_hdr = np.array(
                [len(arrows), vf_result["stride"]], dtype=np.uint32
            ).tobytes()
            vf_scale = np.array(
                [vf_result["scale"]], dtype=np.float32
            ).tobytes()
            payload += vf_hdr + vf_scale + arrows.tobytes()

    return payload


def _handle_slice(msg: dict) -> None:
    """Render a slice and write the binary response (single-slice path)."""
    _write_response(_build_slice_payload(msg))


def _handle_get_viewer_html(msg: dict) -> None:
    """Return the rendered viewer HTML with template substitutions."""
    from importlib.resources import files as _pkg_files

    from arrayview._config import get_viewer_colormaps, get_viewer_rounded_panes, get_viewer_theme
    from arrayview._render import COLORMAP_GRADIENT_STOPS, COMPLEX_MODES, REAL_MODES
    from arrayview._session import COLORMAPS

    template = _pkg_files("arrayview").joinpath("_viewer.html").read_text(encoding="utf-8")
    # In the direct webview (postMessage transport) GSAP cannot be loaded via
    # <script src="/gsap.min.js"> — the webview origin is vscode-webview://…,
    # not the FastAPI server.  Inline the vendored copy instead.
    gsap_js = _pkg_files("arrayview").joinpath("gsap.min.js").read_text(encoding="utf-8")
    template = template.replace(
        '<script src="gsap.min.js"></script>',
        f"<script>{gsap_js}</script>",
    )

    _cfg_colormaps = get_viewer_colormaps()
    _active_colormaps = _cfg_colormaps if _cfg_colormaps is not None else COLORMAPS

    sid = msg.get("sid", "")

    # Build query string from session info — the extension forwards the full
    # SESSION: payload so new features (overlay_sid, compare, etc.) work
    # without touching extension code.
    qs = f"?sid={sid}&transport=postMessage"
    overlay_sid = msg.get("overlay_sid")
    if overlay_sid:
        qs += f"&overlay_sid={overlay_sid}"
    compare_sids = msg.get("compare_sids")
    if compare_sids:
        sids = compare_sids if isinstance(compare_sids, list) else [s.strip() for s in str(compare_sids).split(",") if s.strip()]
        if sids:
            qs += f"&compare_sid={sids[0]}"
            qs += f"&compare_sids={','.join(sids)}"
    query_val = json.dumps(qs) if sid else "null"

    _theme_names = ["dark", "light", "solarized", "nord"]
    _cfg_theme = get_viewer_theme()
    _default_theme_idx = _theme_names.index(_cfg_theme) if _cfg_theme in _theme_names else 0
    _cfg_rounded = get_viewer_rounded_panes()
    _default_rounded_panes = "true" if _cfg_rounded else "false"

    html = (
        template.replace("__COLORMAPS__", str(_active_colormaps))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
        .replace("__DEFAULT_THEME_IDX__", str(_default_theme_idx))
        .replace("__DEFAULT_ROUNDED_PANES__", _default_rounded_panes)
        .replace("__BODY_CLASS__", "av-loading" if sid else "")
    )

    _write_json({"html": html})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _dispatch(msg_type: str, msg: dict) -> None:
    """Dispatch a non-slice request to the appropriate handler."""
    if msg_type == "register":
        _handle_register(msg)
    elif msg_type == "metadata":
        _handle_metadata(msg)
    elif msg_type == "sessions":
        _handle_sessions()
    elif msg_type == "clearcache":
        _handle_clearcache(msg)
    elif msg_type == "fetch-proxy":
        _handle_fetch_proxy(msg)
    elif msg_type == "get-viewer-html":
        _handle_get_viewer_html(msg)
    else:
        _write_error(f"unknown type: {msg_type}")


def run_stdio_server() -> None:
    """Read JSON from stdin, dispatch handlers, write to stdout.

    Uses raw fd reads + select to coalesce rapid-fire slice requests: before
    processing a slice, we non-blocking peek for newer messages and skip stale
    intermediate frames with a minimal 1×1 response.  No threads — zero
    overhead when the queue is empty.
    """
    _init_luts()

    # Bypass Python's buffered IO so select() accurately reflects available
    # data.  We manage our own read buffer on the raw fd.
    _fd = sys.stdin.fileno()
    _buf = b""

    def _read_line() -> str | None:
        """Block until a complete line is available, return it (or None on EOF)."""
        nonlocal _buf
        while b"\n" not in _buf:
            select.select([_fd], [], [])
            chunk = os.read(_fd, 65536)
            if not chunk:
                return None
            _buf += chunk
        line, _, _buf = _buf.partition(b"\n")
        return line.decode()

    def _try_read_line() -> str | None:
        """Return next line if data is immediately available, else None."""
        nonlocal _buf
        if b"\n" in _buf:
            line, _, _buf = _buf.partition(b"\n")
            return line.decode()
        readable, _, _ = select.select([_fd], [], [], 0)
        if not readable:
            return None
        chunk = os.read(_fd, 65536)
        if not chunk:
            return None
        _buf += chunk
        if b"\n" in _buf:
            line, _, _buf = _buf.partition(b"\n")
            return line.decode()
        return None

    def _parse_line(raw: str) -> dict | None:
        """Parse a stripped line as JSON; write error response on failure."""
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            _write_error(f"invalid JSON: {raw[:200]}")
            return None

    while True:
        raw = _read_line()
        if raw is None:
            break
        msg = _parse_line(raw)
        if msg is None:
            continue

        msg_type = msg.get("type", "slice")

        if msg_type == "slice":
            # Coalesce: peek for newer messages without blocking.  Non-slice
            # requests are queued and processed AFTER slices to preserve FIFO
            # order (the extension matches responses to callbacks by position).
            #
            # Pane-aware coalescing: for the SAME (dim_x, dim_y) pane, only the
            # latest request matters — earlier ones get a skip frame.  For
            # DIFFERENT panes (multiview/qMRI), every pane must be rendered.
            # All responses are emitted in arrival order to maintain FIFO.
            pending_slices = [msg]
            deferred = []
            while True:
                next_raw = _try_read_line()
                if next_raw is None:
                    break
                next_msg = _parse_line(next_raw)
                if next_msg is None:
                    continue
                next_type = next_msg.get("type", "slice")
                if next_type == "slice":
                    pending_slices.append(next_msg)
                else:
                    deferred.append((next_type, next_msg))

            # Find the latest arrival index for each pane
            latest_for_pane: dict = {}
            for i, s in enumerate(pending_slices):
                pane_key = (s.get("dim_x", -1), s.get("dim_y", -1))
                latest_for_pane[pane_key] = i

            # Submit non-stale renders to the thread pool so multi-pane modes
            # (qMRI, multiview, compare) don't serialize on the read loop.
            # Per-cache locks on the Session make concurrent renders safe.
            pool = _get_slice_pool() if len(pending_slices) > 1 else None
            payload_futures: dict[int, _futures.Future] = {}
            for i, s in enumerate(pending_slices):
                pane_key = (s.get("dim_x", -1), s.get("dim_y", -1))
                if latest_for_pane[pane_key] != i:
                    continue
                payload_futures[i] = (
                    pool.submit(_build_slice_payload, s) if pool is not None else None
                )

            # Write responses in arrival order to match the extension's FIFO
            # callback matching.
            for i, s in enumerate(pending_slices):
                pane_key = (s.get("dim_x", -1), s.get("dim_y", -1))
                if latest_for_pane[pane_key] != i:
                    _write_skip_response(s)
                    continue
                try:
                    fut = payload_futures.get(i)
                    payload = fut.result() if fut is not None else _build_slice_payload(s)
                    _write_response(payload)
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    _write_error(str(e))

            for def_type, def_msg in deferred:
                try:
                    _dispatch(def_type, def_msg)
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    _write_error(str(e))
        else:
            try:
                _dispatch(msg_type, msg)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                _write_error(str(e))
