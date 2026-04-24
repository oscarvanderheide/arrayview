"""FastAPI application, REST/WebSocket routes, and HTML templates.

This module was extracted from _app.py during the modular refactor.
"""

# ── Imports ───────────────────────────────────────────────────────

import asyncio
import io
import json
import os
import threading

import numpy as np
from fastapi import (
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
)
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.websockets import WebSocketDisconnect
from importlib.resources import files as _pkg_files

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from arrayview._session import (
    _vprint,
    SHELL_SOCKETS,
    PENDING_SESSIONS,
    _render,
    _schedule_prefetch,
    Session,
    SESSIONS,
    COLORMAPS,

    HEAVY_OP_LIMIT_BYTES,
    _estimate_array_bytes,
)
import arrayview._session as _session_mod  # for mutable VIEWER_SOCKETS

from arrayview._render import (
    LUTS,
    COLORMAP_GRADIENT_STOPS,
    COMPLEX_MODES,
    REAL_MODES,
    mosaic_shape,
    _compute_vmin_vmax,
    extract_slice,
    extract_projection,
    PROJECTION_OPS,
    apply_complex_mode,
    _prepare_display,
    _init_luts,
    _ensure_lut,
    _setup_rgb,
    render_rgb_rgba,
    render_rgba,
    render_projection_rgba,
    render_mosaic,
    _run_preload,
)

from arrayview._analysis import (
    _build_metadata,
    _safe_float,
)
from arrayview._diff import (
    _compute_diff,
    _diff_histogram,
    _render_diff_rgba,
    _render_normalized,
    _render_normalized_mosaic,
)
from arrayview._io import load_data
from arrayview._overlays import _composite_overlays
from arrayview._routes_analysis import register_analysis_routes
from arrayview._routes_loading import register_loading_routes
from arrayview._routes_persistence import (
    _CROP_LOCK,
    _CROP_STATE,
    register_persistence_routes,
)
from arrayview._routes_export import register_export_routes
from arrayview._routes_preload import register_preload_routes
from arrayview._routes_query import register_query_routes
from arrayview._routes_rendering import register_rendering_routes
from arrayview._routes_segmentation import register_segmentation_routes
from arrayview._routes_state import register_state_routes
from arrayview._routes_vectorfield import register_vectorfield_routes
from arrayview._config import get_viewer_colormaps, get_viewer_rounded_panes, get_viewer_theme
from arrayview._vectorfield import (
    _MAX_VFIELD_ARROWS,
    _compute_vfield_arrows,
    _configure_vectorfield,
    _get_vfield_layout,
    _resolve_vfield_layout,
    _vfield_counts_for_level,
    _vfield_n_times,
)


# ── Lazy PIL Import ───────────────────────────────────────────────

_pil_image_mod = None
_pil_imageops_mod = None


def _pil_image():
    """Lazy PIL.Image import."""
    global _pil_image_mod
    if _pil_image_mod is None:
        from PIL import Image

        _pil_image_mod = Image
    return _pil_image_mod


def _pil_imageops():
    """Lazy PIL.ImageOps import."""
    global _pil_imageops_mod
    if _pil_imageops_mod is None:
        from PIL import ImageOps

        _pil_imageops_mod = ImageOps
    return _pil_imageops_mod


# ── FastAPI Application ───────────────────────────────────────────

app = FastAPI()


@app.exception_handler(Exception)
async def _generic_exception_handler(request: Request, exc: Exception):
    import traceback

    _vprint(
        f"[ArrayView] Unhandled error on {request.url.path}: {exc}\n"
        + traceback.format_exc(),
        flush=True,
    )
    return JSONResponse(
        status_code=500, content={"error": str(exc), "type": type(exc).__name__}
    )


# ── HTML Templates ────────────────────────────────────────────────

_SHELL_HTML: str = (
    _pkg_files("arrayview").joinpath("_shell.html").read_text(encoding="utf-8")
)
_VIEWER_HTML_TEMPLATE: str = (
    _pkg_files("arrayview").joinpath("_viewer.html").read_text(encoding="utf-8")
)

_GSAP_JS: str = (
    _pkg_files("arrayview").joinpath("gsap.min.js").read_text(encoding="utf-8")
)


@app.get("/gsap.min.js")
def serve_gsap():
    """Serve vendored GSAP library (browser caches via ETag)."""
    return Response(content=_GSAP_JS, media_type="application/javascript")


# ── WebSocket Routes (Shell and Viewer) ───────────────────────────

async def _notify_shells(sid, name, url=None, wait: bool = True) -> bool:
    """Push a new-tab message to all connected webview shell windows.

    Returns True if at least one shell received the message.
    wait=True: poll up to 2 s for a shell to connect (used when a window was just opened).
    wait=False: send immediately to whatever shells are currently connected.
    """
    if wait:
        for _ in range(200):  # Wait up to 2 s for window to connect
            if SHELL_SOCKETS:
                break
            await asyncio.sleep(0.01)
    msg = {"action": "new_tab", "sid": sid, "name": name}
    if url:
        msg["url"] = url
    notified = False
    for ws in SHELL_SOCKETS.copy():
        try:
            await ws.send_json(msg)
            notified = True
        except Exception:
            pass
    return notified


@app.websocket("/ws/shell")
async def shell_websocket(ws: WebSocket):
    await ws.accept()
    SHELL_SOCKETS.append(ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "close":
                sid = msg.get("sid")
                if sid in SESSIONS:
                    SESSIONS[sid].reset_caches()
                    SESSIONS[sid].data = None
                    del SESSIONS[sid]
                    with _CROP_LOCK:
                        _CROP_STATE.pop(sid, None)
    except Exception:
        pass
    finally:
        if ws in SHELL_SOCKETS:
            SHELL_SOCKETS.remove(ws)


@app.websocket("/ws/{sid}")
async def websocket_endpoint(ws: WebSocket, sid: str):
    session = SESSIONS.get(sid)
    # Wait for pending session (data still loading in background thread)
    if not session and sid in PENDING_SESSIONS:
        for _ in range(1200):  # up to 120 s
            await asyncio.sleep(0.1)
            session = SESSIONS.get(sid)
            if session:
                break
    if not session:
        await ws.close()
        return

    await ws.accept()

    # Push metadata as first message — saves the client an HTTP round-trip.
    try:
        meta = _build_metadata(session)
        await ws.send_json({"type": "metadata", **meta})
    except Exception:
        pass  # non-fatal; client falls back to HTTP /metadata endpoint

    _session_mod.VIEWER_SOCKETS += 1
    _session_mod.VIEWER_SIDS.add(sid)
    loop = asyncio.get_running_loop()

    # Single-slot message queue: the receiver always overwrites the pending
    # slot with the newest request so stale intermediate frames are skipped.
    _pending: asyncio.Queue[dict | None] = asyncio.Queue()

    async def _receiver() -> None:
        try:
            while True:
                msg = await ws.receive_json()
                try:
                    _pending.get_nowait()  # evict any still-pending request
                except asyncio.QueueEmpty:
                    pass
                await _pending.put(msg)
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            _pending.put_nowait(None)  # sentinel — always fires, even on cancel

    recv_task = asyncio.create_task(_receiver())
    try:
        while True:
            msg = await _pending.get()
            if msg is None:
                break  # disconnect or receiver error

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
            direction = int(msg.get("direction", 1))  # scroll direction (+1/-1)
            slice_dim = int(msg.get("slice_dim", -1))
            # S4: canvas display resolution hint (device pixels).  When set and the
            # rendered slice is larger, the server thumbnails it down before sending
            # so we skip transmitting pixels the browser discards anyway.
            canvas_w = int(msg.get("canvas_w", 0))
            canvas_h = int(msg.get("canvas_h", 0))
            _mc = msg.get("mosaic_cols")
            mosaic_cols = int(_mc) if _mc is not None else None
            projection_mode = int(msg.get("projection_mode", 0))
            projection_dim = int(msg.get("projection_dim", -1))

            if session.rgb_axis is not None:
                # RGB/RGBA mode — render directly from channel data; skip colormap.
                rgba = await _render(
                    loop,
                    lambda: render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple)),
                )
                h, w = rgba.shape[:2]
                vmin, vmax = 0.0, 255.0
            elif dim_z >= 0:
                rgba = await _render(
                    loop,
                    lambda: render_mosaic(
                        session,
                        dim_x,
                        dim_y,
                        dim_z,
                        idx_tuple,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                        mosaic_cols=mosaic_cols,
                        vmin_override=vmin_override,
                        vmax_override=vmax_override,
                    ),
                )
                h, w = rgba.shape[:2]
                raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
            elif projection_mode > 0 and projection_dim >= 0:
                # Statistical projection mode
                _pm = projection_mode
                _pd = projection_dim
                rgba = await _render(
                    loop,
                    lambda: render_projection_rgba(
                        session,
                        dim_x,
                        dim_y,
                        idx_tuple,
                        _pd,
                        _pm,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                        vmin_override,
                        vmax_override,
                    ),
                )
                h, w = rgba.shape[:2]
                raw = extract_projection(
                    session, dim_x, dim_y, list(idx_tuple), _pd, _pm
                )
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )
            else:
                rgba = await _render(
                    loop,
                    lambda: render_rgba(
                        session,
                        dim_x,
                        dim_y,
                        idx_tuple,
                        colormap,
                        dr,
                        complex_mode,
                        log_scale,
                        vmin_override,
                        vmax_override,
                    ),
                )
                h, w = rgba.shape[:2]
                raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
                _, vmin, vmax = _prepare_display(
                    session,
                    raw,
                    complex_mode,
                    dr,
                    log_scale,
                    vmin_override=vmin_override,
                    vmax_override=vmax_override,
                )

                # Overlay compositing (one or more segmentation masks)
                overlay_sid = msg.get("overlay_sid")
                overlay_colors = msg.get("overlay_colors")
                overlay_alpha = float(msg.get("overlay_alpha", 0.45))
                rgba = _composite_overlays(
                    rgba,
                    overlay_sid,
                    overlay_colors,
                    overlay_alpha,
                    dim_x,
                    dim_y,
                    idx_tuple,
                    (h, w),
                )

            header = np.array([seq, w, h], dtype=np.uint32).tobytes()
            vminmax = np.array([vmin, vmax], dtype=np.float32).tobytes()
            # S4: thumbnail downsample to avoid transmitting pixels the browser
            # discards.  Only shrinks (PIL.thumbnail never enlarges).
            if canvas_w and canvas_h and (w > canvas_w or h > canvas_h):
                pil = _pil_image().fromarray(
                    rgba.astype(np.uint8) if rgba.dtype != np.uint8 else rgba,
                    mode="RGBA",
                )
                pil.thumbnail((canvas_w, canvas_h), _pil_image().LANCZOS)
                rgba = np.array(pil)
                h, w = rgba.shape[:2]
                header = np.array([seq, w, h], dtype=np.uint32).tobytes()
            payload = header + vminmax + rgba.tobytes()
            # Append vectorfield binary trailer when a vfield is attached
            if session.vfield is not None:
                vf_density = int(msg.get("vf_density", 0))
                vf_t = int(msg.get("vf_t", 0))
                vf_result = _compute_vfield_arrows(
                    session, dim_x, dim_y, idx_tuple,
                    t_index=vf_t, density_offset=vf_density,
                )
                if vf_result is not None:
                    arrows = vf_result["arrows"]
                    vf_hdr = np.array([len(arrows), vf_result["stride"]], dtype=np.uint32).tobytes()
                    vf_scale = np.array([vf_result["scale"]], dtype=np.float32).tobytes()
                    payload += vf_hdr + vf_scale + arrows.tobytes()
            await ws.send_bytes(payload)

            # Warm neighbor slices in the background (Phase 3 prefetch)
            if slice_dim >= 0 and not (dim_z >= 0):
                _schedule_prefetch(
                    session, dim_x, dim_y, list(idx_tuple), slice_dim, direction
                )
    except Exception as _ws_exc:
        import traceback

        _vprint(f"[ArrayView] WS/{sid[:8]}: {_ws_exc}", flush=True)
        traceback.print_exc()
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass
        _session_mod.VIEWER_SOCKETS = max(0, _session_mod.VIEWER_SOCKETS - 1)
        # Note: we don't remove from VIEWER_SIDS here because the set is used
        # only to check if a session was *ever* connected, not currently connected.



def get_session_or_404(sid: str) -> "Session":
    """FastAPI dependency: fetch session by sid or raise 404."""
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


register_analysis_routes(app, get_session_or_404)
register_loading_routes(app, notify_shells=_notify_shells, setup_rgb=_setup_rgb)
register_persistence_routes(app)
register_segmentation_routes(app, get_session_or_404)
register_state_routes(app, get_session_or_404)
register_export_routes(app, get_session_or_404=get_session_or_404, pil_image=_pil_image)
register_preload_routes(app)
register_vectorfield_routes(app)
register_rendering_routes(app, get_session_or_404=get_session_or_404)
register_query_routes(
    app,
    get_session_or_404=get_session_or_404,
    pil_image=_pil_image,
    pil_imageops=_pil_imageops,
)

# ── REST Routes: Cache, Metadata, and Session Management ─────────


@app.get("/colormap/{name}")
def get_colormap(name: str):
    """Validate a matplotlib colormap name and return its gradient stops."""
    if not _ensure_lut(name):
        return Response(status_code=404)
    return {"ok": True, "gradient_stops": COLORMAP_GRADIENT_STOPS[name]}


# ── REST Routes: Slice Rendering, Diff, and Oblique ──────────────


@app.get("/shell")
def get_shell():
    """Tabbed shell UI for native webview windows."""
    return HTMLResponse(content=_SHELL_HTML)


@app.get("/ping")
def ping():
    """Health marker so clients can verify this is an ArrayView server."""
    import socket

    return {
        "ok": True,
        "service": "arrayview",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "viewer_sockets": _session_mod.VIEWER_SOCKETS,
    }


# ── Root UI Route ─────────────────────────────────────────────────


@app.get("/")
def get_ui(sid: str = None):
    """Viewer page."""
    # VS Code's asExternalUri() strips query parameters, so ?sid= is often lost
    # before the page loads.  Embed the SID directly in the HTML so the viewer
    # JS can find it regardless of the URL.
    if not sid:
        # No sid in URL — VS Code strips the query string before loading the
        # page, so ?sid= is lost.  Inject the latest valid session
        # server-side so the viewer JS can find it regardless of the URL.
        if SESSIONS:
            latest_sid = list(SESSIONS.keys())[-1]
            query_val = json.dumps(f"?sid={latest_sid}")
        else:
            query_val = "null"  # viewer will show "Session not found or expired"
    else:
        # sid is present in the URL (valid or not) — let the JS fetch /metadata/{sid}
        # and handle errors itself (shows "Session not found or expired" on 404).
        query_val = "null"
    _init_luts()
    _cfg_colormaps = get_viewer_colormaps()
    _active_colormaps = _cfg_colormaps if _cfg_colormaps is not None else COLORMAPS
    _theme_names = ["dark", "light", "solarized", "nord"]
    _cfg_theme = get_viewer_theme()
    _default_theme_idx = _theme_names.index(_cfg_theme) if _cfg_theme in _theme_names else 0
    _cfg_rounded = get_viewer_rounded_panes()
    _default_rounded_panes = "true" if _cfg_rounded else "false"
    html = (
        _VIEWER_HTML_TEMPLATE.replace("__COLORMAPS__", str(_active_colormaps))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
        .replace("__DEFAULT_THEME_IDX__", str(_default_theme_idx))
        .replace("__DEFAULT_ROUNDED_PANES__", _default_rounded_panes)
        .replace("__BODY_CLASS__", "av-loading" if sid else "")
    )
    headers = {"Cache-Control": "no-store"}
    return HTMLResponse(content=html, headers=headers)
