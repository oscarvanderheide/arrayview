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
from arrayview._routes_segmentation import register_segmentation_routes
from arrayview._routes_state import register_state_routes
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


@app.get("/vectorfield/{sid}")
def get_vectorfield(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    t_index: int = 0,
    density_offset: int = 0,
):
    """Return downsampled deformation vector field arrows for the current 2-D view."""
    session = SESSIONS.get(sid)
    if not session or session.vfield is None:
        return Response(status_code=404)
    try:
        idx_tuple = tuple(int(x) for x in indices.split(","))
        result = _compute_vfield_arrows(session, dim_x, dim_y, idx_tuple, t_index, density_offset)
        if result is None:
            return Response(status_code=404)
        return {"arrows": result["arrows"].tolist(), "scale": result["scale"], "stride": result["stride"]}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


@app.post("/attach_vectorfield")
async def attach_vectorfield(request: Request):
    """Attach a vector field to an existing session (for the existing-server code path)."""
    body = await request.json()
    sid = str(body["sid"])
    filepath = str(body["filepath"])
    components_dim = body.get("components_dim")
    session = SESSIONS.get(sid)
    if not session:
        return {"error": f"session {sid} not found"}
    try:
        vf_data = load_data(filepath)
        layout = _configure_vectorfield(session, vf_data, components_dim)
        return {"ok": True, "components_dim": layout["components_dim"]}
    except Exception as e:
        return {"error": str(e)}


# ── REST Routes: Slice Rendering, Diff, and Oblique ──────────────


@app.get("/slice/{sid}")
def get_slice(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    colormap: str = "gray",
    dr: int = 1,
    slice_dim: int = -1,
    dim_z: int = -1,
    complex_mode: int = 0,
    log_scale: bool = False,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
    overlay_sid: str | None = None,
    overlay_colors: str | None = None,
    overlay_alpha: float = 0.45,
    mosaic_cols: int | None = None,
    projection_mode: int = 0,
    projection_dim: int = -1,
    session: "Session" = Depends(get_session_or_404),
):
    idx_tuple = tuple(int(x) for x in indices.split(","))
    if projection_mode > 0 and projection_dim >= 0:
        rgba = render_projection_rgba(
            session,
            dim_x,
            dim_y,
            idx_tuple,
            projection_dim,
            projection_mode,
            colormap,
            dr,
            complex_mode,
            log_scale,
            vmin_override,
            vmax_override,
        )
        raw = extract_projection(
            session, dim_x, dim_y, list(idx_tuple), projection_dim, projection_mode
        )
        _, vmin, vmax = _prepare_display(
            session, raw, complex_mode, dr, log_scale,
            vmin_override=vmin_override, vmax_override=vmax_override,
        )
    elif dim_z >= 0:
        rgba = render_mosaic(
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
        )
        if vmin_override is not None and vmax_override is not None:
            vmin, vmax = float(vmin_override), float(vmax_override)
        else:
            idx_norm = list(idx_tuple)
            idx_norm[dim_z] = 0
            frames_raw = [
                extract_slice(
                    session,
                    dim_x,
                    dim_y,
                    [i if j == dim_z else idx_tuple[j] for j in range(len(session.shape))],
                )
                for i in range(session.shape[dim_z])
            ]
            frames = [apply_complex_mode(frame, complex_mode) for frame in frames_raw]
            if log_scale:
                frames = [np.log1p(np.abs(frame)).astype(np.float32) for frame in frames]
                all_data = np.stack(frames)
                vmin = float(np.percentile(all_data, 1))
                vmax = float(np.percentile(all_data, 99))
            else:
                vmin, vmax = _compute_vmin_vmax(session, np.stack(frames), dr, complex_mode)
    else:
        if session.rgb_axis is not None:
            rgba = render_rgb_rgba(session, dim_x, dim_y, list(idx_tuple))
            vmin, vmax = 0.0, 255.0
        else:
            rgba = render_rgba(
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
            )
            rgba = _composite_overlays(
                rgba,
                overlay_sid,
                overlay_colors,
                overlay_alpha,
                dim_x,
                dim_y,
                idx_tuple,
                rgba.shape[:2],
            )
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
    img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "max-age=300",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
        },
    )


@app.get("/diff/{sid_a}/{sid_b}")
def get_diff(
    sid_a: str,
    sid_b: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    dim_z: int = -1,
    dr: int = 1,
    complex_mode: int = 0,
    log_scale: bool = False,
    diff_mode: int = 1,
    diff_colormap: str = "",
    vmin_override: float = None,
    vmax_override: float = None,
):
    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        return Response(status_code=404)
    try:
        raw, vmin, vmax, colormap, nan_mask = _compute_diff(
            session_a,
            session_b,
            dim_x,
            dim_y,
            indices,
            dim_z,
            dr,
            complex_mode,
            log_scale,
            diff_mode,
        )
    except Exception:
        return Response(status_code=422)
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
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
            "X-ArrayView-Colormap": colormap,
        },
    )


@app.get("/diff-histogram/{sid_a}/{sid_b}")
def get_diff_histogram(
    sid_a: str,
    sid_b: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    dim_z: int = -1,
    dr: int = 1,
    complex_mode: int = 0,
    log_scale: bool = False,
    diff_mode: int = 1,
    bins: int = 64,
):
    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        return Response(status_code=404)
    try:
        raw, _, _, _, _ = _compute_diff(
            session_a,
            session_b,
            dim_x,
            dim_y,
            indices,
            dim_z,
            dr,
            complex_mode,
            log_scale,
            diff_mode,
        )
    except Exception:
        return Response(status_code=422)
    return _diff_histogram(raw, bins)


@app.get("/oblique/{sid}")
def get_oblique(
    sid: str,
    center: str,  # comma-sep floats — full N-dim index (e.g. "32.0,30.0,35.0")
    basis_h: str,  # 3 floats in mv_dims order, unit vector → horizontal
    basis_v: str,  # 3 floats in mv_dims order, unit vector → vertical
    mv_dims: str,  # 3 ints: which array dims are the 3 spatial dims
    size_w: int,
    size_h: int,
    colormap: str = "gray",
    dr: int = 1,
    complex_mode: int = 0,
    log_scale: bool = False,
    vmin_override: float | None = None,
    vmax_override: float | None = None,
    quality: str = "full",
    session: "Session" = Depends(get_session_or_404),
):
    """Render an oblique (arbitrarily-oriented) slice through a 3-D volume."""

    from scipy.ndimage import map_coordinates

    ctr = [float(x) for x in center.split(",")]
    bh = [float(x) for x in basis_h.split(",")]
    bv = [float(x) for x in basis_v.split(",")]
    dims = [int(x) for x in mv_dims.split(",")]

    draft = quality == "draft"
    grid_w = size_w // 2 if draft else size_w
    grid_h = size_h // 2 if draft else size_h

    ndim = len(session.shape)
    hw, hh = size_w / 2.0, size_h / 2.0
    s_arr = np.linspace(-hw, hw, grid_w, dtype=np.float64)
    t_arr = np.linspace(-hh, hh, grid_h, dtype=np.float64)
    ss, tt = np.meshgrid(s_arr, t_arr)  # (grid_h, grid_w)

    # Build full N-dim coordinate grids; non-spatial dims use fixed center value
    coords = np.empty((ndim, grid_h, grid_w), dtype=np.float64)
    for ai in range(ndim):
        if ai in dims:
            ji = dims.index(ai)
            coords[ai] = ctr[ai] + ss * bh[ji] + tt * bv[ji]
        else:
            coords[ai] = ctr[ai]

    data = session.data
    if np.iscomplexobj(data):
        if complex_mode == 1:
            data_f = np.angle(data).astype(np.float32)
        elif complex_mode == 2:
            data_f = data.real.astype(np.float32)
        elif complex_mode == 3:
            data_f = data.imag.astype(np.float32)
        else:
            data_f = np.abs(data).astype(np.float32)
    else:
        data_f = np.nan_to_num(np.asarray(data, dtype=np.float32))

    sampled = map_coordinates(
        data_f, coords, order=0 if draft else 1, mode="constant", cval=0.0
    ).astype(np.float32)

    if log_scale:
        sampled = np.log1p(np.abs(sampled)).astype(np.float32)

    if vmin_override is not None and vmax_override is not None:
        vmin, vmax = float(vmin_override), float(vmax_override)
    else:
        vmin, vmax = _compute_vmin_vmax(session, sampled, dr, complex_mode)

    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    if vmax > vmin:
        normalized = np.clip((sampled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(sampled)
    rgba = lut[(normalized * 255).astype(np.uint8)]

    img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
    if draft and (grid_w != size_w or grid_h != size_h):
        img = img.resize((size_w, size_h), _pil_image().NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-ArrayView-Vmin": str(vmin),
            "X-ArrayView-Vmax": str(vmax),
        },
    )


@app.get("/oblique_vectorfield/{sid}")
def get_oblique_vectorfield(
    sid: str,
    center: str,
    basis_h: str,
    basis_v: str,
    mv_dims: str,
    size_w: int,
    size_h: int,
    density_offset: int = 0,
    t_index: int = 0,
):
    """Return vector field arrows projected onto an oblique slice plane."""
    session = SESSIONS.get(sid)
    if not session or session.vfield is None:
        return Response(status_code=404)

    layout = _get_vfield_layout(session)
    if layout is None:
        return Response(status_code=404)

    try:
        from scipy.ndimage import map_coordinates

        ctr = [float(x) for x in center.split(",")]
        bh = np.array([float(x) for x in basis_h.split(",")], dtype=np.float64)
        bv = np.array([float(x) for x in basis_v.split(",")], dtype=np.float64)
        dims = [int(x) for x in mv_dims.split(",")]

        vf = session.vfield
        spatial_axes = tuple(int(ax) for ax in layout["spatial_axes"])
        comp_dim = int(layout["components_dim"])
        time_dim = layout["time_dim"]

        # Select time slice
        slices = [slice(None)] * vf.ndim
        if time_dim is not None:
            t = max(0, min(int(layout["n_times"]) - 1, t_index))
            slices[int(time_dim)] = t

        # After time selection, extract all spatial+component data
        vf_t = np.asarray(vf[tuple(slices)], dtype=np.float32)

        # Sampling (same logic as _compute_vfield_arrows)
        H, W = size_h, size_w
        base_stride = max(1, max(H, W) // 32)
        n_arrows_target, effective_stride, use_grid = _vfield_counts_for_level(
            density_offset, H, W, base_stride
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
        n_arrows = gy.size

        # Convert 2D sample positions to 3D world coordinates
        hw, hh = W / 2.0, H / 2.0
        sx = gx.astype(np.float64) - hw  # pixel offset from center
        sy = gy.astype(np.float64) - hh

        # Determine axis positions in the time-sliced array
        remaining_axes = [ax for ax, sl in enumerate(slices) if isinstance(sl, slice)]
        axis_map = {ax: i for i, ax in enumerate(remaining_axes)}

        n_comp = vf_t.shape[axis_map[comp_dim]]
        n_spatial = len(spatial_axes)
        # Component offset: components align to last n_comp spatial dims
        comp_offset = n_spatial - n_comp

        # Build N-dim coordinates for each sample point
        ndim = len(ctr)
        coords_3d = np.empty((ndim, n_arrows), dtype=np.float64)
        for ai in range(ndim):
            if ai in dims:
                ji = dims.index(ai)
                coords_3d[ai] = ctr[ai] + sx * bh[ji] + sy * bv[ji]
            else:
                coords_3d[ai] = ctr[ai]

        # For map_coordinates on each component volume, pre-compute the
        # spatial coordinate array (same for all components)
        spatial_remaining = [ax for ax in remaining_axes if ax != comp_dim]
        sp_map = {ax: i for i, ax in enumerate(spatial_remaining)}
        mc_coords = np.empty((len(spatial_remaining), n_arrows), dtype=np.float64)
        for ax in spatial_remaining:
            mc_coords[sp_map[ax]] = coords_3d[ax] if ax < ndim else 0.0

        # Sample each displacement component and project onto the oblique plane
        vecs_h = np.zeros(n_arrows, dtype=np.float64)
        vecs_v = np.zeros(n_arrows, dtype=np.float64)

        for ci in range(n_comp):
            # Select this component from vf_t
            comp_slices = [slice(None)] * vf_t.ndim
            comp_slices[axis_map[comp_dim]] = ci
            vf_comp = vf_t[tuple(comp_slices)]

            sampled = map_coordinates(
                vf_comp, mc_coords, order=1, mode="constant", cval=0.0
            )

            # Component ci corresponds to spatial dim index (ci + comp_offset)
            spatial_idx = ci + comp_offset
            if spatial_idx < n_spatial:
                sa = spatial_axes[spatial_idx]  # array dim this component displaces
                if sa in dims:
                    ji = dims.index(sa)
                    vecs_h += sampled * bh[ji]
                    vecs_v += sampled * bv[ji]

        vx_s = vecs_h.astype(np.float32)
        vy_s = vecs_v.astype(np.float32)

        # Scale (same as _compute_vfield_arrows)
        mags = np.sqrt(vx_s ** 2 + vy_s ** 2)
        nonzero = mags[mags > 0]
        p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        scale = float(effective_stride * 0.75 / max(p95, 1e-9))

        arrows = np.column_stack([gx, gy, vx_s, vy_s]).astype(np.float32)
        return {
            "arrows": arrows.tolist(),
            "scale": scale,
            "stride": int(round(effective_stride)),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


@app.get("/grid/{sid}")
def get_grid(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
    session: "Session" = Depends(get_session_or_404),
):
    if session.rgb_axis is not None:
        return JSONResponse(
            status_code=400, content={"error": "not supported for RGB sessions"}
        )
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]

    # Phase 4 guardrail: grid stacks all n slices — block for large data
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4
    frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
    if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = frame_bytes * n // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Grid blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                    "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                ),
                "too_large": True,
            },
        )

    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(session, dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    vmin = float(np.percentile(all_data, 1))
    vmax = float(np.percentile(all_data, 99))

    rows, cols = mosaic_shape(n)
    H, W = frames[0].shape
    GAP = 2
    total_h = rows * H + (rows - 1) * GAP
    total_w = cols * W + (cols - 1) * GAP
    grid = np.full((total_h, total_w), np.nan, dtype=np.float32)
    for k in range(n):
        r, c = divmod(k, cols)
        grid[r * (H + GAP) : r * (H + GAP) + H, c * (W + GAP) : c * (W + GAP) + W] = (
            all_data[k]
        )

    nan_mask = np.isnan(grid)
    filled = np.where(nan_mask, vmin, grid)
    if vmax > vmin:
        normalized = np.clip((filled - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(filled)

    _init_luts()
    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    rgba[nan_mask] = [22, 22, 22, 255]
    img = _pil_image().fromarray(rgba[:, :, :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/gif/{sid}")
def get_gif(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
    session: "Session" = Depends(get_session_or_404),
):
    if session.rgb_axis is not None:
        return JSONResponse(
            status_code=400, content={"error": "not supported for RGB sessions"}
        )
    idx_list = [int(x) for x in indices.split(",")]
    n = session.shape[slice_dim]

    # Phase 4 guardrail: GIF stacks all n slices — block for large data
    try:
        itemsize = np.dtype(session.data.dtype).itemsize
    except Exception:
        itemsize = 4
    frame_bytes = session.shape[dim_y] * session.shape[dim_x] * itemsize
    if frame_bytes * n > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = frame_bytes * n // (1024 * 1024)
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"GIF blocked: would stack ~{est_mb} MB (limit {limit_mb} MB). "
                    "Increase ARRAYVIEW_HEAVY_OP_LIMIT_MB to override."
                ),
                "too_large": True,
            },
        )

    frames = []
    for i in range(n):
        idx_list[slice_dim] = i
        frames.append(extract_slice(session, dim_x, dim_y, idx_list))

    all_data = np.stack(frames)
    vmin = float(np.percentile(all_data, 1))
    vmax = float(np.percentile(all_data, 99))

    _init_luts()
    lut = LUTS.get(colormap if colormap in LUTS else "gray", LUTS["gray"])
    gif_frames = []
    for frame in frames:
        if vmax > vmin:
            normalized = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
        else:
            normalized = np.zeros_like(frame)
        rgba = lut[(normalized * 255).astype(np.uint8)]
        gif_frames.append(_pil_image().fromarray(rgba[:, :, :3], mode="RGB"))

    buf = io.BytesIO()
    gif_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=100,
    )
    return Response(content=buf.getvalue(), media_type="image/gif")


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
