"""FastAPI application, REST/WebSocket routes, and HTML templates.

This module was extracted from _app.py during the modular refactor.
"""

import asyncio
import io
import itertools
import json
import math
import os
import threading

import numpy as np
from fastapi import (
    Body,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
)
from fastapi.responses import HTMLResponse, JSONResponse
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
    DR_PERCENTILES,
    DR_LABELS,
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
    _extract_overlay_mask,
    _composite_overlay_mask,
    _overlay_is_label_map,
    render_mosaic,
    _run_preload,
)

from arrayview._io import load_data, _SUPPORTED_EXTS, _peek_file_shape
from arrayview._config import get_viewer_colormaps


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
    session: Session, vf_data, components_dim: int | None = None
) -> dict[str, object]:
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


def _get_vfield_layout(session: Session) -> dict[str, object] | None:
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

# ---------------------------------------------------------------------------
# Lazy PIL import (only needed for JPEG/PNG/GIF encoding in routes)
# ---------------------------------------------------------------------------
_pil_image_mod = None


def _pil_image():
    """Lazy PIL.Image import."""
    global _pil_image_mod
    if _pil_image_mod is None:
        from PIL import Image

        _pil_image_mod = Image
    return _pil_image_mod


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _parse_hex_color(hex_str: str) -> np.ndarray | None:
    """Parse a 6-char hex string like 'ff4444' into a uint8 RGB array, or None."""
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return np.array(
            [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.uint8
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
    """Composite one or more overlays onto rgba.  overlay_sid_str is comma-separated."""
    if not overlay_sid_str:
        return rgba
    sids = [s.strip() for s in overlay_sid_str.split(",") if s.strip()]
    colors_raw = (
        [c.strip() for c in overlay_colors_str.split(",")] if overlay_colors_str else []
    )
    for i, sid in enumerate(sids):
        color = _parse_hex_color(colors_raw[i]) if i < len(colors_raw) else None
        ov_raw = _extract_overlay_mask(
            sid, dim_x, dim_y, idx_tuple, expected_shape=shape_hw
        )
        rgba = _composite_overlay_mask(
            rgba,
            ov_raw,
            alpha=overlay_alpha,
            is_label=_overlay_is_label_map(sid, ov_raw),
            override_color=color,
        )
    return rgba


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# HTML Templates (loaded once at import time from package files)
# ---------------------------------------------------------------------------
_SHELL_HTML: str = (
    _pkg_files("arrayview").joinpath("_shell.html").read_text(encoding="utf-8")
)
_VIEWER_HTML_TEMPLATE: str = (
    _pkg_files("arrayview").joinpath("_viewer.html").read_text(encoding="utf-8")
)


# ---------------------------------------------------------------------------
# Shell WebSocket for Webview Tab Management
# ---------------------------------------------------------------------------
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
                    SESSIONS[sid].raw_cache.clear()
                    SESSIONS[sid].rgba_cache.clear()
                    SESSIONS[sid].mosaic_cache.clear()
                    SESSIONS[sid]._raw_bytes = SESSIONS[sid]._rgba_bytes = SESSIONS[
                        sid
                    ]._mosaic_bytes = 0
                    SESSIONS[sid].data = None
                    del SESSIONS[sid]
    except Exception:
        pass
    finally:
        if ws in SHELL_SOCKETS:
            SHELL_SOCKETS.remove(ws)


@app.websocket("/ws/{sid}")
async def websocket_endpoint(ws: WebSocket, sid: str):
    session = SESSIONS.get(sid)
    if not session:
        await ws.close()
        return

    await ws.accept()
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


@app.get("/clearcache/{sid}")
def clear_cache(sid: str):
    session = SESSIONS.get(sid)
    if session:
        session.raw_cache.clear()
        session.rgba_cache.clear()
        session.mosaic_cache.clear()
        session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    return {"status": "ok"}


@app.get("/data_version/{sid}")
def get_data_version(sid: str):
    """Return the current data version for a session (incremented on reload)."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    return {"version": getattr(session, "data_version", 0)}


@app.post("/reload/{sid}")
async def reload_session(sid: str):
    """Reload session data from its source file (used by --watch mode).

    Clears caches and bumps data_version so polling clients know to re-render.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    filepath = session.filepath
    if not filepath or not os.path.isfile(filepath):
        return {"error": "session has no reloadable filepath"}
    try:
        data = await asyncio.to_thread(load_data, filepath)
    except Exception as e:
        return {"error": str(e)}
    # Replace data and recompute stats in-place (keep sid and name).
    session.data = data
    session.shape = data.shape
    session.spatial_shape = data.shape
    session.rgb_axis = None
    session.fft_original_data = None
    session.fft_axes = None
    # Clear all render caches.
    session.raw_cache.clear()
    session.rgba_cache.clear()
    session.mosaic_cache.clear()
    session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    await asyncio.to_thread(session.compute_global_stats)
    session.data_version = getattr(session, "data_version", 0) + 1
    return {"version": session.data_version}


@app.post("/update/{sid}")
async def update_session(sid: str, request: Request):
    """Replace session data with a new numpy array sent as raw .npy bytes.

    The request body must be a valid .npy file (as produced by ``np.save``).
    Clears caches and bumps ``data_version`` so polling clients re-render.
    Used by ``ViewHandle.update(arr)`` in Python.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    raw = await request.body()
    if not raw:
        return Response(status_code=400, content="empty body")
    try:
        arr = np.load(io.BytesIO(raw))
    except Exception as e:
        return Response(status_code=400, content=f"failed to decode array: {e}")
    session.data = arr
    session.shape = arr.shape
    session.spatial_shape = arr.shape
    session.rgb_axis = None
    session.fft_original_data = None
    session.fft_axes = None
    session.raw_cache.clear()
    session.rgba_cache.clear()
    session.mosaic_cache.clear()
    session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    await asyncio.to_thread(session.compute_global_stats)
    session.data_version = getattr(session, "data_version", 0) + 1
    return {"version": session.data_version}


@app.get("/cache_info/{sid}")
def cache_info(sid: str):
    """Phase 5: debug endpoint — returns per-session cache usage and budgets."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    return {
        "raw_cache": {
            "entries": len(session.raw_cache),
            "used_bytes": session._raw_bytes,
            "budget_bytes": session.RAW_CACHE_BYTES,
            "used_mb": round(session._raw_bytes / 1e6, 2),
            "budget_mb": round(session.RAW_CACHE_BYTES / 1e6, 2),
        },
        "rgba_cache": {
            "entries": len(session.rgba_cache),
            "used_bytes": session._rgba_bytes,
            "budget_bytes": session.RGBA_CACHE_BYTES,
            "used_mb": round(session._rgba_bytes / 1e6, 2),
            "budget_mb": round(session.RGBA_CACHE_BYTES / 1e6, 2),
        },
        "mosaic_cache": {
            "entries": len(session.mosaic_cache),
            "used_bytes": session._mosaic_bytes,
            "budget_bytes": session.MOSAIC_CACHE_BYTES,
            "used_mb": round(session._mosaic_bytes / 1e6, 2),
            "budget_mb": round(session.MOSAIC_CACHE_BYTES / 1e6, 2),
        },
        "heavy_op_limit_mb": round(HEAVY_OP_LIMIT_BYTES / 1e6, 1),
    }


@app.get("/colormap/{name}")
def get_colormap(name: str):
    """Validate a matplotlib colormap name and return its gradient stops."""
    if not _ensure_lut(name):
        return Response(status_code=404)
    return {"ok": True, "gradient_stops": COLORMAP_GRADIENT_STOPS[name]}


@app.post("/alpha/{sid}")
async def set_alpha(sid: str, request: Request):
    """Toggle alpha (0=off, 1=transparent below vmin)."""
    session = SESSIONS.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    body = await request.json()
    level = 1 if int(body.get("level", 0)) else 0
    session.alpha_level = level
    session.rgba_cache.clear()
    session._rgba_bytes = 0
    return {"level": level}


@app.post("/preload/{sid}")
async def start_preload(sid: str, request: Request):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}

    body = await request.json()
    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_list = [int(x) for x in body["indices"]]
    colormap = str(body.get("colormap", "gray"))
    dr = int(body.get("dr", 1))
    slice_dim = int(body["slice_dim"])
    dim_z = int(body.get("dim_z", -1))
    complex_mode = int(body.get("complex_mode", 0))
    log_scale = bool(body.get("log_scale", False))

    session.preload_gen += 1
    gen = session.preload_gen
    threading.Thread(
        target=_run_preload,
        args=(
            session,
            gen,
            dim_x,
            dim_y,
            idx_list,
            colormap,
            dr,
            slice_dim,
            dim_z,
            complex_mode,
            log_scale,
        ),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.get("/preload_status/{sid}")
def get_preload_status(sid: str):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}
    with session.preload_lock:
        return {
            "done": session.preload_done,
            "total": session.preload_total,
            "skipped": session.preload_skipped,
        }


def _vfield_n_times(session) -> int:
    """Return number of time frames in the vector field (0 = no vfield, 1 = no time dim)."""
    if session.vfield is None:
        return 0
    return int((_get_vfield_layout(session) or {}).get("n_times", 1))


@app.get("/metadata/{sid}")
async def get_metadata(sid: str):
    session = SESSIONS.get(sid)
    if not session and sid in PENDING_SESSIONS:
        # Session is still loading in a background thread — poll until ready (max 120 s).
        for _ in range(1200):
            await asyncio.sleep(0.1)
            session = SESSIONS.get(sid)
            if session:
                break
    if not session:
        return Response(status_code=404)
    try:
        return {
            "shape": [
                int(s)
                for s in (
                    session.spatial_shape
                    if session.rgb_axis is not None
                    else session.shape
                )
            ],
            "is_complex": bool(np.iscomplexobj(session.data)),
            "name": session.name,
            "has_vectorfield": session.vfield is not None,
            "vfield_n_times": _vfield_n_times(session),
            "is_rgb": session.rgb_axis is not None,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return Response(
            status_code=500, content=str(e).encode(), media_type="text/plain"
        )


def _compute_vfield_arrows(session, dim_x, dim_y, idx_tuple, t_index=0, density_offset=0):
    """Compute downsampled vector field arrows for a 2-D view.

    Returns ``{"arrows": coords, "scale": float, "stride": int}`` where
    *coords* is a float32 numpy array of shape (N, 4), or ``None`` if no
    vector field is available.
    """
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

    # Component mapping: when the image has more spatial dims than the VF
    # has components (e.g. 4-D image with a time-like leading dim but only
    # 3 displacement components), the components align to the *last*
    # n_comp spatial dims.  Dims before that offset have no displacement.
    comp_offset = n_spatial - n_comp
    cy = dim_y - comp_offset
    cx = dim_x - comp_offset
    vy_comp = vf_slice[:, :, cy] if 0 <= cy < n_comp else np.zeros((H, W), dtype=np.float32)
    vx_comp = vf_slice[:, :, cx] if 0 <= cx < n_comp else np.zeros((H, W), dtype=np.float32)

    # Uniform random sampling with a fixed seed derived from (H, W) so that
    # arrow positions are stable across slices (scrolling doesn't rearrange arrows).
    base_stride = max(1, max(H, W) // 32)
    # density_offset: positive = denser (smaller stride), negative = sparser
    # Use √2 per step (half-octave) for finer control than 2x per step
    stride = max(1, round(base_stride * (1.4142**-density_offset)))
    MAX_ARROWS = 4096
    n_arrows = min(max(1, (H // stride) * (W // stride)), MAX_ARROWS)
    rng = np.random.default_rng(int(H) * 10007 + int(W))
    gy = rng.integers(0, H, n_arrows).astype(int)
    gx = rng.integers(0, W, n_arrows).astype(int)

    vx_s = vx_comp[gy, gx]
    vy_s = vy_comp[gy, gx]

    # Scale: stride * 0.75 / p95_magnitude  (image pixels per voxel unit)
    mags = np.sqrt(vx_s**2 + vy_s**2)
    nonzero = mags[mags > 0]
    p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
    scale = float(stride * 0.75 / max(p95, 1e-9))

    # Stack into a single array for faster serialization
    coords = np.column_stack([gx, gy, vx_s, vy_s]).astype(np.float32)
    return {"arrows": coords, "scale": scale, "stride": int(stride)}


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


def _safe_float(v) -> float | None:
    """Convert to float; return None for NaN/Inf (JSON-safe)."""
    f = float(v)
    return f if math.isfinite(f) else None


@app.get("/pixel/{sid}")
def get_pixel(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    px: int,
    py: int,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"value": None}

    idx_tuple = tuple(int(x) for x in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    val = _safe_float(data[py, px]) if (0 <= py < h and 0 <= px < w) else None
    return {"value": val}


@app.post("/roi_freehand/{sid}")
def get_roi_freehand(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    complex_mode: int = 0,
    body: dict = Body(...),
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    points = body.get("points", [])
    if len(points) < 3:
        return {"error": "need at least 3 points"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    from PIL import Image as _PILImage, ImageDraw as _PILDraw
    mask_img = _PILImage.new("L", (w, h), 0)
    _PILDraw.Draw(mask_img).polygon([(p[0], p[1]) for p in points], fill=255)
    mask = np.array(mask_img) > 0
    roi = data[mask]
    if roi.size == 0:
        return {"error": "empty selection"}
    finite = roi[np.isfinite(roi)]
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
    }


@app.get("/roi_circle/{sid}")
def get_roi_circle(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    cx: float,
    cy: float,
    r: float,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    ys, xs = np.ogrid[:h, :w]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r**2
    roi = data[mask]
    if roi.size == 0:
        return {"error": "empty selection"}
    finite = roi[np.isfinite(roi)]
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
    }


@app.get("/roi/{sid}")
def get_roi(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    complex_mode: int = 0,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    xa = max(0, min(x0, x1, w - 1))
    xb = min(w, max(x0, x1) + 1)
    ya = max(0, min(y0, y1, h - 1))
    yb = min(h, max(y0, y1) + 1)
    roi = data[ya:yb, xa:xb]
    if roi.size == 0:
        return {"error": "empty selection"}
    finite = roi[np.isfinite(roi)]
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
    }


@app.get("/roi_multi/{sid}")
def get_roi_multi(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    x0: int = 0,
    y0: int = 0,
    x1: int = 0,
    y1: int = 0,
    complex_mode: int = 0,
):
    """ROI stats across dimension combinations for multi-dim export.

    Uses bounding box (x0,y0,x1,y1) for rect ROIs. Circle and freehand ROIs
    are converted to bounding boxes client-side before calling this endpoint.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    arr = session.data
    ndim = arr.ndim
    idx_list = [int(v) for v in indices.split(",")]

    roi_dims = {dim_x, dim_y}
    other_dims = [d for d in range(ndim) if d not in roi_dims]

    def extract_roi(data_2d):
        """Extract finite ROI pixels from a 2D slice using bounding box."""
        h, w = data_2d.shape
        xa, xb = max(0, min(x0, x1, w - 1)), min(w, max(x0, x1) + 1)
        ya, yb = max(0, min(y0, y1, h - 1)), min(h, max(y0, y1) + 1)
        roi = data_2d[ya:yb, xa:xb]
        if roi.size == 0:
            return np.array([])
        return roi[np.isfinite(roi)]

    rows = []

    # 1. Base slice (just the ROI plane)
    bitmask = ['0'] * ndim
    bitmask[dim_x] = '1'
    bitmask[dim_y] = '1'
    base_slice = extract_slice(session, dim_x, dim_y, idx_list)
    base_data = apply_complex_mode(base_slice, complex_mode)
    finite = extract_roi(base_data)
    if finite.size:
        rows.append({
            "dims": ''.join(bitmask),
            "min": _safe_float(finite.min()),
            "max": _safe_float(finite.max()),
            "mean": _safe_float(finite.mean()),
            "std": _safe_float(finite.std()),
            "n": int(finite.size),
        })

    # 2. Single-dimension extensions
    for ext_dim in other_dims:
        bitmask_ext = list(bitmask)
        bitmask_ext[ext_dim] = '1'
        all_finite = []
        for val in range(arr.shape[ext_dim]):
            idx_copy = list(idx_list)
            idx_copy[ext_dim] = val
            sl = extract_slice(session, dim_x, dim_y, idx_copy)
            data = apply_complex_mode(sl, complex_mode)
            finite = extract_roi(data)
            if finite.size:
                all_finite.append(finite)
        if all_finite:
            combined = np.concatenate(all_finite)
            rows.append({
                "dims": ''.join(bitmask_ext),
                "min": _safe_float(combined.min()),
                "max": _safe_float(combined.max()),
                "mean": _safe_float(combined.mean()),
                "std": _safe_float(combined.std()),
                "n": int(combined.size),
            })

    # 3. All dimensions
    if len(other_dims) > 1:
        bitmask_all = ['1'] * ndim
        all_finite = []
        ranges = [range(arr.shape[d]) for d in other_dims]
        for combo in itertools.product(*ranges):
            idx_copy = list(idx_list)
            for d, val in zip(other_dims, combo):
                idx_copy[d] = val
            sl = extract_slice(session, dim_x, dim_y, idx_copy)
            data = apply_complex_mode(sl, complex_mode)
            finite = extract_roi(data)
            if finite.size:
                all_finite.append(finite)
        if all_finite:
            combined = np.concatenate(all_finite)
            rows.append({
                "dims": ''.join(bitmask_all),
                "min": _safe_float(combined.min()),
                "max": _safe_float(combined.max()),
                "mean": _safe_float(combined.mean()),
                "std": _safe_float(combined.std()),
                "n": int(combined.size),
            })

    return {"rows": rows}


@app.get("/roi_floodfill/{sid}")
def get_roi_floodfill(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    px: int,
    py: int,
    tolerance: float = 0.1,
    complex_mode: int = 0,
):
    """Flood-fill ROI: grow connected region from seed pixel within tolerance."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if session.rgb_axis is not None:
        return {"error": "not supported for RGB sessions"}
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    h, w = data.shape
    if not (0 <= py < h and 0 <= px < w):
        return {"error": "seed out of bounds"}
    seed_val = float(data[py, px])
    # Use scipy flood_fill if available, else manual BFS
    try:
        from scipy.ndimage import label

        abs_tol = tolerance * (np.nanmax(np.abs(data)) - np.nanmin(np.abs(data)) + 1e-10)
        mask = np.abs(data - seed_val) <= abs_tol
        # Label connected components and pick the one containing the seed
        labeled, n_features = label(mask)
        seed_label = labeled[py, px]
        if seed_label == 0:
            return {"error": "seed outside tolerance region"}
        component = labeled == seed_label
    except ImportError:
        # Fallback: simple BFS flood fill
        abs_tol = tolerance * (np.nanmax(np.abs(data)) - np.nanmin(np.abs(data)) + 1e-10)
        component = np.zeros((h, w), dtype=bool)
        stack = [(py, px)]
        component[py, px] = True
        while stack:
            cy, cx = stack.pop()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not component[ny, nx]:
                    if abs(float(data[ny, nx]) - seed_val) <= abs_tol:
                        component[ny, nx] = True
                        stack.append((ny, nx))
    roi = data[component]
    finite = roi[np.isfinite(roi)]
    # Return mask as run-length bounding box for rendering
    ys, xs = np.where(component)
    bbox = {
        "x0": int(xs.min()),
        "y0": int(ys.min()),
        "x1": int(xs.max()),
        "y1": int(ys.max()),
    } if len(xs) > 0 else {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    return {
        "min": _safe_float(finite.min()) if finite.size else None,
        "max": _safe_float(finite.max()) if finite.size else None,
        "mean": _safe_float(finite.mean()) if finite.size else None,
        "std": _safe_float(finite.std()) if finite.size else None,
        "n": int(finite.size),
        "seed_value": _safe_float(seed_val),
        "tolerance": tolerance,
        "bbox": bbox,
        # Encode mask as base64 for rendering
        "mask_b64": _encode_mask_b64(component, bbox),
    }


def _encode_mask_b64(mask: np.ndarray, bbox: dict) -> str:
    """Encode the mask region within bbox as base64 for frontend rendering."""
    import base64

    x0, y0 = bbox["x0"], bbox["y0"]
    x1, y1 = bbox["x1"], bbox["y1"]
    sub = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    return base64.b64encode(sub.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# nnInteractive segmentation
# ---------------------------------------------------------------------------

_seg_overlay_sid: str | None = None  # current segmentation overlay session ID
_seg_label_mask: np.ndarray | None = None  # cumulative multi-label mask (3D)
_seg_current_label: int = 0  # label counter for accept
_seg_vol_axes: tuple[int, ...] | None = None  # which N-D axes map to 3D volume
_seg_fixed_indices: tuple[int, ...] | None = None  # fixed indices for non-volume dims
_seg_full_shape: tuple[int, ...] | None = None  # full N-D shape of source session


def _seg_coord_3d(
    session, dim_x: int, dim_y: int, idx_tuple: tuple[int, ...], px: int, py: int
) -> tuple[int, ...]:
    """Map a canvas click (px, py) to a 3D coordinate in the nnInteractive volume."""
    # Build full N-D index
    idx = list(idx_tuple)
    idx[dim_x] = px
    idx[dim_y] = py
    # Extract only the 3 volume axes
    if _seg_vol_axes is not None:
        return tuple(idx[a] for a in _seg_vol_axes)
    return tuple(idx)


@app.post("/seg/activate/{sid}")
async def seg_activate(sid: str, dim_x: int = 0, dim_y: int = 1, scroll_dim: int = -1,
                       indices: str = ""):
    """Connect to nnInteractive server (auto-launch if needed) and upload volume.

    For >3D data, a 3D subvolume is extracted using dim_x, dim_y, and scroll_dim
    as the three volume axes, with all other dimensions fixed at their current index.
    """
    from arrayview import _segmentation as seg

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    shape = session.spatial_shape if session.rgb_axis is not None else session.shape
    ndim = len(shape)
    if ndim < 3:
        return {"status": "error", "message": f"nnInteractive requires 3D+ data (got {ndim}D)"}

    global _seg_overlay_sid, _seg_label_mask, _seg_current_label
    global _seg_vol_axes, _seg_fixed_indices, _seg_full_shape

    # Connect or launch
    if not seg.is_connected():
        from arrayview._config import get_nninteractive_url
        configured_url = get_nninteractive_url()
        if configured_url:
            if not seg.try_connect(url=configured_url):
                return {"status": "error", "message": f"cannot reach nnInteractive at {configured_url}"}
        elif not seg.try_connect():
            err = await asyncio.to_thread(seg.try_launch)
            if err:
                return {"status": "error", "message": err}

    # If already have a label mask for this session, just resume
    if _seg_label_mask is not None and seg.is_connected():
        return {
            "status": "ok", "message": "resumed",
            "ndim": ndim, "overlay_sid": _seg_overlay_sid,
            "labels": _seg_get_label_info(),
        }

    data = np.asarray(session.data)
    if session.rgb_axis is not None:
        slc = [slice(None)] * data.ndim
        slc[session.rgb_axis] = 0
        data = data[tuple(slc)]

    _seg_full_shape = data.shape

    if ndim == 3:
        vol = data
        _seg_vol_axes = (0, 1, 2)
        _seg_fixed_indices = ()
    else:
        # Pick 3 axes: dim_x, dim_y, and scroll_dim (the axis the user scrolls through)
        if scroll_dim < 0:
            # Auto-pick: first axis that isn't dim_x or dim_y
            for ax in range(ndim):
                if ax != dim_x and ax != dim_y:
                    scroll_dim = ax
                    break
        vol_axes = sorted({dim_x, dim_y, scroll_dim})
        if len(vol_axes) != 3:
            return {"status": "error", "message": "need 3 distinct axes for segmentation"}
        _seg_vol_axes = tuple(vol_axes)

        # Fix remaining dims at current indices
        idx_list = [int(x) for x in indices.split(",")] if indices else [0] * ndim
        slc = [slice(None) if ax in vol_axes else idx_list[ax] for ax in range(ndim)]
        _seg_fixed_indices = tuple(idx_list)
        vol = data[tuple(slc)]

    try:
        await asyncio.to_thread(seg.upload_volume, vol)
    except Exception as exc:
        return {"status": "error", "message": f"upload failed: {exc}"}

    _seg_label_mask = np.zeros(vol.shape, dtype=np.uint8)
    _seg_current_label = 0
    _seg_overlay_sid = None

    return {"status": "ok", "message": "connected", "ndim": ndim}


@app.post("/seg/click/{sid}")
async def seg_click(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    px: int,
    py: int,
    positive: bool = True,
):
    """Send a point interaction to nnInteractive and return overlay session ID."""
    from arrayview import _segmentation as seg

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if not seg.is_connected():
        return {"status": "error", "message": "not connected"}

    idx_tuple = tuple(int(x) for x in indices.split(","))
    coord = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, px, py)

    try:
        mask = await asyncio.to_thread(seg.add_point, coord, positive)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return _seg_apply_mask(mask)


@app.post("/seg/bbox/{sid}")
async def seg_bbox(sid: str, body: dict = Body(...)):
    """Send a bounding box interaction to nnInteractive."""
    from arrayview import _segmentation as seg

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if not seg.is_connected():
        return {"status": "error", "message": "not connected"}

    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_tuple = tuple(int(x) for x in body["indices"].split(","))
    x0, y0 = int(body["x0"]), int(body["y0"])
    x1, y1 = int(body["x1"]), int(body["y1"])

    coord1 = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, x0, y0)
    coord2 = _seg_coord_3d(session, dim_x, dim_y, idx_tuple, x1, y1)

    try:
        mask = await asyncio.to_thread(seg.add_bbox, coord1, coord2, True)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return _seg_apply_mask(mask)


@app.post("/seg/scribble/{sid}")
async def seg_scribble(sid: str, body: dict = Body(...)):
    """Send a scribble interaction (freehand drawn points on current slice)."""
    from arrayview import _segmentation as seg

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if not seg.is_connected() or _seg_label_mask is None:
        return {"status": "error", "message": "not connected"}

    positive = bool(body.get("positive", True))
    mask_3d = _seg_rasterize_drawing(session, body)
    if mask_3d is None:
        return {"status": "error", "message": "no points provided"}

    try:
        mask = await asyncio.to_thread(seg.add_scribble, mask_3d, positive)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return _seg_apply_mask(mask)


@app.post("/seg/lasso/{sid}")
async def seg_lasso(sid: str, body: dict = Body(...)):
    """Send a lasso interaction (filled closed contour on current slice)."""
    from arrayview import _segmentation as seg

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    if not seg.is_connected() or _seg_label_mask is None:
        return {"status": "error", "message": "not connected"}

    positive = bool(body.get("positive", True))
    mask_3d = _seg_rasterize_drawing(session, body, fill=True)
    if mask_3d is None:
        return {"status": "error", "message": "need at least 3 points"}

    try:
        mask = await asyncio.to_thread(seg.add_lasso, mask_3d, positive)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    return _seg_apply_mask(mask)


def _seg_rasterize_drawing(session, body: dict, fill: bool = False) -> np.ndarray | None:
    """Rasterize 2D drawing points into a 3D mask on the current slice.

    Points are (px, py) in the 2D canvas coordinate space.  The mask is
    placed on the correct slice within the 3D volume used by nnInteractive.
    """
    from PIL import Image, ImageDraw

    points = body.get("points", [])
    if len(points) < (3 if fill else 1):
        return None

    dim_x = int(body["dim_x"])
    dim_y = int(body["dim_y"])
    idx_tuple = tuple(int(x) for x in body["indices"].split(","))

    vol_shape = _seg_label_mask.shape  # 3D shape sent to nnInteractive
    mask_3d = np.zeros(vol_shape, dtype=np.uint8)

    # Determine which 2D slice of the 3D volume this drawing is on
    if _seg_vol_axes is not None:
        # Map dim_x, dim_y to their position within the 3 vol_axes
        vol_ax_list = list(_seg_vol_axes)
        # The third axis (not dim_x, not dim_y) is the slice axis
        slice_axis_vol = [i for i, a in enumerate(vol_ax_list)
                          if a != dim_x and a != dim_y]
        if not slice_axis_vol:
            return None
        slice_ax = slice_axis_vol[0]
        slice_idx = idx_tuple[vol_ax_list[slice_ax]]

        # The two drawing axes in vol coordinates
        draw_axes = [i for i in range(3) if i != slice_ax]
        # Map canvas (px, py) to vol dims
        # dim_x maps to vol_ax_list.index(dim_x), dim_y maps to vol_ax_list.index(dim_y)
        vx = vol_ax_list.index(dim_x)
        vy = vol_ax_list.index(dim_y)
        w = vol_shape[vx]
        h = vol_shape[vy]
    else:
        # 3D data directly: dim_x, dim_y are axes, third is the slice axis
        others = [a for a in range(3) if a != dim_x and a != dim_y]
        slice_ax = others[0]
        slice_idx = idx_tuple[slice_ax]
        vx, vy = dim_x, dim_y
        w = vol_shape[vx]
        h = vol_shape[vy]

    # Rasterize points to a 2D image
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    coords = [(int(p[0]), int(p[1])) for p in points]

    if fill and len(coords) >= 3:
        draw.polygon(coords, fill=1)
    else:
        # Draw lines connecting points (scribble)
        if len(coords) >= 2:
            draw.line(coords, fill=1, width=2)
        else:
            # Single point — draw a small dot
            x, y = coords[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=1)

    mask_2d = np.array(img, dtype=np.uint8)

    # Place 2D mask into 3D volume at the correct slice
    slc = [slice(None)] * 3
    slc[slice_ax] = slice_idx
    # mask_2d is (w, h) but we need to align axes: PIL Image(w, h) → array(h, w)
    # Actually PIL.Image("L", (w, h)) → np.array gives shape (h, w)
    # We created Image with size (w, h) where w=vol_shape[vx], h=vol_shape[vy]
    # np.array gives (h, w) = (vol_shape[vy], vol_shape[vx])
    # We need to assign to the slice such that axis vx gets the px dimension
    # and axis vy gets the py dimension
    # If vx < vy (within the 2D slice after removing slice_ax):
    #   the 2D slice has shape (vol_shape[vx], vol_shape[vy]) so we need mask_2d.T
    # If vx > vy: shape is (vol_shape[vy], vol_shape[vx]) so mask_2d is already correct
    remaining = [i for i in range(3) if i != slice_ax]
    if remaining.index(vx) < remaining.index(vy):
        mask_2d = mask_2d.T
    mask_3d[tuple(slc)] = mask_2d

    return mask_3d


@app.post("/seg/reset/{sid}")
async def seg_reset(sid: str):
    """Reset interactions for next object."""
    from arrayview import _segmentation as seg

    if not seg.is_connected():
        return {"status": "error", "message": "not connected"}
    try:
        await asyncio.to_thread(seg.reset_interactions)
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

    # Clear current prediction overlay but keep cumulative label mask
    return _seg_update_overlay()


@app.post("/seg/accept/{sid}")
async def seg_accept(sid: str):
    """Commit current prediction to cumulative label mask, increment label."""
    from arrayview import _segmentation as seg

    global _seg_current_label
    if _seg_overlay_sid is None or _seg_label_mask is None:
        return {"status": "error", "message": "no active segmentation"}

    ov_session = SESSIONS.get(_seg_overlay_sid)
    if ov_session is None:
        return {"status": "error", "message": "overlay session lost"}

    # Merge current binary prediction into cumulative mask
    _seg_current_label += 1
    current_mask = np.asarray(ov_session.data)
    _seg_label_mask[current_mask > 0] = _seg_current_label

    # Reset nnInteractive interactions for next object
    try:
        await asyncio.to_thread(seg.reset_interactions)
    except Exception:
        pass

    # Update overlay to show cumulative labels
    return _seg_update_overlay()


@app.post("/seg/disconnect")
async def seg_disconnect():
    """Disconnect from nnInteractive server."""
    from arrayview import _segmentation as seg

    global _seg_overlay_sid, _seg_label_mask, _seg_current_label
    global _seg_vol_axes, _seg_fixed_indices, _seg_full_shape
    seg.disconnect()
    _seg_overlay_sid = None
    _seg_label_mask = None
    _seg_current_label = 0
    _seg_vol_axes = None
    _seg_fixed_indices = None
    _seg_full_shape = None
    return {"status": "ok"}


def _seg_expand_to_full(mask_3d: np.ndarray) -> np.ndarray:
    """Expand a 3D mask back to the full N-D shape for overlay display."""
    if _seg_full_shape is None or _seg_vol_axes is None:
        return mask_3d
    if len(_seg_full_shape) == 3:
        return mask_3d
    # Build full N-D array with zeros, insert 3D mask at fixed indices
    full = np.zeros(_seg_full_shape, dtype=mask_3d.dtype)
    ndim = len(_seg_full_shape)
    idx = _seg_fixed_indices if _seg_fixed_indices else (0,) * ndim
    slc = [slice(None) if ax in _seg_vol_axes else idx[ax] for ax in range(ndim)]
    full[tuple(slc)] = mask_3d
    return full


def _seg_apply_mask(mask: np.ndarray) -> dict:
    """Store prediction mask (+ cumulative labels) as overlay session."""
    global _seg_overlay_sid

    # Combine: cumulative labels in background, current prediction on top
    if _seg_label_mask is not None:
        combined = _seg_label_mask.copy()
        combined[mask > 0] = _seg_current_label + 1  # next label for current
    else:
        combined = mask

    overlay_data = _seg_expand_to_full(combined)
    _seg_set_overlay(overlay_data)
    return {"status": "ok", "overlay_sid": _seg_overlay_sid}


def _seg_update_overlay() -> dict:
    """Update overlay to show cumulative label mask only (after reset/accept)."""
    if _seg_label_mask is None:
        return {"status": "ok", "overlay_sid": None}

    overlay_data = _seg_expand_to_full(_seg_label_mask.copy())
    _seg_set_overlay(overlay_data)
    return {"status": "ok", "overlay_sid": _seg_overlay_sid, "labels": _seg_get_label_info()}


def _seg_set_overlay(data: np.ndarray) -> None:
    """Create or update the segmentation overlay session."""
    global _seg_overlay_sid
    if _seg_overlay_sid and _seg_overlay_sid in SESSIONS:
        ov = SESSIONS[_seg_overlay_sid]
        ov.data = data
        ov.shape = data.shape
        ov.raw_cache.clear()
        ov.rgba_cache.clear()
        ov.mosaic_cache.clear()
        ov._raw_bytes = 0
        ov._rgba_bytes = 0
        ov._mosaic_bytes = 0
        ov.data_version += 1
    else:
        ov = Session(data, name="nnInteractive segmentation")
        SESSIONS[ov.sid] = ov
        _seg_overlay_sid = ov.sid


# Label colors matching _render.py LABEL_COLORS for frontend display
_LABEL_HEX = [
    "#ff5050", "#50a0ff", "#50d250", "#ffaf32", "#b950ff",
    "#ff64be", "#3cd2c3", "#f0dc32", "#a06e3c", "#b4b4b4",
]

_seg_label_names: dict[int, str] = {}  # label_id → user-assigned name


def _seg_get_label_info() -> list[dict]:
    """Return info about all accepted labels."""
    if _seg_label_mask is None:
        return []
    labels = []
    for lbl in range(1, _seg_current_label + 1):
        voxels = int(np.sum(_seg_label_mask == lbl))
        if voxels == 0:
            continue
        labels.append({
            "label": lbl,
            "name": _seg_label_names.get(lbl, f"segment {lbl}"),
            "color": _LABEL_HEX[(lbl - 1) % len(_LABEL_HEX)],
            "voxels": voxels,
        })
    return labels


@app.get("/seg/labels/{sid}")
def seg_labels(sid: str):
    """Return info about all accepted segmentation labels."""
    return {"labels": _seg_get_label_info(), "overlay_sid": _seg_overlay_sid}


@app.post("/seg/rename/{sid}")
def seg_rename(sid: str, label: int, name: str):
    """Rename a segmentation label."""
    _seg_label_names[label] = name
    return {"status": "ok"}


@app.post("/seg/delete_label/{sid}")
def seg_delete_label(sid: str, label: int):
    """Delete a segmentation label from the cumulative mask."""
    if _seg_label_mask is None:
        return {"status": "error", "message": "no segmentation active"}
    _seg_label_mask[_seg_label_mask == label] = 0
    _seg_label_names.pop(label, None)
    overlay_data = _seg_expand_to_full(_seg_label_mask.copy())
    _seg_set_overlay(overlay_data)
    return {"status": "ok", "labels": _seg_get_label_info(), "overlay_sid": _seg_overlay_sid}


@app.get("/seg/export/{sid}")
def seg_export(sid: str):
    """Export cumulative label mask as downloadable .npy file."""
    if _seg_label_mask is None:
        return Response(status_code=404)
    buf = io.BytesIO()
    # Export in full N-D shape if >3D
    export_data = _seg_expand_to_full(_seg_label_mask)
    np.save(buf, export_data)
    return Response(
        content=buf.getvalue(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=segmentation.npy"},
    )


@app.get("/line_profile/{sid}")
def get_line_profile(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    complex_mode: int = 0,
    log_scale: bool = False,
):
    """Return intensity values sampled along a line between two points."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    if log_scale:
        data = np.where(data > 0, np.log10(data), 0.0)
    h, w = data.shape
    n_samples = 200
    xs = np.linspace(float(x0), float(x1), n_samples)
    ys = np.linspace(float(y0), float(y1), n_samples)
    # Nearest-neighbor sampling, clamped to valid range
    xi = np.clip(xs.astype(int), 0, w - 1)
    yi = np.clip(ys.astype(int), 0, h - 1)
    values = data[yi, xi]
    distance = float(math.hypot(x1 - x0, y1 - y0))
    return {
        "values": [_safe_float(v) for v in values],
        "distance": distance,
    }


@app.get("/histogram/{sid}")
def get_histogram(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    complex_mode: int = 0,
    bins: int = 128,
):
    """Return a histogram of the current 2-D slice as JSON.

    Returns ``{"counts": [...], "edges": [...], "vmin": float, "vmax": float}``
    where ``edges`` has length ``bins + 1``.  Finite values only.
    Used by the W-key histogram strip in the viewer.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    flat = data.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"counts": [], "edges": [], "vmin": 0.0, "vmax": 1.0}
    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        return {
            "counts": [int(finite.size)],
            "edges": [vmin, vmax + 1e-9],
            "vmin": vmin,
            "vmax": vmax,
        }
    bins = max(8, min(bins, 512))
    counts, edges = np.histogram(finite, bins=bins)
    return {
        "counts": counts.tolist(),
        "edges": [float(e) for e in edges],
        "vmin": vmin,
        "vmax": vmax,
    }


@app.get("/lebesgue/{sid}")
def get_lebesgue_slice(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    complex_mode: int = 0,
    log_scale: bool = False,
):
    """Return the raw 2-D slice as float32 binary for Lebesgue integral mode.

    The response is a raw byte buffer of ``height * width`` float32 values in
    row-major (C) order.  Non-finite values are preserved.  If *log_scale* is
    True the values are ``log10(|x| + 1)``.  The client uses this to do
    per-pixel bin lookups without a server round-trip on each hover.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode).astype(np.float32)
    if log_scale:
        data = np.log10(np.abs(data) + 1).astype(np.float32)
    return Response(
        content=data.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-ArrayView-Width": str(data.shape[1]),
            "X-ArrayView-Height": str(data.shape[0]),
            "Cache-Control": "no-cache",
        },
    )


@app.get("/export_slice/{sid}")
def export_slice(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    complex_mode: int = 0,
    save_to_downloads: int = 0,
):
    """Return the current 2-D slice as a downloadable .npy file.

    The slice is the raw floating-point data (before colormap/LUT), with the
    complex mode applied (mag/phase/real/imag). Used by the N-key shortcut.

    When save_to_downloads=1 (PyWebView), saves the file directly to ~/Downloads
    instead of returning it as a download response.
    """
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
    idx_tuple = tuple(int(v) for v in indices.split(","))
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data = apply_complex_mode(raw, complex_mode)
    buf = io.BytesIO()
    np.save(buf, data)
    buf.seek(0)
    # Build a suggested filename: sessionname_dim_x_dim_y_idx.npy
    name_stem = (session.name or "slice").replace(" ", "_").replace("/", "_")
    idx_str = "_".join(str(v) for v in idx_tuple)
    filename = f"{name_stem}_x{dim_x}_y{dim_y}_{idx_str}.npy"
    if save_to_downloads:
        import pathlib
        downloads = pathlib.Path.home() / "Downloads"
        dest = downloads / filename if downloads.is_dir() else pathlib.Path(filename)
        dest.write_bytes(buf.read())
        return JSONResponse({"path": str(dest)})
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/save_file")
async def save_file(request: Request):
    """Save a client-generated file (screenshot, CSV, GIF) to the Downloads folder.

    Accepts JSON: { "filename": "name.png", "data": "<base64-encoded>" }
    Used as fallback when <a download>.click() doesn't work (e.g. PyWebView).
    """
    body = await request.json()
    filename = body.get("filename", "arrayview_export")
    data_b64 = body.get("data", "")
    # Strip data URL prefix if present
    if "," in data_b64:
        data_b64 = data_b64.split(",", 1)[1]
    import base64, pathlib
    raw = base64.b64decode(data_b64)
    # Save to Downloads or current directory
    downloads = pathlib.Path.home() / "Downloads"
    dest = downloads / filename if downloads.is_dir() else pathlib.Path(filename)
    dest.write_bytes(raw)
    return JSONResponse({"path": str(dest)})


@app.get("/info/{sid}")
def get_info(sid: str):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    try:
        dtype_str = str(session.data.dtype)
    except AttributeError:
        dtype_str = "unknown"
    info: dict = {
        "shape": list(session.shape),
        "dtype": dtype_str,
        "ndim": len(session.shape),
        "total_elements": int(np.prod(session.shape)),
        "is_complex": bool(np.iscomplexobj(session.data)),
        "filepath": session.filepath,
    }
    try:
        info["size_mb"] = round(session.data.nbytes / 1024**2, 2)
    except AttributeError:
        info["size_mb"] = None
    # Colormap reason for the info overlay (feature: info-overlay-enhanced)
    try:
        reason = _session_mod._recommend_colormap_reason(
            session.data, session.global_stats
        )
        info["recommended_colormap"] = reason.split(" ")[0]
        info["recommended_colormap_reason"] = reason
    except Exception:
        info["recommended_colormap"] = None
        info["recommended_colormap_reason"] = None
    if session.fft_axes is not None:
        info["fft_axes"] = list(session.fft_axes)
    return info


@app.post("/fft/{sid}")
async def toggle_fft(sid: str, request: Request):
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "Invalid session"}

    body = await request.json()
    axes_str = str(body.get("axes", "")).strip()

    if session.fft_original_data is not None:
        session.data = session.fft_original_data
        session.shape = session.data.shape
        session.fft_original_data = None
        session.fft_axes = None
        session.raw_cache.clear()
        session.rgba_cache.clear()
        session.mosaic_cache.clear()
        session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
        session.compute_global_stats()
        return {"status": "restored", "is_complex": bool(np.iscomplexobj(session.data))}

    try:
        axes = tuple(int(a.strip()) for a in axes_str.split(",") if a.strip())
        if not axes:
            raise ValueError("No axes specified")
    except Exception as e:
        return {"error": str(e)}

    # Phase 4 guardrail: FFT materialises the full array — block for large data
    est = _estimate_array_bytes(session)
    if est > HEAVY_OP_LIMIT_BYTES:
        limit_mb = HEAVY_OP_LIMIT_BYTES // (1024 * 1024)
        est_mb = est // (1024 * 1024)
        return {
            "error": (
                f"FFT blocked: array is ~{est_mb} MB (limit {limit_mb} MB). "
                "Convert to a smaller sub-volume or increase "
                "ARRAYVIEW_HEAVY_OP_LIMIT_MB."
            ),
            "too_large": True,
        }

    session.fft_original_data = session.data
    full = np.array(session.data)
    session.data = np.fft.fftshift(np.fft.fftn(full, axes=axes), axes=axes)
    session.shape = session.data.shape
    session.fft_axes = axes
    session.raw_cache.clear()
    session.rgba_cache.clear()
    session.mosaic_cache.clear()
    session._raw_bytes = session._rgba_bytes = session._mosaic_bytes = 0
    session.compute_global_stats()
    return {
        "status": "fft_applied",
        "axes": list(axes),
        "is_complex": bool(np.iscomplexobj(session.data)),
    }


@app.post("/set_rgb/{sid}")
async def set_rgb_endpoint(sid: str, request: Request):
    """Toggle RGB rendering for a session.

    Body: {"axis": int | null}
    - axis=null  → disable RGB mode (restore full shape)
    - axis=int   → treat that dimension as the RGB/RGBA channel axis
                   (must have size 3 or 4)
    """
    session = SESSIONS.get(sid)
    if not session:
        return {"error": "session not found"}
    body = await request.json()
    axis = body.get("axis")
    if axis is None:
        session.rgb_axis = None
        session.spatial_shape = session.data.shape
    else:
        axis = int(axis)
        if not (0 <= axis < len(session.data.shape)):
            return {
                "error": f"axis {axis} out of range for shape {list(session.data.shape)}"
            }
        if session.data.shape[axis] not in (3, 4):
            return {
                "error": f"dim {axis} has size {session.data.shape[axis]}, need 3 or 4 for RGB/RGBA"
            }
        session.rgb_axis = axis
        session.spatial_shape = tuple(
            s for i, s in enumerate(session.data.shape) if i != axis
        )
    session.rgba_cache.clear()
    session._rgba_bytes = 0
    return {
        "ok": True,
        "is_rgb": session.rgb_axis is not None,
        "spatial_shape": list(session.spatial_shape),
    }


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
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
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
        )
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
            pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
            all_data = np.stack(frames)
            vmin = float(np.percentile(all_data, pct_lo))
            vmax = float(np.percentile(all_data, pct_hi))
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


def _render_normalized(session, dim_x, dim_y, idx_tuple, dr, complex_mode, log_scale):
    """Extract a slice and normalize to [0, 1] float32 using per-slice display range."""
    raw = extract_slice(session, dim_x, dim_y, list(idx_tuple))
    data, vmin, vmax = _prepare_display(session, raw, complex_mode, dr, log_scale)
    if vmax > vmin:
        normalized = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(data)
    return normalized.astype(np.float32)


def _render_normalized_mosaic(
    session, dim_x, dim_y, dim_z, idx_tuple, dr, complex_mode, log_scale
):
    """Return (float32 normalized mosaic grid [0,1], nan_mask) for all z-slices."""
    n = session.shape[dim_z]
    idx_list = list(idx_tuple)
    frames_raw = [
        extract_slice(
            session,
            dim_x,
            dim_y,
            [i if j == dim_z else idx_list[j] for j in range(len(session.shape))],
        )
        for i in range(n)
    ]
    frames = [apply_complex_mode(f, complex_mode) for f in frames_raw]
    if log_scale:
        frames = [np.log1p(np.abs(f)).astype(np.float32) for f in frames]
    all_data = np.stack(frames)
    if log_scale:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))
    else:
        vmin, vmax = _compute_vmin_vmax(session, all_data, dr, complex_mode)
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
    if vmax > vmin:
        normalized = np.clip(
            np.where(nan_mask, 0.0, (grid - vmin) / (vmax - vmin)), 0, 1
        )
    else:
        normalized = np.zeros_like(grid)
    return normalized.astype(np.float32), nan_mask


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
):
    session_a = SESSIONS.get(sid_a)
    session_b = SESSIONS.get(sid_b)
    if not session_a or not session_b:
        return Response(status_code=404)
    idx_tuple = tuple(int(x) for x in indices.split(","))
    # Use the shorter index tuple to handle mismatched dimensionalities
    ndim_a = len(session_a.shape)
    ndim_b = len(session_b.shape)
    idx_a = idx_tuple[:ndim_a]
    idx_b = idx_tuple[:ndim_b]
    nan_mask = None
    try:
        if dim_z >= 0:
            a, nan_mask_a = _render_normalized_mosaic(
                session_a, dim_x, dim_y, dim_z, idx_a, dr, complex_mode, log_scale
            )
            b, nan_mask_b = _render_normalized_mosaic(
                session_b, dim_x, dim_y, dim_z, idx_b, dr, complex_mode, log_scale
            )
            nan_mask = nan_mask_a | nan_mask_b
        else:
            a = _render_normalized(
                session_a, dim_x, dim_y, idx_a, dr, complex_mode, log_scale
            )
            b = _render_normalized(
                session_b, dim_x, dim_y, idx_b, dr, complex_mode, log_scale
            )
    except Exception:
        return Response(status_code=422)
    # Resize b to match a if shapes differ
    if a.shape != b.shape:
        try:
            from PIL import Image as _Image

            b_img = _Image.fromarray((b * 255).astype(np.uint8), mode="L")
            b_img = b_img.resize((a.shape[1], a.shape[0]), _Image.BILINEAR)
            b = np.array(b_img, dtype=np.float32) / 255.0
        except Exception:
            return Response(status_code=422)
    if diff_mode == 1:
        raw = a - b
        vmin, vmax = -1.0, 1.0
        colormap = "RdBu_r"
    elif diff_mode == 2:
        raw = np.abs(a - b)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "afmhot"
    else:  # diff_mode == 3
        raw = np.abs(a - b) / np.maximum(np.abs(a), 1e-6)
        raw = np.clip(raw, 0.0, 2.0).astype(np.float32)
        vmax = float(raw.max()) or 1.0
        vmin = 0.0
        colormap = "afmhot"
    # Allow frontend to override the colormap
    if diff_colormap and _ensure_lut(diff_colormap):
        colormap = diff_colormap
    if vmax > vmin:
        normalized = np.clip((raw - vmin) / (vmax - vmin), 0, 1)
    else:
        normalized = np.zeros_like(raw)
    _ensure_lut(colormap)
    lut = LUTS.get(colormap, LUTS["gray"])
    rgba = lut[(normalized * 255).astype(np.uint8)]
    if nan_mask is not None and nan_mask.shape == rgba.shape[:2]:
        rgba[nan_mask] = [22, 22, 22, 255]  # dark separator (matches mosaic_render)
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
):
    """Render an oblique (arbitrarily-oriented) slice through a 3-D volume."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    from scipy.ndimage import map_coordinates

    ctr = [float(x) for x in center.split(",")]
    bh = [float(x) for x in basis_h.split(",")]
    bv = [float(x) for x in basis_v.split(",")]
    dims = [int(x) for x in mv_dims.split(",")]

    ndim = len(session.shape)
    hw, hh = size_w / 2.0, size_h / 2.0
    s_arr = np.arange(size_w, dtype=np.float64) - hw
    t_arr = np.arange(size_h, dtype=np.float64) - hh
    ss, tt = np.meshgrid(s_arr, t_arr)  # (size_h, size_w)

    # Build full N-dim coordinate grids; non-spatial dims use fixed center value
    coords = np.empty((ndim, size_h, size_w), dtype=np.float64)
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
        data_f, coords, order=1, mode="constant", cval=0.0
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


@app.get("/grid/{sid}")
def get_grid(
    sid: str,
    dim_x: int,
    dim_y: int,
    indices: str,
    slice_dim: int,
    colormap: str = "gray",
    dr: int = 1,
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
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
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

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
):
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)
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
    if dr in session.global_stats:
        vmin, vmax = session.global_stats[dr]
    else:
        pct_lo, pct_hi = DR_PERCENTILES[dr % len(DR_PERCENTILES)]
        vmin = float(np.percentile(all_data, pct_lo))
        vmax = float(np.percentile(all_data, pct_hi))

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


@app.get("/listfiles")
def list_files(directory: str = ""):
    """List supported array files recursively (depth ≤ 4, max 300 files).

    Returns entries sorted by relative path; files directly in the target
    directory use just the filename as ``name``, nested files use a relative
    path (e.g. ``subdir/scan.npy``).  Hidden directories (name starts with
    ``.``) are skipped.
    """
    target = os.path.abspath(directory) if directory else os.getcwd()
    MAX_FILES = 300
    MAX_DEPTH = 4
    results = []
    try:
        for root, dirs, files in os.walk(target):
            rel_root = os.path.relpath(root, target)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            # Prune hidden dirs and stop recursing beyond MAX_DEPTH
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            if depth >= MAX_DEPTH:
                dirs.clear()
            for fname in sorted(files):
                name_lower = fname.lower()
                ext = (
                    ".nii.gz"
                    if name_lower.endswith(".nii.gz")
                    else os.path.splitext(name_lower)[1]
                )
                if ext not in _SUPPORTED_EXTS:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, target)
                name = fname if root == target else rel
                try:
                    file_size = os.path.getsize(fpath)
                except OSError:
                    continue
                shape = _peek_file_shape(fpath, ext)
                results.append(
                    {"name": name, "path": fpath, "size": file_size, "shape": shape}
                )
                if len(results) >= MAX_FILES:
                    break
            if len(results) >= MAX_FILES:
                break
    except Exception as e:
        return {"error": str(e)}
    return results


@app.get("/sessions")
def get_sessions():
    """Returns list of active sessions with metadata for the picker sidebar."""
    result = []
    for s in SESSIONS.values():
        dtype_str = str(getattr(s.data, "dtype", "unknown"))
        result.append({
            "sid": s.sid,
            "name": s.name,
            "shape": [int(x) for x in s.shape],
            "filepath": s.filepath,
            "dtype": dtype_str,
            "estimated_mem": s._estimated_mem,
        })
    return result


@app.get("/thumbnail/{sid}")
async def get_thumbnail(sid: str, w: int = 96, h: int = 72):
    """Return a small JPEG thumbnail of the session's current default view."""
    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    ndim = len(session.shape)
    if ndim < 2:
        rgba = np.full((1, 1, 4), 128, dtype=np.uint8)
    else:
        # Default view: last two dims, middle index for all others
        dim_x = ndim - 1
        dim_y = ndim - 2
        idx_list = [s // 2 for s in session.shape]
        try:
            rgba = await asyncio.to_thread(
                render_rgba, session, dim_x, dim_y, tuple(idx_list),
                "gray", 1, 0, False, None, None,
            )
        except Exception:
            rgba = np.full((1, 1, 4), 128, dtype=np.uint8)

    # Resize to thumbnail dimensions
    Image = _pil_image()
    img = Image.fromarray(rgba[:, :, :3])
    img = img.resize((w, h), Image.NEAREST if max(rgba.shape[:2]) < h else Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=30"},
    )


@app.post("/exploded/{sid}")
async def get_exploded_slices(
    sid: str,
    dim_x: int = Body(...),
    dim_y: int = Body(...),
    scroll_dim: int = Body(...),
    indices: list[int] = Body(...),
    width: int = Body(256),
    colormap: str = Body("gray"),
    dr: int = Body(1),
    complex_mode: int = Body(0),
    log_scale: bool = Body(False),
    vmin_override: float | None = Body(None),
    vmax_override: float | None = Body(None),
):
    """Return JPEG thumbnails for multiple slices along scroll_dim."""
    import base64

    session = SESSIONS.get(sid)
    if not session:
        return Response(status_code=404)

    ndim = len(session.shape)
    if ndim < 3:
        return JSONResponse({"error": "need >= 3D array"}, status_code=400)

    Image = _pil_image()
    results = []

    # Build base index tuple (middle of each dim)
    base_indices = [s // 2 for s in session.shape]

    for slice_idx in indices:
        idx_list = list(base_indices)
        idx_list[scroll_dim] = min(max(0, slice_idx), session.shape[scroll_dim] - 1)

        rgba = await asyncio.to_thread(
            render_rgba, session, dim_x, dim_y, tuple(idx_list),
            colormap, dr, complex_mode, log_scale,
            vmin_override, vmax_override,
        )

        img = Image.fromarray(rgba[:, :, :3])
        # Maintain aspect ratio, fit within width
        aspect = img.height / img.width
        target_h = max(1, int(width * aspect))
        resample = Image.NEAREST if img.width <= width else Image.LANCZOS
        img = img.resize((width, target_h), resample)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        results.append({"index": slice_idx, "image": f"data:image/jpeg;base64,{b64}"})

    return JSONResponse({"slices": results})


@app.post("/load")
async def load_file(request: Request):
    """Load a file into a new session. Optionally notify webview shells."""
    body = await request.json()
    filepath = str(body["filepath"])
    name = str(body.get("name") or os.path.basename(filepath))
    notify = bool(body.get("notify", False))
    # Dedup: if the same file is already loaded, return the existing session.
    abs_path = os.path.abspath(filepath)
    for existing in SESSIONS.values():
        if existing.filepath and os.path.abspath(existing.filepath) == abs_path:
            return {"sid": existing.sid, "name": existing.name, "notified": False}
    # RAM guard: block full-load formats that would exceed available memory.
    if not os.environ.get("ARRAYVIEW_SKIP_RAM_GUARD"):
        from ._io import FULL_LOAD_EXTS
        ext = os.path.splitext(filepath)[1].lower()
        # Handle double extensions like .nii.gz (not relevant here but consistent)
        if filepath.lower().endswith(".nii.gz"):
            ext = ".nii.gz"
        if ext in FULL_LOAD_EXTS:
            try:
                import psutil
                file_size = os.path.getsize(abs_path)
                available = psutil.virtual_memory().available
                if file_size > available:
                    return JSONResponse(
                        {
                            "error": "insufficient_memory",
                            "estimated_bytes": file_size,
                            "available_bytes": available,
                            "filename": os.path.basename(filepath),
                        },
                        status_code=507,
                    )
            except ImportError:
                pass  # psutil not available — skip guard
    try:
        data = await asyncio.to_thread(load_data, filepath)
    except Exception as e:
        return {"error": str(e)}
    session = await asyncio.to_thread(Session, data, filepath=filepath, name=name)
    if body.get("rgb"):
        try:
            await asyncio.to_thread(_setup_rgb, session)
        except ValueError as e:
            return {"error": str(e)}
    SESSIONS[session.sid] = session
    notified = False
    if notify:
        tab_url = None
        if body.get("compare_sids"):
            tab_url = (
                f"/?sid={session.sid}"
                f"&compare_sid={body['compare_sid']}"
                f"&compare_sids={body['compare_sids']}"
            )
        # wait=False: the shell window should already be connected for inject-into-existing.
        # If no shells are connected the native window is gone and the caller must open a new one.
        notified = await _notify_shells(session.sid, name, url=tab_url, wait=False)
    return {"sid": session.sid, "name": name, "notified": notified}


@app.post("/load-upload")
async def load_upload(file: UploadFile = File(...)):
    """Accept a drag-and-dropped .npy or .mat file and create a new session.

    The browser reads the file bytes locally via FileReader and POSTs them here
    as multipart/form-data.  Works in all environments because no file path is
    needed — the bytes travel over HTTP regardless of whether the server is
    local or remote.
    """
    import tempfile

    filename = file.filename or "array"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in (".npy", ".mat"):
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {ext or '(none)'}"
        )

    contents = await file.read()
    # Save to a temp file so the existing load_data() path handles format detection
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        data = await asyncio.to_thread(load_data, tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    name = filename
    session = await asyncio.to_thread(Session, data, name=name)
    SESSIONS[session.sid] = session
    await _notify_shells(session.sid, name, wait=False)
    resp_shape = [
        int(s)
        for s in (
            session.spatial_shape
            if session.rgb_axis is not None
            else session.shape
        )
    ]
    return {"sid": session.sid, "name": name, "shape": resp_shape}


@app.post("/load_bytes")
async def load_bytes_endpoint(request: Request):
    """Relay endpoint: accept a base64-encoded .npy array from a remote machine.

    Used when arrayview runs on a machine that reaches this server via a reverse
    SSH tunnel (``ssh -R PORT:localhost:PORT remote``).  The remote arrayview
    loads the file locally, serialises it as .npy bytes, and POSTs them here.
    This server creates the session and writes the VS Code signal file so Simple
    Browser opens automatically on the tunnel-remote side.
    """
    import base64

    body = await request.json()
    data_b64 = body.get("data_b64", "")
    name = str(body.get("name") or "array")
    rgb = bool(body.get("rgb", False))

    try:
        raw = base64.b64decode(data_b64)
        arr = np.load(io.BytesIO(raw))
    except Exception as e:
        return {"error": f"Failed to decode array: {e}"}

    session = await asyncio.to_thread(Session, arr, name=name)
    if rgb:
        try:
            await asyncio.to_thread(_setup_rgb, session)
        except ValueError as e:
            return {"error": str(e)}
    SESSIONS[session.sid] = session

    # Use the actual server port (not the request URL port, which reflects the
    # client's Host header and would be wrong if arriving via a reverse SSH tunnel).
    import arrayview._session as _sm

    port = _sm.SERVER_PORT or 8000
    url = f"http://localhost:{port}/?sid={session.sid}"

    # Write the signal file so the VS Code extension on this host opens Simple Browser.
    from arrayview._vscode import _open_via_signal_file

    _open_via_signal_file(url)

    return {"sid": session.sid, "url": url}


@app.get("/")
def get_ui(sid: str = None):
    """Viewer page."""
    # VS Code Simple Browser internally calls asExternalUri() which strips query
    # parameters, so ?sid= is often lost before the page loads.  Embed the SID
    # directly in the HTML so the viewer JS can find it regardless of the URL.
    if not sid:
        # No sid in URL — VS Code Simple Browser strips the query string before
        # loading the page, so ?sid= is lost.  Inject the latest valid session
        # server-side so the viewer JS can find it regardless of the URL.
        if SESSIONS:
            latest_sid = list(SESSIONS.keys())[-1]
            query_val = json.dumps(f"?sid={latest_sid}&transport=http")
        else:
            query_val = "null"  # viewer will show "Session not found or expired"
    else:
        # sid is present in the URL (valid or not) — let the JS fetch /metadata/{sid}
        # and handle errors itself (shows "Session not found or expired" on 404).
        query_val = "null"
    _init_luts()
    _cfg_colormaps = get_viewer_colormaps()
    _active_colormaps = _cfg_colormaps if _cfg_colormaps is not None else COLORMAPS
    html = (
        _VIEWER_HTML_TEMPLATE.replace("__COLORMAPS__", str(_active_colormaps))
        .replace("__DR_LABELS__", str(DR_LABELS))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
    )
    headers = {"Cache-Control": "no-store"}
    return HTMLResponse(content=html, headers=headers)
