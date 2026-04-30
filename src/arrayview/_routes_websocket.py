import asyncio

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

import arrayview._session as _session_mod
from arrayview._analysis import _build_metadata
from arrayview._overlays import _composite_overlays
from arrayview._render import (
    _prepare_display,
    extract_projection,
    extract_slice,
    render_mosaic,
    render_projection_rgba,
    render_rgb_rgba,
    render_rgba,
)
from arrayview._session import (
    SESSIONS,
    SHELL_SOCKETS,
    _render,
    _schedule_prefetch,
    _vprint,
    wait_for_session_ready,
)
from arrayview._vectorfield import _compute_vfield_arrows


_pil_image_mod = None


def _pil_image():
    global _pil_image_mod
    if _pil_image_mod is None:
        from PIL import Image

        _pil_image_mod = Image
    return _pil_image_mod


async def _notify_shells(sid, name, url=None, wait: bool = True) -> bool:
    """Push a new-tab message to all connected webview shell windows."""
    if wait:
        for _ in range(200):
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


def register_websocket_routes(app) -> None:
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
                        from arrayview._routes_persistence import _CROP_LOCK, _CROP_STATE

                        with _CROP_LOCK:
                            _CROP_STATE.pop(sid, None)
        except Exception:
            pass
        finally:
            if ws in SHELL_SOCKETS:
                SHELL_SOCKETS.remove(ws)

    @app.websocket("/ws/{sid}")
    async def websocket_endpoint(ws: WebSocket, sid: str):
        session = await wait_for_session_ready(sid)
        if not session:
            await ws.close()
            return

        await ws.accept()

        try:
            meta = _build_metadata(session)
            await ws.send_json({"type": "metadata", **meta})
        except Exception:
            pass

        _session_mod.VIEWER_SOCKETS += 1
        _session_mod.VIEWER_SIDS.add(sid)
        loop = asyncio.get_running_loop()
        _pending: asyncio.Queue[dict | None] = asyncio.Queue()

        async def _receiver() -> None:
            try:
                while True:
                    msg = await ws.receive_json()
                    try:
                        _pending.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await _pending.put(msg)
            except (WebSocketDisconnect, Exception):
                pass
            finally:
                _pending.put_nowait(None)

        recv_task = asyncio.create_task(_receiver())
        try:
            while True:
                msg = await _pending.get()
                if msg is None:
                    break

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
                direction = int(msg.get("direction", 1))
                slice_dim = int(msg.get("slice_dim", -1))
                canvas_w = int(msg.get("canvas_w", 0))
                canvas_h = int(msg.get("canvas_h", 0))
                _mc = msg.get("mosaic_cols")
                mosaic_cols = int(_mc) if _mc is not None else None
                projection_mode = int(msg.get("projection_mode", 0))
                projection_dim = int(msg.get("projection_dim", -1))

                if session.rgb_axis is not None:
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
                if session.vfield is not None:
                    vf_density = int(msg.get("vf_density", 0))
                    vf_t = int(msg.get("vf_t", 0))
                    vf_result = _compute_vfield_arrows(
                        session,
                        dim_x,
                        dim_y,
                        idx_tuple,
                        t_index=vf_t,
                        density_offset=vf_density,
                    )
                    if vf_result is not None:
                        arrows = vf_result["arrows"]
                        vf_hdr = np.array(
                            [len(arrows), vf_result["stride"]], dtype=np.uint32
                        ).tobytes()
                        vf_scale = np.array([vf_result["scale"]], dtype=np.float32).tobytes()
                        payload += vf_hdr + vf_scale + arrows.tobytes()
                await ws.send_bytes(payload)

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
