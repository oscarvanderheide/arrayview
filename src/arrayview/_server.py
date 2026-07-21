"""FastAPI application, REST/WebSocket routes, and HTML templates.

This module was extracted from _app.py during the modular refactor.
"""

# ── Imports ───────────────────────────────────────────────────────

import json
import os
import time
import uuid
from dataclasses import dataclass, replace
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from importlib.resources import files as _pkg_files

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
from arrayview._session import (
    _vprint,
    Session,
    SESSIONS,
    COLORMAPS,
)
from arrayview import __version__ as _av_version
from arrayview._instance_registry import process_start_identity
import arrayview._session as _session_mod  # for mutable VIEWER_SOCKETS

from arrayview._render import (
    COLORMAP_GRADIENT_STOPS,
    COMPLEX_MODES,
    LABEL_COLORS,
    REAL_MODES,
    _init_luts,
    _ensure_lut,
    _mpl_colormaps,
    _setup_rgb,
)
from arrayview._routes_analysis import register_analysis_routes
from arrayview._routes_loading import register_loading_routes
from arrayview._routes_drop import register_drop_routes
from arrayview._routes_persistence import register_persistence_routes
from arrayview._routes_export import register_export_routes
from arrayview._routes_preload import register_preload_routes
from arrayview._routes_query import register_query_routes
from arrayview._routes_rendering import register_rendering_routes
from arrayview._routes_segmentation import register_segmentation_routes
from arrayview._routes_state import register_state_routes
from arrayview._routes_vectorfield import register_vectorfield_routes
from arrayview._routes_websocket import _notify_shells, register_websocket_routes
from arrayview._config import get_viewer_colormaps, get_viewer_rounded_panes, get_viewer_theme


from arrayview._imaging import ensure_image as _pil_image, ensure_imageops as _pil_imageops


# ── FastAPI Application ───────────────────────────────────────────

app = FastAPI()


SERVER_PROTOCOL_VERSION = "1"
SERVER_CAPABILITIES = (
    "health-status",
    "session-registration",
    "identity-fenced-load",
    "identity-fenced-mutations",
    "transactional-relay-display",
    "viewer-websocket",
    "shell-websocket",
    "dir-collection-case-inference",
    "staged-drop-import",
)


def _environment_port() -> int | None:
    value = os.environ.get("ARRAYVIEW_SERVER_PORT")
    if not value:
        return None
    try:
        port = int(value)
    except ValueError:
        return None
    return port if 0 < port < 65536 else None


def _environment_started_at() -> float:
    value = os.environ.get("ARRAYVIEW_STARTED_AT")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return time.time()


@dataclass(frozen=True)
class ServerRuntimeState:
    """Stable identity and ownership metadata for this server process."""

    instance_id: str
    process_start: str
    owner_mode: str
    started_at: float
    port: int | None
    protocol_version: str = SERVER_PROTOCOL_VERSION
    capabilities: tuple[str, ...] = SERVER_CAPABILITIES


if _session_mod.SERVER_RUNTIME is None:
    _session_mod.SERVER_RUNTIME = ServerRuntimeState(
        instance_id=os.environ.get("ARRAYVIEW_INSTANCE_ID") or str(uuid.uuid4()),
        process_start=(
            os.environ.get("ARRAYVIEW_PROCESS_START")
            or process_start_identity(os.getpid())
            or f"pid-only:{os.getpid()}"
        ),
        owner_mode=os.environ.get("ARRAYVIEW_OWNER_MODE", "unknown"),
        started_at=_environment_started_at(),
        port=_environment_port(),
    )


def configure_server_runtime(**changes) -> ServerRuntimeState:
    """Set launch metadata once the listener's final ownership/port is known."""
    _session_mod.SERVER_RUNTIME = replace(_session_mod.SERVER_RUNTIME, **changes)
    return _session_mod.SERVER_RUNTIME


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


def get_session_or_404(sid: str) -> "Session":
    """FastAPI dependency: fetch session by sid or raise 404."""
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


register_analysis_routes(app, get_session_or_404)
register_websocket_routes(app)
register_loading_routes(app, notify_shells=_notify_shells, setup_rgb=_setup_rgb)
register_drop_routes(app)
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


@app.get("/colormaps")
def list_colormaps():
    """Return all available matplotlib colormap names and cached gradient stops."""
    _init_luts()
    from arrayview._render import _mpl_colormaps as mpl_cm
    names = sorted(mpl_cm) if mpl_cm else []
    return {"colormaps": names, "gradient_stops": dict(COLORMAP_GRADIENT_STOPS)}


# ── REST Routes: Slice Rendering, Diff, and Oblique ──────────────


@app.get("/shell")
def get_shell():
    """Tabbed shell UI for native webview windows."""
    return HTMLResponse(content=_SHELL_HTML)


@app.get("/ping")
def ping():
    """Health marker so clients can verify this is an ArrayView server."""
    import socket

    runtime = _session_mod.SERVER_RUNTIME
    return {
        "ok": True,
        "service": "arrayview",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "protocol_version": runtime.protocol_version,
        "package_version": _av_version,
        "instance_id": runtime.instance_id,
        "process_start": runtime.process_start,
        "owner_mode": runtime.owner_mode,
        "started_at": runtime.started_at,
        "port": runtime.port,
        "capabilities": list(runtime.capabilities),
        "active_sessions": len(SESSIONS),
        "active_viewer_sockets": _session_mod.VIEWER_SOCKETS,
        "active_shell_sockets": len(_session_mod.SHELL_SOCKETS),
        "viewer_sockets": _session_mod.VIEWER_SOCKETS,
        "viewer_connections_seen": _session_mod.VIEWER_CONNECTIONS_SEEN,
        "shell_sockets": len(_session_mod.SHELL_SOCKETS),
        "shell_request_ids": sorted(_session_mod.SHELL_REQUEST_IDS),
        "active_viewer_requests": sorted(
            f"{sid}:{request_id}"
            for (sid, request_id), count in _session_mod.VIEWER_REQUEST_COUNTS.items()
            if count > 0
        ),
        "native_ready_requests": sorted(
            f"{sid}:{request_id}"
            for sid, request_id in _session_mod.NATIVE_READY_REQUESTS
        ),
    }


@app.get("/status")
def status():
    """Detailed server identity; equivalent to the compatible ping payload."""
    return ping()


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
    _theme_names = ["dark", "light"]
    _cfg_theme = get_viewer_theme()
    _default_theme_idx = _theme_names.index(_cfg_theme) if _cfg_theme in _theme_names else 0
    _cfg_rounded = get_viewer_rounded_panes()
    _default_rounded_panes = "false" if _cfg_rounded is False else "true"
    html = (
        _VIEWER_HTML_TEMPLATE.replace("__COLORMAPS__", str(_active_colormaps))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__LABEL_COLORS__", json.dumps(LABEL_COLORS.astype(int).tolist()))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
        .replace("__DEFAULT_THEME_IDX__", str(_default_theme_idx))
        .replace("__DEFAULT_ROUNDED_PANES__", _default_rounded_panes)
        .replace("__BODY_CLASS__", "av-loading" if sid else "")
        .replace("__ARRAYVIEW_VERSION__", _av_version)
    )
    headers = {"Cache-Control": "no-store"}
    return HTMLResponse(content=html, headers=headers)
