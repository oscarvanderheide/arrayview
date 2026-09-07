"""FastAPI application, REST/WebSocket routes, and HTML templates.

This module was extracted from _app.py during the modular refactor.
"""

# ── Imports ───────────────────────────────────────────────────────

import json
import os
import time
import uuid
from urllib.parse import urlencode
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
from arrayview._routes_preferences import register_preferences_routes
from arrayview._routes_export import register_export_routes
from arrayview._routes_preload import register_preload_routes
from arrayview._routes_query import register_query_routes
from arrayview._routes_rendering import register_rendering_routes
from arrayview._routes_segmentation import register_segmentation_routes
from arrayview._routes_state import register_state_routes
from arrayview._routes_vectorfield import register_vectorfield_routes
from arrayview._routes_websocket import _notify_shells, register_websocket_routes
from arrayview._config import (
    get_viewer_colormaps,
    get_viewer_dimbar_mode,
    get_viewer_ortho_layout,
    get_viewer_rounded_panes,
    get_viewer_theme,
)


from arrayview._imaging import ensure_image as _pil_image, ensure_imageops as _pil_imageops


# ── FastAPI Application ───────────────────────────────────────────

app = FastAPI()


# Identity, capabilities and the /ping contract are owned by _session.py so a
# booting daemon can answer /ping before this module has imported. Re-exported
# here so callers and tests keep addressing them through the server module.
from arrayview._session import (  # noqa: E402
    SERVER_CAPABILITIES,
    SERVER_PROTOCOL_VERSION,
    ServerRuntimeState,
    configure_server_runtime,
)


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


def _split_viewer_template(template: str) -> tuple[str, str, str]:
    """Split the viewer into a per-launch page and one static script.

    Everything in the main script below ``__AV_STATIC_SCRIPT_BELOW__`` is
    byte-identical on every launch (all substituted values sit above the
    marker), so it is served as ``viewer-<hash>.js`` with an immutable cache
    header instead of being re-sent inside every page. On a VS Code tunnel
    that is ~400 KB gzipped per open that the browser now keeps. The source
    file stays one self-contained HTML document; the split happens here.
    Returns ``(page_template, static_js, static_hash)``; without the marker
    the whole file is served inline exactly as before.
    """
    marker = "__AV_STATIC_SCRIPT_BELOW__"
    marker_at = template.find(marker)
    if marker_at < 0:
        return template, "", ""
    script_start = template.index("\n", marker_at) + 1
    script_end = template.index("</script>", script_start)
    static_js = template[script_start:script_end]
    import hashlib

    digest = hashlib.sha256(static_js.encode("utf-8")).hexdigest()[:16]
    page = (
        template[:script_start]
        + f'    </script>\n    <script src="viewer-{digest}.js">'
        + template[script_end:]
    )
    return page, static_js, digest


_VIEWER_PAGE_TEMPLATE, _VIEWER_STATIC_JS, _VIEWER_STATIC_JS_HASH = (
    _split_viewer_template(_VIEWER_HTML_TEMPLATE)
)
_VIEWER_STATIC_JS_GZIP: bytes | None = None

# The viewer page is ~2 MB of single-file HTML. On loopback that is free, but a
# VS Code tunnel relays every byte through a cloud endpoint, where the same
# payload has been observed taking 8-18 s. gzip cuts it to ~380 KB (5x), so the
# compressed bytes are cached per unique page body and reused across opens.
# Only text routes go through this — frame PNGs are already compressed and must
# not pay the CPU cost, which is why there is no app-wide GZipMiddleware.
_GZIP_CACHE: dict[str, bytes] = {}
_GZIP_CACHE_LIMIT = 4


def _accepts_gzip(request: Request | None) -> bool:
    if request is None:
        return False
    return "gzip" in request.headers.get("accept-encoding", "").lower()


def _text_response(
    body: str,
    *,
    request: Request | None,
    media_type: str,
    headers: dict[str, str] | None = None,
) -> Response:
    """Return ``body``, gzip-encoded when the client advertised support."""
    out = dict(headers or {})
    if not _accepts_gzip(request):
        return Response(content=body, media_type=media_type, headers=out)

    import gzip
    import hashlib

    key = hashlib.sha256(body.encode("utf-8")).hexdigest()
    packed = _GZIP_CACHE.get(key)
    if packed is None:
        packed = gzip.compress(body.encode("utf-8"), 6)
        if len(_GZIP_CACHE) >= _GZIP_CACHE_LIMIT:
            _GZIP_CACHE.pop(next(iter(_GZIP_CACHE)))
        _GZIP_CACHE[key] = packed
    out["Content-Encoding"] = "gzip"
    out["Vary"] = "Accept-Encoding"
    return Response(content=packed, media_type=media_type, headers=out)


@app.get("/viewer-{digest}.js")
def serve_viewer_static_js(digest: str, request: Request):
    """Serve the launch-independent part of the viewer script.

    The digest is in the URL, so a changed viewer gets a new address and the
    browser may keep this one forever; a stale address is simply unknown.
    """
    if not _VIEWER_STATIC_JS or digest != _VIEWER_STATIC_JS_HASH:
        return Response(status_code=404)
    headers = {"Cache-Control": "public, max-age=31536000, immutable"}
    if not _accepts_gzip(request):
        return Response(
            content=_VIEWER_STATIC_JS,
            media_type="application/javascript",
            headers=headers,
        )
    global _VIEWER_STATIC_JS_GZIP
    if _VIEWER_STATIC_JS_GZIP is None:
        import gzip

        _VIEWER_STATIC_JS_GZIP = gzip.compress(_VIEWER_STATIC_JS.encode("utf-8"), 6)
    headers["Content-Encoding"] = "gzip"
    headers["Vary"] = "Accept-Encoding"
    return Response(
        content=_VIEWER_STATIC_JS_GZIP,
        media_type="application/javascript",
        headers=headers,
    )


_GSAP_ETAG = '"' + __import__("hashlib").sha256(_GSAP_JS.encode("utf-8")).hexdigest()[:16] + '"'


@app.get("/gsap.min.js")
def serve_gsap(request: Request):
    """Serve the vendored GSAP library.

    Cached for a day and revalidated by ETag after that, so a repeat launch
    costs one small round trip instead of another 28 KB through the tunnel.
    """
    if request.headers.get("if-none-match") == _GSAP_ETAG:
        return Response(status_code=304, headers={"ETag": _GSAP_ETAG})
    return _text_response(
        _GSAP_JS,
        request=request,
        media_type="application/javascript",
        headers={"Cache-Control": "max-age=86400", "ETag": _GSAP_ETAG},
    )


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
register_preferences_routes(app)
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


@app.get("/colormap_lut/{name}")
def get_colormap_lut(name: str):
    """Return the exact 256-entry RGBA uint8 LUT for a colormap as raw bytes.

    The 32-stop gradient the viewer already carries is an approximation —
    fine for colorbars, wrong for pixels. The window/level drag preview
    remaps the raw slice locally and has to land on the same bytes the
    server would have sent, so it needs the real LUT. 1 KiB, immutable per
    colormap name: fetched once and cached client-side.
    """
    if not _ensure_lut(name):
        return Response(status_code=404)
    from arrayview._render import LUTS

    return Response(
        content=LUTS[name].tobytes(),
        media_type="application/octet-stream",
        headers={"Cache-Control": "max-age=86400"},
    )


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


@app.post("/cold-start-port")
async def cold_start_port(payload: dict):
    """Serve this same server on one more, never-before-forwarded port.

    A forward VS Code has just created does not drop its first requests, while
    one that has sat idle for tens of seconds drops the first two to four.  So
    the viewer page is served from an extra port rather than the main one: a
    fresh port for the first launch, then that same port is leased to later
    launches.  A lease keeps the route alive between handoff and connection,
    including when the previous viewer closes during that gap.  The port is
    released only after its last viewer and its last bounded lease are gone.
    """
    from arrayview import _extra_ports

    request_id = payload.get("requestId")
    ttl_ms = payload.get("ttlMs")
    expected_server_id = payload.get("expectedServerId")
    compatibility_request = not payload
    if compatibility_request:
        # Opener 0.15.47 shipped before correlated leases. Keep that already-
        # loaded host working after the Python package updates, so users are
        # not forced to reload a VS Code window merely to open the next array.
        request_id = f"compat-{uuid.uuid4().hex}"
        ttl_ms = 240_000
        expected_server_id = _session_mod.SERVER_RUNTIME.instance_id
    if not isinstance(request_id, str) or not request_id.strip():
        raise HTTPException(status_code=400, detail="requestId is required")
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise HTTPException(status_code=400, detail="ttlMs must be a positive integer")
    if not isinstance(expected_server_id, str) or not expected_server_id:
        raise HTTPException(status_code=400, detail="expectedServerId is required")
    actual_server_id = _session_mod.SERVER_RUNTIME.instance_id
    if expected_server_id != actual_server_id:
        raise HTTPException(
            status_code=409,
            detail="The requested ArrayView backend no longer owns this port",
        )
    port, reused = await _extra_ports.acquire_viewer_port(request_id, ttl_ms)
    if port is None:
        raise HTTPException(
            status_code=503,
            detail="Could not prepare the private VS Code viewer connection",
        )
    return {
        "port": port,
        "reused": reused,
        "ttlMs": min(ttl_ms, _extra_ports.MAX_LEASE_TTL_MS),
        "compatibility": compatibility_request,
        "requestId": request_id,
    }


@app.get("/ping")
def ping():
    """Health marker so clients can verify this is an ArrayView server."""
    return _session_mod.ping_payload(active_sessions=len(SESSIONS))


@app.get("/status")
def status():
    """Detailed server identity; equivalent to the compatible ping payload."""
    return ping()


# ── Root UI Route ─────────────────────────────────────────────────

def _trace_page_request(
    request: Request,
    *,
    request_id: str | None = None,
    navigation_attempt: str | None = None,
) -> None:
    """Record that a viewer page fetch reached this backend.

    A tunnel-delivered launch that never logs ``script-loaded`` is ambiguous
    from the opener's side alone: the relay may have dropped the page request,
    or the page may have arrived and the script stalled.  This distinguishes
    the two.  Inert unless ``ARRAYVIEW_LAUNCH_TRACE`` is set.

    Also stamps when a page was last served, which the idle nudge in
    ``_routes_websocket`` uses to stay silent while arrays are being opened.
    That stamp is not diagnostic and is kept before the trace guard below.
    """
    _session_mod.LAST_PAGE_SERVED_AT = time.time()
    if not os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
        return
    try:
        from arrayview._launch_trace import emit_launch_event

        params = request.query_params
        emit_launch_event(
            "page.requested",
            request_id=request_id or params.get("_av_launch_request_id"),
            navigation_attempt=(
                navigation_attempt
                or params.get("_av_navigation_attempt")
                or "0"
            ),
        )
    except Exception:
        pass


def _viewer_ui_response(
    request: Request,
    *,
    sid: str | None,
    query_val: str,
    asset_base: str = "",
):
    """Render one viewer page with its launch query already chosen.

    ``asset_base`` is prepended to the page's script addresses. The plain
    route keeps them relative so the page also works behind a Jupyter proxy
    prefix; the private tunnel route passes "/" so the addresses are the
    same on every launch and the browser can reuse the cached script
    (relative addresses under /_av/<tab>/... change with every tab key, which
    made the built-in browser download the 400 KB script on every open).
    """
    _init_luts()
    _cfg_colormaps = get_viewer_colormaps()
    _valid_cfg_colormaps = (
        [name for name in _cfg_colormaps if _ensure_lut(name)]
        if _cfg_colormaps is not None
        else []
    )
    _active_colormaps = _valid_cfg_colormaps or COLORMAPS
    _theme_names = ["dark", "light"]
    _cfg_theme = get_viewer_theme()
    _default_theme_idx = _theme_names.index(_cfg_theme) if _cfg_theme in _theme_names else 0
    _cfg_rounded = get_viewer_rounded_panes()
    _default_rounded_panes = "false" if _cfg_rounded is False else "true"
    _default_ortho_layout = json.dumps(get_viewer_ortho_layout())
    _default_dimbar_mode = json.dumps(get_viewer_dimbar_mode())
    html = (
        _VIEWER_PAGE_TEMPLATE.replace("__COLORMAPS__", str(_active_colormaps))
        .replace("__COLORMAP_GRADIENT_STOPS__", json.dumps(COLORMAP_GRADIENT_STOPS))
        .replace("__LABEL_COLORS__", json.dumps(LABEL_COLORS.astype(int).tolist()))
        .replace("__COMPLEX_MODES__", str(COMPLEX_MODES))
        .replace("__REAL_MODES__", str(REAL_MODES))
        .replace("__ARRAYVIEW_QUERY__", query_val)
        .replace("__DEFAULT_THEME_IDX__", str(_default_theme_idx))
        .replace("__DEFAULT_ROUNDED_PANES__", _default_rounded_panes)
        .replace("__DEFAULT_ORTHO_LAYOUT__", _default_ortho_layout)
        .replace("__DEFAULT_DIMBAR_MODE__", _default_dimbar_mode)
        .replace("__BODY_CLASS__", "av-loading" if sid else "")
        .replace("__ARRAYVIEW_VERSION__", _av_version)
    )
    if asset_base:
        html = html.replace('<script src="gsap.min.js">', f'<script src="{asset_base}gsap.min.js">')
        html = html.replace('<script src="viewer-', f'<script src="{asset_base}viewer-')
    return _text_response(
        html,
        request=request,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/_av/{tab_key}/{navigation_key}")
def get_short_viewer_ui(
    request: Request,
    tab_key: str,
    navigation_key: str,
):
    """Resolve a private integrated-browser launch without exposing its query."""
    # The page's script tags use relative addresses so they also work behind
    # a Jupyter proxy prefix. Under this nested route the browser resolves
    # them to /_av/<tab_key>/<asset>, so the assets are answered here too.
    # (Before this, gsap.min.js silently 404'd on every tunnel launch and the
    # viewer ran without its animation library.)
    if navigation_key == "gsap.min.js":
        return serve_gsap(request)
    if navigation_key.startswith("viewer-") and navigation_key.endswith(".js"):
        return serve_viewer_static_js(navigation_key[len("viewer-"):-3], request)
    try:
        from arrayview._launch_trace import emit_route_launch_event

        emit_route_launch_event(
            "page.route_entered",
            navigation_key=navigation_key,
            tab_key=tab_key,
        )
    except Exception:
        pass
    owner = _session_mod.VIEWER_LAUNCH_ROUTES.get(navigation_key)
    if owner is None:
        raise HTTPException(status_code=404, detail="Viewer launch route expired")
    sid, request_id = owner
    journal = (
        _session_mod.VIEWER_PHASE_JOURNALS
        .get(sid, {})
        .get(request_id)
    )
    if (
        journal is None
        or journal.get("tab_key") != tab_key
        or journal.get("navigation_key") != navigation_key
    ):
        raise HTTPException(status_code=409, detail="Viewer launch route changed")
    try:
        from arrayview._launch_trace import emit_route_launch_event

        emit_route_launch_event(
            "page.route_resolved",
            navigation_key=navigation_key,
            tab_key=tab_key,
            request_id=request_id,
            navigation_attempt=journal["navigation_attempt"],
        )
    except Exception:
        pass
    bootstrap_query = journal["viewer_query"] + "&" + urlencode(
        {
            "_av_integrated_browser": "1",
            "_av_launch_request_id": request_id,
            "_av_launch_server_id": _session_mod.SERVER_RUNTIME.instance_id,
            "_av_launch_window_id": journal["window_id"],
            "_av_launch_token": journal["token"],
            "_av_navigation_attempt": journal["navigation_attempt"],
        }
    )
    _trace_page_request(
        request,
        request_id=request_id,
        navigation_attempt=str(journal["navigation_attempt"]),
    )
    return _viewer_ui_response(
        request,
        sid=sid,
        query_val=json.dumps(bootstrap_query),
        asset_base="/",
    )


@app.get("/")
def get_ui(request: Request, sid: str = None):
    """Viewer page."""
    _trace_page_request(request)
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
    return _viewer_ui_response(request, sid=sid, query_val=query_val)
