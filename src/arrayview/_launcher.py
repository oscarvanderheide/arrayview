"""Entry points, process management, view(), arrayview() CLI.

This module was extracted from _app.py during the modular refactor.
"""

# ── Imports and Lazy Loading ─────────────────────────────────────

import argparse
import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from importlib.resources import files as _pkg_files

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------
# NOTE: numpy, _session, _render, and _io are intentionally NOT imported at
# module level.  They are loaded lazily (inside functions / via _LazyMod) so
# that the CLI fast path — when the server is already alive — costs only the
# Python startup + stdlib, saving ~300–350 ms per invocation.
from arrayview._platform import (
    _in_jupyter,
    _in_vscode_terminal,
    _is_vscode_remote,
    _in_vscode_tunnel,
    _can_native_window,
    _find_vscode_ipc_hook,
    _is_julia_env,
    _in_julia_jupyter,
)
import arrayview._platform as _platform_mod  # for mutable globals

from arrayview._vscode import (
    _configure_vscode_port_preview,
    _ensure_vscode_extension,
    _open_direct_via_signal_file,
    _open_direct_via_shm,
    _print_viewer_location,
    _open_browser,
)

# _server.py (FastAPI) is imported lazily via _server_mod() to keep the
# import-time cost of ``import arrayview`` low (~175 ms saved).
_server_mod_cache = None


def _server_mod():
    global _server_mod_cache
    if _server_mod_cache is None:
        import arrayview._server as _srv  # noqa: PLC0415 — intentional lazy import

        _server_mod_cache = _srv
    return _server_mod_cache


# ---------------------------------------------------------------------------
# Lazy uvicorn import
# ---------------------------------------------------------------------------
_uvicorn_mod = None


def _uvicorn():
    """Lazy uvicorn import."""
    global _uvicorn_mod
    if _uvicorn_mod is None:
        import uvicorn

        _uvicorn_mod = uvicorn
    return _uvicorn_mod


# ---------------------------------------------------------------------------
# Lazy session module proxy
# ---------------------------------------------------------------------------
class _LazyMod:
    """Load a module on first attribute access.

    Keeps the module-level import cost of _launcher.py near zero on the CLI
    fast path (server already alive).  The real module is loaded the first
    time any attribute is accessed — which happens only in the slow path
    (starting the server or registering arrays).
    """

    __slots__ = ("_modname", "_mod")

    def __init__(self, modname: str) -> None:
        object.__setattr__(self, "_modname", modname)
        object.__setattr__(self, "_mod", None)

    def _load(self):
        import importlib

        mod = importlib.import_module(object.__getattribute__(self, "_modname"))
        object.__setattr__(self, "_mod", mod)
        return mod

    def __getattr__(self, attr: str):
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            mod = self._load()
        return getattr(mod, attr)

    def __setattr__(self, attr: str, val):
        if attr in ("_modname", "_mod"):
            object.__setattr__(self, attr, val)
            return
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            mod = self._load()
        setattr(mod, attr, val)


_session_mod = _LazyMod("arrayview._session")


def _vprint(*args, **kwargs) -> None:
    """Proxy for _session_mod._vprint; triggers lazy load only on first call."""
    _session_mod._vprint(*args, **kwargs)


# ── Subprocess GUI Launcher ───────────────────────────────────────

_ICON_PNG_PATH: str | None = None


def _get_icon_png_path() -> str | None:
    """Return path to the bundled ArrayView icon PNG, or None if unavailable."""
    global _ICON_PNG_PATH
    if _ICON_PNG_PATH is not None:
        return _ICON_PNG_PATH or None
    try:
        # Use the pre-rendered icon shipped as package data (avoids PIL at startup).
        icon_ref = _pkg_files("arrayview").joinpath("_icon.png")
        # importlib.resources may return a Path or a traversable; materialise it.
        icon_path = str(icon_ref)
        if os.path.isfile(icon_path):
            _ICON_PNG_PATH = icon_path
        else:
            _ICON_PNG_PATH = ""
    except Exception:
        _ICON_PNG_PATH = ""
    return _ICON_PNG_PATH or None


def _open_webview(
    url: str,
    win_w: int,
    win_h: int,
    capture_stderr: bool = False,
    shell_port: int | None = None,
) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel.

    When shell_port is provided, the shell HTML is embedded as inline content
    (html= parameter) so WKWebView never makes a network request — eliminating
    the white-flash that occurs while waiting for the first HTTP response.
    The shell's WebSocket reconnects to the server once it starts.
    """
    import base64 as _b64

    icon_path = _get_icon_png_path() or ""
    inline_html_b64 = None

    if shell_port is not None:
        try:
            from importlib.resources import files as _pkg_files
            shell_html = _pkg_files("arrayview").joinpath("_shell.html").read_text(encoding="utf-8")
            # Inject inline-mode flag and hardcoded host (location.host is empty in html= mode)
            shell_html = shell_html.replace(
                "</head>",
                f'<script>window.__av_inline=true;</script>\n'
                f'<base href="http://localhost:{shell_port}/">\n'
                f"</head>",
                1,
            )
            # Fix WebSocket URL — location.host is "" in inline html= mode
            shell_html = shell_html.replace(
                "`${proto}//${location.host}/ws/shell`",
                f"`ws://localhost:{shell_port}/ws/shell`",
            )
            inline_html_b64 = _b64.b64encode(shell_html.encode()).decode()
        except Exception:
            pass  # fall back to URL mode

    if inline_html_b64:
        script_lines = [
            "import sys, base64, webview",
            "u, w, h, icon, html_b64 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]",
            "html = base64.b64decode(html_b64.encode()).decode()",
            "win = webview.create_window('ArrayView', html=html, width=w, height=h, background_color='#0c0c0c')",
            "kw = {'gui': 'qt'} if sys.platform.startswith('linux') else {}",
            "def _start_func():",
            "    if icon:",
            "        if sys.platform == 'darwin':",
            "            try:",
            "                import AppKit",
            "                from PyObjCTools import AppHelper",
            "                img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon)",
            "                AppHelper.callAfter(AppKit.NSApplication.sharedApplication().setApplicationIconImage_, img)",
            "            except Exception: pass",
            "        else:",
            "            pass  # icon handled via kw below for non-darwin",
            "if icon and sys.platform != 'darwin':",
            "    kw['icon'] = icon",
            "kw['func'] = _start_func",
            "webview.start(**kw)",
        ]
        script = "\n".join(script_lines)
        return subprocess.Popen(
            [sys.executable, "-c", script, url, str(win_w), str(win_h), icon_path, inline_html_b64],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        )

    # URL mode — direct load (used when shell_port not provided)
    script_lines = [
        "import sys, webview",
        "u, w, h, icon = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]",
        "win = webview.create_window('ArrayView', u, width=w, height=h, background_color='#0c0c0c')",
        "kw = {'gui': 'qt'} if sys.platform.startswith('linux') else {}",
        "def _start_func():",
        "    if icon:",
        "        if sys.platform == 'darwin':",
        "            try:",
        "                import AppKit",
        "                from PyObjCTools import AppHelper",
        "                img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon)",
        "                AppHelper.callAfter(AppKit.NSApplication.sharedApplication().setApplicationIconImage_, img)",
        "            except Exception: pass",
        "        else:",
        "            pass  # icon handled via kw below for non-darwin",
        "if icon and sys.platform != 'darwin':",
        "    kw['icon'] = icon",
        "kw['func'] = _start_func",
        "webview.start(**kw)",
    ]
    script = "\n".join(script_lines)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            url,
            str(win_w),
            str(win_h),
            icon_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def _open_webview_with_fallback(
    url: str,
    win_w: int,
    win_h: int,
    shell_port: int | None = None,
    floating: bool = False,
) -> subprocess.Popen:
    """Launch pywebview, falling back to _open_browser if the subprocess exits immediately
    OR if no viewer WebSocket connects within ~10 s (catches macOS non-framework Python
    zombies that start but show nothing).

    Used from view() (Python API) where the host process stays alive.
    """
    proc = _open_webview(url, win_w, win_h, capture_stderr=True, shell_port=shell_port)
    _vprint(f"[ArrayView] Launching native window (pid={proc.pid})...", flush=True)
    sockets_before = (
        _session_mod.VIEWER_SOCKETS
    )  # capture count so we detect a NEW connection

    def _read_stderr():
        try:
            return proc.stderr.read().decode(errors="replace").strip()
        except Exception:
            return ""

    def _watchdog():
        # Phase 1: watch for an immediate crash (2 s)
        for _ in range(20):
            time.sleep(0.1)
            if proc.poll() is not None:
                stderr_out = _read_stderr()
                _vprint(
                    f"[ArrayView] Native window exited immediately (code {proc.returncode}), opening in browser",
                    flush=True,
                )
                if stderr_out:
                    _vprint(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
                _open_browser(url, floating=floating)
                return

        # Phase 2: process is alive — wait up to 8 s for a NEW viewer WebSocket to connect.
        # We compare against sockets_before so an already-open browser tab doesn't
        # falsely confirm that the native window launched successfully.
        import arrayview._session as _sm

        for _ in range(80):
            time.sleep(0.1)
            if _sm.VIEWER_SOCKETS > sockets_before:
                _vprint("[ArrayView] Native window connected successfully", flush=True)
                if sys.platform == "darwin":
                    subprocess.Popen(
                        [
                            "osascript",
                            "-e",
                            f'tell application "System Events" to set frontmost of'
                            f" (first process whose unix id is {proc.pid}) to true",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                return
            if proc.poll() is not None:
                stderr_out = _read_stderr()
                _vprint(
                    f"[ArrayView] Native window exited (code {proc.returncode}), opening in browser",
                    flush=True,
                )
                if stderr_out:
                    _vprint(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
                _open_browser(url, floating=floating)
                return

        # Phase 3: alive but no UI connection after 10 s — zombie (e.g. non-framework Python on macOS)
        _vprint(
            "[ArrayView] Native window did not connect; falling back to browser",
            flush=True,
        )
        try:
            proc.terminate()
        except Exception:
            pass
        _open_browser(url, floating=floating)

    threading.Thread(target=_watchdog, daemon=True).start()
    return proc


def _open_webview_cli(
    url: str,
    win_w: int,
    win_h: int,
    shell_port: int | None = None,
) -> bool:
    """Launch pywebview from the CLI and synchronously wait to detect an immediate crash.

    Returns True if the window appears to have started (still alive after 2 s).
    Returns False if it crashed; in that case the caller should fall back to browser.
    The CLI process must not exit while the daemon-thread watchdog is still pending,
    so the wait is done synchronously here.
    """
    _vprint("[ArrayView] Launching native window (PyWebView)...", flush=True)
    proc = _open_webview(url, win_w, win_h, capture_stderr=True, shell_port=shell_port)
    for _ in range(20):
        time.sleep(0.1)
        if proc.poll() is not None:
            stderr_out = ""
            try:
                stderr_out = proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                pass
            _vprint(
                f"[ArrayView] Native window exited immediately (code {proc.returncode})",
                flush=True,
            )
            if stderr_out:
                _vprint(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
            return False
    _vprint("[ArrayView] Native window started successfully", flush=True)
    return True


# ── Server Port Utilities ─────────────────────────────────────────


def _server_alive(port: int, timeout: float = 0.5) -> bool:
    """Return True only if an ArrayView server is responding on the port."""
    url = f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("ok") is True and payload.get("service") == "arrayview"
    except Exception:
        return False


def _server_pid(port: int) -> int | None:
    """Return the pid of the responding ArrayView server, or None if unreachable."""
    url = f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=0.5) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return payload.get("pid")
    except Exception:
        pass
    return None


def _server_hostname(port: int, timeout: float = 0.5) -> str | None:
    """Return the hostname reported by the ArrayView server on ``port``, or None."""
    url = f"http://127.0.0.1:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return payload.get("hostname")
    except Exception:
        pass
    return None


def _relay_array_to_server(
    filepath: str,
    port: int,
    name: str,
    rgb: bool = False,
    relay_host: str = "127.0.0.1",
) -> None:
    """Load *filepath* locally and POST the bytes to an ArrayView relay server.

    Used when the local port is a reverse-SSH-forwarded connection to a remote
    ArrayView server (e.g. tunnel-remote).  The relay server registers the
    session and writes its own VS Code signal file so a viewer tab opens there.

    ``relay_host`` defaults to 127.0.0.1; only change it when the relay server
    is genuinely on a different network interface (rare).
    """
    import base64
    import numpy as np
    from arrayview._io import load_data

    print("[ArrayView] Relay mode: sending array to remote server...", flush=True)
    try:
        data = load_data(filepath)
    except Exception as e:
        print(f"[ArrayView] Failed to load {filepath}: {e}", flush=True)
        raise

    buf = io.BytesIO()
    np.save(buf, data)
    data_b64 = base64.b64encode(buf.getvalue()).decode()

    body = json.dumps({"data_b64": data_b64, "name": name, "rgb": rgb}).encode()
    req = urllib.request.Request(
        f"http://{relay_host}:{port}/load_bytes",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read())

    if "error" in result:
        raise RuntimeError(f"[ArrayView] Relay server error: {result['error']}")

    print(
        "[ArrayView] Array sent to relay server. "
        "VS Code viewer tab should open automatically.",
        flush=True,
    )


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


def _wait_for_port(port: int, timeout: float = 10.0, tcp_only: bool = False) -> bool:
    """Wait until the server on *port* is ready.

    ``tcp_only=True`` uses a raw TCP connect instead of an HTTP /ping.  This is
    faster on cold start (saves ~20 ms per poll) and safe when we know the server
    is ours (we just spawned it).  Combined with WS retry in the viewer, the
    browser can open as soon as the port is bound.
    """
    check = _port_in_use if tcp_only else _server_alive
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check(port):
            return True
        time.sleep(0.02)
    return False


def _find_server_port(port: int, search_range: int = 20) -> tuple[int, bool]:
    """Find a usable server port starting from *port*.

    Returns ``(found_port, already_running)`` where ``already_running`` is True
    if an ArrayView server is already live on ``found_port``.
    """
    if _server_alive(port):
        return (port, True)
    for candidate in range(port, port + search_range + 1):
        if not _port_in_use(candidate):
            return (candidate, False)
    return (port + search_range, False)


# Event signalled when the in-process background server is ready to accept
# connections (set right after sock.listen()).  Used by view() to avoid
# the slower _wait_for_port() HTTP polling loop.
_server_ready_event = threading.Event()

# Module-level port set by _serve_background before _server_ready_event fires.
# None when the server is already running (no pre-server needed).
_loading_port: int | None = None

_LOADING_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="dark">
<title>ArrayView</title>
<style>
body {
    margin: 0; background: #0c0c0c; color: #555;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 16px;
}
.av-spinner {
    width: 28px; height: 28px; border-radius: 50%;
    border: 2px solid #1e1e1e; border-top-color: #4cc9f0;
    animation: av-spin 0.8s linear infinite;
}
@keyframes av-spin { to { transform: rotate(360deg); } }
.av-label { font-size: 12px; letter-spacing: 0.03em; }
</style>
</head>
<body>
<div class="av-spinner"></div>
<div class="av-label">Loading ArrayView\u2026</div>
<script>
const target = new URLSearchParams(location.search).get('target');
if (target) {
    (async function poll() {
        try {
            const r = await fetch(target, { signal: AbortSignal.timeout(1000) });
            if (r.ok) { window.location.replace(target); return; }
        } catch (_) {}
        setTimeout(poll, 100);
    })();
}
</script>
</body>
</html>
"""


class _LoadingHandler:
    """Minimal HTTP/1.1 handler: serves _LOADING_HTML on any GET request."""

    def __init__(self, conn: "socket.socket") -> None:
        try:
            body = _LOADING_HTML.encode()
            header = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                b"Connection: close\r\n"
                + f"Content-Length: {len(body)}\r\n\r\n".encode()
            )
            conn.sendall(header + body)
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


def _run_loading_server(
    sock: "socket.socket", timeout: float = 120.0
) -> None:
    """Accept loop for the pre-server loading page. Runs in a daemon thread.

    Serves _LOADING_HTML on every accepted connection until *timeout* seconds
    have elapsed, then closes the socket. The browser's JS poller handles the
    actual redirect — this server just needs to respond to the initial GET.
    """
    sock.settimeout(1.0)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn, _ = sock.accept()
            threading.Thread(
                target=_LoadingHandler, args=(conn,), daemon=True
            ).start()
        except OSError:
            continue
    try:
        sock.close()
    except OSError:
        pass


async def _serve_background(port: int, stop_when_closed: bool = False):
    global _loading_port
    _loading_port = None  # reset for this server lifetime
    _session_mod.SERVER_LOOP = asyncio.get_running_loop()
    _session_mod.SERVER_PORT = port
    import socket as _socket

    # Pre-create the socket with SO_REUSEADDR so we can rebind immediately after
    # a previous server on this port was killed (avoids TIME_WAIT Errno 48).
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    sock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    sock.set_inheritable(True)

    # Bind the loading-page server on an OS-chosen ephemeral port.
    # This uses only stdlib so it starts in microseconds — well before
    # uvicorn's heavy imports finish.  _loading_port is read by the main
    # thread after _server_ready_event fires.
    try:
        _lsock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        _lsock.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        _lsock.bind(("127.0.0.1", 0))
        _lsock.listen(16)
        _loading_port = _lsock.getsockname()[1]
        threading.Thread(
            target=_run_loading_server, args=(_lsock,), daemon=True
        ).start()
    except Exception:
        _loading_port = None

    _server_ready_event.set()  # port is bound — callers can proceed
    config = _uvicorn().Config(
        _server_mod().app,
        log_level="error",
        timeout_keep_alive=30,
        ws_ping_interval=None,
        ws="websockets",  # S7: explicit backend — enables permessage-deflate compression
        ws_per_message_deflate=True,  # S7: negotiate deflate with browser (transparent)
    )
    server = _uvicorn().Server(config)
    if stop_when_closed:
        asyncio.create_task(_stop_server_when_viewer_closes(server))
    await server.serve(sockets=[sock])


def _with_loading(url: str) -> str:
    """Wrap *url* with the loading-page proxy when the pre-server is active.

    Returns a URL pointing to the loading page (which will redirect to *url*
    once the real server responds).  Falls back to *url* unchanged when the
    loading server is not running (e.g. server was already alive).
    """
    if _loading_port is not None:
        encoded = urllib.parse.quote(url, safe="")
        return f"http://127.0.0.1:{_loading_port}/?target={encoded}"
    return url


_OVERLAY_PALETTE = ["ff4444", "44cc44", "4488ff", "ffcc00", "ff44ff", "44ffff"]

_JUPYTER_PROXY_INLINE_CACHE: bool | None = None


type _CSVValues = str | list[str] | tuple[str, ...]
type _CompareSids = list[str] | tuple[str, ...]


def _join_query_values(values: _CSVValues) -> str:
    if isinstance(values, str):
        return values
    return ",".join(str(value) for value in values)


def _viewer_query(
    sid: str,
    *,
    compare_sids: _CompareSids | None = None,
    overlay_sids: _CSVValues | None = None,
    overlay_colors: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    parts = [f"sid={sid}"]
    if overlay_sids:
        parts.append(f"overlay_sid={_join_query_values(overlay_sids)}")
    if overlay_colors:
        parts.append(f"overlay_colors={_join_query_values(overlay_colors)}")
    if compare_sids:
        parts.append(f"compare_sid={compare_sids[0]}")
        parts.append(f"compare_sids={_join_query_values(compare_sids)}")
    if dims is not None:
        parts.append(f"dim_x={dims[0]}")
        parts.append(f"dim_y={dims[1]}")
    if inline:
        parts.append("inline=1")
    return "?" + "&".join(parts)


def _viewer_path(
    sid: str,
    *,
    compare_sids: _CompareSids | None = None,
    overlay_sids: _CSVValues | None = None,
    overlay_colors: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    return "/" + _viewer_query(
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sids,
        overlay_colors=overlay_colors,
        dims=dims,
        inline=inline,
    )


def _viewer_url(
    port: int,
    sid: str,
    *,
    compare_sids: _CompareSids | None = None,
    overlay_sids: _CSVValues | None = None,
    overlay_colors: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    return f"http://localhost:{port}" + _viewer_path(
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sids,
        overlay_colors=overlay_colors,
        dims=dims,
        inline=inline,
    )


def _shell_url(
    port: int, sid: str, name: str, *, compare_sids: _CompareSids | None = None
) -> str:
    parts = [
        f"init_sid={sid}",
        f"init_name={urllib.parse.quote(str(name))}",
    ]
    if compare_sids:
        parts.append(f"init_compare_sid={compare_sids[0]}")
        parts.append(f"init_compare_sids={_join_query_values(compare_sids)}")
    return f"http://localhost:{port}/shell?" + "&".join(parts)


def _jupyter_base_url_prefix() -> str:
    for key in (
        "ARRAYVIEW_JUPYTER_BASE_URL",
        "JUPYTERHUB_SERVICE_PREFIX",
        "NB_PREFIX",
        "JUPYTER_BASE_URL",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            if not value.startswith("/"):
                value = "/" + value
            if not value.endswith("/"):
                value += "/"
            return value
    return "/"


def _should_use_jupyter_proxy_inline() -> bool:
    if not _in_jupyter():
        return False

    forced = os.environ.get("ARRAYVIEW_JUPYTER_PROXY", "").strip().lower()
    if forced:
        return forced not in {"0", "false", "no", "off"}

    global _JUPYTER_PROXY_INLINE_CACHE
    if _JUPYTER_PROXY_INLINE_CACHE is not None:
        return _JUPYTER_PROXY_INLINE_CACHE

    try:
        import importlib.util

        _JUPYTER_PROXY_INLINE_CACHE = (
            importlib.util.find_spec("jupyter_server_proxy") is not None
        )
    except Exception:
        _JUPYTER_PROXY_INLINE_CACHE = False
    return _JUPYTER_PROXY_INLINE_CACHE


def _make_jupyter_proxy_inline_html(viewer_url: str, port: int, height: int):
    from IPython.display import HTML

    parsed = urllib.parse.urlparse(viewer_url)
    proxied_target = f"proxy/{port}{parsed.path or '/'}"
    if parsed.query:
        proxied_target += f"?{parsed.query}"
    initial_src = urllib.parse.urljoin(_jupyter_base_url_prefix(), proxied_target)
    container_id = f"arrayview-inline-{uuid.uuid4().hex}"

    return HTML(
        f"""
<div id={json.dumps(container_id)} style="width:100%;height:{height}px;background:#0c0c0c;overflow:hidden;border:0;border-radius:6px;">
  <iframe
    src={json.dumps(initial_src)}
    title="ArrayView"
    loading="eager"
    referrerpolicy="same-origin"
    allowfullscreen
    style="width:100%;height:100%;border:0;display:block;background:#0c0c0c;"
  ></iframe>
</div>
<script>
(function() {{
  const host = document.getElementById({json.dumps(container_id)});
  if (!host) return;
  const frame = host.querySelector('iframe');
  if (!frame) return;
  const baseCandidates = [
    document.body && document.body.dataset ? document.body.dataset.baseUrl : '',
    window.Jupyter && window.Jupyter.notebook ? window.Jupyter.notebook.base_url : '',
    window.jupyterapp && window.jupyterapp.serviceManager && window.jupyterapp.serviceManager.serverSettings
      ? window.jupyterapp.serviceManager.serverSettings.baseUrl
      : '',
  ];
  let base = baseCandidates.find(value => typeof value === 'string' && value.length) || {json.dumps(_jupyter_base_url_prefix())};
  if (!base.startsWith('/')) base = '/' + base;
  if (!base.endsWith('/')) base += '/';
  frame.src = new URL({json.dumps(proxied_target)}, window.location.origin + base).toString();
}})();
</script>
""".strip()
    )


# ── ViewHandle and view() API ────────────────────────────────────


class ViewHandle(str):
    """Returned by :func:`view`.  Behaves as a URL string for backward compatibility
    and additionally exposes ``.update(arr)`` to push a new array into the viewer
    without reopening a window.

    Example::

        v = view(arr)
        # ... modify arr ...
        v.update(arr2)        # viewer refreshes in-place
    """

    def __new__(cls, url: str, sid: str, port: int):
        obj = super().__new__(cls, url)
        obj._sid = sid
        obj._port = port
        return obj

    @property
    def sid(self) -> str:
        """Session ID for the viewer."""
        return self._sid

    @property
    def url(self) -> str:
        """Viewer URL (same as ``str(handle)``)."""
        return str(self)

    @property
    def port(self) -> int:
        """Port the ArrayView server is listening on."""
        return self._port

    def update(self, arr) -> None:
        """Push *arr* to the viewer, replacing the current data in-place.

        The viewer refreshes automatically (no window reload needed).
        Accepts any array-like that can be converted to numpy.
        """
        import io as _io

        import numpy as _np
        import urllib.request as _req

        if not isinstance(arr, _np.ndarray):
            arr = _np.array(arr)
        buf = _io.BytesIO()
        _np.save(buf, arr)
        body = buf.getvalue()
        request = _req.Request(
            f"http://127.0.0.1:{self._port}/update/{self._sid}",
            data=body,
            method="POST",
        )
        try:
            with _req.urlopen(request, timeout=10) as resp:
                resp.read()
        except Exception as e:
            raise RuntimeError(
                f"[ArrayView] Failed to update viewer: {e}\n"
                f"  URL: http://127.0.0.1:{self._port}/update/{self._sid}\n"
                f"  Is the ArrayView server still running?"
            ) from e


def view(
    *arrays,
    name=None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: str | bool | None = None,
    rgb: bool | list = False,
    overlay=None,
    floating: bool = False,
):
    """
    Launch the viewer. Does not block the main Python process.

    Accepts one to four arrays::

        handle = arrayview.view(a)
        ha, hb = arrayview.view(a, b)
        ha, hb, hc = arrayview.view(a, b, c)
        ha, hb, hc, hd = arrayview.view(a, b, c, d)

    ``name`` and ``rgb`` can each be a scalar (broadcast to all arrays) or a
    list of the same length as the number of arrays.

    ``window`` controls how the viewer opens:
      - ``None``       auto: native window outside Jupyter, inline IFrame inside Jupyter
      - ``True``       native window (falls back to browser if unavailable)
      - ``False``      no automatic opening; returns URL
      - ``'native'``   open in a native desktop window
      - ``'browser'``  open in the system browser
      - ``'vscode'``   open in a VS Code tab
      - ``'inline'``   return an inline IFrame (Jupyter / VS Code notebook)

    Persistent defaults can be set via ``arrayview config set window.<env> <mode>``
    where ``<env>`` is one of: terminal, vscode, jupyter, ssh, julia.
    The ``ARRAYVIEW_WINDOW`` environment variable overrides config file settings.

    ``rgb`` — treat the array as RGB/RGBA. The first or last dimension must
    have size 3 (RGB) or 4 (RGBA). When True, the colorbar is hidden and each
    slice is composited directly from the colour channels.

    ``overlay`` — a single array or list of arrays to composite as overlays.
    Each overlay is assigned an auto-palette color from _OVERLAY_PALETTE.

    Returns a ``ViewHandle`` for a single array, or a tuple of ``ViewHandle``
    objects for multiple arrays (one per array). In inline/Jupyter mode with
    multiple arrays, the IFrame is displayed automatically and a uniform tuple
    of ``ViewHandle`` objects is returned.

    Note: in Jupyter inline mode, single-array calls return an IFrame object for auto-display
    rather than a ViewHandle.
    """
    import numpy as np
    from arrayview._io import _tensor_to_numpy

    # --- Validate array count ---
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("view() requires at least one array.")
    if n_arrays > 4:
        raise ValueError(f"view() accepts at most 4 arrays, got {n_arrays}.")

    # --- Normalise per-array kwargs (name, rgb) ---
    if isinstance(name, list):
        if len(name) != n_arrays:
            raise ValueError(
                f"name list length ({len(name)}) must match number of arrays ({n_arrays})."
            )
        names = list(name)
    else:
        names = [name] * n_arrays  # scalar broadcast (None is valid)

    if isinstance(rgb, list):
        if len(rgb) != n_arrays:
            raise ValueError(
                f"rgb list length ({len(rgb)}) must match number of arrays ({n_arrays})."
            )
        rgbs = list(rgb)
    else:
        rgbs = [rgb] * n_arrays  # scalar broadcast

    # --- Duck-type conversion for ALL arrays ---
    def _convert_array(data):
        if not isinstance(data, np.ndarray):
            _mod = type(data).__module__ or ""
            _is_lazy = "nibabel" in _mod or "zarr" in _mod or "h5py" in _mod
            if not _is_lazy:
                if (
                    hasattr(data, "detach")  # PyTorch
                    or hasattr(data, "numpy")  # PyTorch / TF / JAX
                    or hasattr(data, "__jax_array__")  # JAX
                    or "juliacall" in _mod.lower()  # Julia via PythonCall
                ):
                    data = _tensor_to_numpy(data, "view()")
                else:
                    try:
                        converted = np.array(data)
                        if converted.dtype != object:
                            data = converted
                    except Exception:
                        pass
        return data

    arrays = tuple(_convert_array(arr) for arr in arrays)

    # --- Apply name defaults after conversion (shape is available now) ---
    names = [
        (f"Array {getattr(arrays[i], 'shape', '?')}" if names[i] is None else names[i])
        for i in range(n_arrays)
    ]

    # Convenience: primary data and name
    data = arrays[0]
    name = names[0]
    rgb_primary = rgbs[0]

    # --- Normalise string window modes ---
    _force_browser = False
    _force_vscode = False
    _explicit_inline = inline is not None
    _explicit_window = window is not None
    if isinstance(window, str):
        _w = window.lower()
        if _w == "inline":
            inline = True
            window = False
        elif _w == "native":
            inline = False
            window = True
        elif _w == "browser":
            window = False
            inline = False
            _force_browser = True
        elif _w == "vscode":
            window = False
            inline = False
            _force_vscode = True
        else:
            raise ValueError(
                f"window must be 'inline', 'native', 'browser', or 'vscode', got {window!r}"
            )

    # Remote/tunnel: if the caller didn't override the port and a --serve server
    # is already running on the CLI default (8000), use that instead of 8123.
    _CLI_DEFAULT_PORT = 8000
    if port == 8123 and _is_vscode_remote() and _server_alive(_CLI_DEFAULT_PORT):
        port = _CLI_DEFAULT_PORT

    # --- Julia path: only single-array supported ---
    if _is_julia_env():
        if n_arrays > 1:
            raise NotImplementedError(
                "Multi-array view() is not yet supported in Julia mode"
            )

        # VS Code tunnel/remote: use direct webview mode (SHM + postMessage),
        # same as the Python path.  The subprocess/WebSocket approach doesn't
        # work on tunnels because the port isn't accessible.
        if _is_vscode_remote():
            arr = np.array(data) if not isinstance(data, np.ndarray) else data
            _ensure_vscode_extension()
            _open_direct_via_shm(arr, name=name, title=f"ArrayView: {name}", floating=floating)
            print(
                f"\n  [ArrayView] Opened in VS Code webview (direct mode, no port needed).\n",
                flush=True,
            )
            # Block until the subprocess has read and unlinked the SHM.
            import time as _time
            from arrayview._vscode import _ACTIVE_SHM
            shm_paths = [f"/dev/shm/{shm.name}" for shm in _ACTIVE_SHM]
            _deadline = _time.monotonic() + 30.0
            while _time.monotonic() < _deadline:
                if not any(os.path.exists(p) for p in shm_paths):
                    break
                _time.sleep(0.2)
            return None

        if window is True:
            _inline, _window = False, True
        elif inline is True or window is False:
            _inline, _window = True, False
        elif inline is False:
            _inline, _window = False, False
        else:
            _inline = _in_julia_jupyter()
            _window = False
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window=_window,
            inline=_inline,
            height=height,
            floating=floating,
        )

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter
    if window is None:
        window = not is_jupyter
    if window:
        inline = False

    # User config: apply persistent window preference if no explicit arg was given
    if (
        not _explicit_window
        and not _explicit_inline
        and not _force_browser
        and not _force_vscode
    ):
        from arrayview._config import get_window_default
        from arrayview._platform import detect_environment

        _cfg_window = get_window_default(detect_environment())
        if _cfg_window:
            if _cfg_window == "inline":
                inline = True
                window = False
            elif _cfg_window == "native":
                window = True
            elif _cfg_window == "browser":
                window = False
                _force_browser = True
            elif _cfg_window == "vscode":
                window = False
                _force_vscode = True

    # Auto-detect VS Code terminal: prefer VS Code tab over native window
    if (
        _in_vscode_terminal()
        and not inline
        and not _explicit_window
        and not _explicit_inline
        and not _force_vscode
        and not _force_browser
    ):
        _force_vscode = True
        if window is True:
            window = False

    # Julia/PythonCall (second check after inline/window resolution)
    if not inline and _is_julia_env():
        if n_arrays > 1:
            raise NotImplementedError(
                "Multi-array view() is not yet supported in Julia mode"
            )
        # VS Code tunnel/remote: use direct webview mode (SHM + postMessage).
        if _is_vscode_remote():
            arr = np.array(data) if not isinstance(data, np.ndarray) else data
            _ensure_vscode_extension()
            _open_direct_via_shm(arr, name=name, title=f"ArrayView: {name}", floating=floating)
            print(
                f"\n  [ArrayView] Opened in VS Code webview (direct mode, no port needed).\n",
                flush=True,
            )
            import time as _time
            from arrayview._vscode import _ACTIVE_SHM
            shm_paths = [f"/dev/shm/{shm.name}" for shm in _ACTIVE_SHM]
            _deadline = _time.monotonic() + 30.0
            while _time.monotonic() < _deadline:
                if not any(os.path.exists(p) for p in shm_paths):
                    break
                _time.sleep(0.2)
            return None
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window,
            floating=floating,
        )

    # VS Code tunnel/remote (non-Jupyter): direct webview mode (no port needed).
    # Pass array via shared memory.  We must keep the process alive until the
    # extension subprocess has read the SHM, otherwise atexit unlinks it.
    # This must be checked BEFORE _server_alive() to avoid falling into the
    # port-based path when a stale server happens to be running.
    if not inline and _is_vscode_remote() and not _in_jupyter():
        _ensure_vscode_extension()
        _open_direct_via_shm(data, name=name, title=f"ArrayView: {name}", floating=floating)
        print(
            f"\n  [ArrayView] Opened in VS Code webview (direct mode, no port needed).\n",
            flush=True,
        )
        # Block until the subprocess has read and unlinked the SHM.
        # On Linux, shm_unlink removes /dev/shm/<name> — poll for that.
        # Timeout after 30s to avoid hanging forever.
        import time as _time
        from arrayview._vscode import _ACTIVE_SHM
        shm_paths = [f"/dev/shm/{shm.name}" for shm in _ACTIVE_SHM]
        _deadline = _time.monotonic() + 30.0
        while _time.monotonic() < _deadline:
            if not any(os.path.exists(p) for p in shm_paths):
                break
            _time.sleep(0.2)
        return None

    # VS Code tunnel/remote + Jupyter: open in a VS Code webview tab via
    # direct mode (SHM).  Inline IFrames don't work in VS Code tunnel notebooks
    # because the notebook webview can't reach localhost through the tunnel.
    # The kernel stays alive, so SHM won't be cleaned up prematurely.
    if _is_vscode_remote() and _in_jupyter():
        _ensure_vscode_extension()
        _open_direct_via_shm(data, name=name, title=f"ArrayView: {name}", floating=floating)
        try:
            from IPython.display import HTML, display as _ipy_display

            _ipy_display(
                HTML(
                    f"<div style='padding:8px;color:#888;font-family:monospace;font-size:13px'>"
                    f"[ArrayView] Opened <b>{name}</b> in VS Code tab"
                    f" (inline not available in tunnel sessions)</div>"
                )
            )
        except Exception:
            pass
        return None

    # VS Code tunnel/remote with an existing --serve server: register the array
    # via /load and open the viewer through the signal-file mechanism.
    if not _is_vscode_remote() and _server_alive(port):
        try:
            import tempfile as _tf

            # Load primary array
            with _tf.NamedTemporaryFile(suffix=".npy", delete=False) as _tmp:
                _tmp_path = _tmp.name
            np.save(_tmp_path, data)
            body = json.dumps(
                {"filepath": _tmp_path, "name": name, "rgb": rgb_primary}
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            try:
                os.unlink(_tmp_path)
            except Exception:
                pass
            if "error" in result:
                raise RuntimeError(result["error"])
            sid = result["sid"]

            # Load compare arrays (arrays[1:])
            _compare_sids = []
            for _ci, _carr in enumerate(arrays[1:], start=1):
                with _tf.NamedTemporaryFile(suffix=".npy", delete=False) as _ctmp:
                    _ctmp_path = _ctmp.name
                np.save(_ctmp_path, _carr)
                _cbody = json.dumps(
                    {"filepath": _ctmp_path, "name": names[_ci], "rgb": rgbs[_ci]}
                ).encode()
                _creq = urllib.request.Request(
                    f"http://127.0.0.1:{port}/load",
                    data=_cbody,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(_creq, timeout=300) as _cresp:
                    _cresult = json.loads(_cresp.read())
                try:
                    os.unlink(_ctmp_path)
                except Exception:
                    pass
                if "error" in _cresult:
                    raise RuntimeError(_cresult["error"])
                _compare_sids.append(_cresult["sid"])

            url_viewer = _viewer_url(port, sid, compare_sids=_compare_sids)

            if inline:
                from IPython.display import IFrame, display as _ipy_display

                _inline_url = _viewer_url(
                    port, sid, compare_sids=_compare_sids, inline=True
                )
                if _should_use_jupyter_proxy_inline():
                    _inline_html = _make_jupyter_proxy_inline_html(_inline_url, port, height)
                    if n_arrays == 1:
                        return _inline_html
                    _ipy_display(_inline_html)
                    return tuple(ViewHandle(url_viewer, s, port) for s in [sid] + _compare_sids)
                iframe = IFrame(src=_inline_url, width="100%", height=height)
                if n_arrays == 1:
                    return iframe
                _ipy_display(iframe)
                return tuple(ViewHandle(url_viewer, s, port) for s in [sid] + _compare_sids)

            _open_browser(
                url_viewer, force_vscode=True, blocking=True, title=f"ArrayView: {name}", floating=floating
            )
            _print_viewer_location(url_viewer)
            if n_arrays == 1:
                return ViewHandle(url_viewer, sid, port)
            return tuple(ViewHandle(url_viewer, s, port) for s in [sid] + _compare_sids)
        except Exception as e:
            _vprint(
                f"[ArrayView] Failed to register with --serve server: {e}", flush=True
            )
            # Fall through to subprocess/in-process paths.

    from arrayview._render import _setup_rgb

    session = _session_mod.Session(data, name=name)
    if rgb_primary:
        _setup_rgb(session)
    _session_mod.SESSIONS[session.sid] = session

    # Register compare sessions (arrays[1:])
    _compare_sids = []
    for _ci, _carr in enumerate(arrays[1:], start=1):
        _cname = names[_ci]
        _crgb = rgbs[_ci]
        _csession = _session_mod.Session(_carr, name=_cname)
        if _crgb:
            _setup_rgb(_csession)
        _session_mod.SESSIONS[_csession.sid] = _csession
        _compare_sids.append(_csession.sid)

    # Register overlay sessions.
    _overlay_sids = []
    _overlay_colors = []
    if overlay is not None:
        _ov_list = overlay if isinstance(overlay, (list, tuple)) else [overlay]
        for _i, _ov_arr in enumerate(_ov_list):
            if not isinstance(_ov_arr, np.ndarray):
                try:
                    _ov_arr = np.array(_ov_arr)
                except Exception:
                    pass
            _ov_session = _session_mod.Session(_ov_arr, name=f"overlay {_i + 1}")
            _session_mod.SESSIONS[_ov_session.sid] = _ov_session
            _overlay_sids.append(_ov_session.sid)
            _overlay_colors.append(_OVERLAY_PALETTE[_i % len(_OVERLAY_PALETTE)])

    win_w, win_h = 1400, 900

    # Start (or restart) the background server if it isn't responding or is stale.
    server_pid = _server_pid(port)
    our_pid = os.getpid()
    if server_pid is not None and server_pid != our_pid:
        # A stale ArrayView server (different process) is on this port — sessions
        # stored in our process won't be visible to it.  Kill it so we can bind.
        _vprint(
            f"[ArrayView] Stale server (pid {server_pid}) on port {port}, terminating it...",
            flush=True,
        )
        import signal as _signal

        try:
            os.kill(server_pid, _signal.SIGTERM)
        except Exception:
            pass
        # Wait up to 1 s for a clean exit, then SIGKILL.
        for _ in range(10):
            if not _port_in_use(port):
                break
            time.sleep(0.1)
        if _port_in_use(port):
            try:
                os.kill(server_pid, _signal.SIGKILL)
            except Exception:
                pass
            # Wait up to 2 more seconds after SIGKILL.
            for _ in range(20):
                if not _port_in_use(port):
                    break
                time.sleep(0.1)
        server_pid = None  # treat as not alive

    if server_pid is None:
        if _port_in_use(port) and not _server_alive(port):
            # Port busy by another process — auto-scan for a free one.
            port, _already = _find_server_port(port + 1)
            if _port_in_use(port) and not _server_alive(port):
                raise RuntimeError(
                    f"Port {port} is already in use by another process. "
                    f"Choose a different port in view(..., port=...)."
                )
            _vprint(f"[ArrayView] Default port busy, using port {port}", flush=True)
        if _is_vscode_remote():
            _configure_vscode_port_preview(port)
        _session_mod.SERVER_LOOP = None  # reset so we wait for the new loop below
        _server_ready_event.clear()
        _script = _is_script_mode()
        threading.Thread(
            target=lambda: asyncio.run(
                _serve_background(port, stop_when_closed=_script)
            ),
            daemon=not _script,
            name="arrayview-server",
        ).start()

        can_native_window = _can_native_window() if window else False
        _early_window_opened = False
        if (
            window
            and can_native_window
            and not _force_browser
            and not _force_vscode
            and not inline
        ):
            try:
                wp = _session_mod._window_process
                if wp is None or wp.poll() is not None:
                    url_shell_early = _shell_url(
                        port, session.sid, name, compare_sids=_compare_sids
                    )
                    _session_mod._window_process = _open_webview_with_fallback(
                        url_shell_early, win_w, win_h, shell_port=port, floating=floating
                    )
                    _early_window_opened = True
            except Exception:
                pass

        # Wait for the socket to be ready (event-based, much faster than HTTP polling).
        if not _server_ready_event.wait(timeout=10.0):
            if _port_in_use(port) and not _server_alive(port):
                raise RuntimeError(
                    f"Port {port} is in use by another process (not ArrayView). "
                    f"Choose a different port in view(..., port=...)."
                )
            raise RuntimeError(
                f"ArrayView server did not start on port {port} within timeout."
            )
        _platform_mod._jupyter_server_port = port
    else:
        if _is_vscode_remote():
            _configure_vscode_port_preview(port)
        _platform_mod._jupyter_server_port = port  # server already ours on this port
        can_native_window = _can_native_window() if window else False
        _early_window_opened = False
        global _loading_port
        _loading_port = None  # server is already up — no loading page needed

    # SERVER_LOOP is set as the first statement of _serve_background, before
    # _server_ready_event fires, so it is guaranteed non-None by this point.

    url_viewer = _viewer_url(
        port,
        session.sid,
        compare_sids=_compare_sids,
        overlay_sids=_overlay_sids,
        overlay_colors=_overlay_colors,
    )
    url_shell = _shell_url(
        port, session.sid, name, compare_sids=_compare_sids
    )

    if inline:
        from IPython.display import IFrame, display as _ipy_display

        # Add inline=1 param so the viewer starts in immersive mode
        _inline_url = _viewer_url(
            port,
            session.sid,
            compare_sids=_compare_sids,
            overlay_sids=_overlay_sids,
            overlay_colors=_overlay_colors,
            inline=True,
        )
        if _should_use_jupyter_proxy_inline():
            _inline_html = _make_jupyter_proxy_inline_html(_inline_url, port, height)
            if n_arrays == 1:
                return _inline_html
            _ipy_display(_inline_html)
            handles = tuple(ViewHandle(url_viewer, s, port) for s in [session.sid] + _compare_sids)
            return handles
        iframe = IFrame(src=_inline_url, width="100%", height=height)
        if n_arrays == 1:
            return iframe
        # Multi-array inline: display the IFrame and return a uniform tuple of handles.
        _ipy_display(iframe)
        handles = tuple(ViewHandle(url_viewer, s, port) for s in [session.sid] + _compare_sids)
        return handles

    if window and can_native_window and not _force_browser and not _force_vscode:
        if _early_window_opened:
            # Window already open with inline shell — push initial tab via WebSocket.
            # Fire-and-forget: don't block view() while waiting for shell WS to connect.
            # wait=True inside the coroutine polls up to 2 s for the shell to connect.
            try:
                server_loop = _session_mod.SERVER_LOOP
                if server_loop is not None:
                    tab_url = _viewer_path(
                        session.sid,
                        compare_sids=_compare_sids,
                        overlay_sids=_overlay_sids,
                        overlay_colors=_overlay_colors,
                    )
                    asyncio.run_coroutine_threadsafe(
                        _server_mod()._notify_shells(session.sid, name, url=tab_url, wait=True),
                        server_loop,
                    )
            except Exception:
                pass
        else:
            try:
                wp = _session_mod._window_process
                server_loop = _session_mod.SERVER_LOOP
                window_alive = wp is not None and wp.poll() is None
                notified = False
                if server_loop is not None and (window_alive or _server_alive(port)):
                    future = asyncio.run_coroutine_threadsafe(
                        _server_mod()._notify_shells(
                            session.sid, name, wait=window_alive
                        ),
                        server_loop,
                    )
                    try:
                        notified = future.result(timeout=3.0)
                    except Exception:
                        notified = False
                if not notified:
                    _session_mod._window_process = _open_webview_with_fallback(
                        url_shell, win_w, win_h, shell_port=port, floating=floating
                    )
            except Exception:
                _open_browser(
                    _with_loading(url_viewer), force_vscode=_force_vscode, title=f"ArrayView: {name}", floating=floating
                )
    else:
        if (
            window
            and not can_native_window
            and not _force_browser
            and not _force_vscode
        ):
            _vprint(
                "[ArrayView] Native window unavailable; opening browser fallback",
                flush=True,
            )
        _open_browser(
            _with_loading(url_viewer), force_vscode=_force_vscode, title=f"ArrayView: {name}", floating=floating
        )

    _print_viewer_location(url_viewer)
    if n_arrays == 1:
        return ViewHandle(url_viewer, session.sid, port)
    return tuple(ViewHandle(url_viewer, s, port) for s in [session.sid] + _compare_sids)


# ── Server Lifecycle ──────────────────────────────────────────────


def _is_script_mode() -> bool:
    """True when running as a plain Python script (not interactive REPL, not Jupyter, not Julia)."""
    if _in_jupyter() or _is_julia_env():
        return False
    if sys.flags.interactive or hasattr(sys, "ps1"):
        return False
    return True


async def _stop_server_when_viewer_closes(
    server, connect_timeout: float = 20.0, grace_seconds: float = 1.0
) -> None:
    """Asyncio task: signal uvicorn to stop once the viewer window is fully closed.
    Used in script mode so the non-daemon server thread exits cleanly when done."""
    import arrayview._session as _sm

    deadline = time.monotonic() + connect_timeout
    while _sm.VIEWER_SOCKETS == 0:
        if time.monotonic() > deadline:
            server.should_exit = True  # no viewer connected; give up
            return
        await asyncio.sleep(0.2)
    # At least one viewer connected — now wait for all to disconnect.
    while True:
        while _sm.VIEWER_SOCKETS > 0:
            await asyncio.sleep(0.2)
        # Grace period: let page refreshes reconnect.
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            if _sm.VIEWER_SOCKETS > 0:
                break
            await asyncio.sleep(0.08)
        else:
            server.should_exit = True
            return


def _wait_for_viewer_close(
    grace_seconds: float = 1.0, idle_seconds: float = 0.0
) -> None:
    """Block until all viewer WebSocket connections close.

    Waits for a viewer WebSocket to connect, then all to disconnect, then applies a
    brief grace period so page refreshes don't prematurely kill the server.

    If ``idle_seconds > 0``, the server stays alive that many extra seconds after
    the grace period so the next ``arrayview`` CLI invocation can reuse it without
    spawning a new subprocess.  A new viewer connection resets the countdown.
    """
    import arrayview._session as _sm

    while _sm.VIEWER_SOCKETS == 0:
        time.sleep(0.2)
    while True:
        while _sm.VIEWER_SOCKETS > 0:
            time.sleep(0.2)
        # Grace period: let page refreshes reconnect.
        deadline = time.monotonic() + grace_seconds
        reconnected = False
        while time.monotonic() < deadline:
            if _sm.VIEWER_SOCKETS > 0:
                reconnected = True
                break
            time.sleep(0.08)
        if reconnected:
            continue  # back to Phase 2 (wait for close)
        if idle_seconds <= 0:
            return  # no idle timeout → exit immediately
        # Idle phase: keep the server alive so the next CLI call reuses it.
        # A new viewer resets the timer; total inactivity → exit cleanly.
        idle_deadline = time.monotonic() + idle_seconds
        while time.monotonic() < idle_deadline:
            if _sm.VIEWER_SOCKETS > 0:
                break  # new viewer connected → back to Phase 2
            time.sleep(1.0)
        else:
            return  # idle timeout expired — really done


def _view_julia(
    data: "np.ndarray",
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 500,
    floating: bool = False,
):
    """Julia-specific view() path: run the server in a subprocess so it is
    completely independent of Julia's GIL.
    """

    # Detect VS Code *now*, in the parent process where TERM_PROGRAM and
    # VSCODE_IPC_HOOK_CLI are still available.  The subprocess inherits a
    # stripped environment (Julia/PythonCall, uv run, etc.) so detection
    # there would fail and the VS Code tab would never open.
    force_vscode = _in_vscode_terminal()
    return _view_subprocess(
        data,
        name,
        port,
        window,
        inline=inline,
        height=height,
        force_vscode=force_vscode,
        floating=floating,
    )


def _view_subprocess(
    data: "np.ndarray",
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 500,
    rgb: bool = False,
    force_vscode: bool = False,
    floating: bool = False,
) -> str:
    """Run the viewer in a separate subprocess server.

    Used when the calling process may exit shortly after view() returns
    (Julia, VS Code tunnel one-shot scripts, CLI).  The subprocess server
    survives because it is not a daemon thread.
    """
    import numpy as np
    import tempfile

    # Persist the array to a temp file so the server subprocess can load it.
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name
    np.save(tmp_path, data)

    tab_injected = False  # True when an existing shell window received the new tab
    if _server_alive(port):
        # Existing subprocess server — register the new array via /load.
        # Pass notify=True so the server injects a new tab into any open shell
        # window rather than requiring the caller to open a new native window.
        try:
            body = json.dumps(
                {"filepath": tmp_path, "name": name, "rgb": rgb, "notify": True}
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                raise RuntimeError(result["error"])
            # Data is now in server memory; temp file no longer needed.
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            sid = result["sid"]
            tab_injected = bool(result.get("notified", False))
        except Exception as e:
            _vprint(
                f"[ArrayView] Failed to register with existing server: {e}", flush=True
            )
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
    else:
        if _port_in_use(port):
            # Port busy by another process — auto-scan for a free one.
            port, _already = _find_server_port(port + 1)
            if _port_in_use(port):
                raise RuntimeError(
                    f"Port {port} is already in use by another process. "
                    f"Choose a different port in view(..., port=...)."
                )
            _vprint(f"[ArrayView] Default port busy, using port {port}", flush=True)
        sid = uuid.uuid4().hex
        # Spawn a self-contained server subprocess (same as CLI path).
        script = (
            f"from arrayview._launcher import _serve_daemon;"
            f"_serve_daemon({repr(tmp_path)}, {port}, {repr(sid)}, "
            f"name={repr(name)}, cleanup=True, rgb={rgb})"
        )
        subprocess.Popen(
            [sys.executable, "-c", script],
        )

        if not _wait_for_port(port, timeout=15.0, tcp_only=True):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise RuntimeError(f"ArrayView server failed to start on port {port}.")

    url_viewer = _viewer_url(port, sid)
    url_shell = _shell_url(port, sid, name)
    _print_viewer_location(url_viewer)

    if inline:
        _inline_url = _viewer_url(port, sid, inline=True)
        iframe_html = (
            f"<iframe src='{_inline_url}' width='100%'"
            f" height='{height}' frameborder='0'></iframe>"
        )
        # IJulia kernel: push HTML through Julia's display stack (routes to Jupyter
        # frontend). Must be a side-effect call, not a return value, because
        # PythonCall would convert a Python IFrame object to an opaque Julia value.
        try:
            import juliacall as _jl

            _jl.Main.seval(f'display("text/html", "{iframe_html}")')
            return None
        except Exception:
            pass
        # Plain Python Jupyter kernel fallback.
        try:
            from IPython.display import HTML, display as _ipy_display

            _ipy_display(HTML(iframe_html))
        except Exception:
            pass
        return url_viewer

    if tab_injected:
        # Tab was injected into the existing native shell window — no new window needed.
        _vprint("[ArrayView] New tab injected into existing window", flush=True)
        return ViewHandle(url_viewer, sid, port)

    can_native = _can_native_window()
    if window and can_native:
        if not _open_webview_cli(url_shell, 1400, 900):
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _open_browser(
                url_viewer,
                force_vscode=force_vscode,
                blocking=force_vscode,
                title=f"ArrayView: {name}",
                floating=floating,
            )
    else:
        # blocking=True when force_vscode so signal file is written before
        # returning to Julia (daemon thread would be killed on process exit).
        _open_browser(
            url_viewer,
            force_vscode=force_vscode,
            blocking=force_vscode,
            title=f"ArrayView: {name}",
            floating=floating,
        )
    return ViewHandle(url_viewer, sid, port)


def _serve_empty(port: int) -> None:
    """Background server process with no sessions. Runs until killed.

    Used for ``arrayview --serve`` (pre-warm) and on remote tunnel sessions so
    the port stays alive across multiple tab opens/closes without requiring the
    user to re-run ``--serve`` or re-set port visibility.
    """
    _session_mod.SERVER_PORT = port
    threading.Thread(
        target=lambda: _uvicorn().run(
            _server_mod().app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            timeout_keep_alive=30,
        ),
        daemon=True,
    ).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    os._exit(0)


def _serve_daemon(
    filepath: str,
    port: int,
    sid: str,
    name: str = None,
    cleanup: bool = False,
    overlay_filepaths: list | None = None,
    overlay_sids: list | None = None,
    compare_filepath: str = None,
    compare_sid: str = None,
    vfield_filepath: str = None,
    vfield_components_dim: int = None,
    persist: bool = False,
    rgb: bool = False,
) -> None:
    """Background server process. Loads data, serves it.
    persist=True: never exits (used on remote tunnel so port stays alive).
    persist=False: exits when the UI closes (default, used locally).
    cleanup=True: delete filepath after loading (used when it is a temp file).
    """
    # Register sid as pending so /metadata can poll while data loads.
    _session_mod.PENDING_SESSIONS.add(sid)
    _session_mod.SERVER_PORT = port

    # Start uvicorn immediately — the window can open before data is ready.
    threading.Thread(
        target=lambda: _uvicorn().run(
            _server_mod().app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            timeout_keep_alive=30,
        ),
        daemon=True,
    ).start()

    # Pre-warm colormap LUTs in background (saves ~200 ms on first frame render).
    def _warm_luts():
        try:
            from arrayview._render import _init_luts
            _init_luts()
        except Exception:
            pass
    threading.Thread(target=_warm_luts, daemon=True).start()

    def _load():
        from arrayview._io import load_data, load_data_with_meta

        try:
            data, spatial_meta = load_data_with_meta(filepath)
            if cleanup:
                try:
                    os.unlink(filepath)
                except Exception:
                    pass
            session = _session_mod.Session(
                data, filepath=None if cleanup else filepath, name=name
            )
            session.sid = sid
            session.spatial_meta = spatial_meta
            if spatial_meta is not None:
                session.original_volume = data
            if rgb:
                from arrayview._render import _setup_rgb

                _setup_rgb(session)
            if vfield_filepath:
                try:
                    from arrayview._vectorfield import _configure_vectorfield

                    vf_data = load_data(vfield_filepath)
                    _configure_vectorfield(session, vf_data, vfield_components_dim)
                    _vprint(
                        f"[ArrayView] Loaded vector field {vfield_filepath} shape {vf_data.shape} component_axis={session.vfield_component_dim}",
                        flush=True,
                    )
                except Exception as e:
                    _vprint(
                        f"[ArrayView] Warning: failed to load vector field {vfield_filepath}: {e}",
                        flush=True,
                    )
            _session_mod.SESSIONS[session.sid] = session
            for ov_path, ov_sid in zip(overlay_filepaths or [], overlay_sids or []):
                try:
                    ov_data = load_data(ov_path)
                    ov_session = _session_mod.Session(
                        ov_data, filepath=ov_path, name=os.path.basename(ov_path) or "overlay"
                    )
                    ov_session.sid = ov_sid
                    _session_mod.SESSIONS[ov_sid] = ov_session
                except Exception as e:
                    _vprint(
                        f"[ArrayView] Warning: failed to load overlay {ov_path}: {e}",
                        flush=True,
                    )
            if compare_filepath and compare_sid:
                try:
                    cmp_data = load_data(compare_filepath)
                    cmp_name = os.path.basename(compare_filepath) or "compare"
                    cmp_session = _session_mod.Session(
                        cmp_data, filepath=compare_filepath, name=cmp_name
                    )
                    cmp_session.sid = compare_sid
                    _session_mod.SESSIONS[compare_sid] = cmp_session
                except Exception as e:
                    _vprint(
                        f"[ArrayView] Warning: failed to load compare array {compare_filepath}: {e}",
                        flush=True,
                    )
        finally:
            _session_mod.PENDING_SESSIONS.discard(sid)

    threading.Thread(target=_load, daemon=True).start()

    if persist:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        # Keep the server alive briefly after the last viewer closes (grace period
        # for page refreshes). No extended idle timeout — shut down immediately
        # when the last window closes for cleaner debugging and resource management.
        _wait_for_viewer_close(idle_seconds=0)
    os._exit(0)


# ── Demo Array and File Watching ──────────────────────────────────


def _make_demo_array() -> "np.ndarray":
    """Return a (128, 128, 32, 3) float32 RGB plasma animation.

    Dims: 128x128 canvas, 32 animation frames, 3 colour channels (RGB).
    Each frame is a smooth plasma / interference pattern built from
    overlapping sine waves -- colourful and visually rich at every zoom level.
    Values are in [0, 255] so the array is ready for direct RGB viewing.
    """
    import numpy as np

    H, W, T = 128, 128, 32
    arr = np.zeros((H, W, T, 3), dtype=np.float32)

    # Spatial grids (normalised to [0, 1])
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # shape (H, W)

    for ti in range(T):
        ph = 2.0 * np.pi * ti / T  # animation phase

        # Three overlapping plasma waves with distinct spatial frequencies
        p0 = np.sin(6.0 * np.pi * X + ph) + np.sin(6.0 * np.pi * Y + ph * 0.7)
        p1 = np.sin(9.0 * np.pi * (X + Y) + ph * 1.3) + np.sin(
            np.pi * (np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2) * 12.0 - ph)
        )
        p2 = np.sin(7.0 * np.pi * X * Y + ph * 0.5)

        # Each channel driven by a different mix → distinct hue cycling
        r = 0.5 + 0.5 * np.sin(p0 * 1.0 + ph * 0.0)
        g = 0.5 + 0.5 * np.sin(p1 * 1.0 + ph * 0.33)
        b = 0.5 + 0.5 * np.sin(p2 * 1.0 + ph * 0.67)

        arr[:, :, ti, 0] = (r * 255.0).astype(np.float32)
        arr[:, :, ti, 1] = (g * 255.0).astype(np.float32)
        arr[:, :, ti, 2] = (b * 255.0).astype(np.float32)

    return arr


def _start_watch_thread(filepath: str, sid: str, port: int) -> None:
    """Start a background thread that watches *filepath* for mtime changes.

    On each detected change the thread POSTs to /reload/{sid} on the local
    server so the viewer automatically re-renders the updated array.
    Runs as a daemon thread — exits automatically when the main process ends.
    """
    import urllib.request as _urlreq

    def _watch():
        try:
            last_mtime = os.stat(filepath).st_mtime
        except OSError:
            return
        while True:
            time.sleep(1.0)
            try:
                mtime = os.stat(filepath).st_mtime
            except OSError:
                continue
            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    req = _urlreq.Request(
                        f"http://127.0.0.1:{port}/reload/{sid}",
                        data=b"",
                        method="POST",
                    )
                    with _urlreq.urlopen(req, timeout=10) as resp:
                        result = json.loads(resp.read())
                    ver = result.get("version", "?")
                    print(
                        f"[ArrayView] File changed — reloaded (version {ver})",
                        flush=True,
                    )
                except Exception as e:
                    _vprint(f"[ArrayView] Watch reload error: {e}", flush=True)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()


# ── Config Subcommand ─────────────────────────────────────────────


def _handle_config_command(args: list[str]) -> None:
    """Handle 'arrayview config' subcommands."""
    from arrayview._config import (
        CONFIG_PATH,
        _VALID_ENV_KEYS,
        _VALID_WINDOW_MODES,
        load_config,
        save_config,
    )

    if not args or args[0] == "list":
        cfg = load_config()
        if not cfg:
            print("No configuration set.")
            print(f"Config file: {CONFIG_PATH}")
            return
        for section, values in cfg.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    print(f"{section}.{k} = {v}")
        print(f"\nConfig file: {CONFIG_PATH}")
        return

    if args[0] == "set":
        if len(args) != 3:
            print("Usage: arrayview config set <key> <value>")
            print("  e.g. arrayview config set window.terminal browser")
            print(f"  Valid window modes: {', '.join(sorted(_VALID_WINDOW_MODES))}")
            sys.exit(1)
        key, value = args[1], args[2]
        parts = key.split(".", 1)
        if len(parts) != 2:
            print(f"Key must be section.name, got: {key}")
            sys.exit(1)
        section, name = parts
        if section == "window":
            if name not in _VALID_ENV_KEYS:
                print(f"Unknown window key: {name}")
                print(f"  Valid keys: {', '.join(sorted(_VALID_ENV_KEYS))}")
                sys.exit(1)
            if value not in _VALID_WINDOW_MODES:
                print(f"Invalid window mode: {value}")
                print(f"  Valid modes: {', '.join(sorted(_VALID_WINDOW_MODES))}")
                sys.exit(1)
        cfg = load_config()
        cfg.setdefault(section, {})[name] = value
        save_config(cfg)
        print(f"Set {key} = {value}")
        return

    if args[0] == "get":
        if len(args) != 2:
            print("Usage: arrayview config get <key>")
            sys.exit(1)
        key = args[1]
        parts = key.split(".", 1)
        if len(parts) != 2:
            print(f"Key must be section.name, got: {key}")
            sys.exit(1)
        section, name = parts
        cfg = load_config()
        val = cfg.get(section, {}).get(name)
        if val is None:
            print(f"{key} is not set")
        else:
            print(f"{key} = {val}")
        return

    if args[0] == "reset":
        if os.path.isfile(CONFIG_PATH):
            os.remove(CONFIG_PATH)
            print("Configuration reset.")
        else:
            print("No configuration file to reset.")
        return

    if args[0] == "path":
        print(CONFIG_PATH)
        return

    print(f"Unknown config command: {args[0]}")
    print("Usage: arrayview config [list|set|get|reset|path]")
    sys.exit(1)


# ── CLI Entry Point (arrayview command) ───────────────────────────


def arrayview():
    """Command Line Interface Entry Point."""
    # Handle "arrayview config ..." subcommand before argparse
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        _handle_config_command(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "Array paths. First path is the base array; optional additional paths "
            "are preloaded for compare mode (up to 6 total files)."
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Start a persistent server on the given port without loading any data. "
            "Useful on VS Code remote tunnel: run this first, set the port to Public "
            "in the Ports tab, then use 'arrayview FILE' freely."
        ),
    )
    parser.add_argument(
        "--window",
        choices=["browser", "vscode", "native"],
        default=None,
        help="How to open the viewer: browser, vscode, or native. Overrides config (see 'arrayview config')",
    )
    parser.add_argument(
        "--floating",
        action="store_true",
        help="Open in a VS Code floating window (requires VS Code 1.85+)",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Deprecated: use --window browser",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Kill the ArrayView server running on --port (default 8000) and exit",
    )
    parser.add_argument(
        "--overlay",
        metavar="FILE",
        action="append",
        default=None,
        help=(
            "Segmentation mask to overlay (binary 0/1 array, same spatial shape). "
            "Repeat --overlay to load multiple overlays."
        ),
    )
    parser.add_argument(
        "--compare",
        metavar="FILE",
        help="Deprecated: second array for side-by-side compare mode",
    )
    parser.add_argument(
        "--vectorfield",
        metavar="FILE",
        help=(
            "Deformation vector field to overlay as arrows. "
            "Must have the same spatial shape as the image plus one axis of size 3 "
            "holding the xyz displacement components."
        ),
    )
    parser.add_argument(
        "--vectorfield-components-dim",
        metavar="DIM",
        type=int,
        default=None,
        help=(
            "Axis index of the xyz displacement components in --vectorfield. "
            "If omitted, arrayview auto-detects the unique axis of size 3."
        ),
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help=(
            "Interpret the array as an RGB or RGBA image. "
            "The first or last dimension must have size 3 or 4."
        ),
    )
    parser.add_argument(
        "--relay",
        metavar="[HOST:]PORT",
        help=(
            "Send the array to an existing ArrayView server instead of starting a new one. "
            "Useful over multi-hop SSH: set up a reverse tunnel "
            "(e.g. 'ssh -R 8765:localhost:8000 user@gpu-host') "
            "then run 'arrayview file.npy --relay 8765'. "
            "The remote server registers the session and opens a VS Code tab automatically."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (internal status messages)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print environment detection results and exit (useful for debugging VS Code/tunnel issues)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch FILE for changes and auto-reload the viewer when the file is modified",
    )
    parser.add_argument(
        "--dims",
        metavar="SPEC",
        default=None,
        help=(
            "Force initial x/y viewing dimensions. "
            "Use a comma-separated spec where 'x' and 'y' mark the spatial dims, "
            "e.g. 'x,y,:,:' (first two dims are spatial) or ':,:,x,y' (last two). "
            "Also accepts 0-based integer pair like '2,3'."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["server", "stdio"],
        default="server",
        help="Run mode: server (default HTTP/WS) or stdio (VS Code extension subprocess)",
    )
    parser.add_argument(
        "--shm-name",
        default=None,
        help="Shared memory block name (stdio mode: read array from shared memory instead of file)",
    )
    parser.add_argument(
        "--shm-shape",
        default=None,
        help="Array shape as comma-separated ints (required with --shm-name)",
    )
    parser.add_argument(
        "--shm-dtype",
        default=None,
        help="Array dtype string (required with --shm-name)",
    )
    parser.add_argument(
        "--name",
        default=None,
        dest="array_name",
        help="Display name for the array (stdio mode)",
    )
    args = parser.parse_args()
    _session_mod._verbose = args.verbose
    vfield_components_dim = None

    # --mode stdio: run the stdio server (for VS Code extension subprocess)
    if args.mode == "stdio":
        from pathlib import Path as _Path

        from arrayview._render import _setup_rgb
        from arrayview._stdio_server import run_stdio_server

        if args.shm_name:
            # Read array from shared memory (passed by view() via signal file)
            from multiprocessing.shared_memory import SharedMemory

            import numpy as np
            from arrayview._session import SESSIONS, Session

            shm = SharedMemory(name=args.shm_name)
            shape = tuple(int(x) for x in args.shm_shape.split(","))
            dtype = np.dtype(args.shm_dtype)
            data = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            shm.close()
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            session = Session(
                data=data,
                name=args.array_name or "array",
            )
            if getattr(args, "rgb", False):
                _setup_rgb(session)
            SESSIONS[session.sid] = session
            info = json.dumps(
                {
                    "sid": session.sid,
                    "name": session.name,
                    "shape": [int(s) for s in session.shape],
                }
            )
            print(f"SESSION:{info}", file=sys.stderr)
        elif args.files:
            from arrayview._io import load_data
            from arrayview._session import SESSIONS, Session

            # First file is the main array
            base_path = args.files[0]
            data = load_data(base_path)
            session = Session(
                data=data,
                filepath=base_path,
                name=_Path(base_path).name,
            )
            if getattr(args, "rgb", False):
                _setup_rgb(session)

            # Attach vector field if provided
            if getattr(args, "vectorfield", None):
                from arrayview._vectorfield import _configure_vectorfield

                vf_data = load_data(args.vectorfield)
                _configure_vectorfield(
                    session, vf_data,
                    getattr(args, "vectorfield_components_dim", None),
                )

            SESSIONS[session.sid] = session

            # Load overlay(s) as separate sessions
            overlay_sids = []
            for ov_path in (getattr(args, "overlay", None) or []):
                ov_data = load_data(ov_path)
                ov_session = Session(
                    data=ov_data,
                    filepath=ov_path,
                    name=_Path(ov_path).name or "overlay",
                )
                SESSIONS[ov_session.sid] = ov_session
                overlay_sids.append(ov_session.sid)
            overlay_sid = ",".join(overlay_sids) if overlay_sids else None

            # Load compare files (additional positional args + --compare)
            compare_sids = []
            compare_paths = list(args.files[1:])
            if getattr(args, "compare", None):
                compare_paths.append(args.compare)
            for cp in compare_paths:
                cmp_data = load_data(cp)
                cmp_session = Session(
                    data=cmp_data,
                    filepath=cp,
                    name=_Path(cp).name,
                )
                SESSIONS[cmp_session.sid] = cmp_session
                compare_sids.append(cmp_session.sid)

            info = json.dumps(
                {
                    "sid": session.sid,
                    "name": session.name,
                    "shape": [int(s) for s in session.shape],
                    "has_vectorfield": session.vfield is not None,
                    "overlay_sid": overlay_sid,
                    "compare_sids": compare_sids or None,
                }
            )
            print(f"SESSION:{info}", file=sys.stderr)
        run_stdio_server()
        return

    # --diagnose: print detection results and exit
    if getattr(args, "diagnose", False):
        import json as _json
        from ._platform import (
            _find_vscode_ipc_hook,
            _in_jupyter,
        )

        diag: dict = {
            "env": {
                "TERM_PROGRAM": os.environ.get("TERM_PROGRAM"),
                "VSCODE_IPC_HOOK_CLI": os.environ.get("VSCODE_IPC_HOOK_CLI"),
                "SSH_CONNECTION": os.environ.get("SSH_CONNECTION"),
                "SSH_CLIENT": os.environ.get("SSH_CLIENT"),
                "VSCODE_INJECTION": os.environ.get("VSCODE_INJECTION"),
                "VSCODE_AGENT_FOLDER": os.environ.get("VSCODE_AGENT_FOLDER"),
                "DISPLAY": os.environ.get("DISPLAY"),
                "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY"),
            },
            "detection": {
                "in_vscode_terminal": _in_vscode_terminal(),
                "is_vscode_remote": _is_vscode_remote(),
                "in_vscode_tunnel": _in_vscode_tunnel(),
                "can_native_window": _can_native_window(),
                "in_jupyter": _in_jupyter(),
                "vscode_ipc_hook_recovered": _find_vscode_ipc_hook(),
            },
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "platform": sys.platform,
            "python": sys.executable,
        }
        from arrayview._config import CONFIG_PATH, get_window_default, load_config
        from arrayview._platform import detect_environment

        _det_env = detect_environment()
        diag["config"] = {
            "detected_environment": _det_env,
            "config_file": CONFIG_PATH,
            "config_contents": load_config() or None,
            "ARRAYVIEW_WINDOW": os.environ.get("ARRAYVIEW_WINDOW") or None,
            "resolved_window_pref": get_window_default(_det_env),
        }
        print(_json.dumps(diag, indent=2))
        return

    # Parse --dims spec into (dim_x, dim_y) integers or None
    def _parse_dims_spec(spec: str) -> tuple[int, int] | None:
        parts = [p.strip().lower() for p in spec.split(",")]
        x_idx = next((i for i, p in enumerate(parts) if p == "x"), None)
        y_idx = next((i for i, p in enumerate(parts) if p == "y"), None)
        if x_idx is not None and y_idx is not None and x_idx != y_idx:
            return (x_idx, y_idx)
        # Fallback: try two integer indices like "2,3"
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                if a >= 0 and b >= 0 and a != b:
                    return (a, b)
            except ValueError:
                pass
        return None

    dims_override: tuple[int, int] | None = None
    if args.dims:
        dims_override = _parse_dims_spec(args.dims)
        if dims_override is None:
            parser.error(
                f"--dims {args.dims!r} is invalid. "
                "Use e.g. 'x,y,:,:' or ':,:,x,y' or '0,1'."
            )
    if not args.serve and not args.kill and not args.files:
        # No files given: launch the animated pixel-art demo
        import tempfile as _tempfile
        import numpy as _np_demo

        _demo_arr = _make_demo_array()
        _fd, _tmp_path = _tempfile.mkstemp(suffix=".npy")
        import os as _os_demo

        _os_demo.close(_fd)
        _np_demo.save(_tmp_path, _demo_arr)
        args.files = [_tmp_path]
        args._demo_name = "welcome"
        args._demo_cleanup = True
        args.rgb = True
    if args.files and len(args.files) > 6:
        parser.error(
            "At most six FILE arguments are supported; concat arrays first for larger compare sets."
        )
    if args.compare and len(args.files) > 1:
        parser.error("Use either positional compare files or --compare, not both.")

    # -- --relay: send array bytes to a remote ArrayView server --
    if args.relay:
        if not args.files:
            parser.error("--relay requires a FILE argument.")
        relay_str = args.relay
        if ":" in relay_str:
            relay_host, relay_port_str = relay_str.rsplit(":", 1)
        else:
            relay_host, relay_port_str = "127.0.0.1", relay_str
        try:
            relay_port = int(relay_port_str)
        except ValueError:
            parser.error(f"--relay port must be an integer, got: {relay_port_str!r}")
        relay_file = os.path.abspath(args.files[0])
        if not os.path.isfile(relay_file):
            print(f"Error: file not found: {relay_file}")
            sys.exit(1)
        relay_name = os.path.basename(relay_file)
        if not _server_alive(relay_port):
            print(
                f"[ArrayView] No ArrayView server found on {relay_host}:{relay_port}.\n"
                f"  Make sure the reverse tunnel is active:"
                f" ssh -R {relay_port}:localhost:8000 user@this-host",
                flush=True,
            )
            sys.exit(1)
        try:
            _relay_array_to_server(
                relay_file,
                relay_port,
                relay_name,
                args.rgb,
                relay_host=relay_host,
            )
        except Exception as e:
            print(f"[ArrayView] Relay failed: {e}", flush=True)
            sys.exit(1)
        return

    # -- --kill: stop the server on the given port --
    if args.kill:
        import signal as _signal

        if sys.platform == "win32":
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True
            )
            killed = False
            for line in result.stdout.splitlines():
                if f":{args.port}" in line and "LISTENING" in line:
                    parts = line.split()
                    try:
                        pid = int(parts[-1])
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
                        print(f"[ArrayView] Killed process {pid} on port {args.port}")
                        killed = True
                    except Exception as e:
                        print(f"[ArrayView] Failed to kill process: {e}")
            if not killed:
                print(f"[ArrayView] No process found listening on port {args.port}")
        else:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{args.port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
            )
            pids = [
                int(p) for p in result.stdout.strip().split() if p.strip().isdigit()
            ]
            if not pids:
                print(f"[ArrayView] No process found listening on port {args.port}")
            else:
                for pid in pids:
                    try:
                        os.kill(pid, _signal.SIGTERM)
                        print(f"[ArrayView] Killed process {pid} on port {args.port}")
                    except ProcessLookupError:
                        pass
                deadline = time.time() + 1.0
                while time.time() < deadline:
                    alive = []
                    for pid in pids:
                        try:
                            os.kill(pid, 0)
                            alive.append(pid)
                        except ProcessLookupError:
                            pass
                    if not alive:
                        break
                    time.sleep(0.05)
                else:
                    for pid in alive:
                        try:
                            os.kill(pid, _signal.SIGKILL)
                            print(f"[ArrayView] Force-killed process {pid} on port {args.port}")
                        except ProcessLookupError:
                            pass
                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        alive = []
                        for pid in pids:
                            try:
                                os.kill(pid, 0)
                                alive.append(pid)
                            except ProcessLookupError:
                                pass
                        if not alive:
                            break
                        time.sleep(0.05)
        return

    # -- --serve: start a persistent empty server and exit --
    if args.serve:
        if _server_alive(args.port):
            print(
                f"[ArrayView] Server already running on port {args.port}. "
                "Set port to Public in VS Code Ports tab if not done yet, "
                "then run: arrayview your_file.npy"
            )
            return
        if _port_in_use(args.port):
            if _is_vscode_remote():
                # In tunnel mode the port must be predictable so the user can
                # set the right port to Public.  Auto-scanning would silently
                # pick a different port, leaving the user's Ports tab stale.
                print(
                    f"[ArrayView] Port {args.port} is in use by another process.\n"
                    f"  Run 'arrayview --kill --port {args.port}' to free it, "
                    f"or use --port to specify a different port.",
                    flush=True,
                )
                sys.exit(1)
            # Non-tunnel: auto-scan for a free port.
            args.port, _ = _find_server_port(args.port + 1)
            if _port_in_use(args.port):
                print(
                    f"Error: port {args.port} is in use by another process. "
                    "Use --port to pick another."
                )
                sys.exit(1)
            print(
                f"[ArrayView] Default port busy, using port {args.port}",
                flush=True,
            )
        # Write VS Code port settings BEFORE starting the server so VS Code
        # sees privacy=public when it first detects the new port listening.
        _configure_vscode_port_preview(args.port)
        script = (
            f"from arrayview._launcher import _serve_empty; _serve_empty({args.port})"
        )
        proc = subprocess.Popen([sys.executable, "-c", script])
        if not _wait_for_port(args.port, timeout=15.0, tcp_only=True):
            print(f"Error: ArrayView server failed to start on port {args.port}.")
            sys.exit(1)
        print(
            f"\n  \033[1;36m\u2192 ArrayView server started on port {args.port} (PID {proc.pid})\033[0m\n"
            f"\n  Remote tunnel setup:\n"
            f"    1. VS Code Ports tab \u2192 port {args.port} \u2192 right-click \u2192 Port Visibility \u2192 Public\n"
            f"       (if VS Code did not set it automatically)\n"
            f"    2. Then run: arrayview your_file.npy\n"
            f"\n  Server stays running until you kill it (kill {proc.pid}).\n"
        )
        return

    base_file = os.path.abspath(args.files[0])
    compare_files = [os.path.abspath(p) for p in args.files[1:]]
    if args.compare:
        compare_files.append(os.path.abspath(args.compare))

    if not os.path.isfile(base_file):
        print(f"Error: file not found: {base_file}")
        sys.exit(1)

    name = getattr(args, "_demo_name", None) or os.path.basename(base_file)

    if args.vectorfield:
        try:
            from arrayview._io import load_data
            from arrayview._render import _detect_rgb_axis
            from arrayview._vectorfield import _resolve_vfield_layout

            base_data = load_data(base_file)
            image_shape = tuple(int(s) for s in base_data.shape)
            if args.rgb:
                rgb_axis = _detect_rgb_axis(image_shape)
                image_shape = tuple(
                    s for i, s in enumerate(image_shape) if i != rgb_axis
                )
            vf_data = load_data(args.vectorfield)
            layout = _resolve_vfield_layout(
                tuple(int(s) for s in vf_data.shape),
                image_shape,
                args.vectorfield_components_dim,
            )
            vfield_components_dim = int(layout["components_dim"])
        except Exception as e:
            print(f"Error: invalid vector field {args.vectorfield}: {e}")
            sys.exit(1)

    # Detect SSH early — needed by the reverse-tunnel relay check below.
    _is_ssh = bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"))

    is_arrayview_server = _server_alive(args.port)
    if _port_in_use(args.port) and not is_arrayview_server and _is_ssh:
        # Port occupied but not responding to a fast HTTP check.  When port 8000
        # is bound by a reverse SSH tunnel (ssh -R 8000:localhost:8000), the TCP
        # connection to 127.0.0.1:8000 succeeds immediately (SSH daemon's listener)
        # but the full HTTP round-trip to the tunnel-remote ArrayView server can
        # easily exceed the 0.5 s timeout, or fail entirely if `localhost` on the
        # remote resolves to ::1 while ArrayView only binds to 127.0.0.1.
        # Skip the health-check: just attempt the relay directly.  If the server
        # on the other side of the tunnel is a real ArrayView instance, the relay
        # succeeds.  If not, _relay_array_to_server raises and we fall through to
        # the normal auto-scan path.
        print(
            f"[ArrayView] SSH session — trying relay on port {args.port}...", flush=True
        )
        try:
            _relay_array_to_server(base_file, args.port, name, args.rgb)
            return
        except Exception as _relay_exc:
            print(f"[ArrayView] Relay attempt failed: {_relay_exc}", flush=True)
            # Fall through — not an ArrayView relay server; start our own.
    if _port_in_use(args.port) and not is_arrayview_server:
        if _is_vscode_remote():
            # In tunnel mode the port must be predictable so the user can set
            # the right port to Public.  Auto-scanning would silently pick a
            # different port, leaving the user's Ports tab stale.
            print(
                f"[ArrayView] Port {args.port} is in use by another process.\n"
                f"  Run 'arrayview --kill --port {args.port}' to free it, "
                f"or use --port to specify a different port.",
                flush=True,
            )
            sys.exit(1)
        # Non-tunnel: auto-scan for a free port.
        args.port, is_arrayview_server_new = _find_server_port(args.port + 1)
        is_arrayview_server = is_arrayview_server_new
        if _port_in_use(args.port) and not is_arrayview_server:
            print(
                f"Error: port {args.port} is in use by another process. "
                "Use --port to pick another."
            )
            sys.exit(1)
        print(
            f"[ArrayView] Default port busy, using port {args.port}",
            flush=True,
        )

    # Relay detection: if we're connected via SSH and the existing server on
    # this port is actually on a different machine (reverse SSH tunnel), send
    # the array bytes there instead of a filepath the remote server can't access.
    if is_arrayview_server and _is_ssh:
        import socket as _socket

        # Use a generous timeout: _server_hostname also goes through the SSH tunnel.
        _remote_host = _server_hostname(args.port, timeout=3.0)
        if _remote_host and _remote_host != _socket.gethostname():
            try:
                _relay_array_to_server(base_file, args.port, name, args.rgb)
            except Exception as e:
                print(f"[ArrayView] Relay failed: {e}", flush=True)
                sys.exit(1)
            return  # Resolve --window / --browser into a single window_mode
    if args.browser and not args.window:
        args.window = "browser"
    window_mode = args.window  # None = auto-detect (current behaviour)
    # User config: apply persistent window preference if no explicit --window flag
    if window_mode is None:
        from arrayview._config import get_window_default
        from arrayview._platform import detect_environment

        _cfg_mode = get_window_default(detect_environment())
        if _cfg_mode:
            window_mode = _cfg_mode
    # Auto-detect: prefer VS Code tab in VS Code terminal
    if window_mode is None and _in_vscode_terminal():
        window_mode = "vscode"
    # Explicit vscode requires VS Code terminal
    if window_mode == "vscode":
        if not _in_vscode_terminal():
            print(
                "[ArrayView] --window=vscode requires running from a VS Code integrated terminal.\n"
                "  Use --window=browser to open in your system browser instead.",
                flush=True,
            )
            sys.exit(1)
        # Warn if we can't find the IPC hook (multi-window targeting falls back to PID matching)
        from arrayview._platform import _find_vscode_ipc_hook as _check_ipc_hook

        if not _is_vscode_remote() and not _check_ipc_hook():
            # IPC hook not available — broadcast with focus guard handles this
            _vprint(
                "[ArrayView] No IPC hook; will broadcast to all VS Code windows",
                flush=True,
            )
    # Explicit native is not supported in remote/tunnel environments
    if window_mode == "native" and _is_vscode_remote():
        _vprint(
            "[ArrayView] --window native is not supported on remote tunnel; using vscode instead."
        )
        window_mode = "vscode"
    use_webview = (window_mode == "native") or (
        window_mode is None and _can_native_window()
    )

    if is_arrayview_server:
        # Server already running — register the new array.
        # If using webview, notify the existing shell to inject a new tab.
        try:
            # Register overlay(s) first (no notification) to get their sids
            overlay_sids_list: list[str] = []
            for ov_path in (args.overlay or []):
                ov_body = json.dumps(
                    {
                        "filepath": os.path.abspath(ov_path),
                        "name": os.path.basename(ov_path) or "overlay",
                        "notify": False,
                    }
                ).encode()
                ov_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/load",
                    data=ov_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(ov_req, timeout=300) as resp:
                    ov_result = json.loads(resp.read())
                if "error" in ov_result:
                    print(
                        f"Error from server while loading overlay: {ov_result['error']}"
                    )
                    sys.exit(1)
                ov_sid = ov_result.get("sid")
                if ov_sid:
                    overlay_sids_list.append(ov_sid)
            overlay_sid = ",".join(overlay_sids_list) if overlay_sids_list else None

            compare_sids = []
            for compare_file in compare_files:
                cmp_body = json.dumps(
                    {
                        "filepath": compare_file,
                        "name": os.path.basename(compare_file),
                        "notify": False,
                    }
                ).encode()
                cmp_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/load",
                    data=cmp_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(cmp_req, timeout=300) as resp:
                    cmp_result = json.loads(resp.read())
                if "error" in cmp_result:
                    print(
                        f"Error from server while loading compare array: {cmp_result['error']}"
                    )
                    sys.exit(1)
                compare_sid = cmp_result.get("sid")
                if compare_sid:
                    compare_sids.append(compare_sid)

            notify_webview = use_webview and overlay_sid is None
            body_dict = {
                "filepath": base_file,
                "name": name,
                "notify": notify_webview,
                "rgb": args.rgb,
            }
            if notify_webview and compare_sids:
                body_dict["compare_sid"] = compare_sids[0]
                body_dict["compare_sids"] = ",".join(compare_sids)
            body = json.dumps(body_dict).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/load",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            if "error" in result:
                print(f"Error from server: {result['error']}")
                sys.exit(1)

            # Attach vector field to the newly loaded session
            if args.vectorfield:
                vf_body = json.dumps(
                    {
                        "sid": result["sid"],
                        "filepath": os.path.abspath(args.vectorfield),
                        "components_dim": vfield_components_dim,
                    }
                ).encode()
                vf_req = urllib.request.Request(
                    f"http://127.0.0.1:{args.port}/attach_vectorfield",
                    data=vf_body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(vf_req, timeout=300) as resp:
                    vf_result = json.loads(resp.read())
                if "error" in vf_result:
                    print(f"Error: failed to attach vector field: {vf_result['error']}")
                    sys.exit(1)
        except Exception as e:
            print(
                f"Error: port {args.port} is in use by another process. "
                f"Use --port to pick another. ({e})"
            )
            sys.exit(1)

        sid = result["sid"]
        url = _viewer_url(
            args.port,
            sid,
            compare_sids=compare_sids,
            overlay_sids=overlay_sid,
            dims=dims_override,
        )
        if notify_webview and result.get("notified"):
            # Tab was injected into existing webview window (with or without compare)
            _vprint(f"Injected into existing window (port {args.port})")
        elif notify_webview and not result.get("notified"):
            # Native window was requested but the shell is gone — open a new native window.
            url_shell = _shell_url(args.port, sid, name, compare_sids=compare_sids)
            if not _open_webview_cli(url_shell, 1200, 800):
                _vprint("[ArrayView] Falling back to browser", flush=True)
                _print_viewer_location(url)
                _open_browser(url, blocking=True, title=f"ArrayView: {name}", filepath=base_file, floating=args.floating)
        else:
            if getattr(args, "watch", False):
                _start_watch_thread(base_file, sid, args.port)
            _open_browser(
                url,
                blocking=True,
                force_vscode=(window_mode == "vscode"),
                title=f"ArrayView: {name}",
                filepath=base_file,
                floating=args.floating,
            )
        return

    # Direct webview mode for VS Code tunnel sessions: skip starting the
    # WebSocket server entirely.  The extension spawns a Python subprocess
    # with --mode stdio and bridges via postMessage — no ports needed.
    is_remote = _is_vscode_remote()
    if is_remote:
        _ensure_vscode_extension()
        # Build generic extra CLI args so new flags work without touching the
        # VS Code extension — they're forwarded verbatim to the subprocess.
        extra_args: list[str] = []
        if args.vectorfield:
            extra_args += ["--vectorfield", os.path.abspath(args.vectorfield)]
            if vfield_components_dim is not None:
                extra_args += ["--vectorfield-components-dim", str(vfield_components_dim)]
        for ov_path in (args.overlay or []):
            extra_args += ["--overlay", os.path.abspath(ov_path)]
        if args.rgb:
            extra_args.append("--rgb")
        # Compare files: pass as additional positional args
        for cf in compare_files:
            extra_args.append(cf)
        _open_direct_via_signal_file(
            base_file,
            title=f"ArrayView: {name}",
            extra_args=extra_args or None,
            floating=args.floating,
        )
        return

    sid = uuid.uuid4().hex
    overlay_files = list(args.overlay or [])
    overlay_sids = [uuid.uuid4().hex for _ in overlay_files]
    overlay_sid = ",".join(overlay_sids) if overlay_sids else None

    # Configure VS Code port settings before starting the server.
    if not use_webview:
        _configure_vscode_port_preview(args.port)

    # Spawn background server.
    # On remote tunnel: persist=True so the server survives tab closes and the
    # port stays public across multiple arrayview invocations.
    # Locally: persist=False so the port is freed when the last tab closes.
    is_remote = _is_vscode_remote()
    vfield_abs = os.path.abspath(args.vectorfield) if args.vectorfield else None
    demo_name = getattr(args, "_demo_name", None)
    demo_cleanup = getattr(args, "_demo_cleanup", False)
    script = (
        f"from arrayview._launcher import _serve_daemon;"
        f"_serve_daemon("
        f"{repr(base_file)}, {args.port}, {repr(sid)},"
        f" name={repr(demo_name)},"
        f" cleanup={demo_cleanup},"
        f" overlay_filepaths={repr([os.path.abspath(p) for p in overlay_files])},"
        f" overlay_sids={repr(overlay_sids)},"
        f" vfield_filepath={repr(vfield_abs)},"
        f" vfield_components_dim={repr(vfield_components_dim)},"
        f" persist={is_remote},"
        f" rgb={args.rgb},"
        f")"
    )
    subprocess.Popen(
        [sys.executable, "-c", script],
    )

    if not _wait_for_port(args.port, timeout=15.0, tcp_only=True):
        print(
            f"Error: ArrayView server failed to start on port {args.port}. "
            "Use --port to pick another."
        )
        sys.exit(1)

    compare_sids = []
    for compare_file in compare_files:
        try:
            cmp_body = json.dumps(
                {
                    "filepath": compare_file,
                    "name": os.path.basename(compare_file),
                    "notify": False,
                }
            ).encode()
            cmp_req = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/load",
                data=cmp_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(cmp_req, timeout=300) as resp:
                cmp_result = json.loads(resp.read())
            if "error" in cmp_result:
                print(
                    f"Error from server while loading compare array: {cmp_result['error']}"
                )
                sys.exit(1)
            compare_sid = cmp_result.get("sid")
            if compare_sid:
                compare_sids.append(compare_sid)
        except Exception as e:
            print(f"Error while loading compare array {compare_file}: {e}")
            sys.exit(1)

    url = _viewer_url(
        args.port,
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sid,
        dims=dims_override,
    )

    if use_webview and overlay_sid is None:
        url_shell = _shell_url(args.port, sid, name, compare_sids=compare_sids)
        if not _open_webview_cli(url_shell, 1400, 900):
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _print_viewer_location(url)
            _open_browser(
                url,
                blocking=False,
                force_vscode=(window_mode == "vscode"),
                title=f"ArrayView: {name}",
                floating=args.floating,
            )
    else:
        if use_webview and overlay_sid:
            _vprint(
                "[ArrayView] Overlay mode: opening browser (webview injection not supported with overlay)",
                flush=True,
            )
        _print_viewer_location(url)
        if is_remote and sys.stdin.isatty():
            # New server, tunnel mode: wait for user to set port Public before
            # writing the signal file so the viewer tab opens on first try.
            print(
                f"\n  VS Code Ports tab: right-click port {args.port} "
                f"\u2192 Port Visibility \u2192 Public\n"
                f"  Press Enter once done (or the viewer retries automatically)... ",
                end="",
                flush=True,
            )
            try:
                input()
            except KeyboardInterrupt:
                print(flush=True)
                sys.exit(0)
            except EOFError:
                # stdin reached EOF (e.g. uvx / piped stdin) — don't abort,
                # proceed to open browser immediately without waiting.
                print(flush=True)
            # Suppress duplicate "set to Public" reminder inside _open_browser.
            import arrayview._vscode as _vscode_mod

            _vscode_mod._remote_message_shown = True
        if getattr(args, "watch", False):
            _start_watch_thread(base_file, sid, args.port)
        _open_browser(
            url,
            blocking=True,
            force_vscode=(window_mode == "vscode"),
            title=f"ArrayView: {name}",
            floating=args.floating,
        )
