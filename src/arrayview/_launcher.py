"""Entry points, process management, view(), arrayview() CLI.

This module was extracted from _app.py during the modular refactor.
"""

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


# ---------------------------------------------------------------------------
# Subprocess GUI Launcher
# ---------------------------------------------------------------------------
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
    url: str, win_w: int, win_h: int, capture_stderr: bool = False
) -> subprocess.Popen:
    """Launch pywebview in a fresh subprocess. Uses subprocess.Popen to avoid
    multiprocessing bootstrap errors when called from a Jupyter kernel."""
    icon_path = _get_icon_png_path() or ""
    script = "\n".join(
        [
            "import sys, webview",
            "u, w, h, icon = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]",
            "webview.create_window('ArrayView', u, width=w, height=h, background_color='#111111')",
            "kw = {'gui': 'qt'} if sys.platform.startswith('linux') else {}",
            "if icon:",
            "    if sys.platform == 'darwin':",
            "        def _set_icon():",
            "            try:",
            "                import AppKit",
            "                from PyObjCTools import AppHelper",
            "                img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon)",
            "                AppHelper.callAfter(AppKit.NSApplication.sharedApplication().setApplicationIconImage_, img)",
            "            except Exception: pass",
            "        kw['func'] = _set_icon",
            "    else:",
            "        kw['icon'] = icon",
            "webview.start(**kw)",
        ]
    )
    return subprocess.Popen(
        [sys.executable, "-c", script, url, str(win_w), str(win_h), icon_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
    )


def _open_webview_with_fallback(url: str, win_w: int, win_h: int) -> subprocess.Popen:
    """Launch pywebview, falling back to _open_browser if the subprocess exits immediately
    OR if no viewer WebSocket connects within ~10 s (catches macOS non-framework Python
    zombies that start but show nothing).

    Used from view() (Python API) where the host process stays alive.
    """
    proc = _open_webview(url, win_w, win_h, capture_stderr=True)
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
                _open_browser(url)
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
                _open_browser(url)
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
        _open_browser(url)

    threading.Thread(target=_watchdog, daemon=True).start()
    return proc


def _open_webview_cli(url: str, win_w: int, win_h: int) -> bool:
    """Launch pywebview from the CLI and synchronously wait to detect an immediate crash.

    Returns True if the window appears to have started (still alive after 2 s).
    Returns False if it crashed; in that case the caller should fall back to browser.
    The CLI process must not exit while the daemon-thread watchdog is still pending,
    so the wait is done synchronously here.
    """
    _vprint("[ArrayView] Launching native window (PyWebView)...", flush=True)
    proc = _open_webview(url, win_w, win_h, capture_stderr=True)
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


# ---------------------------------------------------------------------------
# Server port utilities
# ---------------------------------------------------------------------------


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
    session and writes its own VS Code signal file so Simple Browser opens there.

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
        "VS Code Simple Browser should open automatically.",
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


async def _serve_background(port: int, stop_when_closed: bool = False):
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


_OVERLAY_PALETTE = ["ff4444", "44cc44", "4488ff", "ffcc00", "ff44ff", "44ffff"]


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
    data,
    name: str = None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 500,
    window: str | bool | None = None,
    rgb: bool = False,
    overlay=None,
):
    """
    Launch the viewer. Does not block the main Python process.

    ``window`` controls how the viewer opens:
      - ``None``       auto: native window outside Jupyter, inline IFrame inside Jupyter
      - ``True``       native window (falls back to browser if unavailable)
      - ``False``      no automatic opening; returns URL
      - ``'native'``   open in a native desktop window
      - ``'browser'``  open in the system browser
      - ``'vscode'``   open in VS Code Simple Browser
      - ``'inline'``   return an inline IFrame (Jupyter / VS Code notebook)

    ``rgb`` — treat the array as RGB/RGBA. The first or last dimension must
    have size 3 (RGB) or 4 (RGBA). When True, the colorbar is hidden and each
    slice is composited directly from the colour channels.

    ``overlay`` — a single array or list of arrays to composite as overlays.
    Each overlay is assigned an auto-palette color from _OVERLAY_PALETTE.
    """
    import numpy as np
    from arrayview._io import _tensor_to_numpy

    # Normalise string window modes.
    _force_browser = False
    _force_vscode = False
    if isinstance(window, str):
        _w = window.lower()
        if _w == "inline":
            inline = True
            window = False
        elif _w == "native":
            window = True
        elif _w == "browser":
            window = False
            _force_browser = True
        elif _w == "vscode":
            window = False
            _force_vscode = True
        else:
            raise ValueError(
                f"window must be 'inline', 'native', 'browser', or 'vscode', got {window!r}"
            )

    # Duck-typing: accept PyTorch tensors, JAX arrays, Julia/PythonCall arrays, etc.
    # Zarr and nibabel proxy arrays are already numpy-like and work without conversion.
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
                # Generic fallback: copy to numpy for any array-like
                # (catches PythonCall arrays whose module name varies across versions)
                # Use np.array (copy) so Julia's GC can safely reclaim the source after view() returns.
                try:
                    converted = np.array(data)
                    if converted.dtype != object:
                        data = converted
                except Exception:
                    pass  # keep original; let Session handle it lazily

    if name is None:
        name = f"Array {data.shape}"

    # Remote/tunnel: if the caller didn't override the port and a --serve server
    # is already running on the CLI default (8000), use that instead of 8123.
    # This ensures av.view(x) in Python/Jupyter/Julia on a tunnel connects to
    # the persistent server whose port the user has already set to Public.
    _CLI_DEFAULT_PORT = 8000
    if port == 8123 and _is_vscode_remote() and _server_alive(_CLI_DEFAULT_PORT):
        port = _CLI_DEFAULT_PORT

    # Julia/PythonCall: must be handled before the is_jupyter defaults because:
    # 1. _in_jupyter() returns False for IJulia kernels (not ipykernel)
    # 2. that would set inline=False and window=True, overriding user intent
    # The subprocess server is always required here due to GIL.
    if _is_julia_env():
        if window is True:
            _inline, _window = False, True
        elif inline is True or window is False:
            # Explicit "not window" → inline (makes sense in Jupyter; harmless elsewhere)
            _inline, _window = True, False
        elif inline is False:
            _inline, _window = False, False  # explicit inline=False → browser
        else:
            # No args: inline in IJulia, window elsewhere.
            _inline = _in_julia_jupyter()
            _window = False
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window=_window,
            inline=_inline,
            height=height,
        )

    is_jupyter = _in_jupyter()
    if inline is None:
        inline = is_jupyter
    if window is None:
        window = not is_jupyter
    if window:
        inline = False

    # Auto-detect VS Code terminal: prefer Simple Browser over native window
    if _in_vscode_terminal() and not _force_vscode and not _force_browser:
        _force_vscode = True
        if window is True:  # Convert bool to False so we hit the browser path below
            window = False

    # Julia/PythonCall: the GIL is not released reliably between Julia statements,
    # so an in-process uvicorn thread cannot serve requests once view() returns.
    # Use a fully independent subprocess server instead (same approach as the CLI).
    if not inline and _is_julia_env():
        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window,
        )

    # VS Code tunnel/remote with an existing --serve server: register the array
    # via /load and open the viewer through the signal-file mechanism.  This
    # avoids starting a new server on a different port (which wouldn't be Public).
    if _is_vscode_remote() and _server_alive(port):
        try:
            import tempfile as _tf

            with _tf.NamedTemporaryFile(suffix=".npy", delete=False) as _tmp:
                _tmp_path = _tmp.name
            np.save(_tmp_path, data)
            body = json.dumps(
                {"filepath": _tmp_path, "name": name, "rgb": rgb}
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
            url_viewer = f"http://localhost:{port}/?sid={sid}"
            # On a tunnel, inline IFrames don't work (localhost URL in notebook
            # webview can't route through VS Code port forwarding).  Always
            # open via Simple Browser signal file.
            _open_browser(url_viewer, force_vscode=True, blocking=True)
            _print_viewer_location(url_viewer)
            return ViewHandle(url_viewer, sid, port)
        except Exception as e:
            _vprint(
                f"[ArrayView] Failed to register with --serve server: {e}", flush=True
            )
            # Fall through to subprocess/in-process paths.

    # VS Code tunnel/remote: the calling Python process may exit shortly after
    # view() returns (one-shot scripts, non-interactive use).  A daemon-thread
    # server would die with the process, leaving the user staring at
    # "session expired" in the browser.  Use a subprocess server that survives
    # the parent's exit.  (In Jupyter inline mode we stay in-process since the
    # notebook kernel is long-lived.)
    if not inline and _is_vscode_remote() and not _in_jupyter():
        return _view_subprocess(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window,
            inline,
            height,
            rgb=rgb,
            force_vscode=True,
        )

    session = _session_mod.Session(data, name=name)
    if rgb:
        from arrayview._render import _setup_rgb

        _setup_rgb(session)
    _session_mod.SESSIONS[session.sid] = session

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
        _platform_mod._jupyter_server_port = port  # server already ours on this port

    # SERVER_LOOP is set as the first statement of _serve_background, before
    # _server_ready_event fires, so it is guaranteed non-None by this point.

    url_viewer = f"http://localhost:{port}/?sid={session.sid}"
    if _overlay_sids:
        url_viewer += f"&overlay_sid={','.join(_overlay_sids)}"
        url_viewer += f"&overlay_colors={','.join(_overlay_colors)}"
    encoded_name = urllib.parse.quote(name)
    url_shell = (
        f"http://localhost:{port}/shell?init_sid={session.sid}&init_name={encoded_name}"
    )

    if inline:
        from IPython.display import IFrame

        return IFrame(src=url_viewer, width="100%", height=height)

    can_native_window = _can_native_window() if window else False
    if window and can_native_window and not _force_browser and not _force_vscode:
        try:
            wp = _session_mod._window_process
            if wp is not None and wp.poll() is None:
                # Webview already open — inject new tab
                asyncio.run_coroutine_threadsafe(
                    _server_mod()._notify_shells(session.sid, name),
                    _session_mod.SERVER_LOOP,
                )
            else:
                _session_mod._window_process = _open_webview_with_fallback(
                    url_shell, win_w, win_h
                )
        except Exception:
            _open_browser(url_viewer, force_vscode=_force_vscode)
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
        _open_browser(url_viewer, force_vscode=_force_vscode)

    _print_viewer_location(url_viewer)
    return ViewHandle(url_viewer, session.sid, port)


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
):
    """Julia-specific view() path: run the server in a subprocess so it is
    completely independent of Julia's GIL.
    """

    # Detect VS Code *now*, in the parent process where TERM_PROGRAM and
    # VSCODE_IPC_HOOK_CLI are still available.  The subprocess inherits a
    # stripped environment (Julia/PythonCall, uv run, etc.) so detection
    # there would fail and Simple Browser would never open.
    force_vscode = _in_vscode_terminal()
    return _view_subprocess(
        data,
        name,
        port,
        window,
        inline=inline,
        height=height,
        force_vscode=force_vscode,
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

    if _server_alive(port):
        # Existing subprocess server — register the new array via /load.
        try:
            body = json.dumps({"filepath": tmp_path, "name": name, "rgb": rgb}).encode()
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

    url_viewer = f"http://localhost:{port}/?sid={sid}"
    encoded_name = urllib.parse.quote(name)
    url_shell = f"http://localhost:{port}/shell?init_sid={sid}&init_name={encoded_name}"
    _print_viewer_location(url_viewer)

    if inline:
        iframe_html = (
            f"<iframe src='{url_viewer}' width='100%'"
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

    can_native = _can_native_window()
    if window and can_native:
        if not _open_webview_cli(url_shell, 1400, 900):
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _open_browser(url_viewer, force_vscode=force_vscode, blocking=force_vscode)
    else:
        # blocking=True when force_vscode so signal file is written before
        # returning to Julia (daemon thread would be killed on process exit).
        _open_browser(url_viewer, force_vscode=force_vscode, blocking=force_vscode)
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
    overlay_filepath: str = None,
    overlay_sid: str = None,
    compare_filepath: str = None,
    compare_sid: str = None,
    vfield_filepath: str = None,
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

    def _load():
        from arrayview._io import load_data

        try:
            data = load_data(filepath)
            if cleanup:
                try:
                    os.unlink(filepath)
                except Exception:
                    pass
            session = _session_mod.Session(
                data, filepath=None if cleanup else filepath, name=name
            )
            session.sid = sid
            if rgb:
                from arrayview._render import _setup_rgb

                _setup_rgb(session)
            if vfield_filepath:
                try:
                    vf_data = load_data(vfield_filepath)
                    session.vfield = vf_data
                    _vprint(
                        f"[ArrayView] Loaded vector field {vfield_filepath} shape {vf_data.shape}",
                        flush=True,
                    )
                except Exception as e:
                    _vprint(
                        f"[ArrayView] Warning: failed to load vector field {vfield_filepath}: {e}",
                        flush=True,
                    )
            _session_mod.SESSIONS[session.sid] = session
            if overlay_filepath and overlay_sid:
                try:
                    ov_data = load_data(overlay_filepath)
                    ov_session = _session_mod.Session(
                        ov_data, filepath=overlay_filepath, name="overlay"
                    )
                    ov_session.sid = overlay_sid
                    _session_mod.SESSIONS[overlay_sid] = ov_session
                except Exception as e:
                    _vprint(
                        f"[ArrayView] Warning: failed to load overlay {overlay_filepath}: {e}",
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


def arrayview():
    """Command Line Interface Entry Point."""
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
        help="How to open the viewer: browser (system browser), vscode (SimpleBrowser), or native window",
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
        help="Segmentation mask to overlay (binary 0/1 array, same spatial shape)",
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
            "Must have the same spatial shape as the image plus a trailing dimension of size 3 "
            "(displacements along each spatial axis)."
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
            "The remote server registers the session and opens Simple Browser automatically."
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
    args = parser.parse_args()
    _session_mod._verbose = args.verbose

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

    # Detect SSH early — needed by the relay auto-detect retry and the relay check below.
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
    # Auto-detect: prefer VS Code Simple Browser in VS Code terminal
    if window_mode is None and _in_vscode_terminal():
        window_mode = "vscode"
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
            # Register overlay first (no notification) to get overlay_sid
            overlay_sid = None
            if args.overlay:
                ov_body = json.dumps(
                    {
                        "filepath": os.path.abspath(args.overlay),
                        "name": "overlay",
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
                overlay_sid = ov_result.get("sid")

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
                    print(
                        f"Warning: failed to attach vector field: {vf_result['error']}"
                    )
        except Exception as e:
            print(
                f"Error: port {args.port} is in use by another process. "
                f"Use --port to pick another. ({e})"
            )
            sys.exit(1)

        sid = result["sid"]
        encoded_name_inject = urllib.parse.quote(name)
        qs = f"?sid={sid}"
        if overlay_sid:
            qs += f"&overlay_sid={overlay_sid}"
        if compare_sids:
            qs += f"&compare_sid={compare_sids[0]}"
            qs += f"&compare_sids={','.join(compare_sids)}"
        if dims_override:
            qs += f"&dim_x={dims_override[0]}&dim_y={dims_override[1]}"
        if notify_webview and result.get("notified"):
            # Tab was injected into existing webview window (with or without compare)
            _vprint(f"Injected into existing window (port {args.port})")
        elif notify_webview and not result.get("notified"):
            # Native window was requested but the shell is gone — open a new native window.
            init_qs = f"init_sid={sid}&init_name={encoded_name_inject}"
            if compare_sids:
                init_qs += (
                    f"&init_compare_sid={compare_sids[0]}"
                    f"&init_compare_sids={','.join(compare_sids)}"
                )
            url_shell = f"http://localhost:{args.port}/shell?{init_qs}"
            if not _open_webview_cli(url_shell, 1200, 800):
                _vprint("[ArrayView] Falling back to browser", flush=True)
                url = f"http://localhost:{args.port}/{qs}"
                _print_viewer_location(url)
                _open_browser(url, blocking=True)
        else:
            url = f"http://localhost:{args.port}/{qs}"
            if getattr(args, "watch", False):
                _start_watch_thread(base_file, sid, args.port)
            _open_browser(url, blocking=True, force_vscode=(window_mode == "vscode"))
        return

    sid = uuid.uuid4().hex
    overlay_sid = uuid.uuid4().hex if args.overlay else None
    encoded_name = urllib.parse.quote(name)

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
        f" overlay_filepath={repr(os.path.abspath(args.overlay) if args.overlay else None)},"
        f" overlay_sid={repr(overlay_sid)},"
        f" vfield_filepath={repr(vfield_abs)},"
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

    qs = f"?sid={sid}"
    if overlay_sid:
        qs += f"&overlay_sid={overlay_sid}"
    if compare_sids:
        qs += f"&compare_sid={compare_sids[0]}"
        qs += f"&compare_sids={','.join(compare_sids)}"
    if dims_override:
        qs += f"&dim_x={dims_override[0]}&dim_y={dims_override[1]}"

    if use_webview and overlay_sid is None:
        init_qs = f"init_sid={sid}&init_name={encoded_name}"
        if compare_sids:
            init_qs += (
                f"&init_compare_sid={compare_sids[0]}"
                f"&init_compare_sids={','.join(compare_sids)}"
            )
        url_shell = f"http://localhost:{args.port}/shell?{init_qs}"
        if not _open_webview_cli(url_shell, 1400, 900):
            _vprint("[ArrayView] Falling back to browser", flush=True)
            url = f"http://localhost:{args.port}/{qs}"
            _print_viewer_location(url)
            _open_browser(url, blocking=False, force_vscode=(window_mode == "vscode"))
    else:
        if use_webview and overlay_sid:
            _vprint(
                "[ArrayView] Overlay mode: opening browser (webview injection not supported with overlay)",
                flush=True,
            )
        url = f"http://localhost:{args.port}/{qs}"
        _print_viewer_location(url)
        if is_remote and sys.stdin.isatty():
            # New server, tunnel mode: wait for user to set port Public before
            # writing the signal file so Simple Browser opens on first try.
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
        _open_browser(url, blocking=True, force_vscode=(window_mode == "vscode"))
