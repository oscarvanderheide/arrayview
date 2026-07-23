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
from collections.abc import Mapping
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
    _native_window_gui,
    _find_vscode_ipc_hook,
    _is_julia_env,
    _in_julia_jupyter,
)
import arrayview._platform as _platform_mod  # for mutable globals

_vscode_mod_cache = None


def _trace_launch_event(event: str, **attrs: object) -> None:
    """Emit an opt-in launch event without loading tracing on normal paths."""
    if not os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
        return
    from arrayview._launch_trace import emit_launch_event

    emit_launch_event(event, **attrs)


def _launch_trace_tag(value: object) -> str | None:
    if not os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
        return None
    from arrayview._launch_trace import trace_tag

    return trace_tag(value)


def _vscode_mod():
    global _vscode_mod_cache
    if _vscode_mod_cache is None:
        import arrayview._vscode as _vs  # noqa: PLC0415 — intentional lazy import

        _vscode_mod_cache = _vs
    return _vscode_mod_cache


def _configure_vscode_port_preview(*args, **kwargs):
    return _vscode_mod()._configure_vscode_port_preview(*args, **kwargs)


def _ensure_vscode_extension(*args, **kwargs):
    return _vscode_mod()._ensure_vscode_extension(*args, **kwargs)


def _print_viewer_location(*args, **kwargs):
    return _vscode_mod()._print_viewer_location(*args, **kwargs)


def _open_browser(*args, **kwargs):
    result = _vscode_mod()._open_browser(*args, **kwargs)
    if kwargs.get("blocking") and not result:
        detail = getattr(result, "detail", None) or getattr(
            getattr(result, "state", None), "value", "display handoff failed"
        )
        raise RuntimeError(f"ArrayView display handoff failed: {detail}")
    return result

# _server.py (FastAPI) is imported lazily via _server_mod() to keep the
# import-time cost of ``import arrayview`` low (~175 ms saved).
_server_mod_cache = None


def _server_mod():
    global _server_mod_cache
    if _server_mod_cache is None:
        import arrayview._server as _srv  # noqa: PLC0415 — intentional lazy import

        _server_mod_cache = _srv
    return _server_mod_cache


def _register_server_runtime(port: int, owner_mode: str):
    """Register this process identity after its listener has been reserved."""
    from arrayview import __version__
    from arrayview._instance_registry import InstanceRecord, InstanceRegistry

    registry = InstanceRegistry()
    registry._prepare()
    log_directory = registry.directory / "logs"
    log_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(log_directory, 0o700)
    except OSError:
        pass
    record = InstanceRecord.create(
        port=port,
        protocol_version=_server_mod().SERVER_PROTOCOL_VERSION,
        package_version=__version__,
        owner_mode=owner_mode,
        log_path=str(log_directory / f"server-{os.getpid()}.log"),
    )
    registry.write(record)
    _server_mod().configure_server_runtime(
        instance_id=record.instance_id,
        process_start=record.process_start,
        owner_mode=record.owner_mode,
        started_at=record.started_at,
        port=record.port,
    )
    return registry, record


def _stop_verified_server(port: int) -> tuple[str, int | None]:
    """Stop the verified ArrayView process on *port*, never a port occupant."""
    from arrayview._instance_registry import (  # noqa: PLC0415
        InstanceRegistry,
        process_start_identity,
    )

    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/ping", timeout=1.0
        ) as response:
            status = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "No verified ArrayView server found", None

    if status.get("service") != "arrayview" or status.get("port") != port:
        return "Refusing to stop an unverified listener", None
    try:
        pid = int(status["pid"])
        claimed_start = str(status["process_start"])
        instance_id = str(status["instance_id"])
    except (KeyError, TypeError, ValueError):
        return "Refusing to stop an unverified listener", None
    if not claimed_start or not instance_id or process_start_identity(pid) != claimed_start:
        return "Refusing to stop an unverified or stale process", None

    registry = InstanceRegistry()
    matching_record = next(
        (
            record
            for record in registry.discover(clean_stale=True)
            if record.instance_id == instance_id
            and record.port == port
            and record.pid == pid
            and record.process_start == claimed_start
        ),
        None,
    )
    if matching_record is None:
        return "Refusing to stop an unowned ArrayView process", None

    import signal as _signal

    _trace_launch_event(
        "server.stop_requested",
        instance_tag=_launch_trace_tag(instance_id),
        target_pid=pid,
        port=port,
    )
    try:
        os.kill(pid, _signal.SIGTERM)
    except ProcessLookupError:
        pass
    except (OSError, PermissionError) as exc:
        return f"Failed to stop ArrayView process {pid}: {exc}", None

    deadline = time.monotonic() + 1.0
    while process_start_identity(pid) == claimed_start and time.monotonic() < deadline:
        time.sleep(0.05)
    if process_start_identity(pid) == claimed_start:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                os.kill(pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        except (OSError, subprocess.SubprocessError) as exc:
            return f"Failed to stop ArrayView process {pid}: {exc}", None
        deadline = time.monotonic() + 2.0
        while (
            process_start_identity(pid) == claimed_start
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if process_start_identity(pid) == claimed_start:
            return f"Failed to stop ArrayView process {pid}: still running", None

    _trace_launch_event(
        "server.stopped",
        instance_tag=_launch_trace_tag(instance_id),
        target_pid=pid,
        port=port,
    )
    registry.remove(matching_record.instance_id)
    _trace_launch_event(
        "server.unregistered",
        instance_tag=_launch_trace_tag(instance_id),
        port=port,
    )
    return f"Killed process {pid} on port {port}", pid


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
_LOOPBACK_HOST = "localhost"


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


def _build_inline_shell_html(url: str, shell_port: int) -> str | None:
    """Return embedded shell HTML for a cold-start native window."""
    try:
        shell_html = _pkg_files("arrayview").joinpath("_shell.html").read_text(
            encoding="utf-8"
        )
        parsed = urllib.parse.urlparse(url)
        inline_query = parsed.query
        shell_html = shell_html.replace(
            "</head>",
            f"<script>"
            f"window.__av_inline=true;"
            f"window.__av_inlineQuery={inline_query!r};"
            f"</script>\n"
            f'<base href="http://{_LOOPBACK_HOST}:{shell_port}/">\n'
            f"</head>",
            1,
        )
        # Fix WebSocket URL — location.host is "" in inline html= mode
        shell_html = shell_html.replace(
            "`${proto}//${location.host}/ws/shell${shellWsQuery}`",
            f"`ws://{_LOOPBACK_HOST}:{shell_port}/ws/shell${{shellWsQuery}}`",
        )
        return shell_html
    except Exception:
        return None


def _make_loopback_socket(port: int) -> "socket.socket":
    """Bind a single TCP listener on the same loopback host the viewer URLs use."""
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        _LOOPBACK_HOST,
        port,
        type=socket.SOCK_STREAM,
    ):
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.listen(128)
            sock.set_inheritable(True)
            return sock
        except OSError:
            try:
                sock.close()
            except Exception:
                pass
    raise OSError(f"Could not bind {_LOOPBACK_HOST}:{port}")


def _make_loopback_sockets(port: int) -> list["socket.socket"]:
    """Bind loopback listeners on every loopback address that resolves for
    ``_LOOPBACK_HOST`` (typically both ``::1`` and ``127.0.0.1``).

    VS Code's port auto-forward detection looks for IPv4 ``127.0.0.1`` /
    ``0.0.0.0`` listening sockets; an IPv6-only ``[::1]`` listener (the first
    result of ``getaddrinfo("localhost")`` on Linux) is not reliably detected,
    so the tunnel is never forwarded and ``asExternalUri`` returns the
    ``localhost`` URL unchanged — producing a blank viewer tab on the remote
    client.  Binding both families gives VS Code an IPv4 socket to scan while
    preserving the macOS native-shell path that connects via ``localhost`` →
    ``::1``.
    """
    socks: list["socket.socket"] = []
    seen_families: set[int] = set()
    for family, socktype, proto, _, sockaddr in socket.getaddrinfo(
        _LOOPBACK_HOST,
        port,
        type=socket.SOCK_STREAM,
    ):
        if family in seen_families:
            continue
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # On Linux, binding ::1 first then 127.0.0.1 on the same port is
            # fine (different addresses). IPV6_V6ONLY stays default (off on
            # Linux) which is harmless since we bind explicit addresses.
            sock.bind(sockaddr)
            sock.listen(128)
            sock.set_inheritable(True)
            socks.append(sock)
            seen_families.add(family)
        except OSError:
            try:
                sock.close()
            except Exception:
                pass
    if not socks:
        raise OSError(f"Could not bind {_LOOPBACK_HOST}:{port}")
    return socks


def _open_webview(
    url: str,
    win_w: int,
    win_h: int,
    capture_stderr: bool = False,
    shell_port: int | None = None,
    ready_file: str | None = None,
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
    gui_name = _native_window_gui() or ""

    if shell_port is not None:
        shell_html = _build_inline_shell_html(url, shell_port)
        if shell_html is not None:
            inline_html_b64 = _b64.b64encode(shell_html.encode()).decode()

    if inline_html_b64:
        script_lines = [
            "import sys, base64, webview",
            "u, w, h, icon, html_b64, ready_file, gui = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]",
            "html = base64.b64decode(html_b64.encode()).decode()",
            "class Api:",
            "    def set_title(self, title):",
            "        try: webview.windows[0].set_title(str(title)[:240])",
            "        except Exception: pass",
            "win = webview.create_window('ArrayView', html=html, width=w, height=h, background_color='#0c0c0c', js_api=Api())",
            "kw = {'gui': gui} if gui else {}",
            "def _start_func():",
            "    if ready_file:",
            "        try:",
            "            with open(ready_file, 'w') as f: f.write('ready')",
            "        except Exception: pass",
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
                inline_html_b64,
                ready_file or "",
                gui_name,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
            close_fds=True,
            start_new_session=sys.platform != "win32",
        )

    # URL mode — direct load (used when shell_port not provided)
    script_lines = [
        "import sys, webview",
        "u, w, h, icon, ready_file, gui = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]",
        "class Api:",
        "    def set_title(self, title):",
        "        try: webview.windows[0].set_title(str(title)[:240])",
        "        except Exception: pass",
        "win = webview.create_window('ArrayView', u, width=w, height=h, background_color='#0c0c0c', js_api=Api())",
        "kw = {'gui': gui} if gui else {}",
        "def _start_func():",
        "    if ready_file:",
        "        try:",
        "            with open(ready_file, 'w') as f: f.write('ready')",
        "        except Exception: pass",
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
            ready_file or "",
            gui_name,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else subprocess.DEVNULL,
        close_fds=True,
        start_new_session=sys.platform != "win32",
    )


def _open_webview_with_fallback(
    url: str,
    win_w: int,
    win_h: int,
    shell_port: int | None = None,
    floating: bool = False,
    launch_context=None,
    fallback_url: str | None = None,
    title: str | None = None,
) -> subprocess.Popen | None:
    """Resolve Python's native handoff or a confirmed browser fallback."""
    fallback_url = fallback_url or url
    native_request_id = uuid.uuid4().hex
    separator = "&" if "?" in url else "?"
    native_url = (
        f"{url}{separator}native_request_id="
        f"{urllib.parse.quote(native_request_id)}"
    )
    query = urllib.parse.parse_qs(urllib.parse.urlparse(native_url).query)
    sid = (query.get("init_sid") or query.get("sid") or [""])[0]
    expected_server_id = None
    expected_server_pid = os.getpid()
    if launch_context is not None and launch_context.plan.registration.value == "http_load":
        expected_server_id = _planned_server_snapshot(
            launch_context,
            shell_port or launch_context.plan.effective_port,
        ).server_instance_id
        expected_server_pid = None

    _trace_launch_event(
        "display.attempt_started",
        adapter="native",
        stage="python",
    )
    opened = False
    proc = None
    native_failure_evidence = "process_error"
    try:
        opened, proc = _open_webview_cli_tracked(
            native_url,
            win_w,
            win_h,
            shell_port=shell_port,
            trace_stage="python",
        )
        native_failure_evidence = "frame_timeout" if opened else "process_exit"
    except Exception as exc:
        _vprint(f"[ArrayView] Native window could not start: {exc}", flush=True)
        _trace_launch_event(
            "display.process_evidence",
            adapter="native",
            stage="python",
            state="failed",
            evidence="process_error",
            error_type=type(exc).__name__,
        )
    native_ready = bool(
        opened
        and proc is not None
        and sid
        and shell_port is not None
        and _wait_for_native_ready(
            shell_port,
            sid=sid,
            native_request_id=native_request_id,
            expected_server_id=expected_server_id,
            expected_server_pid=expected_server_pid,
        )
    )
    if native_ready:
        _trace_launch_event(
            "display.attempt_finished",
            adapter="native",
            stage="python",
            state="ready",
            evidence="native_frame_ack",
        )
        _vprint("[ArrayView] Native window connected successfully", flush=True)
        if sys.platform == "darwin":
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'tell application "System Events" to set frontmost of'
                    f" (first process whose unix id is {proc.pid}) to true",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        return proc

    _trace_launch_event(
        "display.attempt_finished",
        adapter="native",
        stage="python",
        state="failed",
        evidence=native_failure_evidence,
    )
    _terminate_native_process(proc)
    _vprint("[ArrayView] Native window failed; falling back to browser", flush=True)
    _open_browser(
        fallback_url,
        blocking=True,
        floating=floating,
        title=title,
        launch_context=launch_context,
        use_fallback=True,
    )
    return None


def _open_webview_cli(
    url: str,
    win_w: int,
    win_h: int,
    shell_port: int | None = None,
) -> bool:
    """Launch pywebview from the CLI and synchronously wait to detect an immediate crash.

    Returns True once the child has imported pywebview and created the window.
    Returns False if it crashed; in that case the caller should fall back to browser.
    """
    opened, _proc = _open_webview_cli_tracked(url, win_w, win_h, shell_port=shell_port)
    return opened


def _open_webview_cli_tracked(
    url: str,
    win_w: int,
    win_h: int,
    shell_port: int | None = None,
    trace_stage: str | None = None,
) -> tuple[bool, subprocess.Popen | None]:
    """Launch pywebview and return the child process when startup succeeds."""
    import tempfile

    _vprint("[ArrayView] Launching native window (PyWebView)...", flush=True)
    fd, ready_file = tempfile.mkstemp(prefix="arrayview-webview-ready-", suffix=".flag")
    os.close(fd)
    try:
        os.unlink(ready_file)
    except OSError:
        pass
    proc = _open_webview(
        url,
        win_w,
        win_h,
        capture_stderr=True,
        shell_port=shell_port,
        ready_file=ready_file,
    )
    _trace_launch_event(
        "display.process_started",
        adapter="native",
        stage=trace_stage,
        child_pid=getattr(proc, "pid", None),
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_out = ""
            try:
                stderr_out = proc.stderr.read().decode(errors="replace").strip()
            except Exception:
                pass
            try:
                os.unlink(ready_file)
            except OSError:
                pass
            _vprint(
                f"[ArrayView] Native window exited immediately (code {proc.returncode})",
                flush=True,
            )
            if stderr_out:
                _vprint(f"[ArrayView] webview stderr: {stderr_out}", flush=True)
            _trace_launch_event(
                "display.process_evidence",
                adapter="native",
                stage=trace_stage,
                state="failed",
                evidence="process_exit",
                returncode=proc.returncode,
            )
            return False, None
        if os.path.exists(ready_file):
            try:
                os.unlink(ready_file)
            except OSError:
                pass
            _vprint("[ArrayView] Native window started successfully", flush=True)
            _trace_launch_event(
                "display.process_evidence",
                adapter="native",
                stage=trace_stage,
                state="accepted",
                evidence="ready_flag",
            )
            return True, proc
        time.sleep(0.02)
    try:
        os.unlink(ready_file)
    except OSError:
        pass
    _vprint("[ArrayView] Native window started successfully", flush=True)
    _trace_launch_event(
        "display.process_evidence",
        adapter="native",
        stage=trace_stage,
        state="accepted_unverified",
        evidence="ready_flag_timeout",
    )
    return True, proc


def _server_viewer_connections_seen(port: int, timeout: float = 0.5) -> int:
    """Return the daemon's monotonic viewer WebSocket connection count."""
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return 0
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return int(
                    payload.get("viewer_connections_seen")
                    or payload.get("viewer_sockets")
                    or 0
                )
    except Exception:
        pass
    return 0


def _server_shell_sockets_open(port: int, timeout: float = 0.5) -> int:
    """Return the daemon's current native shell WebSocket count."""
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return 0
            payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return int(payload.get("shell_sockets") or 0)
    except Exception:
        pass
    return 0


def _wait_for_viewer_connection(
    port: int,
    *,
    before: int,
    timeout: float = 8.0,
) -> bool:
    """Wait until the backend has accepted a new viewer WebSocket connection."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_viewer_connections_seen(port, timeout=0.3) > before:
            return True
        time.sleep(0.1)
    return False


def _wait_for_native_shell_or_viewer_connection(
    port: int,
    *,
    viewer_before: int,
    timeout: float = 8.0,
) -> str | None:
    """Return the WebSocket evidence proving that pywebview became reachable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_viewer_connections_seen(port, timeout=0.3) > viewer_before:
            return "viewer_websocket"
        if _server_shell_sockets_open(port, timeout=0.3) > 0:
            return "shell_websocket"
        time.sleep(0.1)
    return None


def _server_native_ready(
    port: int,
    *,
    sid: str,
    native_request_id: str,
    expected_server_id: str | None,
    expected_server_pid: int | None = None,
    timeout: float = 0.5,
) -> bool:
    """Return whether this exact native attempt rendered on this backend."""
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    if payload.get("ok") is not True or payload.get("service") != "arrayview":
        return False
    if expected_server_id is not None:
        if payload.get("instance_id") != expected_server_id:
            return False
    elif expected_server_pid is not None:
        if payload.get("pid") != expected_server_pid:
            return False
    else:
        return False
    return (
        f"{sid}:{native_request_id}"
        in payload.get("native_ready_requests", [])
    )


def _wait_for_native_ready(
    port: int,
    *,
    sid: str,
    native_request_id: str,
    expected_server_id: str | None,
    expected_server_pid: int | None = None,
    timeout: float = 8.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _server_native_ready(
            port,
            sid=sid,
            native_request_id=native_request_id,
            expected_server_id=expected_server_id,
            expected_server_pid=expected_server_pid,
            timeout=0.3,
        ):
            return True
        time.sleep(0.1)
    return False


def _terminate_native_process(proc: subprocess.Popen | None) -> None:
    """Stop an owned native child, escalating when a GUI ignores SIGTERM."""
    _terminate_owned_process(proc)


def _open_cli_native_shell_after_server(
    *,
    port: int,
    sid: str,
    name: str,
    compare_sids: "_CompareSids | None",
    win_w: int,
    win_h: int,
    expected_server_id: str | None,
    expected_server_pid: int | None = None,
) -> bool:
    """Open a CLI native shell and return whether it is usable."""
    native_request_id = uuid.uuid4().hex
    url_shell = _shell_url(
        port,
        sid,
        name,
        compare_sids=compare_sids,
        native_request_id=native_request_id,
    )
    _trace_launch_event(
        "display.attempt_started",
        adapter="native",
        stage="post_tcp",
    )
    opened, proc = _open_webview_cli_tracked(
        url_shell,
        win_w,
        win_h,
        trace_stage="post_tcp",
    )
    connection_evidence = (
        _wait_for_native_ready(
            port,
            sid=sid,
            native_request_id=native_request_id,
            expected_server_id=expected_server_id,
            expected_server_pid=expected_server_pid,
        )
        if opened
        else None
    )
    if connection_evidence:
        _trace_launch_event(
            "display.attempt_finished",
            adapter="native",
            stage="post_tcp",
            state="ready",
            evidence="native_frame_ack",
        )
        return True
    if opened:
        _vprint(
            "[ArrayView] Native window did not connect to the backend; falling back to browser",
            flush=True,
        )
    _terminate_native_process(proc)
    _trace_launch_event(
        "display.attempt_finished",
        adapter="native",
        stage="post_tcp",
        state="failed",
        evidence="connection_timeout" if opened else "process_exit",
    )
    return False


# ── Server Port Utilities ─────────────────────────────────────────


def _server_alive(port: int, timeout: float = 0.5) -> bool:
    """Return True only if an ArrayView server is responding on the port."""
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
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
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
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


def _server_runtime_identity(
    port: int,
    *,
    attempts: int = 3,
    host: str = _LOOPBACK_HOST,
) -> tuple[str | None, str | None, int | None] | None:
    """Return a stable server identity, tolerating brief health-check stalls."""
    payload = _server_runtime_status(port, attempts=attempts, host=host)
    if payload is None:
        return None
    instance_id = payload.get("instance_id")
    process_start = payload.get("process_start")
    pid = payload.get("pid")
    return (
        instance_id if isinstance(instance_id, str) else None,
        process_start if isinstance(process_start, str) else None,
        pid if isinstance(pid, int) else None,
    )


def _server_runtime_status(
    port: int,
    *,
    attempts: int = 3,
    host: str = _LOOPBACK_HOST,
) -> dict | None:
    """Return the complete health contract for a specific server endpoint."""
    url = f"http://{host}:{port}/ping"
    for attempt in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                if resp.status != 200:
                    raise RuntimeError("unexpected health status")
                payload = json.loads(resp.read().decode("utf-8"))
            if payload.get("ok") is True and payload.get("service") == "arrayview":
                return payload
        except Exception:
            pass
        if attempt + 1 < max(1, attempts):
            time.sleep(0.05)
    return None


def _planned_server_snapshot(launch_context, port: int):
    expected = launch_context.evidence.server
    cli_default = launch_context.evidence.cli_default_server
    if cli_default is not None and cli_default.port == port:
        return cli_default
    return expected


def _revalidate_launch_server(launch_context, port: int) -> int:
    """Preserve the server ownership selected by an immutable launch plan."""
    if launch_context.plan.registration.value == "http_load":
        expected = _planned_server_snapshot(launch_context, port)

        current = _server_runtime_identity(port)
        if expected.server_instance_id is None:
            raise RuntimeError(
                "The selected ArrayView server does not expose generation "
                "identity; restart it before loading a new session"
            )
        same_server = (
            current is not None
            and current[0] == expected.server_instance_id
            and (
                expected.server_process_start is None
                or current[1] == expected.server_process_start
            )
            and (expected.server_pid is None or current[2] == expected.server_pid)
        )
        if not same_server:
            raise RuntimeError(
                "The selected existing ArrayView server disappeared or was "
                "replaced before the session could be registered"
            )
        return port

    if not _port_in_use(port):
        return port
    original_port = port
    for candidate in range(port + 1, port + 101):
        if not _port_in_use(candidate):
            _trace_launch_event(
                "server.revalidated",
                decision="alternate_port",
                previous_port=original_port,
                effective_port=candidate,
                reason="planned_owner_preserved",
            )
            return candidate
    raise RuntimeError(
        "Could not preserve planned server ownership: no free ArrayView port"
    )


def _server_hostname(port: int, timeout: float = 0.5) -> str | None:
    """Return the hostname reported by the ArrayView server on ``port``, or None."""
    url = f"http://{_LOOPBACK_HOST}:{port}/ping"
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
    relay_host: str = _LOOPBACK_HOST,
    expected_server_id: str | None = None,
) -> None:
    """Load *filepath* locally and POST the bytes to an ArrayView relay server.

    Used when the local port is a reverse-SSH-forwarded connection to a remote
    ArrayView server (e.g. tunnel-remote).  The relay server registers the
    session and writes its own VS Code signal file so a viewer tab opens there.

    ``relay_host`` defaults to localhost; only change it when the relay server
    is genuinely on a different network interface (rare).
    """
    import base64
    import numpy as np
    from arrayview._io import load_data

    status = _server_runtime_status(port, host=relay_host)
    capabilities = set(status.get("capabilities", [])) if status else set()
    compatible = (
        status is not None
        and isinstance(status.get("instance_id"), str)
        and status.get("protocol_version") == "1"
        and "transactional-relay-display" in capabilities
    )
    if not compatible:
        raise RuntimeError(
            f"No transaction-compatible ArrayView relay found on "
            f"{relay_host}:{port}; update or restart the remote ArrayView server"
        )
    current_server_id = str(status["instance_id"])
    if expected_server_id is not None and expected_server_id != current_server_id:
        raise RuntimeError("The selected relay server was replaced before upload")
    expected_server_id = current_server_id
    requested_sid = uuid.uuid4().hex

    print("[ArrayView] Relay mode: sending array to remote server...", flush=True)
    try:
        data = load_data(filepath)
    except Exception as e:
        print(f"[ArrayView] Failed to load {filepath}: {e}", flush=True)
        raise

    buf = io.BytesIO()
    np.save(buf, data)
    data_b64 = base64.b64encode(buf.getvalue()).decode()

    body = json.dumps(
        {
            "data_b64": data_b64,
            "name": name,
            "rgb": rgb,
            "expected_server_id": expected_server_id,
            "requested_sid": requested_sid,
            "require_display_ack": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{relay_host}:{port}/load_bytes",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())
    except Exception:
        _release_remote_sessions(
            port,
            [requested_sid],
            expected_server_id=expected_server_id,
            host=relay_host,
        )
        raise

    if "error" in result:
        _release_remote_sessions(
            port,
            [requested_sid],
            expected_server_id=expected_server_id,
            host=relay_host,
        )
        raise RuntimeError(f"[ArrayView] Relay server error: {result['error']}")
    if (
        result.get("sid") != requested_sid
        or result.get("display_acknowledged") is not True
    ):
        _release_remote_sessions(
            port,
            [requested_sid],
            expected_server_id=expected_server_id,
            host=relay_host,
        )
        raise RuntimeError(
            "[ArrayView] Relay server did not confirm the exact session display"
        )

    print(
        "[ArrayView] Array sent to relay server. "
        "VS Code viewer tab should open automatically.",
        flush=True,
    )


type _CSVValues = str | list[str] | tuple[str, ...]
type _CompareSids = list[str] | tuple[str, ...]


def _server_json_request(port: int, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"http://{_LOOPBACK_HOST}:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def _release_remote_sessions(
    port: int,
    sids: list[str],
    *,
    expected_server_id: str,
    host: str = _LOOPBACK_HOST,
) -> None:
    """Best-effort rollback fenced to the server that created the leases."""
    for sid in reversed(sids):
        request = urllib.request.Request(
            f"http://{host}:{port}/release/{sid}",
            data=b"",
            headers={
                "X-ArrayView-Expected-Server-ID": expected_server_id,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
        except Exception:
            pass


def _load_session_from_filepath(
    port: int,
    filepath: str,
    name: str,
    *,
    notify: bool = False,
    rgb: bool = False,
    compare_sids: _CompareSids | None = None,
    dir_patterns: list[str] | None = None,
    dir_overlay_specs: list[tuple[str, str]] | None = None,
    dir_case_regex: str | None = None,
    dir_exclude_cases: list[str] | None = None,
    collection_load: str = "lazy",
    collection_stack: str = "auto",
    background: bool = False,
    expected_server_id: str | None = None,
    native_request_id: str | None = None,
    release_on_disconnect: bool = False,
    related_sids: list[str] | None = None,
) -> dict:
    requested_sid = uuid.uuid4().hex
    payload = {
        "filepath": filepath,
        "name": name,
        "notify": notify,
        "requested_sid": requested_sid,
        "release_on_disconnect": release_on_disconnect,
        "related_sids": related_sids or [],
    }
    if rgb:
        payload["rgb"] = True
    if dir_patterns:
        payload["dir_patterns"] = dir_patterns
        payload["dir_overlay_specs"] = dir_overlay_specs or []
        payload["dir_case_regex"] = dir_case_regex
        payload["dir_exclude_cases"] = dir_exclude_cases or []
        payload["load"] = collection_load
        payload["stack"] = collection_stack
    if notify and compare_sids:
        payload["compare_sid"] = compare_sids[0]
        payload["compare_sids"] = _join_query_values(compare_sids)
    if background:
        payload["background"] = True
    if expected_server_id is not None:
        payload["expected_server_id"] = expected_server_id
    if native_request_id is not None:
        payload["native_request_id"] = native_request_id
    try:
        return _server_json_request(port, "/load", payload)
    except Exception:
        if expected_server_id is not None:
            _release_remote_sessions(
                port,
                [requested_sid],
                expected_server_id=expected_server_id,
            )
        raise


def _notify_existing_session(
    port: int,
    sid: str,
    name: str,
    *,
    url: str | None = None,
    wait: bool = False,
    expected_server_id: str | None = None,
) -> dict:
    payload = {"name": name, "wait": wait}
    if url:
        payload["url"] = url
    if expected_server_id is not None:
        payload["expected_server_id"] = expected_server_id
    return _server_json_request(port, f"/notify/{sid}", payload)


def _attach_vectorfield_to_session(
    port: int,
    sid: str,
    filepath: str,
    *,
    components_dim: int | None = None,
    expected_server_id: str | None = None,
) -> dict:
    payload = {
        "sid": sid,
        "filepath": filepath,
        "components_dim": components_dim,
    }
    if expected_server_id is not None:
        payload["expected_server_id"] = expected_server_id
    return _server_json_request(
        port,
        "/attach_vectorfield",
        payload,
    )


def _resolve_cli_window_mode(
    *,
    explicit_window: str | None,
    browser_flag: bool,
    config_window: str | None,
    in_vscode_terminal: bool,
    is_vscode_remote: bool,
    can_native_window: bool,
) -> dict[str, bool | str | None]:
    window_mode = explicit_window
    if browser_flag and window_mode is None:
        window_mode = "browser"
    if window_mode is None and config_window:
        window_mode = config_window
    if window_mode is None and in_vscode_terminal:
        window_mode = "vscode"
    if window_mode == "native" and is_vscode_remote:
        window_mode = "vscode"
    return {
        "window_mode": window_mode,
        "use_native_shell": (window_mode == "native")
        or (window_mode is None and can_native_window),
        "force_vscode": window_mode == "vscode",
        "requires_vscode_terminal": window_mode == "vscode" and not in_vscode_terminal,
        "warn_native_to_vscode": explicit_window == "native" and is_vscode_remote,
    }


def _should_notify_native_shell(
    use_native_shell: bool, overlay_sid: str | None
) -> bool:
    return use_native_shell and overlay_sid is None


def _plan_cli_port_strategy(
    *,
    port_in_use: bool,
    is_arrayview_server: bool,
    is_ssh: bool,
    is_vscode_remote: bool,
) -> dict[str, bool]:
    busy_non_arrayview = port_in_use and not is_arrayview_server
    return {
        "attempt_ssh_relay_before_scan": busy_non_arrayview and is_ssh,
        "requires_fixed_remote_port_error": busy_non_arrayview and is_vscode_remote,
        "should_scan_for_port": busy_non_arrayview and not is_vscode_remote,
        "should_check_existing_ssh_relay": is_arrayview_server and is_ssh,
    }


def _resolve_view_port(
    port: int, *, is_vscode_remote: bool, cli_default_port_alive: bool
) -> int:
    if port == 8123 and is_vscode_remote and cli_default_port_alive:
        return 8000
    return port


def _select_arrayview_launch_path(
    *, is_arrayview_server: bool, is_vscode_remote: bool
) -> str:
    if is_arrayview_server:
        return "existing_server"
    return "spawn_daemon"


def _normalize_view_window_request(
    window: str | bool | None, inline: bool | None
) -> dict[str, bool | str | None]:
    force_browser = False
    force_vscode = False
    explicit_inline = inline is not None
    explicit_window = window is not None
    if isinstance(window, str):
        window_lower = window.lower()
        if window_lower == "inline":
            inline = True
            window = False
        elif window_lower == "native":
            inline = False
            window = True
        elif window_lower == "browser":
            window = False
            inline = False
            force_browser = True
        elif window_lower == "vscode":
            window = False
            inline = False
            force_vscode = True
        elif window_lower == "none":
            window = False
            inline = False
        else:
            raise ValueError(
                "window must be 'inline', 'native', 'browser', 'vscode', or "
                "'none', "
                f"got {window!r}"
            )
    return {
        "window": window,
        "inline": inline,
        "force_browser": force_browser,
        "force_vscode": force_vscode,
        "explicit_inline": explicit_inline,
        "explicit_window": explicit_window,
    }


def _load_compare_sids(
    port: int,
    compare_files: list[str],
    *,
    expected_server_id: str | None = None,
    loaded_sids: list[str] | None = None,
) -> list[str]:
    compare_sids: list[str] = []
    for compare_file in compare_files:
        cmp_result = _load_session_from_filepath(
            port,
            compare_file,
            os.path.basename(compare_file),
            expected_server_id=expected_server_id,
        )
        if "error" in cmp_result:
            raise RuntimeError(
                f"Error from server while loading compare array: {cmp_result['error']}"
            )
        compare_sid = cmp_result.get("sid")
        if compare_sid:
            compare_sid = str(compare_sid)
            compare_sids.append(compare_sid)
            if loaded_sids is not None:
                loaded_sids.append(compare_sid)
    return compare_sids


def _open_cli_existing_server_view(
    *,
    port: int,
    sid: str,
    compare_sids: _CompareSids | None,
    overlay_sid: str | None,
    dims_override: tuple[int, int] | None,
    notify_native_shell: bool,
    notified: bool,
    native_request_id: str | None = None,
    name: str,
    base_file: str,
    watch: bool,
    window_mode: str | None,
    floating: bool,
    launch_context=None,
    overlay_names: list[str] | None = None,
) -> None:
    url = _viewer_url(
        port,
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sid,
        overlay_names=overlay_names,
        dims=dims_override,
    )
    route_kwargs = (
        {"launch_context": launch_context} if launch_context is not None else {}
    )
    fallback_kwargs = (
        {"launch_context": launch_context, "use_fallback": True}
        if launch_context is not None
        else {}
    )
    expected_server_id = (
        _planned_server_snapshot(launch_context, port).server_instance_id
        if launch_context is not None
        else None
    )
    expected_server_pid = None
    if expected_server_id is None:
        identity = _server_runtime_identity(port)
        expected_server_pid = identity[2] if identity is not None else None
    if notify_native_shell and notified:
        if native_request_id and _wait_for_native_ready(
            port,
            sid=sid,
            native_request_id=native_request_id,
            expected_server_id=expected_server_id,
            expected_server_pid=expected_server_pid,
        ):
            _vprint(f"Injected into existing window (port {port})")
            return
        notified = False
    if notify_native_shell:
        native_ready = _open_cli_native_shell_after_server(
            port=port,
            sid=sid,
            name=name,
            compare_sids=compare_sids,
            win_w=1200,
            win_h=800,
            expected_server_id=expected_server_id,
            expected_server_pid=expected_server_pid,
        )
        if not native_ready:
            _trace_launch_event(
                "fallback.applied",
                from_adapter="native",
                to_adapter="system_browser",
                reason="native_connection_failed",
            )
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _print_viewer_location(url, **route_kwargs)
            _open_browser(
                url,
                blocking=True,
                prefer_system_browser=True,
                title=f"ArrayView: {name}",
                floating=floating,
                **fallback_kwargs,
            )
        return
    if watch:
        _start_watch_thread(base_file, sid, port)
    if launch_context is not None and launch_context.plan.display.value == "none":
        return
    _open_browser(
        url,
        blocking=True,
        force_vscode=(window_mode == "vscode"),
        prefer_system_browser=(window_mode == "native"),
        title=f"ArrayView: {name}",
        floating=floating,
        **(
            {
                "launch_context": launch_context,
                "use_fallback": launch_context.plan.display.value == "native",
            }
            if launch_context is not None
            else {}
        ),
    )


def _register_cli_session_with_existing_server_impl(
    *,
    port: int,
    overlay_paths: list[str],
    compare_files: list[str],
    base_file: str,
    name: str,
    rgb: bool,
    use_native_shell: bool,
    vectorfield: str | None,
    vfield_components_dim: int | None,
    dir_patterns: list[str] | None = None,
    dir_overlay_specs: list[tuple[str, str]] | None = None,
    dir_case_regex: str | None = None,
    dir_exclude_cases: list[str] | None = None,
    collection_load: str = "lazy",
    collection_stack: str = "auto",
    overlay_names: list[str] | None = None,
    background_base: bool = False,
    expected_server_id: str | None = None,
    native_request_id: str | None = None,
    _loaded_sids: list[str],
) -> dict[str, object]:
    overlay_sids_list: list[str] = []
    resolved_overlay_names = overlay_names or [
        os.path.basename(path) or f"overlay {i + 1}"
        for i, path in enumerate(overlay_paths)
    ]
    for ov_path, ov_name in zip(overlay_paths, resolved_overlay_names):
        ov_result = _load_session_from_filepath(
            port,
            os.path.abspath(ov_path),
            ov_name,
            expected_server_id=expected_server_id,
        )
        if "error" in ov_result:
            raise RuntimeError(
                f"Error from server while loading overlay: {ov_result['error']}"
            )
        ov_sid = ov_result.get("sid")
        if ov_sid:
            ov_sid = str(ov_sid)
            overlay_sids_list.append(ov_sid)
            _loaded_sids.append(ov_sid)
    overlay_sid = ",".join(overlay_sids_list) if overlay_sids_list else None

    compare_sids = _load_compare_sids(
        port,
        compare_files,
        expected_server_id=expected_server_id,
        loaded_sids=_loaded_sids,
    )
    notify_native_shell = _should_notify_native_shell(use_native_shell, overlay_sid)
    result = _load_session_from_filepath(
        port,
        base_file,
        name,
        notify=notify_native_shell,
        rgb=rgb,
        compare_sids=compare_sids,
        dir_patterns=dir_patterns,
        dir_overlay_specs=dir_overlay_specs,
        dir_case_regex=dir_case_regex,
        dir_exclude_cases=dir_exclude_cases,
        collection_load=collection_load,
        collection_stack=collection_stack,
        background=background_base,
        expected_server_id=expected_server_id,
        native_request_id=native_request_id,
        release_on_disconnect=True,
        related_sids=[*compare_sids, *overlay_sids_list],
    )
    if "error" in result:
        error = str(result["error"])
        if (
            dir_patterns
            and dir_overlay_specs
            and not dir_case_regex
            and "Sparse overlays from --overlay-dir require --case-regex" in error
        ):
            error += (
                "\nThe server on this port is an older ArrayView process that "
                "does not support automatic --overlay-dir case inference yet. "
                f"Restart it with: arrayview --kill --port {port}"
            )
        raise RuntimeError(f"Error from server: {error}")
    _loaded_sids.append(str(result["sid"]))
    if vectorfield:
        vf_result = _attach_vectorfield_to_session(
            port,
            result["sid"],
            os.path.abspath(vectorfield),
            components_dim=vfield_components_dim,
            expected_server_id=expected_server_id,
        )
        if "error" in vf_result:
            raise RuntimeError(
                f"Error: failed to attach vector field: {vf_result['error']}"
            )
    collection_overlay_sids = [str(sid) for sid in result.get("overlay_sids", [])]
    if collection_overlay_sids:
        overlay_sids_list.extend(collection_overlay_sids)
        _loaded_sids.extend(collection_overlay_sids)
        overlay_sid = ",".join(overlay_sids_list)
    session_info = {
        "sid": result["sid"],
        "overlay_sid": overlay_sid,
        "compare_sids": compare_sids,
        "notify_native_shell": notify_native_shell,
        "notified": bool(result.get("notified", False)),
        "native_request_id": native_request_id,
    }
    if result.get("overlay_names"):
        session_info["overlay_names"] = result["overlay_names"]
    return session_info


def _register_cli_session_with_existing_server(**kwargs) -> dict[str, object]:
    loaded_sids: list[str] = []
    expected_server_id = kwargs.get("expected_server_id")
    try:
        return _register_cli_session_with_existing_server_impl(
            **kwargs,
            _loaded_sids=loaded_sids,
        )
    except Exception:
        if isinstance(expected_server_id, str):
            _release_remote_sessions(
                int(kwargs["port"]),
                loaded_sids,
                expected_server_id=expected_server_id,
            )
        raise


def _handle_cli_existing_server(
    *,
    port: int,
    base_file: str,
    name: str,
    compare_files: list[str],
    overlay_files: list[str],
    rgb: bool,
    vectorfield: str | None,
    vfield_components_dim: int | None,
    use_native_shell: bool,
    dims_override: tuple[int, int] | None,
    watch: bool,
    window_mode: str | None,
    floating: bool,
    is_remote: bool = False,
    launch_context=None,
    dir_patterns: list[str] | None = None,
    dir_overlay_specs: list[tuple[str, str]] | None = None,
    dir_case_regex: str | None = None,
    dir_exclude_cases: list[str] | None = None,
    collection_load: str = "lazy",
    collection_stack: str = "auto",
    overlay_names: list[str] | None = None,
) -> None:
    _trace_launch_event("server.decision", decision="reuse", port=port)
    expected_server_id = (
        _planned_server_snapshot(launch_context, port).server_instance_id
        if launch_context is not None
        else None
    )
    native_request_id = uuid.uuid4().hex if use_native_shell else None
    try:
        session_info = _register_cli_session_with_existing_server(
            port=port,
            overlay_paths=overlay_files,
            compare_files=compare_files,
            base_file=base_file,
            name=name,
            rgb=rgb,
            use_native_shell=use_native_shell,
            vectorfield=vectorfield,
            vfield_components_dim=vfield_components_dim,
            dir_patterns=dir_patterns,
            dir_overlay_specs=dir_overlay_specs,
            dir_case_regex=dir_case_regex,
            dir_exclude_cases=dir_exclude_cases,
            collection_load=collection_load,
            collection_stack=collection_stack,
            overlay_names=overlay_names,
            background_base=(
                is_remote
                and not vectorfield
                and not dir_patterns
                and not base_file.endswith((".npz", ".mat"))
            ),
            expected_server_id=expected_server_id,
            native_request_id=native_request_id,
        )
    except Exception as e:
        err = str(e)
        stale_stack_server = (
            dir_patterns
            and "Error from server" in err
            and (
                "Unsupported format" in err
                or "Sparse overlays from --overlay-dir require --case-regex" in err
                or (
                    dir_exclude_cases
                    and "is missing case(s)" in err
                )
            )
        )
        if stale_stack_server:
            fallback_port, already_running = _find_server_port(port + 1)
            if not already_running:
                print(
                    f"[ArrayView] Existing server on port {port} does not support "
                    f"this stack request; starting this checkout on port {fallback_port}.",
                    flush=True,
                )
                replacement_context = None
                if launch_context is not None:
                    from arrayview._launch_plan import (
                        Invocation,
                        LaunchIntent,
                        create_launch_context,
                    )

                    replacement_context = create_launch_context(
                        LaunchIntent(
                            invocation=Invocation.CLI,
                            port=fallback_port,
                            requested_window=window_mode,
                            persistent=is_remote,
                        )
                    )
                    if not replacement_context.plan.ok:
                        raise RuntimeError(
                            "Could not plan a compatible replacement server"
                        )
                _handle_cli_spawned_daemon(
                    port=fallback_port,
                    base_file=base_file,
                    name=name,
                    compare_files=compare_files,
                    overlay_files=overlay_files,
                    dims_override=dims_override,
                    use_native_shell=use_native_shell,
                    watch=watch,
                    window_mode=window_mode,
                    floating=floating,
                    is_remote=is_remote,
                    **(
                        {"launch_context": replacement_context}
                        if replacement_context is not None
                        else {}
                    ),
                    vectorfield=vectorfield,
                    vfield_components_dim=vfield_components_dim,
                    rgb=rgb,
                    demo_name=None,
                    demo_cleanup=False,
                    dir_patterns=dir_patterns,
                    dir_overlay_specs=dir_overlay_specs,
                    dir_case_regex=dir_case_regex,
                    dir_exclude_cases=dir_exclude_cases,
                    collection_load=collection_load,
                    collection_stack=collection_stack,
                    overlay_names=overlay_names,
                )
                return
        if os.path.isdir(base_file) and "Unsupported format" in str(e):
            print(
                f"Error: existing ArrayView server on port {port} does not support "
                "directory stacking. Restart it with "
                f"`arrayview --kill --port {port}` or choose a free port with "
                f"`--port`. ({e})"
            )
            sys.exit(1)
        if "Error from server" in err:
            print(f"Error loading {base_file} on ArrayView server port {port}: {e}")
            sys.exit(1)
        print(
            f"Error: port {port} is in use by another process. "
            f"Use --port to pick another. ({e})"
        )
        sys.exit(1)

    _trace_launch_event(
        "session.registered",
        registration="http_load",
        sid_tag=_launch_trace_tag(session_info["sid"]),
        port=port,
    )
    try:
        _open_cli_existing_server_view(
            port=port,
            sid=str(session_info["sid"]),
            compare_sids=session_info["compare_sids"],
            overlay_sid=session_info["overlay_sid"],
            overlay_names=(
                list(session_info.get("overlay_names", []))
                or overlay_names
                or [
                    os.path.basename(path) or f"overlay {i + 1}"
                    for i, path in enumerate(overlay_files)
                ]
            ),
            dims_override=dims_override,
            notify_native_shell=bool(session_info["notify_native_shell"]),
            notified=bool(session_info["notified"]),
            native_request_id=(
                str(session_info["native_request_id"])
                if session_info.get("native_request_id")
                else None
            ),
            name=name,
            base_file=base_file,
            watch=watch,
            window_mode=window_mode,
            floating=floating,
            **(
                {"launch_context": launch_context}
                if launch_context is not None
                else {}
            ),
        )
    except Exception:
        if expected_server_id is not None:
            rollback_sids = [str(session_info["sid"])]
            rollback_sids.extend(
                str(sid) for sid in session_info["compare_sids"]
            )
            rollback_sids.extend(
                sid
                for sid in str(session_info["overlay_sid"] or "").split(",")
                if sid
            )
            _release_remote_sessions(
                port,
                rollback_sids,
                expected_server_id=expected_server_id,
            )
        raise


def _handle_cli_spawned_daemon(
    *,
    port: int,
    base_file: str,
    name: str,
    compare_files: list[str],
    overlay_files: list[str],
    dims_override: tuple[int, int] | None,
    use_native_shell: bool,
    watch: bool,
    window_mode: str | None,
    floating: bool,
    is_remote: bool,
    vectorfield: str | None,
    vfield_components_dim: int | None,
    rgb: bool,
    demo_name: str | None,
    demo_cleanup: bool,
    dir_patterns: list[str] | None = None,
    dir_overlay_specs: list[tuple[str, str]] | None = None,
    dir_case_regex: str | None = None,
    dir_exclude_cases: list[str] | None = None,
    collection_load: str = "lazy",
    collection_stack: str = "auto",
    overlay_names: list[str] | None = None,
    launch_context=None,
) -> None:
    sid = uuid.uuid4().hex
    overlay_count = len(dir_overlay_specs or []) if dir_patterns is not None else len(overlay_files)
    overlay_sids = [uuid.uuid4().hex for _ in range(overlay_count)]
    overlay_sid = ",".join(overlay_sids) if overlay_sids else None
    resolved_overlay_names = (
        [spec[0] for spec in (dir_overlay_specs or [])]
        if dir_patterns is not None
        else overlay_names
        or [os.path.basename(path) or f"overlay {i + 1}" for i, path in enumerate(overlay_files)]
    )

    vfield_abs = os.path.abspath(vectorfield) if vectorfield else None
    daemon_server_id = None

    from arrayview._instance_registry import InstanceRegistry

    with InstanceRegistry().startup_lock(timeout=20.0):
        server_already_live = (
            _server_alive(port) if launch_context is None else False
        )
        _trace_launch_event(
            "server.decision",
            decision="reuse" if server_already_live else "spawn",
            port=port,
        )
        if server_already_live:
            _handle_cli_existing_server(
                port=port,
                base_file=base_file,
                name=name,
                compare_files=compare_files,
                overlay_files=overlay_files,
                rgb=rgb,
                vectorfield=vectorfield,
                vfield_components_dim=vfield_components_dim,
                use_native_shell=use_native_shell,
                dims_override=dims_override,
                watch=watch,
                window_mode=window_mode,
                floating=floating,
                is_remote=is_remote,
                **(
                    {"launch_context": launch_context}
                    if launch_context is not None
                    else {}
                ),
                dir_patterns=dir_patterns,
                dir_overlay_specs=dir_overlay_specs,
                dir_case_regex=dir_case_regex,
                dir_exclude_cases=dir_exclude_cases,
                collection_load=collection_load,
                collection_stack=collection_stack,
                overlay_names=resolved_overlay_names,
            )
            return

        if launch_context is not None:
            port = _revalidate_launch_server(launch_context, port)

        if (
            launch_context is not None
            and launch_context.plan.display.value == "vscode"
            and is_remote
        ) or (launch_context is None and not use_native_shell):
            _configure_vscode_port_preview(
                port, in_vscode=not is_remote, is_remote=is_remote
            )

        daemon_connect_timeout = (
            _LOCAL_VSCODE_CONNECT_TIMEOUT_SECONDS
            if launch_context is not None
            and launch_context.placement.value == "vscode_local"
            and launch_context.plan.display.value == "vscode"
            else None
        )

        script = (
            f"from arrayview._launcher import _serve_daemon;"
            f"_serve_daemon("
            f"{repr(base_file)}, {port}, {repr(sid)},"
            f" name={repr(demo_name)},"
            f" cleanup={demo_cleanup},"
            f" overlay_filepaths={repr([os.path.abspath(p) for p in overlay_files])},"
            f" overlay_sids={repr(overlay_sids)},"
            f" overlay_names={repr(resolved_overlay_names)},"
            f" vfield_filepath={repr(vfield_abs)},"
            f" vfield_components_dim={repr(vfield_components_dim)},"
            f" persist={is_remote},"
            f" connect_timeout={repr(daemon_connect_timeout)},"
            f" rgb={rgb},"
            f" dir_patterns={repr(dir_patterns)},"
            f" dir_overlay_specs={repr(dir_overlay_specs)},"
            f" dir_case_regex={repr(dir_case_regex)},"
            f" dir_exclude_cases={repr(dir_exclude_cases)},"
            f" collection_load={repr(collection_load)},"
            f" collection_stack={repr(collection_stack)},"
            f")"
        )

        popen_kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
            "start_new_session": sys.platform != "win32",
        }
        if os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
            from arrayview._launch_trace import trace_child_environment

            child_env = trace_child_environment()
            if child_env is not None:
                popen_kwargs["env"] = child_env
        _trace_launch_event(
            "daemon.spawn_requested",
            port=port,
            owner="persistent" if is_remote else "transient",
            sid_tag=_launch_trace_tag(sid),
        )
        daemon_proc = subprocess.Popen(
            [sys.executable, "-c", script],
            **popen_kwargs,
        )
        _trace_launch_event(
            "daemon.spawned",
            child_pid=getattr(daemon_proc, "pid", None),
            sid_tag=_launch_trace_tag(sid),
        )

        if not _wait_for_spawned_server(daemon_proc, port, timeout=15.0):
            _terminate_owned_process(daemon_proc)
            _trace_launch_event("backend.tcp_failed", port=port)
            print(
                f"Error: the spawned ArrayView server did not claim port {port}. "
                "Use --port to pick another."
            )
            sys.exit(1)
        daemon_identity = _server_runtime_identity(port)
        if daemon_identity is None or daemon_identity[0] is None:
            _terminate_owned_process(daemon_proc)
            print(
                "Error: the spawned ArrayView server did not publish a stable "
                "generation identity."
            )
            sys.exit(1)
        daemon_server_id = daemon_identity[0]
        _trace_launch_event(
            "backend.identity_verified",
            port=port,
            child_pid=getattr(daemon_proc, "pid", None),
        )

    try:
        compare_sids = _load_compare_sids(
            port,
            compare_files,
            expected_server_id=daemon_server_id,
        )
    except Exception as e:
        _terminate_owned_process(daemon_proc)
        print(f"Error while loading compare array: {e}")
        sys.exit(1)

    try:
        _open_cli_spawned_view(
            port=port,
            sid=sid,
            compare_sids=compare_sids,
            overlay_sid=overlay_sid,
            overlay_names=resolved_overlay_names,
            dims_override=dims_override,
            use_native_shell=use_native_shell,
            name=name,
            base_file=base_file,
            watch=watch,
            window_mode=window_mode,
            floating=floating,
            is_remote=is_remote,
            **(
                {"launch_context": launch_context}
                if launch_context is not None
                else {}
            ),
            expected_server_id=daemon_server_id,
        )
    except Exception:
        _terminate_owned_process(daemon_proc)
        raise


def _open_cli_spawned_view(
    *,
    port: int,
    sid: str,
    compare_sids: _CompareSids | None,
    overlay_sid: str | None,
    dims_override: tuple[int, int] | None,
    use_native_shell: bool,
    name: str,
    base_file: str,
    watch: bool,
    window_mode: str | None,
    floating: bool,
    is_remote: bool,
    launch_context=None,
    overlay_names: list[str] | None = None,
    expected_server_id: str | None = None,
) -> None:
    url = _viewer_url(
        port,
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sid,
        overlay_names=overlay_names,
        dims=dims_override,
    )
    route_kwargs = (
        {"launch_context": launch_context} if launch_context is not None else {}
    )
    fallback_kwargs = (
        {"launch_context": launch_context, "use_fallback": True}
        if launch_context is not None
        else {}
    )
    if _should_notify_native_shell(use_native_shell, overlay_sid):
        native_ready = _open_cli_native_shell_after_server(
            port=port,
            sid=sid,
            name=name,
            compare_sids=compare_sids,
            win_w=1400,
            win_h=900,
            expected_server_id=expected_server_id,
        )
        if not native_ready:
            _trace_launch_event(
                "fallback.applied",
                from_adapter="native",
                to_adapter="system_browser",
                reason="native_connection_failed",
            )
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _print_viewer_location(url, **route_kwargs)
            _open_browser(
                url,
                blocking=True,
                force_vscode=(window_mode == "vscode"),
                prefer_system_browser=(window_mode == "native"),
                title=f"ArrayView: {name}",
                floating=floating,
                **fallback_kwargs,
            )
        return
    if use_native_shell and overlay_sid:
        _vprint(
            "[ArrayView] Overlay mode: opening browser (native shell injection not supported with overlay)",
            flush=True,
        )
    _print_viewer_location(url, **route_kwargs)
    if watch:
        _start_watch_thread(base_file, sid, port)
    if launch_context is not None and launch_context.plan.display.value == "none":
        return
    _open_browser(
        url,
        blocking=True,
        force_vscode=(window_mode == "vscode"),
        prefer_system_browser=(window_mode == "native"),
        title=f"ArrayView: {name}",
        floating=floating,
        **(
            {
                "launch_context": launch_context,
                "use_fallback": launch_context.plan.display.value == "native",
            }
            if launch_context is not None
            else {}
        ),
    )


def _parse_dims_spec(spec: str) -> tuple[int, int] | None:
    parts = [p.strip().lower() for p in spec.split(",")]
    x_idx = next((i for i, p in enumerate(parts) if p == "x"), None)
    y_idx = next((i for i, p in enumerate(parts) if p == "y"), None)
    if x_idx is not None and y_idx is not None and x_idx != y_idx:
        return (x_idx, y_idx)
    if len(parts) == 2:
        try:
            a, b = int(parts[0]), int(parts[1])
            if a >= 0 and b >= 0 and a != b:
                return (a, b)
        except ValueError:
            pass
    return None


def _resolve_view_display_defaults(
    *,
    inline: bool | None,
    window: str | bool | None,
    is_jupyter: bool,
    explicit_window: bool,
    explicit_inline: bool,
    force_browser: bool,
    force_vscode: bool,
    config_window: str | None,
) -> dict[str, bool | None]:
    if inline is None:
        inline = is_jupyter
    if window is None:
        window = not is_jupyter
    if window:
        inline = False
    if (
        not explicit_window
        and not explicit_inline
        and not force_browser
        and not force_vscode
        and config_window
    ):
        if config_window == "inline":
            inline = True
            window = False
        elif config_window == "native":
            window = True
            inline = False
        elif config_window == "browser":
            window = False
            inline = False
            force_browser = True
        elif config_window == "vscode":
            window = False
            inline = False
            force_vscode = True
    return {
        "inline": inline,
        "window": window,
        "force_browser": force_browser,
        "force_vscode": force_vscode,
    }


def _promote_view_to_vscode_terminal(
    *,
    in_vscode_terminal: bool,
    inline: bool | None,
    window: str | bool | None,
    explicit_window: bool,
    explicit_inline: bool,
    force_vscode: bool,
    force_browser: bool,
) -> dict[str, bool | str | None]:
    if (
        in_vscode_terminal
        and not inline
        and not explicit_window
        and not explicit_inline
        and not force_vscode
        and not force_browser
    ):
        force_vscode = True
        if window is True:
            window = False
    return {
        "window": window,
        "force_vscode": force_vscode,
    }


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection((_LOOPBACK_HOST, port), timeout=0.3):
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


def _terminate_owned_process(proc: subprocess.Popen | None) -> None:
    """Stop only a process created by this invocation."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass
    except Exception:
        pass


def _wait_for_spawned_server(
    proc: subprocess.Popen, port: int, timeout: float = 15.0
) -> bool:
    """Wait for *proc* itself—not merely some listener—to own *port*."""
    expected_pid = getattr(proc, "pid", None)
    if not isinstance(expected_pid, int):
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if proc.poll() is not None:
                return False
        except Exception:
            return False
        if _server_pid(port) == expected_pid:
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


async def _serve_background(
    port: int,
    stop_when_closed: bool = False,
    owner_mode: str = "in_process",
    connect_timeout: float = 20.0,
):
    global _loading_port
    _loading_port = None  # reset for this server lifetime
    _session_mod.SERVER_LOOP = asyncio.get_running_loop()
    _session_mod.SERVER_PORT = port
    # Bind on every loopback address that resolves for ``localhost`` (both
    # ``::1`` and ``127.0.0.1``).  VS Code's port auto-forward detection
    # scans IPv4 loopback; an IPv6-only ``[::1]`` listener is not reliably
    # forwarded in tunnel mode, leaving ``asExternalUri`` returning the
    # localhost URL unchanged (blank viewer tab on the remote client).  The
    # macOS native shell connects via ``localhost`` → ``::1``, so we must
    # keep the IPv6 listener too.
    socks = _make_loopback_sockets(port)
    registry, record = _register_server_runtime(port, owner_mode)

    # Bind the loading-page server on an OS-chosen ephemeral port.
    # This uses only stdlib so it starts in microseconds — well before
    # uvicorn's heavy imports finish.  _loading_port is read by the main
    # thread after _server_ready_event fires.
    try:
        _lsock = _make_loopback_socket(0)
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
        asyncio.create_task(
            _stop_server_when_viewer_closes(
                server, connect_timeout=connect_timeout
            )
        )
    try:
        await server.serve(sockets=socks)
    finally:
        registry.remove(record.instance_id)


def _with_loading(url: str) -> str:
    """Wrap *url* with the loading-page proxy when the pre-server is active.

    Returns a URL pointing to the loading page (which will redirect to *url*
    once the real server responds).  Falls back to *url* unchanged when the
    loading server is not running (e.g. server was already alive).
    """
    if _loading_port is not None:
        encoded = urllib.parse.quote(url, safe="")
        return f"http://{_LOOPBACK_HOST}:{_loading_port}/?target={encoded}"
    return url


_OVERLAY_PALETTE = ["d65b5b", "5aa66a", "5f84d7", "d5ad4f", "c06bb7", "55b9bd"]

_JUPYTER_PROXY_INLINE_CACHE: bool | None = None
_CLI_DAEMON_CONNECT_TIMEOUT_SECONDS = 20.0
_LOCAL_VSCODE_CONNECT_TIMEOUT_SECONDS = 70.0
# CLI-launched transient viewers should shut down promptly once the last viewer
# closes. Keeping a warm idle daemon around caused native-window sessions to
# appear orphaned after close.
_CLI_DAEMON_IDLE_SECONDS = 0.0
# A tunnel server must survive a short VS Code reload, but it must not remain
# alive forever when the opener never connects (for example after an extension
# reload or a failed signal-file handoff).
_PERSIST_DAEMON_CONNECT_TIMEOUT_SECONDS = float(
    os.environ.get("ARRAYVIEW_PERSIST_CONNECT_TIMEOUT_SECONDS", "210")
)
_PERSIST_DAEMON_IDLE_SECONDS = float(
    os.environ.get("ARRAYVIEW_PERSIST_IDLE_SECONDS", "1800")
)


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
    overlay_names: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    parts = [f"sid={sid}"]
    if overlay_sids:
        parts.append(f"overlay_sid={_join_query_values(overlay_sids)}")
    if overlay_colors:
        parts.append(f"overlay_colors={_join_query_values(overlay_colors)}")
    if overlay_names:
        parts.append(
            "overlay_names="
            + ",".join(urllib.parse.quote(str(value)) for value in overlay_names)
        )
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
    overlay_names: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    return "/" + _viewer_query(
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sids,
        overlay_colors=overlay_colors,
        overlay_names=overlay_names,
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
    overlay_names: _CSVValues | None = None,
    dims: tuple[int, int] | None = None,
    inline: bool = False,
) -> str:
    return f"http://localhost:{port}" + _viewer_path(
        sid,
        compare_sids=compare_sids,
        overlay_sids=overlay_sids,
        overlay_colors=overlay_colors,
        overlay_names=overlay_names,
        dims=dims,
        inline=inline,
    )


def _shell_url(
    port: int,
    sid: str,
    name: str,
    *,
    compare_sids: _CompareSids | None = None,
    native_request_id: str | None = None,
) -> str:
    parts = [
        f"init_sid={sid}",
        f"init_name={urllib.parse.quote(str(name))}",
    ]
    if compare_sids:
        parts.append(f"init_compare_sid={compare_sids[0]}")
        parts.append(f"init_compare_sids={_join_query_values(compare_sids)}")
    if native_request_id:
        parts.append(
            f"native_request_id={urllib.parse.quote(native_request_id)}"
        )
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


def _normalize_inline_mode_heights(mode_heights) -> dict[str, int]:
    if mode_heights is None:
        return {}
    if not isinstance(mode_heights, Mapping):
        raise TypeError("mode_heights must be a mapping of mode names to pixel heights")

    normalized = {}
    for mode, value in mode_heights.items():
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("mode_heights keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("mode_heights values must be positive integer pixel heights")
        normalized[mode.strip().lower()] = value
    return normalized


def _normalize_inline_height(height) -> int:
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer pixel height")
    return height


def _script_json(value) -> str:
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _build_jupyter_inline_html(
    viewer_url: str,
    port: int,
    height: int,
    mode_heights: dict[str, int],
    *,
    use_proxy: bool,
):
    parsed = urllib.parse.urlparse(viewer_url)
    proxied_target = f"proxy/{port}{parsed.path or '/'}"
    if parsed.query:
        proxied_target += f"?{parsed.query}"
    initial_src = (
        urllib.parse.urljoin(_jupyter_base_url_prefix(), proxied_target)
        if use_proxy
        else viewer_url
    )
    container_id = f"arrayview-inline-{uuid.uuid4().hex}"

    return f"""
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
  const directSrc = {json.dumps(viewer_url)};
  const useProxy = {json.dumps(use_proxy)};
  const defaultHeight = {height};
  const modeHeights = {_script_json(mode_heights)};
  const modeAliases = {{
    multiview: 'ortho',
    'compare-mv': 'compare-ortho',
  }};
  const resizeForMode = (mode, preferredHeight) => {{
    if (typeof mode !== 'string') return;
    const configured = modeHeights[mode] ?? modeHeights[modeAliases[mode]];
    const automatic = mode === 'multiview' && Number.isFinite(preferredHeight)
      ? Math.min(defaultHeight, Math.max(240, Math.ceil(preferredHeight)))
      : defaultHeight;
    const requested = configured ?? automatic;
    host.style.height = `${{requested}}px`;
  }};
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
  const proxiedSrc = useProxy
    ? new URL({json.dumps(proxied_target)}, window.location.origin + base).toString()
    : directSrc;
  let loaded = false;
  const cleanup = () => {{
    window.removeEventListener('message', onMessage);
    if (fallbackTimer) window.clearTimeout(fallbackTimer);
    removalObserver.disconnect();
  }};
  const onMessage = (event) => {{
    const msg = event && event.data;
    if (!msg || msg.source !== 'arrayview-viewer') return;
    if (event.source !== frame.contentWindow) return;
    if (msg.phase === 'mode-change') {{
      resizeForMode(
        msg.detail && msg.detail.mode,
        msg.detail && msg.detail.preferredHeight,
      );
      return;
    }}
    if (msg.phase !== 'script-loaded') return;
    loaded = true;
    if (fallbackTimer) window.clearTimeout(fallbackTimer);
  }};
  window.addEventListener('message', onMessage);
  const removalObserver = new MutationObserver(() => {{
    if (!host.isConnected) cleanup();
  }});
  removalObserver.observe(document.documentElement, {{ childList: true, subtree: true }});
  const fallbackTimer = window.setTimeout(() => {{
    if (loaded || !useProxy) return;
    frame.src = directSrc;
  }}, 1500);
  frame.src = useProxy ? proxiedSrc : directSrc;
}})();
</script>
""".strip()


def _make_jupyter_inline_html(
    viewer_url: str,
    port: int,
    height: int,
    mode_heights: dict[str, int],
    *,
    use_proxy: bool,
):
    from IPython.display import HTML

    return HTML(
        _build_jupyter_inline_html(
            viewer_url,
            port,
            height,
            mode_heights,
            use_proxy=use_proxy,
        )
    )


def _make_jupyter_proxy_inline_html(
    viewer_url: str,
    port: int,
    height: int,
    mode_heights: dict[str, int] | None = None,
):
    return _make_jupyter_inline_html(
        viewer_url,
        port,
        height,
        mode_heights or {},
        use_proxy=True,
    )


def _make_resizable_jupyter_iframe(
    viewer_url: str,
    port: int,
    height: int,
    mode_heights: dict[str, int],
):
    from types import MethodType

    from IPython.display import IFrame

    iframe = IFrame(src=viewer_url, width="100%", height=height)

    def _repr_html_(self):
        return _make_jupyter_inline_html(
            self.src,
            port,
            self.height,
            mode_heights,
            use_proxy=False,
        ).data

    iframe._repr_html_ = MethodType(_repr_html_, iframe)
    return iframe


# ── ViewHandle and view() API ────────────────────────────────────


class ViewHandle(str):
    """Returned by :func:`view`.  Behaves as a URL string for backward compatibility
    and additionally exposes ``.update(arr)`` and ``.close()``.  It can also be
    used as a context manager when the session should be released at the end of
    a block.

    Example::

        v = view(arr)
        # ... modify arr ...
        v.update(arr2)        # viewer refreshes in-place
    """

    def __new__(
        cls,
        url: str,
        sid: str,
        port: int,
        server_id: str | None = None,
    ):
        obj = super().__new__(cls, url)
        obj._sid = sid
        obj._port = port
        obj._server_id = server_id
        obj._closed = False
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
            f"http://{_LOOPBACK_HOST}:{self._port}/update/{self._sid}",
            data=body,
            headers=(
                {"X-ArrayView-Expected-Server-ID": self._server_id}
                if self._server_id is not None
                else {}
            ),
            method="POST",
        )
        try:
            with _req.urlopen(request, timeout=10) as resp:
                resp.read()
        except Exception as e:
            raise RuntimeError(
                f"[ArrayView] Failed to update viewer: {e}\n"
                f"  URL: http://{_LOOPBACK_HOST}:{self._port}/update/{self._sid}\n"
                f"  Is the ArrayView server still running?"
            ) from e

    def close(self) -> None:
        """Release this viewer session.

        Closing a handle more than once is a local no-op.  If the request
        fails, the handle remains open so the caller can retry.
        """
        if self._closed:
            return

        release_url = (
            f"http://{_LOOPBACK_HOST}:{self._port}/release/{self._sid}"
        )
        request = urllib.request.Request(
            release_url,
            data=b"",
            headers=(
                {"X-ArrayView-Expected-Server-ID": self._server_id}
                if self._server_id is not None
                else {}
            ),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                response.read()
        except Exception as exc:
            raise RuntimeError(
                f"[ArrayView] Failed to close viewer: {exc}\n"
                f"  URL: {release_url}\n"
                f"  Is the ArrayView server still running?"
            ) from exc
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def view(
    *arrays,
    name=None,
    port: int = 8123,
    inline: bool | None = None,
    height: int = 600,
    mode_heights: Mapping[str, int] | None = None,
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

        ``height`` sets the normal-view pixel height of inline notebook IFrames.
        Horizontal ortho view automatically shrinks to its content.
        ``mode_heights`` can override it while a viewer mode is active, for
        example ``{"ortho": 360, "qmri": 480}``.

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

    height = _normalize_inline_height(height)
    _inline_mode_heights = _normalize_inline_mode_heights(mode_heights)

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
            _is_lazy = (
                "nibabel" in _mod
                or "zarr" in _mod
                or "h5py" in _mod
                or getattr(data, "_av_lazy", False)
            )
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
    _raw_window = window
    _window_request = _normalize_view_window_request(window, inline)
    window = _window_request["window"]
    inline = _window_request["inline"]
    _force_browser = bool(_window_request["force_browser"])
    _force_vscode = bool(_window_request["force_vscode"])
    _explicit_inline = bool(_window_request["explicit_inline"])
    _explicit_window = bool(_window_request["explicit_window"])

    from arrayview._launch_plan import (
        CallerScope,
        Display,
        Invocation,
        LaunchIntent,
        Registration,
        create_launch_context,
    )

    if _is_julia_env():
        _invocation = Invocation.JULIA
    elif _platform_mod._in_matlab():
        _invocation = Invocation.MATLAB
    elif _in_jupyter():
        _invocation = Invocation.JUPYTER
    else:
        _invocation = Invocation.PYTHON

    if isinstance(_raw_window, str):
        _requested_window = _raw_window
    elif _raw_window is True:
        _requested_window = "native"
    elif _raw_window is False:
        _requested_window = "none"
    else:
        _requested_window = None

    # PythonCall historically treats window=False as an inline Julia display.
    # Keep that public quirk while routing the resulting mode through the plan.
    if _invocation is Invocation.JULIA and _raw_window is False:
        _requested_window = "inline"
        inline = True
    _is_ijulia = _invocation is Invocation.JULIA and _in_julia_jupyter()
    if _is_ijulia and not _explicit_window and not _explicit_inline:
        _requested_window = "inline"
        inline = True

    if _invocation is Invocation.JUPYTER or _is_ijulia:
        _caller_scope = CallerScope.KERNEL
    elif _invocation is Invocation.MATLAB:
        _caller_scope = CallerScope.EMBEDDED
    elif _invocation is Invocation.PYTHON and _is_script_mode():
        _caller_scope = CallerScope.SCRIPT
    else:
        _caller_scope = CallerScope.INTERACTIVE

    _launch_context = create_launch_context(
        LaunchIntent(
            invocation=_invocation,
            port=port,
            requested_window=_requested_window,
            inline=inline,
            window_explicit=_explicit_window,
            inline_explicit=_explicit_inline,
        ),
        caller_scope=_caller_scope,
    )
    _launch_snapshot = _launch_context.evidence
    _launch_plan = _launch_context.plan
    if not _launch_plan.ok:
        raise ValueError(f"Cannot launch ArrayView: {_launch_plan.failure.value}")

    port = _launch_plan.effective_port
    inline = _launch_plan.display is Display.INLINE
    window = _launch_plan.display is Display.NATIVE
    _force_browser = _launch_plan.display is Display.BROWSER
    _force_vscode = _launch_plan.display is Display.VSCODE
    _suppress_open = _launch_plan.display is Display.NONE

    port = _revalidate_launch_server(_launch_context, port)

    # --- Julia path: only single-array supported ---
    if _invocation is Invocation.JULIA:
        if n_arrays > 1:
            raise NotImplementedError(
                "Multi-array view() is not yet supported in Julia mode"
            )

        return _view_julia(
            np.array(data) if not isinstance(data, np.ndarray) else data,
            name,
            port,
            window=window,
            inline=inline,
            height=height,
            mode_heights=_inline_mode_heights,
            floating=floating,
            launch_context=_launch_context,
        )

    # VS Code tunnel/remote: use the server + WebSocket path.
    # WS works through the devtunnel when the port is public — the extension
    # calls ensurePortPublic() before asExternalUri.
    # Fall through to the server-based path below (don't return early).

    # With a planned existing server: register arrays via /load. Keep display
    # failures outside the registration exception boundary so callers receive
    # the real adapter error.
    if _launch_plan.registration is Registration.HTTP_LOAD:
        _loaded_sids: list[str] = []
        _expected_server_id = _planned_server_snapshot(
            _launch_context, port
        ).server_instance_id
        assert _expected_server_id is not None
        try:
            import tempfile as _tf

            for _array, _name, _rgb in zip(arrays, names, rgbs):
                with _tf.NamedTemporaryFile(suffix=".npy", delete=False) as _tmp:
                    _tmp_path = _tmp.name
                try:
                    np.save(_tmp_path, _array)
                    _result = _load_session_from_filepath(
                        port,
                        _tmp_path,
                        _name,
                        rgb=_rgb,
                        expected_server_id=_expected_server_id,
                        release_on_disconnect=inline,
                    )
                finally:
                    try:
                        os.unlink(_tmp_path)
                    except OSError:
                        pass
                if "error" in _result:
                    raise RuntimeError(_result["error"])
                _loaded_sids.append(str(_result["sid"]))
        except Exception as e:
            _release_remote_sessions(
                port,
                _loaded_sids,
                expected_server_id=_expected_server_id,
            )
            _vprint(
                f"[ArrayView] Failed to register with --serve server: {e}",
                flush=True,
            )
            raise RuntimeError(
                "ArrayView could not register the session with the selected "
                "existing server"
            ) from e

        try:
            sid, *_compare_sids = _loaded_sids
            url_viewer = _viewer_url(port, sid, compare_sids=_compare_sids)

            if inline:
                from IPython.display import display as _ipy_display

                _inline_url = _viewer_url(
                    port, sid, compare_sids=_compare_sids, inline=True
                )
                if _should_use_jupyter_proxy_inline():
                    _inline_html = _make_jupyter_proxy_inline_html(
                        _inline_url, port, height, _inline_mode_heights
                    )
                    if n_arrays == 1:
                        return _inline_html
                    _ipy_display(_inline_html)
                    return tuple(
                        ViewHandle(url_viewer, s, port, _expected_server_id)
                        for s in [sid] + _compare_sids
                    )
                iframe = _make_resizable_jupyter_iframe(
                    _inline_url, port, height, _inline_mode_heights
                )
                if n_arrays == 1:
                    return iframe
                _ipy_display(iframe)
                return tuple(
                    ViewHandle(url_viewer, s, port, _expected_server_id)
                    for s in [sid] + _compare_sids
                )

            if _launch_plan.display is Display.NATIVE:
                _session_mod._window_process = _open_webview_with_fallback(
                    _shell_url(port, sid, name, compare_sids=_compare_sids),
                    1400,
                    900,
                    shell_port=port,
                    floating=floating,
                    launch_context=_launch_context,
                    fallback_url=url_viewer,
                    title=f"ArrayView: {name}",
                )
            elif not _suppress_open:
                _open_browser(
                    url_viewer,
                    force_vscode=_force_vscode,
                    blocking=True,
                    title=f"ArrayView: {name}",
                    floating=floating,
                    launch_context=_launch_context,
                )
            _print_viewer_location(url_viewer, launch_context=_launch_context)
            if n_arrays == 1:
                return ViewHandle(url_viewer, sid, port, _expected_server_id)
            return tuple(
                ViewHandle(url_viewer, s, port, _expected_server_id)
                for s in [sid] + _compare_sids
            )
        except Exception:
            _release_remote_sessions(
                port,
                _loaded_sids,
                expected_server_id=_expected_server_id,
            )
            raise

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

    session.release_on_disconnect = bool(
        inline or _launch_context.caller_scope is CallerScope.EMBEDDED
    )
    session.related_release_sids = [*_compare_sids, *_overlay_sids]

    win_w, win_h = 1400, 900

    # Revalidate and bind while holding the cross-process startup lock. Session
    # objects are caller-owned and must never be left committed after a lost
    # bind race.
    from arrayview._instance_registry import InstanceRegistry

    global _loading_port
    _early_window_opened = False
    can_native_window = _launch_plan.display is Display.NATIVE
    owned_sids = [session.sid, *_compare_sids, *_overlay_sids]
    try:
        with InstanceRegistry().startup_lock(timeout=20.0):
            identity = _server_runtime_identity(port) if _port_in_use(port) else None
            server_pid = identity[2] if identity is not None else None
            our_pid = os.getpid()
            if server_pid != our_pid and _port_in_use(port):
                original_port = port
                for candidate in range(port + 1, min(65535, port + 100) + 1):
                    if not _port_in_use(candidate):
                        port = candidate
                        _trace_launch_event(
                            "server.revalidated",
                            decision="alternate_port",
                            previous_port=original_port,
                            effective_port=port,
                            observed_pid=server_pid,
                        )
                        break
                else:
                    raise RuntimeError(
                        "Could not find a free port for the in-process server"
                    )
                server_pid = None
                _vprint(
                    f"[ArrayView] Port {original_port} was claimed; using {port}",
                    flush=True,
                )

            if server_pid is None:
                _session_mod.SERVER_LOOP = None
                _server_ready_event.clear()
                _script = _launch_context.caller_scope is CallerScope.SCRIPT
                if _script and _launch_plan.display is Display.VSCODE:
                    if _launch_context.placement.value == "vscode_remote":
                        _script_connect_timeout = (
                            _PERSIST_DAEMON_CONNECT_TIMEOUT_SECONDS
                        )
                    elif _launch_context.placement.value == "vscode_local":
                        _script_connect_timeout = (
                            _LOCAL_VSCODE_CONNECT_TIMEOUT_SECONDS
                        )
                    else:
                        _script_connect_timeout = (
                            _CLI_DAEMON_CONNECT_TIMEOUT_SECONDS
                        )
                else:
                    _script_connect_timeout = _CLI_DAEMON_CONNECT_TIMEOUT_SECONDS
                threading.Thread(
                    target=lambda: asyncio.run(
                        _serve_background(
                            port,
                            stop_when_closed=_script,
                            connect_timeout=_script_connect_timeout,
                            owner_mode=(
                                "transient"
                                if _script
                                else "kernel"
                                if _launch_context.caller_scope is CallerScope.KERNEL
                                else "in_process"
                            ),
                        )
                    ),
                    daemon=not _script,
                    name="arrayview-server",
                ).start()
                if not _server_ready_event.wait(timeout=10.0):
                    raise RuntimeError(
                        f"ArrayView server did not bind port {port} within timeout."
                    )
            else:
                _loading_port = None

            if (
                _launch_context.evidence.is_vscode_remote
                and _launch_plan.display is Display.VSCODE
            ):
                _configure_vscode_port_preview(
                    port, in_vscode=False, is_remote=True
                )
            _platform_mod._jupyter_server_port = port
    except Exception:
        for owned_sid in owned_sids:
            _session_mod.SESSIONS.pop(owned_sid, None)
        raise

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
        from IPython.display import display as _ipy_display

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
            _inline_html = _make_jupyter_proxy_inline_html(
                _inline_url, port, height, _inline_mode_heights
            )
            if n_arrays == 1:
                return _inline_html
            _ipy_display(_inline_html)
            handles = tuple(
                ViewHandle(
                    url_viewer,
                    s,
                    port,
                    _session_mod.SERVER_RUNTIME.instance_id,
                )
                for s in [session.sid] + _compare_sids
            )
            return handles
        iframe = _make_resizable_jupyter_iframe(
            _inline_url, port, height, _inline_mode_heights
        )
        if n_arrays == 1:
            return iframe
        # Multi-array inline: display the IFrame and return a uniform tuple of handles.
        _ipy_display(iframe)
        handles = tuple(
            ViewHandle(
                url_viewer,
                s,
                port,
                _session_mod.SERVER_RUNTIME.instance_id,
            )
            for s in [session.sid] + _compare_sids
        )
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
            wp = _session_mod._window_process
            server_loop = _session_mod.SERVER_LOOP
            window_alive = wp is not None and wp.poll() is None
            notified = False
            native_request_id = uuid.uuid4().hex
            try:
                if server_loop is not None and (window_alive or _server_alive(port)):
                    tab_url = _viewer_path(session.sid)
                    tab_url += (
                        "&native_request_id="
                        + urllib.parse.quote(native_request_id)
                    )
                    future = asyncio.run_coroutine_threadsafe(
                        _server_mod()._notify_shells(
                            session.sid,
                            name,
                            url=tab_url,
                            wait=window_alive,
                        ),
                        server_loop,
                    )
                    try:
                        notified = future.result(timeout=3.0)
                    except Exception:
                        notified = False
                    if notified:
                        notified = _wait_for_native_ready(
                            port,
                            sid=session.sid,
                            native_request_id=native_request_id,
                            expected_server_id=None,
                            expected_server_pid=os.getpid(),
                        )
            except Exception:
                notified = False
            if not notified:
                try:
                    _session_mod._window_process = _open_webview_with_fallback(
                        url_shell,
                        win_w,
                        win_h,
                        shell_port=port,
                        floating=floating,
                        launch_context=_launch_context,
                        fallback_url=_with_loading(url_viewer),
                        title=f"ArrayView: {name}",
                    )
                except Exception:
                    for owned_sid in owned_sids:
                        _session_mod.SESSIONS.pop(owned_sid, None)
                    raise
    elif not _suppress_open:
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
        try:
            _open_browser(
                _with_loading(url_viewer),
                blocking=True,
                force_vscode=_force_vscode,
                title=f"ArrayView: {name}",
                floating=floating,
                launch_context=_launch_context,
                use_fallback=_launch_plan.display is Display.NATIVE,
            )
        except Exception:
            for owned_sid in owned_sids:
                _session_mod.SESSIONS.pop(owned_sid, None)
            raise

    _print_viewer_location(url_viewer, launch_context=_launch_context)
    if n_arrays == 1:
        return ViewHandle(
            url_viewer,
            session.sid,
            port,
            _session_mod.SERVER_RUNTIME.instance_id,
        )
    return tuple(
        ViewHandle(
            url_viewer,
            s,
            port,
            _session_mod.SERVER_RUNTIME.instance_id,
        )
        for s in [session.sid] + _compare_sids
    )


# ── Server Lifecycle ──────────────────────────────────────────────


def _is_script_mode() -> bool:
    """True when running as a plain Python script (not interactive REPL, not Jupyter, not Julia)."""
    if _in_jupyter() or _is_julia_env() or _platform_mod._in_matlab():
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

    connections_before = _sm.VIEWER_CONNECTIONS_SEEN
    deadline = time.monotonic() + connect_timeout
    while (
        _sm.VIEWER_SOCKETS == 0
        and _sm.VIEWER_CONNECTIONS_SEEN == connections_before
    ):
        if not _sm.SESSIONS and not _sm.PENDING_SESSIONS:
            server.should_exit = True
            return
        if time.monotonic() > deadline:
            server.should_exit = True  # no viewer connected; give up
            return
        await asyncio.sleep(0.2)
    # At least one viewer connected, even if it already disconnected between
    # polling ticks. Now wait for all viewer sockets to be closed.
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
    grace_seconds: float = 1.0,
    idle_seconds: float = 0.0,
    connect_timeout: float | None = _CLI_DAEMON_CONNECT_TIMEOUT_SECONDS,
) -> None:
    """Block until all viewer WebSocket connections close.

    Waits for a viewer WebSocket to connect, then all to disconnect, then applies a
    brief grace period so page refreshes don't prematurely kill the server.

    If ``connect_timeout`` is set and no viewer WebSocket ever connects within
    that interval, return so caller-owned daemon processes do not orphan after
    a failed or abandoned launch.

    If ``idle_seconds > 0``, the server stays alive that many extra seconds after
    the grace period so the next ``arrayview`` CLI invocation can reuse it without
    spawning a new subprocess.  A new viewer connection resets the countdown.
    """
    import arrayview._session as _sm

    connections_before = _sm.VIEWER_CONNECTIONS_SEEN
    connect_deadline = (
        None if connect_timeout is None else time.monotonic() + connect_timeout
    )
    while (
        _sm.VIEWER_SOCKETS == 0
        and _sm.VIEWER_CONNECTIONS_SEEN == connections_before
    ):
        if connect_deadline is not None and time.monotonic() >= connect_deadline:
            return
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
            if not _sm.SESSIONS and not _sm.PENDING_SESSIONS:
                return  # an explicit panel/window release ended ownership
            time.sleep(1.0)
        else:
            return  # idle timeout expired — really done


def _view_julia(
    data: "np.ndarray",
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 600,
    mode_heights: dict[str, int] | None = None,
    floating: bool = False,
    launch_context=None,
):
    """Julia-specific view() path: run the server in a subprocess so it is
    completely independent of Julia's GIL.
    """

    force_vscode = (
        launch_context.plan.display.value == "vscode"
        if launch_context is not None
        else _in_vscode_terminal()
    )
    return _view_subprocess(
        data,
        name,
        port,
        window,
        inline=inline,
        height=height,
        mode_heights=mode_heights,
        force_vscode=force_vscode,
        floating=floating,
        launch_context=launch_context,
    )


def _view_subprocess(
    data: "np.ndarray",
    name: str,
    port: int,
    window: bool,
    inline: bool = False,
    height: int = 600,
    mode_heights: dict[str, int] | None = None,
    rgb: bool = False,
    force_vscode: bool = False,
    floating: bool = False,
    launch_context=None,
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
    active_server_id = None
    reuse_existing = (
        launch_context.plan.registration.value == "http_load"
        if launch_context is not None
        else _server_alive(port)
    )
    if reuse_existing:
        if launch_context is not None:
            active_server_id = _planned_server_snapshot(
                launch_context, port
            ).server_instance_id
        else:
            identity = _server_runtime_identity(port)
            active_server_id = identity[0] if identity is not None else None
        native_request_id = (
            uuid.uuid4().hex
            if (
                launch_context is not None
                and launch_context.plan.display.value == "native"
            )
            else None
        )
        # Existing subprocess server — register the new array via /load.
        # Pass notify=True so the server injects a new tab into any open shell
        # window rather than requiring the caller to open a new native window.
        try:
            result = _load_session_from_filepath(
                port,
                tmp_path,
                name,
                notify=(
                    launch_context.plan.display.value == "native"
                    if launch_context is not None
                    else True
                ),
                rgb=rgb,
                expected_server_id=active_server_id,
                native_request_id=native_request_id,
                release_on_disconnect=True,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            # Data is now in server memory; temp file no longer needed.
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            sid = result["sid"]
            tab_injected = bool(result.get("notified", False))
            if tab_injected and native_request_id:
                tab_injected = _wait_for_native_ready(
                    port,
                    sid=str(sid),
                    native_request_id=native_request_id,
                    expected_server_id=active_server_id,
                )
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
        sid = uuid.uuid4().hex
        persist_daemon = bool(
            launch_context is not None
            and launch_context.placement.value == "vscode_remote"
            and launch_context.plan.display.value == "vscode"
        )
        daemon_connect_timeout = (
            _LOCAL_VSCODE_CONNECT_TIMEOUT_SECONDS
            if launch_context is not None
            and launch_context.placement.value == "vscode_local"
            and launch_context.plan.display.value == "vscode"
            else None
        )
        from arrayview._instance_registry import InstanceRegistry

        daemon_proc = None
        try:
            with InstanceRegistry().startup_lock(timeout=20.0):
                if launch_context is not None:
                    port = _revalidate_launch_server(launch_context, port)
                elif _port_in_use(port):
                    port, _already = _find_server_port(port + 1)
                    if _port_in_use(port):
                        raise RuntimeError(
                            f"Port {port} is already in use by another process. "
                            "Choose a different port in view(..., port=...)."
                        )
                    _vprint(
                        f"[ArrayView] Default port busy, using port {port}",
                        flush=True,
                    )
                # Spawn a self-contained server subprocess (same as CLI path).
                script = (
                    f"from arrayview._launcher import _serve_daemon;"
                    f"_serve_daemon({repr(tmp_path)}, {port}, {repr(sid)}, "
                    f"name={repr(name)}, cleanup=True, "
                    f"persist={persist_daemon}, "
                    f"connect_timeout={repr(daemon_connect_timeout)}, rgb={rgb})"
                )
                daemon_proc = subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=sys.platform != "win32",
                )
                if not _wait_for_spawned_server(daemon_proc, port, timeout=15.0):
                    raise RuntimeError(
                        f"The spawned ArrayView server did not claim port {port}."
                    )
                daemon_identity = _server_runtime_identity(port)
                if daemon_identity is None or daemon_identity[0] is None:
                    raise RuntimeError(
                        "The spawned ArrayView server did not publish generation "
                        "identity."
                    )
                active_server_id = daemon_identity[0]
        except Exception:
            _terminate_owned_process(daemon_proc)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    url_viewer = _viewer_url(port, sid)
    url_shell = _shell_url(port, sid, name)
    _print_viewer_location(url_viewer, launch_context=launch_context)

    if (
        launch_context is not None
        and launch_context.plan.display.value == "none"
    ):
        return ViewHandle(url_viewer, sid, port, active_server_id)

    if inline:
        _inline_url = _viewer_url(port, sid, inline=True)
        iframe_html = (
            _build_jupyter_inline_html(
                _inline_url,
                port,
                height,
                mode_heights,
                use_proxy=False,
            )
            if mode_heights
            else (
                f"<iframe src='{_inline_url}' width='100%'"
                f" height='{height}' frameborder='0'></iframe>"
            )
        )
        # IJulia kernel: push HTML through Julia's display stack (routes to Jupyter
        # frontend). Must be a side-effect call, not a return value, because
        # PythonCall would convert a Python IFrame object to an opaque Julia value.
        try:
            import juliacall as _jl

            _jl.Main.seval(f'display("text/html", {json.dumps(iframe_html)})')
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
        return ViewHandle(url_viewer, sid, port, active_server_id)

    can_native = (
        launch_context.plan.display.value == "native"
        if launch_context is not None
        else _can_native_window()
    )
    if window and can_native:
        if not _open_cli_native_shell_after_server(
            port=port,
            sid=sid,
            name=name,
            compare_sids=None,
            win_w=1400,
            win_h=900,
            expected_server_id=active_server_id,
        ):
            _vprint("[ArrayView] Falling back to browser", flush=True)
            _open_browser(
                url_viewer,
                force_vscode=force_vscode,
                blocking=True,
                prefer_system_browser=window and not force_vscode,
                title=f"ArrayView: {name}",
                floating=floating,
                launch_context=launch_context,
                use_fallback=True,
            )
    else:
        # blocking=True when force_vscode so signal file is written before
        # returning to Julia (daemon thread would be killed on process exit).
        _open_browser(
            url_viewer,
            force_vscode=force_vscode,
            blocking=True,
            title=f"ArrayView: {name}",
            floating=floating,
            launch_context=launch_context,
        )
    return ViewHandle(url_viewer, sid, port, active_server_id)


def _serve_empty(port: int) -> None:
    """Background server process with no sessions. Runs until killed.

    Used for ``arrayview --serve`` (pre-warm) and on remote tunnel sessions so
    the port stays alive across multiple tab opens/closes without requiring the
    user to re-run ``--serve`` or re-set port visibility.
    """
    _session_mod.SERVER_PORT = port
    socks = _make_loopback_sockets(port)
    registry, record = _register_server_runtime(port, "persistent")

    def _run_empty_uvicorn():
        config = _uvicorn().Config(
            _server_mod().app,
            log_level="error",
            timeout_keep_alive=30,
        )
        server = _uvicorn().Server(config)
        asyncio.run(server.serve(sockets=socks))

    threading.Thread(target=_run_empty_uvicorn, daemon=True).start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    registry.remove(record.instance_id)
    os._exit(0)


def _serve_daemon(
    filepath: str,
    port: int,
    sid: str,
    name: str = None,
    cleanup: bool = False,
    overlay_filepaths: list | None = None,
    overlay_sids: list | None = None,
    overlay_names: list | None = None,
    compare_filepath: str = None,
    compare_sid: str = None,
    vfield_filepath: str = None,
    vfield_components_dim: int | None = None,
    persist: bool = False,
    connect_timeout: float | None = None,
    rgb: bool = False,
    dir_patterns: list[str] | None = None,
    dir_overlay_specs: list[tuple[str, str]] | None = None,
    dir_case_regex: str | None = None,
    dir_exclude_cases: list[str] | None = None,
    collection_load: str = "lazy",
    collection_stack: str = "auto",
) -> None:
    """Background server process. Loads data, serves it.
    persist=True: never exits (used on remote tunnel so port stays alive).
    persist=False: exits when the UI closes (default, used locally).
    cleanup=True: delete filepath after loading (used when it is a temp file).
    """
    sid_tag = _launch_trace_tag(sid)
    _trace_launch_event(
        "daemon.started",
        port=port,
        owner="persistent" if persist else "transient",
    )
    # Register sid as pending so /metadata can poll while data loads.
    _session_mod.PENDING_SESSIONS.add(sid)
    _pending_event = threading.Event()
    _session_mod.PENDING_SESSION_EVENTS[sid] = _pending_event
    _session_mod.SERVER_PORT = port
    _trace_launch_event("session.pending_declared", sid_tag=sid_tag)

    sock = _make_loopback_sockets(port)
    _trace_launch_event("backend.socket_reserved", port=port)
    registry, record = _register_server_runtime(
        port,
        "persistent" if persist else "transient",
    )
    _trace_launch_event(
        "server.registered",
        instance_tag=_launch_trace_tag(record.instance_id),
        owner=record.owner_mode,
        port=record.port,
    )

    def _run_uvicorn_on_socket():
        config = _uvicorn().Config(
            _server_mod().app,
            log_level="error",
            timeout_keep_alive=30,
        )
        server = _uvicorn().Server(config)
        asyncio.run(server.serve(sockets=sock))

    # Start uvicorn immediately — the window can open before data is ready.
    threading.Thread(
        target=_run_uvicorn_on_socket,
        daemon=True,
    ).start()
    _trace_launch_event("backend.http_thread_started", port=port)

    # Pre-warm colormap LUTs in background (saves ~200 ms on first frame render).
    def _warm_luts():
        try:
            from arrayview._render import _init_luts
            _init_luts()
        except Exception:
            pass
    threading.Thread(target=_warm_luts, daemon=True).start()

    def _load():
        from arrayview._io import (
            load_data,
            load_data_with_meta,
            load_dir_collection,
            list_array_keys,
        )
        from arrayview._session import file_signature

        _trace_launch_event("session.load_started", sid_tag=sid_tag)
        try:
            signature_before_load = (
                file_signature(filepath)
                if dir_patterns is None and not cleanup
                else None
            )
            # Multi-array .npz/.mat: load the first array so the session is
            # created, then store keys so the viewer can show a picker.
            _array_keys = None
            if dir_patterns is not None:
                data, spatial_meta, dir_overlay_items, _summary = load_dir_collection(
                    dir_patterns,
                    overlays=dir_overlay_specs or [],
                    case_regex=dir_case_regex,
                    exclude_cases=dir_exclude_cases,
                    load=collection_load,
                    stack=collection_stack,
                )
                collection_spatial_ndim = len(_summary["spatial_shape"])
            elif (filepath.endswith(".npz") or filepath.endswith(".mat")) and not cleanup:
                try:
                    _array_keys = list_array_keys(filepath)
                except Exception:
                    pass
                _load_key = _array_keys[0]["key"] if _array_keys else None
                data, spatial_meta = load_data_with_meta(filepath, key=_load_key)
                dir_overlay_items = None
                collection_spatial_ndim = None
            else:
                _load_key = None
                data, spatial_meta = load_data_with_meta(
                    filepath,
                    key=_load_key,
                    load=collection_load,
                    stack=collection_stack,
                )
                dir_overlay_items = None
                collection_spatial_ndim = None
            if cleanup:
                try:
                    os.unlink(filepath)
                except Exception:
                    pass
            session = _session_mod.Session(
                data, filepath=None if cleanup else filepath, name=name
            )
            session.sid = sid
            if signature_before_load is not None:
                signature_after_load = file_signature(filepath)
                if signature_before_load == signature_after_load:
                    session.file_signature = signature_after_load
            session.spatial_meta = spatial_meta
            if collection_spatial_ndim is not None:
                session.collection_spatial_ndim = collection_spatial_ndim
            if _array_keys and len(_array_keys) > 1:
                session.array_keys = _array_keys
                session.array_filepath = filepath
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
            if dir_overlay_items is not None:
                for item, ov_sid in zip(dir_overlay_items, overlay_sids or []):
                    ov_session = _session_mod.Session(
                        item["data"], filepath=None, name=item["name"]
                    )
                    ov_session.sid = ov_sid
                    _session_mod.SESSIONS[ov_sid] = ov_session
            resolved_overlay_names = overlay_names or [
                os.path.basename(path) or f"overlay {i + 1}"
                for i, path in enumerate(overlay_filepaths or [])
            ]
            for ov_path, ov_sid, ov_name in zip(
                overlay_filepaths or [], overlay_sids or [], resolved_overlay_names
            ):
                try:
                    ov_data = load_data(ov_path)
                    ov_session = _session_mod.Session(
                        ov_data, filepath=ov_path, name=ov_name
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
            _trace_launch_event("session.ready", sid_tag=sid_tag)
        except Exception as exc:
            _trace_launch_event(
                "session.load_failed",
                sid_tag=sid_tag,
                error_type=type(exc).__name__,
            )
            raise
        finally:
            _session_mod.PENDING_SESSIONS.discard(sid)
            _pending_event.set()
            _session_mod.PENDING_SESSION_EVENTS.pop(sid, None)

    threading.Thread(target=_load, daemon=True).start()

    if persist:
        _wait_for_viewer_close(
            idle_seconds=_PERSIST_DAEMON_IDLE_SECONDS,
            connect_timeout=_PERSIST_DAEMON_CONNECT_TIMEOUT_SECONDS,
        )
    else:
        # Transient CLI launches should stop as soon as the last viewer is
        # really gone, aside from the short grace period inside
        # _wait_for_viewer_close for page refreshes.
        _wait_for_viewer_close(
            idle_seconds=_CLI_DAEMON_IDLE_SECONDS,
            connect_timeout=(
                _CLI_DAEMON_CONNECT_TIMEOUT_SECONDS
                if connect_timeout is None
                else connect_timeout
            ),
        )
    _trace_launch_event(
        "daemon.exiting",
        reason="viewer_lifecycle_complete",
        instance_tag=_launch_trace_tag(record.instance_id),
    )
    registry.remove(record.instance_id)
    _trace_launch_event(
        "server.unregistered",
        instance_tag=_launch_trace_tag(record.instance_id),
    )
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
                        f"http://{_LOOPBACK_HOST}:{port}/reload/{sid}",
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
        elif section == "viewer" and name == "rounded_panes":
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                value = True
            elif lowered in {"false", "0", "no", "off"}:
                value = False
            else:
                print(f"Invalid rounded_panes value: {value}")
                print("  Use true/false, yes/no, on/off, or 1/0")
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


def _normalize_dir_overlay_specs(specs):
    if specs is None:
        return []
    if isinstance(specs, dict):
        specs = list(specs.items())
    elif isinstance(specs, str):
        specs = [specs]

    out = []
    for idx, spec in enumerate(specs, start=1):
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            name, pattern = spec
        elif isinstance(spec, str):
            if "=" in spec:
                name, pattern = spec.split("=", 1)
            else:
                pattern = spec
                import glob as _glob
                from arrayview._io import _strip_array_ext

                inferred = _strip_array_ext(pattern)
                name = (
                    inferred
                    if inferred and not _glob.has_magic(inferred)
                    else f"overlay {idx}"
                )
        else:
            raise ValueError(f"Invalid overlay spec {spec!r}.")
        name = str(name).strip()
        pattern = str(pattern).strip()
        if not name or not pattern:
            raise ValueError(f"Invalid overlay spec {spec!r}.")
        out.append((name, pattern))
    return out


def _normalize_file_overlay_specs(specs):
    """Return ``(name, absolute path)`` pairs for ordinary CLI overlays."""
    from arrayview._io import _strip_array_ext

    out = []
    for idx, spec in enumerate(specs or [], start=1):
        if "=" in spec:
            name, path = spec.split("=", 1)
            name = name.strip()
            path = path.strip()
        else:
            path = spec.strip()
            name = _strip_array_ext(path) or f"overlay {idx}"
        if not name or not path:
            raise ValueError(f"Invalid overlay spec {spec!r}.")
        out.append((name, os.path.abspath(path)))
    return out


def _overlay_specs_from_dirs(directory_patterns, case_regex=None):
    """Discover one sparse overlay role per filename in mask directories."""
    from arrayview._io import (
        _collection_case_key,
        _collection_pattern_paths,
        _strip_array_ext,
    )

    if case_regex:
        grouped = {}
        display_patterns = {}
        for directory_pattern in directory_patterns or []:
            base_pattern = os.path.abspath(directory_pattern)
            for path in _collection_pattern_paths(os.path.join(base_pattern, "*")):
                filename = os.path.basename(path)
                case = _collection_case_key(path, case_regex=case_regex)
                by_case = grouped.setdefault(filename, {})
                if case in by_case:
                    raise ValueError(
                        f"Overlay directory pattern {directory_pattern!r} matched "
                        f"multiple files for case {case!r} and filename {filename!r}."
                    )
                by_case[case] = path
                display_patterns[filename] = os.path.join(base_pattern, filename)
        specs = []
        for filename, by_case in grouped.items():
            specs.append(
                (
                    _strip_array_ext(filename),
                    by_case,
                    True,
                    display_patterns[filename],
                )
            )
        return specs

    specs = []
    seen_names = set()
    for directory_pattern in directory_patterns or []:
        pattern = os.path.join(os.path.abspath(directory_pattern), "*")
        for path in _collection_pattern_paths(pattern):
            filename = os.path.basename(path)
            name = _strip_array_ext(filename)
            if name in seen_names:
                continue
            seen_names.add(name)
            specs.append(
                (
                    name,
                    os.path.join(os.path.abspath(directory_pattern), filename),
                    True,
                )
            )
    return specs


class _CliCollectionScanProgress:
    """Small terminal progress display for slow collection header scans."""

    def __init__(self):
        self.count = 0
        self.started = time.monotonic()
        self.last_update = 0.0
        self.interactive = bool(getattr(sys.stderr, "isatty", lambda: False)())

    def update(self, label, _path):
        self.count += 1
        now = time.monotonic()
        if not self.interactive or now - self.last_update < 0.1:
            return
        elapsed = now - self.started
        print(
            f"\r[ArrayView] Scanning {label}: {self.count} files ({elapsed:.1f}s)",
            end="",
            file=sys.stderr,
            flush=True,
        )
        self.last_update = now

    def finish(self):
        if not self.interactive or not self.count:
            return
        elapsed = time.monotonic() - self.started
        print(
            f"\r[ArrayView] Scanned {self.count} files in {elapsed:.1f}s" + " " * 16,
            file=sys.stderr,
            flush=True,
        )


def _print_dir_collection_summary(summary):
    print("[ArrayView] --stack matched collection:")
    print(f"  cases: {len(summary['cases'])}")
    print(f"  spatial shape: {summary['spatial_shape']}")
    print(f"  image shape: {summary['shape']}")
    for i, pattern in enumerate(summary["base_patterns"], start=1):
        print(f"  image pattern {i}: {pattern}")
    for ov in summary["overlays"]:
        extra = ""
        if ov["ignored_cases"]:
            extra = f" ({len(ov['ignored_cases'])} ignored extra case(s))"
        if ov.get("missing_cases"):
            extra += f" ({len(ov['missing_cases'])} missing case(s), shown empty)"
        print(f"  overlay {ov['name']}: {ov['pattern']}{extra}")
    preview = ", ".join(summary["cases"][:6])
    if len(summary["cases"]) > 6:
        preview += ", ..."
    print(f"  case order: {preview}")


def _confirm_partial_overlay_match(error):
    """Ask an interactive CLI user whether cases without an overlay may be dropped."""
    if not bool(getattr(sys.stdin, "isatty", lambda: False)()):
        return False

    keep_count = error.total_cases - len(error.missing_cases)
    if keep_count <= 0:
        return False

    preview = ", ".join(error.missing_cases[:6])
    if len(error.missing_cases) > 6:
        preview += ", ..."
    print(
        f"[ArrayView] Overlay {error.overlay_name!r} has no mask for "
        f"{len(error.missing_cases)} of {error.total_cases} image cases."
    )
    print(f"  Missing cases: {preview}")
    try:
        answer = input(
            f"Continue with the {keep_count} image cases that have masks? [y/N] "
        )
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


# ── CLI Entry Point (arrayview command) ───────────────────────────


def _instance_public_dict(record) -> dict[str, object]:
    value = record.to_dict()
    value.pop("control_token", None)
    return value


def _handle_management_command(argv: list[str]) -> bool:
    """Handle dependency-light management subcommands before the file CLI."""
    if not argv or argv[0] not in {"doctor", "instances", "stop"}:
        return False
    command = argv[0]
    parser = argparse.ArgumentParser(prog=f"arrayview {command}")
    from arrayview._instance_registry import InstanceRegistry

    if command == "instances":
        parser.add_argument("--json", action="store_true")
        args = parser.parse_args(argv[1:])
        rows = [
            _instance_public_dict(record)
            for record in InstanceRegistry().discover(clean_stale=True)
        ]
        if args.json:
            print(json.dumps({"instances": rows}, indent=2, sort_keys=True))
        elif not rows:
            print("No running ArrayView instances.")
        else:
            for row in rows:
                print(
                    f"{row['instance_id']}  port={row['port']}  "
                    f"pid={row['pid']}  owner={row['owner_mode']}  "
                    f"version={row['package_version']}"
                )
        return True

    if command == "stop":
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("instance_id", nargs="?")
        group.add_argument("--all", action="store_true")
        args = parser.parse_args(argv[1:])
        records = InstanceRegistry().discover(clean_stale=True)
        selected = (
            records
            if args.all
            else [record for record in records if record.instance_id == args.instance_id]
        )
        if not selected:
            parser.error("no matching running ArrayView instance")
        for record in selected:
            message, _pid = _stop_verified_server(record.port)
            print(f"[ArrayView] {record.instance_id}: {message}")
        return True

    parser.add_argument("--json", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--window",
        choices=["native", "browser", "vscode", "inline", "none"],
    )
    args = parser.parse_args(argv[1:])
    from arrayview._launch_plan import (
        Invocation,
        LaunchIntent,
        plan_launch,
        snapshot_launch_environment,
    )

    snapshot = snapshot_launch_environment(
        args.port,
        Invocation.CLI,
        requested_window=args.window,
    )
    plan = plan_launch(
        LaunchIntent(
            Invocation.CLI,
            args.port,
            requested_window=args.window,
            window_explicit=args.window is not None,
        ),
        snapshot,
    )
    rows = [
        _instance_public_dict(record)
        for record in InstanceRegistry().discover(clean_stale=True)
    ]
    snapshot_dict = snapshot.to_dict()
    snapshot_dict["env_vars"] = {
        key: "<redacted>"
        for key in sorted(snapshot_dict.get("env_vars", {}))
    }
    remediation = []
    if plan.failure is not None:
        messages = {
            "invalid_port": "Choose a port between 1 and 65535.",
            "invalid_window": "Choose native, browser, vscode, inline, or none.",
            "vscode_unavailable": "Open from a VS Code terminal or choose browser.",
            "remote_port_conflict": "Stop the recorded instance or choose a free forwarded port.",
        }
        remediation.append(
            {
                "code": plan.failure.value,
                "message": messages[plan.failure.value],
            }
        )
    elif snapshot.server.port_busy and not snapshot.server.arrayview_server_alive:
        remediation.append(
            {
                "code": "foreign_port_occupant",
                "message": (
                    "Choose a different port; ArrayView will not stop this listener."
                ),
            }
        )
    report = {
        "snapshot": snapshot_dict,
        "plan": plan.to_dict(),
        "instances": rows,
        "remediation": remediation,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Environment: {plan.environment.value}")
        print(
            f"Server: {plan.server_owner.value} on port {plan.effective_port} "
            f"({plan.registration.value})"
        )
        print(f"Display: {plan.display.value}")
        print("Reasons: " + (", ".join(plan.reasons) or "default policy"))
        print(f"Instances: {len(rows)}")
        for row in rows:
            print(
                f"  {row['instance_id']} port={row['port']} pid={row['pid']} "
                f"owner={row['owner_mode']} package={row['package_version']} "
                f"protocol={row['protocol_version']}"
            )
        for action in remediation:
            print(f"Remediation [{action['code']}]: {action['message']}")
    return True


def arrayview():
    """Command Line Interface Entry Point."""
    # Handle "arrayview config ..." subcommand before argparse
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        _handle_config_command(sys.argv[2:])
        return
    if _handle_management_command(sys.argv[1:]):
        return

    parser = argparse.ArgumentParser(description="Lightning Fast ND Array Viewer")
    from arrayview import __version__ as _av_version
    parser.add_argument("--version", action="version", version=f"arrayview {_av_version}")
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
        choices=["browser", "vscode", "native", "none"],
        default=None,
        help=(
            "How to open the viewer: browser, vscode, native, or none. "
            "Overrides config (see 'arrayview config')"
        ),
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
        metavar="FILE|NAME=PATTERN",
        action="append",
        default=None,
        help=(
            "Segmentation mask overlay. In file mode, pass a concrete mask file. "
            "In --stack mode, pass NAME=PATTERN to add a named overlay role. "
            "Repeat --overlay to load multiple overlays."
        ),
    )
    parser.add_argument(
        "--overlay-dir",
        metavar="PATTERN",
        action="append",
        default=None,
        help=(
            "In --stack mode, discover mask filenames below matching per-case "
            "directories as sparse overlay roles. Missing masks render as empty. "
            "Case directories are inferred automatically; --case-regex can "
            "override unusual layouts. Repeat to include multiple directories."
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
        "--name",
        default=None,
        dest="array_name",
        help="Display name for the array",
    )
    parser.add_argument(
        "--stack",
        action="store_true",
        dest="stack_mode",
        help=(
            "Treat positional FILE arguments as recursive image patterns for an "
            "aligned collection. Repeated patterns become image channels; "
            "--overlay values become overlay role patterns."
        ),
    )
    parser.add_argument(
        "--series",
        default=None,
        help=(
            "Select a DICOM series by displayed index, SeriesNumber, or exact "
            "SeriesInstanceUID when a source contains multiple series."
        ),
    )
    parser.add_argument(
        "--case-regex",
        default=None,
        dest="case_regex",
        help=(
            "Regex used in --stack mode to extract a case id. Must define "
            "a named (?P<case>...) group."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="In --stack mode, print matched cases and roles without opening a viewer.",
    )
    parser.add_argument(
        "--stack-policy",
        nargs="?",
        const="auto",
        choices=("auto", "dense", "ragged"),
        dest="stack_policy",
        help=(
            "Collection stacking policy for --stack mode: auto picks dense when "
            "shapes agree and ragged otherwise; dense requires matching shapes; "
            "ragged always keeps per-item shapes."
        ),
    )
    parser.add_argument(
        "--load",
        choices=("lazy", "eager"),
        default="lazy",
        help="Directory collection loading policy (default: lazy).",
    )
    args = parser.parse_args()
    _session_mod._verbose = args.verbose
    vfield_components_dim = None

    # --diagnose: print detection results and exit
    if getattr(args, "diagnose", False):
        import json as _json
        import importlib.util as _importlib_util
        from arrayview._launch_plan import (
            Invocation,
            LaunchIntent,
            plan_launch,
            snapshot_launch_environment,
        )
        from arrayview._platform import _find_vscode_ipc_hook

        requested_window = args.window
        snapshot = snapshot_launch_environment(
            args.port,
            Invocation.CLI,
            requested_window=requested_window,
        )
        plan = plan_launch(
            LaunchIntent(
                invocation=Invocation.CLI,
                port=args.port,
                requested_window=requested_window,
                browser=args.browser,
                persistent=args.serve,
            ),
            snapshot,
        )

        def _localhost_candidates() -> list[dict[str, object]]:
            out = []
            try:
                infos = socket.getaddrinfo(
                    _LOOPBACK_HOST,
                    args.port,
                    type=socket.SOCK_STREAM,
                )
            except Exception as exc:
                return [{"error": str(exc)}]
            for family, socktype, proto, _canon, sockaddr in infos:
                out.append(
                    {
                        "family": getattr(socket.AddressFamily(family), "name", str(family)),
                        "socktype": socktype,
                        "proto": proto,
                        "sockaddr": str(sockaddr),
                    }
                )
            return out

        def _bind_probe(port: int) -> dict[str, object]:
            try:
                sock = _make_loopback_socket(port)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
            try:
                return {
                    "ok": True,
                    "family": getattr(socket.AddressFamily(sock.family), "name", str(sock.family)),
                    "sockname": str(sock.getsockname()),
                }
            finally:
                sock.close()

        diag: dict = {
            "snapshot": snapshot.to_dict(),
            "plan": plan.to_dict(),
            "loopback": {
                "host": _LOOPBACK_HOST,
                "getaddrinfo": _localhost_candidates(),
                "bind_probe": _bind_probe(0),
            },
            "native_dependencies": {
                "qtpy": _importlib_util.find_spec("qtpy") is not None,
                "gi": _importlib_util.find_spec("gi") is not None,
                "webview": _importlib_util.find_spec("webview") is not None,
            },
            "vscode": {
                "ipc_hook_recovered": _find_vscode_ipc_hook(),
            },
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "platform": sys.platform,
            "python": sys.executable,
        }
        print(_json.dumps(diag, indent=2))
        return

    dims_override: tuple[int, int] | None = None
    if args.dims:
        dims_override = _parse_dims_spec(args.dims)
        if dims_override is None:
            parser.error(
                f"--dims {args.dims!r} is invalid. "
                "Use e.g. 'x,y,:,:' or ':,:,x,y' or '0,1'."
            )
    if args.dry_run and not args.stack_mode:
        parser.error("--dry-run requires --stack.")
    excluded_cases = set()
    if args.stack_mode:
        if not args.files:
            parser.error("--stack requires at least one image pattern.")
        if args.compare or args.vectorfield or args.watch or args.rgb:
            parser.error(
                "--stack is incompatible with --compare, --vectorfield, "
                "--watch, and --rgb."
            )
        if args.relay:
            parser.error("--stack is incompatible with --relay.")
    if args.stack_policy and not args.stack_mode:
        if len(args.files) != 1:
            parser.error(
                "Directory FILE input requires exactly one FILE argument (a directory)."
            )
        if args.compare or args.overlay or args.vectorfield or args.watch:
            parser.error(
                "Directory FILE input is incompatible with --compare, --overlay, "
                "--vectorfield, and --watch."
            )
        _stack_dir = os.path.abspath(args.files[0])
        if not os.path.isdir(_stack_dir):
            print(f"Error: not a directory: {_stack_dir}")
            sys.exit(1)
        if dims_override is None:
            dims_override = (0, 1)
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
    if args.files and len(args.files) > 6 and not args.stack_mode:
        parser.error(
            "At most six FILE arguments are supported; concat arrays first for larger compare sets."
        )
    if args.compare and len(args.files) > 1 and not args.stack_mode:
        parser.error("Use either positional compare files or --compare, not both.")

    # -- --relay: send array bytes to a remote ArrayView server --
    if args.relay:
        if not args.files:
            parser.error("--relay requires a FILE argument.")
        relay_str = args.relay
        if ":" in relay_str:
            relay_host, relay_port_str = relay_str.rsplit(":", 1)
        else:
            relay_host, relay_port_str = _LOOPBACK_HOST, relay_str
        try:
            relay_port = int(relay_port_str)
        except ValueError:
            parser.error(f"--relay port must be an integer, got: {relay_port_str!r}")
        relay_file = os.path.abspath(args.files[0])
        if not os.path.isfile(relay_file):
            print(f"Error: file not found: {relay_file}")
            sys.exit(1)
        relay_name = os.path.basename(relay_file)
        relay_identity = _server_runtime_identity(relay_port, host=relay_host)
        if relay_identity is None or relay_identity[0] is None:
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
                expected_server_id=relay_identity[0],
            )
        except Exception as e:
            print(f"[ArrayView] Relay failed: {e}", flush=True)
            sys.exit(1)
        return

    # -- --kill: stop the server on the given port --
    if args.kill:
        message, _pid = _stop_verified_server(args.port)
        print(f"[ArrayView] {message}")

        from arrayview._vscode_signal import _cleanup_zombie_registrations
        cleaned = _cleanup_zombie_registrations(verbose=True)
        if cleaned:
            print(f"[ArrayView] Cleaned {cleaned} zombie window registration(s)")
        return

    # -- --serve: start a persistent empty server and exit --
    if args.serve:
        from arrayview._instance_registry import InstanceRegistry
        from arrayview._launch_plan import (
            Invocation,
            LaunchIntent,
            Registration,
            create_launch_context,
        )

        serve_context = create_launch_context(
            LaunchIntent(
                invocation=Invocation.CLI,
                port=args.port,
                requested_window="none",
                persistent=True,
            )
        )
        if not serve_context.plan.ok:
            print(
                f"[ArrayView] Cannot start server: "
                f"{serve_context.plan.failure.value}",
                flush=True,
            )
            sys.exit(1)

        proc = None
        with InstanceRegistry().startup_lock(timeout=20.0):
            args.port = serve_context.plan.effective_port
            if serve_context.plan.registration is Registration.HTTP_LOAD:
                _revalidate_launch_server(serve_context, args.port)
                print(
                    f"[ArrayView] Server already running on port {args.port}. "
                    "Set port to Public in VS Code Ports tab if not done yet, "
                    "then run: arrayview your_file.npy"
                )
                return

            if _port_in_use(args.port):
                if serve_context.evidence.is_vscode_remote:
                    print(
                        f"[ArrayView] Port {args.port} was claimed while starting.\n"
                        f"  Run 'arrayview --kill --port {args.port}' to free it, "
                        "or use --port to specify a different port.",
                        flush=True,
                    )
                    sys.exit(1)
                for candidate in range(args.port + 1, args.port + 101):
                    if not _port_in_use(candidate):
                        args.port = candidate
                        break
                else:
                    print("Error: no free ArrayView server port was found.")
                    sys.exit(1)
                print(
                    f"[ArrayView] Default port busy, using port {args.port}",
                    flush=True,
                )

            # Write VS Code settings before binding, using captured placement.
            _configure_vscode_port_preview(
                args.port,
                in_vscode=serve_context.evidence.in_vscode_terminal,
                is_remote=serve_context.evidence.is_vscode_remote,
            )
            script = (
                "from arrayview._launcher import _serve_empty; "
                f"_serve_empty({args.port})"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=sys.platform != "win32",
            )
            if not _wait_for_spawned_server(proc, args.port, timeout=15.0):
                _terminate_owned_process(proc)
                print(
                    f"Error: the spawned ArrayView server did not claim "
                    f"port {args.port}."
                )
                sys.exit(1)
        print(
            f"\n  \033[1;36m\u2192 ArrayView server started on port {args.port} (PID {proc.pid})\033[0m\n"
            f"\n  Run: arrayview your_file.npy\n"
            f"\n  Server stays running until you kill it (kill {proc.pid}).\n"
        )
        return

    if args.stack_mode:
        dir_patterns = [os.path.abspath(p) for p in args.files]
        if len(dir_patterns) == 1 and os.path.isdir(dir_patterns[0]):
            dir_patterns = [os.path.join(dir_patterns[0], "**", "*")]
        dir_overlay_specs = [
            (role, os.path.abspath(pattern))
            for role, pattern in _normalize_dir_overlay_specs(args.overlay or [])
        ]
        try:
            dir_overlay_specs.extend(
                _overlay_specs_from_dirs(args.overlay_dir, case_regex=args.case_regex)
            )
        except Exception as e:
            print(f"Error: --overlay-dir could not discover masks: {e}")
            sys.exit(1)
        base_file = dir_patterns[0]
        compare_files = []
        name = args.array_name or "dir collection"
        from arrayview._io import MissingOverlayCasesError, load_dir_collection

        excluded_cases = set()
        while True:
            try:
                scan_progress = _CliCollectionScanProgress()
                try:
                    data, spatial_meta, overlay_items, summary = load_dir_collection(
                        dir_patterns,
                        overlays=dir_overlay_specs,
                        case_regex=args.case_regex,
                        load=args.load,
                        stack=args.stack_policy or "auto",
                        scan_progress=scan_progress.update,
                        exclude_cases=excluded_cases,
                    )
                finally:
                    scan_progress.finish()
                break
            except MissingOverlayCasesError as e:
                if not _confirm_partial_overlay_match(e):
                    print(f"Error: --stack could not match collection: {e}")
                    sys.exit(1)
                excluded_cases.update(e.missing_cases)
            except Exception as e:
                print(f"Error: --stack could not match collection: {e}")
                sys.exit(1)
        _print_dir_collection_summary(summary)
        if args.dry_run:
            return
    else:
        if args.overlay_dir:
            parser.error("--overlay-dir requires --stack.")
        try:
            file_overlay_specs = _normalize_file_overlay_specs(args.overlay)
        except ValueError as e:
            parser.error(str(e))
        file_overlay_names = [name for name, _path in file_overlay_specs]
        file_overlay_paths = [path for _name, path in file_overlay_specs]
        dir_patterns = None
        dir_overlay_specs = None
        base_file = os.path.abspath(args.files[0])
        compare_files = [os.path.abspath(p) for p in args.files[1:]]
        if args.compare:
            compare_files.append(os.path.abspath(args.compare))
        data = spatial_meta = overlay_items = summary = None
        name = getattr(args, "_demo_name", None) or os.path.basename(base_file)

    if not args.stack_mode and os.path.isdir(base_file):
        from arrayview._dicom import is_dicom_source, resolve_dicom_series_path

        if is_dicom_source(base_file):
            try:
                base_file = resolve_dicom_series_path(base_file, args.series)
            except ValueError as e:
                parser.error(str(e))
        elif args.series is not None:
            parser.error("--series is only valid for a DICOM source.")
    elif args.series is not None:
        from arrayview._dicom import is_dicom_source, resolve_dicom_series_path

        if not is_dicom_source(base_file):
            parser.error("--series is only valid for a DICOM source.")
        try:
            base_file = resolve_dicom_series_path(base_file, args.series)
        except ValueError as e:
            parser.error(str(e))

    if not args.stack_policy and not args.stack_mode and not os.path.exists(base_file):
        print(f"Error: file not found: {base_file}")
        sys.exit(1)

    if args.vectorfield:
        try:
            from arrayview._io import default_array_key, load_data
            from arrayview._render import _detect_rgb_axis
            from arrayview._vectorfield import _resolve_vfield_layout

            base_data = load_data(base_file, key=default_array_key(base_file))
            image_shape = tuple(int(s) for s in base_data.shape)
            if args.rgb:
                rgb_axis = _detect_rgb_axis(image_shape)
                image_shape = tuple(
                    s for i, s in enumerate(image_shape) if i != rgb_axis
                )
            vf_data = load_data(
                args.vectorfield,
                key=default_array_key(args.vectorfield),
            )
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

    from arrayview._launch_plan import (
        Display,
        Invocation,
        LaunchFailure,
        LaunchIntent,
        Registration,
        create_launch_context,
    )

    launch_context = create_launch_context(
        LaunchIntent(
            invocation=Invocation.CLI,
            port=args.port,
            requested_window=args.window,
            browser=args.browser,
        )
    )
    launch_snapshot = launch_context.evidence
    launch_plan = launch_context.plan
    if os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
        from arrayview._launch_trace import configure_launch_trace

        configure_launch_trace(
            launch_id=launch_context.launch_id,
            role="parent",
        )
        _trace_launch_event(
            "launch.started",
            invocation="cli",
            requested_display=args.window,
            explicit_display=args.window is not None or args.browser,
            environment=launch_plan.environment.value,
            vscode_terminal=launch_snapshot.in_vscode_terminal,
            vscode_remote=launch_snapshot.is_vscode_remote,
            placement=launch_context.placement.value,
            completion_target=launch_context.completion_target.value,
        )
        _trace_launch_event(
            "plan.selected",
            primary_display=launch_plan.display.value,
            fallback_display=(
                launch_plan.fallback_display.value
                if launch_plan.fallback_display is not None
                else None
            ),
            fallback_allowed=launch_plan.fallback_allowed,
            server_owner=launch_plan.server_owner.value,
            registration=launch_plan.registration.value,
            requested_port=launch_plan.requested_port,
            effective_port=launch_plan.effective_port,
            failure=(launch_plan.failure.value if launch_plan.failure else None),
        )
    if args.verbose:
        print(
            "[ArrayView] Launch plan: "
            f"environment={launch_plan.environment.value} "
            f"server={launch_plan.server_owner.value} "
            f"registration={launch_plan.registration.value} "
            f"display={launch_plan.display.value} "
            f"port={launch_plan.requested_port}->{launch_plan.effective_port} "
            f"reasons={','.join(launch_plan.reasons) or 'none'}",
            flush=True,
        )
    if launch_plan.failure is LaunchFailure.VSCODE_UNAVAILABLE:
        print(
            "[ArrayView] --window=vscode requires running from a VS Code integrated terminal.\n"
            "  Use --window=browser to open in your system browser instead.",
            flush=True,
        )
        sys.exit(1)
    if (
        not launch_plan.ok
        and launch_plan.failure is not LaunchFailure.REMOTE_PORT_CONFLICT
    ):
        print(f"Error: invalid launch request ({launch_plan.failure.value}).")
        sys.exit(1)

    is_arrayview_server = launch_snapshot.server.arrayview_server_alive
    execution_registration = launch_plan.registration
    busy_non_arrayview = launch_snapshot.server.port_busy and not is_arrayview_server
    if busy_non_arrayview and _is_ssh and not args.stack_mode:
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
    if launch_plan.failure is LaunchFailure.REMOTE_PORT_CONFLICT:
        print(
            f"[ArrayView] Port {args.port} is in use by another process.\n"
            f"  Run 'arrayview --kill --port {args.port}' to free it, "
            f"or use --port to specify a different port.",
            flush=True,
        )
        sys.exit(1)
    if "scan_from_next_port" in launch_plan.reasons:
        # The plan selected a new daemon. Scan for a free port, but never
        # silently adopt a compatible server that appeared after the snapshot.
        args.port = launch_plan.effective_port
        for candidate in range(args.port, min(65535, args.port + 100) + 1):
            if not _port_in_use(candidate):
                args.port = candidate
                break
        else:
            print("Error: no free ArrayView server port was found.")
            sys.exit(1)
        print(
            f"[ArrayView] Default port busy, using port {args.port}",
            flush=True,
        )
    else:
        args.port = launch_plan.effective_port

    if execution_registration is Registration.HTTP_LOAD:
        args.port = _revalidate_launch_server(launch_context, args.port)

    # Relay detection: if we're connected via SSH and the existing server on
    # this port is actually on a different machine (reverse SSH tunnel), send
    # the array bytes there instead of a filepath the remote server can't access.
    if is_arrayview_server and _is_ssh and not args.stack_mode:
        import socket as _socket

        # Use a generous timeout: _server_hostname also goes through the SSH tunnel.
        _remote_host = _server_hostname(args.port, timeout=3.0)
        if _remote_host and _remote_host != _socket.gethostname():
            try:
                _relay_array_to_server(
                    base_file,
                    args.port,
                    name,
                    args.rgb,
                    expected_server_id=_planned_server_snapshot(
                        launch_context, args.port
                    ).server_instance_id,
                )
            except Exception as e:
                print(f"[ArrayView] Relay failed: {e}", flush=True)
                sys.exit(1)
            return

    window_mode = launch_plan.display.value
    use_native_shell = launch_plan.display is Display.NATIVE
    is_remote = launch_snapshot.is_vscode_remote

    if launch_plan.display is Display.VSCODE:
        # Warn if we can't find the IPC hook (multi-window targeting falls back to PID matching)
        from arrayview._platform import _find_vscode_ipc_hook as _check_ipc_hook

        if not launch_snapshot.is_vscode_remote and not _check_ipc_hook():
            # IPC hook not available — broadcast with focus guard handles this
            _vprint(
                "[ArrayView] No IPC hook; will broadcast to all VS Code windows",
                flush=True,
            )
    if "remote_native_redirected_to_vscode" in launch_plan.reasons:
        _vprint(
            "[ArrayView] --window native is not supported on remote tunnel; using vscode instead."
        )

    if execution_registration is Registration.HTTP_LOAD:
        _handle_cli_existing_server(
            port=args.port,
            base_file=base_file,
            name=name,
            compare_files=compare_files,
            overlay_files=[] if args.stack_mode else file_overlay_paths,
            overlay_names=[] if args.stack_mode else file_overlay_names,
            rgb=args.rgb,
            vectorfield=args.vectorfield,
            vfield_components_dim=vfield_components_dim,
            use_native_shell=use_native_shell,
            dims_override=dims_override,
            watch=getattr(args, "watch", False),
            window_mode=window_mode,
            floating=args.floating,
            is_remote=is_remote,
            launch_context=launch_context,
            dir_patterns=dir_patterns,
            dir_overlay_specs=dir_overlay_specs,
            dir_case_regex=args.case_regex,
            dir_exclude_cases=sorted(excluded_cases),
            collection_load=args.load,
            collection_stack=args.stack_policy or "auto",
        )
        return

    demo_name = getattr(args, "_demo_name", None)
    demo_cleanup = getattr(args, "_demo_cleanup", False)
    _handle_cli_spawned_daemon(
        port=args.port,
        base_file=base_file,
        name=name,
        compare_files=compare_files,
        overlay_files=[] if args.stack_mode else file_overlay_paths,
        overlay_names=[] if args.stack_mode else file_overlay_names,
        dims_override=dims_override,
        use_native_shell=use_native_shell,
        watch=getattr(args, "watch", False),
        window_mode=window_mode,
        floating=args.floating,
        is_remote=is_remote,
        launch_context=launch_context,
        vectorfield=args.vectorfield,
        vfield_components_dim=vfield_components_dim,
        rgb=args.rgb,
        demo_name=demo_name,
        demo_cleanup=demo_cleanup,
        dir_patterns=dir_patterns,
        dir_overlay_specs=dir_overlay_specs,
        dir_case_regex=args.case_regex,
        dir_exclude_cases=sorted(excluded_cases),
        collection_load=args.load,
        collection_stack=args.stack_policy or "auto",
    )
