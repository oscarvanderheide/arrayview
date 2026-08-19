"""VS Code signal-file IPC — window targeting, payload writing, and ACKs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from arrayview._session import _vprint
from arrayview._platform import (
    _exact_vscode_window_registration,
    _find_vscode_ipc_hook,
    _is_vscode_remote,
    get_ppid,
)

_VSCODE_SIGNAL_FILENAME = "open-request-v0900.json"
_VSCODE_COMPAT_SIGNAL_FILENAMES: tuple[str, ...] = ("open-request-v0800.json",)
_VSCODE_PORT_SETTINGS_SETTLE_SECONDS = 2.0
_VSCODE_SIGNAL_MAX_AGE_MS = (
    240_000  # survive one slow large-file request ahead in the disk-backed queue
)
_VSCODE_ACK_PROTOCOL_VERSION = 1
_VSCODE_ACK_FILENAME_PREFIX = "open-ack-v0100-"


class AckState(str, Enum):
    """Progress and terminal states reported by the VS Code opener extension."""

    CLAIMED = "claimed"
    PORT_RESOLVED = "port_resolved"
    VISIBILITY_VERIFIED = "visibility_verified"
    PANEL_OPENED = "panel_opened"
    BACKEND_READY = "backend_ready"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID = "invalid"


@dataclass(frozen=True)
class SignalRequest:
    """Metadata for one non-blocking VS Code open request."""

    request_id: str
    window_id: str | None
    server_id: str | None
    ack_path: Path
    written: bool
    extension_version: str | None = None
    failure_reason: str | None = None

    def __bool__(self) -> bool:
        return self.written


@dataclass(frozen=True)
class AckResult:
    """A validated acknowledgement, or a local terminal wait result."""

    state: AckState
    request_id: str
    window_id: str | None = None
    server_id: str | None = None
    message: str | None = None
    payload: dict | None = None


def _vscode_ack_path(request_id: str) -> Path:
    """Return the per-request ACK path (the extension writes it atomically)."""
    return Path(os.path.expanduser("~/.arrayview")) / (
        f"{_VSCODE_ACK_FILENAME_PREFIX}{request_id}.json"
    )


def _public_tunnel_base(
    port: int,
    expected_server_id: str | None = None,
    timeout: float = 45.0,
) -> tuple[str | None, str | None]:
    """Return ``(base_url, reason)`` for a public route to *port*.

    Exactly one is set: a base URL the VS Code client can load, or the reason
    it could not be published. The reason is VS Code's own wording where it
    gave one, because it names the actual obstacle (usually the tunnel's
    forwarded-port limit) and what to do about it.

    Only inline notebook viewers need this. Their iframe renders in a webview
    on the VS Code client, which has no route to this host's localhost and no
    devtunnel cookie, so a private forward answers it with a sign-in redirect
    and the cell stays black. The viewer tab is unaffected: VS Code fetches
    that page itself, on this side, over the private route.

    Asking costs one file write and a poll; the extension does the forwarding.
    """
    signal_dir = Path(os.path.expanduser("~/.arrayview"))
    request_id = uuid.uuid4().hex
    request_path = signal_dir / f"public-port-request-{request_id}.json"
    ack_path = signal_dir / f"public-port-ack-{request_id}.json"
    payload = {
        "port": int(port),
        "requestId": request_id,
        "expectedServerId": expected_server_id,
        "sentAtMs": int(time.time() * 1000),
    }
    try:
        signal_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = request_path.with_suffix(f".tmp-{os.getpid()}")
        tmp_path.write_text(json.dumps(payload))
        os.replace(tmp_path, request_path)
    except OSError as exc:
        return None, f"could not ask VS Code to publish port {port}: {exc}"

    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            try:
                ack = json.loads(ack_path.read_text())
            except (OSError, ValueError):
                time.sleep(0.25)
                continue
            base = ack.get("externalBase")
            if ack.get("ok") and isinstance(base, str) and base:
                return base.rstrip("/"), None
            return None, str(ack.get("error") or "VS Code gave no reason")
        return None, (
            f"VS Code did not answer about port {port} within {timeout:.0f}s"
        )
    finally:
        for path in (request_path, ack_path):
            try:
                path.unlink()
            except OSError:
                pass


def _cleanup_stale_vscode_acks(max_age_seconds: float = 300.0) -> int:
    """Remove ACK files older than *max_age_seconds*."""
    signal_dir = Path(os.path.expanduser("~/.arrayview"))
    try:
        candidates = tuple(signal_dir.glob(f"{_VSCODE_ACK_FILENAME_PREFIX}*.json"))
    except OSError:
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    for path in candidates:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except (FileNotFoundError, OSError):
            continue
    return removed


def _wait_for_vscode_ack(
    request: SignalRequest,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> AckResult:
    """Wait boundedly for a correlated terminal extension ACK."""
    deadline = time.monotonic() + max(0.0, timeout)
    last_result: AckResult | None = None
    while True:
        try:
            raw = request.ack_path.read_text()
        except FileNotFoundError:
            raw = None
        except OSError as exc:
            return AckResult(AckState.INVALID, request.request_id, message=str(exc))

        if raw is not None:
            try:
                payload = json.loads(raw)
                state = AckState(payload["state"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                return AckResult(
                    AckState.INVALID, request.request_id, message=str(exc)
                )
            correlations = (
                ("requestId", request.request_id),
                ("windowId", request.window_id),
                ("serverId", request.server_id),
            )
            for key, expected in correlations:
                if expected is not None and payload.get(key) != expected:
                    return AckResult(
                        AckState.INVALID,
                        request.request_id,
                        message=f"ACK {key} does not match request",
                        payload=payload,
                    )
            if request.extension_version is not None:
                try:
                    actual_version = tuple(
                        int(part) for part in payload["extensionVersion"].split(".")
                    )
                    required_version = tuple(
                        int(part) for part in request.extension_version.split(".")
                    )
                except (AttributeError, KeyError, TypeError, ValueError):
                    return AckResult(
                        AckState.INVALID,
                        request.request_id,
                        message="ACK extensionVersion is missing or invalid",
                        payload=payload,
                    )
                if actual_version < required_version:
                    return AckResult(
                        AckState.INVALID,
                        request.request_id,
                        message="ACK extensionVersion is older than required",
                        payload=payload,
                    )
            result = AckResult(
                state,
                request.request_id,
                payload.get("windowId"),
                payload.get("serverId"),
                payload.get("message"),
                payload,
            )
            if state in {AckState.BACKEND_READY, AckState.FAILED}:
                return result
            last_result = result

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return AckResult(
                AckState.TIMEOUT,
                request.request_id,
                message=(
                    f"last extension state: {last_result.state.value}"
                    if last_result is not None
                    else None
                ),
                payload=last_result.payload if last_result is not None else None,
            )
        time.sleep(min(max(0.001, poll_interval), remaining))

def _find_current_vscode_window_id() -> str | None:
    """Find the current VS Code window by matching ancestor PIDs.

    On Linux, VS Code spawns per-window renderer processes. The extension host
    and the terminal's PTY host are both children of the same renderer, giving
    them a common ancestor that distinguishes them from other windows. We find
    the window whose extension host shares the closest (deepest) common ancestor
    with our own process tree.

    NOTE: On macOS this does NOT work reliably — the extension host
    (Code Helper (Plugin)) and the terminal PTY host (Code Helper) are
    both direct children of the single main Electron process, not of
    per-window renderers. All windows tie on ancestor depth. This function
    returns None on macOS with 2+ windows, letting the caller fall back to
    broadcast with focus guard.

    Returns the window_id (PID string or hookTag string) or None.
    """
    signal_dir = os.path.expanduser("~/.arrayview")
    if not os.path.isdir(signal_dir):
        return None

    # Load all window registrations
    windows = []  # list of (window_id, ext_pid, ext_ppids)
    try:
        for filename in os.listdir(signal_dir):
            if not (filename.startswith("window-") and filename.endswith(".json")):
                continue
            try:
                with open(os.path.join(signal_dir, filename)) as f:
                    data = json.load(f)
                window_id = filename[7:-5]
                ext_pid = data.get("pid")
                ext_ppids = data.get("ppids", [])
                windows.append((window_id, ext_pid, ext_ppids))
            except Exception:
                continue
    except Exception:
        pass

    if not windows:
        return None

    if len(windows) == 1:
        return windows[0][0]

    # On macOS, PID ancestry matching is unreliable with multiple windows:
    # extension hosts and PTY hosts are all direct children of the main
    # Electron process, so scoring always ties.  Return None and let the
    # caller fall back to broadcast with focus guard.
    if sys.platform == "darwin" and len(windows) > 1:
        _vprint(
            f"[ArrayView] window-match: macOS with {len(windows)} windows, "
            f"skipping PID ancestry (ext hosts share same Electron parent)",
            flush=True,
        )
        return None

    # Collect our ancestor PIDs with depth (depth=1 means direct parent)
    our_ancestors: dict[int, int] = {}
    cur = os.getpid()
    for depth in range(1, 25):
        p = get_ppid(cur)
        if p <= 1 or p in our_ancestors:
            break
        our_ancestors[p] = depth
        cur = p

    _vprint(f"[ArrayView] window-match: pid={os.getpid()} ancestors={our_ancestors}", flush=True)
    for window_id, ext_pid, ext_ppids in windows:
        _vprint(f"[ArrayView] window-match: window={window_id} ext_pid={ext_pid} ext_ppids={ext_ppids}", flush=True)

    # Score each window: find the common ancestor with the smallest combined
    # depth (ext_depth + our_depth). Lower = more closely related = this window.
    best_window: str | None = None
    best_score = float("inf")

    for window_id, ext_pid, ext_ppids in windows:
        # Extension host itself is a direct ancestor of our process (rare but possible)
        if ext_pid and ext_pid in our_ancestors:
            _vprint(f"[ArrayView] window-match: ext_pid {ext_pid} is our ancestor → window={window_id}", flush=True)
            return window_id

        # Find the closest common ancestor via the extension's recorded ppids
        for ext_depth, anc_pid in enumerate(ext_ppids, 1):
            if anc_pid in our_ancestors:
                our_depth = our_ancestors[anc_pid]
                score = ext_depth + our_depth
                _vprint(f"[ArrayView] window-match: window={window_id} common_ancestor={anc_pid} ext_depth={ext_depth} our_depth={our_depth} score={score}", flush=True)
                if score < best_score:
                    best_score = score
                    best_window = window_id
                break  # take the shallowest match per window

    if best_window:
        _vprint(f"[ArrayView] window-match: WINNER={best_window} score={best_score}", flush=True)
        return best_window

    return None


def _find_arrayview_window_id() -> str | None:
    """Find ARRAYVIEW_WINDOW_ID from own env or ancestor processes.

    The VS Code extension injects this env var into all terminals via
    EnvironmentVariableCollection.  It identifies which VS Code window
    the terminal belongs to.  ``uv run`` and similar launchers may strip
    env vars, so we also walk the process tree (same as _find_vscode_ipc_hook).
    """
    # Direct env first
    val = os.environ.get("ARRAYVIEW_WINDOW_ID", "")
    if val:
        return val

    pid = os.getpid()
    for _ in range(20):
        pid = get_ppid(pid)
        if pid <= 1:
            break
        # Linux: /proc/<pid>/environ (null-separated KEY=VALUE pairs)
        try:
            with open(f"/proc/{pid}/environ", "rb") as fh:
                for entry in fh.read().split(b"\0"):
                    if entry.startswith(b"ARRAYVIEW_WINDOW_ID="):
                        return entry[len(b"ARRAYVIEW_WINDOW_ID="):].decode()
        except Exception:
            pass
        # macOS: ps ewwww (env vars appended after command arguments)
        try:
            r = subprocess.run(
                ["ps", "ewwww", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=3,
            )
            for token in r.stdout.split():
                if token.startswith("ARRAYVIEW_WINDOW_ID="):
                    return token[len("ARRAYVIEW_WINDOW_ID="):]
        except Exception:
            pass

    # tmux: process tree is detached from VS Code terminal; check all session clients.
    # Only accept the fallback if every attached client for this tmux session
    # reports the same window ID. Returning the first client is unsafe when the
    # same session is attached from multiple VS Code windows.
    if os.environ.get("TERM_PROGRAM") == "tmux":
        tmux_window_ids: set[str] = set()
        try:
            r_sid = subprocess.run(
                ["tmux", "display-message", "-p", "#{session_id}"],
                capture_output=True, text=True, timeout=2,
            )
            session_id = r_sid.stdout.strip()
            if session_id:
                r_clients = subprocess.run(
                    ["tmux", "list-clients", "-t", session_id, "-F", "#{client_pid}"],
                    capture_output=True, text=True, timeout=2,
                )
                for line in r_clients.stdout.strip().splitlines():
                    try:
                        client_pid = int(line.strip())
                    except ValueError:
                        continue
                    if client_pid <= 1:
                        continue
                    # Linux: /proc/<pid>/environ
                    try:
                        with open(f"/proc/{client_pid}/environ", "rb") as fh:
                            for entry in fh.read().split(b"\0"):
                                if entry.startswith(b"ARRAYVIEW_WINDOW_ID="):
                                    val = entry[len(b"ARRAYVIEW_WINDOW_ID="):].decode()
                                    if val:
                                        tmux_window_ids.add(val)
                    except Exception:
                        pass
                    # macOS: ps ewwww
                    try:
                        r = subprocess.run(
                            ["ps", "ewwww", "-p", str(client_pid)],
                            capture_output=True, text=True, timeout=3,
                        )
                        for token in r.stdout.split():
                            if token.startswith("ARRAYVIEW_WINDOW_ID="):
                                val = token[len("ARRAYVIEW_WINDOW_ID="):]
                                if val:
                                    tmux_window_ids.add(val)
                    except Exception:
                        pass
        except Exception:
            pass
        if len(tmux_window_ids) == 1:
            return next(iter(tmux_window_ids))
        if len(tmux_window_ids) > 1:
            _vprint(
                "[ArrayView] signal: tmux clients report multiple "
                f"ARRAYVIEW_WINDOW_ID values: {sorted(tmux_window_ids)}",
                flush=True,
            )

    return None


def _session_name_for_url(url: str) -> str | None:
    """Return the session name behind *url*'s ``?sid=``, or None."""
    try:
        import urllib.parse as _up

        sid = _up.parse_qs(_up.urlparse(url).query).get("sid", [None])[0]
        if not sid:
            return None
        import arrayview._session as _sm

        session = _sm.SESSIONS.get(sid)
        return getattr(session, "name", None) if session else None
    except Exception:
        return None


def _with_viewer_name(url: str, name: str | None) -> str:
    """Add ``av_name`` so the viewer can title its tab before metadata lands.

    The page otherwise shows the bare host until the WebSocket delivers the
    session name, and that WebSocket waits on the array load — seconds for a
    large file.  This stays a query parameter rather than being baked into the
    served HTML because the gzip cache in ``_server`` is keyed on the page
    body; a per-session body would evict on every open and re-compress ~2 MB.
    """
    if not name:
        return url
    try:
        import urllib.parse as _up

        parts = _up.urlsplit(url)
        query = _up.parse_qsl(parts.query, keep_blank_values=True)
        if any(key == "av_name" for key, _ in query):
            return url
        query.append(("av_name", str(name)))
        return _up.urlunsplit(parts._replace(query=_up.urlencode(query)))
    except Exception:
        return url


def _viewer_port_url(
    url: str,
    *,
    request_id: str,
    max_age_ms: int,
    server_id: str | None,
) -> str:
    """Acquire the private tunnel viewer port for one correlated launch.

    Measured 2026-08-06, real host: a forward VS Code has just created never
    loses the viewer's page request (3/3 on genuine cold starts), while one that
    already exists and has been idle for tens of seconds drops the first two to
    four.  A dropped request never arrives, so the opener can only throw the tab
    away and open another — the flicker.

    The backend holds the selected port for this request until its correlated
    viewer connects or the request expires.  That lease prevents the last old
    viewer from closing the port during the handoff.  Failure is explicit: the
    known-stale main port is not a safe fallback for a tunnel viewer.
    """
    import json as _json
    import urllib.request
    from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

    try:
        parts = urlsplit(url)
        # A cold-start launch is handed the loading-page URL (see
        # _with_loading): that stub answers every path, /ping included, with
        # its placeholder HTML rather than proxying to the real backend, so
        # probing it directly always looks like a non-ArrayView response.
        # Unwrap to the real target for the probe and re-wrap the leased
        # result in the same stub so the loading redirect still works.
        loading_wrapper = None
        target = parse_qs(parts.query).get("target")
        if parts.path == "/" and target:
            loading_wrapper = (parts.scheme, parts.netloc)
            parts = urlsplit(target[0])
        if parts.hostname != "localhost" or not parts.port:
            raise RuntimeError("the backend URL is not a private localhost route")
        origin = f"http://localhost:{parts.port}"
        with urllib.request.urlopen(f"{origin}/ping", timeout=1.0) as response:
            status = _json.load(response)
        if status.get("service") != "arrayview":
            raise RuntimeError("the target port is not an ArrayView server")
        body = _json.dumps({
            "requestId": request_id,
            "ttlMs": max_age_ms,
            "expectedServerId": server_id,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{origin}/cold-start-port",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=3.0) as response:
            lease = _json.load(response)
        if lease.get("requestId") != request_id:
            raise RuntimeError(
                "the running server does not support safe viewer handoff"
            )
        port = lease.get("port")
        if not port:
            raise RuntimeError("the server did not provide a viewer connection")
        _vprint(f"[ArrayView] serving this launch from port {port}")
        leased = urlunsplit(
            (parts.scheme, f"localhost:{port}", parts.path, parts.query, parts.fragment)
        )
        if loading_wrapper is not None:
            leased = urlunsplit(
                (
                    loading_wrapper[0],
                    loading_wrapper[1],
                    "/",
                    f"target={quote(leased, safe='')}",
                    "",
                )
            )
        return leased
    except Exception as exc:
        _vprint(f"[ArrayView] private viewer connection unavailable: {exc}")
        if "does not support safe viewer handoff" in str(exc):
            raise RuntimeError(
                "ArrayView needs to restart once before it can open this viewer "
                "safely; run `arrayview --kill`, then retry"
            ) from exc
        raise RuntimeError(
            "ArrayView could not prepare the private VS Code viewer connection; "
            "no display request was sent"
        ) from exc


def _open_via_signal_file(
    url: str,
    delay: float = 0.0,
    title: str | None = None,
    floating: bool = False,
    *,
    server_id: str | None = None,
    window_id: str | None = None,
    is_remote: bool | None = None,
    max_age_ms: int | None = None,
    display_surface: str | None = None,
    cold_start_port: bool = False,
) -> SignalRequest:
    """Write the URL to the versioned ArrayView opener signal file.

    *title* is shown as the VS Code tab label (e.g. "ArrayView: sample.npy").
    If omitted it is auto-derived from the session name embedded in the URL's
    ``?sid=`` query parameter.
    """
    effective_max_age_ms = (
        max(1, int(max_age_ms))
        if max_age_ms is not None
        else _VSCODE_SIGNAL_MAX_AGE_MS
    )
    request_id = uuid.uuid4().hex
    ack_path = _vscode_ack_path(request_id)
    if cold_start_port:
        try:
            url = _viewer_port_url(
                url,
                request_id=request_id,
                max_age_ms=effective_max_age_ms,
                server_id=server_id,
            )
        except RuntimeError as exc:
            from arrayview._vscode_extension import _VSCODE_EXT_VERSION

            detail = str(exc)
            if exc.__cause__ is not None:
                detail = f"{detail} ({exc.__cause__!r})"
            return SignalRequest(
                request_id,
                window_id,
                server_id,
                ack_path,
                False,
                _VSCODE_EXT_VERSION,
                detail,
            )
    session_name = _session_name_for_url(url)
    if title is None and session_name:
        title = f"ArrayView: {session_name}"
    # The viewer titles its own tab from this, so it must be the session name
    # the metadata will later carry — not *title*, which some callers pass
    # pre-formatted and others pass as a bare filename.
    url = _with_viewer_name(url, session_name)

    if window_id is None:
        window_id = _find_arrayview_window_id()
    _cleanup_stale_vscode_acks()
    try:
        ack_path.unlink()
    except FileNotFoundError:
        pass
    payload: dict = {
        "action": "open-preview",
        "url": url,
        "maxAgeMs": effective_max_age_ms,
        "protocolVersion": _VSCODE_ACK_PROTOCOL_VERSION,
        "requestId": request_id,
        "ackPath": str(ack_path),
    }
    from arrayview._vscode_extension import _VSCODE_EXT_VERSION
    payload["requiredExtensionVersion"] = _VSCODE_EXT_VERSION
    remote_was_supplied = is_remote is not None
    if is_remote is None:
        is_remote = _is_vscode_remote()
    if is_remote:
        payload["remoteOnly"] = True
    if window_id:
        payload["windowId"] = window_id
    if server_id:
        payload["serverId"] = server_id
    if display_surface:
        payload["displaySurface"] = display_surface
    if title:
        payload["title"] = title
    if handoff_path := os.environ.get("ARRAYVIEW_HANDOFF_PATH"):
        payload["handoffPath"] = handoff_path
    # Opt-in measurement launch: the opener observes the navigation for a long
    # budget instead of closing and reopening a blank tab after 1.5 s.  Only
    # this launch is affected, so ordinary use continues alongside a run.
    if os.environ.get("ARRAYVIEW_MEASURE_NAVIGATION") == "1":
        payload["measureNavigation"] = True
    if floating:
        payload["floating"] = True
    if remote_was_supplied:
        written = _write_vscode_signal(payload, delay=delay, is_remote=is_remote)
    else:
        written = _write_vscode_signal(payload, delay=delay)
    resolved_window_id = payload.get("windowId")
    return SignalRequest(
        request_id,
        resolved_window_id if isinstance(resolved_window_id, str) else None,
        server_id,
        ack_path,
        written,
        _VSCODE_EXT_VERSION,
    )


def _write_vscode_signal(
    payload: dict,
    delay: float = 0.0,
    skip_compat: bool = False,
    *,
    is_remote: bool | None = None,
) -> bool:
    """Write a versioned control payload for the VS Code opener extension.

    Signal-file targeting strategy (in priority order):

    1. ARRAYVIEW_WINDOW_ID (all platforms):
       The extension injects this env var into every terminal.  It uniquely
       identifies the VS Code window even on macOS where IPC hook recovery
       fails.  If found and the window is still registered, we write directly
       to that window's targeted signal file.

    2. LOCAL VS Code (hookTag):
       When VSCODE_IPC_HOOK_CLI can be found, compute hookTag and write to
       ``~/.arrayview/open-request-ipc-<hookTag>.json``.

    3. LOCAL VS Code (PID ancestry):
       Match ancestor PIDs to find which window spawned this terminal.

    4. REMOTE / TUNNEL:
       Fall back to ARRAYVIEW_WINDOW_ID → PID ancestry → shared file.

    Falls back to the shared signal file when no targeting succeeds.
    """
    import hashlib
    from arrayview._platform import _find_vscode_ipc_hook

    if is_remote is None:
        is_remote = _is_vscode_remote()

    signal_dir = os.path.expanduser("~/.arrayview")
    try:
        os.makedirs(signal_dir, exist_ok=True)
        if delay > 0:
            time.sleep(delay)
        data = dict(payload)
        data.setdefault("sentAtMs", int(time.time() * 1000))
        data.setdefault("maxAgeMs", _VSCODE_SIGNAL_MAX_AGE_MS)
        data.setdefault("requestId", uuid.uuid4().hex)

        # --- Primary: ARRAYVIEW_WINDOW_ID (all platforms) ---
        # The extension injects this env var into every terminal via
        # EnvironmentVariableCollection.  It uniquely identifies the VS Code
        # window even on macOS where IPC hook recovery fails.
        env_wid = _find_arrayview_window_id()
        current_ipc_hook = _find_vscode_ipc_hook()
        exact_registration = _exact_vscode_window_registration(current_ipc_hook)
        if exact_registration is not None:
            exact_wid = exact_registration[0]
            exact_remote_name = exact_registration[1].get("remoteName")
            if isinstance(exact_remote_name, str) and exact_remote_name:
                data["targetRemoteName"] = exact_remote_name
            if env_wid and env_wid != exact_wid:
                _vprint(
                    "[ArrayView] signal: ignoring stale ARRAYVIEW_WINDOW_ID="
                    f"{env_wid}; exact server registration is {exact_wid}",
                    flush=True,
                )
            env_wid = exact_wid
        if current_ipc_hook:
            current_ipc_wid = hashlib.sha256(
                current_ipc_hook.encode()
            ).hexdigest()[:16]
            current_ipc_registration = os.path.join(
                signal_dir, f"window-{current_ipc_wid}.json"
            )
            try:
                with open(current_ipc_registration) as registration_file:
                    current_ipc_data = json.load(registration_file)
                from arrayview._platform import _process_is_alive

                if (
                    _process_is_alive(current_ipc_data.get("pid"))
                    and current_ipc_data.get("hookTag") == current_ipc_wid
                ):
                    if env_wid and env_wid != current_ipc_wid:
                        _vprint(
                            "[ArrayView] signal: ignoring stale ARRAYVIEW_WINDOW_ID="
                            f"{env_wid}; live IPC registration is {current_ipc_wid}",
                            flush=True,
                        )
                    env_wid = current_ipc_wid
            except (OSError, TypeError, ValueError):
                pass
        targeted_via_env = False
        if env_wid:
            reg_file = os.path.join(signal_dir, f"window-{env_wid}.json")
            if os.path.isfile(reg_file):
                try:
                    with open(reg_file) as _rf:
                        _reg_data = json.load(_rf)
                    uses_pid = _reg_data.get("fallbackId", False)
                except Exception:
                    uses_pid = env_wid.isdigit()

                # Trust an exact window ID. Redirecting to a newer same-parent
                # registration can silently open the tab in a live sibling window.
                _prefix = "pid" if uses_pid else "ipc"
                filenames = (f"open-request-{_prefix}-{env_wid}.json",)
                targeted_via_env = True
                _vprint(
                    f"[ArrayView] signal: ARRAYVIEW_WINDOW_ID → {filenames[0]}",
                    flush=True,
                )
            else:
                # Registration file missing: extension host likely restarted.
                # Task 2 (extension.js) makes the ID stable, so this should be
                # rare.  Smart fallback based on how many windows are registered.
                _vprint(
                    f"[ArrayView] signal: ARRAYVIEW_WINDOW_ID={env_wid} registration "
                    f"missing (extension likely restarted), applying smart fallback",
                    flush=True,
                )
                try:
                    _all_windows = [
                        fn for fn in os.listdir(signal_dir)
                        if fn.startswith("window-") and fn.endswith(".json")
                    ]
                except Exception:
                    _all_windows = []

                if len(_all_windows) == 1:
                    _sole_wid = _all_windows[0][7:-5]
                    try:
                        with open(os.path.join(signal_dir, _all_windows[0])) as _rf:
                            _sole_reg = json.load(_rf)
                        _uses_pid_sole = _sole_reg.get("fallbackId", False)
                    except Exception:
                        _uses_pid_sole = _sole_wid.isdigit()
                    _prefix = "pid" if _uses_pid_sole else "ipc"
                    filenames = (f"open-request-{_prefix}-{_sole_wid}.json",)
                    targeted_via_env = True
                    _vprint(
                        f"[ArrayView] signal: single window registered → {filenames[0]}",
                        flush=True,
                    )
                elif len(_all_windows) > 1:
                    print(
                        "[ArrayView] VS Code window is ambiguous: "
                        f"ARRAYVIEW_WINDOW_ID={env_wid!r} is not registered "
                        f"and {len(_all_windows)} VS Code windows are active. "
                        "Open a fresh terminal in the target VS Code window "
                        "and run ArrayView again.",
                        flush=True,
                    )
                    return False
                # else: 0 windows — fall through to subsequent targeting

        if targeted_via_env:
            pass  # filenames already set — skip to write logic below
        elif (ipc_hook := _find_vscode_ipc_hook()) and not is_remote:
            own_tag = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
            data["hookTag"] = own_tag

            # On macOS, the extension host can't walk up to find VSCODE_IPC_HOOK_CLI
            # (it's not in the extension host's process ancestry), so extensions
            # register in PID mode (fallbackId=true, hookTag="").  If no registered
            # extension has our hookTag, fall through to PID-based targeting.
            ext_uses_hook = False
            try:
                for _fname in os.listdir(signal_dir):
                    if _fname.startswith("window-") and _fname.endswith(".json"):
                        with open(os.path.join(signal_dir, _fname)) as _f:
                            _reg = json.load(_f)
                        if _reg.get("hookTag") == own_tag:
                            ext_uses_hook = True
                            break
            except Exception:
                pass

            if ext_uses_hook:
                # Extension found its own hook: use hookTag-based targeted file.
                filenames: tuple[str, ...] = (f"open-request-ipc-{own_tag}.json",)
            else:
                # Extension is in PID mode: find it via ancestor matching and write
                # the pid-based file the extension is actually watching.
                window_id = _find_current_vscode_window_id()
                if window_id:
                    filenames = (
                        f"open-request-pid-{window_id}.json"
                        if window_id.isdigit()
                        else f"open-request-ipc-{window_id}.json",
                    )
                else:
                    from arrayview._platform import _in_vscode_terminal as _in_vsc_t

                    if _in_vsc_t():
                        _wfiles: list[str] = []
                        try:
                            for _fn in os.listdir(signal_dir):
                                if _fn.startswith("window-") and _fn.endswith(".json"):
                                    _wid = _fn[7:-5]
                                    _wfiles.append(
                                        f"open-request-pid-{_wid}.json"
                                        if _wid.isdigit()
                                        else f"open-request-ipc-{_wid}.json"
                                    )
                        except Exception:
                            pass
                        if len(_wfiles) > 1:
                            print(
                                "[ArrayView] VS Code window is ambiguous: "
                                "no exact ARRAYVIEW_WINDOW_ID or PID match is available "
                                f"and {len(_wfiles)} VS Code windows are active. "
                                "Open a fresh terminal in the target VS Code window "
                                "and run ArrayView again.",
                                flush=True,
                            )
                            return False
                        else:
                            filenames = tuple(_wfiles) if _wfiles else (_VSCODE_SIGNAL_FILENAME,)
                    else:
                        filenames = (
                            _VSCODE_SIGNAL_FILENAME,
                            *_VSCODE_COMPAT_SIGNAL_FILENAMES,
                        )
        elif not is_remote:
            # Local VS Code but no IPC hook found: try to find the current window
            # by matching our parent process tree against extension host PIDs.
            window_id = _find_current_vscode_window_id()
            if window_id:
                # Found a match! Write to the window-specific signal file.
                # The window_id is either a hookTag or a PID string.
                filenames = (
                    f"open-request-pid-{window_id}.json"
                    if window_id.isdigit()
                    else f"open-request-ipc-{window_id}.json",
                )
            else:
                # No window match found: check if we're in a VS Code terminal.
                # If so, write to ALL registered windows and let the extension that
                # has the active terminal claim it. If not in VS Code terminal,
                # fall back to shared file with a warning.
                from arrayview._platform import _in_vscode_terminal as _in_vsc_term

                if _in_vsc_term():
                    # Write to all registered window-specific files
                    signal_dir_temp = os.path.expanduser("~/.arrayview")
                    window_files = []
                    try:
                        for fname in os.listdir(signal_dir_temp):
                            if fname.startswith("window-") and fname.endswith(".json"):
                                wid = fname[7:-5]  # extract ID
                                window_files.append(
                                    f"open-request-pid-{wid}.json"
                                    if wid.isdigit()
                                    else f"open-request-ipc-{wid}.json"
                                )
                    except Exception:
                        pass
                    if len(window_files) > 1:
                        print(
                            "[ArrayView] VS Code window is ambiguous: "
                            "no exact ARRAYVIEW_WINDOW_ID or PID match is available "
                            f"and {len(window_files)} VS Code windows are active. "
                            "Open a fresh terminal in the target VS Code window "
                            "and run ArrayView again.",
                            flush=True,
                        )
                        return False
                    else:
                        filenames = (
                            tuple(window_files) if window_files else (_VSCODE_SIGNAL_FILENAME,)
                        )
                else:
                    # Not in VS Code terminal: use shared file
                    filenames = (
                        _VSCODE_SIGNAL_FILENAME,
                        *_VSCODE_COMPAT_SIGNAL_FILENAMES,
                    )
        else:
            # Remote/tunnel: the IPC hook and PID ancestry are shared across
            # all windows in a tunnel, so they can't distinguish windows.
            # Instead, use ARRAYVIEW_WINDOW_ID env var injected by the extension
            # into each terminal via EnvironmentVariableCollection.
            env_wid = _find_arrayview_window_id()
            if not env_wid and ipc_hook:
                # Tunnel windows on the same host have distinct IPC hooks even
                # though their process ancestry may overlap. Prefer the exact
                # live registration over a focus-based broadcast when uv/shell
                # wrappers stripped ARRAYVIEW_WINDOW_ID.
                ipc_wid = hashlib.sha256(ipc_hook.encode()).hexdigest()[:16]
                ipc_registration = os.path.join(
                    signal_dir, f"window-{ipc_wid}.json"
                )
                try:
                    with open(ipc_registration) as registration:
                        registration_data = json.load(registration)
                    if (
                        registration_data.get("remoteName")
                        and registration_data.get("hookTag") == ipc_wid
                    ):
                        env_wid = ipc_wid
                        _vprint(
                            "[ArrayView] signal: recovered exact remote window "
                            f"from IPC hook → {ipc_wid}",
                            flush=True,
                        )
                except (OSError, TypeError, ValueError):
                    pass
            _vprint(f"[ArrayView] signal: remote mode, ARRAYVIEW_WINDOW_ID={env_wid or '(not found)'}", flush=True)

            if env_wid:
                # Verify the window is still registered
                reg_file = os.path.join(signal_dir, f"window-{env_wid}.json")
                if os.path.isfile(reg_file):
                    try:
                        with open(reg_file) as _rf:
                            _reg_data = json.load(_rf)
                        _uses_pid = _reg_data.get("fallbackId", False)
                    except Exception:
                        _uses_pid = env_wid.isdigit()

                    # Trust an exact remote window ID. In tunnels, PID ancestry
                    # can be shared by multiple live windows, so redirecting to
                    # a newer same-parent registration can open the wrong tab.
                    _prefix = "pid" if _uses_pid else "ipc"
                    _vprint(f"[ArrayView] signal: env window match → {_prefix}-{env_wid}", flush=True)
                    filenames = (f"open-request-{_prefix}-{env_wid}.json",)
                else:
                    # Registration missing for remote env wid — smart fallback.
                    _vprint(
                        f"[ArrayView] signal: remote env {env_wid} registration missing, "
                        f"applying smart fallback",
                        flush=True,
                    )
                    try:
                        _all_windows_r = [
                            fn for fn in os.listdir(signal_dir)
                            if fn.startswith("window-") and fn.endswith(".json")
                        ]
                    except Exception:
                        _all_windows_r = []

                    if len(_all_windows_r) == 1:
                        _sole_wid_r = _all_windows_r[0][7:-5]
                        try:
                            with open(os.path.join(signal_dir, _all_windows_r[0])) as _rf:
                                _sole_reg_r = json.load(_rf)
                            _uses_pid_r = _sole_reg_r.get("fallbackId", False)
                        except Exception:
                            _uses_pid_r = _sole_wid_r.isdigit()
                        _prefix_r = "pid" if _uses_pid_r else "ipc"
                        filenames = (f"open-request-{_prefix_r}-{_sole_wid_r}.json",)
                        env_wid = _sole_wid_r  # prevents the `if not env_wid:` fallback below
                        _vprint(
                            f"[ArrayView] signal: remote single window → {filenames[0]}",
                            flush=True,
                        )
                    elif len(_all_windows_r) > 1:
                        print(
                            "[ArrayView] VS Code remote window is ambiguous: "
                            f"ARRAYVIEW_WINDOW_ID={env_wid!r} is not registered "
                            f"and {len(_all_windows_r)} remote windows are active. "
                            "Open a fresh terminal in the target VS Code window "
                            "and run ArrayView again.",
                            flush=True,
                        )
                        return False
                    else:
                        env_wid = None  # 0 windows: keep existing fallback

            if not env_wid:
                try:
                    _all_windows_r = [
                        fn for fn in os.listdir(signal_dir)
                        if fn.startswith("window-") and fn.endswith(".json")
                    ]
                except Exception:
                    _all_windows_r = []
                if len(_all_windows_r) == 1:
                    _sole_wid_r = _all_windows_r[0][7:-5]
                    try:
                        with open(os.path.join(signal_dir, _all_windows_r[0])) as _rf:
                            _sole_reg_r = json.load(_rf)
                        _uses_pid_r = _sole_reg_r.get("fallbackId", False)
                    except Exception:
                        _uses_pid_r = _sole_wid_r.isdigit()
                    _prefix_r = "pid" if _uses_pid_r else "ipc"
                    filenames = (f"open-request-{_prefix_r}-{_sole_wid_r}.json",)
                    _vprint(
                        f"[ArrayView] signal: remote single window → {filenames[0]}",
                        flush=True,
                    )
                elif len(_all_windows_r) > 1:
                    print(
                        "[ArrayView] VS Code remote window is ambiguous: "
                        "no exact ARRAYVIEW_WINDOW_ID or live IPC match is available "
                        f"and {len(_all_windows_r)} remote windows are active. "
                        "Open a fresh terminal in the target VS Code window "
                        "and run ArrayView again.",
                        flush=True,
                    )
                    return False
                else:
                    _vprint(f"[ArrayView] signal: no registered remote window, using shared fallback", flush=True)
                    filenames = (
                        (_VSCODE_SIGNAL_FILENAME,)
                        if skip_compat
                        else (_VSCODE_SIGNAL_FILENAME, *_VSCODE_COMPAT_SIGNAL_FILENAMES)
                    )

        # Keep protocol correlation aligned with the window that will actually
        # consume the request. Target selection can recover from a stale
        # ARRAYVIEW_WINDOW_ID by redirecting to the sole live registration;
        # retaining the stale ID in the payload makes every valid ACK fail
        # correlation. Broadcasts deliberately have no expected window: the
        # focused extension host establishes it when claiming the request.
        resolved_window_id: str | None = None
        if not data.get("broadcast") and len(filenames) == 1:
            target = filenames[0]
            for prefix in ("open-request-pid-", "open-request-ipc-"):
                if target.startswith(prefix) and target.endswith(".json"):
                    resolved_window_id = target[len(prefix) : -len(".json")]
                    break
        if resolved_window_id:
            data["windowId"] = resolved_window_id
            payload["windowId"] = resolved_window_id
        else:
            data.pop("windowId", None)
            payload.pop("windowId", None)

        _vprint(f"[ArrayView] signal: writing to {[f for f in filenames]} broadcast={data.get('broadcast', False)}", flush=True)

        def _registration_supports_queue(filename: str) -> bool:
            """Return whether the live consumer understands queued filenames."""
            window_id: str | None = None
            for prefix in ("open-request-pid-", "open-request-ipc-"):
                if filename.startswith(prefix) and filename.endswith(".json"):
                    window_id = filename[len(prefix) : -len(".json")]
                    break
            if window_id is not None:
                registration_names = (f"window-{window_id}.json",)
            else:
                try:
                    registration_names = tuple(
                        name
                        for name in os.listdir(signal_dir)
                        if name.startswith("window-") and name.endswith(".json")
                    )
                except OSError:
                    return False
                if not registration_names:
                    return False
            for registration_name in registration_names:
                try:
                    with open(os.path.join(signal_dir, registration_name)) as registration:
                        if json.load(registration).get("signalQueueVersion", 0) < 1:
                            return False
                except (OSError, TypeError, ValueError):
                    return False
            return True

        for filename in filenames:
            # Protocol-v1 requests use a unique disk-backed queue entry.  A
            # fixed per-window filename lets simultaneous launchers overwrite
            # each other before the extension can claim the first request.
            # Capability-check the running extension host: installing a newer
            # VSIX does not activate it until VS Code reloads, and older hosts
            # only watch the fixed legacy filename.
            request_id = data.get("requestId")
            if (
                request_id
                and filename.endswith(".json")
                and _registration_supports_queue(filename)
            ):
                filename = f"{filename[:-5]}.request-{request_id}.json"
            else:
                try:
                    os.unlink(os.path.join(signal_dir, filename))
                except FileNotFoundError:
                    pass
            signal_file = os.path.join(signal_dir, filename)
            tmp_file = signal_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump(data, f)
            os.replace(tmp_file, signal_file)  # atomic on POSIX + Windows
        return True
    except Exception:
        return False


def _cleanup_zombie_registrations(verbose: bool = False) -> int:
    """Remove stale window-*.json registrations from ~/.arrayview/.

    Returns count of removed files.
    """
    signal_dir = os.path.expanduser("~/.arrayview")
    if not os.path.isdir(signal_dir):
        return 0

    removed = 0
    registrations = {}

    try:
        files = os.listdir(signal_dir)
    except OSError:
        return 0

    for fn in files:
        if not fn.startswith("window-") or not fn.endswith(".json"):
            continue
        wid = fn[7:-5]
        try:
            with open(os.path.join(signal_dir, fn)) as f:
                registrations[wid] = json.load(f)
        except Exception:
            continue

    for wid, data in list(registrations.items()):
        pid = data.get("pid")
        if pid is None:
            continue
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            try:
                os.unlink(os.path.join(signal_dir, f"window-{wid}.json"))
                removed += 1
                del registrations[wid]
                if verbose:
                    print(
                        f"[ArrayView] Removed zombie registration window-{wid} "
                        f"(pid {pid} dead)",
                        flush=True,
                    )
            except OSError:
                pass

    groups = {}
    for wid, data in registrations.items():
        ppids = data.get("ppids", [])
        pp0 = ppids[0] if ppids else wid
        groups.setdefault(pp0, []).append((wid, data))

    for pp0, regs in groups.items():
        if len(regs) <= 1:
            continue
        fallbacks = [(w, d) for w, d in regs if d.get("fallbackId")]
        non_fallbacks = [(w, d) for w, d in regs if not d.get("fallbackId")]

        if non_fallbacks:
            for wid, data in fallbacks:
                try:
                    os.unlink(os.path.join(signal_dir, f"window-{wid}.json"))
                    removed += 1
                    if verbose:
                        pid = data.get("pid", "?")
                        print(
                            f"[ArrayView] Removed stale fallback registration "
                            f"window-{wid} (pid {pid})",
                            flush=True,
                        )
                except OSError:
                    pass
        else:
            fallbacks.sort(key=lambda x: x[1].get("ts", 0), reverse=True)
            for wid, data in fallbacks[1:]:
                try:
                    os.unlink(os.path.join(signal_dir, f"window-{wid}.json"))
                    removed += 1
                    if verbose:
                        pid = data.get("pid", "?")
                        print(
                            f"[ArrayView] Removed duplicate fallback registration "
                            f"window-{wid} (pid {pid})",
                            flush=True,
                        )
                except OSError:
                    pass

    valid_ids = set(registrations.keys())

    now = time.time()
    for fn in files:
        try:
            fp = os.path.join(signal_dir, fn)
        except Exception:
            continue
        if fn.startswith("open-request-ipc-") or fn.startswith("open-request-pid-"):
            parts = fn[len("open-request-"):].split("-", 1)
            if len(parts) == 2 and parts[1].endswith(".json"):
                wid = parts[1][:-5].split(".request-", 1)[0]
                if wid not in valid_ids:
                    try:
                        os.unlink(fp)
                        removed += 1
                        if verbose:
                            print(
                                f"[ArrayView] Removed stale signal file {fn} "
                                f"(window {wid} not registered)",
                                flush=True,
                            )
                    except OSError:
                        pass
        elif fn in ("open-request-v0900.json", "open-request-v0800.json") or (
            fn.startswith("open-request-v0900.request-") and fn.endswith(".json")
        ):
            try:
                st = os.stat(fp)
                if now - st.st_mtime > 30:
                    os.unlink(fp)
                    removed += 1
                    if verbose:
                        print(
                            f"[ArrayView] Removed stale shared signal file {fn}",
                            flush=True,
                        )
            except OSError:
                pass

    return removed
