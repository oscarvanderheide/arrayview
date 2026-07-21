"""VS Code browser opening, remote/tunnel handling, and SSH guidance."""

from __future__ import annotations

import os
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum

from arrayview._session import _vprint
from arrayview._platform import _in_vscode_terminal, _is_vscode_remote
from arrayview._vscode_extension import _configure_vscode_port_preview, _ensure_vscode_extension
from arrayview._vscode_signal import (
    AckState,
    _open_via_signal_file,
    _schedule_remote_open_retries,
    _wait_for_vscode_ack,
)

# Whether the "set port to Public" message has been printed this session.
_remote_message_shown = False
_ssh_message_shown = False


def _trace_launch_event(event: str, **attrs: object) -> None:
    if not os.environ.get("ARRAYVIEW_LAUNCH_TRACE"):
        return
    from arrayview._launch_trace import emit_launch_event

    emit_launch_event(event, **attrs)


def _trace_tag(value: object) -> str | None:
    if not os.environ.get("ARRAYVIEW_LAUNCH_TRACE") or value is None:
        return None
    from arrayview._launch_trace import trace_tag

    return trace_tag(value)


class OpenState(str, Enum):
    """Best evidence available after requesting a viewer display."""

    ACCEPTED = "accepted"
    OPENED = "opened"
    READY = "ready"
    PRINTED = "printed"
    FAILED = "failed"


@dataclass(frozen=True)
class OpenResult:
    """Outcome of handing a viewer URL to a display mechanism."""

    state: OpenState
    mechanism: str
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.state is not OpenState.FAILED


def _server_id_for_url(url: str) -> str | None:
    """Read the instance ID used to correlate a VS Code readiness ACK."""
    try:
        parsed = urllib.parse.urlsplit(url)
        ping_url = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, "/ping", "", "")
        )
        with urllib.request.urlopen(ping_url, timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if payload.get("service") != "arrayview":
        return None
    instance_id = payload.get("instance_id")
    return instance_id if isinstance(instance_id, str) and instance_id else None


# ---------------------------------------------------------------------------
# Browser opening
# ---------------------------------------------------------------------------


def _print_viewer_location(url: str) -> None:
    """Print a viewer location hint.

    In VS Code remote/tunnel sessions the plain ``http://localhost:<port>/``
    URL is printed to stdout (unconditionally, not gated on verbose) so VS
    Code's terminal-output port-forward detector picks it up and forwards the
    port immediately.  Without this, VS Code relies on its slow proactive
    socket scan, which can take 10+ seconds and sometimes never fires —
    leaving ``asExternalUri`` returning the localhost URL unchanged and the
    viewer tab blank on the remote client.

    Only the host needs to be localhost for the forward to trigger; the
    query string is harmless and helps debugging.  We print the URL without
    ANSI styling so VS Code's detector matches it cleanly.
    """
    if _is_vscode_remote():
        print(url, flush=True)
        return
    _vprint(f"[ArrayView] {url}", flush=True)


def _open_browser(
    url: str,
    blocking: bool = False,
    force_vscode: bool = False,
    prefer_system_browser: bool = False,
    title: str | None = None,
    floating: bool = False,
) -> OpenResult:
    """Open *url* locally, or configure VS Code remote auto-preview behavior.

    *title* is shown as the VS Code tab label (e.g. "ArrayView: sample.npy").
    Passed through to _open_via_signal_file → extension createWebviewPanel.

    Strategy (see log.txt for what was tried and why):
    1. Remote VS Code terminal:
       a. Configure the port as ``silent`` and ``public`` in
          ``remote.portsAttributes``.
       b. Write the signal file; the workspace extension converts the URL via
          asExternalUri and opens a viewer tab in the local VS Code client.
    2. Local VS Code terminal (or force_vscode=True):
       a. Install the helper extension.
       b. Write the signal file so the extension opens a viewer tab locally.
     3. Fallback: open/xdg-open with the http URL (system browser).
     4. Always print the URL.
    """

    _trace_launch_event(
        "display.routing_scheduled",
        blocking=blocking,
        force_vscode=force_vscode,
        prefer_system_browser=prefer_system_browser,
    )

    def _route() -> OpenResult:
        # A failed explicit native-window launch may fall back to a browser,
        # but it must not silently turn back into a VS Code tab. Remote VS Code
        # still takes precedence because a browser on the remote host is not
        # useful to the caller.
        in_vscode = _in_vscode_terminal() and not prefer_system_browser
        is_remote = _is_vscode_remote()
        is_plain_ssh = (
            not is_remote
            and not in_vscode
            and bool(os.environ.get("SSH_CLIENT") or os.environ.get("SSH_CONNECTION"))
        )
        selected_adapter = (
            "vscode"
            if is_remote or force_vscode or in_vscode
            else "ssh-guidance"
            if is_plain_ssh
            else "system-browser"
        )
        _trace_launch_event(
            "display.router_evaluated",
            force_vscode=force_vscode,
            prefer_system_browser=prefer_system_browser,
            detected_vscode=in_vscode,
            detected_remote=is_remote,
            selected_adapter=selected_adapter,
        )
        _trace_launch_event(
            "display.attempt_started",
            adapter=selected_adapter,
            stage="browser_router",
        )
        opened = False
        guidance_printed = False

        try:
            parsed_port = int(url.split(":")[2].split("/")[0].split("?")[0])
        except Exception:
            parsed_port = 8000

        if is_remote:
            # Remote/tunnel: install extension + write signal file.
            ext_ok = _ensure_vscode_extension()
            from arrayview import _vscode_extension as _extension_state
            if not ext_ok and _extension_state._VSCODE_EXT_RELOAD_REQUIRED:
                return OpenResult(
                    OpenState.FAILED,
                    "vscode-extension",
                    "ArrayView updated its VS Code opener; reload this VS Code window once, then retry",
                )
            # URL-based mode: port is forwarded by VS Code and the viewer
            # connects via WebSocket through the devtunnel.
            _configure_vscode_port_preview(parsed_port)
            request = _open_via_signal_file(
                url,
                title=title,
                floating=floating,
                server_id=_server_id_for_url(url),
            )
            _trace_launch_event(
                "vscode.request_written",
                written=bool(request),
                request_tag=_trace_tag(request.request_id),
                window_tag=_trace_tag(request.window_id),
                server_tag=_trace_tag(request.server_id),
            )
            if not ext_ok:
                _vprint(
                    "[ArrayView] extension install could not be verified — signal file written anyway",
                    flush=True,
                )
            if not request:
                return OpenResult(
                    OpenState.FAILED,
                    "vscode-signal",
                    "signal request was not written",
                )
            if not blocking:
                _schedule_remote_open_retries(url, interval=10.0, count=2)
                return OpenResult(OpenState.ACCEPTED, "vscode-signal")
            # Large remote files can remain in PENDING_SESSIONS while the
            # extension is opening the forwarded panel. Keep the terminal
            # alive long enough for that load and the first frame.
            ack = _wait_for_vscode_ack(request, timeout=195.0)
            if ack.state is AckState.BACKEND_READY:
                return OpenResult(OpenState.READY, "vscode-signal", request.request_id)
            return OpenResult(
                OpenState.FAILED,
                "vscode-signal",
                ack.message or ack.state.value,
            )

        if force_vscode or in_vscode:
            # Local VS Code terminal (or --window vscode forced): install extension + signal file.
            _configure_vscode_port_preview(parsed_port)
            ext_ok = _ensure_vscode_extension()
            from arrayview import _vscode_extension as _extension_state
            if not ext_ok and _extension_state._VSCODE_EXT_RELOAD_REQUIRED:
                return OpenResult(
                    OpenState.FAILED,
                    "vscode-extension",
                    "ArrayView updated its VS Code opener; reload this VS Code window once, then retry",
                )
            # Always write the signal file: the extension may already be
            # installed even if _ensure failed (e.g. `code` CLI not found).
            request = _open_via_signal_file(
                url,
                title=title,
                floating=floating,
                server_id=_server_id_for_url(url),
            )
            _trace_launch_event(
                "vscode.request_written",
                written=bool(request),
                request_tag=_trace_tag(request.request_id),
                window_tag=_trace_tag(request.window_id),
                server_tag=_trace_tag(request.server_id),
            )
            # Schedule a retry in case the extension was mid-reload when the
            # first signal was written (e.g. first run with old version removed).
            if not request:
                return OpenResult(
                    OpenState.FAILED,
                    "vscode-signal",
                    "signal request was not written",
                )
            if blocking:
                ack = _wait_for_vscode_ack(request, timeout=15.0)
                if ack.state is AckState.BACKEND_READY:
                    return OpenResult(
                        OpenState.READY,
                        "vscode-signal",
                        request.request_id,
                    )
                return OpenResult(
                    OpenState.FAILED,
                    "vscode-signal",
                    ack.message or ack.state.value,
                )
            _schedule_remote_open_retries(url, interval=10.0, count=2)
            opened = True

        if is_plain_ssh:
            guidance_printed = True
            global _ssh_message_shown
            if not _ssh_message_shown:
                _ssh_message_shown = True
                try:
                    port_hint = int(url.split(":")[2].split("/")[0].split("?")[0])
                except Exception:
                    port_hint = parsed_port
                hostname = platform.node() or "<this-host>"
                print(
                    f"[ArrayView] Plain SSH session detected.\n"
                    f"\n"
                    f"  For the best experience, use VS Code Remote-SSH — arrays open\n"
                    f"  automatically in a VS Code tab with zero setup.\n"
                    f"\n"
                    f"  From a plain terminal, forward port {port_hint} when connecting:\n"
                    f"\n"
                    f"    ssh -L {port_hint}:localhost:{port_hint} <user>@{hostname}\n"
                    f"\n"
                    f"  Then open in your local browser:\n"
                    f"\n"
                    f"    http://localhost:{port_hint}/\n"
                    f"\n"
                    f"  Tip: to avoid typing -L every time, add this to ~/.ssh/config\n"
                    f"  on the machine you SSH from:\n"
                    f"\n"
                    f"    Host {hostname}\n"
                    f"        LocalForward {port_hint} localhost:{port_hint}\n",
                    flush=True,
                )
        if not opened and not is_remote and not force_vscode and not is_plain_ssh:
            # Local fallback: open in system browser
            if sys.platform == "darwin":
                try:
                    r = subprocess.run(["open", url], capture_output=True, timeout=5)
                    opened = r.returncode == 0
                except Exception:
                    pass
            elif sys.platform.startswith("linux"):
                try:
                    r = subprocess.run(
                        ["xdg-open", url], capture_output=True, timeout=5
                    )
                    opened = r.returncode == 0
                except Exception:
                    pass
            elif sys.platform == "win32":
                # os.startfile is the documented way to open a URL with the
                # default handler on Windows; fall back to `cmd /c start` if
                # it is unavailable (rare — constrained environments).
                try:
                    os.startfile(url)  # type: ignore[attr-defined]
                    opened = True
                except Exception:
                    try:
                        subprocess.run(
                            ["cmd", "/c", "start", "", url],
                            capture_output=True, timeout=5, shell=False,
                        )
                        opened = True
                    except Exception:
                        pass

        # Local sessions still benefit from a clickable terminal URL.
        _vprint(f"\n  \033[1;36m→ {url}\033[0m\n", flush=True)
        if opened:
            mechanism = (
                "vscode-signal" if (force_vscode or in_vscode) else "system-browser"
            )
            state = (
                OpenState.ACCEPTED
                if mechanism == "vscode-signal"
                else OpenState.OPENED
            )
            return OpenResult(state, mechanism)
        if guidance_printed:
            return OpenResult(OpenState.PRINTED, "ssh-guidance")
        return OpenResult(OpenState.FAILED, "system-browser", "no opener accepted the URL")

    def _do() -> OpenResult:
        result = _route()
        _trace_launch_event(
            "display.attempt_finished",
            adapter=result.mechanism,
            stage="browser_router",
            state=result.state.value,
        )
        return result

    if blocking:
        result = _do()
        if force_vscode and not result:
            raise RuntimeError(
                "[ArrayView] VS Code viewer failed to become ready: "
                f"{result.detail or result.state.value}"
            )
        return result
    threading.Thread(target=_do, daemon=True).start()
    return OpenResult(OpenState.ACCEPTED, "background-thread")
